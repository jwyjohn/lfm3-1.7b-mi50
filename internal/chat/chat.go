// Package chat renders messages with the LFM3 chat template. It is a direct
// translation of the model's chat_template.jinja; there is no template engine.
package chat

import (
	"fmt"
	"strings"

	"lfm3-1.7b-mi50/internal/ojson"
)

const cfmTag = "CONTINUE_FINAL_MESSAGE_TAG "

// Options are the template's inputs other than messages and tools.
type Options struct {
	AddGenerationPrompt bool
	EnableThinking      bool
	PreserveThinking    bool
}

// RenderError is a template raise_exception: the request is at fault.
type RenderError struct{ Msg string }

func (e *RenderError) Error() string { return e.Msg }

type renderer struct {
	tools []ojson.Value
}

func isMapping(v ojson.Value) bool {
	_, ok := v.(*ojson.Object)
	return ok
}

// renderValue is the template's render_value macro.
func renderValue(v ojson.Value) string {
	switch x := v.(type) {
	case string:
		return x
	case bool:
		if x {
			return "true"
		}
		return "false"
	case nil:
		return "null"
	case *ojson.Object, []ojson.Value:
		return ojson.Encode(x)
	case ojson.Number:
		return string(x)
	}
	return fmt.Sprint(v)
}

func stringify(v ojson.Value) string {
	switch x := v.(type) {
	case string:
		return x
	case nil:
		return "None"
	case bool:
		if x {
			return "True"
		}
		return "False"
	case ojson.Number:
		return string(x)
	}
	return ojson.Encode(v)
}

func textOf(item *ojson.Object) string {
	t, _ := item.Get("text")
	if !ojson.Truthy(t) {
		return ""
	}
	return stringify(t)
}

// parseContent is the template's parse_content macro.
func parseContent(c ojson.Value) string {
	switch x := c.(type) {
	case string:
		return x
	case *ojson.Object:
		switch x.Str("type") {
		case "image":
			return "<|vision_start|><|image|><|vision_end|>"
		case "video":
			return "<|vision_start|><|video|><|vision_end|>"
		case "text":
			return textOf(x)
		}
		return ojson.Encode(x)
	case []ojson.Value:
		var b strings.Builder
		for _, item := range x {
			switch it := item.(type) {
			case string:
				b.WriteString(it)
			case *ojson.Object:
				switch it.Str("type") {
				case "image":
					b.WriteString("<|vision_start|><|image|><|vision_end|>")
				case "video":
					b.WriteString("<|vision_start|><|video|><|vision_end|>")
				case "text":
					b.WriteString(textOf(it))
				default:
					b.WriteString(ojson.Encode(it))
				}
			default:
				b.WriteString(ojson.Encode(item))
			}
		}
		return b.String()
	case nil:
		return ""
	}
	return ojson.Encode(c)
}

func renderToolCalls(calls []ojson.Value) (string, error) {
	var b strings.Builder
	for i, tc := range calls {
		if i > 0 {
			b.WriteString("\n\n")
		}
		t, ok := tc.(*ojson.Object)
		if !ok {
			return "", &RenderError{"tool_calls entries must be objects"}
		}
		fn := t
		if f := t.Obj("function"); f != nil {
			fn = f
		}
		args, _ := fn.Get("arguments")
		if s, isStr := args.(string); isStr {
			if tr := strings.TrimSpace(s); tr != "" && tr != "{}" && tr != "null" {
				return "", &RenderError{"Tool call arguments must be a mapping, got a JSON-encoded string: parse arguments with json.loads() before applying the chat template"}
			}
		}
		b.WriteString("<tool_call>\n<function=" + fn.Str("name") + ">\n")
		if m, ok := args.(*ojson.Object); ok {
			for _, k := range m.Keys {
				b.WriteString("<parameter=" + k + ">\n" + renderValue(m.Vals[k]) + "\n</parameter>\n")
			}
		}
		b.WriteString("</function>\n</tool_call>")
	}
	return b.String(), nil
}

func toolToJSON(tool *ojson.Object) string {
	var b strings.Builder
	b.WriteByte('{')
	first := true
	for _, k := range tool.Keys {
		if k == "defer_loading" {
			continue
		}
		if !first {
			b.WriteString(", ")
		}
		first = false
		ojson.EncodeString(&b, k)
		b.WriteString(": ")
		b.WriteString(ojson.Encode(tool.Vals[k]))
	}
	b.WriteByte('}')
	return b.String()
}

func functionOf(tool ojson.Value) *ojson.Object {
	t, _ := tool.(*ojson.Object)
	if t == nil {
		return ojson.NewObject()
	}
	if f, ok := t.Get("function"); ok {
		if fo, ok := f.(*ojson.Object); ok {
			return fo
		}
	}
	return t
}

func isToolReference(v ojson.Value) bool {
	o, ok := v.(*ojson.Object)
	return ok && o.Str("type") == "tool_reference"
}

func (r *renderer) referencedTool(name string) (string, error) {
	var found *ojson.Object
	for _, t := range r.tools {
		fn := functionOf(t)
		if fn.Str("name") == name {
			found = fn
		}
	}
	if found == nil {
		return "", &RenderError{"Tool reference '" + name + "' not found in available tools. Every tool referenced by a tool_reference must be in `tools` on every request."}
	}
	return toolToJSON(found), nil
}

func toolListText(src []ojson.Value) string {
	parts := make([]string, 0, len(src))
	for _, t := range src {
		if o, ok := t.(*ojson.Object); ok {
			parts = append(parts, toolToJSON(o))
		}
	}
	return strings.Join(parts, "\n")
}

// renderToolResponse is the template's render_tool_response macro.
func (r *renderer) renderToolResponse(content ojson.Value) (string, error) {
	if s, ok := content.(string); ok && strings.Contains(s, "<tool_response>") {
		return s, nil
	}
	var b strings.Builder
	b.WriteString("<tool_response>\n")
	switch x := content.(type) {
	case string:
		b.WriteString(x)
	case *ojson.Object:
		b.WriteString(parseContent(x))
	case []ojson.Value:
		for i, item := range x {
			if i > 0 {
				b.WriteString("\n")
			}
			switch {
			case isToolReference(item):
				if i == 0 || !isToolReference(x[i-1]) {
					b.WriteString("<additional_tools>\n")
				}
				name := item.(*ojson.Object).Str("name")
				t, err := r.referencedTool(name)
				if err != nil {
					return "", err
				}
				b.WriteString(t)
				if i == len(x)-1 || !isToolReference(x[i+1]) {
					b.WriteString("\n</additional_tools>")
				}
			default:
				if o, ok := item.(*ojson.Object); ok {
					if res := o.Obj("result"); res != nil {
						if tl, ok := res.Get("tools"); ok {
							b.WriteString("<additional_tools>\n")
							switch v := tl.(type) {
							case string:
								b.WriteString(v)
							case *ojson.Object:
								b.WriteString(toolToJSON(v))
							case []ojson.Value:
								b.WriteString(toolListText(v))
							}
							b.WriteString("\n</additional_tools>")
							continue
						}
					}
				}
				b.WriteString(parseContent(item))
			}
		}
	case nil:
	default:
		b.WriteString(ojson.Encode(content))
	}
	b.WriteString("\n</tool_response>")
	return b.String(), nil
}

func msgObj(v ojson.Value) (*ojson.Object, error) {
	m, ok := v.(*ojson.Object)
	if !ok {
		return nil, &RenderError{"messages entries must be objects"}
	}
	return m, nil
}

// Render produces the prompt text.
func Render(messages []ojson.Value, tools []ojson.Value, opt Options) (string, error) {
	r := &renderer{tools: tools}
	var b strings.Builder
	b.WriteString("<|startoftext|>")
	system := ""
	if len(messages) > 0 {
		first, err := msgObj(messages[0])
		if err != nil {
			return "", err
		}
		if first.Str("role") == "system" {
			if c, _ := first.Get("content"); ojson.Truthy(c) {
				system = parseContent(c)
			}
			messages = messages[1:]
		}
	}
	if len(tools) > 0 {
		var lines []string
		for _, tool := range tools {
			fn := functionOf(tool)
			dyn := false
			if v, ok := fn.Get("defer_loading"); ok && ojson.Truthy(v) {
				dyn = true
			}
			if t, ok := tool.(*ojson.Object); ok {
				if v, ok := t.Get("defer_loading"); ok && ojson.Truthy(v) {
					dyn = true
				}
			}
			if !dyn {
				lines = append(lines, toolToJSON(fn))
			}
		}
		if len(lines) > 0 {
			if system != "" {
				system += "\n\n"
			}
			system += "<tools>\n" + strings.Join(lines, "\n") + "\n</tools>"
		}
	}
	if system != "" {
		b.WriteString("<|im_start|>system\n" + system + "<|im_end|>\n")
	}
	lastUser := -1
	for i, mv := range messages {
		m, err := msgObj(mv)
		if err != nil {
			return "", err
		}
		if m.Str("role") == "user" {
			lastUser = i
		}
	}
	var pending []ojson.Value
	for i, mv := range messages {
		m, _ := msgObj(mv)
		role := m.Str("role")
		additional, hasAdditional := m.Get("additional_tools")
		additionalSet := hasAdditional && ojson.Truthy(additional)
		merge := false
		if role == "tool" && !additionalSet && i+1 < len(messages) {
			nx, err := msgObj(messages[i+1])
			if err != nil {
				return "", err
			}
			na, naOK := nx.Get("additional_tools")
			id := m.Str("tool_call_id")
			merge = nx.Str("role") == "tool" && !(naOK && ojson.Truthy(na)) && id != "" && nx.Str("tool_call_id") == id
		}
		var items []ojson.Value
		if role == "tool" && !additionalSet {
			c, _ := m.Get("content")
			switch x := c.(type) {
			case string:
				if x != "" {
					o := ojson.NewObject()
					o.Set("type", "text")
					o.Set("text", x)
					items = []ojson.Value{o}
				}
			case *ojson.Object:
				items = []ojson.Value{x}
			case []ojson.Value:
				items = append(items, x...)
			case nil:
			default:
				items = []ojson.Value{x}
			}
		}
		if merge {
			pending = append(pending, items...)
			continue
		}
		b.WriteString("<|im_start|>" + role + "\n")
		switch role {
		case "assistant":
			var thinking ojson.Value
			defined := false
			for _, key := range []string{"thinking", "reasoning", "reasoning_content"} {
				if v, ok := m.Get(key); ok {
					thinking, defined = v, true
					break
				}
			}
			keep := opt.PreserveThinking || i > lastUser
			contentV, _ := m.Get("content")
			contentStr, contentIsStr := contentV.(string)
			switch {
			case defined && keep:
				t := ""
				if ojson.Truthy(thinking) {
					t = stringify(thinking)
				}
				b.WriteString("<think>" + t + "</think>")
			case keep && !(contentIsStr && strings.Contains(contentStr, "</think>")):
				b.WriteString("<think></think>")
			}
			content := ""
			if ojson.Truthy(contentV) {
				content = parseContent(contentV)
			}
			if !keep {
				if idx := strings.LastIndex(content, "</think>"); idx >= 0 {
					content = pyTrim(content[idx+len("</think>"):])
				}
				content = "<think></think>" + content
			}
			hasCfm := false
			if strings.HasSuffix(content, cfmTag) {
				hasCfm = true
				content = content[:len(content)-len(cfmTag)]
			}
			b.WriteString(content)
			if calls := m.Arr("tool_calls"); len(calls) > 0 {
				s, err := renderToolCalls(calls)
				if err != nil {
					return "", err
				}
				b.WriteString(s)
			}
			if hasCfm {
				b.WriteString(cfmTag)
			}
			b.WriteString("<|im_end|>\n")
		case "tool":
			switch {
			case additionalSet:
				var src ojson.Value = additional
				_, isObj := additional.(*ojson.Object)
				_, isArr := additional.([]ojson.Value)
				if !isObj && !isArr {
					src, _ = m.Get("content")
				}
				str, isStr := src.(string)
				switch {
				case isStr && strings.Contains(str, "<tool_response>"):
					b.WriteString(str)
				case isStr && strings.Contains(str, "<additional_tools>"):
					b.WriteString("<tool_response>\n" + str + "\n</tool_response>")
				case ojson.Truthy(src):
					b.WriteString("<tool_response>\n<additional_tools>\n")
					switch v := src.(type) {
					case string:
						b.WriteString(v)
					case *ojson.Object:
						b.WriteString(toolToJSON(v))
					case []ojson.Value:
						b.WriteString(toolListText(v))
					}
					b.WriteString("\n</additional_tools>\n</tool_response>")
				}
			case len(pending) > 0:
				s, err := r.renderToolResponse(append(pending, items...))
				if err != nil {
					return "", err
				}
				b.WriteString(s)
				pending = nil
			default:
				c, _ := m.Get("content")
				if c == nil {
					c = ""
				}
				s, err := r.renderToolResponse(c)
				if err != nil {
					return "", err
				}
				b.WriteString(s)
			}
			b.WriteString("<|im_end|>\n")
		default:
			if c, _ := m.Get("content"); ojson.Truthy(c) {
				b.WriteString(parseContent(c))
			}
			b.WriteString("<|im_end|>\n")
		}
	}
	if opt.AddGenerationPrompt {
		b.WriteString("<|im_start|>assistant\n")
		if opt.EnableThinking {
			b.WriteString("<think>")
		} else {
			b.WriteString("<think></think>")
		}
	}
	return b.String(), nil
}

// pyTrim is Jinja's trim filter: Python str.strip() over Unicode whitespace.
func pyTrim(s string) string { return strings.TrimSpace(s) }
