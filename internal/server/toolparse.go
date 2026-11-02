package server

import (
	"encoding/json"
	"math"
	"strconv"
	"strings"

	"lfm3-1.7b-mi50/internal/ojson"
)

// Event is one parsed piece of assistant output.
type Event struct {
	Reasoning string
	Content   string
	ToolStart *ToolCall // a new tool call, with its name
	ToolArgs  string    // JSON text extending the current call's arguments
	ToolIndex int
}

// ToolCall is one completed call in a non-streaming response.
type ToolCall struct {
	ID   string
	Name string
	Args string
}

type parseState int

const (
	stReasoning parseState = iota
	stContent
	stToolPreamble
	stToolName
	stToolArgs
	stToolBetween
)

const (
	thinkStart = "<think>"
	thinkEnd   = "</think>"
	toolStart  = "<tool_call>"
	toolEnd    = "</tool_call>"
	funcPrefix = "<function="
	funcEnd    = "</function>"
	paramStart = "<parameter="
	paramEnd   = "</parameter>"
)

// OutputParser splits streamed assistant text into reasoning, content and tool calls,
// following the state machine of vLLM's Qwen3 parser. Text that might be the start of
// a tag is held back until it can be classified.
type OutputParser struct {
	state    parseState
	tools    bool
	schemas  map[string]*ojson.Object
	buf      string
	calls    []*ToolCall
	name     string
	args     string // raw body of the current call
	streamed string // arguments JSON already sent for the current call
	newID    func() string
}

// NewOutputParser starts a parser. thinking: the prompt opened a <think> block.
func NewOutputParser(thinking bool, tools []ojson.Value, newID func() string) *OutputParser {
	p := &OutputParser{tools: len(tools) > 0, newID: newID, schemas: map[string]*ojson.Object{}}
	if thinking {
		p.state = stReasoning
	} else {
		p.state = stContent
	}
	for _, t := range tools {
		o, _ := t.(*ojson.Object)
		if o == nil {
			continue
		}
		fn := o.Obj("function")
		if fn == nil {
			fn = o
		}
		if params := fn.Obj("parameters"); params != nil {
			p.schemas[fn.Str("name")] = params.Obj("properties")
		}
	}
	return p
}

func (p *OutputParser) terminals() []string {
	switch p.state {
	case stReasoning:
		if p.tools {
			return []string{thinkStart, thinkEnd, toolStart}
		}
		return []string{thinkStart, thinkEnd}
	case stContent:
		if p.tools {
			return []string{thinkEnd, toolStart, funcPrefix}
		}
		return []string{thinkEnd}
	case stToolPreamble:
		return []string{toolEnd, funcPrefix}
	case stToolName:
		return []string{">", funcEnd, toolEnd, toolStart, funcPrefix}
	case stToolArgs:
		return []string{funcEnd, funcPrefix}
	default:
		return []string{toolEnd, toolStart, funcPrefix}
	}
}

// holdback returns how many trailing bytes of s might begin one of the terminals.
func holdback(s string, terms []string) int {
	best := 0
	for _, t := range terms {
		for n := len(t) - 1; n > best; n-- {
			if n <= len(s) && strings.HasSuffix(s, t[:n]) {
				best = n
				break
			}
		}
	}
	return best
}

// Feed consumes newly generated text.
func (p *OutputParser) Feed(text string) []Event {
	p.buf += text
	var out []Event
	for {
		terms := p.terminals()
		at, which := -1, ""
		for _, t := range terms {
			if i := strings.Index(p.buf, t); i >= 0 && (at < 0 || i < at) {
				at, which = i, t
			}
		}
		if at < 0 {
			keep := holdback(p.buf, terms)
			head := p.buf[:len(p.buf)-keep]
			p.buf = p.buf[len(p.buf)-keep:]
			out = p.plain(head, out)
			return out
		}
		out = p.plain(p.buf[:at], out)
		p.buf = p.buf[at+len(which):]
		out = p.transition(which, out)
	}
}

// Finish flushes the remaining text at the end of generation.
func (p *OutputParser) Finish() []Event {
	var out []Event
	rest := p.buf
	p.buf = ""
	out = p.plain(rest, out)
	if p.state == stToolArgs || p.state == stToolName {
		out = p.closeCall(out)
	}
	return out
}

func (p *OutputParser) plain(s string, out []Event) []Event {
	if s == "" {
		return out
	}
	switch p.state {
	case stReasoning:
		out = append(out, Event{Reasoning: s})
	case stContent:
		out = append(out, Event{Content: s})
	case stToolName:
		p.name += s
	case stToolArgs:
		p.args += s
		out = p.streamArgs(out, false)
	}
	return out
}

func (p *OutputParser) openCall() {
	p.name, p.args, p.streamed = "", "", ""
}

func (p *OutputParser) transition(term string, out []Event) []Event {
	switch p.state {
	case stReasoning:
		switch term {
		case thinkEnd:
			p.state = stContent
		case toolStart:
			p.openCall()
			p.state = stToolPreamble
		}
	case stContent:
		switch term {
		case toolStart:
			p.openCall()
			p.state = stToolPreamble
		case funcPrefix:
			p.openCall()
			p.state = stToolName
		}
	case stToolPreamble:
		switch term {
		case toolEnd:
			p.state = stContent
		case funcPrefix:
			p.state = stToolName
		}
	case stToolName:
		switch term {
		case ">":
			p.state = stToolArgs
			p.name = strings.TrimSpace(p.name)
			call := &ToolCall{ID: p.newID(), Name: p.name}
			p.calls = append(p.calls, call)
			out = append(out, Event{ToolStart: call, ToolIndex: len(p.calls) - 1})
		case funcEnd:
			p.state = stToolBetween
		case toolEnd:
			p.state = stContent
		case toolStart:
			p.openCall()
			p.state = stToolPreamble
		case funcPrefix:
			p.openCall()
		}
	case stToolArgs:
		out = p.closeCall(out)
		switch term {
		case funcEnd:
			p.state = stToolBetween
		case funcPrefix:
			p.openCall()
			p.state = stToolName
		}
	case stToolBetween:
		switch term {
		case toolEnd:
			p.state = stContent
		case toolStart:
			p.openCall()
			p.state = stToolPreamble
		case funcPrefix:
			p.openCall()
			p.state = stToolName
		}
	}
	return out
}

func (p *OutputParser) closeCall(out []Event) []Event {
	if len(p.calls) == 0 || p.state != stToolArgs {
		return out
	}
	out = p.streamArgs(out, true)
	p.calls[len(p.calls)-1].Args = p.streamed
	return out
}

// streamArgs emits the arguments JSON that can no longer change. Values are typed from
// the tool schema only once complete, so a parameter is released when it closes.
func (p *OutputParser) streamArgs(out []Event, final bool) []Event {
	if len(p.calls) == 0 {
		return out
	}
	full := argumentsJSON(p.args, p.schemas[p.name], final)
	if !strings.HasPrefix(full, p.streamed) || len(full) == len(p.streamed) {
		return out
	}
	delta := full[len(p.streamed):]
	p.streamed = full
	return append(out, Event{ToolArgs: delta, ToolIndex: len(p.calls) - 1})
}

// argumentsJSON converts <parameter=K>V</parameter> tags to JSON. Without final, it
// returns only the complete parameters, with no closing brace.
func argumentsJSON(raw string, schema *ojson.Object, final bool) string {
	var b strings.Builder
	b.WriteByte('{')
	n := 0
	rest := raw
	for {
		i := strings.Index(rest, paramStart)
		if i < 0 {
			break
		}
		tail := rest[i+len(paramStart):]
		gt := strings.IndexByte(tail, '>')
		if gt < 0 {
			break
		}
		name := strings.TrimSpace(tail[:gt])
		body := tail[gt+1:]
		end := strings.Index(body, paramEnd)
		nextParam := strings.Index(body, paramStart)
		var value string
		switch {
		case end >= 0 && (nextParam < 0 || end < nextParam):
			value = body[:end]
			rest = body[end+len(paramEnd):]
		case nextParam >= 0:
			value = body[:nextParam]
			rest = body[nextParam:]
		case final:
			value = body
			rest = ""
		default:
			goto done
		}
		if n > 0 {
			b.WriteString(", ")
		}
		ojson.EncodeString(&b, name)
		b.WriteString(": ")
		b.WriteString(encodeArg(trimWrappingNewlines(value), schema, name))
		n++
	}
done:
	if final {
		b.WriteByte('}')
	}
	return b.String()
}

func trimWrappingNewlines(v string) string {
	v = strings.TrimPrefix(v, "\n")
	return strings.TrimSuffix(v, "\n")
}

// encodeArg types a raw value from its JSON Schema, in vLLM's coercion priority order.
func encodeArg(value string, schema *ojson.Object, name string) string {
	str := func() string {
		var b strings.Builder
		ojson.EncodeString(&b, value)
		return b.String()
	}
	if schema == nil {
		return str()
	}
	prop := schema.Obj(name)
	if prop == nil {
		return str()
	}
	var types []string
	if v, ok := prop.Get("type"); ok {
		switch t := v.(type) {
		case string:
			types = []string{t}
		case []ojson.Value:
			for _, e := range t {
				if s, ok := e.(string); ok {
					types = append(types, s)
				}
			}
		}
	}
	if len(types) == 0 {
		return str()
	}
	has := map[string]bool{}
	for _, t := range types {
		switch t = strings.ToLower(strings.TrimSpace(t)); t {
		case "int":
			has["integer"] = true
		case "float", "double":
			has["number"] = true
		case "bool":
			has["boolean"] = true
		case "dict", "map":
			has["object"] = true
		case "list":
			has["array"] = true
		case "str", "text":
			has["string"] = true
		default:
			has[t] = true
		}
	}
	if has["null"] && strings.EqualFold(value, "null") {
		return "null"
	}
	if has["integer"] {
		if i, err := strconv.ParseInt(strings.TrimSpace(value), 10, 64); err == nil {
			return strconv.FormatInt(i, 10)
		}
	}
	if has["number"] {
		if f, err := strconv.ParseFloat(strings.TrimSpace(value), 64); err == nil && !math.IsInf(f, 0) && !math.IsNaN(f) {
			if f == math.Trunc(f) && math.Abs(f) < 1e15 {
				return strconv.FormatInt(int64(f), 10)
			}
			raw, _ := json.Marshal(f)
			return string(raw)
		}
	}
	if has["boolean"] {
		switch strings.ToLower(strings.TrimSpace(value)) {
		case "true", "1":
			return "true"
		case "false", "0":
			return "false"
		}
	}
	if has["object"] || has["array"] {
		if v, err := ojson.Parse([]byte(value)); err == nil {
			return ojson.Encode(v)
		}
	}
	if has["string"] {
		return str()
	}
	if v, err := ojson.Parse([]byte(value)); err == nil {
		return ojson.Encode(v)
	}
	return str()
}
