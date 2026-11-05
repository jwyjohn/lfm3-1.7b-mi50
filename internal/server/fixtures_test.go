package server

import (
	"encoding/json"
	"os"
	"testing"

	"lfm3-1.7b-mi50/internal/chat"
	"lfm3-1.7b-mi50/internal/ojson"
	"lfm3-1.7b-mi50/internal/payload"
	"lfm3-1.7b-mi50/internal/tokenizer"
)

// Fixtures were captured from the vLLM deployment this program replaces. Tokenization
// and template rendering decide the model's input, so they must match exactly.
func loadTok(t *testing.T) *tokenizer.Tokenizer {
	path := os.Getenv("LFM3_PAYLOAD")
	if path == "" {
		t.Skip("LFM3_PAYLOAD not set")
	}
	p, err := payload.Open(path)
	if err != nil {
		t.Fatal(err)
	}
	raw, err := p.Data("tokenizer")
	if err != nil {
		t.Fatal(err)
	}
	tok, err := tokenizer.Load(raw)
	if err != nil {
		t.Fatal(err)
	}
	return tok
}

func firstDiff(a, b []int32) int {
	for i := range a {
		if i >= len(b) || a[i] != b[i] {
			return i
		}
	}
	return len(a)
}

func TestTokenizerFixtures(t *testing.T) {
	tok := loadTok(t)
	raw, err := os.ReadFile("../../testdata/fixtures.json")
	if err != nil {
		t.Fatal(err)
	}
	var fx struct {
		Texts []struct {
			Text   string  `json:"text"`
			Tokens []int32 `json:"tokens"`
		} `json:"texts"`
	}
	if err := json.Unmarshal(raw, &fx); err != nil {
		t.Fatal(err)
	}
	for i, c := range fx.Texts {
		got := tok.Encode(c.Text)
		if len(got) != len(c.Tokens) || firstDiff(got, c.Tokens) != len(got) {
			d := firstDiff(got, c.Tokens)
			lo := max(d-4, 0)
			t.Errorf("text %d (%q...): %d vs %d tokens, first diff at %d:\n got %v\nwant %v", i, trunc(c.Text), len(got), len(c.Tokens), d,
				got[lo:min(d+6, len(got))], c.Tokens[lo:min(d+6, len(c.Tokens))])
		}
		if back := tok.Decode(c.Tokens, false); back != c.Text {
			t.Errorf("text %d: decode round trip differs: %q", i, trunc(back))
		}
	}
}

func trunc(s string) string {
	if len(s) > 50 {
		return s[:50]
	}
	return s
}

func TestChatFixtures(t *testing.T) {
	tok := loadTok(t)
	raw, err := os.ReadFile("../../testdata/fixtures.json")
	if err != nil {
		t.Fatal(err)
	}
	root, err := ojson.Parse(raw)
	if err != nil {
		t.Fatal(err)
	}
	for i, cv := range root.(*ojson.Object).Arr("chats") {
		c := cv.(*ojson.Object)
		req := c.Obj("req")
		opt := chat.Options{AddGenerationPrompt: boolOf(req, "add_generation_prompt", true), EnableThinking: true}
		if kw := req.Obj("chat_template_kwargs"); kw != nil {
			opt.EnableThinking = boolOf(kw, "enable_thinking", true)
		}
		msgs, err := normalizeMessages(req.Arr("messages"))
		if err != nil {
			t.Fatal(err)
		}
		text, err := chat.Render(msgs, normalizeTools(req.Arr("tools")), opt)
		if err != nil {
			t.Fatalf("chat %d: %v", i, err)
		}
		if want := c.Str("text"); text != want {
			t.Errorf("chat %d rendered text differs:\n got %q\nwant %q", i, text, want)
			continue
		}
		v, _ := c.Get("tokens")
		want, _ := tokensOf(v)
		got := tok.Encode(text)
		if len(got) != len(want) || firstDiff(got, want) != len(got) {
			t.Errorf("chat %d tokens differ at %d", i, firstDiff(got, want))
		}
	}
}

func TestOutputParser(t *testing.T) {
	var tools []ojson.Value
	raw, _ := ojson.Parse([]byte(`[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}}}}}]`))
	tools = raw.([]ojson.Value)
	text := "We think.</think>Sure.<tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
	for _, split := range []int{1, 2, 3, 7, 1000} {
		id := 0
		p := NewOutputParser(true, tools, func() string { id++; return "c" })
		var reasoning, content, args string
		name := ""
		run := func(evs []Event) {
			for _, e := range evs {
				reasoning += e.Reasoning
				content += e.Content
				args += e.ToolArgs
				if e.ToolStart != nil {
					name = e.ToolStart.Name
				}
			}
		}
		for i := 0; i < len(text); i += split {
			run(p.Feed(text[i:min(i+split, len(text))]))
		}
		run(p.Finish())
		if reasoning != "We think." || content != "Sure." || name != "get_weather" || args != `{"city": "Paris", "days": 3}` {
			t.Errorf("split %d: reasoning %q content %q name %q args %q", split, reasoning, content, name, args)
		}
		if len(p.calls) != 1 || p.calls[0].Args != args {
			t.Errorf("split %d: stored call %+v", split, p.calls)
		}
	}
}
