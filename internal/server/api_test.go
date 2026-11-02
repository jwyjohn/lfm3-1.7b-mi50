package server

import (
	"bufio"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"os"
	"strings"
	"testing"

	"lfm3-1.7b-mi50/internal/model"
	"lfm3-1.7b-mi50/internal/tokenizer"
)

// scripted replays fixed tokens through the real HTTP stack.
type scripted struct {
	tokens []int32
	last   *model.Request
	cached int
}

func (b *scripted) ResetCache() {}

func (b *scripted) Generate(req *model.Request, emit func(model.Step) bool) (*model.Result, error) {
	b.last = req
	res := &model.Result{PromptTokens: len(req.Prompt), CachedTokens: b.cached}
	for _, t := range b.tokens {
		res.Output++
		stop := !req.IgnoreEOS && req.StopTokens[t]
		if !emit(model.Step{Token: t}) {
			return res, nil
		}
		if stop {
			res.Stopped, res.StopToken = true, t
			return res, nil
		}
		if res.Output >= req.MaxTokens {
			return res, nil
		}
	}
	return res, nil
}

func post(t *testing.T, h http.Handler, path, body string) *httptest.ResponseRecorder {
	t.Helper()
	req := httptest.NewRequest("POST", path, strings.NewReader(body))
	w := httptest.NewRecorder()
	h.ServeHTTP(w, req)
	return w
}

func decodeJSON(t *testing.T, raw string) map[string]any {
	t.Helper()
	var out map[string]any
	if err := json.Unmarshal([]byte(raw), &out); err != nil {
		t.Fatalf("bad JSON %q: %v", raw, err)
	}
	return out
}

func sseEvents(t *testing.T, body string) []map[string]any {
	t.Helper()
	var out []map[string]any
	sc := bufio.NewScanner(strings.NewReader(body))
	sc.Buffer(make([]byte, 1<<20), 1<<24)
	done := false
	for sc.Scan() {
		line := sc.Text()
		if !strings.HasPrefix(line, "data: ") {
			continue
		}
		data := strings.TrimPrefix(line, "data: ")
		if data == "[DONE]" {
			done = true
			continue
		}
		out = append(out, decodeJSON(t, data))
	}
	if !done {
		t.Fatal("stream did not end with [DONE]")
	}
	return out
}

func newTestServer(t *testing.T, tok *tokenizer.Tokenizer, text string, eos bool) (*Server, *scripted) {
	t.Helper()
	ids := tok.Encode(text)
	if eos {
		ids = append(ids, model.EOS)
	}
	b := &scripted{tokens: ids}
	return New(b, tok, 131072), b
}

func TestCompletionText(t *testing.T) {
	tok := loadTok(t)
	s, b := newTestServer(t, tok, " Paris is nice.", true)
	w := post(t, s.Handler(), "/v1/completions", `{"prompt":"Hello","max_tokens":50,"temperature":0}`)
	if w.Code != 200 {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	if got := b.last.Prompt; len(got) != 2 || got[0] != model.BOS || got[1] != 35808 {
		t.Errorf("text prompt must start with BOS as vLLM does: %v", got)
	}
	r := decodeJSON(t, w.Body.String())
	c := r["choices"].([]any)[0].(map[string]any)
	if c["text"] != " Paris is nice." || c["finish_reason"] != "stop" {
		t.Errorf("choice %v", c)
	}
	u := r["usage"].(map[string]any)
	n := float64(len(b.tokens))
	if u["prompt_tokens"] != float64(2) || u["completion_tokens"] != n || u["total_tokens"] != n+2 {
		t.Errorf("usage %v", u)
	}
}

func TestCompletionDefaults(t *testing.T) {
	tok := loadTok(t)
	s, b := newTestServer(t, tok, "x", false)
	post(t, s.Handler(), "/v1/completions", `{"prompt":[1,2,3]}`)
	p := b.last.Params
	if p.Temperature != 0.1 || p.TopK != 50 || p.RepetitionPenalty != 1.1 || p.TopP != 1 {
		t.Errorf("defaults must come from generation_config.json: %+v", p)
	}
	if b.last.MaxTokens != 131072-3 {
		t.Errorf("default max_tokens %d", b.last.MaxTokens)
	}
	if !b.last.StopTokens[model.EOS] {
		t.Error("EOS must stop generation")
	}
	post(t, s.Handler(), "/v1/completions", `{"prompt":[1,2,3],"max_tokens":999999}`)
	if b.last.MaxTokens != 131072-3 {
		t.Errorf("max_tokens must clamp to the context: %d", b.last.MaxTokens)
	}
}

func TestCompletionStream(t *testing.T) {
	tok := loadTok(t)
	s, b := newTestServer(t, tok, "a b c", false)
	body := `{"prompt":[5,6],"max_tokens":3,"stream":true,"stream_options":{"include_usage":true},"return_token_ids":true,"temperature":0}`
	w := post(t, s.Handler(), "/v1/completions", body)
	if ct := w.Header().Get("Content-Type"); !strings.HasPrefix(ct, "text/event-stream") {
		t.Errorf("content type %q", ct)
	}
	evs := sseEvents(t, w.Body.String())
	var text strings.Builder
	var ids []int32
	var finish any
	var usage map[string]any
	for _, e := range evs {
		if u, ok := e["usage"].(map[string]any); ok {
			usage = u
		}
		for _, cv := range e["choices"].([]any) {
			c := cv.(map[string]any)
			text.WriteString(c["text"].(string))
			if tid, ok := c["token_ids"].([]any); ok {
				for _, v := range tid {
					ids = append(ids, int32(v.(float64)))
				}
			}
			if f := c["finish_reason"]; f != nil {
				finish = f
			}
		}
	}
	if text.String() != "a b c" || finish != "length" {
		t.Errorf("text %q finish %v", text.String(), finish)
	}
	if len(ids) != len(b.tokens) || firstDiff(ids, b.tokens) != len(ids) {
		t.Errorf("stream token ids %v, want %v", ids, b.tokens)
	}
	if usage == nil || usage["completion_tokens"] != float64(len(b.tokens)) {
		t.Errorf("usage %v", usage)
	}
}

func TestStopString(t *testing.T) {
	tok := loadTok(t)
	s, _ := newTestServer(t, tok, " 4, 5, 6, 7", true)
	w := post(t, s.Handler(), "/v1/completions", `{"prompt":[1],"temperature":0,"stop":[" 5"]}`)
	c := decodeJSON(t, w.Body.String())["choices"].([]any)[0].(map[string]any)
	if c["text"] != " 4," || c["finish_reason"] != "stop" || c["stop_reason"] != " 5" {
		t.Errorf("non-stream stop %v", c)
	}
	w = post(t, s.Handler(), "/v1/completions", `{"prompt":[1],"temperature":0,"stop":[" 5"],"include_stop_str_in_output":true}`)
	c = decodeJSON(t, w.Body.String())["choices"].([]any)[0].(map[string]any)
	if c["text"] != " 4, 5" {
		t.Errorf("include_stop_str_in_output: %v", c["text"])
	}
	w = post(t, s.Handler(), "/v1/completions", `{"prompt":[1],"temperature":0,"stop":[" 5"],"stream":true}`)
	var text strings.Builder
	for _, e := range sseEvents(t, w.Body.String()) {
		for _, cv := range e["choices"].([]any) {
			text.WriteString(cv.(map[string]any)["text"].(string))
		}
	}
	if text.String() != " 4," {
		t.Errorf("stream stop text %q", text.String())
	}
}

func TestChatReasoning(t *testing.T) {
	tok := loadTok(t)
	s, b := newTestServer(t, tok, "We add.</think>2 + 2 = 4.", true)
	w := post(t, s.Handler(), "/v1/chat/completions", `{"messages":[{"role":"user","content":"What is 2+2?"}],"temperature":0}`)
	if w.Code != 200 {
		t.Fatalf("status %d: %s", w.Code, w.Body)
	}
	if b.last.Prompt[0] != model.BOS {
		t.Error("the chat template starts with BOS")
	}
	r := decodeJSON(t, w.Body.String())
	c := r["choices"].([]any)[0].(map[string]any)
	m := c["message"].(map[string]any)
	if m["content"] != "2 + 2 = 4." || m["reasoning"] != "We add." || m["role"] != "assistant" || c["finish_reason"] != "stop" {
		t.Errorf("message %v finish %v", m, c["finish_reason"])
	}
}

func TestChatThinkingDisabled(t *testing.T) {
	tok := loadTok(t)
	s, _ := newTestServer(t, tok, "Hello!", true)
	w := post(t, s.Handler(), "/v1/chat/completions", `{"messages":[{"role":"user","content":"Say hi"}],"chat_template_kwargs":{"enable_thinking":false}}`)
	m := decodeJSON(t, w.Body.String())["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if m["content"] != "Hello!" || m["reasoning"] != nil {
		t.Errorf("message %v", m)
	}
}

const weatherTools = `[{"type":"function","function":{"name":"get_weather","parameters":{"type":"object","properties":{"city":{"type":"string"},"days":{"type":"integer"}},"required":["city"]}}}]`

func TestChatToolCall(t *testing.T) {
	tok := loadTok(t)
	out := "Use the tool.</think><tool_call>\n<function=get_weather>\n<parameter=city>\nParis\n</parameter>\n<parameter=days>\n3\n</parameter>\n</function>\n</tool_call>"
	s, _ := newTestServer(t, tok, out, true)
	body := `{"messages":[{"role":"user","content":"weather?"}],"tools":` + weatherTools + `,"temperature":0}`
	w := post(t, s.Handler(), "/v1/chat/completions", body)
	c := decodeJSON(t, w.Body.String())["choices"].([]any)[0].(map[string]any)
	m := c["message"].(map[string]any)
	if c["finish_reason"] != "tool_calls" || m["content"] != nil || m["reasoning"] != "Use the tool." {
		t.Errorf("choice %v", c)
	}
	calls := m["tool_calls"].([]any)
	f := calls[0].(map[string]any)["function"].(map[string]any)
	if len(calls) != 1 || f["name"] != "get_weather" || f["arguments"] != `{"city": "Paris", "days": 3}` {
		t.Errorf("tool calls %v", calls)
	}

	w = post(t, s.Handler(), "/v1/chat/completions", strings.Replace(body, `"temperature":0`, `"temperature":0,"stream":true`, 1))
	var reasoning, args strings.Builder
	name, finish := "", ""
	evs := sseEvents(t, w.Body.String())
	if d := evs[0]["choices"].([]any)[0].(map[string]any)["delta"].(map[string]any); d["role"] != "assistant" {
		t.Errorf("first delta %v", d)
	}
	for _, e := range evs {
		for _, cv := range e["choices"].([]any) {
			ch := cv.(map[string]any)
			d := ch["delta"].(map[string]any)
			if r, ok := d["reasoning"].(string); ok {
				reasoning.WriteString(r)
			}
			if tc, ok := d["tool_calls"].([]any); ok {
				fn := tc[0].(map[string]any)["function"].(map[string]any)
				if n, ok := fn["name"].(string); ok {
					name = n
				}
				if a, ok := fn["arguments"].(string); ok {
					args.WriteString(a)
				}
			}
			if f, ok := ch["finish_reason"].(string); ok {
				finish = f
			}
		}
	}
	if reasoning.String() != "Use the tool." || name != "get_weather" || args.String() != `{"city": "Paris", "days": 3}` || finish != "tool_calls" {
		t.Errorf("stream reasoning %q name %q args %q finish %q", reasoning.String(), name, args.String(), finish)
	}
}

func TestToolChoiceNoneLeavesText(t *testing.T) {
	tok := loadTok(t)
	s, b := newTestServer(t, tok, "x</think><tool_call>y</tool_call>", true)
	w := post(t, s.Handler(), "/v1/chat/completions", `{"messages":[{"role":"user","content":"q"}],"tools":`+weatherTools+`,"tool_choice":"none"}`)
	m := decodeJSON(t, w.Body.String())["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if m["content"] != "<tool_call>y</tool_call>" {
		t.Errorf("message %v", m)
	}
	if strings.Contains(tok.Decode(b.last.Prompt, false), "<tools>") {
		t.Error("tool_choice none must not advertise tools in the prompt")
	}
}

func TestRequestErrors(t *testing.T) {
	tok := loadTok(t)
	s, _ := newTestServer(t, tok, "x", true)
	h := s.Handler()
	cases := []struct{ path, body string }{
		{"/v1/completions", `not json`},
		{"/v1/completions", `{"max_tokens":3}`},
		{"/v1/completions", `{"prompt":[]}`},
		{"/v1/completions", `{"prompt":[1],"max_tokens":0}`},
		{"/v1/completions", `{"prompt":[1],"n":2}`},
		{"/v1/completions", `{"prompt":[999999]}`},
		{"/v1/completions", `{"prompt":[1],"response_format":{"type":"json_object"}}`},
		{"/v1/chat/completions", `{"messages":[]}`},
		{"/v1/chat/completions", `{"messages":[{"role":"user","content":"q"}],"tool_choice":"required"}`},
	}
	for _, c := range cases {
		w := post(t, h, c.path, c.body)
		if w.Code != 400 {
			t.Errorf("%s %s: status %d: %s", c.path, c.body, w.Code, w.Body)
			continue
		}
		if decodeJSON(t, w.Body.String())["error"] == nil {
			t.Errorf("%s: no error object", c.body)
		}
	}
	long := make([]string, 131073)
	for i := range long {
		long[i] = "1"
	}
	w := post(t, h, "/v1/completions", `{"prompt":[`+strings.Join(long, ",")+`]}`)
	if w.Code != 400 || !strings.Contains(w.Body.String(), "maximum context length") {
		t.Errorf("oversized prompt: %d %.200s", w.Code, w.Body)
	}
}

func TestTokenizeEndpoint(t *testing.T) {
	tok := loadTok(t)
	s, _ := newTestServer(t, tok, "x", true)
	r := decodeJSON(t, post(t, s.Handler(), "/tokenize", `{"prompt":"Hello"}`).Body.String())
	if got := r["tokens"].([]any); len(got) != 2 || got[0] != float64(model.BOS) {
		t.Errorf("tokens %v", got)
	}
	r = decodeJSON(t, post(t, s.Handler(), "/tokenize", `{"prompt":"Hello","add_special_tokens":false}`).Body.String())
	if got := r["tokens"].([]any); len(got) != 1 {
		t.Errorf("tokens %v", got)
	}
	r = decodeJSON(t, post(t, s.Handler(), "/detokenize", `{"tokens":[124894,35808]}`).Body.String())
	if r["prompt"] != "<|startoftext|>Hello" {
		t.Errorf("detokenize %v", r)
	}
}

// The reference captured from vLLM: the same reasoning and content text must come out
// when its tokens are replayed through this server's parser.
func TestReplayVLLMSample(t *testing.T) {
	tok := loadTok(t)
	raw, err := os.ReadFile("../../testdata/vllm_api_samples.json")
	if err != nil {
		t.Fatal(err)
	}
	var samples struct {
		Plain struct {
			Choices []struct {
				Message struct {
					Content   string `json:"content"`
					Reasoning string `json:"reasoning"`
				} `json:"message"`
			} `json:"choices"`
		} `json:"chat_plain"`
	}
	if err := json.Unmarshal(raw, &samples); err != nil {
		t.Fatal(err)
	}
	want := samples.Plain.Choices[0].Message
	s, _ := newTestServer(t, tok, want.Reasoning+"</think>"+want.Content, true)
	w := post(t, s.Handler(), "/v1/chat/completions", `{"messages":[{"role":"user","content":"What is 2+2? Reply briefly."}],"temperature":0}`)
	m := decodeJSON(t, w.Body.String())["choices"].([]any)[0].(map[string]any)["message"].(map[string]any)
	if m["reasoning"] != want.Reasoning || m["content"] != want.Content {
		t.Errorf("got %v, want %+v", m, want)
	}
}
