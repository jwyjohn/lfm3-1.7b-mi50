package server

import (
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"

	"lfm3-1.7b-mi50/internal/chat"
	"lfm3-1.7b-mi50/internal/model"
	"lfm3-1.7b-mi50/internal/ojson"
)

// chatPrompt renders and tokenizes a chat request. The returned bool is whether the
// prompt ends inside an open <think> block.
func (s *Server) chatPrompt(o *ojson.Object) ([]int32, []ojson.Value, bool, error) {
	messages := o.Arr("messages")
	if len(messages) == 0 {
		return nil, nil, false, errors.New("messages is required")
	}
	tools := o.Arr("tools")
	tc, _ := o.Get("tool_choice")
	switch x := tc.(type) {
	case string:
		switch x {
		case "none":
			tools = nil
		case "auto":
		default:
			return nil, nil, false, fmt.Errorf("tool_choice %q is not supported by this server", x)
		}
	case *ojson.Object:
		return nil, nil, false, errors.New("a named tool_choice is not supported by this server")
	}
	opt := chat.Options{AddGenerationPrompt: boolOf(o, "add_generation_prompt", true), EnableThinking: true}
	if kw := o.Obj("chat_template_kwargs"); kw != nil {
		opt.EnableThinking = boolOf(kw, "enable_thinking", true)
		opt.PreserveThinking = boolOf(kw, "preserve_thinking", false)
	}
	normalized, err := normalizeMessages(messages)
	if err != nil {
		return nil, nil, false, err
	}
	tools = normalizeTools(tools)
	text, err := chat.Render(normalized, tools, opt)
	if err != nil {
		return nil, nil, false, err
	}
	return s.Tok.Encode(text), tools, opt.EnableThinking && opt.AddGenerationPrompt, nil
}

// normalizeTools puts each function definition in the field order vLLM's request model
// serializes: name, description, parameters (null when absent), then anything else. The
// template prints these fields, so the order is part of the prompt.
func normalizeTools(tools []ojson.Value) []ojson.Value {
	out := make([]ojson.Value, len(tools))
	for i, tv := range tools {
		t, ok := tv.(*ojson.Object)
		fn := t.Obj("function")
		if !ok || fn == nil {
			out[i] = tv
			continue
		}
		nf := ojson.NewObject()
		for _, k := range []string{"name", "description", "parameters"} {
			v, _ := fn.Get(k)
			nf.Set(k, v)
		}
		for _, k := range fn.Keys {
			if _, done := nf.Get(k); !done {
				nf.Set(k, fn.Vals[k])
			}
		}
		nt := ojson.NewObject()
		for _, k := range t.Keys {
			nt.Set(k, t.Vals[k])
		}
		nt.Set("function", nf)
		out[i] = nt
	}
	return out
}

// normalizeMessages parses tool-call arguments to objects, as vLLM does before it
// applies the template.
func normalizeMessages(messages []ojson.Value) ([]ojson.Value, error) {
	out := make([]ojson.Value, len(messages))
	for i, mv := range messages {
		m, ok := mv.(*ojson.Object)
		if !ok {
			return nil, fmt.Errorf("message %d is not an object", i)
		}
		out[i] = m
		calls := m.Arr("tool_calls")
		if len(calls) == 0 {
			continue
		}
		copied := ojson.NewObject()
		for _, k := range m.Keys {
			copied.Set(k, m.Vals[k])
		}
		fixed := make([]ojson.Value, len(calls))
		for ci, cv := range calls {
			fixed[ci] = cv
			c, _ := cv.(*ojson.Object)
			fn := c.Obj("function")
			if fn == nil {
				continue
			}
			args, _ := fn.Get("arguments")
			raw, isStr := args.(string)
			if !isStr {
				continue
			}
			parsed, err := ojson.Parse([]byte(raw))
			if err != nil {
				return nil, fmt.Errorf("message %d: tool call arguments are not valid JSON: %w", i, err)
			}
			nf := ojson.NewObject()
			for _, k := range fn.Keys {
				nf.Set(k, fn.Vals[k])
			}
			nf.Set("arguments", parsed)
			nc := ojson.NewObject()
			for _, k := range c.Keys {
				nc.Set(k, c.Vals[k])
			}
			nc.Set("function", nf)
			fixed[ci] = nc
		}
		copied.Set("tool_calls", fixed)
		out[i] = copied
	}
	return out, nil
}

func (s *Server) tokenize(w http.ResponseWriter, r *http.Request) {
	o, err := readBody(w, r)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	var ids []int32
	if _, hasMsgs := o.Get("messages"); hasMsgs {
		ids, _, _, err = s.chatPrompt(o)
	} else {
		ids, err = s.promptTokens(o)
	}
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	if ids == nil {
		ids = []int32{}
	}
	writeJSON(w, 200, map[string]any{"count": len(ids), "max_model_len": s.MaxContext, "tokens": ids, "token_strs": nil})
}

func (s *Server) detokenize(w http.ResponseWriter, r *http.Request) {
	o, err := readBody(w, r)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	v, _ := o.Get("tokens")
	ids, ok := tokensOf(v)
	if !ok {
		writeError(w, 400, "tokens must be a list of token ids")
		return
	}
	writeJSON(w, 200, map[string]any{"prompt": s.Tok.Decode(ids, false)})
}

type chatTokenLogprob struct {
	Token   string  `json:"token"`
	Logprob float64 `json:"logprob"`
	Bytes   []int   `json:"bytes"`
}

type chatContentLogprob struct {
	Token       string             `json:"token"`
	Logprob     float64            `json:"logprob"`
	Bytes       []int              `json:"bytes"`
	TopLogprobs []chatTokenLogprob `json:"top_logprobs"`
}

func (s *Server) chatLogprob(st model.Step, g *genOptions) chatContentLogprob {
	bytesOf := func(id int32) []int {
		raw := s.Tok.TokenBytes(id)
		out := make([]int, len(raw))
		for i, b := range raw {
			out[i] = int(b)
		}
		return out
	}
	e := chatContentLogprob{Token: s.tokenName(st.Token, g), Logprob: safeFloat(st.LogProb), Bytes: bytesOf(st.Token), TopLogprobs: []chatTokenLogprob{}}
	for _, t := range st.Top {
		e.TopLogprobs = append(e.TopLogprobs, chatTokenLogprob{s.tokenName(t.Token, g), safeFloat(t.Value), bytesOf(t.Token)})
	}
	return e
}

func toolCallJSON(index int, c *ToolCall, args string) map[string]any {
	fn := map[string]any{"arguments": args}
	entry := map[string]any{"index": index, "function": fn}
	if c != nil {
		fn["name"] = c.Name
		entry["id"] = c.ID
		entry["type"] = "function"
	}
	return entry
}

func (s *Server) chatCompletions(w http.ResponseWriter, r *http.Request) {
	o, err := readBody(w, r)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	prompt, tools, thinking, err := s.chatPrompt(o)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	g, err := s.parseOptions(o, len(prompt))
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	if boolOf(o, "logprobs", false) {
		g.logprobs, _ = intOf(o, "top_logprobs")
	}
	id := newID("chatcmpl-")
	created := time.Now().Unix()
	parser := NewOutputParser(thinking, tools, func() string { return newID("chatcmpl-tool-") })
	j := s.submit(r, g.request(prompt))
	track := newTextTracker(s.Tok, g)

	if !g.stream {
		var reasoning, content strings.Builder
		var ids []int32
		var lps []chatContentLogprob
		apply := func(events []Event) {
			for _, ev := range events {
				reasoning.WriteString(ev.Reasoning)
				content.WriteString(ev.Content)
			}
		}
		for st := range j.steps {
			ids = append(ids, st.Token)
			if g.logprobs >= 0 {
				lps = append(lps, s.chatLogprob(st, g))
			}
			apply(parser.Feed(track.push(st.Token)))
			if track.hit != "" {
				j.cancel.Store(true)
			}
		}
		apply(parser.Feed(track.flush()))
		apply(parser.Finish())
		result := <-j.done
		if result.err != nil {
			writeError(w, 500, result.err.Error())
			return
		}
		reason, stopReason := finishReason(result.res, track.hit != "")
		if track.hit != "" {
			stopReason = track.hit
		}
		msg := map[string]any{"role": "assistant", "content": nil, "reasoning": nil}
		if content.Len() > 0 {
			msg["content"] = content.String()
		}
		if reasoning.Len() > 0 {
			msg["reasoning"] = reasoning.String()
		}
		if len(parser.calls) > 0 {
			calls := make([]any, len(parser.calls))
			for i, c := range parser.calls {
				calls[i] = map[string]any{"id": c.ID, "type": "function", "function": map[string]any{"name": c.Name, "arguments": c.Args}}
			}
			msg["tool_calls"] = calls
			if reason == "stop" {
				reason = "tool_calls"
			}
		}
		choice := map[string]any{"index": 0, "message": msg, "logprobs": nil, "finish_reason": reason, "stop_reason": stopReason}
		if g.logprobs >= 0 {
			choice["logprobs"] = map[string]any{"content": lps}
		}
		if g.tokenIDs {
			choice["token_ids"] = ids
		}
		resp := map[string]any{"id": id, "object": "chat.completion", "created": created, "model": ModelName,
			"choices": []any{choice}, "usage": usageOf(len(prompt), len(ids), result.res.CachedTokens)}
		if g.tokenIDs {
			resp["prompt_token_ids"] = prompt
		}
		writeJSON(w, 200, resp)
		return
	}

	f := sseHeaders(w)
	chunkOf := func(choice map[string]any) map[string]any {
		return map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": ModelName,
			"choices": []any{choice}}
	}
	send := func(delta map[string]any, finish, stop any, lp any, ids []int32, n int) {
		choice := map[string]any{"index": 0, "delta": delta, "logprobs": lp, "finish_reason": finish, "stop_reason": stop}
		if g.tokenIDs && ids != nil {
			choice["token_ids"] = ids
		}
		chunk := chunkOf(choice)
		if g.continuous {
			chunk["usage"] = usageOf(len(prompt), n, 0)
		}
		if err := sendSSE(w, f, chunk); err != nil {
			j.cancel.Store(true)
		}
	}
	send(map[string]any{"role": "assistant", "content": ""}, nil, nil, nil, nil, 0)
	n := 0
	anyTool := false
	emit := func(events []Event, ids []int32, lp any) bool {
		sent := false
		for _, ev := range events {
			delta := map[string]any{}
			switch {
			case ev.Reasoning != "":
				delta["reasoning"] = ev.Reasoning
			case ev.Content != "":
				delta["content"] = ev.Content
			case ev.ToolStart != nil:
				anyTool = true
				delta["tool_calls"] = []any{toolCallJSON(ev.ToolIndex, ev.ToolStart, "")}
			case ev.ToolArgs != "":
				delta["tool_calls"] = []any{toolCallJSON(ev.ToolIndex, nil, ev.ToolArgs)}
			default:
				continue
			}
			send(delta, nil, nil, lp, ids, n)
			ids, lp = nil, nil
			sent = true
		}
		return sent
	}
	for st := range j.steps {
		n++
		text := track.push(st.Token)
		if track.hit != "" {
			j.cancel.Store(true)
		}
		var lp any
		if g.logprobs >= 0 {
			lp = map[string]any{"content": []chatContentLogprob{s.chatLogprob(st, g)}}
		}
		ids := []int32{st.Token}
		if !emit(parser.Feed(text), ids, lp) && (g.tokenIDs || lp != nil) {
			send(map[string]any{"content": ""}, nil, nil, lp, ids, n)
		}
	}
	emit(parser.Feed(track.flush()), nil, nil)
	emit(parser.Finish(), nil, nil)
	result := <-j.done
	if result.err != nil {
		sendSSE(w, f, map[string]any{"error": map[string]any{"message": result.err.Error(), "code": 500}})
		fmt.Fprint(w, "data: [DONE]\n\n")
		return
	}
	reason, stopReason := finishReason(result.res, track.hit != "")
	if track.hit != "" {
		stopReason = track.hit
	}
	if anyTool && reason == "stop" {
		reason = "tool_calls"
	}
	send(map[string]any{}, reason, stopReason, nil, nil, n)
	if g.includeUsage {
		sendSSE(w, f, map[string]any{"id": id, "object": "chat.completion.chunk", "created": created, "model": ModelName,
			"choices": []any{}, "usage": usageOf(len(prompt), n, result.res.CachedTokens)})
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if f != nil {
		f.Flush()
	}
}
