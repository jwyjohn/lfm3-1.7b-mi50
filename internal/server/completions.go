package server

import (
	"errors"
	"fmt"
	"math"
	"net/http"
	"strings"
	"time"

	"lfm3-1.7b-mi50/internal/model"
	"lfm3-1.7b-mi50/internal/ojson"
)

// encodeText tokenizes a text prompt; like the reference tokenizer it starts with BOS
// unless add_special_tokens is false.
func (s *Server) encodeText(o *ojson.Object, text string) []int32 {
	ids := s.Tok.Encode(text)
	if boolOf(o, "add_special_tokens", true) {
		ids = append([]int32{model.BOS}, ids...)
	}
	return ids
}

func (s *Server) promptTokens(o *ojson.Object) ([]int32, error) {
	v, ok := o.Get("prompt")
	if !ok {
		return nil, errors.New("prompt is required")
	}
	switch x := v.(type) {
	case string:
		return s.encodeText(o, x), nil
	case []ojson.Value:
		if len(x) == 0 {
			return nil, errors.New("the prompt is empty")
		}
		if ids, ok := tokensOf(x); ok {
			return ids, nil
		}
		if len(x) != 1 {
			return nil, errors.New("only one prompt per request is supported")
		}
		if t, isS := x[0].(string); isS {
			return s.encodeText(o, t), nil
		}
		if ids, ok := tokensOf(x[0]); ok {
			return ids, nil
		}
	}
	return nil, errors.New("prompt must be a string or a list of token ids")
}

type completionLogprobs struct {
	TextOffset    []int                `json:"text_offset"`
	TokenLogprobs []*float64           `json:"token_logprobs"`
	Tokens        []string             `json:"tokens"`
	TopLogprobs   []map[string]float64 `json:"top_logprobs"`
}

func safeFloat(f float32) float64 {
	v := float64(f)
	if math.IsInf(v, -1) || math.IsNaN(v) {
		return -9999
	}
	return v
}

func (s *Server) addCompletionLogprob(l *completionLogprobs, st model.Step, offset int, g *genOptions) {
	v := safeFloat(st.LogProb)
	l.Tokens = append(l.Tokens, s.tokenName(st.Token, g))
	l.TokenLogprobs = append(l.TokenLogprobs, &v)
	l.TextOffset = append(l.TextOffset, offset)
	top := map[string]float64{}
	for _, e := range st.Top {
		top[s.tokenName(e.Token, g)] = safeFloat(e.Value)
	}
	l.TopLogprobs = append(l.TopLogprobs, top)
}

func (s *Server) completions(w http.ResponseWriter, r *http.Request) {
	o, err := readBody(w, r)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	prompt, err := s.promptTokens(o)
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	for _, id := range prompt {
		if id < 0 || id >= model.Vocab {
			writeError(w, 400, fmt.Sprintf("token id %d is out of vocabulary", id))
			return
		}
	}
	g, err := s.parseOptions(o, len(prompt))
	if err != nil {
		writeError(w, 400, err.Error())
		return
	}
	if v, ok := intOf(o, "logprobs"); ok {
		g.logprobs = v
	}
	id := newID("cmpl-")
	created := time.Now().Unix()
	base := func(choice map[string]any) map[string]any {
		return map[string]any{"id": id, "object": "text_completion", "created": created, "model": ModelName,
			"choices": []any{choice}}
	}
	j := s.submit(r, g.request(prompt))
	track := newTextTracker(s.Tok, g)

	if !g.stream {
		var text strings.Builder
		var ids []int32
		var lp completionLogprobs
		for st := range j.steps {
			ids = append(ids, st.Token)
			t := track.push(st.Token)
			if g.logprobs >= 0 {
				s.addCompletionLogprob(&lp, st, text.Len(), g)
			}
			text.WriteString(t)
			if track.hit != "" {
				j.cancel.Store(true)
			}
		}
		text.WriteString(track.flush())
		result := <-j.done
		if result.err != nil {
			writeError(w, 500, result.err.Error())
			return
		}
		reason, stopReason := finishReason(result.res, track.hit != "")
		if track.hit != "" {
			stopReason = track.hit
		}
		choice := map[string]any{"index": 0, "text": text.String(), "finish_reason": reason, "stop_reason": stopReason, "logprobs": nil}
		if g.logprobs >= 0 {
			choice["logprobs"] = lp
		}
		if g.tokenIDs {
			choice["token_ids"] = ids
			choice["prompt_token_ids"] = prompt
		}
		resp := base(choice)
		resp["usage"] = usageOf(len(prompt), len(ids), result.res.CachedTokens)
		writeJSON(w, 200, resp)
		return
	}

	f := sseHeaders(w)
	n, offset := 0, 0
	for st := range j.steps {
		n++
		t := track.push(st.Token)
		if track.hit != "" {
			j.cancel.Store(true)
		}
		// A token ending the output carries no text of its own; it still counts.
		choice := map[string]any{"index": 0, "text": t, "logprobs": nil, "finish_reason": nil, "stop_reason": nil}
		if g.tokenIDs {
			choice["token_ids"] = []int32{st.Token}
		}
		if g.logprobs >= 0 {
			var lp completionLogprobs
			s.addCompletionLogprob(&lp, st, offset, g)
			choice["logprobs"] = lp
		}
		offset += len(t)
		chunk := base(choice)
		if g.continuous {
			chunk["usage"] = usageOf(len(prompt), n, 0)
		}
		if err := sendSSE(w, f, chunk); err != nil {
			j.cancel.Store(true)
		}
	}
	rest := track.flush()
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
	sendSSE(w, f, base(map[string]any{"index": 0, "text": rest, "logprobs": nil, "finish_reason": reason, "stop_reason": stopReason}))
	if g.includeUsage {
		chunk := map[string]any{"id": id, "object": "text_completion", "created": created, "model": ModelName,
			"choices": []any{}, "usage": usageOf(len(prompt), n, result.res.CachedTokens)}
		sendSSE(w, f, chunk)
	}
	fmt.Fprint(w, "data: [DONE]\n\n")
	if f != nil {
		f.Flush()
	}
}
