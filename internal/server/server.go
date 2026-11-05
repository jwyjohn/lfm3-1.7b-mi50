// Package server exposes the engine through the OpenAI-compatible HTTP API that vLLM
// served for this model: single stream, requests queue behind the one running.
package server

import (
	"crypto/rand"
	"encoding/hex"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"log"
	"net/http"
	"runtime"
	"strconv"
	"strings"
	"sync/atomic"
	"time"

	"lfm3-1.7b-mi50/internal/model"
	"lfm3-1.7b-mi50/internal/ojson"
	"lfm3-1.7b-mi50/internal/sample"
	"lfm3-1.7b-mi50/internal/tokenizer"
)

// ModelName is the id clients address the model by.
const ModelName = "LiquidAI/LFM3-1.7B"

const maxBodyBytes = 1 << 30

// Defaults from the model's generation_config.json, applied as vLLM does.
const (
	defaultTemperature = 0.1
	defaultTopK        = 50
	defaultRepetition  = 1.1
)

// Backend generates tokens. The GPU engine implements it; tests use a scripted one.
type Backend interface {
	Generate(req *model.Request, emit func(model.Step) bool) (*model.Result, error)
	ResetCache()
}

// Server owns the single GPU worker.
type Server struct {
	backend    Backend
	Tok        *tokenizer.Tokenizer
	MaxContext int
	jobs       chan *job
}

type job struct {
	req    *model.Request
	steps  chan model.Step
	done   chan jobResult
	cancel atomic.Bool
	reset  bool
}

type jobResult struct {
	res *model.Result
	err error
}

// New starts the GPU worker. KFD queues are not thread-bound, but pinning the worker
// keeps its polling loop on one core.
func New(e Backend, tok *tokenizer.Tokenizer, maxContext int) *Server {
	s := &Server{backend: e, Tok: tok, MaxContext: maxContext, jobs: make(chan *job, 1024)}
	go s.worker()
	return s
}

func (s *Server) worker() {
	runtime.LockOSThread()
	for j := range s.jobs {
		if j.reset {
			s.backend.ResetCache()
			j.done <- jobResult{}
			continue
		}
		res, err := s.backend.Generate(j.req, func(st model.Step) bool {
			if j.cancel.Load() {
				return false
			}
			j.steps <- st
			return true
		})
		close(j.steps)
		if err != nil {
			log.Printf("generation failed: %v", err)
		} else {
			log.Printf("prompt %d (cached %d) in %.2f s, output %d at %.1f tok/s, %d rounds, %d verified, %d drafts accepted", res.PromptTokens, res.CachedTokens,
				res.PrefillTime.Seconds(), res.Output, float64(res.Output)/max(res.DecodeTime.Seconds(), 1e-9), res.Rounds, res.Verified, res.Accepted)
		}
		j.done <- jobResult{res, err}
	}
}

// Handler returns the HTTP routes.
func (s *Server) Handler() http.Handler {
	mux := http.NewServeMux()
	mux.HandleFunc("GET /health", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("GET /ping", func(w http.ResponseWriter, _ *http.Request) { w.WriteHeader(http.StatusOK) })
	mux.HandleFunc("GET /v1/models", s.models)
	mux.HandleFunc("POST /tokenize", s.tokenize)
	mux.HandleFunc("POST /detokenize", s.detokenize)
	mux.HandleFunc("POST /reset_prefix_cache", s.resetCache)
	mux.HandleFunc("POST /v1/completions", s.completions)
	mux.HandleFunc("POST /v1/chat/completions", s.chatCompletions)
	return mux
}

func writeJSON(w http.ResponseWriter, code int, v any) {
	w.Header().Set("Content-Type", "application/json")
	w.WriteHeader(code)
	if err := json.NewEncoder(w).Encode(v); err != nil {
		log.Printf("write response: %v", err)
	}
}

func writeError(w http.ResponseWriter, code int, msg string) {
	writeJSON(w, code, map[string]any{"error": map[string]any{
		"message": msg, "type": http.StatusText(code), "param": nil, "code": code,
	}})
}

func (s *Server) models(w http.ResponseWriter, _ *http.Request) {
	writeJSON(w, 200, map[string]any{"object": "list", "data": []any{map[string]any{
		"id": ModelName, "object": "model", "created": time.Now().Unix(), "owned_by": "lfm3-1.7b-mi50",
		"root": ModelName, "parent": nil, "max_model_len": s.MaxContext,
	}}})
}

func (s *Server) resetCache(w http.ResponseWriter, _ *http.Request) {
	j := &job{reset: true, done: make(chan jobResult, 1)}
	s.jobs <- j
	<-j.done
	w.WriteHeader(http.StatusOK)
}

func readBody(w http.ResponseWriter, r *http.Request) (*ojson.Object, error) {
	raw, err := io.ReadAll(http.MaxBytesReader(w, r.Body, maxBodyBytes))
	if err != nil {
		return nil, fmt.Errorf("reading the request: %w", err)
	}
	v, err := ojson.Parse(raw)
	if err != nil {
		return nil, fmt.Errorf("invalid JSON body: %w", err)
	}
	o, ok := v.(*ojson.Object)
	if !ok {
		return nil, errors.New("request body must be a JSON object")
	}
	return o, nil
}

func numOf(o *ojson.Object, k string) (float64, bool) {
	v, ok := o.Get(k)
	if !ok {
		return 0, false
	}
	switch n := v.(type) {
	case ojson.Number:
		f, err := strconv.ParseFloat(string(n), 64)
		return f, err == nil
	}
	return 0, false
}

func intOf(o *ojson.Object, k string) (int, bool) {
	f, ok := numOf(o, k)
	return int(f), ok
}

func boolOf(o *ojson.Object, k string, def bool) bool {
	if v, ok := o.Get(k); ok {
		if b, isB := v.(bool); isB {
			return b
		}
	}
	return def
}

func tokensOf(v ojson.Value) ([]int32, bool) {
	arr, ok := v.([]ojson.Value)
	if !ok {
		return nil, false
	}
	out := make([]int32, len(arr))
	for i, e := range arr {
		n, isNum := e.(ojson.Number)
		if !isNum {
			return nil, false
		}
		t, err := strconv.ParseInt(string(n), 10, 32)
		if err != nil {
			return nil, false
		}
		out[i] = int32(t)
	}
	return out, true
}

// genOptions are the request fields shared by both generation endpoints.
type genOptions struct {
	params       sample.Params
	maxTokens    int
	minTokens    int
	ignoreEOS    bool
	seed         *uint64
	stop         []string
	stopTokens   map[int32]bool
	includeStop  bool
	skipSpecial  bool
	salt         string
	stream       bool
	includeUsage bool
	continuous   bool
	tokenIDs     bool
	idsAsNames   bool
	logprobs     int
}

func (s *Server) parseOptions(o *ojson.Object, promptLen int) (*genOptions, error) {
	for _, unsupported := range []string{"response_format", "structured_outputs", "guided_json", "guided_regex", "guided_grammar", "guided_choice"} {
		if v, ok := o.Get(unsupported); ok && v != nil {
			if rf, isObj := v.(*ojson.Object); !isObj || rf.Str("type") != "text" {
				return nil, fmt.Errorf("%s is not supported by this server", unsupported)
			}
		}
	}
	if n, ok := intOf(o, "n"); ok && n != 1 {
		return nil, errors.New("only n=1 is supported")
	}
	if promptLen == 0 {
		return nil, errors.New("the prompt is empty")
	}
	if promptLen >= s.MaxContext {
		return nil, fmt.Errorf("this model's maximum context length is %d tokens, but the prompt has %d tokens", s.MaxContext, promptLen)
	}
	g := &genOptions{logprobs: -1, skipSpecial: true, stopTokens: map[int32]bool{model.EOS: true}}
	p := &g.params
	p.Temperature, p.TopP, p.TopK, p.RepetitionPenalty = defaultTemperature, 1, defaultTopK, defaultRepetition
	if v, ok := numOf(o, "temperature"); ok {
		p.Temperature = float32(v)
	}
	if v, ok := numOf(o, "top_p"); ok {
		p.TopP = float32(v)
	}
	if v, ok := intOf(o, "top_k"); ok {
		p.TopK = v
	}
	if v, ok := numOf(o, "min_p"); ok {
		p.MinP = float32(v)
	}
	if v, ok := numOf(o, "repetition_penalty"); ok {
		p.RepetitionPenalty = float32(v)
	}
	if v, ok := numOf(o, "frequency_penalty"); ok {
		p.FrequencyPenalty = float32(v)
	}
	if v, ok := numOf(o, "presence_penalty"); ok {
		p.PresencePenalty = float32(v)
	}
	if bias := o.Obj("logit_bias"); bias != nil && len(bias.Keys) > 0 {
		p.LogitBias = map[int32]float32{}
		for _, k := range bias.Keys {
			id, err := strconv.ParseInt(k, 10, 32)
			if err != nil {
				return nil, fmt.Errorf("logit_bias key %q is not a token id", k)
			}
			f, _ := bias.Vals[k].(ojson.Number)
			b, err := strconv.ParseFloat(string(f), 64)
			if err != nil {
				return nil, fmt.Errorf("logit_bias value for %q is not a number", k)
			}
			p.LogitBias[int32(id)] = float32(b)
		}
	}
	if v, ok := o.Get("allowed_token_ids"); ok && v != nil {
		ids, valid := tokensOf(v)
		if !valid {
			return nil, errors.New("allowed_token_ids must be a list of token ids")
		}
		p.AllowedTokens = map[int32]bool{}
		for _, id := range ids {
			p.AllowedTokens[id] = true
		}
	}
	remaining := s.MaxContext - promptLen
	g.maxTokens = remaining
	for _, key := range []string{"max_completion_tokens", "max_tokens"} {
		if v, ok := intOf(o, key); ok {
			if v < 1 {
				return nil, fmt.Errorf("%s must be at least 1", key)
			}
			g.maxTokens = min(v, remaining)
			break
		}
	}
	g.minTokens, _ = intOf(o, "min_tokens")
	g.ignoreEOS = boolOf(o, "ignore_eos", false)
	g.includeStop = boolOf(o, "include_stop_str_in_output", false)
	g.skipSpecial = boolOf(o, "skip_special_tokens", true)
	g.stream = boolOf(o, "stream", false)
	g.tokenIDs = boolOf(o, "return_token_ids", false)
	g.idsAsNames = boolOf(o, "return_tokens_as_token_ids", false)
	g.salt = o.Str("cache_salt")
	if so := o.Obj("stream_options"); so != nil {
		g.includeUsage = boolOf(so, "include_usage", false)
		g.continuous = boolOf(so, "continuous_usage_stats", false)
	}
	if v, ok := numOf(o, "seed"); ok {
		u := uint64(int64(v))
		g.seed = &u
	}
	if v, ok := o.Get("stop"); ok {
		switch x := v.(type) {
		case string:
			g.stop = []string{x}
		case []ojson.Value:
			for _, e := range x {
				if t, isS := e.(string); isS {
					g.stop = append(g.stop, t)
				}
			}
		}
	}
	if v, ok := o.Get("stop_token_ids"); ok && v != nil {
		ids, valid := tokensOf(v)
		if !valid {
			return nil, errors.New("stop_token_ids must be a list of token ids")
		}
		for _, id := range ids {
			g.stopTokens[id] = true
		}
	}
	return g, nil
}

func (g *genOptions) request(prompt []int32) *model.Request {
	return &model.Request{Prompt: prompt, Params: g.params, Salt: g.salt, MaxTokens: g.maxTokens,
		MinTokens: g.minTokens, IgnoreEOS: g.ignoreEOS, StopTokens: g.stopTokens, Seed: g.seed, Logprobs: g.logprobs}
}

// textTracker applies stop strings to the detokenized stream. Text that might still
// grow into a stop string is held back.
type textTracker struct {
	dec     *tokenizer.Stream
	stops   []string
	include bool
	held    string
	hit     string
	maxStop int
}

func newTextTracker(tok *tokenizer.Tokenizer, g *genOptions) *textTracker {
	t := &textTracker{dec: tok.NewStream(g.skipSpecial), stops: g.stop, include: g.includeStop}
	for _, s := range g.stop {
		if len(s) > t.maxStop {
			t.maxStop = len(s)
		}
	}
	return t
}

// push returns text that is safe to emit.
func (t *textTracker) push(id int32) string {
	return t.feed(t.dec.Push(id))
}

func (t *textTracker) feed(s string) string {
	if t.hit != "" {
		return ""
	}
	if len(t.stops) == 0 {
		return s
	}
	t.held += s
	at, which := -1, ""
	for _, st := range t.stops {
		if st == "" {
			continue
		}
		if i := strings.Index(t.held, st); i >= 0 && (at < 0 || i < at) {
			at, which = i, st
		}
	}
	if at >= 0 {
		t.hit = which
		out := t.held[:at]
		if t.include {
			out += which
		}
		t.held = ""
		return out
	}
	keep := 0
	for _, st := range t.stops {
		for n := len(st) - 1; n > keep; n-- {
			if n <= len(t.held) && strings.HasSuffix(t.held, st[:n]) {
				keep = n
				break
			}
		}
	}
	out := t.held[:len(t.held)-keep]
	t.held = t.held[len(t.held)-keep:]
	return out
}

func (t *textTracker) flush() string {
	s := t.feed(t.dec.Flush())
	if t.hit != "" {
		return s
	}
	s += t.held
	t.held = ""
	return s
}

func newID(prefix string) string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		panic(err)
	}
	return prefix + hex.EncodeToString(b[:])
}

// submit queues a job. A generation ends early when the client goes away.
func (s *Server) submit(r *http.Request, req *model.Request) *job {
	j := &job{req: req, steps: make(chan model.Step, 256), done: make(chan jobResult, 1)}
	go func() {
		<-r.Context().Done()
		j.cancel.Store(true)
	}()
	s.jobs <- j
	return j
}

func finishReason(res *model.Result, hitStop bool) (string, any) {
	switch {
	case hitStop:
		return "stop", nil
	case res != nil && res.Stopped:
		if res.StopToken == model.EOS {
			return "stop", nil
		}
		return "stop", res.StopToken
	}
	return "length", nil
}

func usageOf(prompt, completion, cached int) map[string]any {
	return map[string]any{
		"prompt_tokens": prompt, "completion_tokens": completion, "total_tokens": prompt + completion,
		"prompt_tokens_details": map[string]any{"cached_tokens": cached},
	}
}

func sseHeaders(w http.ResponseWriter) http.Flusher {
	h := w.Header()
	h.Set("Content-Type", "text/event-stream; charset=utf-8")
	h.Set("Cache-Control", "no-cache")
	h.Set("X-Accel-Buffering", "no")
	w.WriteHeader(http.StatusOK)
	f, _ := w.(http.Flusher)
	return f
}

func sendSSE(w http.ResponseWriter, f http.Flusher, v any) error {
	raw, err := json.Marshal(v)
	if err != nil {
		return err
	}
	if _, err := fmt.Fprintf(w, "data: %s\n\n", raw); err != nil {
		return err
	}
	if f != nil {
		f.Flush()
	}
	return nil
}

func (s *Server) tokenName(id int32, g *genOptions) string {
	if g.idsAsNames {
		return fmt.Sprintf("token_id:%d", id)
	}
	return s.Tok.Decode([]int32{id}, false)
}
