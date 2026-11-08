package model

import (
	"fmt"
	"math"
	"math/rand/v2"
	"time"

	"lfm3-1.7b-mi50/internal/kfd"
	"lfm3-1.7b-mi50/internal/sample"
)

var negInf = float32(math.Inf(-1))

// Request is one generation.
type Request struct {
	Prompt     []int32
	Params     sample.Params
	Salt       string
	MaxTokens  int
	MinTokens  int
	IgnoreEOS  bool
	StopTokens map[int32]bool
	Seed       *uint64
	Logprobs   int // -1 none; n >= 0 also reports the top n
}

// Step is one generated token handed to the caller.
type Step struct {
	Token   int32
	LogProb float32
	Top     []sample.LogProb
}

// Result summarizes a generation.
type Result struct {
	PromptTokens int
	CachedTokens int
	Output       int
	Stopped      bool // an EOS or stop token ended the output
	StopToken    int32
	PrefillTime  time.Duration
	DecodeTime   time.Duration

	// DSpark rounds of this request: drafts accepted and tokens verified over them.
	Rounds, Accepted, Verified int
}

// sequence tracks the active request's blocks.
type sequence struct {
	tokens     []int32 // prompt + output so far
	hash       uint64  // chain hash of all registered blocks
	registered int     // complete blocks handed to the cache
	entries    []*cacheEntry
	private    []int32 // full blocks whose content is also cached under another block
	full       []int32 // logical -> physical
	slide      []int32
	slideEpoch []int64

	longContext bool // decode started deep in the context: fewer drafts per round
}

// Generate runs one request to completion. Single stream: callers must serialize.
// emit returning false ends generation early.
func (e *Engine) Generate(req *Request, emit func(Step) bool) (*Result, error) {
	if len(req.Prompt) == 0 {
		return nil, fmt.Errorf("empty prompt")
	}
	if len(req.Prompt)+req.MaxTokens > e.cfg.MaxContext {
		return nil, fmt.Errorf("prompt (%d) + max_tokens (%d) exceeds the context limit %d", len(req.Prompt), req.MaxTokens, e.cfg.MaxContext)
	}
	if e.cfg.Layout != LayoutBoth {
		return nil, fmt.Errorf("generation needs both weight layouts on the GPU")
	}
	res := &Result{PromptTokens: len(req.Prompt)}
	start := time.Now()
	seq := &sequence{tokens: append(make([]int32, 0, len(req.Prompt)+req.MaxTokens), req.Prompt...), hash: saltSeed(req.Salt)}
	defer e.release(seq)
	computed := e.reuse(seq, req.Prompt)
	res.CachedTokens = computed

	kernelSamples := req.Params.Greedy() && req.Logprobs < 0 && len(req.Params.LogitBias) == 0 &&
		len(req.Params.AllowedTokens) == 0 && req.Params.FrequencyPenalty == 0 && req.Params.PresencePenalty == 0
	// A stop-token mask until MinTokens needs the logits on the host, every token. With
	// IgnoreEOS nothing stops generation, so there is nothing to hold back.
	speculate := kernelSamples && (req.MinTokens == 0 || req.IgnoreEOS) && !e.cfg.NoDraft && e.cfg.Layout == LayoutBoth
	tapFirst := len(req.Prompt)
	e.ds.tapFrom = -1
	if speculate {
		tapFirst = max(computed, len(req.Prompt)-e.ds.window)
		e.ds.tapFrom = tapFirst
	}
	if err := e.prefill(seq, computed, len(req.Prompt)); err != nil {
		return nil, err
	}
	res.PrefillTime = time.Since(start)

	var rng *rand.Rand
	if req.Seed != nil {
		rng = rand.New(rand.NewPCG(*req.Seed, 0x9e3779b97f4a7c15))
	} else {
		rng = rand.New(rand.NewPCG(rand.Uint64(), rand.Uint64()))
	}
	st := sample.NewState(Vocab, req.Prompt)
	e.InitSeen(req.Prompt, req.Params.RepetitionPenalty)
	scratch := make([]float32, Vocab)

	// deliver hands one token to the caller and reports whether generation goes on.
	deliver := func(tok int32, step Step) bool {
		st.Observe(tok)
		seq.tokens = append(seq.tokens, tok)
		res.Output++
		stop := !req.IgnoreEOS && res.Output > req.MinTokens && req.StopTokens[tok]
		if !emit(step) {
			return false
		}
		if stop {
			res.Stopped, res.StopToken = true, tok
			return false
		}
		return res.Output < req.MaxTokens
	}

	next, step := e.hostSample(req, st, scratch, rng, 0)
	decodeStart := time.Now()
	draftReady := false
	for more := deliver(next, step); more; {
		// The last token is not yet in the KV cache: this step computes it and predicts the next.
		pos := len(seq.tokens) - 1
		if err := e.ensureBlock(seq, pos/BlockTokens); err != nil {
			return nil, err
		}
		if speculate {
			if !draftReady {
				draftReady = true
				seq.longContext = pos >= dsLongFrom
				if err := e.beginDraft(seq, tapFirst, len(req.Prompt)); err != nil {
					return nil, err
				}
			}
			eos := int32(-1)
			if !req.IgnoreEOS && req.StopTokens[EOS] {
				eos = EOS
			}
			out, err := e.draftLaunch(seq, req.MaxTokens-res.Output, eos, req.Params.RepetitionPenalty)
			if err != nil {
				return nil, err
			}
			res.Rounds += out.rounds
			res.Accepted += out.accepted
			res.Verified += out.verified
			delivered := 0
			for _, t := range out.tokens {
				delivered++
				if more = deliver(t, Step{Token: t}); !more {
					break
				}
			}
			if delivered == len(out.tokens) && pos+delivered >= out.boundary && out.boundary%BlockTokens == 0 {
				e.registerAt(seq, out.boundary, out.convEpoch)
			}
			continue
		}
		next = seq.tokens[pos]
		slotFull := int64(seq.full[pos/BlockTokens])*BlockTokens + int64(pos%BlockTokens)
		slotSlide := int64(seq.slide[pos/BlockTokens])*BlockTokens + int64(pos%BlockTokens)
		minActive := res.Output < req.MinTokens
		if kernelSamples && !minActive {
			var err error
			if next, err = e.DecodeStep(next, pos+1, slotFull, slotSlide, 0, SampleRound16); err != nil {
				return nil, err
			}
			step = Step{Token: next}
		} else {
			if err := e.decodeToHost(next, pos+1, slotFull, slotSlide); err != nil {
				return nil, err
			}
			next, step = e.hostSample(req, st, scratch, rng, res.Output)
		}
		if (pos+1)%BlockTokens == 0 {
			e.register(seq, pos+1)
		}
		more = deliver(next, step)
	}
	res.DecodeTime = time.Since(decodeStart)
	if e.ds.trace != nil {
		e.ds.trace.report()
	}
	return res, e.drain(kfd.ScopeAgent)
}

// hostSample draws from the host copy of the logits, as vLLM's sampler would.
func (e *Engine) hostSample(req *Request, st *sample.State, scratch []float32, rng *rand.Rand, produced int) (int32, Step) {
	lg := e.HostLogits()
	copy(scratch, lg)
	p := req.Params
	if produced < req.MinTokens {
		for t := range req.StopTokens {
			scratch[t] = negInf
		}
	}
	sample.Penalize(scratch, &p, st)
	tok := sample.Draw(scratch, &p, rng)
	step := Step{Token: tok}
	if req.Logprobs >= 0 {
		step.LogProb, step.Top = sample.LogSoftmaxTop(lg, tok, req.Logprobs)
	}
	return tok, step
}

// decodeToHost runs a decode step whose logits are sampled on the host.
func (e *Engine) decodeToHost(token int32, seqLen int, slotFull, slotSlide int64) error {
	e.writeStep(token, seqLen, slotFull, slotSlide, 0)
	e.globals[15] = SampleOff
	e.submitDecode()
	return e.finishLogits(true)
}

// reuse maps the longest usable cached prefix into the sequence and restores its conv
// state. At least one prompt token is always left to compute, because its logits are
// needed. Returns the number of reused tokens.
func (e *Engine) reuse(seq *sequence, prompt []int32) int {
	c := e.cache
	limit := (len(prompt) - 1) / BlockTokens
	var chain []*cacheEntry
	h := seq.hash
	for i := 0; i < limit; i++ {
		h = chainHash(h, prompt[i*BlockTokens:(i+1)*BlockTokens])
		ent, ok := c.byHash[h]
		if !ok {
			break
		}
		chain = append(chain, ent)
	}
	hit := 0
	for n := len(chain); n > 0; n-- {
		if !c.convValid(chain[n-1].convEpoch) {
			continue
		}
		ok := true
		// Attention at the next token sees the previous Window - 1 tokens.
		first := (n*BlockTokens - Window + 1)
		if first < 0 {
			first = 0
		}
		for b := first / BlockTokens; b < n; b++ {
			if !c.slideValid(chain[b].slideEpoch) {
				ok = false
				break
			}
		}
		if ok {
			hit = n
			break
		}
	}
	if hit == 0 {
		return 0
	}
	for i := 0; i < hit; i++ {
		ent := chain[i]
		c.pin(ent)
		seq.entries = append(seq.entries, ent)
		seq.full = append(seq.full, ent.full)
		ep := ent.slideEpoch
		s := int32(0)
		if ep >= 0 {
			s = int32(ep % c.slideBlocks)
		}
		seq.slide = append(seq.slide, s)
		seq.slideEpoch = append(seq.slideEpoch, ep)
		e.SetBlock(i, ent.full, s)
		seq.hash = ent.hash
	}
	seq.registered = hit
	e.CopyConvSlot(c.convSlotOf(chain[hit-1].convEpoch), 0)
	return hit * BlockTokens
}

// ensureBlock allocates the physical blocks of a logical block on first use.
func (e *Engine) ensureBlock(seq *sequence, logical int) error {
	for len(seq.full) <= logical {
		i := len(seq.full)
		f, ok := e.cache.allocFull()
		if !ok {
			return fmt.Errorf("KV cache exhausted at block %d", i)
		}
		s, ep := e.cache.allocSlide()
		seq.full = append(seq.full, f)
		seq.slide = append(seq.slide, s)
		seq.slideEpoch = append(seq.slideEpoch, ep)
		e.SetBlock(i, f, s)
	}
	return nil
}

// prefill computes tokens [from, to). Chunks end on block boundaries so the conv state of
// each chunk end can be checkpointed, as vLLM's "align" cache mode does.
func (e *Engine) prefill(seq *sequence, from, to int) error {
	chunk := shortChunk
	if to >= longChunkFrom {
		chunk = maxChunk
	}
	for start := from; start < to; {
		end := start + chunk
		if end > to {
			end = to
		}
		if end < to {
			end = end / BlockTokens * BlockTokens
		}
		if err := e.ensureBlock(seq, (end-1)/BlockTokens); err != nil {
			return err
		}
		// Every chunk checkpoints its last block boundary, including the prompt's tail,
		// the deepest point a following turn of the conversation can resume from.
		boundary := end / BlockTokens * BlockTokens
		var slot int32
		epoch := int64(-1)
		ckpt := 0
		if boundary > start {
			slot, epoch = e.cache.allocConv()
			ckpt = boundary - start
		}
		if err := e.PrefillChunk(seq.tokens[start:end], start, 0, start > 0, end == to, slot, ckpt); err != nil {
			return err
		}
		if ckpt > 0 {
			e.registerAt(seq, boundary, epoch)
		}
		start = end
	}
	return nil
}

// register hands blocks to the cache during decode, when the running conv state is
// exactly the state at the boundary `upto`.
func (e *Engine) register(seq *sequence, upto int) {
	slot, ep := e.cache.allocConv()
	e.CopyConvSlot(0, slot)
	e.registerAt(seq, upto, ep)
}

// registerAt hands every newly complete block up to the boundary `upto` to the cache;
// the last of them owns the conv checkpoint convEpoch.
func (e *Engine) registerAt(seq *sequence, upto int, convEpoch int64) {
	c := e.cache
	n := upto / BlockTokens
	for i := seq.registered; i < n; i++ {
		seq.hash = chainHash(seq.hash, seq.tokens[i*BlockTokens:(i+1)*BlockTokens])
		ent, exists := c.byHash[seq.hash]
		if exists {
			// The same block is cached with other physical pages. Keep using ours for
			// this sequence; the fresher sliding epoch serves later hits better.
			c.pin(ent)
			if ent.full != seq.full[i] {
				seq.private = append(seq.private, seq.full[i])
			}
			ent.slideEpoch = seq.slideEpoch[i]
		} else {
			ent = &cacheEntry{hash: seq.hash, full: seq.full[i], slideEpoch: seq.slideEpoch[i], convEpoch: -1, refs: 1}
			c.byHash[seq.hash] = ent
		}
		seq.entries = append(seq.entries, ent)
		if i == n-1 {
			ent.convEpoch = convEpoch
		}
	}
	if n > seq.registered {
		seq.registered = n
	}
}

// release returns the sequence's blocks to the cache; an incomplete tail block and
// blocks duplicated elsewhere go back to the free list.
func (e *Engine) release(seq *sequence) {
	c := e.cache
	for _, ent := range seq.entries {
		c.unpin(ent)
	}
	c.freeFull = append(c.freeFull, seq.private...)
	for i := seq.registered; i < len(seq.full); i++ {
		c.freeFull = append(c.freeFull, seq.full[i])
	}
}

// ResetCache forgets every cached prefix.
func (e *Engine) ResetCache() { e.cache.reset() }
