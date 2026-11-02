// Package sample turns logits into a token, in the order vLLM's sampler applies its
// steps: logit bias and min tokens, penalties, then temperature, min-p, top-k and
// top-p, then a categorical draw.
package sample

import (
	"math"
	"math/rand/v2"
	"sort"
)

// Params are the per-request sampling settings.
type Params struct {
	Temperature       float32
	TopP              float32
	TopK              int
	MinP              float32
	RepetitionPenalty float32
	FrequencyPenalty  float32
	PresencePenalty   float32
	LogitBias         map[int32]float32
	AllowedTokens     map[int32]bool
}

// GreedyEps is the temperature below which sampling is an argmax.
const GreedyEps = 1e-5

// Greedy reports whether the draw is deterministic.
func (p *Params) Greedy() bool { return p.Temperature < GreedyEps }

// State is the token history the penalties look at.
type State struct {
	Seen   []uint64 // tokens of the prompt and the output
	Counts map[int32]int
}

// NewState marks the prompt tokens.
func NewState(vocab int, prompt []int32) *State {
	s := &State{Seen: make([]uint64, (vocab+63)/64), Counts: map[int32]int{}}
	for _, t := range prompt {
		s.Mark(t)
	}
	return s
}

// Mark records a token as present without counting it as output.
func (s *State) Mark(t int32) {
	if t >= 0 && int(t)>>6 < len(s.Seen) {
		s.Seen[t>>6] |= 1 << (uint(t) & 63)
	}
}

// Observe records an output token.
func (s *State) Observe(t int32) {
	s.Mark(t)
	s.Counts[t]++
}

// Penalize applies logit bias, the allow-list and the penalties in place.
func Penalize(lg []float32, p *Params, st *State) {
	if len(p.AllowedTokens) > 0 {
		for i := range lg {
			if !p.AllowedTokens[int32(i)] {
				lg[i] = float32(math.Inf(-1))
			}
		}
	}
	for t, b := range p.LogitBias {
		if int(t) < len(lg) && t >= 0 {
			lg[t] += b
		}
	}
	if rep := p.RepetitionPenalty; rep != 1 && rep != 0 {
		for w, bits := range st.Seen {
			for bits != 0 {
				b := trailingZeros(bits)
				bits &= bits - 1
				i := w*64 + b
				if i >= len(lg) {
					continue
				}
				if lg[i] > 0 {
					lg[i] /= rep
				} else {
					lg[i] *= rep
				}
			}
		}
	}
	if p.FrequencyPenalty != 0 || p.PresencePenalty != 0 {
		for t, n := range st.Counts {
			if int(t) < len(lg) && t >= 0 {
				lg[t] -= p.FrequencyPenalty*float32(n) + p.PresencePenalty
			}
		}
	}
}

func trailingZeros(x uint64) int {
	n := 0
	for x&1 == 0 {
		x >>= 1
		n++
	}
	return n
}

// Argmax returns the first maximum, as torch.argmax does.
func Argmax(lg []float32) int32 {
	best, bv := 0, float32(math.Inf(-1))
	for i, v := range lg {
		if v > bv {
			best, bv = i, v
		}
	}
	return int32(best)
}

type cand struct {
	v float32
	i int32
}

// topCandidates returns the k largest logits, plus any ties with the k-th, in
// descending order. A scan with a small buffer costs far less than sorting the vocab.
func topCandidates(lg []float32, k int) []cand {
	if k <= 0 || k >= len(lg) {
		out := make([]cand, 0, len(lg))
		for i, v := range lg {
			if !math.IsInf(float64(v), -1) {
				out = append(out, cand{v, int32(i)})
			}
		}
		sort.Slice(out, func(a, b int) bool {
			if out[a].v != out[b].v {
				return out[a].v > out[b].v
			}
			return out[a].i < out[b].i
		})
		return out
	}
	limit := 4*k + 64
	buf := make([]cand, 0, limit)
	floor := float32(math.Inf(-1))
	compact := func() {
		sort.Slice(buf, func(a, b int) bool {
			if buf[a].v != buf[b].v {
				return buf[a].v > buf[b].v
			}
			return buf[a].i < buf[b].i
		})
		if len(buf) > k {
			kth := buf[k-1].v
			n := k
			for n < len(buf) && buf[n].v == kth {
				n++
			}
			buf = buf[:n]
			floor = kth
		}
	}
	for i, v := range lg {
		if v < floor || math.IsInf(float64(v), -1) {
			continue
		}
		buf = append(buf, cand{v, int32(i)})
		if len(buf) == limit {
			compact()
		}
	}
	compact()
	return buf
}

// Draw selects the next token from penalized logits.
func Draw(lg []float32, p *Params, rng *rand.Rand) int32 {
	if p.Greedy() {
		return Argmax(lg)
	}
	cands := topCandidates(lg, p.TopK)
	if len(cands) == 0 {
		return Argmax(lg)
	}
	temp := float64(p.Temperature)
	top := float64(cands[0].v) / temp
	probs := make([]float64, len(cands))
	var sum float64
	for i, c := range cands {
		probs[i] = math.Exp(float64(c.v)/temp - top)
		sum += probs[i]
	}
	if p.MinP > 0 {
		// The threshold is relative to the most likely token and can never remove it.
		keep := float64(p.MinP) * probs[0]
		sum = 0
		for i := range probs {
			if probs[i] < keep {
				probs[i] = 0
			}
			sum += probs[i]
		}
	}
	if p.TopP < 1 && p.TopP > 0 {
		// Drops the low-probability tail whose total mass is at most 1 - p; the most
		// likely token always survives.
		limit := (1 - float64(p.TopP)) * sum
		var tail float64
		for i := len(probs) - 1; i > 0; i-- {
			tail += probs[i]
			if tail > limit {
				break
			}
			probs[i] = 0
		}
		sum = 0
		for _, q := range probs {
			sum += q
		}
	}
	r := rng.Float64() * sum
	var acc float64
	last := 0
	for i, q := range probs {
		if q == 0 {
			continue
		}
		acc += q
		last = i
		if r < acc {
			return cands[i].i
		}
	}
	return cands[last].i
}

// LogProb is one entry of a logprobs response.
type LogProb struct {
	Token int32
	Value float32
}

// LogSoftmaxTop returns the log-probability of chosen and the n most likely tokens.
func LogSoftmaxTop(raw []float32, chosen int32, n int) (float32, []LogProb) {
	maxv := float64(math.Inf(-1))
	for _, v := range raw {
		if float64(v) > maxv {
			maxv = float64(v)
		}
	}
	var sum float64
	for _, v := range raw {
		sum += math.Exp(float64(v) - maxv)
	}
	norm := maxv + math.Log(sum)
	var top []LogProb
	if n > 0 {
		cands := topCandidates(raw, n)
		if len(cands) > n {
			cands = cands[:n]
		}
		top = make([]LogProb, len(cands))
		for i, c := range cands {
			top[i] = LogProb{c.i, float32(float64(c.v) - norm)}
		}
	}
	return float32(float64(raw[chosen]) - norm), top
}
