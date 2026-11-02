package sample

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestArgmaxKeepsFirstMaximum(t *testing.T) {
	if got := Argmax([]float32{1, 3, 3, 2}); got != 1 {
		t.Errorf("got %d", got)
	}
}

func TestRepetitionPenaltySigns(t *testing.T) {
	lg := []float32{2, -2, 2, -2}
	p := &Params{RepetitionPenalty: 2}
	Penalize(lg, p, NewState(4, []int32{0, 1}))
	want := []float32{1, -4, 2, -2}
	for i := range want {
		if lg[i] != want[i] {
			t.Errorf("logit %d = %v, want %v", i, lg[i], want[i])
		}
	}
}

func TestOutputPenalties(t *testing.T) {
	st := NewState(3, nil)
	st.Observe(1)
	st.Observe(1)
	lg := []float32{5, 5, 5}
	Penalize(lg, &Params{RepetitionPenalty: 1, FrequencyPenalty: 0.5, PresencePenalty: 1}, st)
	if lg[0] != 5 || lg[1] != 3 {
		t.Errorf("got %v", lg)
	}
}

func TestLogitBiasAndAllowList(t *testing.T) {
	lg := []float32{0, 1, 2}
	Penalize(lg, &Params{RepetitionPenalty: 1, LogitBias: map[int32]float32{0: 10}}, NewState(3, nil))
	if Argmax(lg) != 0 {
		t.Errorf("bias ignored: %v", lg)
	}
	lg = []float32{0, 1, 2}
	Penalize(lg, &Params{RepetitionPenalty: 1, AllowedTokens: map[int32]bool{0: true}}, NewState(3, nil))
	if !math.IsInf(float64(lg[2]), -1) || Argmax(lg) != 0 {
		t.Errorf("allow list ignored: %v", lg)
	}
}

func rng() *rand.Rand { return rand.New(rand.NewPCG(1, 2)) }

func TestDrawGreedy(t *testing.T) {
	if got := Draw([]float32{0, 9, 1}, &Params{Temperature: 0, TopP: 1}, rng()); got != 1 {
		t.Errorf("got %d", got)
	}
}

func histogram(lg []float32, p *Params, n int) []int {
	r := rng()
	h := make([]int, len(lg))
	for i := 0; i < n; i++ {
		h[Draw(lg, p, r)]++
	}
	return h
}

func TestDrawFollowsSoftmax(t *testing.T) {
	lg := []float32{float32(math.Log(0.2)), float32(math.Log(0.3)), float32(math.Log(0.5))}
	h := histogram(lg, &Params{Temperature: 1, TopP: 1}, 200000)
	for i, want := range []float64{0.2, 0.3, 0.5} {
		if got := float64(h[i]) / 200000; math.Abs(got-want) > 0.01 {
			t.Errorf("token %d frequency %.3f, want %.3f", i, got, want)
		}
	}
}

func TestTemperatureSharpens(t *testing.T) {
	lg := []float32{0, 1}
	h := histogram(lg, &Params{Temperature: 0.1, TopP: 1}, 20000)
	if h[1] < 19900 {
		t.Errorf("low temperature should almost always pick the best: %v", h)
	}
}

func TestTopK(t *testing.T) {
	h := histogram([]float32{5, 4, 3, 2, 1, 0}, &Params{Temperature: 1, TopK: 2, TopP: 1}, 5000)
	if h[2]+h[3]+h[4]+h[5] != 0 || h[0] == 0 || h[1] == 0 {
		t.Errorf("top-k kept the wrong tokens: %v", h)
	}
}

func TestTopKKeepsTies(t *testing.T) {
	h := histogram([]float32{3, 2, 2, 1}, &Params{Temperature: 1, TopK: 2, TopP: 1}, 5000)
	if h[1] == 0 || h[2] == 0 || h[3] != 0 {
		t.Errorf("tied k-th logits must all stay: %v", h)
	}
}

func TestTopP(t *testing.T) {
	// Probabilities 0.5, 0.3, 0.15, 0.05 with p = 0.75: the tail with mass <= 0.25 goes.
	lg := []float32{float32(math.Log(0.5)), float32(math.Log(0.3)), float32(math.Log(0.15)), float32(math.Log(0.05))}
	h := histogram(lg, &Params{Temperature: 1, TopP: 0.75}, 20000)
	if h[2] != 0 || h[3] != 0 || h[0] == 0 || h[1] == 0 {
		t.Errorf("top-p kept the wrong tokens: %v", h)
	}
}

func TestTopPKeepsBestAlways(t *testing.T) {
	h := histogram([]float32{0, 10}, &Params{Temperature: 1, TopP: 0.0001}, 1000)
	if h[1] != 1000 {
		t.Errorf("got %v", h)
	}
}

func TestMinP(t *testing.T) {
	lg := []float32{float32(math.Log(1)), float32(math.Log(0.5)), float32(math.Log(0.01))}
	h := histogram(lg, &Params{Temperature: 1, TopP: 1, MinP: 0.1}, 20000)
	if h[2] != 0 || h[1] == 0 {
		t.Errorf("min-p kept the wrong tokens: %v", h)
	}
}

func TestLargeVocabularyTopK(t *testing.T) {
	lg := make([]float32, 128000)
	for i := range lg {
		lg[i] = float32(i%1000) / 1000
	}
	lg[777] = 50
	lg[12345] = 49
	got := topCandidates(lg, 2)
	if len(got) != 2 || got[0].i != 777 || got[1].i != 12345 {
		t.Errorf("got %+v", got)
	}
}

func TestLogSoftmaxTop(t *testing.T) {
	lg := []float32{float32(math.Log(0.25)), float32(math.Log(0.75))}
	v, top := LogSoftmaxTop(lg, 0, 1)
	if math.Abs(float64(v)-math.Log(0.25)) > 1e-5 {
		t.Errorf("logprob %v", v)
	}
	if len(top) != 1 || top[0].Token != 1 || math.Abs(float64(top[0].Value)-math.Log(0.75)) > 1e-5 {
		t.Errorf("top %+v", top)
	}
}
