package model

import (
	"fmt"
	"log"
	"os"
	"sort"
	"strconv"
	"strings"
	"unsafe"

	"lfm3-1.7b-mi50/internal/kfd"
)

// Development aid: LFM3_DS_TRACE=<block> records the DSpark kernel's phase marks of that block
// (kernels/dspark.hip trace_mark) and logs the time per phase and round at the end of each request.
// Launches then run at most 4 rounds, which the kernel's 4090 marks per launch hold whole; the marks
// go to VRAM (a mark stored across PCIe would hold up every barrier that follows it).

var dsTraceNames = map[int]string{1: "start", 2: "ctx_fc", 3: "ctx_kv", 4: "ctx_ring", 5: "norm", 6: "mix", 7: "conv", 8: "attn",
	9: "comb", 10: "out", 11: "w13", 12: "w2", 13: "dhead", 14: "markov", 15: "vhead", 16: "commit", 17: "wait", 18: "post", 19: "tail", 22: "query"}

type dsTracer struct {
	buf    *kfd.Buffer // host copy
	vram   *kfd.Buffer // the kernel's marks: VRAM, so a mark is not a store across PCIe
	blk    int
	ticks  map[string]int64
	rounds int
}

const dsTraceDwords = 8194 * 2

// arm resets the marks before a launch; fetch queues their copy to the host after it.
func (e *Engine) armTrace(t *dsTracer) {
	e.upload(t.vram.VA, []uint32{0, 0, uint32(t.blk), 0})
}
func (e *Engine) fetchTrace(t *dsTracer) {
	e.uploadRaw(t.vram.VA, t.buf.VA, dsTraceDwords)
}

func (e *Engine) initDraftTrace() error {
	v := os.Getenv("LFM3_DS_TRACE")
	if v == "" {
		return nil
	}
	blk, err := strconv.Atoi(v)
	if err != nil {
		return fmt.Errorf("LFM3_DS_TRACE: %w", err)
	}
	t := &dsTracer{ticks: map[string]int64{}, blk: blk}
	if t.buf, err = e.dev.Alloc(dsTraceDwords*4+64, kfd.GTT); err != nil {
		return err
	}
	if t.vram, err = e.dev.Alloc(dsTraceDwords*4+64, kfd.VRAM); err != nil {
		return err
	}
	e.ds.trace = t
	e.ds.globals.trace = t.vram.VA
	return nil
}

func dsTagName(tag int64) string {
	base, sec := int(tag), "T"
	if base >= 100 {
		base, sec = base-100, "D"
	}
	n, ok := dsTraceNames[base]
	if !ok {
		n = strconv.Itoa(base)
	}
	return sec + ":" + n
}

// collect adds one launch's marks. A mark's time goes to its own tag; a wait (17) to the phase it follows.
func (t *dsTracer) collect(rounds int) {
	w := unsafe.Slice((*int64)(t.buf.Ptr()), 8194)
	n := int(w[0])
	prev := ""
	for i := 2; i <= n && i+1 < len(w); i += 2 {
		tag := w[i]
		name := dsTagName(tag)
		if tag == 17 && prev != "" {
			name = "wait<" + prev
		}
		if i > 2 {
			t.ticks[name] += w[i+1] - w[i-1]
		}
		if tag != 17 {
			prev = dsTagName(tag)
		}
	}
	t.rounds += rounds
}

func (t *dsTracer) report() {
	if t.rounds == 0 {
		return
	}
	type kv struct {
		k string
		v int64
	}
	var all []kv
	var tot int64
	for k, v := range t.ticks {
		all = append(all, kv{k, v})
		tot += v
	}
	sort.Slice(all, func(i, j int) bool { return all[i].v > all[j].v })
	var b strings.Builder
	us := func(v int64) float64 { return float64(v) / memrealtimeHz * 1e6 / float64(t.rounds) }
	fmt.Fprintf(&b, "trace: %d rounds, %.1f us/round:", t.rounds, us(tot))
	for _, x := range all {
		fmt.Fprintf(&b, " %s=%.1f", x.k, us(x.v))
	}
	log.Print(b.String())
	t.ticks = map[string]int64{}
	t.rounds = 0
}
