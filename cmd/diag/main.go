// diag checks the GPU path of the server, one code path at a time:
//
//	decode-determinism   the decode kernel must give bit-identical logits on every run;
//	                     on a difference it reports which kernel phase and blocks diverged
//	prefill-determinism  the same for the prefill pipeline
//	decode-timing        kernel time against wall time per token, by context length
//	queue-churn          creates a queue and exits; run it in a loop beside the determinism
//	                     modes to test that they survive preemption
//
// Each mode loads one weight layout, so it can run beside another GPU process with enough
// free VRAM: about 3.3 GB for decode-determinism, 3 GB for prefill-determinism and 4 GB for
// decode-timing. queue-churn needs almost none.
package main

import (
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"runtime"

	"lfm3-1.7b-mi50/internal/kfd"
	"lfm3-1.7b-mi50/internal/model"
	"lfm3-1.7b-mi50/internal/payload"
)

const (
	decodeBlocks = 60
	blockWaves   = 4
)

func main() {
	runtime.LockOSThread()
	payloadPath := flag.String("payload", "", "payload file or executable with an attached payload (default: this program's own path)")
	mode := flag.String("mode", "decode-determinism", "decode-determinism | prefill-determinism | decode-timing | queue-churn")
	steps := flag.Int("n", 200, "decode steps or prefill tokens")
	flag.Parse()

	path := *payloadPath
	if path == "" {
		self, err := os.Executable()
		if err != nil {
			log.Fatal(err)
		}
		path = self
	}
	if *mode == "queue-churn" {
		queueChurn()
		return
	}
	pay, err := payload.Open(path)
	if err != nil {
		log.Fatal(err)
	}
	cfg := model.Config{MaxContext: 8192, FullTokens: 8192 + 256, SlideBlocks: 256, ConvSlots: 64, Layout: model.LayoutDecodeOnly}
	switch *mode {
	case "prefill-determinism":
		cfg.Layout = model.LayoutPrefillOnly
	case "decode-timing":
		cfg.MaxContext, cfg.FullTokens = 131072, 131072+256
	}
	e, err := model.Open(cfg, pay)
	if err != nil {
		log.Fatal(err)
	}
	defer e.Close()

	switch *mode {
	case "decode-determinism":
		decodeDeterminism(e, tokenSequence(*steps))
	case "prefill-determinism":
		prefillDeterminism(e, tokenSequence(*steps))
	case "decode-timing":
		decodeTiming(e)
	default:
		log.Fatalf("unknown mode %q", *mode)
	}
}

// tokenSequence is a fixed pseudo-random token stream, so runs compare across machines.
func tokenSequence(n int) []int32 {
	out := make([]int32, n)
	x := uint32(12345)
	for i := range out {
		x = x*1664525 + 1013904223
		out[i] = int32(1000 + x>>8%50000)
	}
	return out
}

func clone(src []float32) []float32 {
	return append([]float32(nil), src...)
}

func maxDiff(a, b []float32) (float64, int) {
	worst, count := 0.0, 0
	for i := range a {
		if a[i] != b[i] {
			count++
			worst = math.Max(worst, math.Abs(float64(a[i]-b[i])))
		}
	}
	return worst, count
}

func decodeDeterminism(e *model.Engine, tokens []int32) {
	var refLogits, refWS [][]float32
	for run := 0; run < 3; run++ {
		var logits, ws [][]float32
		err := e.TraceDecode(tokens, func(int) {
			logits = append(logits, clone(e.HostLogits()))
			w := make([]float32, model.WorkspaceFloats)
			if err := e.ReadWorkspace(w); err != nil {
				log.Fatal(err)
			}
			ws = append(ws, w)
		})
		if err != nil {
			log.Fatal(err)
		}
		if run == 0 {
			refLogits, refWS = logits, ws
			continue
		}
		first := -1
		for s := range logits {
			if _, n := maxDiff(logits[s], refLogits[s]); n > 0 {
				first = s
				break
			}
		}
		if first < 0 {
			fmt.Printf("run %d: all %d steps bit-identical to run 0\n", run, len(logits))
			continue
		}
		d, n := maxDiff(logits[first], refLogits[first])
		fmt.Printf("run %d: first difference at step %d (%d logits, max %.6f)\n", run, first, n, d)
		reportPhases(ws[first], refWS[first])
	}
}

// reportPhases shows which phase outputs diverged and, for the conv input projection,
// which dispatch blocks own the channels that differ.
func reportPhases(a, b []float32) {
	for _, r := range model.WorkspaceRegions {
		d, n := maxDiff(a[r.Off:r.Off+r.Len], b[r.Off:r.Off+r.Len])
		fmt.Printf("  %-4s %5d of %5d differ, max %.5f\n", r.Name, n, r.Len, d)
		if r.Name != "bcx" {
			continue
		}
		byBlock := make([]int, decodeBlocks)
		for c := 0; c < model.H; c++ {
			for k := 0; k < 3; k++ {
				if a[r.Off+3*c+k] != b[r.Off+3*c+k] {
					byBlock[c%(decodeBlocks*blockWaves)/blockWaves]++
					break
				}
			}
		}
		fmt.Printf("       channels differing per block: %v\n", byBlock)
	}
}

func prefillDeterminism(e *model.Engine, tokens []int32) {
	var ref []float32
	for run := 0; run < 3; run++ {
		lg, err := e.PrefillLogits(tokens)
		if err != nil {
			log.Fatal(err)
		}
		if run == 0 {
			ref = clone(lg)
			continue
		}
		if d, n := maxDiff(lg, ref); n > 0 {
			fmt.Printf("run %d: %d logits differ, max %.6f\n", run, n, d)
		} else {
			fmt.Printf("run %d: logits bit-identical to run 0\n", run)
		}
	}
}

func decodeTiming(e *model.Engine) {
	for _, ctx := range []int{1024, 8192, 32768, 65536, 126976} {
		k, w, err := e.MeasureDecode(ctx, 60)
		if err != nil {
			log.Fatal(err)
		}
		fmt.Printf("context %6d: kernel %.3f ms, wall %.3f ms (%.1f tok/s), driver overhead %.3f ms\n", ctx, k, w, 1e3/w, w-k)
	}
}

// queueChurn creates a queue and exits. KFD rebuilds its runlist for both, which preempts
// every other process's running waves, so running this in a loop beside another mode tests
// that their context save and restore work.
func queueChurn() {
	d, err := kfd.Open()
	if err != nil {
		log.Fatal(err)
	}
	if _, err := d.NewQueue(); err != nil {
		log.Fatal(err)
	}
}
