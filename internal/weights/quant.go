// Package weights holds the int8 weight format shared by the pack tool and the runtime:
// one fp16 scale per 128 inputs, stored biased as the decode kernel expects.
package weights

import (
	"math"
	"runtime"
	"sync"

	"lfm3-1.7b-mi50/internal/fp16"
)

// Group is the number of inputs that share one scale.
const Group = 128

// Packed is a quantized matrix in the decode kernel's layout.
type Packed struct {
	Rows, K int
	Biased  []byte   // [Rows][K], q + 128
	Scales  []uint16 // Rows x SRow, see SRow
}

// SRow is the padded scale-row length: [quarter][i] with 4 x SQ entries.
func SRow(k int) int {
	if k/512 == 4 {
		return 16
	}
	return 64
}

// SQ is the padded number of scales per quarter.
func SQ(k int) int { return SRow(k) / 4 }

// Parallel runs fn over [0, n) on all CPUs.
func Parallel(n int, fn func(i int)) {
	var wg sync.WaitGroup
	workers := runtime.NumCPU()
	chunk := n/(workers*8) + 1
	var next int
	var mu sync.Mutex
	for w := 0; w < workers; w++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				mu.Lock()
				a := next
				next += chunk
				mu.Unlock()
				if a >= n {
					return
				}
				b := a + chunk
				if b > n {
					b = n
				}
				for i := a; i < b; i++ {
					fn(i)
				}
			}
		}()
	}
	wg.Wait()
}

// Pack mirrors vLLM's decode_v2_weights.pack for an fp16 [rows, k] matrix.
func Pack(w []uint16, rows, k int) *Packed {
	groups := k / Group
	ni := groups / 4
	sq := SQ(k)
	p := &Packed{Rows: rows, K: k, Biased: make([]byte, rows*k), Scales: make([]uint16, rows*4*sq)}
	Parallel(rows, func(r int) {
		src := w[r*k : (r+1)*k]
		dst := p.Biased[r*k : (r+1)*k]
		sc := p.Scales[r*4*sq : (r+1)*4*sq]
		for g := 0; g < groups; g++ {
			var m float32
			for j := 0; j < Group; j++ {
				if v := abs32(fp16.ToFloat32(src[g*Group+j])); v > m {
					m = v
				}
			}
			if m < 1e-8 {
				m = 1e-8
			}
			h := fp16.FromFloat32(m / 127)
			s := fp16.ToFloat32(h)
			for j := 0; j < Group; j++ {
				q := roundEven(fp16.ToFloat32(src[g*Group+j]) / s)
				if q > 127 {
					q = 127
				} else if q < -127 {
					q = -127
				}
				dst[g*Group+j] = byte(int32(q) + 128)
			}
			quarter, i := g%4, g/4
			if i < ni {
				sc[quarter*sq+i] = h
			}
		}
	})
	return p
}

func abs32(f float32) float32 {
	return math.Float32frombits(math.Float32bits(f) &^ (1 << 31))
}

// roundEven is torch.round on a float32: halfway cases go to the even integer.
func roundEven(f float32) float32 {
	return float32(math.RoundToEven(float64(f)))
}
