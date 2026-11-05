package weights

import (
	"testing"

	"lfm3-1.7b-mi50/internal/fp16"
)

func TestQuantizeRowsInt8(t *testing.T) {
	w := make([]uint16, 8)
	for i, v := range []float32{1, -0.5, 0.25, 0, 0, 0, 0, 0} {
		w[i] = fp16.FromFloat32(v)
	}
	q, s := QuantizeRowsInt8(w, 2, 4)
	if want := float32(1.0 / 127); s[0] != want {
		t.Errorf("scale %v, want %v", s[0], want)
	}
	for i, want := range []int8{127, -64, 32, 0} {
		if q[i] != want {
			t.Errorf("q[%d] = %d, want %d", i, q[i], want)
		}
	}
	// An all-zero row keeps a finite scale and zero values.
	if want := float32(1e-8) / 127; s[1] != want {
		t.Errorf("zero row scale %v, want %v", s[1], want)
	}
	for i := 4; i < 8; i++ {
		if q[i] != 0 {
			t.Errorf("q[%d] = %d, want 0", i, q[i])
		}
	}
}

// Each row quantizes on its own, which is what lets the head crop reuse LM head rows and the
// context projection reuse the drafter's mixer rows.
func TestPackRowsAreIndependent(t *testing.T) {
	const rows, k = 3, 2048
	w := make([]uint16, rows*k)
	for i := range w {
		w[i] = fp16.FromFloat32(float32(i%251-125) * 1e-3)
	}
	all := Pack(w, rows, k)
	one := Pack(w[k:2*k], 1, k)
	for i := range k {
		if all.Biased[k+i] != one.Biased[i] {
			t.Fatalf("value %d differs when packed alone", i)
		}
	}
	sr := SRow(k)
	for i := range sr {
		if all.Scales[sr+i] != one.Scales[i] {
			t.Fatalf("scale %d differs when packed alone", i)
		}
	}
}
