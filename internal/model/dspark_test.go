package model

import (
	"testing"
	"unsafe"
)

func TestStateWordsMatchKernel(t *testing.T) {
	// struct State in kernels/dspark.hip: 22 scalar words, then tokens[MT].
	if sTokens != 22 || dsStateWords != 22+dsMaxTokens {
		t.Fatalf("State has %d scalars and %d words", sTokens, dsStateWords)
	}
	if unsafe.Sizeof(dsGlobals{}) != 296 {
		t.Fatalf("Globals is %d bytes", unsafe.Sizeof(dsGlobals{}))
	}
	if dsLayerWords != 36 {
		t.Fatalf("Layer has %d words", dsLayerWords)
	}
	if dsHostState+dsStateWords*4 > dsHostOut {
		t.Fatal("state overlaps the tokens in the host mailbox")
	}
}

func TestTapSlot(t *testing.T) {
	for li := range Layers {
		want := -1
		for s, l := range dsTapLayers {
			if l == li {
				want = s
			}
		}
		if got := tapSlot(li); got != want {
			t.Errorf("layer %d: slot %d, want %d", li, got, want)
		}
	}
}
