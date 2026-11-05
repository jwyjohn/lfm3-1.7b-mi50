package main

import (
	"encoding/binary"
	"testing"
)

func TestTransposeLanes(t *testing.T) {
	const rows, rank = 128, 256
	src := make([]byte, rows*rank)
	for i := range src {
		src[i] = byte(i * 31)
	}
	dst := transposeLanes(src)
	for g := range 2 {
		for r := range 64 {
			for p := range 16 {
				for b := range 16 {
					from := (g*64+r)*rank + p*16 + b
					to := g*64*rank + p*64*16 + r*16 + b
					if dst[to] != src[from] {
						t.Fatalf("group %d row %d piece %d byte %d", g, r, p, b)
					}
				}
			}
		}
	}
}

func TestCropIDs(t *testing.T) {
	if len(cropIDs) != cropRows*4 {
		t.Fatalf("%d bytes", len(cropIDs))
	}
	seen := map[uint32]bool{}
	for i := range cropRows {
		id := binary.LittleEndian.Uint32(cropIDs[4*i:])
		if id >= 128000 || seen[id] {
			t.Fatalf("entry %d: token %d out of range or repeated", i, id)
		}
		seen[id] = true
	}
}
