package payload

import (
	"bytes"
	"math/rand/v2"
	"path/filepath"
	"testing"
)

func TestCompressedRoundTrip(t *testing.T) {
	rng := rand.New(rand.NewPCG(1, 2))
	// Crosses a chunk boundary and leaves a partial element at the end of a stride-2 chunk.
	raw := make([]byte, 2*chunkBytes+12345)
	for i := range raw {
		raw[i] = byte(rng.IntN(32) << 3)
	}
	stored := []byte("stored as is")
	path := filepath.Join(t.TempDir(), "payload.bin")
	w, err := Create(path)
	if err != nil {
		t.Fatal(err)
	}
	for _, step := range []func() error{
		func() error { return w.AddCompressed("bytes", raw, Entropy, 1) },
		func() error { return w.AddCompressed("planes", raw, Entropy, 2) },
		func() error { return w.AddCompressed("text", raw[:1000], Text, 1) },
		func() error { return w.AddCompressed("empty", nil, Entropy, 1) },
		func() error { return w.Add("stored", stored) },
		func() error { return w.Close() },
	} {
		if err := step(); err != nil {
			t.Fatal(err)
		}
	}
	p, err := Open(path)
	if err != nil {
		t.Fatal(err)
	}
	for name, want := range map[string][]byte{
		"bytes": raw, "planes": raw, "text": raw[:1000], "stored": stored,
	} {
		got, err := p.Data(name)
		if err != nil {
			t.Fatalf("%s: %v", name, err)
		}
		if !bytes.Equal(got, want) {
			t.Errorf("%s: decoded bytes differ", name)
		}
		size, err := p.RawSize(name)
		if err != nil || size != uint64(len(want)) {
			t.Errorf("%s: raw size %d, %v, want %d", name, size, err, len(want))
		}
	}
	// Three random bits in eight per byte: entropy coding must find them.
	if e := p.Man.Entries["bytes"]; e.Size > e.Raw*3/4 {
		t.Errorf("coded %d of %d bytes", e.Size, e.Raw)
	}
	if got, err := p.Data("empty"); err != nil || len(got) != 0 {
		t.Errorf("empty entry: %d bytes, %v", len(got), err)
	}
	if _, err := p.Data("missing"); err == nil {
		t.Error("a missing entry must fail")
	}
}

func TestPlanes(t *testing.T) {
	raw := []byte{1, 2, 3, 4, 5, 6, 7}
	planes := split(raw, 2)
	if !bytes.Equal(planes, []byte{1, 3, 5, 2, 4, 6, 7}) {
		t.Fatalf("planes %v", planes)
	}
	out := make([]byte, len(raw))
	join(out, planes, 2)
	if !bytes.Equal(out, raw) {
		t.Fatalf("joined %v", out)
	}
}
