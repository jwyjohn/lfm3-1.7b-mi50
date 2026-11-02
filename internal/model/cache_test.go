package model

import "testing"

func tokens(seed int32) []int32 {
	t := make([]int32, BlockTokens)
	for i := range t {
		t[i] = seed + int32(i)
	}
	return t
}

func TestChainHashDependsOnHistoryAndSalt(t *testing.T) {
	a := chainHash(saltSeed("x"), tokens(1))
	if a != chainHash(saltSeed("x"), tokens(1)) {
		t.Error("not deterministic")
	}
	if a == chainHash(saltSeed("y"), tokens(1)) {
		t.Error("the salt must isolate caches")
	}
	if chainHash(a, tokens(2)) == chainHash(chainHash(saltSeed("x"), tokens(9)), tokens(2)) {
		t.Error("a block's identity must include its prefix")
	}
}

func TestFullBlockLRU(t *testing.T) {
	c := newPrefixCache(2, 8, 4)
	f0, _ := c.allocFull()
	f1, _ := c.allocFull()
	if _, ok := c.allocFull(); ok {
		t.Fatal("pool has only two blocks")
	}
	e0 := &cacheEntry{hash: 10, full: f0, refs: 1}
	e1 := &cacheEntry{hash: 11, full: f1, refs: 1}
	c.byHash[10], c.byHash[11] = e0, e1
	if _, ok := c.allocFull(); ok {
		t.Fatal("pinned blocks must not be evicted")
	}
	c.unpin(e0)
	c.unpin(e1)
	c.pin(e0)
	c.unpin(e0) // e0 is now the most recently used
	got, ok := c.allocFull()
	if !ok || got != f1 {
		t.Errorf("evicted %d, want the coldest block %d", got, f1)
	}
	if _, still := c.byHash[11]; still {
		t.Error("an evicted block must leave the index")
	}
	if _, still := c.byHash[10]; !still {
		t.Error("the warm block must stay")
	}
}

func TestSlideRingValidity(t *testing.T) {
	c := newPrefixCache(1, 300, 4)
	b, ep := c.allocSlide()
	if b != 0 || ep != 0 {
		t.Fatalf("first block %d epoch %d", b, ep)
	}
	if !c.slideValid(ep) {
		t.Error("a fresh block is valid")
	}
	for i := 0; i < 300-slideReuseMargin-2; i++ {
		c.allocSlide()
	}
	if !c.slideValid(ep) {
		t.Error("still outside the reuse margin")
	}
	c.allocSlide()
	if c.slideValid(ep) {
		t.Error("a block about to be overwritten must not be reused")
	}
	if c.slideValid(-1) {
		t.Error("no block is never valid")
	}
}

func TestConvRingSkipsRunningSlot(t *testing.T) {
	c := newPrefixCache(1, 300, 4)
	seen := map[int32]bool{}
	for i := 0; i < 9; i++ {
		s, ep := c.allocConv()
		if s == 0 {
			t.Fatal("slot 0 holds the running state")
		}
		seen[s] = true
		if c.convSlotOf(ep) != s {
			t.Errorf("slot of epoch %d = %d, want %d", ep, c.convSlotOf(ep), s)
		}
	}
	if len(seen) != 3 {
		t.Errorf("used %d slots, want 3", len(seen))
	}
	_, first := c.allocConv()
	for i := 0; i < 3; i++ {
		c.allocConv()
	}
	if c.convValid(first) {
		t.Error("an overwritten checkpoint must not be valid")
	}
}

func TestRopeTableShapeConstants(t *testing.T) {
	if nFullLayers+nSlideLayers+nConvLayers != Layers {
		t.Error("layer counts disagree")
	}
	counts := map[int]int{}
	for _, k := range LayerKinds {
		counts[k]++
	}
	if counts[0] != nConvLayers || counts[1] != nFullLayers || counts[2] != nSlideLayers {
		t.Errorf("layer kinds %v", counts)
	}
}
