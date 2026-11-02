package model

import (
	"encoding/binary"
	"hash/fnv"
)

// Prefix cache. Full-attention KV blocks are kept per 32-token block and evicted LRU.
// Sliding-attention blocks and conv checkpoints live in FIFO rings: a cache hit only
// needs the most recent window of them, so old entries can simply age out.

const (
	// Keeps a reused sliding block out of the ring's overwrite path for as long as it
	// can stay inside the attention window of the sequence reusing it.
	slideReuseMargin = 128
)

type cacheEntry struct {
	hash       uint64
	full       int32
	slideEpoch int64 // ring position of its sliding block, -1 if none
	convEpoch  int64 // ring position of its conv checkpoint, -1 if none
	refs       int
	prev, next *cacheEntry
}

type prefixCache struct {
	byHash   map[uint64]*cacheEntry
	freeFull []int32
	head     *cacheEntry // most recently used, unreferenced
	tail     *cacheEntry // least recently used, unreferenced

	slideBlocks int64
	slideNext   int64
	convSlots   int64 // checkpoint slots, excluding the running slot
	convNext    int64
}

func newPrefixCache(fullBlocks, slideBlocks, convSlots int) *prefixCache {
	c := &prefixCache{byHash: map[uint64]*cacheEntry{}, slideBlocks: int64(slideBlocks), convSlots: int64(convSlots - 1)}
	c.freeFull = make([]int32, 0, fullBlocks)
	for i := fullBlocks - 1; i >= 0; i-- {
		c.freeFull = append(c.freeFull, int32(i))
	}
	return c
}

func (c *prefixCache) lruRemove(e *cacheEntry) {
	if e.prev != nil {
		e.prev.next = e.next
	} else if c.head == e {
		c.head = e.next
	}
	if e.next != nil {
		e.next.prev = e.prev
	} else if c.tail == e {
		c.tail = e.prev
	}
	e.prev, e.next = nil, nil
}

func (c *prefixCache) lruPush(e *cacheEntry) {
	e.prev, e.next = nil, c.head
	if c.head != nil {
		c.head.prev = e
	}
	c.head = e
	if c.tail == nil {
		c.tail = e
	}
}

// allocFull returns a free full-attention block, evicting the coldest cached one.
func (c *prefixCache) allocFull() (int32, bool) {
	if n := len(c.freeFull); n > 0 {
		b := c.freeFull[n-1]
		c.freeFull = c.freeFull[:n-1]
		return b, true
	}
	v := c.tail
	if v == nil {
		return 0, false
	}
	c.lruRemove(v)
	delete(c.byHash, v.hash)
	return v.full, true
}

func (c *prefixCache) pin(e *cacheEntry) {
	if e.refs == 0 {
		c.lruRemove(e)
	}
	e.refs++
}

func (c *prefixCache) unpin(e *cacheEntry) {
	e.refs--
	if e.refs == 0 {
		c.lruPush(e)
	}
}

// allocSlide returns the next ring block, and the epoch recording it.
func (c *prefixCache) allocSlide() (int32, int64) {
	ep := c.slideNext
	c.slideNext++
	return int32(ep % c.slideBlocks), ep
}

func (c *prefixCache) slideValid(ep int64) bool {
	return ep >= 0 && c.slideNext-ep < c.slideBlocks-slideReuseMargin
}

// allocConv returns the next checkpoint slot (slot 0 is the running state).
func (c *prefixCache) allocConv() (int32, int64) {
	ep := c.convNext
	c.convNext++
	return int32(ep%c.convSlots) + 1, ep
}

func (c *prefixCache) convValid(ep int64) bool {
	return ep >= 0 && c.convNext-ep < c.convSlots
}

func (c *prefixCache) convSlotOf(ep int64) int32 { return int32(ep%c.convSlots) + 1 }

func (c *prefixCache) reset() {
	for _, e := range c.byHash {
		if e.refs == 0 {
			c.freeFull = append(c.freeFull, e.full)
		}
	}
	c.byHash = map[uint64]*cacheEntry{}
	c.head, c.tail = nil, nil
}

// Chain hashing: a block's identity includes the salt and every token before it.
func saltSeed(salt string) uint64 {
	h := fnv.New64a()
	h.Write([]byte("prefix-v1\x00"))
	h.Write([]byte(salt))
	return h.Sum64()
}

func chainHash(prev uint64, tokens []int32) uint64 {
	var buf [8 + 4*BlockTokens]byte
	binary.LittleEndian.PutUint64(buf[:], prev)
	for i, t := range tokens {
		binary.LittleEndian.PutUint32(buf[8+4*i:], uint32(t))
	}
	h := fnv.New64a()
	h.Write(buf[:])
	return h.Sum64()
}
