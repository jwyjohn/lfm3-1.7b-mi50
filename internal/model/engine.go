// Package model runs LFM3-1.7B on the MI50: weights, KV cache, prefill and decode.
package model

import (
	"encoding/json"
	"fmt"
	"math"
	"runtime/debug"
	"strings"
	"sync/atomic"
	"time"
	"unsafe"

	"lfm3-1.7b-mi50/internal/fp16"
	"lfm3-1.7b-mi50/internal/kfd"
	"lfm3-1.7b-mi50/internal/payload"
	"lfm3-1.7b-mi50/internal/weights"
	"lfm3-1.7b-mi50/kernels"
)

// Model geometry (config.json of LFM3-1.7B).
const (
	H           = 2048
	I           = 7680
	QH          = 16
	KVH         = 4
	D           = 128
	Layers      = 24
	Vocab       = 128000
	Rec         = 264
	BlockTokens = 32
	KVBlock     = BlockTokens * KVH * Rec
	Window      = 256
	ConvState   = 6
	EOS         = 124900
	BOS         = 124894
)

// LayerKinds lists layer types in order: 0 conv, 1 full attention, 2 sliding attention.
var LayerKinds = [Layers]int{0, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0, 2, 0, 1, 0}

// Layer descriptor word offsets; must match struct Layer in kernels/decode.hip.
const (
	lKind = iota
	lOpNorm
	lFfnNorm
	lPostOpNorm
	lPostFfnNorm
	lWMix
	lSMix
	lWOut
	lSOut
	lW13
	lS13
	lW2
	lS2
	lConvW
	lConvState
	lCsSlot
	lCsDim
	lCsT
	lStateIndex
	lQNorm
	lKNorm
	lRope
	lRotary
	lKV
	lKVBlock
	lKVHead
	lKVSlot
	lBlockSize
	lBlockTable
	lSeqLens
	lSlotMapping
	lWindow
	layerWords
)

const (
	decodeBlocks   = 60
	decodeThreads  = 256
	workspaceBytes = 311872 * 4 // WS_FLOATS of kernels/decode.hip
	seenWords      = (Vocab + 31) / 32
	gpuTimeout     = 120 * time.Second
)

// Offsets inside the per-step control page. Host-written words and words the
// kernels hit with atomics are kept in separate cache lines.
const (
	stepInput     = 0
	stepSeqLen    = 8
	stepSlotFull  = 16
	stepSlotSlide = 24
	stepConvSlot  = 32
	stepRep       = 40
	stepBest      = 256
	stepDone      = 264
	stepBarrier   = 512
	stepPageBytes = 4096
)

// Config sizes the run-time allocations.
type Config struct {
	MaxContext   int     // longest prompt + output, tokens
	VRAMFraction float64 // share of VRAM to use; the full-attention KV pool takes what remains

	// Layout restricts which weight layout is loaded, so diagnostics of one code path fit
	// beside another GPU process. Generate needs LayoutBoth.
	Layout Layout

	// NoDraft keeps greedy requests on the one-token decode kernel.
	NoDraft bool

	// Zero values pick defaults sized for MaxContext and VRAMFraction.
	FullTokens  int // KV capacity of full-attention layers, tokens
	SlideBlocks int // ring of KV blocks for sliding-attention layers
	ConvSlots   int // conv state slots, including the running one
}

// Layout selects the weight layouts on the GPU.
type Layout int

const (
	LayoutBoth Layout = iota
	LayoutDecodeOnly
	LayoutPrefillOnly
)

// Default ring sizes: enough to keep the last window and conv checkpoint of a few
// hundred cached prefixes, at under 700 MiB.
const (
	defaultSlideBlocks = 2048
	defaultConvSlots   = 1024
	nFullLayers        = 6
	nSlideLayers       = 5
	nConvLayers        = 13
)

// Engine owns the device, the weights and every GPU allocation.
type Engine struct {
	cfg Config
	dev *kfd.Device
	q   *kfd.Queue

	kDecode *kfd.Kernel
	ops     opKernels
	signal  *kfd.Signal
	globals [19]uint64 // decode kernel arguments

	blob  []byte                   // the weight arena in the payload
	index map[string]payload.Entry // tensor -> place in blob
	vram  map[string]uint64        // tensor -> GPU address
	arena *kfd.Buffer
	embed *kfd.Buffer

	headBuf *kfd.Buffer
	ds      draftState

	prefillArena *kfd.Buffer
	pfLayers     [Layers]prefillLayer
	pf           prefillBuffers

	ropeFull, ropeSlide   *kfd.Buffer
	layers, ws, logits    *kfd.Buffer
	step, seen            *kfd.Buffer
	tableFull, tableSlide *kfd.Buffer
	kvFull, kvSlide       *kfd.Buffer
	convState             *kfd.Buffer
	hostToken             *kfd.Buffer // token and logits markers the kernels write

	blitWindow *kfd.Buffer // host memory behind loadToVRAM
	staging    *kfd.Buffer // host memory behind upload
	stagingOff uint64

	fullBlocks int
	cache      *prefixCache
}

func (e *Engine) arenaVA(name string) uint64 {
	va, ok := e.vram[name]
	if !ok {
		panic("missing arena tensor " + name)
	}
	return va
}

// Open loads weights and kernels onto the GPU.
func Open(cfg Config, pay *payload.Payload) (*Engine, error) {
	dev, err := kfd.Open()
	if err != nil {
		return nil, err
	}
	q, err := dev.NewQueue()
	if err != nil {
		return nil, err
	}
	e := &Engine{cfg: cfg, dev: dev, q: q, signal: q.NewSignal()}
	e.ds.tapFrom = -1 // no prefill captures until a request asks for it
	if err := e.checkFootprint(pay); err != nil {
		return nil, err
	}
	if err := e.loadKernels(); err != nil {
		return nil, err
	}
	head, err := e.loadWeights(pay)
	if err != nil {
		return nil, err
	}
	if err := e.allocRuntime(); err != nil {
		return nil, err
	}
	var targetRows []uint64
	if cfg.Layout != LayoutPrefillOnly {
		targetRows = e.buildLayers()
	}
	e.cache = newPrefixCache(e.fullBlocks, e.cfg.SlideBlocks, e.cfg.ConvSlots)
	if cfg.Layout != LayoutDecodeOnly {
		if err := e.buildPrefillWeights(); err != nil {
			return nil, err
		}
	}
	if err := e.allocPrefillBuffers(); err != nil {
		return nil, err
	}
	if cfg.Layout == LayoutBoth {
		if err := e.initDSpark(pay, head, targetRows); err != nil {
			return nil, err
		}
	}
	// The host copies of the weights have served their purpose.
	e.blob = nil
	debug.FreeOSMemory()
	return e, nil
}

func (e *Engine) loadKernels() error {
	o := &e.ops
	if err := e.loadDraftKernel(); err != nil {
		return err
	}
	for _, obj := range []struct {
		name  string
		image []byte
		want  map[string]**kfd.Kernel
	}{
		{"decode", kernels.Decode, map[string]**kfd.Kernel{"decode_kernel": &e.kDecode}},
		{"prefill attention", kernels.PrefillAttn, map[string]**kfd.Kernel{
			"quantize_q": &o.quantizeQ, "prefill_attention": &o.attention,
		}},
		{"gemm", kernels.Gemm, map[string]**kfd.Kernel{
			"quantize_packILb0": &o.quantizePack, "quantize_packILb1": &o.quantizePackGated, "4gemmE": &o.gemm,
		}},
		{"ops", kernels.Ops, map[string]**kfd.Kernel{
			"embed_gather": &o.embed, "8rms_normE": &o.rms, "post_add_norm": &o.postAddNorm,
			"conv_load_state": &o.convLoad, "conv_bx": &o.convBx, "conv_apply": &o.convApply,
			"conv_store_state": &o.convStore, "q_norm_rope": &o.qNormRope, "kv_write": &o.kvWrite,
			"gate_mul": &o.gateMul, "head_fp16": &o.headFp16, "logits_to_host": &o.logitsToHost,
			"state_copy": &o.stateCopy, "6uploadE": &o.upload, "4blitE": &o.blit,
		}},
	} {
		prog, err := e.dev.LoadCodeObject(obj.image)
		if err != nil {
			return fmt.Errorf("%s code object: %w", obj.name, err)
		}
		for name, dst := range obj.want {
			if *dst, err = prog.Kernel(name); err != nil {
				return err
			}
		}
	}
	return nil
}

// loadWeights places the stored tensors on the GPU and derives the LM head, which it returns
// in host memory when this layout needs it.
func (e *Engine) loadWeights(pay *payload.Payload) (*weights.Packed, error) {
	raw, err := pay.Data("arena.index")
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &e.index); err != nil {
		return nil, fmt.Errorf("arena index: %w", err)
	}
	blob, err := pay.Data("arena")
	if err != nil {
		return nil, err
	}
	e.vram = make(map[string]uint64, len(e.index))
	e.blob = blob
	if e.cfg.Layout == LayoutPrefillOnly {
		if err := e.loadSmallTensors(blob); err != nil {
			return nil, err
		}
	} else {
		if err := e.checkVRAM(uint64(len(blob))); err != nil {
			return nil, err
		}
		if e.arena, err = e.dev.Alloc(uint64(len(blob)), kfd.VRAM); err != nil {
			return nil, fmt.Errorf("weight arena: %w", err)
		}
		if err := e.loadToVRAM(e.arena.VA, blob); err != nil {
			return nil, err
		}
		for name, ent := range e.index {
			e.vram[name] = e.arena.VA + ent.Off
		}
	}
	embed, err := pay.Data("embed")
	if err != nil {
		return nil, err
	}
	if e.embed, err = e.dev.Alloc(uint64(len(embed)), kfd.VRAM); err != nil {
		return nil, fmt.Errorf("embedding: %w", err)
	}
	if err := e.loadToVRAM(e.embed.VA, embed); err != nil {
		return nil, err
	}
	var head *weights.Packed
	if e.cfg.Layout != LayoutPrefillOnly {
		if head, err = e.buildHead(embed); err != nil {
			return nil, err
		}
	}
	if e.ropeFull, err = e.buildRope(1e6, 0.5); err != nil {
		return nil, err
	}
	if e.ropeSlide, err = e.buildRope(1e4, 1.0); err != nil {
		return nil, err
	}
	return head, nil
}

// loadSmallTensors places everything except the int8 decode matrices.
func (e *Engine) loadSmallTensors(blob []byte) error {
	var total uint64
	const align = 4096
	keep := func(name string) bool { return !strings.HasSuffix(name, ".w") }
	for name, ent := range e.index {
		if keep(name) {
			total += (ent.Size + align - 1) &^ (align - 1)
		}
	}
	var err error
	if e.arena, err = e.dev.Alloc(total, kfd.VRAM); err != nil {
		return fmt.Errorf("small tensors: %w", err)
	}
	dst := make([]byte, total)
	var off uint64
	for name, ent := range e.index {
		if keep(name) {
			copy(dst[off:], blob[ent.Off:ent.Off+ent.Size])
			e.vram[name] = e.arena.VA + off
			off += (ent.Size + align - 1) &^ (align - 1)
		}
	}
	return e.loadToVRAM(e.arena.VA, dst)
}

// vramHeadroom is kept free on the card. Filling VRAM makes the kernel evict other
// processes' buffers into host RAM, which can exhaust it and invoke the OOM killer.
const vramHeadroom = 512 << 20

// checkFootprint refuses to start, before allocating anything, when the card cannot
// hold the weights and one full-length sequence.
func (e *Engine) checkFootprint(pay *payload.Payload) error {
	arena, err := pay.RawSize("arena")
	if err != nil {
		return err
	}
	embed, err := pay.RawSize("embed")
	if err != nil {
		return err
	}
	need := embed + 512<<20 // tensors and small buffers
	if e.cfg.Layout != LayoutPrefillOnly {
		need += arena + headBytes
	}
	if e.cfg.Layout != LayoutDecodeOnly {
		need += prefillWeightBytes() + chunkBufferBytes()
	}
	if e.cfg.Layout == LayoutBoth {
		need += dsparkBufferBytes(dsWindowMax)
	}
	tokens := e.cfg.FullTokens
	if tokens == 0 {
		tokens = e.cfg.MaxContext
	}
	need += uint64(tokens/BlockTokens+2) * nFullLayers * KVBlock
	return e.checkVRAM(need)
}

// checkVRAM refuses an allocation that would push the card near full.
func (e *Engine) checkVRAM(need uint64) error {
	free, err := e.dev.FreeVRAM()
	if err != nil {
		return err
	}
	if need+vramHeadroom > free {
		return fmt.Errorf("need %d MiB of VRAM but only %d MiB is free (keeping %d MiB of headroom); stop the other GPU process or lower --kv-tokens",
			need>>20, free>>20, uint64(vramHeadroom)>>20)
	}
	return nil
}

// copyParallel copies from all CPUs.
func copyParallel(dst, src []byte) {
	const chunk = 16 << 20
	n := (len(src) + chunk - 1) / chunk
	weights.Parallel(n, func(i int) {
		a := i * chunk
		b := a + chunk
		if b > len(src) {
			b = len(src)
		}
		copy(dst[a:b], src[a:b])
	})
}

// buildRope fills cos | sin per position, as vLLM's zero-tail rotary embedding does:
// fp32 frequencies, rounded to fp16.
func (e *Engine) buildRope(base float64, factor float64) (*kfd.Buffer, error) {
	angles := int(factor*D) / 2
	var inv [D / 2]float32
	for i := 0; i < angles; i++ {
		exp := float32(2*i) / D
		inv[i] = 1 / float32(math.Pow(base, float64(exp)))
	}
	n := e.cfg.MaxContext + 1
	buf, err := e.dev.Alloc(uint64(n)*D*2, kfd.VRAM)
	if err != nil {
		return nil, fmt.Errorf("rope cache: %w", err)
	}
	out := make([]uint16, n*D)
	const chunk = 4096
	weights.Parallel((n+chunk-1)/chunk, func(c int) {
		var row [D]uint16
		for p := c * chunk; p < (c+1)*chunk && p < n; p++ {
			t := float32(p)
			for i := 0; i < D/2; i++ {
				f := float64(t * inv[i])
				row[i] = fp16.FromFloat32(float32(math.Cos(f)))
				row[D/2+i] = fp16.FromFloat32(float32(math.Sin(f)))
			}
			copy(out[p*D:(p+1)*D], row[:])
		}
	})
	raw := unsafe.Slice((*byte)(unsafe.Pointer(&out[0])), len(out)*2)
	if err := e.loadToVRAM(buf.VA, raw); err != nil {
		return nil, err
	}
	return buf, nil
}

func (e *Engine) allocRuntime() error {
	d := e.dev
	var err error
	alloc := func(dst **kfd.Buffer, size uint64, kind kfd.Kind, what string) {
		if err != nil {
			return
		}
		if *dst, err = d.Alloc(size, kind); err != nil {
			err = fmt.Errorf("%s: %w", what, err)
		}
	}
	if e.cfg.SlideBlocks == 0 {
		e.cfg.SlideBlocks = defaultSlideBlocks
	}
	if e.cfg.ConvSlots == 0 {
		e.cfg.ConvSlots = defaultConvSlots
	}
	slideBytes := uint64(nSlideLayers) * uint64(e.cfg.SlideBlocks) * KVBlock
	convBytes := uint64(nConvLayers) * uint64(e.cfg.ConvSlots) * H * ConvState * 2
	if e.cfg.FullTokens == 0 {
		// Prefill weights, chunk tensors and drafter buffers are allocated after this.
		later := prefillWeightBytes() + chunkBufferBytes() + dsparkBufferBytes(dsWindowMax) + 128<<20
		budget := uint64(float64(d.VRAMBytes()) * e.cfg.VRAMFraction)
		used := e.arena.Size + e.embed.Size + e.headBuf.Size + e.ropeFull.Size + e.ropeSlide.Size + slideBytes + convBytes + later
		if budget <= used {
			return fmt.Errorf("VRAM budget of %d MiB is too small (fixed needs %d MiB)", budget>>20, used>>20)
		}
		e.fullBlocks = int((budget - used) / (nFullLayers * KVBlock))
	} else {
		e.fullBlocks = (e.cfg.FullTokens + BlockTokens - 1) / BlockTokens
	}
	if err := e.checkVRAM(uint64(e.fullBlocks)*nFullLayers*KVBlock + slideBytes + convBytes); err != nil {
		return err
	}
	if minBlocks := e.cfg.MaxContext/BlockTokens + 2; e.fullBlocks < minBlocks {
		return fmt.Errorf("KV pool of %d blocks cannot hold one %d-token sequence", e.fullBlocks, e.cfg.MaxContext)
	}
	tableEntries := uint64(e.cfg.MaxContext/BlockTokens + 2)
	alloc(&e.layers, Layers*layerWords*8, kfd.VRAM, "layer table")
	alloc(&e.ws, workspaceBytes, kfd.VRAM, "workspace")
	alloc(&e.logits, Vocab*4, kfd.VRAM, "logits")
	alloc(&e.step, stepPageBytes, kfd.VRAM, "step page")
	alloc(&e.seen, seenWords*4, kfd.VRAM, "seen mask")
	alloc(&e.tableFull, tableEntries*4, kfd.VRAM, "block table")
	alloc(&e.tableSlide, tableEntries*4, kfd.VRAM, "block table")
	alloc(&e.hostToken, 4096, kfd.GTT, "token mailbox")
	alloc(&e.staging, 1<<20, kfd.GTT, "upload staging")
	alloc(&e.kvFull, nFullLayers*uint64(e.fullBlocks)*KVBlock, kfd.VRAM, "full KV")
	alloc(&e.kvSlide, slideBytes, kfd.VRAM, "sliding KV")
	alloc(&e.convState, convBytes, kfd.VRAM, "conv state")
	if err != nil {
		return err
	}
	// Nothing has read these yet, so a GPU clear needs no cache care. The barrier words
	// and control page must start zeroed, and the kernels use no other zero assumption.
	e.uploadRaw(0, e.ws.VA, workspaceBytes/4)
	e.uploadRaw(0, e.step.VA, stepPageBytes/4)
	e.uploadRaw(0, e.seen.VA, seenWords)
	if err := e.drain(kfd.ScopeAgent); err != nil {
		return fmt.Errorf("clearing control memory: %w", err)
	}
	return nil
}

// buildLayers writes the decode kernel's layer table and arguments and returns the table.
func (e *Engine) buildLayers() []uint64 {
	rows := make([]uint64, Layers*layerWords)
	stepVA := e.step.VA
	iFull, iSlide, iConv := uint64(0), uint64(0), uint64(0)
	slideBlocks := uint64(e.cfg.SlideBlocks)
	for li := 0; li < Layers; li++ {
		k := func(s string) uint64 { return e.arenaVA(fmt.Sprintf("l%d.%s", li, s)) }
		var r [layerWords]uint64
		r[lKind] = uint64(LayerKinds[li])
		r[lOpNorm], r[lFfnNorm] = k("operator_norm"), k("ffn_norm")
		r[lPostOpNorm], r[lPostFfnNorm] = k("post_operator_norm"), k("post_ffn_norm")
		r[lWMix], r[lSMix] = k("mix.w"), k("mix.s")
		r[lWOut], r[lSOut] = k("out.w"), k("out.s")
		r[lW13], r[lS13] = k("w13.w"), k("w13.s")
		r[lW2], r[lS2] = k("w2.w"), k("w2.s")
		switch LayerKinds[li] {
		case 0:
			r[lConvW] = k("conv")
			r[lConvState] = e.convState.VA + iConv*uint64(e.cfg.ConvSlots)*H*ConvState*2
			r[lCsSlot], r[lCsDim], r[lCsT] = H*ConvState, ConvState, 1
			r[lStateIndex] = stepVA + stepConvSlot
			iConv++
		default:
			r[lQNorm], r[lKNorm] = k("q_norm"), k("k_norm")
			r[lRotary] = D
			r[lKVBlock], r[lKVHead], r[lKVSlot] = KVBlock, Rec, KVH*Rec
			r[lBlockSize] = BlockTokens
			r[lSeqLens] = stepVA + stepSeqLen
			if LayerKinds[li] == 1 {
				r[lRope] = e.ropeFull.VA
				r[lKV] = e.kvFull.VA + iFull*uint64(e.fullBlocks)*KVBlock
				r[lBlockTable] = e.tableFull.VA
				r[lSlotMapping] = stepVA + stepSlotFull
				iFull++
			} else {
				r[lRope] = e.ropeSlide.VA
				r[lKV] = e.kvSlide.VA + iSlide*slideBlocks*KVBlock
				r[lBlockTable] = e.tableSlide.VA
				r[lSlotMapping] = stepVA + stepSlotSlide
				r[lWindow] = Window
				iSlide++
			}
		}
		copy(rows[li*layerWords:], r[:])
	}
	g := &e.globals
	g[0] = e.layers.VA
	g[1] = stepVA + stepInput
	g[2] = e.embed.VA
	g[3] = e.arenaVA("final_norm")
	g[4], g[5] = e.arenaVA("head.w"), e.arenaVA("head.s")
	g[6] = e.logits.VA
	g[7] = 0 // hidden_out: unused
	g[8] = e.ws.VA
	g[9] = stepVA + stepBarrier
	g[10] = Layers
	g[11] = Vocab
	g[12] = 0 // trace
	g[13] = e.seen.VA
	g[14] = stepVA + stepRep
	g[15] = 2 // sample rounded to fp16, the dtype vLLM's sampler sees
	g[16] = stepVA + stepBest
	g[17] = stepVA + stepDone
	g[18] = e.hostToken.VA
	e.upload(e.layers.VA, unsafe.Slice((*uint32)(unsafe.Pointer(&rows[0])), len(rows)*2))
	return rows
}

// submitDecode queues the megakernel with the fences ROCr uses for it. Agent scope is
// required, not just cheaper: a system-scope acquire invalidates the L2, and the kernels'
// results (KV records, conv state, uploads) live there until something flushes them.
func (e *Engine) submitDecode() {
	e.q.Submit(&kfd.Dispatch{
		Kernel:  e.kDecode,
		Grid:    [3]uint32{decodeBlocks * decodeThreads},
		Block:   [3]uint16{decodeThreads},
		Args:    unsafe.Pointer(&e.globals),
		ArgSize: unsafe.Sizeof(e.globals),
		Acquire: kfd.ScopeAgent,
		Release: kfd.ScopeAgent,
	})
}

// Sampling modes of the decode kernel.
const (
	SampleOff     = 0
	SampleRound16 = 2
)

// InitSeen sets the repetition-penalty mask to exactly the given tokens.
func (e *Engine) InitSeen(tokens []int32, rep float32) {
	if rep == 0 {
		rep = 1
	}
	mask := make([]uint32, seenWords)
	for _, t := range tokens {
		if t >= 0 && t < Vocab {
			mask[t>>5] |= 1 << (uint(t) & 31)
		}
	}
	e.upload(e.seen.VA, mask)
	e.upload(e.step.VA+stepRep, []uint32{math.Float32bits(rep)})
}

// writeStep queues the per-token control words the decode kernel reads.
func (e *Engine) writeStep(token int32, seqLen int, slotFull, slotSlide int64, convSlot int32) {
	var w [10]uint32
	w[stepInput/4] = uint32(token)
	w[stepSeqLen/4] = uint32(seqLen)
	w[stepSlotFull/4], w[stepSlotFull/4+1] = uint32(slotFull), uint32(slotFull>>32)
	w[stepSlotSlide/4], w[stepSlotSlide/4+1] = uint32(slotSlide), uint32(slotSlide>>32)
	w[stepConvSlot/4] = uint32(convSlot)
	e.upload(e.step.VA, w[:])
}

// DecodeStep runs one token through the megakernel. The caller has placed its KV
// slot and table entries; the kernel writes logits and, when sampling, the token.
func (e *Engine) DecodeStep(token int32, seqLen int, slotFull, slotSlide int64, convSlot int32, sample uint64) (int32, error) {
	e.writeStep(token, seqLen, slotFull, slotSlide, convSlot)
	e.globals[15] = sample
	box := (*int32)(e.hostToken.Ptr())
	atomic.StoreInt32(box, tokenPending)
	e.submitDecode()
	// Agent scope: only the token crosses to the host, and it goes straight to uncached
	// host memory. A system-scope release would write back the whole L2 on every token,
	// which costs up to 0.9 ms. The sentinel covers the token's own arrival.
	if err := e.drain(kfd.ScopeAgent); err != nil {
		return 0, fmt.Errorf("decode step: %w", err)
	}
	if sample == SampleOff {
		return 0, nil
	}
	return e.awaitToken(box)
}

// tokenPending marks the mailbox empty; no token id is negative.
const tokenPending = -1

// awaitToken waits for the kernel's token, which may land just after the barrier signal.
func (e *Engine) awaitToken(box *int32) (int32, error) {
	deadline := time.Now().Add(time.Second)
	for {
		for i := 0; i < 4096; i++ {
			if t := atomic.LoadInt32(box); t != tokenPending {
				return t, nil
			}
		}
		if time.Now().After(deadline) {
			return 0, fmt.Errorf("decode step: the kernel finished but no token arrived")
		}
	}
}

// SetBlock maps a logical block index of a sequence to physical blocks.
func (e *Engine) SetBlock(logical int, physFull, physSlide int32) {
	e.upload(e.tableFull.VA+uint64(logical)*4, []uint32{uint32(physFull)})
	e.upload(e.tableSlide.VA+uint64(logical)*4, []uint32{uint32(physSlide)})
}

// KVCapacityTokens is the size of the full-attention KV pool.
func (e *Engine) KVCapacityTokens() int { return e.fullBlocks * BlockTokens }

// Close stops the queue.
func (e *Engine) Close() error { return e.q.Destroy() }
