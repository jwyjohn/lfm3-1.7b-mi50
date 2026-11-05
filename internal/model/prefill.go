package model

import (
	"encoding/binary"
	"fmt"
	"math"
	"sync/atomic"
	"unsafe"

	"lfm3-1.7b-mi50/internal/fp16"
	"lfm3-1.7b-mi50/internal/kfd"
	"lfm3-1.7b-mi50/internal/weights"
)

const (
	gemmTile    = 128
	gemmThreads = 256
	halfPacks   = 16
	qrowDwords  = D/4 + 4
	mixColumns  = 3088
	// maxChunk serves prompts from longChunkFrom tokens on: at 7680 tokens a chunk maps to
	// 1920 attention tiles, whole rounds of the 60 CUs, and each layer's fixed costs spread
	// over more tokens. Short prompts stay at shortChunk, which ends sooner.
	maxChunk      = 7680
	shortChunk    = 2048
	longChunkFrom = 32768
)

// g128Weight is one linear layer in the GEMM kernel's packed layout.
type g128Weight struct {
	packed, scales uint64
	outputs        int
	columns        int
}

type prefillLayer struct {
	mix, out, w13, w2 g128Weight
}

type opKernels struct {
	embed, rms, postAddNorm         *kfd.Kernel
	convLoad, convBx, convApply     *kfd.Kernel
	convStore, qNormRope, kvWrite   *kfd.Kernel
	gateMul, headFp16, logitsToHost *kfd.Kernel
	stateCopy, upload, blit         *kfd.Kernel
	quantizePack, quantizePackGated *kfd.Kernel
	gemm, quantizeQ, attention      *kfd.Kernel
}

// prefillBuffers are the fixed chunk-sized activation tensors.
type prefillBuffers struct {
	ids               *kfd.Buffer
	h, n, f, bcx, bxs *kfd.Buffer
	y, o, a, q, gu, d *kfd.Buffer
	aPacked, aScales  *kfd.Buffer
	qbuf              *kfd.Buffer
	hostLogits        *kfd.Buffer
}

func g128Sizes(rows, k int) (packed, scales uint64) {
	tiles := (rows + gemmTile - 1) / gemmTile
	return uint64(tiles * gemmTile * k), uint64(k/weights.Group*tiles*gemmTile) * 4
}

// matrixSpec names a matrix and how prefill row r maps to the decode-layout row.
type matrixSpec struct {
	name    string
	rows, k int
	rowMap  func(r int) int
}

func identity(r int) int { return r }

func interleaved(n, parts int) func(int) int {
	return func(r int) int { return parts*(r%n) + r/n }
}

func layerMatrices(li int) []matrixSpec {
	k := func(s string) string { return fmt.Sprintf("l%d.%s", li, s) }
	m := []matrixSpec{
		{k("w13"), 2 * I, H, interleaved(I, 2)},
		{k("w2"), H, I, identity},
		{k("out"), H, H, identity},
	}
	if LayerKinds[li] == 0 {
		return append(m, matrixSpec{k("mix"), 3 * H, H, interleaved(H, 3)})
	}
	return append(m, matrixSpec{k("mix"), mixColumns, H, identity})
}

// prefillWeightBytes is the size of the GEMM-layout weight arena.
func prefillWeightBytes() uint64 {
	const align = 64 << 10
	var total uint64
	for li := 0; li < Layers; li++ {
		for _, m := range layerMatrices(li) {
			p, s := g128Sizes(m.rows, m.k)
			total += (p + align - 1) &^ (align - 1)
			total += (s + align - 1) &^ (align - 1)
		}
	}
	return total
}

// buildPrefillWeights derives the GEMM layout from the decode layout in the payload:
// the int8 values are identical, only rows and packing differ.
func (e *Engine) buildPrefillWeights() error {
	blob := e.blob
	var err error
	const align = 64 << 10
	total := prefillWeightBytes()
	if err := e.checkVRAM(total); err != nil {
		return err
	}
	if e.prefillArena, err = e.dev.Alloc(total, kfd.VRAM); err != nil {
		return fmt.Errorf("prefill weights: %w", err)
	}
	// Built in host memory, in the arena's final order, then streamed once.
	host := make([]byte, total)
	dst := host
	var off uint64
	for li := 0; li < Layers; li++ {
		pl := &e.pfLayers[li]
		for _, m := range layerMatrices(li) {
			ws, ok1 := e.index[m.name+".w"]
			ss, ok2 := e.index[m.name+".s"]
			if !ok1 || !ok2 {
				return fmt.Errorf("arena has no %s", m.name)
			}
			packedBytes, scaleBytes := g128Sizes(m.rows, m.k)
			w := g128Weight{e.prefillArena.VA + off, e.prefillArena.VA + off + ((packedBytes + align - 1) &^ (align - 1)), 0, m.k}
			w.outputs = m.rows
			fillPacked(dst[off:off+packedBytes], blob[ws.Off:ws.Off+ws.Size], m)
			off += (packedBytes + align - 1) &^ (align - 1)
			fillScales(dst[off:off+scaleBytes], blob[ss.Off:ss.Off+ss.Size], m)
			off += (scaleBytes + align - 1) &^ (align - 1)
			switch {
			case m.name == fmt.Sprintf("l%d.w13", li):
				pl.w13 = w
			case m.name == fmt.Sprintf("l%d.w2", li):
				pl.w2 = w
			case m.name == fmt.Sprintf("l%d.out", li):
				pl.out = w
			default:
				pl.mix = w
			}
		}
	}
	return e.loadToVRAM(e.prefillArena.VA, host)
}

// fillPacked writes uint32 [K/64][tiles][16][128], one signed byte per input. Each
// (half, tile) region is built in a local buffer, then copied in one burst.
func fillPacked(dst, biased []byte, m matrixSpec) {
	tiles := (m.rows + gemmTile - 1) / gemmTile
	halves := m.k / 64
	const regionDwords = halfPacks * gemmTile
	weights.Parallel(halves*tiles, func(i int) {
		h, t := i/tiles, i%tiles
		var local [regionDwords]uint32
		for r := 0; r < gemmTile; r++ {
			row := t*gemmTile + r
			if row >= m.rows {
				break
			}
			src := biased[m.rowMap(row)*m.k+h*64:]
			for p := 0; p < halfPacks; p++ {
				local[p*gemmTile+r] = binary.LittleEndian.Uint32(src[4*p:]) ^ 0x80808080
			}
		}
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&local[0])), regionDwords*4)
		copy(dst[uint64(i)*regionDwords*4:], raw)
	})
}

// fillScales writes fp32 [K/128][rows padded to the tile] from the decode layout's scales.
func fillScales(dst, packedScales []byte, m matrixSpec) {
	sq := weights.SQ(m.k)
	srow := 4 * sq
	fillGroupScales(dst, m.rows, m.k/weights.Group, func(r, g int) uint16 {
		at := (m.rowMap(r)*srow + (g%4)*sq + g/4) * 2
		return binary.LittleEndian.Uint16(packedScales[at:])
	})
}

// fillGroupScales writes fp32 [groups][rows padded to the tile], padding scale 1.
func fillGroupScales(dst []byte, rows, groups int, scale func(r, g int) uint16) {
	tiles := (rows + gemmTile - 1) / gemmTile
	padded := tiles * gemmTile
	weights.Parallel(groups, func(g int) {
		col := make([]uint32, padded)
		for r := range col {
			col[r] = 0x3f800000 // 1.0
		}
		for r := 0; r < rows; r++ {
			col[r] = math.Float32bits(fp16.ToFloat32(scale(r, g)))
		}
		raw := unsafe.Slice((*byte)(unsafe.Pointer(&col[0])), padded*4)
		copy(dst[g*padded*4:], raw)
	})
}

// chunkSizes lists the chunk tensors, for allocation and for the VRAM budget.
func (p *prefillBuffers) chunkSizes() map[**kfd.Buffer]uint64 {
	const fp16Bytes = 2
	return map[**kfd.Buffer]uint64{
		&p.ids:     maxChunk * 4,
		&p.h:       maxChunk * H * fp16Bytes,
		&p.n:       maxChunk * H * fp16Bytes,
		&p.f:       maxChunk * mixColumns * fp16Bytes,
		&p.bcx:     maxChunk * 3 * H * fp16Bytes,
		&p.bxs:     (maxChunk + ConvState) * H * fp16Bytes,
		&p.y:       maxChunk * H * fp16Bytes,
		&p.o:       maxChunk * H * fp16Bytes,
		&p.a:       maxChunk * H * fp16Bytes,
		&p.q:       maxChunk * H * fp16Bytes,
		&p.gu:      maxChunk * 2 * I * fp16Bytes,
		&p.d:       maxChunk * H * fp16Bytes,
		&p.aPacked: maxChunk * I,
		&p.aScales: maxChunk * I / weights.Group * 4,
		&p.qbuf:    maxChunk * QH * qrowDwords * 4,
	}
}

// chunkBufferBytes is the VRAM of the chunk tensors, rounded as the allocator does.
func chunkBufferBytes() uint64 {
	var total uint64
	var p prefillBuffers
	for _, size := range p.chunkSizes() {
		total += (size + 2<<20 - 1) &^ (2<<20 - 1)
	}
	return total
}

func (e *Engine) allocPrefillBuffers() error {
	var err error
	for dst, size := range e.pf.chunkSizes() {
		if err == nil {
			*dst, err = e.dev.Alloc(size, kfd.VRAM)
		}
	}
	p := &e.pf
	if err != nil {
		return fmt.Errorf("prefill buffers: %w", err)
	}
	if p.hostLogits, err = e.dev.Alloc(Vocab*4, kfd.GTT); err != nil {
		return fmt.Errorf("host logits: %w", err)
	}
	return nil
}

// launch enqueues one kernel with agent-scope fences: results stay in the GPU's L2,
// which every later kernel on this queue shares. See submitDecode for why no dispatch
// may use a system-scope acquire.
func (e *Engine) launch(k *kfd.Kernel, gridX, gridY uint32, block uint16, args unsafe.Pointer, size uintptr) {
	e.q.Submit(&kfd.Dispatch{
		Kernel:  k,
		Grid:    [3]uint32{gridX * uint32(block), gridY},
		Block:   [3]uint16{block},
		Args:    args,
		ArgSize: size,
		Acquire: kfd.ScopeAgent,
		Release: kfd.ScopeAgent,
	})
}

// upload queues a GPU copy of dwords from host memory to VRAM. The staging ring holds
// more than one chunk's worth of data, so entries survive until their dispatch runs.
func (e *Engine) upload(dst uint64, data []uint32) {
	n := uint64(len(data))
	if e.stagingOff+n*4 > e.staging.Size {
		e.stagingOff = 0
	}
	src := e.staging.VA + e.stagingOff
	copy(unsafe.Slice((*uint32)(unsafe.Add(e.staging.Ptr(), e.stagingOff)), n), data)
	e.stagingOff += n*4 + 60
	e.stagingOff &^= 63
	e.uploadRaw(src, dst, int32(n))
}

// blitChunk is the size of the host window used to stream weights to the GPU.
const blitChunk = 128 << 20

// loadToVRAM streams data into private VRAM through the GPU's L2, in chunks that fill
// from all CPUs. Writing through the BAR instead would leave any older L2 lines for
// those addresses in place, and the weights' streaming reads rarely replace them.
func (e *Engine) loadToVRAM(dst uint64, data []byte) error {
	if e.blitWindow == nil {
		var err error
		if e.blitWindow, err = e.dev.Alloc(blitChunk, kfd.GTT); err != nil {
			return fmt.Errorf("blit window: %w", err)
		}
	}
	host := e.blitWindow.Bytes()
	for off := 0; off < len(data); off += blitChunk {
		end := off + blitChunk
		if end > len(data) {
			end = len(data)
		}
		copyParallel(host[:end-off], data[off:end])
		// A trailing partial 16-byte unit copies whatever follows in the window, which
		// lands inside the padded destination allocation.
		a := struct {
			src, dst uint64
			bytes    int64
		}{e.blitWindow.VA, dst + uint64(off), int64(end - off)}
		blocks := uint32((end - off + 4095) / 4096)
		e.launch(e.ops.blit, blocks, 1, 256, unsafe.Pointer(&a), 24)
		if err := e.drain(kfd.ScopeAgent); err != nil {
			return fmt.Errorf("blit: %w", err)
		}
	}
	return nil
}

func (e *Engine) uploadRaw(src, dst uint64, dwords int32) {
	a := struct {
		src, dst uint64
		dwords   int32
	}{src, dst, dwords}
	blocks := (uint32(dwords) + 255) / 256
	e.launch(e.ops.upload, blocks, 1, 256, unsafe.Pointer(&a), 20)
}

// drain waits for everything submitted so far. A bounded in-flight window matters on
// gfx906: long chains of queued dispatches have wedged its compute ring.
func (e *Engine) drain(release kfd.FenceScope) error {
	s := e.signal
	s.Reset()
	e.q.SubmitBarrier(s, release)
	return e.q.Wait(s, gpuTimeout)
}

// g128 runs y = x W^T: quantize and pack the fp16 activations, then the GEMM.
func (e *Engine) g128(w *g128Weight, x uint64, xStride int64, rows int, gated bool, out uint64, outStride int64) {
	p := &e.pf
	qa := struct {
		x             uint64
		xStride       int64
		packed, scale uint64
		rows, columns int32
	}{x, xStride, p.aPacked.VA, p.aScales.VA, int32(rows), int32(w.columns)}
	tiles := uint32((rows + gemmTile - 1) / gemmTile)
	k := e.ops.quantizePack
	if gated {
		k = e.ops.quantizePackGated
	}
	e.launch(k, tiles, uint32(w.columns/weights.Group), gemmThreads, unsafe.Pointer(&qa), 40)
	m := uint32((rows + gemmTile - 1) / gemmTile)
	n := uint32((w.outputs + gemmTile - 1) / gemmTile)
	ga := struct {
		packedA, aScales, packedB, bScales, out uint64
		outStride                               int64
		residual                                uint64
		residualStride                          int64
		rows, outputs, columns, pad             int32
	}{p.aPacked.VA, p.aScales.VA, w.packed, w.scales, out, outStride, 0, 0, int32(rows), int32(w.outputs), int32(w.columns), 0}
	e.launch(e.ops.gemm, ((m+3)/4)*n*4, 1, gemmThreads, unsafe.Pointer(&ga), 80)
}

// PrefillChunk runs up to maxChunk tokens through every layer. The table entries and
// the conv slot of the sequence must already be in place. With wantLogits, the logits
// of the last token reach host memory, rounded to fp16. When ckptTokens > 0, the conv
// state after that many tokens of this chunk is also saved to ckptSlot, so a later
// request can resume from that block boundary.
func (e *Engine) PrefillChunk(ids []int32, context int, convSlot int32, hasInitial, wantLogits bool, ckptSlot int32, ckptTokens int) error {
	T := len(ids)
	if T == 0 || T > maxChunk {
		return fmt.Errorf("chunk of %d tokens", T)
	}
	p := &e.pf
	e.upload(p.ids.VA, unsafe.Slice((*uint32)(unsafe.Pointer(&ids[0])), T))
	o := &e.ops
	rows := uint32(T)
	h, n := p.h.VA, p.n.VA

	ea := struct {
		ids, weight, out uint64
		stride           int64
	}{p.ids.VA, e.embed.VA, h, H}
	e.launch(o.embed, rows, 1, 256, unsafe.Pointer(&ea), 32)
	ra := struct {
		in       uint64
		inStride int64
		w, out   uint64
		stride   int64
	}{h, H, e.arenaVA("l0.operator_norm"), n, H}
	e.launch(o.rms, rows, 1, 256, unsafe.Pointer(&ra), 40)

	initial := int32(0)
	if hasInitial {
		initial = 1
	}
	iConv := uint64(0)
	iFull, iSlide := uint64(0), uint64(0)
	slideBlocks := uint64(e.cfg.SlideBlocks)
	for li := 0; li < Layers; li++ {
		L := &e.pfLayers[li]
		arena := func(s string) uint64 { return e.arenaVA(fmt.Sprintf("l%d.%s", li, s)) }
		if LayerKinds[li] == 0 {
			e.g128(&L.mix, n, H, T, false, p.bcx.VA, 3*H)
			state := e.convState.VA + iConv*uint64(e.cfg.ConvSlots)*H*ConvState*2 + uint64(convSlot)*H*ConvState*2
			iConv++
			la := struct {
				state, bxs uint64
				flag       int32
			}{state, p.bxs.VA, initial}
			e.launch(o.convLoad, ConvState, 1, 256, unsafe.Pointer(&la), 20)
			ba := struct{ bcx, bxs uint64 }{p.bcx.VA, p.bxs.VA}
			e.launch(o.convBx, rows, 1, 256, unsafe.Pointer(&ba), 16)
			aa := struct{ bxs, bcx, w, y uint64 }{p.bxs.VA, p.bcx.VA, arena("conv"), p.y.VA}
			e.launch(o.convApply, rows, 1, 256, unsafe.Pointer(&aa), 32)
			sa := struct {
				bxs, state uint64
				tokens     int32
			}{p.bxs.VA, state, int32(T)}
			e.launch(o.convStore, ConvState, 1, 256, unsafe.Pointer(&sa), 20)
			if ckptTokens > 0 {
				ca := struct {
					bxs, state uint64
					tokens     int32
				}{p.bxs.VA, state + uint64(ckptSlot-convSlot)*H*ConvState*2, int32(ckptTokens)}
				e.launch(o.convStore, ConvState, 1, 256, unsafe.Pointer(&ca), 20)
			}
			e.g128(&L.out, p.y.VA, H, T, false, p.o.VA, H)
		} else {
			e.g128(&L.mix, n, H, T, false, p.f.VA, mixColumns)
			full := LayerKinds[li] == 1
			var rope, kv, table uint64
			window := int32(0)
			if full {
				rope, table = e.ropeFull.VA, e.tableFull.VA
				kv = e.kvFull.VA + iFull*uint64(e.fullBlocks)*KVBlock
				iFull++
			} else {
				rope, table = e.ropeSlide.VA, e.tableSlide.VA
				kv = e.kvSlide.VA + iSlide*slideBlocks*KVBlock
				iSlide++
				window = Window
			}
			qa := struct {
				f, w, rope, q   uint64
				context, stride int32
			}{p.f.VA, arena("q_norm"), rope, p.q.VA, int32(context), mixColumns}
			e.launch(o.qNormRope, rows*QH, 1, 64, unsafe.Pointer(&qa), 40)
			ka := struct {
				f, kNorm, rope, kv                 uint64
				kvBlock, kvHead, kvSlot            int64
				table                              uint64
				blockSize, context, fStride, pad32 int32
			}{p.f.VA, arena("k_norm"), rope, kv, KVBlock, Rec, KVH * Rec, table, BlockTokens, int32(context), mixColumns, 0}
			e.launch(o.kvWrite, rows*KVH, 1, 64, unsafe.Pointer(&ka), 80)
			pa := struct {
				q, out, kv              uint64
				kvBlock, kvHead, kvSlot int64
				table                   uint64
				blockSize, tokens       int32
				context, window         int32
				qStride, outStride      int64
				qbuf                    uint64
			}{p.q.VA, p.a.VA, kv, KVBlock, Rec, KVH * Rec, table, BlockTokens, int32(T), int32(context), window, H, H, p.qbuf.VA}
			e.launch(o.quantizeQ, rows*QH, 1, 64, unsafe.Pointer(&pa), 96)
			blocks := (rows + 15) / 16
			e.launch(o.attention, blocks, KVH, 256, unsafe.Pointer(&pa), 96)
			ga := struct {
				a, f   uint64
				stride int32
			}{p.a.VA, p.f.VA, mixColumns}
			e.launch(o.gateMul, rows, 1, 256, unsafe.Pointer(&ga), 20)
			e.g128(&L.out, p.a.VA, H, T, false, p.o.VA, H)
		}
		// Residual after the mixer, then the feed-forward block.
		pn := struct {
			o       uint64
			oStride int64
			post, h uint64
			next, n uint64
		}{p.o.VA, H, arena("post_operator_norm"), h, arena("ffn_norm"), n}
		e.launch(o.postAddNorm, rows, 1, 256, unsafe.Pointer(&pn), 48)
		e.g128(&L.w13, n, H, T, false, p.gu.VA, 2*I)
		e.g128(&L.w2, p.gu.VA, 2*I, T, true, p.d.VA, H)
		next := e.arenaVA("final_norm")
		if li+1 < Layers {
			next = e.arenaVA(fmt.Sprintf("l%d.operator_norm", li+1))
		}
		pf := struct {
			o       uint64
			oStride int64
			post, h uint64
			next, n uint64
		}{p.d.VA, H, arena("post_ffn_norm"), h, next, n}
		e.launch(o.postAddNorm, rows, 1, 256, unsafe.Pointer(&pf), 48)
		if slot := tapSlot(li); slot >= 0 && e.ds.tapFrom >= 0 {
			e.captureTap(slot, h, context, T)
		}
		if err := e.drain(kfd.ScopeAgent); err != nil {
			return fmt.Errorf("prefill layer %d: %w", li, err)
		}
	}
	if wantLogits {
		ha := struct {
			weight, x, logits uint64
			rows              int32
		}{e.embed.VA, n + uint64(T-1)*H*2, e.logits.VA, Vocab}
		e.launch(o.headFp16, decodeBlocks, 1, 256, unsafe.Pointer(&ha), 28)
		return e.finishLogits(false)
	}
	return e.drain(kfd.ScopeAgent)
}

// finishLogits copies the logits to host memory and waits until they are all there.
// Nothing here needs a system-scope release, which would write back the whole L2: the
// copy waits for its own writes, and the marker written after it proves they finished.
func (e *Engine) finishLogits(rounded bool) error {
	box := (*int32)(unsafe.Add(e.hostToken.Ptr(), logitsMarkerOff))
	atomic.StoreInt32(box, tokenPending)
	e.LogitsToHost(rounded)
	e.upload(e.hostToken.VA+logitsMarkerOff, []uint32{1})
	if err := e.drain(kfd.ScopeAgent); err != nil {
		return err
	}
	_, err := e.awaitToken(box)
	return err
}

// logitsMarkerOff keeps the marker in its own cache line of the token mailbox page.
const logitsMarkerOff = 64

// LogitsToHost queues a copy of the VRAM logits to host memory.
func (e *Engine) LogitsToHost(rounded bool) {
	flag := int32(0)
	if rounded {
		flag = 1
	}
	la := struct {
		src, dst       uint64
		count, rounded int32
	}{e.logits.VA, e.pf.hostLogits.VA, Vocab, flag}
	blocks := uint32((Vocab + 256*4 - 1) / (256 * 4))
	e.launch(e.ops.logitsToHost, blocks, 1, 256, unsafe.Pointer(&la), 24)
}

// HostLogits is the host copy of the logits, valid after finishLogits.
func (e *Engine) HostLogits() []float32 {
	return unsafe.Slice((*float32)(e.pf.hostLogits.Ptr()), Vocab)
}

// CopyConvSlot copies a conv state slot across all conv layers on the GPU.
func (e *Engine) CopyConvSlot(src, dst int32) {
	slot := uint64(H * ConvState * 2)
	ca := struct {
		base        uint64
		layerStride int64
		srcOff      int64
		dstOff      int64
	}{e.convState.VA, int64(uint64(e.cfg.ConvSlots) * slot), int64(uint64(src) * slot), int64(uint64(dst) * slot)}
	const chunks = H * ConvState * 2 / 4096
	e.launch(e.ops.stateCopy, chunks, 13, 256, unsafe.Pointer(&ca), 32)
}
