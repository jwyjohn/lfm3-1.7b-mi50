package model

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sync/atomic"
	"unsafe"

	"lfm3-1.7b-mi50/internal/kfd"
	"lfm3-1.7b-mi50/internal/payload"
	"lfm3-1.7b-mi50/internal/weights"
	"lfm3-1.7b-mi50/kernels"
)

// DSpark speculative decode. One launch of kernels/dspark.hip runs speculative rounds for
// the active request until it reaches the next block boundary: project the tapped states of
// the tokens committed last round into the drafter's KV ring, draft with the 3 drafter
// layers, verify anchor + drafts through the 24 target layers with greedy tokens computed in
// the kernel, and commit the accepted prefix plus the bonus token.

// Geometry shared with kernels/dspark.hip.
const (
	dsMaxTokens       = 9 // MT: tokens one verify pass can hold
	dsDraftLayers     = 3 // DRAFT_LAYERS
	dsTaps            = 5 // NUM_TAPS
	dsRows            = Layers + dsDraftLayers
	dsWorkspaceFloats = 1759744               // WS_FLOATS
	dsStateWords      = sTokens + dsMaxTokens // struct State
	dsFreshBytes      = 32 * KVH * Rec        // [32 tokens][KVH][Rec] of verified K/V
	dsConvRingBytes   = 32 * H * 2            // RING positions of fp16 conv history
	dsRingPositions   = 4096                  // drafter KV ring
	dsRingBlock       = 16                    // ring addressing unit, positions
	dsRingBlockBytes  = dsRingBlock * KVH * Rec
	dsRingBytes       = dsRingPositions * KVH * Rec
	dsCropRows        = 16384 // head crop rows
	dsCropExtra       = 4     // confidence row + padding rows
	dsMarkovRank      = 256
	dsCkvRows         = dsDraftLayers * 1024
	dsWindowMax       = 2048 // drafter attention window the context buffers hold
)

// Layer table words beyond the decode kernel's; must match struct Layer in kernels/dspark.hip.
const (
	lRing = layerWords + iota
	lTapSlot
	lFresh
	lPad
	dsLayerWords
)

// Words of struct State.
const (
	sA = iota
	sAnchor
	sCount
	sRounds
	sLimit
	sBudget
	sEOS
	sM
	sCtxNew
	sCtxPos
	sCtxLo
	sRun
	sForce
	sDone
	sAccepted
	sVerified
	sMiss
	sSkip
	sSkipAfter
	sSkipRounds
	sStopAt
	sCkptSlot
	sTokens
)

// Speculation policy: the threshold was chosen by replaying fixed token sequences through the
// drafter; the switch to fewer drafts moved from 32768 to 57344 once verify attention became one
// pass over all verified tokens (measured over 18 streams: 7 drafts win at 32k and 48k, 3 at 64k
// and 128k, where a verified token still costs ~55 us per full-attention layer at 128k).
const (
	dsRoundsPerLaunch = 32
	dsMaxDrafts       = 7
	dsMaxDraftsLong   = 3 // from dsLongFrom on, where a verified token costs more
	dsLongFrom        = 57344
	dsConfThreshold   = -0.4
	dsMaxBudget       = 512
)

// dsTapLayers are the target layers whose outputs feed the drafter.
var dsTapLayers = [dsTaps]int{2, 6, 12, 18, 23}

func tapSlot(layer int) int {
	for i, l := range dsTapLayers {
		if l == layer {
			return i
		}
	}
	return -1
}

// Offsets in the control buffer; atomics and polled words stay out of each other's lines.
const (
	dsCtlState   = 0
	dsCtlBest    = 256
	dsCtlBarrier = 512
	dsCtlOut     = 4096
	dsCtlBytes   = 3 * 4096
)

// Offsets in the host mailbox.
const (
	dsHostMarker = 0
	dsHostState  = 64
	dsHostOut    = 256
	dsHostBytes  = 4096
)

// dsGlobals is struct Globals in kernels/dspark.hip.
type dsGlobals struct {
	layers, embed, finalNorm, wHead, sHead, draftNorm uint64
	wCrop, sCrop, cropIDs, w2Crop, w2Scale, w1q, w1s  uint64
	c2, base, tapNorms, wFc, sFc, hiddenNorm          uint64
	wCkv, sCkv, ws, barrier, st, seen, rep, best, out uint64
	trace, dbgLogits                                  uint64
	vocab, cropRows, maskToken, maxDrafts, attnMask   int64
	confBias, confThr, invRep, pad                    float32
}

// sizeof(Globals): an underflow in either direction fails the build.
const (
	_ = uint(unsafe.Sizeof(dsGlobals{}) - 296)
	_ = uint(296 - unsafe.Sizeof(dsGlobals{}))
)

// draftState is everything the drafter path owns.
type draftState struct {
	kernel  *kfd.Kernel
	globals dsGlobals
	window  int

	layers, ws, ctl, base *kfd.Buffer
	rings, locals         *kfd.Buffer
	ringTable             *kfd.Buffer
	derived               *kfd.Buffer // cropped head and k/v projection in decode layout
	host                  *kfd.Buffer // state and tokens the host reads
	fc, ckv               g128Weight  // the same projections for the prompt, GEMM layout
	projWeights           *kfd.Buffer
	tapX, tapC, tapC2     *kfd.Buffer
	tapKV                 *kfd.Buffer
	trace                 *dsTracer

	// tapFrom is the first position the running prefill captures, or -1.
	tapFrom int

	// Drafter context kept between launches of a request.
	ctxNew, ctxPos, ctxLo int32
	miss, skip            int32

	// The ring keeps positions [ringLo, ringEnd) of ringTokens across requests.
	ringLo, ringEnd int
	ringTokens      []int32
}

// dsparkBufferBytes is the VRAM of everything allocated in this file.
func dsparkBufferBytes(window int) uint64 {
	const padded = 2 << 20
	c := func(n uint64) uint64 { return (n + padded - 1) &^ (padded - 1) }
	w := uint64(window)
	total := c(dsWorkspaceFloats*4) + c(dsMaxTokens*(dsCropRows+dsCropExtra)*4) + c(dsDraftLayers*dsRingBytes)
	total += c(nConvLayers*dsConvRingBytes + (nFullLayers+nSlideLayers+dsDraftLayers)*dsFreshBytes)
	total += c((dsCropRows+dsCropExtra)*H + (dsCropRows+dsCropExtra)*16*2 + dsCkvRows*H + dsCkvRows*16*2)
	p0, s0 := g128Sizes(H, dsTaps*H)
	p1, s1 := g128Sizes(dsCkvRows, H)
	total += c(p0 + s0 + p1 + s1 + 4<<16)
	total += c(w * dsTaps * H * 2)
	total += c(w*H*2)*2 + c(w*dsCkvRows*2)
	return total + 1<<20
}

// initDSpark allocates the drafter's buffers, derives the tensors the payload does not store
// and loads the layer table. targetRows is the decode kernel's table.
func (e *Engine) initDSpark(pay *payload.Payload, head *weights.Packed, targetRows []uint64) error {
	d := &e.ds
	raw, err := pay.Data("draft.meta")
	if err != nil {
		return err
	}
	var meta payload.DraftMeta
	if err := json.Unmarshal(raw, &meta); err != nil {
		return fmt.Errorf("drafter settings: %w", err)
	}
	if meta.Rows != dsCropRows || meta.Window <= 0 || meta.Window > dsWindowMax {
		return fmt.Errorf("unsupported drafter: %d crop rows, window %d", meta.Rows, meta.Window)
	}
	d.window = meta.Window

	const fp16Bytes = 2
	var aerr error
	alloc := func(dst **kfd.Buffer, size uint64, kind kfd.Kind) {
		if aerr == nil {
			*dst, aerr = e.dev.Alloc(size, kind)
		}
	}
	w := uint64(d.window)
	alloc(&d.layers, dsRows*dsLayerWords*8, kfd.VRAM)
	alloc(&d.ws, dsWorkspaceFloats*4, kfd.VRAM)
	alloc(&d.ctl, dsCtlBytes, kfd.VRAM)
	alloc(&d.base, dsMaxTokens*(dsCropRows+dsCropExtra)*4, kfd.VRAM)
	alloc(&d.rings, dsDraftLayers*dsRingBytes, kfd.VRAM)
	alloc(&d.locals, nConvLayers*dsConvRingBytes+(nFullLayers+nSlideLayers+dsDraftLayers)*dsFreshBytes, kfd.VRAM)
	tableEntries := uint64(e.cfg.MaxContext/dsRingBlock + 2)
	alloc(&d.ringTable, tableEntries*4, kfd.VRAM)
	alloc(&d.host, dsHostBytes, kfd.GTT)
	alloc(&d.tapX, w*dsTaps*H*fp16Bytes, kfd.VRAM)
	alloc(&d.tapC, w*H*fp16Bytes, kfd.VRAM)
	alloc(&d.tapC2, w*H*fp16Bytes, kfd.VRAM)
	alloc(&d.tapKV, w*dsCkvRows*fp16Bytes, kfd.VRAM)
	if aerr != nil {
		return fmt.Errorf("drafter buffers: %w", aerr)
	}
	// The kernel assumes zeroed rings and an empty barrier.
	e.uploadRaw(0, d.ws.VA, dsWorkspaceFloats)
	e.uploadRaw(0, d.ctl.VA, dsCtlBytes/4)
	e.uploadRaw(0, d.rings.VA, dsDraftLayers*dsRingBytes/4)
	e.uploadRaw(0, d.locals.VA, int32(d.locals.Size/4))
	table := make([]uint32, tableEntries)
	for i := range table {
		table[i] = uint32(i) % (dsRingPositions / dsRingBlock)
	}
	e.upload(d.ringTable.VA, table)
	if err := e.drain(kfd.ScopeAgent); err != nil {
		return fmt.Errorf("clearing drafter memory: %w", err)
	}

	if err := e.buildDraftWeights(head); err != nil {
		return err
	}
	e.buildDraftLayers(targetRows)
	e.buildDraftGlobals(&meta)
	if err := e.initDraftTrace(); err != nil {
		return err
	}
	return e.drain(kfd.ScopeAgent)
}

// scaleRow is the padded scale-row length of an H-input matrix, weights.SRow(H).
const scaleRow = 16

// entry returns the arena bytes of a tensor while the blob is still loaded.
func (e *Engine) entry(name string) ([]byte, error) {
	ent, ok := e.index[name]
	if !ok {
		return nil, fmt.Errorf("arena has no %s", name)
	}
	return e.blob[ent.Off : ent.Off+ent.Size], nil
}

// buildDraftWeights derives what the payload leaves out, on the same row-independent
// quantization as everything else: the head crop from the LM head rows plus the confidence
// row, and the context k/v projection from the drafter's mixer rows.
func (e *Engine) buildDraftWeights(head *weights.Packed) error {
	d := &e.ds
	ids, err := e.entry("crop_ids")
	if err != nil {
		return err
	}
	confRaw, err := e.entry("conf_row")
	if err != nil {
		return err
	}
	rows := dsCropRows + dsCropExtra
	tail := make([]uint16, dsCropExtra*H)
	for i := range H {
		tail[i] = binary.LittleEndian.Uint16(confRaw[2*i:])
	}
	tailPack := weights.Pack(tail, dsCropExtra, H)
	cropBytes := uint64(rows * H)
	cropScaleBytes := uint64(rows * scaleRow * 2)
	ckvBytes := uint64(dsCkvRows * H)
	ckvScaleBytes := uint64(dsCkvRows * scaleRow * 2)
	const align = 64 << 10
	up := func(n uint64) uint64 { return (n + align - 1) &^ (align - 1) }
	total := up(cropBytes) + up(cropScaleBytes) + up(ckvBytes) + ckvScaleBytes
	if d.derived, err = e.dev.Alloc(total, kfd.VRAM); err != nil {
		return fmt.Errorf("derived drafter tensors: %w", err)
	}
	host := make([]byte, total)
	cropW, cropS := host[:cropBytes], host[up(cropBytes):up(cropBytes)+cropScaleBytes]
	off := up(cropBytes) + up(cropScaleBytes)
	ckvW, ckvS := host[off:off+ckvBytes], host[off+up(ckvBytes):off+up(ckvBytes)+ckvScaleBytes]
	for r := range dsCropRows {
		id := int(binary.LittleEndian.Uint32(ids[4*r:]))
		copy(cropW[r*H:(r+1)*H], head.Biased[id*H:(id+1)*H])
		for i := range scaleRow {
			binary.LittleEndian.PutUint16(cropS[(r*scaleRow+i)*2:], head.Scales[id*scaleRow+i])
		}
	}
	copy(cropW[dsCropRows*H:], tailPack.Biased)
	for i, v := range tailPack.Scales {
		binary.LittleEndian.PutUint16(cropS[(dsCropRows*scaleRow+i)*2:], v)
	}
	for l := range dsDraftLayers {
		mixW, err := e.entry(fmt.Sprintf("d%d.mix.w", l))
		if err != nil {
			return err
		}
		mixS, err := e.entry(fmt.Sprintf("d%d.mix.s", l))
		if err != nil {
			return err
		}
		copy(ckvW[l*1024*H:(l+1)*1024*H], mixW[H*H:])
		copy(ckvS[l*1024*scaleRow*2:(l+1)*1024*scaleRow*2], mixS[H*scaleRow*2:])
	}
	if err := e.loadToVRAM(d.derived.VA, host); err != nil {
		return err
	}
	base := d.derived.VA
	e.vram["crop.w"] = base
	e.vram["crop.s"] = base + up(cropBytes)
	e.vram["ckv.w"] = base + off
	e.vram["ckv.s"] = base + off + up(ckvBytes)

	// GEMM layouts for projecting the prompt.
	fcW, err := e.entry("fc.w")
	if err != nil {
		return err
	}
	fcS, err := e.entry("fc.s")
	if err != nil {
		return err
	}
	fcRows := make([]byte, dsTaps*H*H)
	for r := range H {
		for b := range dsTaps {
			copy(fcRows[(r*dsTaps+b)*H:(r*dsTaps+b+1)*H], fcW[(b*H+r)*H:])
		}
	}
	p0, s0 := g128Sizes(H, dsTaps*H)
	p1, s1 := g128Sizes(dsCkvRows, H)
	gtotal := up(p0) + up(s0) + up(p1) + up(s1)
	if d.projWeights, err = e.dev.Alloc(gtotal, kfd.VRAM); err != nil {
		return fmt.Errorf("drafter projection weights: %w", err)
	}
	g := make([]byte, gtotal)
	fcSpec := matrixSpec{"fc", H, dsTaps * H, identity}
	fillPacked(g[:p0], fcRows, fcSpec)
	// One scale per 128 inputs: group g of the concatenated row is group g%16 of band g/16.
	fillGroupScales(g[up(p0):up(p0)+s0], H, dsTaps*H/weights.Group, func(r, grp int) uint16 {
		band, local := grp/16, grp%16
		at := ((band*H+r)*scaleRow + (local%4)*(scaleRow/4) + local/4) * 2
		return binary.LittleEndian.Uint16(fcS[at:])
	})
	ck := up(p0) + up(s0)
	ckvSpec := matrixSpec{"ckv", dsCkvRows, H, identity}
	fillPacked(g[ck:ck+p1], ckvW, ckvSpec)
	fillScales(g[ck+up(p1):ck+up(p1)+s1], ckvS, ckvSpec)
	if err := e.loadToVRAM(d.projWeights.VA, g); err != nil {
		return err
	}
	va := d.projWeights.VA
	d.fc = g128Weight{packed: va, scales: va + up(p0), outputs: H, columns: dsTaps * H}
	d.ckv = g128Weight{packed: va + ck, scales: va + ck + up(p1), outputs: dsCkvRows, columns: H}
	return nil
}

// buildDraftLayers writes the kernel's table: the target rows with their extra per-layer
// buffers, then the drafter rows.
func (e *Engine) buildDraftLayers(targetRows []uint64) {
	d := &e.ds
	rows := make([]uint64, dsRows*dsLayerWords)
	next := d.locals.VA
	take := func(n uint64) uint64 {
		va := next
		next += n
		return va
	}
	minusOne := ^uint64(0)
	for li := range Layers {
		r := rows[li*dsLayerWords : (li+1)*dsLayerWords]
		copy(r, targetRows[li*layerWords:(li+1)*layerWords])
		r[lTapSlot] = minusOne
		if s := tapSlot(li); s >= 0 {
			r[lTapSlot] = uint64(s)
		}
		if LayerKinds[li] == 0 {
			r[lRing] = take(dsConvRingBytes)
		} else {
			r[lFresh] = take(dsFreshBytes)
		}
	}
	for l := range dsDraftLayers {
		k := func(s string) uint64 { return e.arenaVA(fmt.Sprintf("d%d.%s", l, s)) }
		r := rows[(Layers+l)*dsLayerWords : (Layers+l+1)*dsLayerWords]
		r[lKind] = 3
		r[lOpNorm], r[lFfnNorm] = k("operator_norm"), k("ffn_norm")
		r[lWMix], r[lSMix] = k("mix.w"), k("mix.s")
		r[lWOut], r[lSOut] = k("out.w"), k("out.s")
		r[lW13], r[lS13] = k("w13.w"), k("w13.s")
		r[lW2], r[lS2] = k("w2.w"), k("w2.s")
		r[lQNorm], r[lKNorm] = k("q_norm"), k("k_norm")
		r[lRope], r[lRotary] = e.ropeFull.VA, D
		r[lKV] = d.rings.VA + uint64(l)*dsRingBytes
		r[lKVBlock], r[lKVHead], r[lKVSlot] = dsRingBlockBytes, Rec, KVH*Rec
		r[lBlockSize] = dsRingBlock
		r[lWindow] = uint64(d.window)
		r[lTapSlot] = minusOne
		r[lFresh] = take(dsFreshBytes)
	}
	e.upload(d.layers.VA, unsafe.Slice((*uint32)(unsafe.Pointer(&rows[0])), len(rows)*2))
}

func (e *Engine) buildDraftGlobals(meta *payload.DraftMeta) {
	d := &e.ds
	var attn int64
	for li, k := range LayerKinds {
		if k != 0 {
			attn |= 1 << li
		}
	}
	d.globals = dsGlobals{
		layers: d.layers.VA, embed: e.embed.VA, finalNorm: e.arenaVA("final_norm"),
		wHead: e.arenaVA("head.w"), sHead: e.arenaVA("head.s"), draftNorm: e.arenaVA("draft_norm"),
		wCrop: e.arenaVA("crop.w"), sCrop: e.arenaVA("crop.s"), cropIDs: e.arenaVA("crop_ids"),
		w2Crop: e.arenaVA("markov_w2.q"), w2Scale: e.arenaVA("markov_w2.s"),
		w1q: e.arenaVA("markov_w1.q"), w1s: e.arenaVA("markov_w1.s"), c2: e.arenaVA("conf_markov"),
		base: d.base.VA, tapNorms: e.arenaVA("tap_norms"),
		wFc: e.arenaVA("fc.w"), sFc: e.arenaVA("fc.s"), hiddenNorm: e.arenaVA("hidden_norm"),
		wCkv: e.arenaVA("ckv.w"), sCkv: e.arenaVA("ckv.s"), ws: d.ws.VA,
		barrier: d.ctl.VA + dsCtlBarrier, st: d.ctl.VA + dsCtlState, seen: e.seen.VA,
		rep: e.step.VA + stepRep, best: d.ctl.VA + dsCtlBest, out: d.ctl.VA + dsCtlOut,
		vocab: Vocab, cropRows: dsCropRows, maskToken: int64(meta.MaskToken), maxDrafts: dsMaxDrafts,
		attnMask: attn, confBias: meta.ConfBias, confThr: dsConfThreshold, invRep: 1,
	}
}

// captureTap normalizes the part of a tapped layer's output the drafter needs into its slot
// of the concatenated input of the context projection.
func (e *Engine) captureTap(slot int, h uint64, context, tokens int) {
	d := &e.ds
	first := max(context, d.tapFrom)
	rows := context + tokens - first
	if rows <= 0 {
		return
	}
	ra := struct {
		in       uint64
		inStride int64
		w, out   uint64
		stride   int64
	}{
		h + uint64(first-context)*H*2, H,
		e.arenaVA("tap_norms") + uint64(slot)*H*2,
		d.tapX.VA + uint64(first-d.tapFrom)*dsTaps*H*2 + uint64(slot)*H*2, dsTaps * H,
	}
	e.launch(e.ops.rms, uint32(rows), 1, 256, unsafe.Pointer(&ra), 40)
}

// projectPrompt runs the captured positions [first, end) through the context projection and
// stores them in the drafter's KV rings: fc, hidden norm, the k/v rows of each drafter layer,
// k norm and RoPE, int8. These are the operations of the prefill's own K/V path.
func (e *Engine) projectPrompt(first, end int) error {
	d := &e.ds
	rows := end - first
	e.g128(&d.fc, d.tapX.VA, dsTaps*H, rows, false, d.tapC.VA, H)
	ra := struct {
		in       uint64
		inStride int64
		w, out   uint64
		stride   int64
	}{d.tapC.VA, H, e.arenaVA("hidden_norm"), d.tapC2.VA, H}
	e.launch(e.ops.rms, uint32(rows), 1, 256, unsafe.Pointer(&ra), 40)
	e.g128(&d.ckv, d.tapC2.VA, H, rows, false, d.tapKV.VA, dsCkvRows)
	const fKColumn = 2048 // where kv_write looks for k inside a row
	for l := range dsDraftLayers {
		// The kernel addresses k at column 2048 and v 512 later; these rows hold them at l * 1024.
		ka := struct {
			f, kNorm, rope, kv                 uint64
			kvBlock, kvHead, kvSlot            int64
			table                              uint64
			blockSize, context, fStride, pad32 int32
		}{
			d.tapKV.VA + uint64(l*1024)*2 - fKColumn*2, e.arenaVA(fmt.Sprintf("d%d.k_norm", l)), e.ropeFull.VA,
			d.rings.VA + uint64(l)*dsRingBytes, dsRingBlockBytes, Rec, KVH * Rec,
			d.ringTable.VA, dsRingBlock, int32(first), dsCkvRows, 0,
		}
		e.launch(e.ops.kvWrite, uint32(rows)*KVH, 1, 64, unsafe.Pointer(&ka), 80)
	}
	return e.drain(kfd.ScopeAgent)
}

// beginDraft prepares the drafter context of a request whose prefill captured [first, end)
// of the prompt. Earlier positions stay usable when the ring still holds them for the same
// tokens, which is what a prefix-cache hit needs: its prefill replays only the tail.
func (e *Engine) beginDraft(seq *sequence, first, end int) error {
	d := &e.ds
	lo := first
	if first < end {
		if err := e.projectPrompt(first, end); err != nil {
			return err
		}
	}
	if keep := max(d.ringLo, d.ringEnd-dsRingPositions); keep < lo && d.ringEnd >= lo && lo <= len(seq.tokens) && lo <= len(d.ringTokens) &&
		sameTokens(seq.tokens[keep:lo], d.ringTokens[keep:lo]) {
		lo = keep
	}
	a := len(seq.tokens) - 1
	d.ctxNew, d.ctxPos, d.ctxLo = 0, int32(a), int32(lo)
	d.miss, d.skip = 0, 0
	d.ringLo, d.ringEnd = lo, end
	d.ringTokens = seq.tokens
	return nil
}

func sameTokens(a, b []int32) bool {
	if len(a) != len(b) {
		return false
	}
	for i := range a {
		if a[i] != b[i] {
			return false
		}
	}
	return true
}

// draftResult is what one launch hands back.
type draftResult struct {
	tokens                     []int32
	rounds, accepted, verified int
	boundary                   int   // the block boundary this launch ran to
	convEpoch                  int64 // checkpoint of the conv state there, valid if the launch reached it
}

// draftLaunch runs speculative rounds from the sequence's last token until it reaches the next
// block boundary. Verify passes may cross the boundary, so its position never shortens them:
// the blocks they reach are mapped here, and the kernel saves the conv state at the boundary
// from its history ring. The result says whether the boundary was reached with that state.
func (e *Engine) draftLaunch(seq *sequence, budget int, eos int32, rep float32) (draftResult, error) {
	d := &e.ds
	a := len(seq.tokens) - 1
	stop := min((a/BlockTokens+1)*BlockTokens, e.cfg.MaxContext)
	limit := min(stop+dsMaxTokens, e.cfg.MaxContext)
	budget = min(budget, dsMaxBudget)
	if err := e.ensureBlock(seq, (limit-1)/BlockTokens); err != nil {
		return draftResult{}, err
	}
	ckpt, epoch := e.cache.allocConv()

	var s [dsStateWords]uint32
	s[sA], s[sAnchor], s[sRounds] = uint32(a), uint32(seq.tokens[a]), dsRoundsPerLaunch
	s[sLimit], s[sBudget], s[sEOS] = uint32(limit), uint32(budget), uint32(eos)
	s[sCtxNew], s[sCtxPos], s[sCtxLo] = uint32(d.ctxNew), uint32(d.ctxPos), uint32(d.ctxLo)
	s[sRun] = 1
	if d.trace != nil {
		s[sRounds] = 4 // what the trace buffer holds whole
	}
	s[sMiss], s[sSkip] = uint32(d.miss), uint32(d.skip)
	s[sSkipRounds] = 1
	s[sStopAt], s[sCkptSlot] = uint32(stop), uint32(ckpt)
	e.upload(d.ctl.VA+dsCtlState, s[:])
	e.uploadRaw(0, d.ctl.VA+dsCtlBest, dsMaxTokens*2)
	e.uploadRaw(0, d.ctl.VA+dsCtlBarrier, 2)

	g := &d.globals
	g.maxDrafts = dsMaxDrafts
	if seq.longContext {
		g.maxDrafts = dsMaxDraftsLong
	}
	if d.trace != nil {
		e.armTrace(d.trace)
	}
	g.invRep = 1
	if rep != 0 {
		g.invRep = 1 / rep
	}
	e.q.Submit(&kfd.Dispatch{
		Kernel:  d.kernel,
		Grid:    [3]uint32{decodeBlocks * decodeThreads},
		Block:   [3]uint16{decodeThreads},
		Args:    unsafe.Pointer(g),
		ArgSize: unsafe.Sizeof(*g),
		Acquire: kfd.ScopeAgent,
		Release: kfd.ScopeAgent,
	})

	// The state and tokens cross to uncached host memory; the marker written after them
	// proves they arrived. See submitDecode for why nothing here widens a fence.
	box := (*int32)(unsafe.Add(d.host.Ptr(), dsHostMarker))
	atomic.StoreInt32(box, tokenPending)
	maxOut := int32(min(limit-a, budget))
	if d.trace != nil {
		e.fetchTrace(d.trace)
	}
	e.uploadRaw(d.ctl.VA+dsCtlState, d.host.VA+dsHostState, dsStateWords)
	e.uploadRaw(d.ctl.VA+dsCtlOut, d.host.VA+dsHostOut, maxOut)
	e.upload(d.host.VA+dsHostMarker, []uint32{1})
	if err := e.drain(kfd.ScopeAgent); err != nil {
		return draftResult{}, fmt.Errorf("speculative decode: %w", err)
	}
	if _, err := e.awaitToken(box); err != nil {
		return draftResult{}, fmt.Errorf("speculative decode: %w", err)
	}
	st := unsafe.Slice((*int32)(unsafe.Add(d.host.Ptr(), dsHostState)), dsStateWords)
	n := atomic.LoadInt32(&st[sCount])
	if n < 1 || n > maxOut {
		return draftResult{}, fmt.Errorf("speculative decode: %d tokens from a launch of at most %d", n, maxOut)
	}
	if d.trace != nil {
		d.trace.collect(int(st[sDone]))
	}
	d.ctxNew, d.ctxPos = st[sCtxNew], st[sCtxPos]
	d.miss, d.skip = st[sMiss], st[sSkip]
	d.ringEnd, d.ringTokens = int(d.ctxPos), seq.tokens
	out := unsafe.Slice((*int32)(unsafe.Add(d.host.Ptr(), dsHostOut)), n)
	tokens := make([]int32, n)
	copy(tokens, out)
	return draftResult{tokens, int(st[sDone]), int(st[sAccepted]), int(st[sVerified]), stop, epoch}, nil
}

// loadDraftKernel resolves the kernel in its code object.
func (e *Engine) loadDraftKernel() error {
	prog, err := e.dev.LoadCodeObject(kernels.DSpark)
	if err != nil {
		return fmt.Errorf("dspark code object: %w", err)
	}
	e.ds.kernel, err = prog.Kernel("dspark_kernel")
	return err
}

// buildHead quantizes the LM head from the embedding: same rows, int8 with one fp16 scale per
// 128 inputs. Only the embedding is stored in the payload.
func (e *Engine) buildHead(embed []byte) (*weights.Packed, error) {
	src := make([]uint16, len(embed)/2)
	weights.Parallel(len(src), func(i int) { src[i] = binary.LittleEndian.Uint16(embed[2*i:]) })
	p := weights.Pack(src, Vocab, H)
	const align = 64 << 10
	wBytes := uint64(len(p.Biased))
	sBytes := uint64(len(p.Scales)) * 2
	total := (wBytes+align-1)&^(align-1) + sBytes
	buf, err := e.dev.Alloc(total, kfd.VRAM)
	if err != nil {
		return nil, fmt.Errorf("LM head: %w", err)
	}
	host := make([]byte, total)
	copy(host, p.Biased)
	sOff := (wBytes + align - 1) &^ (align - 1)
	weights.Parallel(len(p.Scales), func(i int) {
		binary.LittleEndian.PutUint16(host[sOff+uint64(i)*2:], p.Scales[i])
	})
	if err := e.loadToVRAM(buf.VA, host); err != nil {
		return nil, err
	}
	e.vram["head.w"], e.vram["head.s"] = buf.VA, buf.VA+sOff
	e.headBuf = buf
	return p, nil
}

const headBytes = uint64(Vocab*H) + uint64(Vocab*16*2) + 2<<20
