package model

import (
	"time"
	"unsafe"

	"lfm3-1.7b-mi50/internal/kfd"
)

// Diagnostics used by cmd/diag. The server does not call any of this.

// memrealtimeHz is the rate of the s_memrealtime counter the decode kernel's trace reads.
const memrealtimeHz = 25e6

// WorkspaceRegion is one kernel phase's output in the decode workspace, in floats.
type WorkspaceRegion struct {
	Name string
	Off  int
	Len  int
}

// WorkspaceFloats is the size of the decode workspace.
const WorkspaceFloats = workspaceBytes / 4

// WorkspaceRegions lists the phase outputs, as laid out in kernels/decode.hip.
var WorkspaceRegions = []WorkspaceRegion{
	{"h0", 0, H}, {"h1", H, H}, {"mix", 2 * H, 6144}, {"y", 10240, H}, {"o", 12288, H},
	{"a", 14336, I}, {"d", 22016, H}, {"bcx", 24064, 6144}, {"g13", 30208, 2 * I},
}

func (e *Engine) resetConv(slot int32) {
	slotBytes := uint64(H * ConvState * 2)
	layerBytes := uint64(e.cfg.ConvSlots) * slotBytes
	for i := uint64(0); i < nConvLayers; i++ {
		e.uploadRaw(0, e.convState.VA+i*layerBytes+uint64(slot)*slotBytes, int32(slotBytes/4))
	}
}

// TraceDecode pushes a fixed token sequence through the decode kernel, without sampling,
// calling after(step) once each step's raw logits are in HostLogits.
func (e *Engine) TraceDecode(tokens []int32, after func(step int)) error {
	e.resetConv(0)
	e.InitSeen(nil, 1)
	for i, t := range tokens {
		b, off := i/BlockTokens, i%BlockTokens
		if off == 0 {
			e.SetBlock(b, int32(b), int32(b%e.cfg.SlideBlocks))
		}
		e.writeStep(t, i+1, int64(b*BlockTokens+off), int64((b%e.cfg.SlideBlocks)*BlockTokens+off), 0)
		e.globals[15] = SampleOff
		e.submitDecode()
		if err := e.finishLogits(false); err != nil {
			return err
		}
		after(i)
	}
	return nil
}

// ReadWorkspace copies the decode kernel's workspace to dst.
func (e *Engine) ReadWorkspace(dst []float32) error {
	a := struct {
		src, dst uint64
		bytes    int64
	}{e.ws.VA, e.blitWindow.VA, workspaceBytes}
	e.launch(e.ops.blit, (workspaceBytes+4095)/4096, 1, 256, unsafe.Pointer(&a), 24)
	if err := e.drain(kfd.ScopeSystem); err != nil {
		return err
	}
	copy(dst, unsafe.Slice((*float32)(e.blitWindow.Ptr()), WorkspaceFloats))
	return nil
}

// PrefillLogits prefills prompt from scratch and returns the host copy of the last
// token's logits. The slice is overwritten by the next call.
func (e *Engine) PrefillLogits(prompt []int32) ([]float32, error) {
	seq := &sequence{tokens: prompt, hash: saltSeed("diag")}
	defer e.release(seq)
	if err := e.prefill(seq, 0, len(prompt)); err != nil {
		return nil, err
	}
	return e.HostLogits(), nil
}

// MeasureDecode runs kernel-sampled decode steps at a fixed context length, over zeroed
// KV, and returns mean milliseconds inside the kernel and on the host wall clock. The
// difference is what the driver adds per token.
func (e *Engine) MeasureDecode(context, steps int) (kernelMs, wallMs float64, err error) {
	tr, err := e.dev.Alloc(8192, kfd.GTT)
	if err != nil {
		return 0, 0, err
	}
	for b := 0; b <= (context+steps)/BlockTokens; b++ {
		e.SetBlock(b, int32(b), int32(b%e.cfg.SlideBlocks))
	}
	e.InitSeen(nil, 1.1)
	e.globals[12] = tr.VA
	defer func() { e.globals[12] = 0 }()
	ts := unsafe.Slice((*int64)(tr.Ptr()), 1024)
	const warmup = 5
	for i := 0; i < steps; i++ {
		pos := context + i
		b, off := pos/BlockTokens, pos%BlockTokens
		clear(ts)
		start := time.Now()
		if _, err := e.DecodeStep(int32(1000+i), pos+1, int64(b*BlockTokens+off),
			int64((b%e.cfg.SlideBlocks)*BlockTokens+off), 0, SampleRound16); err != nil {
			return 0, 0, err
		}
		wall := time.Since(start).Seconds() * 1e3
		last := 0
		for last+1 < len(ts) && ts[last+1] != 0 {
			last++
		}
		if i >= warmup {
			kernelMs += float64(ts[last]-ts[0]) / memrealtimeHz * 1e3
			wallMs += wall
		}
	}
	n := float64(steps - warmup)
	return kernelMs / n, wallMs / n, nil
}
