package kfd

import (
	"fmt"
	"sync/atomic"
	"syscall"
	"time"
	"unsafe"
)

// MMIO accessors. atomic.Store is XCHG on amd64, a locked bus cycle that must not
// target device registers; x86 keeps uncached stores in order, so a plain MOV is
// both sufficient and correct here.
//
//go:noinline
func mmioStore32(p *uint32, v uint32) { *p = v }

//go:noinline
func mmioLoad32(p *uint32) uint32 { return *p }

//go:noinline
func mmioStore64(p *uint64, v uint64) { *p = v }

// AQL packet header fields (hsa.h).
const (
	packetDispatch    = 2
	packetBarrierAnd  = 3
	headerBarrier     = 1 << 8
	acquireShift      = 9
	releaseShift      = 11
	offWriteDispatch  = 56
	offLdsApertureHi  = 64
	offScrApertureHi  = 68
	offMaxCU          = 72
	offMaxWave        = 76
	offReadDispatch   = 128
	offReadFieldBase  = 136
	offProperties     = 180
	ringPackets       = 4096
	packetBytes       = 64
	signalBytes       = 64
	signalKindUser    = 1
	kernargPoolBytes  = 2 << 20
	kernargAlign      = 64
	signalPoolCount   = 64
	debugBytesPerWave = 32
)

// Every dispatch declares three dimensions, as ROCr does; unused axes have size 1.
const dispatchDims = 3

// FenceScope is an AQL memory fence scope.
type FenceScope uint16

const (
	ScopeNone   FenceScope = 0
	ScopeAgent  FenceScope = 1
	ScopeSystem FenceScope = 2
)

// Queue is one in-order AQL compute queue.
type Queue struct {
	dev      *Device
	id       uint32
	desc     *Buffer
	ring     *Buffer
	eop      *Buffer
	cwsr     *Buffer
	doorbell *uint64
	writeIdx uint64
	wptr     *uint64
	rptr     *uint64

	kernargs *Buffer
	kernOff  uint64
	signals  *Buffer
	sigNext  int
}

// Signal is a user-mode completion signal polled from host memory.
type Signal struct {
	VA    uint64
	value *int64
}

func (d *Device) debugMemorySize() uint64 {
	cu := d.Props["simd_count"] / d.Props["simd_per_cu"]
	waves := cu * 40
	if perEngine := d.Props["array_count"] / d.Props["simd_arrays_per_engine"] * 512; perEngine < waves {
		waves = perEngine
	}
	return (waves*debugBytesPerWave + 63) &^ 63
}

// NewQueue creates an AQL queue. KFD validates the exact size of each of these
// mappings against the topology, so none may be padded.
func (d *Device) NewQueue() (*Queue, error) {
	q := &Queue{dev: d}
	var err error
	if q.desc, err = d.Alloc(pageSize, GTT); err != nil {
		return nil, err
	}
	if q.ring, err = d.Alloc(ringPackets*packetBytes, GTT); err != nil {
		return nil, err
	}
	// 4 KiB is what KFD expects on gfx9 and is not exposed by the topology.
	if q.eop, err = d.alloc(pageSize, VRAMCode, pageSize); err != nil {
		return nil, err
	}
	cwsrSize, ctlSize := d.Props["cwsr_size"], d.Props["ctl_stack_size"]
	if cwsrSize == 0 || ctlSize == 0 {
		return nil, fmt.Errorf("topology has no CWSR sizes")
	}
	debugSize := d.debugMemorySize()
	total := (cwsrSize + debugSize + pageSize - 1) &^ (pageSize - 1)
	if q.cwsr, err = d.Alloc(total, GTT); err != nil {
		return nil, err
	}
	// HsaUserContextSaveAreaHeader; the kernel's trap handler fills in the rest.
	hdr := (*[10]uint32)(q.cwsr.Ptr())
	hdr[4] = uint32(cwsrSize)  // DebugOffset
	hdr[5] = uint32(debugSize) // DebugSize
	if q.kernargs, err = d.Alloc(kernargPoolBytes, GTT); err != nil {
		return nil, err
	}
	if q.signals, err = d.Alloc(signalPoolCount*signalBytes, GTT); err != nil {
		return nil, err
	}

	base := uintptr(q.desc.VA)
	q.wptr = (*uint64)(unsafe.Pointer(base + offWriteDispatch))
	q.rptr = (*uint64)(unsafe.Pointer(base + offReadDispatch))
	*(*uint32)(unsafe.Pointer(base + offLdsApertureHi)) = uint32(d.Apertures.LdsBase >> 32)
	*(*uint32)(unsafe.Pointer(base + offScrApertureHi)) = uint32(d.Apertures.ScratchBase >> 32)
	*(*uint32)(unsafe.Pointer(base + offMaxCU)) = uint32(d.Props["simd_count"]/d.Props["simd_per_cu"]) - 1
	*(*uint32)(unsafe.Pointer(base + offMaxWave)) = uint32(d.Props["max_waves_per_simd"]*d.Props["simd_per_cu"]) - 1
	*(*uint32)(unsafe.Pointer(base + offReadFieldBase)) = offReadDispatch
	*(*uint32)(unsafe.Pointer(base + offProperties)) = 1 << 1 // is_ptr64

	// The ring must not contain a valid packet before the CP starts reading it.
	ring := unsafe.Slice((*uint32)(q.ring.Ptr()), ringPackets*packetBytes/4)
	for i := 0; i < ringPackets; i++ {
		ring[i*packetBytes/4] = 1 // HSA_PACKET_TYPE_INVALID
	}

	args := createQueueArgs{
		RingBase:           q.ring.VA,
		WritePtr:           q.desc.VA + offWriteDispatch,
		ReadPtr:            q.desc.VA + offReadDispatch,
		RingSize:           uint32(q.ring.Size),
		GpuID:              d.GpuID,
		QueueType:          queueTypeComputeAQL,
		QueuePercent:       queuePercent,
		QueuePriority:      queuePriorityNormal,
		EopBase:            q.eop.VA,
		EopSize:            q.eop.Size,
		CtxSaveRestore:     q.cwsr.VA,
		CtxSaveRestoreSize: uint32(cwsrSize),
		CtlStackSize:       uint32(ctlSize),
	}
	if err := ioctl(d.kfd, iocCreateQueue, unsafe.Pointer(&args)); err != nil {
		return nil, fmt.Errorf("KFD create queue: %w (cwsr %#x ctl %#x total %#x)", err, cwsrSize, ctlSize, total)
	}
	q.id = args.QueueID
	const doorbellPage = 8192
	pageOff := args.DoorbellOffset &^ (doorbellPage - 1)
	db, err := mmap(0, doorbellPage, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED, d.kfd, pageOff)
	if err != nil {
		return nil, fmt.Errorf("map doorbell: %w", err)
	}
	q.doorbell = (*uint64)(unsafe.Pointer(db + uintptr(args.DoorbellOffset&(doorbellPage-1))))
	return q, nil
}

// Destroy stops the queue. Exiting the process has the same effect.
func (q *Queue) Destroy() error {
	a := destroyQueueArgs{QueueID: q.id}
	return ioctl(q.dev.kfd, iocDestroyQueue, unsafe.Pointer(&a))
}

// NewSignal returns a signal from the queue's pool, reset to value 1.
func (q *Queue) NewSignal() *Signal {
	i := q.sigNext % signalPoolCount
	q.sigNext++
	base := uintptr(q.signals.VA) + uintptr(i)*signalBytes
	*(*int64)(unsafe.Pointer(base)) = signalKindUser
	s := &Signal{VA: uint64(base), value: (*int64)(unsafe.Pointer(base + 8))}
	s.Reset()
	return s
}

// Reset arms the signal for one more dispatch.
func (s *Signal) Reset() { atomic.StoreInt64(s.value, 1) }

// Done reports completion without blocking.
func (s *Signal) Done() bool { return atomic.LoadInt64(s.value) <= 0 }

// Dispatch describes one kernel launch. Grid and Block are in work-items.
type Dispatch struct {
	Kernel  *Kernel
	Grid    [3]uint32
	Block   [3]uint16
	Args    unsafe.Pointer // user kernel arguments, copied to the kernarg segment
	ArgSize uintptr
	Acquire FenceScope
	Release FenceScope
}

func (q *Queue) allocKernarg(n uint64) (uint64, unsafe.Pointer) {
	n = (n + kernargAlign - 1) &^ (kernargAlign - 1)
	if q.kernOff+n > q.kernargs.Size {
		q.kernOff = 0
	}
	off := q.kernOff
	q.kernOff += n
	return q.kernargs.VA + off, unsafe.Pointer(uintptr(q.kernargs.VA) + uintptr(off))
}

// Submit enqueues the dispatch in order (barrier bit set) and rings the doorbell.
func (q *Queue) Submit(c *Dispatch) {
	k := c.Kernel
	block := c.Block
	grid := c.Grid
	for i := range grid {
		if grid[i] == 0 {
			grid[i] = 1
		}
		if block[i] == 0 {
			block[i] = 1
		}
	}
	if c.ArgSize > uintptr(k.KernargSize) {
		panic(fmt.Sprintf("kernel %s: %d bytes of arguments, segment holds %d", k.Name, c.ArgSize, k.KernargSize))
	}
	kaVA, ka := q.allocKernarg(uint64(k.KernargSize))
	dst := unsafe.Slice((*byte)(ka), k.KernargSize)
	clear(dst)
	if c.ArgSize > 0 {
		copy(dst, unsafe.Slice((*byte)(c.Args), c.ArgSize))
	}
	k.writeHidden(dst, grid, block)

	for q.writeIdx-atomic.LoadUint64(q.rptr) >= ringPackets {
		// Ring full: the CP is far behind, which only long prefills can reach.
		time.Sleep(10 * time.Microsecond)
	}
	idx := q.writeIdx
	p := unsafe.Pointer(uintptr(q.ring.VA) + uintptr(idx%ringPackets)*packetBytes)
	pk := (*[8]uint64)(p)
	w := (*[16]uint32)(p)
	// Dword 0 is published last: it is what makes the packet valid for the CP.
	w[1] = uint32(block[0]) | uint32(block[1])<<16
	w[2] = uint32(block[2])
	w[3], w[4], w[5] = grid[0], grid[1], grid[2]
	w[6] = k.PrivateSize
	w[7] = k.GroupSize
	pk[4] = k.Object
	pk[5] = kaVA
	pk[6], pk[7] = 0, 0 // completion is signalled by a barrier packet
	header := uint32(packetDispatch) | headerBarrier |
		uint32(c.Acquire)<<acquireShift | uint32(c.Release)<<releaseShift |
		uint32(dispatchDims)<<16
	atomic.StoreUint32(&w[0], header)
	q.writeIdx = idx + 1
	atomic.StoreUint64(q.wptr, q.writeIdx)
	// The doorbell takes the id of the last valid packet, not the write index.
	mmioStore64(q.doorbell, idx)
}

// SubmitBarrier enqueues a barrier-AND packet that signals s once every earlier
// packet has completed. Its release scope decides how far their writes are flushed.
func (q *Queue) SubmitBarrier(s *Signal, release FenceScope) {
	for q.writeIdx-atomic.LoadUint64(q.rptr) >= ringPackets {
		time.Sleep(10 * time.Microsecond)
	}
	idx := q.writeIdx
	p := unsafe.Pointer(uintptr(q.ring.VA) + uintptr(idx%ringPackets)*packetBytes)
	pk := (*[8]uint64)(p)
	pk[1], pk[2], pk[3], pk[4], pk[5], pk[6] = 0, 0, 0, 0, 0, 0
	pk[7] = s.VA
	header := uint32(packetBarrierAnd) | headerBarrier | uint32(release)<<releaseShift
	atomic.StoreUint32((*uint32)(p), header)
	q.writeIdx = idx + 1
	atomic.StoreUint64(q.wptr, q.writeIdx)
	mmioStore64(q.doorbell, idx)
}

// Wait spins until the signal completes, then falls back to sleeping once it has waited
// long enough that the extra latency is negligible. It gives up at timeout so a wedged
// GPU surfaces as an error. Decode steps stay inside the spin: a sleep overshoots by tens
// of microseconds, which is visible at thousands of tokens per second of budget.
func (q *Queue) Wait(s *Signal, timeout time.Duration) error {
	const spinFor = 20 * time.Millisecond
	start := time.Now()
	for {
		for i := 0; i < 1024; i++ {
			if s.Done() {
				return nil
			}
		}
		elapsed := time.Since(start)
		if elapsed < spinFor {
			continue
		}
		if elapsed > timeout {
			return fmt.Errorf("GPU timeout after %v: read %d of %d packets; check dmesg for amdgpu faults", timeout, atomic.LoadUint64(q.rptr), q.writeIdx)
		}
		time.Sleep(50 * time.Microsecond)
	}
}
