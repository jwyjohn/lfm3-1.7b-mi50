// Package kfd is a minimal pure-Go driver for AMD's KFD compute interface
// (/dev/kfd). It replaces libhsakmt and ROCr for the only device this program
// supports: a Vega 20 (gfx906) MI50. No cgo, no shared libraries.
package kfd

import (
	"fmt"
	"syscall"
	"unsafe"
)

const (
	iocWrite = 1
	iocRead  = 2
)

func ioc(dir, nr, size uintptr) uintptr {
	return dir<<30 | size<<16 | uintptr('K')<<8 | nr
}

type getVersionArgs struct {
	Major, Minor uint32
}

type createQueueArgs struct {
	RingBase, WritePtr, ReadPtr, DoorbellOffset uint64
	RingSize, GpuID, QueueType, QueuePercent    uint32
	QueuePriority, QueueID                      uint32
	EopBase, EopSize, CtxSaveRestore            uint64
	CtxSaveRestoreSize, CtlStackSize            uint32
	SdmaEngineID, MetadataRingSize              uint32
}

type destroyQueueArgs struct {
	QueueID, Pad uint32
}

type setMemoryPolicyArgs struct {
	AltBase, AltSize                              uint64
	GpuID, DefaultPolicy, AltPolicy, MiscProcFlag uint32
}

type processDeviceApertures struct {
	LdsBase, LdsLimit, ScratchBase, ScratchLimit, GpuvmBase, GpuvmLimit uint64
	GpuID, Pad                                                          uint32
}

type getProcessAperturesNewArgs struct {
	Ptr        uint64
	NumOfNodes uint32
	Pad        uint32
}

type acquireVMArgs struct {
	DrmFd, GpuID uint32
}

type allocMemoryArgs struct {
	VA, Size, Handle, MmapOffset uint64
	GpuID, Flags                 uint32
}

type runtimeEnableArgs struct {
	RDebug       uint64
	ModeMask     uint32
	Capabilities uint32
}

type mapMemoryArgs struct {
	Handle, DeviceIDs uint64
	NDevices, NSucc   uint32
}

var (
	iocGetVersion      = ioc(iocRead, 0x01, unsafe.Sizeof(getVersionArgs{}))
	iocCreateQueue     = ioc(iocRead|iocWrite, 0x02, unsafe.Sizeof(createQueueArgs{}))
	iocDestroyQueue    = ioc(iocRead|iocWrite, 0x03, unsafe.Sizeof(destroyQueueArgs{}))
	iocSetMemoryPolicy = ioc(iocWrite, 0x04, unsafe.Sizeof(setMemoryPolicyArgs{}))
	iocGetApertures    = ioc(iocRead|iocWrite, 0x14, unsafe.Sizeof(getProcessAperturesNewArgs{}))
	iocAcquireVM       = ioc(iocWrite, 0x15, unsafe.Sizeof(acquireVMArgs{}))
	iocAllocMemory     = ioc(iocRead|iocWrite, 0x16, unsafe.Sizeof(allocMemoryArgs{}))
	iocMapMemory       = ioc(iocRead|iocWrite, 0x18, unsafe.Sizeof(mapMemoryArgs{}))
	iocSetXnackMode    = ioc(iocRead|iocWrite, 0x21, 4)
	iocRuntimeEnable   = ioc(iocRead|iocWrite, 0x25, unsafe.Sizeof(runtimeEnableArgs{}))
)

// Allocation flags (linux/kfd_ioctl.h).
const (
	memVRAM         uint32 = 1 << 0
	memGTT          uint32 = 1 << 1
	memWritable     uint32 = 1 << 31
	memExecutable   uint32 = 1 << 30
	memPublic       uint32 = 1 << 29
	memNoSubstitute uint32 = 1 << 28
	memCoherent     uint32 = 1 << 26
	memUncached     uint32 = 1 << 25
)

const (
	queueTypeComputeAQL = 2
	queuePercent        = 100
	queuePriorityNormal = 7

	cachePolicyCoherent    = 0
	cachePolicyNonCoherent = 1

	mmapGpuIDShift = 46
)

func ioctl(fd int, req uintptr, arg unsafe.Pointer) error {
	for {
		_, _, e := syscall.Syscall(syscall.SYS_IOCTL, uintptr(fd), req, uintptr(arg))
		switch e {
		case 0:
			return nil
		case syscall.EINTR, syscall.EAGAIN:
			continue
		}
		return e
	}
}

// mmap wraps the raw syscall: unlike syscall.Mmap it can place fixed mappings
// (GPU and CPU must see one address) and does not track them in the Go runtime.
func mmap(addr, length uintptr, prot, flags, fd int, offset uint64) (uintptr, error) {
	r, _, e := syscall.Syscall6(syscall.SYS_MMAP, addr, length, uintptr(prot), uintptr(flags), uintptr(fd), uintptr(offset))
	if e != 0 {
		return 0, fmt.Errorf("mmap(len=%#x off=%#x): %w", length, offset, e)
	}
	return r, nil
}

func munmap(addr, length uintptr) error {
	_, _, e := syscall.Syscall(syscall.SYS_MUNMAP, addr, length, 0)
	if e != 0 {
		return e
	}
	return nil
}
