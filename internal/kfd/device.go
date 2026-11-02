package kfd

import (
	"fmt"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"syscall"
	"time"
	"unsafe"
)

const (
	vega20TargetVersion = 90006
	pageSize            = 4096
	hugePage            = 2 << 20

	// The shared CPU/GPU virtual address range ROCr gives KFD on this ASIC.
	svmBase  = 0x200000
	svmLimit = 0x7fffffffffff + 1

	// How long Open lets the kernel's page-table updates finish; see its comment.
	vmSettleTime = 200 * time.Millisecond
)

// Device is an open MI50 with its GPU virtual address space acquired.
type Device struct {
	kfd, drm  int
	GpuID     uint32
	Node      string
	Props     map[string]uint64
	Apertures processDeviceApertures
	hdp       *uint32
}

func readProps(path string) (map[string]uint64, error) {
	raw, err := os.ReadFile(path)
	if err != nil {
		return nil, err
	}
	out := map[string]uint64{}
	for _, line := range strings.Split(string(raw), "\n") {
		f := strings.Fields(line)
		if len(f) == 2 {
			if v, err := strconv.ParseUint(f[1], 10, 64); err == nil {
				out[f[0]] = v
			}
		}
	}
	return out, nil
}

// Open finds the first gfx906 node in the KFD topology and binds to it.
func Open() (*Device, error) {
	nodes, err := filepath.Glob("/sys/class/kfd/kfd/topology/nodes/*")
	if err != nil {
		return nil, err
	}
	d := &Device{}
	for _, n := range nodes {
		p, err := readProps(n + "/properties")
		if err != nil || p["gfx_target_version"] != vega20TargetVersion {
			continue
		}
		raw, err := os.ReadFile(n + "/gpu_id")
		if err != nil {
			return nil, err
		}
		id, err := strconv.ParseUint(strings.TrimSpace(string(raw)), 10, 32)
		if err != nil {
			return nil, err
		}
		d.GpuID, d.Node, d.Props = uint32(id), n, p
		break
	}
	if d.GpuID == 0 {
		return nil, fmt.Errorf("no gfx906 device in the KFD topology")
	}
	if d.kfd, err = syscall.Open("/dev/kfd", syscall.O_RDWR|syscall.O_CLOEXEC, 0); err != nil {
		return nil, fmt.Errorf("open /dev/kfd: %w (needs root or the render group)", err)
	}
	var ver getVersionArgs
	if err := ioctl(d.kfd, iocGetVersion, unsafe.Pointer(&ver)); err != nil {
		return nil, fmt.Errorf("KFD get version: %w", err)
	}
	if ver.Major != 1 {
		return nil, fmt.Errorf("unsupported KFD ABI %d.%d", ver.Major, ver.Minor)
	}
	render := fmt.Sprintf("/dev/dri/renderD%d", d.Props["drm_render_minor"])
	if d.drm, err = syscall.Open(render, syscall.O_RDWR|syscall.O_CLOEXEC, 0); err != nil {
		return nil, fmt.Errorf("open %s: %w", render, err)
	}
	// Two steps: newer kernels reject a count above the real number of GPU nodes.
	var ga getProcessAperturesNewArgs
	if err := ioctl(d.kfd, iocGetApertures, unsafe.Pointer(&ga)); err != nil {
		return nil, fmt.Errorf("KFD aperture count: %w", err)
	}
	if ga.NumOfNodes == 0 {
		return nil, fmt.Errorf("KFD reports no GPU for this process")
	}
	aps := make([]processDeviceApertures, ga.NumOfNodes)
	ga.Ptr = uint64(uintptr(unsafe.Pointer(&aps[0])))
	if err := ioctl(d.kfd, iocGetApertures, unsafe.Pointer(&ga)); err != nil {
		return nil, fmt.Errorf("KFD apertures: %w", err)
	}
	for i := 0; i < int(ga.NumOfNodes); i++ {
		if aps[i].GpuID == d.GpuID {
			d.Apertures = aps[i]
		}
	}
	if d.Apertures.GpuID == 0 {
		return nil, fmt.Errorf("gpu %d has no process aperture", d.GpuID)
	}
	// Opening the render node starts asynchronous page-table updates in the kernel. If
	// KFD acquires the VM while they run, they overwrite the page directory entries it
	// writes for its trap handler. Nothing fails until another process changes its queues
	// and a running wave is preempted: the wave cannot fetch the trap handler and the queue
	// hangs. There is no ioctl to wait for those updates. 1 ms was enough on an idle GPU.
	time.Sleep(vmSettleTime)
	vm := acquireVMArgs{DrmFd: uint32(d.drm), GpuID: d.GpuID}
	if err := ioctl(d.kfd, iocAcquireVM, unsafe.Pointer(&vm)); err != nil {
		return nil, fmt.Errorf("KFD acquire VM: %w", err)
	}
	mp := setMemoryPolicyArgs{
		AltBase: svmBase, AltSize: svmLimit - svmBase,
		GpuID: d.GpuID, DefaultPolicy: cachePolicyNonCoherent, AltPolicy: cachePolicyCoherent,
	}
	if err := ioctl(d.kfd, iocSetMemoryPolicy, unsafe.Pointer(&mp)); err != nil {
		return nil, fmt.Errorf("KFD memory policy: %w", err)
	}
	// XNACK must be off. The kernels are built for gfx906 without replay; when it is on,
	// translation retries under heavy memory load replay memory instructions, which
	// silently corrupts results once enough compute units run at once.
	xnack := int32(0)
	if err := ioctl(d.kfd, iocSetXnackMode, unsafe.Pointer(&xnack)); err != nil {
		return nil, fmt.Errorf("KFD disable XNACK: %w", err)
	}
	rt := runtimeEnableArgs{ModeMask: 1}
	if err := ioctl(d.kfd, iocRuntimeEnable, unsafe.Pointer(&rt)); err != nil {
		return nil, fmt.Errorf("KFD runtime enable: %w", err)
	}
	// HDP flush register page: CPU writes through the BAR sit in the host data
	// path until flushed, and the GPU may otherwise read stale VRAM.
	off := uint64(d.GpuID) << mmapGpuIDShift
	hdp, err := mmap(0, pageSize, syscall.PROT_READ|syscall.PROT_WRITE, syscall.MAP_SHARED, d.kfd, off)
	if err != nil {
		return nil, fmt.Errorf("map HDP flush register: %w", err)
	}
	d.hdp = (*uint32)(unsafe.Pointer(hdp))
	return d, nil
}

// VRAMBytes is the total local memory reported by the topology.
func (d *Device) VRAMBytes() uint64 {
	p, err := readProps(d.Node + "/mem_banks/0/properties")
	if err != nil {
		return 0
	}
	return p["size_in_bytes"]
}

// FreeVRAM reports the card's unused VRAM across all processes.
func (d *Device) FreeVRAM() (uint64, error) {
	dir := fmt.Sprintf("/sys/class/drm/renderD%d/device", d.Props["drm_render_minor"])
	read := func(name string) (uint64, error) {
		raw, err := os.ReadFile(dir + "/" + name)
		if err != nil {
			return 0, err
		}
		return strconv.ParseUint(strings.TrimSpace(string(raw)), 10, 64)
	}
	total, err := read("mem_info_vram_total")
	if err != nil {
		return 0, fmt.Errorf("read VRAM size: %w", err)
	}
	used, err := read("mem_info_vram_used")
	if err != nil {
		return 0, fmt.Errorf("read VRAM use: %w", err)
	}
	if used > total {
		return 0, nil
	}
	return total - used, nil
}

// Buffer is memory mapped at the same virtual address for the CPU and the GPU.
type Buffer struct {
	VA   uint64
	Size uint64
}

// Bytes views the buffer from the CPU. Only GTT and VRAMCode buffers have a CPU mapping.
func (b *Buffer) Bytes() []byte {
	return unsafe.Slice((*byte)(unsafe.Pointer(uintptr(b.VA))), b.Size)
}

func (b *Buffer) Ptr() unsafe.Pointer { return unsafe.Pointer(uintptr(b.VA)) }

// Kind selects the placement of an allocation.
type Kind int

const (
	// VRAM is coarse-grained local memory with no CPU mapping, as ROCr allocates tensors.
	// Data reaches it through the GPU (see Engine.upload): host writes through the BAR
	// bypass the L2, which can keep serving older values for those addresses.
	VRAM Kind = iota
	// VRAMCode is CPU-writable and executable, for code objects written before first use.
	VRAMCode
	// GTT is cached, snooped host memory, uncached in the GPU's L2, so both sides
	// see each other's writes without cache maintenance.
	GTT
)

func (d *Device) alloc(size uint64, kind Kind, align uint64) (*Buffer, error) {
	if size == 0 || size%pageSize != 0 {
		return nil, fmt.Errorf("alloc size %#x is not a positive multiple of the page size", size)
	}
	span := uintptr(size + align)
	base, err := mmap(0, span, syscall.PROT_NONE, syscall.MAP_PRIVATE|syscall.MAP_ANONYMOUS|syscall.MAP_NORESERVE, -1, 0)
	if err != nil {
		return nil, err
	}
	va := (uint64(base) + align - 1) &^ (align - 1)
	// These match what ROCr requests for the same kinds of memory.
	flags := memWritable | memNoSubstitute
	switch kind {
	case VRAM:
		flags |= memVRAM
	case VRAMCode:
		flags |= memVRAM | memPublic | memExecutable
	case GTT:
		flags |= memGTT | memCoherent | memUncached | memExecutable
	}
	args := allocMemoryArgs{VA: va, Size: size, GpuID: d.GpuID, Flags: flags}
	if err := ioctl(d.kfd, iocAllocMemory, unsafe.Pointer(&args)); err != nil {
		munmap(base, span)
		return nil, fmt.Errorf("KFD alloc %d MiB (kind %d): %w", size>>20, kind, err)
	}
	if kind != VRAM {
		if _, err := mmap(uintptr(va), uintptr(size), syscall.PROT_READ|syscall.PROT_WRITE,
			syscall.MAP_SHARED|syscall.MAP_FIXED, d.drm, args.MmapOffset); err != nil {
			return nil, err
		}
	}
	ids := []uint32{d.GpuID}
	ma := mapMemoryArgs{Handle: args.Handle, DeviceIDs: uint64(uintptr(unsafe.Pointer(&ids[0]))), NDevices: 1}
	if err := ioctl(d.kfd, iocMapMemory, unsafe.Pointer(&ma)); err != nil {
		return nil, fmt.Errorf("KFD map %d MiB: %w", size>>20, err)
	}
	return &Buffer{VA: va, Size: size}, nil
}

// Alloc rounds size up to pages. VRAM of 2 MiB or more is 2 MiB aligned, so the
// GPU page tables use huge pages: the kernels stream gigabytes per token and
// TLB misses are directly visible in tokens/s.
func (d *Device) Alloc(size uint64, kind Kind) (*Buffer, error) {
	size = (size + pageSize - 1) &^ (pageSize - 1)
	align := uint64(pageSize)
	if kind != GTT && size >= hugePage {
		size = (size + hugePage - 1) &^ (hugePage - 1)
		align = hugePage
	}
	return d.alloc(size, kind, align)
}

// FlushHDP makes every prior CPU write through the BAR visible to the GPU.
func (d *Device) FlushHDP() {
	mmioStore32(d.hdp, 1)
	_ = mmioLoad32(d.hdp)
}
