package kfd

import (
	"bytes"
	"debug/elf"
	"encoding/binary"
	"fmt"
	"strings"
)

// Kernel is a loaded kernel descriptor plus the metadata needed to launch it.
type Kernel struct {
	Name        string
	Object      uint64 // GPU address of the 64-byte kernel descriptor
	KernargSize uint32
	GroupSize   uint32
	PrivateSize uint32
	hidden      []hiddenArg
}

type hiddenArg struct {
	kind string
	off  uint32
}

// writeHidden fills the implicit arguments HIP kernels read. Global offsets stay zero;
// hostcall, heap and queue slots stay null because no kernel here prints or allocates.
func (k *Kernel) writeHidden(dst []byte, grid [3]uint32, block [3]uint16) {
	le := binary.LittleEndian
	for _, h := range k.hidden {
		b := dst[h.off:]
		sep := strings.LastIndexByte(h.kind, '_')
		field, axis := h.kind[:sep], int(h.kind[sep+1])-'x'
		switch field {
		case "hidden_block_count":
			le.PutUint32(b, grid[axis]/uint32(block[axis]))
		case "hidden_group_size":
			le.PutUint16(b, block[axis])
		case "hidden_remainder":
			le.PutUint16(b, uint16(grid[axis]%uint32(block[axis])))
		case "hidden_grid":
			le.PutUint16(b, dispatchDims)
		}
	}
}

// Program is a code object loaded into GPU-executable memory.
type Program struct {
	kernels map[string]*Kernel
}

// Relocation types of the AMDGPU ELF ABI.
const (
	rAMDGPUAbs64      = 2
	rAMDGPURelative64 = 13
)

// LoadCodeObject maps an amdgcn ELF code object into VRAM and resolves its kernels.
func (d *Device) LoadCodeObject(image []byte) (*Program, error) {
	f, err := elf.NewFile(bytes.NewReader(image))
	if err != nil {
		return nil, fmt.Errorf("code object: %w", err)
	}
	var span uint64
	for _, p := range f.Progs {
		if p.Type == elf.PT_LOAD && p.Vaddr+p.Memsz > span {
			span = p.Vaddr + p.Memsz
		}
	}
	if span == 0 {
		return nil, fmt.Errorf("code object has no loadable segment")
	}
	buf, err := d.Alloc(span, VRAMCode)
	if err != nil {
		return nil, err
	}
	mem := make([]byte, buf.Size)
	for _, p := range f.Progs {
		if p.Type != elf.PT_LOAD || p.Filesz == 0 {
			continue
		}
		if _, err := p.ReadAt(mem[p.Vaddr:p.Vaddr+p.Filesz], 0); err != nil {
			return nil, fmt.Errorf("segment at %#x: %w", p.Vaddr, err)
		}
	}
	syms, err := f.DynamicSymbols()
	if err != nil {
		return nil, fmt.Errorf("dynamic symbols: %w", err)
	}
	for _, s := range f.Sections {
		if s.Type != elf.SHT_RELA {
			continue
		}
		raw, err := s.Data()
		if err != nil {
			return nil, err
		}
		for i := 0; i+24 <= len(raw); i += 24 {
			off := binary.LittleEndian.Uint64(raw[i:])
			info := binary.LittleEndian.Uint64(raw[i+8:])
			add := binary.LittleEndian.Uint64(raw[i+16:])
			switch t := uint32(info); t {
			case rAMDGPURelative64:
				binary.LittleEndian.PutUint64(mem[off:], buf.VA+add)
			case rAMDGPUAbs64:
				sym := syms[info>>32-1]
				binary.LittleEndian.PutUint64(mem[off:], buf.VA+sym.Value+add)
			default:
				return nil, fmt.Errorf("unsupported relocation %d in %s", t, s.Name)
			}
		}
	}
	// Written once, before the GPU ever fetches it.
	copy(buf.Bytes(), mem)
	d.FlushHDP()

	meta, err := readMetadata(f)
	if err != nil {
		return nil, err
	}
	prog := &Program{kernels: map[string]*Kernel{}}
	for _, s := range syms {
		if !strings.HasSuffix(s.Name, ".kd") {
			continue
		}
		name := strings.TrimSuffix(s.Name, ".kd")
		k, ok := meta[name]
		if !ok {
			return nil, fmt.Errorf("kernel %s has no metadata", name)
		}
		k.Object = buf.VA + s.Value
		prog.kernels[name] = k
	}
	return prog, nil
}

// Kernel finds a kernel by a unique substring of its symbol, so callers need not spell
// C++ mangled names.
func (p *Program) Kernel(name string) (*Kernel, error) {
	var found *Kernel
	for n, k := range p.kernels {
		if strings.Contains(n, name) {
			if found != nil {
				return nil, fmt.Errorf("kernel name %q is ambiguous", name)
			}
			found = k
		}
	}
	if found == nil {
		return nil, fmt.Errorf("kernel %q not found", name)
	}
	if found.PrivateSize != 0 {
		return nil, fmt.Errorf("kernel %s needs scratch memory, which this driver does not provide", found.Name)
	}
	return found, nil
}

const ntAMDGPUMetadata = 32

func readMetadata(f *elf.File) (map[string]*Kernel, error) {
	for _, p := range f.Progs {
		if p.Type != elf.PT_NOTE {
			continue
		}
		raw := make([]byte, p.Filesz)
		if _, err := p.ReadAt(raw, 0); err != nil {
			return nil, err
		}
		for len(raw) >= 12 {
			le := binary.LittleEndian
			nameSz, descSz, typ := le.Uint32(raw), le.Uint32(raw[4:]), le.Uint32(raw[8:])
			nameEnd := 12 + (int(nameSz)+3)&^3
			descEnd := nameEnd + (int(descSz)+3)&^3
			if descEnd > len(raw) {
				break
			}
			if typ == ntAMDGPUMetadata {
				v, _, err := msgpackDecode(raw[nameEnd : nameEnd+int(descSz)])
				if err != nil {
					return nil, fmt.Errorf("metadata: %w", err)
				}
				return parseKernels(v)
			}
			raw = raw[descEnd:]
		}
	}
	return nil, fmt.Errorf("code object has no AMDGPU metadata note")
}

func parseKernels(v any) (map[string]*Kernel, error) {
	root, ok := v.(map[string]any)
	if !ok {
		return nil, fmt.Errorf("metadata root is not a map")
	}
	list, ok := root["amdhsa.kernels"].([]any)
	if !ok {
		return nil, fmt.Errorf("metadata has no amdhsa.kernels")
	}
	out := map[string]*Kernel{}
	for _, e := range list {
		m, ok := e.(map[string]any)
		if !ok {
			return nil, fmt.Errorf("kernel entry is not a map")
		}
		name, _ := m[".name"].(string)
		k := &Kernel{
			Name:        name,
			KernargSize: uint32(mpUint(m[".kernarg_segment_size"])),
			GroupSize:   uint32(mpUint(m[".group_segment_fixed_size"])),
			PrivateSize: uint32(mpUint(m[".private_segment_fixed_size"])),
		}
		args, _ := m[".args"].([]any)
		for _, a := range args {
			am, _ := a.(map[string]any)
			kind, _ := am[".value_kind"].(string)
			if strings.HasPrefix(kind, "hidden_") {
				k.hidden = append(k.hidden, hiddenArg{kind, uint32(mpUint(am[".offset"]))})
			}
		}
		out[name] = k
	}
	return out, nil
}

func mpUint(v any) uint64 {
	if u, ok := v.(uint64); ok {
		return u
	}
	return 0
}

// msgpackDecode handles the subset used by AMDGPU metadata; integers decode to uint64.
func msgpackDecode(b []byte) (any, []byte, error) {
	if len(b) == 0 {
		return nil, nil, fmt.Errorf("truncated msgpack")
	}
	c := b[0]
	b = b[1:]
	be := binary.BigEndian
	str := func(n int) (any, []byte, error) {
		if len(b) < n {
			return nil, nil, fmt.Errorf("truncated msgpack string")
		}
		return string(b[:n]), b[n:], nil
	}
	arr := func(n int) (any, []byte, error) {
		out := make([]any, n)
		var err error
		for i := range out {
			if out[i], b, err = msgpackDecode(b); err != nil {
				return nil, nil, err
			}
		}
		return out, b, nil
	}
	mp := func(n int) (any, []byte, error) {
		out := make(map[string]any, n)
		for i := 0; i < n; i++ {
			var k, v any
			var err error
			if k, b, err = msgpackDecode(b); err != nil {
				return nil, nil, err
			}
			if v, b, err = msgpackDecode(b); err != nil {
				return nil, nil, err
			}
			ks, ok := k.(string)
			if !ok {
				return nil, nil, fmt.Errorf("non-string msgpack key")
			}
			out[ks] = v
		}
		return out, b, nil
	}
	switch {
	case c < 0x80:
		return uint64(c), b, nil
	case c < 0x90:
		return mp(int(c & 0xf))
	case c < 0xa0:
		return arr(int(c & 0xf))
	case c < 0xc0:
		return str(int(c & 0x1f))
	case c >= 0xe0:
		return uint64(int8(c)), b, nil
	}
	need := func(n int) error {
		if len(b) < n {
			return fmt.Errorf("truncated msgpack value")
		}
		return nil
	}
	switch c {
	case 0xc0:
		return nil, b, nil
	case 0xc2:
		return false, b, nil
	case 0xc3:
		return true, b, nil
	case 0xcc, 0xd0:
		if err := need(1); err != nil {
			return nil, nil, err
		}
		return uint64(b[0]), b[1:], nil
	case 0xcd, 0xd1:
		if err := need(2); err != nil {
			return nil, nil, err
		}
		return uint64(be.Uint16(b)), b[2:], nil
	case 0xce, 0xd2:
		if err := need(4); err != nil {
			return nil, nil, err
		}
		return uint64(be.Uint32(b)), b[4:], nil
	case 0xcf, 0xd3:
		if err := need(8); err != nil {
			return nil, nil, err
		}
		return be.Uint64(b), b[8:], nil
	case 0xd9:
		if err := need(1); err != nil {
			return nil, nil, err
		}
		n := int(b[0])
		b = b[1:]
		return str(n)
	case 0xda:
		if err := need(2); err != nil {
			return nil, nil, err
		}
		n := int(be.Uint16(b))
		b = b[2:]
		return str(n)
	case 0xdc:
		if err := need(2); err != nil {
			return nil, nil, err
		}
		n := int(be.Uint16(b))
		b = b[2:]
		return arr(n)
	case 0xde:
		if err := need(2); err != nil {
			return nil, nil, err
		}
		n := int(be.Uint16(b))
		b = b[2:]
		return mp(n)
	}
	return nil, nil, fmt.Errorf("unsupported msgpack tag %#x", c)
}
