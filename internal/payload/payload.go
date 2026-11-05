// Package payload is the model data appended to the executable. It lives after the
// ELF image rather than in a Go section because the Go linker caps data at 2 GB.
package payload

import (
	"bytes"
	"compress/flate"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"runtime"
	"sync"
	"sync/atomic"
	"syscall"
)

const (
	magic      = "LFM3PAY1"
	footerSize = 32
	// Align is the file alignment of every entry.
	Align = 4096
	// chunkBytes is the raw size of one independently coded piece of a compressed entry.
	chunkBytes = 8 << 20
)

// Entry locates one blob relative to the start of the payload.
type Entry struct {
	Off  uint64 `json:"off"`
	Size uint64 `json:"size"`
	// Compressed entries only: the raw size, the coded size of each chunk, and the element
	// width whose byte planes are coded apart.
	Raw    uint64   `json:"raw,omitempty"`
	Chunks []uint64 `json:"chunks,omitempty"`
	Stride int      `json:"stride,omitempty"`
}

// DraftMeta holds the scalar settings of the packed DSpark drafter.
type DraftMeta struct {
	ConfBias  float32 `json:"conf_bias"`
	MaskToken int     `json:"mask_token"`
	Window    int     `json:"window"`
	Rows      int     `json:"crop_rows"`
}

// Manifest describes the payload contents.
type Manifest struct {
	Entries map[string]Entry `json:"entries"`
}

// Writer streams blobs to a payload file.
type Writer struct {
	f   *os.File
	pos uint64
	m   Manifest
}

// Create starts a new payload file.
func Create(path string) (*Writer, error) {
	f, err := os.Create(path)
	if err != nil {
		return nil, err
	}
	return &Writer{f: f, m: Manifest{Entries: map[string]Entry{}}}, nil
}

func (w *Writer) pad(align uint64) error {
	if rem := w.pos % align; rem != 0 {
		if _, err := w.f.Write(make([]byte, align-rem)); err != nil {
			return err
		}
		w.pos += align - rem
	}
	return nil
}

// Add appends a named blob.
func (w *Writer) Add(name string, data []byte) error {
	if err := w.pad(Align); err != nil {
		return err
	}
	if _, dup := w.m.Entries[name]; dup {
		return fmt.Errorf("duplicate payload entry %q", name)
	}
	if _, err := w.f.Write(data); err != nil {
		return err
	}
	w.m.Entries[name] = Entry{Off: w.pos, Size: uint64(len(data))}
	w.pos += uint64(len(data))
	return nil
}

// Compression levels of AddCompressed. Weights are close to random in their low bits, so
// matching finds nothing in them: entropy coding alone keeps nearly all of the gain and
// inflates much faster.
const (
	Entropy = flate.HuffmanOnly
	Text    = flate.BestCompression
)

// AddCompressed appends a named blob coded in parallel chunks. With stride > 1 each chunk is
// split into byte planes first, which suits multi-byte floats whose planes differ in entropy.
func (w *Writer) AddCompressed(name string, data []byte, level, stride int) error {
	n := (len(data) + chunkBytes - 1) / chunkBytes
	coded := make([][]byte, n)
	var firstErr atomic.Value
	parallel(n, func(i int) {
		raw := data[i*chunkBytes : min((i+1)*chunkBytes, len(data))]
		var buf bytes.Buffer
		fw, err := flate.NewWriter(&buf, level)
		if err == nil {
			if _, err = fw.Write(split(raw, stride)); err == nil {
				err = fw.Close()
			}
		}
		if err != nil {
			firstErr.Store(err)
			return
		}
		coded[i] = buf.Bytes()
	})
	if err, ok := firstErr.Load().(error); ok {
		return err
	}
	if err := w.pad(Align); err != nil {
		return err
	}
	if _, dup := w.m.Entries[name]; dup {
		return fmt.Errorf("duplicate payload entry %q", name)
	}
	e := Entry{Off: w.pos, Raw: uint64(len(data)), Stride: stride, Chunks: make([]uint64, n)}
	for i, c := range coded {
		if _, err := w.f.Write(c); err != nil {
			return err
		}
		e.Chunks[i] = uint64(len(c))
		e.Size += uint64(len(c))
	}
	w.m.Entries[name] = e
	w.pos += e.Size
	return nil
}

// split separates the byte planes of elements stride bytes wide.
func split(raw []byte, stride int) []byte {
	if stride <= 1 {
		return raw
	}
	out := make([]byte, len(raw))
	elems := len(raw) / stride
	for b := 0; b < stride; b++ {
		dst := out[b*elems : (b+1)*elems]
		for i := range dst {
			dst[i] = raw[i*stride+b]
		}
	}
	copy(out[elems*stride:], raw[elems*stride:])
	return out
}

// join is the inverse of split.
func join(dst, planes []byte, stride int) {
	elems := len(dst) / stride
	for b := 0; b < stride; b++ {
		src := planes[b*elems : (b+1)*elems]
		for i, v := range src {
			dst[i*stride+b] = v
		}
	}
	copy(dst[elems*stride:], planes[elems*stride:])
}

func parallel(n int, fn func(i int)) {
	var next atomic.Int64
	var wg sync.WaitGroup
	for range runtime.NumCPU() {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for {
				i := int(next.Add(1)) - 1
				if i >= n {
					return
				}
				fn(i)
			}
		}()
	}
	wg.Wait()
}

// Close writes the manifest and footer. The footer's payload offset is zero in a
// standalone file; attaching to an ELF rewrites it.
func (w *Writer) Close() error {
	if err := w.pad(Align); err != nil {
		return err
	}
	raw, err := json.Marshal(w.m)
	if err != nil {
		return err
	}
	if _, err := w.f.Write(raw); err != nil {
		return err
	}
	if err := writeFooter(w.f, 0, w.pos, uint64(len(raw))); err != nil {
		return err
	}
	return w.f.Close()
}

func writeFooter(f io.Writer, base, manifestOff, manifestLen uint64) error {
	var b [footerSize]byte
	copy(b[:], magic)
	binary.LittleEndian.PutUint64(b[8:], base)
	binary.LittleEndian.PutUint64(b[16:], manifestOff)
	binary.LittleEndian.PutUint64(b[24:], manifestLen)
	_, err := f.Write(b[:])
	return err
}

// Attach writes elf + padding + payload to out and marks it executable.
func Attach(elfPath, payloadPath, out string) error {
	src, err := os.Open(elfPath)
	if err != nil {
		return err
	}
	defer src.Close()
	pay, err := os.Open(payloadPath)
	if err != nil {
		return err
	}
	defer pay.Close()
	st, err := pay.Stat()
	if err != nil {
		return err
	}
	if st.Size() < footerSize {
		return fmt.Errorf("%s is too short to be a payload", payloadPath)
	}
	var foot [footerSize]byte
	if _, err := pay.ReadAt(foot[:], st.Size()-footerSize); err != nil {
		return err
	}
	if string(foot[:8]) != magic {
		return fmt.Errorf("%s has no payload footer", payloadPath)
	}
	dst, err := os.OpenFile(out, os.O_CREATE|os.O_TRUNC|os.O_WRONLY, 0o755)
	if err != nil {
		return err
	}
	defer dst.Close()
	n, err := io.Copy(dst, src)
	if err != nil {
		return err
	}
	const pageAlign = 2 << 20
	pad := (pageAlign - n%pageAlign) % pageAlign
	if _, err := dst.Write(make([]byte, pad)); err != nil {
		return err
	}
	base := uint64(n + pad)
	if _, err := io.CopyN(dst, pay, st.Size()-footerSize); err != nil {
		return err
	}
	if err := writeFooter(dst, base, binary.LittleEndian.Uint64(foot[16:]), binary.LittleEndian.Uint64(foot[24:])); err != nil {
		return err
	}
	return dst.Chmod(0o755)
}

// Payload is a read-only view of the attached data, mapped from the page cache.
type Payload struct {
	data []byte
	base uint64
	Man  Manifest
}

// Open maps path, which is an executable with an attached payload or a bare payload file.
func Open(path string) (*Payload, error) {
	f, err := os.Open(path)
	if err != nil {
		return nil, err
	}
	defer f.Close()
	st, err := f.Stat()
	if err != nil {
		return nil, err
	}
	data, err := syscall.Mmap(int(f.Fd()), 0, int(st.Size()), syscall.PROT_READ, syscall.MAP_SHARED)
	if err != nil {
		return nil, fmt.Errorf("mmap %s: %w", path, err)
	}
	if len(data) < footerSize || string(data[len(data)-footerSize:len(data)-footerSize+8]) != magic {
		return nil, fmt.Errorf("%s carries no model payload", path)
	}
	foot := data[len(data)-footerSize:]
	p := &Payload{data: data, base: binary.LittleEndian.Uint64(foot[8:])}
	off := p.base + binary.LittleEndian.Uint64(foot[16:])
	n := binary.LittleEndian.Uint64(foot[24:])
	if err := json.Unmarshal(data[off:off+n], &p.Man); err != nil {
		return nil, fmt.Errorf("payload manifest: %w", err)
	}
	return p, nil
}

// Get returns a named blob; the slice aliases the mapping and is read-only.
func (p *Payload) Get(name string) ([]byte, error) {
	e, ok := p.Man.Entries[name]
	if !ok {
		return nil, fmt.Errorf("payload has no entry %q", name)
	}
	return p.data[p.base+e.Off : p.base+e.Off+e.Size], nil
}

// RawSize is the size of a named blob once decoded.
func (p *Payload) RawSize(name string) (uint64, error) {
	e, ok := p.Man.Entries[name]
	if !ok {
		return 0, fmt.Errorf("payload has no entry %q", name)
	}
	if e.Chunks != nil {
		return e.Raw, nil
	}
	return e.Size, nil
}

// Data returns a named blob decoded. An entry stored as is aliases the mapping and is
// read-only; a compressed entry lands in a new slice on all CPUs.
func (p *Payload) Data(name string) ([]byte, error) {
	e, ok := p.Man.Entries[name]
	if !ok {
		return nil, fmt.Errorf("payload has no entry %q", name)
	}
	src := p.data[p.base+e.Off : p.base+e.Off+e.Size]
	if e.Chunks == nil {
		return src, nil
	}
	starts := make([]uint64, len(e.Chunks))
	var off uint64
	for i, c := range e.Chunks {
		starts[i] = off
		off += c
	}
	out := make([]byte, e.Raw)
	var firstErr atomic.Value
	parallel(len(e.Chunks), func(i int) {
		dst := out[i*chunkBytes : min((i+1)*chunkBytes, len(out))]
		planes := dst
		if e.Stride > 1 {
			planes = make([]byte, len(dst))
		}
		r := flate.NewReader(bytes.NewReader(src[starts[i] : starts[i]+e.Chunks[i]]))
		if _, err := io.ReadFull(r, planes); err != nil {
			firstErr.Store(fmt.Errorf("payload entry %q chunk %d: %w", name, i, err))
			return
		}
		if e.Stride > 1 {
			join(dst, planes, e.Stride)
		}
	})
	if err, ok := firstErr.Load().(error); ok {
		return nil, err
	}
	return out, nil
}
