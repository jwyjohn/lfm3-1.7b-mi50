// Package payload is the model data appended to the executable. It lives after the
// ELF image rather than in a Go section because the Go linker caps data at 2 GB.
package payload

import (
	"encoding/binary"
	"encoding/json"
	"fmt"
	"io"
	"os"
	"syscall"
)

const (
	magic      = "LFM3PAY1"
	footerSize = 32
	// Align is the file alignment of every entry.
	Align = 4096
)

// Entry locates one blob relative to the start of the payload.
type Entry struct {
	Off  uint64 `json:"off"`
	Size uint64 `json:"size"`
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
	w.m.Entries[name] = Entry{w.pos, uint64(len(data))}
	w.pos += uint64(len(data))
	return nil
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
