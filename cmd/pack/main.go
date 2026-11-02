// pack quantizes the LFM3-1.7B safetensors into the model payload, or attaches a finished
// payload to a built server executable.
package main

import (
	"encoding/binary"
	"encoding/json"
	"flag"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"syscall"

	"lfm3-1.7b-mi50/internal/fp16"
	"lfm3-1.7b-mi50/internal/payload"
	"lfm3-1.7b-mi50/internal/tokenizer"
	"lfm3-1.7b-mi50/internal/weights"
)

type tensorInfo struct {
	Dtype   string   `json:"dtype"`
	Shape   []int    `json:"shape"`
	Offsets [2]int64 `json:"data_offsets"`
}

type safetensors struct {
	data []byte
	base int64
	info map[string]tensorInfo
}

func openSafetensors(path string) (*safetensors, error) {
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
		return nil, err
	}
	n := int64(binary.LittleEndian.Uint64(data))
	raw := map[string]json.RawMessage{}
	if err := json.Unmarshal(data[8:8+n], &raw); err != nil {
		return nil, err
	}
	s := &safetensors{data: data, base: 8 + n, info: map[string]tensorInfo{}}
	for k, v := range raw {
		if k == "__metadata__" {
			continue
		}
		var ti tensorInfo
		if err := json.Unmarshal(v, &ti); err != nil {
			return nil, err
		}
		s.info[k] = ti
	}
	return s, nil
}

// half loads a tensor as vLLM's --dtype half does: bfloat16 values rounded to fp16.
func (s *safetensors) half(name string, shape ...int) []uint16 {
	ti, ok := s.info[name]
	if !ok {
		log.Fatalf("tensor %s not found", name)
	}
	if ti.Dtype != "BF16" {
		log.Fatalf("tensor %s is %s, expected BF16", name, ti.Dtype)
	}
	if fmt.Sprint(ti.Shape) != fmt.Sprint(shape) {
		log.Fatalf("tensor %s has shape %v, expected %v", name, ti.Shape, shape)
	}
	raw := s.data[s.base+ti.Offsets[0] : s.base+ti.Offsets[1]]
	out := make([]uint16, len(raw)/2)
	weights.Parallel(len(out), func(i int) {
		bf := uint32(binary.LittleEndian.Uint16(raw[2*i:]))
		out[i] = fp16.FromFloat32(math.Float32frombits(bf << 16))
	})
	return out
}

func u16le(v []uint16) []byte {
	out := make([]byte, 2*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint16(out[2*i:], x)
	}
	return out
}

// arena mirrors decode_v2.py _consolidate: one blob, tensors on 64 KiB boundaries.
type arena struct {
	blob    []byte
	offsets map[string]payload.Entry
}

func (a *arena) add(name string, b []byte) {
	const align = 64 << 10
	off := uint64(len(a.blob))
	a.offsets[name] = payload.Entry{Off: off, Size: uint64(len(b))}
	a.blob = append(a.blob, b...)
	if rem := len(b) % align; rem != 0 {
		a.blob = append(a.blob, make([]byte, align-rem)...)
	}
}

func (a *arena) addPacked(name string, p *weights.Packed) {
	a.add(name+".w", p.Biased)
	a.add(name+".s", u16le(p.Scales))
}

func pack(model, out string) error {
	st, err := openSafetensors(filepath.Join(model, "model.safetensors"))
	if err != nil {
		return err
	}
	var cfg struct {
		LayerTypes []string `json:"layer_types"`
	}
	raw, err := os.ReadFile(filepath.Join(model, "config.json"))
	if err != nil {
		return err
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return err
	}
	if len(cfg.LayerTypes) != 24 {
		return fmt.Errorf("expected 24 layers, config has %d", len(cfg.LayerTypes))
	}
	const H, I = 2048, 7680
	a := &arena{blob: make([]byte, 0, 1800<<20), offsets: map[string]payload.Entry{}}
	for li, kind := range cfg.LayerTypes {
		pre := fmt.Sprintf("model.layers.%d.", li)
		key := func(s string) string { return fmt.Sprintf("l%d.%s", li, s) }
		for _, n := range []string{"operator_norm", "ffn_norm", "post_operator_norm", "post_ffn_norm"} {
			a.add(key(n), u16le(st.half(pre+n+".weight", H)))
		}
		// w13: gate and up rows interleaved so one wave owns whole pairs.
		w1 := st.half(pre+"feed_forward.w1.weight", I, H)
		w3 := st.half(pre+"feed_forward.w3.weight", I, H)
		w13 := make([]uint16, 2*I*H)
		for r := 0; r < I; r++ {
			copy(w13[2*r*H:], w1[r*H:(r+1)*H])
			copy(w13[(2*r+1)*H:], w3[r*H:(r+1)*H])
		}
		a.addPacked(key("w13"), weights.Pack(w13, 2*I, H))
		a.addPacked(key("w2"), weights.Pack(st.half(pre+"feed_forward.w2.weight", H, I), H, I))
		if kind == "conv" {
			in := st.half(pre+"conv.in_proj.weight", 3*H, H)
			mix := make([]uint16, 3*H*H)
			for c := 0; c < H; c++ {
				for ch := 0; ch < 3; ch++ {
					copy(mix[(3*c+ch)*H:], in[(ch*H+c)*H:(ch*H+c+1)*H])
				}
			}
			a.addPacked(key("mix"), weights.Pack(mix, 3*H, H))
			a.addPacked(key("out"), weights.Pack(st.half(pre+"conv.out_proj.weight", H, H), H, H))
			a.add(key("conv"), u16le(st.half(pre+"conv.conv.weight", H, 1, 7)))
			continue
		}
		const rows = H + 512 + 512 + 16
		mix := make([]uint16, 0, rows*H)
		mix = append(mix, st.half(pre+"self_attn.q_proj.weight", H, H)...)
		mix = append(mix, st.half(pre+"self_attn.k_proj.weight", 512, H)...)
		mix = append(mix, st.half(pre+"self_attn.v_proj.weight", 512, H)...)
		mix = append(mix, st.half(pre+"self_attn.gate_proj.weight", 16, H)...)
		a.addPacked(key("mix"), weights.Pack(mix, rows, H))
		a.addPacked(key("out"), weights.Pack(st.half(pre+"self_attn.o_proj.weight", H, H), H, H))
		a.add(key("q_norm"), u16le(st.half(pre+"self_attn.q_norm.weight", 128)))
		a.add(key("k_norm"), u16le(st.half(pre+"self_attn.k_norm.weight", 128)))
		log.Printf("layer %d (%s) packed, arena %d MiB", li, kind, len(a.blob)>>20)
	}
	embed := st.half("model.embed_tokens.weight", 128000, H)
	a.addPacked("head", weights.Pack(embed, 128000, H))
	a.add("final_norm", u16le(st.half("model.embedding_norm.weight", H)))
	log.Printf("arena %d MiB", len(a.blob)>>20)

	w, err := payload.Create(out)
	if err != nil {
		return err
	}
	if err := w.Add("arena", a.blob); err != nil {
		return err
	}
	index, err := json.Marshal(a.offsets)
	if err != nil {
		return err
	}
	if err := w.Add("arena.index", index); err != nil {
		return err
	}
	if err := w.Add("embed", u16le(embed)); err != nil {
		return err
	}
	tokRaw, err := os.ReadFile(filepath.Join(model, "tokenizer.json"))
	if err != nil {
		return err
	}
	compiled, err := tokenizer.Compile(tokRaw)
	if err != nil {
		return fmt.Errorf("tokenizer: %w", err)
	}
	if err := w.Add("tokenizer", compiled); err != nil {
		return err
	}
	return w.Close()
}

func main() {
	model := flag.String("model", "", "LFM3-1.7B snapshot directory (with model.safetensors)")
	out := flag.String("out", "", "payload file to create")
	attachElf := flag.String("elf", "", "attach mode: built executable")
	attachPayload := flag.String("payload", "", "attach mode: payload file")
	flag.Parse()
	var err error
	switch {
	case *attachElf != "":
		err = payload.Attach(*attachElf, *attachPayload, *out)
	case *model != "":
		err = pack(*model, *out)
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		log.Fatal(err)
	}
}
