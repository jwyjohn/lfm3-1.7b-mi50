// pack quantizes the LFM3-1.7B and DSpark drafter safetensors into the model payload, or
// attaches a finished payload to a built server executable.
package main

import (
	_ "embed"
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

// float16 loads a tensor stored as IEEE half precision.
func (s *safetensors) float16(name string, shape ...int) []uint16 {
	raw := s.raw(name, "F16", shape)
	out := make([]uint16, len(raw)/2)
	for i := range out {
		out[i] = binary.LittleEndian.Uint16(raw[2*i:])
	}
	return out
}

// float32s loads a tensor stored as IEEE single precision.
func (s *safetensors) float32s(name string, shape ...int) []float32 {
	raw := s.raw(name, "F32", shape)
	out := make([]float32, len(raw)/4)
	for i := range out {
		out[i] = math.Float32frombits(binary.LittleEndian.Uint32(raw[4*i:]))
	}
	return out
}

func (s *safetensors) raw(name, dtype string, shape []int) []byte {
	ti, ok := s.info[name]
	if !ok {
		log.Fatalf("tensor %s not found", name)
	}
	if ti.Dtype != dtype {
		log.Fatalf("tensor %s is %s, expected %s", name, ti.Dtype, dtype)
	}
	if fmt.Sprint(ti.Shape) != fmt.Sprint(shape) {
		log.Fatalf("tensor %s has shape %v, expected %v", name, ti.Shape, shape)
	}
	return s.data[s.base+ti.Offsets[0] : s.base+ti.Offsets[1]]
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

// cropIDs lists the 16384 most frequent tokens, most frequent first, little-endian int32:
// the rows of the drafter's cropped LM head.
//
//go:embed crop_ids.bin
var cropIDs []byte

const cropRows = 16384

// packDrafter adds the DSpark drafter. Its k/v projection in the context path and the head
// crop are derived at startup from tensors stored here and from the embedding.
func packDrafter(a *arena, dir string) (*payload.DraftMeta, error) {
	st, err := openSafetensors(filepath.Join(dir, "model.safetensors"))
	if err != nil {
		return nil, err
	}
	var cfg struct {
		Architectures []string `json:"architectures"`
		Layers        int      `json:"num_hidden_layers"`
		Taps          []int    `json:"target_layer_ids"`
		Window        int      `json:"draft_sliding_window"`
		MarkovRank    int      `json:"markov_rank"`
		DFlash        struct {
			MaskToken int `json:"mask_token_id"`
		} `json:"dflash_config"`
	}
	raw, err := os.ReadFile(filepath.Join(dir, "config.json"))
	if err != nil {
		return nil, err
	}
	if err := json.Unmarshal(raw, &cfg); err != nil {
		return nil, err
	}
	if len(cfg.Architectures) != 1 || cfg.Architectures[0] != "Lfm3DSparkDraftModel" {
		return nil, fmt.Errorf("%s is not a DSpark drafter: %v", dir, cfg.Architectures)
	}
	if cfg.Layers != 3 || fmt.Sprint(cfg.Taps) != "[2 6 12 18 23]" || cfg.MarkovRank != 256 {
		return nil, fmt.Errorf("unsupported drafter shape: %d layers, taps %v, rank %d", cfg.Layers, cfg.Taps, cfg.MarkovRank)
	}
	if len(cropIDs) != cropRows*4 {
		return nil, fmt.Errorf("crop list has %d bytes", len(cropIDs))
	}
	const H, I, Vocab, Rank = 2048, 7680, 128000, 256
	for l := range 3 {
		pre := fmt.Sprintf("layers.%d.", l)
		key := func(s string) string { return fmt.Sprintf("d%d.%s", l, s) }
		a.add(key("operator_norm"), u16le(st.float16(pre+"input_layernorm.weight", H)))
		a.add(key("ffn_norm"), u16le(st.float16(pre+"post_attention_layernorm.weight", H)))
		mix := make([]uint16, 0, 3072*H)
		mix = append(mix, st.float16(pre+"self_attn.q_proj.weight", H, H)...)
		mix = append(mix, st.float16(pre+"self_attn.k_proj.weight", 512, H)...)
		mix = append(mix, st.float16(pre+"self_attn.v_proj.weight", 512, H)...)
		a.addPacked(key("mix"), weights.Pack(mix, 3072, H))
		a.addPacked(key("out"), weights.Pack(st.float16(pre+"self_attn.o_proj.weight", H, H), H, H))
		gate := st.float16(pre+"mlp.gate_proj.weight", I, H)
		up := st.float16(pre+"mlp.up_proj.weight", I, H)
		w13 := make([]uint16, 2*I*H)
		for r := range I {
			copy(w13[2*r*H:], gate[r*H:(r+1)*H])
			copy(w13[(2*r+1)*H:], up[r*H:(r+1)*H])
		}
		a.addPacked(key("w13"), weights.Pack(w13, 2*I, H))
		w2 := weights.Pack(st.float16(pre+"mlp.down_proj.weight", H, I), H, I)
		// The kernel's streaming loads may run a little past the last row.
		a.add(key("w2.w"), append(w2.Biased, make([]byte, 4096)...))
		a.add(key("w2.s"), u16le(w2.Scales))
		a.add(key("q_norm"), u16le(st.float16(pre+"self_attn.q_norm.weight", 128)))
		a.add(key("k_norm"), u16le(st.float16(pre+"self_attn.k_norm.weight", 128)))
		log.Printf("drafter layer %d packed, arena %d MiB", l, len(a.blob)>>20)
	}
	// fc in five bands of 2048 inputs, one per tapped layer.
	fc := st.float16("fc.weight", H, 5*H)
	var fcW []byte
	var fcS []uint16
	band := make([]uint16, H*H)
	for b := range 5 {
		for r := range H {
			copy(band[r*H:(r+1)*H], fc[r*5*H+b*H:])
		}
		p := weights.Pack(band, H, H)
		fcW = append(fcW, p.Biased...)
		fcS = append(fcS, p.Scales...)
	}
	a.add("fc.w", fcW)
	a.add("fc.s", u16le(fcS))
	var tapNorms []uint16
	for i := range 5 {
		tapNorms = append(tapNorms, st.float16(fmt.Sprintf("tap_norms.%d.weight", i), H)...)
	}
	a.add("tap_norms", u16le(tapNorms))
	a.add("hidden_norm", u16le(st.float16("hidden_norm.weight", H)))
	a.add("draft_norm", u16le(st.float16("norm.weight", H)))
	a.add("crop_ids", cropIDs)

	// Markov matrices: w1 for every token, w2 only for the head crop, in the kernel's
	// [64 rows][16 pieces][64 lanes][16 B] order so one wave load is one contiguous KiB.
	m1 := st.float16("markov_head.markov_w1.weight", Vocab, Rank)
	m2 := st.float16("markov_head.markov_w2.weight", Vocab, Rank)
	w1q, w1s := weights.QuantizeRowsInt8(m1, Vocab, Rank)
	a.add("markov_w1.q", int8s(w1q))
	a.add("markov_w1.s", f32le(w1s))
	cropped := make([]uint16, cropRows*Rank)
	for r := range cropRows {
		id := int(binary.LittleEndian.Uint32(cropIDs[4*r:]))
		copy(cropped[r*Rank:(r+1)*Rank], m2[id*Rank:(id+1)*Rank])
	}
	w2q, w2s := weights.QuantizeRowsInt8(cropped, cropRows, Rank)
	a.add("markov_w2.q", transposeLanes(int8s(w2q)))
	a.add("markov_w2.s", f32le(w2s))

	conf := st.float32s("confidence_head.proj.weight", 1, H+Rank)
	bias := st.float32s("confidence_head.proj.bias", 1)
	c2 := make([]float32, Vocab)
	weights.Parallel(Vocab, func(r int) {
		var sum float64
		for j, h := range m1[r*Rank : (r+1)*Rank] {
			sum += float64(fp16.ToFloat32(h)) * float64(conf[H+j])
		}
		c2[r] = float32(sum)
	})
	a.add("conf_markov", f32le(c2))
	confRow := make([]uint16, H)
	for i := range confRow {
		confRow[i] = fp16.FromFloat32(conf[i])
	}
	a.add("conf_row", u16le(confRow))
	log.Printf("drafter packed, arena %d MiB", len(a.blob)>>20)
	return &payload.DraftMeta{ConfBias: bias[0], MaskToken: cfg.DFlash.MaskToken, Window: cfg.Window, Rows: cropRows}, nil
}

// transposeLanes turns [groups][64 rows][16 pieces][16 B] into [groups][16 pieces][64 rows][16 B].
func transposeLanes(src []byte) []byte {
	const row, piece, group = 256, 16, 64 * 256
	dst := make([]byte, len(src))
	for g := 0; g+group <= len(src); g += group {
		for r := range 64 {
			for p := range 16 {
				copy(dst[g+p*64*piece+r*piece:], src[g+r*row+p*piece:g+r*row+(p+1)*piece])
			}
		}
	}
	return dst
}

func int8s(v []int8) []byte {
	out := make([]byte, len(v))
	for i, x := range v {
		out[i] = byte(x)
	}
	return out
}

func f32le(v []float32) []byte {
	out := make([]byte, 4*len(v))
	for i, x := range v {
		binary.LittleEndian.PutUint32(out[4*i:], math.Float32bits(x))
	}
	return out
}

func pack(model, drafter, out string) error {
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
	// The int8 LM head is the quantized embedding and is derived from it at startup.
	embed := st.half("model.embed_tokens.weight", 128000, H)
	a.add("final_norm", u16le(st.half("model.embedding_norm.weight", H)))
	log.Printf("arena %d MiB", len(a.blob)>>20)
	meta, err := packDrafter(a, drafter)
	if err != nil {
		return fmt.Errorf("drafter: %w", err)
	}

	w, err := payload.Create(out)
	if err != nil {
		return err
	}
	if err := w.AddCompressed("arena", a.blob, payload.Entropy, 1); err != nil {
		return err
	}
	index, err := json.Marshal(a.offsets)
	if err != nil {
		return err
	}
	if err := w.Add("arena.index", index); err != nil {
		return err
	}
	metaRaw, err := json.Marshal(meta)
	if err != nil {
		return err
	}
	if err := w.Add("draft.meta", metaRaw); err != nil {
		return err
	}
	if err := w.AddCompressed("embed", u16le(embed), payload.Entropy, 2); err != nil {
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
	if err := w.AddCompressed("tokenizer", compiled, payload.Text, 1); err != nil {
		return err
	}
	return w.Close()
}

func main() {
	model := flag.String("model", "", "LFM3-1.7B snapshot directory (with model.safetensors)")
	drafter := flag.String("drafter", "", "DSpark drafter directory (with model.safetensors and config.json)")
	out := flag.String("out", "", "payload file to create")
	attachElf := flag.String("elf", "", "attach mode: built executable")
	attachPayload := flag.String("payload", "", "attach mode: payload file")
	flag.Parse()
	var err error
	switch {
	case *attachElf != "":
		err = payload.Attach(*attachElf, *attachPayload, *out)
	case *model != "":
		err = pack(*model, *drafter, *out)
	default:
		flag.Usage()
		os.Exit(2)
	}
	if err != nil {
		log.Fatal(err)
	}
}
