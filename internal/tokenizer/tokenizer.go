// Package tokenizer implements the LFM3 tokenizer: BPE with GPT-2 byte-level
// symbols, a fixed pre-tokenizer split, and literal added tokens.
package tokenizer

import (
	"bytes"
	"container/heap"
	"encoding/binary"
	"encoding/json"
	"fmt"
	"sort"
	"strings"
	"unicode"
	"unicode/utf8"
)

const compiledMagic = "LFM3TOK1"

// byteLevelTable is GPT-2's bytes_to_unicode mapping: rune -> raw byte.
func byteLevelTable() map[rune]byte {
	out := map[rune]byte{}
	n := 0
	for b := 0; b < 256; b++ {
		if (b >= '!' && b <= '~') || (b >= 0xA1 && b <= 0xAC) || (b >= 0xAE && b <= 0xFF) {
			out[rune(b)] = byte(b)
		} else {
			out[rune(256+n)] = byte(b)
			n++
		}
	}
	return out
}

func symbolsToBytes(s string, table map[rune]byte) ([]byte, error) {
	out := make([]byte, 0, len(s))
	for _, r := range s {
		b, ok := table[r]
		if !ok {
			return nil, fmt.Errorf("token %q has a rune outside the byte-level alphabet", s)
		}
		out = append(out, b)
	}
	return out, nil
}

// Compile converts tokenizer.json to the compact form stored in the payload.
func Compile(raw []byte) ([]byte, error) {
	var tj struct {
		AddedTokens []struct {
			ID      int32  `json:"id"`
			Content string `json:"content"`
			Special bool   `json:"special"`
		} `json:"added_tokens"`
		Model struct {
			Type   string           `json:"type"`
			Vocab  map[string]int32 `json:"vocab"`
			Merges [][2]string      `json:"merges"`
		} `json:"model"`
	}
	if err := json.Unmarshal(raw, &tj); err != nil {
		return nil, err
	}
	if tj.Model.Type != "BPE" {
		return nil, fmt.Errorf("model type %q is not BPE", tj.Model.Type)
	}
	table := byteLevelTable()
	var maxID int32
	for _, id := range tj.Model.Vocab {
		if id > maxID {
			maxID = id
		}
	}
	for _, a := range tj.AddedTokens {
		if a.ID > maxID {
			maxID = a.ID
		}
	}
	n := int(maxID) + 1
	toks := make([][]byte, n)
	flags := make([]byte, n)
	for s, id := range tj.Model.Vocab {
		b, err := symbolsToBytes(s, table)
		if err != nil {
			return nil, err
		}
		toks[id] = b
	}
	for _, a := range tj.AddedTokens {
		toks[a.ID] = []byte(a.Content)
		flags[a.ID] = flagAdded
		if a.Special {
			flags[a.ID] |= flagSpecial
		}
	}
	var buf bytes.Buffer
	buf.WriteString(compiledMagic)
	le := binary.LittleEndian
	binary.Write(&buf, le, uint32(n))
	for i := 0; i < n; i++ {
		buf.WriteByte(flags[i])
		binary.Write(&buf, le, uint16(len(toks[i])))
		buf.Write(toks[i])
	}
	binary.Write(&buf, le, uint32(len(tj.Model.Merges)))
	for _, m := range tj.Model.Merges {
		l, ok1 := tj.Model.Vocab[m[0]]
		r, ok2 := tj.Model.Vocab[m[1]]
		v, ok3 := tj.Model.Vocab[m[0]+m[1]]
		if !ok1 || !ok2 || !ok3 {
			return nil, fmt.Errorf("merge %q + %q is not in the vocabulary", m[0], m[1])
		}
		binary.Write(&buf, le, l)
		binary.Write(&buf, le, r)
		binary.Write(&buf, le, v)
	}
	return buf.Bytes(), nil
}

const (
	flagAdded   = 1
	flagSpecial = 2
)

type mergeRule struct {
	rank int32
	id   int32
}

// Tokenizer encodes and decodes. It is safe for concurrent use after Load.
type Tokenizer struct {
	tokens  [][]byte
	flags   []byte
	whole   map[string]int32
	merges  map[uint64]mergeRule
	byteIDs [256]int32
	added   []addedToken // longest first
	addedBy map[byte][]int
}

type addedToken struct {
	text string
	id   int32
}

// Load parses the compiled form.
func Load(b []byte) (*Tokenizer, error) {
	if len(b) < 12 || string(b[:8]) != compiledMagic {
		return nil, fmt.Errorf("bad compiled tokenizer")
	}
	le := binary.LittleEndian
	n := int(le.Uint32(b[8:]))
	p := 12
	t := &Tokenizer{tokens: make([][]byte, n), flags: make([]byte, n), whole: make(map[string]int32, n)}
	for i := 0; i < n; i++ {
		t.flags[i] = b[p]
		l := int(le.Uint16(b[p+1:]))
		t.tokens[i] = b[p+3 : p+3+l]
		p += 3 + l
		if l > 0 && t.flags[i]&flagAdded == 0 {
			t.whole[string(t.tokens[i])] = int32(i)
		}
		if t.flags[i]&flagAdded != 0 {
			t.added = append(t.added, addedToken{string(t.tokens[i]), int32(i)})
		}
	}
	for i := range t.byteIDs {
		t.byteIDs[i] = -1
		if id, ok := t.whole[string([]byte{byte(i)})]; ok {
			t.byteIDs[i] = id
		}
	}
	nm := int(le.Uint32(b[p:]))
	p += 4
	t.merges = make(map[uint64]mergeRule, nm)
	for i := 0; i < nm; i++ {
		l, r, v := le.Uint32(b[p:]), le.Uint32(b[p+4:]), le.Uint32(b[p+8:])
		p += 12
		t.merges[uint64(l)<<32|uint64(r)] = mergeRule{int32(i), int32(v)}
	}
	sort.SliceStable(t.added, func(i, j int) bool { return len(t.added[i].text) > len(t.added[j].text) })
	t.addedBy = map[byte][]int{}
	for i, a := range t.added {
		t.addedBy[a.text[0]] = append(t.addedBy[a.text[0]], i)
	}
	return t, nil
}

func isL(r rune) bool { return unicode.IsLetter(r) }
func isN(r rune) bool { return unicode.Is(unicode.N, r) }
func isS(r rune) bool { return unicode.IsSpace(r) }

// splitEnd returns the end of the pre-tokenizer match starting at i. It is a
// hand-written equivalent of the model's split regex, whose lookahead Go's regexp
// cannot express:
//
//	(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}|
//	 ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+
func splitEnd(s string, i int) int {
	r0, w0 := utf8.DecodeRuneInString(s[i:])
	if r0 == '\'' {
		rest := s[i+1:]
		if len(rest) > 0 {
			switch rest[0] {
			case 's', 'S', 't', 'T', 'm', 'M', 'd', 'D':
				return i + 2
			case 'r', 'R':
				if len(rest) > 1 && (rest[1] == 'e' || rest[1] == 'E') {
					return i + 3
				}
			case 'v', 'V':
				if len(rest) > 1 && (rest[1] == 'e' || rest[1] == 'E') {
					return i + 3
				}
			case 'l', 'L':
				if len(rest) > 1 && (rest[1] == 'l' || rest[1] == 'L') {
					return i + 3
				}
			}
		}
	}
	letters := func(j int) int {
		for j < len(s) {
			r, w := utf8.DecodeRuneInString(s[j:])
			if !isL(r) {
				break
			}
			j += w
		}
		return j
	}
	if isL(r0) {
		return letters(i + w0)
	}
	if r0 != '\r' && r0 != '\n' && !isN(r0) {
		if e := letters(i + w0); e > i+w0 {
			return e
		}
	}
	if isN(r0) {
		j := i + w0
		for c := 1; c < 3 && j < len(s); c++ {
			r, w := utf8.DecodeRuneInString(s[j:])
			if !isN(r) {
				break
			}
			j += w
		}
		return j
	}
	symbol := func(r rune) bool { return !isS(r) && !isL(r) && !isN(r) }
	start := i
	if r0 == ' ' {
		start = i + 1
	}
	if start < len(s) {
		if r, _ := utf8.DecodeRuneInString(s[start:]); symbol(r) {
			j := start
			for j < len(s) {
				r, w := utf8.DecodeRuneInString(s[j:])
				if !symbol(r) {
					break
				}
				j += w
			}
			for j < len(s) && (s[j] == '\r' || s[j] == '\n') {
				j++
			}
			return j
		}
	}
	// Remaining alternatives start with whitespace; find the run.
	j := i
	lastNL := -1
	var lastW int
	for j < len(s) {
		r, w := utf8.DecodeRuneInString(s[j:])
		if !isS(r) {
			break
		}
		j += w
		lastW = w
		if r == '\r' || r == '\n' {
			lastNL = j
		}
	}
	if j == i {
		return i + w0 // unreachable for valid text; keeps the scanner advancing
	}
	if lastNL >= 0 {
		return lastNL
	}
	if j == len(s) {
		return j
	}
	if j-lastW > i {
		return j - lastW
	}
	return j
}

type bpeItem struct {
	rank  int32
	pos   int32
	left  int32
	right int32
}

type bpeHeap []bpeItem

func (h bpeHeap) Len() int { return len(h) }
func (h bpeHeap) Less(a, b int) bool {
	if h[a].rank != h[b].rank {
		return h[a].rank < h[b].rank
	}
	return h[a].pos < h[b].pos
}
func (h bpeHeap) Swap(a, b int) { h[a], h[b] = h[b], h[a] }
func (h *bpeHeap) Push(x any)   { *h = append(*h, x.(bpeItem)) }
func (h *bpeHeap) Pop() any {
	o := *h
	v := o[len(o)-1]
	*h = o[:len(o)-1]
	return v
}

// bpe encodes one pre-tokenized piece, appending to out.
func (t *Tokenizer) bpe(piece string, out []int32) []int32 {
	// ignore_merges: a piece that is itself a vocabulary entry is emitted whole.
	if id, ok := t.whole[piece]; ok {
		return append(out, id)
	}
	n := len(piece)
	ids := make([]int32, 0, n)
	for i := 0; i < n; i++ {
		if id := t.byteIDs[piece[i]]; id >= 0 {
			ids = append(ids, id)
		}
	}
	m := len(ids)
	if m == 0 {
		return out
	}
	next := make([]int32, m)
	prev := make([]int32, m)
	alive := make([]bool, m)
	for i := 0; i < m; i++ {
		next[i], prev[i], alive[i] = int32(i+1), int32(i-1), true
	}
	h := make(bpeHeap, 0, m)
	push := func(i int32) {
		j := next[i]
		if int(j) >= m {
			return
		}
		if rule, ok := t.merges[uint64(ids[i])<<32|uint64(ids[j])]; ok {
			heap.Push(&h, bpeItem{rule.rank, i, ids[i], ids[j]})
		}
	}
	for i := 0; i < m-1; i++ {
		push(int32(i))
	}
	for h.Len() > 0 {
		it := heap.Pop(&h).(bpeItem)
		i := it.pos
		if !alive[i] || ids[i] != it.left {
			continue
		}
		j := next[i]
		if int(j) >= m || ids[j] != it.right {
			continue
		}
		rule := t.merges[uint64(it.left)<<32|uint64(it.right)]
		ids[i] = rule.id
		alive[j] = false
		next[i] = next[j]
		if int(next[j]) < m {
			prev[next[j]] = i
		}
		if prev[i] >= 0 {
			push(prev[i])
		}
		push(i)
	}
	for i := int32(0); int(i) < m; i = next[i] {
		out = append(out, ids[i])
	}
	return out
}

func (t *Tokenizer) encodePlain(s string, out []int32) []int32 {
	s = strings.ToValidUTF8(s, "�")
	for i := 0; i < len(s); {
		e := splitEnd(s, i)
		out = t.bpe(s[i:e], out)
		i = e
	}
	return out
}

// Encode tokenizes text. Added tokens are recognized anywhere in it, as they are
// by the reference tokenizer for rendered chat prompts and plain completions.
func (t *Tokenizer) Encode(s string) []int32 {
	out := make([]int32, 0, len(s)/3+4)
	start := 0
	for i := 0; i < len(s); i++ {
		cands, ok := t.addedBy[s[i]]
		if !ok {
			continue
		}
		for _, c := range cands {
			a := t.added[c]
			if strings.HasPrefix(s[i:], a.text) {
				out = t.encodePlain(s[start:i], out)
				out = append(out, a.id)
				i += len(a.text) - 1
				start = i + 1
				break
			}
		}
	}
	return t.encodePlain(s[start:], out)
}

// IsSpecial reports whether id is a special token, which output text omits.
func (t *Tokenizer) IsSpecial(id int32) bool {
	return int(id) < len(t.flags) && t.flags[id]&flagSpecial != 0
}

// TokenBytes returns the raw bytes of id; unknown ids have none.
func (t *Tokenizer) TokenBytes(id int32) []byte {
	if id < 0 || int(id) >= len(t.tokens) {
		return nil
	}
	return t.tokens[id]
}

// Decode converts ids to text, replacing invalid UTF-8.
func (t *Tokenizer) Decode(ids []int32, skipSpecial bool) string {
	var b []byte
	for _, id := range ids {
		if skipSpecial && t.IsSpecial(id) {
			continue
		}
		b = append(b, t.TokenBytes(id)...)
	}
	return strings.ToValidUTF8(string(b), "�")
}

// Stream is an incremental detokenizer: it holds back a trailing partial UTF-8
// sequence so streamed text never splits a character.
type Stream struct {
	t           *Tokenizer
	skipSpecial bool
	pending     []byte
}

// NewStream starts incremental decoding.
func (t *Tokenizer) NewStream(skipSpecial bool) *Stream {
	return &Stream{t: t, skipSpecial: skipSpecial}
}

// Push adds one token and returns any text now complete.
func (s *Stream) Push(id int32) string {
	if s.skipSpecial && s.t.IsSpecial(id) {
		return ""
	}
	s.pending = append(s.pending, s.t.TokenBytes(id)...)
	cut := len(s.pending)
	for back := 1; back <= 3 && back <= len(s.pending); back++ {
		c := s.pending[len(s.pending)-back]
		if c < 0x80 {
			break
		}
		if c >= 0xC0 {
			need := 2
			if c >= 0xF0 {
				need = 4
			} else if c >= 0xE0 {
				need = 3
			}
			if back < need {
				cut = len(s.pending) - back
			}
			break
		}
	}
	out := strings.ToValidUTF8(string(s.pending[:cut]), "�")
	s.pending = append(s.pending[:0], s.pending[cut:]...)
	return out
}

// Flush returns whatever is left when the stream ends.
func (s *Stream) Flush() string {
	out := strings.ToValidUTF8(string(s.pending), "�")
	s.pending = s.pending[:0]
	return out
}
