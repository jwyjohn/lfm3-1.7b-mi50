// Package ojson is a JSON value model that keeps object key order and number text.
// The chat template prints tool schemas with `tojson`, so both affect the prompt and
// therefore the output: encoding/json's map decoding would reorder keys.
package ojson

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
	"unicode/utf8"
)

// Value is one of: nil, bool, Number, string, []Value, *Object.
type Value any

// Number keeps the source text.
type Number string

// Object is an insertion-ordered map.
type Object struct {
	Keys []string
	Vals map[string]Value
}

// NewObject returns an empty object.
func NewObject() *Object { return &Object{Vals: map[string]Value{}} }

// Set adds or replaces a key.
func (o *Object) Set(k string, v Value) {
	if _, ok := o.Vals[k]; !ok {
		o.Keys = append(o.Keys, k)
	}
	o.Vals[k] = v
}

// Get returns a value and whether the key exists.
func (o *Object) Get(k string) (Value, bool) {
	if o == nil {
		return nil, false
	}
	v, ok := o.Vals[k]
	return v, ok
}

// Str returns a string field, or "" when absent or of another type.
func (o *Object) Str(k string) string {
	v, _ := o.Get(k)
	s, _ := v.(string)
	return s
}

// Obj returns an object field, or nil.
func (o *Object) Obj(k string) *Object {
	v, _ := o.Get(k)
	c, _ := v.(*Object)
	return c
}

// Arr returns an array field, or nil.
func (o *Object) Arr(k string) []Value {
	v, _ := o.Get(k)
	c, _ := v.([]Value)
	return c
}

// Truthy follows Python/Jinja truthiness.
func Truthy(v Value) bool {
	switch x := v.(type) {
	case nil:
		return false
	case bool:
		return x
	case string:
		return x != ""
	case []Value:
		return len(x) > 0
	case *Object:
		return x != nil && len(x.Keys) > 0
	case Number:
		f, err := strconv.ParseFloat(string(x), 64)
		return err != nil || f != 0
	}
	return true
}

type parser struct {
	s string
	i int
}

// Parse decodes one JSON document.
func Parse(data []byte) (Value, error) {
	p := &parser{s: string(data)}
	v, err := p.value()
	if err != nil {
		return nil, err
	}
	p.ws()
	if p.i != len(p.s) {
		return nil, fmt.Errorf("trailing data at offset %d", p.i)
	}
	return v, nil
}

func (p *parser) ws() {
	for p.i < len(p.s) {
		switch p.s[p.i] {
		case ' ', '\t', '\n', '\r':
			p.i++
		default:
			return
		}
	}
}

func (p *parser) value() (Value, error) {
	p.ws()
	if p.i >= len(p.s) {
		return nil, fmt.Errorf("unexpected end of JSON")
	}
	switch c := p.s[p.i]; {
	case c == '{':
		p.i++
		o := NewObject()
		p.ws()
		if p.i < len(p.s) && p.s[p.i] == '}' {
			p.i++
			return o, nil
		}
		for {
			p.ws()
			if p.i >= len(p.s) || p.s[p.i] != '"' {
				return nil, fmt.Errorf("expected object key at offset %d", p.i)
			}
			k, err := p.str()
			if err != nil {
				return nil, err
			}
			p.ws()
			if p.i >= len(p.s) || p.s[p.i] != ':' {
				return nil, fmt.Errorf("expected ':' at offset %d", p.i)
			}
			p.i++
			v, err := p.value()
			if err != nil {
				return nil, err
			}
			o.Set(k, v)
			p.ws()
			if p.i >= len(p.s) {
				return nil, fmt.Errorf("unterminated object")
			}
			if p.s[p.i] == ',' {
				p.i++
				continue
			}
			if p.s[p.i] == '}' {
				p.i++
				return o, nil
			}
			return nil, fmt.Errorf("expected ',' or '}' at offset %d", p.i)
		}
	case c == '[':
		p.i++
		out := []Value{}
		p.ws()
		if p.i < len(p.s) && p.s[p.i] == ']' {
			p.i++
			return out, nil
		}
		for {
			v, err := p.value()
			if err != nil {
				return nil, err
			}
			out = append(out, v)
			p.ws()
			if p.i >= len(p.s) {
				return nil, fmt.Errorf("unterminated array")
			}
			if p.s[p.i] == ',' {
				p.i++
				continue
			}
			if p.s[p.i] == ']' {
				p.i++
				return out, nil
			}
			return nil, fmt.Errorf("expected ',' or ']' at offset %d", p.i)
		}
	case c == '"':
		return p.str()
	case strings.HasPrefix(p.s[p.i:], "true"):
		p.i += 4
		return true, nil
	case strings.HasPrefix(p.s[p.i:], "false"):
		p.i += 5
		return false, nil
	case strings.HasPrefix(p.s[p.i:], "null"):
		p.i += 4
		return nil, nil
	case c == '-' || (c >= '0' && c <= '9'):
		st := p.i
		p.i++
		for p.i < len(p.s) && strings.IndexByte("0123456789+-.eE", p.s[p.i]) >= 0 {
			p.i++
		}
		text := p.s[st:p.i]
		if _, err := strconv.ParseFloat(text, 64); err != nil {
			return nil, fmt.Errorf("bad number %q", text)
		}
		return Number(text), nil
	}
	return nil, fmt.Errorf("unexpected character %q at offset %d", p.s[p.i], p.i)
}

func (p *parser) str() (string, error) {
	st := p.i
	p.i++
	escaped := false
	for p.i < len(p.s) {
		c := p.s[p.i]
		if c == '\\' {
			escaped = true
			p.i += 2
			continue
		}
		if c == '"' {
			raw := p.s[st : p.i+1]
			p.i++
			if !escaped {
				return raw[1 : len(raw)-1], nil
			}
			var out string
			if err := json.Unmarshal([]byte(raw), &out); err != nil {
				return "", err
			}
			return out, nil
		}
		p.i++
	}
	return "", fmt.Errorf("unterminated string")
}

// Encode writes v like Python's json.dumps(v, ensure_ascii=False).
func Encode(v Value) string {
	var b strings.Builder
	encode(&b, v)
	return b.String()
}

func encode(b *strings.Builder, v Value) {
	switch x := v.(type) {
	case nil:
		b.WriteString("null")
	case bool:
		if x {
			b.WriteString("true")
		} else {
			b.WriteString("false")
		}
	case Number:
		b.WriteString(string(x))
	case string:
		EncodeString(b, x)
	case []Value:
		b.WriteByte('[')
		for i, e := range x {
			if i > 0 {
				b.WriteString(", ")
			}
			encode(b, e)
		}
		b.WriteByte(']')
	case *Object:
		b.WriteByte('{')
		for i, k := range x.Keys {
			if i > 0 {
				b.WriteString(", ")
			}
			EncodeString(b, k)
			b.WriteString(": ")
			encode(b, x.Vals[k])
		}
		b.WriteByte('}')
	default:
		panic(fmt.Sprintf("ojson: cannot encode %T", v))
	}
}

// EncodeString writes a JSON string, leaving non-ASCII text unescaped.
func EncodeString(b *strings.Builder, s string) {
	b.WriteByte('"')
	for i := 0; i < len(s); {
		c := s[i]
		if c < utf8.RuneSelf {
			switch {
			case c == '"':
				b.WriteString(`\"`)
			case c == '\\':
				b.WriteString(`\\`)
			case c == '\n':
				b.WriteString(`\n`)
			case c == '\r':
				b.WriteString(`\r`)
			case c == '\t':
				b.WriteString(`\t`)
			case c == '\b':
				b.WriteString(`\b`)
			case c == '\f':
				b.WriteString(`\f`)
			case c < 0x20:
				fmt.Fprintf(b, `\u%04x`, c)
			default:
				b.WriteByte(c)
			}
			i++
			continue
		}
		_, w := utf8.DecodeRuneInString(s[i:])
		b.WriteString(s[i : i+w])
		i += w
	}
	b.WriteByte('"')
}
