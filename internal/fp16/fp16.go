// Package fp16 converts IEEE half precision with round-to-nearest-even, the rule
// used by the GPU and by torch, so packed weights match the running model.
package fp16

import "math"

// FromFloat32 rounds f to the nearest half, ties to even.
func FromFloat32(f float32) uint16 {
	b := math.Float32bits(f)
	sign := uint16(b>>16) & 0x8000
	biased := int32(b>>23) & 0xff
	man := b & 0x7fffff
	if biased == 0xff {
		if man != 0 {
			return sign | 0x7e00
		}
		return sign | 0x7c00
	}
	exp := biased - 127 + 15
	if exp >= 31 {
		return sign | 0x7c00
	}
	if exp <= 0 {
		if exp < -10 {
			return sign
		}
		man |= 0x800000
		shift := uint32(14 - exp)
		m := man >> shift
		rem := man & (1<<shift - 1)
		half := uint32(1) << (shift - 1)
		if rem > half || (rem == half && m&1 == 1) {
			m++
		}
		return sign | uint16(m)
	}
	m := man >> 13
	rem := man & 0x1fff
	out := sign | uint16(exp)<<10 | uint16(m)
	if rem > 0x1000 || (rem == 0x1000 && m&1 == 1) {
		out++ // a carry into the exponent is the correct next representable value
	}
	return out
}

// ToFloat32 is exact.
func ToFloat32(h uint16) float32 {
	sign := uint32(h&0x8000) << 16
	exp := uint32(h>>10) & 0x1f
	man := uint32(h & 0x3ff)
	switch exp {
	case 0:
		if man == 0 {
			return math.Float32frombits(sign)
		}
		e := uint32(127 - 15 + 1)
		for man&0x400 == 0 {
			man <<= 1
			e--
		}
		return math.Float32frombits(sign | e<<23 | (man&0x3ff)<<13)
	case 31:
		return math.Float32frombits(sign | 0xff<<23 | man<<13)
	}
	return math.Float32frombits(sign | (exp+127-15)<<23 | man<<13)
}
