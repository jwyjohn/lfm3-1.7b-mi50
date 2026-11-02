// Package kernels carries the precompiled gfx906 code objects. Build them with build.sh;
// nothing is compiled or JIT-generated at run time.
package kernels

import _ "embed"

//go:embed decode.co
var Decode []byte

//go:embed prefill_attn.co
var PrefillAttn []byte

//go:embed gemm.co
var Gemm []byte

//go:embed ops.co
var Ops []byte
