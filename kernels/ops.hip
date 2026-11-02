// Native gfx906 kernels for everything in the LFM3 prefill path that vLLM ran through
// torch or Triton: embedding gather, RMS norm, causal conv, Q/K norm + RoPE, int8 KV
// store, per-head gate, the fp16 LM head and small host/state copies.
//
// Precision follows the vLLM ops they replace: values are computed in fp32 and rounded
// to fp16 at the same points, so later layers see the same activations.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace ops {

constexpr int THREADS = 256, WAVES = 4;
constexpr int H = 2048, D = 128;
constexpr int CONV_ROWS = 6;                 // conv state length (kernel width 7)
constexpr int BCX_B = 0, BCX_C = H, BCX_X = 2 * H;
constexpr int F_K = 2048, F_V = 2560, F_GATE = 3072;  // columns of the fused q/k/v/gate GEMM
constexpr float EPS = 1e-5f;

#define OPS_DPP(v, ctrl) __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), ctrl, 0xf, 0xf, false))
#define OPS_LANE(v, l) __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), l))

// Reductions over all 64 lanes of a wave: DPP inside 16-lane rows, readlane across them.
__device__ __forceinline__ float wave_sum(float v) {
  v += OPS_DPP(v, 0xb1);
  v += OPS_DPP(v, 0x4e);
  v += OPS_DPP(v, 0x141);
  v += OPS_DPP(v, 0x140);
  return (OPS_LANE(v, 0) + OPS_LANE(v, 16)) + (OPS_LANE(v, 32) + OPS_LANE(v, 48));
}

__device__ __forceinline__ float wave_max(float v) {
  v = fmaxf(v, OPS_DPP(v, 0xb1));
  v = fmaxf(v, OPS_DPP(v, 0x4e));
  v = fmaxf(v, OPS_DPP(v, 0x141));
  v = fmaxf(v, OPS_DPP(v, 0x140));
  return fmaxf(fmaxf(OPS_LANE(v, 0), OPS_LANE(v, 16)), fmaxf(OPS_LANE(v, 32), OPS_LANE(v, 48)));
}

__device__ __forceinline__ float round16(float v) { return __half2float(__float2half_rn(v)); }

// Eight consecutive fp16 values <-> fp32 registers, one 16-byte memory access.
__device__ __forceinline__ void unpack8(const uint4 u, float (&v)[8]) {
  const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    __half2 h;
    __builtin_memcpy(&h, &w[j], 4);
    const float2 f = __half22float2(h);
    v[2 * j] = f.x;
    v[2 * j + 1] = f.y;
  }
}

__device__ __forceinline__ uint4 pack8(const float (&v)[8]) {
  uint32_t w[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const __half2 h = __floats2half2_rn(v[2 * j], v[2 * j + 1]);
    __builtin_memcpy(&w[j], &h, 4);
  }
  return make_uint4(w[0], w[1], w[2], w[3]);
}

__device__ __forceinline__ const uint4* cu4(const __half* p) { return reinterpret_cast<const uint4*>(p); }
__device__ __forceinline__ uint4* mu4(__half* p) { return reinterpret_cast<uint4*>(p); }

// 1 / rms of a row held across all 256 threads, 8 elements each. `red` is block-shared.
__device__ __forceinline__ float row_inverse_rms(const float (&x)[8], float* red) {
  float s = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) s = fmaf(x[k], x[k], s);
  s = wave_sum(s);
  if ((threadIdx.x & 63) == 0) red[threadIdx.x >> 6] = s;
  __syncthreads();
  const float total = red[0] + red[1] + red[2] + red[3];
  return rsqrtf(total / H + EPS);
}

// ---------------------------------------------------------------- embedding
__global__ __launch_bounds__(THREADS) void embed_gather(const int32_t* ids, const __half* weight, __half* out,
                                                         int64_t out_stride) {
  const int row = blockIdx.x;
  const int id = ids[row];
  mu4(out + row * out_stride)[threadIdx.x] = cu4(weight + (int64_t)id * H)[threadIdx.x];
}

// ---------------------------------------------------------------- RMS norm
// out = round16(x / rms(x) * w)
__global__ __launch_bounds__(THREADS) void rms_norm(const __half* in, int64_t in_stride, const __half* w, __half* out,
                                                     int64_t out_stride) {
  __shared__ float red[WAVES];
  const int row = blockIdx.x, t = threadIdx.x;
  float x[8], wv[8];
  unpack8(cu4(in + row * in_stride)[t], x);
  unpack8(cu4(w)[t], wv);
  const float inv = row_inverse_rms(x, red);
#pragma unroll
  for (int k = 0; k < 8; ++k) x[k] = x[k] * inv * wv[k];
  mu4(out + row * out_stride)[t] = pack8(x);
}

// The residual step between two sublayers, fused:
//   r = round16(o / rms(o) * post_w);  h = round16(h + r);  n = round16(h / rms(h) * next_w)
__global__ __launch_bounds__(THREADS) void post_add_norm(const __half* o, int64_t o_stride, const __half* post_w,
                                                          __half* h, const __half* next_w, __half* n) {
  __shared__ float red[WAVES];
  const int row = blockIdx.x, t = threadIdx.x;
  float ov[8], hv[8], pw[8], nw[8];
  unpack8(cu4(o + row * o_stride)[t], ov);
  unpack8(cu4(h + row * H)[t], hv);
  unpack8(cu4(post_w)[t], pw);
  unpack8(cu4(next_w)[t], nw);
  const float a = row_inverse_rms(ov, red);
#pragma unroll
  for (int k = 0; k < 8; ++k) hv[k] = round16(hv[k] + round16(ov[k] * a * pw[k]));
  mu4(h + row * H)[t] = pack8(hv);
  __syncthreads();
  const float b = row_inverse_rms(hv, red);
#pragma unroll
  for (int k = 0; k < 8; ++k) hv[k] = hv[k] * b * nw[k];
  mu4(n + row * H)[t] = pack8(hv);
}

// ---------------------------------------------------------------- causal conv
// bxs rows: 0..5 the previous six B*x values (newest last), 6.. this chunk's B*x.
__global__ __launch_bounds__(THREADS) void conv_load_state(const __half* state, __half* bxs, int has_initial) {
  const int k = blockIdx.x, t = threadIdx.x;
  float v[8];
  if (has_initial) {
    const __half* s = state + (int64_t)(8 * t) * CONV_ROWS + k;
#pragma unroll
    for (int c = 0; c < 8; ++c) v[c] = __half2float(s[c * CONV_ROWS]);
  } else {
#pragma unroll
    for (int c = 0; c < 8; ++c) v[c] = 0.f;
  }
  mu4(bxs + k * H)[t] = pack8(v);
}

__global__ __launch_bounds__(THREADS) void conv_bx(const __half* bcx, __half* bxs) {
  const int row = blockIdx.x, t = threadIdx.x;
  float b[8], x[8];
  const __half* r = bcx + (int64_t)row * 3 * H;
  unpack8(cu4(r + BCX_B)[t], b);
  unpack8(cu4(r + BCX_X)[t], x);
#pragma unroll
  for (int k = 0; k < 8; ++k) b[k] *= x[k];
  mu4(bxs + (int64_t)(row + CONV_ROWS) * H)[t] = pack8(b);
}

// y = round16(C * round16(sum_k w[c][k] * bxs[row + k][c]))
__global__ __launch_bounds__(THREADS) void conv_apply(const __half* bxs, const __half* bcx, const __half* w, __half* y) {
  const int row = blockIdx.x, t = threadIdx.x;
  float tap[7][8], c[8], acc[8];
  // 8 channels x 7 taps is 56 consecutive fp16 values of [channel][tap].
  const uint4* wv = cu4(w + (int64_t)(8 * t) * 7);
#pragma unroll
  for (int i = 0; i < 7; ++i) unpack8(wv[i], tap[i]);
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = 0.f;
#pragma unroll
  for (int i = 0; i < 7; ++i) {
    float x[8];
    unpack8(cu4(bxs + (int64_t)(row + i) * H)[t], x);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
      const int flat = k * 7 + i;  // tap i of channel k in the flattened 7 x 8 registers
      acc[k] = fmaf(tap[flat >> 3][flat & 7], x[k], acc[k]);
    }
  }
  unpack8(cu4(bcx + (int64_t)row * 3 * H + BCX_C)[t], c);
#pragma unroll
  for (int k = 0; k < 8; ++k) acc[k] = c[k] * round16(acc[k]);
  mu4(y + (int64_t)row * H)[t] = pack8(acc);
}

__global__ __launch_bounds__(THREADS) void conv_store_state(const __half* bxs, __half* state, int tokens) {
  const int k = blockIdx.x, t = threadIdx.x;
  float v[8];
  unpack8(cu4(bxs + (int64_t)(tokens + k) * H)[t], v);
  __half* s = state + (int64_t)(8 * t) * CONV_ROWS + k;
#pragma unroll
  for (int c = 0; c < 8; ++c) s[c * CONV_ROWS] = __float2half_rn(v[c]);
}

// ---------------------------------------------------------------- attention inputs
// One block = one (token, head) of 64 lanes; lane l owns dims l and l + 64, a RoPE pair.
struct RotPair {
  float a, b;
};

__device__ __forceinline__ RotPair norm_rotate(float a, float b, const __half* w, const __half* cs) {
  const int l = threadIdx.x;
  const float inv = rsqrtf(wave_sum(a * a + b * b) / D + EPS);
  const float a1 = round16(a * inv * __half2float(w[l]));
  const float b1 = round16(b * inv * __half2float(w[l + 64]));
  const float c = __half2float(cs[l]), s = __half2float(cs[l + 64]);
  return {round16(a1 * c - b1 * s), round16(b1 * c + a1 * s)};
}

__global__ __launch_bounds__(64) void q_norm_rope(const __half* f, const __half* w, const __half* rope, __half* q,
                                                   int context, int f_stride) {
  const int row = blockIdx.x, tok = row / 16, head = row % 16, l = threadIdx.x;
  const __half* src = f + (int64_t)tok * f_stride + head * D;
  const RotPair r = norm_rotate(__half2float(src[l]), __half2float(src[l + 64]), w, rope + (int64_t)(context + tok) * D);
  __half* dst = q + (int64_t)tok * (16 * D) + head * D;
  dst[l] = __float2half_rn(r.a);
  dst[l + 64] = __float2half_rn(r.b);
}

// Per (token, head) scale; round half away from zero; matches vLLM's int8 cache writer.
__device__ __forceinline__ float quantize_pair(float a, float b, int8_t* rec) {
  const float amax = wave_max(fmaxf(fabsf(a), fabsf(b)));
  const float scale = fmaxf(amax / 127.f, 1e-6f);
  const float inv = 1.f / scale;
  float qa = a * inv, qb = b * inv;
  qa = qa >= 0.f ? qa + 0.5f : qa - 0.5f;
  qb = qb >= 0.f ? qb + 0.5f : qb - 0.5f;
  const int l = threadIdx.x;
  rec[l] = (int8_t)truncf(fminf(fmaxf(qa, -128.f), 127.f));
  rec[l + 64] = (int8_t)truncf(fminf(fmaxf(qb, -128.f), 127.f));
  return scale;
}

struct KvArgs {
  const __half* f;
  const __half* k_norm;
  const __half* rope;
  int8_t* kv;
  int64_t kv_block, kv_head, kv_slot;
  const int32_t* block_table;
  int32_t block_size, context, f_stride, pad;
};

__global__ __launch_bounds__(64) void kv_write(KvArgs p) {
  const int row = blockIdx.x, tok = row / 4, head = row % 4, l = threadIdx.x;
  const int pos = p.context + tok;
  const __half* src = p.f + (int64_t)tok * p.f_stride;
  const __half* ks = src + F_K + head * D;
  const __half* vs = src + F_V + head * D;
  const RotPair k = norm_rotate(__half2float(ks[l]), __half2float(ks[l + 64]), p.k_norm, p.rope + (int64_t)pos * D);
  const int64_t blk = p.block_table[pos / p.block_size];
  int8_t* rec = p.kv + blk * p.kv_block + head * p.kv_head + (int64_t)(pos % p.block_size) * p.kv_slot;
  const float k_scale = quantize_pair(k.a, k.b, rec);
  const float v_scale = quantize_pair(__half2float(vs[l]), __half2float(vs[l + 64]), rec + 132);
  if (l == 0) {
    *reinterpret_cast<float*>(rec + 128) = k_scale;
    *reinterpret_cast<float*>(rec + 260) = v_scale;
  }
}

// a[token][head][d] *= sigmoid(gate[token][head])
__global__ __launch_bounds__(THREADS) void gate_mul(__half* a, const __half* f, int f_stride) {
  const int row = blockIdx.x, t = threadIdx.x, head = t >> 4;
  const float g = __half2float(f[(int64_t)row * f_stride + F_GATE + head]);
  const float s = round16(1.f / (1.f + __expf(-g)));
  float v[8];
  uint4* p = mu4(a + (int64_t)row * (16 * D)) + t;
  unpack8(*p, v);
#pragma unroll
  for (int k = 0; k < 8; ++k) v[k] *= s;
  *p = pack8(v);
}

// ---------------------------------------------------------------- LM head
// logits = W x with the exact fp16 weights, rounded to fp16: what vLLM computes for the
// first token. Wave w serves rows w, w + nw, ...; fp32 accumulation via v_dot2.
__global__ __launch_bounds__(THREADS) void head_fp16(const __half* weight, const __half* x, float* logits, int rows) {
  __shared__ uint32_t xs[H / 2];
  const int t = threadIdx.x, lane = t & 63;
  const int wave = __builtin_amdgcn_readfirstlane(t >> 6);
  reinterpret_cast<uint4*>(xs)[t] = cu4(x)[t];
  __syncthreads();
  const int nw = gridDim.x * WAVES;
  for (int r = blockIdx.x * WAVES + wave; r < rows; r += nw) {
    const uint4* wv = cu4(weight + (int64_t)r * H + lane * 32);
    const uint4* xv = reinterpret_cast<const uint4*>(xs + lane * 16);
    float acc = 0.f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      const uint4 a = wv[i], b = xv[i];
      const uint32_t aw[4] = {a.x, a.y, a.z, a.w}, bw[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        typedef _Float16 h2 __attribute__((ext_vector_type(2)));
        h2 ha, hb;
        __builtin_memcpy(&ha, &aw[j], 4);
        __builtin_memcpy(&hb, &bw[j], 4);
        acc = __builtin_amdgcn_fdot2(ha, hb, acc, false);
      }
    }
    const float total = wave_sum(acc);
    if (lane == 0) logits[r] = round16(total);
  }
}

// Copies logits to host memory, optionally rounding to fp16 (the dtype vLLM samples from).
__global__ __launch_bounds__(THREADS) void logits_to_host(const float* src, float* dst, int count, int rounded) {
  const int i = (blockIdx.x * THREADS + threadIdx.x) * 4;
  if (i >= count) return;
  float4 v = *reinterpret_cast<const float4*>(src + i);
  if (rounded) {
    v.x = round16(v.x);
    v.y = round16(v.y);
    v.z = round16(v.z);
    v.w = round16(v.w);
  }
  *reinterpret_cast<float4*>(dst + i) = v;
  // Host memory is written straight through; wait for the write to complete so the
  // dispatch cannot finish before its data is visible.
  asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
}

// Moves host data into VRAM through the GPU's own L2. Host writes through the BAR bypass
// L2, which can keep serving old values for those addresses, so anything the host
// rewrites while kernels run must arrive this way. src == nullptr writes zeros.
__global__ __launch_bounds__(THREADS) void upload(const uint32_t* src, uint32_t* dst, int dwords) {
  const int i = blockIdx.x * THREADS + threadIdx.x;
  if (i >= dwords) return;
  dst[i] = src ? src[i] : 0u;
}

// Bulk host-to-VRAM copy through L2, 16 bytes per thread, 4 KiB per block.
__global__ __launch_bounds__(THREADS) void blit(const char* src, char* dst, int64_t bytes) {
  const int64_t off = (int64_t)blockIdx.x * 4096 + threadIdx.x * 16;
  if (off >= bytes) return;
  *reinterpret_cast<uint4*>(dst + off) = *reinterpret_cast<const uint4*>(src + off);
}

// Copies one conv-state slot across every conv layer: y = layer, x = 4 KiB chunk.
__global__ __launch_bounds__(THREADS) void state_copy(char* base, int64_t layer_stride, int64_t src_off, int64_t dst_off) {
  const int64_t chunk = (int64_t)blockIdx.y * layer_stride + (int64_t)blockIdx.x * 4096;
  const uint4 v = reinterpret_cast<const uint4*>(base + chunk + src_off)[threadIdx.x];
  reinterpret_cast<uint4*>(base + chunk + dst_off)[threadIdx.x] = v;
}

}  // namespace ops
