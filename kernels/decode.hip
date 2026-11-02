// LFM3 single-token decode megakernel for gfx906.
//
// One persistent launch runs all 24 layers, the final norm and the LM head.
// Weights are int8 with a fp16 scale per 128 input elements, stored biased
// (u = q + 128) so a byte becomes the fp16 value 1024 + u with one v_perm.
// Activations stay fp16, every dot product accumulates in fp32, the residual
// stream stays fp32. Attention reads vLLM's paged int8_per_token_head cache
// (K | K scale | V | V scale per slot) and keeps the softmax in fp32.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace lfm3v2 {

constexpr int H = 2048, I = 7680, QH = 16, KVH = 4, D = 128;
constexpr int THREADS = 256, WAVES = THREADS / 64;
constexpr int REC = 264;          // bytes per (slot, kv head) record
constexpr int SUB = 16;           // tokens per contiguous sub-tile
constexpr int SPLITS = 128;       // max attention splits (token ranges)
constexpr int QKVG = 2048 + 512 + 512 + 16;  // q + k + v + per-head gate = 3088 rows
static_assert(QKVG == 3088);
constexpr float LOG2E = 1.4426950408889634f;

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

// Streaming loads through the global address space: generic (flat) loads also count
// against lgkmcnt, so every LDS wait would drain them.
template <typename T>
__device__ __forceinline__ T gload_nt(const T* p) {
  typedef __attribute__((address_space(1))) const T GT;
  return __builtin_nontemporal_load((GT*)p);
}
template <typename T>
__device__ __forceinline__ T gload(const T* p) {
  typedef __attribute__((address_space(1))) const T GT;
  return *(GT*)p;
}

// Value-parameter bit casts: clang's __builtin_bit_cast on an ext_vector element
// (v.y) reads element 0, so always cast from a plain scalar.
__device__ __forceinline__ half2_t as_h2(uint32_t u) {
  half2_t h;
  __builtin_memcpy(&h, &u, 4);
  return h;
}

struct Layer {
  int64_t kind;  // 0 conv, 1 full attention, 2 sliding attention
  const __half* op_norm;
  const __half* ffn_norm;
  const __half* post_op_norm;
  const __half* post_ffn_norm;
  const uint8_t* w_mix; const __half* s_mix;  // conv in_proj (channel-interleaved) | qkv+gate
  const uint8_t* w_out; const __half* s_out;  // conv out_proj | o_proj
  const uint8_t* w13; const __half* s13;      // gate/up rows interleaved
  const uint8_t* w2; const __half* s2;
  const __half* conv_w;                       // [2048][7]
  __half* conv_state; int64_t cs_slot, cs_dim, cs_t; const int32_t* state_index;
  const __half* q_norm; const __half* k_norm; const __half* rope; int64_t rotary;
  int8_t* kv; int64_t kv_block, kv_head, kv_slot, block_size;
  const int32_t* block_table; const int32_t* seq_lens; const int64_t* slot_mapping;
  int64_t window;
};

struct Globals {
  const Layer* layers;
  const int32_t* input_ids;
  const __half* embed;
  const __half* final_norm;
  const uint8_t* w_head; const __half* s_head;
  float* logits;
  __half* hidden_out;
  float* ws;
  uint32_t* barrier;
  int64_t num_layers;
  int64_t vocab;
  int64_t* trace;  // optional: phase end timestamps (s_memrealtime, 100 MHz) from block 0
  // In-kernel greedy sampling (sample != 0): token = argmax of the logits after vLLM's
  // repetition penalty over the tokens marked in `seen` (prompt and outputs; the input
  // token is added at kernel start). The logits buffer itself stays raw.
  uint32_t* seen;           // [vocab / 32] token-presence bits
  const float* rep;         // repetition penalty (device scalar)
  int64_t sample;           // 0 off; 1 sample the fp32 logits; 2 sample them rounded to fp16
                            // (the dtype vLLM's sampler receives), so ties and penalties match
  uint64_t* best;           // packed (ordered logit, ~index) running max
  uint32_t* done;           // blocks finished with the head (last one publishes)
  int32_t* token;           // sampled token
};

constexpr int SEEN_WORDS_MAX = 128000 / 32;

// Workspace layout (floats).
constexpr int WS_H0 = 0, WS_H1 = WS_H0 + H, WS_MIX = WS_H1 + H;
constexpr int WS_Y = WS_MIX + 6144, WS_O = WS_Y + H, WS_A = WS_O + H, WS_D = WS_A + I;
constexpr int WS_BCX = WS_D + H;            // conv in_proj rows (channel-interleaved)
constexpr int WS_G13 = WS_BCX + 6144;        // gate/up rows (interleaved)
constexpr int WS_TRASH = WS_G13 + 2 * I;     // sink for inactive lanes of branch-free stores
constexpr int WS_PART = WS_TRASH + 64;       // [QH][SPLITS][D + 2]
constexpr int WS_FLOATS = WS_PART + QH * SPLITS * (D + 2);

// Attention tile: SUB tokens x 4 kv heads, records laid out [token][head][REC].
// Per-wave attention scratch: each wave owns one kv head, so the token tile is
// private to the wave and needs no block barrier.
constexpr int BT_MAX = 512;  // block-table entries staged in LDS per split (>= 131072 / 60 / 16 + 2)

struct AttnShared {
  alignas(16) uint8_t tile[2][SUB * KVH * REC];  // two 16-token groups, [token][kv head][record] (NHD)
  alignas(16) uint8_t fresh[KVH * REC];
  alignas(16) int8_t q8[KVH][4][D];              // query heads quantized per 32-dim group
  float qs8[KVH][4][4];                          // their group scales
  alignas(16) __half p[WAVES][4][2 * SUB];       // scaled fp16 probabilities x V scale (two groups)
  int32_t blocks[BT_MAX];                        // this split's block-table entries
};

struct GemvShared {
  alignas(16) __half x[I];
  float xs[I / 8];  // 1152 * sum of each 8-element piece of x
};

struct HeadShared {
  GemvShared g;               // x of the head GEMV (same offset as Shared::g)
  uint32_t seen[SEEN_WORDS_MAX];
};

union alignas(16) Shared {
  GemvShared g;
  AttnShared attn;
  HeadShared head;
};

struct Smem {
  Shared u;
  float red[WAVES * 4];
};

// Full-wave reductions (all 64 lanes active): DPP inside each 16-lane row, then the four
// row results via readlane. Avoids ds_bpermute round trips through the LDS pipeline.
#define LFM3_DPP(v, ctrl) __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), ctrl, 0xf, 0xf, false))
#define LFM3_LANE(v, l) __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), l))

__device__ __forceinline__ float wave_sum(float v) {
  v += LFM3_DPP(v, 0xb1);
  v += LFM3_DPP(v, 0x4e);
  v += LFM3_DPP(v, 0x141);
  v += LFM3_DPP(v, 0x140);
  return (LFM3_LANE(v, 0) + LFM3_LANE(v, 16)) + (LFM3_LANE(v, 32) + LFM3_LANE(v, 48));
}

__device__ __forceinline__ float row_max16(float v) {
  v = fmaxf(v, LFM3_DPP(v, 0xb1));
  v = fmaxf(v, LFM3_DPP(v, 0x4e));
  v = fmaxf(v, LFM3_DPP(v, 0x141));
  return fmaxf(v, LFM3_DPP(v, 0x140));
}

__device__ __forceinline__ float wave_max(float v) {
  v = row_max16(v);
  return fmaxf(fmaxf(LFM3_LANE(v, 0), LFM3_LANE(v, 16)), fmaxf(LFM3_LANE(v, 32), LFM3_LANE(v, 48)));
}

// Max over each 32-lane half; every lane gets its half's result.
__device__ __forceinline__ float half_max(float v) {
  v = row_max16(v);
  const float lo = fmaxf(LFM3_LANE(v, 0), LFM3_LANE(v, 16)), hi = fmaxf(LFM3_LANE(v, 32), LFM3_LANE(v, 48));
  return (threadIdx.x & 32) ? hi : lo;
}

// Split grid barrier: arrive publishes this block's writes; wait blocks until every block
// arrived. Work that does not read other blocks' data (weight / KV prefetch) can be issued
// between the two so its memory latency overlaps the barrier.
__device__ __forceinline__ void grid_arrive(uint32_t* bar, uint32_t gen) {
  __syncthreads();
  if (threadIdx.x == 0) {
    uint32_t ticket = __hip_atomic_fetch_add(bar, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
    if (ticket == gridDim.x - 1) {
      __hip_atomic_store(bar, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      __hip_atomic_store(bar + 1, gen + 1, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

__device__ __forceinline__ void grid_wait(uint32_t* bar, uint32_t& gen) {
  if (threadIdx.x == 0)
    while (__hip_atomic_load(bar + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) == gen)
      __builtin_amdgcn_s_sleep(1);
  gen += 1;
  __syncthreads();
  __atomic_thread_fence(__ATOMIC_ACQUIRE);
}

__device__ __forceinline__ void trace_mark(int64_t* trace, int& n) {
  if (trace && blockIdx.x == 0 && threadIdx.x == 0) trace[n] = (int64_t)__builtin_amdgcn_s_memrealtime();
  ++n;
}

// ---------------------------------------------------------------- GEMV
// Lane l owns the 8-byte pieces l + 64 i of a row (i < K / 512). The group of
// piece p is p / 16 = 4 i + l / 16; scales are stored per row as
// [quarter = l / 16][i] so a lane reads its NI scales contiguously.
template <int K>
struct Geo {
  static constexpr int NI = K / 512;
  static constexpr int SROW = K == 2048 ? 16 : 64;   // fp16 scales per row (padded)
  static constexpr int SQ = K == 2048 ? 4 : 16;      // scales per quarter (padded)
};

__device__ __forceinline__ float dot8(u32x2 w, u32x4 x, float acc) {
  const uint32_t c = 0x64646464u;
  const uint32_t w0 = w.x, w1 = w.y, x0 = x.x, x1 = x.y, x2 = x.z, x3 = x.w;
  acc = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(c, w0, 0x04010400u)), as_h2(x0), acc, false);
  acc = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(c, w0, 0x04030402u)), as_h2(x1), acc, false);
  acc = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(c, w1, 0x04010400u)), as_h2(x2), acc, false);
  acc = __builtin_amdgcn_fdot2(as_h2(__builtin_amdgcn_perm(c, w1, 0x04030402u)), as_h2(x3), acc, false);
  return acc;
}

// Wave-wide sum with DPP inside 16-lane rows and readlane across rows: no LDS
// traffic and no memory counters, so it does not serialize pending loads.
__device__ __forceinline__ float wave_total(float v) {
  int i = __builtin_bit_cast(int, v);
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(i, 0xb1, 0xf, 0xf, false));  // quad_perm 1,0,3,2
  i = __builtin_bit_cast(int, v);
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(i, 0x4e, 0xf, 0xf, false));  // quad_perm 2,3,0,1
  i = __builtin_bit_cast(int, v);
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(i, 0x141, 0xf, 0xf, false)); // row_half_mirror
  i = __builtin_bit_cast(int, v);
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(i, 0x140, 0xf, 0xf, false)); // row_mirror
  i = __builtin_bit_cast(int, v);
  return (__builtin_bit_cast(float, __builtin_amdgcn_readlane(i, 0)) +
          __builtin_bit_cast(float, __builtin_amdgcn_readlane(i, 16))) +
         (__builtin_bit_cast(float, __builtin_amdgcn_readlane(i, 32)) +
          __builtin_bit_cast(float, __builtin_amdgcn_readlane(i, 48)));
}

template <int K, int R>
struct GemvBuf {
  u32x2 w[R][Geo<K>::NI];
  u32x2 s[R][Geo<K>::SQ / 4];
};

template <int K, int R>
__device__ __forceinline__ void gemv_load(GemvBuf<K, R>& b, const uint8_t* __restrict__ W,
                                          const __half* __restrict__ S, int base, int last) {
  using G = Geo<K>;
  const int lane = threadIdx.x & 63, quarter = lane >> 4;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int row = min(base + r, last);
    const u32x2* src = reinterpret_cast<const u32x2*>(W + (size_t)row * K);
#pragma unroll
    for (int i = 0; i < G::NI; ++i) b.w[r][i] = gload_nt(src + lane + 64 * i);
    const u32x2* sp = reinterpret_cast<const u32x2*>(S + (size_t)row * G::SROW + quarter * G::SQ);
#pragma unroll
    for (int i = 0; i < G::SQ / 4; ++i) b.s[r][i] = gload(sp + i);
  }
}

// x pieces and biases, held in registers across the GEMV.
template <int K>
struct GemvX {
  u32x4 x[Geo<K>::NI];
  float xs[Geo<K>::NI];
};

template <int K>
__device__ __forceinline__ void gemv_xreg(GemvX<K>& gx, const __half* x) {
  const int lane = threadIdx.x & 63;
  const float* xs = reinterpret_cast<const GemvShared*>(x)->xs;
#pragma unroll
  for (int i = 0; i < Geo<K>::NI; ++i) {
    gx.x[i] = *reinterpret_cast<const u32x4*>(x + 8 * (lane + 64 * i));
    gx.xs[i] = xs[lane + 64 * i];
  }
}

template <int K, int R>
__device__ __forceinline__ void gemv_compute_reg(const GemvBuf<K, R>& b, const GemvX<K>& gx, int base, int r1,
                                                 float* out, float* trash) {
  using G = Geo<K>;
  const int lane = threadIdx.x & 63;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
  for (int i = 0; i < G::NI; ++i) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint32_t word = (i >> 1) & 1 ? b.s[r][i >> 2].y : b.s[r][i >> 2].x;
      const float sc = __half2float(__ushort_as_half((unsigned short)(i & 1 ? word >> 16 : word & 0xffffu)));
      acc[r] = fmaf(sc, dot8(b.w[r][i], gx.x[i], 0.f) - gx.xs[i], acc[r]);
    }
  }
  float outv = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float t = wave_total(acc[r]);
    outv = lane == r ? t : outv;
  }
  const int row = base + lane;
  float* p = (lane < R && row < r1) ? out + row : trash + lane;
  *p = outv;
}

// Same arithmetic as gemv_compute_reg; returns the value lane r would store (row base + r).
template <int K, int R>
__device__ __forceinline__ float gemv_compute_reg_value(const GemvBuf<K, R>& b, const GemvX<K>& gx) {
  using G = Geo<K>;
  const int lane = threadIdx.x & 63;
  float acc[R];
#pragma unroll
  for (int r = 0; r < R; ++r) acc[r] = 0.f;
#pragma unroll
  for (int i = 0; i < G::NI; ++i) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const uint32_t word = (i >> 1) & 1 ? b.s[r][i >> 2].y : b.s[r][i >> 2].x;
      const float sc = __half2float(__ushort_as_half((unsigned short)(i & 1 ? word >> 16 : word & 0xffffu)));
      acc[r] = fmaf(sc, dot8(b.w[r][i], gx.x[i], 0.f) - gx.xs[i], acc[r]);
    }
  }
  float outv = 0.f;
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const float t = wave_total(acc[r]);
    outv = lane == r ? t : outv;
  }
  return outv;
}

// out = W x over `units` units of R consecutive rows. Unit u goes to wave
// u % NW so that, at any moment, all waves stream one contiguous region of the
// matrix (rows owned by a single wave would put every wave at the same
// power-of-two offset and camp on the same HBM channels). NB register buffers
// keep NB-1 units in flight; tail units clamp to the last row and store to trash.
// pre() runs (on every thread) after this wave's first weight loads are issued and before
// x is read: it waits for the grid barrier and builds x, so the loads overlap both.
template <int K, int R, int NB, typename Pre>
__device__ __forceinline__ void gemv_store(const uint8_t* __restrict__ W, const __half* __restrict__ S,
                                           int units, const __half* x, float* out, float* trash, Pre&& pre) {
  const int nw = gridDim.x * WAVES;
  const int w = blockIdx.x * WAVES + __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const bool active = w < units;
  const int rows = units * R, last = rows - 1;
  GemvBuf<K, R> buf[NB];
  if (active) {
#pragma unroll
    for (int s = 0; s < NB - 1; ++s) {
      gemv_load<K, R>(buf[s], W, S, (w + s * nw) * R, last);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  pre();
  if (!active) return;
  GemvX<K> gx;
  gemv_xreg<K>(gx, x);
  for (int u = w; u < units; u += NB * nw) {
#pragma unroll
    for (int s = 0; s < NB; ++s) {
      gemv_load<K, R>(buf[(s + NB - 1) % NB], W, S, (u + (s + NB - 1) * nw) * R, last);
      // No fence here: the next unit's loads may interleave with this unit's math.
      gemv_compute_reg<K, R>(buf[s], gx, (u + s * nw) * R, rows, out, trash);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
}

// ---------------------------------------------------------------- norms
// h_new = h_old + post(v) (when v != nullptr); x = norm(h_new) * w -> LDS fp16
// plus the GEMV bias table. Thread t owns elements [8 t, 8 t + 8) (one piece).
// One block reduction: with a = 1 / rms(v), |h_new|^2 = |h|^2 + 2 a (h.v.pw) + a^2 |v.pw|^2.
// Inputs of residual_norm that do not depend on the phase just finished (h_old was written
// at least one grid barrier earlier; the weights are constant): loaded before grid_wait.
struct NormIn {
  float hv[8], pw[8], nw[8];
};

__device__ __forceinline__ void norm_prefetch(NormIn& n, const float* h_old, const __half* post_w,
                                              const __half* norm_w) {
  const int i0 = 8 * threadIdx.x;
  const float4* h4 = reinterpret_cast<const float4*>(h_old + i0);
  const float4 a = h4[0], b = h4[1];
  n.hv[0] = a.x; n.hv[1] = a.y; n.hv[2] = a.z; n.hv[3] = a.w; n.hv[4] = b.x; n.hv[5] = b.y; n.hv[6] = b.z; n.hv[7] = b.w;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    n.pw[k] = post_w ? __half2float(post_w[i0 + k]) : 0.f;
    n.nw[k] = __half2float(norm_w[i0 + k]);
  }
}

__device__ __forceinline__ void residual_norm_fin(const NormIn& n, const float* v, float* dst, __half* x,
                                                  float* red) {
  const int i0 = 8 * threadIdx.x;
  float hv[8], vv[8], pw[8], nw[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    hv[k] = n.hv[k];
    pw[k] = n.pw[k];
    nw[k] = n.nw[k];
  }
  if (v) {
    const float4* v4 = reinterpret_cast<const float4*>(v + i0);
    const float4 a = v4[0], b = v4[1];
    vv[0] = a.x; vv[1] = a.y; vv[2] = a.z; vv[3] = a.w; vv[4] = b.x; vv[5] = b.y; vv[6] = b.z; vv[7] = b.w;
  } else {
#pragma unroll
    for (int k = 0; k < 8; ++k) vv[k] = 0.f;
  }
  float s_hh = 0.f, s_vv = 0.f, s_hv = 0.f, s_ww = 0.f;
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float vw = vv[k] * pw[k];
    s_hh = fmaf(hv[k], hv[k], s_hh);
    s_vv = fmaf(vv[k], vv[k], s_vv);
    s_hv = fmaf(hv[k], vw, s_hv);
    s_ww = fmaf(vw, vw, s_ww);
  }
  s_hh = wave_total(s_hh); s_vv = wave_total(s_vv); s_hv = wave_total(s_hv); s_ww = wave_total(s_ww);
  const int w = threadIdx.x >> 6;
  if ((threadIdx.x & 63) == 0) { red[4 * w] = s_hh; red[4 * w + 1] = s_vv; red[4 * w + 2] = s_hv; red[4 * w + 3] = s_ww; }
  __syncthreads();
  float t_hh = 0.f, t_vv = 0.f, t_hv = 0.f, t_ww = 0.f;
#pragma unroll
  for (int i = 0; i < WAVES; ++i) {
    t_hh += red[4 * i]; t_vv += red[4 * i + 1]; t_hv += red[4 * i + 2]; t_ww += red[4 * i + 3];
  }
  const float a = v ? rsqrtf(t_vv / H + 1e-5f) : 0.f;
  const float hh = t_hh + 2.f * a * t_hv + a * a * t_ww;
  const float inv = rsqrtf(fmaxf(hh, 0.f) / H + 1e-5f);
  float xs = 0.f;
  __half xh[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    hv[k] = fmaf(vv[k] * a, pw[k], hv[k]);
    xh[k] = __float2half_rn(hv[k] * inv * nw[k]);
    xs += __half2float(xh[k]);
  }
  *reinterpret_cast<uint4*>(x + i0) = *reinterpret_cast<const uint4*>(xh);
  reinterpret_cast<GemvShared*>(x)->xs[threadIdx.x] = 1152.f * xs;
  if (blockIdx.x == 0) {
    float4* d4 = reinterpret_cast<float4*>(dst + i0);
    d4[0] = make_float4(hv[0], hv[1], hv[2], hv[3]);
    d4[1] = make_float4(hv[4], hv[5], hv[6], hv[7]);
  }
  __syncthreads();
}

// x = src (fp32) -> LDS fp16 plus the GEMV bias table, one 8-element piece at a time.
template <int N>
__device__ __forceinline__ void load_x(const float* src, __half* x) {
  float* xs = reinterpret_cast<GemvShared*>(x)->xs;
  for (int p = threadIdx.x; p < N / 8; p += THREADS) {
    const float4* s4 = reinterpret_cast<const float4*>(src + 8 * p);
    const float4 a = s4[0], b = s4[1];
    __half xh[8] = {__float2half_rn(a.x), __float2half_rn(a.y), __float2half_rn(a.z), __float2half_rn(a.w),
                    __float2half_rn(b.x), __float2half_rn(b.y), __float2half_rn(b.z), __float2half_rn(b.w)};
    float t = 0.f;
#pragma unroll
    for (int k = 0; k < 8; ++k) t += __half2float(xh[k]);
    *reinterpret_cast<uint4*>(x + 8 * p) = *reinterpret_cast<const uint4*>(xh);
    xs[p] = 1152.f * t;
  }
  __syncthreads();
}

// Quantizes one 128-vector held as lane values (v0 at d = lane, v1 at lane + 64)
// exactly like vLLM's per-token-head int8 cache writer.
__device__ __forceinline__ float quantize_pair(float v0, float v1, int8_t* out) {
  float amax = wave_max(fmaxf(fabsf(v0), fabsf(v1)));
  float scale = fmaxf(amax / 127.f, 1e-6f);
  float inv = 1.f / scale;
  int lane = threadIdx.x & 63;
  float a = v0 * inv, b = v1 * inv;
  a = a >= 0 ? a + 0.5f : a - 0.5f;
  b = b >= 0 ? b + 0.5f : b - 0.5f;
  out[lane] = (int8_t)fminf(fmaxf(truncf(a), -128.f), 127.f);
  out[lane + 64] = (int8_t)fminf(fmaxf(truncf(b), -128.f), 127.f);
  return scale;
}

// RMS-norm one head (weights w), then rotate pairs (d, d + rotary / 2).
// Per-lane constants of norm_rope: norm weights of dims (lane, lane + 64) and the rotation
// (cos, sin) at pos; loaded before the phase's grid_wait.
struct RopeIn {
  float w0, w1, c, s;
};

__device__ __forceinline__ RopeIn rope_prefetch(const __half* w, const Layer& L, int pos) {
  const int lane = threadIdx.x & 63;
  const __half* cs = L.rope + (int64_t)pos * L.rotary;
  return {__half2float(w[lane]), __half2float(w[lane + 64]), __half2float(cs[lane]), __half2float(cs[64 + lane])};
}

__device__ __forceinline__ void norm_rope(float& v0, float& v1, const RopeIn& r) {
  // rotary == head_dim for LFM3 (zero-tail partial RoPE), so the pair is (lane, lane + 64).
  const float w0 = r.w0, w1 = r.w1, c = r.c, s = r.s;
  const float inv = rsqrtf(wave_sum(v0 * v0 + v1 * v1) / D + 1e-5f);
  const float a = v0 * inv * w0, b = v1 * inv * w1;
  v0 = a * c - b * s;
  v1 = b * c + a * s;
}

// DPP reductions inside a 16-lane row: every lane of the row gets the result.
__device__ __forceinline__ float row_max(float v) {
  v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0xb1, 0xf, 0xf, false)));
  v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x4e, 0xf, 0xf, false)));
  v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x141, 0xf, 0xf, false)));
  v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x140, 0xf, 0xf, false)));
  return v;
}

__device__ __forceinline__ float row_sum(float v) {
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0xb1, 0xf, 0xf, false));
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x4e, 0xf, 0xf, false));
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x141, 0xf, 0xf, false));
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x140, 0xf, 0xf, false));
  return v;
}

// One 16-token group of all 4 kv heads is contiguous in vLLM's NHD cache
// (block_size % 16 == 0): 16896 bytes = 1056 pieces of 16 bytes.
constexpr int TILE_BYTES = SUB * KVH * REC;
constexpr int TILE_LOADS = (TILE_BYTES / 16 + THREADS - 1) / THREADS;
constexpr float PSCALE = 4096.f;  // keeps p * v_scale in the fp16 normal range

__device__ __forceinline__ void tile_load(u32x4 (&r)[TILE_LOADS], const Layer& L, int t0) {
  const int64_t bs = L.block_size;
  const uint8_t* base = reinterpret_cast<const uint8_t*>(L.kv) + (int64_t)L.block_table[t0 / bs] * L.kv_block +
                        (t0 % bs) * L.kv_slot;
#pragma unroll
  for (int k = 0; k < TILE_LOADS; ++k) {
    const int p = min((int)threadIdx.x + THREADS * k, TILE_BYTES / 16 - 1);
    r[k] = gload_nt(reinterpret_cast<const u32x4*>(base) + p);
  }
}

// Block barrier that only orders LDS traffic: __syncthreads() also waits for every
// outstanding global load, which would drain the K/V prefetch each iteration.
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// Paged KV geometry held in scalar registers for the attention loop: reading the Layer
// descriptor (global memory) inside the loop needs vmcnt(0), which drains the K/V prefetch.
struct KvView {
  const uint8_t* kv;
  int64_t kv_block, kv_slot;
  int bs;
};

__device__ __forceinline__ int64_t uniform64(int64_t v) {
  const uint64_t u = (uint64_t)v;
  return (int64_t)(((uint64_t)(uint32_t)__builtin_amdgcn_readfirstlane((int)(u >> 32)) << 32) |
                   (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)u));
}

// Pieces [k0, k1) of tile_load_staged: the loads are spread through the compute so a
// single wave per SIMD does not stall issuing ten 16-byte loads back to back.
__device__ __forceinline__ void tile_load_part(u32x4 (&r)[TILE_LOADS], const KvView& kv, int t0,
                                               const int32_t* blocks, int b0, int k0, int k1) {
  const uint8_t* base = kv.kv + (int64_t)blocks[t0 / kv.bs - b0] * kv.kv_block + (t0 % kv.bs) * kv.kv_slot;
#pragma unroll
  for (int k = 0; k < TILE_LOADS; ++k) {
    if (k < k0 || k >= k1) continue;
    const int p = min((int)threadIdx.x + THREADS * k, TILE_BYTES / 16 - 1);
    r[k] = gload_nt(reinterpret_cast<const u32x4*>(base) + p);
  }
}

__device__ __forceinline__ void tile_store(const u32x4 (&r)[TILE_LOADS], uint8_t* tile) {
#pragma unroll
  for (int k = 0; k < TILE_LOADS; ++k) {
    const int p = threadIdx.x + THREADS * k;
    if (p < TILE_BYTES / 16) reinterpret_cast<u32x4*>(tile)[p] = r[k];
  }
}

__device__ __forceinline__ float xor_add(float v, int mask) { return v + __shfl_xor(v, mask); }

// Block = one token range (split) over all 4 kv heads; wave w owns kv head w and
// its 4 query heads. QK lanes: 16 * dim quarter + token; softmax lanes:
// 16 * query head + token; PV lanes: 16 * query head + 8-dim group.
// Writes (m, l, acc) partials per query head and split.
// pre() (every thread, every block) runs after the first K/V tiles are requested and
// before mix (written by other blocks) is read.
template <typename Pre>
__device__ __forceinline__ void attention_phase(const Layer& L, const float* mix, float* part,
                                                AttnShared& sm, int splits, Pre&& pre) {
  const int split = blockIdx.x;
  if (split >= splits) {
    pre();
    return;
  }
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), lane = threadIdx.x & 63;
  const int len = L.seq_lens[0];
  const int pos = len - 1;
  const int start = L.window > 0 ? max(0, len - (int)L.window) : 0;
  const int g_begin = start / SUB, g_total = (len + SUB - 1) / SUB - g_begin;
  const int g0 = g_begin + g_total * split / splits, g1 = g_begin + g_total * (split + 1) / splits;

  u32x4 nxt[TILE_LOADS];
  if (g0 < g1) tile_load(nxt, L, g0 * SUB);
  // Stage this split's block ids (visible after the barrier that ends the q prologue).
  // The host guarantees nb <= BT_MAX (decode_v2.py checks max_model_len).
  const int b0 = g0 * SUB / (int)L.block_size;
  const int nb = g0 < g1 ? min((g1 * SUB - 1) / (int)L.block_size - b0 + 1, BT_MAX) : 0;
  KvView kvv;
  kvv.kv = reinterpret_cast<const uint8_t*>(uniform64(reinterpret_cast<int64_t>(L.kv)));
  kvv.kv_block = uniform64(L.kv_block);
  kvv.kv_slot = uniform64(L.kv_slot);
  kvv.bs = __builtin_amdgcn_readfirstlane((int)L.block_size);
  for (int i = threadIdx.x; i < nb; i += THREADS) sm.blocks[i] = L.block_table[b0 + i];
  u32x4 nxt2[TILE_LOADS];
  if (g0 + 1 < g1) tile_load(nxt2, L, (g0 + 1) * SUB);
  const RopeIn rq = rope_prefetch(L.q_norm, L, pos), rk = rope_prefetch(L.k_norm, L, pos);
  pre();

  // Fresh q (fp16), k and v (int8 records) for kv head `wave`; all loads issued up front.
  {
    float qv0[4], qv1[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      qv0[j] = mix[(wave * 4 + j) * D + lane];
      qv1[j] = mix[(wave * 4 + j) * D + lane + 64];
    }
    float k0 = mix[2048 + wave * D + lane], k1 = mix[2048 + wave * D + lane + 64];
    float v0 = mix[2560 + wave * D + lane], v1 = mix[2560 + wave * D + lane + 64];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      norm_rope(qv0[j], qv1[j], rq);
      // int8 per 32-dim group: dims lane (group lane / 32) and lane + 64 (group 2 + lane / 32).
      float m0 = fabsf(qv0[j]), m1 = fabsf(qv1[j]);
      m0 = half_max(m0);
      m1 = half_max(m1);
      const float s0 = fmaxf(m0 / 127.f, 1e-20f), s1 = fmaxf(m1 / 127.f, 1e-20f);
      sm.q8[wave][j][lane] = (int8_t)rintf(qv0[j] / s0);
      sm.q8[wave][j][lane + 64] = (int8_t)rintf(qv1[j] / s1);
      if ((lane & 31) == 0) {
        sm.qs8[wave][j][lane >> 5] = s0;
        sm.qs8[wave][j][2 + (lane >> 5)] = s1;
      }
    }
    uint8_t* rec = sm.fresh + wave * REC;
    norm_rope(k0, k1, rk);
    float ks = quantize_pair(k0, k1, reinterpret_cast<int8_t*>(rec));
    float vs = quantize_pair(v0, v1, reinterpret_cast<int8_t*>(rec + 132));
    if (lane == 0) {
      *reinterpret_cast<float*>(rec + 128) = ks;
      *reinterpret_cast<float*>(rec + 260) = vs;
    }
  }
  __syncthreads();
  if (g0 <= pos / SUB && pos / SUB < g1) {
    int64_t slot = L.slot_mapping[0];
    int64_t blk = slot / L.block_size, off = slot % L.block_size;
    for (int i = threadIdx.x; i < KVH * REC / 4; i += THREADS) {
      const int h = i / (REC / 4), w = i % (REC / 4);
      uint32_t* dst = reinterpret_cast<uint32_t*>(reinterpret_cast<uint8_t*>(L.kv) + blk * L.kv_block +
                                                  h * L.kv_head + off * L.kv_slot);
      dst[w] = reinterpret_cast<const uint32_t*>(sm.fresh + h * REC)[w];
    }
  }

  const int row = lane >> 4, col = lane & 15;  // row: dim quarter (QK) / query head (softmax, PV)
  const float qk_scale = 0.08838834764831845f * LOG2E;
  // int8 q dims [32 row, 32 row + 32) of the 4 query heads and their group scales.
  uint32_t qreg[4][8];
  float qsc[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
#pragma unroll
    for (int c = 0; c < 8; ++c) qreg[j][c] = reinterpret_cast<const uint32_t*>(sm.q8[wave][j])[row * 8 + c];
    qsc[j] = sm.qs8[wave][j][row] * qk_scale;
  }
  // Softmax state of query head `row` (identical across its 16 lanes). PV lanes: tokens
  // 8 half .. +7 of each group, dims 4 j4 .. +3, all 4 query heads (V converted once).
  const int half = lane >> 5, j4 = lane & 31;
  float m_run = -INFINITY, l_run = 0.f;
  float acc[4][4];
#pragma unroll
  for (int h = 0; h < 4; ++h)
#pragma unroll
    for (int d = 0; d < 4; ++d) acc[h][d] = 0.f;
  const uint32_t c64 = 0x64646464u;
  const half2_t bias = {(_Float16)1152.f, (_Float16)1152.f};

  auto qk = [&](const uint8_t* rec, int (&si)[4]) {
    const u32x2* kr = reinterpret_cast<const u32x2*>(rec + row * 32);
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const u32x2 kk = kr[c];
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        si[j] = __builtin_amdgcn_sdot4(kk.x, qreg[j][2 * c], si[j], false);
        si[j] = __builtin_amdgcn_sdot4(kk.y, qreg[j][2 * c + 1], si[j], false);
      }
    }
  };

  // Both 16-token groups of an iteration at once: one softmax update and one P exchange per
  // 32 tokens, and two independent instruction streams through QK.
  auto process2 = [&](const uint8_t* ta, const uint8_t* tb, int t0, bool two, auto&& mid) {
    mid(0);
    const uint8_t* ra = ta + (col * KVH + wave) * REC;
    const uint8_t* rb = tb + (col * KVH + wave) * REC;
    int sa[4] = {0, 0, 0, 0}, sb[4] = {0, 0, 0, 0};
    qk(ra, sa);
    qk(rb, sb);
    mid(1);
    float fa[4], fb[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      fa[j] = (float)sa[j] * qsc[j];
      fb[j] = (float)sb[j] * qsc[j];
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      fa[j] = xor_add(fa[j], 16);
      fb[j] = xor_add(fb[j], 16);
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      fa[j] = xor_add(fa[j], 32);
      fb[j] = xor_add(fb[j], 32);
    }
    // Softmax lanes: row = query head, col = token (of group a and of group b).
    const float ya = row == 0 ? fa[0] : row == 1 ? fa[1] : row == 2 ? fa[2] : fa[3];
    const float yb = row == 0 ? fb[0] : row == 1 ? fb[1] : row == 2 ? fb[2] : fb[3];
    const int tka = t0 + col, tkb = t0 + SUB + col;
    const bool va = tka >= start && tka < len, vb = two && tkb >= start && tkb < len;
    const float ksa = *reinterpret_cast<const float*>(ra + 128), vsa = *reinterpret_cast<const float*>(ra + 260);
    const float ksb = *reinterpret_cast<const float*>(rb + 128), vsb = *reinterpret_cast<const float*>(rb + 260);
    const float s_a = va ? ya * ksa : -INFINITY, s_b = vb ? yb * ksb : -INFINITY;
    const float m_new = fmaxf(m_run, row_max(fmaxf(s_a, s_b)));
    const float pa = va ? exp2f(s_a - m_new) : 0.f, pb = vb ? exp2f(s_b - m_new) : 0.f;
    const float corr = m_run == -INFINITY ? 0.f : exp2f(m_run - m_new);
    l_run = l_run * corr + row_sum(pa + pb);
    m_run = m_new;
    // Invalid tokens contribute exactly 0 (their cache bytes may be stale).
    sm.p[wave][row][col] = va ? __float2half_rn(pa * vsa * PSCALE) : (__half)0.f;
    sm.p[wave][row][SUB + col] = vb ? __float2half_rn(pb * vsb * PSCALE) : (__half)0.f;
    mid(2);  // next tiles' loads, spread between compute
    float cr[4];
#pragma unroll
    for (int h = 0; h < 4; ++h) cr[h] = __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, corr), 16 * h));
    __builtin_amdgcn_wave_barrier();
    // LDS ops of one wave complete in order; wait only for LDS (a fence would also wait
    // for the in-flight K/V prefetch).
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    u32x4 pq[4][2];
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      pq[h][0] = *reinterpret_cast<const u32x4*>(&sm.p[wave][h][8 * half]);
      pq[h][1] = *reinterpret_cast<const u32x4*>(&sm.p[wave][h][SUB + 8 * half]);
    }
    uint32_t vw[2][8];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      vw[0][i] = *reinterpret_cast<const uint32_t*>(ta + ((8 * half + i) * KVH + wave) * REC + 132 + 4 * j4);
      vw[1][i] = *reinterpret_cast<const uint32_t*>(tb + ((8 * half + i) * KVH + wave) * REC + 132 + 4 * j4);
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int h = 0; h < 4; ++h)
#pragma unroll
      for (int d = 0; d < 4; ++d) acc[h][d] *= cr[h];
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      if (g == 1) mid(3);
#pragma unroll
      for (int i = 0; i < 4; ++i) {
        const uint32_t a = vw[g][2 * i] ^ 0x80808080u, b = vw[g][2 * i + 1] ^ 0x80808080u;
        const uint32_t i0 = __builtin_amdgcn_perm(b, a, 0x05010400u), i1 = __builtin_amdgcn_perm(b, a, 0x07030602u);
        // (token 2i, token 2i + 1) pairs of dims 0..3: exact fp16 integers via the 0x64 bias.
        const half2_t vd[4] = {as_h2(__builtin_amdgcn_perm(c64, i0, 0x04010400u)) - bias,
                               as_h2(__builtin_amdgcn_perm(c64, i0, 0x04030402u)) - bias,
                               as_h2(__builtin_amdgcn_perm(c64, i1, 0x04010400u)) - bias,
                               as_h2(__builtin_amdgcn_perm(c64, i1, 0x04030402u)) - bias};
#pragma unroll
        for (int h = 0; h < 4; ++h) {
          const uint32_t pw[4] = {pq[h][g].x, pq[h][g].y, pq[h][g].z, pq[h][g].w};
          const half2_t pp = as_h2(pw[i]);
#pragma unroll
          for (int d = 0; d < 4; ++d) acc[h][d] = __builtin_amdgcn_fdot2(vd[d], pp, acc[h][d], false);
        }
      }
    }
    __builtin_amdgcn_wave_barrier();
  };

  // Two 16-token groups per block barrier; the next pair is loaded while this one computes.
  for (int g = g0; g < g1; g += 2) {
    const bool two = g + 1 < g1;
    lds_barrier();
    tile_store(nxt, sm.tile[0]);
    // Unconditional (tile 1 is ignored when !two) so no path leaves nxt2 loads pending;
    // a pending path makes the compiler drain all loads before nxt2 is reused.
    tile_store(nxt2, sm.tile[1]);

    for (int k = 0; k < (two ? 2 : 1); ++k) {
      const int t0 = (g + k) * SUB;
      if (pos >= t0 && pos < t0 + SUB)
        for (int i = threadIdx.x; i < KVH * REC / 4; i += THREADS)
          reinterpret_cast<uint32_t*>(sm.tile[k] + (pos - t0) * KVH * REC)[i] =
              reinterpret_cast<const uint32_t*>(sm.fresh)[i];
    }
    lds_barrier();
    process2(sm.tile[0], sm.tile[1], g * SUB, two, [&](int point) {
      // Pieces of the next two tiles: 3 + 2 of tile a, then 3 + 2 of tile b.
      if (point == 0 && g + 2 < g1) tile_load_part(nxt, kvv, (g + 2) * SUB, sm.blocks, b0, 0, 3);
      if (point == 1 && g + 2 < g1) tile_load_part(nxt, kvv, (g + 2) * SUB, sm.blocks, b0, 3, TILE_LOADS);
      if (point == 2 && g + 3 < g1) tile_load_part(nxt2, kvv, (g + 3) * SUB, sm.blocks, b0, 0, 3);
      if (point == 3 && g + 3 < g1) tile_load_part(nxt2, kvv, (g + 3) * SUB, sm.blocks, b0, 3, TILE_LOADS);
    });
  }

  // Sum the two token halves; lanes 0..31 write dims 4 j4 .. +3 of every head, and lane
  // 16 h (row h) writes head h's softmax state.
#pragma unroll
  for (int h = 0; h < 4; ++h)
#pragma unroll
    for (int d = 0; d < 4; ++d) acc[h][d] = xor_add(acc[h][d], 32);
  if (half == 0) {
#pragma unroll
    for (int h = 0; h < 4; ++h) {
      float* dst = part + ((size_t)(wave * 4 + h) * SPLITS + split) * (D + 2);
#pragma unroll
      for (int d = 0; d < 4; ++d) dst[4 * j4 + d] = acc[h][d] * (1.f / PSCALE);  // rows are 130 floats
    }
  }
  if (col == 0) {
    float* dst = part + ((size_t)(wave * 4 + row) * SPLITS + split) * (D + 2);
    dst[D] = m_run;
    dst[D + 1] = l_run;
  }
}

__device__ __forceinline__ void combine_phase(const float* mix, const float* part, float* y, int splits,
                                              float* red) {
  if (blockIdx.x >= QH) return;
  const int qh = blockIdx.x;
  const float* base = part + (size_t)qh * SPLITS * (D + 2);
  __shared__ float fac[SPLITS], lsum[2], osum[D];
  const int t = threadIdx.x;
  float m = t < splits ? base[t * (D + 2) + D] : -INFINITY;
  float l = t < splits ? base[t * (D + 2) + D + 1] : 0.f;
  float M = wave_max(m);
  if ((t & 63) == 0) red[t >> 6] = M;
  __syncthreads();
  M = fmaxf(fmaxf(red[0], red[1]), fmaxf(red[2], red[3]));
  float f = (t < splits && m != -INFINITY) ? exp2f(m - M) : 0.f;
  if (t < SPLITS) fac[t] = f;
  float lt = wave_sum(f * l);
  __syncthreads();
  if ((t & 63) == 0) red[t >> 6] = lt;
  __syncthreads();
  const float L = red[0] + red[1] + red[2] + red[3];
  const int d = t & (D - 1), half = t >> 7;
  float o = 0.f;
  for (int s = half; s < splits; s += 2) o += fac[s] * base[s * (D + 2) + d];
  if (half == 1) osum[d] = o;
  __syncthreads();
  if (half == 0) {
    float gate = 1.f / (1.f + __expf(-mix[3072 + qh]));
    y[qh * D + d] = (L > 0 ? (o + osum[d]) / L : 0.f) * gate;
  }
}

// ---------------------------------------------------------------- phases
// Conv in_proj rows are [B_c, C_c, x_c] per channel c; each wave owns whole
// channels, so after its GEMV rows it runs the depthwise conv for them.
template <typename Pre>
__device__ __forceinline__ void conv_in_phase(const Layer& L, const __half* x, float* ws, Pre&& pre) {
  const int lane = threadIdx.x & 63;
  const int nw = gridDim.x * WAVES, w = blockIdx.x * WAVES + __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  float* bcx = ws + WS_BCX;
  // This wave's channels are w, w + nw, ...; one lane per channel. The conv taps and the
  // channel's state (this layer's, untouched since the previous token) load under the GEMV.
  const int c = w + lane * nw;
  const bool own = lane < 64 && c < H;
  const int cc = own ? c : 0;
  const int32_t slot = L.state_index[0];
  __half* st = L.conv_state + (int64_t)slot * L.cs_slot + (int64_t)cc * L.cs_dim;
  const __half* cw = L.conv_w + cc * 7;
  float s[6], tap[7];
#pragma unroll
  for (int k = 0; k < 6; ++k) s[k] = __half2float(st[k * L.cs_t]);
#pragma unroll
  for (int k = 0; k < 7; ++k) tap[k] = __half2float(cw[k]);
  gemv_store<2048, 3, 3>(L.w_mix, L.s_mix, H, x, bcx, ws + WS_TRASH, pre);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_wave_barrier();
  if (own) {
    const float B = bcx[3 * c], C = bcx[3 * c + 1], X = bcx[3 * c + 2];
    const __half bxh = __float2half_rn(B * X);
    float conv = tap[6] * __half2float(bxh);
#pragma unroll
    for (int k = 0; k < 6; ++k) conv = fmaf(tap[k], s[k], conv);
#pragma unroll
    for (int k = 0; k < 5; ++k) st[k * L.cs_t] = __float2half_rn(s[k + 1]);
    st[5 * L.cs_t] = bxh;
    ws[WS_Y + c] = C * conv;
  }
}

template <int K, int R, int NB, typename Pre>
__device__ __forceinline__ void plain_phase(const uint8_t* W, const __half* S, int rows,
                                            const __half* x, float* out, float* trash, Pre&& pre) {
  gemv_store<K, R, NB>(W, S, rows / R, x, out, trash, pre);
}

// Gate/up rows are interleaved (gate_i, up_i); each wave owns whole pairs.
template <typename Pre>
__device__ __forceinline__ void w13_phase(const Layer& L, const __half* x, float* ws, Pre&& pre) {
  const int lane = threadIdx.x & 63;
  const int nw = gridDim.x * WAVES, w = blockIdx.x * WAVES + __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  float* g13 = ws + WS_G13;
  // Unit u = rows [4u, 4u + 4) = pairs 2u and 2u + 1.
  constexpr int units = I / 2;
  gemv_store<2048, 4, 2>(L.w13, L.s13, units, x, g13, ws + WS_TRASH, pre);
  __builtin_amdgcn_s_waitcnt(0);
  __builtin_amdgcn_wave_barrier();
  for (int j = lane;; j += 64) {
    const int u = w + (j >> 1) * nw;
    if (u >= units) break;
    const int p = 2 * u + (j & 1);
    const float gt = g13[2 * p], up = g13[2 * p + 1];
    ws[WS_A + p] = gt / (1.f + __expf(-gt)) * up;
  }
}

// ---------------------------------------------------------------- head + greedy sampling
__device__ __forceinline__ uint64_t argmax_key(float v, int idx) {
  // Larger key = larger logit, then smaller index (torch.argmax keeps the first maximum).
  const uint32_t u = __float_as_uint(v);
  const uint32_t ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((uint64_t)ord << 32) | (uint32_t)(0xffffffffu - (uint32_t)idx);
}

__device__ __forceinline__ uint64_t umax64(uint64_t a, uint64_t b) { return a > b ? a : b; }

// LM head GEMV (as plain_phase<2048, 4, 2>) that, when sampling, also tracks the argmax of
// the penalised logits of this wave's rows and folds it into g.best.
template <typename Pre>
__device__ __forceinline__ void head_phase(const Globals& g, const __half* x, float* trash, uint32_t* seen_lds,
                                           Pre&& pre) {
  constexpr int K = 2048, R = 4, NB = 2;
  const int units = (int)g.vocab / R;
  const int nw = gridDim.x * WAVES;
  const int w = blockIdx.x * WAVES + __builtin_amdgcn_readfirstlane(threadIdx.x >> 6);
  const int lane = threadIdx.x & 63;
  const bool active = w < units;
  const int rows = units * R, last = rows - 1;
  const bool sample = g.sample != 0;
  GemvBuf<K, R> buf[NB];
  if (active) {
#pragma unroll
    for (int s = 0; s < NB - 1; ++s) {
      gemv_load<K, R>(buf[s], g.w_head, g.s_head, (w + s * nw) * R, last);
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  if (sample) {
    const int words = ((int)g.vocab + 31) / 32;
    for (int i = threadIdx.x; i < words; i += THREADS) seen_lds[i] = gload(g.seen + i);
  }
  pre();  // ends with a block barrier (residual_norm), which also publishes seen_lds
  const float rep = sample ? *g.rep : 1.f;
  uint64_t best = 0;
  if (active) {
    GemvX<K> gx;
    gemv_xreg<K>(gx, x);
    for (int u = w; u < units; u += NB * nw) {
#pragma unroll
      for (int s = 0; s < NB; ++s) {
        gemv_load<K, R>(buf[(s + NB - 1) % NB], g.w_head, g.s_head, (u + (s + NB - 1) * nw) * R, last);
        // No fence here: the next unit's loads may interleave with this unit's math.
        const int base = (u + s * nw) * R;
        const float v = gemv_compute_reg_value<K, R>(buf[s], gx);
        const int row = base + lane;
        const bool mine = lane < R && row < rows && u + s * nw < units;
        *((mine ? g.logits + row : trash + lane)) = v;
        if (sample && mine) {
          const float vr = g.sample == 2 ? __half2float(__float2half_rn(v)) : v;
          const bool hit = (seen_lds[row >> 5] >> (row & 31)) & 1u;
          const float pv = hit ? (vr > 0.f ? vr / rep : vr * rep) : vr;
          best = umax64(best, argmax_key(pv, row));
        }
        __builtin_amdgcn_sched_barrier(0);
      }
    }
  }
  if (sample) {
#pragma unroll
    for (int o = 32; o > 0; o >>= 1) {
      const uint32_t lo = (uint32_t)best, hi = (uint32_t)(best >> 32);
      const uint64_t other = ((uint64_t)(uint32_t)__shfl_xor((int)hi, o) << 32) | (uint32_t)__shfl_xor((int)lo, o);
      best = umax64(best, other);
    }
    if (lane == 0 && best) __hip_atomic_fetch_max(g.best, best, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    __syncthreads();
    if (threadIdx.x == 0) {
      const uint32_t ticket = __hip_atomic_fetch_add(g.done, 1u, __ATOMIC_ACQ_REL, __HIP_MEMORY_SCOPE_AGENT);
      if (ticket == gridDim.x - 1) {
        const uint64_t key = __hip_atomic_load(g.best, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
        *g.token = (int32_t)(0xffffffffu - (uint32_t)key);
        __hip_atomic_store(g.done, 0u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
      }
    }
  }
}

// ---------------------------------------------------------------- kernel
// The layer loop is flattened into a phase list so each phase body is
// register-allocated on its own; only the phase cursor lives across phases.
__global__ __launch_bounds__(THREADS, 1) void decode_kernel(Globals g) {
  __shared__ Smem sm;
  uint32_t gen = __hip_atomic_load(g.barrier + 1, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT);
  float* ws = g.ws;
  __half* x = sm.u.g.x;
  const int attn_splits = min(SPLITS, (int)gridDim.x);
  const int nl = (int)g.num_layers;
  int tn = 0;
  trace_mark(g.trace, tn);
  if (g.sample && blockIdx.x == 0 && threadIdx.x == 0) {
    // Published to the head phase by the grid barriers in between.
    *g.best = 0;
    const int t = g.input_ids[0];
    __hip_atomic_fetch_or(g.seen + (t >> 5), 1u << (t & 31), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  // cur: which residual buffer holds h; flips at each residual_norm.
  int cur = 0;
  // Every phase ends with grid_arrive; the next phase issues its weight (or KV) loads and
  // only then waits, so their latency overlaps the barrier.
  bool pending = false;
  auto wait_pending = [&]() {
    if (pending) grid_wait(g.barrier, gen);
    pending = false;
  };
  auto arrive = [&]() {
    grid_arrive(g.barrier, gen);
    pending = true;
  };
  for (int step = -1; step <= 6 * nl; ++step) {
    int li = step < 0 ? -1 : step / 6;
    int stage = step < 0 ? -1 : step % 6;
    if (li == nl) {
      if (stage != 0) break;
      auto pre = [&]() {
        NormIn nin;
        norm_prefetch(nin, ws + WS_H0 + cur * H, g.layers[nl - 1].post_ffn_norm, g.final_norm);
        wait_pending();
        residual_norm_fin(nin, ws + WS_D, ws + WS_H0 + (cur ^ 1) * H, x, sm.red);
        if (blockIdx.x == 0 && g.hidden_out)
          for (int i = threadIdx.x; i < H; i += THREADS) g.hidden_out[i] = x[i];
        trace_mark(g.trace, tn);
      };
      if (g.logits) head_phase(g, x, ws + WS_TRASH, sm.u.head.seen, pre);
      else pre();
      __syncthreads();
      trace_mark(g.trace, tn);
      break;
    }
    if (li < 0) {
      if (blockIdx.x == 0) {
        const __half* e = g.embed + (int64_t)g.input_ids[0] * H;
        for (int i = threadIdx.x; i < H; i += THREADS) ws[WS_H0 + i] = __half2float(e[i]);
      }
      arrive();
      trace_mark(g.trace, tn);
      continue;
    }
    const Layer* L = g.layers + li;
    const int kind = (int)L->kind;
    switch (stage) {
      case 0: {  // residual + operator norm + mixer input projection
        auto pre = [&]() {
          NormIn nin;
          // Layer 0 reads h0, which block 0 wrote in the phase just finished, so unlike
          // every later residual it may only be read after the barrier.
          if (li == 0) wait_pending();
          norm_prefetch(nin, ws + WS_H0 + cur * H, li == 0 ? nullptr : g.layers[li - 1].post_ffn_norm, L->op_norm);
          wait_pending();
          residual_norm_fin(nin, li == 0 ? nullptr : ws + WS_D, ws + WS_H0 + (cur ^ 1) * H, x, sm.red);
          cur ^= 1;
          trace_mark(g.trace, tn);
        };
        if (kind == 0) conv_in_phase(*L, x, ws, pre);
        else plain_phase<2048, 2, 4>(L->w_mix, L->s_mix, QKVG, x, ws + WS_MIX, ws + WS_TRASH, pre);
        arrive();
        trace_mark(g.trace, tn);
        break;
      }
      case 1: {  // attention partials (or nothing for conv layers)
        if (kind == 0) break;
        attention_phase(*L, ws + WS_MIX, ws + WS_PART, sm.u.attn, attn_splits, wait_pending);
        arrive();
        trace_mark(g.trace, tn);
        break;
      }
      case 2: {
        if (kind == 0) break;
        wait_pending();
        combine_phase(ws + WS_MIX, ws + WS_PART, ws + WS_Y, attn_splits, sm.red);
        arrive();
        trace_mark(g.trace, tn);
        break;
      }
      case 3: {
        auto pre = [&]() {
          wait_pending();
          load_x<H>(ws + WS_Y, x);
          trace_mark(g.trace, tn);
        };
        plain_phase<2048, 2, 3>(L->w_out, L->s_out, H, x, ws + WS_O, ws + WS_TRASH, pre);
        arrive();
        trace_mark(g.trace, tn);
        break;
      }
      case 4: {
        auto pre = [&]() {
          NormIn nin;
          norm_prefetch(nin, ws + WS_H0 + cur * H, L->post_op_norm, L->ffn_norm);
          wait_pending();
          residual_norm_fin(nin, ws + WS_O, ws + WS_H0 + (cur ^ 1) * H, x, sm.red);
          cur ^= 1;
          trace_mark(g.trace, tn);
        };
        w13_phase(*L, x, ws, pre);
        arrive();
        trace_mark(g.trace, tn);
        break;
      }
      case 5: {
        auto pre = [&]() {
          wait_pending();
          load_x<I>(ws + WS_A, x);
          trace_mark(g.trace, tn);
        };
        plain_phase<7680, 1, 2>(L->w2, L->s2, H, x, ws + WS_D, ws + WS_TRASH, pre);
        arrive();
        trace_mark(g.trace, tn);
        break;
      }
    }
  }
}

}  // namespace lfm3v2
// Launched by the Go driver (internal/model): 60 blocks x 256 threads, one block per CU.
