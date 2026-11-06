// LFM3 prefill attention for gfx906: int8 QK and int8 PV.
//
// One block = 32 query tokens x one kv head = 128 query rows (row 4 t + j is token t, query head
// 4 * kv_head + j) against key tiles of 64 tokens; 512 threads, two waves per SIMD. Thread (tr, tc),
// tr = 4 * wave + lane / 16, tc = lane % 16:
//   QK      rows 4 tr .. +3 x keys 4 tc .. +3: int8 Q (per 32-dim group scales) x int8 K with v_dot4;
//           each group's int32 sum starts from a magic bit pattern and reads back as a float;
//   softmax fp32, the running max only moves when a tile beats it by more than 2^8;
//   P       P' = p x V scale, quantized to u8 per row and 64-key tile against the tile's largest P'
//           (round to nearest); --exact-attention selects prefill_attn_exact.hip, which keeps fp16 P;
//   PV      rows 4 tr .. +3 x dims 4 tc .. +3 and 64 + 4 tc .. +3 in two passes: u8 P' x (V + 128)
//           with v_dot4 into u32, then acc += scale x sum, and a per-row bias for the + 128.
// Waves 0-3 (tokens 0-15) run QK + softmax of tile s and then its PV; waves 4-7 (tokens 16-31) run
// PV of tile s - 1 and then QK + softmax of tile s. Each SIMD holds one wave of each group, and QK
// runs at a higher issue priority than PV (PV above the softmax), so the LDS-heavy phases of the two
// waves interleave. Two barriers per tile; K, V and the scales are double-buffered.
// A chunk can be split along the keys (grid z, attnSplits in internal/model/prefill.go): each split
// writes an unnormalized partial and the combine kernel merges them.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace lfm3pa8 {

constexpr int QH = 16, KVH = 4, D = 128, REC = 264;
constexpr int QT = 32;              // query tokens per block
constexpr int ROWS = QT * 4;        // query rows per block
constexpr int KT = 64;              // keys per tile
constexpr int THREADS = 512;
constexpr int G = 4;                // 32-dim groups
constexpr float LOG2E = 1.4426950408889634f;
constexpr int QROW = D / 4 + G;     // quantized query row: 32 dwords + 4 scales
constexpr float LAZY = 8.f;
// The KV cache layout is fixed by the engine: 32-token blocks of [slot][kv head][264 bytes].
// Compile-time strides turn the per-tile address math into shifts and 24-bit multiplies.
constexpr int BS = 32, KV_SLOT = KVH * REC, KV_BLOCK = BS * KV_SLOT;         // rescale the accumulators only when the max grows by more
// QK group sums start from the bit pattern of 1.5 x 2^23: a sum within +-2^22 then reads back as the
// float 12582912 + sum, so one fma per group replaces a conversion and an fma.
constexpr int MAGIC_I = 0x4B400000;
constexpr float MAGIC_F = 12582912.f;

typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

template <int CTRL>
__device__ __forceinline__ float dpp(float v) {
  return __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), CTRL, 0xf, 0xf, false));
}
template <int CTRL>
__device__ __forceinline__ uint32_t dppu(uint32_t v) {
  return (uint32_t)__builtin_amdgcn_mov_dpp((int)v, CTRL, 0xf, 0xf, false);
}

// reductions over the 16 lanes of a DPP row
__device__ __forceinline__ float max16(float v) {
  v = fmaxf(v, dpp<0xb1>(v));
  v = fmaxf(v, dpp<0x4e>(v));
  v = fmaxf(v, dpp<0x141>(v));
  return fmaxf(v, dpp<0x140>(v));
}
__device__ __forceinline__ float sum16(float v) {
  v += dpp<0xb1>(v);
  v += dpp<0x4e>(v);
  v += dpp<0x141>(v);
  return v + dpp<0x140>(v);
}
__device__ __forceinline__ float max4(float a, float b, float c, float d) {
  float r;
  asm("v_max3_f32 %0, %1, %2, %3\n\tv_max_f32 %0, %0, %4" : "=&v"(r) : "v"(a), "v"(b), "v"(c), "v"(d));
  return r;
}

__device__ __forceinline__ void max16x4(float& a, float& b, float& c, float& d) {
  asm volatile(
      "s_nop 1\n\t"
      "v_max_f32_dpp %0, %0, %0 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %1, %1, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %2, %2, %2 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %3, %3, %3 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf\n\t"
      "s_nop 0\n\t"
      "v_max_f32_dpp %0, %0, %0 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %1, %1, %1 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %2, %2, %2 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %3, %3, %3 quad_perm:[2,3,0,1] row_mask:0xf bank_mask:0xf\n\t"
      "s_nop 0\n\t"
      "v_max_f32_dpp %0, %0, %0 row_half_mirror row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %1, %1, %1 row_half_mirror row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %2, %2, %2 row_half_mirror row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %3, %3, %3 row_half_mirror row_mask:0xf bank_mask:0xf\n\t"
      "s_nop 0\n\t"
      "v_max_f32_dpp %0, %0, %0 row_mirror row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %1, %1, %1 row_mirror row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %2, %2, %2 row_mirror row_mask:0xf bank_mask:0xf\n\t"
      "v_max_f32_dpp %3, %3, %3 row_mirror row_mask:0xf bank_mask:0xf\n\t"
      "s_nop 1"
      : "+v"(a), "+v"(b), "+v"(c), "+v"(d));
}

struct Params {
  const __half* q;        // [T, 16, 128] fp16
  __half* out;            // [T, 16, 128]
  const int8_t* kv;
  int64_t kv_block, kv_head, kv_slot;
  const int32_t* block_table;
  int32_t block_size;
  int32_t tokens;
  int32_t context;
  int32_t window;
  int64_t q_stride;
  int64_t out_stride;
  uint32_t* qbuf;         // [T * 16][QROW]
  int32_t splits;         // > 1: block z handles part of the keys and writes a partial result
  int32_t pad_;
  float* ws;              // partials: [token * 16 + head][splits][PART]
};

constexpr int PART = D + 4;   // acc[128], max, sum, bias, unused

// Per (token, head): int8 per 32-dim group (round to nearest) + fp32 scales.
__global__ __launch_bounds__(64) void quantize_q(Params p) {
  const int row = blockIdx.x;
  const int t = row / QH, h = row % QH, lane = threadIdx.x;
  const __half* src = p.q + (int64_t)t * p.q_stride + h * D;
  const float a = __half2float(src[2 * lane]), b = __half2float(src[2 * lane + 1]);
  float m = fmaxf(fabsf(a), fabsf(b));
#pragma unroll
  for (int o = 1; o < 16; o <<= 1) m = fmaxf(m, __shfl_xor(m, o));
  const float scale = fmaxf(m / 127.f, 1e-20f), inv = 1.f / scale;
  const int qa = (int)rintf(a * inv), qb = (int)rintf(b * inv);
  const uint32_t pair = (uint32_t)(qa & 0xff) | ((uint32_t)(qb & 0xff) << 8);
  const uint32_t other = __shfl_xor(pair, 1);
  uint32_t* dst = p.qbuf + (int64_t)row * QROW;
  if ((lane & 1) == 0) dst[lane / 2] = pair | (other << 16);
  if ((lane & 15) == 0) reinterpret_cast<float*>(dst)[D / 4 + lane / 16] = scale;
}

struct Shared {
  alignas(16) uint32_t qt[D / 4][ROWS];       // Q transposed (int8 dwords)
  alignas(16) uint32_t kt[2][D / 4][KT];      // K transposed: dword c of each key
  alignas(16) uint32_t v4[2][KT / 4][D];      // V + 128 as u8, the 4 keys of a quad in one dword
  alignas(16) uint32_t pt[KT / 4][ROWS + 4];  // u8 P' of a key quad, per row
  alignas(16) float ks[2][KT], vs[2][KT];
  alignas(16) float qs[G][ROWS];
  alignas(16) float qc[ROWS];                 // -MAGIC_F x sum of the row's group scales
};

__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__device__ __forceinline__ uint32_t rec_off(const Params& p, int head, int key) {
  const uint32_t blk = (uint32_t)p.block_table[key / BS];
  return __umul24(blk, KV_BLOCK) + head * REC + __umul24(key % BS, KV_SLOT);   // < 4 GB per layer
}

// K and the scales of a tile, by group B (gt = thread index in the group): key gt / 4,
// K bytes 32 (gt % 4) .. +31; gt % 4 == 0 also the K scale, == 1 the V scale.
struct KLoad {
  u32x2 r[4];
  uint32_t scale;   // one dword in one variable (a pair's halves under two conditions miscompiled)
};
__device__ __forceinline__ void k_load(KLoad& l, const Params& p, int head, int k0, int kend, int gt) {
  const int key = min(k0 + (gt >> 2), kend - 1), sub = gt & 3;
  const int8_t* src = p.kv + rec_off(p, head, key);
#pragma unroll
  for (int j = 0; j < 4; ++j) l.r[j] = *reinterpret_cast<const u32x2*>(src + 32 * sub + 8 * j);
  if (sub < 2) l.scale = *reinterpret_cast<const uint32_t*>(src + (sub == 0 ? 128 : 260));
}
__device__ __forceinline__ void k_store(const KLoad& l, Shared& sm, int buf, int gt, float kscale) {
  const int key = gt >> 2, sub = gt & 3;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    sm.kt[buf][8 * sub + 2 * j][key] = l.r[j].x;
    sm.kt[buf][8 * sub + 2 * j + 1][key] = l.r[j].y;
  }
  if (sub == 0) sm.ks[buf][key] = __builtin_bit_cast(float, l.scale) * kscale;
  if (sub == 1) sm.vs[buf][key] = __builtin_bit_cast(float, l.scale);
}

// V of a tile, by group A: key quads gt / 32 and 8 + gt / 32 at dims 4 (gt % 32) .. +3, transposed
// in registers into the v4 layout.
struct VLoad {
  uint32_t r[2][4];
};
__device__ __forceinline__ void v_load(VLoad& l, const Params& p, int head, int k0, int kend, int gt) {
  const int dc = gt & 31;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    // the 4 keys of a quad share a 32-token block (keys past the end read slots of a block that
    // exists, and are masked)
    const int key0 = min(k0 + 4 * (8 * j + (gt >> 5)), (kend - 1) & ~3);
    const int8_t* src = p.kv + rec_off(p, head, key0) + 132 + 4 * dc;
#pragma unroll
    for (int i = 0; i < 4; ++i) l.r[j][i] = *reinterpret_cast<const uint32_t*>(src + i * KV_SLOT);
  }
}
__device__ __forceinline__ void v_store(const VLoad& l, Shared& sm, int buf, int gt) {
  const int dc = gt & 31;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
    const int kq = 8 * j + (gt >> 5);
    const uint32_t x0 = l.r[j][0] ^ 0x80808080u, x1 = l.r[j][1] ^ 0x80808080u;
    const uint32_t x2 = l.r[j][2] ^ 0x80808080u, x3 = l.r[j][3] ^ 0x80808080u;
    const uint32_t a0 = __builtin_amdgcn_perm(x1, x0, 0x05010400u), a1 = __builtin_amdgcn_perm(x1, x0, 0x07030602u);
    const uint32_t b0 = __builtin_amdgcn_perm(x3, x2, 0x05010400u), b1 = __builtin_amdgcn_perm(x3, x2, 0x07030602u);
    *reinterpret_cast<u32x4*>(&sm.v4[buf][kq][4 * dc]) =
        u32x4{__builtin_amdgcn_perm(b0, a0, 0x05040100u), __builtin_amdgcn_perm(b0, a0, 0x07060302u),
              __builtin_amdgcn_perm(b1, a1, 0x05040100u), __builtin_amdgcn_perm(b1, a1, 0x07060302u)};
  }
}

__device__ __forceinline__ void qk_softmax(Shared& sm, int buf, int tr, int tc, int k0, int qpos, int kbegin, int window, bool row_ok, bool masked,
                                           bool first, float (&m)[4], float (&lp)[4], float (&cb)[4], float (&acc)[4][8], float (&sc)[4]) {
  // QK: 4 rows (token tr) x 4 keys (4 tc .. 4 tc + 3), fully unrolled.
  float s[4][4];
  {
    const float4 c4 = *reinterpret_cast<const float4*>(&sm.qc[4 * tr]);
    const float c[4] = {c4.x, c4.y, c4.z, c4.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) s[i][k] = c[i];
  }
  __builtin_amdgcn_s_setprio(2);                      // QK's LDS reads ahead of the other group's PV
  u32x4 qv = *reinterpret_cast<const u32x4*>(&sm.qt[0][4 * tr]);
  u32x4 kv = *reinterpret_cast<const u32x4*>(&sm.kt[buf][0][4 * tc]);
  u32x4 kv1 = *reinterpret_cast<const u32x4*>(&sm.kt[buf][1][4 * tc]);
#pragma unroll
  for (int g = 0; g < G; ++g) {
    int a[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) a[i][k] = MAGIC_I;
#pragma unroll
    for (int cc = 0; cc < 8; ++cc) {
      const int c1 = min(8 * g + cc + 1, D / 4 - 1), c2 = min(8 * g + cc + 2, D / 4 - 1);
      const u32x4 qn = *reinterpret_cast<const u32x4*>(&sm.qt[c1][4 * tr]);   // Q one column ahead
      const u32x4 kn = *reinterpret_cast<const u32x4*>(&sm.kt[buf][c2][4 * tc]);   // K two columns ahead
      const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w}, kw[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) a[i][k] = __builtin_amdgcn_sdot4(qw[i], kw[k], a[i][k], false);
      __builtin_amdgcn_sched_barrier(0);
      qv = qn;
      kv = kv1;
      kv1 = kn;
    }
    const float4 qg = *reinterpret_cast<const float4*>(&sm.qs[g][4 * tr]);
    const float qsc[4] = {qg.x, qg.y, qg.z, qg.w};
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) s[i][k] = fmaf(__builtin_bit_cast(float, a[i][k]), qsc[i], s[i][k]);
  }
  __builtin_amdgcn_s_setprio(0);
  const float4 ks4 = *reinterpret_cast<const float4*>(&sm.ks[buf][4 * tc]);
  const float4 vs4 = *reinterpret_cast<const float4*>(&sm.vs[buf][4 * tc]);
  const float kss[4] = {ks4.x, ks4.y, ks4.z, ks4.w}, vss[4] = {vs4.x, vs4.y, vs4.z, vs4.w};
  // u = log2 p relative to the running max m (0 until the first tile sets it); m only moves when
  // a tile beats it by LAZY, so p stays below 2^LAZY.
  float u[4][4];
#pragma unroll
  for (int i = 0; i < 4; ++i)
#pragma unroll
    for (int k = 0; k < 4; ++k) u[i][k] = fmaf(s[i][k], kss[k], -m[i]);
  if (masked) {                                       // block-uniform: the edge tiles only
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int kpos = k0 + 4 * tc + k;
      if (!(kpos <= qpos && kpos >= kbegin && (window == 0 || kpos > qpos - window) && row_ok)) {
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i][k] = -INFINITY;
      }
    }
  }
  bool grow = false;
#pragma unroll
  for (int i = 0; i < 4; ++i)
    grow |= max4(u[i][0], u[i][1], u[i][2], u[i][3]) > LAZY;
  if (__builtin_amdgcn_ballot_w64(grow || first) != 0) {   // the first tile, then rarely
    float tm[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      tm[i] = max4(u[i][0], u[i][1], u[i][2], u[i][3]) + m[i];                                  // u + m is the logit (m is finite)
    }
    max16x4(tm[0], tm[1], tm[2], tm[3]);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      // uniform across the 16 lanes of the row; a row without valid keys yet keeps m. Rows that
      // keep their max scale by 1: a select over the accumulators would double their registers.
      const bool up = tm[i] > -INFINITY && (first || tm[i] > m[i] + LAZY);
      const float d = up ? m[i] - tm[i] : 0.f;
      const float corr = __builtin_amdgcn_exp2f(d);
      m[i] = up ? tm[i] : m[i];
      lp[i] *= corr;
      cb[i] *= corr;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] *= corr;
#pragma unroll
      for (int k = 0; k < 4; ++k) u[i][k] += d;
    }
  }
  // p = 2^u, P' = p x V scale, u8 against the row's largest P' in the tile.
  float pr[4][4], pm[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    float e[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      e[k] = __builtin_amdgcn_exp2f(u[i][k]);
      pr[i][k] = e[k] * vss[k];
    }
    lp[i] += (e[0] + e[1]) + (e[2] + e[3]);
    pm[i] = max4(pr[i][0], pr[i][1], pr[i][2], pr[i][3]);
  }
  max16x4(pm[0], pm[1], pm[2], pm[3]);
  uint32_t pw[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float inv = 255.f * __builtin_amdgcn_rcpf(fmaxf(pm[i], 1e-30f));   // a row without valid keys: q = 0
    sc[i] = pm[i] * (1.f / 255.f);
    uint32_t q = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) q = __builtin_amdgcn_cvt_pk_u8_f32(pr[i][k] * inv, k, q);  // nearest even, saturates
    pw[i] = q;
    // this lane's share of 128 x sum(q) x scale; the 16 lanes of the row are summed at the end
    cb[i] = fmaf((float)__builtin_amdgcn_udot4(q, 0x01010101u, 0u, false), 128.f * sc[i], cb[i]);
  }
  *reinterpret_cast<u32x4*>(&sm.pt[tc][4 * tr]) = u32x4{pw[0], pw[1], pw[2], pw[3]};
}

// PV of one tile for rows 4 tr .. +3: two passes over the dims (16 accumulators each).
__device__ __forceinline__ void pv_tile(Shared& sm, int buf, int tr, int tc, const float (&sc)[4], float (&acc)[4][8]) {
  __builtin_amdgcn_s_setprio(1);
#pragma unroll
  for (int h = 0; h < 2; ++h) {
    uint32_t au[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) au[i][e] = 0;
    u32x4 pv = *reinterpret_cast<const u32x4*>(&sm.pt[0][4 * tr]);
    u32x4 va = *reinterpret_cast<const u32x4*>(&sm.v4[buf][0][64 * h + 4 * tc]);
#pragma unroll
    for (int kq = 0; kq < KT / 4; ++kq) {
      const int kn = min(kq + 1, KT / 4 - 1);         // one key quad ahead
      const u32x4 pn = *reinterpret_cast<const u32x4*>(&sm.pt[kn][4 * tr]);
      const u32x4 van = *reinterpret_cast<const u32x4*>(&sm.v4[buf][kn][64 * h + 4 * tc]);
      const uint32_t pw[4] = {pv.x, pv.y, pv.z, pv.w};
      const uint32_t vw[4] = {va.x, va.y, va.z, va.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 4; ++e) au[i][e] = __builtin_amdgcn_udot4(pw[i], vw[e], au[i][e], false);
      __builtin_amdgcn_sched_barrier(0);
      pv = pn;
      va = van;
    }
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int e = 0; e < 4; ++e) acc[i][4 * h + e] = fmaf((float)au[i][e], sc[i], acc[i][4 * h + e]);
  }
  __builtin_amdgcn_s_setprio(0);
}

// Distinct asm per group: identical barriers in both branches let the compiler merge them.
__device__ __forceinline__ void abarrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier ; A" ::: "memory"); }
__device__ __forceinline__ void bbarrier() { asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier ; B" ::: "memory"); }

__global__ __launch_bounds__(THREADS, 1) void prefill_attention(Params p) {
  __shared__ Shared sm;
  const int head = blockIdx.y;
  const int t0 = (gridDim.x - 1 - blockIdx.x) * QT;   // the longest rows first
  const int ntok = min(QT, p.tokens - t0);
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), lane = threadIdx.x & 63;
  const bool group_a = wave < 4;
  const int gt = threadIdx.x & 255;                   // thread index within the group
  const int tr = wave * 4 + (lane >> 4), tc = lane & 15;   // A: tokens 0-15, B: 16-31
  const int qpos = p.context + t0 + tr;
  const int qfirst = p.context + t0, qlast = p.context + t0 + ntok - 1;
  const int kend = qlast + 1;
  const int kbegin = p.window > 0 ? max(0, qfirst - p.window + 1) : 0;
  const int kt_begin = kbegin / KT * KT;
  const int all_tiles = (kend - kt_begin + KT - 1) / KT;
  const int z = blockIdx.z, nz = p.splits;
  const int tile_lo = all_tiles * z / nz;
  const int ntiles = all_tiles * (z + 1) / nz - tile_lo;
  const int kt_first = kt_begin + tile_lo * KT;           // this split's first tile
  const float kscale = 0.08838834764831845f * LOG2E;

  for (int i = threadIdx.x; i < ROWS * (D / 4); i += THREADS) {
    const int r = i % ROWS, c = i / ROWS;
    const int tok = min(r >> 2, ntok - 1);
    sm.qt[c][r] = p.qbuf[(int64_t)((t0 + tok) * QH + head * 4 + (r & 3)) * QROW + c];
  }
  if (threadIdx.x < ROWS) {
    const int r = threadIdx.x, tok = min(r >> 2, ntok - 1);
    const float* s = reinterpret_cast<const float*>(p.qbuf + (int64_t)((t0 + tok) * QH + head * 4 + (r & 3)) * QROW + D / 4);
#pragma unroll
    for (int g = 0; g < G; ++g) sm.qs[g][r] = s[g];
    sm.qc[r] = -MAGIC_F * (s[0] + s[1] + s[2] + s[3]);
  }

  float m[4], lp[4], cb[4], acc[4][8], sc[4];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    m[i] = 0.f;
    lp[i] = 0.f;
    cb[i] = 0.f;
    sc[i] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  }

  if (group_a) {
    VLoad vl;
    if (ntiles > 0) v_load(vl, p, head, kt_first, kend, gt);   // V of the first tile
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (ntiles > 0) v_store(vl, sm, 0, gt);
    abarrier();
    for (int s = 0; s <= ntiles; ++s) {
      const int buf = s & 1, k0 = kt_first + s * KT;
      if (s < ntiles) {
        if (s > 0) {
          asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
          v_store(vl, sm, buf, gt);                    // V(s), loaded during the last PV
        }
        const bool interior = k0 + KT - 1 <= qfirst && k0 >= kbegin && (p.window == 0 || k0 > qlast - p.window);
        qk_softmax(sm, buf, tr, tc, k0, qpos, kbegin, p.window, tr < ntok, !interior, s == 0, m, lp, cb, acc, sc);
      }
      abarrier();                                      // X
      if (s < ntiles) {
        if (s + 1 < ntiles) v_load(vl, p, head, k0 + KT, kend, gt);
        pv_tile(sm, buf, tr, tc, sc, acc);
      }
      abarrier();                                      // Y
    }
  } else {
    KLoad kl;
    if (ntiles > 0) k_load(kl, p, head, kt_first, kend, gt);   // K of the first tile
    asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
    if (ntiles > 0) k_store(kl, sm, 0, gt, kscale);
    bbarrier();
    for (int s = 0; s <= ntiles; ++s) {
      const int buf = s & 1, k0 = kt_first + s * KT;
      if (s + 1 < ntiles) k_load(kl, p, head, k0 + KT, kend, gt);
      if (s >= 1) pv_tile(sm, buf ^ 1, tr, tc, sc, acc);   // PV(s - 1)
      bbarrier();                                      // X
      if (s + 1 < ntiles) {
        asm volatile("s_waitcnt vmcnt(0)" ::: "memory");
        k_store(kl, sm, buf ^ 1, gt, kscale);          // K(s + 1): nobody reads that buffer this step
      }
      if (s < ntiles) {
        const bool interior = k0 + KT - 1 <= qfirst && k0 >= kbegin && (p.window == 0 || k0 > qlast - p.window);
        qk_softmax(sm, buf, tr, tc, k0, qpos, kbegin, p.window, tr < ntok, !interior, s == 0, m, lp, cb, acc, sc);
      }
      bbarrier();                                      // Y
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float l = sum16(lp[i]), bias = sum16(cb[i]);
    if (tr >= ntok) continue;
    if (nz > 1) {                                       // partial: unnormalized, combined later
      float* part = p.ws + ((int64_t)((t0 + tr) * QH + head * 4 + i) * nz + z) * PART;
      *reinterpret_cast<float4*>(part + 4 * tc) = float4{acc[i][0], acc[i][1], acc[i][2], acc[i][3]};
      *reinterpret_cast<float4*>(part + 64 + 4 * tc) = float4{acc[i][4], acc[i][5], acc[i][6], acc[i][7]};
      if (tc == 0) {
        part[D] = ntiles > 0 ? m[i] : -INFINITY;
        part[D + 1] = l;
        part[D + 2] = bias;
      }
      continue;
    }
    const float inv = l > 0.f ? 1.f / l : 0.f;
    __half h[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) h[e] = __float2half_rn((acc[i][e] - bias) * inv);
    __half* dst = p.out + (int64_t)(t0 + tr) * p.out_stride + (head * 4 + i) * D + 4 * tc;
    *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(h);
    *reinterpret_cast<uint2*>(dst + 64) = *reinterpret_cast<const uint2*>(h + 4);
  }
}

// Combine the splits of one (token, head) row: out = sum_z 2^(m_z - M) (acc_z - bias_z) / sum_z 2^(m_z - M) l_z.
__global__ __launch_bounds__(64) void combine(Params p) {
  const int row = blockIdx.x, t = row / QH, h = row % QH, lane = threadIdx.x;
  const float* part = p.ws + (int64_t)row * p.splits * PART;
  float mx = -INFINITY;
  for (int z = 0; z < p.splits; ++z) mx = fmaxf(mx, part[z * PART + D]);
  float l = 0.f, a0 = 0.f, a1 = 0.f;
  for (int z = 0; z < p.splits; ++z) {
    const float* pz = part + z * PART;
    if (pz[D] == -INFINITY) continue;
    const float w = __builtin_amdgcn_exp2f(pz[D] - mx);
    l = fmaf(w, pz[D + 1], l);
    a0 = fmaf(w, pz[2 * lane] - pz[D + 2], a0);
    a1 = fmaf(w, pz[2 * lane + 1] - pz[D + 2], a1);
  }
  const float inv = l > 0.f ? 1.f / l : 0.f;
  *reinterpret_cast<__half2*>(p.out + (int64_t)t * p.out_stride + h * D + 2 * lane) = __floats2half2_rn(a0 * inv, a1 * inv);
}

}  // namespace lfm3pa8
