// LFM3 prefill attention for gfx906: register-tiled (GEMM-style) flash attention.
//
// One block = 16 query tokens x one kv head = 64 query rows (row 4 t + j is
// token t, query head 4 * kv_head + j) against key tiles of 64 tokens.
// Thread (tr, tc), tr = 4 * wave + lane / 16, tc = lane % 16:
//   QK      rows 4 tr .. 4 tr + 3 (token tr, its 4 heads) x keys 4 tc .. 4 tc + 3,
//           int8 Q (per 32-dim group scales) x int8 K with v_dot4, fp32 rescale;
//   softmax fp32; the 16 lanes of a row share a DPP row;
//   PV      rows 4 tr .. +3 x dims 4 tc .. +3 and 64 + 4 tc .. +3, fp16 P' (P x V scale) x exact fp16 V
//           (key pairs) with v_dot2 and fp32 accumulation.
// LDS holds K transposed (kt[c][key]), raw V, V as fp16 key pairs, Q
// transposed and P' transposed (pt[key pair][row]).
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

namespace lfm3pa7 {

constexpr int QH = 16, KVH = 4, D = 128, REC = 264;
constexpr int QT = 16;              // query tokens per block
constexpr int ROWS = QT * 4;        // query rows per block
constexpr int KT = 64;              // keys per tile
constexpr int THREADS = 256;
constexpr int G = 4;                // 32-dim groups
constexpr float LOG2E = 1.4426950408889634f;
constexpr float PSCALE = 4096.f;
constexpr int QROW = D / 4 + G;     // quantized query row: 32 dwords + 4 scales

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ half2_t as_h2(uint32_t u) {
  half2_t h;
  __builtin_memcpy(&h, &u, 4);
  return h;
}

__device__ __forceinline__ uint32_t as_u(half2_t h) {
  uint32_t u;
  __builtin_memcpy(&u, &h, 4);
  return u;
}

__device__ __forceinline__ uint32_t i8pair(uint32_t u_biased, uint32_t sel) {
  const half2_t bias = {(_Float16)1152.f, (_Float16)1152.f};
  return as_u(as_h2(__builtin_amdgcn_perm(0x64646464u, u_biased, sel)) - bias);
}

__device__ __forceinline__ float dpp_max16(float v) {
  v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0xb1, 0xf, 0xf, false)));
  v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x4e, 0xf, 0xf, false)));
  v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x141, 0xf, 0xf, false)));
  v = fmaxf(v, __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x140, 0xf, 0xf, false)));
  return v;
}

__device__ __forceinline__ float dpp_sum16(float v) {
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0xb1, 0xf, 0xf, false));
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x4e, 0xf, 0xf, false));
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x141, 0xf, 0xf, false));
  v += __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), 0x140, 0xf, 0xf, false));
  return v;
}

// exp2 for x <= 0: v_ldexp + degree 4 polynomial; ~1e-4 relative accuracy is plenty for the fp16 P'.
__device__ __forceinline__ float fast_exp2(float x) {
  const float fl = floorf(x);
  const float f = x - fl;
  float p = fmaf(f, 0.0135557f, 0.0520323f);
  p = fmaf(f, p, 0.2413793f);
  p = fmaf(f, p, 0.6930886f);
  p = fmaf(f, p, 1.0000026f);
  return __builtin_ldexpf(p, (int)fmaxf(fl, -150.f));
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
};

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

constexpr int PIECES = REC / 8;   // 33 eight-byte pieces per record
constexpr int PER_THREAD = 9;     // thread t: key t / 4, pieces (t % 4) + 4 j

struct Shared {
  alignas(16) uint32_t kt[D / 4][KT + 4];   // K transposed: dword c (dims 4c..4c+3) of each key
  alignas(16) uint8_t vraw[KT][D];          // V int8
  alignas(16) uint32_t v2[KT / 2][D];       // V fp16 key pairs
  alignas(16) uint32_t qt[D / 4][ROWS];     // Q transposed (int8 dwords)
  alignas(16) uint32_t pt[KT / 2][ROWS + 4];  // P' fp16 key pairs per row
  float ks[KT], vs[KT];
  alignas(16) float qs[G][ROWS];
};

// Block barrier that only orders LDS: __syncthreads() also waits for every outstanding
// global load, which would drain the next tile's prefetch issued just before it.
__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

// Block-table entry of this thread's key in the tile at k0 (fetched one tile ahead so the
// dependent K/V loads do not wait for it).
__device__ __forceinline__ int block_of(const Params& p, int k0) {
  const int key = k0 + (threadIdx.x >> 2);
  return p.block_table[key / p.block_size];
}

__device__ __forceinline__ void tile_load(u32x2 (&r)[PER_THREAD], const Params& p, int head, int k0, int blk32) {
  const int key = k0 + (threadIdx.x >> 2);
  const int64_t blk = blk32;
  const int8_t* src = p.kv + blk * p.kv_block + head * p.kv_head + (key % p.block_size) * p.kv_slot;
  const int sub = threadIdx.x & 3;
#pragma unroll
  for (int j = 0; j < PER_THREAD; ++j) {
    const int piece = min(sub + 4 * j, PIECES - 1);
    r[j] = *reinterpret_cast<const u32x2*>(src + piece * 8);
  }
}

// Scatter one tile's records: K dwords -> kt, V bytes -> vraw, scales -> ks/vs.
__device__ __forceinline__ void tile_store(const u32x2 (&r)[PER_THREAD], Shared& sm, float kscale) {
  const int key = threadIdx.x >> 2, sub = threadIdx.x & 3;
#pragma unroll
  for (int j = 0; j < PER_THREAD; ++j) {
    const int piece = sub + 4 * j;
    const uint32_t lo = r[j].x, hi = r[j].y;
    if (piece < 16) {                      // K bytes 8 piece .. 8 piece + 7
      sm.kt[2 * piece][key] = lo;
      sm.kt[2 * piece + 1][key] = hi;
    } else if (piece == 16) {              // K scale | V dims 0..3
      sm.ks[key] = __builtin_bit_cast(float, lo) * kscale;
      *reinterpret_cast<uint32_t*>(&sm.vraw[key][0]) = hi;
    } else if (piece < 32) {               // V dims 8 (piece - 17) + 4 .. + 11
      *reinterpret_cast<uint32_t*>(&sm.vraw[key][8 * (piece - 17) + 4]) = lo;
      *reinterpret_cast<uint32_t*>(&sm.vraw[key][8 * (piece - 17) + 8]) = hi;
    } else if (piece == 32) {              // V dims 124..127 | V scale
      *reinterpret_cast<uint32_t*>(&sm.vraw[key][124]) = lo;
      sm.vs[key] = __builtin_bit_cast(float, hi) * PSCALE;
    }
  }
}

__global__ __launch_bounds__(THREADS, 1) void prefill_attention(Params p) {
  __shared__ Shared sm;
  const int head = blockIdx.y;
  const int t0 = blockIdx.x * QT;
  const int ntok = min(QT, p.tokens - t0);
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 6), lane = threadIdx.x & 63;
  const int tr = wave * 4 + (lane >> 4), tc = lane & 15;   // token tr; key group / dim group tc
  const int qpos = p.context + t0 + tr;                      // position of this thread's token
  const int kend = p.context + t0 + ntok;
  const int kbegin = p.window > 0 ? max(0, p.context + t0 - p.window + 1) : 0;
  const int kt_begin = kbegin / KT * KT;
  const float kscale = 0.08838834764831845f * LOG2E;

  u32x2 nxt[PER_THREAD];
  tile_load(nxt, p, head, kt_begin, block_of(p, kt_begin));
  int blk_next = block_of(p, min(kt_begin + KT, kend - 1) / KT * KT);

  // Q tile: rows r = 4 token + j -> qt[c][r], qs[r][g].
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
  }
  __syncthreads();

  float m[4], lp[4], acc[4][8];
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    m[i] = -INFINITY;
    lp[i] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[i][e] = 0.f;
  }

  for (int k0 = kt_begin; k0 < kend; k0 += KT) {
    lds_barrier();                                    // previous PV / QK done with LDS
    tile_store(nxt, sm, kscale);
    if (k0 + KT < kend) {
      tile_load(nxt, p, head, k0 + KT, blk_next);
      blk_next = block_of(p, min(k0 + 2 * KT, kend - 1) / KT * KT);
    }
    lds_barrier();
    // V -> fp16 key pairs: item -> key pair item / 32, dims 4 (item % 32) .. +3.
#pragma unroll
    for (int j = 0; j < KT / 2 * (D / 4) / THREADS; ++j) {
      const int item = threadIdx.x + j * THREADS, kp = item >> 5, dc = item & 31;
      const uint32_t x = *reinterpret_cast<const uint32_t*>(&sm.vraw[2 * kp][4 * dc]) ^ 0x80808080u;
      const uint32_t y = *reinterpret_cast<const uint32_t*>(&sm.vraw[2 * kp + 1][4 * dc]) ^ 0x80808080u;
      const uint32_t i0 = __builtin_amdgcn_perm(y, x, 0x05010400u), i1 = __builtin_amdgcn_perm(y, x, 0x07030602u);
      u32x4 o;
      o.x = i8pair(i0, 0x04010400u);
      o.y = i8pair(i0, 0x04030402u);
      o.z = i8pair(i1, 0x04010400u);
      o.w = i8pair(i1, 0x04030402u);
      *reinterpret_cast<u32x4*>(&sm.v2[kp][4 * dc]) = o;
    }
    // QK: 4 rows (token tr) x 4 keys (4 tc .. 4 tc + 3).
    float s[4][4];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
      for (int k = 0; k < 4; ++k) s[i][k] = 0.f;
    u32x4 qv = *reinterpret_cast<const u32x4*>(&sm.qt[0][4 * tr]);
    u32x4 kv = *reinterpret_cast<const u32x4*>(&sm.kt[0][4 * tc]);
#pragma unroll 1
    for (int g = 0; g < G; ++g) {
      int a[4][4];
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) a[i][k] = 0;
#pragma unroll
      for (int cc = 0; cc < 8; ++cc) {
        const int cn = min(8 * g + cc + 1, D / 4 - 1);   // prefetch the next dword column
        const u32x4 qn = *reinterpret_cast<const u32x4*>(&sm.qt[cn][4 * tr]);
        const u32x4 kn = *reinterpret_cast<const u32x4*>(&sm.kt[cn][4 * tc]);
        const uint32_t qw[4] = {qv.x, qv.y, qv.z, qv.w}, kw[4] = {kv.x, kv.y, kv.z, kv.w};
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
          for (int k = 0; k < 4; ++k) a[i][k] = __builtin_amdgcn_sdot4(qw[i], kw[k], a[i][k], false);
        __builtin_amdgcn_sched_barrier(0);
        qv = qn;
        kv = kn;
      }
      const float4 qg = *reinterpret_cast<const float4*>(&sm.qs[g][4 * tr]);
      const float qsc[4] = {qg.x, qg.y, qg.z, qg.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int k = 0; k < 4; ++k) s[i][k] = fmaf((float)a[i][k], qsc[i], s[i][k]);
    }
    // Online softmax (fp32) per row; P' = p * V scale (x PSCALE) -> fp16 key pairs.
    float kss[4], vss[4];
    bool valid[4];
#pragma unroll
    for (int k = 0; k < 4; ++k) {
      const int kpos = k0 + 4 * tc + k;
      kss[k] = sm.ks[4 * tc + k];
      vss[k] = sm.vs[4 * tc + k];
      valid[k] = kpos <= qpos && kpos >= kbegin && (p.window == 0 || kpos > qpos - p.window) && tr < ntok;
    }
    uint32_t p01[4], p23[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
      float mx = -INFINITY;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        s[i][k] = valid[k] ? s[i][k] * kss[k] : -INFINITY;
        mx = fmaxf(mx, s[i][k]);
      }
      const float mn = fmaxf(m[i], dpp_max16(mx));
      const float corr = m[i] == -INFINITY ? 0.f : fast_exp2(m[i] - mn);
      m[i] = mn;
      float pr[4], ps = 0.f;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        pr[k] = valid[k] ? fast_exp2(s[i][k] - mn) : 0.f;
        ps += pr[k];
      }
      lp[i] = lp[i] * corr + ps;
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[i][e] *= corr;
      p01[i] = as_u(half2_t{(_Float16)(pr[0] * vss[0]), (_Float16)(pr[1] * vss[1])});
      p23[i] = as_u(half2_t{(_Float16)(pr[2] * vss[2]), (_Float16)(pr[3] * vss[3])});
    }
    *reinterpret_cast<u32x4*>(&sm.pt[2 * tc][4 * tr]) = u32x4{p01[0], p01[1], p01[2], p01[3]};
    *reinterpret_cast<u32x4*>(&sm.pt[2 * tc + 1][4 * tr]) = u32x4{p23[0], p23[1], p23[2], p23[3]};
    lds_barrier();                                    // pt and v2 complete
    // PV: rows 4 tr .. +3 x dims 4 tc .. +3 and 64 + 4 tc .. +3.
    u32x4 pv = *reinterpret_cast<const u32x4*>(&sm.pt[0][4 * tr]);
    u32x4 va = *reinterpret_cast<const u32x4*>(&sm.v2[0][4 * tc]);
    u32x4 vb = *reinterpret_cast<const u32x4*>(&sm.v2[0][64 + 4 * tc]);
#pragma unroll 2
    for (int kp = 0; kp < KT / 2; ++kp) {
      const int kn = min(kp + 1, KT / 2 - 1);
      const u32x4 pn = *reinterpret_cast<const u32x4*>(&sm.pt[kn][4 * tr]);
      const u32x4 van = *reinterpret_cast<const u32x4*>(&sm.v2[kn][4 * tc]);
      const u32x4 vbn = *reinterpret_cast<const u32x4*>(&sm.v2[kn][64 + 4 * tc]);
      const uint32_t pw[4] = {pv.x, pv.y, pv.z, pv.w};
      const uint32_t vw[8] = {va.x, va.y, va.z, va.w, vb.x, vb.y, vb.z, vb.w};
#pragma unroll
      for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e)
          acc[i][e] = __builtin_amdgcn_fdot2(as_h2(pw[i]), as_h2(vw[e]), acc[i][e], false);
      __builtin_amdgcn_sched_barrier(0);
      pv = pn;
      va = van;
      vb = vbn;
    }
  }
#pragma unroll
  for (int i = 0; i < 4; ++i) {
    const float l = dpp_sum16(lp[i]);
    if (tr < ntok) {
      const float inv = l > 0.f ? 1.f / (l * PSCALE) : 0.f;
      __half h[8];
#pragma unroll
      for (int e = 0; e < 8; ++e) h[e] = __float2half_rn(acc[i][e] * inv);
      __half* dst = p.out + (int64_t)(t0 + tr) * p.out_stride + (head * 4 + i) * D + 4 * tc;
      *reinterpret_cast<uint2*>(dst) = *reinterpret_cast<const uint2*>(h);
      *reinterpret_cast<uint2*>(dst + 64) = *reinterpret_cast<const uint2*>(h + 4);
    }
  }
}

}  // namespace lfm3pa7
