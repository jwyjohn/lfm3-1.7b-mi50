// W8A8 GEMM with 128-column group scales on both operands (gfx906, v_dot4).
//
// y[m][n] = sum_g sa[g][m] * sb[g][n] * sum_{k in g} qa[m][k] * qb[n][k]
//
// A and B use the direct-LDS packed layout uint32[K/64][ceil(rows/128)][16][128]
// (dword = 4 consecutive k of one row).  Scales are fp32 [K/128][rows padded to 128].
// Tile 128 x 128, 256 threads, 8 x 8 outputs per thread; each 128-k group ends with
// an int32 -> fp32 rescale, so a block holds both int32 and fp32 accumulators and
// runs one per CU.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

using DirectLdsResource = int32_t __attribute__((ext_vector_type(4)));
using DirectLdsPointer = __attribute__((address_space(3))) uint32_t*;

__device__ void direct_lds_raw_load(DirectLdsResource resource,
    DirectLdsPointer destination, int size, int voffset,
    int soffset, int offset, int cache)
    __asm("llvm.amdgcn.raw.buffer.load.lds");

namespace lfm3g128 {
constexpr int THREADS = 256;
constexpr int TILE = 128;
constexpr int HALF_PACKS = 16;

struct HalfTile {
  uint32_t a[HALF_PACKS][TILE];
  uint32_t b[HALF_PACKS][TILE];
};

struct SharedTile {
  HalfTile half[2];
};

__device__ __forceinline__ DirectLdsResource resource_for(const void* pointer, uint32_t bytes) {
  uint64_t address = reinterpret_cast<uintptr_t>(pointer);
  return DirectLdsResource{static_cast<int32_t>(address), static_cast<int32_t>(address >> 32),
                           static_cast<int32_t>(bytes), 0x00020000};
}

// One block = 128 rows x one 128-column group of a row-major fp16 matrix.
// Thread t: row t % 128, half t / 128 (64 columns -> 16 packed dwords).
// GATED: the row holds [gate | up] (2 x columns) and the value is silu(gate) * up.
__device__ __forceinline__ void load64(const __half* src, float (&v)[64]) {
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint4 u = reinterpret_cast<const uint4*>(src)[i];
    const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      __half2 h;
      __builtin_memcpy(&h, &w[j], 4);
      const float2 f = __half22float2(h);
      v[8 * i + 2 * j] = f.x;
      v[8 * i + 2 * j + 1] = f.y;
    }
  }
}

template <bool GATED>
__global__ __launch_bounds__(THREADS) void quantize_pack(
    const __half* __restrict__ x, int64_t x_stride, uint32_t* __restrict__ packed,
    float* __restrict__ scales, int rows, int columns) {
  __shared__ float partner[THREADS];
  const int tile = blockIdx.x, group = blockIdx.y;
  const int tiles = gridDim.x;
  const int local_row = threadIdx.x % TILE, half = threadIdx.x / TILE;
  const int row = tile * TILE + local_row;
  float v[64];
  if (row < rows) {
    const __half* src = x + row * x_stride + group * 128 + half * 64;
    load64(src, v);
    if (GATED) {
      float u[64];
      load64(src + columns, u);
#pragma unroll
      for (int i = 0; i < 64; ++i) v[i] = v[i] / (1.f + __expf(-v[i])) * u[i];
    }
  } else {
#pragma unroll
    for (int i = 0; i < 64; ++i) v[i] = 0.f;
  }
  float m = 0.f;
#pragma unroll
  for (int i = 0; i < 64; ++i) m = fmaxf(m, fabsf(v[i]));
  partner[threadIdx.x] = m;
  __syncthreads();
  m = fmaxf(m, partner[threadIdx.x ^ TILE]);
  const float scale = m > 0.f ? m / 127.f : 1.f;
  const float inv = 1.f / scale;
  if (half == 0) scales[(int64_t)group * tiles * TILE + row] = scale;
  uint32_t* dst = packed + ((int64_t)(2 * group + half) * tiles + tile) * HALF_PACKS * TILE + local_row;
#pragma unroll
  for (int p = 0; p < HALF_PACKS; ++p) {
    uint32_t d = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const int q = (int)rintf(fminf(fmaxf(v[4 * p + j] * inv, -127.f), 127.f));
      d |= (uint32_t)(q & 0xff) << (8 * j);
    }
    dst[p * TILE] = d;
  }
}

template <int HALF>
__device__ __forceinline__ void copy_half(SharedTile* shared, DirectLdsResource a_resource,
    DirectLdsResource b_resource, int tile_m, int tile_n, int m_tiles, int n_tiles, int k_half, int wave) {
#pragma unroll
  for (int chunk = 0; chunk < HALF_PACKS * TILE / THREADS; ++chunk) {
    uint32_t source = ((uint32_t(k_half) * m_tiles + tile_m) * HALF_PACKS * TILE + chunk * THREADS + threadIdx.x) * 4;
    uint32_t destination = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&shared->half[HALF].a[0][0] + chunk * THREADS + wave * 64)));
    direct_lds_raw_load(a_resource, reinterpret_cast<DirectLdsPointer>(destination), 4,
                        static_cast<int32_t>(source), 0, 0, 0);
  }
#pragma unroll
  for (int chunk = 0; chunk < HALF_PACKS * TILE / THREADS; ++chunk) {
    uint32_t source = ((uint32_t(k_half) * n_tiles + tile_n) * HALF_PACKS * TILE + chunk * THREADS + threadIdx.x) * 4;
    uint32_t destination = __builtin_amdgcn_readfirstlane(static_cast<uint32_t>(
        reinterpret_cast<uintptr_t>(&shared->half[HALF].b[0][0] + chunk * THREADS + wave * 64)));
    direct_lds_raw_load(b_resource, reinterpret_cast<DirectLdsPointer>(destination), 4,
                        static_cast<int32_t>(source), 0, 0, 0);
  }
}

typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));

// Thread (lm, ln) owns rows {4 lm .. +3, 64 + 4 lm .. +3} and the same columns of ln,
// so each operand is two 16-byte LDS reads per pack.
__device__ __forceinline__ void load_pack(const uint32_t (&src)[TILE], int local, uint32_t (&v)[8]) {
  const u32x4 lo = *reinterpret_cast<const u32x4*>(&src[4 * local]);
  const u32x4 hi = *reinterpret_cast<const u32x4*>(&src[64 + 4 * local]);
  v[0] = lo.x; v[1] = lo.y; v[2] = lo.z; v[3] = lo.w;
  v[4] = hi.x; v[5] = hi.y; v[6] = hi.z; v[7] = hi.w;
}

template <int HALF>
__device__ __forceinline__ void compute_half(SharedTile* shared, int (&accum)[8][8], int local_m, int local_n) {
  uint32_t a[8], b[8];
  load_pack(shared->half[HALF].a[0], local_m, a);
  load_pack(shared->half[HALF].b[0], local_n, b);
#pragma unroll
  for (int pack = 0; pack < HALF_PACKS; ++pack) {
    uint32_t an[8], bn[8];
    const int next = pack + 1 < HALF_PACKS ? pack + 1 : pack;
    load_pack(shared->half[HALF].a[next], local_m, an);
    load_pack(shared->half[HALF].b[next], local_n, bn);
#pragma unroll
    for (int row = 0; row < 8; ++row)
#pragma unroll
      for (int col = 0; col < 8; ++col)
        asm("v_dot4_i32_i8 %0, %1, %2, %3" : "=v"(accum[row][col]) : "v"(a[row]), "v"(b[col]), "v"(accum[row][col]));
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int i = 0; i < 8; ++i) {
      a[i] = an[i];
      b[i] = bn[i];
    }
  }
}

struct GemmArgs {
  const uint32_t* packed_a;
  const float* a_scales;
  const uint32_t* packed_b;
  const float* b_scales;
  __half* out;
  int64_t out_stride;
  const __half* residual;     // optional: out = residual + y
  int64_t residual_stride;
  int32_t rows, outputs, columns, pad;
};

__global__ __launch_bounds__(THREADS, 1) void gemm(GemmArgs g) {
  __shared__ SharedTile shared;
  const int m_tiles = (g.rows + TILE - 1) / TILE;
  const int n_tiles = (g.outputs + TILE - 1) / TILE;
  const int m_group = blockIdx.x / (n_tiles * 4);
  const int within = blockIdx.x % (n_tiles * 4);
  const int tile_m = m_group * 4 + within % 4, tile_n = within / 4;
  if (tile_m >= m_tiles) return;
  const int local_m = threadIdx.x / 16, local_n = threadIdx.x % 16;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x / 64);
  const DirectLdsResource a_resource = resource_for(g.packed_a, uint32_t(m_tiles) * TILE * g.columns);
  const DirectLdsResource b_resource = resource_for(g.packed_b, uint32_t(n_tiles) * TILE * g.columns);
  const int groups = g.columns / 128;
  const float* sa = g.a_scales + tile_m * TILE + 4 * local_m;
  const float* sb = g.b_scales + tile_n * TILE + 4 * local_n;
  const int sa_stride = m_tiles * TILE, sb_stride = n_tiles * TILE;
  float acc[8][8];
#pragma unroll
  for (int r = 0; r < 8; ++r)
#pragma unroll
    for (int c = 0; c < 8; ++c) acc[r][c] = 0.f;
  copy_half<0>(&shared, a_resource, b_resource, tile_m, tile_n, m_tiles, n_tiles, 0, wave);
  asm volatile("s_waitcnt vmcnt(0)" : : : "memory");
  __syncthreads();
  for (int group = 0; group < groups; ++group) {
    float ra[8], rb[8];
    {
      const float4 a0 = *reinterpret_cast<const float4*>(sa + group * sa_stride);
      const float4 a1 = *reinterpret_cast<const float4*>(sa + group * sa_stride + 64);
      const float4 b0 = *reinterpret_cast<const float4*>(sb + group * sb_stride);
      const float4 b1 = *reinterpret_cast<const float4*>(sb + group * sb_stride + 64);
      ra[0] = a0.x; ra[1] = a0.y; ra[2] = a0.z; ra[3] = a0.w; ra[4] = a1.x; ra[5] = a1.y; ra[6] = a1.z; ra[7] = a1.w;
      rb[0] = b0.x; rb[1] = b0.y; rb[2] = b0.z; rb[3] = b0.w; rb[4] = b1.x; rb[5] = b1.y; rb[6] = b1.z; rb[7] = b1.w;
    }
    int accum[8][8] = {};
    copy_half<1>(&shared, a_resource, b_resource, tile_m, tile_n, m_tiles, n_tiles, group * 2 + 1, wave);
    compute_half<0>(&shared, accum, local_m, local_n);
    asm volatile("s_waitcnt vmcnt(0)" : : : "memory");
    __syncthreads();
    if (group + 1 < groups)
      copy_half<0>(&shared, a_resource, b_resource, tile_m, tile_n, m_tiles, n_tiles, group * 2 + 2, wave);
    compute_half<1>(&shared, accum, local_m, local_n);
#pragma unroll
    for (int r = 0; r < 8; ++r)
#pragma unroll
      for (int c = 0; c < 8; ++c) acc[r][c] = fmaf((float)accum[r][c] * rb[c], ra[r], acc[r][c]);
    asm volatile("s_waitcnt vmcnt(0)" : : : "memory");
    __syncthreads();
  }
#pragma unroll
  for (int r = 0; r < 8; ++r) {
    const int m = tile_m * TILE + (r < 4 ? 4 * local_m + r : 64 + 4 * local_m + r - 4);
    if (m >= g.rows) continue;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
      const int n = tile_n * TILE + 64 * h + 4 * local_n;
      float y[4] = {acc[r][4 * h], acc[r][4 * h + 1], acc[r][4 * h + 2], acc[r][4 * h + 3]};
      if (n + 3 < g.outputs) {
        if (g.residual) {
          const uint2 u = *reinterpret_cast<const uint2*>(g.residual + (int64_t)m * g.residual_stride + n);
          const __half2 r0 = *reinterpret_cast<const __half2*>(&u.x), r1 = *reinterpret_cast<const __half2*>(&u.y);
          y[0] += __low2float(r0); y[1] += __high2float(r0); y[2] += __low2float(r1); y[3] += __high2float(r1);
        }
        __half2 o0 = __floats2half2_rn(y[0], y[1]), o1 = __floats2half2_rn(y[2], y[3]);
        uint2 u;
        __builtin_memcpy(&u.x, &o0, 4);
        __builtin_memcpy(&u.y, &o1, 4);
        *reinterpret_cast<uint2*>(g.out + (int64_t)m * g.out_stride + n) = u;
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          if (n + c < g.outputs) {
            float v = y[c];
            if (g.residual) v += __half2float(g.residual[(int64_t)m * g.residual_stride + n + c]);
            g.out[(int64_t)m * g.out_stride + n + c] = __float2half_rn(v);
          }
      }
    }
  }
}
// GATED quantization for the down projection (silu(gate) * up), plain for everything else.
template __global__ __launch_bounds__(THREADS) void quantize_pack<true>(
    const __half* __restrict__, int64_t, uint32_t* __restrict__, float* __restrict__, int, int);
template __global__ __launch_bounds__(THREADS) void quantize_pack<false>(
    const __half* __restrict__, int64_t, uint32_t* __restrict__, float* __restrict__, int, int);
}  // namespace lfm3g128
