// LFM3 + DSpark speculative decode megakernel for gfx906.
//
// One persistent launch runs `rounds` speculative rounds of one request:
//   context  project the taps of the tokens committed last round and append them to the
//            drafter KV ring (W8 fc band GEMVs, k/v GEMV, k_norm + RoPE + int8);
//   draft    3 attention-only drafter layers over [anchor, mask x 8], base logits on a
//            frequency-cropped LM head, Markov steps with confidence-based early stop;
//   verify   24 target layers + full LM head over M = 1 + drafts tokens, every position's
//            greedy token (repetition penalty included) computed in the kernel;
//   commit   accepted prefix + bonus token; K/V of committed positions go to the KV pages,
//            conv history lives in a position-indexed ring so a rejection needs no rollback.
//
// Weights are int8 with a fp16 scale per 128 inputs stored biased (u = q + 128), activations
// fp16 (exactly the v2 decode kernel), attention reads vLLM's int8_per_token_head pages.
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <stdint.h>

using DirectLdsResource = int32_t __attribute__((ext_vector_type(4)));
using DirectLdsPointer = __attribute__((address_space(3))) uint32_t*;
__device__ void direct_lds_raw_load(DirectLdsResource resource, DirectLdsPointer destination, int size, int voffset,
                                    int soffset, int offset, int cache) __asm("llvm.amdgcn.raw.buffer.load.lds");

// Load slots per GEMV phase (NB - 1 units in flight per wave), by token count; deeper is worse past ~12 KB per wave.
#ifndef LFM3_F16
#define LFM3_X8                 // int8 x in the GEMVs; -DLFM3_F16 keeps fp16 x (slower, closer to the fp16 reference)
#endif
#ifndef LFM3_NBX
#define LFM3_NBX 2
#endif
#ifndef LFM3_NB1
#define LFM3_NB1 4
#endif
#ifndef LFM3_NB2
#define LFM3_NB2 4
#endif
#ifndef LFM3_NB3
#define LFM3_NB3 4
#endif
#ifndef LFM3_NB4
#define LFM3_NB4 3
#endif

namespace lfm3ds {

// The thread id lives in a VGPR the register allocator never touches (the kernels are limited to v0..v254):
// reading it is one v_mov, while a spilled id would need a scratch reload whose s_waitcnt vmcnt(0) drains
// every weight load in flight. The volatile read also keeps lane-derived values inside their phase, so LICM
// cannot hoist them above the round/step loops where they would stay live through every other phase.
#define LFM3_KERNEL_ATTR __attribute__((amdgpu_num_vgpr(255)))
__device__ __forceinline__ void tidx_init() {
  const int r = threadIdx.x;
  asm volatile("v_mov_b32 v255, %0" : : "v"(r) : "v255");
}
__device__ __forceinline__ int tidx() {
  int r;
  asm volatile("v_mov_b32 %0, v255" : "=v"(r) : : "v255");
  return r;
}

constexpr int H = 2048, I = 7680, QH = 16, KVH = 4, D = 128;
constexpr int THREADS = 256, WAVES = THREADS / 64;
constexpr int REC = 264;          // bytes per (slot, kv head) record
constexpr int SUB = 16;           // keys per tile (one contiguous NHD group of all kv heads)
constexpr int MT = 9;             // max tokens in one pass (drafter block: 9, as trained; verify: anchor + 8 drafts)
constexpr int DRAFT_LAYERS = 3;
constexpr int NUM_TAPS = 5;
constexpr int SPLITS = 60;
constexpr int RING = 32;          // conv history ring (positions)
constexpr int CTX_MASK = 255;     // drafter KV ring: 256 groups of 16 keys = 4096 positions
constexpr int QKVG = 3088;
constexpr int QKV = 3072;
constexpr int MIXS = 6144;        // floats per token of the mixer projection workspace
constexpr float LOG2E = 1.4426950408889634f;
constexpr float PSCALE = 4096.f;
constexpr int TILE_BYTES = SUB * KVH * REC;
constexpr int TILE_DWORDS = TILE_BYTES / 4;
constexpr int BT_MAX = 512;
constexpr int SEEN_WORDS = 128000 / 32;

typedef _Float16 half2_t __attribute__((ext_vector_type(2)));
typedef uint32_t u32x2 __attribute__((ext_vector_type(2)));
typedef uint32_t u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

// A local that is only defined under a condition (`if (x) load; ... if (x) use`) has a CFG path from a use back to the previous
// iteration's definition, because after inlining its slot lives for the whole kernel: mem2reg then carries the value in phis across
// the interpreter's back edge, and those long live ranges are what ends up spilled. An empty asm defines it at its scope: no
// instruction, no phi.
#ifdef __HIP_DEVICE_COMPILE__
#define LFM3_UNDEF(x) asm volatile("" : "=v"(x))
#else
#define LFM3_UNDEF(x) (void)0
#endif

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
__device__ __forceinline__ half2_t as_h2(uint32_t u) {
  half2_t h;
  __builtin_memcpy(&h, &u, 4);
  return h;
}

struct Layer {
  int64_t kind;  // 0 conv, 1 full attention, 2 sliding attention, 3 drafter attention
  const __half* op_norm;
  const __half* ffn_norm;
  const __half* post_op_norm;
  const __half* post_ffn_norm;
  const uint8_t* w_mix; const __half* s_mix;
  const uint8_t* w_out; const __half* s_out;
  const uint8_t* w13; const __half* s13;
  const uint8_t* w2; const __half* s2;
  const __half* conv_w;
  __half* conv_state; int64_t cs_slot, cs_dim, cs_t; const int32_t* state_index;
  const __half* q_norm; const __half* k_norm; const __half* rope; int64_t rotary;
  int8_t* kv; int64_t kv_block, kv_head, kv_slot, block_size;
  const int32_t* block_table; const int32_t* seq_lens; const int64_t* slot_mapping;
  int64_t window;
  __half* ring;         // [RING][2048] conv history by position
  int64_t tap_slot;     // index into target_layer_ids, or -1
  uint8_t* fresh;       // [MT][KVH][REC]: K/V of this layer's verified tokens
  int64_t pad;
};

struct State {
  int32_t a;            // position of the anchor (next input)
  int32_t anchor;       // token at position a
  int32_t count;        // tokens emitted by this launch
  int32_t rounds_left;
  int32_t limit;        // positions < limit may be processed in this launch
  int32_t budget;       // max tokens this launch may emit
  int32_t eos;          // stop token (or -1)
  int32_t M;            // tokens verified this round
  int32_t ctx_new;      // context positions waiting for the drafter ring
  int32_t ctx_pos;      // their first position
  int32_t ctx_lo;       // oldest valid position in the drafter ring
  int32_t run;          // 0: stop
  int32_t force_m;      // debug: >0 verify force_m tokens from `tokens`, no drafter
  int32_t rounds_done;
  int32_t accepted;     // accepted draft tokens over this launch
  int32_t verified;     // verified tokens over this launch
  int32_t miss;         // consecutive speculative rounds that accepted nothing
  int32_t skip;         // upcoming rounds without drafter
  int32_t skip_after;   // misses that trigger skipping (0: never skip)
  int32_t skip_rounds;  // rounds to skip
  int32_t stop_at;      // no round starts from this position on; limit only bounds the verified tokens
  int32_t ckpt_slot;    // conv slot that receives the state at stop_at once the launch reaches it, or -1
  int32_t tokens[MT];   // tokens[0] = anchor, tokens[1..M-1] = drafts
};

struct Globals {
  const Layer* layers;      // 24 target layers then DRAFT_LAYERS drafter layers
  const __half* embed;
  const __half* final_norm;
  const uint8_t* w_head; const __half* s_head;
  const __half* d_final_norm;
  const uint8_t* w_crop; const __half* s_crop;   // cropped LM head rows + confidence row
  const int32_t* crop_ids;
  const int8_t* w2_crop;    // [N][256] int8 Markov output
  const float* w2_scale;    // [N]
  const int8_t* w1q;        // [vocab][256] int8 Markov input
  const float* w1s;         // [vocab]
  const float* c2;          // [vocab]: confidence contribution of the previous token (fp32)
  float* base;              // [MT][N + 4] base logits + confidence hidden term
  const __half* tap_norms;  // [5][2048]
  const uint8_t* w_fc; const __half* s_fc;       // [5][2048 rows][2048]
  const __half* hidden_norm;
  const uint8_t* w_ckv; const __half* s_ckv;     // 3 x (k, v) rows: [3072][2048]
  float* ws;
  uint32_t* barrier;
  State* st;
  uint32_t* seen;
  const float* rep;         // repetition penalty
  uint64_t* best;           // [MT] packed (ordered logit, ~index)
  int32_t* out_tokens;
  int64_t* trace;
  float* dbg_logits;        // optional: [MT][vocab] raw logits of the verify head
  int64_t vocab;
  int64_t crop_rows;        // N (multiple of 64)
  int64_t mask_token;
  int64_t max_drafts;
  int64_t attn_mask;        // bit li: target layer li has attention K/V
  float conf_bias;
  float conf_thr;           // stop drafting when the confidence logit falls below this
  float inv_rep;
  float pad;
};

// Workspace layout (floats).
constexpr int WS_H0 = 0, WS_H1 = WS_H0 + MT * H, WS_MIX = WS_H1 + MT * H;
constexpr int WS_Y = WS_MIX + MT * MIXS, WS_O = WS_Y + MT * H, WS_A = WS_O + MT * H;
constexpr int WS_D = WS_A + MT * I;                   // [4 bands][MT][H]
constexpr int WS_G13 = WS_D + 4 * MT * H;             // [MT][2 I]
constexpr int WS_TAP = WS_G13 + MT * 2 * I;           // [5][MT][H]
constexpr int WS_CP = WS_TAP + NUM_TAPS * MT * H;     // fc band outputs [5][MT][H]
constexpr int WS_CKV = WS_CP + NUM_TAPS * MT * H;     // [MT][3072]
constexpr int WS_TRASH = WS_CKV + MT * QKV;
constexpr int WS_PART = WS_TRASH + 64;                // [MT * QH][SPLITS][D + 2]
constexpr int WS_X16 = WS_PART + MT * QH * SPLITS * (D + 2);  // normalized x, fp16 [MT][H]
constexpr int WS_Q = WS_X16 + MT * H / 2;                 // int8 Q of the block's tokens: [MT][QH][D] bytes, then [MT][QH][4] scales
constexpr int WS_FLOATS = WS_Q + MT * (QH * D + QH * 16) / 4;
// WS_Y and WS_A hold fp16 too ([MT][H] and [MT][I]): every reader rounds to fp16 anyway.

// x of one band per token, flat: token t starts at t * band columns. Capacity: 9 x 2048, 4 x 4096, 2 x 7680.
struct GemvShared {
#ifdef LFM3_X8
  alignas(16) int8_t x8[MT * H];     // int8 x, one scale per 16 columns
  float xs[MT][H / 16];
#else
  alignas(16) __half x[MT * H];
#endif
};

// Row totals of a GEMV in batches: the 4 row-of-16 sums per (wave, token, row), written by 4 lanes, summed by all lanes.
constexpr int RB = 8;

struct HeadShared {
  GemvShared g;
  uint32_t seen[SEEN_WORDS];
};

// Rows consumed on the wave that computed them (gate/up, conv B/C/X) never touch global memory.
struct StagedShared {
  GemvShared g;
  float rows[WAVES][MT][64];
};

// One iteration's K, V and scales per wave (98 IT dwords): K [32 IT], V as fp16 pairs of adjacent keys [64 IT], k scale [IT],
// v scale [IT]. The tile sits right after the q8 of the pass's tokens, so a one-token pass can hold 32 keys.
constexpr int TILE_DW = 98 * 32;

struct AttnShared {
  int32_t blocks[BT_MAX];
  float qs8[4][QH][4];
  alignas(16) int8_t q8[4][QH][D];
  uint32_t tile_room[(WAVES * TILE_DW + 2 * QH * D / 4) - 4 * QH * D / 4 + 64];
};

union alignas(16) Shared {
  GemvShared g;
  HeadShared head;
  StagedShared staged;
  AttnShared attn;
};

struct Smem {
  Shared u;
  float4 rows[WAVES][MT][RB];
  float4 red[WAVES][MT];
};

// Row-of-16 sums of N accumulators, one VALU per step and accumulator (the same order as row_sum16). A DPP read of a VGPR
// written by a VALU in the previous two wait states is undefined and the compiler cannot see into inline asm, so every
// group carries its own s_nop and ties its operands: nothing can be scheduled between the wait and the reads.
#define LFM3_DPP1(r, swz) "v_add_f32_dpp %" #r ", %" #r ", %" #r " " swz " row_mask:0xf bank_mask:0xf\n\t"
#define LFM3_DPP_G4(NOP, a, b, c, d, swz) \
  asm volatile(NOP LFM3_DPP1(0, swz) LFM3_DPP1(1, swz) LFM3_DPP1(2, swz) LFM3_DPP1(3, swz) : "+v"(a), "+v"(b), "+v"(c), "+v"(d))
#define LFM3_DPP_G2(NOP, a, b, swz) asm volatile(NOP LFM3_DPP1(0, swz) LFM3_DPP1(1, swz) : "+v"(a), "+v"(b))
#define LFM3_DPP_G1(NOP, a, swz) asm volatile(NOP LFM3_DPP1(0, swz) : "+v"(a))
// One step over N accumulators. Only the batch's first group waits for the VALU that wrote the inputs. With N >= 3 the same
// register comes back at least 2 wait states later, so later groups need no s_nop; the sched_barrier keeps the writers out.
#define LFM3_DPP_STEP(FIRST, swz)                                                                                      \
  do {                                                                                                                 \
    _Pragma("unroll") for (int g = 0; g < N / 4; ++g) {                                                                \
      if ((FIRST) && g == 0) LFM3_DPP_G4("s_nop 1\n\t", v[4 * g], v[4 * g + 1], v[4 * g + 2], v[4 * g + 3], swz);       \
      else LFM3_DPP_G4("", v[4 * g], v[4 * g + 1], v[4 * g + 2], v[4 * g + 3], swz);                                    \
    }                                                                                                                  \
    if (N < 4) {                                                                                                       \
      if (N % 4 >= 2) LFM3_DPP_G2("s_nop 1\n\t", v[N / 4 * 4], v[N / 4 * 4 + 1], swz);                                  \
      if (N % 2 == 1) LFM3_DPP_G1("s_nop 1\n\t", v[N - 1], swz);                                                        \
    } else {                                                                                                           \
      if (N % 4 >= 2) LFM3_DPP_G2("", v[N / 4 * 4], v[N / 4 * 4 + 1], swz);                                             \
      if (N % 2 == 1) LFM3_DPP_G1("", v[N - 1], swz);                                                                   \
    }                                                                                                                  \
  } while (0)

template <int N>
__device__ __forceinline__ void row_sum16_batch(float* v) {
  __builtin_amdgcn_sched_barrier(0);
  LFM3_DPP_STEP(true, "quad_perm:[1,0,3,2]");
  LFM3_DPP_STEP(false, "quad_perm:[2,3,0,1]");
  LFM3_DPP_STEP(false, "row_half_mirror");
  LFM3_DPP_STEP(false, "row_mirror");
}

#define LFM3_DPPM1(r, swz) "v_max_f32_dpp %" #r ", %" #r ", %" #r " " swz " row_mask:0xf bank_mask:0xf\n\t"
#define LFM3_DPP_MAX4(a, b, c, d, swz) \
  asm volatile("s_nop 1\n\t" LFM3_DPPM1(0, swz) LFM3_DPPM1(1, swz) LFM3_DPPM1(2, swz) LFM3_DPPM1(3, swz) : "+v"(a), "+v"(b), "+v"(c), "+v"(d))
template <int N>
__device__ __forceinline__ void row_max16_batch(float* v) {
  static_assert(N == 4);
  LFM3_DPP_MAX4(v[0], v[1], v[2], v[3], "quad_perm:[1,0,3,2]");
  LFM3_DPP_MAX4(v[0], v[1], v[2], v[3], "quad_perm:[2,3,0,1]");
  LFM3_DPP_MAX4(v[0], v[1], v[2], v[3], "row_half_mirror");
  LFM3_DPP_MAX4(v[0], v[1], v[2], v[3], "row_mirror");
}

#define LFM3_DPP(v, ctrl) __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), ctrl, 0xf, 0xf, false))
#define LFM3_LANE(v, l) __builtin_bit_cast(float, __builtin_amdgcn_readlane(__builtin_bit_cast(int, v), l))

__device__ __forceinline__ float wave_total(float v) {
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

__device__ __forceinline__ float row_sum16(float v) {
  v += LFM3_DPP(v, 0xb1);
  v += LFM3_DPP(v, 0x4e);
  v += LFM3_DPP(v, 0x141);
  v += LFM3_DPP(v, 0x140);
  return v;
}

__device__ __forceinline__ float wave_max(float v) {
  v = row_max16(v);
  return fmaxf(fmaxf(LFM3_LANE(v, 0), LFM3_LANE(v, 16)), fmaxf(LFM3_LANE(v, 32), LFM3_LANE(v, 48)));
}

__device__ __forceinline__ float half_max(float v) {
  v = row_max16(v);
  const float lo = fmaxf(LFM3_LANE(v, 0), LFM3_LANE(v, 16)), hi = fmaxf(LFM3_LANE(v, 32), LFM3_LANE(v, 48));
  return (tidx() & 32) ? hi : lo;
}

__device__ __forceinline__ void lds_barrier() {
  asm volatile("s_waitcnt lgkmcnt(0)\n\ts_barrier" ::: "memory");
}

__device__ __forceinline__ int uniform_i32(int v) { return __builtin_amdgcn_readfirstlane(v); }
__device__ __forceinline__ int64_t uniform64(int64_t v) {
  const uint64_t u = (uint64_t)v;
  return (int64_t)(((uint64_t)(uint32_t)__builtin_amdgcn_readfirstlane((int)(u >> 32)) << 32) |
                   (uint32_t)__builtin_amdgcn_readfirstlane((int)(uint32_t)u));
}

// Scalar load of data that is read-only during a launch, at a uniform address: the barrier's L1 invalidate does not touch the scalar
// cache, and one lane's worth of bytes does not cross the CU's return path 64 times.
template <typename T>
__device__ __forceinline__ T sload(const T* p) {
  return gload(p);
}

// Grid barrier: a monotonic arrival counter (zeroed by the host before the launch) that wave 0 of each block bumps once all
// its block's stores are complete. Nothing waits for VMEM on the way out, so the next phase's loads, issued between arrive
// and wait, stay in flight through the barrier and L1 is invalidated without draining them.
__device__ __forceinline__ void grid_arrive(uint32_t* bar, uint32_t gen) {
  asm volatile("s_waitcnt vmcnt(0) lgkmcnt(0)\n\ts_barrier" ::: "memory");
  if (tidx() == 0) __hip_atomic_fetch_add(bar + 1, 1u, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}
__device__ __forceinline__ void grid_wait(uint32_t* bar, uint32_t& gen) {
  gen += gridDim.x;
  // Wave 0 polls on the scalar path: a vector load would come back only after this wave's own weight loads in flight.
  if (__builtin_amdgcn_readfirstlane(tidx() >> 6) == 0) {
    const uint64_t cnt = reinterpret_cast<uint64_t>(bar + 1);
    for (;;) {
      uint32_t v;
      asm volatile("s_load_dword %0, %1, 0x0 glc\n\ts_waitcnt lgkmcnt(0)" : "=s"(v) : "s"(cnt) : "memory");
      if ((int32_t)(v - gen) >= 0) break;
      __builtin_amdgcn_s_sleep(1);
    }
  }
  asm volatile("s_barrier\n\tbuffer_wbinvl1_vol" ::: "memory");
}

__device__ __forceinline__ void trace_mark(int64_t* trace, int& n, int tag) {
  if (trace && blockIdx.x == trace[1] && tidx() == 0 && n < 8180) {  // trace[1]: the traced block
    trace[n + 2] = tag;
    trace[n + 3] = (int64_t)__builtin_amdgcn_s_memrealtime();
    trace[0] = n + 2;
  }
  n += 2;
}

// ---------------------------------------------------------------- GEMV over M tokens
// One unit = 2 rows on one wave; lane l owns the 8-byte pieces l + 64 i of a 2048-column band and a row sums
// over the whole wave. Every linear layer goes through one code path (Plan + exact token count) so a layer's
// code fits the instruction cache; a band of a wider matrix (K = 7680) is the same path with an offset.
struct Plan {
  const uint8_t* W;
  const __half* S;
  int64_t rowbytes;
  int srow, sq, i0, rows;   // scale stride per row / per quarter, first piece index, rows of the matrix (band)
  int i3;                   // piece group of the 4th load: 3, or 2 when the band ends after 3 (its scale is 0, nothing is read twice)
  bool triple;              // three consecutive units (two conv channels) stay on one wave
  bool staged;              // rows go to this wave's LDS stage instead of out
  float* out;
  int stride;
};

template <int V>
struct IntC { static constexpr int value = V; };

// The k-th unit of wave w: rows stay consecutive in groups of 2 (6 for conv: two channels) and the groups interleave over the waves.
template <int R>
__device__ __forceinline__ int unit_at(bool triple, int k, int w, int nw) {
  const int G = (triple ? 6 : 2) / R;
  return G * (w + (k / G) * nw) + k % G;
}
template <int R>
__device__ __forceinline__ int units_of_wave(bool triple, int units, int w, int nw) {
  const int G = (triple ? 6 : 2) / R;
  return G * ((units / G - w + nw - 1) / nw);
}

// One GEMV phase: R rows per unit, NB load slots, so NB - 1 units are in flight per wave. Throughput is bounded by in-flight
// bytes / latency, so big phases want 8 KB per wave; short ones keep R = 2 for balance. pre() (barrier wait, x -> LDS) runs
// with the first units' loads already out. No conditional load in the loop: a load on one path only merges in-flight state and
// makes the compiler drain at the next use. Loads past the end re-read the last unit (an L1 hit).
// Lane layout of a GEMV unit: lane l owns columns 16 l .. +15 and 1024 + 16 l .. +15 of a 2048-column band, one 16-byte load
// each. Both 16-column pieces lie inside one scale group, so every row has 2 scales and 2 biases per lane, not 4.
template <int R>
struct BufR {
  u32x4 w[R][2];
  _Float16 s[R][2];
};

template <int M>
struct XRegs2 {
#ifdef LFM3_X8
  u32x4 x[M][2];
#else
  u32x4 x[M][4];
#endif
  float xs[M][2];
};

// x of this lane's two 16-column pieces, and the bias -1152 * sum that seeds each piece's first dot (weights are stored as 1024 +
// u, u = q + 128).
#ifdef LFM3_X8
template <int M>
__device__ __forceinline__ void xregs2_load(XRegs2<M>& xr, const GemvShared& gs, int lane) {
#pragma unroll
  for (int t = 0; t < M; ++t)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      xr.x[t][j] = *reinterpret_cast<const u32x4*>(gs.x8 + t * H + 16 * lane + 1024 * j);
      xr.xs[t][j] = gs.xs[t][lane + 64 * j];
    }
  __builtin_amdgcn_sched_barrier(0);
}
#else
template <int M>
__device__ __forceinline__ void xregs2_load(XRegs2<M>& xr, const GemvShared& gs, int lane) {
  const half2_t ones = {(_Float16)1.f, (_Float16)1.f};
#pragma unroll
  for (int t = 0; t < M; ++t)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const int c = 16 * lane + 1024 * j;
      xr.x[t][2 * j] = *reinterpret_cast<const u32x4*>(gs.x + t * H + c);
      xr.x[t][2 * j + 1] = *reinterpret_cast<const u32x4*>(gs.x + t * H + c + 8);
    }
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (int t = 0; t < M; ++t)
#pragma unroll
    for (int j = 0; j < 2; ++j) {
      const u32x4 a = xr.x[t][2 * j], b = xr.x[t][2 * j + 1];
      float sum = __builtin_amdgcn_fdot2(as_h2(a.x), ones, 0.f, false);
      sum = __builtin_amdgcn_fdot2(as_h2(a.y), ones, sum, false);
      sum = __builtin_amdgcn_fdot2(as_h2(a.z), ones, sum, false);
      sum = __builtin_amdgcn_fdot2(as_h2(a.w), ones, sum, false);
      sum = __builtin_amdgcn_fdot2(as_h2(b.x), ones, sum, false);
      sum = __builtin_amdgcn_fdot2(as_h2(b.y), ones, sum, false);
      sum = __builtin_amdgcn_fdot2(as_h2(b.z), ones, sum, false);
      sum = __builtin_amdgcn_fdot2(as_h2(b.w), ones, sum, false);
      xr.xs[t][j] = -1152.f * sum;
    }
}
#endif

template <int R, int NB>
__device__ __forceinline__ void bufr_undef(BufR<R> (&buf)[NB]) {
#pragma unroll
  for (int s = 0; s < NB; ++s)
#pragma unroll
    for (int r = 0; r < R; ++r)
#pragma unroll
      for (int j = 0; j < 2; ++j) {
        LFM3_UNDEF(buf[s].w[r][j]);
        LFM3_UNDEF(buf[s].s[r][j]);
      }
}

// Row pointers of the next unit to load, advanced by a few scalar adds instead of recomputed from the unit number.
struct Cursor {
  uint64_t wp, sp;
  int left, gc;
};

template <int R>
__device__ __forceinline__ void bufr_load_at(BufR<R>& b, const Cursor& c, uint32_t rowb, uint32_t srowb, uint32_t off_w0, uint32_t off_w1, uint32_t off_s0, uint32_t off_s1) {
#pragma unroll
  for (int r = 0; r < R; ++r) {
    typedef const __attribute__((address_space(1))) uint8_t B1;
    typedef const __attribute__((address_space(1))) u32x4 W1;
    typedef const __attribute__((address_space(1))) _Float16 S1;
    B1* bw = (B1*)(c.wp + (uint64_t)((uint32_t)r * rowb));
    B1* bs = (B1*)(c.sp + (uint64_t)((uint32_t)r * srowb));
    b.w[r][0] = __builtin_nontemporal_load((W1*)(bw + off_w0));
    b.w[r][1] = __builtin_nontemporal_load((W1*)(bw + off_w1));
    b.s[r][0] = *(S1*)(bs + off_s0);
    b.s[r][1] = *(S1*)(bs + off_s1);
  }
}

template <int R>
__device__ __forceinline__ void bufr_load(BufR<R>& b, const Plan& p, int unit, uint32_t off_w0, uint32_t off_w1, uint32_t off_s0, uint32_t off_s1) {
#pragma unroll
  for (int r = 0; r < R; ++r) {
    const int row = R * unit + r;
    uint64_t bw = reinterpret_cast<uint64_t>(p.W) + (int64_t)row * p.rowbytes + 512 * p.i0;
    uint64_t bs = reinterpret_cast<uint64_t>(p.S) + 2 * ((int64_t)row * p.srow + p.i0);
    asm("" : "+s"(bw), "+s"(bs));
    b.w[r][0] = gload_nt(reinterpret_cast<const u32x4*>(reinterpret_cast<const uint8_t*>(bw) + off_w0));
    b.w[r][1] = gload_nt(reinterpret_cast<const u32x4*>(reinterpret_cast<const uint8_t*>(bw) + off_w1));
    b.s[r][0] = gload(reinterpret_cast<const _Float16*>(reinterpret_cast<const uint8_t*>(bs) + off_s0));
    b.s[r][1] = gload(reinterpret_cast<const _Float16*>(reinterpret_cast<const uint8_t*>(bs) + off_s1));
  }
}

#ifdef LFM3_X8
template <int M, int R>
__device__ __forceinline__ void unit_dot_r(const BufR<R>& b, const XRegs2<M>& xr, float (&acc)[R][M]) {
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < M; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const u32x4 wv = b.w[r][j];
      const uint32_t w0 = wv.x ^ 0x80808080u, w1 = wv.y ^ 0x80808080u, w2 = wv.z ^ 0x80808080u, w3 = wv.w ^ 0x80808080u;
      const float sc = (float)b.s[r][j];
#pragma unroll
      for (int t = 0; t < M; ++t) {
        const u32x4 xa = xr.x[t][j];
        int d = __builtin_amdgcn_sdot4(w0, xa.x, 0, false);
        d = __builtin_amdgcn_sdot4(w1, xa.y, d, false);
        d = __builtin_amdgcn_sdot4(w2, xa.z, d, false);
        d = __builtin_amdgcn_sdot4(w3, xa.w, d, false);
        acc[r][t] = fmaf(sc * xr.xs[t][j], (float)d, acc[r][t]);
      }
    }
  }
}
#else
template <int M, int R>
__device__ __forceinline__ void unit_dot_r(const BufR<R>& b, const XRegs2<M>& xr, float (&acc)[R][M]) {
  const uint32_t c = 0x64646464u;
#pragma unroll
  for (int r = 0; r < R; ++r)
#pragma unroll
    for (int t = 0; t < M; ++t) acc[r][t] = 0.f;
#pragma unroll
  for (int j = 0; j < 2; ++j) {
#pragma unroll
    for (int r = 0; r < R; ++r) {
      const u32x4 wv = b.w[r][j];
      const uint32_t w0 = wv.x, w1 = wv.y, w2 = wv.z, w3 = wv.w;
      const half2_t h0 = as_h2(__builtin_amdgcn_perm(c, w0, 0x04010400u)), h1 = as_h2(__builtin_amdgcn_perm(c, w0, 0x04030402u));
      const half2_t h2 = as_h2(__builtin_amdgcn_perm(c, w1, 0x04010400u)), h3 = as_h2(__builtin_amdgcn_perm(c, w1, 0x04030402u));
      const half2_t h4 = as_h2(__builtin_amdgcn_perm(c, w2, 0x04010400u)), h5 = as_h2(__builtin_amdgcn_perm(c, w2, 0x04030402u));
      const half2_t h6 = as_h2(__builtin_amdgcn_perm(c, w3, 0x04010400u)), h7 = as_h2(__builtin_amdgcn_perm(c, w3, 0x04030402u));
      const float sc = (float)b.s[r][j];
      // Round-robin over the tokens: no two consecutive instructions depend on each other.
      float dd[M];
      const half2_t hh[8] = {h0, h1, h2, h3, h4, h5, h6, h7};
#pragma unroll
      for (int t = 0; t < M; ++t) {
        
        dd[t] = __builtin_amdgcn_fdot2(hh[0], as_h2(xr.x[t][2 * j].x), xr.xs[t][j], false);
      }
      __builtin_amdgcn_sched_barrier(0);
#define LFM3_RR(k, field, q)                                                                 \
  _Pragma("unroll") for (int t = 0; t < M; ++t) dd[t] = __builtin_amdgcn_fdot2(hh[k], as_h2(xr.x[t][2 * j + q].field), dd[t], false); \
  __builtin_amdgcn_sched_barrier(0);
      LFM3_RR(1, y, 0) LFM3_RR(2, z, 0) LFM3_RR(3, w, 0) LFM3_RR(4, x, 1) LFM3_RR(5, y, 1) LFM3_RR(6, z, 1) LFM3_RR(7, w, 1)
#pragma unroll
      for (int t = 0; t < M; ++t) acc[r][t] = fmaf(sc, dd[t], acc[r][t]);
    }
  }
}

#endif

template <int M, int R, int NB, typename Pre>
__device__ __forceinline__ void gemv_phase(const Plan& p, const GemvShared& gs, float4 (&rows)[MT][RB], float (&stage)[MT][64], int lane, int w, int nw, Pre&& pre) {
  const int units = p.rows / R;
  const bool active = w < units / ((p.triple ? 6 : 2) / R);
  const int nk = units_of_wave<R>(p.triple, units, w, nw);
  // Scale group of column c is c / 128 = q + 4 i, stored at q * sq + i. The pieces of lane l are groups l / 8 and 8 + l / 8.
  // The last band of a 7680-column matrix has 3 quarters of its columns: lanes past them read their first piece again, and
  // their scale is 0.
  const uint32_t off_w0 = 16u * lane, off_w1 = (p.i3 == 3 || lane < 32) ? off_w0 + 1024u : off_w0;
  const uint32_t off_s0 = 2u * (((lane >> 3) & 3) * p.sq + (lane >> 5)), off_s1 = off_s0 + 4u;
  BufR<R> buf[NB];
  bufr_undef<R, NB>(buf);
  // Units of a group are consecutive, the next group is G nw - (G - 1) units on. Past the last unit a load re-reads it.
  const int G = (p.triple ? 6 : 2) / R, jump = G * nw - (G - 1);
  const uint32_t rowb = (uint32_t)p.rowbytes, srowb = 2u * (uint32_t)p.srow;
  const uint32_t step1w = (uint32_t)R * rowb, step1s = (uint32_t)R * srowb;
  const uint32_t stepjw = (uint32_t)jump * step1w, stepjs = (uint32_t)jump * step1s;
  const int u0 = unit_at<R>(p.triple, 0, w, nw);
  Cursor cur = {reinterpret_cast<uint64_t>(p.W) + (uint64_t)(uint32_t)u0 * step1w + 512u * (uint32_t)p.i0,
                reinterpret_cast<uint64_t>(p.S) + (uint64_t)(uint32_t)u0 * step1s + 2u * (uint32_t)p.i0, nk, 0};
  auto advance = [&]() {
    const int gn = cur.gc + 1;
    const bool wrap = gn == G, more = cur.left > 1;
    cur.wp += more ? (uint64_t)(wrap ? stepjw : step1w) : 0u;
    cur.sp += more ? (uint64_t)(wrap ? stepjs : step1s) : 0u;
    cur.gc = wrap ? 0 : gn;
    cur.left -= 1;
  };
  if (active) {
#pragma unroll
    for (int s = 0; s < NB - 1; ++s) {
      bufr_load_at<R>(buf[s], cur, rowb, srowb, off_w0, off_w1, off_s0, off_s1);
      advance();
    }
  }
  __builtin_amdgcn_sched_barrier(0);
  pre();
  if (!active) return;
  XRegs2<M> xr;
  xregs2_load<M>(xr, gs, lane);
  // Everything issued so far is complete here. Telling the waitcnt pass keeps its merged state at the loop head exact; otherwise the
  // first slot of every iteration can get a full vmcnt(0), which throws away the units in flight.
  __builtin_amdgcn_s_waitcnt(0x70);
  const bool row_head = (lane & 15) == 0;
  const int q = lane >> 4;
  auto flush = [&](int idx0, int n) {
    asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
    if (lane < n) {
      const int idx = idx0 + lane;
      if (p.staged) {
#pragma unroll
        for (int t = 0; t < M; ++t) {
          const float4 r = rows[t][lane];
          stage[t][idx] = (r.x + r.y) + (r.z + r.w);
        }
      } else {
        // Only conv rows come in groups of 6, and those are staged.
        const int row = 2 * (w + (idx >> 1) * nw) + (idx & 1);
#pragma unroll
        for (int t = 0; t < M; ++t) {
          const float4 r = rows[t][lane];
          p.out[(int64_t)t * p.stride + row] = (r.x + r.y) + (r.z + r.w);
        }
      }
    }
    __builtin_amdgcn_sched_barrier(0);
  };
  auto slot = [&](auto sc, int k) {
    constexpr int s = decltype(sc)::value;
    bufr_load_at<R>(buf[(s + NB - 1) % NB], cur, rowb, srowb, off_w0, off_w1, off_s0, off_s1);
    advance();
    float acc[R][M];
    unit_dot_r<M, R>(buf[s], xr, acc);
    row_sum16_batch<R * M>(&acc[0][0]);
    const int kr = (k * R) & (RB - 1);
    if (row_head) {
#pragma unroll
      for (int r = 0; r < R; ++r)
#pragma unroll
        for (int t = 0; t < M; ++t) reinterpret_cast<float*>(&rows[t][kr + r])[q] = acc[r][t];
    }
    __builtin_amdgcn_sched_barrier(0);
    if ((((k + 1) * R) & (RB - 1)) == 0) flush((k + 1) * R - RB, RB);
  };
  int k = 0;
  for (;;) {
    slot(IntC<0>{}, k);
    if (++k >= nk) break;
    slot(IntC<1>{}, k);
    if (++k >= nk) break;
    if constexpr (NB > 2) {
      slot(IntC<2>{}, k);
      if (++k >= nk) break;
    }
    if constexpr (NB > 3) {
      slot(IntC<3>{}, k);
      if (++k >= nk) break;
    }
    if constexpr (NB > 4) {
      slot(IntC<4>{}, k);
      if (++k >= nk) break;
    }
    if constexpr (NB > 5) {
      slot(IntC<5>{}, k);
      if (++k >= nk) break;
    }
  }
  const int tail = (nk * R) & (RB - 1);
  if (tail) flush(nk * R - tail, tail);
}

// ---------------------------------------------------------------- norms
// h_new = h_old + a pw v; x = rms-normalized(h_new) * nw. Block t computes token t once (fp16 x to x16, h_new to dst and
// tap); after the phase barrier every block copies x16 to LDS, so the phase reads each vector once instead of 60 times.
// a = 1 / rms(v) with sandwich post norm; a = 1 and pw = 1 without (post_w == nullptr and v set).
__device__ __forceinline__ void norm_compute(const float* h_old, const float* v, int bands, int vstride, const __half* post_w,
                                             const __half* norm_w, bool sandwich, float* dst, float* tap, __half* x16,
                                             float4 (&red)[WAVES][MT], int m) {
  const int t = blockIdx.x;
  if (t >= m) return;
  const int tid = tidx(), i0 = 8 * tid, w_id = tid >> 6;
  // One 16-byte load per array instead of 16 fp16 loads: every lane of a sub-dword load is one L1 access.
  const u32x4 nw4 = gload(reinterpret_cast<const u32x4*>(norm_w + i0));
  u32x4 pw4 = {0, 0, 0, 0};
  if (post_w) pw4 = gload(reinterpret_cast<const u32x4*>(post_w + i0));
  f32x4 h4[2], v4[NUM_TAPS][2];
#pragma unroll
  for (int bd = 0; bd < NUM_TAPS; ++bd) {
    LFM3_UNDEF(v4[bd][0]);
    LFM3_UNDEF(v4[bd][1]);
  }
  if (h_old) {
    const f32x4* hp = reinterpret_cast<const f32x4*>(h_old + t * H + i0);
    h4[0] = gload(hp);
    h4[1] = gload(hp + 1);
  } else {
    h4[0] = (f32x4){0.f, 0.f, 0.f, 0.f};
    h4[1] = h4[0];
  }
#pragma unroll
  for (int bd = 0; bd < NUM_TAPS; ++bd) {
    if (bd < bands) {
      const f32x4* vp = reinterpret_cast<const f32x4*>(v + (size_t)bd * vstride + t * H + i0);
      v4[bd][0] = gload(vp);
      v4[bd][1] = gload(vp + 1);
    }
  }
  __builtin_amdgcn_sched_barrier(0);
  float hv[8] = {h4[0].x, h4[0].y, h4[0].z, h4[0].w, h4[1].x, h4[1].y, h4[1].z, h4[1].w};
  float vv[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
#pragma unroll
  for (int bd = 0; bd < NUM_TAPS; ++bd) {
    if (bd < bands) {
      vv[0] += v4[bd][0].x; vv[1] += v4[bd][0].y; vv[2] += v4[bd][0].z; vv[3] += v4[bd][0].w;
      vv[4] += v4[bd][1].x; vv[5] += v4[bd][1].y; vv[6] += v4[bd][1].z; vv[7] += v4[bd][1].w;
    }
  }
  float pw[8], nw[8];
  const uint32_t nwd[4] = {nw4.x, nw4.y, nw4.z, nw4.w}, pwd[4] = {pw4.x, pw4.y, pw4.z, pw4.w};
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    const half2_t n2 = as_h2(nwd[k]);
    nw[2 * k] = (float)n2.x;
    nw[2 * k + 1] = (float)n2.y;
    if (post_w) {
      const half2_t p2 = as_h2(pwd[k]);
      pw[2 * k] = (float)p2.x;
      pw[2 * k + 1] = (float)p2.y;
    } else {
      pw[2 * k] = 1.f;
      pw[2 * k + 1] = 1.f;
    }
  }
  float sums[4] = {0.f, 0.f, 0.f, 0.f};  // hh, vv, hv, ww
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    const float vw = vv[k] * pw[k];
    sums[0] = fmaf(hv[k], hv[k], sums[0]);
    sums[1] = fmaf(vv[k], vv[k], sums[1]);
    sums[2] = fmaf(hv[k], vw, sums[2]);
    sums[3] = fmaf(vw, vw, sums[3]);
  }
  row_sum16_batch<4>(sums);
  float wt[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) wt[j] = (LFM3_LANE(sums[j], 0) + LFM3_LANE(sums[j], 16)) + (LFM3_LANE(sums[j], 32) + LFM3_LANE(sums[j], 48));
  if ((tid & 63) == 0) red[w_id][0] = make_float4(wt[0], wt[1], wt[2], wt[3]);
  lds_barrier();
  const float4 r0 = red[0][0], r1 = red[1][0], r2 = red[2][0], r3 = red[3][0];
  const float t_hh = (r0.x + r1.x) + (r2.x + r3.x), t_vv = (r0.y + r1.y) + (r2.y + r3.y);
  const float t_hv = (r0.z + r1.z) + (r2.z + r3.z), t_ww = (r0.w + r1.w) + (r2.w + r3.w);
  const float a = bands == 0 ? 0.f : (sandwich ? rsqrtf(t_vv / H + 1e-5f) : 1.f);
  const float hh = t_hh + 2.f * a * t_hv + a * a * t_ww;
  const float inv = rsqrtf(fmaxf(hh, 0.f) / H + 1e-5f);
  float hn[8];
#ifdef LFM3_X8
  // x as int8 + one scale per 16 columns, once here instead of once per block.
  float xf[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    hn[k] = fmaf(vv[k] * a, pw[k], hv[k]);
    xf[k] = hn[k] * inv * nw[k];
  }
  float xm = fabsf(xf[0]);
#pragma unroll
  for (int k = 1; k < 8; ++k) xm = fmaxf(xm, fabsf(xf[k]));
  xm = fmaxf(xm, LFM3_DPP(xm, 0xb1));
  const float xsc = fmaxf(xm / 127.f, 1e-20f), xinv = __builtin_amdgcn_rcpf(xsc);
  uint32_t qlo = 0, qhi = 0;
#pragma unroll
  for (int k = 0; k < 4; ++k) {
    qlo |= ((uint32_t)(int)rintf(xf[k] * xinv) & 255u) << (8 * k);
    qhi |= ((uint32_t)(int)rintf(xf[k + 4] * xinv) & 255u) << (8 * k);
  }
  int8_t* x8g = reinterpret_cast<int8_t*>(x16);
  *reinterpret_cast<u32x2*>(x8g + t * H + 8 * tid) = (u32x2){qlo, qhi};
  reinterpret_cast<float*>(x8g + MT * H)[t * (H / 16) + (tid >> 1)] = xsc;
#else
  __half xh[8];
#pragma unroll
  for (int k = 0; k < 8; ++k) {
    hn[k] = fmaf(vv[k] * a, pw[k], hv[k]);
    xh[k] = __float2half_rn(hn[k] * inv * nw[k]);
  }
  *reinterpret_cast<uint4*>(x16 + t * H + i0) = *reinterpret_cast<const uint4*>(xh);
#endif
  if (dst) {
    float4* d4 = reinterpret_cast<float4*>(dst + t * H + i0);
    d4[0] = make_float4(hn[0], hn[1], hn[2], hn[3]);
    d4[1] = make_float4(hn[4], hn[5], hn[6], hn[7]);
  }
  if (tap) {
    float4* d4 = reinterpret_cast<float4*>(tap + t * H + i0);
    d4[0] = make_float4(hn[0], hn[1], hn[2], hn[3]);
    d4[1] = make_float4(hn[4], hn[5], hn[6], hn[7]);
  }
}

#ifdef LFM3_X8
// x already quantized by the norm phase -> LDS.
__device__ __forceinline__ void x_fill8(const __half* x16, GemvShared& gs, int m) {
  const int tid = tidx();
  const int8_t* x8g = reinterpret_cast<const int8_t*>(x16);
  const float* xsg = reinterpret_cast<const float*>(x8g + MT * H);
  u32x2 v[MT];
  float sc[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    LFM3_UNDEF(v[t]);
    LFM3_UNDEF(sc[t]);
  }
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    if (t < m) {
      v[t] = gload(reinterpret_cast<const u32x2*>(x8g + t * H + 8 * tid));
      sc[t] = gload(xsg + t * (H / 16) + (tid >> 1));
    }
  }
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    if (t < m) {
      *reinterpret_cast<u32x2*>(gs.x8 + t * H + 8 * tid) = v[t];
      gs.xs[t][tid >> 1] = sc[t];
    }
  }
  lds_barrier();
}

// x[t] = src[t * stride + offset .. + 2048) fp16 -> int8 + one scale per 16 columns in LDS. All loads go out before the first is used.
__device__ __forceinline__ void x_fill(const __half* src, int stride, int offset, GemvShared& gs, int m) {
  const int tid = tidx();
  u32x4 v[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t) LFM3_UNDEF(v[t]);
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    if (t < m) v[t] = gload(reinterpret_cast<const u32x4*>(src + (size_t)t * stride + offset + 8 * tid));
  }
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    if (t < m) {
      const half2_t h0 = as_h2(v[t].x), h1 = as_h2(v[t].y), h2 = as_h2(v[t].z), h3 = as_h2(v[t].w);
      const float f[8] = {(float)h0.x, (float)h0.y, (float)h1.x, (float)h1.y, (float)h2.x, (float)h2.y, (float)h3.x, (float)h3.y};
      float mx = fabsf(f[0]);
#pragma unroll
      for (int k = 1; k < 8; ++k) mx = fmaxf(mx, fabsf(f[k]));
      mx = fmaxf(mx, LFM3_DPP(mx, 0xb1));
      const float sc = fmaxf(mx / 127.f, 1e-20f), inv = __builtin_amdgcn_rcpf(sc);
      uint32_t lo = 0, hi = 0;
#pragma unroll
      for (int k = 0; k < 4; ++k) {
        lo |= ((uint32_t)(int)rintf(f[k] * inv) & 255u) << (8 * k);
        hi |= ((uint32_t)(int)rintf(f[k + 4] * inv) & 255u) << (8 * k);
      }
      *reinterpret_cast<u32x2*>(gs.x8 + t * H + 8 * tid) = (u32x2){lo, hi};
      gs.xs[t][tid >> 1] = sc;
    }
  }
  lds_barrier();
}
#else
// x[t] = src[t * stride + offset .. + 2048) fp16 -> LDS. All loads go out before the first is used.
__device__ __forceinline__ void x_fill(const __half* src, int stride, int offset, GemvShared& gs, int m) {
  const int tid = tidx();
  u32x4 v[MT];
#pragma unroll
  for (int t = 0; t < MT; ++t) LFM3_UNDEF(v[t]);
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    if (t < m) v[t] = gload(reinterpret_cast<const u32x4*>(src + (size_t)t * stride + offset + 8 * tid));
  }
  __builtin_amdgcn_sched_barrier(0);
#pragma unroll
  for (int t = 0; t < MT; ++t) {
    if (t < m) *reinterpret_cast<u32x4*>(gs.x + t * H + 8 * tid) = v[t];
  }
  lds_barrier();
}

#endif

// ---------------------------------------------------------------- attention
__device__ __forceinline__ float xor_add(float v, int mask) { return v + __shfl_xor(v, mask); }

__device__ __forceinline__ float quantize_pair(float v0, float v1, int8_t* out) {
  float amax = wave_max(fmaxf(fabsf(v0), fabsf(v1)));
  float scale = fmaxf(amax / 127.f, 1e-6f);
  float inv = 1.f / scale;
  int lane = tidx() & 63;
  float a = v0 * inv, b = v1 * inv;
  a = a >= 0 ? a + 0.5f : a - 0.5f;
  b = b >= 0 ? b + 0.5f : b - 0.5f;
  out[lane] = (int8_t)fminf(fmaxf(truncf(a), -128.f), 127.f);
  out[lane + 64] = (int8_t)fminf(fmaxf(truncf(b), -128.f), 127.f);
  return scale;
}

__device__ __forceinline__ void norm_rope(float& v0, float& v1, float w0, float w1, float c, float s) {
  const float inv = rsqrtf(wave_total(v0 * v0 + v1 * v1) / D + 1e-5f);
  const float a = v0 * inv * w0, b = v1 * inv * w1;
  v0 = a * c - b * s;
  v1 = b * c + a * s;
}

// Raw-buffer DMA of one tile into LDS: memory image, no VGPRs held while it is in flight.
__device__ __forceinline__ void tile_dma(const uint8_t* src, uint8_t* tile) {
  const uint64_t addr = (uint64_t)uniform64((int64_t)src);
  const DirectLdsResource res = {static_cast<int32_t>(addr), static_cast<int32_t>(addr >> 32), TILE_BYTES, 0x00020000};
  const int wave = __builtin_amdgcn_readfirstlane(tidx() >> 6);
  const uint32_t base = (uint32_t)reinterpret_cast<uintptr_t>(tile);
  for (int chunk = 0; chunk * THREADS + wave * 64 < TILE_DWORDS; ++chunk) {
    const int idx = chunk * THREADS + (int)tidx();
    const uint32_t dest = __builtin_amdgcn_readfirstlane(base + 4u * (chunk * THREADS + wave * 64));
    if (idx < TILE_DWORDS)
      direct_lds_raw_load(res, reinterpret_cast<DirectLdsPointer>(dest), 4, idx * 4, 0, 0, 0);
  }
}

__device__ __forceinline__ uint64_t argmax_key(float v, int idx) {
  const uint32_t u = __float_as_uint(v);
  const uint32_t ord = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ((uint64_t)ord << 32) | (uint32_t)(0xffffffffu - (uint32_t)idx);
}

__device__ __forceinline__ uint64_t umax64(uint64_t a, uint64_t b) { return a > b ? a : b; }

__device__ __forceinline__ int attn_start(int kind, int win, int a, int ctx_lo) {
  if (kind == 3) return max(ctx_lo, a - win);
  return win > 0 ? max(0, a - win + 1) : 0;
}

// exp2 for x <= 0 (P of a softmax): v_ldexp + degree 4 polynomial; accuracy ~1e-4 is plenty for fp16 P.
__device__ __forceinline__ float hw_exp2(float x) {
  float r;
  asm("v_exp_f32 %0, %1" : "=v"(r) : "v"(x));
  return r;
}

__device__ __forceinline__ float fast_exp2(float x) {
  const float fl = floorf(x);
  const float f = x - fl;
  float p = fmaf(f, 0.0135557f, 0.0520323f);
  p = fmaf(f, p, 0.2413793f);
  p = fmaf(f, p, 0.6930886f);
  p = fmaf(f, p, 1.0000026f);
  return __builtin_ldexpf(p, (int)fmaxf(fl, -150.f));
}

// Token t's query (int8 per 32-dim group) and fresh K/V, once per layer, block t: wave = kv head. Everything is staged in this block's
// LDS and leaves in 16-byte pieces, one load and one store per wave instead of hundreds of byte stores.
__device__ __forceinline__ void query_fresh(const Layer& L, const float* mix, int8_t* qg, AttnShared& sm, int t, int a, int wave, int lane) {
  const float* mx = mix + (size_t)t * MIXS;
  const __half* c = L.rope + (int64_t)(a + t) * L.rotary;
  const half2_t cs2 = {c[lane], c[64 + lane]};
  const float qw0 = __half2float(L.q_norm[lane]), qw1 = __half2float(L.q_norm[lane + 64]);
  const float kw0 = __half2float(L.k_norm[lane]), kw1 = __half2float(L.k_norm[lane + 64]);
  float v0[4], v1[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    v0[j] = gload(mx + (wave * 4 + j) * D + lane);
    v1[j] = gload(mx + (wave * 4 + j) * D + lane + 64);
  }
  float k0 = gload(mx + 2048 + wave * D + lane), k1 = gload(mx + 2048 + wave * D + lane + 64);
  float vk0 = gload(mx + 2560 + wave * D + lane), vk1 = gload(mx + 2560 + wave * D + lane + 64);
  __builtin_amdgcn_sched_barrier(0);
  const float cs = (float)cs2.x, sn = (float)cs2.y;
  float sq[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) sq[j] = v0[j] * v0[j] + v1[j] * v1[j];
  row_sum16_batch<4>(sq);
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float inv = rsqrtf(((LFM3_LANE(sq[j], 0) + LFM3_LANE(sq[j], 16)) + (LFM3_LANE(sq[j], 32) + LFM3_LANE(sq[j], 48))) / D + 1e-5f);
    const float a0 = v0[j] * inv * qw0, b0 = v1[j] * inv * qw1;
    v0[j] = a0 * cs - b0 * sn;
    v1[j] = b0 * cs + a0 * sn;
  }
  float m0[4], m1[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) { m0[j] = fabsf(v0[j]); m1[j] = fabsf(v1[j]); }
  row_max16_batch<4>(m0);
  row_max16_batch<4>(m1);
  const bool hi = (lane & 32) != 0;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float lo0 = fmaxf(LFM3_LANE(m0[j], 0), LFM3_LANE(m0[j], 16)), hi0 = fmaxf(LFM3_LANE(m0[j], 32), LFM3_LANE(m0[j], 48));
    const float lo1 = fmaxf(LFM3_LANE(m1[j], 0), LFM3_LANE(m1[j], 16)), hi1 = fmaxf(LFM3_LANE(m1[j], 32), LFM3_LANE(m1[j], 48));
    const float s0 = fmaxf((hi ? hi0 : lo0) / 127.f, 1e-20f), s1 = fmaxf((hi ? hi1 : lo1) / 127.f, 1e-20f);
    sm.q8[0][wave * 4 + j][lane] = (int8_t)rintf(v0[j] * __builtin_amdgcn_rcpf(s0));
    sm.q8[0][wave * 4 + j][lane + 64] = (int8_t)rintf(v1[j] * __builtin_amdgcn_rcpf(s1));
    if ((lane & 31) == 0) {
      sm.qs8[0][wave * 4 + j][lane >> 5] = s0;
      sm.qs8[0][wave * 4 + j][2 + (lane >> 5)] = s1;
    }
  }
  norm_rope(k0, k1, kw0, kw1, cs, sn);
  uint8_t* rec_lds = reinterpret_cast<uint8_t*>(sm.q8[1][wave * 4]);
  const float kss = quantize_pair(k0, k1, reinterpret_cast<int8_t*>(rec_lds));
  const float vss = quantize_pair(vk0, vk1, reinterpret_cast<int8_t*>(rec_lds + 132));
  if (lane == 0) {
    *reinterpret_cast<float*>(rec_lds + 128) = kss;
    *reinterpret_cast<float*>(rec_lds + 260) = vss;
  }
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_wave_barrier();
  // LDS -> global: 512 bytes of query (lanes < 32), 64 bytes of scales (lanes < 16), one 264-byte record per wave.
  int8_t* qt = qg + (size_t)t * QH * D + wave * 4 * D;
  float* st = reinterpret_cast<float*>(qg + (size_t)MT * QH * D) + (size_t)t * QH * 4 + wave * 16;
  uint8_t* rec = L.fresh + (size_t)(t * KVH + wave) * REC;
  if (lane < 32) *reinterpret_cast<u32x4*>(qt + 16 * lane) = *reinterpret_cast<const u32x4*>(sm.q8[0][wave * 4] + 16 * lane);
  if (lane < 16) st[lane] = reinterpret_cast<const float*>(sm.qs8[0][wave * 4])[lane];
  if (lane < 33) *reinterpret_cast<u32x2*>(rec + 8 * lane) = *reinterpret_cast<const u32x2*>(rec_lds + 8 * lane);
}

// One pass = up to 4 tokens over all context keys, plus (split 0) the verified tokens' fresh K/V.
// Wave = kv head. Lane (tr = lane >> 4, col = lane & 15). QK: lane holds its token's 4 query heads x dims
// [64 (col & 1), +64) of key (col >> 1) of an 8-key tile; both halves of a key sum over one DPP step. PV:
// lane accumulates 4 query heads x dims 8 col .. +7; P pairs rotate over the DPP row so no LDS is touched.
// Every lane-byte of a load crosses the CU's ~16 B/clk return path, and rows of different tokens want the same keys: K and V
// of one iteration arrive once per wave (one 16-byte load per 8 keys, the next iteration's loads in flight during the compute),
// go to this wave's LDS tile, and rows read them from there.
template <int R, bool RING, typename Pre>
__device__ __forceinline__ void attention_pass(const Layer& L, const float* mix, float* part, AttnShared& sm, int t0_pass, int nq, int m,
                                               int a, int ctx_lo, bool causal, int64_t* trace, int& tn, Pre&& pre) {
  constexpr int TK = 8;                                  // keys of one row of lanes per iteration
  constexpr int IT = TK * R;                             // keys per iteration (contiguous: IT divides the page size)
  const int bs = uniform_i32((int)L.block_size);
  const int win = uniform_i32((int)L.window);
  const int kind = RING ? 3 : 1;
  const bool ring = RING;
  const int start = attn_start(kind, win, a, ctx_lo);
  const int g_begin = start / IT;
  const int g_total = a > start ? (a - 1) / IT - g_begin + 1 : 0;
  const int splits = max(1, min(g_total, (int)gridDim.x));
  const int split = blockIdx.x;
  if (split >= splits) {
    pre();
    return;
  }
  const int wave = __builtin_amdgcn_readfirstlane(tidx() >> 6), lane = tidx() & 63;
  const int tr = lane >> 4, col = lane & 15, keyl = col >> 1, dh = col & 1;
  const int rt = R == 4 ? tr : (R == 2 ? tr >> 1 : 0);                    // this row's keys among its token's rows
  const int tq = R == 4 ? 0 : (R == 2 ? tr & 1 : min(tr, nq - 1));        // this row's token within the pass
  const int keyr = keyl + TK * rt;
  // The fresh keys and block 0's fresh K/V count as the first tiles of the partition, so block 0 gets that many context tiles less.
  const int ft = (m + IT - 1) / IT, gt = g_total + ft;
  const int g0 = g_begin + max((int)((int64_t)gt * split / splits) - ft, 0), g1 = g_begin + max((int)((int64_t)gt * (split + 1) / splits) - ft, 0);
  const int b0 = g0 * IT / bs;
  const int nb = g0 < g1 ? min((g1 * IT - 1) / bs - b0 + 1, BT_MAX) : 0;
  const uint8_t* kv = reinterpret_cast<const uint8_t*>(uniform64(reinterpret_cast<int64_t>(L.kv)));
  const int64_t kv_block = uniform64(L.kv_block), kv_slot = uniform64(L.kv_slot);
  const uint8_t* fresh_base = reinterpret_cast<const uint8_t*>(uniform64(reinterpret_cast<int64_t>(L.fresh)));
  auto source = [&](int g) -> const uint8_t* {
    const int t0 = g * IT;
    int64_t blk;
    if constexpr (RING) blk = (int64_t)((t0 / bs) & CTX_MASK);
    else blk = (int64_t)uniform_i32(sm.blocks[t0 / bs - b0]);
    return kv + blk * kv_block + (t0 % bs) * kv_slot;
  };
  constexpr int TW = 98 * IT;                            // tile dwords per wave
  uint32_t* T = reinterpret_cast<uint32_t*>(sm.q8[R == 4 ? 1 : (R == 2 ? 2 : 4)]) + wave * TW;
  u32x4 nk[R];
  u32x2 nv[R][2];
  float nks, nvs;
#pragma unroll
  for (int c = 0; c < R; ++c) {
    LFM3_UNDEF(nk[c]);
    LFM3_UNDEF(nv[c][0]);
    LFM3_UNDEF(nv[c][1]);
  }
  LFM3_UNDEF(nks);
  LFM3_UNDEF(nvs);
  // Tile loads: K: lane l owns key (l >> 3) + 8 c, 16-byte piece l & 7. V: lane l owns dims 8 (l & 15) .. +7 of the key pair
  // 2 (l >> 4), 2 (l >> 4) + 1 of group c. V is expanded once here, not once per row of tokens.
  const uint32_t off_k = (uint32_t)(((lane >> 3) * KVH + wave) * REC + 16 * (lane & 7));
  const uint32_t off_v = (uint32_t)((2 * (lane >> 4) * KVH + wave) * REC + 132 + 8 * col);
  const uint32_t off_scale = (uint32_t)(((lane & (IT - 1)) * KVH + wave) * REC);
  auto issue = [&](const uint8_t* tile) {
#pragma unroll
    for (int c = 0; c < R; ++c) {
      const uint8_t* tc = tile + (size_t)c * TK * KVH * REC;
      nk[c] = gload_nt(reinterpret_cast<const u32x4*>(tc + off_k));
      nv[c][0] = gload_nt(reinterpret_cast<const u32x2*>(tc + off_v));
      nv[c][1] = gload_nt(reinterpret_cast<const u32x2*>(tc + off_v + KVH * REC));
    }
    nks = gload(reinterpret_cast<const float*>(tile + off_scale + 128));
    nvs = gload(reinterpret_cast<const float*>(tile + off_scale + 260));
    __builtin_amdgcn_sched_barrier(0);
  };
  const uint32_t c64 = 0x64646464u;
  const half2_t bias = {(_Float16)1152.f, (_Float16)1152.f};
  auto stage = [&]() {
#pragma unroll
    for (int c = 0; c < R; ++c) {
      *reinterpret_cast<u32x4*>(T + 32 * ((lane >> 3) + TK * c) + 4 * (lane & 7)) = nk[c];
      const uint32_t a0 = nv[c][0].x ^ 0x80808080u, a1 = nv[c][0].y ^ 0x80808080u, b0_ = nv[c][1].x ^ 0x80808080u, b1_ = nv[c][1].y ^ 0x80808080u;
      const uint32_t i0 = __builtin_amdgcn_perm(b0_, a0, 0x05010400u), i1 = __builtin_amdgcn_perm(b0_, a0, 0x07030602u);
      const uint32_t i2 = __builtin_amdgcn_perm(b1_, a1, 0x05010400u), i3 = __builtin_amdgcn_perm(b1_, a1, 0x07030602u);
      const half2_t vd[8] = {as_h2(__builtin_amdgcn_perm(c64, i0, 0x04010400u)) - bias,
                             as_h2(__builtin_amdgcn_perm(c64, i0, 0x04030402u)) - bias,
                             as_h2(__builtin_amdgcn_perm(c64, i1, 0x04010400u)) - bias,
                             as_h2(__builtin_amdgcn_perm(c64, i1, 0x04030402u)) - bias,
                             as_h2(__builtin_amdgcn_perm(c64, i2, 0x04010400u)) - bias,
                             as_h2(__builtin_amdgcn_perm(c64, i2, 0x04030402u)) - bias,
                             as_h2(__builtin_amdgcn_perm(c64, i3, 0x04010400u)) - bias,
                             as_h2(__builtin_amdgcn_perm(c64, i3, 0x04030402u)) - bias};
      uint32_t* vout = T + 32 * IT + 128 * (TK / 2 * c + (lane >> 4)) + 8 * col;
      reinterpret_cast<half2_t*>(vout)[0] = vd[0]; reinterpret_cast<half2_t*>(vout)[1] = vd[1];
      reinterpret_cast<half2_t*>(vout)[2] = vd[2]; reinterpret_cast<half2_t*>(vout)[3] = vd[3];
      reinterpret_cast<half2_t*>(vout)[4] = vd[4]; reinterpret_cast<half2_t*>(vout)[5] = vd[5];
      reinterpret_cast<half2_t*>(vout)[6] = vd[6]; reinterpret_cast<half2_t*>(vout)[7] = vd[7];
    }
    reinterpret_cast<float*>(T)[96 * IT + (lane & (IT - 1))] = nks;
    reinterpret_cast<float*>(T)[97 * IT + (lane & (IT - 1))] = nvs;
    __builtin_amdgcn_sched_barrier(0);
  };
  const int gfresh = g1;                       // pseudo tile index of the fresh keys
  const bool ctx_any = g0 < g1;
  auto tile_of = [&](int g, const uint8_t*& tile) {
    if (g >= gfresh) tile = fresh_base + (size_t)(g - gfresh) * IT * KVH * REC;
    else tile = source(g);
  };
  if constexpr (!RING)
    for (int i = tidx(); i < nb; i += THREADS) sm.blocks[i] = gload(L.block_table + b0 + i);
  if (ctx_any) {
    const int t0 = g0 * IT;
    int64_t blk;
    if constexpr (RING) blk = (int64_t)((t0 / bs) & CTX_MASK);
    else blk = (int64_t)uniform_i32(sload(L.block_table + t0 / bs));
    issue(kv + blk * kv_block + (t0 % bs) * kv_slot);
  }
  pre();
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_wave_barrier();
  trace_mark(trace, tn, 20);

  // This kv head's 4 query heads of the pass's tokens, from the query phase, into LDS.
  {
    const int8_t* qg = reinterpret_cast<const int8_t*>(part) + (size_t)(WS_Q - WS_PART) * 4;
    const float* qsg = reinterpret_cast<const float*>(qg + (size_t)MT * QH * D);
    u32x4 qv[4];
    float qsv[4];
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      LFM3_UNDEF(qv[n]);
      LFM3_UNDEF(qsv[n]);
    }
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      if (n < nq) {
        const int t = t0_pass + n;
        qv[n] = gload(reinterpret_cast<const u32x4*>(qg + (size_t)t * QH * D + wave * 4 * D + 16 * (lane & 31)));
        qsv[n] = gload(qsg + (size_t)t * QH * 4 + wave * 16 + (lane & 15));
      }
    }
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int n = 0; n < 4; ++n) {
      if (n < nq) {
        if (lane < 32) *reinterpret_cast<u32x4*>(sm.q8[n][wave * 4] + 16 * lane) = qv[n];
        if (lane < 16) reinterpret_cast<float*>(sm.qs8[n][wave * 4])[lane] = qsv[n];
      }
    }
  }
  trace_mark(trace, tn, 21);
  // Fresh K/V of the verified tokens: block 0 writes them for the final tiles and the commit.
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_wave_barrier();
  trace_mark(trace, tn, 22);

  const float qk_scale = 0.08838834764831845f * LOG2E;
  uint32_t qreg[4][16];
  float qsc[4][2];
  {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const u32x4* src = reinterpret_cast<const u32x4*>(sm.q8[tq][wave * 4 + j] + 64 * dh);
#pragma unroll
      for (int rd = 0; rd < 4; ++rd) {
        const u32x4 q = src[rd];
        qreg[j][rd * 4] = q.x;
        qreg[j][rd * 4 + 1] = q.y;
        qreg[j][rd * 4 + 2] = q.z;
        qreg[j][rd * 4 + 3] = q.w;
      }
      qsc[j][0] = sm.qs8[tq][wave * 4 + j][2 * dh] * qk_scale;
      qsc[j][1] = sm.qs8[tq][wave * 4 + j][2 * dh + 1] * qk_scale;
    }
  }

  float m_run[4], l_run[4], acc[4][8];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    m_run[j] = -1e30f;     // not -inf: an invalid key's score is -inf, and -inf - -inf would be NaN below
    l_run[j] = 0.f;
#pragma unroll
    for (int e = 0; e < 8; ++e) acc[j][e] = 0.f;
  }
  const int tok = t0_pass + tq;
  const bool lane_tok = R > 1 || tr < nq;
  const float pen0 = lane_tok ? 0.f : -INFINITY;
  const uint32_t* v_lds[TK / 2];
#pragma unroll
  for (int prr = 0; prr < TK / 2; ++prr) v_lds[prr] = T + 32 * IT + 128 * (TK / 2 * rt + (((col >> 2) + prr) & 3)) + 8 * col;
  trace_mark(trace, tn, 23);

  const int ctx_iters = g1 - g0;
  const int fresh_tiles = split == 0 ? (m + IT - 1) / IT : 0;
  const int total = ctx_iters + fresh_tiles;
  if (ctx_iters == 0 && total > 0) issue(fresh_base);
  // Inside a page the next tile is contiguous; a table lookup only at a page boundary or at the fresh keys.
  const uint8_t* tile_next = nullptr;
  if (ctx_iters > 0) tile_of(g0, tile_next);
  const int tiles_page = max(bs / IT, 1);
  int page_pos = (g0 + 1) % tiles_page;
  // Iterations in [it_a, it_b) have all their keys valid. Everything per-iteration is a few scalar adds: at long context this loop is
  // compute-limited and every instruction counts.
  const int t_min = (win != 0 && !ring) ? max(start, a + t0_pass + nq - win + 1) : start;
  const int it_a = (t_min + IT - 1) / IT - g0, it_b = min(a / IT - g0, ctx_iters);
  for (int it = 0; it < total; ++it) {
    stage();
    {
      if (__builtin_expect(it + 1 >= ctx_iters || page_pos == 0, 0)) tile_of(g0 + min(it + 1, total - 1), tile_next);
      else tile_next += (size_t)IT * KVH * REC;
      page_pos = page_pos + 1 == tiles_page ? 0 : page_pos + 1;
      issue(tile_next);
    }
    // Declared here, not outside the loop: an uninitialised aggregate outside gets a phi at the loop head and stays live across
    // the interpreter's back edge, 18 VGPRs that end up reloaded from scratch after every phase.
    struct { u32x4 k[4]; float ks, vs; } kb;
#pragma unroll
    for (int rd = 0; rd < 4; ++rd) kb.k[rd] = *reinterpret_cast<const u32x4*>(T + keyr * 32 + 16 * dh + 4 * rd);
    kb.ks = reinterpret_cast<const float*>(T)[96 * IT + keyr];
    kb.vs = reinterpret_cast<const float*>(T)[97 * IT + keyr];
    float s[4] = {0.f, 0.f, 0.f, 0.f};
#pragma unroll
    for (int g = 0; g < 2; ++g) {
      int ai[4] = {0, 0, 0, 0};
#pragma unroll
      for (int rd = 0; rd < 2; ++rd) {
        const u32x4 kk = kb.k[2 * g + rd];
        const uint32_t kw[4] = {kk.x, kk.y, kk.z, kk.w};
#pragma unroll
        for (int j = 0; j < 4; ++j)
#pragma unroll
          for (int c = 0; c < 4; ++c) ai[j] = __builtin_amdgcn_sdot4(kw[c], qreg[j][8 * g + 4 * rd + c], ai[j], false);
      }
#pragma unroll
      for (int j = 0; j < 4; ++j) s[j] = fmaf((float)ai[j], qsc[j][g], s[j]);
    }
    // Both halves of a key: lanes col and col ^ 1.
#pragma unroll
    for (int j = 0; j < 4; ++j) s[j] += LFM3_DPP(s[j], 0xb1);
    // Nearly every tile is valid for all its keys: per-lane checks only at the ends of the range, otherwise -inf from pen0 alone.
    float pen = pen0;
    if (__builtin_expect(it < it_a || it >= it_b, 0)) {
      const bool fresh = it >= ctx_iters;
      const int t0 = (g0 + it) * IT;
      const int fresh_base_idx = (it - ctx_iters) * IT;
      const int qpos = a + tok;
      const int kpos = fresh ? a + fresh_base_idx + keyr : t0 + keyr;
      bool valid;
      if (fresh) valid = fresh_base_idx + keyr < m && (!causal || fresh_base_idx + keyr <= tok);
      else valid = kpos >= start && kpos < a && (win == 0 || ring || kpos > qpos - win);
      pen = valid ? pen0 : -INFINITY;
    }
    // The next V pair is always on its way from LDS during the current compute: 4 waves share the LDS port.
    u32x4 vc0 = *reinterpret_cast<const u32x4*>(v_lds[0]), vc1 = *(reinterpret_cast<const u32x4*>(v_lds[0]) + 1);
    __builtin_amdgcn_sched_barrier(0);
    uint32_t pp[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      const float sc = fmaf(s[j], kb.ks, pen);
      const float mo = m_run[j];
      float mn = mo;
      // The running max rarely moves: the row max and the rescale only run when some lane exceeded it.
      // (Uniform branch: the DPP reduction below needs every lane active.)
      if (__builtin_expect(__builtin_amdgcn_uicmp(sc > mo ? 1u : 0u, 0u, 33) != 0, 0)) {
        // The running max is per token: rows of one token share it so their sums can simply be added at the end.
        const float rm = row_max16(sc);
        if constexpr (R == 1) {
          mn = fmaxf(mo, rm);
        } else {
          const float a0 = LFM3_LANE(rm, 0), a1 = LFM3_LANE(rm, 16), a2 = LFM3_LANE(rm, 32), a3 = LFM3_LANE(rm, 48);
          const float tm = R == 4 ? fmaxf(fmaxf(a0, a1), fmaxf(a2, a3)) : ((tr & 1) ? fmaxf(a1, a3) : fmaxf(a0, a2));
          mn = fmaxf(mo, tm);
        }
        if (mn != mo) {
          const float corr = mo == -1e30f ? 0.f : fast_exp2(mo - mn);
          l_run[j] *= corr;
#pragma unroll
          for (int e = 0; e < 8; ++e) acc[j][e] *= corr;
        }
      }
      const float pr = hw_exp2(sc - mn);
      l_run[j] += pr;
      m_run[j] = mn;
      const __half ph = __float2half_rn(pr * kb.vs * PSCALE);
      const uint32_t p0 = (uint32_t)__half_as_ushort(ph);
      // (P_2p, P_2p+1): the 4 lanes of a quad hold the 2 keys of pair p, so a quad swizzle packs them.
      pp[j] = (uint32_t)__builtin_amdgcn_mov_dpp((int)p0, 0x44, 0xf, 0xf, false) | ((uint32_t)__builtin_amdgcn_mov_dpp((int)p0, 0xee, 0xf, 0xf, false) << 16);
    }
#pragma unroll
    for (int prr = 0; prr < TK / 2; ++prr) {
      u32x4 vn0 = vc0, vn1 = vc1;
      if (prr + 1 < TK / 2) {
        vn0 = *reinterpret_cast<const u32x4*>(v_lds[prr + 1]);
        vn1 = *(reinterpret_cast<const u32x4*>(v_lds[prr + 1]) + 1);
      }
      __builtin_amdgcn_sched_barrier(0);
      const uint32_t vd[8] = {vc0.x, vc0.y, vc0.z, vc0.w, vc1.x, vc1.y, vc1.z, vc1.w};
#pragma unroll
      for (int j = 0; j < 4; ++j) {
        const half2_t ppj = as_h2(pp[j]);
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[j][d] = __builtin_amdgcn_fdot2(as_h2(vd[d]), ppj, acc[j][d], false);
        pp[j] = (uint32_t)__builtin_amdgcn_mov_dpp((int)pp[j], 0x120 + 12, 0xf, 0xf, false);
      }
      __builtin_amdgcn_sched_barrier(0);
      vc0 = vn0;
      vc1 = vn1;
    }
    __builtin_amdgcn_sched_barrier(0);
  }
  trace_mark(trace, tn, 24);
  // Partials: lanes write dims 8 col .. +7 of their token's 4 query heads; col 0 writes m and l.
  float l_tot[4], m_tot[4];
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    const float lr = row_sum16(l_run[j]);
    if constexpr (R == 1) {
      l_tot[j] = 0.5f * lr;
    } else {
      const float a0 = LFM3_LANE(lr, 0), a1 = LFM3_LANE(lr, 16), a2 = LFM3_LANE(lr, 32), a3 = LFM3_LANE(lr, 48);
      l_tot[j] = 0.5f * (R == 4 ? (a0 + a1) + (a2 + a3) : ((tr & 1) ? a1 + a3 : a0 + a2));
    }
    m_tot[j] = row_max16(m_run[j]);
  }
  if constexpr (R > 1) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[j][e] = xor_add(acc[j][e], 32);
  }
  if constexpr (R == 4) {
#pragma unroll
    for (int j = 0; j < 4; ++j)
#pragma unroll
      for (int e = 0; e < 8; ++e) acc[j][e] = xor_add(acc[j][e], 16);
  }
  if (lane_tok && rt == 0) {
#pragma unroll
    for (int j = 0; j < 4; ++j) {
      float* dst = part + ((size_t)(tok * QH + wave * 4 + j) * SPLITS + split) * (D + 2);
      float4* d4 = reinterpret_cast<float4*>(dst + 8 * col);
      d4[0] = make_float4(acc[j][0], acc[j][1], acc[j][2], acc[j][3]) * (1.f / PSCALE);
      d4[1] = make_float4(acc[j][4], acc[j][5], acc[j][6], acc[j][7]) * (1.f / PSCALE);
      if (col == 0) {
        dst[D] = m_tot[j];
        dst[D + 1] = l_tot[j];
      }
    }
  }
}

// ---------------------------------------------------------------- verify attention, page-tiled (v2)
// One pass = NT <= 4 of the verified tokens over every context page plus (split 0) the fresh keys, exactly the arithmetic of
// attention_pass: int8 Q (per 32-dim group) x int8 K in int32, fp32 scores and an exact running max, P = fp16(p x v_scale x
// PSCALE), PV with v_dot2_f32_f16 into fp32.
// Wave = kv head h. QK lanes (j = lane >> 4: query head 4 h + j, kk = (lane >> 1) & 7: key of an 8-key sub-tile, dh = lane & 1:
// dims [64 dh, +64)) hold every pass token's query in registers, so the 64 lanes stay busy for any NT. PV lanes (j, col =
// lane & 15: dims 8 col .. +7) accumulate all NT tokens of head j, so V is read once per sub-tile and reused NT times; P goes
// through LDS (one b16 store per score, one b128 load per token and sub-tile). One iteration = one 32-key KV page, the next
// page's K/V in registers meanwhile.
__device__ u32x4 vraw_b128(DirectLdsResource rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v4i32");
__device__ u32x2 vraw_b64(DirectLdsResource rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.v2i32");
__device__ float vraw_f32(DirectLdsResource rsrc, int voffset, int soffset, int aux) __asm("llvm.amdgcn.raw.buffer.load.f32");
__device__ __forceinline__ DirectLdsResource vrsrc(const void* base) {
  const uint64_t addr = (uint64_t)uniform64((int64_t)base);
  return DirectLdsResource{(int32_t)(uint32_t)addr, (int32_t)(uint32_t)(addr >> 32), -1, 0x00020000};
}
// DPP with bound_ctrl, which lets the compiler fold the move into the VALU op that uses it (v_add_f32_dpp, v_cvt_f32_i32_dpp).
#define V3_DPP(v, ctrl) __builtin_bit_cast(float, __builtin_amdgcn_mov_dpp(__builtin_bit_cast(int, v), ctrl, 0xf, 0xf, true))
#define V3_DPPI(v, ctrl) __builtin_amdgcn_mov_dpp(v, ctrl, 0xf, 0xf, true)
// dst = fp16 pair (signed byte B of a, signed byte B of b): two SDWA converts, the second keeps the low half.
#define V3_CVT_PAIR(dst, a, b, B)                                                                                          \
  do {                                                                                                                     \
    asm("v_cvt_f16_i16_sdwa %0, sext(%1) dst_sel:WORD_0 dst_unused:UNUSED_PAD src0_sel:BYTE_" #B : "=v"(dst) : "v"(a));   \
    asm("v_cvt_f16_i16_sdwa %0, sext(%1) dst_sel:WORD_1 dst_unused:UNUSED_PRESERVE src0_sel:BYTE_" #B : "+v"(dst) : "v"(b)); \
  } while (0)

constexpr int VP = 32;                        // keys per iteration: one KV page
constexpr int KSTR = 36;                      // K row stride (dwords): keys 4 banks apart, so the 8 lanes of a b128 phase hit 32 banks
constexpr int VK_DW = VP * KSTR;              // K: [32 keys][128 bytes + pad]
constexpr int VV_DW = (VP / 2) * 128;         // V: fp16 pairs of adjacent keys, [16 pairs][2 halves][16 col][4 dims]: conflict-free b128
constexpr int VS_DW = 2 * VP;                 // k scale [32], v scale x PSCALE [32]
constexpr int KV_SLOT = KVH * REC;
template <int NT>
constexpr int vwave_dw() {
  return VK_DW + VV_DW + VS_DW + NT * 4 * VP / 2;
}
static_assert(4 * vwave_dw<4>() * 4 <= (int)sizeof(AttnShared), "verify attention tiles exceed the attention LDS");

__device__ __forceinline__ float vadd_x1(float a) {
  float r;
  asm volatile("s_nop 1\n\tv_add_f32_dpp %0, %1, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf" : "=v"(r) : "v"(a));
  return r;
}
// a + (b of the lane ^ 1)
__device__ __forceinline__ float vadd_x1b(float a, float b) {
  float r;
  asm volatile("s_nop 1\n\tv_add_f32_dpp %0, %2, %1 quad_perm:[1,0,3,2] row_mask:0xf bank_mask:0xf" : "=v"(r) : "v"(a), "v"(b));
  return r;
}
// Max / sum over the 8 lanes of one parity in a 16-lane row (lanes 2 k + p, k = 0..7).
__device__ __forceinline__ float max8p(float v) {
  v = fmaxf(v, LFM3_DPP(v, 0x4e));   // quad_perm [2,3,0,1]: k ^ 1
  v = fmaxf(v, LFM3_DPP(v, 0x124));  // row_ror 4: k - 2
  return fmaxf(v, LFM3_DPP(v, 0x128));  // row_ror 8: k - 4
}
__device__ __forceinline__ float sum8p(float v) {
  v += LFM3_DPP(v, 0x4e);
  v += LFM3_DPP(v, 0x124);
  return v + LFM3_DPP(v, 0x128);
}

// Whether a target layer's verify attention runs on the page partition, one call over all m tokens (attention_v2 / v3), or as passes
// of up to 4 tokens of attention_pass (8 R-key tiles): short key ranges (sliding layers, short contexts) keep attention_pass's finer
// partition, which spreads them over more blocks.
__device__ __forceinline__ bool use_paged(int start, int a, int m) {
  const int pt = a > start ? (a - 1) / VP - start / VP + 1 : 0;
  return m <= 8 && (m <= 2 || pt >= 12);
}

__device__ __forceinline__ int verify_splits(int start, int a) {
  const int pt = a > start ? (a - 1) / VP - start / VP + 1 : 0;
  return max(1, min(pt, (int)gridDim.x));
}

template <int NT, typename Pre, int PF = 1>
__device__ __forceinline__ void attention_v2(const Layer& L, float* part, AttnShared& sm, int t0_pass, int m, int a, Pre&& pre) {
  const int win = uniform_i32((int)L.window);
  const int kind = uniform_i32((int)L.kind);
  const int start = attn_start(kind, win, a, 0);
  const int p_begin = start / VP;
  const int p_total = a > start ? (a - 1) / VP - p_begin + 1 : 0;
  const int splits = max(1, min(p_total, (int)gridDim.x));
  const int split = blockIdx.x;
  if (split >= splits) {
    pre();
    return;
  }
  // The fresh keys count as the first page of the partition, so block 0 gets one context page less.
  const int gt = p_total + 1;
  const int g0 = p_begin + max((int)((int64_t)gt * split / splits) - 1, 0);
  const int g1 = p_begin + max((int)((int64_t)gt * (split + 1) / splits) - 1, 0);
  const int nctx = g1 - g0;
  const int total = nctx + (split == 0 ? 1 : 0);
  const int wave = __builtin_amdgcn_readfirstlane(tidx() >> 6), lane = tidx() & 63;
  const int j = lane >> 4, c16 = lane & 15, kk = c16 >> 1, dh = c16 & 1;

  const uint8_t* kv = reinterpret_cast<const uint8_t*>(uniform64(reinterpret_cast<int64_t>(L.kv)));
  const int64_t kv_block = uniform64(L.kv_block);
  const uint8_t* fresh_base = reinterpret_cast<const uint8_t*>(uniform64(reinterpret_cast<int64_t>(L.fresh)));
  const __attribute__((address_space(4))) int32_t* btab =
      (const __attribute__((address_space(4))) int32_t*)(uintptr_t)uniform64(reinterpret_cast<int64_t>(L.block_table));

  // Loads of one page. K: lane l owns 16-byte piece l & 7 of key (l >> 3) + 8 c. V: dims 8 (l & 15) .. +7 of keys 2 (l >> 4) + 8 c
  // and the next one. Scales: lane l the k (l < 32) or v (l >= 32) scale of key l & 31.
  const uint32_t off_k = (uint32_t)((lane >> 3) * KV_SLOT + wave * REC + 16 * (lane & 7));
  const uint32_t off_v = (uint32_t)(2 * (lane >> 4) * KV_SLOT + wave * REC + 132 + 8 * c16);
  const uint32_t off_s = (uint32_t)((lane & 31) * KV_SLOT + wave * REC + (lane < 32 ? 128 : 260));
  const float s_mul = lane < 32 ? 1.f : PSCALE;
  u32x4 nk[PF][4];
  u32x2 nv[PF][4][2];
  float ns[PF];
  // Buffer loads: the page offset sits in soffset (SGPR), the lane offset in voffset: no per-load address math.
  const DirectLdsResource rs_kv = vrsrc(kv), rs_fresh = vrsrc(fresh_base);
  auto issue = [&](int b, bool fresh, int soff) {
    const DirectLdsResource r = fresh ? rs_fresh : rs_kv;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      nk[b][c] = vraw_b128(r, (int)off_k, soff + c * 8 * KV_SLOT, 0);
      nv[b][c][0] = vraw_b64(r, (int)off_v, soff + c * 8 * KV_SLOT, 0);
      nv[b][c][1] = vraw_b64(r, (int)off_v, soff + c * 8 * KV_SLOT + KV_SLOT, 0);
    }
    ns[b] = vraw_f32(r, (int)off_s, soff, 0);
  };

  uint32_t* W = reinterpret_cast<uint32_t*>(&sm) + wave * vwave_dw<NT>();
  uint32_t* WK = W;
  uint32_t* WV = W + VK_DW;
  float* WS = reinterpret_cast<float*>(W + VK_DW + VV_DW);
  __half* WP = reinterpret_cast<__half*>(W + VK_DW + VV_DW + VS_DW);
  auto stage = [&](int b) {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      *reinterpret_cast<u32x4*>(WK + ((lane >> 3) + 8 * c) * KSTR + 4 * (lane & 7)) = nk[b][c];
      // V as fp16 pairs of adjacent keys: one SDWA convert per half (signed byte -> fp16).
      uint32_t vw[8];
      V3_CVT_PAIR(vw[0], nv[b][c][0].x, nv[b][c][1].x, 0);
      V3_CVT_PAIR(vw[1], nv[b][c][0].x, nv[b][c][1].x, 1);
      V3_CVT_PAIR(vw[2], nv[b][c][0].x, nv[b][c][1].x, 2);
      V3_CVT_PAIR(vw[3], nv[b][c][0].x, nv[b][c][1].x, 3);
      V3_CVT_PAIR(vw[4], nv[b][c][0].y, nv[b][c][1].y, 0);
      V3_CVT_PAIR(vw[5], nv[b][c][0].y, nv[b][c][1].y, 1);
      V3_CVT_PAIR(vw[6], nv[b][c][0].y, nv[b][c][1].y, 2);
      V3_CVT_PAIR(vw[7], nv[b][c][0].y, nv[b][c][1].y, 3);
      uint32_t* vo = WV + (4 * c + (lane >> 4)) * 128 + 4 * c16;
      *reinterpret_cast<u32x4*>(vo) = u32x4{vw[0], vw[1], vw[2], vw[3]};
      *reinterpret_cast<u32x4*>(vo + 64) = u32x4{vw[4], vw[5], vw[6], vw[7]};
    }
    WS[lane] = ns[b] * s_mul;
  };

  const int kvb = uniform_i32((int)L.kv_block);
  auto blk_of = [&](int q) { return uniform_i32(btab[g0 + min(q, max(nctx - 1, 0))]); };
  auto issue_page = [&](int b, int q, int blk) { issue(b, q >= nctx, q >= nctx ? 0 : blk * kvb); };
  // A fresh page (split 0 without context pages) holds K/V that other blocks wrote in the phase just before: after the barrier.
  if (nctx >= PF) {
#pragma unroll
    for (int b = 0; b < PF; ++b) issue_page(b, b, blk_of(b));
  }
  pre();
  if (nctx < PF) {
#pragma unroll
    for (int b = 0; b < PF; ++b) issue_page(b, b, blk_of(b));
  }

  // This lane's query: dims [64 dh, +64) of head 4 wave + j for every pass token (int8, 16 dwords) and its two group scales.
  const float qk_scale = 0.08838834764831845f * LOG2E;
  // Even NT: the two lanes of a key split the tokens after the QK sum (lane dh owns tokens [H dh, H dh + H)); lane dh = 1 holds
  // its queries rotated by H, so one DPP add per owned token completes the score (the other half comes from register r + H).
  constexpr bool DEDUP = NT % 2 == 0;
  constexpr int H = DEDUP ? NT / 2 : NT;
  auto qtok = [&](int r) { return DEDUP && dh ? (r + H) % NT : r; };  // token held in query register r
  auto otok = [&](int r) { return DEDUP ? r + H * dh : r; };           // token of owned row r
  uint32_t qreg[NT][16];
  float qsc[NT][2];
  {
    const int8_t* qg = reinterpret_cast<const int8_t*>(part) + (size_t)(WS_Q - WS_PART) * 4;
    const float* qsg = reinterpret_cast<const float*>(qg + (size_t)MT * QH * D);
#pragma unroll
    for (int t = 0; t < NT; ++t) {
      const int tok = t0_pass + qtok(t);
      const u32x4* src = reinterpret_cast<const u32x4*>(qg + ((size_t)tok * QH + wave * 4 + j) * D + 64 * dh);
#pragma unroll
      for (int r = 0; r < 4; ++r) {
        const u32x4 q = gload(src + r);
        qreg[t][4 * r] = q.x;
        qreg[t][4 * r + 1] = q.y;
        qreg[t][4 * r + 2] = q.z;
        qreg[t][4 * r + 3] = q.w;
      }
      qsc[t][0] = gload(qsg + ((size_t)tok * QH + wave * 4 + j) * 4 + 2 * dh) * qk_scale;
      qsc[t][1] = gload(qsg + ((size_t)tok * QH + wave * 4 + j) * 4 + 2 * dh + 1) * qk_scale;
    }
  }

  // The query loads (and with them the first pages) complete here, not at their first use inside the loop: a pending load at
  // the loop entry would make every iteration wait for the newest page.
  __builtin_amdgcn_s_waitcnt(0x0f70);  // vmcnt(0)
  float m_run[H], l_run[H], acc[NT][8];
#pragma unroll
  for (int r = 0; r < H; ++r) {
    m_run[r] = -1e30f;  // not -inf: an invalid key's score is -inf, and -inf - -inf would be NaN
    l_run[r] = 0.f;
  }
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int d = 0; d < 8; ++d) acc[t][d] = 0.f;
  // Pages whose 32 keys are valid for every pass token: no per-key checks.
  const int last_tok = t0_pass + NT - 1;
  const int kmin_all = (win != 0) ? max(start, a + last_tok - win + 1) : start;
  const int it_a = (kmin_all + VP - 1) / VP - g0;   // first page fully at or above kmin_all
  const int it_b = min(a / VP - g0, nctx);         // pages below this one end before a

  // Block id of the page PF iterations ahead, loaded (scalar) one iteration before its use.
  int blk_n = blk_of(PF);
  auto body = [&](int it, int b) {
    stage(b);
    issue_page(b, it + PF, blk_n);
    blk_n = blk_of(it + PF + 1);
    __builtin_amdgcn_sched_barrier(0);
    const bool edge = it < it_a || it >= it_b;
    // Per sub-tile: QK (the K half of one 32-dim group pair in flight while the other computes), the running max, P to LDS.
    uint32_t ka[8], kb[8];
    auto khalf = [&](uint32_t (&dst)[8], int c, int h) {
      const u32x4* src = reinterpret_cast<const u32x4*>(WK + (8 * c + kk) * KSTR + 16 * dh + 8 * h);
      const u32x4 x = src[0], y = src[1];
      dst[0] = x.x; dst[1] = x.y; dst[2] = x.z; dst[3] = x.w;
      dst[4] = y.x; dst[5] = y.y; dst[6] = y.z; dst[7] = y.w;
    };
    khalf(ka, 0, 0);
    khalf(kb, 0, 1);
    // The whole page's scores first: a running max that moves inside the page must also rescale P already written.
    float sc[4][H], vsv[4];
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      const float ks = WS[8 * c + kk];
      vsv[c] = WS[VP + 8 * c + kk];
      __builtin_amdgcn_sched_barrier(0);
      int ai0[NT], ai1[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        ai0[t] = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) ai0[t] = __builtin_amdgcn_sdot4(ka[w], qreg[t][w], ai0[t], false);
      }
      if (c + 1 < 4) khalf(ka, c + 1, 0);
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        ai1[t] = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) ai1[t] = __builtin_amdgcn_sdot4(kb[w], qreg[t][8 + w], ai1[t], false);
      }
      if (c + 1 < 4) khalf(kb, c + 1, 1);
      __builtin_amdgcn_sched_barrier(0);
      float sv[NT];
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        sv[t] = fmaf((float)ai0[t], qsc[t][0], 0.f);
        sv[t] = fmaf((float)ai1[t], qsc[t][1], sv[t]);
      }
#pragma unroll
      for (int r = 0; r < H; ++r) sc[c][r] = (DEDUP ? vadd_x1b(sv[r], sv[r + H]) : vadd_x1(sv[r])) * ks;
    }
    if (__builtin_expect(edge, 0)) {
      const bool fresh = it >= nctx;
#pragma unroll
      for (int c = 0; c < 4; ++c)
#pragma unroll
        for (int r = 0; r < H; ++r) {
          const int tok = t0_pass + otok(r), qpos = a + tok, fk = 8 * c + kk;
          bool valid;
          if (fresh) valid = fk < m && fk <= tok;
          else {
            const int kpos = (g0 + it) * VP + fk;
            valid = kpos >= start && kpos < a && (win == 0 || kpos > qpos - win);
          }
          sc[c][r] = valid ? sc[c][r] : -INFINITY;
        }
    }
    // Running max: only when some lane's score beats its row's max (uniform branch: the row max needs every lane).
    float mx[H];
    uint64_t beat = 0;
#pragma unroll
    for (int r = 0; r < H; ++r) {
      mx[r] = fmaxf(fmaxf(sc[0][r], sc[1][r]), fmaxf(sc[2][r], sc[3][r]));
      beat |= __builtin_amdgcn_uicmp(mx[r] > m_run[r] ? 1u : 0u, 0u, 33);
    }
    if (__builtin_expect(beat != 0, 0)) {
      float corr[H];
#pragma unroll
      for (int r = 0; r < H; ++r) {
        const float mn = fmaxf(m_run[r], DEDUP ? max8p(mx[r]) : row_max16(mx[r]));
        corr[r] = hw_exp2(m_run[r] - mn);
        m_run[r] = mn;
        l_run[r] *= corr[r];
      }
      // PV lanes hold every token of head j: token t's factor comes from a lane of its owner parity in the same quad.
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        float ct = corr[t % H];
        if (DEDUP) ct = t < H ? LFM3_DPP(ct, 0x00) : LFM3_DPP(ct, 0x55);
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[t][d] *= ct;
      }
    }
    // All exponents back to back (quarter-rate ops, independent), then their uses: interleaved, every use would wait for its exp.
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int r = 0; r < H; ++r) sc[c][r] = sc[c][r] - m_run[r];
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int r = 0; r < H; ++r) sc[c][r] = hw_exp2(sc[c][r]);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int c = 0; c < 4; ++c)
#pragma unroll
      for (int r = 0; r < H; ++r) {
        l_run[r] += sc[c][r];
        WP[(otok(r) * 4 + j) * VP + 8 * c + kk] = __float2half_rn(sc[c][r] * vsv[c]);
      }
    __builtin_amdgcn_sched_barrier(0);
    // PV: dims 8 col .. +7 of head 4 wave + j for every token over the 16 key pairs. The next pair's V and the next P half
    // are on their way from LDS during each step (one wave's LDS ops complete in order: P is read after it was written).
    uint32_t pc[NT][2], pn[NT][2];
    constexpr int VDIST = 1, VB = VDIST + 1;
    u32x4 vb[VB][2];
    auto pload = [&](uint32_t (&dst)[NT][2], int c, int h) {
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const u32x2 x = *reinterpret_cast<const u32x2*>(reinterpret_cast<const uint32_t*>(WP) + (t * 4 + j) * (VP / 2) + 4 * c + 2 * h);
        dst[t][0] = x.x;
        dst[t][1] = x.y;
      }
    };
    auto vload = [&](int s, int b) {
      const uint32_t* vp = WV + s * 128 + 4 * c16;
      vb[b][0] = *reinterpret_cast<const u32x4*>(vp);
      vb[b][1] = *reinterpret_cast<const u32x4*>(vp + 64);
    };
    pload(pc, 0, 0);
#pragma unroll
    for (int s = 0; s < VDIST; ++s) vload(s, s % VB);
#pragma unroll
    for (int s = 0; s < 16; ++s) {
      if (s + VDIST < 16) vload(s + VDIST, (s + VDIST) % VB);
      // pairs (0, 1) of a sub-tile use pc, (2, 3) pn; each half is requested one step before its first use
      if ((s & 1) == 0 && s + 1 < 16) {
        if ((s & 3) == 0) pload(pn, s >> 2, 1);
        else pload(pc, (s >> 2) + 1, 0);
      }
      __builtin_amdgcn_sched_barrier(0);
      const uint32_t vd[8] = {vb[s % VB][0].x, vb[s % VB][0].y, vb[s % VB][0].z, vb[s % VB][0].w,
                              vb[s % VB][1].x, vb[s % VB][1].y, vb[s % VB][1].z, vb[s % VB][1].w};
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const half2_t pp = as_h2((s & 2) ? pn[t][s & 1] : pc[t][s & 1]);
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[t][d] = __builtin_amdgcn_fdot2(as_h2(vd[d]), pp, acc[t][d], false);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
  };
  for (int it = 0; it < total; it += PF) {
    body(it, 0);
    if (PF > 1 && it + 1 < total) body(it + 1, 1 % PF);
  }

  // Partials: lane (j, col) writes dims 8 col .. +7 of every token's head 4 wave + j; col 0 adds m and l (both lanes of a key
  // summed the same p, hence the half).
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    const int tok = t0_pass + t;
    float* dst = part + ((size_t)(tok * QH + wave * 4 + j) * SPLITS + split) * (D + 2);
    float4* d4 = reinterpret_cast<float4*>(dst + 8 * c16);
    d4[0] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]) * (1.f / PSCALE);
    d4[1] = make_float4(acc[t][4], acc[t][5], acc[t][6], acc[t][7]) * (1.f / PSCALE);
  }
  // m and l of the owned rows: lane (j, kk = 0, dh) for dedup (each p summed once), lane col 0 otherwise (both lanes of a key
  // summed the same p, hence the half).
#pragma unroll
  for (int r = 0; r < H; ++r) {
    const float lt = DEDUP ? sum8p(l_run[r]) : 0.5f * row_sum16(l_run[r]);
    const int tok = t0_pass + otok(r);
    float* dst = part + ((size_t)(tok * QH + wave * 4 + j) * SPLITS + split) * (D + 2);
    if (DEDUP ? c16 < 2 : c16 == 0) {
      dst[D] = m_run[r];
      dst[D + 1] = lt;
    }
  }
}

// ---------------------------------------------------------------- verify attention, one K/V pass for up to 8 tokens (v3)
// attention_v2's partition, page loads, staging and partials, but every verified token in one pass, so K/V cross the memory bus
// once per layer whatever M is. QK lanes: j = lane >> 4 (query head 4 wave + j), kk = (lane >> 2) & 3 (key 4 c + kk of sub-tile c,
// 8 sub-tiles per page), dq = lane & 3 (dims [32 dq, +32): one query scale group). A lane keeps 8 dwords of query per token, so 8
// tokens take 64 VGPRs; register r holds token (r & ~3) | ((r & 3) ^ dq), which lets the 4 lanes of a key reduce-scatter their
// partial scores in 3 DPP ops per 4 tokens: lane dq then owns tokens dq and 4 + dq of its key. PV lanes (j, col = lane & 15)
// accumulate dims 8 col .. +7 of every token of head j as in v2; P comes through LDS in the K tile (K is dead by then).
template <int NT>
constexpr int v3_pj(int j) {  // P tile offset of head j (dwords): the 32 dwords of one P store land in 32 banks
  return NT == 8 ? j * 128 + (j & 1) * 4 + (j >> 1) * 16 : j * 72;
}
constexpr int V3_DW = VK_DW + VV_DW + VS_DW;
static_assert(4 * V3_DW * 4 <= (int)sizeof(AttnShared), "verify attention v3 tiles exceed the attention LDS");
static_assert(v3_pj<8>(3) + 16 * 8 <= VK_DW && v3_pj<4>(3) + 16 * 4 <= VK_DW, "P tile exceeds the K tile");

template <int NT, typename Pre>
__device__ __forceinline__ void attention_v3(const Layer& L, float* part, AttnShared& sm, int m, int a, Pre&& pre) {
  static_assert(NT == 4 || NT == 8, "v3 runs 4 or 8 token slots");
  constexpr int G = NT / 4;  // owned tokens per lane and key
  const int win = uniform_i32((int)L.window);
  const int kind = uniform_i32((int)L.kind);
  const int start = attn_start(kind, win, a, 0);
  const int p_begin = start / VP;
  const int p_total = a > start ? (a - 1) / VP - p_begin + 1 : 0;
  const int splits = max(1, min(p_total, (int)gridDim.x));
  const int split = blockIdx.x;
  if (split >= splits) {
    pre();
    return;
  }
  const int gt = p_total + 1;
  const int g0 = p_begin + max((int)((int64_t)gt * split / splits) - 1, 0);
  const int g1 = p_begin + max((int)((int64_t)gt * (split + 1) / splits) - 1, 0);
  const int nctx = g1 - g0;
  const int total = nctx + (split == 0 ? 1 : 0);
  const int wave = __builtin_amdgcn_readfirstlane(tidx() >> 6), lane = tidx() & 63;
  const int j = lane >> 4, c16 = lane & 15, kk = (lane >> 2) & 3, dq = lane & 3;

  const uint8_t* kv = reinterpret_cast<const uint8_t*>(uniform64(reinterpret_cast<int64_t>(L.kv)));
  const uint8_t* fresh_base = reinterpret_cast<const uint8_t*>(uniform64(reinterpret_cast<int64_t>(L.fresh)));
  const int32_t* btab = reinterpret_cast<const int32_t*>(uniform64(reinterpret_cast<int64_t>(L.block_table)));

  const uint32_t off_k = (uint32_t)((lane >> 3) * KV_SLOT + wave * REC + 16 * (lane & 7));
  const uint32_t off_v = (uint32_t)(2 * (lane >> 4) * KV_SLOT + wave * REC + 132 + 8 * c16);
  const uint32_t off_s = (uint32_t)((lane & 31) * KV_SLOT + wave * REC + (lane < 32 ? 128 : 260));
  const float s_mul = lane < 32 ? 1.f : PSCALE;
  u32x4 nk[4];
  u32x2 nv[4][2];
  float ns;
  const DirectLdsResource rs_kv = vrsrc(kv), rs_fresh = vrsrc(fresh_base);
  auto issue = [&](bool fresh, int soff) {
    const DirectLdsResource r = fresh ? rs_fresh : rs_kv;
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      nk[c] = vraw_b128(r, (int)off_k, soff + c * 8 * KV_SLOT, 0);
      nv[c][0] = vraw_b64(r, (int)off_v, soff + c * 8 * KV_SLOT, 0);
      nv[c][1] = vraw_b64(r, (int)off_v, soff + c * 8 * KV_SLOT + KV_SLOT, 0);
    }
    ns = vraw_f32(r, (int)off_s, soff, 0);
  };

  uint32_t* W = reinterpret_cast<uint32_t*>(&sm) + wave * V3_DW;
  uint32_t* WK = W;
  uint32_t* WV = W + VK_DW;
  float* WS = reinterpret_cast<float*>(W + VK_DW + VV_DW);
  __half* WP = reinterpret_cast<__half*>(W);
  // K and the scales first: the QK's first reads queue behind these stores only, and the V conversion covers their latency.
  auto stage_k = [&]() {
#pragma unroll
    for (int c = 0; c < 4; ++c) *reinterpret_cast<u32x4*>(WK + ((lane >> 3) + 8 * c) * KSTR + 4 * (lane & 7)) = nk[c];
    // Scales by (kk, c) of their key 4 c + kk, so a QK lane finds its 8 keys' scales together: k at [8 kk + c], v at [32 + 8 kk + c].
    const int key = lane & 31;
    WS[(lane & 32) + 8 * (key & 3) + (key >> 2)] = ns * s_mul;
  };
  // V as fp16 pairs of adjacent keys: one SDWA convert per half (signed byte -> fp16), 16 per 8 keys and 8 dims.
  auto stage_v = [&]() {
#pragma unroll
    for (int c = 0; c < 4; ++c) {
      uint32_t vw[8];
      V3_CVT_PAIR(vw[0], nv[c][0].x, nv[c][1].x, 0);
      V3_CVT_PAIR(vw[1], nv[c][0].x, nv[c][1].x, 1);
      V3_CVT_PAIR(vw[2], nv[c][0].x, nv[c][1].x, 2);
      V3_CVT_PAIR(vw[3], nv[c][0].x, nv[c][1].x, 3);
      V3_CVT_PAIR(vw[4], nv[c][0].y, nv[c][1].y, 0);
      V3_CVT_PAIR(vw[5], nv[c][0].y, nv[c][1].y, 1);
      V3_CVT_PAIR(vw[6], nv[c][0].y, nv[c][1].y, 2);
      V3_CVT_PAIR(vw[7], nv[c][0].y, nv[c][1].y, 3);
      uint32_t* vo = WV + (4 * c + (lane >> 4)) * 128 + 4 * c16;
      *reinterpret_cast<u32x4*>(vo) = u32x4{vw[0], vw[1], vw[2], vw[3]};
      *reinterpret_cast<u32x4*>(vo + 64) = u32x4{vw[4], vw[5], vw[6], vw[7]};
    }
  };

  // Block ids of this split's pages, one per lane, read with v_readlane: no scalar load (and its lgkmcnt(0), which would also
  // drain the LDS stores) inside the loop. A split has at most 69 pages at the 131072-token limit (4097 pages over 60 blocks).
  const int kvb = uniform_i32((int)L.kv_block);
  const int qmax = max(nctx - 1, 0);
  const int bt0 = gload(btab + g0 + min(lane, qmax)), bt1 = gload(btab + g0 + min(64 + lane, qmax));
  auto blk_of = [&](int q) {
    q = min(q, min(qmax, 127));
    return q < 64 ? __builtin_amdgcn_readlane(bt0, q) : __builtin_amdgcn_readlane(bt1, q - 64);
  };
  auto issue_page = [&](int q) { issue(q >= nctx, q >= nctx ? 0 : blk_of(q) * kvb); };

  // Query dims [32 dq, +32) of head 4 wave + j, register r = token (r & ~3) | ((r & 3) ^ dq); tokens from m on get a zero scale.
  const float qk_scale = 0.08838834764831845f * LOG2E;
  auto qtok = [&](int r) { return (r & ~3) | ((r & 3) ^ dq); };
  uint32_t qreg[NT][8];
  float qsc[NT];
  {
    const int8_t* qg = reinterpret_cast<const int8_t*>(part) + (size_t)(WS_Q - WS_PART) * 4;
    const float* qsg = reinterpret_cast<const float*>(qg + (size_t)MT * QH * D);
    // The block ids arrive first (the vmcnt below), then the first page goes out while the query loads.
    // (A fresh page, split 0 without context pages, holds K/V other blocks wrote in the phase just before: after the barrier.)
    __builtin_amdgcn_s_waitcnt(0x0f70);
    if (nctx > 0) issue_page(0);
    pre();
    if (nctx == 0) issue_page(0);
#pragma unroll
    for (int r = 0; r < NT; ++r) {
      const int tok = qtok(r);
      const u32x4* src = reinterpret_cast<const u32x4*>(qg + ((size_t)tok * QH + wave * 4 + j) * D + 32 * dq);
      const u32x4 q0 = gload(src), q1 = gload(src + 1);
      qreg[r][0] = q0.x; qreg[r][1] = q0.y; qreg[r][2] = q0.z; qreg[r][3] = q0.w;
      qreg[r][4] = q1.x; qreg[r][5] = q1.y; qreg[r][6] = q1.z; qreg[r][7] = q1.w;
      const float s = gload(qsg + ((size_t)tok * QH + wave * 4 + j) * 4 + dq) * qk_scale;
      qsc[r] = tok < m ? s : 0.f;
    }
  }
  __builtin_amdgcn_s_waitcnt(0x0f70);  // vmcnt(0): see attention_v2
  // The scale of the partner lane (dq ^ 1) for odd registers: the first reduce step is one v_fmac_f32_dpp per token pair.
  float qsp[NT / 2];
#pragma unroll
  for (int r = 0; r < NT / 2; ++r) qsp[r] = V3_DPP(qsc[2 * r + 1], 0xB1);

  float m_run[G], l_run[G], acc[NT][8];
#pragma unroll
  for (int g = 0; g < G; ++g) {
    m_run[g] = -1e30f;
    l_run[g] = 0.f;
  }
#pragma unroll
  for (int t = 0; t < NT; ++t)
#pragma unroll
    for (int d = 0; d < 8; ++d) acc[t][d] = 0.f;
  const int last_tok = min(m, NT) - 1;
  const int kmin_all = (win != 0) ? max(start, a + last_tok - win + 1) : start;
  const int it_a = (kmin_all + VP - 1) / VP - g0;
  const int it_b = min(a / VP - g0, nctx);
  const uint32_t* WPw = reinterpret_cast<const uint32_t*>(W) + v3_pj<NT>(j);

  for (int it = 0; it < total; ++it) {
    uint32_t kb[2][8];
    float ksc[2], vsc[8];
    auto kload = [&](int c, int b) {
      const u32x4* src = reinterpret_cast<const u32x4*>(WK + (4 * c + kk) * KSTR + 8 * dq);
      const u32x4 x = src[0], y = src[1];
      kb[b][0] = x.x; kb[b][1] = x.y; kb[b][2] = x.z; kb[b][3] = x.w;
      kb[b][4] = y.x; kb[b][5] = y.y; kb[b][6] = y.z; kb[b][7] = y.w;
      ksc[b] = WS[8 * kk + c];
      vsc[c] = WS[32 + 8 * kk + c];
    };
    stage_k();
    kload(0, 0);
    stage_v();
    issue_page(it + 1);
    __builtin_amdgcn_sched_barrier(0);
    const bool edge = it < it_a || it >= it_b;
    // QK over the 8 sub-tiles: this lane's 32 dims of key 4 c + kk against every token, then the reduce-scatter.
    float sc[8][G];
#pragma unroll
    for (int c = 0; c < 8; ++c) {
      if (c + 1 < 8) kload(c + 1, (c + 1) & 1);
      __builtin_amdgcn_sched_barrier(0);
      int si[NT];
#pragma unroll
      for (int r = 0; r < NT; ++r) {
        si[r] = 0;
#pragma unroll
        for (int w = 0; w < 8; ++w) si[r] = __builtin_amdgcn_sdot4(kb[c & 1][w], qreg[r][w], si[r], false);
      }
#pragma unroll
      for (int g = 0; g < G; ++g) {
        // Token pair sums over lanes dq, dq ^ 1 (the partner's integer sum converts on the way in: v_cvt_f32_i32_dpp), then
        // over dq ^ 2 (v_add_f32_dpp).
        const float s0 = fmaf((float)V3_DPPI(si[4 * g + 1], 0xB1), qsp[2 * g], (float)si[4 * g] * qsc[4 * g]);
        const float s1 = fmaf((float)V3_DPPI(si[4 * g + 3], 0xB1), qsp[2 * g + 1], (float)si[4 * g + 2] * qsc[4 * g + 2]);
        sc[c][g] = (s0 + V3_DPP(s1, 0x4E)) * ksc[c & 1];
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (__builtin_expect(edge, 0)) {
      const bool fresh = it >= nctx;
#pragma unroll
      for (int c = 0; c < 8; ++c)
#pragma unroll
        for (int g = 0; g < G; ++g) {
          const int tok = 4 * g + dq, qpos = a + tok, fk = 4 * c + kk;
          bool valid;
          if (fresh) valid = fk < m && fk <= tok;
          else {
            const int kpos = (g0 + it) * VP + fk;
            valid = kpos >= start && kpos < a && (win == 0 || kpos > qpos - win);
          }
          sc[c][g] = valid ? sc[c][g] : -INFINITY;
        }
    }
    float mx[G];
    uint64_t beat = 0;
#pragma unroll
    for (int g = 0; g < G; ++g) {
      mx[g] = fmaxf(fmaxf(fmaxf(sc[0][g], sc[1][g]), fmaxf(sc[2][g], sc[3][g])), fmaxf(fmaxf(sc[4][g], sc[5][g]), fmaxf(sc[6][g], sc[7][g])));
      beat |= __builtin_amdgcn_uicmp(mx[g] > m_run[g] ? 1u : 0u, 0u, 33);
    }
    if (__builtin_expect(beat != 0, 0)) {
      float corr[G];
#pragma unroll
      for (int g = 0; g < G; ++g) {
        float r = fmaxf(mx[g], LFM3_DPP(mx[g], 0x124));  // row_ror 4: kk + 1
        r = fmaxf(r, LFM3_DPP(r, 0x128));                // row_ror 8: kk + 2
        const float mn = fmaxf(m_run[g], r);
        corr[g] = hw_exp2(m_run[g] - mn);
        m_run[g] = mn;
        l_run[g] *= corr[g];
      }
      // PV lanes hold every token of head j: token t's factor is in lane dq = t & 3 of every quad of the row.
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const float cg = corr[t >> 2];
        const float ct = (t & 3) == 0 ? LFM3_DPP(cg, 0x00) : (t & 3) == 1 ? LFM3_DPP(cg, 0x55) : (t & 3) == 2 ? LFM3_DPP(cg, 0xAA) : LFM3_DPP(cg, 0xFF);
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[t][d] *= ct;
      }
    }
    // PV operands, one key pair ahead: V of the pair (8 dims) and P of every token (NT dwords). V of the first pairs is in LDS
    // since the staging, so it goes out before the P stores and its latency hides behind the exponentials.
    constexpr int VB = 2;
    u32x4 vb[VB][2];
    uint32_t pb[VB][NT];
    auto vload = [&](int s, int b) {
      const uint32_t* vp = WV + s * 128 + 4 * c16;
      vb[b][0] = *reinterpret_cast<const u32x4*>(vp);
      vb[b][1] = *reinterpret_cast<const u32x4*>(vp + 64);
    };
    auto pload = [&](int s, int b) {
#pragma unroll
      for (int q = 0; q < NT / 4; ++q) {
        const u32x4 x = *reinterpret_cast<const u32x4*>(WPw + s * NT + 4 * q);
        pb[b][4 * q] = x.x; pb[b][4 * q + 1] = x.y; pb[b][4 * q + 2] = x.z; pb[b][4 * q + 3] = x.w;
      }
    };
    vload(0, 0);
#pragma unroll
    for (int c = 0; c < 8; ++c)
#pragma unroll
      for (int g = 0; g < G; ++g) sc[c][g] = sc[c][g] - m_run[g];
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int c = 0; c < 8; ++c)
#pragma unroll
      for (int g = 0; g < G; ++g) sc[c][g] = hw_exp2(sc[c][g]);
    __builtin_amdgcn_sched_barrier(0);
    // P (fp16, x v scale x PSCALE) of token 4 g + dq, key 4 c + kk: half (kk & 1) of dword pair (2 c + (kk >> 1)) of token row.
#pragma unroll
    for (int c = 0; c < 8; ++c)
#pragma unroll
      for (int g = 0; g < G; ++g) {
        l_run[g] += sc[c][g];
        WP[2 * (v3_pj<NT>(j) + (2 * c + (kk >> 1)) * NT + 4 * g + dq) + (kk & 1)] = __float2half_rn(sc[c][g] * vsc[c]);
      }
    pload(0, 0);
    __builtin_amdgcn_sched_barrier(0);
#pragma unroll
    for (int s = 0; s < 16; ++s) {
      if (s + 1 < 16) {
        vload(s + 1, (s + 1) % VB);
        pload(s + 1, (s + 1) % VB);
      }
      __builtin_amdgcn_sched_barrier(0);
      const uint32_t vd[8] = {vb[s % VB][0].x, vb[s % VB][0].y, vb[s % VB][0].z, vb[s % VB][0].w,
                              vb[s % VB][1].x, vb[s % VB][1].y, vb[s % VB][1].z, vb[s % VB][1].w};
#pragma unroll
      for (int t = 0; t < NT; ++t) {
        const half2_t pp = as_h2(pb[s % VB][t]);
#pragma unroll
        for (int d = 0; d < 8; ++d) acc[t][d] = __builtin_amdgcn_fdot2(as_h2(vd[d]), pp, acc[t][d], false);
      }
      __builtin_amdgcn_sched_barrier(0);
    }
  }

  // Partials of tokens < m: lane (j, col) writes dims 8 col .. +7 of head 4 wave + j; the owner lanes (kk = 0) add m and l.
#pragma unroll
  for (int t = 0; t < NT; ++t) {
    if (t >= m) break;
    float* dst = part + ((size_t)(t * QH + wave * 4 + j) * SPLITS + split) * (D + 2);
    float4* d4 = reinterpret_cast<float4*>(dst + 8 * c16);
    d4[0] = make_float4(acc[t][0], acc[t][1], acc[t][2], acc[t][3]) * (1.f / PSCALE);
    d4[1] = make_float4(acc[t][4], acc[t][5], acc[t][6], acc[t][7]) * (1.f / PSCALE);
  }
#pragma unroll
  for (int g = 0; g < G; ++g) {
    float lt = l_run[g] + LFM3_DPP(l_run[g], 0x124);
    lt = lt + LFM3_DPP(lt, 0x128);
    const int tok = 4 * g + dq;
    float* dst = part + ((size_t)(tok * QH + wave * 4 + j) * SPLITS + split) * (D + 2);
    if (kk == 0 && tok < m) {
      dst[D] = m_run[g];
      dst[D + 1] = lt;
    }
  }
}

// One pass over all m verified tokens on the page partition (m <= 8).
template <typename Pre>
__device__ __forceinline__ void verify_attention_paged(const Layer& L, float* part, AttnShared& sm, int m, int a, Pre&& pre) {
  if (m == 1) attention_v2<1>(L, part, sm, 0, m, a, pre);
  else if (m == 2) attention_v2<2>(L, part, sm, 0, m, a, pre);
  else if (m == 3) attention_v2<3>(L, part, sm, 0, m, a, pre);
  else if (m == 4) attention_v3<4>(L, part, sm, m, a, pre);
  else attention_v3<8>(L, part, sm, m, a, pre);
}

// y[t][qh][d] = gate(t, qh) * (sum_s fac_s acc_s) / (sum_s fac_s l_s). One (t, qh) task per pair of waves, one lane per dim: the two
// pairs of a block run side by side, so 64 tasks over 60 blocks do not double the phase for the 4 blocks that get a second task.
__device__ __forceinline__ void combine_phase(const float* mix, const float* part, __half* y, int start, int a, int m, bool gated,
                                              float4 (&red4)[WAVES][MT]) {
  __shared__ float fac[2][SPLITS];
  float (&red)[WAVES] = reinterpret_cast<float (&)[WAVES]>(red4[0][0]);
  const int wave = __builtin_amdgcn_readfirstlane(tidx() >> 6), pair = wave >> 1;
  const int lane = tidx() & 63, d = lane + 64 * (wave & 1);
  const bool lead = (wave & 1) == 0;
  const int tasks = m * QH, slots = 2 * gridDim.x, slot = 2 * blockIdx.x + pair;
  const int rounds = (tasks + slots - 1) / slots;
  typedef float f32x2 __attribute__((ext_vector_type(2)));
  for (int r = 0; r < rounds; ++r) {
    const int task = slot + r * slots;
    const bool valid = task < tasks;
    const int tok = task / QH, qh = task % QH;
    const float* base = part + (size_t)task * SPLITS * (D + 2);
    int nq = min(m - (tok & ~3), 4);
    const int IT = gated ? 8 * (nq == 1 ? 4 : (nq == 2 ? 2 : 1)) : 8;
    const int splits = gated && use_paged(start, a, m) ? verify_splits(start, a)
                                                     : max(1, min(a > start ? (a - 1) / IT - start / IT + 1 : 0, (int)gridDim.x));
    if (valid && lead) {
      // The (max, sum) of all splits fit on the first wave of the pair.
      f32x2 ml = {-INFINITY, 0.f};
      if (lane < splits) ml = gload(reinterpret_cast<const f32x2*>(base + lane * (D + 2) + D));
      const float mm = ml.x, l = ml.y;
      const float M = wave_max(mm);
      const float fe = hw_exp2(mm - M);
      const float f = (lane < splits && mm != -INFINITY) ? fe : 0.f;
      if (lane < SPLITS) fac[pair][lane] = f;
      const float lt = wave_total(f * l);
      if (lane == 0) red[pair] = lt;
    }
    lds_barrier();
    if (valid) {
      const float L = red[pair];
      // All splits of this dim: 8 loads issued together per round, or every lane waits for memory once per split.
      float o = 0.f;
      for (int s0 = 0; s0 < splits; s0 += 8) {
        float v[8], fc[8];
#pragma unroll
        for (int k = 0; k < 8; ++k) {
          const int sk = s0 + k;
          v[k] = sk < splits ? gload(base + sk * (D + 2) + d) : 0.f;
          fc[k] = sk < splits ? fac[pair][sk] : 0.f;
        }
#pragma unroll
        for (int k = 0; k < 8; ++k) o = fmaf(fc[k], v[k], o);
      }
      const float gate = gated ? __builtin_amdgcn_rcpf(1.f + hw_exp2(-mix[(size_t)tok * MIXS + 3072 + qh] * LOG2E)) : 1.f;
      y[(size_t)tok * H + qh * D + d] = __float2half_rn((L > 0 ? o * __builtin_amdgcn_rcpf(L) : 0.f) * gate);
    }
  }
}

// ---------------------------------------------------------------- conv
// Conv in_proj rows are [B_c, C_c, x_c] per channel. A wave owns channel pairs (one triple of units), one lane per
// channel, so the conv runs right after its rows without a phase of its own. History lives in the position ring, a
// rejected token is simply overwritten by the next round.
__device__ __forceinline__ void conv_wave(const Layer& L, float (&stage)[MT][64], __half* y16, int m, int a, int w, int nw, int lane) {
  const int c = 2 * (w + (lane >> 1) * nw) + (lane & 1);
  if (c < H) {
    const __half* cw = L.conv_w + c * 7;
    const int base = 6 * (lane >> 1) + 3 * (lane & 1);  // this channel's B, C, X in the stage
    float s[6], tap[7];
#pragma unroll
    for (int k = 0; k < 6; ++k) s[k] = __half2float(L.ring[(size_t)((a - 6 + k) & (RING - 1)) * H + c]);
#pragma unroll
    for (int k = 0; k < 7; ++k) tap[k] = __half2float(cw[k]);
    for (int t = 0; t < m; ++t) {
      const float B = stage[t][base], C = stage[t][base + 1], X = stage[t][base + 2];
      const __half bxh = __float2half_rn(B * X);
      float conv = tap[6] * __half2float(bxh);
#pragma unroll
      for (int k = 0; k < 6; ++k) conv = fmaf(tap[k], s[k], conv);
#pragma unroll
      for (int k = 0; k < 5; ++k) s[k] = s[k + 1];
      s[5] = __half2float(bxh);
      L.ring[(size_t)((a + t) & (RING - 1)) * H + c] = bxh;
      y16[(size_t)t * H + c] = __float2half_rn(C * conv);
    }
  }
}

// ---------------------------------------------------------------- LM heads
// DRAFT: out[t * stride + row] on the cropped rows. VERIFY: argmax of the penalised fp16-rounded logits per
// position, folded into g.best[t]; drafts not yet in `seen` count as seen only for the positions after them.
template <bool DRAFT, int M>
__device__ __forceinline__ void head_run(const Globals& g, const Plan& p, BufR<2> (&buf)[2], const GemvShared& gs, const uint32_t* seen_lds,
                                         float4 (&rows)[MT][RB], int lane, int w, int nw) {
  const float irep = g.inv_rep, rep = DRAFT ? 1.f : *g.rep;
  int32_t dt[MT];
  int isnew = 0;
  if constexpr (!DRAFT) {
#pragma unroll
    for (int t = 1; t < M; ++t) {
      dt[t] = uniform_i32(gload(&g.st->tokens[t]));
      if (uniform_i32(((seen_lds[dt[t] >> 5] >> (dt[t] & 31)) & 1) ? 1 : 0) == 0) isnew |= 1 << t;
    }
  }
  XRegs2<M> xr;
  xregs2_load<M>(xr, gs, lane);
  uint64_t best[M];
#pragma unroll
  for (int t = 0; t < M; ++t) best[t] = 0;
  const int units = p.rows / 2, last = units - 1;
  const int nk = (units - w + nw - 1) / nw;
  const uint32_t off_w0 = 16u * lane, off_w1 = off_w0 + 1024u;
  const uint32_t off_s0 = 2u * (((lane >> 3) & 3) * p.sq + (lane >> 5)), off_s1 = off_s0 + 4u;
  const bool row_head = (lane & 15) == 0;
  const int q = lane >> 4;
  // Row pointers of unit 1 (unit 0 was loaded before the barrier); the last unit is read again past the end.
  const uint32_t rowb = (uint32_t)p.rowbytes, srowb = 2u * (uint32_t)p.srow, stepw = 2u * (uint32_t)nw * rowb, steps = 2u * (uint32_t)nw * srowb;
  const int u1 = min(w + nw, last);
  Cursor cur = {reinterpret_cast<uint64_t>(p.W) + (uint64_t)(uint32_t)u1 * 2u * rowb + 512u * (uint32_t)p.i0,
                reinterpret_cast<uint64_t>(p.S) + (uint64_t)(uint32_t)u1 * 2u * srowb + 2u * (uint32_t)p.i0, 0, 0};
  for (int k0 = 0; k0 < nk; k0 += 2) {
#pragma unroll
    for (int s = 0; s < 2; ++s) {
      bufr_load_at<2>(buf[s ^ 1], cur, rowb, srowb, off_w0, off_w1, off_s0, off_s1);
      cur.wp += (k0 + s + 1 < nk - 1) ? (uint64_t)stepw : 0u;
      cur.sp += (k0 + s + 1 < nk - 1) ? (uint64_t)steps : 0u;
      float acc[2][M];
      unit_dot_r<M, 2>(buf[s], xr, acc);
      row_sum16_batch<2 * M>(&acc[0][0]);
      const int kr = 2 * ((k0 + s) & 3);
      if (row_head) {
#pragma unroll
        for (int t = 0; t < M; ++t) {
          reinterpret_cast<float*>(&rows[t][kr])[q] = acc[0][t];
          reinterpret_cast<float*>(&rows[t][kr + 1])[q] = acc[1][t];
        }
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    if (((k0 + 2) & 3) == 0 || k0 + 2 >= nk) {
      asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
      const int kb = k0 & ~3;
      const int n = min(2 * ((k0 & 3) + 2), 2 * (nk - kb));
      if (lane < n) {
        const int row = 2 * (w + (kb + (lane >> 1)) * nw) + (lane & 1);
        if constexpr (DRAFT) {
#pragma unroll
          for (int t = 0; t < M; ++t) {
            const float4 r = rows[t][lane];
            p.out[(int64_t)t * p.stride + row] = (r.x + r.y) + (r.z + r.w);
          }
        } else {
          const bool hit0 = ((seen_lds[row >> 5] >> (row & 31)) & 1u) != 0;
#pragma unroll
          for (int t = 0; t < M; ++t) {
            const float4 r = rows[t][lane];
            const float v = (r.x + r.y) + (r.z + r.w);
            if (g.dbg_logits) g.dbg_logits[(size_t)t * g.vocab + row] = v;
            const float vr = __half2float(__float2half_rn(v));
            bool hit = hit0;
            if (isnew) {
#pragma unroll
              for (int j = 1; j < M; ++j)
                if (j <= t && ((isnew >> j) & 1) && row == dt[j]) hit = true;
            }
            const float pv = hit ? (vr > 0.f ? vr * irep : vr * rep) : vr;
            best[t] = umax64(best[t], argmax_key(pv, row));
          }
        }
      }
      __builtin_amdgcn_sched_barrier(0);
    }
  }
  if constexpr (!DRAFT) {
    // Lanes 0..RB-1 hold candidates; three DPP steps fold them into lane 0. All lanes are active here.
#pragma unroll
    for (int t = 0; t < M; ++t) {
      uint64_t bb = best[t];
#pragma unroll
      for (int c = 0; c < 3; ++c) {
        const uint32_t lo = (uint32_t)bb, hi = (uint32_t)(bb >> 32);
        uint32_t olo, ohi;
        if (c == 0) { olo = __builtin_amdgcn_mov_dpp((int)lo, 0xb1, 0xf, 0xf, false); ohi = __builtin_amdgcn_mov_dpp((int)hi, 0xb1, 0xf, 0xf, false); }
        else if (c == 1) { olo = __builtin_amdgcn_mov_dpp((int)lo, 0x4e, 0xf, 0xf, false); ohi = __builtin_amdgcn_mov_dpp((int)hi, 0x4e, 0xf, 0xf, false); }
        else { olo = __builtin_amdgcn_mov_dpp((int)lo, 0x141, 0xf, 0xf, false); ohi = __builtin_amdgcn_mov_dpp((int)hi, 0x141, 0xf, 0xf, false); }
        bb = umax64(bb, ((uint64_t)ohi << 32) | olo);
      }
      if (lane == 0 && bb) __hip_atomic_fetch_max(g.best + t, bb, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
    }
  }
}

// One Markov step: z = base[j] + W2 . W1[prev]; a wave takes 64 rows at a time (W2 is stored [64 rows][16 pieces][64 lanes][16 B], so
// one load is one contiguous KB), argmax folded into best[j]. Row groups past one per wave spread over the blocks.
__device__ __forceinline__ void markov_phase(const Globals& g, int j, int prev) {
  const int N = (int)g.crop_rows;
  const int nw = gridDim.x * WAVES;
  const int w = blockIdx.x * WAVES + __builtin_amdgcn_readfirstlane(tidx() >> 6);
  const int lane = tidx() & 63;
  const int stride = N + 4;
  const int nrb = N / 64, rounds = nrb / nw, extra = nrb % nw, every = extra ? nw / extra : 1;
  const int iters = rounds + ((extra && w % every == 0 && w / every < extra) ? 1 : 0);
  const u32x4* up = reinterpret_cast<const u32x4*>(g.w1q + (size_t)prev * 256);
  u32x4 uw[16];
#pragma unroll
  for (int i = 0; i < 16; ++i) uw[i] = sload(up + i);
  const float su = sload(g.w1s + prev);
  uint64_t best = 0;
  for (int it = 0; it < iters; ++it) {
    const int rb = it < rounds ? w + it * nw : rounds * nw + w / every;
    const int row = rb * 64 + lane;
    const u32x4* wp = reinterpret_cast<const u32x4*>(g.w2_crop + (size_t)rb * 64 * 256);
    u32x4 ww[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) ww[i] = gload_nt(wp + i * 64 + lane);
    const float sc = gload(g.w2_scale + row);
    const float b = gload(g.base + (size_t)j * stride + row);
    int acc = 0;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
      const uint32_t a0 = ww[i].x, a1 = ww[i].y, a2 = ww[i].z, a3 = ww[i].w;
      const uint32_t b0 = uw[i].x, b1 = uw[i].y, b2 = uw[i].z, b3 = uw[i].w;
      acc = __builtin_amdgcn_sdot4(a0, b0, acc, false);
      acc = __builtin_amdgcn_sdot4(a1, b1, acc, false);
      acc = __builtin_amdgcn_sdot4(a2, b2, acc, false);
      acc = __builtin_amdgcn_sdot4(a3, b3, acc, false);
    }
    const float z = fmaf((float)acc, sc * su, b);
    best = umax64(best, argmax_key(z, row));
  }
#pragma unroll
  for (int o = 32; o > 0; o >>= 1) {
    const uint32_t lo = (uint32_t)best, hi = (uint32_t)(best >> 32);
    const uint64_t other = ((uint64_t)(uint32_t)__shfl_xor((int)hi, o) << 32) | (uint32_t)__shfl_xor((int)lo, o);
    best = umax64(best, other);
  }
  uint64_t* slot = g.best + j;
  if (lane == 0 && best) __hip_atomic_fetch_max(slot, best, __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
}

// ---------------------------------------------------------------- kernel
enum Tag { T_START = 1, T_CTX_FC, T_CTX_KV, T_CTX_RING, T_NORM, T_MIX, T_CONV, T_ATTN, T_COMB, T_OUT, T_W13, T_W2,
           T_DHEAD, T_MARKOV, T_VHEAD, T_COMMIT, T_WAIT };

__device__ __forceinline__ int64_t ptr_u(const void* p) { return uniform64(reinterpret_cast<int64_t>(p)); }

template <typename T>
__device__ __forceinline__ const T* uptr(const T* p) { return reinterpret_cast<const T*>(ptr_u(p)); }

// The layer pass is a small interpreter over (section, layer, stage): every linear layer of every section runs
// through the single GEMV site below, so one layer's instruction footprint stays inside the instruction cache.
__global__ __launch_bounds__(THREADS, 1) LFM3_KERNEL_ATTR void dspark_kernel(Globals g) {
  __shared__ Smem sm;
  tidx_init();
  uint32_t gen = 0;
  float* ws = g.ws;
  GemvShared& gs = sm.u.g;
  float4 (&gemv_rows)[MT][RB] = sm.rows[__builtin_amdgcn_readfirstlane(tidx() >> 6)];
  float (&stage_rows)[MT][64] = sm.u.staged.rows[__builtin_amdgcn_readfirstlane(tidx() >> 6)];
  uint32_t* seen_lds = sm.u.head.seen;
  int tn = 0;
  bool pending = false;
  auto arrive = [&]() {
    grid_arrive(g.barrier, gen);
    pending = true;
  };
  auto wait_pending = [&]() {
    if (pending) {
      grid_wait(g.barrier, gen);
      trace_mark(g.trace, tn, T_WAIT);
    }
    pending = false;
  };
#ifdef LFM3_ARRIVALS
  int bar_n = 0;
#endif
  // Layer rows are read-only during a launch: scalar loads, which the barrier's L1 invalidate does not touch.
  const Layer* const slayers = (const Layer*)(const __attribute__((address_space(4))) Layer*)(uintptr_t)g.layers;
  State* st = g.st;
  __half* x16 = reinterpret_cast<__half*>(ws + WS_X16);
  __half* a16 = reinterpret_cast<__half*>(ws + WS_A);
  constexpr int NLAYERS = 24;
  const int b = blockIdx.x, nb = gridDim.x;
  const int wave = uniform_i32(tidx() >> 6);
  const int w = b * WAVES + wave, nw = nb * WAVES;
  trace_mark(g.trace, tn, T_START);

  // Conv history: the running state -> position ring.
  {
    const int a0 = uniform_i32(gload(&st->a));
    int ci = 0;
    for (int li = 0; li < NLAYERS; ++li) {
      const Layer* L = slayers + li;
      if (uniform_i32((int)L->kind) != 0) continue;
      if ((ci++ % nb) == b) {
        const int32_t slot = L->state_index[0];
        const __half* st0 = L->conv_state + (int64_t)slot * L->cs_slot;
        for (int c = tidx(); c < H; c += THREADS)
#pragma unroll
          for (int k = 0; k < 6; ++k)
            L->ring[(size_t)((a0 - 6 + k) & (RING - 1)) * H + c] = st0[(int64_t)c * L->cs_dim + k * L->cs_t];
      }
    }
  }
  if (b == 0 && tidx() == 0) {
    const int t = st->anchor;
    __hip_atomic_fetch_or(g.seen + (t >> 5), 1u << (t & 31), __ATOMIC_RELAXED, __HIP_MEMORY_SCOPE_AGENT);
  }
  arrive();

  for (;;) {
    wait_pending();
    if (!uniform_i32(gload(&st->run))) break;
    const int skip_now = uniform_i32(gload(&st->skip));
    const int force_raw = uniform_i32(gload(&st->force_m));
    const int a = uniform_i32(gload(&st->a));
    const int force_m = skip_now > 0 ? 1 : force_raw;
    const int ctx_new = uniform_i32(gload(&st->ctx_new));
    const int ctx_pos = uniform_i32(gload(&st->ctx_pos));
    const int ctx_lo = uniform_i32(gload(&st->ctx_lo));

    // sec: 0 context projection, 1 drafter layers, 2 target layers, 3 done.
    int sec = (ctx_new > 0 && !force_raw) ? 0 : (force_m ? 2 : 1);
    int m = sec == 0 ? ctx_new : (sec == 1 ? MT : force_m);
    int li = 0, stage = 0, cur = 0;
    bool embed = sec != 0, first = true;

    while (sec < 3) {
      const int lane = tidx() & 63;
      const bool drafter = sec == 1;
      const Layer* L = slayers + (drafter ? NLAYERS + li : li);
      const int kind = sec == 0 ? 3 : uniform_i32((int)L->kind);
      const int TB = drafter ? 100 : 0;

      // ------ specials
      if (embed) {
        wait_pending();
        // Block t embeds token t, 8 dims per lane.
        if (b < m) {
          int tok;
          if (drafter) tok = b == 0 ? uniform_i32(gload(&st->anchor)) : (int)g.mask_token;
          else tok = uniform_i32(gload(&st->tokens[b]));
          const u32x4 e4 = gload(reinterpret_cast<const u32x4*>(g.embed + (int64_t)tok * H + 8 * tidx()));
          const half2_t e0 = as_h2(e4.x), e1 = as_h2(e4.y), e2 = as_h2(e4.z), e3 = as_h2(e4.w);
          float4* d4 = reinterpret_cast<float4*>(ws + WS_H0 + (size_t)b * H + 8 * tidx());
          d4[0] = make_float4((float)e0.x, (float)e0.y, (float)e1.x, (float)e1.y);
          d4[1] = make_float4((float)e2.x, (float)e2.y, (float)e3.x, (float)e3.y);
        }
        if (b == 0 && tidx() == 0) {
          if (!drafter) st->M = m;
          const int nbest = drafter ? MT : m;
          for (int t = 0; t < nbest; ++t) g.best[t] = 0;
        }
        arrive();
        embed = false;
        first = true;
        cur = 0;
        continue;
      }
      if (sec == 0 && stage == 6) {  // drafter KV ring
        wait_pending();
        const int task = w;
        const int tasks = m * DRAFT_LAYERS * KVH;
        if (task < tasks) {
          const int t = task / (DRAFT_LAYERS * KVH), lh = task % (DRAFT_LAYERS * KVH);
          const int l = lh / KVH, h = lh % KVH;
          const Layer& DL = slayers[NLAYERS + l];
          const float* row = ws + WS_CKV + (size_t)t * QKV + l * 1024;
          float k0 = row[h * D + lane], k1 = row[h * D + lane + 64];
          const float v0 = row[512 + h * D + lane], v1 = row[512 + h * D + lane + 64];
          const int pos = ctx_pos + t;
          const __half* c = DL.rope + (int64_t)pos * DL.rotary;
          norm_rope(k0, k1, __half2float(DL.k_norm[lane]), __half2float(DL.k_norm[lane + 64]), __half2float(c[lane]),
                    __half2float(c[64 + lane]));
          uint8_t* rec = reinterpret_cast<uint8_t*>(DL.kv) + (size_t)((pos / SUB) & CTX_MASK) * DL.kv_block +
                         (size_t)h * DL.kv_head + (size_t)(pos % SUB) * DL.kv_slot;
          const float ks = quantize_pair(k0, k1, reinterpret_cast<int8_t*>(rec));
          const float vs = quantize_pair(v0, v1, reinterpret_cast<int8_t*>(rec + 132));
          if (lane == 0) {
            *reinterpret_cast<float*>(rec + 128) = ks;
            *reinterpret_cast<float*>(rec + 260) = vs;
          }
        }
        arrive();
        trace_mark(g.trace, tn, T_CTX_RING);
        sec = force_m ? 2 : 1;
        m = force_m ? force_m : MT;
        embed = true;
        stage = 0;
        continue;
      }
      if (sec != 0 && stage == 1) {  // attention (conv layers skip this stage)
        // Block t: token t's query and fresh K/V.
        wait_pending();
        if (b < m) query_fresh(*L, ws + WS_MIX, reinterpret_cast<int8_t*>(ws + WS_Q), sm.u.attn, b, a, wave, lane);
        arrive();
        trace_mark(g.trace, tn, 22);
        if (!drafter && use_paged(attn_start(kind, (int)L->window, a, ctx_lo), a, m)) {
          verify_attention_paged(*L, ws + WS_PART, sm.u.attn, m, a, wait_pending);
        } else {
          for (int p0 = 0, nq = 0; p0 < m; p0 += nq) {
            if (p0) __syncthreads();
            nq = min(m - p0, 4);
            if (drafter) attention_pass<1, true>(*L, ws + WS_MIX, ws + WS_PART, sm.u.attn, p0, nq, m, a, ctx_lo, false, g.trace, tn, wait_pending);
            else if (nq == 1) attention_pass<4, false>(*L, ws + WS_MIX, ws + WS_PART, sm.u.attn, p0, nq, m, a, ctx_lo, true, g.trace, tn, wait_pending);
            else if (nq == 2) attention_pass<2, false>(*L, ws + WS_MIX, ws + WS_PART, sm.u.attn, p0, nq, m, a, ctx_lo, true, g.trace, tn, wait_pending);
            else attention_pass<1, false>(*L, ws + WS_MIX, ws + WS_PART, sm.u.attn, p0, nq, m, a, ctx_lo, true, g.trace, tn, wait_pending);
          }
        }
        arrive();
        trace_mark(g.trace, tn, T_ATTN + TB);
        stage = 2;
        continue;
      }
      if (sec != 0 && stage == 2) {
        // Memory is idle through the attention's end and the combine: the next GEMV's weights (4.2 MB) fit in L2.
        {
          const uint64_t wo = reinterpret_cast<uint64_t>(uptr(L->w_out));
          const uint32_t off_t = (uint32_t)(64 * lane);
          const int units = H / 2, nku = (units - w + nw - 1) / nw;
          for (int ti = 0; ti < nku; ++ti) {
            const uint64_t ta = wo + (uint64_t)(uint32_t)(w + ti * nw) * 2u * H;
            asm volatile("global_load_dword v254, %0, %1" : : "v"(off_t), "s"(ta) : "v254");
          }
        }
        wait_pending();
        const int start = attn_start(kind, (int)L->window, a, ctx_lo);
        combine_phase(ws + WS_MIX, ws + WS_PART, reinterpret_cast<__half*>(ws + WS_Y), start, a, m, kind != 3, sm.red);
        arrive();
        trace_mark(g.trace, tn, T_COMB + TB);
        stage = 3;
        continue;
      }
      if (sec != 0 && li == (drafter ? DRAFT_LAYERS : NLAYERS)) {
        // ------ drafter output: final norm, cropped head, Markov steps; target output: final norm, full head.
        const Layer* Lp = slayers + (drafter ? NLAYERS + DRAFT_LAYERS - 1 : NLAYERS - 1);
        Plan p = {};
        p.W = uptr(drafter ? g.w_crop : g.w_head);
        p.S = uptr(drafter ? g.s_crop : g.s_head);
        p.rowbytes = H; p.srow = 16; p.sq = 4; p.i0 = 0; p.i3 = 3; p.triple = false; p.staged = false;
        const int N = (int)g.crop_rows;
        p.rows = drafter ? N + 4 : (int)g.vocab;
        p.out = g.base;
        p.stride = N + 4;
        if (drafter) {
          wait_pending();
          norm_compute(ws + WS_H0 + (size_t)cur * MT * H, ws + WS_D, 4, MT * H, nullptr, g.d_final_norm, false, nullptr, nullptr, x16, sm.red, m);
          arrive();
          auto pre = [&]() {
            wait_pending();
#ifdef LFM3_X8
            x_fill8(x16, gs, m);
#else
            x_fill(x16, H, 0, gs, m);
#endif
            trace_mark(g.trace, tn, T_NORM + TB);
          };
          gemv_phase<MT, 2, LFM3_NBX>(p, gs, gemv_rows, stage_rows, lane, w, nw, pre);
          arrive();
          trace_mark(g.trace, tn, T_DHEAD);
        } else
        {
        BufR<2> buf[2];
        bufr_undef<2, 2>(buf);
        if (w < p.rows / 2) {
          const uint32_t off_w0 = 16u * lane, off_s0 = 2u * (((lane >> 3) & 3) * p.sq + (lane >> 5));
          bufr_load<2>(buf[0], p, w, off_w0, off_w0 + 1024u, off_s0, off_s0 + 4u);
        }
        __builtin_amdgcn_sched_barrier(0);
        wait_pending();
        norm_compute(ws + WS_H0 + (size_t)cur * MT * H, ws + WS_D, 4, MT * H, drafter ? nullptr : Lp->post_ffn_norm,
                     drafter ? g.d_final_norm : g.final_norm, !drafter, nullptr,
                     drafter ? nullptr : ws + WS_TAP + (size_t)Lp->tap_slot * MT * H, x16, sm.red, m);
        arrive();
        wait_pending();
#ifdef LFM3_X8
        x_fill8(x16, gs, m);
#else
        x_fill(x16, H, 0, gs, m);
#endif
        if (!drafter) {
          for (int i = tidx(); i < SEEN_WORDS; i += THREADS) seen_lds[i] = gload(g.seen + i);
          lds_barrier();
        }
        trace_mark(g.trace, tn, T_NORM + TB);
        if (w < p.rows / 2) {
          if (drafter) {
            head_run<true, MT>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw);
          } else {
            switch (m) {
              case 1: head_run<false, 1>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw); break;
              case 2: head_run<false, 2>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw); break;
              case 3: head_run<false, 3>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw); break;
              case 4: head_run<false, 4>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw); break;
              case 5: head_run<false, 5>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw); break;
              case 6: head_run<false, 6>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw); break;
              case 7: head_run<false, 7>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw); break;
              case 8: head_run<false, 8>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw); break;
              default: head_run<false, 9>(g, p, buf, gs, seen_lds, gemv_rows, lane, w, nw); break;
            }
          }
        }
        arrive();
        trace_mark(g.trace, tn, drafter ? T_DHEAD : T_VHEAD);
        }
        if (drafter) {
          // Sequential Markov steps with early stop on the confidence head.
          wait_pending();
          const int max_drafts = (int)g.max_drafts;
          const int limit = uniform_i32(gload(&st->limit));
          const int budget = uniform_i32(gload(&st->budget));
          const int count = uniform_i32(gload(&st->count));
          const int cap = min(min(max_drafts, MT - 1), min(limit - a, budget - count) - 1);
          int prev = uniform_i32(gload(&st->anchor));
          int drafts = 0;
          for (int j = 0; j < cap; ++j) {
            const float conf = gload(g.base + (size_t)j * (N + 4) + N) + sload(g.c2 + prev) + g.conf_bias;
            if (!uniform_i32(conf >= g.conf_thr ? 1 : 0)) break;
            markov_phase(g, j, prev);
            arrive();
            wait_pending();
            const uint64_t key = uniform64((int64_t)__hip_atomic_load(g.best + j, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT));
            const int row = (int)(0xffffffffu - (uint32_t)key);
            prev = uniform_i32(sload(g.crop_ids + row));
            drafts = j + 1;
            if (b == 0 && tidx() == 0) st->tokens[j + 1] = prev;
            trace_mark(g.trace, tn, T_MARKOV);
          }
          if (b == 0 && tidx() == 0) st->tokens[0] = st->anchor;
          m = uniform_i32(1 + drafts);
          sec = 2;
          li = 0;
          stage = 0;
          embed = true;
          arrive();
          continue;
        }
        sec = 3;
        continue;
      }

      // ------ the GEMV site: plan, x source, then the exact token-count loop
      Plan p = {};
      int pk = 0;                  // 0: norm, 1: load_x
      const __half* xsrc = nullptr; int xstride = 0, xoff = 0;
      const float* h_old = nullptr; const float* vin = nullptr; int bands = 0, vstride = 0;
      const __half* post_w = nullptr; const __half* norm_w = nullptr; bool sandwich = false;
      float* dst = nullptr; float* tap = nullptr; bool flip = false;
      bool last_of_stage = true;
      bool waitb = true;
      int tag = T_MIX;
      p.rowbytes = H; p.srow = 16; p.sq = 4; p.i0 = 0; p.i3 = 3; p.triple = false; p.staged = false;
      int gw = w, gnw = nw;    // this wave's rank among the waves of the phase
      bool big = false;        // long phase: 8 KB in flight per wave in 4-row units
      if (sec == 0) {
        p.stride = H;
        if (stage < NUM_TAPS) {
          p.W = uptr(g.w_fc + (size_t)stage * H * H);
          p.S = uptr(g.s_fc + (size_t)stage * H * 16);
          p.rows = H;
          p.out = ws + WS_CP + (size_t)stage * MT * H;
          pk = 0;
          h_old = ws + WS_TAP + (size_t)stage * MT * H;
          norm_w = g.tap_norms + stage * H;
          last_of_stage = stage == NUM_TAPS - 1;
          waitb = stage == 0;
          tag = T_CTX_FC;
        } else {
          p.W = uptr(g.w_ckv);
          p.S = uptr(g.s_ckv);
          p.rows = QKV;
          p.out = ws + WS_CKV;
          p.stride = QKV;
          pk = 0;
          vin = ws + WS_CP; bands = NUM_TAPS; vstride = MT * H;
          norm_w = g.hidden_norm;
          tag = T_CTX_KV;
        }
      } else {
        sandwich = !drafter;
        switch (stage) {
          case 0: {
            p.W = uptr(L->w_mix);
            p.S = uptr(L->s_mix);
            p.rows = kind == 0 ? 6144 : (kind == 3 ? QKV : QKVG);
            p.triple = kind == 0;
            p.staged = kind == 0;
            p.out = ws + WS_MIX;
            p.stride = MIXS;
            pk = 0;
            h_old = ws + WS_H0 + (size_t)cur * MT * H;
            if (!first) {
              vin = ws + WS_D; bands = 4; vstride = MT * H;
              const Layer* Lp = L - 1;
              if (sandwich) post_w = Lp->post_ffn_norm;
              if (!drafter) {
                const int ts = uniform_i32((int)Lp->tap_slot);
                if (ts >= 0) tap = ws + WS_TAP + (size_t)ts * MT * H;
              }
            }
            norm_w = L->op_norm;
            dst = ws + WS_H0 + (size_t)(cur ^ 1) * MT * H;
            flip = true;
            tag = kind == 0 ? T_CONV : T_MIX;
            break;
          }
          case 3: {
            p.W = uptr(L->w_out);
            p.S = uptr(L->s_out);
            p.rows = H;
            p.out = ws + WS_O;
            p.stride = H;
            pk = 1;
            xsrc = reinterpret_cast<const __half*>(ws + WS_Y); xstride = H; xoff = 0;
            tag = T_OUT;
            break;
          }
          case 4: {
            p.W = uptr(L->w13);
            p.S = uptr(L->s13);
            p.rows = 2 * I;
            big = true;
            p.staged = true;
            p.out = ws + WS_G13;
            p.stride = 2 * I;
            pk = 0;
            h_old = ws + WS_H0 + (size_t)cur * MT * H;
            vin = ws + WS_O; bands = 1; vstride = 0;
            if (sandwich) post_w = L->post_op_norm;
            norm_w = L->ffn_norm;
            dst = ws + WS_H0 + (size_t)(cur ^ 1) * MT * H;
            flip = true;
            tag = T_W13;
            break;
          }
          default: {  // down projection: each quarter of the blocks owns one 2048-column band of all rows
            const int band = b * 4 / nb, b0 = (band * nb + 3) / 4, b1 = ((band + 1) * nb + 3) / 4;
            gw = (b - b0) * WAVES + wave;
            gnw = (b1 - b0) * WAVES;
            p.W = uptr(L->w2);
            p.S = uptr(L->s2);
            p.rowbytes = I; p.srow = 64; p.sq = 16; p.i0 = 4 * band;
            if (band == 3) p.i3 = 2;
            p.rows = H;
            big = true;
            p.out = ws + WS_D + (size_t)band * MT * H;
            p.stride = H;
            pk = 1;
            xsrc = a16; xstride = I; xoff = band * H;
            tag = T_W2;
            break;
          }
        }
      }
      if (pk == 0) {
        // The norm phase comes first: its arrival must not wait for the next GEMV's weights (vmcnt is in order).
        wait_pending();
        norm_compute(h_old, vin, bands, vstride, post_w, norm_w, sandwich, dst, tap, x16, sm.red, m);
        arrive();
        trace_mark(g.trace, tn, 18 + TB);
        if (flip) cur ^= 1;
        first = false;
        xsrc = x16; xstride = H; xoff = 0;
      }
      {
        auto pre = [&]() {
          wait_pending();
#ifdef LFM3_X8
          if (pk == 0) x_fill8(x16, gs, m);
          else
#endif
          x_fill(xsrc, xstride, xoff, gs, m);
          if (pk == 0) trace_mark(g.trace, tn, T_NORM + TB);
        };
        // One copy of the code per token count: a second copy for the short phases costs more in instruction fetch than its depth gains.
        switch (m) {
          case 1: gemv_phase<1, 2, LFM3_NB1>(p, gs, gemv_rows, stage_rows, lane, gw, gnw, pre); break;
          case 2: gemv_phase<2, 2, LFM3_NB2>(p, gs, gemv_rows, stage_rows, lane, gw, gnw, pre); break;
          case 3: gemv_phase<3, 2, LFM3_NB3>(p, gs, gemv_rows, stage_rows, lane, gw, gnw, pre); break;
          case 4: gemv_phase<4, 2, LFM3_NB4>(p, gs, gemv_rows, stage_rows, lane, gw, gnw, pre); break;
          case 5: gemv_phase<5, 2, LFM3_NBX>(p, gs, gemv_rows, stage_rows, lane, gw, gnw, pre); break;
          case 6: gemv_phase<6, 2, LFM3_NBX>(p, gs, gemv_rows, stage_rows, lane, gw, gnw, pre); break;
          case 7: gemv_phase<7, 2, LFM3_NBX>(p, gs, gemv_rows, stage_rows, lane, gw, gnw, pre); break;
          case 8: gemv_phase<8, 2, LFM3_NBX>(p, gs, gemv_rows, stage_rows, lane, gw, gnw, pre); break;
          default: gemv_phase<9, 2, LFM3_NBX>(p, gs, gemv_rows, stage_rows, lane, gw, gnw, pre); break;
        }
      }
      if (sec != 0 && stage == 0 && kind == 0) {
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        conv_wave(*L, stage_rows, reinterpret_cast<__half*>(ws + WS_Y), m, a, w, nw, lane);
      }
      if (sec != 0 && stage == 4) {
        // Gate/up rows of a pair stay on one wave: silu(gate) * up right here.
        asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
        // Pair j of this wave: row pair 2 unit + (j & 1) with 4-row units, unit j with 2-row units.
        const int u = w + lane * nw;
        if (u < I && lane < 32) {
          for (int t = 0; t < m; ++t) {
            const float gt = stage_rows[t][2 * lane], up = stage_rows[t][2 * lane + 1];
            a16[(size_t)t * I + u] = __float2half_rn(gt * __builtin_amdgcn_rcpf(1.f + hw_exp2(-gt * LOG2E)) * up);
          }
        }
      }
      if (last_of_stage) {
        arrive();
        trace_mark(g.trace, tn, tag + TB);
      }
      // ------ advance
      if (sec == 0) {
        ++stage;
      } else if (stage == 0) {
        stage = kind == 0 ? 3 : 1;
      } else if (stage < 5) {
        ++stage;
      } else {
        ++li;
        stage = 0;
      }
    }

    // ---- commit: every block copies its share of the accepted K/V, one (layer, token) each; block 0 keeps the books.
    // The books touch nothing the other blocks read here (tokens[0], counters), so they stay in the same phase.
    wait_pending();
    {
      const int aa = a;
      int acc = 0;
      for (int t = 0; t < m - 1; ++t) {
        const int bt = uniform_i32((int)(0xffffffffu - (uint32_t)gload(g.best + t)));
        const int dt = uniform_i32(gload(&st->tokens[t + 1]));
        if (bt != dt) break;
        ++acc;
      }
      const int commit = acc + 1;
      const uint32_t mask = (uint32_t)uniform_i32((int)g.attn_mask);
      const int tasks = __builtin_popcount(mask) * commit;
      for (int task = b; task < tasks; task += nb) {
        const int c = task % commit;
        int nth = task / commit, li2 = 0;
        uint32_t mm = mask;
        for (; nth > 0; --nth) mm &= mm - 1;
        li2 = __builtin_ctz(mm);
        const Layer* L = slayers + li2;
        const int64_t bs = uniform64(L->block_size);
        const uint8_t* kv = reinterpret_cast<const uint8_t*>(L->kv);
        const int pos = aa + c;
        const int blk = uniform_i32(sload(L->block_table + (int)(pos / bs)));
        uint8_t* dstk = const_cast<uint8_t*>(kv) + (int64_t)blk * L->kv_block + (pos % bs) * L->kv_slot;
        const uint32_t* src = reinterpret_cast<const uint32_t*>(L->fresh + (size_t)c * KVH * REC);
        for (int i = tidx(); i < KVH * REC / 4; i += THREADS) reinterpret_cast<uint32_t*>(dstk)[i] = src[i];
      }
      if (b == 0) {
      const int nb_token = uniform_i32((int)(0xffffffffu - (uint32_t)gload(g.best + acc)));
      if (tidx() == 0) {
        int count = st->count;
        for (int i = 1; i <= acc; ++i) g.out_tokens[count++] = st->tokens[i];
        g.out_tokens[count++] = nb_token;
        for (int i = 1; i <= acc; ++i) {
          const int tk = st->tokens[i];
          g.seen[tk >> 5] |= 1u << (tk & 31);
        }
        g.seen[nb_token >> 5] |= 1u << (nb_token & 31);
        bool stop = false;
        if (st->eos >= 0)
          for (int i = count - commit; i < count; ++i) stop = stop || g.out_tokens[i] == st->eos;
        st->count = count;
        st->ctx_new = commit;
        st->ctx_pos = aa;
        st->a = aa + commit;
        st->anchor = nb_token;
        st->accepted += acc;
        st->verified += m;
        if (skip_now > 0) {
          st->skip -= 1;
        } else if (acc == 0) {
          st->miss += 1;
          if (st->skip_after > 0 && st->miss >= st->skip_after) {
            st->skip = st->skip_rounds;
            st->miss = 0;
          }
        } else {
          st->miss = 0;
        }
        st->tokens[0] = nb_token;
        st->rounds_done += 1;
        st->rounds_left -= 1;
        st->run = (!stop && st->rounds_left > 0 && st->a < st->stop_at && count < st->budget) ? 1 : 0;
      }
      }
    }
    arrive();
    trace_mark(g.trace, tn, T_COMMIT);
  }

  // Conv history -> the running state, and the checkpoint at stop_at. A launch passes stop_at by
  // less than one verify pass, so the ring still holds the positions before it.
  {
    const int a1 = uniform_i32(gload(&st->a));
    const int stop = uniform_i32(gload(&st->stop_at));
    const int ckpt = uniform_i32(gload(&st->ckpt_slot));
    int ci = 0;
    for (int li = 0; li < NLAYERS; ++li) {
      const Layer* L = slayers + li;
      if (uniform_i32((int)L->kind) != 0) continue;
      if ((ci++ % nb) == b) {
        const int32_t slot = L->state_index[0];
        __half* st0 = L->conv_state + (int64_t)slot * L->cs_slot;
        for (int c = tidx(); c < H; c += THREADS)
#pragma unroll
          for (int k = 0; k < 6; ++k)
            st0[(int64_t)c * L->cs_dim + k * L->cs_t] = L->ring[(size_t)((a1 - 6 + k) & (RING - 1)) * H + c];
        if (ckpt >= 0 && a1 >= stop) {
          __half* ck = L->conv_state + (int64_t)ckpt * L->cs_slot;
          for (int c = tidx(); c < H; c += THREADS)
#pragma unroll
            for (int k = 0; k < 6; ++k)
              ck[(int64_t)c * L->cs_dim + k * L->cs_t] = L->ring[(size_t)((stop - 6 + k) & (RING - 1)) * H + c];
        }
      }
    }
  }
  __syncthreads();
  trace_mark(g.trace, tn, 19);
}

}  // namespace lfm3ds
using namespace lfm3ds;

extern "C" {
int lfm3_dspark_workspace_floats() { return WS_FLOATS; }
int lfm3_dspark_layer_bytes() { return sizeof(Layer); }
int lfm3_dspark_globals_bytes() { return sizeof(Globals); }
int lfm3_dspark_state_bytes() { return sizeof(State); }
int lfm3_dspark_fresh_bytes() { return 32 * KVH * REC; }
int lfm3_dspark_ring_tokens() { return RING; }
int lfm3_dspark_max_tokens() { return MT; }
int lfm3_dspark_lds_bytes() { return sizeof(Smem); }

int lfm3_dspark_launch(const void* globals, int blocks, hipStream_t stream) {
  Globals g = *reinterpret_cast<const Globals*>(globals);
  hipLaunchKernelGGL(dspark_kernel, dim3(blocks), dim3(THREADS), 0, stream, g);
  return (int)hipGetLastError();
}
}
