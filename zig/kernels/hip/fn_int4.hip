// GPTQ int4 W4A16 matmuls of the INT4-AutoRound Flash Next checkpoint on gfx11 (wave32 WMMA): the routed experts
// (grouped by the experts plan, gate/up with the SwiGLU epilogue, down) and the lm_head (dense rows). Ours (no Python
// counterpart); the HIP rewrite of zig/kernels/cuda/fn_int4.cu with the same arguments.
//
// Arithmetic: a weight (q - 8) * scale, q - 8 exact in bf16. Per group of GS inputs one output's products run in one
// fp32 WMMA chain (v_wmma_f32_16x16x16_bf16, the group's k16 steps in order, C = 0 at the first), then
// acc = fmaf(group sum, scale, acc) in group order. A WMMA's output (m, n) reads row m of A and column n of B only,
// so a row's bits never depend on the other rows, the row count, the plan's items, NT, MT or the grid.
//
// Packed layout (cuda_int4.zig packWords): words uint32 [N/16/ST][K/GS][ST][GS/32][16 columns][4]: super tile of ST
// n16 tiles (ST = gcd(N/16, 16)), group, tile, k32 chunk, column c, then the chunk's 32 inputs of column c as 4 words
// of 8 nibbles, input i at nibble i, each the two's complement 4-bit q - 8 (GPTQ's code xor 8). Scales fp16
// [N/16/ST][K/GS][ST][16]. A group's tiles side by side: the waves of a launch, all near the same group, read DRAM
// in long runs (a wave streaming its own tile's groups reads 1 KiB runs: 145 vs 237 GB/s, int4_probe.hip;
// super tiles of 4, 16 or 32 measure the same).
//
// WMMA operands on gfx11 wave32 (int4_probe.py): lane l holds A row l % 16 and B column l % 16, element i = k i
// (lanes 16-31 repeat 0-15); D element i of lane l is (row 2i + l / 16, column l % 16).

#include <hip/hip_runtime.h>
#include <stdint.h>

namespace tf_fn_int4 {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));

constexpr int WAVES = 4;

// Signed nibble j of w as fp32 (exact).
__device__ __forceinline__ float nib(uint32_t w, int j) { return (float)((int)(w << (28 - 4 * j)) >> 28); }

// The bf16 pair (lo, hi) of two fp32s whose low halves are zero.
__device__ __forceinline__ uint32_t pk(float lo, float hi) {
  return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo), 0x07060302u);
}

// A k16 step's B fragment: words w0 (inputs 0..7) and w1 (8..15) of the lane's column.
__device__ __forceinline__ v16s dq(uint32_t w0, uint32_t w1) {
  u8v r;
#pragma unroll
  for (int j = 0; j < 4; ++j) {
    r[j] = pk(nib(w0, 2 * j), nib(w0, 2 * j + 1));
    r[4 + j] = pk(nib(w1, 2 * j), nib(w1, 2 * j + 1));
  }
  return (v16s)r;
}

__device__ __forceinline__ uint16_t bf16(float x) {   // round to nearest even (cuda's __float2bfloat16_rn)
  const uint32_t u = __float_as_uint(x);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float bf(float x) { return __uint_as_float((uint32_t)bf16(x) << 16); }

// experts.cuh's SwiGLU: gate and up rounded to bf16, silu rounded, times up.
__device__ __forceinline__ float swiglu(float g, float u, float limit) {
  float gv = bf(g), uv = bf(u);
  if (limit > 0.f) {
    gv = fminf(gv, limit);
    uv = fminf(fmaxf(uv, -limit), limit);
  }
  return bf(gv / (1.f + expf(-gv))) * uv;
}

// A barrier that orders shared memory only: __syncthreads' workgroup fence also waits for every global load in flight
// (vmcnt(0) in WGP mode), which would drain the weights' prefetch at each group.
__device__ __forceinline__ void lds_barrier() {
  __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
  __builtin_amdgcn_s_barrier();
  __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// n16 tiles a super tile of an [N, K] matrix: gcd(N / 16, 16) (cuda_int4.zig superTiles).
__device__ __forceinline__ int super_tiles(int N) {
  const int t = N / 16;
  return min(t & -t, 16);
}

// A group's operands in flight: each (matrix, n16 tile)'s words and scale of the wave, and the thread's XN 16-byte
// chunks of the block's activation rows (bound for shared memory).
template <int GS, int NT, int MATS, int XN>
struct Stage {
  u4v w[MATS][NT][GS / 32];
  _Float16 s[MATS][NT];   // converted at use: a conversion here would wait for the load at once
  u4v x[XN];
};

template <int GS, int NT, int MATS, int XN>
__device__ __forceinline__ void load_stage(Stage<GS, NT, MATS, XN>& st, const uint32_t* const (&w)[MATS][NT],
                                           const _Float16* const (&s)[MATS][NT], const uint16_t* const (&xp)[XN],
                                           int g, int gw) {
#pragma unroll
  for (int i = 0; i < XN; ++i) st.x[i] = reinterpret_cast<const u4v*>(xp[i] + g * GS)[0];   // zeroed at use
#pragma unroll
  for (int m = 0; m < MATS; ++m)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int q = 0; q < GS / 32; ++q)
        st.w[m][j][q] = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(w[m][j] + g * gw * 2 * GS + q * 64));
      st.s[m][j] = s[m][j][g * gw * 16];
    }
}

// One group: a fresh WMMA chain over its k16 steps for every (row tile, matrix, n16 tile), then the scale's fmaf.
// `xs`: the group's rows in shared memory, RS bf16 apart (rows past the item are zero). Every row tile runs, live or
// not: a run-time bound on the row-tile loops (or a branch to a one-tile copy) makes the compiler spill.
template <int GS, int NT, int MT, int MATS, int XN, int RS>
__device__ __forceinline__ void compute_group(v8f (&acc)[MT][MATS][NT], const Stage<GS, NT, MATS, XN>& cur,
                                              const uint16_t* xs, int c) {
  v8f d[MT][MATS][NT];
#pragma unroll
  for (int r = 0; r < MT; ++r)
#pragma unroll
    for (int m = 0; m < MATS; ++m)
#pragma unroll
      for (int j = 0; j < NT; ++j) d[r][m][j] = v8f{};
#pragma unroll
  for (int k = 0; k < GS / 16; ++k) {
    v16s a[MT];
#pragma unroll
    for (int r = 0; r < MT; ++r) {
        const u4v* p = reinterpret_cast<const u4v*>(xs + (16 * r + c) * RS + 16 * k);
        const u4v lo = p[0], hi = p[1];
        a[r] = (v16s)u8v{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
      }
#pragma unroll
    for (int m = 0; m < MATS; ++m)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const u4v q = cur.w[m][j][k / 2];
        const v16s b = dq(q[2 * (k & 1)], q[2 * (k & 1) + 1]);
#pragma unroll
        for (int r = 0; r < MT; ++r) d[r][m][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[r], b, d[r][m][j]);
      }
  }
#pragma unroll
  for (int r = 0; r < MT; ++r)
#pragma unroll
    for (int m = 0; m < MATS; ++m)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][m][j][i] = fmaf(d[r][m][j][i], (float)cur.s[m][j], acc[r][m][j][i]);
}


// A block a unit: (item, WAVES x NT n16 tiles of every matrix), a wave NT of them. Items (expert, first member, count)
// from the experts plan, or, with `items` null, the dense rows [16 i, 16 i + 16) of `rows` (the head). `slots`: member
// p reads input row p / slots (gate/up), 0: row p (down). A pass takes MT row tiles of 16 against one decode of the
// weights; the block stages each group's rows in shared memory (two buffers, one barrier a group).
template <int GS, int NT, int MT, int MATS, int EPI>
__device__ __forceinline__ void int4_body(const uint16_t* __restrict__ X, int x_stride, int slots,
                                          const uint32_t* __restrict__ W, const _Float16* __restrict__ S, int K, int N,
                                          const int* __restrict__ items, const int* __restrict__ counts,
                                          const int* __restrict__ members, int rows, void* __restrict__ out,
                                          int out_stride, float limit, int skip) {
  constexpr int BR = 16 * MT;                     // rows a pass
  constexpr int CH = GS / 8;                      // 16-byte chunks a row's group
  constexpr int XN = BR * CH / (WAVES * 32);      // chunks a thread stages
  constexpr int RS = GS + 8;                      // shared row stride (bf16): rows 16 bytes off the banks' period
  constexpr int DS = 2;   // groups in flight (even: the shared buffers alternate); 4 or 8 measured slower
  static_assert(XN * WAVES * 32 == BR * CH, "whole chunks a thread");
  constexpr int WM = WAVES / MATS;                // waves a matrix: wave w takes matrix w % MATS
  __shared__ __attribute__((aligned(16))) uint16_t xs[2][BR * RS];
  float* ep = reinterpret_cast<float*>(&xs[0][0]);   // after the K loop: the up waves' sums for the SwiGLU epilogue
  static_assert(MATS == 1 || WM * MT * NT * 8 * 32 * 4 <= sizeof(xs), "the epilogue fits the row buffers");
  const int lane = threadIdx.x & 31, c = lane & 15, h = lane >> 4;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  const int KG = K / GS, NB = N / (16 * NT * WM);
  const int mw = wave % MATS, tw = wave / MATS;   // the wave's matrix and n16-tile slot
  const int ST = super_tiles(N);
  const int per = items ? BR : 16;   // rows an item
  const int n_items = items ? __builtin_amdgcn_readfirstlane(counts[0]) : (rows + 15) / 16;
  const int units = n_items * NB;
  const size_t mat_words = (size_t)N * K / 8, mat_scales = (size_t)N * KG;
  for (int unit = blockIdx.x; unit < units; unit += gridDim.x) {
    const int it = unit / NB, cb = unit - it * NB;
    int e = 0, first = per * it, cnt = min(per, rows - per * it);
    if (items) {
      e = __builtin_amdgcn_readfirstlane(items[3 * it]);
      first = __builtin_amdgcn_readfirstlane(items[3 * it + 1]);
      cnt = __builtin_amdgcn_readfirstlane(items[3 * it + 2]);
      if (e == skip) continue;
    }
    const uint32_t* w[1][NT];
    const _Float16* s[1][NT];
#pragma unroll
    for (int j = 0; j < NT; ++j) {
      const size_t mi = (size_t)e * MATS + mw, tile = ((size_t)cb * WM + tw) * NT + j;
      const size_t at = (tile / ST * KG) * ST + tile % ST;   // (super tile, group 0, tile in it)
      w[0][j] = W + mi * mat_words + at * 2 * GS + c * 4;
      s[0][j] = S + mi * mat_scales + at * 16 + c;
    }
    for (int r0 = 0; r0 < cnt; r0 += BR) {
      const int live = min(MT, (cnt - r0 + 15) / 16);   // row tiles holding rows
      const uint16_t* xp[XN];   // the thread's chunks of the pass's rows (row 0 of X past the item: loaded, zeroed)
      bool xv[XN];
#pragma unroll
      for (int i = 0; i < XN; ++i) {
        const int q = threadIdx.x + WAVES * 32 * i, a = r0 + q / CH;
        xv[i] = a < cnt;
        const int p = xv[i] ? (items ? members[first + a] : first + a) : 0;
        xp[i] = X + (size_t)(slots ? p / slots : p) * x_stride + (q % CH) * 8;
      }
      v8f acc[MT][1][NT];
#pragma unroll
      for (int r = 0; r < MT; ++r)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[r][0][j] = v8f{};
      Stage<GS, NT, 1, XN> st[DS];
#pragma unroll
      for (int b = 0; b < DS - 1; ++b)
        if (b < KG) load_stage<GS, NT, 1, XN>(st[b], w, s, xp, min(b, KG - 1), ST);
      lds_barrier();   // the previous pass no longer reads the buffers
      for (int g0 = 0; g0 < KG; g0 += DS) {
#pragma unroll
        for (int b = 0; b < DS; ++b) {   // stage b, buffer b % 2 hold group g0 + b (known at compile time: no scratch)
          const int g = g0 + b;
          if (g >= KG) break;
#pragma unroll
          for (int i = 0; i < XN; ++i) {
            const int q = threadIdx.x + WAVES * 32 * i;
            *reinterpret_cast<u4v*>(xs[b & 1] + (q / CH) * RS + (q % CH) * 8) = xv[i] ? st[b].x[i] : u4v{};
          }
          lds_barrier();   // buffer b % 2 written; the other (group g - 1) read by every wave
          // unconditional (the last groups again past the end, from the cache): a load under a branch makes the
          // waitcnt pass wait at the join as if no load were in flight
          load_stage<GS, NT, 1, XN>(st[(b + DS - 1) % DS], w, s, xp, min(g + DS - 1, KG - 1), ST);
          compute_group<GS, NT, MT, 1, XN, RS>(acc, st[b], xs[b & 1], c);
        }
      }
      if constexpr (MATS == 2) {   // the up waves hand their sums to the gate waves of the same tiles
        lds_barrier();   // every wave is done with the row buffers
        if (mw == 1)
#pragma unroll
          for (int r = 0; r < MT; ++r)
#pragma unroll
            for (int j = 0; j < NT; ++j)
#pragma unroll
              for (int i = 0; i < 8; ++i) ep[(((tw * MT + r) * NT + j) * 8 + i) * 32 + lane] = acc[r][0][j][i];
        lds_barrier();
      }
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        if (r >= live || (MATS == 2 && mw == 1)) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int a = r0 + 16 * r + 2 * i + h;
          if (a >= cnt) continue;
          const size_t row = (size_t)(items ? members[first + a] : first + a) * out_stride;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int col = ((cb * WM + tw) * NT + j) * 16 + c;
            const float v = acc[r][0][j][i];
            if constexpr (EPI == 0) {
              reinterpret_cast<float*>(out)[row + col] = v;
            } else if constexpr (EPI == 2) {
              reinterpret_cast<uint16_t*>(out)[row + col] =
                  bf16(swiglu(v, ep[(((tw * MT + r) * NT + j) * 8 + i) * 32 + lane], limit));
            } else {
              reinterpret_cast<uint16_t*>(out)[row + col] = bf16(v);
            }
          }
        }
      }
    }
  }
}

}  // namespace tf_fn_int4

// Entry points tf_int4_<gs>_<nt>_<mt>_<mats>_<epi> (cuda_int4.zig symbol): gate/up SwiGLU (2 matrices, groups of
// 128), down fp32 / bf16 (groups of 128, or 64 at a TP=2 rank), the head (bf16 / fp32); NT 1 or 2 n16 tiles a wave;
// MT 1 row tile a pass (decode, items of 16 pairs: NT 1 measured best) or 2 (prompt, items of 64: NT 2). The same
// bits at every NT, MT. NT 4 measured slower (registers) and is left out.
#define TF_INT4(GS, NT, MT, MATS, EPI)                                                                              \
  extern "C" __global__ void __launch_bounds__(tf_fn_int4::WAVES * 32) tf_int4_##GS##_##NT##_##MT##_##MATS##_##EPI( \
      const uint16_t* X, int x_stride, int slots, const uint32_t* W, const _Float16* S, int K, int N,              \
      const int* items, const int* counts, const int* members, int rows, void* out, int out_stride, float limit,   \
      int skip) {                                                                                                  \
    tf_fn_int4::int4_body<GS, NT, MT, MATS, EPI>(X, x_stride, slots, W, S, K, N, items, counts, members, rows, out,  \
                                                 out_stride, limit, skip);                                         \
  }
#define TF_INT4_NT(GS, MATS, EPI) \
  TF_INT4(GS, 1, 1, MATS, EPI) TF_INT4(GS, 2, 1, MATS, EPI) TF_INT4(GS, 1, 2, MATS, EPI) TF_INT4(GS, 2, 2, MATS, EPI)
TF_INT4_NT(128, 2, 2)
TF_INT4_NT(128, 1, 0)
TF_INT4_NT(128, 1, 3)
TF_INT4_NT(64, 1, 0)
TF_INT4_NT(64, 1, 3)
