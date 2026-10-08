// GPTQ int4 W4A16 matmuls of the INT4-AutoRound Flash Next checkpoint on gfx11 (wave32 WMMA): the routed experts
// (grouped by the experts plan, gate/up with the SwiGLU epilogue, down) and the lm_head (dense rows). Ours (no Python
// counterpart); the HIP rewrite of zig/kernels/cuda/fn_int4.cu with the same arguments.
//
// Arithmetic: a weight (q - 8) * scale, q - 8 exact in bf16. Per group of GS inputs one output's products run in one
// fp32 WMMA chain (v_wmma_f32_16x16x16_bf16, the group's k16 steps in order, C = 0 at the first), then
// acc = fmaf(group sum, scale, acc) in group order. A WMMA's output (m, n) reads row m of A and column n of B only,
// so a row's bits never depend on the other rows, the row count, the plan's items, NT, MT or the grid.
//
// Packed layout (cuda_int4.zig packWords): words uint32 [N/16][K/GS][GS/32][16 columns][4]: n16 tile, group, k32
// chunk, column c, then the chunk's 32 inputs of column c as 4 words of 8 nibbles, input i at nibble i, each the
// two's complement 4-bit q - 8 (GPTQ's code xor 8). Scales fp16 [N/16][K/GS][16].
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
#ifndef TF_INT4_ABL
#define TF_INT4_ABL 0   // bench ablations: 1 no dequant, 2 no WMMA, 3 no activation loads
#endif

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

// A group's operands in flight: every (matrix, n16 tile)'s words and scales, and with one row tile a pass (XS) the
// tile's activations too (loaded with the weights, so waiting for this stage never waits for the next one's loads).
template <int GS, int NT, int MATS, bool XS>
struct Stage {
  u4v w[MATS][NT][GS / 32];
  float s[MATS][NT];
  u4v x[XS ? GS / 8 : 1];
};

__device__ __forceinline__ v16s xfrag(const uint16_t* p, bool v) {
  u8v r = u8v{};
  if (v && TF_INT4_ABL != 3) {
    const u4v lo = reinterpret_cast<const u4v*>(p)[0], hi = reinterpret_cast<const u4v*>(p)[1];
    r = u8v{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
  }
  return (v16s)r;
}

template <int GS, int NT, int MATS, bool XS>
__device__ __forceinline__ void load_stage(Stage<GS, NT, MATS, XS>& st, const uint32_t* const (&w)[MATS][NT],
                                           const _Float16* const (&s)[MATS][NT], const uint16_t* xr, bool xv, int g) {
  if constexpr (XS)
#pragma unroll
    for (int q = 0; q < GS / 8; ++q)
      st.x[q] = xv && TF_INT4_ABL != 3 ? reinterpret_cast<const u4v*>(xr + g * GS)[q] : u4v{};
#pragma unroll
  for (int m = 0; m < MATS; ++m)
#pragma unroll
    for (int j = 0; j < NT; ++j) {
#pragma unroll
      for (int q = 0; q < GS / 32; ++q)
        st.w[m][j][q] = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(w[m][j] + g * 2 * GS + q * 64));
      st.s[m][j] = (float)s[m][j][g * 16];
    }
}

// One group: a fresh WMMA chain over its k16 steps for every (row tile, matrix, n16 tile), then the scale's fmaf.
template <int GS, int NT, int MT, int MATS, bool XS>
__device__ __forceinline__ void compute_group(v8f (&acc)[MT][MATS][NT], const Stage<GS, NT, MATS, XS>& cur,
                                              const uint16_t* const (&xr)[MT], const bool (&xv)[MT], int live, int g) {
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
      if constexpr (XS) {
        const u4v lo = cur.x[2 * k], hi = cur.x[2 * k + 1];
        a[r] = (v16s)u8v{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
      } else {
        a[r] = xfrag(xr[r] + g * GS + 16 * k, xv[r]);
      }
    }
#pragma unroll
    for (int m = 0; m < MATS; ++m)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const u4v q = cur.w[m][j][k / 2];
#if TF_INT4_ABL == 1
        const v16s b = (v16s)u8v{q[0], q[1], q[2], q[3], q[0], q[1], q[2], q[3]};
#else
        const v16s b = dq(q[2 * (k & 1)], q[2 * (k & 1) + 1]);
#endif
#pragma unroll
        for (int r = 0; r < MT; ++r)
#if TF_INT4_ABL == 2
          if (r < live) d[r][m][j][0] += __builtin_bit_cast(float, ((u8v)b)[0] ^ ((u8v)a[r])[k & 7]);
#else
          if (r < live) d[r][m][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a[r], b, d[r][m][j]);
#endif
      }
  }
#pragma unroll
  for (int r = 0; r < MT; ++r)
#pragma unroll
    for (int m = 0; m < MATS; ++m)
#pragma unroll
      for (int j = 0; j < NT; ++j)
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[r][m][j][i] = fmaf(d[r][m][j][i], cur.s[m][j], acc[r][m][j][i]);
}

// A wave a unit: (item, NT n16 tiles of every matrix). Items (expert, first member, count) from the experts plan, or,
// with `items` null, the dense rows [16 i, 16 i + 16) of `rows` (the head). `slots`: member p reads input row
// p / slots (gate/up), 0: row p (down). A pass takes MT row tiles of 16 against one decode of the weights.
template <int GS, int NT, int MT, int MATS, int EPI>
__device__ __forceinline__ void int4_body(const uint16_t* __restrict__ X, int x_stride, int slots,
                                          const uint32_t* __restrict__ W, const _Float16* __restrict__ S, int K, int N,
                                          const int* __restrict__ items, const int* __restrict__ counts,
                                          const int* __restrict__ members, int rows, void* __restrict__ out,
                                          int out_stride, float limit, int skip) {
  const int lane = threadIdx.x & 31, c = lane & 15, h = lane >> 4;
  const int KG = K / GS, NB = N / (16 * NT);
  const int per = items ? 16 * MT : 16;   // rows an item
  const int n_items = items ? __builtin_amdgcn_readfirstlane(counts[0]) : (rows + 15) / 16;
  const int units = n_items * NB;
  const size_t mat_words = (size_t)N * K / 8, mat_scales = (size_t)N * KG;
  const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
  for (int unit = blockIdx.x * WAVES + wave; unit < units; unit += gridDim.x * WAVES) {
    const int it = unit / NB, cb = unit - it * NB;
    int e = 0, first = per * it, cnt = min(per, rows - per * it);
    if (items) {
      e = __builtin_amdgcn_readfirstlane(items[3 * it]);
      first = __builtin_amdgcn_readfirstlane(items[3 * it + 1]);
      cnt = __builtin_amdgcn_readfirstlane(items[3 * it + 2]);
      if (e == skip) continue;
    }
    const uint32_t* w[MATS][NT];
    const _Float16* s[MATS][NT];
#pragma unroll
    for (int m = 0; m < MATS; ++m)
#pragma unroll
      for (int j = 0; j < NT; ++j) {
        const size_t mi = (size_t)e * MATS + m, tile = (size_t)cb * NT + j;
        w[m][j] = W + mi * mat_words + tile * KG * 2 * GS + c * 4;
        s[m][j] = S + mi * mat_scales + tile * KG * 16 + c;
      }
    for (int r0 = 0; r0 < cnt; r0 += 16 * MT) {
      const uint16_t* xr[MT];
      bool xv[MT];
      int live = 0;   // row tiles holding rows (wave-uniform)
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        const int a = r0 + 16 * r + c;
        xv[r] = a < cnt;
        const int p = xv[r] ? (items ? members[first + a] : first + a) : 0;
        xr[r] = X + (size_t)(slots ? p / slots : p) * x_stride;
        live += r0 + 16 * r < cnt;
      }
      v8f acc[MT][MATS][NT];
#pragma unroll
      for (int r = 0; r < MT; ++r)
#pragma unroll
        for (int m = 0; m < MATS; ++m)
#pragma unroll
          for (int j = 0; j < NT; ++j) acc[r][m][j] = v8f{};
      constexpr bool XS = MT == 1;
      Stage<GS, NT, MATS, XS> st[2];
      load_stage<GS, NT, MATS, XS>(st[0], w, s, xr[0], xv[0], 0);
      for (int g0 = 0; g0 < KG; g0 += 2) {
#pragma unroll
        for (int b = 0; b < 2; ++b) {   // stage b holds group g0 + b (indices known at compile time: no scratch)
          const int g = g0 + b;
          if (g >= KG) break;
          if (g + 1 < KG) load_stage<GS, NT, MATS, XS>(st[b ^ 1], w, s, xr[0], xv[0], g + 1);
          compute_group<GS, NT, MT, MATS, XS>(acc, st[b], xr, xv, live, g);
        }
      }
#pragma unroll
      for (int r = 0; r < MT; ++r) {
        if (r >= live) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int a = r0 + 16 * r + 2 * i + h;
          if (a >= cnt) continue;
          const size_t row = (size_t)(items ? members[first + a] : first + a) * out_stride;
#pragma unroll
          for (int j = 0; j < NT; ++j) {
            const int col = (cb * NT + j) * 16 + c;
            const float v = acc[r][0][j][i];
            if constexpr (EPI == 0) {
              reinterpret_cast<float*>(out)[row + col] = v;
            } else if constexpr (EPI == 2) {
              reinterpret_cast<uint16_t*>(out)[row + col] = bf16(swiglu(v, acc[r][MATS - 1][j][i], limit));
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

// Entry points tf_int4_<gs>_<nt>_<mt>_<mats>_<epi> (cuda_int4.zig hipSymbol): gate/up SwiGLU (2 matrices, groups of
// 128), down fp32 / bf16 (groups of 128, or 64 at a TP=2 rank), the head (bf16 / fp32); NT 1, 2, 4 n16 tiles a
// wave; MT 1 row tile a pass (decode: items of 16 pairs) or 4 (prompt: items of 64). The same bits at every NT, MT.
#define TF_INT4(GS, NT, MT, MATS, EPI)                                                                              \
  extern "C" __global__ void __launch_bounds__(tf_fn_int4::WAVES * 32) tf_int4_##GS##_##NT##_##MT##_##MATS##_##EPI( \
      const uint16_t* X, int x_stride, int slots, const uint32_t* W, const _Float16* S, int K, int N,              \
      const int* items, const int* counts, const int* members, int rows, void* out, int out_stride, float limit,   \
      int skip) {                                                                                                  \
    tf_fn_int4::int4_body<GS, NT, MT, MATS, EPI>(X, x_stride, slots, W, S, K, N, items, counts, members, rows, out,  \
                                                 out_stride, limit, skip);                                         \
  }
#define TF_INT4_NT(GS, MATS, EPI)                                                                       \
  TF_INT4(GS, 1, 1, MATS, EPI) TF_INT4(GS, 2, 1, MATS, EPI) TF_INT4(GS, 4, 1, MATS, EPI)                \
  TF_INT4(GS, 1, 4, MATS, EPI) TF_INT4(GS, 2, 4, MATS, EPI) TF_INT4(GS, 4, 4, MATS, EPI)
TF_INT4_NT(128, 2, 2)
TF_INT4_NT(128, 1, 0)
TF_INT4_NT(128, 1, 3)
TF_INT4_NT(64, 1, 0)
TF_INT4_NT(64, 1, 3)
