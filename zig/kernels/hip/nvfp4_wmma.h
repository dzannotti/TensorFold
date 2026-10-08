// NVFP4 routed experts on gfx11 (wave32 WMMA): the one per-unit body of fn_nvfp4_experts.hip, fn_nvfp4_shape.hip and
// fn_experts_prompt.hip (ours; HIP rewrites of the .cu files of those names, same arguments and block layout).
//
// Arithmetic of every output (row, column): for each 32-input group g in order, for each 16-input block h:
// p = v_wmma_f32_16x16x16_bf16(x row, the block's e2m1 weights as exact bf16, C = 0), acc = fmaf(p, e4m3 scale, acc);
// then acc *= the (expert, matrix) fp32 scale and experts.cuh's epilogue. The structure of nvfp4_expert_kernel's
// mma.sync / fmaf chain with the WMMA's own sum of the 16 exact products. A WMMA output (m, n) reads row m of A and
// column n of B only, so a row's bits never depend on the other rows, the row count, items, T, the grid or which of
// the three kernels runs it.
//
// Block layout (experts.py `pack`, cuda_layouts.zig packExpert, unchanged): per (expert, 32-column block, group) MS
// matrices of 36 uint4: 128 code words, then 64 scale bytes. Word (gq * 4 + t) * 4 + j holds column 8 j + gq, input
// 16 h + 4 t + 2 a + b at nibble 2 h + a + 4 b; scale byte 16 t' + 8 h + 2 j + c is column 8 j + 2 t' + c, block h.
//
// WMMA operands (gfx11 wave32): lane l holds A row l % 16 and B column l % 16, element i = input i of the block (lanes
// 16-31 repeat 0-15); D element i of lane l is (row 2 i + l / 16, column l % 16). A wave's unit is one plan item's rows
// x 16 or 32 columns (CT n16 tiles), each column's four code words loaded by the lanes that hold it in B.
#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_bf16.h>
#include <stdint.h>

#pragma clang fp contract(off)

namespace tf_nvfp4w {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));

constexpr int BLOCKW = 144;  // uint32 a (32 columns, 32 inputs) matrix block: 128 code words, 16 of scales

// e2m1 magnitude (code & 7) -> its bf16 pattern's high and low bytes, as v_perm_b32 tables (bytes 0-3 in the second)
constexpr uint32_t HI_A = 0x40404040u, HI_B = 0x3F3F3F00u, LO_A = 0xC0804000u, LO_B = 0xC0800000u;

// A code word -> d[h][a]: the bf16 pair (nibble 2 h + a, nibble 2 h + a + 4), i.e. inputs 16 h + 4 t + 2 a + {0, 1}.
__device__ __forceinline__ void decode(uint32_t w, uint32_t (&d)[2][2]) {
  const uint32_t ml = w & 0x07070707u, mh = (w >> 4) & 0x07070707u;
  const uint32_t hl = __builtin_amdgcn_perm(HI_A, HI_B, ml) | ((w << 4) & 0x80808080u);
  const uint32_t ll = __builtin_amdgcn_perm(LO_A, LO_B, ml);
  const uint32_t hh = __builtin_amdgcn_perm(HI_A, HI_B, mh) | (w & 0x80808080u);
  const uint32_t lh = __builtin_amdgcn_perm(LO_A, LO_B, mh);
  d[0][0] = __builtin_amdgcn_perm(hl, ll, 0x06020400u);
  d[1][0] = __builtin_amdgcn_perm(hl, ll, 0x07030501u);
  d[0][1] = __builtin_amdgcn_perm(hh, lh, 0x06020400u);
  d[1][1] = __builtin_amdgcn_perm(hh, lh, 0x07030501u);
}

// e4m3fn -> fp32, exact (NaN for 0x7F / 0xFF, as __nv_cvt_fp8_to_halfraw)
__device__ __forceinline__ float e4m3f(uint32_t b) {
  const uint32_t e = (b >> 3) & 15u, m = b & 7u;
  float v = e ? __uint_as_float(((e + 120u) << 23) | (m << 20)) : (float)m * 0x1p-9f;
  if ((b & 0x7Fu) == 0x7Fu) v = __uint_as_float(0x7FC00000u);
  return (b & 0x80u) ? -v : v;
}

__device__ __forceinline__ uint16_t bf16(float x) {  // round to nearest even (__float2bfloat16_rn)
  const uint32_t u = __float_as_uint(x);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

__device__ __forceinline__ float bf(float x) { return __uint_as_float((uint32_t)bf16(x) << 16); }

// experts.cuh's SwiGLU
__device__ __forceinline__ float swiglu(float g, float u, float limit) {
  float gv = bf(g), uv = bf(u);
  if (limit > 0.f) {
    gv = fminf(gv, limit);
    uv = fminf(fmaxf(uv, -limit), limit);
  }
  return bf(gv / (1.f + expf(-gv))) * uv;
}

// A group's loads: the lane's CT columns' 4 code words and 2 scale bytes of each matrix, T row tiles' 32 inputs.
template <int M, int T, int CT>
struct Stage {
  uint32_t w[M][CT][4];
  uint32_t s[M][CT][2];
  u4v x[T][4];
};

// woff / soff: the lane's first column's code word and scale byte offsets (column + 16: + 2 words, + 4 bytes)
template <int M, int T, int CT, int MS>
__device__ __forceinline__ void load(Stage<M, T, CT>& st, const uint32_t* blk, int g, int woff, int soff,
                                     const uint16_t* const (&xr)[T], const bool (&xv)[T]) {
  const uint32_t* b = blk + (size_t)g * (MS * BLOCKW);
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int ct = 0; ct < CT; ++ct) {
#pragma unroll
      for (int t = 0; t < 4; ++t)
        st.w[m][ct][t] = __builtin_nontemporal_load(b + m * BLOCKW + woff + 2 * ct + 4 * t);
      const uint8_t* sb = reinterpret_cast<const uint8_t*>(b + m * BLOCKW) + soff + 4 * ct;
      st.s[m][ct][0] = sb[0];
      st.s[m][ct][1] = sb[8];
    }
#pragma unroll
  for (int r = 0; r < T; ++r) {
    const u4v* p = reinterpret_cast<const u4v*>(xr[r] + g * 32);
#pragma unroll
    for (int q = 0; q < 4; ++q) st.x[r][q] = xv[r] ? p[q] : u4v{0u, 0u, 0u, 0u};
  }
}

template <int M, int T, int CT>
__device__ __forceinline__ void compute(v8f (&acc)[T][M][CT], const Stage<M, T, CT>& st) {
#pragma unroll
  for (int m = 0; m < M; ++m)
#pragma unroll
    for (int ct = 0; ct < CT; ++ct) {
      uint32_t d[4][2][2];
#pragma unroll
      for (int t = 0; t < 4; ++t) decode(st.w[m][ct][t], d[t]);
#pragma unroll
      for (int h = 0; h < 2; ++h) {
        const u8v bv = {d[0][h][0], d[0][h][1], d[1][h][0], d[1][h][1],
                        d[2][h][0], d[2][h][1], d[3][h][0], d[3][h][1]};
        const v16s b = (v16s)bv;
        const float s = e4m3f(st.s[m][ct][h]);
#pragma unroll
        for (int r = 0; r < T; ++r) {
          const u8v av = {st.x[r][2 * h][0], st.x[r][2 * h][1], st.x[r][2 * h][2], st.x[r][2 * h][3],
                          st.x[r][2 * h + 1][0], st.x[r][2 * h + 1][1], st.x[r][2 * h + 1][2], st.x[r][2 * h + 1][3]};
          const v8f p = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32((v16s)av, b, v8f{});
#pragma unroll
          for (int i = 0; i < 8; ++i) acc[r][m][ct][i] = __builtin_fmaf(p[i], s, acc[r][m][ct][i]);
        }
      }
    }
}

// X, x_stride, slots, W, scale, KG, NB, items, counts, members, out, N, limit, skip: nvfp4_expert_kernel's arguments.
// EPI 0 fp32 out, 2 SwiGLU(gate, up) bf16, 3 bf16, 5 SwiGLU(bf16 `gate` [pairs, N], this pass) bf16. MS matrices are
// stored a block group, this pass reads M of them from MOFF. A wave's unit: an item's rows x 16 CT columns, T row
// tiles a pass, D groups' loads in flight; waves `wave`, `wave + waves`, .. take the units.
template <int M, int EPI, int T, int CT, int D, int MS = M, int MOFF = 0>
__device__ __forceinline__ void body(const uint16_t* __restrict__ X, int x_stride, int slots,
                                     const uint32_t* __restrict__ W, const float* __restrict__ scale, int KG, int NB,
                                     const int* __restrict__ items, const int* __restrict__ counts,
                                     const int* __restrict__ members, void* __restrict__ out, int N, float limit,
                                     int skip, const uint16_t* __restrict__ gate, int wave, int waves) {
  constexpr int PARTS = 2 / CT;
  const int lane = threadIdx.x & 31, n = lane & 15, hi = lane >> 4;
  const int units = counts[0] * NB * PARTS;
  for (int unit = wave; unit < units; unit += waves) {
    const int whole = unit / PARTS, part = unit - whole * PARTS;
    const int it = whole / NB, cb = whole - it * NB;
    const int e = items[3 * it], first = items[3 * it + 1], cnt = items[3 * it + 2];
    if (e == skip) continue;
    const uint32_t* blk = W + ((size_t)e * NB + cb) * (size_t)KG * (MS * BLOCKW) + MOFF * BLOCKW;
    const int c = part * 16 + n;  // the lane's first column in the 32-column block (B and D); then c + 16
    const int woff = (c & 7) * 16 + (c >> 3);
    const int soff = 512 + 16 * ((c & 7) >> 1) + 2 * (c >> 3) + (c & 1);
    const int col = cb * 32 + c;
    for (int p0 = 0; p0 < cnt; p0 += 16 * T) {
      const uint16_t* xr[T];
      bool xv[T];
#pragma unroll
      for (int r = 0; r < T; ++r) {
        const int i = p0 + 16 * r + n;
        xv[r] = i < cnt;
        const int pr = xv[r] ? members[first + i] : 0;
        xr[r] = X + (size_t)(slots ? pr / slots : pr) * x_stride;
      }
      v8f acc[T][M][CT];
#pragma unroll
      for (int r = 0; r < T; ++r)
#pragma unroll
        for (int m = 0; m < M; ++m)
#pragma unroll
          for (int ct = 0; ct < CT; ++ct) acc[r][m][ct] = v8f{};
      // a ring of D stages: group g + D - 1 loads while group g computes
      Stage<M, T, CT> st[D];
#pragma unroll
      for (int d = 0; d < D - 1; ++d)
        if (d < KG) load<M, T, CT, MS>(st[d], blk, d, woff, soff, xr, xv);
      for (int g0 = 0; g0 < KG; g0 += D) {
#pragma unroll
        for (int d = 0; d < D; ++d) {
          const int g = g0 + d;
          if (g < KG) {
            if (g + D - 1 < KG) load<M, T, CT, MS>(st[(d + D - 1) % D], blk, g + D - 1, woff, soff, xr, xv);
            compute<M, T, CT>(acc, st[d]);
          }
        }
      }
      float gs[M];
#pragma unroll
      for (int m = 0; m < M; ++m) gs[m] = scale[e * MS + MOFF + m];
#pragma unroll
      for (int r = 0; r < T; ++r)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          const int ii = p0 + 16 * r + 2 * i + hi;
          if (ii >= cnt) continue;
          const size_t row = (size_t)members[first + ii] * N + col;
#pragma unroll
          for (int ct = 0; ct < CT; ++ct) {
            const size_t o = row + 16 * ct;
            const float a0 = acc[r][0][ct][i] * gs[0];
            if constexpr (EPI == 0) {
              reinterpret_cast<float*>(out)[o] = a0;
            } else {
              float v;
              if constexpr (EPI == 2) v = swiglu(a0, acc[r][M - 1][ct][i] * gs[M - 1], limit);
              else if constexpr (EPI == 5) v = swiglu(__uint_as_float((uint32_t)gate[o] << 16), a0, limit);
              else v = a0;
              reinterpret_cast<uint16_t*>(out)[o] = bf16(v);
            }
          }
        }
    }
  }
}

}  // namespace tf_nvfp4w
