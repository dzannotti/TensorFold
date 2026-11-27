// SPDX-License-Identifier: MIT
// Routed-expert prompt tiles (gate|up pair + down) with register-resident NVFP4 weights; the Loom kernels' ABI.
// Kernel structure adapted from gufo (MIT, Copyright (c) 2026 gufo contributors),
// src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp RoutedPairF16Kernel / RoutedDownF16Kernel:
// activations the only LDS stage, one K16 half decoded a lane and v_permlanex16 for the other, gate and up rows in
// one wave tile, SwiGLU in registers. Changed: NVFP4 g16 decode from the row-major bake, bf16 WMMA, plan items.
#ifndef TF_QUANT_GUFO_ROUTED_HPP_
#define TF_QUANT_GUFO_ROUTED_HPP_

#include <hip/hip_runtime.h>
#include <algorithm>
#include <cstdint>

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef unsigned v8u __attribute__((ext_vector_type(8)));
typedef _Float16 gr_h2 __attribute__((ext_vector_type(2)));
typedef float gr_f2 __attribute__((ext_vector_type(2)));
typedef unsigned gr_u2 __attribute__((ext_vector_type(2)));

__device__ __forceinline__ unsigned gr_SwapHalves(unsigned v) {
  unsigned old;
  asm("" : "=v"(old));
  return (unsigned)__builtin_amdgcn_permlanex16((int)old, (int)v, 0x76543210, 0xfedcba98, false, false);
}
__device__ __forceinline__ v8f gr_Wmma(v16s a, v16s b, v8f c) { return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c); }
__device__ __forceinline__ float gr_SigmoidF(float x) { return 1.0f / (1.0f + __expf(-x)); }
__device__ __forceinline__ unsigned short gr_Bf16(float v) {
  const unsigned b = __builtin_bit_cast(unsigned, v);
  return (unsigned short)((b + ((b >> 16) & 1u) + 0x7fffu) >> 16);
}
__device__ __forceinline__ unsigned gr_scale16(unsigned e) {
  const unsigned b = e | (e << 16);
  const unsigned bits = ((b << 7) & 0x3f803f80u) | ((b << 8) & 0x80008000u);
  const gr_h2 s = __builtin_bit_cast(gr_h2, bits) * __builtin_bit_cast(gr_h2, 0x5c005c00u);
  return __builtin_bit_cast(unsigned, s);
}
__device__ __forceinline__ unsigned gr_pr(unsigned w, int P, unsigned sc) {
  unsigned r;
  if (P < 3) r = ((w << (9 - 4 * P)) & 0x0e000e00u) | ((w << (12 - 4 * P)) & 0x80008000u);
  else r = ((w >> 3) & 0x0e000e00u) | (w & 0x80008000u);
  const gr_h2 v = __builtin_bit_cast(gr_h2, r) * __builtin_bit_cast(gr_h2, sc);
  const gr_u2 bits = __builtin_bit_cast(gr_u2, __builtin_convertvector(v, gr_f2));
  return __builtin_amdgcn_perm(bits.y, bits.x, 0x07060302u);
}

#define GR_BN 128
#define GR_BK 2
#define GR_TT (GR_BN / 16)
#define GR_AST (GR_BN + 1)
#define GR_AFETCH (GR_BN * GR_BK * 4 / 256)

// Shared body: `wrow_w`/`wrow_s` = this lane's weight row, f_c = the K16 half it decodes.
#define GR_ROUTED_BODY(KDIM)                                                                                              \
  const int num_kb = (KDIM) / 32;                                                                                      \
  const uint16_t* a_src[GR_AFETCH];                                                                                       \
  int a_slot[GR_AFETCH];                                                                                                  \
  for (int i = 0; i < GR_AFETCH; ++i) {                                                                                   \
    const int chunk = tid + (i * 256);                                                                                 \
    const int t = chunk / (GR_BK * 4), sub = chunk % (GR_BK * 4);                                                            \
    const int c_row = t_local + t;                                                                                     \
    const int src = c_row < bucket_rows ? (int)((unsigned)members[first + c_row] / (unsigned)x_div) : -1;                                          \
    a_src[i] = src >= 0 ? x + ((size_t)src * (KDIM)) + (sub * 8) : nullptr;                                            \
    a_slot[i] = (sub * GR_AST) + t;                                                                                       \
  }                                                                                                                    \
  gr_u2 w_codes[2][GR_BK];                                                                                                   \
  unsigned w_scale[2][GR_BK];                                                                                             \
  uint4 a_data[GR_AFETCH];                                                                                                \
  for (int i = 0; i < GR_AFETCH; ++i) a_data[i] = make_uint4(0u, 0u, 0u, 0u);                                             \
  const auto fetch_stage = [&]<int S>(int kb0) {                                                                       \
    for (int b = 0; b < GR_BK; ++b) {                                                                                     \
      __builtin_memcpy(&w_codes[S][b], wrow_w + (kb0 + b) * 4 + 2 * f_c, 8);                                           \
      w_scale[S][b] = wrow_s[(kb0 + b) * 2 + f_c];                                                                     \
    }                                                                                                                  \
    for (int i = 0; i < GR_AFETCH; ++i)                                                                                   \
      if (a_src[i] != nullptr) a_data[i] = *reinterpret_cast<const uint4*>(a_src[i] + (kb0 * 32));                     \
  };                                                                                                                   \
  v8f acc[GR_TT];                                                                                                         \
  for (int j = 0; j < GR_TT; ++j) acc[j] = v8f{0, 0, 0, 0, 0, 0, 0, 0};                                                   \
  const auto compute_stage = [&]<int S>() {                                                                            \
    for (int kb = 0; kb < GR_BK; ++kb) {                                                                                  \
      const unsigned sc = gr_scale16(w_scale[S][kb]);                                                                     \
      const gr_u2 c = w_codes[S][kb];                                                                                     \
      const v8u own = {gr_pr(c.x, 0, sc), gr_pr(c.x, 1, sc), gr_pr(c.x, 2, sc), gr_pr(c.x, 3, sc),                                 \
                       gr_pr(c.y, 0, sc), gr_pr(c.y, 1, sc), gr_pr(c.y, 2, sc), gr_pr(c.y, 3, sc)};                                \
      v8u peer;                                                                                                        \
      for (int i = 0; i < 8; ++i) peer[i] = gr_SwapHalves(own[i]);                                                        \
      const v16s a_lo = __builtin_bit_cast(v16s, own);                                                                 \
      const v16s a_hi = __builtin_bit_cast(v16s, peer);                                                                \
      for (int j = 0; j < GR_TT; ++j) {                                                                                   \
        asm volatile("" ::: "memory");                                                                                 \
        if (j >= live_tok_tiles) continue;                                                                             \
        const uint4* frag = s_act + ((kb * 4) * GR_AST) + (j * 16) + sub_lane;                                            \
        uint4 b[4];                                                                                                    \
        for (int q = 0; q < 4; ++q) b[q] = frag[q * GR_AST];                                                              \
        v16s b_lo, b_hi;                                                                                               \
        __builtin_memcpy(&b_lo, &b[0], 32);                                                                            \
        __builtin_memcpy(&b_hi, &b[2], 32);                                                                            \
        acc[j] = gr_Wmma(a_lo, b_lo, acc[j]);                                                                             \
        acc[j] = gr_Wmma(a_hi, b_hi, acc[j]);                                                                             \
      }                                                                                                                \
    }                                                                                                                  \
  };                                                                                                                   \
  const auto run_stage = [&]<int S>(int kb0) {                                                                         \
    for (int i = 0; i < GR_AFETCH; ++i) s_act[a_slot[i]] = a_data[i];                                                     \
    __syncthreads();                                                                                                   \
    if (kb0 + GR_BK < num_kb) fetch_stage.template operator()<1 - S>(kb0 + GR_BK);                                           \
    compute_stage.template operator()<S>();                                                                            \
    __syncthreads();                                                                                                   \
  };                                                                                                                   \
  fetch_stage.template operator()<0>(0);                                                                               \
  for (int kb0 = 0; kb0 < num_kb; kb0 += 2 * GR_BK) {                                                                     \
    run_stage.template operator()<0>(kb0);                                                                             \
    if (kb0 + GR_BK < num_kb) run_stage.template operator()<1>(kb0 + GR_BK);                                                 \
  }


#define GR_TILE_HEAD                                                                                                   \
  const unsigned z = blockIdx.z;                                                                                       \
  const int expert = items[3 * z], first = items[3 * z + 1], bucket_rows = items[3 * z + 2];                           \
  const int t_local = (int)blockIdx.y * GR_BN;                                                                         \
  if (t_local >= bucket_rows) return;                                                                                  \
  const int live_tok_tiles = std::min(GR_TT, (bucket_rows - t_local + 15) / 16);                                       \
  const int tid = (int)threadIdx.x;                                                                                    \
  const int wave_id = tid >> 5, lane_id = tid & 31, sub_lane = lane_id & 15, half_id = lane_id >> 4;                   \
  const int f_c = half_id ^ (sub_lane & 1);

// act = bf16(silu(gate) x up): grid (640 / 64, row tiles of 128, items).
extern "C" __launch_bounds__(256) __global__ void gufo_expert_gate_up(const uint16_t* __restrict__ x,
    const unsigned* __restrict__ words, const uint8_t* __restrict__ scales, const float* __restrict__ globals,
    const int* __restrict__ items, const int* __restrict__ members, uint16_t* __restrict__ out, int x_div, int n, int k) {
  constexpr int K = 2560, N = 640;
  if (n != N || k != K) return;
  constexpr int kRows = 64, kPairStride = kRows + 2;
  constexpr int kActBytes = GR_BK * 4 * GR_AST * 16, kEpiBytes = GR_BN * kPairStride * 2;
  __shared__ __attribute__((aligned(16))) uint8_t lds[kActBytes > kEpiBytes ? kActBytes : kEpiBytes];
  auto* s_act = reinterpret_cast<uint4*>(lds);
  GR_TILE_HEAD
  const int r_block = (int)blockIdx.x * kRows;
  const int row = r_block + (wave_id * 8) + (sub_lane % 8) + (sub_lane < 8 ? 0 : N);
  const unsigned* wrow_w = words + ((size_t)expert * 2 * N + row) * (K / 8);
  const uint8_t* wrow_s = scales + ((size_t)expert * 2 * N + row) * (K / 16);
  GR_ROUTED_BODY(K)
  const float gsg = globals[2 * expert] * 16384.0f, gsu = globals[2 * expert + 1] * 16384.0f;
  uint16_t* scratch = reinterpret_cast<uint16_t*>(lds);
  for (int j = 0; j < GR_TT; ++j) {
    if (j >= live_tok_tiles) continue;
    for (int l = 0; l < 4; ++l) {
      const float g = acc[j][l] * gsg;
      float product = (acc[j][l + 4] * gsu) * g;
      asm volatile("" : "+v"(product));
      float value = product * gr_SigmoidF(g);
      asm volatile("" : "+v"(value));
      scratch[(j * 16 + sub_lane) * kPairStride + wave_id * 8 + 2 * l + half_id] = gr_Bf16(value);
    }
  }
  __syncthreads();
  for (int unit = 0; unit < GR_BN * kRows / (2 * 256); ++unit) {
    const int flat = (unit * 256 + tid) * 2;
    const int t = t_local + flat / kRows;
    const int r = r_block + flat % kRows;
    if (t < bucket_rows)
      *reinterpret_cast<uint32_t*>(out + (size_t)(unsigned)members[first + t] * N + r) =
          *reinterpret_cast<const uint32_t*>(scratch + (flat / kRows) * kPairStride + flat % kRows);
  }
}

// Down sums: fp32 rows, or bf16 rows when n is negative; grid (2560 / 128, row tiles of 128, items).
extern "C" __launch_bounds__(256) __global__ void gufo_expert_down(const uint16_t* __restrict__ x,
    const unsigned* __restrict__ words, const uint8_t* __restrict__ scales, const float* __restrict__ globals,
    const int* __restrict__ items, const int* __restrict__ members, float* __restrict__ out, int x_div, int n, int k) {
  constexpr int K = 640, N = 2560;
  const bool half = n < 0;
  if ((half ? -n : n) != N || k != K) return;
  constexpr int kOutStride = 128 + 4, kEpiTokens = 32;
  constexpr int kHalfStride = 128 + 8, kHalfTokens = 64;
  constexpr int kActBytes = GR_BK * 4 * GR_AST * 16, kEpiBytes = kHalfTokens * kHalfStride * 2;
  static_assert(kEpiBytes >= kEpiTokens * kOutStride * 4 && kEpiBytes >= kActBytes);
  __shared__ __attribute__((aligned(16))) uint8_t lds[kEpiBytes];
  auto* s_act = reinterpret_cast<uint4*>(lds);
  GR_TILE_HEAD
  const int r_block = (int)blockIdx.x * 128;
  const int row = r_block + (wave_id * 16) + sub_lane;
  const unsigned* wrow_w = words + ((size_t)expert * N + row) * (K / 8);
  const uint8_t* wrow_s = scales + ((size_t)expert * N + row) * (K / 16);
  GR_ROUTED_BODY(K)
  const float gs = globals[expert] * 16384.0f;
  if (half) {
    uint16_t* scratch = reinterpret_cast<uint16_t*>(lds);
    uint16_t* out16 = reinterpret_cast<uint16_t*>(out);
    for (int part = 0; part < GR_BN / kHalfTokens; ++part) {
      if (part * (kHalfTokens / 16) >= live_tok_tiles) break;
      if (part > 0) __syncthreads();
      for (int jj = 0; jj < kHalfTokens / 16; ++jj) {
        const int j = part * (kHalfTokens / 16) + jj;
        for (int i = 0; i < 8; ++i)
          scratch[((jj * 16) + sub_lane) * kHalfStride + (wave_id * 16) + (2 * i) + half_id] = gr_Bf16(acc[j][i] * gs);
      }
      __syncthreads();
      for (int unit = 0; unit < kHalfTokens * 16 / 256; ++unit) {
        const int flat = (unit * 256) + tid;
        const int tr = flat / 16, chunk = flat % 16;
        const int t = t_local + (part * kHalfTokens) + tr;
        if (t >= bucket_rows) continue;
        *reinterpret_cast<uint4*>(out16 + (size_t)(unsigned)members[first + t] * N + r_block + chunk * 8) =
            *reinterpret_cast<const uint4*>(scratch + (tr * kHalfStride) + (chunk * 8));
      }
    }
    return;
  }
  float* scratch = reinterpret_cast<float*>(lds);
  for (int part = 0; part < GR_BN / kEpiTokens; ++part) {
    if (part * (kEpiTokens / 16) >= live_tok_tiles) break;
    if (part > 0) __syncthreads();
    for (int jj = 0; jj < kEpiTokens / 16; ++jj) {
      const int j = part * (kEpiTokens / 16) + jj;
      for (int i = 0; i < 8; ++i)
        scratch[((jj * 16) + sub_lane) * kOutStride + (wave_id * 16) + (2 * i) + half_id] = acc[j][i] * gs;
    }
    __syncthreads();
    for (int unit = 0; unit < kEpiTokens * 32 / 256; ++unit) {
      const int flat = (unit * 256) + tid;
      const int tr = flat / 32, chunk = flat % 32;
      const int t = t_local + (part * kEpiTokens) + tr;
      if (t >= bucket_rows) continue;
      *reinterpret_cast<float4*>(out + (size_t)(unsigned)members[first + t] * N + r_block + chunk * 4) =
          *reinterpret_cast<const float4*>(scratch + (tr * kOutStride) + (chunk * 4));
    }
  }
}

#endif  // TF_QUANT_GUFO_ROUTED_HPP_
