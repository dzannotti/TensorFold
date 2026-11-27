// SPDX-License-Identifier: MIT
// Adapted from gufo (MIT, Copyright (c) 2026 gufo contributors): W8A8BlockedWmmaGEMMKernel<64, 128, 4, 2, 4, kHcDown>,
// GroupedRasterTile and the tiled Q8 activation layout, src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp.
// Changes: TensorFold's gates epilogue (bf16 silu rows, the inject rows under them), no tail kernel.
#pragma once

#include "../ops/common.hpp"

namespace {

struct HcQ8Block {
    uint16_t d;
    int8_t qs[32];
} __attribute__((packed));
typedef int hc_q8_i32x4 __attribute__((__vector_size__(4 * sizeof(int))));
typedef int hc_q8_i32x8 __attribute__((__vector_size__(8 * sizeof(int))));
typedef float hc_q8_f32x4 __attribute__((ext_vector_type(4)));
__device__ __forceinline__ hc_q8_i32x8 hc_q8_wmma(hc_q8_i32x4 a, hc_q8_i32x4 b, hc_q8_i32x8 c) {
    return __builtin_amdgcn_wmma_i32_16x16x16_iu8_w32(true, a, true, b, c, true);
}
__device__ __forceinline__ float hc_q8_rb(float v) { return bf16_to_float(float_to_bf16(v)); }
constexpr size_t kHcQ8TileTokens = 16, kHcQ8TileBytes = 576, kHcQ8ScaleOffset = 512;
constexpr unsigned kHcQ8RasterTokens = 4096;
struct HcQ8RasterTile {
    unsigned token_tile, group_tile, row_tile, group_width;
};
template <int BN>
__device__ __forceinline__ HcQ8RasterTile hc_q8_raster_tile() {
    constexpr unsigned kGroupTiles = kHcQ8RasterTokens / BN;
    const unsigned per_group = kGroupTiles * gridDim.y;
    const unsigned linear = (blockIdx.y * gridDim.x) + blockIdx.x;
    const unsigned group = linear / per_group;
    const unsigned first = group * kGroupTiles;
    const unsigned width = min(kGroupTiles, gridDim.x - first);
    const unsigned rem = linear - (group * per_group);
    return {first + (rem % width), rem % width, rem / width, width};
}

// Grid (token tiles of 128, row tiles of 64): z = xq . wq^T by int8 WMMA with a rescale a K32 block, then the gates:
// act bf16 [batch][low] = silu, inject bf16 [batch][4] = 2 sigmoid for rows from `low`.
extern "C" __global__ void __launch_bounds__(256) tf_fn_hc_down_q8(const void* __restrict__ w, const void* __restrict__ x_blocks,
                                                uint16_t* __restrict__ out, size_t batch, size_t m,
                                                size_t k, uint16_t* inject, int low) {
  constexpr int BM = 64, BN = 128, BK = 4, WM = 2, WN = 4;
  constexpr int kRowTiles = BM / 16, kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM, kWaveTokTiles = kTokTiles / WN;
  constexpr int kABytes = BK * kRowTiles * 32 * 16, kDwBytes = BK * kRowTiles * 16 * 4;
  constexpr int kBBytes = BK * kTokTiles * 32 * 16, kDxBytes = BK * kTokTiles * 16 * 4;
  constexpr int kLdsBytes = kABytes + kDwBytes + kBBytes + kDxBytes;
  __shared__ __attribute__((aligned(16))) uint8_t lds[kLdsBytes];
  auto* s_a = reinterpret_cast<hc_q8_i32x4(*)[kRowTiles][32]>(lds);
  auto* s_dw = reinterpret_cast<float (*)[kRowTiles][16]>(lds + kABytes);
  auto* s_b = reinterpret_cast<hc_q8_i32x4(*)[kTokTiles][32]>(lds + kABytes + kDwBytes);
  auto* s_dx = reinterpret_cast<float (*)[kTokTiles][16]>(lds + kABytes + kDwBytes + kBBytes);
  const auto* w_blocks = static_cast<const HcQ8Block*>(w);
  const size_t num_blocks = k / 32;
  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5, lane_id = tid & 31, sub_lane = lane_id & 15, half_id = lane_id >> 4;
  const int wave_row = wave_id / WN, wave_tok = wave_id % WN;
  const HcQ8RasterTile tile = hc_q8_raster_tile<BN>();
  const size_t r_block = static_cast<size_t>(tile.row_tile) * BM;
  const size_t t_block = static_cast<size_t>(tile.token_tile) * BN;
  const size_t tt_block = t_block / kHcQ8TileTokens;
  float acc[kWaveRowTiles][kWaveTokTiles][8];
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i)
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j)
#pragma unroll
      for (int l = 0; l < 8; ++l) acc[i][j][l] = 0.0F;
  constexpr int kPrefetch = (BM * BK) / 256;
  hc_q8_i32x4 r_q0[kPrefetch], r_q1[kPrefetch];
  float r_dw[kPrefetch];
  hc_q8_i32x4 r_b[BK];
  float r_dx[BK];
  const int b_tile = wave_id;
  const int num_kb = static_cast<int>(num_blocks);
  const int m_i = static_cast<int>(m);
  const HcQ8Block* w_row[kPrefetch];
  float row_live[kPrefetch];
#pragma unroll
  for (int p = 0; p < kPrefetch; ++p) {
    const int r = static_cast<int>(r_block) + (((p * 256) + tid) / BK);
    const int r_clamped = (r < m_i) ? r : (m_i - 1);
    w_row[p] = w_blocks + (static_cast<size_t>(r_clamped) * num_blocks);
    row_live[p] = (r < m_i) ? 1.0F : 0.0F;
  }
  const auto* b_base = static_cast<const int8_t*>(x_blocks) +
                       ((tt_block + static_cast<size_t>(b_tile)) * num_blocks * kHcQ8TileBytes);
  const bool b_live = b_tile < kTokTiles;
  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
      const HcQ8Block& blk = w_row[p][kb_clamped];
      __builtin_memcpy(&r_q0[p], blk.qs + 0, 16);
      __builtin_memcpy(&r_q1[p], blk.qs + 16, 16);
      r_dw[p] = static_cast<float>(__builtin_bit_cast(_Float16, blk.d)) * ((kb < num_kb) ? row_live[p] : 0.0F);
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        const int kb = kb0 + i;
        const int kb_clamped = (kb < num_kb) ? kb : (num_kb - 1);
        const auto* t = b_base + (static_cast<size_t>(kb_clamped) * kHcQ8TileBytes);
        r_b[i] = reinterpret_cast<const hc_q8_i32x4*>(t)[lane_id];
        r_dx[i] = ((kb < num_kb) ? 1.0F : 0.0F) * reinterpret_cast<const float*>(t + kHcQ8ScaleOffset)[lane_id & 15];
      }
    }
  };
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kPrefetch; ++p) {
      const int idx = (p * 256) + tid;
      const int rr = idx / BK, kk = idx % BK, rs = rr / 16, rl = rr % 16;
      s_a[kk][rs][rl] = r_q0[p];
      s_a[kk][rs][16 + rl] = r_q1[p];
      s_dw[kk][rs][(rl % 2 == 0) ? (rl / 2) : (8 + (rl / 2))] = r_dw[p];
    }
    if (b_live) {
#pragma unroll
      for (int i = 0; i < BK; ++i) {
        s_b[i][b_tile][lane_id] = r_b[i];
        if (lane_id < 16) s_dx[i][b_tile][lane_id] = r_dx[i];
      }
    }
  };
  fetch_stage(0);
  for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
    commit_stage();
    __syncthreads();
    if (kb0 + BK < num_kb) fetch_stage(kb0 + BK);
#pragma unroll
    for (int kb = 0; kb < BK; ++kb) {
      hc_q8_i32x4 a0[kWaveRowTiles], a1[kWaveRowTiles];
      float dw[kWaveRowTiles][8];
#pragma unroll
      for (int i = 0; i < kWaveRowTiles; ++i) {
        const int rs = (wave_row * kWaveRowTiles) + i;
        a0[i] = s_a[kb][rs][sub_lane];
        a1[i] = s_a[kb][rs][16 + sub_lane];
        const hc_q8_f32x4 lo = *reinterpret_cast<const hc_q8_f32x4*>(&s_dw[kb][rs][half_id * 8]);
        const hc_q8_f32x4 up = *reinterpret_cast<const hc_q8_f32x4*>(&s_dw[kb][rs][(half_id * 8) + 4]);
        dw[i][0] = lo[0]; dw[i][1] = lo[1]; dw[i][2] = lo[2]; dw[i][3] = lo[3];
        dw[i][4] = up[0]; dw[i][5] = up[1]; dw[i][6] = up[2]; dw[i][7] = up[3];
      }
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j) {
        const int ts = (wave_tok * kWaveTokTiles) + j;
        const hc_q8_i32x4 b0 = s_b[kb][ts][sub_lane];
        const hc_q8_i32x4 b1 = s_b[kb][ts][16 + sub_lane];
        const float dx = s_dx[kb][ts][sub_lane];
#pragma unroll
        for (int i = 0; i < kWaveRowTiles; ++i) {
          hc_q8_i32x8 c = {0, 0, 0, 0, 0, 0, 0, 0};
          c = hc_q8_wmma(a0[i], b0, c);
          c = hc_q8_wmma(a1[i], b1, c);
#pragma unroll
          for (int l = 0; l < 8; ++l)
            acc[i][j][l] = __fmaf_rn(__fmul_rn(dx, static_cast<float>(c[l])), dw[i][l], acc[i][j][l]);
        }
      }
      __builtin_amdgcn_sched_barrier(0);
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i)
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      const size_t token = t_block + ((wave_tok * kWaveTokTiles) + j) * 16 + sub_lane;
#pragma unroll
      for (int l = 0; l < 8; ++l) {
        const int row = static_cast<int>(r_block) + (wave_row * kWaveRowTiles + i) * 16 + 2 * l + half_id;
        if (token < batch && row < m_i) {
          const float v = hc_q8_rb(hc_q8_rb(acc[i][j][l]) * 0.25f);
          const float e = 1.0f + expf(-v);
          if (row < low) out[token * low + row] = static_cast<uint16_t>(float_to_bf16(v / e));
          else inject[token * 4 + row - low] = static_cast<uint16_t>(float_to_bf16(2.0f * hc_q8_rb(1.0f / e)));
        }
      }
    }
}

}  // namespace
