// Dense prompt GEMM tile ported from gufo (MIT, Copyright (c) 2026 gufo contributors), commit 1866ed9:
// src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp, DenseF16GEMMKernel, GroupedRasterTile and helpers.
// Changes for TensorFold: weights are FP4-Local MXFP8 (e4m3 bytes [m][k], e8m0 block scales [m][k / 32]) decoded to
// bf16 at the LDS commit, bf16 activations and bf16 WMMA; the attention, HC-mix and SSM epilogues are not ported.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and
// this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
#pragma once

#include <cstddef>
#include <cstdint>

#include "quant/fp4local_bake.hpp"
#include "tiles/epilogue.hpp"

namespace tf {
namespace rocm {
namespace gufo {

using kv16 = std::uint16_t;
using v16h = __attribute__((__vector_size__(16 * sizeof(kv16)))) kv16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
typedef __bf16 gufo_v16bf __attribute__((ext_vector_type(16)));
__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(gufo_v16bf, a), __builtin_bit_cast(gufo_v16bf, b), c);
}
__device__ __forceinline__ v8f WmmaBf16(v16h a, v16h b, v8f c) { return Wmma(a, b, c); }
constexpr unsigned kSsmProjectionTileTokens = 32;
__device__ __forceinline__ int XorLane(int v, int offset) {
#if 1
  switch (offset) {
    case 1:
      return __builtin_amdgcn_update_dpp(0, v, 0x161, 0xF, 0xF, false);
    case 2:
      return __builtin_amdgcn_update_dpp(0, v, 0x162, 0xF, 0xF, false);
    case 4:
      return __builtin_amdgcn_update_dpp(0, v, 0x164, 0xF, 0xF, false);
    case 8:
      return __builtin_amdgcn_update_dpp(0, v, 0x168, 0xF, 0xF, false);
    case 16:
      return __builtin_amdgcn_permlanex16(v, v, 0x76543210, 0xfedcba98, false,
                                          false);
  }
#endif
  return __shfl_xor(v, offset);
}

__device__ __forceinline__ unsigned XorLane(unsigned v, int offset) {
  return static_cast<unsigned>(XorLane(static_cast<int>(v), offset));
}

__device__ __forceinline__ float XorLane(float v, int offset) {
  return __builtin_bit_cast(float, XorLane(__builtin_bit_cast(int, v), offset));
}
constexpr unsigned kRasterTokens = 4096;
struct RasterTile {
  unsigned token_tile;  // absolute token tile
  unsigned group_tile;  // token tile within its group
  unsigned row_tile;
  unsigned group_width;  // token tiles in this group
};
template<int BN>
__device__ __forceinline__ RasterTile GroupedRasterTile() {
  constexpr unsigned kGroupTiles = kRasterTokens / BN;
  const unsigned per_group = kGroupTiles * gridDim.y;
  const unsigned linear = (blockIdx.y * gridDim.x) + blockIdx.x;
  const unsigned group = linear / per_group;
  const unsigned first = group * kGroupTiles;
  const unsigned width = min(kGroupTiles, gridDim.x - first);
  const unsigned rem = linear - (group * per_group);
  return {first + (rem % width), rem % width, rem / width, width};
}
/// Dense F16 WMMA GEMM over Q8_0 or F16 weights: block = BM rows x BN tokens,
/// BK 32-element K blocks per LDS stage, waves = WM row groups x WN token
/// groups. The codes are dequantized to F16 once per stage as they are
/// committed to LDS (magic-number F16 construction, exact for a Q8_0 code),
/// and the activations are F16 rows [batch][k], so the matrix cores
/// accumulate in F32 with no per-block scaling. SSM waves retain their
/// decoded weight rows in registers instead of staging them in LDS. F16 weights
/// skip decoding and use the same ordered K16 products. y is [batch][m].
template<int BM, int BN, int BK, int WM, int WN, int kRowGroup = 1,
         bool kHcMix = false, bool kSsmConv = false, bool kAttention = false,
         bool kHalfWeights = false, bool kBf16 = false, bool kStrided = false,
         bool kYarn = false>
__launch_bounds__(256) __global__ void DenseF16GEMMKernel(tf::rocm::Affine aff) {
  const void* __restrict__ w = aff.words;
  const kv16* __restrict__ x = static_cast<const kv16*>(aff.x);
  float* __restrict__ y = aff.out;
  kv16* __restrict__ y16 = static_cast<kv16*>(static_cast<void*>(aff.out16));
  const std::size_t batch = aff.m, m = aff.n, k = aff.k;
  const std::uint8_t* __restrict__ w_scale = static_cast<const std::uint8_t*>(aff.scale.p);
  const std::size_t input_stride = 0;
  const std::uint32_t first_row_tile = 0;
  static_assert(WM * WN == 8, "256 threads is 8 waves");
  static_assert(BM % (16 * WM) == 0 && BN % (16 * WN) == 0);
  static_assert(!(kBf16 && kHalfWeights));
  // 16-bit weight rows staged as they are: F16, or BF16 with BF16
  // activations.
  constexpr bool kRawWeights = kHalfWeights || kBf16;
  constexpr bool kRegisterWeights = kSsmConv;
  static_assert(!kRegisterWeights ||
                (BM == 256 && BK == 2 && WM == 8 && WN == 1));
  constexpr int kRowTiles = BM / 16;
  constexpr int kTokTiles = BN / 16;
  constexpr int kWaveRowTiles = kRowTiles / WM;
  constexpr int kWaveTokTiles = kTokTiles / WN;
  // Weight and activation K blocks staged per thread; a unit past the
  // stage's block count is idle.
  constexpr int kAUnits = BM * BK;
  constexpr int kBUnits = BN * BK;
  constexpr int kAPer = (kAUnits + 255) / 256;
  constexpr int kBPer = (kBUnits + 255) / 256;

  // One K block of one row or token is four 16-byte chunks; the chunks of
  // nearby rows are permuted so a fragment read (one row per lane, 64-byte
  // stride) covers all bank groups.
  constexpr int kLdsChunks = BK * ((kRegisterWeights ? 0 : BM) + BN) * 4;
  constexpr int kTransposeChunks = 8 *
                                   (kSsmConv ? kSsmProjectionTileTokens : 16) *
                                   36 * sizeof(float) / sizeof(uint4);
  constexpr int kLdsStorage =
      kLdsChunks < kTransposeChunks ? kTransposeChunks : kLdsChunks;
  __shared__ __attribute__((aligned(16))) uint4 s_lds[kLdsStorage];
  auto* s_a = reinterpret_cast<uint4(*)[BM][4]>(s_lds);
  auto* s_b = reinterpret_cast<uint4(*)[BN][4]>(
      s_lds + (kRegisterWeights ? 0 : BK * BM * 4));
  const auto swizzle = [](int row, int c) { return c ^ ((row >> 1) & 3); };

  const int num_kb = static_cast<int>(k / 32);
  const int m_i = static_cast<int>(m);
  const auto* w_bytes = static_cast<const std::uint8_t*>(w);

  const int tid = static_cast<int>(threadIdx.x);
  const int wave_id = tid >> 5;
  const int lane_id = tid & 31;
  const int sub_lane = lane_id & 15;
  const int half_id = lane_id >> 4;
  const int wave_row = wave_id / WN;
  const int wave_tok = wave_id % WN;
  // Group row tiles for the narrow HC up projection. The tile grid's row
  // count is divisible by kRowGroup; every dot product keeps its K order.
  const RasterTile tile = GroupedRasterTile<BN>();
  const unsigned row_group = tile.row_tile / kRowGroup;
  const unsigned within =
      (tile.row_tile % kRowGroup) * tile.group_width + tile.group_tile;
  const int r_block =
      (first_row_tile + row_group * kRowGroup + within % kRowGroup) * BM;
  const int t_block =
      (tile.token_tile - tile.group_tile + within / kRowGroup) * BN;
  const auto token_of = [&](int local) { return t_block + local; };

  // SSM thread tid owns row tid and K block p of the stage. Other paths
  // use row (p*256+tid)/BK and K block (p*256+tid)%BK; rows past m read
  // the last row with a zero scale.
  const std::uint8_t* a_ptr[kAPer];
  std::size_t a_srow[kAPer];
  bool a_live[kAPer];
#pragma unroll
  for (int p = 0; p < kAPer; ++p) {
    const int idx = (p * 256) + tid;
    const int r = r_block + (kRegisterWeights ? tid : idx / BK);
    a_live[p] = idx < kAUnits && r < m_i;
    // Interleave the four gate streams within the tile, without repacking
    // weights. Each group of four rows produces one mixed hidden element.
    const int weight_row = kHcMix ? (r % 4) * (m_i / 4) + r / 4 : r;
    a_ptr[p] = w_bytes +
               static_cast<std::size_t>(a_live[p] ? weight_row : (m_i - 1)) *
                   static_cast<std::size_t>(num_kb) * (kRawWeights ? 64 : 32);
    a_srow[p] = static_cast<std::size_t>(a_live[p] ? weight_row : (m_i - 1)) *
                static_cast<std::size_t>(num_kb);
  }
  const kv16* b_ptr[kBPer];
#pragma unroll
  for (int p = 0; p < kBPer; ++p) {
    const int idx = (p * 256) + tid;
    const int t = token_of(idx / BK);
    b_ptr[p] =
        idx < kBUnits && t < static_cast<int>(batch)
            ? x + (static_cast<std::size_t>(t) * (kStrided ? input_stride : k))
            : nullptr;
  }
  uint4 a_codes[kAPer][2];
  uint4 a_half[kRawWeights ? kAPer : 1][4];
  std::uint32_t a_d[kAPer];
  uint4 b_data[kBPer][4];

  const auto fetch_stage = [&](int kb0) {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int kb = kb0 + (kRegisterWeights ? p : (((p * 256) + tid) % BK));
      const bool live = a_live[p] && kb < num_kb;
      if constexpr (kRawWeights) {
        const auto* src = reinterpret_cast<const uint4*>(
            a_ptr[p] + static_cast<std::size_t>(kb) * 64);
#pragma unroll
        for (int c = 0; c < 4; ++c)
          a_half[p][c] = live ? src[c] : make_uint4(0u, 0u, 0u, 0u);
      } else {
        const std::size_t kbi = static_cast<std::size_t>(live ? kb : (num_kb - 1));
        const std::uint8_t* blk = a_ptr[p] + (kbi * 32);
        a_d[p] = live ? w_scale[a_srow[p] + kbi] : 0xFFFFFFFFU;
        a_codes[p][0] = *reinterpret_cast<const uint4*>(blk);
        a_codes[p][1] = *reinterpret_cast<const uint4*>(blk + 16);
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int kb = kb0 + (((p * 256) + tid) % BK);
      if (b_ptr[p] != nullptr && kb < num_kb) {
        const auto* src = reinterpret_cast<const uint4*>(b_ptr[p] + (kb * 32));
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = src[c];
        }
      } else {
#pragma unroll
        for (int c = 0; c < 4; ++c) {
          b_data[p][c] = make_uint4(0u, 0u, 0u, 0u);
        }
      }
    }
  };

  // Preserve this stage's rounded fragments before prefetch overwrites
  // the raw Q8 codes. Each SSM wave owns both of its 16-row output tiles.
  v16h own_a_lo[BK];
  v16h own_a_hi[BK];
  const auto commit_stage = [&]() {
#pragma unroll
    for (int p = 0; p < kAPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kAUnits) {
        break;
      }
      const int row = idx / BK;
      const int kk = idx % BK;
      if constexpr (kRawWeights) {
#pragma unroll
        for (int c = 0; c < 4; ++c)
          s_a[kk][row][swizzle(row, c)] = a_half[p][c];
      } else {
        const std::uint32_t words[8] = {
            a_codes[p][0].x, a_codes[p][0].y, a_codes[p][0].z, a_codes[p][0].w,
            a_codes[p][1].x, a_codes[p][1].y, a_codes[p][1].z, a_codes[p][1].w};
        // MXFP8: e4m3 bytes times the block's e8m0 multiplier, exact bf16 pairs; a dead unit decodes to zero.
        const float mult = a_d[p] == 0xFFFFFFFFU ? 0.0F : tf::rocm::mxfp8_multiplier(a_d[p]);
        std::uint32_t h[16];
#pragma unroll
        for (int i = 0; i < 8; ++i) {
          h[2 * i] = tf::rocm::mxfp8_pair(words[i], 0, mult);
          h[2 * i + 1] = tf::rocm::mxfp8_pair(words[i], 2, mult);
        }
        if constexpr (kRegisterWeights) {
          __builtin_memcpy(&own_a_lo[p], &h[0], 32);
          __builtin_memcpy(&own_a_hi[p], &h[8], 32);
        } else {
#pragma unroll
          for (int c = 0; c < 4; ++c) {
            uint4 v;
            __builtin_memcpy(&v, &h[4 * c], 16);
            s_a[kk][row][swizzle(row, c)] = v;
          }
        }
      }
    }
#pragma unroll
    for (int p = 0; p < kBPer; ++p) {
      const int idx = (p * 256) + tid;
      if (idx >= kBUnits) {
        break;
      }
      const int t = idx / BK;
      const int kk = idx % BK;
#pragma unroll
      for (int c = 0; c < 4; ++c) {
        s_b[kk][t][swizzle(t, c)] = b_data[p][c];
      }
    }
  };

  v8f acc[kWaveRowTiles][kWaveTokTiles];
  // Separate K16 chains reduce FP32 accumulation error for the sensitive
  // unquantized router/gate projections, with the same order in every chunk.
  v8f acc_high[kHalfWeights ? kWaveRowTiles : 1]
              [kHalfWeights ? kWaveTokTiles : 1]{};
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; ++j) {
      acc[i][j] = v8f{0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F, 0.0F};
    }
  }

  // Keep full tiles free of per-K bounds branches. The wide SSM paths can
  // omit whole 16-token subtiles in their final partial tile while retaining
  // the original WMMA order for every live output. One launch preserves
  // weight reuse between neighboring token tiles.
  const auto accumulate = [&]<bool partial>() {
    fetch_stage(0);
    for (int kb0 = 0; kb0 < num_kb; kb0 += BK) {
      commit_stage();
      __syncthreads();
      if (kb0 + BK < num_kb) {
        fetch_stage(kb0 + BK);
      }
      // Keeps the next stage's loads ahead of the matrix work: scheduled
      // freely, the compiler sinks them below it and the commit waits on
      // the full memory latency.
      __builtin_amdgcn_sched_barrier(0);
#pragma unroll
      for (int kb = 0; kb < BK; ++kb) {
        v16h a_lo[kWaveRowTiles];
        v16h a_hi[kWaveRowTiles];
        if constexpr (kRegisterWeights) {
          using v8u = __attribute__((__vector_size__(32))) std::uint32_t;
          const v8u own_lo = __builtin_bit_cast(v8u, own_a_lo[kb]);
          const v8u own_hi = __builtin_bit_cast(v8u, own_a_hi[kb]);
          v8u peer_lo;
          v8u peer_hi;
#pragma unroll
          for (int q = 0; q < 8; ++q) {
            peer_lo[q] = XorLane(own_lo[q], 16);
            peer_hi[q] = XorLane(own_hi[q], 16);
          }
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            a_lo[i] = __builtin_bit_cast(v16h, half_id == i ? own_lo : peer_lo);
            a_hi[i] = __builtin_bit_cast(v16h, half_id == i ? own_hi : peer_hi);
          }
        } else {
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            const int row = (((wave_row * kWaveRowTiles) + i) * 16) + sub_lane;
            uint4 c[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              c[q] = s_a[kb][row][swizzle(row, q)];
            }
            __builtin_memcpy(&a_lo[i], &c[0], 32);
            __builtin_memcpy(&a_hi[i], &c[2], 32);
          }
        }
#pragma unroll
        for (int j = 0; j < kWaveTokTiles; ++j) {
          if constexpr (partial) {
            if (token_of(((wave_tok * kWaveTokTiles) + j) * 16) >= batch)
              continue;
          }
          const int t = (((wave_tok * kWaveTokTiles) + j) * 16) + sub_lane;
          uint4 c[4];
#pragma unroll
          for (int q = 0; q < 4; ++q) {
            c[q] = s_b[kb][t][swizzle(t, q)];
          }
          v16h b_lo;
          v16h b_hi;
          __builtin_memcpy(&b_lo, &c[0], 32);
          __builtin_memcpy(&b_hi, &c[2], 32);
#pragma unroll
          for (int i = 0; i < kWaveRowTiles; ++i) {
            if constexpr (kBf16) {
              // One K16 chain in K order for every token.
              acc[i][j] = WmmaBf16(a_lo[i], b_lo, acc[i][j]);
              acc[i][j] = WmmaBf16(a_hi[i], b_hi, acc[i][j]);
            } else if constexpr (kHalfWeights) {
              acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
              acc_high[i][j] = Wmma(a_hi[i], b_hi, acc_high[i][j]);
            } else {
              acc[i][j] = Wmma(a_lo[i], b_lo, acc[i][j]);
              acc[i][j] = Wmma(a_hi[i], b_hi, acc[i][j]);
            }
          }
        }
      }
      __syncthreads();
    }
  };
  if constexpr (kSsmConv || kStrided) {
    if (t_block + BN <= batch)
      accumulate.template operator()<false>();
    else
      accumulate.template operator()<true>();
  } else {
    accumulate.template operator()<false>();
  }

  if constexpr (kHalfWeights) {
#pragma unroll
    for (int i = 0; i < kWaveRowTiles; ++i) {
#pragma unroll
      for (int j = 0; j < kWaveTokTiles; ++j)
        acc[i][j] += acc_high[i][j];
    }
  }


  // Transpose the result through LDS, two row tiles at a time, so every
  // global store covers 32 consecutive rows of one token: a full 128-byte
  // line (half lines cost a read-modify-write on the fabric).
  // Four padding floats reduce scatter bank conflicts and retain float4
  // alignment for both output stores and the fused convolution's reads.
  constexpr unsigned kOutputStride = 36;
  // Pair token tiles for SSM: half as many convolutions cross a tile edge.
  // The larger transpose still fits the projection's existing LDS allocation.
  constexpr unsigned kOutputTokens = kSsmConv ? kSsmProjectionTileTokens : 16;
  constexpr unsigned kOutputGroups = kOutputTokens / 16;
  static_assert(8 * kOutputTokens * kOutputStride * sizeof(float) <=
                sizeof(s_lds));
  static_assert(kWaveRowTiles % 2 == 0, "the epilogue pairs row tiles");
  float* tile_scratch =
      reinterpret_cast<float*>(s_lds) + wave_id * kOutputTokens * kOutputStride;
#pragma unroll
  for (int i = 0; i < kWaveRowTiles; i += 2) {
#pragma unroll
    for (int j = 0; j < kWaveTokTiles; j += kOutputGroups) {
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // scratch[token][row], with 32 output rows per token.
#pragma unroll
        for (int l = 0; l < 8; ++l) {
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + (2 * l) +
                       half_id] = acc[i][j + group][l];
          tile_scratch[((sub_lane + group * 16) * kOutputStride) + 16 +
                       (2 * l) + half_id] = acc[i + 1][j + group][l];
        }
      }
      __builtin_amdgcn_wave_barrier();
      const std::size_t r0 =
          static_cast<std::size_t>(r_block) +
          static_cast<std::size_t>((((wave_row * kWaveRowTiles) + i) * 16));
      const std::size_t t0 = static_cast<std::size_t>(
          token_of(((wave_tok * kWaveTokTiles) + j) * 16));
#pragma unroll
      for (unsigned group = 0; group < kOutputGroups; ++group) {
        // Lane pair (2p, 2p + 1) stores token p's 32 rows as eight float4.
        const int tok_l = (lane_id >> 1) + group * 16;
        const int row_l = (lane_id & 1) * 16;
        const std::size_t tok = t0 + static_cast<std::size_t>(tok_l);
        const auto* src = reinterpret_cast<const float4*>(
            tile_scratch + (tok_l * kOutputStride) + row_l);
          if (y16 != nullptr) {
            if (tok < batch) {
#pragma unroll
              for (int q = 0; q < 16; ++q) {
                const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
                if (r < m) {
                  y16[(tok * m) + r] = __builtin_bit_cast(
                      kv16, tf::rocm::narrow<tf::rocm::DotBF16>(
                                tile_scratch[(tok_l * kOutputStride) + row_l + q]));
                }
              }
            }
          } else if (tok < batch && r0 + 32 <= m && ((tok * m) % 4 == 0)) {
            auto* dst = reinterpret_cast<float4*>(
                y + (tok * m) + r0 + static_cast<std::size_t>(row_l));
#pragma unroll
            for (int q = 0; q < 4; ++q) {
              dst[q] = src[q];
            }
          } else if (tok < batch) {
#pragma unroll
            for (int q = 0; q < 16; ++q) {
              const std::size_t r = r0 + static_cast<std::size_t>(row_l + q);
              if (r < m) {
                y[(tok * m) + r] =
                    tile_scratch[(tok_l * kOutputStride) + row_l + q];
              }
            }
          }
      }
      __builtin_amdgcn_wave_barrier();
    }
  }
}

}  // namespace gufo
}  // namespace rocm
}  // namespace tf
