// Prompt attention ported from gufo (MIT, Copyright (c) 2026 gufo contributors), commit 1866ed9:
// src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp, WmmaCausalAttentionKernel and its helpers.
// Changes for TensorFold's cache: K/V are bf16 [kv head][capacity][256] (bf16 WMMA, bf16 q and p), output rounded to bf16.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
// documentation files (the "Software"), to deal in the Software without restriction, including without limitation the
// rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit
// persons to whom the Software is furnished to do so, subject to the following conditions: The above copyright notice and
// this permission notice shall be included in all copies or substantial portions of the Software.
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND.
#pragma once

#include <hip/hip_runtime.h>

#include <cmath>
#include <cstddef>
#include <cstdint>

namespace gufo_attn {

using kv16 = std::uint16_t;
__device__ __forceinline__ kv16 FloatToBf16(float v) {
  const std::uint32_t b = __float_as_uint(v);
  return static_cast<kv16>((b + 0x7FFFu + ((b >> 16) & 1u)) >> 16);
}
__device__ __forceinline__ float Bf16ToFloat(kv16 v) { return __uint_as_float(static_cast<std::uint32_t>(v) << 16); }
__device__ __forceinline__ float SigmoidF(float x) { return 1.0f / (1.0f + __expf(-x)); }
using v16h = __attribute__((__vector_size__(16 * sizeof(kv16)))) kv16;
using v8f = __attribute__((__vector_size__(8 * sizeof(float)))) float;
typedef __bf16 v16bf __attribute__((ext_vector_type(16)));
__device__ __forceinline__ v8f Wmma(v16h a, v16h b, v8f c) {
  return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(v16bf, a), __builtin_bit_cast(v16bf, b), c);
}
__device__ __forceinline__ v16h LoadFrag(const kv16* p) {
  union { v16h f; uint4 u[2]; } cvt;
  cvt.u[0] = *reinterpret_cast<const uint4*>(p);
  cvt.u[1] = *reinterpret_cast<const uint4*>(p + 8);
  return cvt.f;
}
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
// A barrier for kernels whose waves exchange data through LDS only.
// __syncthreads also waits for every outstanding global load and drops the
// L0 cache, which serializes register prefetches behind the barrier. The
// global inputs are read-only here, so completing LDS traffic is enough.
__device__ __forceinline__ __attribute__((convergent)) void SyncLds() {
  asm volatile("s_waitcnt lgkmcnt(0)" ::: "memory");
  __builtin_amdgcn_s_barrier();
  asm volatile("" ::: "memory");
}

constexpr std::uint32_t kWmmaHeadDim = 256;
constexpr std::uint32_t kWmmaQueryHeads = 24;
constexpr std::uint32_t kWmmaKvHeads = 2;
constexpr std::uint32_t kWmmaGqa = kWmmaQueryHeads / kWmmaKvHeads;
constexpr std::uint32_t kWmmaAttnWidth = kWmmaQueryHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaKvWidth = kWmmaKvHeads * kWmmaHeadDim;
constexpr std::uint32_t kWmmaHeads = 2;  // query heads per block, divides GQA
// Sixteen queries keep the causal-tail accumulator in registers.
constexpr std::uint32_t kWmmaQueryRows = 16;
constexpr std::uint32_t kWmmaKeys = 16;
constexpr std::uint32_t kWmmaRatio = 4;
constexpr std::uint32_t kWmmaKSteps = kWmmaHeadDim / 16;
// Row padding keeps a 16-byte-per-lane fragment read off a single bank group.
constexpr std::uint32_t kWmmaKStride = kWmmaHeadDim + 8;
// The sparse window builds its union mask in passes of 256 threads x eight
// words (one bit per 4-token block, 262,144 visible tokens per pass), so the
// live extent is not bounded by the pass. A compact list entry keeps the block
// index below its membership bits, which bounds the extent at 2^28 blocks.
constexpr std::uint32_t kWmmaPassWords = 256 * 8;
constexpr std::uint32_t kWmmaMemberShift = 28;

/// Two row layouts share the kernel. The dense window packs 16 queries x 2
/// heads (a row block is 16 queries of one head, grid.y over head pairs).
/// The sparse window packs four queries x twelve heads into three row
/// blocks without padding, with adjacent blocks covering both KV heads: the key
/// tiles are gathered from the union of the block's query selections and
/// four selections overlap far less than 32 (measured at 16k depth: 846
/// versus 2,478 selected blocks against 512 per query).
template<std::uint32_t kQueryRows, std::uint32_t kKeys, bool kPackHeads>
__launch_bounds__(256, 2) __global__ void WmmaCausalAttentionKernel(
    const float* __restrict__ q, const float* __restrict__ gate,
    const kv16* __restrict__ k_cache, const kv16* __restrict__ v_cache,
    const std::uint32_t* __restrict__ mask, std::uint32_t mask_words,
    float* __restrict__ out, std::uint32_t start_pos, std::uint32_t n_tokens,
    std::uint32_t first_query_group, std::uint64_t capacity) {
  // The launcher accepts only four-token selection blocks. Keep that
  // geometry constant throughout the key sweep.
  constexpr std::uint32_t ratio = kWmmaRatio;
  constexpr std::uint32_t kHeadDim = kWmmaHeadDim;
  constexpr std::uint32_t kRowBlocks = kPackHeads
                                           ? (kQueryRows * kWmmaGqa + 15) / 16
                                           : (kQueryRows / 16) * kWmmaHeads;
  static_assert(!kPackHeads || kQueryRows * kWmmaGqa == 48,
                "four queries pack twelve heads into three tiles");
  constexpr std::uint32_t kKeyBlocks = kKeys / 16;
  constexpr std::uint32_t kSTiles = kRowBlocks * kKeyBlocks;
  constexpr std::uint32_t kKStepsPerWave = kWmmaKSteps / 2;
  constexpr std::uint32_t kRows = kRowBlocks * 16;
  constexpr std::uint32_t kOTilesPerWave = (kRowBlocks * kWmmaKSteps) / 8;
  constexpr std::uint32_t kSoftmaxLanes = 4;
  constexpr std::uint32_t kVtStride = kKeys + 8;
  static_assert(kSTiles == (kPackHeads ? 3 : 2), "two waves per S tile");
  static_assert(kOTilesPerWave == 2 * kRowBlocks, "O tiles per wave");
  static_assert(kKeys % 16 == 0 && (kPackHeads || kQueryRows % 16 == 0),
                "16-row WMMA tiles");
  static_assert(kKStepsPerWave * 2 == kWmmaKSteps, "k split");
  static_assert(kWmmaKStride % 8 == 0 && kVtStride % 8 == 0,
                "fragment rows must start on a 16-byte boundary");
  static_assert(kSoftmaxLanes * (kKeys / kSoftmaxLanes) == kKeys, "softmax");

  const std::uint32_t tid = threadIdx.x;
  const std::uint32_t lane = tid & 31u;
  const std::uint32_t wave = tid >> 5u;
  const std::uint32_t sub = lane & 15u;
  const std::uint32_t half_id = lane >> 4u;

  // Keep the two KV heads of nearby queries together in the launch order.
  // This improves cache reuse without changing any query's key sweep.
  const std::uint32_t linear = blockIdx.y * gridDim.x + blockIdx.x;
  const std::uint32_t query_group =
      first_query_group + (kPackHeads ? linear / kWmmaKvHeads : blockIdx.x);
  const std::uint32_t query_start = query_group * kQueryRows;
  const std::uint32_t kv_head =
      kPackHeads ? linear % kWmmaKvHeads : blockIdx.y / (kWmmaGqa / kWmmaHeads);
  const std::uint32_t first_query_head =
      kPackHeads ? kv_head * kWmmaGqa
                 : (kv_head * kWmmaGqa) +
                       ((blockIdx.y % (kWmmaGqa / kWmmaHeads)) * kWmmaHeads);
  constexpr float attention_scale = 1.0F / 16.0F;

  // Row (row block rb, row r) -> (local query, query head, live).
  const auto row_query = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? query_start + (rb * 16 + r) / kWmmaGqa
                      : query_start + ((rb / kWmmaHeads) * 16) + r;
  };
  const auto row_head = [&](std::uint32_t rb, std::uint32_t r) {
    return kPackHeads ? first_query_head + (rb * 16 + r) % kWmmaGqa
                      : first_query_head + (rb % kWmmaHeads);
  };
  const auto row_live = [&](std::uint32_t rb, std::uint32_t r) {
    return row_query(rb, r) < n_tokens &&
           (!kPackHeads || rb * 16 + r < kQueryRows * kWmmaGqa);
  };

  // K and V^T have their own buffers so the next tile's S overlaps this
  // tile's PV (see the key sweep below).
  __shared__ kv16 k_lds[kKeys * kWmmaKStride];
  __shared__ __attribute__((aligned(16))) kv16 v_lds[kHeadDim * kVtStride];
  // Pad score rows for the four-lane softmax reads and probability rows
  // for the WMMA fragments. Both transposes otherwise repeat LDS banks.
  // Dense score rows are padded to 20 floats: the WMMA scatter stays nearly
  // conflict-free and each softmax lane reads its four scores as a float4.
  // The sparse window keeps 17-float rows and scalar reads (faster there).
  constexpr bool kVectorSoftmax = !kPackHeads;
  constexpr std::uint32_t kScoreStride = kVectorSoftmax ? 20 : 17;
  static_assert(kKeys / kSoftmaxLanes == 4, "one float4 of scores per lane");
  __shared__
      __attribute__((aligned(16))) float s_lds[2][kSTiles][16][kScoreStride];
  __shared__ __attribute__((aligned(16))) kv16 p_lds[kRows][kKeys + 8];
  __shared__ float row_sum[kRows];
  // Rescale factors grouped as [row block][row parity][row / 2]: a lane of
  // the PV phase needs the eight rows of its parity, read as two float4.
  __shared__ __attribute__((aligned(16))) float row_scale[kRows];

  const std::uint32_t s_tile = wave % kSTiles;
  const std::uint32_t s_kh = wave / kSTiles;
  const std::uint32_t s_rb = s_tile % kRowBlocks;
  const std::uint32_t s_kb = s_tile / kRowBlocks;

  v16h q_frag[kKStepsPerWave];
  if (wave < 2 * kSTiles) {
    const std::uint32_t local_query = row_query(s_rb, sub);
    const bool live = row_live(s_rb, sub);
    const float* q_row =
        q +
        (static_cast<std::size_t>(live ? local_query : 0) * kWmmaAttnWidth) +
        (static_cast<std::size_t>(live ? row_head(s_rb, sub) : 0) * kHeadDim);
#pragma unroll
    for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
      const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
      const auto* qp = reinterpret_cast<const float4*>(q_row + d0);
#pragma unroll
      for (std::uint32_t v = 0; v < 4; ++v) {
        const float4 f = live ? qp[v] : make_float4(0.0F, 0.0F, 0.0F, 0.0F);
        q_frag[ks][(v * 4) + 0] = FloatToBf16(f.x * attention_scale);
        q_frag[ks][(v * 4) + 1] = FloatToBf16(f.y * attention_scale);
        q_frag[ks][(v * 4) + 2] = FloatToBf16(f.z * attention_scale);
        q_frag[ks][(v * 4) + 3] = FloatToBf16(f.w * attention_scale);
      }
    }
  }

  // This wave owns dim tiles `wave` and `wave + 8` for every row block.
  v8f o_acc[kRowBlocks][2] = {};
  // The same four threads own each softmax row throughout the key sweep.
  // Keep its running statistics in registers; only the final denominator
  // and each tile's rescale factor need to cross waves.
  float running_max = -INFINITY;
  float running_sum = 0.0F;

  const std::uint32_t context_end = start_pos + n_tokens;
  const std::uint32_t max_visible =
      min(context_end, start_pos + query_start + kQueryRows);

  // Key tiles are gathered from `kBlocksPerTile` selection blocks (ratio
  // keys each). In the sparse window the blocks come from the union of this
  // block's query masks (every row still applies its own mask below), then
  // every block from the earliest row's tail on; the dense window walks the
  // blocks in order. The sweep therefore costs the union of the selected
  // windows, not the context.
  constexpr std::uint32_t kBlocksPerTile = kKeys / 4;
  static_assert(kBlocksPerTile == 4, "a tile is four selection blocks");
  const std::uint32_t n_blocks = (max_visible + ratio - 1) / ratio;
  const std::uint32_t tail_block =
      ((start_pos + query_start + 1) / ratio);  // first always-visible block
  constexpr bool sparse = kPackHeads;
  // Four 512-block selections plus their incomplete tail. The same LDS
  // stores the union words of wider selections used by operator callers
  // while they fit.
  constexpr unsigned kListCapacity = 4 * 512 + 4;
  // A compact entry also carries which of the packed queries selected the
  // block, so the softmax reads its mask bits from the tile, not global.
  constexpr unsigned kMaskRows = kPackHeads ? kQueryRows : 1;
  constexpr unsigned kMemberShift = kWmmaMemberShift;
  static_assert(kMaskRows <= 32 - kMemberShift, "membership bits fit");
  __shared__ unsigned union_words[kListCapacity];
  __shared__ unsigned wave_counts[8];
  bool compact = false;
  unsigned selected = 0;
  const unsigned word_count = (n_blocks + 31) / 32;
  // A non-compact union keeps its words in LDS while they fit; past that the
  // walk recomputes each word from the rows' masks.
  const bool words_in_lds = word_count <= kListCapacity;
  const unsigned live_rows = min(kQueryRows, n_tokens - query_start);
  // Union word `w` once the always-visible tail and the live extent apply.
  const auto clip_word = [&](unsigned w, unsigned bits) {
    if (w * 32 >= tail_block)
      bits = ~0u;
    else if ((w + 1) * 32 > tail_block)
      bits |= ~0u << (tail_block % 32);
    if ((w + 1) * 32 > n_blocks)
      bits &= (1u << (n_blocks % 32)) - 1u;
    return bits;
  };
  if (sparse) {
    // Consecutive word ranges per thread make the prefix scan preserve
    // increasing block order, including ties and the always-visible tail.
    // A pass covers 256 x 8 words; wider extents take several in block order.
    unsigned local_words[8];
    unsigned row_words[8][kMaskRows];
    unsigned pass_first = 0;
    unsigned pass_words = 0;
    unsigned words_per_thread = 0;
    const auto load_pass = [&](unsigned first_word) {
      pass_first = first_word;
      pass_words = min(word_count - first_word, kWmmaPassWords);
      words_per_thread = (pass_words + 255) / 256;
      unsigned count = 0;
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const unsigned offset = tid * words_per_thread + j;
        const unsigned w = first_word + offset;
        unsigned bits = 0;
#pragma unroll
        for (unsigned r = 0; r < kMaskRows; ++r)
          row_words[j][r] = 0;
        if (j < words_per_thread && offset < pass_words) {
#pragma unroll
          for (unsigned r = 0; r < kMaskRows; ++r) {
            if (r < live_rows) {
              row_words[j][r] = mask[size_t(query_start + r) * mask_words + w];
              bits |= row_words[j][r];
            }
          }
          bits = clip_word(w, bits);
        }
        local_words[j] = bits;
        count += __popc(bits);
      }
      return count;
    };
    // Block-wide exclusive prefix of `count` in thread order, and the total.
    const auto scan_counts = [&](unsigned count, unsigned* prefix) {
      unsigned scan = count;
#pragma unroll
      for (unsigned offset = 1; offset < 32; offset *= 2) {
        unsigned before = __shfl_up(scan, offset, 32);
        if (lane >= offset)
          scan += before;
      }
      if (lane == 31)
        wave_counts[wave] = scan;
      __syncthreads();
      unsigned before = scan - count;
      unsigned total = 0;
#pragma unroll
      for (unsigned w = 0; w < 8; ++w) {
        if (w < wave)
          before += wave_counts[w];
        total += wave_counts[w];
      }
      *prefix = before;
      return total;
    };
    // This thread's list entries of the loaded pass from `index` on; a
    // bounded emission drops the entries past the list capacity.
    const auto emit_entries = [&](unsigned index, bool bounded) {
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const unsigned word = pass_first + tid * words_per_thread + j;
        unsigned bits = local_words[j];
        while (bits) {
          unsigned bit = __builtin_ctz(bits);
          unsigned members = 0;
#pragma unroll
          for (unsigned r = 0; r < kMaskRows; ++r)
            members |= ((row_words[j][r] >> bit) & 1u) << r;
          if (!bounded || index < kListCapacity)
            union_words[index] = (word * 32 + bit) | (members << kMemberShift);
          ++index;
          bits &= bits - 1;
        }
      }
    };
    const auto store_words = [&] {
#pragma unroll
      for (unsigned j = 0; j < 8; ++j) {
        const unsigned offset = tid * words_per_thread + j;
        if (j < words_per_thread && offset < pass_words)
          union_words[pass_first + offset] = local_words[j];
      }
    };
    if (word_count <= kWmmaPassWords) {
      unsigned prefix = 0;
      selected = scan_counts(load_pass(0), &prefix);
      compact = selected <= kListCapacity;
      if (compact)
        emit_entries(prefix, false);
      else
        store_words();
    } else {
      // Count and emit in one read of the masks. Entries keep block order
      // across passes, so the first kListCapacity are exact; an overflowing
      // union is rewritten as words when they fit LDS.
      for (unsigned first = 0; first < word_count; first += kWmmaPassWords) {
        if (first != 0)
          __syncthreads();  // every wave has read wave_counts
        unsigned prefix = 0;
        const unsigned total = scan_counts(load_pass(first), &prefix);
        emit_entries(selected + prefix, true);
        selected += total;
      }
      compact = selected <= kListCapacity;
      if (!compact && words_in_lds) {
        __syncthreads();  // the dropped list is overwritten
        for (unsigned first = 0; first < word_count; first += kWmmaPassWords) {
          load_pass(first);
          store_words();
        }
      }
    }
    __syncthreads();
  }
  // Word `w` of the non-compact union.
  const auto union_word = [&](std::uint32_t w) -> std::uint32_t {
    if (words_in_lds)
      return union_words[w];
    unsigned bits = 0;
#pragma unroll
    for (unsigned r = 0; r < kMaskRows; ++r) {
      if (r < live_rows)
        bits |= mask[size_t(query_start + r) * mask_words + w];
    }
    return clip_word(w, bits);
  };
  // First visible block at or after `b` (n_blocks when none). Uniform across
  // the block: every thread walks the same words.
  const auto next_block = [&](std::uint32_t b) -> std::uint32_t {
    if (!sparse) {
      return b;
    }
    while (b < tail_block && b < n_blocks) {
      const std::uint32_t bits = union_word(b / 32) >> (b % 32);
      // Never jump past the tail: from tail_block on every block is visible.
      if (bits != 0) {
        return min(b + static_cast<std::uint32_t>(__builtin_ctz(bits)),
                   tail_block);
      }
      b = min(((b / 32) + 1) * 32u, tail_block);
    }
    return b;
  };
  struct Tile {
    std::uint32_t block[kBlocksPerTile];
    std::uint32_t count;
    std::uint32_t members;  // kMaskRows bits per block, compact lists only
  };
  // Gathers the next tile starting the search at block `b`; returns the
  // block to continue from.
  const auto gather_tile = [&](std::uint32_t b, Tile& tile) {
    if constexpr (!sparse) {
      // The dense window's tiles are four consecutive blocks.
      tile.count = min(kBlocksPerTile, n_blocks - min(b, n_blocks));
      tile.members = 0;
#pragma unroll
      for (std::uint32_t i = 0; i < kBlocksPerTile; ++i)
        tile.block[i] = b + i < n_blocks ? b + i : n_blocks;
      return b + tile.count;
    }
    if (compact) {
      tile.count = min(kBlocksPerTile, selected - min(b, selected));
      tile.members = 0;
#pragma unroll
      for (unsigned i = 0; i < kBlocksPerTile; ++i) {
        const unsigned entry = b + i < selected ? union_words[b + i] : 0u;
        tile.block[i] =
            b + i < selected ? entry & ((1u << kMemberShift) - 1u) : n_blocks;
        tile.members |= (entry >> kMemberShift) << (i * kMaskRows);
      }
      return b + tile.count;
    }
    tile.count = 0;
    tile.members = 0;
#pragma unroll
    for (std::uint32_t i = 0; i < kBlocksPerTile; ++i) {
      b = next_block(b);
      const bool have = b < n_blocks;
      tile.block[i] = have ? b : n_blocks;
      tile.count += have ? 1u : 0u;
      b = have ? b + 1 : b;
    }
    return b;
  };
  // Key position of tile row `r` (0..15), or context_end past the tile.
  const auto tile_key = [&](const Tile& tile, std::uint32_t r) {
    if constexpr (!sparse) {
      const std::uint32_t blk = tile.block[0] + (r / ratio);
      return blk < n_blocks ? (blk * ratio) + (r % ratio) : context_end;
    }
    const std::uint32_t i = r / ratio;
    std::uint32_t blk = n_blocks;
#pragma unroll
    for (std::uint32_t j = 0; j < kBlocksPerTile; ++j) {
      if (i == j) {
        blk = tile.block[j];
      }
    }
    return blk < n_blocks ? (blk * ratio) + (r % ratio) : context_end;
  };

  // One key per lane for V, a 16-dim slice per thread: a strided global read
  // in exchange for conflict-free transpose writes.
  constexpr std::uint32_t kVRegs = (kKeys * kHeadDim) / (256 * 8);
  constexpr std::uint32_t kKRegs = (kKeys * (kHeadDim / 8)) / 256;
  static_assert(kKRegs * 256 == kKeys * (kHeadDim / 8), "K stages evenly");
  const std::uint32_t v_key = lane % kKeys;
  const std::uint32_t v_slice = (tid / kKeys) * (kVRegs * 8);
  const auto* v_base =
      v_cache + (static_cast<std::size_t>(kv_head) * capacity * kHeadDim) + v_slice;
  const auto* k_base = k_cache + (static_cast<std::size_t>(kv_head) * capacity * kHeadDim);

  const auto load_v = [&](const Tile& tile, uint4* dst) {
    const std::uint32_t key_position = tile_key(tile, v_key);
    const bool live = key_position < context_end;
    const auto* src =
        v_base +
        (static_cast<std::size_t>(live ? key_position : 0) * kHeadDim);
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      dst[j] = live ? *reinterpret_cast<const uint4*>(src + (j * 8))
                    : make_uint4(0u, 0u, 0u, 0u);
    }
  };
  // Coalesced: a wave reads one key row's 512 contiguous bytes.
  const auto load_k = [&](const Tile& tile, uint4* dst) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_position = tile_key(tile, idx / (kHeadDim / 8));
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      const bool live = key_position < context_end;
      dst[n] =
          live ? *reinterpret_cast<const uint4*>(
                     k_base +
                     (static_cast<std::size_t>(key_position) * kHeadDim) +
                     d8)
               : make_uint4(0u, 0u, 0u, 0u);
    }
  };

  const auto stage_k = [&](const uint4* regs) {
#pragma unroll
    for (std::uint32_t n = 0; n < kKRegs; ++n) {
      const std::uint32_t idx = tid + (n * 256);
      const std::uint32_t key_row = idx / (kHeadDim / 8);
      const std::uint32_t d8 = (idx % (kHeadDim / 8)) * 8;
      *reinterpret_cast<uint4*>(&k_lds[(key_row * kWmmaKStride) + d8]) =
          regs[n];
    }
  };
  const auto stage_v = [&](const uint4* regs) {
    // Neighbouring lanes hold neighbouring keys of the same dims. Swapping
    // one half with the partner lets each lane store a key pair of one dim
    // as a word: the even lane the even dim, the odd lane the odd dim.
    const bool odd = (v_key & 1u) != 0u;
#pragma unroll
    for (std::uint32_t j = 0; j < kVRegs; ++j) {
      const std::uint32_t words[4] = {regs[j].x, regs[j].y, regs[j].z,
                                      regs[j].w};
#pragma unroll
      for (std::uint32_t w = 0; w < 4; ++w) {
        const std::uint32_t own = words[w];
        const auto partner =
            static_cast<std::uint32_t>(__builtin_amdgcn_mov_dpp(
                static_cast<int>(own), 0xB1, 0xF, 0xF, false));
        const std::uint32_t pair =
            odd ? __builtin_amdgcn_perm(partner, own, 0x03020706U)
                : __builtin_amdgcn_perm(partner, own, 0x05040100U);
        const std::uint32_t dim = v_slice + (j * 8) + (2 * w) + (odd ? 1 : 0);
        *reinterpret_cast<std::uint32_t*>(
            &v_lds[(dim * kVtStride) + (v_key & ~1u)]) = pair;
      }
    }
  };

  // --- S = Q K^T of the tile in k_lds. Each half writes its own slot; the
  // reader sums them.
  const auto scores = [&] {
    if (wave < 2 * kSTiles) {
      v8f s_acc = {};
#pragma unroll
      for (std::uint32_t ks = 0; ks < kKStepsPerWave; ++ks) {
        const std::uint32_t d0 = ((s_kh * kKStepsPerWave) + ks) * 16;
        const v16h k_frag =
            LoadFrag(&k_lds[(((s_kb * 16) + sub) * kWmmaKStride) + d0]);
        s_acc = Wmma(q_frag[ks], k_frag, s_acc);
        if constexpr (kPackHeads) {
          // Bound temporary K fragments without changing the dot-product
          // sequence. Hoisting all eight LDS fragments raises register use.
          if ((ks + 1) % 2 == 0)
            __builtin_amdgcn_sched_barrier(0);
        }
      }
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        s_lds[s_kh][s_tile][(2 * i) + half_id][sub] = s_acc[i];
      }
    }
  };

  // --- online softmax of the scores in s_lds: kSoftmaxLanes threads per
  // query row ---
  const auto softmax = [&](const Tile& tile) {
    const std::uint32_t rg = tid / kSoftmaxLanes;
    if (rg < kRows) {
      const std::uint32_t seg = tid % kSoftmaxLanes;
      constexpr std::uint32_t kPerLane = kKeys / kSoftmaxLanes;
      const std::uint32_t rb = rg / 16;
      const std::uint32_t row = rg % 16;
      const std::uint32_t local_query = row_query(rb, row);
      const bool live_row = row_live(rb, row);
      const std::uint32_t absolute_query = start_pos + local_query;
      // Keys in the query's own incomplete block are always visible; earlier
      // blocks follow the selection mask.
      const std::uint32_t tail_start = ((absolute_query + 1) / ratio) * ratio;
      // The dense window runs without a selection mask.
      const std::uint32_t* words =
          !sparse || mask == nullptr || !live_row
              ? nullptr
              : mask + static_cast<std::size_t>(local_query) * mask_words;
      float part_max = -INFINITY;
      float vals[kPerLane];
      float s0v[kPerLane];
      float s1v[kPerLane];
      // A lane's four keys form one selection block, so the sparse window
      // decides its mask bit once per lane instead of once per key.
      static_assert(kPerLane == ratio, "one selection block per lane");
      bool block_selected = true;
      std::uint32_t blk = 0;
      if constexpr (sparse) {
        blk = n_blocks;
#pragma unroll
        for (std::uint32_t j = 0; j < kBlocksPerTile; ++j) {
          if (seg == j) {
            blk = tile.block[j];
          }
        }
        if (words != nullptr && blk < n_blocks && blk * ratio < tail_start) {
          const std::uint32_t member =
              (seg * kMaskRows) + (local_query - query_start);
          block_selected = compact
                               ? ((tile.members >> member) & 1u) != 0u
                               : ((words[blk / 32] >> (blk % 32)) & 1u) != 0u;
        }
      }
      if constexpr (kVectorSoftmax) {
        const float4 s0 = *reinterpret_cast<const float4*>(
            &s_lds[0][rb][row][seg * kPerLane]);
        const float4 s1 = *reinterpret_cast<const float4*>(
            &s_lds[1][rb][row][seg * kPerLane]);
        s0v[0] = s0.x, s0v[1] = s0.y, s0v[2] = s0.z, s0v[3] = s0.w;
        s1v[0] = s1.x, s1v[1] = s1.y, s1v[2] = s1.z, s1v[3] = s1.w;
      } else {
        // Only lanes with a visible block read their scores, which saves
        // LDS bandwidth, but under one condition so the reads share a wait.
#pragma unroll
        for (std::uint32_t m = 0; m < kPerLane; ++m)
          s0v[m] = s1v[m] = 0.0F;
        if (live_row && block_selected && blk < n_blocks) {
#pragma unroll
          for (std::uint32_t m = 0; m < kPerLane; ++m) {
            const std::uint32_t col = (seg * kPerLane) + m;
            const std::uint32_t s_index = ((col / 16) * kRowBlocks) + rb;
            s0v[m] = s_lds[0][s_index][row][col % 16];
            s1v[m] = s_lds[1][s_index][row][col % 16];
          }
        }
      }
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const std::uint32_t col = (seg * kPerLane) + m;
        const std::uint32_t key_position = tile_key(tile, col);
        const bool valid = live_row && key_position <= absolute_query &&
                           key_position < context_end && block_selected;
        vals[m] = valid ? (s0v[m] + s1v[m]) : -INFINITY;
        part_max = fmaxf(part_max, vals[m]);
      }
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_max = fmaxf(part_max, XorLane(part_max, off));
      }
      const float prev_max = running_max;
      const float next_max = fmaxf(prev_max, part_max);
      const float prior_scale =
          isfinite(prev_max) ? __expf(prev_max - next_max) : 0.0F;
      float part_sum = 0.0F;
      kv16 p[kPerLane];
#pragma unroll
      for (std::uint32_t m = 0; m < kPerLane; ++m) {
        const float w = isfinite(vals[m]) ? __expf(vals[m] - next_max) : 0.0F;
        part_sum += w;
        p[m] = FloatToBf16(w);
        if constexpr (!kVectorSoftmax)
          p_lds[rg][(seg * kPerLane) + m] = p[m];
      }
      if constexpr (kVectorSoftmax)
        __builtin_memcpy(&p_lds[rg][seg * kPerLane], p, sizeof(p));
#pragma unroll
      for (std::uint32_t off = 1; off < kSoftmaxLanes; off <<= 1) {
        part_sum += XorLane(part_sum, off);
      }
      running_max = next_max;
      // The completed partial sum, then one fused update: written out so
      // fast-math cannot reassociate the reduction into the update.
      running_sum = __fmaf_rn(prior_scale, running_sum, part_sum);
      if (seg == 0) {
        row_scale[kVectorSoftmax ? (rb * 16) + ((row % 2) * 8) + (row / 2)
                                 : rg] = prior_scale;
      }
    }
  };

  // --- rescale the running O by the tile's new maximum, then O += P V. A
  // lane touches only rows 2i + half_id, so the factors are read once per
  // row block.
  const auto accumulate = [&](const Tile& tile) {
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
      float scale[8];
      if constexpr (kVectorSoftmax) {
        const auto* factors = reinterpret_cast<const float4*>(
            &row_scale[(rb * 16) + (half_id * 8)]);
        const float4 f0 = factors[0];
        const float4 f1 = factors[1];
        scale[0] = f0.x, scale[1] = f0.y, scale[2] = f0.z, scale[3] = f0.w;
        scale[4] = f1.x, scale[5] = f1.y, scale[6] = f1.z, scale[7] = f1.w;
      } else {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i)
          scale[i] = row_scale[(rb * 16) + (2 * i) + half_id];
      }
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t) {
#pragma unroll
        for (std::uint32_t i = 0; i < 8; ++i) {
          o_acc[rb][t][i] *= scale[i];
        }
      }
    }
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
      v16h v_frag[kKeyBlocks];
#pragma unroll
      for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
        v_frag[kb] =
            LoadFrag(&v_lds[(((dim_tile * 16) + sub) * kVtStride) + (kb * 16)]);
      }
#pragma unroll
      for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
        for (std::uint32_t kb = 0; kb < kKeyBlocks; ++kb) {
          const v16h p_frag = LoadFrag(&p_lds[(rb * 16) + sub][kb * 16]);
          v8f next = Wmma(p_frag, v_frag[kb], o_acc[rb][t]);
          if constexpr (!kPackHeads) {
            const std::uint32_t first_key = tile.block[0] * ratio;
            // WMMA's dot accumulation can change its last bits when zero
            // probabilities multiply populated future values instead of the
            // zero padding at a chunk boundary. Accumulate each query's final
            // partial tile over its visible keys only. Earlier complete tiles
            // retain the matrix-core path.
            if (first_key + kKeys - 1 > start_pos + query_start) {
#pragma unroll
              for (std::uint32_t i = 0; i < 8; ++i) {
                const std::uint32_t row = (2 * i) + half_id;
                const std::uint32_t absolute_query =
                    start_pos + row_query(rb, row);
                if (absolute_query < first_key + kKeys - 1) {
                  float acc = o_acc[rb][t][i];
                  for (std::uint32_t key = 0;
                       key < kKeys && first_key + key <= absolute_query;
                       ++key) {
                    acc = fmaf(
                        Bf16ToFloat(p_lds[(rb * 16) + row][key]),
                        Bf16ToFloat(
                            v_lds[((dim_tile * 16) + sub) * kVtStride + key]),
                        acc);
                  }
                  next[i] = acc;
                }
              }
            }
          }
          o_acc[rb][t] = next;
        }
      }
    }
  };

  // Software-pipelined key sweep. Phase A of step j accumulates tile j's PV
  // while the score waves compute tile j + 1's S; phase B runs tile j + 1's
  // softmax while staging K for tile j + 2 and V for tile j + 1, whose
  // registers were loaded one step earlier. Every row sees the same S, P,
  // rescale and accumulation sequence as a sequential sweep.
  uint4 k_regs[kKRegs] = {};
  uint4 v_regs[kVRegs] = {};
  Tile cur;     // tile j: P in p_lds, V in v_lds
  Tile next;    // tile j + 1: K in k_lds
  Tile k_tile;  // tile j + 2: K in k_regs
  Tile v_tile;  // tile j + 1 V in v_regs
  std::uint32_t cursor = gather_tile(0, cur);
  if (cur.count != 0) {
    load_k(cur, k_regs);
    load_v(cur, v_regs);
  }
  stage_k(k_regs);
  stage_v(v_regs);
  cursor = gather_tile(cursor, next);
  if (next.count != 0) {
    load_k(next, k_regs);
    load_v(next, v_regs);
  }
  SyncLds();
  if (cur.count != 0)
    scores();
  SyncLds();
  if (cur.count != 0)
    softmax(cur);
  stage_k(k_regs);
  v_tile = next;
  cursor = gather_tile(cursor, k_tile);
  if (k_tile.count != 0)
    load_k(k_tile, k_regs);
  SyncLds();

  while (cur.count != 0) {
    // Phase A: k_lds = K(j + 1), v_lds = V(j), p_lds = P(j).
    if (next.count != 0)
      scores();
    accumulate(cur);
    // Finish PV before the barrier. Left alone, the compiler sinks these
    // WMMAs into phase B behind the softmax, which is the critical path.
#pragma unroll
    for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb)
#pragma unroll
      for (std::uint32_t t = 0; t < 2; ++t)
        asm volatile("" : "+v"(o_acc[rb][t]));
    SyncLds();
    // Phase B.
    if (next.count != 0)
      softmax(next);
    stage_v(v_regs);
    stage_k(k_regs);
    cur = next;
    next = k_tile;
    v_tile = next;
    cursor = gather_tile(cursor, k_tile);
    if (k_tile.count != 0)
      load_k(k_tile, k_regs);
    if (v_tile.count != 0)
      load_v(v_tile, v_regs);
    SyncLds();
  }
  if (tid < kRows * kSoftmaxLanes && tid % kSoftmaxLanes == 0)
    row_sum[tid / kSoftmaxLanes] = running_sum;
  SyncLds();

  // --- epilogue: normalize and apply the sigmoid output gate ---
#pragma unroll
  for (std::uint32_t rb = 0; rb < kRowBlocks; ++rb) {
#pragma unroll
    for (std::uint32_t t = 0; t < 2; ++t) {
      const std::uint32_t dim_tile = wave + (t * 8);
#pragma unroll
      for (std::uint32_t i = 0; i < 8; ++i) {
        const std::uint32_t row = (2 * i) + half_id;
        if (!row_live(rb, row)) {
          continue;
        }
        const float denominator = row_sum[(rb * 16) + row];
        const std::size_t offset =
            (static_cast<std::size_t>(row_query(rb, row)) * kWmmaAttnWidth) +
            (static_cast<std::size_t>(row_head(rb, row)) * kHeadDim) +
            (dim_tile * 16) + sub;
        float value =
            (denominator > 0.0F) ? (o_acc[rb][t][i] / denominator) : 0.0F;
        if (gate != nullptr) {
          value *= SigmoidF(gate[offset]);
        }
        out[offset] = Bf16ToFloat(FloatToBf16(value));
      }
    }
  }
}
template __global__ void WmmaCausalAttentionKernel<4, 16, true>(const float*, const float*, const kv16*, const kv16*, const std::uint32_t*, std::uint32_t, float*, std::uint32_t, std::uint32_t, std::uint32_t, std::uint64_t);

// Row r's selection as one bit a block for the kernel above, from tf_fn_idx_select's key list: the listed blocks below
// the row's complete blocks, or every block when the row is not sparse. One workgroup a row, `words` <= kMaskWords.
constexpr int kMaskWords = 4096;
extern "C" __global__ void __launch_bounds__(256)
tf_fn_idx_mask(const int* ids, long long id_stride, const int* count, const int* sparse, const int* complete,
               unsigned* mask, unsigned words, int ratio) {
  __shared__ unsigned bits[kMaskWords];
  const unsigned r = blockIdx.x, t = threadIdx.x;
  const bool listed = sparse[r] != 0;
  for (unsigned w = t; w < words; w += 256) bits[w] = listed ? 0u : ~0u;
  __syncthreads();
  if (listed) {
    const int n = count[r] / ratio, nb = complete[r];
    const int* row = ids + r * id_stride;
    for (int i = t; i < n; i += 256) {
      const int b = row[static_cast<long long>(i) * ratio] / ratio;
      if (b < nb) atomicOr(&bits[b >> 5], 1u << (b & 31));
    }
  }
  __syncthreads();
  unsigned* out = mask + static_cast<std::size_t>(r) * words;
  for (unsigned w = t; w < words; w += 256) out[w] = bits[w];
}

}  // namespace gufo_attn
