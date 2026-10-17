// ggml block-quantized projections read in place from a GGUF file (quant/gguf.hpp's decoders): the product of any row
// count (decode and lane rounds), the dequantizing prompt tile, and the row gather (embeddings, tests).
//
//   gguf_product  each output is, per group of one (s, mn) in k order, the WMMA chain of x against the group's exact
//                 integer codes folded acc = fma(dot, s, acc), acc = fma(sum, mn, acc) with the row's x sum over the
//                 group (gguf_sums): the group's weights are exactly ggml's (codes below 128 enter as 128 + u, the fold
//                 taking mn - 128 s rounded once: within 2^-17 of a quantization step). A block's four waves take every fourth superblock and add their
//                 sums in wave order, so a row's bits depend on its x and (K, N) alone, never on m or the tile.
//   gguf_prompt   prompt_dq_tile over the decoders: a stage's weights as ggml's fp32 values rounded once to BF16.
//   gguf_chain    the prompt tile's sums for a prompt's few rows (up to 128): a wave's 16 columns, one WMMA chain
//                 over all of k of the same BF16 weights, so its rows carry the tile's bits.
//   gguf_rows     rows by id as ggml's fp32 values (fp32 out exactly dequantize_row_*), or rounded to the out kind.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "common/arch.hpp"
#include "quant/gguf.hpp"
#include "tiles/matrix_prompt.hpp"

namespace tf {
namespace rocm {

// ggml's iq3s_grid (ggml/src/ggml-common.h).
__constant__ uint32_t gg_iq3s_grid[512] = {
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
};

// One product: blocks [n, row_bytes], x bf16 [m, k], out [m, n] of out_kind (0 fp32, 1 bf16, 2 fp16); sums the
// rows' x sums a group (gguf_sums' layout), or null for the chain and prompt kernels.
struct Gguf {
    const uint8_t* w;
    const void* x;
    void* out;
    const float* sums;
    int m, n, k, row_bytes, out_kind;
};

// Where row r's sum of group g sits: 16-row tiles, a lane's eight rows 2 e + h of a tile as two 16-byte words.
__device__ inline long long gg_sum_at(int g, int r, int m) {
    const int mp = (m + 15) & ~15;
    return static_cast<long long>(g) * mp + (r & ~15) + (r & 1) * 8 + ((r & 15) >> 1);
}

// The rows' x sums over each G-element group, added in k order from zero in fp32: a thread a (row, group).
template <int G>
__global__ void __launch_bounds__(256) gguf_sums(const __bf16* x, float* sums, int m, int k) {
    const int groups = k / G;
    const long long i = static_cast<long long>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= static_cast<long long>(groups) * m) return;
    const int r = static_cast<int>(i / groups), g = static_cast<int>(i % groups);
    const uint32_t* src = reinterpret_cast<const uint32_t*>(x + static_cast<long long>(r) * k + static_cast<long long>(g) * G);
    float run = 0.f;
#pragma unroll
    for (int j = 0; j < G / 2; ++j) {
        const uint32_t w = src[j];
        run += __uint_as_float(w << 16);
        run += __uint_as_float(w & 0xffff0000u);
    }
    sums[gg_sum_at(g, r, m)] = run;
}

__device__ inline void gg_store(void* out, int kind, long long at, float v) {
    if (kind == 0) {
        static_cast<float*>(out)[at] = v;
    } else if (kind == 1) {
        static_cast<uint16_t*>(out)[at] = __builtin_bit_cast(uint16_t, narrow<WmmaBF16>(v));
    } else {
        static_cast<__half*>(out)[at] = __float2half_rn(v);
    }
}

constexpr int kGgWaves = 4;

// Columns 16 blockIdx.x.., rows 16 MT blockIdx.y..; wave w sums superblocks w, w + 4, ...
template <int T, int MT>
__global__ void __launch_bounds__(32 * kGgWaves) gguf_product(Gguf a) {
#if TF_DEVICE_WMMA_GFX11
    using Q = GguQ<T>;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int col = blockIdx.x * 16 + (lane & 15);
    const int col_s = col < a.n ? col : a.n - 1;
    const int m0 = blockIdx.y * 16 * MT;
    const uint8_t* row = a.w + static_cast<long long>(col_s) * a.row_bytes;
    const int nsb = a.k >> 8;
    // a lane's x rows (lane % 16 of each tile); rows past the call read its last row and are never stored
    const __bf16* xr[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int r = m0 + 16 * mt + (lane & 15);
        xr[mt] = static_cast<const __bf16*>(a.x) + static_cast<long long>(r < a.m ? r : a.m - 1) * a.k;
    }
    constexpr int G = Q::kGroup, PG = G / 16;  // pieces a group
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    f32x8 acc[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) acc[mt] = zero;
    for (int sb = wave; sb < nsb; sb += kGgWaves) {
        const uint8_t* blk = row + static_cast<long long>(sb) * Q::kBytes;
#pragma unroll
        for (int p0 = 0; p0 < 16; p0 += PG) {
            bf16x16 b[PG];
            float s, mn;
#pragma unroll
            for (int q = 0; q < PG; ++q) {
                uint32_t u[4];
                Q::piece(blk, p0 + q, u, s, mn);  // a group's pieces share s and mn
                uint32_t f[8];
                if constexpr (Q::kSmall) {
                    gg_pairs128(u, f);
                } else {
                    gg_pairs(u, f);
                }
                b[q] = __builtin_bit_cast(bf16x16, u32x8{f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]});
            }
            if constexpr (Q::kSmall) mn = fmaf(-128.0f, s, mn);  // the operands are 128 + u
            const int g = (sb * 256 + 16 * p0) / G;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                f32x8 d = zero;
#pragma unroll
                for (int q = 0; q < PG; ++q) d = WmmaBF16::mma(frag16(reinterpret_cast<const uint32_t*>(xr[mt] + sb * 256 + 16 * (p0 + q))), b[q], d);
                const int rt = m0 + 16 * mt < a.m ? m0 + 16 * mt : 0;  // a tile past the call: any sums, never stored
                const u32x4* sp = reinterpret_cast<const u32x4*>(a.sums + gg_sum_at(g, rt, a.m) + (lane >> 4) * 8);
                const u32x4 s0 = sp[0], s1 = sp[1];
                const float sx[8] = {__uint_as_float(s0.x), __uint_as_float(s0.y), __uint_as_float(s0.z), __uint_as_float(s0.w),
                                     __uint_as_float(s1.x), __uint_as_float(s1.y), __uint_as_float(s1.z), __uint_as_float(s1.w)};
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc[mt][e] = fmaf(d[e], s, acc[mt][e]);
                    acc[mt][e] = fmaf(sx[e], mn, acc[mt][e]);
                }
            }
        }
    }
    // the waves' sums in wave order: ((w0 + w1) + w2) + w3
    __shared__ float red[kGgWaves - 1][MT * 8][32];
    if (wave > 0) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int e = 0; e < 8; ++e) red[wave - 1][mt * 8 + e][lane] = acc[mt][e];
    }
    __syncthreads();
    if (wave != 0) return;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            float v = acc[mt][e];
#pragma unroll
            for (int w = 0; w < kGgWaves - 1; ++w) v += red[w][mt * 8 + e][lane];
            const int r = m0 + 16 * mt + 2 * e + (lane >> 4);
            if (r < a.m && col < a.n) gg_store(a.out, a.out_kind, static_cast<long long>(r) * a.n + col, v);
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

// A piece's weights fma(s, u, mn) rounded once to BF16 (prompt_pack_rn), as the prompt tile stages them: pairs (2 i, 2 i + 1).
__device__ inline void gg_dq_pairs(const uint32_t (&u)[4], float s, float mn, uint32_t (&f)[8]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t w = u[i];
        asm("" : "+v"(w));
        const float c0 = static_cast<float>(w & 0xffu), c1 = static_cast<float>((w >> 8) & 0xffu);
        const float c2 = static_cast<float>((w >> 16) & 0xffu), c3 = static_cast<float>(w >> 24);
        f[2 * i] = prompt_pack_rn(fmaf(s, c0, mn), fmaf(s, c1, mn));
        f[2 * i + 1] = prompt_pack_rn(fmaf(s, c2, mn), fmaf(s, c3, mn));
    }
}

// Columns 16 (4 blockIdx.x + wave).., rows 16 MT blockIdx.y..: one chain over k in order from zero, the prompt tile's.
template <int T, int MT>
__global__ void __launch_bounds__(32 * kGgWaves) gguf_chain(Gguf a) {
#if TF_DEVICE_WMMA_GFX11
    using Q = GguQ<T>;
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int col = (blockIdx.x * kGgWaves + wave) * 16 + (lane & 15);
    const int col_s = col < a.n ? col : a.n - 1;
    const int m0 = blockIdx.y * 16 * MT;
    const uint8_t* row = a.w + static_cast<long long>(col_s) * a.row_bytes;
    const int nsb = a.k >> 8;
    const __bf16* xr[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int r = m0 + 16 * mt + (lane & 15);
        xr[mt] = static_cast<const __bf16*>(a.x) + static_cast<long long>(r < a.m ? r : a.m - 1) * a.k;
    }
    f32x8 acc[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) acc[mt] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int sb = 0; sb < nsb; ++sb) {
        const uint8_t* blk = row + static_cast<long long>(sb) * Q::kBytes;
#pragma unroll
        for (int p = 0; p < 16; ++p) {
            uint32_t u[4];
            float s, mn;
            Q::piece(blk, p, u, s, mn);
            uint32_t f[8];
            gg_dq_pairs(u, s, mn, f);
            const bf16x16 b = __builtin_bit_cast(bf16x16, u32x8{f[0], f[1], f[2], f[3], f[4], f[5], f[6], f[7]});
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) acc[mt] = WmmaBF16::mma(frag16(reinterpret_cast<const uint32_t*>(xr[mt] + sb * 256 + 16 * p)), b, acc[mt]);
        }
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int r = m0 + 16 * mt + 2 * e + (lane >> 4);
            if (r < a.m && col < a.n) gg_store(a.out, a.out_kind, static_cast<long long>(r) * a.n + col, acc[mt][e]);
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

// prompt_dq_tile's hook (256 x 128 blocks: two threads a column, a piece each a stage): stage s is superblock s / 8,
// pieces 2 (s % 8) + part; each weight fma(s, u, mn) (ggml's fp32 value) rounded once to BF16.
template <int T>
struct GgufDq {
    template <int P>
    struct Hook {
        static_assert(P == 2, "two threads a column");
        using Q = GguQ<T>;
        static constexpr int kDepth = 4;
        const uint8_t* row;
        int part;
        typename Q::Raw R[kDepth];
        int pc[kDepth];

        __device__ Hook(const Affine& a, int col, int part_)
            : row(reinterpret_cast<const uint8_t*>(a.words) + static_cast<long long>(col) * (a.k >> 8) * Q::kBytes), part(part_) {}

        template <int J>
        __device__ void fetch(int s) {
            pc[J] = 2 * (s & 7) + part;
            Q::load(row + static_cast<long long>(s >> 3) * Q::kBytes, pc[J], R[J]);
        }

        template <int J, class Store>
        __device__ void put(Store st) const {
            uint32_t u[4];
            float sc, mn;
            Q::decode(R[J], pc[J], u, sc, mn);
            uint32_t f[8];
            gg_dq_pairs(u, sc, mn, f);
            st(0, u32x4{f[0], f[1], f[2], f[3]});
            st(1, u32x4{f[4], f[5], f[6], f[7]});
        }
    };
};

template <int T>
__global__ void __launch_bounds__(256) gguf_prompt(Affine a) {
    prompt_dq_tile<ActOut<WmmaBF16>, GgufDq<T>::template Hook, 256, 128>(a);
}

// Row blockIdx.x of the output: blocks row ids[blockIdx.x] (blockIdx.x without ids), k values of out_kind.
template <int T>
__global__ void __launch_bounds__(256) gguf_rows(const uint8_t* w, int row_bytes, const int* ids, int k, void* out, int out_kind) {
    using Q = GguQ<T>;
    const long long id = ids != nullptr ? ids[blockIdx.x] : blockIdx.x;
    const uint8_t* row = w + id * row_bytes;
    for (int p = threadIdx.x; p < (k >> 4); p += blockDim.x) {
        uint32_t u[4];
        float s, mn;
        Q::piece(row + static_cast<long long>(p >> 4) * Q::kBytes, p & 15, u, s, mn);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            const float v = fmaf(s, static_cast<float>((u[i >> 2] >> (8 * (i & 3))) & 0xffu), mn);
            gg_store(out, out_kind, static_cast<long long>(blockIdx.x) * k + 16 * p + i, v);
        }
    }
}

#define TF_GGUF(T)                                                                                         \
    template __global__ void gguf_product<T, 1>(Gguf);                                                    \
    template __global__ void gguf_product<T, 2>(Gguf);                                                    \
    template __global__ void gguf_product<T, 4>(Gguf);                                                    \
    template __global__ void gguf_product<T, 8>(Gguf);                                                    \
    template __global__ void gguf_chain<T, 1>(Gguf);                                                      \
    template __global__ void gguf_chain<T, 2>(Gguf);                                                      \
    template __global__ void gguf_chain<T, 4>(Gguf);                                                      \
    template __global__ void gguf_prompt<T>(Affine);                                                      \
    template __global__ void gguf_rows<T>(const uint8_t*, int, const int*, int, void*, int);
TF_GGUF(8)
TF_GGUF(11)
TF_GGUF(12)
TF_GGUF(13)
TF_GGUF(14)
TF_GGUF(20)
TF_GGUF(21)
TF_GGUF(23)
#undef TF_GGUF
template __global__ void gguf_sums<16>(const __bf16*, float*, int, int);
template __global__ void gguf_sums<32>(const __bf16*, float*, int, int);

}  // namespace rocm
}  // namespace tf
