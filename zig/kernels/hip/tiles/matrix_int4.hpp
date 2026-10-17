#pragma once

// Opt-in W4A4 prompt GEMM (TF_PROMPT_INT4) for MLX 4-bit group 64 with BF16 tables, gfx11: x is quantized per row and
// 64-group to signed int4 (absmax / 7, round to nearest) and each group's codes meet the weight codes in iu4 WMMAs
// (int32), then y += sx (sw dot + bw sum(codes)). Changes the bits. Measured on the 27B: KL vs fp32 0.112 -> 0.277,
// top-1 91.1% -> 84.2% (fails the gate); the tile is a correct but unpipelined reference (~15 TOPS vs the bf16 tile's 43).

#include "common/wmma.hpp"
#include "tiles/epilogue.hpp"

namespace tf {
namespace rocm {

struct Int4Prompt {
    const void* x;          // (m, k) bf16
    uint32_t* xq;           // (m, k / 8): x's int4 codes, low nibble first
    float* xs;              // (k / 64, mp): x's group scales, rows permuted by int4_row_at
    float* sums;            // (k / 64, mp): the quantized x's group sums, same order
    const uint32_t* words;  // (n, k / 8) MLX codes
    const uint16_t* scale;  // (n, k / 64) bf16
    const uint16_t* bias;
    uint16_t* out;  // (m, n) bf16
    int m, n, k, mp;  // mp: m rounded up to 128 (the tables' row stride)
    void* fake;       // set: also write x's dequantized codes (m, k) bf16 here (TF_PROMPT_INT4=2's check path)
    int rotate;       // the check path only: quantize the group's 64-point Hadamard rotation, rotate back (exact W H)
};

// A lane's eight rows 2 i + h of a 16-row tile as eight neighbours.
__device__ inline int int4_row_at(int r) { return (r & ~15) | ((r & 1) << 3) | ((r & 15) >> 1); }

// A thread a (row, group).
__device__ inline void int4_quant_x(const Int4Prompt& a) {
    const int groups = a.k >> 6;
    const long long t = blockIdx.x * static_cast<long long>(blockDim.x) + threadIdx.x;
    if (t >= static_cast<long long>(a.m) * groups) return;
    const int r = static_cast<int>(t / groups), g = static_cast<int>(t % groups);
    const u32x4* src = reinterpret_cast<const u32x4*>(static_cast<const uint16_t*>(a.x) + static_cast<long long>(r) * a.k + 64 * g);
    float v[64];
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const u32x4 u = src[i];
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            v[8 * i + 2 * j] = __uint_as_float(w[j] << 16);
            v[8 * i + 2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
        }
    }
    auto fwht = [&]() {
#pragma unroll
        for (int h = 1; h < 64; h <<= 1) {
#pragma unroll
            for (int i = 0; i < 64; ++i) {
                if (i & h) continue;
                const float p = v[i], q = v[i + h];
                v[i] = p + q;
                v[i + h] = p - q;
            }
        }
#pragma unroll
        for (int i = 0; i < 64; ++i) v[i] *= 0.125f;
    };
    if (a.rotate) fwht();
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) amax = fmaxf(amax, fabsf(v[i]));
    const float inv = amax > 0.f ? 7.f / amax : 0.f;
    uint32_t* dst = a.xq + static_cast<long long>(r) * (a.k >> 3) + 8 * g;
    int codes = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        uint32_t w = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c = __float2int_rn(v[8 * i + j] * inv);
            codes += c;
            w |= (static_cast<uint32_t>(c) & 15u) << (4 * j);
        }
        dst[i] = w;
    }
    // The bias term takes the quantized x's sum: y's error is then sum (s q + b) e over the centred w, where the exact
    // sum would leave s sum q e over the uncentred codes (measured ~2x the output error).
    const float s = amax / 7.f * static_cast<float>(codes);
    if (a.fake) {  // the check path: x's dequantized codes as bf16 for the bf16 tile
        u32x4* fx = reinterpret_cast<u32x4*>(static_cast<uint16_t*>(a.fake) + static_cast<long long>(r) * a.k + 64 * g);
        const float d = amax / 7.f;
#pragma unroll
        for (int i = 0; i < 64; ++i) v[i] = d * static_cast<float>(__float2int_rn(v[i] * inv));
        if (a.rotate) fwht();
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            uint32_t w[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float lo = v[8 * i + 2 * j];
                const float hi = v[8 * i + 2 * j + 1];
                w[j] = ((__float_as_uint(lo) + 0x8000u) >> 16) | ((__float_as_uint(hi) + 0x8000u) & 0xffff0000u);  // d code to bf16
            }
            fx[i] = u32x4{w[0], w[1], w[2], w[3]};
        }
    }
    const long long at = static_cast<long long>(g) * a.mp + int4_row_at(r);
    a.xs[at] = amax / 7.f;
    a.sums[at] = s;
}

typedef int int4_v2i __attribute__((ext_vector_type(2)));
typedef int int4_v8i __attribute__((ext_vector_type(8)));

// 128 x 128 a block, 8 waves of 32 x 64 (2 x 4 WMMA tiles), operands straight from global one group ahead.
// ponytail: no LDS staging or pipelining yet; the quality verdict comes first.
__device__ inline void int4_prompt_tile(const Int4Prompt& a) {
#if TF_DEVICE_WMMA_GFX11
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
    const int ntiles = (a.n + 127) / 128;
    const int m0 = (blockIdx.x / ntiles) * 128, n0 = (blockIdx.x % ntiles) * 128;
    const int wm = wave & 3, wn = wave >> 2, rl = lane & 15, rh = lane >> 4;
    const int groups = a.k >> 6, kw = a.k >> 3;
    const u32x4* xa[2];
    const u32x4* wa[4];
    int cols[4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
        const int r = min(m0 + wm * 32 + mt * 16 + rl, a.m - 1);
        xa[mt] = reinterpret_cast<const u32x4*>(a.xq + static_cast<long long>(r) * kw);
    }
#pragma unroll
    for (int nt = 0; nt < 4; ++nt) {
        cols[nt] = min(n0 + wn * 64 + nt * 16 + rl, a.n - 1);
        wa[nt] = reinterpret_cast<const u32x4*>(a.words + static_cast<long long>(cols[nt]) * kw);
    }
    const int rb = m0 + wm * 32 + rh * 8;
    f32x8 acc[2][4] = {};
    u32x4 X[2][2], W[4][2];
    auto load = [&](int g) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) X[mt][0] = xa[mt][2 * g], X[mt][1] = xa[mt][2 * g + 1];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) W[nt][0] = wa[nt][2 * g], W[nt][1] = wa[nt][2 * g + 1];
    };
    load(0);
    for (int g = 0; g < groups; ++g) {
        u32x4 Xc[2][2], Wc[4][2];
#pragma unroll
        for (int i = 0; i < 2; ++i) Xc[i][0] = X[i][0], Xc[i][1] = X[i][1];
#pragma unroll
        for (int i = 0; i < 4; ++i) Wc[i][0] = W[i][0], Wc[i][1] = W[i][1];
        float sw[4], bw[4], sx[2][8], su[2][8];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            sw[nt] = __uint_as_float(static_cast<uint32_t>(a.scale[static_cast<long long>(cols[nt]) * groups + g]) << 16);
            bw[nt] = __uint_as_float(static_cast<uint32_t>(a.bias[static_cast<long long>(cols[nt]) * groups + g]) << 16);
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            const float* p = a.xs + static_cast<long long>(g) * a.mp + rb + 16 * mt;
            const float* q = a.sums + static_cast<long long>(g) * a.mp + rb + 16 * mt;
#pragma unroll
            for (int i = 0; i < 8; ++i) sx[mt][i] = p[i], su[mt][i] = q[i];
        }
        if (g + 1 < groups) load(g + 1);
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                int4_v8i d = {0, 0, 0, 0, 0, 0, 0, 0};
                const uint32_t xw[8] = {Xc[mt][0].x, Xc[mt][0].y, Xc[mt][0].z, Xc[mt][0].w, Xc[mt][1].x, Xc[mt][1].y, Xc[mt][1].z, Xc[mt][1].w};
                const uint32_t ww[8] = {Wc[nt][0].x, Wc[nt][0].y, Wc[nt][0].z, Wc[nt][0].w, Wc[nt][1].x, Wc[nt][1].y, Wc[nt][1].z, Wc[nt][1].w};
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const int4_v2i A = {static_cast<int>(xw[2 * t]), static_cast<int>(xw[2 * t + 1])};
                    const int4_v2i B = {static_cast<int>(ww[2 * t]), static_cast<int>(ww[2 * t + 1])};
                    d = __builtin_amdgcn_wmma_i32_16x16x16_iu4_w32(true, A, false, B, d, false);  // x signed, codes unsigned
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[mt][nt][i] = fmaf(static_cast<float>(d[i]), sx[mt][i] * sw[nt], acc[mt][nt][i]);
                    acc[mt][nt][i] = fmaf(su[mt][i], bw[nt], acc[mt][nt][i]);
                }
            }
        }
    }
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
            if (row >= a.m) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int c = n0 + wn * 64 + nt * 16 + rl;
                if (c < a.n) a.out[static_cast<long long>(row) * a.n + c] = __builtin_bit_cast(uint16_t, narrow<WmmaBF16>(acc[mt][nt][i]));
            }
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
