// SPDX-License-Identifier: MIT
// Adapted from gufo (MIT, Copyright (c) 2026 gufo contributors): HcCombineMoeF16Kernel,
// src/models/qwen38_flash_next/kernels/rocm/kernels.hip.cpp. Changes: bf16 expert rows, residual streams and norm
// (FP4-Local's activations), the branch rounded to bf16 before the stream weights, no Q8 copy, no inject partials.
#pragma once

#include "../ops/common.hpp"

namespace {

constexpr int kHcMoeStreams = 4;
constexpr int kHcMoeDims = 2560;
constexpr int kHcMoeThreads = kHcMoeDims / 8;
constexpr int kHcMoeWaves = kHcMoeThreads / 32;
constexpr int kHcMoeSlots = 10;

typedef uint32_t hc_moe_u32x4 __attribute__((ext_vector_type(4)));
typedef float hc_moe_f32x4 __attribute__((ext_vector_type(4)));

__device__ __forceinline__ float hc_moe_lo(uint32_t w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float hc_moe_hi(uint32_t w) { return __uint_as_float(w & 0xffff0000u); }
__device__ __forceinline__ uint32_t hc_moe_pack(float lo, float hi) {
    return static_cast<uint32_t>(float_to_bf16(lo)) | (static_cast<uint32_t>(float_to_bf16(hi)) << 16);
}

// Block r, a thread 8 dims of every stream: branch = bf16(sum of the routed slots' bf16 rows y by wts in slot order
// + shared by the last weight); h_s = bf16(h_s + bf16(branch inject[s])) in place; x = bf16((h rinv) scale).
extern "C" __global__ void __launch_bounds__(kHcMoeThreads)
tf_fn_hc_moe_norm(uint16_t* h, const uint16_t* y, const float* wts, const float* shared, const uint16_t* inject,
                  const float* scale, uint16_t* x, int slots, float eps) {
    __shared__ float part[kHcMoeStreams][kHcMoeWaves];
    const long long r = blockIdx.x;
    const int i = threadIdx.x * 8;
    const float* w = wts + r * slots;
    const uint16_t* rows = y + r * slots * kHcMoeDims + i;
    float b[8] = {0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f, 0.f};
    if (slots == kHcMoeSlots + 1) {
        hc_moe_u32x4 e[kHcMoeSlots];
        float wk[kHcMoeSlots];
#pragma unroll
        for (int k = 0; k < kHcMoeSlots; ++k) {
            e[k] = *reinterpret_cast<const hc_moe_u32x4*>(rows + static_cast<long long>(k) * kHcMoeDims);
            wk[k] = w[k];
        }
#pragma unroll
        for (int k = 0; k < kHcMoeSlots; ++k) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                b[2 * j] += hc_moe_lo(e[k][j]) * wk[k];
                b[2 * j + 1] += hc_moe_hi(e[k][j]) * wk[k];
            }
        }
    } else {
        for (int k = 0; k < slots - 1; ++k) {
            const hc_moe_u32x4 e = *reinterpret_cast<const hc_moe_u32x4*>(rows + static_cast<long long>(k) * kHcMoeDims);
            const float wk = w[k];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                b[2 * j] += hc_moe_lo(e[j]) * wk;
                b[2 * j + 1] += hc_moe_hi(e[j]) * wk;
            }
        }
    }
    if (shared != nullptr) {
        const float g = w[slots - 1];
        const float* sh = shared + r * kHcMoeDims + i;
        const hc_moe_f32x4 s0 = *reinterpret_cast<const hc_moe_f32x4*>(sh);
        const hc_moe_f32x4 s1 = *reinterpret_cast<const hc_moe_f32x4*>(sh + 4);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            b[j] += s0[j] * g;
            b[4 + j] += s1[j] * g;
        }
    }
#pragma unroll
    for (int j = 0; j < 8; ++j) b[j] = bf16_to_float(float_to_bf16(b[j]));
    float v[kHcMoeStreams][8];
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5;
#pragma unroll
    for (int s = 0; s < kHcMoeStreams; ++s) {
        uint16_t* hs = h + (r * kHcMoeStreams + s) * kHcMoeDims + i;
        const float inj = bf16_to_float(inject[r * kHcMoeStreams + s]);
        const hc_moe_u32x4 hv = *reinterpret_cast<const hc_moe_u32x4*>(hs);
        hc_moe_u32x4 out;
        float sum = 0.f;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float lo = hc_moe_lo(hv[j]) + bf16_to_float(float_to_bf16(b[2 * j] * inj));
            const float hi = hc_moe_hi(hv[j]) + bf16_to_float(float_to_bf16(b[2 * j + 1] * inj));
            out[j] = hc_moe_pack(lo, hi);
            v[s][2 * j] = hc_moe_lo(out[j]);
            v[s][2 * j + 1] = hc_moe_hi(out[j]);
            sum += v[s][2 * j] * v[s][2 * j];
            sum += v[s][2 * j + 1] * v[s][2 * j + 1];
        }
        *reinterpret_cast<hc_moe_u32x4*>(hs) = out;
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
        if (lane == 0) part[s][wave] = sum;
    }
    __syncthreads();
#pragma unroll
    for (int s = 0; s < kHcMoeStreams; ++s) {
        float total = 0.f;
#pragma unroll
        for (int q = 0; q < kHcMoeWaves; ++q) total += part[s][q];
        const float rinv = rsqrtf(total / static_cast<float>(kHcMoeDims) + eps);
        const float* g = scale + s * kHcMoeDims + i;
        const hc_moe_f32x4 g0 = *reinterpret_cast<const hc_moe_f32x4*>(g);
        const hc_moe_f32x4 g1 = *reinterpret_cast<const hc_moe_f32x4*>(g + 4);
        hc_moe_u32x4 out;
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            out[j] = hc_moe_pack((v[s][2 * j] * rinv) * g0[2 * j], (v[s][2 * j + 1] * rinv) * g0[2 * j + 1]);
            out[2 + j] = hc_moe_pack((v[s][4 + 2 * j] * rinv) * g1[2 * j], (v[s][5 + 2 * j] * rinv) * g1[2 * j + 1]);
        }
        *reinterpret_cast<hc_moe_u32x4*>(x + (r * kHcMoeStreams + s) * kHcMoeDims + i) = out;
    }
}

}  // namespace
