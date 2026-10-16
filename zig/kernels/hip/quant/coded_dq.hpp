#pragma once

// prompt_dq_tile's hooks for the coded weights (quant/coded.hpp): a prompt's products dequantize them to BF16 in LDS.
//   Fp8Dq   a stage's 32 codes of a column times its group's fp32 scale, each rounded once to BF16
//   Bf16Dq  a stage's 32 BF16 as stored (exact)

#include "quant/coded.hpp"
#include "tiles/matrix_prompt.hpp"

namespace tf {
namespace rocm {

template <int P>
struct Fp8Dq {
    static_assert(P == 1 || P == 2, "16 or 32 codes a thread");
    static constexpr int kDepth = 4;
    static constexpr int kVecs = 2 / P;  // 16-byte code loads a stage (16 codes each)
    __amdgpu_buffer_rsrc_t words, scale, bias;
    uint32_t wo, to;  // a lane's byte offsets
    u32x4 W[kDepth][kVecs];
    float S[kDepth];

    __device__ Fp8Dq(const Affine& a, int col, int part)
        : words(prompt_rsrc(a.words)),
          scale(prompt_rsrc(a.scale.p)),
          bias(prompt_rsrc(a.bias.p)),
          wo(static_cast<uint32_t>(col) * static_cast<uint32_t>(a.k) + part * 16 * kVecs),
          to(static_cast<uint32_t>(col) * static_cast<uint32_t>(a.k >> 7) * 2) {}

    template <int J>
    __device__ void fetch(int s) {
#pragma unroll
        for (int v = 0; v < kVecs; ++v) W[J][v] = prompt_load<u32x4>(words, wo, 32 * s + 16 * v);
        const uint32_t lo = prompt_load<unsigned short>(scale, to, 2 * (s >> 2));
        const uint32_t hi = prompt_load<unsigned short>(bias, to, 2 * (s >> 2));
        S[J] = __uint_as_float(lo | hi << 16);
    }

    template <int J, class Store>
    __device__ void put(Store st) const {
        const float sc = S[J];
#pragma unroll
        for (int v = 0; v < kVecs; ++v) {
            const uint32_t c[4] = {W[J][v].x, W[J][v].y, W[J][v].z, W[J][v].w};
#pragma unroll
            for (int h = 0; h < 2; ++h) {
                uint32_t d[4];
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const uint32_t q = c[2 * h + j];
                    const uint32_t lo = fp8_pair(q);
                    const uint32_t hi = fp8_pair(q >> 16);
                    d[2 * j] = prompt_pack_rn(__uint_as_float(lo << 16) * sc, __uint_as_float(lo & 0xffff0000u) * sc);
                    d[2 * j + 1] = prompt_pack_rn(__uint_as_float(hi << 16) * sc, __uint_as_float(hi & 0xffff0000u) * sc);
                }
                st(2 * v + h, u32x4{d[0], d[1], d[2], d[3]});
            }
        }
    }
};

template <int P>
struct Bf16Dq {
    static_assert(P == 1 || P == 2, "16 or 32 values a thread");
    static constexpr int kDepth = 4;
    static constexpr int kVecs = 4 / P;  // 16-byte loads a stage (8 values each)
    __amdgpu_buffer_rsrc_t words;
    uint32_t wo;
    u32x4 W[kDepth][kVecs];

    __device__ Bf16Dq(const Affine& a, int col, int part)
        : words(prompt_rsrc(a.words)), wo(static_cast<uint32_t>(col) * static_cast<uint32_t>(a.k) * 2 + part * 16 * kVecs) {}

    template <int J>
    __device__ void fetch(int s) {
#pragma unroll
        for (int v = 0; v < kVecs; ++v) W[J][v] = prompt_load<u32x4>(words, wo, 64 * s + 16 * v);
    }

    template <int J, class Store>
    __device__ void put(Store st) const {
#pragma unroll
        for (int v = 0; v < kVecs; ++v) st(v, W[J][v]);
    }
};

}  // namespace rocm
}  // namespace tf
