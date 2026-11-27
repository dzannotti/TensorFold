#pragma once

// Bf16Decoder with the decode stream tile's terms: FP4-Local's BF16 products (lm_head, mixers, router) on the shared
// tiles. A live chunk folds with scale 1 (fma(d, 1, acc) is acc + d), a dead one with 0; the prompt fold is a.group.

#include "quant/bf16_decoder.hpp"

namespace tf {
namespace rocm {

struct Bf16FoldDecoder : Bf16Decoder {
    __device__ static float fold_scale(float acc, float dot, float scale) { return fmaf(dot, scale, acc); }

    static constexpr int kWideBytes = 16;
    static constexpr bool kGapPairs = false;
    template <int I>
    static constexpr int lo() { return 2 * I; }
    static constexpr int gap() { return 1; }

    __device__ static bool wide_aligned(const Args& a) {
        return ((reinterpret_cast<uintptr_t>(a.words) | static_cast<uintptr_t>(a.k) * 2) & 15) == 0;
    }

    template <class Sides>
    __device__ static void bind(Args& a, const Sides& sides, int s) {
        a.words = sides.words[s];
    }

    template <typename T, int I, class Raw>
    __device__ static typename T::pair stream_pair(const uint32_t (&w)[kWords], const Raw&) {
        return __builtin_bit_cast(typename T::pair, w[I]);
    }

    struct StreamTerms {
        struct Raw {};
        __device__ explicit StreamTerms(const Args&) {}
        __device__ long long row(long long) const { return 0; }
        __device__ Raw raw(long long, int, int) const { return {}; }
        __device__ float scale_value(Raw) const { return 1.f; }
        __device__ float bias_value(Raw) const { return 0.f; }
    };

    // A non-finite stream sum: the fp64 sum when an operand is non-finite, else the sum as it is.
    template <typename T>
    __device__ static float repair_stream(const Args& a, const typename T::elem* x, int col, float original) {
        const unsigned short* w = reinterpret_cast<const unsigned short*>(a.words) + static_cast<long long>(col) * a.k;
        bool exceptional = false;
        double y = 0.0;
        for (int i = 0; i < a.k; ++i) {
            const double wv = __uint_as_float(static_cast<uint32_t>(w[i]) << 16);
            const double v = static_cast<float>(x[i]);
            exceptional |= !__builtin_isfinite(v) || !__builtin_isfinite(wv);
            y += wv * v;
        }
        return exceptional ? static_cast<float>(y) : original;
    }
};

}  // namespace rocm
}  // namespace tf
