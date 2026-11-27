#pragma once

// The FP4-Local NVFP4 WeightDecoder: baked words [n][k / 8] (nibbles c0 c2 c4 c6 c1 c3 c5 c7), e4m3 scales [n][k / 16]
// (a.scale.p), weight_scale_2 in a.global as a.global_layout lays it out. A routed stack is [E][n][...], globals [E][per].
// Weights are baked into exact bf16 e2m1 x e4m3 x 2^-14 (fp4local_bake.hpp); output() multiplies by global x 2^14.

#include <type_traits>

#include "common/vec.hpp"
#include "quant/fp4local_bake.hpp"
#include "quant/mlx.hpp"
#include "quant/mxfp8_decoder.hpp"

namespace tf {
namespace rocm {

struct Nvfp4Decoder {
    using Args = Affine;
    static constexpr int kBits = 4;
    static constexpr int kWords = 4;
    static constexpr bool kBias = false;

    __device__ static int fold_k(const Args&) { return kFp4LocalFold; }

    using Stage = Nvfp4Scales;
    __device__ static Stage term_stage(const Args& a, long long col, int stage) {
        const unsigned short* s = static_cast<const unsigned short*>(a.scale.p);
        return nvfp4_scales(s[col * (a.k >> 5) + stage]);
    }

    // The sum times the tensor's (or the expert's, or its gate's or up's) weight_scale_2, undoing the weights' 2^-14.
    __device__ static float output(const Args& a, int col, float v) {
        if (a.global_layout == kGlobalNone) return v;
        const int at = a.global_layout == kGlobalExpertPair && col >= (a.n >> 1) ? 1 : 0;
        return v * (a.global[at] * 0x1p14f);
    }

    // Item z: its expert's words, scales and globals, and its rows.
    __device__ static bool take_item(Args& a, int z) {
        if (!a.route.items) return true;
        const int* item = a.route.items + 3 * z;
        const int count = item[2];
        if (count <= 0) return false;
        const long long expert = item[0];
        a.words += expert * a.n * static_cast<long long>(a.k >> 3);
        a.scale.p = static_cast<const char*>(a.scale.p) + expert * a.n * static_cast<long long>(a.k >> 4);
        if (a.global_layout == kGlobalExpert) a.global += expert;
        if (a.global_layout == kGlobalExpertPair) a.global += 2 * expert;
        a.route.first = item[1];
        a.m = count;
        return true;
    }

    __device__ static const uint32_t* words(const Args& a) { return a.words; }

    template <typename Idx>
    __device__ static Idx row_words(const Args& a) {
        return static_cast<Idx>(a.k) / 8;
    }

    template <bool WIDE>
    __device__ static void load(const uint32_t* src, uint32_t (&w)[kWords]) {
        const u32x4 v = *reinterpret_cast<const u32x4*>(src);
        w[0] = v.x;
        w[1] = v.y;
        w[2] = v.z;
        w[3] = v.w;
    }

    // Weights t and t + 1 of a stage (t even): nibbles (t % 8) / 2 and + 4 of word t / 8, group t / 16's scale.
    template <typename T>
    __device__ static typename T::pair pair(const uint32_t (&w)[kWords], int t, Stage s) {
        static_assert(std::is_same_v<typename T::elem, __bf16>, "FP4-Local weights take BF16 activations");
        return __builtin_bit_cast(typename T::pair, nvfp4_pair(w[t >> 3], (t & 7) >> 1, t < 16 ? s.g0 : s.g1));
    }

    struct Term {};
    __device__ static Term term(const Args&, long long) { return {}; }
    __device__ static float scale_value(const Args&, const Term&) { return 1.f; }
    __device__ static float bias_value(const Args&, const Term&) { return 0.f; }
    __device__ static float fold_scale(float acc, float dot, float scale) { return fmaf(dot, scale, acc); }
    __device__ static float fold_bias(float acc, float, float) { return acc; }

    // ---- the decode stream tile ----

    static constexpr int kWideBytes = 16;
    static constexpr bool kGapPairs = false;
    template <int I>
    static constexpr int lo() { return 2 * I; }
    static constexpr int gap() { return 1; }

    __device__ static bool wide_aligned(const Args& a) {
        return ((reinterpret_cast<uintptr_t>(a.words) | static_cast<uintptr_t>(a.k >> 1)) & 15) == 0;
    }

    template <class Sides>
    __device__ static void bind(Args& a, const Sides& sides, int s) {
        a.words = sides.words[s];
        a.scale.p = sides.scale[s];
    }

    template <typename T, int I, class Raw>
    __device__ static typename T::pair stream_pair(const uint32_t (&w)[kWords], const Raw& raw) {
        const Stage s = nvfp4_scales(raw.two);
        return __builtin_bit_cast(typename T::pair, nvfp4_pair(w[I >> 2], I & 3, I < 8 ? s.g0 : s.g1));
    }

    // A chunk's raw terms are its two e4m3 scales (one u16); the scales are in the weights, so a live chunk folds with 1.
    struct StreamTerms {
        const unsigned short* s;
        long long chunks;
        struct Raw {
            uint32_t two;
        };
        __device__ explicit StreamTerms(const Args& a) : s(static_cast<const unsigned short*>(a.scale.p)), chunks(a.k >> 5) {}
        __device__ long long row(long long col) const { return col * chunks; }
        __device__ Raw raw(long long base, int q, int) const { return {s[base + q]}; }
        __device__ float scale_value(Raw) const { return 1.f; }
        __device__ float bias_value(Raw) const { return 0.f; }
    };

    // A non-finite stream sum: the fp64 sum of the decoded weights (global applied by output) when an x is non-finite.
    template <typename T>
    __device__ static float repair_stream(const Args& a, const typename T::elem* x, int col, float original) {
        const uint32_t* w = a.words + static_cast<long long>(col) * (a.k >> 3);
        const unsigned short* s = static_cast<const unsigned short*>(a.scale.p) + static_cast<long long>(col) * (a.k >> 5);
        bool exceptional = false;
        double y = 0.0;
        for (int i = 0; i < a.k; i += 2) {
            const Stage st = nvfp4_scales(s[i >> 5]);
            const uint32_t bf = nvfp4_pair(w[i >> 3], (i & 7) >> 1, (i & 31) < 16 ? st.g0 : st.g1);
            const double x0 = static_cast<float>(x[i]);
            const double x1 = static_cast<float>(x[i + 1]);
            exceptional |= !__builtin_isfinite(x0) || !__builtin_isfinite(x1);
            y += static_cast<double>(__uint_as_float(bf << 16)) * x0 + static_cast<double>(__uint_as_float(bf & 0xffff0000u)) * x1;
        }
        return exceptional ? static_cast<float>(y) : original;
    }
};

}  // namespace rocm
}  // namespace tf
