#pragma once

// The FP4-Local MXFP8 WeightDecoder: e4m3 bytes [n][k] and e8m0 block scales [n][k / 32] (a.scale.p), no global.
// Each weight is baked into its exact bf16 (fp4local_bake.hpp); the prompt tiles fold every 128 inputs with scale 1.

#include <type_traits>

#include "common/vec.hpp"
#include "quant/fp4local_bake.hpp"
#include "quant/mlx.hpp"

namespace tf {
namespace rocm {

// The prefill fold of FP4-Local (core/quant/fp4local_decode.zig `fold`): fixed, another value changes every bit.
constexpr int kFp4LocalFold = 128;

struct Mxfp8Decoder {
    using Args = Affine;
    static constexpr int kBits = 8;
    static constexpr int kWords = 8;
    static constexpr bool kBias = false;

    __device__ static int fold_k(const Args&) { return kFp4LocalFold; }

    // A 32-input stage is one e8m0 block: its multiplier.
    struct Stage {
        float mult;
    };
    __device__ static Stage term_stage(const Args& a, long long col, int stage) {
        const uint8_t* e = static_cast<const uint8_t*>(a.scale.p);
        return {mxfp8_multiplier(e[col * (a.k >> 5) + stage])};
    }
    __device__ static float output(const Args&, int, float v) { return v; }

    __device__ static bool take_item(Args& a, int) { return a.route.items == nullptr; }

    __device__ static const uint32_t* words(const Args& a) { return a.words; }

    template <typename Idx>
    __device__ static Idx row_words(const Args& a) {
        return static_cast<Idx>(a.k) / 4;
    }

    template <bool WIDE>
    __device__ static void load(const uint32_t* src, uint32_t (&w)[kWords]) {
#pragma unroll
        for (int i = 0; i < kWords / 4; ++i) {
            const u32x4 v = reinterpret_cast<const u32x4*>(src)[i];
            w[4 * i] = v.x;
            w[4 * i + 1] = v.y;
            w[4 * i + 2] = v.z;
            w[4 * i + 3] = v.w;
        }
    }

    template <typename T>
    __device__ static typename T::pair pair(const uint32_t (&w)[kWords], int t, Stage s) {
        static_assert(std::is_same_v<typename T::elem, __bf16>, "FP4-Local weights take BF16 activations");
        return __builtin_bit_cast(typename T::pair, mxfp8_pair(w[t >> 2], t & 2, s.mult));
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
        return ((reinterpret_cast<uintptr_t>(a.words) | static_cast<uintptr_t>(a.k)) & 15) == 0;
    }

    template <class Sides>
    __device__ static void bind(Args& a, const Sides& sides, int s) {
        a.words = sides.words[s];
        a.scale.p = sides.scale[s];
    }

    template <typename T, int I, class Raw>
    __device__ static typename T::pair stream_pair(const uint32_t (&w)[kWords], const Raw& raw) {
        return __builtin_bit_cast(typename T::pair, mxfp8_pair(w[I >> 1], (I & 1) * 2, mxfp8_multiplier(raw.e)));
    }

    // A chunk's raw term is its e8m0 byte; the scale is in the weights, so a live chunk folds with 1.
    struct StreamTerms {
        const uint8_t* e;
        long long chunks;
        struct Raw {
            uint32_t e;
        };
        __device__ explicit StreamTerms(const Args& a) : e(static_cast<const uint8_t*>(a.scale.p)), chunks(a.k >> 5) {}
        __device__ long long row(long long col) const { return col * chunks; }
        __device__ Raw raw(long long base, int q, int) const { return {e[base + q]}; }
        __device__ float scale_value(Raw) const { return 1.f; }
        __device__ float bias_value(Raw) const { return 0.f; }
    };

    // A non-finite stream sum: the fp64 sum of the decoded weights when an x is non-finite, else the sum as it is.
    template <typename T>
    __device__ static float repair_stream(const Args& a, const typename T::elem* x, int col, float original) {
        const uint8_t* codes = reinterpret_cast<const uint8_t*>(a.words) + static_cast<long long>(col) * a.k;
        const uint8_t* e = static_cast<const uint8_t*>(a.scale.p) + static_cast<long long>(col) * (a.k >> 5);
        bool exceptional = false;
        double y = 0.0;
        for (int i = 0; i < a.k; ++i) {
            const uint32_t b = codes[i];
            const uint32_t h = fp4l_e4m3_half2(b);
            const double w = static_cast<double>(__half2float(__ushort_as_half(static_cast<unsigned short>(h)))) *
                             static_cast<double>(mxfp8_multiplier(e[i >> 5]));
            const double v = static_cast<float>(x[i]);
            exceptional |= !__builtin_isfinite(v);
            y += w * v;
        }
        return exceptional ? static_cast<float>(y) : original;
    }
};

}  // namespace rocm
}  // namespace tf
