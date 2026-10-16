#pragma once

// Decoders for weights that are not affine codes, over the shared decode tiles (stream, WMMA verify and its split):
//   Fp8Decoder   block FP8: e4m3 codes [N, K] as stored (four a word), one fp32 scale a column and 128 k (a 128 x 128
//                checkpoint block's scale_inv repeated down its rows), kept as two 16-bit tables: low half in `scale`,
//                high half in `bias`, so the tiles' (scale | bias << 16) term word is the fp32 itself. y = sum_g s_g dot_g.
//   Bf16Decoder  bf16 [N, K] as stored (two a word), no terms: y = sum_g dot_g.
// Codes become BF16 exactly (an e4m3 has 4 significant bits), so each group's dot is the WMMA / dot2 sum of x times the
// weights, and the fold order is the affine decoder's: a row's bits depend on its own x and K alone.

#include <cstdint>

#include "common/vec.hpp"
#include "common/wmma.hpp"
#include "quant/mlx.hpp"

namespace tf {
namespace rocm {

// Two e4m3 codes (the low 16 bits of `w`) as a BF16 pair, exactly: placed as two fp16 (exponent bias 15 against 7,
// e4m3 subnormals land on fp16 subnormals), widened to fp32, times 2^8, top halves. 0x7f / 0xff (NaN) read as 480.
__device__ inline uint32_t fp8_pair(uint32_t w) {
    const uint32_t x = __builtin_amdgcn_perm(0u, w, 0x010c000cu);  // byte 0 -> bits 8-15, byte 1 -> bits 24-31
    const uint32_t h = ((x >> 1) & 0x3f803f80u) | (x & 0x80008000u);
    const float lo = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(h))) * 256.0f;
    const float hi = static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(h >> 16))) * 256.0f;
    return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo), 0x07060302u);
}

// What both coded decoders share with MlxDecoder's interface: the plan, the words, the loads and the sides.
template <int BITS>
struct CodedBase {
    static constexpr bool kCoded = true;
    using Args = Affine;
    static constexpr int kBits = BITS;
    static constexpr int kWords = BITS;  // words of a 32-code chunk
    static constexpr int kWideBytes = 16;
    static constexpr bool kGapPairs = false;

    __device__ static bool take_item(Args& a, int z) { return tf::rocm::take_item(a, z); }
    __device__ static const uint32_t* words(const Args& a) { return a.words; }

    template <typename Idx>
    __device__ static Idx row_words(const Args& a) {
        return static_cast<Idx>(a.k) * BITS / 32;
    }

    template <bool WIDE>
    __device__ static void load(const uint32_t* src, uint32_t (&w)[BITS]) {
        if constexpr (WIDE) {
#pragma unroll
            for (int i = 0; i < BITS / 4; ++i) {
                const u32x4 v = reinterpret_cast<const u32x4*>(src)[i];
                w[4 * i] = v.x;
                w[4 * i + 1] = v.y;
                w[4 * i + 2] = v.z;
                w[4 * i + 3] = v.w;
            }
        } else {
#pragma unroll
            for (int i = 0; i < BITS; ++i) w[i] = src[i];
        }
    }

    __device__ static bool wide_aligned(const Args& a) {
        const long long words_row = static_cast<long long>(a.k) * BITS / 32;
        return ((reinterpret_cast<uintptr_t>(a.words) | static_cast<uintptr_t>(words_row * 4)) & 15) == 0;
    }

    template <class Sides>
    __device__ static void bind(Args& a, const Sides& sides, int s) {
        a.words = sides.words[s];
        a.scale.p = sides.scale[s];
        a.bias.p = sides.bias[s];
    }

    template <int I>
    static constexpr int lo() {
        return 2 * I;
    }
    static constexpr int gap() { return 1; }
};

struct Fp8Decoder : CodedBase<8> {
    static constexpr bool kTerms = true;

    // codes 2 I and 2 I + 1 of a chunk as the BF16 pair
    template <typename T, int I>
    __device__ static typename T::pair stream_pair(const uint32_t (&w)[8]) {
        return __builtin_bit_cast(typename T::pair, fp8_pair(w[I >> 1] >> (16 * (I & 1))));
    }

    // codes 16 T .. 16 T + 15 as a WMMA B fragment, in K order
    template <int T, class Frag>
    __device__ static Frag frag(const uint32_t (&w)[8]) {
        u32x8 f;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            f[2 * i] = fp8_pair(w[4 * T + i]);
            f[2 * i + 1] = fp8_pair(w[4 * T + i] >> 16);
        }
        return __builtin_bit_cast(Frag, f);
    }

    // the group's fp32 scale from the term word; no bias
    template <class Terms>
    __device__ static float term_scale(const Terms&, uint32_t g) {
        return __uint_as_float(g);
    }
    template <class Terms>
    __device__ static float term_bias(const Terms&, uint32_t) {
        return 0.f;
    }
    __device__ static float fold_scale(float acc, float dot, float scale) { return fmaf(dot, scale, acc); }
    __device__ static float fold_bias(float acc, float, float) { return acc; }

    // the stream tile's terms: the two 16-bit halves as stored, joined where used
    struct StreamTerms {
        const unsigned short* scale;
        const unsigned short* bias;
        __device__ explicit StreamTerms(const Args& a)
            : scale(static_cast<const unsigned short*>(a.scale.p)), bias(static_cast<const unsigned short*>(a.bias.p)) {}
        __device__ unsigned short scale_raw(long long at) const { return scale[at]; }
        __device__ unsigned short bias_raw(long long at) const { return bias[at]; }
    };
    __device__ static float stream_scale(const StreamTerms&, unsigned short s, unsigned short b) {
        return __uint_as_float(static_cast<uint32_t>(s) | static_cast<uint32_t>(b) << 16);
    }
    __device__ static float stream_bias(const StreamTerms&, unsigned short, unsigned short) { return 0.f; }
};

struct Bf16Decoder : CodedBase<16> {
    static constexpr bool kTerms = false;

    template <typename T, int I>
    __device__ static typename T::pair stream_pair(const uint32_t (&w)[16]) {
        return __builtin_bit_cast(typename T::pair, w[I]);
    }

    template <int T, class Frag>
    __device__ static Frag frag(const uint32_t (&w)[16]) {
        u32x8 f;
#pragma unroll
        for (int i = 0; i < 8; ++i) f[i] = w[8 * T + i];
        return __builtin_bit_cast(Frag, f);
    }

    template <class Terms>
    __device__ static float term_scale(const Terms&, uint32_t) {
        return 1.f;
    }
    template <class Terms>
    __device__ static float term_bias(const Terms&, uint32_t) {
        return 0.f;
    }
    // the scale is 1, or 0 for a stream lane past the row's last chunk
    __device__ static float fold_scale(float acc, float dot, float scale) { return fmaf(dot, scale, acc); }
    __device__ static float fold_bias(float acc, float, float) { return acc; }

    // no tables: nothing is loaded
    struct StreamTerms {
        __device__ explicit StreamTerms(const Args&) {}
        __device__ unsigned short scale_raw(long long) const { return 0; }
        __device__ unsigned short bias_raw(long long) const { return 0; }
        const unsigned short* scale = nullptr;
        const unsigned short* bias = nullptr;
    };
    __device__ static float stream_scale(const StreamTerms&, unsigned short, unsigned short) { return 1.f; }
    __device__ static float stream_bias(const StreamTerms&, unsigned short, unsigned short) { return 0.f; }
};

}  // namespace rocm
}  // namespace tf
