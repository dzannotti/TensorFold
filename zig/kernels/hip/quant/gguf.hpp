#pragma once

// ggml's block quantizations on the GPU, as a GGUF file stores them (ggml-org/llama.cpp ggml/src/ggml-common.h): a
// projection row [K] is K / 256 superblocks (q8_0 and iq4_nl: eight 32-element blocks make one), each 16 pieces of 16
// consecutive elements. GguQ<T>::piece gives piece p of superblock `sb` as 16 unsigned codes u (four bytes a word, k
// order) with fp32 s and mn such that ggml's dequantize_row_* value is exactly s u + mn:
//   q4_K, q5_K  s = d sc_j, mn = -dmin m_j (get_scale_min_k4), u the 4 or 5-bit code
//   q6_K        s = d sc_p, mn = -32 s, u the 6-bit code (ggml's q - 32)
//   q3_K        s = d (sc_p - 32), mn = -4 s, u = q + 4 hbit (ggml's q - (hbit ? 0 : 4))
//   q8_0        s = d, mn = -128 d, u = q + 128
//   iq4_nl/xs   s = d (iq4_xs: d (ls - 32)), mn = -127 s, u = kvalues_iq4nl[q] + 127
//   iq3_s       s = d (1 + 2 sc), mn = -16 s, u = 16 +- grid magnitude
// Every s u + mn is exact in fp32 (a zero may come out +0 where ggml's product gives -0) (s has at most 18 significant bits, u at most 8), so a sum s (x . u) + mn (x . 1)
// is the dot with ggml's weights up to the fp32 roundings of the two dots.
// Loads may sit at any byte (q6_K's 210-byte blocks, q8_0's 34): the backend's queues run with unaligned access on.

#include <cstdint>

#include "common/vec.hpp"

namespace tf {
namespace rocm {

// ggml type numbers (ggml.h enum ggml_type).
enum GgmlType : int { kQ8_0 = 8, kQ3_K = 11, kQ4_K = 12, kQ5_K = 13, kQ6_K = 14, kIQ4_NL = 20, kIQ3_S = 21, kIQ4_XS = 23 };

__device__ inline uint32_t gg_ld32(const uint8_t* p) { return *reinterpret_cast<const uint32_t*>(p); }
__device__ inline u32x4 gg_ld128(const uint8_t* p) { return *reinterpret_cast<const u32x4*>(p); }
__device__ inline float gg_half(uint32_t v) { return __half2float(__ushort_as_half(static_cast<unsigned short>(v & 0xffffu))); }

// kvalues_iq4nl + 127 as bytes, four a word: codes 0-3, 4-7, 8-11, 12-15.
__device__ inline uint32_t gg_iq4_lookup(uint32_t nib) {
    // kvalues_iq4nl = -127 -104 -83 -65 -49 -35 -22 -10 1 13 25 38 53 69 89 113, plus 127
    constexpr uint32_t t0 = 0x3e2c1700u, t1 = 0x75695c4eu, t2 = 0xa5988c80u, t3 = 0xf0d8c4b4u;
    const uint32_t sel = nib & 0x07070707u;
    const uint32_t lo = __builtin_amdgcn_perm(t1, t0, sel);
    const uint32_t hi = __builtin_amdgcn_perm(t3, t2, sel);
    const uint32_t m = ((nib >> 3) & 0x01010101u) * 0xffu;
    return (hi & m) | (lo & ~m);
}

// The 6-bit scale and min of sub-block j of a q4_K / q5_K block's 12 scale bytes (s0 = bytes 0-3, s1 = 4-7, s2 = 8-11).
__device__ inline void gg_scale_min_k4(int j, uint32_t s0, uint32_t s1, uint32_t s2, uint32_t& sc, uint32_t& m) {
    const int b = 8 * (j & 3);
    const uint32_t q0 = (s0 >> b) & 0xffu;  // q[j] (j < 4) / q[j - 4]
    const uint32_t q4 = (s1 >> b) & 0xffu;  // q[j + 4] (j < 4) / q[j]
    const uint32_t q8 = (s2 >> b) & 0xffu;  // q[j + 4] (j >= 4)
    if (j < 4) {
        sc = q0 & 63;
        m = q4 & 63;
    } else {
        sc = (q8 & 0xf) | ((q0 >> 6) << 4);
        m = (q8 >> 4) | ((q4 >> 6) << 4);
    }
}

extern __constant__ uint32_t gg_iq3s_grid[512];

template <int T>
struct GguQ;

// Each decoder: Raw, the registers a piece's loads fill (load issues them, nothing waits), and decode, the piece from
// them; piece is both.
#define TF_GG_PIECE                                                                                \
    __device__ static void piece(const uint8_t* b, int p, uint32_t (&u)[4], float& s, float& mn) { \
        Raw r;                                                                                     \
        load(b, p, r);                                                                             \
        decode(r, p, u, s, mn);                                                                    \
    }

template <>
struct GguQ<kQ4_K> {
    static constexpr int kBytes = 144;  // a superblock
    struct Raw {
        u32x4 h, q;
    };
    __device__ static void load(const uint8_t* b, int p, Raw& r) {
        r.h = gg_ld128(b);
        r.q = gg_ld128(b + 16 + 32 * (p >> 2) + 16 * (p & 1));
    }
    __device__ static void decode(const Raw& r, int p, uint32_t (&u)[4], float& s, float& mn) {
        const int j = p >> 1, hi = j & 1;
        uint32_t sc, m;
        gg_scale_min_k4(j, r.h.y, r.h.z, r.h.w, sc, m);
        s = gg_half(r.h.x) * static_cast<float>(sc);
        mn = -(gg_half(r.h.x >> 16) * static_cast<float>(m));
        const uint32_t w[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = (w[i] >> (4 * hi)) & 0x0f0f0f0fu;
    }
    TF_GG_PIECE
};

template <>
struct GguQ<kQ5_K> {
    static constexpr int kBytes = 176;
    struct Raw {
        u32x4 h, qh, q;
    };
    __device__ static void load(const uint8_t* b, int p, Raw& r) {
        r.h = gg_ld128(b);
        r.qh = gg_ld128(b + 16 + 16 * (p & 1));
        r.q = gg_ld128(b + 48 + 32 * (p >> 2) + 16 * (p & 1));
    }
    __device__ static void decode(const Raw& r, int p, uint32_t (&u)[4], float& s, float& mn) {
        const int j = p >> 1, hi = j & 1;
        uint32_t sc, m;
        gg_scale_min_k4(j, r.h.y, r.h.z, r.h.w, sc, m);
        s = gg_half(r.h.x) * static_cast<float>(sc);
        mn = -(gg_half(r.h.x >> 16) * static_cast<float>(m));
        const uint32_t w[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
        const uint32_t hb[4] = {r.qh.x, r.qh.y, r.qh.z, r.qh.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = ((w[i] >> (4 * hi)) & 0x0f0f0f0fu) | (((hb[i] >> j) & 0x01010101u) << 4);
    }
    TF_GG_PIECE
};

template <>
struct GguQ<kQ6_K> {
    static constexpr int kBytes = 210;
    struct Raw {
        u32x4 ql, qh;
        uint32_t sw, d;
    };
    __device__ static void load(const uint8_t* b, int p, Raw& r) {
        const int n = p >> 3, qt = (p >> 1) & 3, h = p & 1;
        r.ql = gg_ld128(b + 64 * n + 32 * (qt & 1) + 16 * h);
        r.qh = gg_ld128(b + 128 + 32 * n + 16 * h);
        r.sw = gg_ld32(b + 192 + (p & ~3));  // scales[p & ~3] ..
        r.d = gg_ld32(b + 206);              // d in the high half
    }
    __device__ static void decode(const Raw& r, int p, uint32_t (&u)[4], float& s, float& mn) {
        const int qt = (p >> 1) & 3;
        const int sc = static_cast<int8_t>((r.sw >> (8 * (p & 3))) & 0xffu);
        s = gg_half(r.d >> 16) * static_cast<float>(sc);
        mn = -32.0f * s;
        const uint32_t l[4] = {r.ql.x, r.ql.y, r.ql.z, r.ql.w};
        const uint32_t hh[4] = {r.qh.x, r.qh.y, r.qh.z, r.qh.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = ((l[i] >> (4 * (qt >> 1))) & 0x0f0f0f0fu) | (((hh[i] >> (2 * qt)) & 0x03030303u) << 4);
    }
    TF_GG_PIECE
};

template <>
struct GguQ<kQ3_K> {
    static constexpr int kBytes = 110;
    struct Raw {
        u32x4 hm, qs;
        uint32_t a0, a1, tmp, d;
    };
    __device__ static void load(const uint8_t* b, int p, Raw& r) {
        const int n = p >> 3, h = p & 1;
        r.hm = gg_ld128(b + 16 * h);
        r.qs = gg_ld128(b + 32 + 32 * n + 16 * h);
        r.a0 = gg_ld32(b + 96);
        r.a1 = gg_ld32(b + 100);
        r.tmp = gg_ld32(b + 104);
        r.d = gg_ld32(b + 106);  // d in the high half
    }
    __device__ static void decode(const Raw& r, int p, uint32_t (&u)[4], float& s, float& mn) {
        const int n = p >> 3, j = (p >> 1) & 3;
        // dequantize_row_q3_K's aux[p / 4], byte p % 4
        const int i4 = p >> 2;
        const uint32_t lo = (i4 & 1) ? r.a1 : r.a0;
        const uint32_t aux = ((i4 < 2 ? lo : lo >> 4) & 0x0f0f0f0fu) | (((r.tmp >> (2 * i4)) & 0x03030303u) << 4);
        const int sc = static_cast<int8_t>((aux >> (8 * (p & 3))) & 0xffu);
        s = gg_half(r.d >> 16) * static_cast<float>(sc - 32);
        mn = -4.0f * s;
        const uint32_t q[4] = {r.qs.x, r.qs.y, r.qs.z, r.qs.w};
        const uint32_t m[4] = {r.hm.x, r.hm.y, r.hm.z, r.hm.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = ((q[i] >> (2 * j)) & 0x03030303u) | (((m[i] >> (4 * n + j)) & 0x01010101u) << 2);
    }
    TF_GG_PIECE
};

template <>
struct GguQ<kQ8_0> {
    static constexpr int kBytes = 8 * 34;
    struct Raw {
        u32x4 q;
        uint32_t d;
    };
    __device__ static void load(const uint8_t* b, int p, Raw& r) {
        const uint8_t* blk = b + 34 * (p >> 1);
        r.q = gg_ld128(blk + 2 + 16 * (p & 1));
        r.d = *reinterpret_cast<const unsigned short*>(blk);
    }
    __device__ static void decode(const Raw& r, int, uint32_t (&u)[4], float& s, float& mn) {
        s = gg_half(r.d);
        mn = -128.0f * s;
        u[0] = r.q.x ^ 0x80808080u;
        u[1] = r.q.y ^ 0x80808080u;
        u[2] = r.q.z ^ 0x80808080u;
        u[3] = r.q.w ^ 0x80808080u;
    }
    TF_GG_PIECE
};

template <>
struct GguQ<kIQ4_NL> {
    static constexpr int kBytes = 8 * 18;
    struct Raw {
        u32x4 q;
        uint32_t d;
    };
    __device__ static void load(const uint8_t* b, int p, Raw& r) {
        const uint8_t* blk = b + 18 * (p >> 1);
        r.q = gg_ld128(blk + 2);
        r.d = *reinterpret_cast<const unsigned short*>(blk);
    }
    __device__ static void decode(const Raw& r, int p, uint32_t (&u)[4], float& s, float& mn) {
        s = gg_half(r.d);
        mn = -127.0f * s;
        const uint32_t w[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = gg_iq4_lookup((w[i] >> (4 * (p & 1))) & 0x0f0f0f0fu);
    }
    TF_GG_PIECE
};

template <>
struct GguQ<kIQ4_XS> {
    static constexpr int kBytes = 136;
    struct Raw {
        u32x4 q;
        uint32_t h0, sl;
    };
    __device__ static void load(const uint8_t* b, int p, Raw& r) {
        r.h0 = gg_ld32(b);
        r.sl = gg_ld32(b + 4);
        r.q = gg_ld128(b + 8 + 16 * (p >> 1));
    }
    __device__ static void decode(const Raw& r, int p, uint32_t (&u)[4], float& s, float& mn) {
        const int ib = p >> 1;
        const int ls = static_cast<int>(((r.sl >> (8 * (ib >> 1) + 4 * (ib & 1))) & 0xfu) | (((r.h0 >> (16 + 2 * ib)) & 3u) << 4));
        s = gg_half(r.h0) * static_cast<float>(ls - 32);
        mn = -127.0f * s;
        const uint32_t w[4] = {r.q.x, r.q.y, r.q.z, r.q.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) u[i] = gg_iq4_lookup((w[i] >> (4 * (p & 1))) & 0x0f0f0f0fu);
    }
    TF_GG_PIECE
};

template <>
struct GguQ<kIQ3_S> {
    static constexpr int kBytes = 110;
    struct Raw {
        uint32_t d, qs, qh, sg, sc;
    };
    __device__ static void load(const uint8_t* b, int p, Raw& r) {
        const int ib = p >> 1, h = p & 1;
        r.d = *reinterpret_cast<const unsigned short*>(b);
        r.qs = gg_ld32(b + 2 + 8 * ib + 4 * h);  // qs[8 ib + 2 l + g], l = 2 h, 2 h + 1
        r.qh = b[66 + ib];
        r.sg = *reinterpret_cast<const unsigned short*>(b + 74 + 4 * ib + 2 * h);  // signs[4 ib + l]
        r.sc = b[106 + (ib >> 1)];
    }
    __device__ static void decode(const Raw& r, int p, uint32_t (&u)[4], float& s, float& mn) {
        const int ib = p >> 1, h = p & 1;
        s = gg_half(r.d) * static_cast<float>(1 + 2 * ((r.sc >> (4 * (ib & 1))) & 0xfu));
        mn = -16.0f * s;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int l = 2 * h + (i >> 1), g = i & 1;
            const uint32_t idx = ((r.qs >> (8 * i)) & 0xffu) | ((r.qh << (8 - 2 * l - g)) & 256u);
            const uint32_t mag = gg_iq3s_grid[idx];
            const uint32_t bits = (r.sg >> (8 * (i >> 1) + 4 * g)) & 0xfu;
            const uint32_t neg = ((bits | (bits << 7) | (bits << 14) | (bits << 21)) & 0x01010101u) * 0xffu;
            u[i] = ((0x10101010u - mag) & neg) | ((0x10101010u + mag) & ~neg);
        }
    }
    TF_GG_PIECE
};

#undef TF_GG_PIECE

// Four code words as a WMMA operand's eight BF16 pairs (codes 2 i, 2 i + 1), exactly: each byte to float, top halves.
__device__ inline void gg_pairs(const uint32_t (&u)[4], uint32_t (&f)[8]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        uint32_t w = u[i];
        asm("" : "+v"(w));  // whole bytes, so each converts with v_cvt_f32_ubyteN
        const float c0 = static_cast<float>(w & 0xffu);
        const float c1 = static_cast<float>((w >> 8) & 0xffu);
        const float c2 = static_cast<float>((w >> 16) & 0xffu);
        const float c3 = static_cast<float>(w >> 24);
        f[2 * i] = __builtin_amdgcn_perm(__float_as_uint(c1), __float_as_uint(c0), 0x07060302u);
        f[2 * i + 1] = __builtin_amdgcn_perm(__float_as_uint(c3), __float_as_uint(c2), 0x07060302u);
    }
}

}  // namespace rocm
}  // namespace tf
