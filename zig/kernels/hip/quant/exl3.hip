// EXL3 trellis projections (docs/recipes/exl3.md): the input rotation, a WMMA product whose rows never see each other, and
// the output rotation. A row's bits depend on its own x and on (K, N) alone, so any row count or chunking gives them.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "common/arch.hpp"

namespace tf {
namespace rocm {

typedef _Float16 h16x2 __attribute__((ext_vector_type(2)));
typedef _Float16 h16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8e __attribute__((ext_vector_type(8)));
typedef unsigned u32x8e __attribute__((ext_vector_type(8)));
typedef unsigned u32x4e __attribute__((ext_vector_type(4)));

// One product: xh [m, k] rotated fp16, the trellis tiles (tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb +
// (nt % 8) * 4 K2 dwords), out [m, n] of `out_kind` (0 fp32, 1 bf16, 2 fp16), or fp32 partial sums [splits, m, n].
struct Exl3 {
    const __half* xh;
    const uint32_t* trellis;
    long long stride_k;
    long long stride_nb;
    const __half* svh;
    const __half* bias;
    void* out;
    float* partial;
    int m, k, n, splits, out_kind;
};

// E(p): one past the last stream bit of value p's 16-bit window, K2 = twice the bits.
template <int K2>
__host__ __device__ constexpr int exl3_end(int p) {
    return (K2 & 1) ? ((p + 1) * K2 - ((p + 1) & 1)) / 2 : (p + 1) * (K2 / 2);
}

// The first stream value of window w of a lane holding column c's rows of half h: four values, rows 2a.. and 8 + 2a..
__host__ __device__ constexpr int exl3_first(int c, int h, int w) { return 32 * (c & 7) + 4 * (c >> 3) + 8 * (2 * h + w); }

// Words a window spans at most, over every lane: two up to 5 bits, three above.
template <int K2>
__host__ __device__ constexpr int exl3_window_words() {
    int most = 0;
    for (int c = 0; c < 16; ++c)
        for (int h = 0; h < 2; ++h)
            for (int w = 0; w < 2; ++w) {
                const int q = exl3_first(c, h, w);
                const int start = (exl3_end<K2>(q) - 16 + 128 * K2) % (128 * K2);
                const int need = start % 32 + exl3_end<K2>(q + 3) - exl3_end<K2>(q) + 16;
                most = need > most ? need : most;
            }
    return (most + 31) / 32;
}

// s0 * C and s1 * C mod 2^32 for two 16-bit states packed in `p` (s0 low): the high halves from one packed 16-bit
// multiply, the low ones from two 16-bit multiply-adds; v_mul_lo_u32 is quarter rate (asm: the compiler refolds it).
template <uint32_t C>
__device__ __forceinline__ void exl3_mul2(uint32_t p, uint32_t& x0, uint32_t& x1) {
    uint32_t hi;
    asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(hi) : "v"(p), "s"((C >> 16) * 0x10001u));
    const uint32_t c0 = hi << 16, c1 = hi & 0xffff0000u;
    asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(x0) : "v"(p), "s"(C & 0xffffu), "v"(c0));
    asm("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(x1) : "v"(p), "s"(C & 0xffffu), "v"(c1));
}

// Two codebook values from two 16-bit states packed in `p` (the first low), as a half2 (one rounding each).
template <int CB>
__device__ __forceinline__ uint32_t exl3_decode2(uint32_t p) {
    if constexpr (CB == 2) {
        uint32_t x0, x1;
        exl3_mul2<0x83DCD12Du>(p, x0, x1);
        // fp16 bits of 1024 + the byte sum (below 1024, so the integer lands in the mantissa exactly)
        const uint32_t h0 = __builtin_amdgcn_udot4(x0, 0x01010101u, 0x6400u, false);
        const uint32_t h1 = __builtin_amdgcn_udot4(x1, 0x01010101u, 0x6400u, false);
        const h16x2 h = __builtin_bit_cast(h16x2, __builtin_amdgcn_perm(h1, h0, 0x05040100u));
        const h16x2 scale = {__builtin_bit_cast(_Float16, static_cast<unsigned short>(0x1EEE)), __builtin_bit_cast(_Float16, static_cast<unsigned short>(0x1EEE))};
        const h16x2 bias = {__builtin_bit_cast(_Float16, static_cast<unsigned short>(0xC931)), __builtin_bit_cast(_Float16, static_cast<unsigned short>(0xC931))};
        return __builtin_bit_cast(uint32_t, __builtin_elementwise_fma(h, scale, bias));
    } else {
        uint32_t x0, x1;
        if constexpr (CB == 1) {
            exl3_mul2<0xCBAC1FEDu>(p, x0, x1);
        } else {
            exl3_mul2<89226354u>(p, x0, x1);
            x0 += 64248484u;
            x1 += 64248484u;
        }
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        const h16x2 lo = __builtin_bit_cast(h16x2, __builtin_amdgcn_perm(x1, x0, 0x05040100u));
        const h16x2 hi = __builtin_bit_cast(h16x2, __builtin_amdgcn_perm(x1, x0, 0x07060302u));
        return __builtin_bit_cast(uint32_t, lo + hi);
    }
}

// Unscaled Walsh-Hadamard transform of 128 fp32 values, values 4 l .. 4 l + 3 in lane l, in a fixed butterfly order.
__device__ __forceinline__ void exl3_fwht(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
#pragma unroll
    for (int m = 1; m < 32; m <<= 1) {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float o = __shfl_xor(v[j], m, 32);
            v[j] = (lane & m) ? o - v[j] : v[j] + o;
        }
    }
}

constexpr float kExl3Had = 0.08838834764831845f;  // 1 / sqrt(128)

__device__ __forceinline__ float exl3_half(const __half* p, int i) {
    return static_cast<float>(__builtin_bit_cast(_Float16, reinterpret_cast<const unsigned short*>(p)[i]));
}

// A row's 128 finished outputs from its sums (4 a lane): rotated, scaled by svh, biased, stored in the output kind.
__device__ __forceinline__ void exl3_store(const Exl3& a, float (&v)[4], int lane, int row, int col0) {
    exl3_fwht(v, lane);
    const int col = col0 + 4 * lane;
    const uint2 sv = *reinterpret_cast<const uint2*>(a.svh + col);
    const h16x2 s01 = __builtin_bit_cast(h16x2, sv.x), s23 = __builtin_bit_cast(h16x2, sv.y);
    const float s[4] = {static_cast<float>(s01[0]), static_cast<float>(s01[1]), static_cast<float>(s23[0]), static_cast<float>(s23[1])};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        v[j] = v[j] * kExl3Had * s[j];
        if (a.bias != nullptr) v[j] += exl3_half(a.bias, col + j);
    }
    const long long at = static_cast<long long>(row) * a.n + col;
    if (a.out_kind == 0) {
        *reinterpret_cast<float4*>(static_cast<float*>(a.out) + at) = make_float4(v[0], v[1], v[2], v[3]);
    } else if (a.out_kind == 1) {
        uint32_t u[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const uint16_t lo = __builtin_bit_cast(uint16_t, static_cast<__bf16>(v[2 * j]));
            const uint16_t hi = __builtin_bit_cast(uint16_t, static_cast<__bf16>(v[2 * j + 1]));
            u[j] = lo | (static_cast<uint32_t>(hi) << 16);
        }
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(a.out) + at) = make_uint2(u[0], u[1]);
    } else {
        const h16x2 lo = {static_cast<_Float16>(v[0]), static_cast<_Float16>(v[1])};
        const h16x2 hi = {static_cast<_Float16>(v[2]), static_cast<_Float16>(v[3])};
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(a.out) + at) = make_uint2(__builtin_bit_cast(uint32_t, lo), __builtin_bit_cast(uint32_t, hi));
    }
}

// x [m, k] (kind 1 bf16, 2 fp16) times suh, rotated by 128-blocks, scaled, rounded to fp16: one wave a block of a row.
extern "C" __global__ void __launch_bounds__(32) tf_exl3_rotate_in(const void* x, int kind, const __half* suh, __half* xh, int k) {
    const int lane = threadIdx.x;
    const int col = blockIdx.x * 128 + 4 * lane;
    const long long at = static_cast<long long>(blockIdx.y) * k + col;
    const uint2 raw = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(x) + at);
    const uint2 su = *reinterpret_cast<const uint2*>(suh + col);
    float v[4];
    const uint32_t w[2] = {raw.x, raw.y};
    const uint32_t s[2] = {su.x, su.y};
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const h16x2 sh = __builtin_bit_cast(h16x2, s[j]);
        if (kind == 1) {
            v[2 * j] = __uint_as_float(w[j] << 16);
            v[2 * j + 1] = __uint_as_float(w[j] & 0xffff0000u);
        } else {
            const h16x2 xv = __builtin_bit_cast(h16x2, w[j]);
            v[2 * j] = static_cast<float>(xv[0]);
            v[2 * j + 1] = static_cast<float>(xv[1]);
        }
        v[2 * j] *= static_cast<float>(sh[0]);
        v[2 * j + 1] *= static_cast<float>(sh[1]);
    }
    exl3_fwht(v, lane);
    const h16x2 lo = {static_cast<_Float16>(v[0] * kExl3Had), static_cast<_Float16>(v[1] * kExl3Had)};
    const h16x2 hi = {static_cast<_Float16>(v[2] * kExl3Had), static_cast<_Float16>(v[3] * kExl3Had)};
    *reinterpret_cast<uint2*>(xh + at) = make_uint2(__builtin_bit_cast(uint32_t, lo), __builtin_bit_cast(uint32_t, hi));
}

// The split sums of a row's 128 columns added in split order and finished: one wave a block of a row.
extern "C" __global__ void __launch_bounds__(32) tf_exl3_finish(Exl3 a) {
    const int lane = threadIdx.x, row = blockIdx.y, col0 = blockIdx.x * 128;
    const long long at = static_cast<long long>(row) * a.n + col0 + 4 * lane;
    float4 s = *reinterpret_cast<const float4*>(a.partial + at);
    for (int q = 1; q < a.splits; ++q) {
        const float4 u = *reinterpret_cast<const float4*>(a.partial + static_cast<long long>(q) * a.m * a.n + at);
        s.x += u.x;
        s.y += u.y;
        s.z += u.z;
        s.w += u.w;
    }
    float v[4] = {s.x, s.y, s.z, s.w};
    exl3_store(a, v, lane, row, col0);
}

// k steps a slab stages (a prompt's 8 row tiles); a split's tiles are a multiple of it and of a round's 4 in flight.
constexpr int kExl3Slab = 4;

// A tile's WMMA B fragment for this lane from its two windows' words (window w at bit `off[w]` of r[w][0..]): column
// c = lane % 16, rows 8 h .. decoded here (h = lane / 16), the other half's through the LDS crossbar.
template <int K2, int CB, int LW>
__device__ __forceinline__ h16x16 exl3_fragment(const uint32_t (&r)[2][LW], const int (&off)[2], int c) {
    uint32_t own[2][2];  // window w: rows 2a, 2a + 1 and 8 + 2a, 9 + 2a for a = 2 h + w
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        const uint32_t hi = static_cast<uint32_t>((((static_cast<uint64_t>(r[w][0]) << 32) | r[w][1]) << off[w]) >> 32);
        uint32_t lo;
        if constexpr (LW > 2) {
            lo = static_cast<uint32_t>((((static_cast<uint64_t>(r[w][1]) << 32) | r[w][LW > 2 ? 2 : 1]) << off[w]) >> 32);
        } else {
            lo = r[w][1] << off[w];
        }
        const uint64_t st = (static_cast<uint64_t>(hi) << 32) | lo;
        // state b in the low 16 bits of v[b] (one funnel shift each), packed in pairs
        uint32_t v[4];
#pragma unroll
        for (int b = 0; b < 4; ++b) v[b] = static_cast<uint32_t>(st >> (48 - (exl3_end<K2>(b) - exl3_end<K2>(0))));
        own[w][0] = exl3_decode2<CB>(__builtin_amdgcn_perm(v[1], v[0], 0x05040100u));
        own[w][1] = exl3_decode2<CB>(__builtin_amdgcn_perm(v[3], v[2], 0x05040100u));
    }
    // dword d holds rows 2 d, 2 d + 1: half 0 decoded dwords 0, 1, 4, 5 and half 1 dwords 2, 3, 6, 7
    u32x8e f;
#pragma unroll
    for (int j = 0; j < 2; ++j)
#pragma unroll
        for (int w = 0; w < 2; ++w) {
            f[4 * j + w] = __builtin_amdgcn_ds_bpermute(4 * c, own[w][j]);
            f[4 * j + 2 + w] = __builtin_amdgcn_ds_bpermute(4 * (c + 16), own[w][j]);
        }
    return __builtin_bit_cast(h16x16, f);
}

// Columns 128 blockIdx.x.. over split blockIdx.y of K for rows 16 MT blockIdx.z..: wave w owns tile column 8 bx + w, a
// lane column c = lane % 16 of it, and rows 8 h .. of its WMMA B fragment (h = lane / 16) from two windows of the tile's
// stream; the halves swap theirs. Every row's sum is the same WMMA chain over its K range in K order.
// A round's row tiles (MT 1, 2) keep a ring of each lane's window words and x fragments in registers; a prompt's (MT 8)
// stages a slab of k steps (the block's 8 tiles each and its rows' x) through LDS, one slab ahead, a staged tile
// repeating its first 4 dwords after its end so a window never wraps. Both run the same WMMA chain.
template <int K2, int CB, int MT>
__global__ void __launch_bounds__(256) exl3_product(Exl3 a) {
#if TF_DEVICE_WMMA_GFX11
    constexpr int TD = 4 * K2;      // dwords a tile
    constexpr int RING = 128 * K2;  // bits a tile
    constexpr int LW = exl3_window_words<K2>();
    constexpr int R = 16 * MT;      // rows of the block
    __shared__ __attribute__((aligned(16))) float red[16][132];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int c = lane & 15, h = lane >> 4;
    const int tiles = (a.k >> 4) / a.splits;
    const int kt0 = blockIdx.y * tiles;
    const int m0 = blockIdx.z * R;
    // the lane's windows: first dword in its tile and bit offset, the same for every tile
    int start[2], off[2];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        start[w] = ((exl3_end<K2>(exl3_first(c, h, w)) - 16 + RING) % RING) >> 5;
        off[w] = ((exl3_end<K2>(exl3_first(c, h, w)) - 16 + RING) % RING) & 31;
    }
    f32x8e acc[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) acc[mt] = f32x8e{0, 0, 0, 0, 0, 0, 0, 0};

    if constexpr (MT <= 2) {
        constexpr int D = 4;  // tiles in flight a wave
        // uniform bases with per-lane byte offsets: a load takes a scalar address and a 32-bit lane offset
        const char* tp = reinterpret_cast<const char*>(a.trellis + static_cast<long long>(blockIdx.x) * a.stride_nb + wave * TD + static_cast<long long>(kt0) * a.stride_k);
        const long long tstep = a.stride_k * 4;
        const char* xp = reinterpret_cast<const char*>(a.xh + kt0 * 16);
        uint32_t idx[2][LW];
#pragma unroll
        for (int w = 0; w < 2; ++w)
#pragma unroll
            for (int i = 0; i < LW; ++i) idx[w][i] = 4u * ((start[w] + i) % TD);
        // rows past the call read its last row: their outputs are never stored
        uint32_t xo[MT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int r = m0 + mt * 16 + c;
            xo[mt] = static_cast<uint32_t>(r < a.m ? r : a.m - 1) * static_cast<uint32_t>(a.k) * 2u;
        }
        uint32_t ring[D][2][LW];
        u32x8e xa[D][MT];
        // every load is unconditional on a clamped tile, so the wait counters never drain the ring
        auto fetch = [&](int i, uint32_t (&r)[2][LW], u32x8e (&x)[MT]) {
            const int t = i < tiles ? i : tiles - 1;
            const char* tile = tp + t * tstep;
#pragma unroll
            for (int w = 0; w < 2; ++w)
#pragma unroll
                for (int j = 0; j < LW; ++j) r[w][j] = *reinterpret_cast<const uint32_t*>(tile + idx[w][j]);
            const char* xt = xp + t * 32;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) x[mt] = *reinterpret_cast<const u32x8e*>(xt + xo[mt]);
        };
#pragma unroll
        for (int i = 0; i < D - 1; ++i) fetch(i, ring[i], xa[i]);
        // the launch takes `tiles` a multiple of D
        for (int i0 = 0; i0 < tiles; i0 += D) {
#pragma unroll
            for (int rb = 0; rb < D; ++rb) {
                fetch(i0 + rb + D - 1, ring[(rb + D - 1) % D], xa[(rb + D - 1) % D]);
                const h16x16 b = exl3_fragment<K2, CB, LW>(ring[rb], off, c);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16x16, xa[rb][mt]), b, acc[mt]);
            }
        }
    } else {
        constexpr int KS = kExl3Slab;
        constexpr int TP = TD + 4;                          // dwords a staged tile
        constexpr int XP = KS * 32 + 16;                    // bytes a staged x row
        constexpr int TPIECES = KS * 8 * K2;                // 16-byte pieces of a slab's tiles
        constexpr int XPIECES = R * KS * 2;                 // and of its x
        constexpr int PT = (TPIECES + 255) / 256;
        constexpr int PX = (XPIECES + 255) / 256;
        constexpr int TBYTES = KS * 8 * TP * 4;
        __shared__ __attribute__((aligned(16))) char lds[2][TBYTES + R * XP];
        const int slabs = tiles / KS;
        const char* tbase = reinterpret_cast<const char*>(a.trellis + static_cast<long long>(blockIdx.x) * a.stride_nb + static_cast<long long>(kt0) * a.stride_k);
        const long long tslab = a.stride_k * 4 * KS;
        const char* xbase = reinterpret_cast<const char*>(a.xh + kt0 * 16);
        // a thread's staging pieces: where each comes from (past the slab base) and goes, the same every slab; pieces
        // past the slab repeat the last one, so every load is unconditional
        uint32_t tg[PT], tl[PT], xg[PX], xl[PX];
        bool dup[PT];
#pragma unroll
        for (int j = 0; j < PT; ++j) {
            const int p = tid + 256 * j < TPIECES ? tid + 256 * j : TPIECES - 1;
            const int step = p / (8 * K2), tile = (p / K2) % 8, q = p % K2;
            tg[j] = static_cast<uint32_t>(step * a.stride_k * 4 + (tile * TD + 4 * q) * 4);
            tl[j] = static_cast<uint32_t>(((step * 8 + tile) * TP + 4 * q) * 4);
            dup[j] = q == 0;
        }
#pragma unroll
        for (int j = 0; j < PX; ++j) {
            const int p = tid + 256 * j < XPIECES ? tid + 256 * j : XPIECES - 1;
            const int r = p / (KS * 2), q = p % (KS * 2);
            // rows past the call read its last row: their outputs are never stored
            const int row = m0 + r < a.m ? m0 + r : a.m - 1;
            xg[j] = static_cast<uint32_t>(row) * static_cast<uint32_t>(a.k) * 2u + 16u * q;
            xl[j] = static_cast<uint32_t>(TBYTES + r * XP + 16 * q);
        }
        u32x4e treg[PT], xreg[PX];
        auto load = [&](int s0) {
            const int sl = s0 < slabs ? s0 : slabs - 1;
            const char* tb = tbase + sl * tslab;
            const char* xb = xbase + sl * KS * 32;
#pragma unroll
            for (int j = 0; j < PT; ++j) treg[j] = *reinterpret_cast<const u32x4e*>(tb + tg[j]);
#pragma unroll
            for (int j = 0; j < PX; ++j) xreg[j] = *reinterpret_cast<const u32x4e*>(xb + xg[j]);
        };
        auto store = [&](int buf) {
            char* l = lds[buf];
#pragma unroll
            for (int j = 0; j < PT; ++j) {
                *reinterpret_cast<u32x4e*>(l + tl[j]) = treg[j];
                if (dup[j]) *reinterpret_cast<u32x4e*>(l + tl[j] + TD * 4) = treg[j];
            }
#pragma unroll
            for (int j = 0; j < PX; ++j) *reinterpret_cast<u32x4e*>(l + xl[j]) = xreg[j];
        };
        uint32_t wat[2];
#pragma unroll
        for (int w = 0; w < 2; ++w) wat[w] = static_cast<uint32_t>((wave * TP + start[w]) * 4);
        const uint32_t xat = static_cast<uint32_t>(TBYTES + c * XP);
        load(0);
        store(0);
        load(1);
        __syncthreads();
        for (int sl = 0; sl < slabs; ++sl) {
            const char* l = lds[sl & 1];
#pragma unroll
            for (int i = 0; i < KS; ++i) {
                uint32_t r[2][LW];
#pragma unroll
                for (int w = 0; w < 2; ++w)
#pragma unroll
                    for (int j = 0; j < LW; ++j) r[w][j] = reinterpret_cast<const uint32_t*>(l + wat[w] + i * 8 * TP * 4)[j];
                const h16x16 b = exl3_fragment<K2, CB, LW>(r, off, c);
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    const u32x8e x = *reinterpret_cast<const u32x8e*>(l + xat + mt * 16 * XP + i * 32);
                    acc[mt] = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16x16, x), b, acc[mt]);
                }
            }
            store((sl + 1) & 1);
            load(sl + 2);
            __syncthreads();
        }
    }

    const int col0 = blockIdx.x * 128;
    if (a.splits > 1) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = m0 + mt * 16 + 2 * e + h;
                if (row < a.m) a.partial[(static_cast<long long>(blockIdx.y) * a.m + row) * a.n + col0 + wave * 16 + c] = acc[mt][e];
            }
        return;
    }
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        __syncthreads();
#pragma unroll
        for (int e = 0; e < 8; ++e) red[2 * e + h][wave * 16 + c] = acc[mt][e];
        __syncthreads();
#pragma unroll
        for (int q = 0; q < 2; ++q) {
            const int r = wave + 8 * q;
            const int row = m0 + mt * 16 + r;
            if (row >= a.m) continue;
            const float4 u = *reinterpret_cast<const float4*>(&red[r][4 * lane]);
            float v[4] = {u.x, u.y, u.z, u.w};
            exl3_store(a, v, lane, row, col0);
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

#if TF_MATRIX
#define TF_EXL3_MT(K2, CB)                                                    \
    template __global__ void exl3_product<K2, CB, 1>(Exl3);                   \
    template __global__ void exl3_product<K2, CB, 2>(Exl3);                   \
    template __global__ void exl3_product<K2, CB, 8>(Exl3);
#define TF_EXL3_CB(K2) TF_EXL3_MT(K2, 0) TF_EXL3_MT(K2, 1) TF_EXL3_MT(K2, 2)
TF_EXL3_CB(4)
TF_EXL3_CB(6)
TF_EXL3_CB(8)
TF_EXL3_CB(10)
TF_EXL3_CB(12)
TF_EXL3_CB(14)
TF_EXL3_CB(16)
TF_EXL3_MT(3, 2)
TF_EXL3_MT(5, 2)
TF_EXL3_MT(7, 2)
#undef TF_EXL3_CB
#undef TF_EXL3_MT
#endif

}  // namespace rocm
}  // namespace tf
