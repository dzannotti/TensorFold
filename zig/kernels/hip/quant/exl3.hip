// EXL3 trellis projections (docs/recipes/exl3.md): the input rotation, a WMMA product whose rows never see each other, and
// the output rotation. A row's bits depend on its own x and on (K, N) alone, so any row count or chunking gives them.

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>
#include <cstdint>

#include "common/arch.hpp"
#include "tiles/matrix_prompt.hpp"

namespace tf {
namespace rocm {

typedef _Float16 h16x2 __attribute__((ext_vector_type(2)));
typedef _Float16 h16x16 __attribute__((ext_vector_type(16)));
typedef float f32x8e __attribute__((ext_vector_type(8)));
typedef unsigned u32x8e __attribute__((ext_vector_type(8)));
typedef unsigned u32x4e __attribute__((ext_vector_type(4)));

// One product: xh rotated fp16 ([k / 16, m, 16] for a round's row tiles, MT <= 2; [m, k] for a prompt's), the trellis tiles (tile (kt, nt) at kt * stride_k + (nt / 8) * stride_nb +
// (nt % 8) * 4 K2 dwords), out [m, n] of `out_kind` (0 fp32, 1 bf16, 2 fp16; 3, the prompt tile's: bf16 out += the
// bf16 outputs), or fp32 partial sums [splits, m, n].
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

// s * C + D mod 2^32 for the 16-bit state s in the low (HI false) or high half of `src`, the rest of it ignored: the
// high product half from one packed 16-bit multiply(-add) whose low lane gives D's low half, the rest from one 16-bit
// multiply-add; v_mul_lo_u32 is quarter rate (asm: the compiler refolds it).
template <uint32_t C, uint32_t D, bool HI>
__device__ __forceinline__ uint32_t exl3_hash(uint32_t src) {
    uint32_t r, x;
    if constexpr (D == 0) {
        if constexpr (HI) asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(r) : "v"(src), "s"(C & 0xffff0000u));
        else asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(r) : "v"(src), "s"(C & 0xffff0000u));
    } else {
        if constexpr (HI) asm("v_pk_mad_u16 %0, %1, %2, %3" : "=v"(r) : "v"(src), "s"(C & 0xffff0000u), "s"(D));
        else asm("v_pk_mad_u16 %0, %1, %2, %3 op_sel_hi:[0,1,1]" : "=v"(r) : "v"(src), "s"(C & 0xffff0000u), "s"(D));
    }
    if constexpr (HI) asm("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(x) : "v"(src), "s"(C & 0xffffu), "v"(r));
    else asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(x) : "v"(src), "s"(C & 0xffffu), "v"(r));
    return x;
}

// Two codebook values from two 16-bit states (each in the HI0 / HI1 half of its register), as a half2, first low (one
// rounding each).
template <int CB, bool HI0, bool HI1>
__device__ __forceinline__ uint32_t exl3_decode2(uint32_t a, uint32_t b) {
    if constexpr (CB == 2) {
        const uint32_t x0 = exl3_hash<0x83DCD12Du, 0, HI0>(a), x1 = exl3_hash<0x83DCD12Du, 0, HI1>(b);
        // fp16 bits of 1024 + each byte sum (below 1024, so the integer lands in the mantissa exactly), packed
        const uint32_t s = __builtin_amdgcn_sad_hi_u8(x1, 0u, __builtin_amdgcn_sad_u8(x0, 0u, 0x64006400u));
        const h16x2 h = __builtin_bit_cast(h16x2, s);
        const h16x2 scale = {__builtin_bit_cast(_Float16, static_cast<unsigned short>(0x1EEE)), __builtin_bit_cast(_Float16, static_cast<unsigned short>(0x1EEE))};
        const h16x2 bias = {__builtin_bit_cast(_Float16, static_cast<unsigned short>(0xC931)), __builtin_bit_cast(_Float16, static_cast<unsigned short>(0xC931))};
        return __builtin_bit_cast(uint32_t, __builtin_elementwise_fma(h, scale, bias));
    } else {
        constexpr uint32_t C = CB == 1 ? 0xCBAC1FEDu : 89226354u;
        constexpr uint32_t D = CB == 1 ? 0u : 64248484u;
        uint32_t x0 = exl3_hash<C, D, HI0>(a), x1 = exl3_hash<C, D, HI1>(b);
        x0 = (x0 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        x1 = (x1 & 0x8FFF8FFFu) ^ 0x3B603B60u;
        const h16x2 lo = __builtin_bit_cast(h16x2, __builtin_amdgcn_perm(x1, x0, 0x05040100u));
        const h16x2 hi = __builtin_bit_cast(h16x2, __builtin_amdgcn_perm(x1, x0, 0x07060302u));
        return __builtin_bit_cast(uint32_t, lo + hi);
    }
}

// The register holding the state SH bits up in hi:lo (exl3_window_pair takes the high half at 48 and 16): halves and
// whole words need no shift, the rest one.
template <int SH>
__device__ __forceinline__ uint32_t exl3_state(uint32_t hi, uint32_t lo) {
    if constexpr (SH == 48 || SH == 32) return hi;
    else if constexpr (SH > 32) return hi >> (SH - 32);
    else if constexpr (SH == 16 || SH == 0) return lo;
    else return __builtin_amdgcn_alignbit(hi, lo, SH);
}

// Values 2 j, 2 j + 1 of a window's four, from its 64 stream bits hi:lo (the first state's last bit at bit 48).
template <int K2, int CB, int J>
__device__ __forceinline__ uint32_t exl3_window_pair(uint32_t hi, uint32_t lo) {
    constexpr int S0 = 48 - (exl3_end<K2>(2 * J) - exl3_end<K2>(0));
    constexpr int S1 = 48 - (exl3_end<K2>(2 * J + 1) - exl3_end<K2>(0));
    return exl3_decode2<CB, S0 == 48 || S0 == 16, S1 == 48 || S1 == 16>(exl3_state<S0>(hi, lo), exl3_state<S1>(hi, lo));
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
// xh is [m, k], or with `steps` (the call's m) [k / 16, m, 16]: a round's k step of every row in one run.
// A float rounded to bf16 (round to nearest even, NaN to 0x7fc0: ops/common.hpp's float_to_bf16), as its bits.
__device__ __forceinline__ uint32_t exl3_bf16(float f) {
    if (f != f) return 0x7fc0u;
    const uint32_t u = __float_as_uint(f);
    return (u + ((u >> 16) & 1u) + 0x7fffu) >> 16;
}

__device__ __forceinline__ float exl3_widen(uint32_t h, int kind) {
    return kind == 1 ? __uint_as_float(h << 16) : static_cast<float>(__builtin_bit_cast(_Float16, static_cast<unsigned short>(h)));
}

__device__ __forceinline__ uint32_t exl3_round(float v, int kind) {
    return kind == 1 ? exl3_bf16(v) : static_cast<uint32_t>(__builtin_bit_cast(unsigned short, __float2half_rn(v)));
}

// tf_silu_mul's value in x's kind (1 bf16, 2 fp16): silu(gate) rounded to the kind, times up, rounded again; as bits.
__device__ __forceinline__ uint32_t exl3_silu_mul(uint32_t g, uint32_t u, int kind) {
    const float x = exl3_widen(g, kind);
    const float s = exl3_widen(exl3_round(x / (1.0f + expf(-x)), kind), kind);
    return exl3_round(s * exl3_widen(u, kind), kind);
}

template <bool SILU = false>
__device__ __forceinline__ void exl3_rotate_body(const void* x, int kind, const __half* suh, __half* xh, int k, int steps, const void* up = nullptr) {
    const int lane = threadIdx.x;
    const int col = blockIdx.x * 128 + 4 * lane;
    const long long at = static_cast<long long>(blockIdx.y) * k + col;
    const long long to = steps > 0 ? (static_cast<long long>(col >> 4) * steps + blockIdx.y) * 16 + (col & 15) : at;
    uint2 raw = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(x) + at);
    if constexpr (SILU) {
        // x is the gate: the down product's input is tf_silu_mul's row, made here
        const uint2 ur = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(up) + at);
        const uint32_t g[2] = {raw.x, raw.y}, u[2] = {ur.x, ur.y};
        uint32_t o[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) o[j] = exl3_silu_mul(g[j] & 0xffffu, u[j] & 0xffffu, kind) | (exl3_silu_mul(g[j] >> 16, u[j] >> 16, kind) << 16);
        raw = make_uint2(o[0], o[1]);
    }
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
    *reinterpret_cast<uint2*>(xh + to) = make_uint2(__builtin_bit_cast(uint32_t, lo), __builtin_bit_cast(uint32_t, hi));
}

extern "C" __global__ void __launch_bounds__(32) tf_exl3_rotate_in(const void* x, int kind, const __half* suh, __half* xh, int k, int steps) {
    exl3_rotate_body(x, kind, suh, xh, k, steps);
}

// tf_exl3_rotate_in of tf_silu_mul(gate, up) without the row between them: the same bits.
extern "C" __global__ void __launch_bounds__(32) tf_exl3_rotate_silu(const void* gate, const void* up, int kind, const __half* suh, __half* xh, int k, int steps) {
    exl3_rotate_body<true>(gate, kind, suh, xh, k, steps, up);
}

// out [rows, k] = x [rows, k] times blockdiag(H128) / sqrt(128), rounded once to x's kind (act numbering: 1 fp16, 2
// bf16): the input rotation of ciru_q8's experts (weights stored as W H, so W x = (W H)(H x)). A wave a 128-block, 8
// a block; each row on its own, so a row's bits never depend on the rows beside it. `out` may be `x`.
extern "C" __global__ void __launch_bounds__(256) tf_had128(const void* x, int kind, void* out, int k, long long waves) {
    const int lane = threadIdx.x & 31;
    const long long w = static_cast<long long>(blockIdx.x) * 8 + (threadIdx.x >> 5);
    if (w >= waves) return;
    const int per = k >> 7;
    const long long at = (w / per) * k + (w % per) * 128 + 4 * lane;
    const uint2 raw = *reinterpret_cast<const uint2*>(static_cast<const uint16_t*>(x) + at);
    const uint32_t r[2] = {raw.x, raw.y};
    float v[4];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        if (kind == 2) {
            v[2 * j] = __uint_as_float(r[j] << 16);
            v[2 * j + 1] = __uint_as_float(r[j] & 0xffff0000u);
        } else {
            const h16x2 xv = __builtin_bit_cast(h16x2, r[j]);
            v[2 * j] = static_cast<float>(xv[0]);
            v[2 * j + 1] = static_cast<float>(xv[1]);
        }
    }
    exl3_fwht(v, lane);
    uint32_t o[2];
#pragma unroll
    for (int j = 0; j < 2; ++j) {
        const float lo = v[2 * j] * kExl3Had, hi = v[2 * j + 1] * kExl3Had;
        if (kind == 2) {
            o[j] = __builtin_bit_cast(uint16_t, static_cast<__bf16>(lo)) | (static_cast<uint32_t>(__builtin_bit_cast(uint16_t, static_cast<__bf16>(hi))) << 16);
        } else {
            o[j] = __builtin_bit_cast(uint32_t, h16x2{static_cast<_Float16>(lo), static_cast<_Float16>(hi)});
        }
    }
    *reinterpret_cast<uint2*>(static_cast<uint16_t*>(out) + at) = make_uint2(o[0], o[1]);
}

// Arrivals of a product's split blocks at each (column block, row block): the last one finishes the sums and resets
// its count, so every launch on a stream finds them zero. Launches index it by (row block, column block).
constexpr int kExl3Counters = 65536;
__device__ int exl3_arrivals[kExl3Counters];

// k steps a slab stages (a prompt's 8 row tiles); a split's tiles are a multiple of it and of a round's 4 in flight.
constexpr int kExl3Slab = 4;

// A raw buffer over `p`: loads take a lane's 32-bit offset and a uniform one, with no per-lane address arithmetic.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t exl3_buffer(const void* p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, 0x7fffffff, 0x31004000);
}

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
        own[w][0] = exl3_window_pair<K2, CB, 0>(hi, lo);
        own[w][1] = exl3_window_pair<K2, CB, 1>(hi, lo);
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

// Columns 128 NT blockIdx.x.. over split blockIdx.y of K for rows 16 MT blockIdx.z..: wave w owns tile columns
// 8 (NT bx + n) + w (n < NT, one x fragment feeding the NT tiles' WMMAs) and of each a
// lane column c = lane % 16 of it, and rows 8 h .. of its WMMA B fragment (h = lane / 16) from two windows of the tile's
// stream; the halves swap theirs. Every row's sum is the same WMMA chain over its K range in K order.
// A round's row tiles (MT 1, 2) keep a ring of each lane's window words and x fragments in registers; a prompt's (MT 8)
// stages a slab of k steps (the block's 8 tiles each and its rows' x) through LDS, one slab ahead, a staged tile
// repeating its first 4 dwords after its end so a window never wraps. Both run the same WMMA chain: even and odd k steps
// on two sums added at the end, or with ONE a single chain over k, exl3_prompt's sums to the bit.
template <int K2, int CB, int MT, int NT, bool ONE = false>
__device__ __forceinline__ void exl3_product_body(const Exl3& a, int bx, int by, int bz, int arrival) {
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
    const int kt0 = by * tiles;
    const int m0 = bz * R;
    // the lane's windows: first dword in its tile and bit offset, the same for every tile
    int start[2], off[2];
#pragma unroll
    for (int w = 0; w < 2; ++w) {
        start[w] = ((exl3_end<K2>(exl3_first(c, h, w)) - 16 + RING) % RING) >> 5;
        off[w] = ((exl3_end<K2>(exl3_first(c, h, w)) - 16 + RING) % RING) & 31;
    }
    // even and odd k steps chain into separate sums (independent WMMAs interleave), added once at the end
    f32x8e acc[NT][MT], odd[NT][MT];
#pragma unroll
    for (int n = 0; n < NT; ++n)
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) acc[n][mt] = odd[n][mt] = f32x8e{0, 0, 0, 0, 0, 0, 0, 0};

    if constexpr (MT <= 2) {
        constexpr int D = 4;  // tiles in flight a wave
        // buffers over the block's tiles and x: a load takes a lane's byte offset and the tile's uniform one
        const __amdgpu_buffer_rsrc_t tb = exl3_buffer(a.trellis + static_cast<long long>(bx) * NT * a.stride_nb + wave * TD + static_cast<long long>(kt0) * a.stride_k);
        const int tstep = static_cast<int>(a.stride_k * 4);
        // x by k steps ([k / 16, m, 16], tf_exl3_rotate_in): a step's rows are one run
        const __amdgpu_buffer_rsrc_t xb = exl3_buffer(a.xh + static_cast<long long>(kt0) * 16 * a.m);
        const int xstep = 32 * a.m;
        uint32_t idx[NT][2][LW];
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int w = 0; w < 2; ++w)
#pragma unroll
                for (int i = 0; i < LW; ++i) idx[n][w][i] = 4u * ((start[w] + i) % TD + n * a.stride_nb);
        // rows past the call read its last row: their outputs are never stored
        uint32_t xo[MT];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const int r = m0 + mt * 16 + c;
            xo[mt] = static_cast<uint32_t>(r < a.m ? r : a.m - 1) * 32u;
        }
        // the tiles' words D - 1 k steps ahead (DRAM), x XD - 1 ahead (L2: a round's 16 rows miss L0 often enough
        // that one step ahead leaves them waiting)
        constexpr int XD = 3;
        uint32_t ring[D][NT][2][LW];
        u32x8e xa[D][MT];
        // every load is unconditional on a clamped tile, so the wait counters never drain the ring
        auto fetch = [&](int i, uint32_t (&r)[NT][2][LW]) {
            const int t = i < tiles ? i : tiles - 1;
#pragma unroll
            for (int n = 0; n < NT; ++n)
#pragma unroll
                for (int w = 0; w < 2; ++w)
#pragma unroll
                    for (int j = 0; j < LW; ++j) r[n][w][j] = __builtin_amdgcn_raw_buffer_load_b32(tb, idx[n][w][j], t * tstep, 0);
        };
        auto xfetch = [&](int i, u32x8e (&x)[MT]) {
            const int t = i < tiles ? i : tiles - 1;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const u32x4e lo = __builtin_amdgcn_raw_buffer_load_b128(xb, xo[mt], t * xstep, 0);
                const u32x4e hi = __builtin_amdgcn_raw_buffer_load_b128(xb, xo[mt] + 16, t * xstep, 0);
                x[mt] = __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
            }
        };
#pragma unroll
        for (int i = 0; i < D - 1; ++i) fetch(i, ring[i]);
#pragma unroll
        for (int i = 0; i < XD - 1; ++i) xfetch(i, xa[i]);
        // the launch takes `tiles` a multiple of D
        for (int i0 = 0; i0 < tiles; i0 += D) {
#pragma unroll
            for (int rb = 0; rb < D; ++rb) {
                fetch(i0 + rb + D - 1, ring[(rb + D - 1) % D]);
                xfetch(i0 + rb + XD - 1, xa[(rb + XD - 1) % D]);
#pragma unroll
                for (int n = 0; n < NT; ++n) {
                    const h16x16 b = exl3_fragment<K2, CB, LW>(ring[rb][n], off, c);
#pragma unroll
                    for (int mt = 0; mt < MT; ++mt) {
                        f32x8e& sum = !ONE && (rb & 1) ? odd[n][mt] : acc[n][mt];
                        sum = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16x16, xa[rb][mt]), b, sum);
                    }
                }
            }
        }
    } else {
        static_assert(NT == 1, "a prompt's row tiles take one column tile a wave");
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
        const char* tbase = reinterpret_cast<const char*>(a.trellis + static_cast<long long>(bx) * a.stride_nb + static_cast<long long>(kt0) * a.stride_k);
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
                    f32x8e& sum = !ONE && (i & 1) ? odd[0][mt] : acc[0][mt];
                    sum = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(__builtin_bit_cast(h16x16, x), b, sum);
                }
            }
            store((sl + 1) & 1);
            load(sl + 2);
            __syncthreads();
        }
    }

    if constexpr (!ONE) {
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) acc[n][mt] += odd[n][mt];
    }
    if (a.splits > 1) {
#pragma unroll
        for (int n = 0; n < NT; ++n)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    const int row = m0 + mt * 16 + 2 * e + h;
                    if (row < a.m) a.partial[(static_cast<long long>(by) * a.m + row) * a.n + (bx * NT + n) * 128 + wave * 16 + c] = acc[n][mt][e];
                }
        // the last split block to arrive adds every split's sums in split order (the same bits whichever arrives last)
        __shared__ int last;
        __threadfence();
        __syncthreads();
        int* arrived = &exl3_arrivals[arrival];
        if (tid == 0) last = atomicAdd(arrived, 1) == a.splits - 1;
        __syncthreads();
        if (!last) return;
        __threadfence();
        for (int n = 0; n < NT; ++n) {
            const int col0 = (bx * NT + n) * 128;
            for (int r = wave; r < R; r += 8) {
                const int row = m0 + r;
                if (row >= a.m) break;
                const long long at = static_cast<long long>(row) * a.n + col0 + 4 * lane;
                // volatile loads bypass the non-coherent vector caches, where another CU's sums may be stale
                float v[4] = {0, 0, 0, 0};
                for (int q = 0; q < a.splits; ++q) {
                    const u32x4e u = *reinterpret_cast<const volatile u32x4e*>(a.partial + static_cast<long long>(q) * a.m * a.n + at);
#pragma unroll
                    for (int j = 0; j < 4; ++j) v[j] += __uint_as_float(u[j]);
                }
                exl3_store(a, v, lane, row, col0);
            }
        }
        if (tid == 0) *arrived = 0;
        return;
    }
#pragma unroll
    for (int n = 0; n < NT; ++n) {
        const int col0 = (bx * NT + n) * 128;
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            __syncthreads();
#pragma unroll
            for (int e = 0; e < 8; ++e) red[2 * e + h][wave * 16 + c] = acc[n][mt][e];
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
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

template <int K2, int CB, int MT, int NT>
__global__ void __launch_bounds__(256) exl3_product(Exl3 a) {
    exl3_product_body<K2, CB, MT, NT>(a, blockIdx.x, blockIdx.y, blockIdx.z, blockIdx.z * gridDim.x + blockIdx.x);
}

// A prompt's rows below exl3_prompt's (mul1, one split): the product's tiles on one chain, exl3_prompt's bits.
template <int K2, int MT, int NT>
__global__ void __launch_bounds__(256) exl3_chain(Exl3 a) {
    exl3_product_body<K2, 2, MT, NT, true>(a, blockIdx.x, blockIdx.y, blockIdx.z, blockIdx.z * gridDim.x + blockIdx.x);
}

// ---- a prompt's product: the trellis decoded into LDS as FP16 on prompt_dq_core ----

// The 16-bit state window starting S bits into the words `w` (stream bits MSB first): the register holding it in its
// high half (exl3_window_hi) or low half, the other half ignored by exl3_hash. Halves of a word are free; any other
// window is the high half of the 32 stream bits from S, or the low half of those from S - 16, keyed so the register
// starts in a word's second half: two windows 16 bits apart (4-bit values 4 apart) share one alignbit.
template <int S>
constexpr bool exl3_window_hi() { return S % 32 == 0 || S % 32 > 16; }

// exl3_hash's two steps (D = 0) apart, so a batch can issue each over all its states before the next.
template <uint32_t C, bool HI>
__device__ __forceinline__ uint32_t exl3_mul_hi(uint32_t src) {
    uint32_t r;
    if constexpr (HI) asm("v_pk_mul_lo_u16 %0, %1, %2" : "=v"(r) : "v"(src), "s"(C & 0xffff0000u));
    else asm("v_pk_mul_lo_u16 %0, %1, %2 op_sel_hi:[0,1]" : "=v"(r) : "v"(src), "s"(C & 0xffff0000u));
    return r;
}
template <uint32_t C, bool HI>
__device__ __forceinline__ uint32_t exl3_mul_lo(uint32_t src, uint32_t r) {
    uint32_t x;
    if constexpr (HI) asm("v_mad_u32_u16 %0, %1, %2, %3 op_sel:[1,0,0,0]" : "=v"(x) : "v"(src), "s"(C & 0xffffu), "v"(r));
    else asm("v_mad_u32_u16 %0, %1, %2, %3" : "=v"(x) : "v"(src), "s"(C & 0xffffu), "v"(r));
    return x;
}
template <int S>
__device__ __forceinline__ uint32_t exl3_window(const uint32_t* w) {
    constexpr int o = S % 32;
    if constexpr (o == 0 || o == 16) return w[S / 32];
    constexpr int K = o > 16 ? S : S - 16;  // K % 32 in (16, 32)
    return __builtin_amdgcn_alignbit(w[K / 32], w[K / 32 + 1], 32 - K % 32);
}

// prompt_dq_core's hook over a whole-bit trellis (K2 even), FP16 exact. A stage's 32 k of BN columns are 2 k steps x
// BN / 16 tiles x 8 runs; run r of a tile is its 32 stream values 32 r.., which are columns r and r + 8 of the tile
// over all 16 k (exl3.md's placement), its bits the run's B dwords plus the dword before (the window of value 0 starts
// 16 bits back, wrapping to the tile's end). P = 1: a thread a run, both columns. P = 2: a run's two threads sit in two
// waves at the same lane, the first wave decoding column r, the second r + 8 (a wave-uniform branch).
template <int K2, int CB>
struct Exl3Prompt {
    template <int P>
    struct Dq {
        static_assert(K2 % 2 == 0 && (P == 1 || P == 2), "whole bits; one or two threads a run");
        static constexpr bool kPlaced = true;
        static constexpr bool kHalf = true;
        static constexpr int kDepth = 4;
        static constexpr int B = K2 / 2;  // bits a value: dwords a run
        static constexpr int TD = 4 * K2;  // dwords a tile
        static constexpr int NW = B + 1;
        __amdgpu_buffer_rsrc_t t;
        uint32_t pre, own, sstep;  // byte offsets: the dword before the run, the run's first; bytes a stage
        int col, chunk, half;
        uint32_t W[kDepth][NW];

        __device__ Dq(const Exl3& a, int n0, int tid, int bn) : t(prompt_rsrc(a.trellis)) {
            const int u = P == 1 ? tid : (tid >> 6) * 32 + (tid & 31);
            half = P == 1 ? 0 : __builtin_amdgcn_readfirstlane((tid >> 5) & 1);
            const int per = bn / 2;  // runs a k step
            const int ks = u / per, ct = (u % per) >> 3, r = u & 7;
            const int last = (a.n >> 4) - 1;
            const int nt = (n0 >> 4) + ct < last ? (n0 >> 4) + ct : last;
            const uint32_t tile = static_cast<uint32_t>(ks * a.stride_k + (nt >> 3) * a.stride_nb + (nt & 7) * TD);
            own = 4 * (tile + r * B);
            pre = 4 * (tile + (r * B + TD - 1) % TD);
            sstep = static_cast<uint32_t>(8 * a.stride_k);
            col = ct * 16 + r + 8 * half;
            chunk = 2 * ks;
        }

        template <int J>
        __device__ void fetch(int s) {
            const uint32_t so = s * sstep;
            W[J][0] = __builtin_amdgcn_raw_buffer_load_b32(t, pre, so, 0);
            int i = 0;
#pragma unroll
            for (; i + 4 <= B; i += 4) {
                const u32x4 v = prompt_load<u32x4>(t, own + 4 * i, so);
                W[J][1 + i] = v.x, W[J][2 + i] = v.y, W[J][3 + i] = v.z, W[J][4 + i] = v.w;
            }
            if constexpr (B % 4 == 3) {
                const auto v = __builtin_amdgcn_raw_buffer_load_b96(t, own + 4 * (B - 3), so, 0);
                W[J][B - 2] = v[0], W[J][B - 1] = v[1], W[J][B] = v[2];
            } else if constexpr (B % 4 == 2) {
                const uint2 v = prompt_load<uint2>(t, own + 4 * (B - 2), so);
                W[J][B - 1] = v.x, W[J][B] = v.y;
            } else if constexpr (B % 4 == 1) {
                W[J][B] = __builtin_amdgcn_raw_buffer_load_b32(t, own + 4 * (B - 1), so, 0);
            }
        }

        // Element e of columns H0.. (16 a column, k order): column H0 + e / 16 is r + 8 H of the run, its dword d = e % 16
        // / 2 holding k 2d, 2d + 1: values 8d + 4H, + 1 for d < 4, 8 (d - 4) + 4H + 2, + 3 above.
        template <int H0, int E>
        static constexpr int value() {
            constexpr int h = H0 + E / 16, d = (E % 16) / 2;
            return 8 * (d % 4) + 4 * h + 2 * (d / 4) + E % 2;
        }
        template <int H0, int E>
        static constexpr int window() { return 16 + (value<H0, E>() + 1) * B; }

        // NC columns from H0 into words d[c][0..7] (k pairs as half2, first low): mul1, each step of the codebook over
        // every value before the next, so the dependent steps sit 16 NC instructions apart.
        template <int H0, int NC>
        __device__ static void columns(const uint32_t* w, uint32_t (&d)[NC][8]) {
            static_assert(CB == 2, "the prompt tile decodes mul1");
            constexpr uint32_t C = 0x83DCD12Du;
            uint32_t s[16 * NC], x[16 * NC];
            [&]<int... E>(std::integer_sequence<int, E...>) {
                ((s[E] = exl3_window<window<H0, E>()>(w)), ...);
                // the high product half (s C_hi mod 2^16) << 16, then + s C_lo: s C mod 2^32 (exl3_hash)
                ((x[E] = exl3_mul_hi<C, exl3_window_hi<window<H0, E>()>()>(s[E])), ...);
                ((x[E] = exl3_mul_lo<C, exl3_window_hi<window<H0, E>()>()>(s[E], x[E])), ...);
            }(std::make_integer_sequence<int, 16 * NC>{});
            // fp16 bits of 1024 + each byte sum, packed, then one fma each: h / 147.7 - 10.39
            uint32_t p[8 * NC];
#pragma unroll
            for (int i = 0; i < 8 * NC; ++i) p[i] = __builtin_amdgcn_sad_u8(x[2 * i], 0u, 0x64006400u);
#pragma unroll
            for (int i = 0; i < 8 * NC; ++i) p[i] = __builtin_amdgcn_sad_hi_u8(x[2 * i + 1], 0u, p[i]);
            const h16x2 scale = {__builtin_bit_cast(_Float16, static_cast<unsigned short>(0x1EEE)), __builtin_bit_cast(_Float16, static_cast<unsigned short>(0x1EEE))};
            const h16x2 bias = {__builtin_bit_cast(_Float16, static_cast<unsigned short>(0xC931)), __builtin_bit_cast(_Float16, static_cast<unsigned short>(0xC931))};
#pragma unroll
            for (int i = 0; i < 8 * NC; ++i) d[i / 8][i % 8] = __builtin_bit_cast(uint32_t, __builtin_elementwise_fma(__builtin_bit_cast(h16x2, p[i]), scale, bias));
        }

        template <int J, class Store>
        __device__ void put(Store st) const {
            auto store = [&](const uint32_t (&d)[8], int c) {
                st(c, chunk, u32x4{d[0], d[1], d[2], d[3]});
                st(c, chunk + 1, u32x4{d[4], d[5], d[6], d[7]});
            };
            if constexpr (P == 1) {
                uint32_t d[2][8];
                columns<0, 2>(W[J], d);
                store(d[0], col);
                store(d[1], col + 8);
            } else if (half) {
                uint32_t d[1][8];
                columns<1, 1>(W[J], d);
                store(d[0], col);
            } else {
                uint32_t d[1][8];
                columns<0, 1>(W[J], d);
                store(d[0], col);
            }
        }
    };
};

// A row's four finished outputs at col.. in the output kind (exl3_store's tail).
__device__ __forceinline__ void exl3_put(const Exl3& a, const float (&v)[4], int row, int col) {
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
    } else if (a.out_kind == 3) {
        // the residual: out (bf16) += the outputs as kind 1 stores them, then tf_add's sum and rounding
        uint16_t* const o = static_cast<uint16_t*>(a.out) + at;
        const uint2 xr = *reinterpret_cast<const uint2*>(o);
        const uint32_t xw[2] = {xr.x, xr.y};
        uint32_t u[2];
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float y0 = static_cast<float>(static_cast<__bf16>(v[2 * j]));
            const float y1 = static_cast<float>(static_cast<__bf16>(v[2 * j + 1]));
            u[j] = exl3_bf16(__uint_as_float(xw[j] << 16) + y0) | (exl3_bf16(__uint_as_float(xw[j] & 0xffff0000u) + y1) << 16);
        }
        *reinterpret_cast<uint2*>(o) = make_uint2(u[0], u[1]);
    } else {
        const h16x2 lo = {static_cast<_Float16>(v[0]), static_cast<_Float16>(v[1])};
        const h16x2 hi = {static_cast<_Float16>(v[2]), static_cast<_Float16>(v[3])};
        *reinterpret_cast<uint2*>(static_cast<uint16_t*>(a.out) + at) = make_uint2(__builtin_bit_cast(uint32_t, lo), __builtin_bit_cast(uint32_t, hi));
    }
}

// Lane L's partner L ^ M of v: DPP within a 16-lane row, permlanex16 across the two.
template <int M>
__device__ __forceinline__ float exl3_xor(float v) {
    const int u = __builtin_bit_cast(int, v);
    if constexpr (M == 16) return __builtin_bit_cast(float, __builtin_amdgcn_permlanex16(0, u, 0x76543210, 0xfedcba98, false, false));
    else return __builtin_bit_cast(float, __builtin_amdgcn_update_dpp(0, u, 0x160 + M, 0xf, 0xf, false));
}

template <int M>
__device__ __forceinline__ void exl3_fwht_lanes(float (&v)[4], int lane) {
    float o[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) o[j] = exl3_xor<M>(v[j]);
#pragma unroll
    for (int j = 0; j < 4; ++j) v[j] = (lane & M) ? o[j] - v[j] : v[j] + o[j];
}

// exl3_fwht's butterflies in its order, the partners through DPP and permlanex16 instead of LDS (the same sums).
__device__ __forceinline__ void exl3_fwht_dpp(float (&v)[4], int lane) {
    const float a = v[0] + v[1], b = v[0] - v[1], c = v[2] + v[3], d = v[2] - v[3];
    v[0] = a + c;
    v[1] = b + d;
    v[2] = a - c;
    v[3] = b - d;
    exl3_fwht_lanes<1>(v, lane);
    exl3_fwht_lanes<2>(v, lane);
    exl3_fwht_lanes<4>(v, lane);
    exl3_fwht_lanes<8>(v, lane);
    exl3_fwht_lanes<16>(v, lane);
}

// y [m, n] = rot_in(x) W_q rotated out (exl3_store), BM x BN a block, x already rotated into a.xh [m, k]. Each output
// is one fp32 WMMA chain over k in order: its bits depend on its row and (K, N) alone, never on m or the block shape.
template <int K2, int CB, int BM, int BN>
__global__ void __launch_bounds__(BM * BN / 128) exl3_prompt(Exl3 a) {
#if TF_DEVICE_WMMA_GFX11
    constexpr int WGM = BM / 64;
    constexpr int WAVES = BM * BN / 4096;
    constexpr int PITCH = BN + 4;  // floats a staged row
    __shared__ __attribute__((aligned(16))) uint32_t lds[2 * (BM + BN) * 16];
    // a pass stages a 16-row tile of WMP wave rows: all of them where they fit, else half
    constexpr int WMP = 16 * WGM * PITCH * 4 <= sizeof(lds) ? WGM : WGM / 2;
    constexpr int SR = 16 * WMP;  // staged rows
    static_assert(SR * PITCH * 4 <= sizeof(lds), "a pass's sums fit the tile's LDS");
    int m0, n0;
    prompt_place_at<BM, BN, 1>(a, blockIdx.x, m0, n0);  // N fastest: 2 and 4 M tiles a band measured 0-2% and 1-5% slower
    f32x8 acc[4][4];
    prompt_dq_core<Exl3Prompt<K2, CB>::template Dq, BM, BN>(a, a.xh, m0, n0, lds, acc);

    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int wm = wave % WGM;
    const int wn = wave / WGM;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    float* const red = reinterpret_cast<float*>(lds);
    // a lane's four columns of each 128-column block: svh and the bias, read once
    float sc[BN / 128][4], bi[BN / 128][4];
#pragma unroll
    for (int cb = 0; cb < BN / 128; ++cb) {
        const int col = (n0 + cb * 128 < a.n ? n0 + cb * 128 : a.n - 128) + 4 * lane;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sc[cb][j] = exl3_half(a.svh, col + j);
            bi[cb][j] = a.bias != nullptr ? exl3_half(a.bias, col + j) : 0.f;
        }
    }
#pragma unroll
    for (int g = 0; g < WGM / WMP; ++g) {
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
            if (wm / WMP == g) {
#pragma unroll
                for (int i = 0; i < 8; ++i)
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) red[((wm % WMP) * 16 + 2 * i + rh) * PITCH + wn * 64 + nt * 16 + rl] = acc[mt][nt][i];
            }
            __syncthreads();
            for (int it = wave; it < SR * (BN / 128); it += WAVES) {
                const int sr = it % SR, cb = it / SR;
                const int row = m0 + (g * WMP + (sr >> 4)) * 64 + mt * 16 + (sr & 15);
                const int col0 = n0 + cb * 128;
                if (row < a.m && col0 < a.n) {
                    const float4 u = *reinterpret_cast<const float4*>(red + sr * PITCH + cb * 128 + 4 * lane);
                    float v[4] = {u.x, u.y, u.z, u.w};
                    exl3_fwht_dpp(v, lane);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        v[j] = v[j] * kExl3Had * (cb ? sc[BN / 128 - 1][j] : sc[0][j]);
                        if (a.bias != nullptr) v[j] += cb ? bi[BN / 128 - 1][j] : bi[0][j];
                    }
                    exl3_put(a, v, row, col0 + 4 * lane);
                }
            }
            __syncthreads();
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

#if TF_MATRIX
#define TF_EXL3_PROMPT(K2)                                                    \
    template __global__ void exl3_prompt<K2, 2, 128, 256>(Exl3);              \
    template __global__ void exl3_prompt<K2, 2, 256, 128>(Exl3);              \
    template __global__ void exl3_chain<K2, 1, 1>(Exl3);                      \
    template __global__ void exl3_chain<K2, 1, 2>(Exl3);                      \
    template __global__ void exl3_chain<K2, 2, 1>(Exl3);                      \
    template __global__ void exl3_chain<K2, 2, 2>(Exl3);                      \
    template __global__ void exl3_chain<K2, 8, 1>(Exl3);
TF_EXL3_PROMPT(4)
TF_EXL3_PROMPT(6)
TF_EXL3_PROMPT(8)
TF_EXL3_PROMPT(10)
TF_EXL3_PROMPT(12)
TF_EXL3_PROMPT(14)
TF_EXL3_PROMPT(16)
#undef TF_EXL3_PROMPT
#endif

#if TF_MATRIX
#define TF_EXL3_MT(K2, CB)                                                    \
    template __global__ void exl3_product<K2, CB, 1, 1>(Exl3);                \
    template __global__ void exl3_product<K2, CB, 1, 2>(Exl3);                \
    template __global__ void exl3_product<K2, CB, 2, 1>(Exl3);                \
    template __global__ void exl3_product<K2, CB, 2, 2>(Exl3);                \
    template __global__ void exl3_product<K2, CB, 8, 1>(Exl3);
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
