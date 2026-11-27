#pragma once

#include "../ops/common.hpp"
#include "common/arch.hpp"

// sparse_parts on gfx11 WMMA, 12 heads, bf16 cache, 256 dims: the order of zig/kernels/loom/sparse_attention/README.md.

namespace {

#if TF_DEVICE_WMMA_GFX11

typedef __bf16 sw_bf16x16 __attribute__((ext_vector_type(16)));
typedef float sw_f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned sw_u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned sw_u32x4 __attribute__((ext_vector_type(4)));

constexpr int kSwGroup = 12, kSwDims = 256, kSwTile = 16, kSwParts = 16, kSwBlockParts = 4;
constexpr int kSwQPitch = kSwDims + 8;

__device__ __forceinline__ sw_f32x8 sw_mma(sw_u32x8 a, sw_u32x8 b, sw_f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(sw_bf16x16, a),
                                                       __builtin_bit_cast(sw_bf16x16, b), c);
}

__device__ __forceinline__ sw_u32x8 sw_join(sw_u32x4 lo, sw_u32x4 hi) {
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}

__device__ __forceinline__ sw_u32x8 sw_frag(const uint16_t* p) {
    return sw_join(*reinterpret_cast<const sw_u32x4*>(p), *reinterpret_cast<const sw_u32x4*>(p + 8));
}

__device__ __forceinline__ uint32_t sw_top(float x) { return __float_as_uint(x) & 0xffff0000u; }

// x = hi + mid + lo exactly, each a bf16 (their upper halves), +0 for a zero part.
struct SwSplit {
    uint32_t hi, mid, lo;
};

__device__ __forceinline__ SwSplit sw_split(float x) {
    const uint32_t a = sw_top(x);
    const float r = x - __uint_as_float(a);
    const uint32_t b = sw_top(r);
    const float s = r - __uint_as_float(b);
    return {sw_top(__uint_as_float(a) + 0.f), sw_top(__uint_as_float(b) + 0.f), sw_top(s + 0.f)};
}

__device__ __forceinline__ uint32_t sw_pack(uint32_t lo_elem, uint32_t hi_elem) { return hi_elem | (lo_elem >> 16); }

// Block (g, r, z), wave pair i: part z * kSwBlockParts + i of row r's keys for kv head g, wave side s of the pair
// owning dims [128 s, 128 s + 128); writes the part's o, m and l for tf_fn_sparse_merge.
__device__ void sparse_wmma(const float* q, const uint16_t* k16, const uint16_t* v16, const int* ids, long long id_stride,
                            const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads,
                            long long cap, float scale) {
    __shared__ __attribute__((aligned(16))) uint16_t qp[3][16][kSwQPitch];
    __shared__ __attribute__((aligned(16))) uint32_t slots[2 * kSwBlockParts][256];
    const int g = blockIdx.x, r = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5, col = lane & 15, half = lane >> 4, side = wave & 1;
    const int part = blockIdx.z * kSwBlockParts + (wave >> 1);
    const int n = count[r];
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int lo = static_cast<int>(static_cast<long long>(part) * n / kSwParts);
    const int hi = static_cast<int>(static_cast<long long>(part + 1) * n / kSwParts);
    const long long head_base = static_cast<long long>(g) * cap * kSwDims;
    const float* qrow = q + (static_cast<long long>(r) * heads + g * kSwGroup) * kSwDims;
    for (int i = t; i < 16 * kSwDims; i += 2 * kSwBlockParts * 32) {
        const int h = i / kSwDims, d = i % kSwDims;
        const SwSplit s = h < kSwGroup ? sw_split(qrow[h * kSwDims + d] * scale) : SwSplit{0, 0, 0};
        qp[0][h][d] = static_cast<uint16_t>(s.hi >> 16);
        qp[1][h][d] = static_cast<uint16_t>(s.mid >> 16);
        qp[2][h][d] = static_cast<uint16_t>(s.lo >> 16);
    }
    __syncthreads();
    const sw_f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    sw_f32x8 o[8];
#pragma unroll
    for (int i = 0; i < 8; ++i) o[i] = zero;
    uint32_t* const mine = slots[wave];
    const uint32_t* const other = slots[wave ^ 1];
    uint16_t* const tw = reinterpret_cast<uint16_t*>(mine);
    const uint16_t* const qh = &qp[0][col][side * (kSwDims / 2)];
    // every wave runs the longest part's tile count: the barriers stay block-uniform; extra tiles do nothing
    const int iters = ((n + kSwParts - 1) / kSwParts + kSwTile - 1) / kSwTile;
    for (int it = 0; it < iters; ++it) {
        const int t0 = lo + it * kSwTile;
        const int tn = min(kSwTile, hi - t0);
        const bool live = col < tn;
        long long key = 0;
        sw_f32x8 x = zero;
        if (tn > 0) {
            const int at = t0 + (live ? col : 0);
            key = listed ? ids[r * id_stride + at] : at;
            // hi parts of the side's 8 chunks one WMMA each from +0, summed in chunk order, plus mid, lo chain
            const uint16_t* kh = k16 + head_base + key * kSwDims + side * (kSwDims / 2);
            sw_f32x8 hs = zero, ch = zero;
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                sw_u32x8 a = sw_frag(kh + c * 16);
                if (!live) a = sw_u32x8{0, 0, 0, 0, 0, 0, 0, 0};
                const sw_f32x8 dc = sw_mma(a, sw_frag(qh + c * 16), zero);
                if (c == 0) hs = dc;
                else hs = hs + dc;
                ch = sw_mma(a, sw_frag(qh + 16 * kSwQPitch + c * 16), ch);
                ch = sw_mma(a, sw_frag(qh + 32 * kSwQPitch + c * 16), ch);
            }
            x = hs + ch;
        }
        *reinterpret_cast<sw_f32x8*>(mine + lane * 8) = x;
        __syncthreads();
        sw_u32x8 ph, pm, pl;
        sw_f32x8 fe;
        if (tn > 0) {
            const sw_f32x8 y = *reinterpret_cast<const sw_f32x8*>(other + lane * 8);
            const sw_f32x8 s = side ? y + x : x + y;
            // lane: head col; element e: key 2e + half
            float top = -INFINITY;
#pragma unroll
            for (int e = 0; e < 8; ++e)
                if (2 * e + half < tn) top = fmaxf(top, s[e]);
            top = fmaxf(top, __shfl_xor(top, 16, 32));
            top = fmaxf(top, m);
            sw_f32x8 p;
#pragma unroll
            for (int e = 0; e < 8; ++e) p[e] = 2 * e + half < tn && col < kSwGroup ? expf(s[e] - top) : 0.f;
            float sum = p[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) sum += p[e];
            sum += __shfl_xor(sum, 16, 32);
            const float f = expf(m - top);
            l = l * f + sum;
            m = top;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                fe[e] = __shfl(f, 2 * e + half, 32);
                const float oth = __shfl_xor(p[e], 16, 32);
                const SwSplit x0 = sw_split(half ? oth : p[e]), x1 = sw_split(half ? p[e] : oth);
                ph[e] = sw_pack(x0.hi, x1.hi);
                pm[e] = sw_pack(x0.mid, x1.mid);
                pl[e] = sw_pack(x0.lo, x1.lo);
            }
        }
        __syncthreads();
        if (tn > 0) {
            // values: lanes of half h stage N tile 2j + h of key col (element (dim d, key c) of staged tile b at
            // (b * 16 + d) * 16 + (c ^ 8 (d & 1))), read back as one dim's 16 keys; d = lo, mid, hi from +0,
            // o = fma(o, f, d)
            const uint16_t* vh = v16 + head_base + key * kSwDims + side * (kSwDims / 2) + half * 16;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sw_u32x8 w = sw_frag(vh + 32 * j);
                if (!live) w = sw_u32x8{0, 0, 0, 0, 0, 0, 0, 0};
                uint16_t* const th = tw + half * 256;
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    th[(2 * e) * 16 + col] = static_cast<uint16_t>(w[e]);
                    th[(2 * e + 1) * 16 + (col ^ 8)] = static_cast<uint16_t>(w[e] >> 16);
                }
                __builtin_amdgcn_wave_barrier();
#pragma unroll
                for (int b = 0; b < 2; ++b) {
                    const uint16_t* row = tw + (b * 16 + col) * 16;
                    const int sw = (col & 1) * 8;
                    const sw_u32x8 bv = sw_join(*reinterpret_cast<const sw_u32x4*>(row + sw),
                                                *reinterpret_cast<const sw_u32x4*>(row + (8 ^ sw)));
                    sw_f32x8 d = sw_mma(pl, bv, zero);
                    d = sw_mma(pm, bv, d);
                    d = sw_mma(ph, bv, d);
#pragma unroll
                    for (int e = 0; e < 8; ++e) o[2 * j + b][e] = __builtin_fmaf(o[2 * j + b][e], fe[e], d[e]);
                }
                __builtin_amdgcn_wave_barrier();
            }
        }
    }
    const long long row0 = (static_cast<long long>(r) * heads + g * kSwGroup) * kSwParts + part;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int h = 2 * e + half;
        if (h >= kSwGroup) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i)
            parts_o[(row0 + h * kSwParts) * kSwDims + side * (kSwDims / 2) + i * 16 + col] = o[i][e];
    }
    if (side == 0 && half == 0 && col < kSwGroup) {
        parts_ms[(row0 + col * kSwParts) * 2] = m;
        parts_ms[(row0 + col * kSwParts) * 2 + 1] = l;
    }
}

extern "C" __global__ void __launch_bounds__(2 * kSwBlockParts * 32)
tf_fn_sparse_wmma12_bf16(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                         const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads,
                         int dims, long long cap, float scale, int parts) {
    if (dims != kSwDims || heads != kv_heads * kSwGroup || parts != kSwParts) return;
    sparse_wmma(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride, count, sparse,
                parts_o, parts_ms, heads, cap, scale);
}

#endif

}  // namespace
