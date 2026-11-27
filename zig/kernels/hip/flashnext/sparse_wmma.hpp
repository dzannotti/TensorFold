#pragma once

#include "../ops/common.hpp"
#include "common/arch.hpp"

// Sparse attention on gfx11 WMMA, 12 heads, bf16 cache, 256 dims: zig/kernels/loom/sparse_attention's sparse_attend,
// byte for byte (README.md, Sum order), for decode, verify and staged prompt rows.

namespace {

#if TF_DEVICE_WMMA_GFX11

typedef __bf16 sw_bf16x16 __attribute__((ext_vector_type(16)));
typedef float sw_f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned sw_u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned sw_u32x4 __attribute__((ext_vector_type(4)));

constexpr int kSwGroup = 12, kSwDims = 256, kSwTile = 32, kSwParts = 4, kSwNTiles = kSwDims / 16;
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

// CUDA's tf_exp (common/glue_math.cuh): a degree-7 Taylor polynomial after x = j ln2 + f, 2^j in two exact factors.
__device__ __forceinline__ float sw_exp(float x) {
    if (x != x) return x;
    if (x > 104.0f) return __int_as_float(0x7f800000);
    if (x < -104.0f) return 0.0f;
    const float j = __builtin_fmaf(x, 0x1.715476p+0f, 0x1.8p+23f) - 0x1.8p+23f;
    float f = __builtin_fmaf(j, -0x1.62e4p-1f, x);
    f = __builtin_fmaf(j, -0x1.7f7d1cp-20f, f);
    float p = 0x1.a01a02p-13f;
    p = __builtin_fmaf(p, f, 0x1.6c16c2p-10f);
    p = __builtin_fmaf(p, f, 0x1.111112p-7f);
    p = __builtin_fmaf(p, f, 0x1.555556p-5f);
    p = __builtin_fmaf(p, f, 0x1.555556p-3f);
    p = __builtin_fmaf(p, f, 0.5f);
    p = __builtin_fmaf(p, f, 1.0f);
    p = __builtin_fmaf(p, f, 1.0f);
    const int i = static_cast<int>(j);
    const float a = __int_as_float(i > 0 ? 0x7f000000 : 0x02000000);
    const float b = __int_as_float(i > 0 ? i << 23 : (i + 250) << 23);
    return (p * a) * b;
}

// Head 2e + half's copy of a value lane = head holds.
__device__ __forceinline__ sw_f32x8 sw_heads(float x, int half) {
    sw_f32x8 r;
#pragma unroll
    for (int e = 0; e < 8; ++e) r[e] = __shfl(x, 2 * e + half, 32);
    return r;
}

struct SwWeights {
    float a, b, m, l;
};

// CUDA's fn_attn_merge step: a part (m, l) folded into the running (mm, ll).
__device__ __forceinline__ SwWeights sw_weights(float mm, float ll, float m, float l) {
    const bool active = l > 0.0f;
    const float next = active ? fmaxf(mm, m) : mm;
    const float a = active ? (mm == -INFINITY ? 0.0f : sw_exp(mm - next)) : 1.0f;
    const float b = active ? sw_exp(m - next) : 0.0f;
    return {a, b, next, __builtin_fmaf(l, b, ll * a)};
}

// Block (g, r), wave p: part p = [p n / 4, (p + 1) n / 4) of row r's keys for kv head g in 32-key tiles; the parts
// merged in order in LDS by wave 0, out = bf16(o / l) kept as fp32.
__device__ void sparse_wmma(const float* q, const uint16_t* k16, const uint16_t* v16, const int* ids, long long id_stride,
                            const int* count, const int* sparse, float* out, int heads, long long cap, float scale) {
    __shared__ __attribute__((aligned(16))) uint32_t arena[16 * kSwQPitch / 2 + kSwParts * 256];
    uint16_t* const qp = reinterpret_cast<uint16_t*>(arena);
    float* const sc = reinterpret_cast<float*>(arena);
    constexpr int lo_at = 0, hi_at = kSwNTiles * 32 * 4, ms_at = hi_at + kSwNTiles * 32 * 2;
    const int g = blockIdx.x, r = blockIdx.y;
    const int t = threadIdx.x, lane = t & 31, part = t >> 5, col = lane & 15, half = lane >> 4;
    const int n = count[r];
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const int* const idr = listed ? ids + r * id_stride : nullptr;
    const long long base = static_cast<long long>(g) * cap;
    const long long qrow = static_cast<long long>(r) * heads + g * kSwGroup;
    for (int i = t; i < 16 * kSwDims; i += 32 * kSwParts) {
        const int h = i / kSwDims, d = i % kSwDims;
        qp[h * kSwQPitch + d] = h < kSwGroup ? float_to_bf16(q[qrow * kSwDims + h * kSwDims + d]) : 0;
    }
    __syncthreads();
    uint16_t* const tw = reinterpret_cast<uint16_t*>(arena + 16 * kSwQPitch / 2 + part * 256);
    const uint16_t* const qc = qp + col * kSwQPitch;
    const sw_f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    float m = -INFINITY, l = 0.f;
    sw_f32x8 o[kSwNTiles];
#pragma unroll
    for (int j = 0; j < kSwNTiles; ++j) o[j] = zero;
    const int lo = static_cast<int>(static_cast<long long>(part) * n / kSwParts);
    const int hi = static_cast<int>(static_cast<long long>(part + 1) * n / kSwParts);
    for (int t0 = lo; t0 < hi; t0 += kSwTile) {
        const int tn = min(kSwTile, hi - t0), last = t0 + tn - 1;
        const int i0 = min(t0 + col, last), i1 = min(t0 + 16 + col, last);
        const long long key0 = base + (listed ? idr[i0] : i0), key1 = base + (listed ? idr[i1] : i1);
        const uint16_t* const k0 = k16 + key0 * kSwDims;
        const uint16_t* const k1 = k16 + key1 * kSwDims;
        sw_f32x8 s0 = zero, s1 = zero;
#pragma unroll
        for (int c = 0; c < kSwNTiles; ++c) {
            const sw_u32x8 b = sw_frag(qc + 16 * c);
            s0 = sw_mma(sw_frag(k0 + 16 * c), b, s0);
            s1 = sw_mma(sw_frag(k1 + 16 * c), b, s1);
        }
        sw_f32x8 x0, x1;
        float tm = -INFINITY;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            x0[e] = 2 * e + half < tn ? s0[e] * scale : -INFINITY;
            x1[e] = 16 + 2 * e + half < tn ? s1[e] * scale : -INFINITY;
            tm = fmaxf(tm, fmaxf(x0[e], x1[e]));
        }
        tm = fmaxf(tm, __shfl_xor(tm, 16, 32));
        const bool active = tm != -INFINITY;
        const float next = active ? fmaxf(m, tm) : m;
        const float alpha = active ? (m == -INFINITY ? 0.0f : sw_exp(m - next)) : 1.0f;
        sw_f32x8 p0, p1;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            p0[e] = x0[e] == -INFINITY ? 0.0f : sw_exp(x0[e] - next);
            p1[e] = x1[e] == -INFINITY ? 0.0f : sw_exp(x1[e] - next);
        }
        float sum = 0.0f;
#pragma unroll
        for (int e = 0; e < 8; ++e) sum = sum + p0[e];
#pragma unroll
        for (int e = 0; e < 8; ++e) sum = sum + p1[e];
        sum = sum + __shfl_xor(sum, 16, 32);
        l = __builtin_fmaf(l, alpha, sum);
        m = next;
        sw_u32x8 a0, a1;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const float y0 = __shfl_xor(p0[e], 16, 32), y1 = __shfl_xor(p1[e], 16, 32);
            a0[e] = (static_cast<uint32_t>(float_to_bf16(half ? p0[e] : y0)) << 16) | float_to_bf16(half ? y0 : p0[e]);
            a1[e] = (static_cast<uint32_t>(float_to_bf16(half ? p1[e] : y1)) << 16) | float_to_bf16(half ? y1 : p1[e]);
        }
        const sw_f32x8 fe = sw_heads(alpha, half);
        // values: lanes of half h stage fragment h's 16 keys of N tile j (element (dim d, key c) at
        // (h * 16 + d) * 16 + (c ^ 8 (d & 1))), read back as one dim's 16 keys; o = mma(p1, v1, mma(p0, v0, o f))
        const uint16_t* const vrow = v16 + (half ? key1 : key0) * kSwDims;
        uint16_t* const th = tw + half * 256;
#pragma unroll
        for (int j = 0; j < kSwNTiles; ++j) {
            const sw_u32x8 w = sw_frag(vrow + 16 * j);
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                th[(2 * e) * 16 + col] = static_cast<uint16_t>(w[e]);
                th[(2 * e + 1) * 16 + (col ^ 8)] = static_cast<uint16_t>(w[e] >> 16);
            }
            __builtin_amdgcn_wave_barrier();
            sw_u32x8 bv[2];
#pragma unroll
            for (int b = 0; b < 2; ++b) {
                const uint16_t* row = tw + (b * 16 + col) * 16;
                const int sw = (col & 1) * 8;
                bv[b] = sw_join(*reinterpret_cast<const sw_u32x4*>(row + sw), *reinterpret_cast<const sw_u32x4*>(row + (8 ^ sw)));
            }
            o[j] = sw_mma(a1, bv[1], sw_mma(a0, bv[0], o[j] * fe));
            __builtin_amdgcn_wave_barrier();
        }
    }
    __syncthreads();
    float mm = -INFINITY, ll = 0.0f;
    sw_f32x8 oo[kSwNTiles];
#pragma unroll
    for (int j = 0; j < kSwNTiles; ++j) oo[j] = zero;
    for (int p = 0; p < kSwParts; ++p) {
        if (p > 0) {
            if (part == p) {
#pragma unroll
                for (int j = 0; j < kSwNTiles; ++j) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) sc[lo_at + (j * 32 + lane) * 4 + e] = o[j][e];
#pragma unroll
                    for (int e = 0; e < 2; ++e) sc[hi_at + (j * 32 + lane) * 2 + e] = o[j][4 + e];
                }
                if (half == 0) {
                    sc[ms_at + col * 2] = m;
                    sc[ms_at + col * 2 + 1] = l;
                }
            }
            __syncthreads();
        }
        if (part == 0) {
            const float pm = p > 0 ? sc[ms_at + col * 2] : m, pl = p > 0 ? sc[ms_at + col * 2 + 1] : l;
            const SwWeights w = sw_weights(mm, ll, pm, pl);
            const sw_f32x8 a = sw_heads(w.a, half), b = sw_heads(w.b, half);
            mm = w.m;
            ll = w.l;
#pragma unroll
            for (int j = 0; j < kSwNTiles; ++j) {
                sw_f32x8 x = o[j];
                if (p > 0) {
#pragma unroll
                    for (int e = 0; e < 4; ++e) x[e] = sc[lo_at + (j * 32 + lane) * 4 + e];
#pragma unroll
                    for (int e = 0; e < 2; ++e) x[4 + e] = sc[hi_at + (j * 32 + lane) * 2 + e];
                    x[6] = 0.0f;
                    x[7] = 0.0f;
                }
#pragma unroll
                for (int e = 0; e < 8; ++e) oo[j][e] = __builtin_fmaf(x[e], b[e], oo[j][e] * a[e]);
            }
        }
        if (p > 0) __syncthreads();
    }
    if (part == 0) {
        const sw_f32x8 lh = sw_heads(ll, half);
#pragma unroll
        for (int j = 0; j < kSwNTiles; ++j)
#pragma unroll
            for (int e = 0; e < 6; ++e)
                out[(qrow + 2 * e + half) * kSwDims + 16 * j + col] =
                    lh[e] > 0.0f ? bf16_to_float(float_to_bf16(oo[j][e] / lh[e])) : 0.0f;
    }
}

extern "C" __global__ void __launch_bounds__(32 * kSwParts)
tf_fn_sparse_wmma12_bf16(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                         const int* count, const int* sparse, float* out, int heads, int kv_heads, int dims,
                         long long cap, float scale) {
    if (dims != kSwDims || heads != kv_heads * kSwGroup) return;
    sparse_wmma(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride, count, sparse, out,
                heads, cap, scale);
}

#endif

}  // namespace
