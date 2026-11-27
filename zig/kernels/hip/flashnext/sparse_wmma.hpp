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


// Block (g, r, z), wave pair i: part z * kSwBlockParts + i of row r's keys for kv head g, wave side s of the pair
// owning dims [128 s, 128 s + 128); writes the part's o, m and l for tf_fn_sparse_merge.
__device__ void sparse_wmma(const float* q, const uint16_t* k16, const uint16_t* v16, const int* ids, long long id_stride,
                            const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads,
                            long long cap, float scale) {
    __shared__ __attribute__((aligned(16))) uint16_t qp[1][16][kSwQPitch];
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
        qp[0][h][d] = h < kSwGroup ? float_to_bf16(qrow[h * kSwDims + d]) : 0;
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
            // the side's 8 chunks as one chain from +0
            const uint16_t* kh = k16 + head_base + key * kSwDims + side * (kSwDims / 2);
#pragma unroll
            for (int c = 0; c < 8; ++c) {
                sw_u32x8 a = sw_frag(kh + c * 16);
                if (!live) a = sw_u32x8{0, 0, 0, 0, 0, 0, 0, 0};
                x = sw_mma(a, sw_frag(qh + c * 16), x);
            }
        }
        *reinterpret_cast<sw_f32x8*>(mine + lane * 8) = x;
        __syncthreads();
        sw_u32x8 pb;
        sw_f32x8 fe;
        if (tn > 0) {
            const sw_f32x8 y = *reinterpret_cast<const sw_f32x8*>(other + lane * 8);
            const sw_f32x8 s = (side ? y + x : x + y) * scale;
            // lane: head col; element e: key 2e + half
            float top = -INFINITY;
#pragma unroll
            for (int e = 0; e < 8; ++e)
                if (2 * e + half < tn) top = fmaxf(top, s[e]);
            top = fmaxf(top, __shfl_xor(top, 16, 32));
            top = fmaxf(top, m);
            sw_f32x8 p;
#pragma unroll
            for (int e = 0; e < 8; ++e) p[e] = 2 * e + half < tn && col < kSwGroup ? sw_exp(s[e] - top) : 0.f;
            float sum = p[0];
#pragma unroll
            for (int e = 1; e < 8; ++e) sum += p[e];
            sum += __shfl_xor(sum, 16, 32);
            const float f = sw_exp(m - top);
            l = l * f + sum;
            m = top;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                fe[e] = __shfl(f, 2 * e + half, 32);
                const float oth = __shfl_xor(p[e], 16, 32);
                const uint32_t x0 = float_to_bf16(half ? oth : p[e]), x1 = float_to_bf16(half ? p[e] : oth);
                pb[e] = (x1 << 16) | x0;
            }
        }
        __syncthreads();
        if (tn > 0) {
            // values: lanes of half h stage N tile 2j + h of key col (element (dim d, key c) of staged tile b at
            // (b * 16 + d) * 16 + (c ^ 8 (d & 1))), read back as one dim's 16 keys; o = mma(bf16 p, v, o f)
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
                    o[2 * j + b] = sw_mma(pb, bv, o[2 * j + b] * fe);
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

// Block (h, r), a thread a dim: the parts of head h of row r in part order, w_p = tf_exp(m_p - max m), out =
// bf16(acc / total) kept as fp32.
extern "C" __global__ void tf_fn_sparse_wmma_merge(const float* parts_o, const float* parts_ms, float* out, int heads,
                                                   int dims, int parts) {
    const int h = blockIdx.x, r = blockIdx.y;
    const long long at = (static_cast<long long>(r) * heads + h) * parts;
    float top = -INFINITY;
    for (int p = 0; p < parts; ++p) top = fmaxf(top, parts_ms[(at + p) * 2]);
    for (int d = threadIdx.x; d < dims; d += blockDim.x) {
        float total = 0.f, acc = 0.f;
        for (int p = 0; p < parts; ++p) {
            const float mp = parts_ms[(at + p) * 2];
            const float w = mp == -INFINITY ? 0.f : sw_exp(mp - top);
            total += parts_ms[(at + p) * 2 + 1] * w;
            acc += parts_o[(at + p) * dims + d] * w;
        }
        out[(static_cast<long long>(r) * heads + h) * dims + d] = total > 0.f ? bf16_to_float(float_to_bf16(acc / total)) : 0.f;
    }
}

#endif

}  // namespace
