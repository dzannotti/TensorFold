#pragma once

#include "../ops/common.hpp"
#include "common/arch.hpp"

// CUDA's attn_chunks (zig/kernels/cuda/flashnext_attn.cu) on gfx11 bf16 WMMA for 12 query heads a kv head over a bf16
// cache at head_dim 256: its 512-key chunks, 32-key tiles, rounding points and sum order (sum order: README.md of
// zig/kernels/loom/sparse_attention).

namespace {

#if TF_DEVICE_WMMA_GFX11

typedef __bf16 sc_bf16x16 __attribute__((ext_vector_type(16)));
typedef float sc_f32x8 __attribute__((ext_vector_type(8)));
typedef unsigned sc_u32x8 __attribute__((ext_vector_type(8)));
typedef unsigned sc_u32x4 __attribute__((ext_vector_type(4)));

constexpr int kScGroup = 12, kScDims = 256, kScTile = 32, kScChunk = 512, kScChunks = 5;
constexpr int kScQPitch = kScDims + 8, kScPPitch = kScTile + 8;

__device__ __forceinline__ sc_f32x8 sc_mma(sc_u32x8 a, sc_u32x8 b, sc_f32x8 c) {
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(__builtin_bit_cast(sc_bf16x16, a),
                                                       __builtin_bit_cast(sc_bf16x16, b), c);
}

__device__ __forceinline__ sc_u32x8 sc_frag(const uint16_t* p) {
    const sc_u32x4 lo = *reinterpret_cast<const sc_u32x4*>(p), hi = *reinterpret_cast<const sc_u32x4*>(p + 8);
    return __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7);
}

// CUDA's tf_exp (common/glue_math.cuh): a degree-7 Taylor polynomial after x = j ln2 + f, 2^j in two exact factors.
__device__ __forceinline__ float sc_exp(float x) {
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

// Block (r, g, c), 2 waves: chunk c (keys [512 c, 512 c + 512)) of row r for kv head g, wave w the tile's keys
// [16 w, 16 w + 16) for the scores and dims [128 w, 128 w + 128) for the values; writes the chunk's o, m and l.
__device__ void sparse_chunks(const float* q, const uint16_t* k16, const uint16_t* v16, const int* ids, long long id_stride,
                              const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads,
                              long long cap, float scale) {
    __shared__ __attribute__((aligned(32))) uint16_t qs[16][kScQPitch];
    __shared__ __attribute__((aligned(32))) uint16_t ps[16][kScPPitch];
    __shared__ __attribute__((aligned(32))) uint16_t vt[2][2][16][16];
    __shared__ float red_m[2][16], red_l[4][16];
    const int r = blockIdx.x, g = blockIdx.y, c = blockIdx.z;
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5, col = lane & 15, half = lane >> 4;
    const int n = count[r];
    if (c * kScChunk >= n) return;
    const bool listed = ids != nullptr && sparse != nullptr && sparse[r] != 0;
    const long long head_base = static_cast<long long>(g) * cap * kScDims;
    const float* qrow = q + (static_cast<long long>(r) * heads + g * kScGroup) * kScDims;
    for (int i = t; i < 16 * kScDims; i += 64) {
        const int h = i / kScDims, d = i % kScDims;
        qs[h][d] = h < kScGroup ? float_to_bf16(qrow[h * kScDims + d]) : 0;
    }
    const sc_f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    float m[8], l[8];
    sc_f32x8 o[8];
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        m[e] = -INFINITY;
        l[e] = 0.f;
        o[e] = zero;
    }
    const int start = c * kScChunk, tiles = min(n - start, kScChunk);
    for (int t0 = 0; t0 < tiles; t0 += kScTile) {
        __syncthreads();
        // scores: lane col of wave w takes key 16 w + col of the tile (element e: head 2e + half)
        const int ki = start + t0 + 16 * wave + col;
        const bool valid = ki < n;
        const long long key = valid ? (listed ? ids[r * id_stride + ki] : ki) : 0;
        const uint16_t* krow = k16 + head_base + key * kScDims;
        sc_f32x8 s = zero;
#pragma unroll
        for (int ks = 0; ks < kScDims / 16; ++ks) {
            sc_u32x8 b = sc_frag(krow + ks * 16);
            if (!valid) b = sc_u32x8{0, 0, 0, 0, 0, 0, 0, 0};
            s = sc_mma(sc_frag(&qs[col][ks * 16]), b, s);
        }
        float tm[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            s[e] = valid ? s[e] * scale : -INFINITY;
            float x = s[e];
            x = fmaxf(x, __shfl_xor(x, 1, 32));
            x = fmaxf(x, __shfl_xor(x, 2, 32));
            x = fmaxf(x, __shfl_xor(x, 4, 32));
            x = fmaxf(x, __shfl_xor(x, 8, 32));
            tm[e] = x;
        }
        if (col == 0) {
#pragma unroll
            for (int e = 0; e < 8; ++e) red_m[wave][2 * e + half] = tm[e];
        }
        __syncthreads();
        float next_m[8], alpha[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int h = 2 * e + half;
            const float x = fmaxf(red_m[0][h], red_m[1][h]);
            const bool active = x != -INFINITY;
            next_m[e] = active ? fmaxf(m[e], x) : m[e];
            alpha[e] = active ? (m[e] == -INFINITY ? 0.f : sc_exp(m[e] - next_m[e])) : 1.f;
            const float p = s[e] == -INFINITY ? 0.f : sc_exp(s[e] - next_m[e]);
            ps[h][16 * wave + col] = float_to_bf16(p);
            // an 8-key group's sum as CUDA's warp: pairs, then pairs of pairs, then the two halves
            float a = p + __shfl_xor(p, 1, 32);
            a = a + __shfl_xor(a, 2, 32);
            a = a + __shfl_xor(a, 4, 32);
            if ((col & 7) == 0) red_l[2 * wave + (col >> 3)][h] = a;
        }
        __syncthreads();
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int h = 2 * e + half;
            const float sum = ((red_l[0][h] + red_l[1][h]) + red_l[2][h]) + red_l[3][h];
            l[e] = __builtin_fmaf(l[e], alpha[e], sum);
            m[e] = next_m[e];
        }
#pragma unroll
        for (int i = 0; i < 8; ++i)
#pragma unroll
            for (int e = 0; e < 8; ++e) o[i][e] = o[i][e] * alpha[e];
        // values: per 16-key half kk, lanes of half hh stage key 16 kk + col's dims of N tile 2j + hh, read back by dim
#pragma unroll
        for (int kk = 0; kk < 2; ++kk) {
            const int vi = start + t0 + 16 * kk + col;
            const bool vvalid = vi < n;
            const long long vkey = vvalid ? (listed ? ids[r * id_stride + vi] : vi) : 0;
            const uint16_t* vrow = v16 + head_base + vkey * kScDims + wave * (kScDims / 2) + half * 16;
            const sc_u32x8 pa = sc_frag(&ps[col][16 * kk]);
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                sc_u32x8 w = sc_frag(vrow + 32 * j);
                if (!vvalid) w = sc_u32x8{0, 0, 0, 0, 0, 0, 0, 0};
                uint16_t(*tw)[16] = vt[wave][half];
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    tw[2 * e][col] = static_cast<uint16_t>(w[e]);
                    tw[2 * e + 1][col] = static_cast<uint16_t>(w[e] >> 16);
                }
                __builtin_amdgcn_wave_barrier();
#pragma unroll
                for (int b = 0; b < 2; ++b) o[2 * j + b] = sc_mma(pa, sc_frag(&vt[wave][b][col][0]), o[2 * j + b]);
                __builtin_amdgcn_wave_barrier();
            }
        }
    }
    const long long base = (static_cast<long long>(r) * kScChunks + c) * heads + g * kScGroup;
#pragma unroll
    for (int e = 0; e < 8; ++e) {
        const int h = 2 * e + half;
        if (h >= kScGroup) continue;
#pragma unroll
        for (int i = 0; i < 8; ++i) parts_o[(base + h) * kScDims + wave * (kScDims / 2) + i * 16 + col] = o[i][e];
        if (wave == 0 && col == 0) {
            parts_ms[(base + h) * 2] = m[e];
            parts_ms[(base + h) * 2 + 1] = l[e];
        }
    }
}

extern "C" __global__ void __launch_bounds__(64)
tf_fn_sparse_chunks12_bf16(const float* q, const void* k, const void* v, const int* ids, long long id_stride,
                           const int* count, const int* sparse, float* parts_o, float* parts_ms, int heads, int kv_heads,
                           int dims, long long cap, float scale) {
    if (dims != kScDims || heads != kv_heads * kScGroup) return;
    sparse_chunks(q, static_cast<const uint16_t*>(k), static_cast<const uint16_t*>(v), ids, id_stride, count, sparse,
                  parts_o, parts_ms, heads, cap, scale);
}

// Block (r, h), a thread a dim: CUDA's fn_attn_merge, the row's chunks in order, o / l rounded to bf16 (kept as fp32).
extern "C" __global__ void __launch_bounds__(256)
tf_fn_sparse_chunks_merge(const float* parts_o, const float* parts_ms, const int* count, float* out, int heads) {
    const int r = blockIdx.x, h = blockIdx.y, d = threadIdx.x;
    const int chunks = (count[r] + kScChunk - 1) / kScChunk;
    float m = -INFINITY, l = 0.f, o = 0.f;
    for (int c = 0; c < chunks; ++c) {
        const long long base = (static_cast<long long>(r) * kScChunks + c) * heads + h;
        const float cm = parts_ms[base * 2], cl = parts_ms[base * 2 + 1], co = parts_o[base * kScDims + d];
        const bool active = cl > 0.f;
        const float next = active ? fmaxf(m, cm) : m;
        const float a = active ? (m == -INFINITY ? 0.f : sc_exp(m - next)) : 1.f;
        const float b = active ? sc_exp(cm - next) : 0.f;
        o = __builtin_fmaf(co, b, o * a);
        l = __builtin_fmaf(cl, b, l * a);
        m = next;
    }
    out[(static_cast<long long>(r) * heads + h) * kScDims + d] = bf16_to_float(float_to_bf16(o / l));
}

#endif

}  // namespace
