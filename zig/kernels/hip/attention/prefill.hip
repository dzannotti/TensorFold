// Prefill attention for long prompts: 64 query rows share a 64-key tile; dot2 products and online softmax in fp32.

#include <hip/hip_bfloat16.h>
#include <hip/hip_fp16.h>
#include <hip/hip_runtime.h>

#include "decode/pages.hpp"

namespace {

constexpr int kRows = 64;     // query rows a block
constexpr int kKeys = 64;     // keys a tile
constexpr int kChunk = 64;    // head dimensions a staged key or value chunk
constexpr int kThreads = 256;
static_assert(kKeys == kPageTokens, "a key tile is one page");

// A cache element type's pair for dot2, the rounding into it and the product.
template <typename Cache>
struct Dot;

template <>
struct Dot<__half> {
    using elem = _Float16;
    using pair = _Float16 __attribute__((ext_vector_type(2)));
    static __device__ __forceinline__ float dot(pair a, pair b, float acc) {
        return __builtin_amdgcn_fdot2(a, b, acc, false);
    }
};

template <>
struct Dot<hip_bfloat16> {
    using elem = __bf16;
    using pair = __bf16 __attribute__((ext_vector_type(2)));
    static __device__ __forceinline__ float dot(pair a, pair b, float acc) {
#if TF_DOT2_BF16
        return __builtin_amdgcn_fdot2_f32_bf16(a, b, acc, false);
#else
        // RDNA2 has no bf16 dot2: its caches are fp16, this only keeps the kernel building
        return fmaf(static_cast<float>(a.y), static_cast<float>(b.y), fmaf(static_cast<float>(a.x), static_cast<float>(b.x), acc));
#endif
    }
};

// A 16-byte LDS read as four pairs.
template <typename Pair>
__device__ __forceinline__ void pairs4(const void* at, Pair out[4]) {
    const uint4 w = *reinterpret_cast<const uint4*>(at);
    out[0] = __builtin_bit_cast(Pair, w.x);
    out[1] = __builtin_bit_cast(Pair, w.y);
    out[2] = __builtin_bit_cast(Pair, w.z);
    out[3] = __builtin_bit_cast(Pair, w.w);
}

// q rounds to the cache type once and stays in LDS; with PAGED, k and v are a layer's pools and a 64-key tile a page.
template <typename Cache, bool PAGED>
__device__ void fa_dot2(const float* __restrict__ q, const Cache* __restrict__ k, const Cache* __restrict__ v,
                        float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d, float scale,
                        int q_pos0, long long k_sb, long long k_sh, long long k_ss, long long v_sb, long long v_sh,
                        long long v_ss, const unsigned* table, int count) {
    constexpr int kQChunks = 256 / 8;      // 16-byte chunks a q row
    constexpr int kKChunks = kChunk / 8;   // a staged key row's
    constexpr int kPLd = kRows + 4;        // a key pair's probabilities, row-indexed
    __shared__ uint4 sQ[kRows * kQChunks];
    __shared__ uint4 sKV[kKeys * kKChunks];   // a key chunk (key, 8 dims) or a value chunk (key pair, 4 dims)
    using T = Dot<Cache>;
    using elem = typename T::elem;
    using pair_t = typename T::pair;
    __shared__ alignas(16) pair_t sH[kKeys / 2 * kPLd];
    __shared__ float sA[kRows], sL[kRows];

    const int q0 = blockIdx.x * kRows;
    const int head = blockIdx.y;
    const int batch = blockIdx.z;
    if (q0 >= qlen) return;
    const int tid = threadIdx.x;
    int q_hi = qlen - q0;
    if (q_hi > kRows) q_hi = kRows;
    int max_vis = q_pos0 + q0 + q_hi;
    if (max_vis > span) max_vis = span;
    const int kv = head / (heads / kv_heads);
    const float* qrows = q + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    float* orows = out + ((static_cast<long long>(batch) * heads + head) * qlen + q0) * d;
    const Cache* kbase = PAGED ? k : k + static_cast<long long>(batch) * k_sb + static_cast<long long>(kv) * k_sh;
    const Cache* vbase = PAGED ? v : v + static_cast<long long>(batch) * v_sb + static_cast<long long>(kv) * v_sh;
    const long long kstride = PAGED ? d : k_ss;
    const long long vstride = PAGED ? d : v_ss;
    const int chunks = d / 8;

    for (int e = tid; e < kRows * chunks; e += kThreads) {
        const int row = e / chunks, c = e % chunks;
        elem h[8];
#pragma unroll
        for (int m = 0; m < 8; ++m)
            h[m] = static_cast<elem>(row < q_hi ? qrows[static_cast<long long>(row) * d + 8 * c + m] : 0.f);
        sQ[row * kQChunks + (c ^ (row & 7))] = __builtin_bit_cast(uint4, h);
    }

    // scores: rows sr + 16i against keys 4sk .. 4sk + 3; PV: rows pr .. pr + 3 at dims 4pc .. 4pc + 3 of each chunk
    const int sr = tid / 16, sk = tid % 16;
    const int pr = (tid / 16) * 4, pc = tid % 16;
    float m_run[4], l_run[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) m_run[i] = -INFINITY, l_run[i] = 0.f;
    float o[4][16];
#pragma unroll
    for (int i = 0; i < 4; ++i)
#pragma unroll
        for (int m = 0; m < 16; ++m) o[i][m] = 0.f;

    for (int k0 = 0; k0 < max_vis; k0 += kKeys) {
        // the tile's first key row in the keys' and the values' storage
        const Cache* ktile = PAGED ? kbase + page_row(table[k0 / kPageTokens], count, kv, 0, d) : kbase + static_cast<long long>(k0) * k_ss;
        const Cache* vtile = PAGED ? vbase + page_row(table[k0 / kPageTokens], count, kv, 0, d) : vbase + static_cast<long long>(k0) * v_ss;
        float acc[4][4];
#pragma unroll
        for (int i = 0; i < 4; ++i)
#pragma unroll
            for (int j = 0; j < 4; ++j) acc[i][j] = 0.f;
        for (int c0 = 0; c0 < chunks; c0 += kKChunks) {
            uint4 staged[2];
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int e = tid + t * kThreads, key = e / kKChunks, c = e % kKChunks;
                staged[t] = k0 + key < span
                                ? *reinterpret_cast<const uint4*>(ktile + static_cast<long long>(key) * kstride + 8 * (c0 + c))
                                : uint4{0, 0, 0, 0};
            }
            __syncthreads();
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                const int e = tid + t * kThreads, key = e / kKChunks, c = e % kKChunks;
                sKV[key * kKChunks + (c ^ ((key >> 2) & 7))] = staged[t];
            }
            __syncthreads();
#pragma unroll 2
            for (int c = 0; c < kKChunks; ++c) {
                pair_t qv[4][4], kf[4][4];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    const int row = sr + 16 * i;
                    pairs4(&sQ[row * kQChunks + ((c0 + c) ^ (row & 7))], qv[i]);
                }
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    const int key = 4 * sk + j;
                    pairs4(&sKV[key * kKChunks + (c ^ ((key >> 2) & 7))], kf[j]);
                }
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int j = 0; j < 4; ++j)
#pragma unroll
                        for (int m = 0; m < 4; ++m) acc[i][j] = T::dot(qv[i][m], kf[j][m], acc[i][j]);
            }
        }
        // the online softmax in registers: a row's 64 keys sit on its 16 lanes
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = sr + 16 * i;
            int vis = q_pos0 + q0 + row + 1;
            if (vis > span) vis = span;
            float tile_max = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[i][j] = (row < q_hi && k0 + 4 * sk + j < vis) ? acc[i][j] * scale : -INFINITY;
                tile_max = fmaxf(tile_max, acc[i][j]);
            }
#pragma unroll
            for (int lane = 8; lane > 0; lane >>= 1) tile_max = fmaxf(tile_max, __shfl_xor(tile_max, lane, 16));
            const float next_m = fmaxf(m_run[i], tile_max);
            const float alpha = next_m == -INFINITY ? 1.f : expf(m_run[i] - next_m);
            float p[4], sum = 0.f;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                p[j] = next_m == -INFINITY ? 0.f : expf(acc[i][j] - next_m);
                sum += p[j];
            }
#pragma unroll
            for (int lane = 8; lane > 0; lane >>= 1) sum += __shfl_xor(sum, lane, 16);
            l_run[i] = alpha * l_run[i] + sum;
            m_run[i] = next_m;
            sH[(2 * sk) * kPLd + row] = pair_t{static_cast<elem>(p[0]), static_cast<elem>(p[1])};
            sH[(2 * sk + 1) * kPLd + row] = pair_t{static_cast<elem>(p[2]), static_cast<elem>(p[3])};
            if (sk == 0) sA[row] = alpha;
        }
#pragma unroll
        for (int at = 0; at < 256 / kChunk; ++at) {
            if (at * kKChunks >= chunks) break;
            const int c0 = at * kKChunks;
            // a value chunk: key pair `pair`, dims 8g .. 8g + 7 interleaved as (key, key + 1) fp16 pairs
            const int pair = tid / kKChunks, g = tid % kKChunks;
            const int key = k0 + 2 * pair;
            const uint4 a = key < span ? *reinterpret_cast<const uint4*>(vtile + static_cast<long long>(2 * pair) * vstride + 8 * (c0 + g))
                                       : uint4{0, 0, 0, 0};
            const uint4 b = key + 1 < span
                                ? *reinterpret_cast<const uint4*>(vtile + static_cast<long long>(2 * pair + 1) * vstride + 8 * (c0 + g))
                                : uint4{0, 0, 0, 0};
            __syncthreads();
            const elem* ha = reinterpret_cast<const elem*>(&a);
            const elem* hb = reinterpret_cast<const elem*>(&b);
            pair_t lo[4], hi[4];
#pragma unroll
            for (int m = 0; m < 4; ++m) lo[m] = pair_t{ha[m], hb[m]}, hi[m] = pair_t{ha[4 + m], hb[4 + m]};
            sKV[pair * 16 + 2 * g] = __builtin_bit_cast(uint4, lo);
            sKV[pair * 16 + 2 * g + 1] = __builtin_bit_cast(uint4, hi);
            __syncthreads();
            float pv[4][4];
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) pv[i][m] = 0.f;
#pragma unroll 4
            for (int pp = 0; pp < kKeys / 2; ++pp) {
                pair_t pr4[4], vv[4];
                pairs4(sH + pp * kPLd + pr, pr4);
                pairs4(&sKV[pp * 16 + pc], vv);
#pragma unroll
                for (int i = 0; i < 4; ++i)
#pragma unroll
                    for (int m = 0; m < 4; ++m) pv[i][m] = T::dot(pr4[i], vv[m], pv[i][m]);
            }
#pragma unroll
            for (int i = 0; i < 4; ++i)
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    float& acc_o = o[i][4 * at + m];
                    acc_o = fmaf(sA[pr + i], acc_o, pv[i][m]);
                }
        }
    }
    if (sk == 0)
#pragma unroll
        for (int i = 0; i < 4; ++i) sL[sr + 16 * i] = l_run[i];
    __syncthreads();
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int row = pr + i;
        if (row >= q_hi) continue;
        const float inv = sL[row] == 0.f ? 0.f : 1.f / sL[row];
#pragma unroll
        for (int at = 0; at < 256 / kChunk; ++at) {
            if (at * kKChunks >= chunks) break;
#pragma unroll
            for (int m = 0; m < 4; ++m)
                orows[static_cast<long long>(row) * d + 64 * at + 4 * pc + m] = o[i][4 * at + m] * inv;
        }
    }
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
               int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh, long long k_ss,
               long long v_sb, long long v_sh, long long v_ss) {
    fa_dot2<__half, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb, v_sh, v_ss, nullptr, 0);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_wide_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                int heads, int kv_heads, int d, float scale, int q_pos0, long long k_sb, long long k_sh,
                long long k_ss, long long v_sb, long long v_sh, long long v_ss) {
    fa_dot2<hip_bfloat16, false>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, k_sb, k_sh, k_ss, v_sb,
                                 v_sh, v_ss, nullptr, 0);
}

// The same tile over a stream's pages: `k` and `v` are a layer's pools, `table` the stream's page ids.
extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_paged_f16(const float* q, const __half* k, const __half* v, float* out, int qlen, int span, int heads,
                int kv_heads, int d, float scale, int q_pos0, const unsigned* table, int count) {
    fa_dot2<__half, true>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, 0, 0, 0, 0, 0, 0, table, count);
}

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_paged_bf16(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                 int heads, int kv_heads, int d, float scale, int q_pos0, const unsigned* table, int count) {
    fa_dot2<hip_bfloat16, true>(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, 0, 0, 0, 0, 0, 0, table, count);
}

// ---- the paged bf16 tile on gfx11's matrix cores: fa_dot2's bits (a WMMA chain is the dot2 chain in k order) ----

#include "common/wmma.hpp"

namespace {

// __shfl_xor(x, ln, 16) on DPP row_xmask (ln a power of two below 16, unrolled to a constant).
__device__ __forceinline__ float xmask16(float x, int ln) {
    const int u = __float_as_int(x);
    switch (ln) {
        case 8: return __int_as_float(__builtin_amdgcn_update_dpp(0, u, 0x168, 0xf, 0xf, false));
        case 4: return __int_as_float(__builtin_amdgcn_update_dpp(0, u, 0x164, 0xf, 0xf, false));
        case 2: return __int_as_float(__builtin_amdgcn_update_dpp(0, u, 0x162, 0xf, 0xf, false));
        default: return __int_as_float(__builtin_amdgcn_update_dpp(0, u, 0x161, 0xf, 0xf, false));
    }
}

// fa_dot2<hip_bfloat16, true> with q.k and p.v on WMMA, k order kept; the softmax is fa_dot2's, on scores from LDS.
// Keys and values come 128 dims a chunk, each fetched a chunk ahead; scores, then values, reuse the keys' LDS.
__device__ void fa_wmma(const float* __restrict__ q, const hip_bfloat16* __restrict__ k, const hip_bfloat16* __restrict__ v,
                        float* __restrict__ out, int qlen, int span, int heads, int kv_heads, int d, float scale, int q_pos0,
                        const unsigned* table, int count) {
#if TF_DEVICE_WMMA_GFX11
    using tf::rocm::bf16x16;
    using tf::rocm::f32x8;
    using elem = __bf16;
    constexpr int kQChunks = 256 / 8;
    constexpr int kCDims = 128;           // dims a key or value chunk
    constexpr int kC = kCDims / 8;        // 16-byte pieces of a staged key row
    constexpr int kSLd = kKeys;           // scores, fp32, row-major
    constexpr int kPLd = kKeys + 8;       // probabilities, bf16, row-major
    constexpr int kVLd = kKeys / 2 + 4;   // a value chunk transposed: (dim, key pair) words
    __shared__ uint4 sQ[kRows * kQChunks];
    __shared__ uint4 sKV[kCDims * kVLd / 4];  // a key chunk (key, 16 pieces swizzled), the scores, or a value chunk
    __shared__ alignas(16) elem sP[kRows * kPLd];
    __shared__ float sA[kRows], sL[kRows];
    float* sS = reinterpret_cast<float*>(sKV);
    uint32_t* sV = reinterpret_cast<uint32_t*>(sKV);

    const int q0 = blockIdx.x * kRows;
    const int head = blockIdx.y;
    if (q0 >= qlen) return;
    const int tid = threadIdx.x;
    int q_hi = qlen - q0;
    if (q_hi > kRows) q_hi = kRows;
    int max_vis = q_pos0 + q0 + q_hi;
    if (max_vis > span) max_vis = span;
    const int kv = head / (heads / kv_heads);
    const float* qrows = q + (static_cast<long long>(head) * qlen + q0) * d;
    float* orows = out + (static_cast<long long>(head) * qlen + q0) * d;
    const int chunks = d / 8;

    for (int e = tid; e < kRows * chunks; e += kThreads) {
        const int row = e / chunks, c = e % chunks;
        elem h[8];
#pragma unroll
        for (int m = 0; m < 8; ++m)
            h[m] = static_cast<elem>(row < q_hi ? qrows[static_cast<long long>(row) * d + 8 * c + m] : 0.f);
        sQ[row * kQChunks + (c ^ (row & 7))] = __builtin_bit_cast(uint4, h);
    }

    // wave (rw, kw): rows 16 rw.., keys 32 kw.. (scores) or dims 64 kw.. of a chunk (values); the softmax keeps
    // fa_dot2's (sr, sk) lanes
    const int lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int rw = wave & 3, kw = wave >> 2;
    const int rl = lane & 15, rh = lane >> 4;
    const int sr = tid / 16, sk = tid % 16;
    float m_run[4], l_run[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) m_run[i] = -INFINITY, l_run[i] = 0.f;
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    f32x8 o[2][4];
#pragma unroll
    for (int at = 0; at < 2; ++at)
#pragma unroll
        for (int j = 0; j < 4; ++j) o[at][j] = zero;
    auto frag = [](const uint4& lo, const uint4& hi) {
        const tf::rocm::u32x4 a = __builtin_bit_cast(tf::rocm::u32x4, lo), b = __builtin_bit_cast(tf::rocm::u32x4, hi);
        return __builtin_bit_cast(bf16x16, __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7));
    };

    // a tile's chunks in order: keys (idx < nc), then values; each fetched a chunk ahead into R
    const int nc = (d + kCDims - 1) / kCDims;
    const int vp = tid % 32, vg = tid / 32;  // a value chunk's thread: key pair, 16 dims
    uint4 R[4];
    auto fetch = [&](int kk0, int idx) {
        const long long tile = page_row(table[kk0 / kPageTokens], count, kv, 0, d);
        if (idx < nc) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int e = tid + t * kThreads, key = e / kC, c = idx * kC + e % kC;
                R[t] = kk0 + key < span && c < chunks ? *reinterpret_cast<const uint4*>(k + tile + static_cast<long long>(key) * d + 8 * c)
                                                      : uint4{0, 0, 0, 0};
            }
        } else {
            const int dim = (idx - nc) * kCDims + 16 * vg;
            const hip_bfloat16* vt = v + tile + dim;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int key = 2 * vp + (t >> 1);
                R[t] = kk0 + key < span && dim < d ? *reinterpret_cast<const uint4*>(vt + static_cast<long long>(key) * d + 8 * (t & 1))
                                                   : uint4{0, 0, 0, 0};
            }
        }
    };
    auto fetch_after = [&](int kk0, int idx) {
        if (idx + 1 < 2 * nc) fetch(kk0, idx + 1);
        else if (kk0 + kKeys < max_vis) fetch(kk0 + kKeys, 0);
    };
    fetch(0, 0);

    for (int k0 = 0; k0 < max_vis; k0 += kKeys) {
        f32x8 s[2] = {zero, zero};
        for (int ci = 0; ci < nc; ++ci) {
            __syncthreads();
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int e = tid + t * kThreads, key = e / kC, c = e % kC;
                sKV[key * kC + (c ^ (key & 15))] = R[t];
            }
            __syncthreads();
            fetch_after(k0, ci);
            const int steps = (chunks - ci * kC) / 2 < kC / 2 ? (chunks - ci * kC) / 2 : kC / 2;
#pragma unroll
            for (int t = 0; t < kC / 2; ++t) {
                if (t >= steps) break;
                const int row = 16 * rw + rl;
                const int qc = ci * kC + 2 * t;
                const bf16x16 af = frag(sQ[row * kQChunks + (qc ^ (row & 7))], sQ[row * kQChunks + ((qc + 1) ^ (row & 7))]);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int key = 32 * kw + 16 * j + rl;
                    const bf16x16 bf = frag(sKV[key * kC + ((2 * t) ^ (key & 15))], sKV[key * kC + ((2 * t + 1) ^ (key & 15))]);
                    s[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af, bf, s[j]);
                }
            }
        }
        __syncthreads();
#pragma unroll
        for (int j = 0; j < 2; ++j)
#pragma unroll
            for (int i = 0; i < 8; ++i) sS[(16 * rw + 2 * i + rh) * kSLd + 32 * kw + 16 * j + rl] = s[j][i];
        __syncthreads();
        // fa_dot2's online softmax on the same scores: a row's 64 keys on its 16 lanes
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = sr + 16 * i;
            const float4 sc4 = *reinterpret_cast<const float4*>(&sS[row * kSLd + 4 * sk]);
            float acc[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
            int vis = q_pos0 + q0 + row + 1;
            if (vis > span) vis = span;
            float tile_max = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[j] = (row < q_hi && k0 + 4 * sk + j < vis) ? acc[j] * scale : -INFINITY;
                tile_max = fmaxf(tile_max, acc[j]);
            }
#pragma unroll
            for (int ln = 8; ln > 0; ln >>= 1) tile_max = fmaxf(tile_max, xmask16(tile_max, ln));
            const float next_m = fmaxf(m_run[i], tile_max);
            const float alpha = next_m == -INFINITY ? 1.f : expf(m_run[i] - next_m);
            float sum = 0.f;
            elem h[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float p = next_m == -INFINITY ? 0.f : expf(acc[j] - next_m);
                sum += p;
                h[j] = static_cast<elem>(p);
            }
#pragma unroll
            for (int ln = 8; ln > 0; ln >>= 1) sum += xmask16(sum, ln);
            l_run[i] = alpha * l_run[i] + sum;
            m_run[i] = next_m;
            *reinterpret_cast<uint2*>(&sP[row * kPLd + 4 * sk]) = __builtin_bit_cast(uint2, h);
            if (sk == 0) sA[row] = alpha;
        }
        float alpha_r[8];
#pragma unroll
        for (int at = 0; at < 2; ++at) {
            if (at >= nc) break;
            // a value chunk: thread (key pair p, dims 16 g..) writes the (key 2 p, key 2 p + 1) word of each dim
            __syncthreads();
            if (at == 0) {
#pragma unroll
                for (int i = 0; i < 8; ++i) alpha_r[i] = sA[16 * rw + 2 * i + rh];
            }
#pragma unroll
            for (int hlf = 0; hlf < 2; ++hlf) {
                const uint4 va = R[hlf], vb = R[2 + hlf];
                const uint32_t wa[4] = {va.x, va.y, va.z, va.w};
                const uint32_t wb[4] = {vb.x, vb.y, vb.z, vb.w};
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int dim = 16 * vg + 8 * hlf + 2 * m;
                    sV[dim * kVLd + vp] = (wa[m] & 0xffffu) | (wb[m] << 16);
                    sV[(dim + 1) * kVLd + vp] = (wa[m] >> 16) | (wb[m] & 0xffff0000u);
                }
            }
            __syncthreads();
            fetch_after(k0, nc + at);
            bf16x16 pf[4];  // the probabilities, keys 16 t.. of rows 16 rw..
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint4* src = reinterpret_cast<const uint4*>(&sP[(16 * rw + rl) * kPLd + 16 * t]);
                pf[t] = frag(src[0], src[1]);
            }
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                if (at * kCDims + 64 * kw + 16 * j >= d) break;
                const int dim = 64 * kw + 16 * j + rl;
                f32x8 pv = zero;
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    const uint4* src = reinterpret_cast<const uint4*>(&sV[dim * kVLd + 8 * t]);
                    pv = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf[t], frag(src[0], src[1]), pv);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) o[at][j][i] = fmaf(alpha_r[i], o[at][j][i], pv[i]);
            }
        }
    }
    if (sk == 0)
#pragma unroll
        for (int i = 0; i < 4; ++i) sL[sr + 16 * i] = l_run[i];
    __syncthreads();
    // the lane again from mbcnt: nothing lane-indexed stays live through the tiles for the stores
    const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_lo(~0u, 0u));
    const int erl = ln & 15, erh = ln >> 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 16 * rw + 2 * i + erh;
        if (row >= q_hi) continue;
        const float inv = sL[row] == 0.f ? 0.f : 1.f / sL[row];
#pragma unroll
        for (int at = 0; at < 2; ++at) {
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const int dim = at * kCDims + 64 * kw + 16 * j;
                if (dim < d) orows[static_cast<long long>(row) * d + dim + erl] = o[at][j][i] * inv;
            }
        }
    }
#else
    __builtin_trap();
#endif
}

}  // namespace

extern "C" __global__ void __launch_bounds__(kThreads, 1)
tf_fa_paged_wmma(const float* q, const hip_bfloat16* k, const hip_bfloat16* v, float* out, int qlen, int span,
                 int heads, int kv_heads, int d, float scale, int q_pos0, const unsigned* table, int count) {
    fa_wmma(q, k, v, out, qlen, span, heads, kv_heads, d, scale, q_pos0, table, count);
}
