#pragma once

#include "../common/wmma.hpp"
#include "sparse.hpp"

// A prompt's dense rows (each row's first count[r] keys) on gfx11's matrix cores: a block is 16 rows times 4 query
// heads of one kv head, 64 keys a tile, the online softmax in fp32 per 64-key tile from key 0. q.k and p.v are WMMA
// chains over BF16 with q split into NQ BF16 planes and p into NP (q = hi + mid + lo, each the top bits of what is
// left): three planes carry an fp32's 24 bits exactly, so the products are as exact as the fp32 kernel's (the keys and
// values are BF16 already). A row's bits depend on its own q and keys alone: never on the rows beside it, the row count or
// where a prompt is cut.

namespace {

constexpr int kApRows = 64;    // query rows a block: 16 prompt rows x 4 heads
constexpr int kApBand = 16;    // prompt rows a block
constexpr int kApKeys = 64;    // keys a tile
constexpr int kApDims = 256;   // head dims (the kernel's only width)
constexpr int kApThreads = 256;

// __shfl_xor(x, ln, 16) on DPP row_xmask (ln a power of two below 16, unrolled to a constant).
__device__ __forceinline__ float ap_xmask16(float x, int ln) {
    const int u = __float_as_int(x);
    switch (ln) {
        case 8: return __int_as_float(__builtin_amdgcn_update_dpp(0, u, 0x168, 0xf, 0xf, false));
        case 4: return __int_as_float(__builtin_amdgcn_update_dpp(0, u, 0x164, 0xf, 0xf, false));
        case 2: return __int_as_float(__builtin_amdgcn_update_dpp(0, u, 0x162, 0xf, 0xf, false));
        default: return __int_as_float(__builtin_amdgcn_update_dpp(0, u, 0x161, 0xf, 0xf, false));
    }
}

// A raw buffer over `bytes` bytes from `p`: a load is a scalar base plus a 32-bit lane offset, and past the end reads 0.
__device__ __forceinline__ __amdgpu_buffer_rsrc_t ap_rsrc(const void* p, uint32_t bytes) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, static_cast<int>(bytes), 0x31004000);
}

__device__ __forceinline__ uint4 ap_load(__amdgpu_buffer_rsrc_t r, uint32_t at) {
    return __builtin_bit_cast(uint4, __builtin_amdgcn_raw_buffer_load_b128(r, at, 0, 0));
}

// x's 16 floats as N BF16 planes: plane p the top 16 bits of what planes 0 .. p - 1 leave (each difference is exact,
// so three planes hold an fp32's 24 significant bits exactly; one plane is x truncated to BF16).
template <int N>
__device__ __forceinline__ void ap_split(const float (&x)[16], tf::rocm::bf16x16 (&out)[N]) {
    tf::rocm::u32x8 w[N];
#pragma unroll
    for (int e = 0; e < 16; e += 2) {
        float r0 = x[e], r1 = x[e + 1];
#pragma unroll
        for (int p = 0; p < N; ++p) {
            const uint32_t h0 = __float_as_uint(r0) & 0xffff0000u, h1 = __float_as_uint(r1) & 0xffff0000u;
            w[p][e / 2] = (h0 >> 16) | h1;
            r0 -= __uint_as_float(h0);
            r1 -= __uint_as_float(h1);
        }
    }
#pragma unroll
    for (int p = 0; p < N; ++p) out[p] = __builtin_bit_cast(tf::rocm::bf16x16, w[p]);
}

// p as N BF16 planes, as ap_split.
template <int N>
__device__ __forceinline__ void ap_split1(float p, __bf16 (&out)[N]) {
#pragma unroll
    for (int i = 0; i < N; ++i) {
        const uint32_t h = __float_as_uint(p) & 0xffff0000u;
        out[i] = __builtin_bit_cast(__bf16, static_cast<unsigned short>(h >> 16));
        p -= __uint_as_float(h);
    }
}

// Block (band from the last, quad): rows 16 band.., heads 4 quad..; query row i is head 4 quad + i / 16, row i % 16.
template <int NQ, int NP>
__device__ void attn_prompt(const float* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
                            const int* __restrict__ count, float* __restrict__ out, int rows, int heads, int kv_heads,
                            long long cap, float scale) {
#if TF_DEVICE_WMMA_GFX11
    using tf::rocm::bf16x16;
    using tf::rocm::f32x8;
    using elem = __bf16;
    constexpr int d = kApDims;
    constexpr int kCDims = 128;           // dims a key or value chunk
    constexpr int kC = kCDims / 8;        // 16-byte pieces of a staged key row
    constexpr int nc = d / kCDims;
    constexpr int kSLd = kApKeys;         // scores, fp32, row-major
    constexpr int kPLd = kApKeys + 8;     // probabilities, bf16, row-major
    constexpr int kVLd = kApKeys / 2 + 4; // a value chunk transposed: (dim, key pair) words
    __shared__ uint4 sKV[kCDims * kVLd / 4];  // a key chunk (key, 16 pieces swizzled), the scores, or a value chunk
    __shared__ alignas(16) elem sP[NP][kApRows * kPLd];
    __shared__ float sA[kApRows], sL[kApRows];
    float* sS = reinterpret_cast<float*>(sKV);
    uint32_t* sV = reinterpret_cast<uint32_t*>(sKV);

    const int band = gridDim.x - 1 - blockIdx.x;  // the longest rows first
    const int quad = blockIdx.y;
    const int r0 = band * kApBand;
    const int nr = rows - r0 < kApBand ? rows - r0 : kApBand;
    const int g = 4 * quad / (heads / kv_heads);
    const int tid = threadIdx.x;
    const int max_vis = count[r0 + nr - 1];
    // the kv head's first max_vis keys and values: past them, zeros
    const __amdgpu_buffer_rsrc_t kbuf = ap_rsrc(k + static_cast<long long>(g) * cap * d, static_cast<uint32_t>(max_vis) * d * 2);
    const __amdgpu_buffer_rsrc_t vbuf = ap_rsrc(v + static_cast<long long>(g) * cap * d, static_cast<uint32_t>(max_vis) * d * 2);

    // wave (rw, kw): query rows 16 rw.. (head 4 quad + rw), keys 32 kw.. (scores) or dims 64 kw.. of a chunk (values);
    // the softmax: lanes (sr, sk), query rows sr + 16 i (prompt row sr of each head), keys 4 sk..
    const int lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int rw = wave & 3, kw = wave >> 2;
    const int rl = lane & 15;
    const int sr = tid / 16, sk = tid % 16;
    const int vis = sr < nr ? count[r0 + sr] : 0;
    // each band row's keys (0 past the rows), and their least and most
    __shared__ int sVis[kApBand];
    if (tid < kApBand) sVis[tid] = tid < nr ? count[r0 + tid] : 0;
    int vmin = max_vis, vmax = 0;
    for (int j = 0; j < nr; ++j) vmin = min(vmin, count[r0 + j]), vmax = max(vmax, count[r0 + j]);
    const int my_vis = rl < nr ? count[r0 + rl] : 0;  // the A row this lane holds
    const bf16x16 zero16 = __builtin_bit_cast(bf16x16, tf::rocm::u32x8{0, 0, 0, 0, 0, 0, 0, 0});
    // the A row this lane holds in the scores: prompt row rl of head 4 quad + rw, fp32 from memory (zero past the rows)
    // (the band's rows alone are in the buffer: past them, zeros)
    const __amdgpu_buffer_rsrc_t qbuf = ap_rsrc(q + static_cast<long long>(r0) * heads * d, static_cast<uint32_t>(nr * heads * d * 4));
    const uint32_t qrow = static_cast<uint32_t>((rl * heads + 4 * quad + rw) * d * 4);
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
    const int vp = tid % 32, vg = tid / 32;  // a value chunk's thread: key pair, 16 dims
    uint4 R[4];
    auto fetch = [&](int kk0, int idx) {
        if (idx < nc) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int e = tid + t * kApThreads, key = e / kC, c = idx * kC + e % kC;
                R[t] = ap_load(kbuf, static_cast<uint32_t>(((kk0 + key) * d + 8 * c) * 2));
            }
        } else {
            const int dim = (idx - nc) * kCDims + 16 * vg;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int key = 2 * vp + (t >> 1);
                R[t] = ap_load(vbuf, static_cast<uint32_t>(((kk0 + key) * d + dim + 8 * (t & 1)) * 2));
            }
        }
    };
    auto fetch_after = [&](int kk0, int idx) {
        if (idx + 1 < 2 * nc) fetch(kk0, idx + 1);
        else if (kk0 + kApKeys < max_vis) fetch(kk0 + kApKeys, 0);
    };
    fetch(0, 0);

    for (int k0 = 0; k0 < max_vis; k0 += kApKeys) {
        f32x8 s[2] = {zero, zero};
        for (int ci = 0; ci < nc; ++ci) {
            __syncthreads();
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int e = tid + t * kApThreads, key = e / kC, c = e % kC;
                sKV[key * kC + (c ^ (key & 15))] = R[t];
            }
            __syncthreads();
            fetch_after(k0, ci);
#pragma unroll 2
            for (int t = 0; t < kC / 2; ++t) {
                float x[16];
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const float4 f = __builtin_bit_cast(float4, ap_load(qbuf, qrow + (ci * kCDims + 16 * t + 4 * e) * 4));
                    x[4 * e] = f.x, x[4 * e + 1] = f.y, x[4 * e + 2] = f.z, x[4 * e + 3] = f.w;
                }
                bf16x16 af[NQ];
                ap_split<NQ>(x, af);
#pragma unroll
                for (int j = 0; j < 2; ++j) {
                    const int key = 32 * kw + 16 * j + rl;
                    const bf16x16 bf = frag(sKV[key * kC + ((2 * t) ^ (key & 15))], sKV[key * kC + ((2 * t + 1) ^ (key & 15))]);
#pragma unroll
                    for (int p = 0; p < NQ; ++p) s[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[p], bf, s[j]);
                }
            }
        }
        __syncthreads();
        {
            const int rh = lane >> 4;
#pragma unroll
            for (int j = 0; j < 2; ++j)
#pragma unroll
                for (int i = 0; i < 8; ++i) sS[(16 * rw + 2 * i + rh) * kSLd + 32 * kw + 16 * j + rl] = s[j][i];
        }
        __syncthreads();
        // the online softmax: a query row's 64 keys on its 16 lanes
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const int row = sr + 16 * i;
            const float4 sc4 = *reinterpret_cast<const float4*>(&sS[row * kSLd + 4 * sk]);
            float acc[4] = {sc4.x, sc4.y, sc4.z, sc4.w};
            float tile_max = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[j] = k0 + 4 * sk + j < vis ? acc[j] * scale : -INFINITY;
                tile_max = fmaxf(tile_max, acc[j]);
            }
#pragma unroll
            for (int ln = 8; ln > 0; ln >>= 1) tile_max = fmaxf(tile_max, ap_xmask16(tile_max, ln));
            const float next_m = fmaxf(m_run[i], tile_max);
            const float alpha = next_m == -INFINITY ? 1.f : expf(m_run[i] - next_m);
            float sum = 0.f;
            elem hp[NP][4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float p = next_m == -INFINITY ? 0.f : expf(acc[j] - next_m);
                sum += p;
                elem pp[NP];
                ap_split1<NP>(p, pp);
#pragma unroll
                for (int pl = 0; pl < NP; ++pl) hp[pl][j] = pp[pl];
            }
#pragma unroll
            for (int ln = 8; ln > 0; ln >>= 1) sum += ap_xmask16(sum, ln);
            l_run[i] = alpha * l_run[i] + sum;
            m_run[i] = next_m;
#pragma unroll
            for (int pl = 0; pl < NP; ++pl) *reinterpret_cast<uint2*>(&sP[pl][row * kPLd + 4 * sk]) = __builtin_bit_cast(uint2, hp[pl]);
            if (sk == 0) sA[row] = alpha;
        }
        float alpha_r[8];
#pragma unroll
        for (int at = 0; at < nc; ++at) {
            // a value chunk: thread (key pair p, dims 16 g..) writes the (key 2 p, key 2 p + 1) word of each dim
            __syncthreads();
            if (at == 0) {
                const int rh = lane >> 4;
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
                    const int dm = 16 * vg + 8 * hlf + 2 * m;
                    sV[dm * kVLd + vp] = (wa[m] & 0xffffu) | (wb[m] << 16);
                    sV[(dm + 1) * kVLd + vp] = (wa[m] >> 16) | (wb[m] & 0xffff0000u);
                }
            }
            __syncthreads();
            fetch_after(k0, nc + at);
            // the probabilities' planes, keys 16 t.. of query rows 16 rw.., times the values: dims 16 j.. of the chunk
            auto planes = [&](int t, bf16x16 (&pf)[NP]) {
#pragma unroll
                for (int pl = 0; pl < NP; ++pl) {
                    const uint4* src = reinterpret_cast<const uint4*>(&sP[pl][(16 * rw + rl) * kPLd + 16 * t]);
                    pf[pl] = frag(src[0], src[1]);
                }
            };
            auto values = [&](int j, int t) {
                const uint4* src = reinterpret_cast<const uint4*>(&sV[(64 * kw + 16 * j + rl) * kVLd + 8 * t]);
                return frag(src[0], src[1]);
            };
            if (vmin >= k0 + kApKeys || vmax <= k0) {
                f32x8 pv[4] = {zero, zero, zero, zero};
#pragma unroll
                for (int t = 0; t < 4; ++t) {
                    bf16x16 pf[NP];
                    planes(t, pf);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const bf16x16 vf = values(j, t);
#pragma unroll
                        for (int pl = 0; pl < NP; ++pl) pv[j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(pf[pl], vf, pv[j]);
                    }
                }
#pragma unroll
                for (int j = 0; j < 4; ++j)
#pragma unroll
                    for (int i = 0; i < 8; ++i) o[at][j][i] = fmaf(alpha_r[i], o[at][j][i], pv[j][i]);
            } else {
                // a tile some rows see only in part: in a step a row sees in part, that row's own product over its
                // values past its keys zeroed (a WMMA's sum depends on the values its zero probabilities meet; an
                // all-zero row adds exactly nothing), so a row's bits never depend on the rows beside it
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    f32x8 pv = zero;
#pragma unroll 1
                    for (int t = 0; t < 4; ++t) {
                        const int kb = k0 + 16 * t;
                        const bool part = my_vis > kb && my_vis < kb + 16;
                        bf16x16 pf[NP];
                        planes(t, pf);
                        const bf16x16 vf = values(j, t);
#pragma unroll
                        for (int pl = 0; pl < NP; ++pl) pv = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(part ? zero16 : pf[pl], vf, pv);
#pragma unroll 1
                        for (int jr = 0; jr < kApBand; ++jr) {
                            const int n = sVis[jr] - kb;
                            if (n <= 0 || n >= 16) continue;
                            tf::rocm::u32x8 w = __builtin_bit_cast(tf::rocm::u32x8, vf);
#pragma unroll
                            for (int e = 0; e < 8; ++e) w[e] = 2 * e + 1 < n ? w[e] : 2 * e < n ? (w[e] & 0xffffu) : 0u;
#pragma unroll
                            for (int pl = 0; pl < NP; ++pl)
                                pv = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(rl == jr ? pf[pl] : zero16, __builtin_bit_cast(bf16x16, w), pv);
                        }
                    }
#pragma unroll
                    for (int i = 0; i < 8; ++i) o[at][j][i] = fmaf(alpha_r[i], o[at][j][i], pv[i]);
                }
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
        const int br = 2 * i + erh;
        if (br >= nr) continue;
        const int row = 16 * rw + br;
        const float inv = sL[row] == 0.f ? 0.f : 1.f / sL[row];
        float* dst = out + (static_cast<long long>(r0 + br) * heads + 4 * quad + rw) * d;
#pragma unroll
        for (int at = 0; at < nc; ++at)
#pragma unroll
            for (int j = 0; j < 4; ++j) dst[at * kCDims + 64 * kw + 16 * j + erl] = o[at][j][i] * inv;
    }
#else
    __builtin_trap();
#endif
}

// A prompt's listed rows (the indexer's keys; a row unlisted takes its first count[r] keys): block (r, g) is row r's
// query heads of kv head g (up to 16: one WMMA tile, the rest zero), 64 listed keys a tile, the dense kernel's
// arithmetic per row (q and p in NQ and NP planes, the online softmax per 64 keys of the row's list from its first).
// Past a row's count its keys and values read as zero, so its bits depend on its own q and listed keys alone.
template <int NQ, int NP>
__device__ void attn_listed(const float* __restrict__ q, const uint16_t* __restrict__ k, const uint16_t* __restrict__ v,
                            const int* __restrict__ ids, long long id_stride, const int* __restrict__ count,
                            const int* __restrict__ sparse, float* __restrict__ out, int heads, int kv_heads,
                            long long cap, float scale) {
#if TF_DEVICE_WMMA_GFX11
    using tf::rocm::bf16x16;
    using tf::rocm::f32x8;
    using elem = __bf16;
    constexpr int d = kApDims;
    constexpr int kCDims = 128;
    constexpr int kC = kCDims / 8;
    constexpr int nc = d / kCDims;
    constexpr int kQLd = d + 8;            // a staged q row's halves
    constexpr int kSLd = kApKeys;
    constexpr int kPLd = kApKeys + 8;
    constexpr int kVLd = kApKeys / 2 + 4;
    __shared__ alignas(16) elem sQ[NQ][16 * kQLd];
    __shared__ uint4 sKV[kCDims * kVLd / 4];  // a key chunk (key, 16 pieces swizzled) or a value chunk (transposed)
    __shared__ alignas(16) float sS[2][16 * kSLd];  // the scores' two halves of the dims
    __shared__ alignas(16) elem sP[NP][16 * kPLd];
    __shared__ float sA[16], sL[16];
    __shared__ int sKeys[2][kApKeys];
    uint32_t* sV = reinterpret_cast<uint32_t*>(sKV);

    const int r = blockIdx.x, g = blockIdx.y;
    const int group = heads / kv_heads;
    const int tid = threadIdx.x;
    const int n = count[r];
    const int* list = ids != nullptr && sparse != nullptr && sparse[r] != 0 ? ids + r * id_stride : nullptr;
    const __amdgpu_buffer_rsrc_t kbuf = ap_rsrc(k + static_cast<long long>(g) * cap * d, static_cast<uint32_t>(cap * d * 2));
    const __amdgpu_buffer_rsrc_t vbuf = ap_rsrc(v + static_cast<long long>(g) * cap * d, static_cast<uint32_t>(cap * d * 2));
    constexpr uint32_t kPast = 0xfffffff0u;  // an offset past every buffer: reads zero

    // q rows: head g group + i (zero past the group), as NQ planes
    for (int e = tid; e < 16 * (d / 16); e += kApThreads) {
        const int row = e / (d / 16), c = e % (d / 16);
        float x[16];
        const float* src = q + (static_cast<long long>(r) * heads + g * group + row) * d + 16 * c;
#pragma unroll
        for (int i = 0; i < 16; ++i) x[i] = row < group ? src[i] : 0.f;
        bf16x16 pl[NQ];
        ap_split<NQ>(x, pl);
#pragma unroll
        for (int p = 0; p < NQ; ++p) {
            const tf::rocm::u32x8 w = __builtin_bit_cast(tf::rocm::u32x8, pl[p]);
            uint4* dst = reinterpret_cast<uint4*>(&sQ[p][row * kQLd + 16 * c]);
            dst[0] = uint4{w[0], w[1], w[2], w[3]};
            dst[1] = uint4{w[4], w[5], w[6], w[7]};
        }
    }
    auto key_of = [&](int j) { return j < n ? (list ? list[j] : j) : -1; };
    if (tid < kApKeys) sKeys[0][tid] = key_of(tid);

    const int lane = tid & 31, wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int rl = lane & 15;
    const int nt = wave & 3, hk = wave >> 2;  // scores: keys 16 nt.., k-steps 4 hk.. of each chunk; values: dims 16 wave..
    const int sr = tid / 16, sk = tid % 16;
    float m_run = -INFINITY, l_run = 0.f;
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    f32x8 o[nc] = {zero, zero};
    auto frag = [](const uint4& lo, const uint4& hi) {
        const tf::rocm::u32x4 a = __builtin_bit_cast(tf::rocm::u32x4, lo), b = __builtin_bit_cast(tf::rocm::u32x4, hi);
        return __builtin_bit_cast(bf16x16, __builtin_shufflevector(a, b, 0, 1, 2, 3, 4, 5, 6, 7));
    };
    const int vp = tid % 32, vg = tid / 32;
    uint4 R[4];
    // chunk idx of tile `tile` (keys from sKeys[tile & 1]): keys (idx < nc), then values
    auto fetch = [&](int tile, int idx) {
        const int* keys = sKeys[tile & 1];
        if (idx < nc) {
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int e = tid + t * kApThreads, key = keys[e / kC], c = idx * kC + e % kC;
                R[t] = ap_load(kbuf, key < 0 ? kPast : static_cast<uint32_t>((key * d + 8 * c) * 2));
            }
        } else {
            const int dim = (idx - nc) * kCDims + 16 * vg;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int key = keys[2 * vp + (t >> 1)];
                R[t] = ap_load(vbuf, key < 0 ? kPast : static_cast<uint32_t>((key * d + dim + 8 * (t & 1)) * 2));
            }
        }
    };
    const int tiles = (n + kApKeys - 1) / kApKeys;
    __syncthreads();
    fetch(0, 0);

    for (int tile = 0; tile < tiles; ++tile) {
        const int k0 = tile * kApKeys;
        // the next tile's keys, read by its first fetch (after this tile's values are staged)
        if (tid < kApKeys) sKeys[(tile + 1) & 1][tid] = key_of(k0 + kApKeys + tid);
        f32x8 s = zero;
        for (int ci = 0; ci < nc; ++ci) {
            __syncthreads();
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int e = tid + t * kApThreads, key = e / kC, c = e % kC;
                sKV[key * kC + (c ^ (key & 15))] = R[t];
            }
            __syncthreads();
            fetch(tile, ci + 1);
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const int ks = 4 * hk + t;  // the chunk's k-step
                const int key = 16 * nt + rl;
                const bf16x16 bf = frag(sKV[key * kC + ((2 * ks) ^ (key & 15))], sKV[key * kC + ((2 * ks + 1) ^ (key & 15))]);
#pragma unroll
                for (int p = 0; p < NQ; ++p) {
                    const uint4* src = reinterpret_cast<const uint4*>(&sQ[p][rl * kQLd + ci * kCDims + 16 * ks]);
                    s = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(frag(src[0], src[1]), bf, s);
                }
            }
        }
        {
            const int rh = lane >> 4;
#pragma unroll
            for (int i = 0; i < 8; ++i) sS[hk][(2 * i + rh) * kSLd + 16 * nt + rl] = s[i];
        }
        __syncthreads();
        {
            const float4 a4 = *reinterpret_cast<const float4*>(&sS[0][sr * kSLd + 4 * sk]);
            const float4 b4 = *reinterpret_cast<const float4*>(&sS[1][sr * kSLd + 4 * sk]);
            float acc[4] = {a4.x + b4.x, a4.y + b4.y, a4.z + b4.z, a4.w + b4.w};
            float tile_max = -INFINITY;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                acc[j] = k0 + 4 * sk + j < n ? acc[j] * scale : -INFINITY;
                tile_max = fmaxf(tile_max, acc[j]);
            }
#pragma unroll
            for (int ln = 8; ln > 0; ln >>= 1) tile_max = fmaxf(tile_max, ap_xmask16(tile_max, ln));
            const float next_m = fmaxf(m_run, tile_max);
            const float alpha = next_m == -INFINITY ? 1.f : expf(m_run - next_m);
            float sum = 0.f;
            elem hp[NP][4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                float p = next_m == -INFINITY ? 0.f : expf(acc[j] - next_m);
                sum += p;
                elem pp[NP];
                ap_split1<NP>(p, pp);
#pragma unroll
                for (int pl = 0; pl < NP; ++pl) hp[pl][j] = pp[pl];
            }
#pragma unroll
            for (int ln = 8; ln > 0; ln >>= 1) sum += ap_xmask16(sum, ln);
            l_run = alpha * l_run + sum;
            m_run = next_m;
#pragma unroll
            for (int pl = 0; pl < NP; ++pl) *reinterpret_cast<uint2*>(&sP[pl][sr * kPLd + 4 * sk]) = __builtin_bit_cast(uint2, hp[pl]);
            if (sk == 0) sA[sr] = alpha;
        }
        float alpha_r[8];
#pragma unroll
        for (int at = 0; at < nc; ++at) {
            __syncthreads();
            if (at == 0) {
                const int rh = lane >> 4;
#pragma unroll
                for (int i = 0; i < 8; ++i) alpha_r[i] = sA[2 * i + rh];
            }
#pragma unroll
            for (int hlf = 0; hlf < 2; ++hlf) {
                const uint4 va = R[hlf], vb = R[2 + hlf];
                const uint32_t wa[4] = {va.x, va.y, va.z, va.w};
                const uint32_t wb[4] = {vb.x, vb.y, vb.z, vb.w};
#pragma unroll
                for (int m = 0; m < 4; ++m) {
                    const int dm = 16 * vg + 8 * hlf + 2 * m;
                    sV[dm * kVLd + vp] = (wa[m] & 0xffffu) | (wb[m] << 16);
                    sV[(dm + 1) * kVLd + vp] = (wa[m] >> 16) | (wb[m] & 0xffff0000u);
                }
            }
            __syncthreads();
            if (at + 1 < nc) fetch(tile, nc + at + 1);
            else if (tile + 1 < tiles) fetch(tile + 1, 0);
            f32x8 pv = zero;
#pragma unroll
            for (int t = 0; t < 4; ++t) {
                const uint4* vs = reinterpret_cast<const uint4*>(&sV[(16 * wave + rl) * kVLd + 8 * t]);
                const bf16x16 vf = frag(vs[0], vs[1]);
#pragma unroll
                for (int pl = 0; pl < NP; ++pl) {
                    const uint4* ps = reinterpret_cast<const uint4*>(&sP[pl][rl * kPLd + 16 * t]);
                    pv = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(frag(ps[0], ps[1]), vf, pv);
                }
            }
#pragma unroll
            for (int i = 0; i < 8; ++i) o[at][i] = fmaf(alpha_r[i], o[at][i], pv[i]);
        }
    }
    if (sk == 0) sL[sr] = l_run;
    __syncthreads();
    const int ln = static_cast<int>(__builtin_amdgcn_mbcnt_lo(~0u, 0u));
    const int erl = ln & 15, erh = ln >> 4;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        const int row = 2 * i + erh;
        if (row >= group) continue;
        const float inv = sL[row] == 0.f ? 0.f : 1.f / sL[row];
        float* dst = out + (static_cast<long long>(r) * heads + g * group + row) * d;
#pragma unroll
        for (int at = 0; at < nc; ++at) dst[at * kCDims + 16 * wave + erl] = o[at][i] * inv;
    }
#else
    __builtin_trap();
#endif
}

#define TF_ATTN_PROMPT(NQ, NP)                                                                                         \
    extern "C" __global__ void __launch_bounds__(kApThreads, 1) tf_fn_attn_prompt##NQ##NP(                             \
        const float* q, const uint16_t* k, const uint16_t* v, const int* count, float* out, int rows, int heads,     \
        int kv_heads, long long cap, float scale) {                                                                     \
        attn_prompt<NQ, NP>(q, k, v, count, out, rows, heads, kv_heads, cap, scale);                                  \
    }
#define TF_ATTN_LISTED(NQ, NP)                                                                                         \
    extern "C" __global__ void __launch_bounds__(kApThreads) tf_fn_attn_listed##NQ##NP(                               \
        const float* q, const uint16_t* k, const uint16_t* v, const int* ids, long long id_stride, const int* count,  \
        const int* sparse, float* out, int heads, int kv_heads, long long cap, float scale) {                         \
        attn_listed<NQ, NP>(q, k, v, ids, id_stride, count, sparse, out, heads, kv_heads, cap, scale);               \
    }
TF_ATTN_LISTED(1, 1)
TF_ATTN_LISTED(2, 2)
TF_ATTN_LISTED(3, 3)
TF_ATTN_LISTED(3, 1)
TF_ATTN_LISTED(1, 3)
#undef TF_ATTN_LISTED
TF_ATTN_PROMPT(1, 1)
TF_ATTN_PROMPT(2, 2)
TF_ATTN_PROMPT(3, 3)
TF_ATTN_PROMPT(3, 1)
TF_ATTN_PROMPT(1, 3)
#undef TF_ATTN_PROMPT

}  // namespace
