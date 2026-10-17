// The DFlash2 drafter's own kernels (families/qwen3_5/backend/hip/draft): tap rows, rings, dynamic conv, block attention.

#include <hip/hip_bfloat16.h>
#include <hip/hip_runtime.h>
#include <cstdint>

#include "common/arch.hpp"
#include "common/wmma.hpp"

namespace {


__device__ __forceinline__ float bf(uint16_t h) { return __uint_as_float(static_cast<uint32_t>(h) << 16); }

// Round to nearest even, as torch's bfloat16.
__device__ __forceinline__ uint16_t tobf(float f) {
    if (f != f) return 0x7fc0;
    const uint32_t u = __float_as_uint(f);
    return static_cast<uint16_t>((u + ((u >> 16) & 1u) + 0x7fffu) >> 16);
}

__device__ __forceinline__ float rbf(float x) { return bf(tobf(x)); }

__device__ __forceinline__ float wave_sum(float x) {
#pragma unroll
    for (int m = 16; m; m >>= 1) x += __shfl_xor(x, m, 32);
    return x;
}

// Per-head RMS norm (fp32 weight) then rotate-half RoPE at `pos` of HD values (128 or 256), HD / 32 a lane; returns
// this lane's.
template <int HD>
__device__ __forceinline__ void norm_rope4(const uint16_t* x, const float* w, float eps, float theta, long long pos, float out[HD / 32]) {
    constexpr int E = HD / 32;
    const int t = threadIdx.x & 31;
    float v[E], acc = 0.f;
#pragma unroll
    for (int i = 0; i < E; i++) {
        v[i] = bf(x[E * t + i]);
        acc += v[i] * v[i];
    }
    acc = wave_sum(acc);
    const float inv = rsqrtf(acc / static_cast<float>(HD) + eps);
    float n[E];
#pragma unroll
    for (int i = 0; i < E; i++) n[i] = rbf(v[i] * inv * w[E * t + i]);
    // the partner of element d is d +- HD / 2: lane t's partners sit on lane t ^ 16
#pragma unroll
    for (int i = 0; i < E; i++) {
        const int d = E * t + i;
        const float partner = __shfl_xor(n[i], 16, 32);
        const int j = d < HD / 2 ? d : d - HD / 2;
        const float angle = static_cast<float>(pos) * powf(theta, -static_cast<float>(j) / static_cast<float>(HD / 2));
        const float sn = sinf(angle), cs = cosf(angle);
        out[i] = rbf(d < HD / 2 ? n[i] * cs - partner * sn : n[i] * cs + partner * sn);
    }
}

}  // namespace

// One absorbed row: its features' address, its stream's rings and its position (draft.zig's AbsorbRow).
struct DftRow {
    uint64_t src;
    uint64_t ring;
    long long pos;
};

// A held stream: its rings, the first position its context keeps and the block's first position (draft.zig's HoldDesc).
struct DftDesc {
    uint64_t ring;
    long long begin;
    long long pos0;
};

// OUT row i = the `width` bytes at rows[i].src (16-byte multiples). Grid (n), 256 threads.
extern "C" __global__ void tf_dft_gather(const DftRow* rows, int width, uint4* out) {
    const uint4* src = reinterpret_cast<const uint4*>(rows[blockIdx.x].src);
    uint4* dst = out + static_cast<size_t>(blockIdx.x) * (width / 16);
    for (int i = threadIdx.x; i < width / 16; i += blockDim.x) dst[i] = src[i];
}

// Absorbed rows' keys and values into every layer's ring: KV row (stride `stride`) holds layer l's k at 2 l kvw, v after.
// A layer's ring: keys (R, kv_heads, HD) then values transposed by 16-slot tiles (kv_heads, R / 16, HD, 16). Grid (rows, layers * kv_heads), 32 threads.
template <int HD>
__device__ __forceinline__ void dft_ring_put(const uint16_t* KV, long long stride, const DftRow* rows, const float* knorm, int kv_heads, int R,
                                             float eps, float theta) {
    constexpr int E = HD / 32;
    const int r = blockIdx.x;
    const int layer = blockIdx.y / kv_heads, h = blockIdx.y % kv_heads;
    const int kvw = kv_heads * HD;
    const long long pos = rows[r].pos;
    const size_t slot = static_cast<size_t>(pos % R);
    uint16_t* ring = reinterpret_cast<uint16_t*>(rows[r].ring) + static_cast<size_t>(layer) * 2 * R * kvw;
    const uint16_t* k = KV + r * stride + static_cast<size_t>(layer) * 2 * kvw + h * HD;
    float out[E];
    norm_rope4<HD>(k, knorm + layer * HD, eps, theta, pos, out);
    const int t = threadIdx.x;
#pragma unroll
    for (int i = 0; i < E; i++) ring[slot * kvw + h * HD + E * t + i] = tobf(out[i]);
    const uint16_t* v = k + kvw;
    uint16_t* vt = ring + static_cast<size_t>(R) * kvw + static_cast<size_t>(h) * HD * R;
#pragma unroll
    for (int i = 0; i < E; i++) vt[(slot / 16) * HD * 16 + static_cast<size_t>(E * t + i) * 16 + slot % 16] = v[E * t + i];
}

extern "C" __global__ void __launch_bounds__(32) tf_dft_ring_put(const uint16_t* KV, long long stride, const DftRow* rows, const float* knorm,
                                                                 int kv_heads, int R, float eps, float theta) {
    dft_ring_put<128>(KV, stride, rows, knorm, kv_heads, R, eps, theta);
}
extern "C" __global__ void __launch_bounds__(32) tf_dft_ring_put_256(const uint16_t* KV, long long stride, const DftRow* rows,
                                                                     const float* knorm, int kv_heads, int R, float eps, float theta) {
    dft_ring_put<256>(KV, stride, rows, knorm, kv_heads, R, eps, theta);
}

// Block rows' q or k heads normed and rotated at desc[row / block].pos0 + row % block; OUT packed (rows, heads, HD).
// Grid (rows * heads), 32 threads.
extern "C" __global__ void __launch_bounds__(32) tf_dft_norm_rope(const uint16_t* X, long long stride, const float* W, uint16_t* OUT,
                                                                  const DftDesc* desc, int block, int heads, float eps, float theta) {
    constexpr int HD = 128;
    const int head = blockIdx.x % heads;
    const int row = blockIdx.x / heads;
    const long long pos = desc[row / block].pos0 + row % block;
    float out[4];
    norm_rope4<HD>(X + row * stride + head * HD, W, eps, theta, pos, out);
    uint16_t* o = OUT + (static_cast<size_t>(row) * heads + head) * HD + 4 * threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; i++) o[i] = tobf(out[i]);
}

// ops/act.hip's rms_kernel on bf16 rows of at most 8192, its bits: lane l of one wave sums the squares of l, l + 32, ...
// in order, the xor tree, then x * inv * w rounded. The row staged in LDS by the block, the wave's reads 16 ahead of its
// adds; each of a row's gridDim.y blocks writes its share. Grid (rows, parts), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_dft_rms(const hip_bfloat16* X, const float* W, hip_bfloat16* Y, int width, float eps) {
    __shared__ float row_v[8192];
    __shared__ float inv_s;
    const int row = blockIdx.x;
    const hip_bfloat16* in = X + static_cast<long long>(row) * width;
    for (int i = threadIdx.x; i < width; i += 256) row_v[i] = static_cast<float>(in[i]);
    __syncthreads();
    if (threadIdx.x < 32) {
        float sum = 0.f;
        int i = threadIdx.x;
        for (; i + 32 * 15 < width; i += 32 * 16) {
            float v[16];
#pragma unroll
            for (int j = 0; j < 16; ++j) v[j] = row_v[i + 32 * j];
#pragma unroll
            for (int j = 0; j < 16; ++j) sum += v[j] * v[j];
        }
        for (; i < width; i += 32) sum += row_v[i] * row_v[i];
#pragma unroll
        for (int mask = 16; mask > 0; mask >>= 1) sum += __shfl_xor(sum, mask, 32);
        sum = __shfl(sum, 0, 32);
        if (threadIdx.x == 0) inv_s = rsqrtf(sum / static_cast<float>(width) + eps);
    }
    __syncthreads();
    const float inv = inv_s;
    const int per = (width + gridDim.y - 1) / gridDim.y;
    const int end = min(width, (static_cast<int>(blockIdx.y) + 1) * per);
    hip_bfloat16* out = Y + static_cast<long long>(row) * width;
    for (int i = blockIdx.y * per + threadIdx.x; i < end; i += 256) {
        float value = row_v[i] * inv;
        value *= W[i];
        out[i] = hip_bfloat16(value);
    }
}

// tf_dft_norm_rope on the q heads (rows x heads blocks) then the k heads at X + koff (rows x kv_heads), in one launch.
template <int HD>
__device__ __forceinline__ void dft_norm_rope_qk(const uint16_t* X, long long stride, long long koff, const float* WQ, const float* WK,
                                                 uint16_t* QO, uint16_t* KO, const DftDesc* desc, int block, int rows, int heads,
                                                 int kv_heads, float eps, float theta) {
    constexpr int E = HD / 32;
    const bool q = static_cast<int>(blockIdx.x) < rows * heads;
    const int b = q ? blockIdx.x : blockIdx.x - rows * heads;
    const int hs = q ? heads : kv_heads;
    const int head = b % hs;
    const int row = b / hs;
    const long long pos = desc[row / block].pos0 + row % block;
    float out[E];
    norm_rope4<HD>((q ? X : X + koff) + row * stride + head * HD, q ? WQ : WK, eps, theta, pos, out);
    uint16_t* o = (q ? QO : KO) + (static_cast<size_t>(row) * hs + head) * HD + E * threadIdx.x;
#pragma unroll
    for (int i = 0; i < E; i++) o[i] = tobf(out[i]);
}

extern "C" __global__ void __launch_bounds__(32) tf_dft_norm_rope_qk(const uint16_t* X, long long stride, long long koff, const float* WQ,
                                                                     const float* WK, uint16_t* QO, uint16_t* KO, const DftDesc* desc,
                                                                     int block, int rows, int heads, int kv_heads, float eps, float theta) {
    dft_norm_rope_qk<128>(X, stride, koff, WQ, WK, QO, KO, desc, block, rows, heads, kv_heads, eps, theta);
}
extern "C" __global__ void __launch_bounds__(32) tf_dft_norm_rope_qk_256(const uint16_t* X, long long stride, long long koff,
                                                                         const float* WQ, const float* WK, uint16_t* QO, uint16_t* KO,
                                                                         const DftDesc* desc, int block, int rows, int heads, int kv_heads,
                                                                         float eps, float theta) {
    dft_norm_rope_qk<256>(X, stride, koff, WQ, WK, QO, KO, desc, block, rows, heads, kv_heads, eps, theta);
}

// y[t, c] = sum over lags l of (BASE[l, c] + DYN[t, l, c / group]) * x[t - l, c] in bf16 steps, x before a block's first row 0;
// RES (when given) added last. Grid (rows, D / 256 rounded up), 256 threads: a channel a thread.
extern "C" __global__ void tf_dft_dyn_conv(const uint16_t* X, const uint16_t* DYN, const uint16_t* BASE, const uint16_t* RES, uint16_t* OUT,
                                           int D, int group, int branch, int block) {
    const int t = blockIdx.x;
    const int groups = D / group;
    const int lags = t % block == 0 ? 1 : 2;
    const int c = blockIdx.y * blockDim.x + threadIdx.x;
    if (c < D) {
        const int g = c / group;
        float total = 0.f;
        for (int l = 0; l < lags; l++) {
            const float x = bf(X[static_cast<size_t>(t - l) * D + c]);
            total = rbf(total + rbf(bf(BASE[(branch * 2 + l) * D + c]) * x));
            total = rbf(total + rbf(bf(DYN[static_cast<size_t>(t) * (4 * groups) + (branch * 2 + l) * groups + g]) * x));
        }
        if (RES != nullptr) total = rbf(total + bf(RES[static_cast<size_t>(t) * D + c]));
        OUT[static_cast<size_t>(t) * D + c] = tobf(total);
    }
}

namespace {

using tf::rocm::bf16x16;
using tf::rocm::f32x8;

typedef uint32_t u32x8v __attribute__((ext_vector_type(8)));

__device__ __forceinline__ bf16x16 load16(const uint16_t* p) {
    const uint4 a = reinterpret_cast<const uint4*>(p)[0];
    const uint4 b = reinterpret_cast<const uint4*>(p)[1];
    const u32x8v v = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
    return __builtin_bit_cast(bf16x16, v);
}

__device__ __forceinline__ f32x8 mma(bf16x16 a, bf16x16 b, f32x8 c) {
#if TF_DEVICE_WMMA_GFX11
    return __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(a, b, c);
#else
    (void)a;
    (void)b;
    __builtin_trap();
    return c;
#endif
}

constexpr int kTile = 16;
constexpr int kQueries = 64;  // a stream's block rows times the query heads of a key head: four waves of 16

}  // namespace

// Block attention, split over the keys: query (m, g) of a stream's key head sees ring positions
// [max(begin, pos0 + m - window + 1), pos0) and every block row. Split y of `splits` walks its share of 16-position tiles,
// writing unnormalized PART (splits, 64 queries, HD) fp32 and STATS (max, sum) per (stream, key head).
// Q (rows, heads, HD), KB (rows, kv_heads, HD) packed; VB rows `vstride` apart. Grid (streams * kv_heads, splits, HD / HO),
// 128 threads: z owns output dims [HO z, HO z + HO) (a 256-wide head's whole output would not fit the registers).
template <int HD>
__device__ __forceinline__ void dft_attn(const uint16_t* Q, const uint16_t* KB, const uint16_t* VB, long long vstride, const DftDesc* desc,
                                         int layer, int block, int heads, int kv_heads, int R, int window, float* PART, float* STATS) {
    constexpr int HO = HD < 128 ? HD : 128;
    const int od = HO * blockIdx.z;
    const int stream = blockIdx.x / kv_heads, kvh = blockIdx.x % kv_heads;
    const int split = blockIdx.y, splits = gridDim.y;
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = lane & 15, half = lane >> 4;
    const int G = heads / kv_heads;
    const int kvw = kv_heads * HD;
    const DftDesc d = desc[stream];
    const uint16_t* RK = reinterpret_cast<const uint16_t*>(d.ring) + static_cast<size_t>(layer) * 2 * R * kvw;
    const uint16_t* RVT = RK + static_cast<size_t>(R) * kvw + static_cast<size_t>(kvh) * HD * R;
    // this lane's query column: q = 16 wave + col, block row q / G, head kvh G + q % G
    const int q = 16 * wave + col;
    const bool qlive = q < block * G;
    const int m = qlive ? q / G : 0;
    const int head = kvh * G + (qlive ? q % G : 0);
    const uint16_t* qp = Q + (static_cast<size_t>(stream * block + m) * heads + head) * HD;
    bf16x16 qf[HD / 16];
#pragma unroll
    for (int t = 0; t < HD / 16; t++) qf[t] = load16(qp + 16 * t);
    const float scale = rsqrtf(static_cast<float>(HD));
    long long first = d.pos0 + m - window + 1;
    if (first < d.begin) first = d.begin;
    long long lo = d.pos0 - window + 1;
    if (lo < d.begin) lo = d.begin;
    if (lo > d.pos0) lo = d.pos0;
    const long long base = (lo / kTile) * kTile;
    const int tiles = static_cast<int>((d.pos0 - base + kTile - 1) / kTile);
    const int t0 = static_cast<int>(static_cast<long long>(tiles) * split / splits);
    const int t1 = static_cast<int>(static_cast<long long>(tiles) * (split + 1) / splits);
    f32x8 o[HO / 16];
#pragma unroll
    for (int i = 0; i < HO / 16; i++) o[i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    float mx = -INFINITY, l = 0.f;
    // one tile of 16 keys: s[i] the score of key 2 i + half for this lane's query; vfrag(i) V^T's rows 16 i.. of the tile
    auto step = [&](f32x8 s, auto&& vfrag) {
        float tmax = -INFINITY;
#pragma unroll
        for (int i = 0; i < 8; i++) tmax = fmaxf(tmax, s[i]);
        tmax = fmaxf(tmax, __shfl_xor(tmax, 16, 32));
        const float mn = fmaxf(mx, tmax);
        // no branch on a lane's query (the products take the whole wave): nothing visible yet keeps everything 0
        const float mref = mn == -INFINITY ? 0.f : mn;
        const float alpha = __expf(mx - mref);
        float p[8], psum = 0.f;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            p[i] = __expf(s[i] - mref);
            psum += p[i];
        }
        psum += __shfl_xor(psum, 16, 32);
        l = l * alpha + psum;
        mx = mn;
        // P^T's fragment for this lane's query: key k from this lane when k % 2 == half, else from lane ^ 16
        uint16_t mine[8], other[8];
#pragma unroll
        for (int i = 0; i < 8; i++) mine[i] = tobf(p[i]);
#pragma unroll
        for (int i = 0; i < 8; i += 2) {
            const uint32_t packed = static_cast<uint32_t>(mine[i]) | (static_cast<uint32_t>(mine[i + 1]) << 16);
            const uint32_t got = __shfl_xor(packed, 16, 32);
            other[i] = static_cast<uint16_t>(got & 0xffff);
            other[i + 1] = static_cast<uint16_t>(got >> 16);
        }
        uint16_t pk[16];
#pragma unroll
        for (int k = 0; k < 16; k++) pk[k] = (k & 1) == half ? mine[k >> 1] : other[k >> 1];
        const bf16x16 pf = __builtin_bit_cast(bf16x16, pk);
#pragma unroll
        for (int i = 0; i < HO / 16; i++) {
            f32x8 c = o[i];
#pragma unroll
            for (int e = 0; e < 8; e++) c[e] *= alpha;
            o[i] = mma(vfrag(i), pf, c);
        }
    };
    for (int t = t0; t < t1; t++) {
        const long long p0 = base + static_cast<long long>(t) * kTile;
        const size_t slot0 = static_cast<size_t>(p0 % R);
        // S^T = K Q^T: A the keys (this lane's key `col`), B the queries
        f32x8 s = {0, 0, 0, 0, 0, 0, 0, 0};
        const uint16_t* kp = RK + (slot0 + col) * kvw + kvh * HD;
#pragma unroll
        for (int u = 0; u < HD / 16; u++) s = mma(load16(kp + 16 * u), qf[u], s);
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const long long kpos = p0 + 2 * i + half;
            s[i] = (qlive && kpos >= first && kpos < d.pos0) ? s[i] * scale : -INFINITY;
        }
        // the tile's values: (HD, 16) back to back
        step(s, [&](int i) { return load16(RVT + slot0 * HD + static_cast<size_t>(od + 16 * i + col) * 16); });
    }
    if (split == splits - 1) {
        // the block's own rows, every one visible to every block row (zeros past the block)
        uint16_t kz[16];
        bf16x16 kf[HD / 16];
        const bool klive = col < block;
        const uint16_t* kb = KB + (static_cast<size_t>(stream * block + (klive ? col : 0)) * kv_heads + kvh) * HD;
#pragma unroll
        for (int u = 0; u < HD / 16; u++) {
            if (klive) {
                kf[u] = load16(kb + 16 * u);
            } else {
#pragma unroll
                for (int e = 0; e < 16; e++) kz[e] = 0;
                kf[u] = __builtin_bit_cast(bf16x16, kz);
            }
        }
        f32x8 s = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
        for (int u = 0; u < HD / 16; u++) s = mma(kf[u], qf[u], s);
#pragma unroll
        for (int i = 0; i < 8; i++) s[i] = (qlive && 2 * i + half < block) ? s[i] * scale : -INFINITY;
        const uint16_t* vb = VB + static_cast<size_t>(stream * block) * vstride + kvh * HD;
        step(s, [&](int i) {
            uint16_t vv[16];
#pragma unroll
            for (int k = 0; k < 16; k++) vv[k] = k < block ? vb[k * vstride + od + 16 * i + col] : 0;
            return __builtin_bit_cast(bf16x16, vv);
        });
    }
    // O^T's tile i: this lane's query, dims 16 i + 2 e + half
    const size_t at = (static_cast<size_t>(blockIdx.x) * splits + split) * kQueries + q;
    float* part = PART + at * HD + od;
#pragma unroll
    for (int i = 0; i < HO / 16; i++) {
#pragma unroll
        for (int e = 0; e < 8; e++) part[16 * i + 2 * e + half] = o[i][e];
    }
    if (half == 0 && blockIdx.z == 0) {
        STATS[2 * at] = mx;
        STATS[2 * at + 1] = l;
    }
}

extern "C" __global__ void __launch_bounds__(128) tf_dft_attn(const uint16_t* Q, const uint16_t* KB, const uint16_t* VB, long long vstride,
                                                             const DftDesc* desc, int layer, int block, int heads, int kv_heads, int R,
                                                             int window, float* PART, float* STATS) {
    dft_attn<128>(Q, KB, VB, vstride, desc, layer, block, heads, kv_heads, R, window, PART, STATS);
}
extern "C" __global__ void __launch_bounds__(128) tf_dft_attn_256(const uint16_t* Q, const uint16_t* KB, const uint16_t* VB,
                                                                 long long vstride, const DftDesc* desc, int layer, int block, int heads,
                                                                 int kv_heads, int R, int window, float* PART, float* STATS) {
    dft_attn<256>(Q, KB, VB, vstride, desc, layer, block, heads, kv_heads, R, window, PART, STATS);
}

// The splits of tf_dft_attn joined: OUT (rows, heads, HD) bf16, rows ostride apart. Grid (streams * kv_heads, 64 queries), HD threads.
template <int HD>
__device__ __forceinline__ void dft_attn_merge(const float* PART, const float* STATS, int splits, int block, int heads, int kv_heads,
                                               uint16_t* OUT, long long ostride) {
    const int stream = blockIdx.x / kv_heads, kvh = blockIdx.x % kv_heads;
    const int q = blockIdx.y;
    const int G = heads / kv_heads;
    if (q >= block * G) return;
    const int dim = threadIdx.x;
    float mx = -INFINITY;
    for (int s = 0; s < splits; s++) mx = fmaxf(mx, STATS[2 * ((static_cast<size_t>(blockIdx.x) * splits + s) * kQueries + q)]);
    float l = 0.f, acc = 0.f;
    for (int s = 0; s < splits; s++) {
        const size_t at = (static_cast<size_t>(blockIdx.x) * splits + s) * kQueries + q;
        const float ms = STATS[2 * at];
        if (ms == -INFINITY) continue;
        const float w = __expf(ms - mx);
        l += STATS[2 * at + 1] * w;
        acc += PART[at * HD + dim] * w;
    }
    const int m = q / G, head = kvh * G + q % G;
    OUT[static_cast<size_t>(stream * block + m) * ostride + head * HD + dim] = tobf(acc / l);
}

extern "C" __global__ void tf_dft_attn_merge(const float* PART, const float* STATS, int splits, int block, int heads, int kv_heads,
                                             uint16_t* OUT, long long ostride) {
    dft_attn_merge<128>(PART, STATS, splits, block, heads, kv_heads, OUT, ostride);
}
extern "C" __global__ void tf_dft_attn_merge_256(const float* PART, const float* STATS, int splits, int block, int heads, int kv_heads,
                                                 uint16_t* OUT, long long ostride) {
    dft_attn_merge<256>(PART, STATS, splits, block, heads, kv_heads, OUT, ostride);
}

namespace {

// A finite value's bf16 bits, round to nearest even (dequantized weights are never NaN).
__device__ __forceinline__ uint32_t rne(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u + 0x7fffu + ((u >> 16) & 1u)) >> 16;
}

// 16 weights of a column from two code words (low nibble first): scale q + bias, rounded to bf16, two a dword.
__device__ __forceinline__ u32x8v dequant16(uint32_t w0, uint32_t w1, float s, float b) {
    u32x8v v;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = rne(static_cast<float>((w0 >> (8 * i)) & 15u) * s + b) | (rne(static_cast<float>((w0 >> (8 * i + 4)) & 15u) * s + b) << 16);
        v[4 + i] = rne(static_cast<float>((w1 >> (8 * i)) & 15u) * s + b) | (rne(static_cast<float>((w1 >> (8 * i + 4)) & 15u) * s + b) << 16);
    }
    return v;
}

}  // namespace

// OUT (rows, n) bf16 = X (rows, k) bf16 times an affine-4 group-64 matrix (n, k) in the drafter's tiled layout, rows at most
// 64: every weight read once whatever the rows (the target's stream tile reads them once per 4 rows). The layout: blocks of
// 32 columns, each group of a block 32 lanes x 32 bytes (lane l column 32 b + l, its 64 nibbles low first) back to back, then
// the scales and biases as (block, group, 32) bf16. A workgroup owns a block, its 4 waves every 4th group (adjacent bytes at
// once), summed through LDS. Lane l dequantizes column l's group once; the halves trade fragments; the group runs against
// every row tile. Grid (blocks), 128 threads.
extern "C" __global__ void __launch_bounds__(128) tf_dft_q4_rows(const uint16_t* X, const uint4* W, const uint16_t* S, const uint16_t* Bz,
                                                                 uint16_t* OUT, int rows, int n, int k) {
    constexpr int kWaves = 4;
    __shared__ float red[kWaves - 1][4][2][8][32];
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = lane & 15, half = lane >> 4;
    const int block = blockIdx.x;
    const int n0 = block * 32;
    const int tiles = (rows + 15) / 16;
    const int groups = k / 64;
    const uint4* w = W + (static_cast<size_t>(block) * groups * 32 + lane) * 2;
    const uint16_t* sp = S + static_cast<size_t>(block) * groups * 32 + lane;
    const uint16_t* bp = Bz + static_cast<size_t>(block) * groups * 32 + lane;
    f32x8 acc[4][2];
#pragma unroll
    for (int t = 0; t < 4; t++) acc[t][0] = acc[t][1] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    const uint16_t* xr[4];
#pragma unroll
    for (int t = 0; t < 4; t++) xr[t] = X + static_cast<size_t>(min(16 * t + col, rows - 1)) * k;
    for (int g = wave; g < groups; g += kWaves) {
        const uint4 p0 = w[64 * g], p1 = w[64 * g + 1];
        const float s = bf(sp[32 * g]), b = bf(bp[32 * g]);
        // 16 k a step: this lane's column (lanes 0-15 the block's first 16, 16-31 the next); the halves trade by a shuffle
        auto step = [&](int u, uint32_t lo, uint32_t hi) __attribute__((always_inline)) {
            const u32x8v mine = dequant16(lo, hi, s, b);
            const u32x8v other = {__shfl_xor(mine[0], 16, 32), __shfl_xor(mine[1], 16, 32), __shfl_xor(mine[2], 16, 32), __shfl_xor(mine[3], 16, 32),
                                  __shfl_xor(mine[4], 16, 32), __shfl_xor(mine[5], 16, 32), __shfl_xor(mine[6], 16, 32), __shfl_xor(mine[7], 16, 32)};
            const u32x8v lo16 = half == 0 ? mine : other, hi16 = half == 0 ? other : mine;
            const bf16x16 fa = __builtin_bit_cast(bf16x16, lo16);
            const bf16x16 fb = __builtin_bit_cast(bf16x16, hi16);
#pragma unroll
            for (int t = 0; t < 4; t++) {
                if (t < tiles) {
                    const bf16x16 x = load16(xr[t] + 64 * g + 16 * u);
                    acc[t][0] = mma(x, fa, acc[t][0]);
                    acc[t][1] = mma(x, fb, acc[t][1]);
                }
            }
        };
        step(0, p0.x, p0.y);
        step(1, p0.z, p0.w);
        step(2, p1.x, p1.y);
        step(3, p1.z, p1.w);
    }
    if (wave > 0) {
#pragma unroll
        for (int t = 0; t < 4; t++) {
            if (t < tiles) {
#pragma unroll
                for (int j = 0; j < 2; j++) {
#pragma unroll
                    for (int i = 0; i < 8; i++) red[wave - 1][t][j][i][lane] = acc[t][j][i];
                }
            }
        }
    }
    __syncthreads();
    if (wave != 0) return;
#pragma unroll
    for (int t = 0; t < 4; t++) {
        if (t >= tiles) continue;
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = 16 * t + 2 * i + half;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                float v = acc[t][j][i];
#pragma unroll
                for (int x = 0; x < kWaves - 1; x++) v += red[x][t][j][i][lane];
                const int c = n0 + 16 * j + col;
                if (r < rows && c < n) OUT[static_cast<size_t>(r) * n + c] = tobf(v);
            }
        }
    }
}

// OUT (rows, n) bf16 = X (rows, k) bf16 times W (n, k) bf16 transposed, rows at most 64, k a multiple of 32:
// --drafter-bits 0's products. A wave owns 32 columns over every row, so the weights stream once. Grid (n / 128 up), 128 threads.
extern "C" __global__ void __launch_bounds__(128) tf_dft_bf16_rows(const uint16_t* X, const uint16_t* W, uint16_t* OUT, int rows, int n, int k) {
    const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int col = lane & 15, half = lane >> 4;
    const int n0 = (blockIdx.x * 4 + wave) * 32;
    if (n0 >= n) return;  // a whole wave leaves: the products take the whole wave
    const int tiles = (rows + 15) / 16;
    const uint16_t* w0 = W + static_cast<size_t>(min(n0 + col, n - 1)) * k;
    const uint16_t* w1 = W + static_cast<size_t>(min(n0 + 16 + col, n - 1)) * k;
    f32x8 acc[4][2];
#pragma unroll
    for (int t = 0; t < 4; t++) acc[t][0] = acc[t][1] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    const uint16_t* xr[4];
#pragma unroll
    for (int t = 0; t < 4; t++) xr[t] = X + static_cast<size_t>(min(16 * t + col, rows - 1)) * k;
    for (int kk = 0; kk < k; kk += 32) {
        bf16x16 b[2][2];
#pragma unroll
        for (int u = 0; u < 2; u++) {
            b[u][0] = load16(w0 + kk + 16 * u);
            b[u][1] = load16(w1 + kk + 16 * u);
        }
#pragma unroll
        for (int t = 0; t < 4; t++) {
            if (t < tiles) {
#pragma unroll
                for (int u = 0; u < 2; u++) {
                    const bf16x16 a = load16(xr[t] + kk + 16 * u);
                    acc[t][0] = mma(a, b[u][0], acc[t][0]);
                    acc[t][1] = mma(a, b[u][1], acc[t][1]);
                }
            }
        }
    }
#pragma unroll
    for (int t = 0; t < 4; t++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
            const int r = 16 * t + 2 * i + half;
            if (t >= tiles || r >= rows) continue;
#pragma unroll
            for (int j = 0; j < 2; j++) {
                const int c = n0 + 16 * j + col;
                if (c < n) OUT[static_cast<size_t>(r) * n + c] = tobf(acc[t][j][i]);
            }
        }
    }
}

namespace {

// A monotone 16-bit key of a value's bits (-0 and +0 share one), as ops/draw.hpp's tf_topk orders them.
__device__ __forceinline__ uint32_t order_key(uint16_t bits) {
    if (bits == 0x8000) bits = 0;
    return (bits & 0x8000) ? (~bits & 0xffffu) : (bits | 0x8000u);
}

}  // namespace

// Row r's ks[r] largest by (value desc, id asc) out of its S chunks' own top k (tf_topk over the row as S rows of
// `chunk`): CI/CV (rows, S, k) chunk ids and values, ids returned as the row's. The same set as tf_topk over the whole
// row, best first. Grid (rows), S x k threads (at most 1024).
extern "C" __global__ void tf_dft_topk_merge(const int* CI, const uint16_t* CV, int S, int k, int chunk, const int* ks, int stride,
                                             int* ids, uint16_t* values) {
    const int r = blockIdx.x, want = ks[r];
    if (want <= 0) return;
    __shared__ uint32_t key[1024];
    __shared__ int id[1024];
    const int n = S * k, i = threadIdx.x;
    if (i < n) {
        key[i] = order_key(CV[static_cast<size_t>(r) * n + i]);
        id[i] = (i / k) * chunk + CI[static_cast<size_t>(r) * n + i];
    }
    __syncthreads();
    if (i >= n) return;
    int rank = 0;
    for (int j = 0; j < n; j++) rank += key[j] > key[i] || (key[j] == key[i] && id[j] < id[i]);
    if (rank < want) {
        ids[r * stride + rank] = id[i];
        values[r * stride + rank] = CV[static_cast<size_t>(r) * n + i];
    }
}

// ---- the drafter's MLX affine-4 group-64 products at up to 16 rows (tiles/wmma_verify.hpp's split tile, same bits) ----

// One matrix of a product: MLX words (n, ld / 8), bf16 scales and biases (n, ld / 64), its bf16 output (rows, n).
struct DftMat {
    const uint32_t* words;
    const uint16_t* scale;
    const uint16_t* bias;
    uint16_t* out;
    int n;
    int ld;  // a row's k as stored (zero codes past k, never read), 0: k
};

// X (m <= 16 rows, k a multiple of 256, rows ldx apart) bf16 against `a` (with PAIR, `a`'s column tiles then `b`'s).
struct DftMlx {
    const uint16_t* x;
    int m;
    int k;
    DftMat a;
    DftMat b;
    // with CONV, a.out = tf_dft_dyn_conv(product, dyn, base, res) on branch 1: the product's rows, `block` a stream
    const uint16_t* dyn;
    const uint16_t* base;
    const uint16_t* res;
    int group;
    int block;
    int ldx;  // 0: k
};

namespace {

using tf::rocm::u32x4;
using tf::rocm::u32x8;

// Codes 16 T .. 16 T + 15 of a 32-code chunk (4 words, low nibble first) as a WMMA B fragment: verify_frag<4, T>.
template <int T>
__device__ __forceinline__ bf16x16 dft_frag(const uint32_t (&w)[4]) {
    u32x8 f;
#pragma unroll
    for (int h = 0; h < 2; ++h) {
        const uint32_t n0 = w[2 * T + h] & 0x0f0f0f0fu;
        const uint32_t n1 = (w[2 * T + h] >> 4) & 0x0f0f0f0fu;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const uint32_t lo = __float_as_uint(static_cast<float>((n0 >> (8 * j)) & 0xffu));
            const uint32_t hi = __float_as_uint(static_cast<float>((n1 >> (8 * j)) & 0xffu));
            f[4 * h + j] = __builtin_amdgcn_perm(hi, lo, 0x07060302u);
        }
    }
    return __builtin_bit_cast(bf16x16, f);
}

}  // namespace

// The split tile's sums for one 16-column tile a block: per group a WMMA chain from zero over its 64 k, then the fold
// acc = fma(dot, scale, acc); acc = fma(sum_x, bias, acc) in k order (sum_x: the row's x over the group, lo then hi of
// each pair). Wave w makes trips (256 k) w, w + WV, ... and parks its sums in LDS, then every lane folds its elements.
template <int WV, bool CONV, bool PAIR>
__device__ __forceinline__ void dft_mlx_tile(const DftMlx& p) {
#if TF_DEVICE_WMMA_GFX11
    constexpr int GPT = 4;                  // groups a trip
    constexpr int EPL = 256 / (32 * WV);    // elements of the tile a lane folds
    __shared__ float dl[WV][GPT][8][32];
    __shared__ float sxl[WV][GPT][16];
    __shared__ uint32_t tl[WV][GPT][16];
    // CONV: the rounded product's tile, in dl's room once the folds are done (more LDS would cost a block a WGP)
    uint16_t (*ob)[16] = reinterpret_cast<uint16_t (*)[16]>(&dl[0][0][0][0]);
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int half = lane >> 4;
    const int live = p.m < 16 ? p.m : 16;
    const int K = p.k;
    const int ntrips = K >> 8;
    // PAIR: the first blocks a's tiles, the rest b's (two products of the same x in one launch)
    const int blocks_a = (p.a.n + 15) / 16;
    const bool second = PAIR && static_cast<int>(blockIdx.x) >= blocks_a;
    const int bx = second ? blockIdx.x - blocks_a : blockIdx.x;
    const DftMat& om = second ? p.b : p.a;
    const int col_s = bx * 16 + (lane & 15) < om.n ? bx * 16 + (lane & 15) : om.n - 1;
    const int ld = om.ld ? om.ld : K;
    const uint32_t* wp = om.words + static_cast<long long>(col_s) * (ld / 8);
    // lane half 0 reads the scales, half 1 the biases
    const uint32_t* sp = reinterpret_cast<const uint32_t*>((half ? om.bias : om.scale) + static_cast<long long>(col_s) * (ld / 64));
    const int xrow = (lane & 15) < live ? (lane & 15) : live - 1;
    const uint32_t* xg = reinterpret_cast<const uint32_t*>(p.x + static_cast<long long>(xrow) * (p.ldx ? p.ldx : K));

    float acc[EPL];
#pragma unroll
    for (int i = 0; i < EPL; ++i) acc[i] = 0.f;

    for (int ss = 0; ss < ntrips; ss += WV) {
        const int t = ss + wave;
        // every load on a clamped trip, so none is conditional
        const int tc = t < ntrips ? t : ntrips - 1;
        u32x4 xv[GPT][8];
        auto xload = [&](int q, u32x4 (&v)[8]) {
#pragma unroll
            for (int i = 0; i < 8; ++i) v[i] = *reinterpret_cast<const u32x4*>(xg + (tc * 256 + q * 64) / 2 + 4 * i);
        };
        // x before the codes: x sits in L2, the codes come from memory
#pragma unroll
        for (int q = 0; q < GPT; ++q) xload(q, xv[q]);
        u32x4 wr[4];
#pragma unroll
        for (int j = 0; j < 4; ++j) wr[j] = *reinterpret_cast<const u32x4*>(wp + static_cast<long long>(tc * 8 + 4 * half + j) * 4);
        const uint2 own = *reinterpret_cast<const uint2*>(sp + tc * 2);
        uint32_t g[GPT];
        {
            const uint32_t ow[2] = {own.x, own.y};
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                const uint32_t other = __builtin_amdgcn_permlanex16(ow[v], ow[v], 0x76543210u, 0xfedcba98u, false, false);
                const uint32_t sw = half ? other : ow[v];
                const uint32_t bw = half ? ow[v] : other;
                g[2 * v] = (sw & 0xffffu) | (bw << 16);
                g[2 * v + 1] = (sw >> 16) | (bw & 0xffff0000u);
            }
        }
#pragma unroll
        for (int q = 0; q < GPT; ++q) {
            u32x4 (&x)[8] = xv[q];
            float run = 0.f;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const uint32_t q4[4] = {x[v].x, x[v].y, x[v].z, x[v].w};
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    run += __uint_as_float(q4[h] << 16);
                    run += __uint_as_float(q4[h] & 0xffff0000u);
                }
            }
            f32x8 d = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
            for (int cc = 0; cc < 2; ++cc) {
                const int J = q * 2 + cc;
                const uint32_t mine[4] = {wr[J % 4].x, wr[J % 4].y, wr[J % 4].z, wr[J % 4].w};
                uint32_t w[4];
#pragma unroll
                for (int b = 0; b < 4; ++b) {
                    const uint32_t other = __builtin_amdgcn_permlanex16(mine[b], mine[b], 0x76543210u, 0xfedcba98u, false, false);
                    w[b] = half == J / 4 ? mine[b] : other;
                }
                const bf16x16 xa = __builtin_bit_cast(bf16x16, __builtin_shufflevector(x[4 * cc], x[4 * cc + 1], 0, 1, 2, 3, 4, 5, 6, 7));
                const bf16x16 xb = __builtin_bit_cast(bf16x16, __builtin_shufflevector(x[4 * cc + 2], x[4 * cc + 3], 0, 1, 2, 3, 4, 5, 6, 7));
                d = mma(xa, dft_frag<0>(w), d);
                d = mma(xb, dft_frag<1>(w), d);
            }
#pragma unroll
            for (int e = 0; e < 8; ++e) dl[wave][q][e][lane] = d[e];
            if (lane < 16) {
                sxl[wave][q][lane] = run;
                tl[wave][q][lane] = g[q];
            }
        }
        __syncthreads();
        const int trips = ntrips - ss < WV ? ntrips - ss : WV;
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            const int e = i * 32 * WV + wave * 32 + lane;
            const int row = e >> 4, cl = e & 15;
            if (row >= live) continue;
            float v = acc[i];
            for (int tt = 0; tt < trips; ++tt) {
#pragma unroll
                for (int q = 0; q < GPT; ++q) {
                    const uint32_t gq = tl[tt][q][cl];
                    v = fmaf(dl[tt][q][row >> 1][(row & 1) * 16 + cl], bf(static_cast<uint16_t>(gq & 0xffffu)), v);
                    v = fmaf(sxl[tt][q][row], bf(static_cast<uint16_t>(gq >> 16)), v);
                }
            }
            acc[i] = v;
        }
        __syncthreads();
    }

#pragma unroll
    for (int i = 0; i < EPL; ++i) {
        const int e = i * 32 * WV + wave * 32 + lane;
        const int row = e >> 4, cl = e & 15;
        const int col = bx * 16 + cl;
        if (row >= p.m) continue;
        if constexpr (CONV) {
            ob[row][cl] = tobf(acc[i]);
        } else if (col < om.n) {
            om.out[static_cast<long long>(row) * om.n + col] = tobf(acc[i]);
        }
    }
    if constexpr (CONV) {
        // tf_dft_dyn_conv's branch 1 with its residual over the rounded product, row t - 1 from the tile
        __syncthreads();
        const int D = p.a.n;
        const int groups = D / p.group;
#pragma unroll
        for (int i = 0; i < EPL; ++i) {
            const int e = i * 32 * WV + wave * 32 + lane;
            const int t = e >> 4, cl = e & 15;
            const int col = bx * 16 + cl;
            if (t >= p.m || col >= D) continue;
            const int g = col / p.group;
            const int lags = t % p.block == 0 ? 1 : 2;
            float total = 0.f;
            for (int l = 0; l < lags; l++) {
                const float xv = bf(ob[t - l][cl]);
                total = rbf(total + rbf(bf(p.base[(2 + l) * D + col]) * xv));
                total = rbf(total + rbf(bf(p.dyn[static_cast<size_t>(t) * (4 * groups) + (2 + l) * groups + g]) * xv));
            }
            total = rbf(total + bf(p.res[static_cast<size_t>(t) * D + col]));
            p.a.out[static_cast<size_t>(t) * D + col] = tobf(total);
        }
    }
#else
    (void)p;
    __builtin_trap();
#endif
}

extern "C" __global__ void __launch_bounds__(128) tf_dft_mlx(DftMlx p) { dft_mlx_tile<4, false, false>(p); }
extern "C" __global__ void __launch_bounds__(128) tf_dft_mlx_pair(DftMlx p) { dft_mlx_tile<4, false, true>(p); }
extern "C" __global__ void __launch_bounds__(128) tf_dft_mlx_conv(DftMlx p) { dft_mlx_tile<4, true, false>(p); }

// ---- DSpark's sequential Markov head over the whole vocabulary ----
// vLLM qwen3_dspark.py DSparkMarkovHead.bias + v1/worker/gpu/spec_decode/dspark/speculator.py _sample_sequential (no
// dspark_draft_topk), llama.cpp src/models/dflash.cpp build_dspark_markov_head: row i's draft is
// argmax_t L[s, i, t] + markov_w2[t] . markov_w1[prev], prev the block's anchor (i 0) or row i - 1's draft, the first
// (lowest) id of equal scores (torch argmax, ggml_argmax). The bias in fp32 as ggml's mul_mat; L the head's bf16 logits.

namespace {

// A score and its token as one max-ordered key: the score's ordered bits, then ~token (equal scores: the lower id).
__device__ __forceinline__ unsigned long long markov_key(float f, uint32_t t) {
    const uint32_t u = __float_as_uint(f);
    const uint32_t k = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
    return (static_cast<unsigned long long>(k) << 32) | static_cast<uint32_t>(~t);
}

__device__ __forceinline__ unsigned long long shfl_max64(unsigned long long v, int m) {
    const uint32_t lo = __shfl_xor(static_cast<uint32_t>(v), m, 32);
    const uint32_t hi = __shfl_xor(static_cast<uint32_t>(v >> 32), m, 32);
    const unsigned long long o = (static_cast<unsigned long long>(hi) << 32) | lo;
    return o > v ? o : v;
}

// Rank 256: lane l of a wave takes elements 32 (l % 8) .. + 31 of token 4 i + l / 8 (four tokens a wave a pass).
// W2 bf16 rows of 256 (Q4 false) or affine-4 group 64: words (V, 32) low nibble first, bf16 scales and biases (V, 4).
template <bool Q4>
__device__ __forceinline__ void markov_step(const uint16_t* L, int ldl, const uint16_t* W1, const void* W2, const uint16_t* W2s,
                                            const uint16_t* W2b, const int* ids, unsigned long long* keys, int V, int block,
                                            int step) {
    const int s = blockIdx.y, lane = threadIdx.x & 31, wave = threadIdx.x >> 5, j = lane & 7, q = lane >> 3;
    const int row = s * block + step;
    const uint32_t prev = step == 0 ? static_cast<uint32_t>(ids[s * block]) : ~static_cast<uint32_t>(keys[row - 1]);
    float e[32], esum = 0.f;
    {
        const uint4* p = reinterpret_cast<const uint4*>(W1 + static_cast<size_t>(prev) * 256 + 32 * j);
#pragma unroll
        for (int c = 0; c < 4; c++) {
            const uint4 v = p[c];
            const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int h = 0; h < 4; h++) {
                e[8 * c + 2 * h] = bf(static_cast<uint16_t>(w[h] & 0xffffu));
                e[8 * c + 2 * h + 1] = bf(static_cast<uint16_t>(w[h] >> 16));
            }
        }
#pragma unroll
        for (int m = 0; m < 32; m++) esum += e[m];
    }
    const uint16_t* lrow = L + static_cast<size_t>(row) * ldl;
    unsigned long long best = 0;
    const int first = blockIdx.x * 1024 + wave * 128;
    for (int it = 0; it < 32; it++) {
        const int t = first + 4 * it + q;
        float dot = 0.f;
        if (t < V) {
            if constexpr (Q4) {
                const uint4 v = reinterpret_cast<const uint4*>(static_cast<const uint32_t*>(W2) + static_cast<size_t>(t) * 32)[j];
                const uint32_t w[4] = {v.x, v.y, v.z, v.w};
                float acc = 0.f;
#pragma unroll
                for (int h = 0; h < 4; h++)
#pragma unroll
                    for (int i = 0; i < 8; i++) acc = fmaf(static_cast<float>((w[h] >> (4 * i)) & 15u), e[8 * h + i], acc);
                dot = fmaf(bf(W2s[static_cast<size_t>(t) * 4 + (j >> 1)]), acc, bf(W2b[static_cast<size_t>(t) * 4 + (j >> 1)]) * esum);
            } else {
                const uint4* p = reinterpret_cast<const uint4*>(static_cast<const uint16_t*>(W2) + static_cast<size_t>(t) * 256 + 32 * j);
#pragma unroll
                for (int c = 0; c < 4; c++) {
                    const uint4 v = p[c];
                    const uint32_t w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
                    for (int h = 0; h < 4; h++) {
                        dot = fmaf(bf(static_cast<uint16_t>(w[h] & 0xffffu)), e[8 * c + 2 * h], dot);
                        dot = fmaf(bf(static_cast<uint16_t>(w[h] >> 16)), e[8 * c + 2 * h + 1], dot);
                    }
                }
            }
        }
        dot += __shfl_xor(dot, 1, 32);
        dot += __shfl_xor(dot, 2, 32);
        dot += __shfl_xor(dot, 4, 32);
        if (t < V) {
            const float score = bf(lrow[t]) + dot;
            if (score == score) {
                const unsigned long long k = markov_key(score, static_cast<uint32_t>(t));
                best = k > best ? k : best;
            }
        }
    }
    best = shfl_max64(best, 8);
    best = shfl_max64(best, 16);
    __shared__ unsigned long long waves[8];
    if (lane == 0) waves[wave] = best;
    __syncthreads();
    if (threadIdx.x == 0) {
        for (int w = 1; w < 8; w++) best = waves[w] > best ? waves[w] : best;
        atomicMax(keys + row, best);
    }
}

}  // namespace

// Step `step` of every stream's chain: keys[s * block + step] max-reduced (zero before; tf_dft_markov_take zeroes).
// Grid (V / 1024 rounded up, streams), 256 threads.
extern "C" __global__ void __launch_bounds__(256) tf_dft_markov_step(const uint16_t* L, int ldl, const uint16_t* W1, const uint16_t* W2,
                                                                     const int* ids, unsigned long long* keys, int V, int block, int step) {
    markov_step<false>(L, ldl, W1, W2, nullptr, nullptr, ids, keys, V, block, step);
}
extern "C" __global__ void __launch_bounds__(256) tf_dft_markov_step_q4(const uint16_t* L, int ldl, const uint16_t* W1, const uint32_t* W2,
                                                                        const uint16_t* W2s, const uint16_t* W2b, const int* ids,
                                                                        unsigned long long* keys, int V, int block, int step) {
    markov_step<true>(L, ldl, W1, W2, W2s, W2b, ids, keys, V, block, step);
}

// The chains' tokens out of their keys, the keys zeroed for the next hold. Grid (n / 256 rounded up), 256 threads.
extern "C" __global__ void tf_dft_markov_take(unsigned long long* keys, int* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = static_cast<int>(~static_cast<uint32_t>(keys[i]));
    keys[i] = 0;
}
