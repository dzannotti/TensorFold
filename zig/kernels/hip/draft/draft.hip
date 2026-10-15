// The DFlash2 drafter's own kernels (families/qwen3_5/backend/hip/draft): tap rows, rings, dynamic conv, block attention.

#include <hip/hip_runtime.h>
#include <cstdint>

#include "common/arch.hpp"
#include "common/wmma.hpp"

namespace {

constexpr int HD = 128;  // the drafter's head width (q, k, v)

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

// Per-head RMS norm (fp32 weight) then rotate-half RoPE at `pos` of 128 values, 4 a lane; returns this lane's 4.
__device__ __forceinline__ void norm_rope4(const uint16_t* x, const float* w, float eps, float theta, long long pos, float out[4]) {
    const int t = threadIdx.x & 31;
    float v[4], acc = 0.f;
#pragma unroll
    for (int i = 0; i < 4; i++) {
        v[i] = bf(x[4 * t + i]);
        acc += v[i] * v[i];
    }
    acc = wave_sum(acc);
    const float inv = rsqrtf(acc / static_cast<float>(HD) + eps);
    float n[4];
#pragma unroll
    for (int i = 0; i < 4; i++) n[i] = rbf(v[i] * inv * w[4 * t + i]);
    // the partner of element d is d +- HD / 2: lane t's partners sit on lane t ^ 16
#pragma unroll
    for (int i = 0; i < 4; i++) {
        const int d = 4 * t + i;
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
// A layer's ring: keys (R, kv_heads, HD) then values transposed (kv_heads, HD, R). Grid (rows, layers * kv_heads), 32 threads.
extern "C" __global__ void __launch_bounds__(32) tf_dft_ring_put(const uint16_t* KV, long long stride, const DftRow* rows, const float* knorm,
                                                                 int kv_heads, int R, float eps, float theta) {
    const int r = blockIdx.x;
    const int layer = blockIdx.y / kv_heads, h = blockIdx.y % kv_heads;
    const int kvw = kv_heads * HD;
    const long long pos = rows[r].pos;
    const size_t slot = static_cast<size_t>(pos % R);
    uint16_t* ring = reinterpret_cast<uint16_t*>(rows[r].ring) + static_cast<size_t>(layer) * 2 * R * kvw;
    const uint16_t* k = KV + r * stride + static_cast<size_t>(layer) * 2 * kvw + h * HD;
    float out[4];
    norm_rope4(k, knorm + layer * HD, eps, theta, pos, out);
    const int t = threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; i++) ring[slot * kvw + h * HD + 4 * t + i] = tobf(out[i]);
    const uint16_t* v = k + kvw;
    uint16_t* vt = ring + static_cast<size_t>(R) * kvw + static_cast<size_t>(h) * HD * R;
#pragma unroll
    for (int i = 0; i < 4; i++) vt[static_cast<size_t>(4 * t + i) * R + slot] = v[4 * t + i];
}

// Block rows' q or k heads normed and rotated at desc[row / block].pos0 + row % block; OUT packed (rows, heads, HD).
// Grid (rows * heads), 32 threads.
extern "C" __global__ void __launch_bounds__(32) tf_dft_norm_rope(const uint16_t* X, long long stride, const float* W, uint16_t* OUT,
                                                                  const DftDesc* desc, int block, int heads, float eps, float theta) {
    const int head = blockIdx.x % heads;
    const int row = blockIdx.x / heads;
    const long long pos = desc[row / block].pos0 + row % block;
    float out[4];
    norm_rope4(X + row * stride + head * HD, W, eps, theta, pos, out);
    uint16_t* o = OUT + (static_cast<size_t>(row) * heads + head) * HD + 4 * threadIdx.x;
#pragma unroll
    for (int i = 0; i < 4; i++) o[i] = tobf(out[i]);
}

// y[t, c] = sum over lags l of (BASE[l, c] + DYN[t, l, c / group]) * x[t - l, c] in bf16 steps, x before a block's first row 0;
// RES (when given) added last. Grid (rows), 256 threads.
extern "C" __global__ void tf_dft_dyn_conv(const uint16_t* X, const uint16_t* DYN, const uint16_t* BASE, const uint16_t* RES, uint16_t* OUT,
                                           int D, int group, int branch, int block) {
    const int t = blockIdx.x;
    const int groups = D / group;
    const int lags = t % block == 0 ? 1 : 2;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
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
constexpr int kQueries = 32;  // a stream's block rows times the query heads of a key head: two waves of 16

}  // namespace

// Block attention, split over the keys: query (m, g) of a stream's key head sees ring positions
// [max(begin, pos0 + m - window + 1), pos0) and every block row. Split y of `splits` walks its share of 16-position tiles,
// writing unnormalized PART (splits, 32 queries, HD) fp32 and STATS (max, sum) per (stream, key head).
// Q (rows, heads, HD), KB (rows, kv_heads, HD) packed; VB rows `vstride` apart. Grid (streams * kv_heads, splits), 64 threads.
extern "C" __global__ void __launch_bounds__(64) tf_dft_attn(const uint16_t* Q, const uint16_t* KB, const uint16_t* VB, long long vstride,
                                                             const DftDesc* desc, int layer, int block, int heads, int kv_heads, int R,
                                                             int window, float* PART, float* STATS) {
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
    f32x8 o[HD / 16];
#pragma unroll
    for (int i = 0; i < HD / 16; i++) o[i] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
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
        for (int i = 0; i < HD / 16; i++) {
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
        step(s, [&](int i) { return load16(RVT + static_cast<size_t>(16 * i + col) * R + slot0); });
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
            for (int k = 0; k < 16; k++) vv[k] = k < block ? vb[k * vstride + 16 * i + col] : 0;
            return __builtin_bit_cast(bf16x16, vv);
        });
    }
    // O^T's tile i: this lane's query, dims 16 i + 2 e + half
    const size_t at = (static_cast<size_t>(blockIdx.x) * splits + split) * kQueries + q;
    float* part = PART + at * HD;
#pragma unroll
    for (int i = 0; i < HD / 16; i++) {
#pragma unroll
        for (int e = 0; e < 8; e++) part[16 * i + 2 * e + half] = o[i][e];
    }
    if (half == 0) {
        STATS[2 * at] = mx;
        STATS[2 * at + 1] = l;
    }
}

// The splits of tf_dft_attn joined: OUT (rows, heads, HD) bf16. Grid (streams * kv_heads, 32 queries), HD threads.
extern "C" __global__ void tf_dft_attn_merge(const float* PART, const float* STATS, int splits, int block, int heads, int kv_heads,
                                             uint16_t* OUT) {
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
    OUT[(static_cast<size_t>(stream * block + m) * heads + head) * HD + dim] = tobf(acc / l);
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
