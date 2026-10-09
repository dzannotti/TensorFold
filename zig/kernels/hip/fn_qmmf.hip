// Block FP8 (FP8G: e4m3 weights in fn_qmmf.cu's fragment order, an fp32 scale per 64 inputs and column) on gfx1151:
// a rewrite of fn_qmmf.cu's qmmf_kernel<3, BM, 64, 1, 4, 4, F32, CLUSTER, FUSE> for RDNA3 WMMA (wave32, bf16 16x16x16).
// Each 64 inputs' bf16 products sum in fp32 (four WMMA steps from zero), scaled into the slice's fp32 sum in group
// order; slices add left to right (sum = s0 + s1 + ...). FUSE: a wave runs the slices one after another; otherwise
// the block holds SK slices side by side (128 * SK threads) and adds them in LDS in the same order: the same bits.
// A row's bits depend on nothing but its own inputs (no M-dependent order, no atomics). TF_QMMF_LD: rows `ldo` apart.
// qmmw_kernel (from 33 rows): the same arithmetic per output on 64/128-row tiles staged in LDS, so the same bits.
#include <hip/hip_bf16.h>
#include <hip/hip_runtime.h>
#include <stdint.h>

#ifdef TF_QMMF_LD
#define TF_NS tf_fn_qmmf_ld
#else
#define TF_NS tf_fn_qmmf
#endif

namespace TF_NS {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));

constexpr int GS = 64;                                    // inputs a group (one fp32 scale a column)
constexpr int MAX_SLICES = 4;                             // the host's split_k on gfx1151 (<= 512 threads)

// two e4m3 codes (the low 16 bits of `w`) -> two bf16 in one word, exactly: as fp16 (bias 15 against 7, e4m3
// subnormals fp16 subnormals), to fp32 and times 2^8; fp32 denormals never occur, so their flush mode does not matter
__device__ __forceinline__ uint32_t fp8pair(uint32_t w) {
    const uint32_t x = __builtin_amdgcn_perm(0u, w, 0x010c000cu);   // byte 0 -> bits 8-15, byte 1 -> bits 24-31
    union { uint32_t u; _Float16 h[2]; } t;
    t.u = ((x >> 1) & 0x3F803F80u) | (x & 0x80008000u);
    const float lo = static_cast<float>(t.h[0]) * 256.0f, hi = static_cast<float>(t.h[1]) * 256.0f;
    return __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo), 0x07060302u);
}

// a lane's 16 weight bytes of one WMMA step -> its B fragment (16 bf16, the column's inputs of that step)
__device__ __forceinline__ v16s bfrag(const uint4& b) {
    union { uint32_t u[8]; v16s v; } r;
    const uint32_t c[4] = {b.x, b.y, b.z, b.w};
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        r.u[2 * i] = fp8pair(c[i]);
        r.u[2 * i + 1] = fp8pair(c[i] >> 16);
    }
    return r.v;
}

// step q's A fragment: the row's input pairs q, q + 4, ..., q + 28 of the group (the weight bytes' order)
__device__ __forceinline__ v16s afrag(const uint4 (&a)[8], int q) {
    union { uint32_t u[8]; v16s v; } r;
#pragma unroll
    for (int i = 0; i < 8; ++i) r.u[i] = q == 0 ? a[i].x : q == 1 ? a[i].y : q == 2 ? a[i].z : a[i].w;
    return r.v;
}

// 16 weight bytes; NT: a nontemporal load (decode tiles read each byte once: +8-12% GB/s at 1-32 rows, notes/fp8dec.md)
template <bool NT>
__device__ __forceinline__ uint4 wload(const unsigned char* p) {
    if constexpr (!NT) return *reinterpret_cast<const uint4*>(p);
    const u4v v = __builtin_nontemporal_load(reinterpret_cast<const u4v*>(p));
    return make_uint4(v[0], v[1], v[2], v[3]);
}

__device__ __forceinline__ int2 tile_of(int b, int M, int N, int BM, int BN, int group) {
    const int rows_t = (M + BM - 1) / BM, cols_t = (N + BN - 1) / BN, band = group * cols_t;
    const int first = b / band * group, in_band = b % band, height = min(group, rows_t - first);
    return make_int2((first + in_band % height) * BM, in_band / height * BN);
}

template <int MODE, int BM, int BN, int WM, int WN, int STAGES, bool F32, bool CLUSTER, bool FUSE = false>
__global__ void __launch_bounds__(128 * MAX_SLICES) qmmf_kernel(
        const __hip_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group
#ifdef TF_QMMF_LD
        , int ldo
#endif
        ) {
    static_assert(MODE == 3 && BN == 64 && WN == 4 && !CLUSTER, "gfx1151: block FP8 on 64-column tiles, no clusters");
    constexpr int MT = BM / 16, PA = MT <= 2 ? MT : 0;   // row tiles; PA > 0: the next group's loads in flight
#ifndef TF_QMMF_LD
    const int ldo = N;
#endif
    __shared__ float red[FUSE ? 1 : 4 * MT * 8 * 32];     // a slice's sums, parked for slice 0
    const int lane = threadIdx.x & 31, wave = threadIdx.x >> 5, wn = wave & 3;
    const int KG = K / GS, per = KG / SK, slice = FUSE ? 0 : wave >> 2;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y, c = wn * 16 + (lane & 15);    // the lane's column within the tile
    const unsigned char* wc = w + static_cast<size_t>(n0 / 64) * KG * 4096 + (c / 8) * 512 + (c % 8) * 64;
    const float* sc = reinterpret_cast<const float*>(bs) + static_cast<size_t>(n0 / 64) * KG * 64 + c;
    const __hip_bfloat16* xr[MT];
#pragma unroll
    for (int i = 0; i < MT; ++i) xr[i] = x + static_cast<size_t>(min(m0 + i * 16 + (lane & 15), M - 1)) * ldx;

    v8f acc[MT], tot[MT];
    for (int s = FUSE ? 0 : slice; s < (FUSE ? SK : slice + 1); ++s) {
#pragma unroll
        for (int i = 0; i < MT; ++i) acc[i] = v8f{};
        const int g1 = (s + 1) * per;
        uint4 wn_[4], an[PA > 0 ? PA : 1][8];                          // the next group's words (and rows, PA tiles), in flight
        float sn;
        auto fetch = [&](int g) {
#pragma unroll
            for (int q = 0; q < 4; ++q) wn_[q] = wload<(PA > 0)>(wc + static_cast<size_t>(g) * 4096 + q * 16);
            sn = sc[static_cast<size_t>(g) * 64];
#pragma unroll
            for (int i = 0; i < PA; ++i)
#pragma unroll
                for (int u = 0; u < 8; ++u) an[i][u] = reinterpret_cast<const uint4*>(xr[i] + g * GS)[u];
        };
        if (PA > 0) fetch(s * per);
        for (int g = s * per; g < g1; ++g) {
            if (PA == 0) fetch(g);                        // 64-row tiles: no registers to spare
            v16s b[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) b[q] = bfrag(wn_[q]);
            const float sg = sn;
            uint4 ac[PA > 0 ? PA : 1][8];
#pragma unroll
            for (int i = 0; i < PA; ++i)
#pragma unroll
                for (int u = 0; u < 8; ++u) ac[i][u] = an[i][u];
            if (PA > 0 && g + 1 < g1) fetch(g + 1);
#pragma unroll
            for (int i = 0; i < MT; ++i) {
                uint4 a[8];
#pragma unroll
                for (int u = 0; u < 8; ++u) a[u] = i < PA ? ac[i < PA ? i : 0][u] : reinterpret_cast<const uint4*>(xr[i] + g * GS)[u];
                v8f d = v8f{};
#pragma unroll
                for (int q = 0; q < 4; ++q) d = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(afrag(a, q), b[q], d);
#pragma unroll
                for (int e = 0; e < 8; ++e) acc[i][e] = __builtin_fmaf(d[e], sg, acc[i][e]);
            }
        }
#pragma unroll
        for (int i = 0; i < MT; ++i) tot[i] = s == 0 ? acc[i] : tot[i] + acc[i];
    }
    if constexpr (!FUSE) {                                // slices 1.. add into slice 0's sum, in slice order
        for (int s = 1; s < SK; ++s) {
            if (slice == s)
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int e = 0; e < 8; ++e) red[((wn * MT + i) * 8 + e) * 32 + lane] = tot[i][e];
            __syncthreads();
            if (slice == 0)
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int e = 0; e < 8; ++e) tot[i][e] = tot[i][e] + red[((wn * MT + i) * 8 + e) * 32 + lane];
            __syncthreads();
        }
        if (slice != 0) return;
    }
    const int col = n0 + c;
    if (col >= N) return;
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int e = 0; e < 8; ++e) {                     // WMMA D: lane column, rows 2e + lane / 16
            const int row = m0 + i * 16 + 2 * e + (lane >> 4);
            if (row >= M) continue;
            const float v = tot[i][e] * scale;
            if (F32) reinterpret_cast<float*>(out)[static_cast<size_t>(row) * ldo + col] = v;
            else reinterpret_cast<__hip_bfloat16*>(out)[static_cast<size_t>(row) * ldo + col] = __float2bfloat16(v);
        }
}


// The wide tile: BM x BN outputs a block of WR x WC waves (each MT x NT WMMA tiles; 2 BM threads), the group's inputs
// and weights staged in LDS (weights converted to bf16 there), double-buffered through registers, one barrier a group;
// CU mode (the block's LDS on one CU: +10-30% over WGP mode). With K slices (SPLIT) 64 columns: tot needs registers.
// Each output's arithmetic is qmmf_kernel's: the same fragments, the 4-step chain from zero, the fma into the slice's
// sum and the slices added left to right, so a row's bits equal every other form's.
template <int BM, int BN> struct Wide;
template <> struct Wide<128, 128> { static constexpr int WR = 2, WC = 4, MT = 4, NT = 2; static constexpr bool SPLIT = false; };
template <> struct Wide<128, 64> { static constexpr int WR = 4, WC = 2, MT = 2, NT = 2; static constexpr bool SPLIT = true; };
template <> struct Wide<64, 128> { static constexpr int WR = 1, WC = 4, MT = 4, NT = 2; static constexpr bool SPLIT = false; };
template <> struct Wide<64, 64> { static constexpr int WR = 2, WC = 2, MT = 2, NT = 2; static constexpr bool SPLIT = true; };

// a 128-byte LDS row (a row's or a column's group: 8 chunks, chunks 2q and 2q + 1 step q's fragment), swizzled so
// 16 lanes reading one chunk of 16 rows hit 64 distinct banks
__device__ __forceinline__ int swz(int r, int k) { return r * 8 + (k ^ ((r >> 1) & 7)); }

// a barrier ordering LDS only (__syncthreads' fence also waits for the global loads in flight)
__device__ __forceinline__ void lds_barrier() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

template <int BM, int BN, bool F32>
__global__ void __launch_bounds__(256) __attribute__((target("cumode"))) qmmw_kernel(
        const __hip_bfloat16* __restrict__ x, const unsigned char* __restrict__ w, const uint8_t* __restrict__ bs,
        float scale, void* __restrict__ out, float* __restrict__ part, int M, int N, int K, int SK, int npad, int ldx,
        int group
#ifdef TF_QMMF_LD
        , int ldo
#endif
        ) {
    using C = Wide<BM, BN>;
    constexpr int WR = C::WR, WC = C::WC, MT = C::MT, NT = C::NT, T = 32 * WR * WC, BU = BN * 4 / T;
    static_assert(BM * 8 / T == 4 && BU >= 1, "a thread stages 4 input words and BU weight words");
#ifndef TF_QMMF_LD
    const int ldo = N;
#endif
    __shared__ uint4 lds[2][(BM + BN) * 8];
    const int t = threadIdx.x, lane = t & 31, wave = t >> 5, wr = wave / WC, wc = wave % WC;
    const int KG = K / GS, per = KG / SK;
    const int2 at = tile_of(blockIdx.x, M, N, BM, BN, group);
    const int m0 = at.x, n0 = at.y;
    // staging: thread t holds row t / 2's input words 4 (t % 2) .. + 3 and weight words t * BU .. (column, step)
    const __hip_bfloat16* xs = x + static_cast<size_t>(min(m0 + t / 2, M - 1)) * ldx + (t % 2) * 32;
    const unsigned char* ws[BU];
#pragma unroll
    for (int p = 0; p < BU; ++p) {
        const int e = t * BU + p, col = n0 + e / 4;
        ws[p] = w + static_cast<size_t>(col / 64) * KG * 4096 + (col % 64) * 64 + (e % 4) * 16;
    }
    const float* sc[NT];                                  // the lane's columns' scales
#pragma unroll
    for (int j = 0; j < NT; ++j) {
        const int col = n0 + (wc * NT + j) * 16 + (lane & 15);
        sc[j] = reinterpret_cast<const float*>(bs) + static_cast<size_t>(col / 64) * KG * 64 + col % 64;
    }
    uint4 ra[4], rb[BU];
    float sn[NT], sg[NT];
    auto fetch = [&](int g) {
#pragma unroll
        for (int p = 0; p < 4; ++p) ra[p] = reinterpret_cast<const uint4*>(xs + g * GS)[p];
#pragma unroll
        for (int p = 0; p < BU; ++p) rb[p] = *reinterpret_cast<const uint4*>(ws[p] + static_cast<size_t>(g) * 4096);
#pragma unroll
        for (int j = 0; j < NT; ++j) sn[j] = sc[j][static_cast<size_t>(g) * 64];
    };
    auto stage = [&](uint4* buf) {                        // inputs in fragment order, weights as bf16
        const int r = t / 2, h = t % 2;
#pragma unroll
        for (int q = 0; q < 4; ++q) {
            uint4 v;
            v.x = q == 0 ? ra[0].x : q == 1 ? ra[0].y : q == 2 ? ra[0].z : ra[0].w;
            v.y = q == 0 ? ra[1].x : q == 1 ? ra[1].y : q == 2 ? ra[1].z : ra[1].w;
            v.z = q == 0 ? ra[2].x : q == 1 ? ra[2].y : q == 2 ? ra[2].z : ra[2].w;
            v.w = q == 0 ? ra[3].x : q == 1 ? ra[3].y : q == 2 ? ra[3].z : ra[3].w;
            buf[swz(r, 2 * q + h)] = v;
        }
#pragma unroll
        for (int p = 0; p < BU; ++p) {
            const int e = t * BU + p, r = BM + e / 4, q = e % 4;
            union { v16s v; uint4 u[2]; } b;
            b.v = bfrag(rb[p]);
            buf[swz(r, 2 * q)] = b.u[0];
            buf[swz(r, 2 * q + 1)] = b.u[1];
        }
#pragma unroll
        for (int j = 0; j < NT; ++j) sg[j] = sn[j];
    };
    auto frag = [&](const uint4* buf, int r, int q) {
        union { v16s v; uint4 u[2]; } f;
        f.u[0] = buf[swz(r, 2 * q)];
        f.u[1] = buf[swz(r, 2 * q + 1)];
        return f.v;
    };

    v8f acc[MT][NT], tot[C::SPLIT ? MT : 1][C::SPLIT ? NT : 1];
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) acc[i][j] = v8f{};
    fetch(0);
    stage(lds[0]);
    lds_barrier();
    for (int g = 0; g < KG; ++g) {
        if (g + 1 < KG) fetch(g + 1);
        const uint4* buf = lds[g & 1];
        v16s bf[NT][4], af[2][4];                         // the group's weight fragments; a row tile's, the next in flight
#pragma unroll
        for (int i = 0; i < MT; i += 2) {                // two row tiles at a time: 2 NT independent chains
#pragma unroll
            for (int q = 0; q < 4; ++q) {                 // in the order the steps use them
                if (i == 0)
#pragma unroll
                    for (int j = 0; j < NT; ++j) bf[j][q] = frag(buf, BM + (wc * NT + j) * 16 + (lane & 15), q);
#pragma unroll
                for (int h = 0; h < 2; ++h) af[h][q] = frag(buf, (wr * MT + i + h) * 16 + (lane & 15), q);
            }
            v8f d[2][NT];
#pragma unroll
            for (int q = 0; q < 4; ++q)
#pragma unroll
                for (int h = 0; h < 2; ++h)
#pragma unroll
                    for (int j = 0; j < NT; ++j)
                        d[h][j] = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32(af[h][q], bf[j][q], q == 0 ? v8f{} : d[h][j]);
#pragma unroll
            for (int h = 0; h < 2; ++h)
#pragma unroll
                for (int j = 0; j < NT; ++j)
#pragma unroll
                    for (int e = 0; e < 8; ++e) acc[i + h][j][e] = __builtin_fmaf(d[h][j][e], sg[j], acc[i + h][j][e]);
        }
        if (g + 1 < KG) stage(lds[(g + 1) & 1]);
        lds_barrier();
        if constexpr (C::SPLIT) {
            if ((g + 1) % per == 0) {                     // a slice's end: add into the sum, in slice order
#pragma unroll
                for (int i = 0; i < MT; ++i)
#pragma unroll
                    for (int j = 0; j < NT; ++j) {
                        tot[i][j] = g + 1 == per ? acc[i][j] : tot[i][j] + acc[i][j];
                        acc[i][j] = v8f{};
                    }
            }
        }
    }
#pragma unroll
    for (int i = 0; i < MT; ++i)
#pragma unroll
        for (int j = 0; j < NT; ++j) {
            const int col = n0 + (wc * NT + j) * 16 + (lane & 15);
            if (col >= N) continue;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                const int row = m0 + (wr * MT + i) * 16 + 2 * e + (lane >> 4);
                if (row >= M) continue;
                const float v = (C::SPLIT ? tot[C::SPLIT ? i : 0][C::SPLIT ? j : 0][e] : acc[i][j][e]) * scale;
                if (F32) reinterpret_cast<float*>(out)[static_cast<size_t>(row) * ldo + col] = v;
                else reinterpret_cast<__hip_bfloat16*>(out)[static_cast<size_t>(row) * ldo + col] = __float2bfloat16(v);
            }
        }
}

} // namespace TF_NS

// fn_qmmf.cu's instantiations without the cluster ones (gfx1151 has none; its uninstantiated split-K reduce_kernel
// is not needed either): row tiles 16, 32, 64 and the fused form
#ifdef TF_QMMF_LD
#define TF_QMMF(BM, FUSE) template __global__ void tf_fn_qmmf_ld::qmmf_kernel<3, BM, 64, 1, 4, 4, false, false, \
    FUSE>(const __hip_bfloat16*, const unsigned char*, const uint8_t*, float, void*, float*, int, int, int, int, int, \
    int, int, int);
TF_QMMF(16, false)
TF_QMMF(32, false)
TF_QMMF(64, false)
TF_QMMF(64, true)
#define TF_QMMW(BM, BN) template __global__ void tf_fn_qmmf_ld::qmmw_kernel<BM, BN, false>(const __hip_bfloat16*, \
    const unsigned char*, const uint8_t*, float, void*, float*, int, int, int, int, int, int, int, int);
TF_QMMW(128, 128)
TF_QMMW(128, 64)
TF_QMMW(64, 128)
TF_QMMW(64, 64)
#else
#define TF_QMMF(BM, F32, FUSE) template __global__ void tf_fn_qmmf::qmmf_kernel<3, BM, 64, 1, 4, 4, F32, false, \
    FUSE>(const __hip_bfloat16*, const unsigned char*, const uint8_t*, float, void*, float*, int, int, int, int, int, \
    int, int);
TF_QMMF(16, false, false)
TF_QMMF(32, false, false)
TF_QMMF(64, false, false)
TF_QMMF(16, true, false)
TF_QMMF(32, true, false)
TF_QMMF(64, true, false)
TF_QMMF(64, false, true)
TF_QMMF(64, true, true)
#define TF_QMMW(BM, BN, F32) template __global__ void tf_fn_qmmf::qmmw_kernel<BM, BN, F32>(const __hip_bfloat16*, \
    const unsigned char*, const uint8_t*, float, void*, float*, int, int, int, int, int, int, int);
TF_QMMW(128, 128, false)
TF_QMMW(128, 64, false)
TF_QMMW(64, 128, false)
TF_QMMW(64, 64, false)
TF_QMMW(128, 128, true)
TF_QMMW(128, 64, true)
TF_QMMW(64, 128, true)
TF_QMMW(64, 64, true)
#endif
