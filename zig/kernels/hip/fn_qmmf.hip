// Block FP8 (FP8G: e4m3 weights in fn_qmmf.cu's fragment order, an fp32 scale per 64 inputs and column) on gfx1151:
// a rewrite of fn_qmmf.cu's qmmf_kernel<3, BM, 64, 1, 4, 4, F32, CLUSTER, FUSE> for RDNA3 WMMA (wave32, bf16 16x16x16).
// Each 64 inputs' bf16 products sum in fp32 (four WMMA steps from zero), scaled into the slice's fp32 sum in group
// order; slices add left to right (sum = s0 + s1 + ...). FUSE: a wave runs the slices one after another; otherwise
// the block holds SK slices side by side (128 * SK threads) and adds them in LDS in the same order: the same bits.
// A row's bits depend on nothing but its own inputs (no M-dependent order, no atomics). TF_QMMF_LD: rows `ldo` apart.
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

constexpr int GS = 64;                                    // inputs a group (one fp32 scale a column)
constexpr int MAX_SLICES = 4;                             // the host's split_k on gfx1151 (<= 512 threads)

// two e4m3 codes (the low 16 bits of `w`) -> two bf16 in one word, exactly (fp32 denormals kept for e4m3 subnormals)
__device__ __forceinline__ uint32_t fp8pair(uint32_t w) {
    const float lo = __uint_as_float(((w & 0x7Fu) << 20) | ((w & 0x80u) << 24)) * 0x1p120f;
    const float hi = __uint_as_float(((w & 0x7F00u) << 12) | ((w & 0x8000u) << 16)) * 0x1p120f;
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
            for (int q = 0; q < 4; ++q) wn_[q] = *reinterpret_cast<const uint4*>(wc + static_cast<size_t>(g) * 4096 + q * 16);
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
#endif
