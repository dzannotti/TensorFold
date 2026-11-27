#pragma once

// qmmf (zig/kernels/cuda/qmmf.cu) on gfx11: rows in buckets of 16/32/64 share one WMMA, weights read once a launch.
// Deviations: 16x16x16 fragments read straight from memory (no smem stage), a wave is 16 columns x one K slice,
// K slices join through LDS in slice order, the e8m0 block scale is folded into the exact bf16 weight.

#include "common/wmma.hpp"
#include "quant/fp4local_bake.hpp"
#include "quant/mlx.hpp"
#include "tiles/epilogue.hpp"

namespace tf {
namespace rocm {

struct QmmfSides {
    unsigned long long out[4];
    int cuts[3];
    int count;
};

template <int MODE, int MT>
__device__ __forceinline__ void qmmf_cols(Affine a, int sk, QmmfSides qs) {
    constexpr int U = MT == 4 ? 2 : 4;
#if TF_DEVICE_WMMA_GFX11
    __shared__ float part[8][MT * 8][32];
    const int lane = threadIdx.x & 31;
    const int wave = __builtin_amdgcn_readfirstlane(threadIdx.x >> 5);
    const int cgl = sk == 1 ? 3 : sk == 2 ? 2 : sk == 4 ? 1 : 0;
    const int cg = 1 << cgl;
    const int slice = wave >> cgl, cgi = wave & (cg - 1);
    const int cl = lane & 15, rh = lane >> 4;
    const int n0 = (blockIdx.x * cg + cgi) * 16, m0 = blockIdx.y * (16 * MT);
    const int col = n0 + cl < a.n ? n0 + cl : a.n - 1;
    const int kts = (a.k >> 4) >> (3 - cgl);
    const int kt0 = slice * kts;
    const uint32_t* xr[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        const int row = m0 + mt * 16 + cl;
        xr[mt] = reinterpret_cast<const uint32_t*>(static_cast<const __bf16*>(a.x) + static_cast<long long>(row < a.m ? row : a.m - 1) * a.k);
    }
    const uint8_t* w8 = reinterpret_cast<const uint8_t*>(a.words) + static_cast<long long>(col) * a.k;
    const uint32_t* w16 = reinterpret_cast<const uint32_t*>(reinterpret_cast<const __bf16*>(a.words) + static_cast<long long>(col) * a.k);
    const uint8_t* e = static_cast<const uint8_t*>(a.scale.p) + static_cast<long long>(col) * (a.k >> 5);
    f32x8 acc[MT];
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) acc[mt] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    for (int kt = kt0; kt < kt0 + kts; kt += U) {
        bf16x16 wf[U];
        bf16x16 xf[MT][U];
        u32x4 raw[U];
        float mult[U];
        if constexpr (MODE == 1) {
#pragma unroll
            for (int u = 0; u < U; u += 2) raw[u] = *reinterpret_cast<const u32x4*>(w8 + (kt + u + rh) * 16);
#pragma unroll
            for (int u = 0; u < U; u += 2) mult[u] = mult[u + 1] = mxfp8_multiplier(e[(kt + u) >> 1]);
        } else {
#pragma unroll
            for (int u = 0; u < U; ++u) wf[u] = frag16(w16 + (kt + u) * 8);
        }
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            if (m0 + mt * 16 >= a.m) continue;
#pragma unroll
            for (int u = 0; u < U; ++u) xf[mt][u] = frag16(xr[mt] + (kt + u) * 8);
        }
        if constexpr (MODE == 1) {
            // a lane half decodes one fragment of the pair; permlanex16 hands each to the other half
#pragma unroll
            for (int u = 0; u < U; u += 2) {
                const uint32_t q[4] = {raw[u].x, raw[u].y, raw[u].z, raw[u].w};
                uint32_t p[8], lo[8], hi[8];
#pragma unroll
                for (int i = 0; i < 4; ++i) {
                    p[2 * i] = mxfp8_pair(q[i], 0, 1.f);
                    p[2 * i + 1] = mxfp8_pair(q[i], 2, 1.f);
                }
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const uint32_t other = __builtin_amdgcn_permlanex16(p[i], p[i], 0x76543210u, 0xfedcba98u, false, false);
                    lo[i] = rh ? other : p[i];
                    hi[i] = rh ? p[i] : other;
                }
                wf[u] = __builtin_bit_cast(bf16x16, u32x8{lo[0], lo[1], lo[2], lo[3], lo[4], lo[5], lo[6], lo[7]});
                wf[u + 1] = __builtin_bit_cast(bf16x16, u32x8{hi[0], hi[1], hi[2], hi[3], hi[4], hi[5], hi[6], hi[7]});
            }
        }
        if constexpr (MODE == 1) {
            // qmmf MXFP8: a 32-input block from zero (mma0, mma), then fma(d, block scale, acc)
#pragma unroll
            for (int u = 0; u < U; u += 2)
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (m0 + mt * 16 >= a.m) continue;
                    f32x8 d = WmmaBF16::mma(xf[mt][u], wf[u], f32x8{0, 0, 0, 0, 0, 0, 0, 0});
                    d = WmmaBF16::mma(xf[mt][u + 1], wf[u + 1], d);
#pragma unroll
                    for (int i = 0; i < 8; ++i) acc[mt][i] = fmaf(d[i], mult[u], acc[mt][i]);
                }
        } else {
#pragma unroll
            for (int u = 0; u < U; ++u)
#pragma unroll
                for (int mt = 0; mt < MT; ++mt) {
                    if (m0 + mt * 16 >= a.m) continue;
                    acc[mt] = WmmaBF16::mma(xf[mt][u], wf[u], acc[mt]);
                }
        }
    }
    if (sk > 1) {
#pragma unroll
        for (int mt = 0; mt < MT; ++mt)
#pragma unroll
            for (int i = 0; i < 8; ++i) part[wave][mt * 8 + i][lane] = acc[mt][i];
        __syncthreads();
        if (slice != 0) return;
        for (int s = 1; s < sk; ++s)
#pragma unroll
            for (int mt = 0; mt < MT; ++mt)
#pragma unroll
                for (int i = 0; i < 8; ++i) acc[mt][i] = acc[mt][i] + part[s * cg + cgi][mt * 8 + i][lane];
    }
    if (n0 + cl >= a.n) return;
#pragma unroll
    for (int mt = 0; mt < MT; ++mt)
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + mt * 16 + 2 * i + rh;
            if (row >= a.m) continue;
            if (qs.count == 0) {
                GemmOut<WmmaBF16>::store(a, row, n0 + cl, acc[mt][i]);
            } else {
                const int c = n0 + cl;
                const int sd = (c >= qs.cuts[0]) + (c >= qs.cuts[1]) + (c >= qs.cuts[2]);
                const int base = sd == 0 ? 0 : qs.cuts[sd - 1];
                const int width = (sd < 3 ? qs.cuts[sd] : a.n) - base;
                reinterpret_cast<__bf16*>(qs.out[sd])[static_cast<long long>(row) * width + (c - base)] = narrow<WmmaBF16>(acc[mt][i]);
            }
        }
#else
    (void)a, (void)sk, (void)qs;
    __builtin_trap();
#endif
}

#define TF_QMMF(NAME, MODE, MT) \
    extern "C" __global__ void __launch_bounds__(256) NAME(Affine a, int sk, QmmfSides qs) { qmmf_cols<MODE, MT>(a, sk, qs); }
TF_QMMF(tf_qmmf_mx_16, 1, 1)
TF_QMMF(tf_qmmf_mx_32, 1, 2)
TF_QMMF(tf_qmmf_mx_64, 1, 4)
TF_QMMF(tf_qmmf_bf_16, 2, 1)
TF_QMMF(tf_qmmf_bf_32, 2, 2)
TF_QMMF(tf_qmmf_bf_64, 2, 4)
#undef TF_QMMF

}  // namespace rocm
}  // namespace tf
