#pragma once

// gfx11 decode round's routed experts on the matrix core: a wave is (item, 16 columns), x rows of the item's pairs (16 a
// pass) are the A fragment, the expert's baked NVFP4 words decoded to exact bf16 in registers are the B fragment, so an
// expert is read once an item. nvfp4_expert_waves (kernels/cuda/nvfp4_experts.cu) with v_wmma_f32_16x16x16_bf16: one
// m16n8k16 is two WMMA halves, no cp.async ring (loads are issued four chunks ahead in registers), the scale folded into
// the weights as the baked format does, not applied to the mma result.

#include "common/wmma.hpp"
#include "quant/nvfp4_decoder.hpp"
#include "tiles/epilogue.hpp"

namespace tf {
namespace rocm {

constexpr int kExpertWaves = 4;
constexpr int kExpertAhead = 4;

#if TF_DEVICE_WMMA_GFX11
// Weights 16 j .. 16 j + 15 of a chunk as the B fragment: words wa, wb of the chunk, group scale g.
__device__ __forceinline__ bf16x16 expert_frag(uint32_t wa, uint32_t wb, __half2 g) {
    u32x8 f;
#pragma unroll
    for (int p = 0; p < 4; ++p) {
        f[p] = nvfp4_pair(wa, p, g);
        f[4 + p] = nvfp4_pair(wb, p, g);
    }
    return __builtin_bit_cast(bf16x16, f);
}

struct ExpertChunk {
    u32x4 w[2];
    uint32_t s[2];
    u32x4 x[4];
};
#endif

// PAIR: a.n is the (gate | up) stack, the output silu(gate) x up in a.out16 (pairs, n / 2); else a.out (or out16) (pairs, n).
template <bool PAIR>
__device__ __forceinline__ void wmma_experts_tile(Affine& a, float limit) {
#if TF_DEVICE_WMMA_GFX11
    if (!Nvfp4Decoder::take_item(a, blockIdx.z)) return;
    const int count = a.m;
    const int cols = PAIR ? a.n / 2 : a.n;
    const int tile = blockIdx.x * kExpertWaves + (threadIdx.x >> 5);
    if (tile * 16 >= cols) return;
    const int lane = threadIdx.x & 31;
    const int l16 = lane & 15;
    const int h = lane >> 4;
    const int chunks = a.k >> 5;
    constexpr int kSides = PAIR ? 2 : 1;
    const int col0 = tile * 16 + l16;
    const uint32_t* wrow[kSides];
    const unsigned short* srow[kSides];
#pragma unroll
    for (int s = 0; s < kSides; ++s) {
        const long long col = col0 + s * cols;
        wrow[s] = a.words + col * (a.k >> 3);
        srow[s] = static_cast<const unsigned short*>(a.scale.p) + col * chunks;
    }
    const int first = a.route.first;
    for (int r0 = 0; r0 < count; r0 += 16) {
        const bool live = r0 + l16 < count;
        const int pair = live ? a.route.members[first + r0 + l16] : 0;
        const u32x4* xr = reinterpret_cast<const u32x4*>(static_cast<const char*>(a.x) + static_cast<long long>(live ? pair / a.route.x_div : 0) * a.k * 2);
        f32x8 acc[kSides][2];
#pragma unroll
        for (int s = 0; s < kSides; ++s) acc[s][0] = acc[s][1] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
        const u32x4 none = {0, 0, 0, 0};
        for (int c0 = 0; c0 < chunks; c0 += kExpertAhead) {
            ExpertChunk ch[kExpertAhead][kSides];
#pragma unroll
            for (int d = 0; d < kExpertAhead; ++d) {
                const int c = c0 + d;
#pragma unroll
                for (int s = 0; s < kSides; ++s) {
                    ch[d][s].w[0] = *reinterpret_cast<const u32x4*>(wrow[s] + c * 4);
                    ch[d][s].s[0] = srow[s][c];
                }
                const u32x4* xc = xr + c * 4;
#pragma unroll
                for (int v = 0; v < 4; ++v) ch[d][0].x[v] = live ? xc[v] : none;
            }
#pragma unroll
            for (int d = 0; d < kExpertAhead; ++d) {
#pragma unroll
                for (int s = 0; s < kSides; ++s) {
                    const Nvfp4Scales sc = nvfp4_scales(ch[d][s].s[0]);
                    const u32x4 w = ch[d][s].w[0];
#pragma unroll
                    for (int st = 0; st < 2; ++st) {
                        const u32x4 xlo = ch[d][0].x[2 * st], xhi = ch[d][0].x[2 * st + 1];
                        const bf16x16 af = __builtin_bit_cast(bf16x16, __builtin_shufflevector(xlo, xhi, 0, 1, 2, 3, 4, 5, 6, 7));
                        const bf16x16 bf = st == 0 ? expert_frag(w.x, w.y, sc.g0) : expert_frag(w.z, w.w, sc.g1);
                        acc[s][d & 1] = WmmaBF16::mma(af, bf, acc[s][d & 1]);
                    }
                }
            }
        }
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int r = r0 + 2 * i + h;
            if (r >= count) continue;
            const int out = a.route.members[first + r];
            const float g = Nvfp4Decoder::output(a, col0, acc[0][0][i] + acc[0][1][i]);
            if constexpr (PAIR) {
                const float u = Nvfp4Decoder::output(a, col0 + cols, acc[1][0][i] + acc[1][1][i]);
                static_cast<__bf16*>(static_cast<void*>(a.out16))[static_cast<long long>(out) * cols + col0] = swiglu<WmmaBF16>(g, u, limit);
            } else {
                GemmOut<WmmaBF16>::store(a, r, col0, g);
            }
        }
    }
#else
    if (threadIdx.x < 100u) __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
