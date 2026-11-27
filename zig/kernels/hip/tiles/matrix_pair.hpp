#pragma once

// gfx11 WMMA prompt GEMM over a (gate | up) stack with silu(gate) x up as its epilogue: block bx stages x once for gate
// features 64 bx.. and the same up features, so a routed item's activations come out in one launch. Each sum is
// wmma_gemm_tile's (SKEW staging, the same stage order and fold), so act equals that tile's fp32 product then moe_act.
// For decoders without a bias term (kBias false): FP4-Local's.

#include "common/wmma.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/gemm.hpp"
#include "tiles/plan.hpp"

namespace tf {
namespace rocm {

// a.n is the stack's rows (gate then up, n / 2 each); a.out16 takes act (pairs, n / 2) in T; `limit` clamps as moe_act.
template <class Dec, class Act, class T>
__device__ __forceinline__ void wmma_pair_tile(typename Dec::Args& a, float limit) {
#if TF_DEVICE_WMMA_GFX11
    static_assert(!Dec::kBias, "the pair tile folds no bias");
    if (!Dec::take_item(a, blockIdx.z)) return;
    int bx, by;
    band_tile(bx, by);
    const int m0 = by * kGemmM;
    if (m0 >= a.m) return;
    constexpr int kLdw = kGemmK / 2;
    constexpr int kGroupW = 8 * kLdw + 12;
    constexpr int kStage = kGemmM / 8 * kGroupW;
    auto rowat = [](int r) { return (r >> 3) * kGroupW + (r & 7) * kLdw + (((r >> 1) & 3) << 2); };
    // x and the weights' stages, double-buffered; the up sums reuse them in the epilogue
    __shared__ __attribute__((aligned(16))) uint32_t lds[4 * kStage];
    uint32_t* const xs[2] = {lds, lds + kStage};
    uint32_t* const ws[2] = {lds + 2 * kStage, lds + 3 * kStage};

    const int tid = threadIdx.x;
    const int half = a.n >> 1;
    const int f0 = bx * (kGemmN / 2);
    const int words_row = Dec::template row_words<int>(a);
    const int fold = Dec::fold_k(a);
    const int stages = a.k / kGemmK;
    const int per = fold / kGemmK;
    const int per_log2 = __builtin_ctz(per);
    auto group_start = [&](int st) { return (st & (per - 1)) == 0; };
    auto group_end = [&](int st) { return (st & (per - 1)) == per - 1; };
    const int rows_left = a.m - m0;
    const int line = tid & (kGemmN - 1);
    const int side = __builtin_amdgcn_readfirstlane(tid >> 7);
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    const int lane = tid & 31;
    (void)per_log2;

    // line < 64: gate feature f0 + line; else up feature f0 + line - 64 (its row half past the gate's)
    const int feat = f0 + (line & 63) < half ? f0 + (line & 63) : half - 1;
    const int col_s = feat + (line >= 64 ? half : 0);
    const uint32_t* wsrc = Dec::words(a) + static_cast<long long>(col_s) * words_row;
    const int xrow = tid >> 1;
    const int xhalf = tid & 1;
    const bool xlive = wave * 16 < rows_left;
    const int row_s = m0 + xrow < a.m ? m0 + xrow : 0;
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(Act::row(a, row_s));
    uint32_t wreg[2][Dec::kWords];
    u32x4 xreg[4];
    typename Dec::Stage stage[2];
    auto fetch = [&](int st0) {
        const int st1 = st0 + 1 < stages ? st0 + 1 : stages - 1;
        const int st = st0 < stages ? st0 : stages - 1;
        Dec::template load<true>(wsrc + st * Dec::kWords, wreg[0]);
        Dec::template load<true>(wsrc + st1 * Dec::kWords, wreg[1]);
        stage[0] = Dec::term_stage(a, col_s, st);
        stage[1] = Dec::term_stage(a, col_s, st1);
        const int sx_ = xhalf ? st1 : st;
        if (xlive) {
#pragma unroll
            for (int v = 0; v < 4; ++v) xreg[v] = xsrc[sx_ * 4 + v];
        }
    };
    auto stash = [&]<int J>(std::integral_constant<int, J>, int buf) {
        u32x4* dst = reinterpret_cast<u32x4*>(ws[buf] + rowat(line) + side * 8);
        auto decode = [&]<int S>(std::integral_constant<int, S>) {
#pragma unroll
            for (int v = 0; v < 2; ++v) {
                uint32_t d[4];
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    const int t = S * 16 + v * 8 + h * 2;
                    d[h] = __builtin_bit_cast(uint32_t, Dec::template pair<T>(wreg[J], t, stage[J]));
                }
                dst[v] = u32x4{d[0], d[1], d[2], d[3]};
            }
        };
        if (side == 0) decode(std::integral_constant<int, 0>{}); else decode(std::integral_constant<int, 1>{});
        if (xhalf == J && xlive) {
            u32x4* xdst = reinterpret_cast<u32x4*>(xs[buf] + rowat(xrow));
#pragma unroll
            for (int v = 0; v < 4; ++v) xdst[v] = xreg[v];
        }
    };

    // Wave (wm, wn): rows 32 wm.., wn 0 the gate features, wn 1 the same up features; a lane: column lane % 16.
    const int wm = wave & 3;
    const int wn = wave >> 2;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const bool live0 = wm * 32 < rows_left;
    const bool live1 = wm * 32 + 16 < rows_left;
    f32x8 acc[2][4];
    f32x8 d[2][4];
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            acc[mt][nt] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
            d[mt][nt] = acc[mt][nt];
        }
    }
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
    fetch(0);
    stash(std::integral_constant<int, 0>{}, 0);
    __syncthreads();
    auto step = [&]<int P>(std::integral_constant<int, P>, int s) {
        constexpr int cur = P;
        if constexpr (P == 1) fetch(s + 1);
        if (live0) {
#pragma unroll
            for (int t = 0; t < 2; ++t) {
                bf16x16 bf[4];
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) bf[nt] = frag16(ws[cur] + (wn * 8 + nt * 2) * kGroupW + rowat(rl) + t * 8);
                const bool first = t == 0 && group_start(s);
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (mt == 1 && !live1) continue;
                    const bf16x16 af = frag16(xs[cur] + (wm * 4 + mt * 2) * kGroupW + rowat(rl) + t * 8);
                    if (first) {
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt) d[mt][nt] = T::mma(af, bf[nt], zero);
                    } else {
#pragma unroll
                        for (int nt = 0; nt < 4; ++nt) d[mt][nt] = T::mma(af, bf[nt], d[mt][nt]);
                    }
                }
            }
            if (group_end(s)) {
#pragma unroll
                for (int mt = 0; mt < 2; ++mt) {
                    if (mt == 1 && !live1) continue;
#pragma unroll
                    for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                        for (int i = 0; i < 8; ++i) acc[mt][nt][i] = Dec::fold_scale(acc[mt][nt][i], d[mt][nt][i], 1.f);
                    }
                }
            }
        }
        stash(std::integral_constant<int, P ^ 1>{}, cur ^ 1);
        __syncthreads();
    };
    for (int s = 0; s < stages; s += 2) {
        step(std::integral_constant<int, 0>{}, s);
        if (s + 1 < stages) step(std::integral_constant<int, 1>{}, s + 1);
    }
    // the up waves hand their sums (global applied) to the gate waves through the stages' LDS
    constexpr int kRed = kGemmN / 2 + 1;
    static_assert(kGemmM * kRed <= 4 * kStage, "the up sums fit the stages");
    float* const red = reinterpret_cast<float*>(lds);
    if (wn == 1 && live0) {
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int f = f0 + nt * 16 + rl;
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    red[(wm * 32 + mt * 16 + 2 * i + rh) * kRed + nt * 16 + rl] = Dec::output(a, half + f, acc[mt][nt][i]);
                }
            }
        }
    }
    __syncthreads();
    if (wn == 1) return;
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int rr = wm * 32 + mt * 16 + 2 * i + rh;
            const int row = m0 + rr;
            if (row >= a.m) continue;
            typename T::elem* out = static_cast<typename T::elem*>(static_cast<void*>(a.out16)) + out_row(a, row) * half;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int f = f0 + nt * 16 + rl;
                if (f < half) out[f] = swiglu<T>(Dec::output(a, f, acc[mt][nt][i]), red[rr * kRed + nt * 16 + rl], limit);
            }
        }
    }
#else
    (void)a;
    (void)limit;
    __builtin_trap();
#endif
}

}  // namespace rocm
}  // namespace tf
