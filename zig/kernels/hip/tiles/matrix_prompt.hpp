#pragma once

// gfx11 WMMA prompt GEMM, MLX 4-bit group 64: wmma_gemm_tile's bits from a pipelined tile and precomputed x group sums.

#include "common/wmma.hpp"
#include "quant/mlx.hpp"

namespace tf {
namespace rocm {

constexpr int kPromptLdw = 20;  // words a staged row: 16 of codes or x and 4 of pad, so eight 16-byte reads hit eight bank groups

// Word w's eight codes as four BF16 pairs (codes 2j, 2j + 1): bytes of the even and odd nibbles to float, top halves.
__device__ inline void prompt_decode(uint32_t w, uint32_t* p) {
    uint32_t e = w & 0x0f0f0f0fu;
    uint32_t o = (w >> 4) & 0x0f0f0f0fu;
    asm("" : "+v"(e), "+v"(o));  // hide the masks so each byte converts with v_cvt_f32_ubyteN
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float lo = static_cast<float>((e >> (8 * j)) & 0xffu);
        const float hi = static_cast<float>((o >> (8 * j)) & 0xffu);
        p[j] = __builtin_amdgcn_perm(__float_as_uint(hi), __float_as_uint(lo), 0x07060302u);
    }
}

// A stored 16-bit scale or bias as float (BF16 or FP16 tables).
__device__ inline float prompt_widen(uint32_t v, bool half) {
    return half ? __half2float(__ushort_as_half(static_cast<unsigned short>(v))) : __uint_as_float(v << 16);
}

// __syncthreads for LDS alone: the global loads in flight stay in flight.
__device__ inline void prompt_sync() {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "workgroup", "local");
    __builtin_amdgcn_s_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "workgroup", "local");
}

// sx is kept so a lane's eight rows 2 i + h of a 16-row tile are eight neighbours.
__device__ inline int prompt_sx_at(int r) { return (r & ~15) | ((r & 1) << 3) | ((r & 15) >> 1); }

// Tile t of the launch: groups of 4 M tiles, M fastest, so the blocks in flight share their weights and x in L2.
__device__ inline void prompt_place(const Affine& a, int t, int& m0, int& n0) {
    constexpr int G = 4;
    const int nt = (a.n + 127) >> 7;
    const int mt = (a.m + 127) >> 7;
    const int band = t / (G * nt);
    const int rows = mt - band * G < G ? mt - band * G : G;
    const int in = t - band * G * nt;
    n0 = (in / rows) * 128;
    m0 = (band * G + in % rows) * 128;
}

template <class Epi>
__device__ __forceinline__ void prompt_tile(const Affine& a) {
#if TF_DEVICE_WMMA_GFX11
    constexpr int L = kPromptLdw;
    __shared__ __attribute__((aligned(16))) uint32_t xs[2][128 * L];
    __shared__ __attribute__((aligned(16))) uint32_t ws[2][128 * L];
    __shared__ __attribute__((aligned(16))) float sx[128];
    __shared__ float sc[128];
    __shared__ float bi[128];

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    int m0, n0;
    prompt_place(a, blockIdx.x, m0, n0);
    const int rows = a.m - m0;
    const int groups = a.k >> 6;
    const int stages = 2 * groups;
    const bool half = a.scale.kind == kScaleF16;

    // Codes: column c, codes 16 h.. of each stage. x: row r, elements 16 q.. of each stage. Terms and sums: thread c's
    // column's scale (h = 0) or bias (h = 1), and row c's group sum (h = 0). Past an edge: the last column, row m0.
    const int c = tid & 127;
    const int h = tid >> 7;
    const int r = tid >> 1;
    const int q = tid & 1;
    const int col = n0 + c < a.n ? n0 + c : a.n - 1;
    const uint2* wsrc = reinterpret_cast<const uint2*>(a.words + static_cast<long long>(col) * (a.k >> 3)) + h;
    const u32x4* xsrc = reinterpret_cast<const u32x4*>(static_cast<const uint16_t*>(a.x) + static_cast<long long>(m0 + (r < rows ? r : 0)) * a.k) + 2 * q;
    const unsigned short* tsrc = static_cast<const unsigned short*>(h ? a.bias.p : a.scale.p) + static_cast<long long>(col) * groups;
    const float* ssrc = a.sums + m0 + (c < rows ? c : 0);

    // Group slots (parity of g): the codes of both stages, the term, the sum. Stage slots (parity of s): x.
    uint2 W[2][2];
    uint32_t T[2];  // a whole register each: two halves of one would make a wait for either wait for both
    float S[2];
    u32x4 X[2][2];

    auto load_group = [&](int g, int j) {
        W[j][0] = wsrc[4 * g];
        W[j][1] = wsrc[4 * g + 2];
        T[j] = tsrc[g];
        S[j] = ssrc[static_cast<long long>(g) * a.m];
    };
    auto load_x = [&](int s, int p) {
        X[p][0] = xsrc[4 * s];
        X[p][1] = xsrc[4 * s + 1];
    };
    // Stage parity p's codes (from W[j][p]) and x (from X[xp]) into buffer p.
    auto stash = [&](int j, int p, int xp) {
        uint32_t d[8];
        prompt_decode(W[j][p].x, d);
        prompt_decode(W[j][p].y, d + 4);
        u32x4* wd = reinterpret_cast<u32x4*>(ws[p] + c * L + 8 * h);
        wd[0] = u32x4{d[0], d[1], d[2], d[3]};
        wd[1] = u32x4{d[4], d[5], d[6], d[7]};
        u32x4* xd = reinterpret_cast<u32x4*>(xs[p] + r * L + 8 * q);
        xd[0] = X[xp][0];
        xd[1] = X[xp][1];
    };
    auto terms = [&](int j) {
        const float v = prompt_widen(T[j], half);
        if (h) {
            bi[c] = v;
        } else {
            sc[c] = v;
            sx[prompt_sx_at(c)] = S[j];
        }
    };

    // Wave (wm, wn) owns rows 32 wm.., columns 64 wn.. (2 x 4 tiles); a lane: column lane % 16, rows 2 i + lane / 16.
    const int wm = wave & 3;
    const int wn = wave >> 2;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const bool live0 = wm * 32 < rows;
    const bool live1 = wm * 32 + 16 < rows;
    f32x8 acc[2][4];
    f32x8 d[2][4];
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = zero;
    }

    auto compute = [&](int buf, bool first) {
        if (!live0) return;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            bf16x16 bf[4];
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) bf[nt] = frag16(ws[buf] + (wn * 64 + nt * 16 + rl) * L + t * 8);
#pragma unroll
            for (int mt = 0; mt < 2; ++mt) {
                if (mt == 1 && !live1) continue;
                const bf16x16 af = frag16(xs[buf] + (wm * 32 + mt * 16 + rl) * L + t * 8);
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) d[mt][nt] = WmmaBF16::mma(af, bf[nt], first && t == 0 ? zero : d[mt][nt]);
            }
        }
    };
    auto fold = [&]() {
        if (!live0) return;
        float s[4], b[4];
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) {
            s[nt] = sc[wn * 64 + nt * 16 + rl];
            b[nt] = bi[wn * 64 + nt * 16 + rl];
        }
#pragma unroll
        for (int mt = 0; mt < 2; ++mt) {
            if (mt == 1 && !live1) continue;
            const float4* xp = reinterpret_cast<const float4*>(sx + wm * 32 + mt * 16 + rh * 8);
            const float4 x0 = xp[0];
            const float4 x1 = xp[1];
            const float xv[8] = {x0.x, x0.y, x0.z, x0.w, x1.x, x1.y, x1.z, x1.w};
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int i = 0; i < 8; ++i) {
                    acc[mt][nt][i] = fmaf(d[mt][nt][i], s[nt], acc[mt][nt][i]);
                    acc[mt][nt][i] = fmaf(xv[i], b[nt], acc[mt][nt][i]);
                }
            }
        }
    };

    // Prologue in the loop's issue order (x, then the group), so its waits count the same loads in flight.
    load_group(0, 0);
    load_x(0, 0);
    __builtin_amdgcn_sched_barrier(0);
    load_x(1, 1);
    __builtin_amdgcn_sched_barrier(0);
    load_group(groups > 1 ? 1 : 0, 1);
    __builtin_amdgcn_sched_barrier(0);
    stash(0, 0, 0);
    load_x(stages > 2 ? 2 : stages - 1, 0);
    prompt_sync();

    // Group g: stage 2 g from buffer 0 while 2 g + 1 is staged into 1, then 2 g + 1 while 2 g + 2 goes into 0.
    auto group = [&]<int J>(std::integral_constant<int, J>, int g) {
        stash(J, 1, 1);
        terms(J);
        load_x(2 * g + 3 < stages ? 2 * g + 3 : stages - 1, 1);
        load_group(g + 2 < groups ? g + 2 : groups - 1, J);
        compute(0, true);
        prompt_sync();
        if (g + 1 < groups) {
            stash(J ^ 1, 0, 0);
            load_x(2 * g + 4 < stages ? 2 * g + 4 : stages - 1, 0);
        }
        compute(1, false);
        fold();
        prompt_sync();
    };
    int g = 0;
    for (; g + 1 < groups; g += 2) {
        group(std::integral_constant<int, 0>{}, g);
        group(std::integral_constant<int, 1>{}, g + 1);
    }
    if (g < groups) group(std::integral_constant<int, 0>{}, g);

#pragma unroll
    for (int mt = 0; mt < 2; ++mt) {
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int row = m0 + wm * 32 + mt * 16 + 2 * i + rh;
            if (row >= a.m) continue;
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
                const int cc = n0 + wn * 64 + nt * 16 + rl;
                if (cc < a.n) Epi::store(a, row, cc, acc[mt][nt][i]);
            }
        }
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

// x's group sums for the prompt tile: sums[g * m + r] = x[r, 64 g..64 g + 63] added in order from zero, in fp32.
__device__ inline void prompt_sums(const Affine& a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    const int groups = a.k >> 6;
    if (i >= a.m * groups) return;
    const int r = i / groups;  // a wave reads a row's groups in a run
    const int g = i % groups;
    const u32x4* src = reinterpret_cast<const u32x4*>(static_cast<const uint16_t*>(a.x) + static_cast<long long>(r) * a.k + 64 * g);
    float run = 0.f;
#pragma unroll
    for (int v = 0; v < 8; ++v) {
        const u32x4 u = src[v];
        const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            run += __uint_as_float(w[j] << 16);
            run += __uint_as_float(w[j] & 0xffff0000u);
        }
    }
    const_cast<float*>(a.sums)[static_cast<long long>(g) * a.m + r] = run;
}

}  // namespace rocm
}  // namespace tf
