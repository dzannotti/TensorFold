#pragma once

// gfx11 WMMA decode tile for any row count: a wave owns 16 columns over all of K and every row of the call, so each
// weight byte is read once; per element the sums are wmma_gemm_tile's (a WMMA chain a group, fma scale, fma bias).

#include "common/wmma.hpp"
#include "quant/act.hpp"
#include "quant/mlx_decoder.hpp"
#include "quant/mlx_tiles.hpp"
#include "tiles/epilogue.hpp"

namespace tf {
namespace rocm {

// MT 16-row tiles and WV waves a block: the K of x a staged slab (8 KB) and the slab's row pitch in BF16.
template <int MT, int WV>
struct VerifyShape {
    static constexpr int rows = 16 * MT;
    static constexpr int waves = WV;
    static constexpr int ks = 4096 / rows;
    static constexpr int ld = ks + 8;
};

// Codes 16 T .. 16 T + 15 of a 32-code chunk as a WMMA B fragment, in K order.
template <int BITS, int T>
__device__ inline bf16x16 verify_frag(const uint32_t (&w)[BITS]) {
    u32x8 f;
    if constexpr (BITS == 4) {
        // a word's even codes are the low nibbles of its bytes, odd codes the high: one byte convert a code
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
    } else if constexpr (BITS == 8) {
#pragma unroll
        for (int h = 0; h < 4; ++h) {
            const uint32_t v = w[4 * T + h];
#pragma unroll
            for (int j = 0; j < 2; ++j) {
                const uint32_t lo = __float_as_uint(static_cast<float>((v >> (16 * j)) & 0xffu));
                const uint32_t hi = __float_as_uint(static_cast<float>((v >> (16 * j + 8)) & 0xffu));
                f[2 * h + j] = __builtin_amdgcn_perm(hi, lo, 0x07060302u);
            }
        }
    } else {
#pragma unroll
        for (int p = 0; p < 8; ++p) f[p] = __builtin_bit_cast(uint32_t, MlxDecoder<BITS>::template pair<DotBF16>(w, 16 * T + 2 * p));
    }
    return __builtin_bit_cast(bf16x16, f);
}

// f(integral_constant<0>) .. f(integral_constant<N - 1>), unrolled.
template <int N, class F>
__device__ __forceinline__ void verify_for(F&& f) {
    [&]<int... I>(std::integer_sequence<int, I...>) { (f(std::integral_constant<int, I>{}), ...); }(std::make_integer_sequence<int, N>{});
}

// Where row r's sum of x sits in a slab's sums: a lane reads its eight rows (2 e + half) as two 16-byte words.
__device__ inline int verify_sum_at(int r) { return (r & ~15) + (r & 1) * 8 + ((r & 15) >> 1); }

// Columns 16 W bx.. of the product (or of side s of a group) for rows 16 MT blockIdx.y..; x and sums of x in LDS.
template <int BITS, int MT, int G, int WV>
__global__ void __launch_bounds__(32 * WV) affine_wmma_verify(Affine a, StreamSides sides) {
#if TF_DEVICE_WMMA_GFX11
    using S = VerifyShape<MT, WV>;
    using Dec = MlxDecoder<BITS>;
    using T = WmmaBF16;
    constexpr int MP = S::rows, W = S::waves, KS = S::ks, LD = S::ld, CPS = KS / 32;
    constexpr int SPT = 8 / CPS;                 // slabs a trip of 8 chunks (256 K)
    constexpr int D = BITS <= 4 && MT <= 4 ? 3 : 2;  // trips of codes a lane holds: D - 1 in flight
    constexpr int NPT = MP * KS / 8 / (32 * W);  // 16-byte pieces of a slab a thread stages
    __shared__ __attribute__((aligned(16))) unsigned short xs[3][MP * LD];
    __shared__ __attribute__((aligned(16))) float sxs[3][KS / 32][MP];

    int bx = blockIdx.x;
    if (sides.count > 0) {
        int s = 0;
        while (s + 1 < sides.count && bx >= sides.first[s + 1]) ++s;
        bx -= sides.first[s];
        Dec::bind(a, sides, s);
        a.n = sides.n[s];
        if (sides.out_half) {
            a.out16 = static_cast<__half*>(sides.out[s]);
        } else {
            a.out = static_cast<float*>(sides.out[s]);
        }
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int m0 = blockIdx.y * MP;
    const int live = a.m - m0 < MP ? a.m - m0 : MP;
    const int K = a.k;
    constexpr int CPG = G / 32;   // chunks a group
    constexpr int GPT = 8 / CPG;  // groups a trip of 8 chunks
    constexpr int cpgm = CPG - 1;
    const int nch = K >> 5;
    const int nslab = (K + KS - 1) / KS;
    const int col = (bx * W + wave) * 16 + (lane & 15);
    const int col_s = col < a.n ? col : a.n - 1;
    const uint32_t* wp = Dec::words(a) + static_cast<long long>(col_s) * nch * BITS;
    const long long tb = static_cast<long long>(col_s) * (K / G);
    const typename Dec::StreamTerms terms(a);

    // ---- a trip's codes and group terms: lane half h loads chunks 4 h .. 4 h + 3 of its column, 64 bytes at 4 bits ----
    const int half = lane >> 4;
    const int ntrips = (nch + 7) >> 3;
    uint32_t ring[D][4][BITS];
    uint32_t tr[D][GPT];
    // Every load below is unconditional on clamped addresses, so the compiler can count them and never drains the
    // prefetch ring (vmcnt(0)); the launch takes K a multiple of 256 and term tables on 2 GPT bytes.
    auto fetch = [&](int t, uint32_t (&w)[4][BITS], uint32_t (&g)[GPT]) {
        const int tt = t < ntrips ? t : ntrips - 1;
#pragma unroll
        for (int j = 0; j < 4; ++j) Dec::template load<true>(wp + static_cast<long long>(tt * 8 + 4 * half + j) * BITS, w[j]);
        // lane half 0 loads the trip's scales, half 1 its biases; each takes the other's from its partner lane
        constexpr int NW = GPT / 2 > 0 ? GPT / 2 : 1;
        uint32_t own[NW];
        using V = std::conditional_t<GPT == 8, u32x4, std::conditional_t<GPT == 4, uint2, uint32_t>>;
        *reinterpret_cast<V*>(own) = *reinterpret_cast<const V*>((half ? terms.bias : terms.scale) + tb + tt * GPT);
#pragma unroll
        for (int v = 0; v < NW; ++v) {
            const uint32_t other = __builtin_amdgcn_permlanex16(own[v], own[v], 0x76543210u, 0xfedcba98u, false, false);
            const uint32_t sw = half ? other : own[v];
            const uint32_t bw = half ? own[v] : other;
            g[2 * v] = (sw & 0xffffu) | (bw << 16);
            if (2 * v + 1 < GPT) g[2 * v + 1] = (sw >> 16) | (bw & 0xffff0000u);
        }
    };

    // ---- x: a slab of KS a row through registers into one of three LDS buffers ----
    u32x4 xr[NPT];
    auto xload = [&](int s0) {
        const int s = s0 < nslab ? s0 : nslab - 1;
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            const int p = tid + j * 32 * W;
            const int r = p / (KS / 8);
            const int k = s * KS + (p % (KS / 8)) * 8;
            // rows past the call read its last row: their outputs are never stored
            xr[j] = *reinterpret_cast<const u32x4*>(static_cast<const __bf16*>(a.x) + static_cast<long long>(m0 + (r < live ? r : live - 1)) * K + k);
        }
    };
    auto xstore = [&](int buf) {
#pragma unroll
        for (int j = 0; j < NPT; ++j) {
            const int p = tid + j * 32 * W;
            *reinterpret_cast<u32x4*>(&xs[buf][(p / (KS / 8)) * LD + (p % (KS / 8)) * 8]) = xr[j];
        }
    };
    // A row's sum of x over each group, lo then hi of each pair in K order (the GEMM tiles' sumx); a group past the
    // slab carries its run to the next.
    // a thread may run several chains (fewer threads than rows x segments); the carry only exists when KS < G, one
    // segment a slab, so a thread owns one row's carry
    constexpr int LEN = KS < G ? KS : G;
    constexpr int CHAINS = MP * (KS / LEN);
    constexpr int CPT = (CHAINS + 32 * W - 1) / (32 * W);
    static_assert(KS >= G || CHAINS <= 32 * W, "a carried group sum needs a thread a chain");
    float carry = 0.f;
    auto sums = [&](int s) {
#pragma unroll
        for (int ci = 0; ci < CPT; ++ci) {
            const int chain = tid + ci * 32 * W;
            if (chain >= CHAINS) break;
            const int r = chain % MP;
            const int seg = chain / MP;
            const int k0 = s * KS + seg * LEN;
            if (r >= live || k0 >= K) continue;
            float run = KS < G && k0 % G != 0 ? carry : 0.f;
            const u32x4* src = reinterpret_cast<const u32x4*>(&xs[s % 3][r * LD + seg * LEN]);
            for (int v = 0; v < LEN / 8; ++v) {
                const u32x4 u = src[v];
                const uint32_t q[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    run += __uint_as_float(q[h] << 16);
                    run += __uint_as_float(q[h] & 0xffff0000u);
                }
            }
            if ((k0 + LEN) % G == 0) {
                sxs[s % 3][seg][verify_sum_at(r)] = run;
            } else {
                carry = run;
            }
        }
    };

    f32x8 acc[MT];
    f32x8 d[MT];
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
        acc[mt] = zero;
        d[mt] = zero;
    }
    // Chunk J of a trip from ring buffer RB (half J / 4's words; the other half's by permlanex16), against slab
    // buffer `buf` at chunk i of the slab: two WMMA steps, and at a group's end its fold.
    auto chunk = [&]<int RB, int J>(std::integral_constant<int, RB>, std::integral_constant<int, J>, int c, int buf, int i) {
        uint32_t w[BITS];
#pragma unroll
        for (int b = 0; b < BITS; ++b) {
            const uint32_t own = ring[RB][J % 4][b];
            const uint32_t other = __builtin_amdgcn_permlanex16(own, own, 0x76543210u, 0xfedcba98u, false, false);
            w[b] = half == J / 4 ? own : other;
        }
        const bf16x16 b0 = verify_frag<BITS, 0>(w);
        const bf16x16 b1 = verify_frag<BITS, 1>(w);
        if ((c & cpgm) == 0) {
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) d[mt] = zero;
        }
        const unsigned short* xb = &xs[buf][(lane & 15) * LD + i * 32];
#pragma unroll
        for (int mt = 0; mt < MT; ++mt) {
            const uint32_t* xp = reinterpret_cast<const uint32_t*>(xb + mt * 16 * LD);
            d[mt] = T::mma(frag16(xp), b0, d[mt]);
            d[mt] = T::mma(frag16(xp + 8), b1, d[mt]);
        }
        if ((c & cpgm) == cpgm) {
            const uint32_t g = tr[RB][J / CPG];
            const float scale = terms.widen(static_cast<unsigned short>(g & 0xffffu));
            const float bias = terms.widen(static_cast<unsigned short>(g >> 16));
            const int seg = KS >= G ? (i * 32) / G : 0;
#pragma unroll
            for (int mt = 0; mt < MT; ++mt) {
                const u32x4* sp = reinterpret_cast<const u32x4*>(&sxs[buf][seg][mt * 16 + (lane >> 4) * 8]);
                const u32x4 s0 = sp[0];
                const u32x4 s1 = sp[1];
                const float sx[8] = {__uint_as_float(s0.x), __uint_as_float(s0.y), __uint_as_float(s0.z), __uint_as_float(s0.w),
                                     __uint_as_float(s1.x), __uint_as_float(s1.y), __uint_as_float(s1.z), __uint_as_float(s1.w)};
#pragma unroll
                for (int e = 0; e < 8; ++e) {
                    acc[mt][e] = Dec::fold_scale(acc[mt][e], d[mt][e], scale);
                    acc[mt][e] = Dec::fold_bias(acc[mt][e], sx[e], bias);
                }
            }
        }
    };

    verify_for<D - 1>([&](auto t) { fetch(decltype(t)::value, ring[decltype(t)::value], tr[decltype(t)::value]); });
    xload(0);
    xstore(0);
    xload(1);
    xstore(1);
    __syncthreads();
    sums(0);
    __syncthreads();
    // whole rounds of D trips: past the last trip the loads repeat it and the work is skipped
    for (int t0 = 0; t0 < ntrips; t0 += D) {
        verify_for<D>([&](auto rb) {
            constexpr int RB = decltype(rb)::value;
            const int t = t0 + RB;
            verify_for<SPT>([&](auto sl) {
                constexpr int SL = decltype(sl)::value;
                const int s = t * SPT + SL;
                // x before the trip's codes: the in-order load counter then lets xstore wait without draining them
                xload(s + 2);
                if constexpr (SL == 0) fetch(t + D - 1, ring[(RB + D - 1) % D], tr[(RB + D - 1) % D]);
                if (s + 1 < nslab) sums(s + 1);
                if (s < nslab) {
                    verify_for<CPS>([&](auto i) {
                        constexpr int I = decltype(i)::value;
                        chunk(rb, std::integral_constant<int, SL * CPS + I>{}, s * CPS + I, s % 3, I);
                    });
                }
                xstore((s + 2) % 3);
                __syncthreads();
            });
        });
    }

#pragma unroll
    for (int mt = 0; mt < MT; ++mt) {
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            const int row = m0 + mt * 16 + 2 * e + (lane >> 4);
            if (row >= a.m || col >= a.n) continue;
            const long long at = static_cast<long long>(row) * a.n + col;
            if (a.out16 != nullptr) {
                static_cast<__bf16*>(static_cast<void*>(a.out16))[at] = narrow<T>(acc[mt][e]);
            } else {
                a.out[at] = acc[mt][e];
            }
        }
    }
#else
    (void)a;
    (void)sides;
    __builtin_trap();
#endif
}

// Few columns over a long K at few rows (Flash Next's hyper-connection down, 324 x 10240, in a round of at most 4
// rows): the verify tile's sums with K cut across a block's waves. Wave w makes groups [16 w, 16 w + 16) of the block's
// 16 columns, each group's WMMA chain from zero as the verify tile makes it, and keeps each group's dots of the rows
// below 2 E (E of a lane's 8 sums); lanes 0-15 put the groups' sums of x in LDS. Then the folds run in K order: wave w
// folds its groups onto wave w - 1's running sums, handed over in LDS. Every product and sum is the verify tile's, in
// its order: the same bits. Codes and terms run eight groups ahead, x three. W = blockDim / 32 is 1 to 20.
template <int BITS, int E>
__global__ void __launch_bounds__(640) affine_wmma_deal(Affine a, StreamSides sides) {
#if TF_DEVICE_WMMA_GFX11
    using Dec = MlxDecoder<BITS>;
    using T = WmmaBF16;
    constexpr int L = 16;  // groups a wave
    constexpr int A = 3;   // groups of x in flight
    constexpr int P = 8;   // groups of codes in flight
    constexpr int ROWS = 2 * E;
    __shared__ float sxs[20 * L][ROWS];  // a group's sums of x, a row each
    __shared__ float hand[32][E];        // the running sums a wave hands the next, a lane's E

    int bx = blockIdx.x;
    if (sides.count > 0) {
        int s = 0;
        while (s + 1 < sides.count && bx >= sides.first[s + 1]) ++s;
        bx -= sides.first[s];
        Dec::bind(a, sides, s);
        a.n = sides.n[s];
        if (sides.out_half) {
            a.out16 = static_cast<__half*>(sides.out[s]);
        } else {
            a.out = static_cast<float*>(sides.out[s]);
        }
    }
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = tid >> 5;
    const int W = blockDim.x >> 5;
    const int live = a.m < ROWS ? a.m : ROWS;
    const int K = a.k;
    const int groups = K >> 5;
    const int col = bx * 16 + (lane & 15);
    const int col_s = col < a.n ? col : a.n - 1;
    const uint32_t* wp = Dec::words(a) + static_cast<long long>(col_s) * groups * BITS;
    const long long tb = static_cast<long long>(col_s) * groups;
    const typename Dec::StreamTerms terms(a);
    // the A fragment's row: rows past the call read its last row, whose sums are never stored
    const int xrow = (lane & 15) < live ? (lane & 15) : live - 1;
    const __bf16* xp0 = static_cast<const __bf16*>(a.x) + static_cast<long long>(xrow) * K;
    const int g0 = wave * L;

    // codes and terms P groups ahead, unconditional on clamped groups so no load waits on another
    uint32_t w[P][BITS];
    uint32_t tp[P];
    uint32_t tr[L];
    auto cload = [&](int i, uint32_t (&c)[BITS], uint32_t& t) {
        const int g = g0 + i < groups ? g0 + i : groups - 1;
        Dec::template load<true>(wp + static_cast<long long>(g) * BITS, c);
        t = static_cast<uint32_t>(terms.scale_raw(tb + g)) | static_cast<uint32_t>(terms.bias_raw(tb + g)) << 16;
    };
#pragma unroll
    for (int i = 0; i < P; ++i) cload(i, w[i], tp[i]);
    u32x4 xr[A][4];
    auto xload = [&](int i, u32x4 (&v)[4]) {
        const int g = g0 + i < groups ? g0 + i : groups - 1;
        const u32x4* src = reinterpret_cast<const u32x4*>(xp0 + static_cast<long long>(g) * 32);
#pragma unroll
        for (int q = 0; q < 4; ++q) v[q] = src[q];
    };
#pragma unroll
    for (int i = 0; i < A - 1; ++i) xload(i, xr[i]);

    float dl[L][E];
    const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
#pragma unroll
    for (int i = 0; i < L; ++i) {
        xload(i + A - 1 < L ? i + A - 1 : L - 1, xr[(i + A - 1) % A]);
        const u32x4(&v)[4] = xr[i % A];
        const bf16x16 lo = __builtin_bit_cast(bf16x16, __builtin_shufflevector(v[0], v[1], 0, 1, 2, 3, 4, 5, 6, 7));
        const bf16x16 hi = __builtin_bit_cast(bf16x16, __builtin_shufflevector(v[2], v[3], 0, 1, 2, 3, 4, 5, 6, 7));
        f32x8 d = T::mma(lo, verify_frag<BITS, 0>(w[i % P]), zero);
        d = T::mma(hi, verify_frag<BITS, 1>(w[i % P]), d);
        tr[i] = tp[i % P];
        if (i + P < L) cload(i + P, w[i % P], tp[i % P]);
#pragma unroll
        for (int e = 0; e < E; ++e) dl[i][e] = d[e];
        if (lane < ROWS && g0 + i < groups) {
            // the verify tile's sum of x over a group: from zero, in K order
            float run = 0.f;
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const uint32_t u[4] = {v[q].x, v[q].y, v[q].z, v[q].w};
#pragma unroll
                for (int h = 0; h < 4; ++h) {
                    run += __uint_as_float(u[h] << 16);
                    run += __uint_as_float(u[h] & 0xffff0000u);
                }
            }
            sxs[g0 + i][lane] = run;
        }
    }
    __syncthreads();

    // the folds in K order, a wave at a time: a lane's sums of x first, so a fold waits only on the handover
    const int half = lane >> 4;
    float sv[L][E];
#pragma unroll
    for (int i = 0; i < L; ++i)
#pragma unroll
        for (int e = 0; e < E; ++e) sv[i][e] = sxs[g0 + i < groups ? g0 + i : groups - 1][2 * e + half];
    for (int s = 0; s < W; ++s) {
        if (wave == s) {
            float acc[E];
#pragma unroll
            for (int e = 0; e < E; ++e) acc[e] = s > 0 ? hand[lane][e] : 0.f;
#pragma unroll
            for (int i = 0; i < L; ++i) {
                if (g0 + i < groups) {
                    const float scale = terms.widen(static_cast<unsigned short>(tr[i] & 0xffffu));
                    const float bias = terms.widen(static_cast<unsigned short>(tr[i] >> 16));
#pragma unroll
                    for (int e = 0; e < E; ++e) {
                        acc[e] = Dec::fold_scale(acc[e], dl[i][e], scale);
                        acc[e] = Dec::fold_bias(acc[e], sv[i][e], bias);
                    }
                }
            }
            if (s + 1 < W) {
#pragma unroll
                for (int e = 0; e < E; ++e) hand[lane][e] = acc[e];
            } else {
#pragma unroll
                for (int e = 0; e < E; ++e) {
                    const int row = 2 * e + half;
                    if (row >= a.m || col >= a.n) continue;
                    const long long at = static_cast<long long>(row) * a.n + col;
                    if (a.out16 != nullptr) {
                        static_cast<__bf16*>(static_cast<void*>(a.out16))[at] = narrow<T>(acc[e]);
                    } else {
                        a.out[at] = acc[e];
                    }
                }
            }
        }
        __syncthreads();
    }
#else
    (void)a;
    (void)sides;
    __builtin_trap();
#endif
}

#if TF_MATRIX
// group 32 at 2, 3 or 4 bits; E = 1 (two rows) or 2 (four)
#define TF_DEAL(BITS)                                                        \
    template __global__ void affine_wmma_deal<BITS, 1>(Affine, StreamSides); \
    template __global__ void affine_wmma_deal<BITS, 2>(Affine, StreamSides);
TF_DEAL(2)
TF_DEAL(3)
TF_DEAL(4)
#undef TF_DEAL
#define TF_VERIFY_W(BITS, G, WV)                                                  \
    template __global__ void affine_wmma_verify<BITS, 1, G, WV>(Affine, StreamSides); \
    template __global__ void affine_wmma_verify<BITS, 2, G, WV>(Affine, StreamSides);
#define TF_VERIFY_G(BITS, G) TF_VERIFY_W(BITS, G, 2) TF_VERIFY_W(BITS, G, 4)
#define TF_VERIFY(BITS) TF_VERIFY_G(BITS, 32) TF_VERIFY_G(BITS, 64) TF_VERIFY_G(BITS, 128)
TF_VERIFY(2)
TF_VERIFY(3)
TF_VERIFY(4)
TF_VERIFY(5)
TF_VERIFY(6)
TF_VERIFY(8)
#undef TF_VERIFY
#undef TF_VERIFY_G
#undef TF_VERIFY_W
#endif

}  // namespace rocm
}  // namespace tf
