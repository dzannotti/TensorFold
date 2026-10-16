#pragma once

// gfx11 WMMA prompt GEMMs, MLX 4-bit group 64: prompt_tile keeps wmma_gemm_tile's bits (an exact per-group fold over
// precomputed x group sums); prompt_dq_tile dequantizes the weights to bf16 and runs one WMMA chain over all of k.

#include <type_traits>
#include <utility>

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

// Tile t of a BM x BN launch: bands of 4 M tiles, M fastest, so the blocks in flight share their weights and x in L2.
template <int BM, int BN>
__device__ inline void prompt_place_at(const Affine& a, int t, int& m0, int& n0) {
    constexpr int G = 4;
    const int nt = (a.n + BN - 1) / BN;
    const int mt = (a.m + BM - 1) / BM;
    const int band = t / (G * nt);
    const int rows = mt - band * G < G ? mt - band * G : G;
    const int in = t - band * G * nt;
    n0 = (in / rows) * BN;
    m0 = (band * G + in % rows) * BM;
}

__device__ inline void prompt_place(const Affine& a, int t, int& m0, int& n0) { prompt_place_at<128, 128>(a, t, m0, n0); }

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

// ---- the dequantizing prompt tile: weights to bf16 in LDS, one WMMA chain over all of K ----

// A raw buffer over `p` (gfx11's word 3: 32-bit dwords, raw out-of-range checks) so a load is a scalar base and
// offset plus one vector offset, no 64-bit address adds.
__device__ inline __amdgpu_buffer_rsrc_t prompt_rsrc(const void* p) {
    return __builtin_amdgcn_make_buffer_rsrc(const_cast<void*>(p), 0, 0x7fffffff, 0x31004000);
}

template <class T>
__device__ inline T prompt_load(__amdgpu_buffer_rsrc_t r, uint32_t lane, uint32_t uniform) {
    if constexpr (sizeof(T) == 16) {
        return __builtin_bit_cast(T, __builtin_amdgcn_raw_buffer_load_b128(r, lane, uniform, 0));
    } else if constexpr (sizeof(T) == 8) {
        return __builtin_bit_cast(T, __builtin_amdgcn_raw_buffer_load_b64(r, lane, uniform, 0));
    } else {
        static_assert(sizeof(T) == 2, "16, 8 or 2 bytes");
        return __builtin_bit_cast(T, __builtin_amdgcn_raw_buffer_load_b16(r, lane, uniform, 0));
    }
}

// Two floats as a BF16 pair (lo, hi), each rounded to nearest (ties away from zero: one add, not RNE's two).
__device__ inline uint32_t prompt_pack_rn(float lo, float hi) {
    return __builtin_amdgcn_perm(__float_as_uint(hi) + 0x8000u, __float_as_uint(lo) + 0x8000u, 0x07060302u);
}

// The dequant hook of prompt_dq_tile for MLX 4-bit group 64 with BF16 tables: a thread owns 32 / P codes of one
// column in each 32-k stage (part `part` of P) and writes them as w = scale q + bias, one fp32 fma rounded once to
// BF16 (MLX's qmm dequantizes into the activation type before its matmul). A hook keeps kDepth stages in flight in
// register slots: fetch<J>(s) issues stage s's loads into slot J; put<J>(st) hands slot J's BF16 to st(chunk, words),
// chunk v of 4 / P being its 8 k from 8 (part * 4 / P + v). The core fetches slots in order, each after it was put.
// Another format's decoder (EXL3's trellis) is another hook over the same core.
template <int P>
struct MlxDq4 {
    static_assert(P == 1 || P == 2, "16 or 32 codes a thread");
    static constexpr int kDepth = 4;
    static constexpr int kWords = 4 / P;  // code words a stage
    using Codes = std::conditional_t<P == 1, u32x4, uint2>;
    __amdgpu_buffer_rsrc_t words, scale, bias;
    uint32_t wo, to;  // a lane's byte offsets
    Codes W[kDepth];
    uint32_t S[kDepth / 2], B[kDepth / 2];  // a group's terms: loaded with its second stage, after its first was put

    __device__ MlxDq4(const Affine& a, int col, int part)
        : words(prompt_rsrc(a.words)),
          scale(prompt_rsrc(a.scale.p)),
          bias(prompt_rsrc(a.bias.p)),
          wo(static_cast<uint32_t>(col) * static_cast<uint32_t>(a.k >> 1) + part * sizeof(Codes)),
          to(static_cast<uint32_t>(col) * static_cast<uint32_t>(a.k >> 5)) {}

    template <int J>
    __device__ void fetch(int s) {
        W[J] = prompt_load<Codes>(words, wo, 16 * s);
        if constexpr (J & 1) {
            S[J / 2] = prompt_load<unsigned short>(scale, to, 2 * (s >> 1));
            B[J / 2] = prompt_load<unsigned short>(bias, to, 2 * (s >> 1));
        }
    }

    template <int J, class Store>
    __device__ void put(Store st) const {
        const float sc = __uint_as_float(S[J / 2] << 16);
        const float bi = __uint_as_float(B[J / 2] << 16);
        uint32_t w[kWords];
        if constexpr (P == 1) {
            w[0] = W[J].x, w[1] = W[J].y, w[2] = W[J].z, w[3] = W[J].w;
        } else {
            w[0] = W[J].x, w[1] = W[J].y;
        }
#pragma unroll
        for (int v = 0; v < kWords; ++v) {
            uint32_t e = w[v] & 0x0f0f0f0fu;
            uint32_t o = (w[v] >> 4) & 0x0f0f0f0fu;
            asm("" : "+v"(e), "+v"(o));  // keep the bytes whole so each converts with v_cvt_f32_ubyteN
            uint32_t d[4];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                const float lo = fmaf(sc, static_cast<float>((e >> (8 * j)) & 0xffu), bi);
                const float hi = fmaf(sc, static_cast<float>((o >> (8 * j)) & 0xffu), bi);
                d[j] = prompt_pack_rn(lo, hi);
            }
            st(v, u32x4{d[0], d[1], d[2], d[3]});
        }
    }
};

// A staged row is 32 k (64 bytes, four 16-byte chunks) with no pad: chunk c of row r sits at c ^ ((r >> 1) & 3), so
// each 8 rows of a fragment read (a pass of 128 bytes) hit 8 different bank groups.
__device__ inline int prompt_at(int r, int c) { return r * 16 + 4 * (c ^ ((r >> 1) & 3)); }

// BM x BN a block of BM * BN / 4096 waves, 64 x 64 a wave (4 x 4 WMMA tiles), 32 k a stage, double-buffered in LDS; x
// one stage ahead, the weights Hook::kDepth. Each output's sum is one WMMA chain over k in order from zero: its bits
// depend on k alone, never on m, the block shape or where a prompt is cut.
template <class Epi, template <int> class HookOf, int BM, int BN>
__device__ __forceinline__ void prompt_dq_tile(const Affine& a) {
#if TF_DEVICE_WMMA_GFX11
    constexpr int NT = BM * BN / 128;  // threads: a wave per 64 x 64
    constexpr int WGM = BM / 64;       // waves down M
    constexpr int XC = BM * 4 / NT;    // x chunks a thread a stage
    constexpr int XR = NT / 4;         // rows a pass of x chunks
    constexpr int P = NT / BN;         // threads a column
    using Hook = HookOf<P>;
    constexpr int D = Hook::kDepth;
    static_assert(D % 2 == 0, "a stage's buffer is its slot's parity");
    __shared__ __attribute__((aligned(16))) uint32_t lds[2 * (BM + BN) * 16];  // x's two buffers, then the weights'
    uint32_t* const xs[2] = {lds, lds + BM * 16};
    uint32_t* const ws[2] = {lds + 2 * BM * 16, lds + 2 * BM * 16 + BN * 16};

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int wave = __builtin_amdgcn_readfirstlane(tid >> 5);
    int m0, n0;
    prompt_place_at<BM, BN>(a, blockIdx.x, m0, n0);
    const int rows = a.m - m0;
    const int stages = a.k >> 5;

    // x: rows tid / 4 + XR i, chunk tid % 4 of each stage. Weights: column tid / P, part tid % P. Past an edge: row
    // m0, the last column. Stages past the end reload the last (their slots are never put).
    const int xr = tid >> 2;
    const int xq = tid & 3;
    const __amdgpu_buffer_rsrc_t xbase = prompt_rsrc(a.x);
    uint32_t xo[XC];  // a lane's byte offsets of its rows' chunks
#pragma unroll
    for (int i = 0; i < XC; ++i) {
        const int r = xr + XR * i;
        xo[i] = static_cast<uint32_t>(m0 + (r < rows ? r : 0)) * static_cast<uint32_t>(2 * a.k) + 16 * xq;
    }
    const int c = tid / P;
    const int part = tid % P;
    Hook hook(a, n0 + c < a.n ? n0 + c : a.n - 1, part);
    auto last = [&](int s) { return s < stages ? s : stages - 1; };

    u32x4 X[XC];
    auto fetch_x = [&](int s) {
#pragma unroll
        for (int i = 0; i < XC; ++i) X[i] = prompt_load<u32x4>(xbase, xo[i], 64 * last(s));
    };
    auto put_x = [&](int p) {
#pragma unroll
        for (int i = 0; i < XC; ++i) *reinterpret_cast<u32x4*>(xs[p] + prompt_at(xr + XR * i, xq)) = X[i];
    };
    auto put_w = [&]<int J>(std::integral_constant<int, J>, int p) {
        hook.template put<J>([&](int v, u32x4 d) { *reinterpret_cast<u32x4*>(ws[p] + prompt_at(c, part * (4 / P) + v)) = d; });
    };

    // Wave (wm, wn): rows 64 wm.., columns 64 wn..; a lane reads row (or column) lane % 16 of each 16, its chunks
    // swizzled by lane / 2 % 4 (16-row tiles keep r % 16), and holds rows 2 i + lane / 16 of each output tile.
    const int wm = wave % WGM;
    const int wn = wave / WGM;
    const int rl = lane & 15;
    const int rh = lane >> 4;
    const int swz = (rl >> 1) & 3;
    const bool live = wm * 64 < rows;
    f32x8 acc[4][4];
#pragma unroll
    for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
        for (int nt = 0; nt < 4; ++nt) acc[mt][nt] = f32x8{0, 0, 0, 0, 0, 0, 0, 0};
    }
    auto frag = [&](const uint32_t* row, int t) {
        const u32x4 lo = *reinterpret_cast<const u32x4*>(row + 4 * ((2 * t) ^ swz));
        const u32x4 hi = *reinterpret_cast<const u32x4*>(row + 4 * ((2 * t + 1) ^ swz));
        return __builtin_bit_cast(bf16x16, __builtin_shufflevector(lo, hi, 0, 1, 2, 3, 4, 5, 6, 7));
    };
    auto compute = [&](int p) {
        if (!live) return;
#pragma unroll
        for (int t = 0; t < 2; ++t) {
            bf16x16 af[4], bf[4];
#pragma unroll
            for (int i = 0; i < 4; ++i) {
                af[i] = frag(xs[p] + (wm * 64 + i * 16 + rl) * 16, t);
                bf[i] = frag(ws[p] + (wn * 64 + i * 16 + rl) * 16, t);
            }
#pragma unroll
            for (int nt = 0; nt < 4; ++nt) {
#pragma unroll
                for (int mt = 0; mt < 4; ++mt) acc[mt][nt] = WmmaBF16::mma(af[mt], bf[nt], acc[mt][nt]);
            }
        }
    };

    // Stage s (slot s % D, buffer s % 2) computes while s + 1 goes into the other buffer, x of s + 2 and the weights
    // of s + 1 + D are fetched into the registers just freed.
    [&]<int... J>(std::integer_sequence<int, J...>) { (hook.template fetch<J>(last(J)), ...); }(std::make_integer_sequence<int, D>{});
    fetch_x(0);
    put_x(0);
    put_w(std::integral_constant<int, 0>{}, 0);
    hook.template fetch<0>(last(D));
    fetch_x(1);
    prompt_sync();
    auto step = [&]<int J>(std::integral_constant<int, J>, int s) {
        constexpr int N = (J + 1) % D;
        if (s + 1 < stages) {
            put_x(N & 1);
            put_w(std::integral_constant<int, N>{}, N & 1);
            fetch_x(s + 2);
            hook.template fetch<N>(last(s + 1 + D));
        }
        compute(J & 1);
        prompt_sync();
    };
    for (int s = 0; s < stages; s += D) {
        [&]<int... J>(std::integer_sequence<int, J...>) {
            ((s + J < stages ? step(std::integral_constant<int, J>{}, s + J) : void()), ...);
        }(std::make_integer_sequence<int, D>{});
    }

    if (a.out16 == nullptr) {
#pragma unroll
        for (int mt = 0; mt < 4; ++mt) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const int row = m0 + wm * 64 + mt * 16 + 2 * i + rh;
                if (row >= a.m) continue;
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) {
                    const int cc = n0 + wn * 64 + nt * 16 + rl;
                    if (cc < a.n) Epi::store(a, row, cc, acc[mt][nt][i]);
                }
            }
        }
        return;
    }
    // BF16 out through the wave's own LDS (no barrier): 32 rows of 64 at a time, 144-byte rows, then a row a lane in
    // 16-byte stores.
    constexpr int kPitch = 72;  // halves a staged row
    static_assert(NT / 32 * 32 * kPitch * 2 <= sizeof(lds), "the staged halves fit the tile's LDS");
    uint16_t* const st = reinterpret_cast<uint16_t*>(lds) + wave * 32 * kPitch;
    uint16_t* const out = static_cast<uint16_t*>(static_cast<void*>(a.out16));
#pragma unroll
    for (int h = 0; h < 2; ++h) {
#pragma unroll
        for (int m2 = 0; m2 < 2; ++m2) {
#pragma unroll
            for (int i = 0; i < 8; ++i) {
#pragma unroll
                for (int nt = 0; nt < 4; ++nt) st[(m2 * 16 + 2 * i + rh) * kPitch + nt * 16 + rl] = __builtin_bit_cast(uint16_t, narrow<WmmaBF16>(acc[2 * h + m2][nt][i]));
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
        const int row = m0 + wm * 64 + h * 32 + lane;
        const int c0 = n0 + wn * 64;
        if (row < a.m) {
            const u32x4* src = reinterpret_cast<const u32x4*>(st + lane * kPitch);
            uint16_t* dst = out + static_cast<long long>(row) * a.n + c0;
            if (c0 + 64 <= a.n && (a.n & 7) == 0) {
#pragma unroll
                for (int j = 0; j < 8; ++j) reinterpret_cast<u32x4*>(dst)[j] = src[j];
            } else {
                for (int j = 0; j < 64 && c0 + j < a.n; ++j) dst[j] = st[lane * kPitch + j];
            }
        }
        __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront", "local");
        __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront", "local");
    }
#else
    (void)a;
    __builtin_trap();
#endif
}

// x's group sums for the prompt tile: sums[g * m + r] = x[r, 64 g..64 g + 63] added in order from zero, in fp32.
// A wave a row: 32 groups at a time come in whole lines through LDS, then each lane adds one group.
__device__ inline void prompt_sums(const Affine& a) {
    constexpr int kLd = 36;  // words a staged group: 32 and 4 of pad
    __shared__ __attribute__((aligned(16))) uint32_t st[8][32 * kLd];
    const int lane = threadIdx.x & 31;
    const int wave = threadIdx.x >> 5;
    const int r = blockIdx.x * 8 + wave;
    const bool live = r < a.m;
    const int groups = a.k >> 6;
    const u32x4* src = reinterpret_cast<const u32x4*>(static_cast<const uint16_t*>(a.x) + static_cast<long long>(live ? r : 0) * a.k);
    uint32_t* mine = st[wave];
    for (int g0 = 0; g0 < groups; g0 += 32) {
        const int n = groups - g0 < 32 ? groups - g0 : 32;
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const int c = lane + 32 * j;  // chunk c: group c / 8, 16 bytes c % 8
            if (c < 8 * n) *reinterpret_cast<u32x4*>(mine + (c >> 3) * kLd + 4 * (c & 7)) = src[8 * g0 + c];
        }
        __syncthreads();
        if (live && lane < n) {
            float run = 0.f;
#pragma unroll
            for (int v = 0; v < 8; ++v) {
                const u32x4 u = *reinterpret_cast<const u32x4*>(mine + lane * kLd + 4 * v);
                const uint32_t w[4] = {u.x, u.y, u.z, u.w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    run += __uint_as_float(w[j] << 16);
                    run += __uint_as_float(w[j] & 0xffff0000u);
                }
            }
            const_cast<float*>(a.sums)[static_cast<long long>(g0 + lane) * a.m + r] = run;
        }
        __syncthreads();
    }
}

}  // namespace rocm
}  // namespace tf
