// HIP decode: moe._topk_rows and experts.cu's one-block plan_kernel in one launch (fn_ops.cu includes it on HIP).
// Phase 1, a wave a row: Triton's _topk_rows as gfx1151 runs it (the AOT code object's instructions): the TOPK largest
// fp32 logits of the NE routed ones (the lower id among equal), ex = exp(l_k - l_0) by Triton's exp (v_exp_f32 of
// fma(x, log2 e, 64 if x log2 e < -126 else 0), times 2^-64 then), the running total fused as the compiler fuses it
// (fma(e, scale, total) from 0), weights bf16(ex / total) (IEEE divide, Triton's rounding: nearest even, NaN 0x7FFF),
// then the shared expert NE with bf16(1 / fma(e, scale, 1)) of -bf16(its logit). Phase 2: plan_kernel on the picks
// (P <= 1024 pairs), the same members, items and counts. tools/rocm/check_topk_plan is glue-check's comparison.

namespace tf_topk_plan {

constexpr int THREADS = 1024;
constexpr int EMAX = 1024;
constexpr int PSMALL = 1024;
constexpr int PER = 16;           // logits a lane holds (NE <= 512)

__device__ __forceinline__ float tri_exp(float x, float& raw) {
    const float log2e = 1.44269502162933349609375f;          // 0x3fb8aa3b
    const bool small = __fmul_rn(x, log2e) < -126.0f;
    raw = __builtin_amdgcn_exp2f(__builtin_fmaf(x, log2e, small ? 64.0f : 0.0f));
    return small ? 0x1p-64f : 1.0f;
}

__device__ __forceinline__ float tri_bf16(float v) {
    const uint32_t u = __float_as_uint(v);
    const uint32_t b = v != v ? 0x7FFFu : (u + ((u >> 16) & 1u) + 0x7FFFu) >> 16;
    return __uint_as_float(b << 16);
}

// experts.cu place_items (the same scan; T pairs an item)
__device__ int place_items(int c, int E, int T, int* __restrict__ items, int* __restrict__ counts) {
    __shared__ int wsum[3][32];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int tiles = (c + T - 1) / T;
    int a = c, b = tiles, u = c > 0;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int ua = __shfl_up(a, o, 32), ub = __shfl_up(b, o, 32), uu = __shfl_up(u, o, 32);
        if (lane >= o) {
            a += ua;
            b += ub;
            u += uu;
        }
    }
    if (lane == 31) {
        wsum[0][warp] = a;
        wsum[1][warp] = b;
        wsum[2][warp] = u;
    }
    __syncthreads();
    if (warp == 0) {
#pragma unroll
        for (int s = 0; s < 3; ++s) {
            int v = wsum[s][lane];
#pragma unroll
            for (int o = 1; o < 32; o <<= 1) {
                const int w = __shfl_up(v, o, 32);
                if (lane >= o) v += w;
            }
            wsum[s][lane] = v;
        }
    }
    __syncthreads();
    const int off = (warp ? wsum[0][warp - 1] : 0) + a - c;
    const int ioff = (warp ? wsum[1][warp - 1] : 0) + b - tiles;
    if (tid == 0) {
        counts[0] = wsum[1][31];
        counts[1] = wsum[2][31];
    }
    if (tid < E)
        for (int j = 0; j < tiles; ++j) {
            int* it = items + 3 * (ioff + j);
            it[0] = tid;
            it[1] = off + T * j;
            it[2] = min(T, c - T * j);
        }
    return off;
}

} // namespace tf_topk_plan

// logits [R, NL] fp32 (NE routed, then the shared gate) -> picks/weights [R, TOPK + 1] and the plan of the R (TOPK + 1)
// pairs over E = NE + 1 experts in items of T pairs. 1024 threads (32 waves), one block; R (TOPK + 1) <= 1024.
extern "C" __global__ void __launch_bounds__(1024) tf_fn_topk_plan_kernel(
        const float* __restrict__ L, int R, int NE, int NL, int TOPK, int* __restrict__ pick, float* __restrict__ wts,
        int T, int* __restrict__ members, int* __restrict__ items, int* __restrict__ counts) {
    using namespace tf_topk_plan;
    __shared__ int pk[PSMALL];
    __shared__ int cnt[EMAX];
    __shared__ int off[EMAX];
    const int tid = threadIdx.x, lane = tid & 31, warp = tid >> 5;
    const int slots = TOPK + 1, E = NE + 1, P = R * slots;
    for (int e = tid; e < EMAX; e += THREADS) cnt[e] = 0;
    for (int r = warp; r < R; r += THREADS / 32) {
        float v[PER];
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int e = j * 32 + lane;
            v[j] = e < NE ? L[static_cast<size_t>(r) * NL + e] : -__builtin_inff();
        }
        float top = -__builtin_inff();
        bool first = true;
        float total = 0.0f, mine = 0.0f;                    // lane k keeps ex_k
        int idx_k = 0;
        for (int k = 0; k < TOPK; ++k) {
            float m = v[0];
#pragma unroll
            for (int j = 1; j < PER; ++j) m = fmaxf(m, v[j]);
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) m = fmaxf(m, __shfl_xor(m, o, 32));
            int idx = 1 << 30;
#pragma unroll
            for (int j = PER - 1; j >= 0; --j)
                if (v[j] == m) idx = j * 32 + lane;
#pragma unroll
            for (int o = 16; o > 0; o >>= 1) idx = min(idx, __shfl_xor(idx, o, 32));
            if (first) {
                top = m;
                first = false;
            }
            float raw;
            const float sc = tri_exp(__fsub_rn(m, top), raw);
            const float ex = __fmul_rn(raw, sc);
            total = __builtin_fmaf(raw, sc, total);
            if (lane == k) {
                mine = ex;
                idx_k = idx;
            }
#pragma unroll
            for (int j = 0; j < PER; ++j)
                if (j * 32 + lane == idx) v[j] = -__builtin_inff();
        }
        const size_t base = static_cast<size_t>(r) * slots;
        if (lane < TOPK) {
            pick[base + lane] = idx_k;
            pk[base + lane] = idx_k;
            wts[base + lane] = tri_bf16(__fdiv_rn(mine, total));
        } else if (lane == TOPK) {
            const float sg = tri_bf16(L[static_cast<size_t>(r) * NL + NE]);
            float raw;
            const float sc = tri_exp(-sg, raw);
            pick[base + lane] = NE;
            pk[base + lane] = NE;
            wts[base + lane] = tri_bf16(__fdiv_rn(1.0f, __builtin_fmaf(raw, sc, 1.0f)));
        }
    }
    __syncthreads();
    // experts.cu plan_kernel on the shared picks
    for (int p = tid; p < P; p += THREADS) atomicAdd(&cnt[pk[p]], 1);
    __syncthreads();
    const int o = tf_topk_plan::place_items(tid < E ? cnt[tid] : 0, E, T, items, counts);
    if (tid < E) off[tid] = o;
    __syncthreads();
    for (int p = tid; p < P; p += THREADS) {
        const int e = pk[p];
        int rank = 0;
        for (int q = 0; q < p; ++q) rank += pk[q] == e;
        members[off[e] + rank] = p;
    }
}
