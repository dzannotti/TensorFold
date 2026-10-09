// fn_gdn_io.cu, plus prompt forms that run U (row, head) units a block (blockDim (128, U)): each unit's threads,
// warps, lanes and operations are front_kernel's / back_kernel's, so the same bits with U-times fewer blocks.
#include "../cuda/fn_gdn_io.cu"

namespace tf_fn_gdn_io {

// front_kernel, unit (blockIdx.x, blockIdx.y U + threadIdx.y); a block's units are all key heads or all value heads.
template <int NK, int NV, int U>
__global__ void __launch_bounds__(128 * U) front_kernel_u(
        const __nv_bfloat16* __restrict__ P, const long long* __restrict__ conv_ptrs, const int* __restrict__ sid,
        const int* __restrict__ win, const __nv_bfloat16* __restrict__ cw, const float* __restrict__ a_log,
        const float* __restrict__ dt_bias, float* __restrict__ q, float* __restrict__ k,
        __nv_bfloat16* __restrict__ v, float* __restrict__ g, float* __restrict__ beta) {
    static_assert(NK % U == 0 && NV % U == 0);
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int r = blockIdx.x, head = blockIdx.y * U + threadIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    const auto* cs = reinterpret_cast<const __nv_bfloat16*>(conv_ptrs[sid[r]]);
    __shared__ float xs_[U][2][DK];
    auto& xs = xs_[threadIdx.y];
    if (blockIdx.y < NK / U) {
        xs[0][t] = conv_act(P, cs, cw, win, r, head * DK + t, C, PW);
        xs[1][t] = conv_act(P, cs, cw, win, r, NK * DK + head * DK + t, C, PW);
        __syncthreads();
        if (warp < 2) {
            float v4[4], ss = 0.0f;
#pragma unroll
            for (int i = 0; i < 4; ++i) { v4[i] = xs[warp][lane * 4 + i]; ss = ss + v4[i] * v4[i]; }
            ss = warp_sum(ss);
            float inv = 1.0f / sqrtf(ss + 1e-6f);
            if (warp == 0) inv = inv * (1.0f / sqrtf((float)DK));
            float* out = (warp == 0 ? q : k) + ((size_t)r * NK + head) * DK + lane * 4;
#pragma unroll
            for (int i = 0; i < 4; ++i) out[i] = v4[i] * inv;
        }
        return;
    }
    const int hv = head - NK;
    v[((size_t)r * NV + hv) * DV + t] = __float2bfloat16_rn(conv_act(P, cs, cw, win, r, 2 * NK * DK + hv * DV + t,
                                                                     C, PW));
    if (t == 0) {
        const float b = __bfloat162float(P[(size_t)r * PW + C + NV * DV + hv]);
        const float a = __bfloat162float(P[(size_t)r * PW + C + NV * DV + NV + hv]);
        g[r * NV + hv] = expf(-expf(a_log[hv]) * softplusf_(a + dt_bias[hv]));
        beta[r * NV + hv] = bf(sigmoidf_(b));
    }
}

// back_kernel, unit (blockIdx.x, blockIdx.y U + threadIdx.y).
template <int NK, int NV, int U>
__global__ void __launch_bounds__(128 * U) back_kernel_u(
        const __nv_bfloat16* __restrict__ y, const __nv_bfloat16* __restrict__ P,
        const __nv_bfloat16* __restrict__ norm_w, float eps, __nv_bfloat16* __restrict__ out,
        float* __restrict__ xs) {
    static_assert(NV % U == 0);
    constexpr int C = 2 * NK * DK + NV * DV, PW = C + NV * DV + 2 * NV;
    const int r = blockIdx.x, hv = blockIdx.y * U + threadIdx.y, t = threadIdx.x, warp = t >> 5, lane = t & 31;
    __shared__ float ys_[U][DV];
    __shared__ float rinv_[U];
    float* ys = ys_[threadIdx.y];
    float& rinv = rinv_[threadIdx.y];
    ys[t] = __bfloat162float(y[((size_t)r * NV + hv) * DV + t]);
    __syncthreads();
    if (warp == 0) {
        float ss = 0.0f;
#pragma unroll
        for (int i = 0; i < 4; ++i) { const float yy = ys[lane * 4 + i]; ss = ss + yy * yy; }
        ss = warp_sum(ss);
        if (lane == 0) rinv = 1.0f / sqrtf(ss / (float)DV + eps);
    }
    __syncthreads();
    const float yn = bf(bf(ys[t] * rinv) * __bfloat162float(norm_w[t]));
    const float z = __bfloat162float(P[(size_t)r * PW + C + hv * DV + t]);
    const float o = bf(yn * sigmoidf_(z));
    out[(size_t)r * NV * DV + hv * DV + t] = __float2bfloat16_rn(o);
    const float gs = warp_sum(o);
    if (lane == 0) xs[(size_t)r * (NV * DV / 32) + hv * (DV / 32) + warp] = gs;
}

}  // namespace tf_fn_gdn_io

// One GPU (16, 48), 8 units a block (cuda_kernels.zig gdnFront / gdnBack on prompt rows).
template __global__ void tf_fn_gdn_io::front_kernel_u<16, 48, 8>(const __nv_bfloat16*, const long long*, const int*,
    const int*, const __nv_bfloat16*, const float*, const float*, float*, float*, __nv_bfloat16*, float*, float*);
template __global__ void tf_fn_gdn_io::back_kernel_u<16, 48, 8>(const __nv_bfloat16*, const __nv_bfloat16*,
    const __nv_bfloat16*, float, __nv_bfloat16*, float*);
