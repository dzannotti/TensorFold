// fn_nvfp4_shape.cu's expert_nt_kernel on gfx11 (HIP rewrite, same template and arguments): the one-warp-a-block
// launch of nvfp4_expert_kernel's units. Every wave's unit is nvfp4_wmma.h's 16-column unit whatever D and NT say
// (they set CUDA's load depth and n8 tiles a warp), so the bits are fn_nvfp4_experts.hip's.
#include "nvfp4_wmma.h"

namespace tf_fn_nvfp4_shape {

template <int M, int EPI, int WARPS, int D, int NT>
__global__ void __launch_bounds__(WARPS * 32)
    expert_nt_kernel(const __hip_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                     const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                     const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                     float limit, int skip) {
  tf_nvfp4w::body<M, EPI, 1, 1, 4>(reinterpret_cast<const uint16_t*>(X), x_stride, slots,
                             reinterpret_cast<const uint32_t*>(W), scale, KG, NB, items, counts, members, out, N,
                             limit, skip, nullptr, blockIdx.x * WARPS + (threadIdx.x >> 5), gridDim.x * WARPS);
}

}  // namespace tf_fn_nvfp4_shape

template __global__ void tf_fn_nvfp4_shape::expert_nt_kernel<2, 2, 1, 2, 1>(const __hip_bfloat16*, int, int,
    const uint4*, const float*, int, int, const int*, const int*, const int*, void*, int, float, int);
