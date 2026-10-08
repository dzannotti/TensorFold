// Flash Next's routed NVFP4 experts on gfx11: the HIP rewrite of zig/kernels/cuda/fn_nvfp4_experts.cu, the same
// template, arguments and instantiations; arithmetic and layout in nvfp4_wmma.h (no Python counterpart on ROCm).
// A wave takes a whole CUDA unit (32 columns: the host's grid gives one a warp); 2 (gate/up) or 4 (down) groups of
// loads in flight, the fastest at 1-128 decode rows (tools/rocm/mtp_experts_check.py --bench, A/B on a shared GPU).
#include "nvfp4_wmma.h"

namespace tf_fn_nvfp4_experts {

template <int M, int EPI, int WARPS>
__global__ void __launch_bounds__(WARPS * 32)
    nvfp4_expert_kernel(const __hip_bfloat16* __restrict__ X, int x_stride, int slots, const uint4* __restrict__ W,
                        const float* __restrict__ scale, int KG, int NB, const int* __restrict__ items,
                        const int* __restrict__ counts, const int* __restrict__ members, void* __restrict__ out, int N,
                        float limit, int skip) {
  tf_nvfp4w::body<M, EPI, 1, 2, (M == 2 ? 2 : 4)>(reinterpret_cast<const uint16_t*>(X), x_stride, slots,
                             reinterpret_cast<const uint32_t*>(W), scale, KG, NB, items, counts, members, out, N,
                             limit, skip, nullptr, blockIdx.x * WARPS + (threadIdx.x >> 5), gridDim.x * WARPS);
}

}  // namespace tf_fn_nvfp4_experts

// gate/up SwiGLU (M 2, epilogue 2), down fp32 (0) and bf16 (3)
#define TF_NVFP4(M, EPI) template __global__ void tf_fn_nvfp4_experts::nvfp4_expert_kernel<M, EPI, 4>( \
    const __hip_bfloat16*, int, int, const uint4*, const float*, int, int, const int*, const int*, const int*, \
    void*, int, float, int);
TF_NVFP4(2, 2)
TF_NVFP4(1, 0)
TF_NVFP4(1, 3)
