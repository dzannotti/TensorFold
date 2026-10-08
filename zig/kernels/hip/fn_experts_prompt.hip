// fn_experts_prompt.cu on gfx11 (HIP rewrite, the same extern "C" entry points and arguments): nvfp4_wmma.h's unit
// with T row tiles of 16 a pass, each group's code words decoded and scales converted once for all T. Every row's
// bits are fn_nvfp4_experts.hip's (the same WMMA and fmaf sequence a row; T moves only which rows share a decode).
#include "nvfp4_wmma.h"

#define TF_PROMPT_ARGS                                                                                              \
  const __hip_bfloat16 *__restrict__ X, int x_stride, int slots, const uint4 *__restrict__ W,                     \
      const float *__restrict__ scale, int KG, int NB, const int *__restrict__ items,                              \
      const int *__restrict__ counts, const int *__restrict__ members, void *__restrict__ out, int N, float limit, \
      int skip
#define TF_PROMPT_BODY(M, EPI, T, CT, MS, MOFF, GATE)                                                           \
  tf_nvfp4w::body<M, EPI, T, CT, 1, MS, MOFF>(reinterpret_cast<const uint16_t*>(X), x_stride, slots,             \
                                              reinterpret_cast<const uint32_t*>(W), scale, KG, NB, items, counts,  \
                                              members, out, N, limit, skip, GATE, blockIdx.x * 4 + (threadIdx.x >> 5), \
                                              gridDim.x * 4)

// gate/up SwiGLU (M 2, epilogue 2), down fp32 (M 1, epilogue 0) and bf16 (3); 4 waves a block, one group's loads at a
// time (the waves hide them). Rows a pass (T) and n16 tiles a wave (CT) are the fastest on gfx1151 for each name's
// plan items (2048-row chunk, 96 experts, A/B on a shared GPU: gate/up T 2 CT 2 3.2-3.4 ms vs 4.6 for
// nvfp4_expert_kernel; down T 2 CT 2 2.2 ms, T 4 CT 2 1.7 at 64-pair items, vs 3.2); T 3 and 4 gate/up spill and so
// take T 2, and the CUDA names' "o" forms (4 CTAs an SM) are their plain ones. None of this moves a bit.
#define TF_PROMPT4(NAME, M, EPI, T)                                                                \
  extern "C" __global__ void __launch_bounds__(128) NAME(TF_PROMPT_ARGS) {                         \
    TF_PROMPT_BODY(M, EPI, T, 2, M, 0, nullptr);                                                   \
  }
TF_PROMPT4(fn_prompt4_gu_t2, 2, 2, 2)
TF_PROMPT4(fn_prompt4_gu_t3, 2, 2, 2)
TF_PROMPT4(fn_prompt4_gu_t4, 2, 2, 2)
TF_PROMPT4(fn_prompt4_f32_t2, 1, 0, 2)
TF_PROMPT4(fn_prompt4_f32_t4, 1, 0, 4)
TF_PROMPT4(fn_prompt4_b16_t2, 1, 3, 2)
TF_PROMPT4(fn_prompt4_b16_t4, 1, 3, 4)
TF_PROMPT4(fn_prompt4_gu_t2o, 2, 2, 2)
TF_PROMPT4(fn_prompt4_f32_t2o, 1, 0, 2)
TF_PROMPT4(fn_prompt4_b16_t2o, 1, 3, 2)

// gate/up in two one-matrix passes: the gate's bf16(sums x scale) into `out`, then the up pass's SwiGLU reading them
extern "C" __global__ void __launch_bounds__(128) fn_prompt4_gate_t2o(TF_PROMPT_ARGS) {
  TF_PROMPT_BODY(1, 3, 2, 2, 2, 0, nullptr);
}
extern "C" __global__ void __launch_bounds__(128)
    fn_prompt4_up_t2o(TF_PROMPT_ARGS, const __hip_bfloat16* __restrict__ gate) {
  TF_PROMPT_BODY(1, 5, 2, 2, 2, 1, reinterpret_cast<const uint16_t*>(gate));
}
