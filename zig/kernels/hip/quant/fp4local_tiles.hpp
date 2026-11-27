#pragma once

// FP4-Local kernels: the shared stream and gfx11 WMMA tiles over the NVFP4, MXFP8 and BF16-fold decoders.
// Decode bits follow k and the column alone (stream.hpp); prompt bits follow the fold of 128 (or a.group for BF16).

#include <type_traits>

#include "quant/act.hpp"
#include "quant/bf16_fold_decoder.hpp"
#include "quant/mlx_tiles.hpp"
#include "quant/mxfp8_decoder.hpp"
#include "quant/nvfp4_decoder.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/matrix_gemm.hpp"
#include "tiles/matrix_item.hpp"
#include "tiles/matrix_pair.hpp"
#include "tiles/stream.hpp"

namespace tf {
namespace rocm {

// The decoders by the index launches/fp4local.zig numbers them.
template <int F>
using Fp4LocalDecoder = std::conditional_t<F == 0, Nvfp4Decoder, std::conditional_t<F == 1, Mxfp8Decoder, Bf16FoldDecoder>>;

// The decode stream tile (or a plan's items, or a group's sides); PAIR: (gate | up) rows, out silu(gate) * up.
template <int F, int R, int CB, bool PAIR>
__global__ void __launch_bounds__(32 * kStreamWaves) fp4l_stream(Affine a, StreamSides sides, int lpc_log2) {
#if TF_DEVICE_BF16_DOT2
    using Epi = std::conditional_t<PAIR, StreamSwiglu<DotBF16>, StreamRound<DotBF16>>;
    stream_tile<Fp4LocalDecoder<F>, BF16Act, DotBF16, Epi, R, CB>(a, sides, lpc_log2, 0);
#else
    (void)a, (void)sides, (void)lpc_log2;
    __builtin_trap();
#endif
}

// The prompt's 128 x 128 matrix-core tile over the plain product or every item of a plan.
template <int F>
__global__ void __launch_bounds__(256) fp4l_wmma(Affine a) {
    wmma_gemm_tile<Fp4LocalDecoder<F>, BF16Act, WmmaBF16, GemmOut<WmmaBF16>, true>(a);
}

// The routed prompt's 64-row item tile: the same staging and sums, so the same bytes; no K-parallel passes.
struct Fp4LocalNoFew {
    using Dot = DotBF16;
    using Epi = GemmOut<DotBF16>;
    static constexpr bool every_pass = true;
};

template <int F>
__global__ void __launch_bounds__(256) fp4l_item(Affine a) {
    a.route.few = 0;
    wmma_item_tile<Fp4LocalDecoder<F>, BF16Act, WmmaBF16, GemmOut<WmmaBF16>, 4, Fp4LocalNoFew>(a);
}

// A routed (gate | up) prompt product with silu(gate) x up in the epilogue: act (pairs, n / 2) in a.out16.
template <int F>
__global__ void __launch_bounds__(256) fp4l_pair(Affine a, float limit) {
    wmma_pair_tile<Fp4LocalDecoder<F>, BF16Act, WmmaBF16>(a, limit);
}

}  // namespace rocm
}  // namespace tf
