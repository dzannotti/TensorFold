#pragma once

// The coded decoders' kernels (quant/coded.hpp), named as launches/coded.zig looks them up: the stream tile for a
// window's products whose K is not a multiple of 256, the WMMA verify tile and its split for the rest, and the
// dequantizing prompt tile for a prompt's. Groups of 128 (a block-FP8 scale's k); BF16 has no terms.

#include "quant/act.hpp"
#include "quant/coded.hpp"
#include "quant/coded_dq.hpp"
#include "quant/mlx_tiles.hpp"
#include "tiles/epilogue.hpp"
#include "tiles/stream.hpp"
#include "tiles/wmma_verify.hpp"

namespace tf {
namespace rocm {

template <class Dec, int R, int CB>
__global__ void __launch_bounds__(32 * kStreamWaves) coded_dot2_stream(Affine a, StreamSides sides, int lpc_log2, int gshift) {
    stream_tile<Dec, IdentityAct<DotBF16>, DotBF16, StreamRound<DotBF16>, R, CB>(a, sides, lpc_log2, gshift);
}

template <class Dec, int BM>
__global__ void __launch_bounds__(256) coded_prompt_dq(Affine a) {
    if constexpr (requires { Dec::kTerms; } && Dec::kTerms) {
        prompt_dq_tile<ActOut<WmmaBF16>, Fp8Dq, BM, 32768 / BM>(a);
    } else {
        prompt_dq_tile<ActOut<WmmaBF16>, Bf16Dq, BM, 32768 / BM>(a);
    }
}

#define TF_CODED_STREAM(DEC, R, CB) template __global__ void coded_dot2_stream<DEC, R, CB>(Affine, StreamSides, int, int);
#define TF_CODED_ALL(DEC)                                                                    \
    TF_CODED_STREAM(DEC, 1, 2) TF_CODED_STREAM(DEC, 1, 4) TF_CODED_STREAM(DEC, 2, 2)          \
    TF_CODED_STREAM(DEC, 2, 4) TF_CODED_STREAM(DEC, 4, 2) TF_CODED_STREAM(DEC, 4, 4)
TF_CODED_ALL(Fp8Decoder)
TF_CODED_ALL(Bf16Decoder)
#undef TF_CODED_ALL
#undef TF_CODED_STREAM

#if TF_MATRIX
// BF16's 16-word chunks leave one-tile verify blocks and the 128-row prompt block short of registers (they spill):
// BF16 takes the two-tile verify block and the 256-row prompt block alone (the same bits at every shape)
#define TF_CODED_WMMA(DEC)                                                                 \
    template __global__ void coded_wmma_verify<DEC, 2, 128, 2>(Affine, StreamSides);       \
    template __global__ void coded_wmma_verify<DEC, 2, 128, 4>(Affine, StreamSides);       \
    template __global__ void coded_wmma_verify_split<DEC, 128, 4>(Affine, StreamSides);    \
    template __global__ void coded_wmma_verify_split<DEC, 128, 8>(Affine, StreamSides);    \
    template __global__ void coded_prompt_dq<DEC, 256>(Affine);
TF_CODED_WMMA(Fp8Decoder)
TF_CODED_WMMA(Bf16Decoder)
template __global__ void coded_wmma_verify<Fp8Decoder, 1, 128, 2>(Affine, StreamSides);
template __global__ void coded_wmma_verify<Fp8Decoder, 1, 128, 4>(Affine, StreamSides);
template __global__ void coded_prompt_dq<Fp8Decoder, 128>(Affine);
#undef TF_CODED_WMMA
#endif

}  // namespace rocm
}  // namespace tf
