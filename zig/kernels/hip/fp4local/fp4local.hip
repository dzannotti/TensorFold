// FP4-Local's NVFP4, MXFP8 and BF16-fold products on the shared tiles, as launches/fp4local.zig resolves them.

#include "quant/fp4local_tiles.hpp"

namespace tf {
namespace rocm {

#define TF_FP4L_STREAM(F, R, CB) template __global__ void fp4l_stream<F, R, CB, false>(Affine, StreamSides, int);
#define TF_FP4L_PAIR(F, R, CB) template __global__ void fp4l_stream<F, R, CB, true>(Affine, StreamSides, int);
#define TF_FP4L_ROWS(MACRO, F, CB) MACRO(F, 1, CB) MACRO(F, 2, CB) MACRO(F, 4, CB)

// NVFP4: 4 code words a chunk, so up to 8 columns a lane; MXFP8: 8 words, up to 4; BF16: 16 words, 2.
TF_FP4L_ROWS(TF_FP4L_STREAM, 0, 2)
TF_FP4L_ROWS(TF_FP4L_STREAM, 0, 4)
TF_FP4L_STREAM(0, 1, 8)
TF_FP4L_STREAM(0, 2, 8)
TF_FP4L_ROWS(TF_FP4L_PAIR, 0, 2)
TF_FP4L_ROWS(TF_FP4L_PAIR, 0, 4)
TF_FP4L_PAIR(0, 1, 8)
TF_FP4L_PAIR(0, 2, 8)
TF_FP4L_ROWS(TF_FP4L_STREAM, 1, 2)
TF_FP4L_ROWS(TF_FP4L_STREAM, 1, 4)
TF_FP4L_ROWS(TF_FP4L_PAIR, 1, 2)
TF_FP4L_ROWS(TF_FP4L_PAIR, 1, 4)
TF_FP4L_ROWS(TF_FP4L_STREAM, 2, 2)

template __global__ void fp4l_wmma<0>(Affine);
template __global__ void fp4l_wmma<1>(Affine);
template __global__ void fp4l_wmma<2>(Affine);
template __global__ void fp4l_item<0>(Affine);
template __global__ void gufo::DenseF16GEMMKernel<256, 128, 2, 8, 1>(Affine);
template __global__ void gufo::DenseF16GEMMKernel<256, 128, 2, 8, 1, 5>(Affine);
template __global__ void fp4l_pair<0>(Affine, float);
template __global__ void fp4l_pair<1>(Affine, float);

}  // namespace rocm
}  // namespace tf
