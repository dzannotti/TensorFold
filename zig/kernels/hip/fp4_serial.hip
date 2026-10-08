// fn_ops.cu's tf_fn_fp4_serial_kernel on gfx11 (HIP rewrite, the same extern "C" name, arguments and launch:
// grid ((M + 15) / 16, N / 32), 128 threads): nvfp4.matmul with one K slice and bf16-pattern tables (`_fp4mm` SK 1,
// PACKED 0) in AMD Triton's bits. Its AMDGCN on gfx1151 (triton 3.6, amd_wmma version 1): for each 16-input block b
// in order, p = v_wmma_f32_16x16x16_bf16(x rows, the block's 16 x 16 weights, C = 0) and acc = v_fmac(p, S[b, n],
// acc), the bf16 store rounding to nearest. Here each of the CTA's first two waves owns 16 rows x 16 columns (its
// n16 tile of the CTA's 32) and walks all of K, D blocks of loads in flight; a WMMA output reads one row and one
// column, so the tile choice moves no bit. X [M, K] bf16 rows x_stride apart; W the 64-column tiles of 16 x 64
// blocks (tile t at t * (K / 64) * 64 * 64 elements, block b at b * 16 * 64, row k, column n % 64); S [K / 16, N] fp32.
#include <hip/hip_runtime.h>
#include <stdint.h>

#pragma clang fp contract(off)

namespace {

typedef short v16s __attribute__((ext_vector_type(16)));
typedef float v8f __attribute__((ext_vector_type(8)));
typedef uint32_t u4v __attribute__((ext_vector_type(4)));
typedef uint32_t u8v __attribute__((ext_vector_type(8)));

constexpr int D = 8;  // blocks of loads in flight a wave

struct Stage {
  u8v a, b;  // the lane's x row and weight column over the block's 16 inputs (element i = input i)
  float s;
};

__device__ __forceinline__ void load(Stage& st, const uint16_t* xr, bool xv, const uint16_t* wc, const float* sc,
                                     uint32_t b, uint32_t N) {
  if (xv) {
    const u4v* p = reinterpret_cast<const u4v*>(xr + b * 16);
    const u4v lo = p[0], hi = p[1];
    st.a = u8v{lo[0], lo[1], lo[2], lo[3], hi[0], hi[1], hi[2], hi[3]};
  } else {
    st.a = u8v{};
  }
  const uint16_t* w = wc + (size_t)b * 16 * 64;
#pragma unroll
  for (int i = 0; i < 8; ++i) st.b[i] = (uint32_t)w[(2 * i) * 64] | ((uint32_t)w[(2 * i + 1) * 64] << 16);
  st.s = sc[(size_t)b * N];
}

__device__ __forceinline__ uint16_t bf16(float x) {  // round to nearest even
  const uint32_t u = __float_as_uint(x);
  if ((u & 0x7FFFFFFFu) > 0x7F800000u) return (uint16_t)((u >> 16) | 0x40u);
  return (uint16_t)((u + 0x7FFFu + ((u >> 16) & 1u)) >> 16);
}

}  // namespace

extern "C" __global__ void __launch_bounds__(128) tf_fn_fp4_serial_kernel(
        const uint16_t* __restrict__ X, const uint16_t* __restrict__ W, const float* __restrict__ S, void* OUT,
        uint32_t M, uint32_t N, uint32_t K, uint32_t x_stride, uint32_t f32_out) {
  const int wave = threadIdx.x >> 5, lane = threadIdx.x & 31, n = lane & 15, hi = lane >> 4;
  if (wave >= 2) return;
  const uint32_t m0 = blockIdx.x * 16, col = blockIdx.y * 32 + wave * 16 + n;
  const uint32_t blocks = K / 16;
  const bool xv = m0 + n < M;
  const uint16_t* xr = X + (size_t)(xv ? m0 + n : 0) * x_stride;
  const uint16_t* wc = W + (size_t)(col / 64) * ((K / 64) * 64 * 64) + (col % 64);
  const float* sc = S + col;
  v8f acc = v8f{};
  Stage st[D];
#pragma unroll
  for (int d = 0; d < D - 1; ++d)
    if ((uint32_t)d < blocks) load(st[d], xr, xv, wc, sc, d, N);
  for (uint32_t b0 = 0; b0 < blocks; b0 += D) {
#pragma unroll
    for (int d = 0; d < D; ++d) {
      const uint32_t b = b0 + d;
      if (b < blocks) {
        if (b + D - 1 < blocks) load(st[(d + D - 1) % D], xr, xv, wc, sc, b + D - 1, N);
        const v8f p = __builtin_amdgcn_wmma_f32_16x16x16_bf16_w32((v16s)st[d].a, (v16s)st[d].b, v8f{});
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = __builtin_fmaf(p[i], st[d].s, acc[i]);
      }
    }
  }
#pragma unroll
  for (int i = 0; i < 8; ++i) {
    const uint32_t row = m0 + 2 * i + hi;
    if (row >= M) continue;
    if (f32_out) static_cast<float*>(OUT)[(size_t)row * N + col] = acc[i];
    else static_cast<uint16_t*>(OUT)[(size_t)row * N + col] = bf16(acc[i]);
  }
}
