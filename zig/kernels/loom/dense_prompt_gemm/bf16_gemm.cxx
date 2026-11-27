// SPDX-License-Identifier: MIT

// Dense bf16 prompt GEMMs: y = x . W^T, bf16 x and W, fp32 y; the HIP matrix tile's sums (README.md).
// bf16_gemm_<rows>x<cols>: one block a <rows> x <cols> tile.
#include <loomcxx/kernel.h>

#include "bf16_rows.h"
#include "gemm_tile.h"

typedef k5::Geometry<4, 2, 4> G128x128;
typedef k5::Geometry<4, 4, 4> G128x256;

template <class G>
static loom::kernel::configuration configure(unsigned m, unsigned n) {
  return {{(n + G::kCols - 1) / G::kCols, (m + G::kRows - 1) / G::kRows, 1}, {G::kThreads, 1, 1}};
}

#define K5_STORE(I, J)                                                                       \
  if (G::NT > J && wr * 32 + I * 16 < rows) {                                                \
    const unsigned c = n0 + wc * 16 * G::NT + J * 16 + (lane & 15);                          \
    [[loom::unroll]] for (unsigned e = 0; e < 8; e++) {                                      \
      const unsigned r = wr * 32 + I * 16 + 2 * e + (lane >> 4);                             \
      if (r < rows && c < un) out[(m0 + r) * un + c] = acc.a##I##J[e];                       \
    }                                                                                        \
  }

// Rows past m and columns past n are neither read past the buffers nor stored. K: a multiple of 64.
template <class G>
LOOM_INLINE void bf16_gemm(const __bf16* x, const __bf16* wt, float* out, int m, int n, int k, __bf16* lds) {
  const unsigned lane = loom::kernel::workitem::id.x & 31;
  const unsigned n0 = loom::kernel::workgroup::id.x * G::kCols, m0 = loom::kernel::workgroup::id.y * G::kRows;
  const unsigned um = (unsigned)m, un = (unsigned)n, uk = (unsigned)k;
  if (m0 >= um || n0 >= un || uk < 64 || uk % 64 != 0) return;
  const unsigned rows = um - m0 < G::kRows ? um - m0 : G::kRows;
  const unsigned cols = un - n0 < G::kCols ? un - n0 : G::kCols;
  const k5::Bf16Rows::Weights w = {(const k5::u32x4*)wt, uk};
  const k5::Acc acc = k5::gemm_tile<k5::Bf16Rows, G>(w, (const k5::u32x4*)x, uk, m0, rows, n0, cols, lds);
  const unsigned wave = (unsigned)loom::kernel::subgroup::id(), wr = wave % G::WR, wc = wave / G::WR;
  K5_STORE(0, 0) K5_STORE(0, 1) K5_STORE(0, 2) K5_STORE(0, 3) K5_STORE(0, 4)
  K5_STORE(1, 0) K5_STORE(1, 1) K5_STORE(1, 2) K5_STORE(1, 3) K5_STORE(1, 4)
}

#undef K5_STORE

[[loom::kernel(configure<G128x128>)]]
void bf16_gemm_128x128(const __bf16* x, const __bf16* w, float* out, int m, int n, int k) {
  LOOM_WORKGROUP __bf16 lds[G128x128::kArena];
  bf16_gemm<G128x128>(x, w, out, m, n, k, lds);
}

[[loom::kernel(configure<G128x256>)]]
void bf16_gemm_128x256(const __bf16* x, const __bf16* w, float* out, int m, int n, int k) {
  LOOM_WORKGROUP __bf16 lds[G128x256::kArena];
  bf16_gemm<G128x256>(x, w, out, m, n, k, lds);
}
