// SPDX-License-Identifier: MIT

// bf16 weights for the GEMM tile: a 16-byte chunk is 8 inputs of one row, staged as they are.
#ifndef K5_BF16_ROWS_H_
#define K5_BF16_ROWS_H_

#include <loomcxx/kernel.h>

namespace k5 {

struct Bf16Rows {
  typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
  struct Weights {
    const u32x4* w;  // [rows][k / 8]
    unsigned k;
  };
  typedef u32x4 Raw;
  // Chunk `part` (0..3) of each stage of row `row`, in 16-byte units.
  LOOM_INLINE static unsigned base(const Weights& w, unsigned row, unsigned part) { return row * (w.k / 8) + part; }
  LOOM_INLINE static Raw fetch(const Weights& w, unsigned base, unsigned stage) { return w.w[base + 4 * stage]; }
  LOOM_INLINE static Raw zero() { return {0, 0, 0, 0}; }
  LOOM_INLINE static Raw raw(u32x4 v) { return v; }
  LOOM_INLINE static u32x4 chunk(const Raw& r) { return r; }
  LOOM_INLINE static u32x4 decode(const Raw& r) { return r; }
};

}  // namespace k5

#endif  // K5_BF16_ROWS_H_
