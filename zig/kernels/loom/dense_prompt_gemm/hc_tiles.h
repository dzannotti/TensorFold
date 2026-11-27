// SPDX-License-Identifier: MIT

// The hyper-connection read-out's plug-in for the GEMM tile: the up projection's columns in stream order.
#ifndef K5_HC_TILES_H_
#define K5_HC_TILES_H_

#include <loomcxx/kernel.h>

#include "gemm_tile.h"

namespace k5 {

constexpr unsigned kStreams = 4;

typedef float f32x4 __attribute__((ext_vector_type(4)));

// The up's bf16 weights [kStreams dims][k]: column L (block of C = SPAN kStreams) is stream L % C / SPAN, dim below.
template <unsigned SPAN>
struct StreamRows {
  typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
  struct Weights {
    const u32x4* w;
    unsigned k, dims;
  };
  typedef u32x4 Raw;
  // Dimension SPAN (L / C) + L % SPAN of that stream.
  LOOM_INLINE static unsigned base(const Weights& w, unsigned col, unsigned part) {
    const unsigned c = SPAN * kStreams, j = col % c;
    return ((j / SPAN) * w.dims + (col / c) * SPAN + j % SPAN) * (w.k / 8) + part;
  }
  LOOM_INLINE static Raw fetch(const Weights& w, unsigned base, unsigned stage) { return w.w[base + 4 * stage]; }
  LOOM_INLINE static Raw zero() { return {0, 0, 0, 0}; }
  LOOM_INLINE static Raw raw(u32x4 v) { return v; }
  LOOM_INLINE static u32x4 chunk(const Raw& r) { return r; }
  LOOM_INLINE static u32x4 decode(const Raw& r) { return r; }
};

}  // namespace k5

#endif  // K5_HC_TILES_H_
