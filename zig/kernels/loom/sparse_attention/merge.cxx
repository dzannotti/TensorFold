// SPDX-License-Identifier: MIT

// CUDA's fn_attn_merge for sparse_attend (attend.cxx): a row's chunks in order, o / l rounded to bf16 (fp32 out).
#include <loomcxx/kernel.h>

#include "chunk_tile.h"

static loom::kernel::configuration configure_merge(unsigned rows, unsigned heads) {
  return {{rows, heads, 1}, {256, 1, 1}};
}

[[loom::kernel(configure_merge)]]
void sparse_merge(const float* parts_o, const float* parts_ms, const int* count, float* out, int heads) {
  using namespace sc;
  const unsigned r = loom::kernel::workgroup::id.x, h = loom::kernel::workgroup::id.y;
  const unsigned d = loom::kernel::workitem::id.x;
  const int chunks = (count[r] + (int)kChunk - 1) / (int)kChunk;
  float m = neg_inf(), l = 0.0f, o = 0.0f;
  for (int c = 0; c < chunks; c++) {
    const long long base = ((long long)r * kChunks + c) * heads + h;
    const float cm = parts_ms[base * 2], cl = parts_ms[base * 2 + 1], co = parts_o[base * kDims + d];
    const bool active = cl > 0.0f;
    const float next = active ? loom::scalar::maxnumf(m, cm) : m;
    const float a = active ? (m == neg_inf() ? 0.0f : tf_exp(m - next)) : 1.0f;
    const float b = active ? tf_exp(cm - next) : 0.0f;
    o = loom::scalar::fmaf(co, b, o * a);
    l = loom::scalar::fmaf(cl, b, l * a);
    m = next;
  }
  out[((long long)r * heads + h) * kDims + d] = bits((unsigned)to_bf16(o / l) << 16);
}
