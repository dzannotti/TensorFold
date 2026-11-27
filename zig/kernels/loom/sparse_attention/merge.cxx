// SPDX-License-Identifier: MIT

// The parts merge of sparse_attend (attend.cxx); ABI and sum order in README.md.
#include <loomcxx/kernel.h>

#include "wmma_tile.h"

static loom::kernel::configuration configure_merge(unsigned heads, unsigned rows) {
  return {{heads, rows, 1}, {256, 1, 1}};
}

// Block (h, r), a work-item a dim: the parts of head h of row r combined in part order (tf_fn_sparse_merge).
[[loom::kernel(configure_merge)]]
void sparse_merge(const float* parts_o, const float* parts_ms, float* out, int heads, int dims, int parts) {
  const unsigned h = loom::kernel::workgroup::id.x, r = loom::kernel::workgroup::id.y;
  const unsigned d = loom::kernel::workitem::id.x;
  const long long at = ((long long)r * heads + h) * parts;
  float top = sa::neg_inf();
  for (int p = 0; p < parts; p++) top = loom::scalar::maxnumf(top, parts_ms[(at + p) * 2]);
  for (int dd = (int)d; dd < dims; dd += 256) {
    float total = 0.0f, acc = 0.0f;
    for (int p = 0; p < parts; p++) {
      const float mp = parts_ms[(at + p) * 2];
      const float w = mp == sa::neg_inf() ? 0.0f : sa::tf_exp(mp - top);
      total = total + parts_ms[(at + p) * 2 + 1] * w;
      acc = acc + parts_o[(at + p) * dims + dd] * w;
    }
    out[((long long)r * heads + h) * dims + dd] = total > 0.0f ? sa::bits((unsigned)sa::to_bf16(acc / total) << 16) : 0.0f;
  }
}
