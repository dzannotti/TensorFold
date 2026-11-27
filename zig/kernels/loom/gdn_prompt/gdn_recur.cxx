// SPDX-License-Identifier: MIT

// gdn_recur: a prompt's gated delta rule for 128 x 128 heads, the bytes of TensorFold's tf_gdn_prompt_128; see README.
#include <loomcxx/kernel.h>

#include "recur.h"

static loom::kernel::configuration configure_recur(unsigned value_heads) {
  return {{(unsigned)(gdn::kDim / gdn::kBlockRows), value_heads, 1}, {(unsigned)gdn::kThreads, 1, 1}};
}

[[loom::kernel(configure_recur)]]
void gdn_recur(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* state,
               float* y, int length, int key_heads, int value_heads) {
  LOOM_WORKGROUP float lds[gdn::kLds];
  gdn::recur(q, k, v, gate, beta, state, y, length, key_heads, value_heads, lds);
}
