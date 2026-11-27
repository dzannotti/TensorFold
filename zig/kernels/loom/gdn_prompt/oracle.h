// SPDX-License-Identifier: MIT

// Inputs and the reference scan for checks.cxx: a wave a state row, lane l sums keys l + 32 i, then lane0_sum.
#ifndef GDN_ORACLE_H_
#define GDN_ORACLE_H_

#include <loomcxx/kernel.h>

namespace gdno {

constexpr int kDim = 128;

static loom::kernel::configuration per_item(unsigned n) { return {{(n + 255) / 256, 1, 1}, {256, 1, 1}}; }

static loom::kernel::configuration per_row(unsigned value_heads) {
  return {{(unsigned)kDim, value_heads, 1}, {32, 1, 1}};
}

LOOM_INLINE unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

// Value i of stream `seed`: uniform in [lo, hi) on a 2^-24 grid.
[[loom::kernel(per_item)]]
void gen_uniform(unsigned n, float* out, unsigned seed, float lo, float hi) {
  const unsigned i = loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x;
  if (i < n) out[i] = lo + (hi - lo) * ((float)(mix(i * 0x9e3779b9u ^ seed) >> 8) * 5.9604645e-8f);
}

LOOM_INLINE float lane0_sum(float x) {
  [[loom::unroll]] for (unsigned s = 0; s < 5; s++) x = x + loom::kernel::subgroup::shuffle_xor(x, 16u >> s, 32).value;
  return x;
}

// tf_gated_delta's one-row wave (gated_delta_kernel<128>): the reference order, one launch over `length` rows.
[[loom::kernel(per_row)]]
void oracle_scan(const float* q, const float* k, const float* v, const float* gate, const float* beta, float* state,
                 float* y, int length, int key_heads, int value_heads) {
  const int lane = (int)loom::kernel::workitem::id.x, row = (int)loom::kernel::workgroup::id.x;
  const int head = (int)loom::kernel::workgroup::id.y, key_head = head / (value_heads / key_heads);
  float* at = state + ((long long)head * kDim + row) * kDim;
  float s[4];
  [[loom::unroll]] for (int i = 0; i < 4; i++) s[i] = at[lane + 32 * i];
  for (int t = 0; t < length; t++) {
    const long long step = (long long)t * value_heads + head;
    const float decay = gate[step], sb = beta[step];
    const float value = v[step * kDim + row];
    const long long key = ((long long)t * key_heads + key_head) * kDim;
    float remembered = 0.0f, ks[4];
    [[loom::unroll]] for (int i = 0; i < 4; i++) {
      ks[i] = k[key + lane + 32 * i];
      s[i] = s[i] * decay;
      remembered = remembered + s[i] * ks[i];
    }
    const float delta = (value - lane0_sum(remembered)) * sb;
    [[loom::unroll]] for (int i = 0; i < 4; i++) s[i] = s[i] + ks[i] * delta;
    float readout = 0.0f;
    [[loom::unroll]] for (int i = 0; i < 4; i++) readout = readout + s[i] * q[key + lane + 32 * i];
    readout = lane0_sum(readout);
    if (lane == 0) y[step * kDim + row] = readout;
  }
  [[loom::unroll]] for (int i = 0; i < 4; i++) at[lane + 32 * i] = s[i];
}

}  // namespace gdno

#endif  // GDN_ORACLE_H_
