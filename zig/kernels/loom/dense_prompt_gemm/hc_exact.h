// SPDX-License-Identifier: MIT

// The hyper-connection read-out's pointwise math by hipcc's gfx11 sequences (ROCm 10.1, -ffp-contract=off): same bits.
#ifndef K5_HC_EXACT_H_
#define K5_HC_EXACT_H_

#include <loomcxx/kernel.h>
#include <loomcxx/math.h>

namespace k5 {

LOOM_INLINE float bits(unsigned u) { return __builtin_bit_cast(float, u); }

// bf16 round to nearest even, NaN to 0x7fc0: the 16 bits. Both arms are formed first, so the choice is a select.
LOOM_INLINE unsigned bf16_bits(float v) {
  const unsigned b = __builtin_bit_cast(unsigned, v);
  const unsigned rn = (b + ((b >> 16) & 1u) + 0x7fffu) >> 16;
  return v != v ? 0x7fc0u : rn;
}

// Two bf16 inputs packed in a word, each bf16((x rv) s): the low half takes sl, the high half sh.
LOOM_INLINE unsigned norm_pair(unsigned w, float rv, float sl, float sh) {
  const float lo = bits(w << 16), hi = bits(w & 0xffff0000u);
  return bf16_bits((lo * rv) * sl) | (bf16_bits((hi * rv) * sh) << 16);
}

// v rounded to bf16 and widened again.
LOOM_INLINE float bf16_round(float v) { return bits(bf16_bits(v) << 16); }

// expf: x log2(e) in two parts, round to even, v_exp_f32, ldexp; 0 below -103.97, +inf above 88.72 (NaN kept).
LOOM_INLINE float exp_exact(float x) {
  const float ph = x * bits(0x3fb8aa3bu);
  float pl = loom::scalar::fmaf(bits(0x3fb8aa3bu), x, -ph);
  const float e = loom::scalar::roundevenf(ph);
  pl = loom::scalar::fmaf(x, bits(0x32a5705fu), pl);
  const float t = (ph - e) + pl;
  float r = loom::scalar::ldexpf(loom::scalar::approximate::exp2f(t), (int)e);
  r = bits(0xc2ce8ed0u) > x ? 0.0f : r;
  r = bits(0x42b17218u) < x ? bits(0x7f800000u) : r;
  return r;
}

LOOM_INLINE float sigmoid_exact(float x) { return 1.0f / (1.0f + exp_exact(-x)); }
LOOM_INLINE float silu_exact(float x) { return x / (1.0f + exp_exact(-x)); }

// rsqrtf: v_rsq_f32, inputs below the least normal scaled by 2^24 and the result by 2^12.
LOOM_INLINE float rsqrt_exact(float x) {
  const bool tiny = x < bits(0x00800000u);
  const float xs = x * bits(0x4b800000u);
  const float r = loom::scalar::approximate::rsqrtf(tiny ? xs : x);
  const float rs = r * bits(0x45800000u);
  return tiny ? rs : r;
}

}  // namespace k5

#endif  // K5_HC_EXACT_H_
