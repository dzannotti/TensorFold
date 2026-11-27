// SPDX-License-Identifier: MIT

// hipcc's IEEE expf on gfx11 (ROCm 10.1 ocml, -ffp-contract=off): the same instructions, so the same bits.
#ifndef SA_EXP_EXACT_H_
#define SA_EXP_EXACT_H_

#include <loomcxx/kernel.h>
#include <loomcxx/math.h>

namespace sa {

LOOM_INLINE float bits(unsigned u) { return __builtin_bit_cast(float, u); }

// x log2(e) split in two parts, round to even, v_exp_f32, ldexp; 0 below -103.97, +inf above 88.72 (NaN kept).
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

}  // namespace sa

#endif  // SA_EXP_EXACT_H_
