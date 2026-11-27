// SPDX-License-Identifier: MIT

// silu(g) x u to bf16 by hipcc's IEEE expf sequence (README: SwiGLU), so the bytes equal the HIP kernels'.
#ifndef K4_SWIGLU_EXACT_H_
#define K4_SWIGLU_EXACT_H_

#include <loomcxx/kernel.h>
#include <loomcxx/math.h>

namespace k4 {

LOOM_INLINE float exp_neg(float g) {
  const float ph = g * __builtin_bit_cast(float, 0xbfb8aa3bu);
  float pl = loom::scalar::fmaf(__builtin_bit_cast(float, 0xbfb8aa3bu), g, -ph);
  const float e = loom::scalar::roundevenf(ph);
  pl = loom::scalar::fmaf(g, __builtin_bit_cast(float, 0xb2a5705fu), pl);
  const float t = (ph - e) + pl;
  float r = loom::scalar::ldexpf(loom::scalar::approximate::exp2f(t), (int)e);
  r = __builtin_bit_cast(float, 0x42ce8ed0u) < g ? 0.0f : r;
  r = __builtin_bit_cast(float, 0xc2b17218u) > g ? __builtin_bit_cast(float, 0x7f800000u) : r;
  return r;
}

LOOM_INLINE __bf16 swiglu_bf16(float g, float u) {
  const float v = (g / (1.0f + exp_neg(g))) * u;
  const unsigned b = __builtin_bit_cast(unsigned, v);
  const unsigned r = v != v ? 0x7fc0u : (b + ((b >> 16) & 1u) + 0x7fffu) >> 16;
  return __builtin_bit_cast(__bf16, (unsigned short)r);
}

}  // namespace k4

#endif  // K4_SWIGLU_EXACT_H_
