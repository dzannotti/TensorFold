// SPDX-License-Identifier: MIT

// loom-test / loom-benchmark cases: bytes against a dot2 replica of the stream tile's order, and row-count identity.
#include <loomcxx/check.h>
#include <loomcxx/kernel.h>
#include <loomcxx/vector.h>

#include "skinny_mxfp8.cxx"

namespace skc {

typedef __bf16 bf16x2 __attribute__((ext_vector_type(2)));
typedef float f32x1 __attribute__((ext_vector_type(1)));

static loom::kernel::configuration per_thread(unsigned total) { return {{(total + 255) / 256, 1, 1}, {256, 1, 1}}; }

// The bf16 of e4m3 code b x 2^-8 (exact), as mxfp8_cols.h decodes it.
LOOM_INLINE __bf16 code(unsigned b) {
  const unsigned short hb = (unsigned short)(((b & 127u) << 7) | ((b & 128u) << 8));
  return (__bf16)(float)__builtin_bit_cast(_Float16, hb);
}

// out[r][c] in README.md's order: lanes l, chunks q = ((t + rot) mod T) L + l, dot2 chains, fma by 2^(E - 119), xor tree.
template <unsigned K>
LOOM_INLINE void replica(unsigned total, const unsigned char* w, const unsigned char* e, const __bf16* x, float* out,
                         unsigned n, unsigned c1, unsigned c2, unsigned c3) {
  constexpr unsigned L = sk::lanes_of(K), C = K / 32, T = C / L;
  const unsigned i = loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x;
  if (i >= total) return;
  const unsigned r = i / n, c = i % n;
  const unsigned base = c >= c3 ? c3 : c >= c2 ? c2 : c >= c1 ? c1 : 0, rot = ((c - base) / 8) % T;
  float acc[L];
  for (unsigned l = 0; l < L; l++) {
    float a = 0.0f;
    for (unsigned t = 0; t < T; t++) {
      const unsigned q = ((t + rot) % T) * L + l;
      f32x1 d = {0.0f};
      for (unsigned p = 0; p < 16; p++) {
        const unsigned k = 32 * q + 2 * p;
        const bf16x2 wv = {code(w[c * K + k]), code(w[c * K + k + 1])};
        const bf16x2 xv = {x[r * K + k], x[r * K + k + 1]};
        d = loom::vector::dot2f(wv, xv, d);
      }
      a = loom::scalar::fmaf(d[0], sk::e8m0_mult(e[c * C + q]), a);
    }
    acc[l] = a;
  }
  for (unsigned o = L / 2; o >= 1; o /= 2)
    for (unsigned l = 0; l < o; l++) acc[l] = acc[l] + acc[l + o];
  out[i] = acc[0];
}

[[loom::kernel(per_thread)]]
void replica_2560(unsigned total, const unsigned char* w, const unsigned char* e, const __bf16* x, float* out,
                  unsigned n, unsigned c1, unsigned c2, unsigned c3) {
  replica<2560>(total, w, e, x, out, n, c1, c2, c3);
}

[[loom::kernel(per_thread)]]
void replica_6144(unsigned total, const unsigned char* w, const unsigned char* e, const __bf16* x, float* out,
                  unsigned n, unsigned c1, unsigned c2, unsigned c3) {
  replica<6144>(total, w, e, x, out, n, c1, c2, c3);
}

[[loom::kernel(per_thread)]]
void replica_640(unsigned total, const unsigned char* w, const unsigned char* e, const __bf16* x, float* out,
                 unsigned n, unsigned c1, unsigned c2, unsigned c3) {
  replica<640>(total, w, e, x, out, n, c1, c2, c3);
}

}  // namespace skc

// Codes: multiples of 5 (no NaN code 0x7f / 0xff), both signs; scales 2^-9 .. 2^4; x a ramp with +-0 inside.
#define SKC_INPUTS(M, N, K)                                                                                       \
  constexpr unsigned m = M, n = N, k = K, outs = m * n;                                                           \
  const auto w = loom::check::iota<n * k, unsigned char>(0, 5, 50);                                             \
  const auto e = loom::check::iota<n * k / 32, unsigned char>(118, 1, 14);                                        \
  const auto x = loom::check::iota<m * k, __bf16>((__bf16)-1.0f, (__bf16)0.0078125f, 257);                        \
  const auto want = loom::check::fill<outs, float>(0.0f);

// Fp32 out against the replica; the same rows from a 64-row launch byte-equal (rows are independent of the count).
#define SKC_SINGLE(NAME, KERNEL, KERNEL64, REPLICA, M, N, K)                                                      \
  LOOM_CHECK_CASE(NAME) {                                                                                         \
    SKC_INPUTS(M, N, K)                                                                                           \
    const auto out = loom::check::fill<outs, float>(0.0f);                                                        \
    const auto x64 = loom::check::iota<64 * k, __bf16>((__bf16)-1.0f, (__bf16)0.0078125f, 257);                   \
    const auto out64 = loom::check::fill<64 * n, float>(0.0f);                                                    \
    loom::check::launch<KERNEL>(loom::kernel::workload(m, n), (int)m, (int)n, w, e, x, out);                      \
    loom::check::launch<KERNEL64>(loom::kernel::workload(64u, n), 64, (int)n, w, e, x64, out64);                  \
    loom::check::launch<skc::REPLICA>(loom::kernel::workload(outs), outs, w, e, x, want, n, n, n, n);             \
    loom::check::expect_bitwise(out, want);                                                                       \
    loom::check::expect_bitwise(loom::check::slice<outs>(out64, 0), out);                                           \
  }

SKC_SINGLE(k2560_rows_1, skinny_mxfp8_k2560_f1_t4_f32, skinny_mxfp8_k2560_f4_t2_f32, replica_2560, 1, 1280, 2560)
SKC_SINGLE(k2560_rows_20, skinny_mxfp8_k2560_f2_t4_f32, skinny_mxfp8_k2560_f4_t2_f32, replica_2560, 20, 1280, 2560)
SKC_SINGLE(k2560_rows_37, skinny_mxfp8_k2560_f3_t2_f32, skinny_mxfp8_k2560_f4_t2_f32, replica_2560, 37, 1288, 2560)
SKC_SINGLE(k6144_rows_5, skinny_mxfp8_k6144_f1_t2_f32, skinny_mxfp8_k6144_f4_t1_f32, replica_6144, 5, 2560, 6144)
SKC_SINGLE(k6144_rows_64, skinny_mxfp8_k6144_f4_t1_f32, skinny_mxfp8_k6144_f4_t1_f32, replica_6144, 64, 2560, 6144)
SKC_SINGLE(k640_rows_24, skinny_mxfp8_k640_f2_t1_f32, skinny_mxfp8_k640_f4_t1_f32, replica_640, 24, 2560, 640)
