// SPDX-License-Identifier: MIT

// The skinny tile's arguments and stores: the non-finite repair, bf16 rounding and the four-way sides.
#ifndef SK_SKINNY_OUT_H_
#define SK_SKINNY_OUT_H_

#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

#include "mxfp8_cols.h"

namespace sk {

struct Args {
  const unsigned char* w;  // e4m3 [n][K]
  const unsigned char* e;  // e8m0 [n][K / 32]
  const __bf16* x;         // [m][K]
  unsigned m, n;
};

// The product's sides: [0, c1), [c1, c2), [c2, c3), [c3, n) when count is 4, else one side [0, n).
struct Sides {
  unsigned c1, c2, c3, n, count;
};

// Where a block works: its side, the side's first column and width, the tile's first group, its rotation.
struct Place {
  unsigned side, base, width, group0, rot;
};

// fp32 [m][n] (kind 0), bf16 [m][n] (1), or bf16 into side s's [m][width] (2).
struct Outs {
  void *o0, *o1, *o2, *o3;
  unsigned kind;
};

// val, or the f64 sum of row r against column col (w_i x_i in i order, no fma) when val is non-finite and an x is.
template <unsigned K>
LOOM_INLINE float repair(const Args& a, float val, unsigned r, unsigned col) {
  if (loom::scalar::isfinitef(val)) return val;
  double y = 0.0;
  bool bad = false;
  for (unsigned i = 0; i < K; i++) {
    const unsigned b = a.w[col * K + i];
    const unsigned short hb = (unsigned short)(((b & 127u) << 7) | ((b & 128u) << 8));
    const double wf = (double)__builtin_bit_cast(_Float16, hb);
    const double sd = (double)e8m0_mult(a.e[col * (K / 32) + i / 32]);
    const double wd = wf * sd;
    const double xd = (double)(float)a.x[r * K + i];
    const double pr = wd * xd;
    y = y + pr;
    if (!loom::scalar::isfinitef(xd)) bad = true;
  }
  return bad ? (float)y : val;
}

// val rounded to bf16 as the stream tile's epilogue does: nearest even, any NaN to 0x7fc0.
LOOM_INLINE unsigned short bf16_bits(float val) {
  const unsigned u = __builtin_bit_cast(unsigned, val);
  const unsigned rn = (u + ((u >> 16) & 1u) + 32767u) >> 16;
  return loom::scalar::isnanf(val) ? (unsigned short)0x7fc0 : (unsigned short)rn;
}

static LOOM_INLINE void put(const Outs& out, const Place& pl, unsigned r, unsigned local, unsigned n, float v) {
  if (out.kind == 0) {
    ((float*)out.o0)[r * n + pl.base + local] = v;
  } else if (out.kind == 1) {
    ((unsigned short*)out.o0)[r * n + pl.base + local] = bf16_bits(v);
  } else {
    void* d = pl.side == 0 ? out.o0 : pl.side == 1 ? out.o1 : pl.side == 2 ? out.o2 : out.o3;
    ((unsigned short*)d)[r * pl.width + local] = bf16_bits(v);
  }
}

}  // namespace sk

#endif  // SK_SKINNY_OUT_H_
