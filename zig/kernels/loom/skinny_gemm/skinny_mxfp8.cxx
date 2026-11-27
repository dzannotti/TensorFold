// SPDX-License-Identifier: MIT

// Skinny MXFP8 x bf16 products for 1-64 rows (README.md): skinny_mxfp8_k<K>_f<F>_t<NT>_<out>, rows <= 16 F.
// <out>: f32 or bf16 [m][n]; x4: bf16 into four outputs split at columns c1 <= c2 <= c3 (multiples of 16).
#include <loomcxx/kernel.h>

#include "skinny_tile.h"

template <class G>
static unsigned blocks_of(unsigned w) {
  return sk::side_blocks(w, G::kTile, G::T);
}

template <class G>
static loom::kernel::configuration one(unsigned m, unsigned n) {
  (void)m;
  return {{blocks_of<G>(n), 1, 1}, {G::kThreads, 1, 1}};
}

template <class G>
static loom::kernel::configuration four(unsigned m, unsigned n, unsigned c1, unsigned c2, unsigned c3) {
  (void)m;
  const unsigned b = blocks_of<G>(c1) + blocks_of<G>(c2 - c1) + blocks_of<G>(c3 - c2) + blocks_of<G>(n - c3);
  return {{b, 1, 1}, {G::kThreads, 1, 1}};
}

template <class G>
LOOM_INLINE void single(int m, int n, const unsigned char* w, const unsigned char* e, const __bf16* x, void* o,
                        unsigned kind, float* sc, float* red) {
  const unsigned un = (unsigned)n;
  if (m < 1 || m > (int)(16 * G::F) || n < 1) return;
  const sk::Args a = {w, e, x, (unsigned)m, un};
  const sk::Sides s = {un, un, un, un, 1};
  const sk::Outs out = {o, o, o, o, kind};
  sk::skinny<G>(a, s, out, sc, red);
}

template <class G>
LOOM_INLINE void sides(int m, int n, int c1, int c2, int c3, const unsigned char* w, const unsigned char* e,
                       const __bf16* x, void* o0, void* o1, void* o2, void* o3, float* sc, float* red) {
  const unsigned un = (unsigned)n;
  if (m < 1 || m > (int)(16 * G::F) || n < 1 || c1 < 0 || c1 > c2 || c2 > c3 || c3 > n) return;
  if ((c1 | c2 | c3) % 16 != 0) return;
  const sk::Args a = {w, e, x, (unsigned)m, un};
  const sk::Sides s = {(unsigned)c1, (unsigned)c2, (unsigned)c3, un, 4};
  const sk::Outs out = {o0, o1, o2, o3, 2};
  sk::skinny<G>(a, s, out, sc, red);
}

#define SK_LDS(G)                 \
  LOOM_WORKGROUP float sc[G::kScales]; \
  LOOM_WORKGROUP float red[G::kRed];

#define SK_SINGLE(K, F, NT, A)                                                                                    \
  typedef sk::Geo<K, F, NT> G_##K##_##F##_##NT;                                                                \
  [[loom::kernel(one<G_##K##_##F##_##NT>)]]                                                                       \
  void skinny_mxfp8_k##K##_f##F##_t##NT##_f32(int m, int n, const unsigned char* w, const unsigned char* e,       \
                                              const __bf16* x, float* o) {                                        \
    SK_LDS(G_##K##_##F##_##NT)                                                                                    \
    single<G_##K##_##F##_##NT>(m, n, w, e, x, o, 0, sc, red);                                                            \
  }                                                                                                               \
  [[loom::kernel(one<G_##K##_##F##_##NT>)]]                                                                       \
  void skinny_mxfp8_k##K##_f##F##_t##NT##_bf16(int m, int n, const unsigned char* w, const unsigned char* e,      \
                                               const __bf16* x, __bf16* o) {                                      \
    SK_LDS(G_##K##_##F##_##NT)                                                                                    \
    single<G_##K##_##F##_##NT>(m, n, w, e, x, o, 1, sc, red);                                                            \
  }

#define SK_SIDES(K, F, NT, A)                                                                                     \
  [[loom::kernel(four<G_##K##_##F##_##NT>)]]                                                                      \
  void skinny_mxfp8_k##K##_f##F##_t##NT##_x4(int m, int n, int c1, int c2, int c3, const unsigned char* w,        \
                                             const unsigned char* e, const __bf16* x, __bf16* o0, __bf16* o1,     \
                                             __bf16* o2, __bf16* o3) {                                            \
    SK_LDS(G_##K##_##F##_##NT)                                                                                    \
    sides<G_##K##_##F##_##NT>(m, n, c1, c2, c3, w, e, x, o0, o1, o2, o3, sc, red);                                \
  }

SK_SINGLE(2560, 1, 4, 1)
SK_SINGLE(2560, 2, 4, 1)
SK_SINGLE(2560, 3, 2, 1)
SK_SINGLE(2560, 4, 2, 1)
SK_SIDES(2560, 1, 4, 1)
SK_SIDES(2560, 2, 4, 1)
SK_SIDES(2560, 3, 2, 1)
SK_SIDES(2560, 4, 2, 1)
SK_SINGLE(6144, 1, 2, 2)
SK_SINGLE(6144, 2, 2, 2)
SK_SINGLE(6144, 3, 1, 2)
SK_SINGLE(6144, 4, 1, 2)
SK_SINGLE(640, 1, 1, 1)
SK_SINGLE(640, 2, 1, 1)
SK_SINGLE(640, 3, 1, 1)
SK_SINGLE(640, 4, 1, 1)

#undef SK_SINGLE
#undef SK_SIDES
#undef SK_LDS
