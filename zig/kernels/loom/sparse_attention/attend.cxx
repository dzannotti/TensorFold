// SPDX-License-Identifier: MIT

// Prompt attention core (listed or first-n keys), bf16 KV, head_dim 256, 12 query heads a KV head: CUDA attn_chunks'
// 512-key chunks on gfx11 bf16 WMMA, each chunk's o, m and l out for sparse_merge (merge.cxx). README.md: ABI, order.
#include <loomcxx/kernel.h>

#include "chunk_tile.h"

static loom::kernel::configuration configure_attend(unsigned rows, unsigned kv_heads) {
  return {{rows, kv_heads, sc::kChunks}, {64, 1, 1}};
}

namespace sc {

// Element e of a tile's softmax step (head 2e + half): CUDA's active rule, alpha, p and its bf16, the group sum.
struct Step {
  float next, alpha, p, group;
};

static LOOM_INLINE Step step(float m, float s, float tile_max) {
  const bool active = tile_max != neg_inf();
  const float next = active ? loom::scalar::maxnumf(m, tile_max) : m;
  const float alpha = active ? (m == neg_inf() ? 0.0f : tf_exp(m - next)) : 1.0f;
  const float p = s == neg_inf() ? 0.0f : tf_exp(s - next);
  return {next, alpha, p, sum8(p)};
}

}  // namespace sc

[[loom::kernel(configure_attend)]]
void sparse_attend(const float* q, const __bf16* k, const __bf16* v, const int* ids, const int* count,
                   const int* sparse, float* parts_o, float* parts_ms, long long id_stride, long long cap, int heads,
                   int kv_heads, int dims, float scale) {
  using namespace sc;
  if (dims != (int)kDims || heads != kv_heads * (int)kGroup) return;
  LOOM_WORKGROUP unsigned short qs[16 * kQPitch];
  LOOM_WORKGROUP unsigned short ps[16 * kPPitch];
  LOOM_WORKGROUP unsigned short vt[2 * 2 * 16 * 16];
  LOOM_WORKGROUP float red_m[2 * 16];
  LOOM_WORKGROUP float red_l[4 * 16];
  const unsigned r = loom::kernel::workgroup::id.x, g = loom::kernel::workgroup::id.y, c = loom::kernel::workgroup::id.z;
  const unsigned tid = loom::kernel::workitem::id.x, lane = tid & 31, col = lane & 15, half = lane >> 4;
  const unsigned wave = (unsigned)loom::kernel::subgroup::id();
  const int n = count[r];
  if ((int)(c * kChunk) >= n) return;
  const bool listed = sparse[r] != 0;
  const int* const row_ids = ids + (long long)r * id_stride;
  const unsigned short* const kh = (const unsigned short*)k + (long long)g * cap * kDims;
  const unsigned short* const vh = (const unsigned short*)v + (long long)g * cap * kDims;
  const float* const qrow = q + ((long long)r * heads + g * kGroup) * kDims;
  for (unsigned i = tid; i < 16 * kDims; i += 64) {
    const unsigned h = i / kDims, d = i % kDims;
    qs[h * kQPitch + d] = h < kGroup ? to_bf16(qrow[h * kDims + d]) : (unsigned short)0;
  }
  const f32x8 zero = {0, 0, 0, 0, 0, 0, 0, 0};
  const f32x8 ninf = {neg_inf(), neg_inf(), neg_inf(), neg_inf(), neg_inf(), neg_inf(), neg_inf(), neg_inf()};
  Part st = {ninf, zero, zero, zero, zero, zero, zero, zero, zero, zero};
  const int start = (int)(c * kChunk), tiles = n - start < (int)kChunk ? n - start : (int)kChunk;
  const unsigned short* const qcol = qs + col * kQPitch;
  for (int t0 = 0; t0 < tiles; t0 += (int)kTile) {
    lds_barrier();
    // scores: lane col of wave w takes key 16 w + col of the tile (element e: head 2e + half)
    const int ki = start + t0 + (int)(16 * wave + col);
    const bool valid = ki < n;
    const long long key = valid ? (listed ? row_ids[ki] : ki) : 0;
    const unsigned short* const krow = kh + key * kDims;
    const unsigned km = valid ? 0xffffffffu : 0u;
    f32x8 s = zero;
    [[loom::pipeline(2)]] for (unsigned ks = 0; ks < kDims / 16; ks++)
      s = mma(frag(qcol + ks * 16), masked(frag(krow + ks * 16), km), s);
#define SC_S(E) (valid ? s[E] * scale : neg_inf())
    const f32x8 sv = SC_V8(SC_S);
#undef SC_S
#define SC_T(E) max16(sv[E])
    const f32x8 tm = SC_V8(SC_T);
#undef SC_T
    if (col == 0) {
#define SC_W(E) red_m[wave * 16 + 2 * (E) + half] = tm[E];
      SC_W(0) SC_W(1) SC_W(2) SC_W(3) SC_W(4) SC_W(5) SC_W(6) SC_W(7)
#undef SC_W
    }
    lds_barrier();
#define SC_X(E) step(st.m[E], sv[E], loom::scalar::maxnumf(red_m[2 * (E) + half], red_m[16 + 2 * (E) + half]))
    const Step e0 = SC_X(0), e1 = SC_X(1), e2 = SC_X(2), e3 = SC_X(3), e4 = SC_X(4), e5 = SC_X(5), e6 = SC_X(6),
               e7 = SC_X(7);
#undef SC_X
#define SC_P(E, X)                                                        \
  ps[(2 * (E) + half) * kPPitch + 16 * wave + col] = to_bf16(X.p);        \
  if ((col & 7) == 0) red_l[(2 * wave + (col >> 3)) * 16 + 2 * (E) + half] = X.group;
    SC_P(0, e0) SC_P(1, e1) SC_P(2, e2) SC_P(3, e3) SC_P(4, e4) SC_P(5, e5) SC_P(6, e6) SC_P(7, e7)
#undef SC_P
    lds_barrier();
#define SC_SUM(E) (((red_l[2 * (E) + half] + red_l[16 + 2 * (E) + half]) + red_l[32 + 2 * (E) + half]) + red_l[48 + 2 * (E) + half])
    const f32x8 sum = SC_V8(SC_SUM);
#undef SC_SUM
    const f32x8 alpha = {e0.alpha, e1.alpha, e2.alpha, e3.alpha, e4.alpha, e5.alpha, e6.alpha, e7.alpha};
    using loom::scalar::fmaf;
#define SC_L(E) fmaf(st.l[E], alpha[E], sum[E])
    st.l = SC_V8(SC_L);
#undef SC_L
    st.m = f32x8{e0.next, e1.next, e2.next, e3.next, e4.next, e5.next, e6.next, e7.next};
    st.o0 = st.o0 * alpha;
    st.o1 = st.o1 * alpha;
    st.o2 = st.o2 * alpha;
    st.o3 = st.o3 * alpha;
    st.o4 = st.o4 * alpha;
    st.o5 = st.o5 * alpha;
    st.o6 = st.o6 * alpha;
    st.o7 = st.o7 * alpha;
    // values: per 16-key half kk, lanes of half hh stage key 16 kk + col's dims of N tile 2j + hh, read back by dim
    unsigned short* const tw = vt + wave * 512;
    [[loom::unroll]] for (unsigned kk = 0; kk < 2; kk++) {
      const int vi = start + t0 + (int)(16 * kk + col);
      const bool vvalid = vi < n;
      const long long vkey = vvalid ? (listed ? row_ids[vi] : vi) : 0;
      const unsigned short* const vrow = vh + vkey * kDims + wave * (kDims / 2) + half * 16;
      const unsigned vm = vvalid ? 0xffffffffu : 0u;
      const u32x8 pa = frag(ps + col * kPPitch + 16 * kk);
#define SC_ROUND(J, A, B)                                                                   \
  {                                                                                         \
    const u32x8 w = masked(frag(vrow + 32 * (J)), vm);                                      \
    unsigned short* const th = tw + half * 256 + col;                                       \
    [[loom::unroll]] for (unsigned e = 0; e < 8; e++) {                                     \
      th[(2 * e) * 16] = (unsigned short)w[e];                                              \
      th[(2 * e + 1) * 16] = (unsigned short)(w[e] >> 16);                                  \
    }                                                                                       \
    wave_fence();                                                                           \
    st.A = mma(pa, frag(tw + col * 16), st.A);                                              \
    st.B = mma(pa, frag(tw + 256 + col * 16), st.B);                                        \
    wave_fence();                                                                           \
  }
      SC_ROUND(0, o0, o1) SC_ROUND(1, o2, o3) SC_ROUND(2, o4, o5) SC_ROUND(3, o6, o7)
#undef SC_ROUND
    }
  }
  const long long base = ((long long)r * kChunks + c) * heads + g * kGroup;
  [[loom::unroll]] for (unsigned e = 0; e < 8; e++) {
    const unsigned h = 2 * e + half;
    if (h < kGroup) {
      float* const o = parts_o + (base + h) * kDims + wave * (kDims / 2) + col;
      o[0 * 16] = st.o0[e];
      o[1 * 16] = st.o1[e];
      o[2 * 16] = st.o2[e];
      o[3 * 16] = st.o3[e];
      o[4 * 16] = st.o4[e];
      o[5 * 16] = st.o5[e];
      o[6 * 16] = st.o6[e];
      o[7 * 16] = st.o7[e];
      if (wave == 0 && col == 0) {
        parts_ms[(base + h) * 2] = st.m[e];
        parts_ms[(base + h) * 2 + 1] = st.l[e];
      }
    }
  }
}
