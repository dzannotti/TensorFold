// SPDX-License-Identifier: MIT

// Prompt attention core (listed or first-n keys), bf16 KV, head_dim 256, 12 query heads a KV head; ABI in README.md.
#include <loomcxx/kernel.h>

#include "exp_exact.h"
#include "lanes_asm.h"

#ifndef SA_PARTS
#define SA_PARTS 16
#endif

constexpr unsigned kDims = 256, kGroup = 12, kParts = SA_PARTS, kTile = 32, kLaneDims = kDims / 32;
constexpr unsigned kMergeHeads = 2;
static_assert(32 * kParts == kMergeHeads * kDims, "the merge maps a work-item to a (head, dim) of a round");

typedef unsigned u32x4 __attribute__((ext_vector_type(4)));
typedef float f32x4 __attribute__((ext_vector_type(4)));

static loom::kernel::configuration configure_attend(unsigned kv_heads, unsigned rows) {
  return {{kv_heads, rows, 1}, {32 * kParts, 1, 1}};
}

namespace sa {

LOOM_INLINE float lo16(unsigned w) { return __builtin_bit_cast(float, w << 16); }
LOOM_INLINE float hi16(unsigned w) { return __builtin_bit_cast(float, w & 0xffff0000u); }

LOOM_INLINE void lds_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

LOOM_INLINE void wave_fence() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::subgroup, ordering::acq_rel>();
}

LOOM_INLINE float wave_max(float x) {
  [[loom::unroll]] for (unsigned s = 0; s < 5; s++)
    x = loom::scalar::maxnumf(x, loom::kernel::subgroup::shuffle_xor(x, 16u >> s, 32).value);
  return x;
}

// The butterfly of __shfl_xor over masks 16, 8, 4, 2, 1: every lane ends with the same tree's sum.
LOOM_INLINE float wave_sum(float x) {
  [[loom::unroll]] for (unsigned s = 0; s < 5; s++) x = x + loom::kernel::subgroup::shuffle_xor(x, 16u >> s, 32).value;
  return x;
}

}  // namespace sa

[[loom::kernel(configure_attend)]]
void sparse_attend(const float* q, const __bf16* k, const __bf16* v, const int* ids, const int* count,
                   const int* sparse, float* out, long long id_stride, long long cap, int heads, int kv_heads, int dims,
                   float scale, int parts) {
  if (dims != (int)kDims || heads != kv_heads * (int)kGroup || parts != (int)kParts) return;
  LOOM_WORKGROUP float qs[kGroup * kDims];
  LOOM_WORKGROUP float ms[kParts * kGroup * 2];
  LOOM_WORKGROUP float scratch[kMergeHeads * kParts * kDims];
  LOOM_WORKGROUP int kids[kParts * kTile];
  float* const red = scratch;
  const unsigned g = loom::kernel::workgroup::id.x, r = loom::kernel::workgroup::id.y;
  const unsigned tid = loom::kernel::workitem::id.x, lane = tid & 31;
  const unsigned part = (unsigned)loom::kernel::subgroup::id();
  const int n = count[r];
  const bool listed = sparse[r] != 0;
  const int lo = (int)((long long)part * n / kParts), hi = (int)((long long)(part + 1) * n / kParts);
  const long long base = (long long)g * cap;
  const float* qrow = q + ((long long)r * heads + g * kGroup) * kDims;
  for (unsigned i = tid; i < kGroup * kDims; i += 32 * kParts) qs[i] = qrow[i] * scale;
  sa::lds_barrier();

  float m[kGroup], l[kGroup], o[kGroup][kLaneDims];
  [[loom::unroll]] for (unsigned h = 0; h < kGroup; h++) {
    m[h] = -sa::bits(0x7f800000u);
    l[h] = 0.0f;
    [[loom::unroll]] for (unsigned i = 0; i < kLaneDims; i++) o[h][i] = 0.0f;
  }
  int* const kid = kids + part * kTile;
  for (int t0 = lo; t0 < hi; t0 += (int)kTile) {
    const int tn = hi - t0 < (int)kTile ? hi - t0 : (int)kTile;
    const bool live = (int)lane < tn;
    const int at = t0 + (live ? (int)lane : 0);
    const int key = listed ? ids[(long long)r * id_stride + at] : at;
    const u32x4* krow = (const u32x4*)(k + (base + key) * kDims);
    float sc[kGroup];
    [[loom::unroll]] for (unsigned h = 0; h < kGroup; h++) sc[h] = 0.0f;
    // 16 dims a step, the next step's key words in flight during this one's products
    u32x4 w0 = krow[0], w1 = krow[1];
    for (unsigned c = 0; c < kDims / 16; c++) {
      const unsigned nc = c + 1 < kDims / 16 ? 2 * (c + 1) : 2 * c;
      const u32x4 n0 = krow[nc], n1 = krow[nc + 1];
      float kk[16] = {sa::lo16(w0.x), sa::hi16(w0.x), sa::lo16(w0.y), sa::hi16(w0.y), sa::lo16(w0.z), sa::hi16(w0.z),
                      sa::lo16(w0.w), sa::hi16(w0.w), sa::lo16(w1.x), sa::hi16(w1.x), sa::lo16(w1.y), sa::hi16(w1.y),
                      sa::lo16(w1.z), sa::hi16(w1.z), sa::lo16(w1.w), sa::hi16(w1.w)};
      [[loom::unroll]] for (unsigned g6 = 0; g6 < kGroup / 6; g6++) {
        const float* qq = qs + 6 * g6 * kDims + c * 16 + (lane & 15);
        float* const a = sc + 6 * g6;
        const sa::F6 r6 = sa::qk16x6({a[0], a[1], a[2], a[3], a[4], a[5]},
                                     {qq[0], qq[kDims], qq[2 * kDims], qq[3 * kDims], qq[4 * kDims], qq[5 * kDims]},
                                     kk[0], kk[1], kk[2], kk[3], kk[4], kk[5], kk[6], kk[7], kk[8], kk[9], kk[10],
                                     kk[11], kk[12], kk[13], kk[14], kk[15]);
        a[0] = r6.a0;
        a[1] = r6.a1;
        a[2] = r6.a2;
        a[3] = r6.a3;
        a[4] = r6.a4;
        a[5] = r6.a5;
      }
      w0 = n0;
      w1 = n1;
    }
    sa::wave_fence();
    kid[lane] = key;
    float ep[kGroup];
    [[loom::unroll]] for (unsigned h = 0; h < kGroup; h++) {
      float top = sa::wave_max(live ? sc[h] : -sa::bits(0x7f800000u));
      top = loom::scalar::maxnumf(top, m[h]);
      const float e = live ? sa::exp_exact(sc[h] - top) : 0.0f;
      const float sum = sa::wave_sum(e);
      const float f = sa::exp_exact(m[h] - top);
      l[h] = l[h] * f + sum;
      m[h] = top;
      ep[h] = e;
      [[loom::unroll]] for (unsigned i = 0; i < kLaneDims; i++) o[h][i] = o[h][i] * f;
    }
    sa::wave_fence();
    // values 4 keys a step, the next 4 keys' rows in flight; keys past tn load a live row and add nothing
    const u32x4* vlane = (const u32x4*)v + lane;
    u32x4 x0 = vlane[(base + kid[0]) * (kDims / 8)];
    u32x4 x1 = vlane[(base + kid[1 < tn ? 1 : 0]) * (kDims / 8)];
    u32x4 x2 = vlane[(base + kid[2 < tn ? 2 : 0]) * (kDims / 8)];
    u32x4 x3 = vlane[(base + kid[3 < tn ? 3 : 0]) * (kDims / 8)];
    for (int j0 = 0; j0 < tn; j0 += 4) {
      const int a0 = j0 + 4 < tn ? j0 + 4 : 0, a1 = j0 + 5 < tn ? j0 + 5 : 0;
      const int a2 = j0 + 6 < tn ? j0 + 6 : 0, a3 = j0 + 7 < tn ? j0 + 7 : 0;
      const u32x4 y0 = vlane[(base + kid[a0]) * (kDims / 8)], y1 = vlane[(base + kid[a1]) * (kDims / 8)];
      const u32x4 y2 = vlane[(base + kid[a2]) * (kDims / 8)], y3 = vlane[(base + kid[a3]) * (kDims / 8)];
      [[loom::unroll]] for (unsigned e = 0; e < 4; e++) {
        const int j = j0 + (int)e;
        if (j < tn) {
          const u32x4 w = e == 0 ? x0 : (e == 1 ? x1 : (e == 2 ? x2 : x3));
          const float v0 = sa::lo16(w.x), v1 = sa::hi16(w.x), v2 = sa::lo16(w.y), v3 = sa::hi16(w.y);
          const float v4 = sa::lo16(w.z), v5 = sa::hi16(w.z), v6 = sa::lo16(w.w), v7 = sa::hi16(w.w);
          [[loom::unroll]] for (unsigned h = 0; h < kGroup; h++) {
            const sa::O8 r8 = sa::pv8({o[h][0], o[h][1], o[h][2], o[h][3], o[h][4], o[h][5], o[h][6], o[h][7]}, ep[h],
                                      j, v0, v1, v2, v3, v4, v5, v6, v7);
            o[h][0] = r8.a0;
            o[h][1] = r8.a1;
            o[h][2] = r8.a2;
            o[h][3] = r8.a3;
            o[h][4] = r8.a4;
            o[h][5] = r8.a5;
            o[h][6] = r8.a6;
            o[h][7] = r8.a7;
          }
        }
      }
      x0 = y0;
      x1 = y1;
      x2 = y2;
      x3 = y3;
    }
  }
  sa::lds_barrier();

  // Merge in part order, as tf_fn_sparse_merge: w_p = e^(m_p - top), acc = sum_p o_p w_p, total = sum_p l_p w_p.
  if (lane == 0) {
    [[loom::unroll]] for (unsigned h = 0; h < kGroup; h++) {
      ms[(part * kGroup + h) * 2] = m[h];
      ms[(part * kGroup + h) * 2 + 1] = l[h];
    }
  }
  sa::lds_barrier();
  [[loom::unroll]] for (unsigned round = 0; round < kGroup / kMergeHeads; round++) {
    [[loom::unroll]] for (unsigned hh = 0; hh < kMergeHeads; hh++) {
      const unsigned h = round * kMergeHeads + hh;
      float top = -sa::bits(0x7f800000u);
      for (unsigned pp = 0; pp < kParts; pp++) top = loom::scalar::maxnumf(top, ms[(pp * kGroup + h) * 2]);
      const float w = m[h] == -sa::bits(0x7f800000u) ? 0.0f : sa::exp_exact(m[h] - top);
      float* dst = red + (hh * kParts + part) * kDims + lane * kLaneDims;
      f32x4 c0 = {o[h][0] * w, o[h][1] * w, o[h][2] * w, o[h][3] * w};
      f32x4 c1 = {o[h][4] * w, o[h][5] * w, o[h][6] * w, o[h][7] * w};
      *(f32x4*)dst = c0;
      *(f32x4*)(dst + 4) = c1;
    }
    sa::lds_barrier();
    {
      const unsigned hh = tid / kDims, d = tid % kDims, h = round * kMergeHeads + hh;
      float top = -sa::bits(0x7f800000u);
      for (unsigned pp = 0; pp < kParts; pp++) top = loom::scalar::maxnumf(top, ms[(pp * kGroup + h) * 2]);
      float total = 0.0f, acc = 0.0f;
      for (unsigned pp = 0; pp < kParts; pp++) {
        const float mp = ms[(pp * kGroup + h) * 2];
        const float w = mp == -sa::bits(0x7f800000u) ? 0.0f : sa::exp_exact(mp - top);
        total = total + ms[(pp * kGroup + h) * 2 + 1] * w;
        acc = acc + red[(hh * kParts + pp) * kDims + d];
      }
      out[((long long)r * heads + g * kGroup + h) * kDims + d] = total > 0.0f ? acc / total : 0.0f;
    }
    sa::lds_barrier();
  }
}

