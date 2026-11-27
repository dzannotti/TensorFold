// SPDX-License-Identifier: MIT

// The gated delta rule over a prompt's rows, a value head's state rows split over lanes; sums in lane0_sum's tree.
#ifndef GDN_RECUR_H_
#define GDN_RECUR_H_

#include <loomcxx/kernel.h>

namespace gdn {

// a + b c with the product and the sum each rounded, as the reference (hipcc -ffp-contract=off).
LOOM_INLINE float mad(float a, float b, float c) { return a + b * c; }

typedef float f32x4 __attribute__((ext_vector_type(4)));

constexpr int kDim = 128, kSteps = 8;
constexpr int L = 4, R = 2, W = 4, J = 32 / L, E = 4 * J, G = 32 / L;
constexpr int kThreads = 32 * W, kBlockRows = W * G * R;
constexpr int kLogL = 2, kLogJ = 3;
static_assert((1 << kLogL) == L && (1 << kLogJ) == J, "the tree's levels");
static_assert(kDim % kBlockRows == 0, "blocks tile a head's rows");

LOOM_INLINE void lds_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

// lane0_sum's tree on virtual lanes vl = p + L j: levels over j in registers, then the L real lanes by xor.
struct Parts {
  float v[J];
};

LOOM_INLINE float tree(Parts part) {
  [[loom::unroll]] for (int lv = 0; lv < kLogJ; lv++)
    [[loom::unroll]] for (int j = 0; j < J / 2; j++)
      if (j < (J >> (lv + 1))) part.v[j] = part.v[j] + part.v[j + (J >> (lv + 1))];
  float c = part.v[0];
  [[loom::unroll]] for (int lv = 0; lv < kLogL; lv++)
    c = c + loom::kernel::subgroup::shuffle_xor(c, (unsigned)(L >> (lv + 1)), 32).value;
  return c;
}

// Element e = p + L j + 32 i: lane p, register j + J i; float4 c of lane p staged at c L + p (adjacent 16 bytes).
LOOM_INLINE int slot(int e) {
  const int p = e % L, rest = e / L, m = rest % J + J * (rest / J);
  return ((m / 4) * L + p) * 4 + m % 4;
}

// A block's staging: k and q a step [128] in slot() order, its rows' v, the steps' gates and betas.
constexpr int kLds = 2 * kSteps * kDim + kSteps * kBlockRows + 2 * kSteps;
constexpr int kKq = kSteps * kDim / 4 / kThreads, kV = kSteps * kBlockRows / 4 / kThreads;
static_assert(kKq * kThreads * 4 == kSteps * kDim && kV * kThreads * 4 == kSteps * kBlockRows, "whole float4 a lane");
static_assert(kThreads >= 2 * kSteps, "a lane a gate or beta");

static_assert(kKq == 2 && kV == 1, "a stage's fetch: two float4 of k and of q, one of v a work-item");

// Float4 c of work-item tid's stage share: its step and its first element (k and q rows of kDim, v rows of kBlockRows).
LOOM_INLINE int kq_step(int tid, int c) { return (tid + c * kThreads) / (kDim / 4); }
LOOM_INLINE int kq_elem(int tid, int c) { return ((tid + c * kThreads) % (kDim / 4)) * 4; }

// Row t0 + step, or the prompt's last row past it (its values are never used).
LOOM_INLINE long long clamp_row(int t0, int step, int length) { return t0 + step < length ? t0 + step : length - 1; }

LOOM_INLINE void put4(float* row, int e, float x, float y, float z, float w) {
  row[slot(e)] = x, row[slot(e + 1)] = y, row[slot(e + 2)] = z, row[slot(e + 3)] = w;
}

// A float4 kept as four scalars: a loop-carried vector would live in scratch.
#define UNPACK(N, X)                                                                                             \
  {                                                                                                              \
    const f32x4 u_ = (X);                                                                                        \
    N##x = u_.x, N##y = u_.y, N##z = u_.z, N##w = u_.w;                                                          \
  }
#define PACK(N) N##x, N##y, N##z, N##w

// Work-item tid's stage share from row `ahead` on into k0 .. gb.
#define FETCH(AHEAD)                                                                                             \
  {                                                                                                              \
    const int ahead = (AHEAD);                                                                                   \
    const long long a0 = (clamp_row(ahead, kq_step(tid, 0), length) * key_heads + key_head) * kDim + kq_elem(tid, 0); \
    const long long a1 = (clamp_row(ahead, kq_step(tid, 1), length) * key_heads + key_head) * kDim + kq_elem(tid, 1); \
    const long long av = (clamp_row(ahead, vstep, length) * value_heads + head) * kDim + first + velem;          \
    const long long ag = clamp_row(ahead, tid % kSteps, length) * value_heads + head;                            \
    UNPACK(k0, *(const f32x4*)&k[a0]) UNPACK(k1, *(const f32x4*)&k[a1])                                          \
    UNPACK(q0, *(const f32x4*)&q[a0]) UNPACK(q1, *(const f32x4*)&q[a1])                                          \
    UNPACK(v0, *(const f32x4*)&v[av])                                                                            \
    gb = tid < kSteps ? gate[ag] : tid < 2 * kSteps ? beta[ag] : 0.0f;                                           \
  }

// One launch, lds kLds floats: grid (128 / kBlockRows, value_heads) of kThreads; state [heads][128][128] in place.
LOOM_INLINE void recur(const float* q, const float* k, const float* v, const float* gate, const float* beta,
                       float* state, float* y, int length, int key_heads, int value_heads, float* lds) {
  if (length <= 0) return;
  float* ks = lds;
  float* qs = ks + kSteps * kDim;
  float* vs = qs + kSteps * kDim;
  float* gs = vs + kSteps * kBlockRows;
  float* bs = gs + kSteps;
  const int tid = (int)loom::kernel::workitem::id.x;
  const int lane = tid & 31, wave = tid >> 5, p = lane % L, g = lane / L;
  const int first = (int)loom::kernel::workgroup::id.x * kBlockRows;
  const int head = (int)loom::kernel::workgroup::id.y;
  const int key_head = head / (value_heads / key_heads);
  int rows[R];
  [[loom::unroll]] for (int r = 0; r < R; r++) rows[r] = first + wave * G * R + r * G + g;
  float s[R][E];
  [[loom::unroll]] for (int r = 0; r < R; r++) {
    const float* at = state + ((long long)head * kDim + rows[r]) * kDim;
    [[loom::unroll]] for (int j = 0; j < J; j++)
      [[loom::unroll]] for (int i = 0; i < 4; i++) s[r][j + J * i] = at[p + L * j + 32 * i];
  }
  const int vstep = tid / (kBlockRows / 4), velem = (tid % (kBlockRows / 4)) * 4;
  // the stage's share in registers, fetched a stage ahead
  float k0x, k0y, k0z, k0w, k1x, k1y, k1z, k1w, q0x, q0y, q0z, q0w, q1x, q1y, q1z, q1w, v0x, v0y, v0z, v0w, gb;
  FETCH(0);
  for (int t0 = 0; t0 < length; t0 += kSteps) {
    lds_barrier();
    put4(ks + kq_step(tid, 0) * kDim, kq_elem(tid, 0), PACK(k0));
    put4(ks + kq_step(tid, 1) * kDim, kq_elem(tid, 1), PACK(k1));
    put4(qs + kq_step(tid, 0) * kDim, kq_elem(tid, 0), PACK(q0));
    put4(qs + kq_step(tid, 1) * kDim, kq_elem(tid, 1), PACK(q1));
    vs[tid * 4] = v0x, vs[tid * 4 + 1] = v0y, vs[tid * 4 + 2] = v0z, vs[tid * 4 + 3] = v0w;
    if (tid < 2 * kSteps) gs[tid] = gb;
    FETCH(t0 + kSteps < length ? t0 + kSteps : t0);
    const int n = length - t0 < kSteps ? length - t0 : kSteps;
    lds_barrier();
    for (int tt = 0; tt < n; tt++) {
      const float decay = gs[tt], sb = bs[tt];
      float kv[E], qv[E];
      const f32x4* k4 = (const f32x4*)&ks[tt * kDim];
      const f32x4* q4 = (const f32x4*)&qs[tt * kDim];
      [[loom::unroll]] for (int c = 0; c < E / 4; c++) {
        const f32x4 a = k4[c * L + p], b = q4[c * L + p];
        kv[4 * c] = a.x, kv[4 * c + 1] = a.y, kv[4 * c + 2] = a.z, kv[4 * c + 3] = a.w;
        qv[4 * c] = b.x, qv[4 * c + 1] = b.y, qv[4 * c + 2] = b.z, qv[4 * c + 3] = b.w;
      }
      float out[R];
      [[loom::unroll]] for (int r = 0; r < R; r++) {
        Parts part;
        [[loom::unroll]] for (int j = 0; j < J; j++) part.v[j] = 0.0f;
        [[loom::unroll]] for (int i = 0; i < 4; i++)
          [[loom::unroll]] for (int j = 0; j < J; j++) {
            const int m = j + J * i;
            s[r][m] = s[r][m] * decay;
            part.v[j] = mad(part.v[j], s[r][m], kv[m]);
          }
        const float delta = (vs[tt * kBlockRows + rows[r] - first] - tree(part)) * sb;
        [[loom::unroll]] for (int j = 0; j < J; j++) part.v[j] = 0.0f;
        [[loom::unroll]] for (int i = 0; i < 4; i++)
          [[loom::unroll]] for (int j = 0; j < J; j++) {
            const int m = j + J * i;
            s[r][m] = mad(s[r][m], kv[m], delta);
            part.v[j] = mad(part.v[j], s[r][m], qv[m]);
          }
        out[r] = tree(part);
      }
      if (p == 0) {
        float* yo = y + ((long long)(t0 + tt) * value_heads + head) * kDim;
        [[loom::unroll]] for (int r = 0; r < R; r++) yo[rows[r]] = out[r];
      }
    }
  }
  [[loom::unroll]] for (int r = 0; r < R; r++) {
    float* at = state + ((long long)head * kDim + rows[r]) * kDim;
    [[loom::unroll]] for (int j = 0; j < J; j++)
      [[loom::unroll]] for (int i = 0; i < 4; i++) at[p + L * j + 32 * i] = s[r][j + J * i];
  }
}

}  // namespace gdn

#undef FETCH
#undef UNPACK
#undef PACK

#endif  // GDN_RECUR_H_
