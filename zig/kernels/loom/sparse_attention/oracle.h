// SPDX-License-Identifier: MIT

// Serial replicas of the HIP reference's arithmetic (README: Sum order), one work-item an output, for checks.cxx.
#ifndef SA_ORACLE_H_
#define SA_ORACLE_H_

#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

#include "exp_exact.h"

namespace sao {

constexpr unsigned kHd = 256, kGroup = 12, kParts = 16, kTile = 32, kIdx = 128, kIdxHeads = 4;

static loom::kernel::configuration per_item(unsigned n) { return {{(n + 255) / 256, 1, 1}, {256, 1, 1}}; }

LOOM_INLINE unsigned item() { return loom::kernel::workgroup::id.x * 256 + loom::kernel::workitem::id.x; }

LOOM_INLINE float bf(__bf16 x) { return (float)x; }

LOOM_INLINE unsigned mix(unsigned x) {
  x ^= x >> 16;
  x *= 0x7feb352du;
  x ^= x >> 15;
  x *= 0x846ca68bu;
  return x ^ (x >> 16);
}

// A rough normal of std ~ `scale` rounded to bf16, some +0 and -0: the bits of value i of stream `seed`.
LOOM_INLINE unsigned short draw(unsigned i, unsigned seed, float scale) {
  const unsigned h = mix(i * 0x9e3779b9u ^ seed);
  const float u = (float)(h & 255) + (float)((h >> 8) & 255) + (float)((h >> 16) & 255) - 382.5f;
  const float x = u * (scale / 128.0f) * ((h >> 24) % 61 == 0 ? 8.0f : 1.0f);
  const unsigned b = __builtin_bit_cast(unsigned, x);
  const unsigned short r = (unsigned short)((b + 0x7fffu + ((b >> 16) & 1u)) >> 16);
  return (h >> 24) % 97 == 0 ? 0 : ((h >> 24) % 89 == 0 ? (unsigned short)0x8000 : r);
}

[[loom::kernel(per_item)]]
void gen_bf16(unsigned n, __bf16* out, unsigned seed, float scale) {
  const unsigned i = item();
  if (i < n) out[i] = __builtin_bit_cast(__bf16, draw(i, seed, scale));
}

[[loom::kernel(per_item)]]
void gen_f32(unsigned n, float* out, unsigned seed, float scale) {
  const unsigned i = item();
  if (i < n) out[i] = __builtin_bit_cast(float, (unsigned)draw(i, seed, scale) << 16);
}

// Rows first .. first + n - 1: ends = position + 1, complete = ends / ratio.
[[loom::kernel(per_item)]]
void gen_rows(unsigned n, int* ends, int* complete, int first, int ratio) {
  const unsigned i = item();
  if (i < n) {
    ends[i] = first + (int)i + 1;
    complete[i] = (first + (int)i + 1) / ratio;
  }
}

// scores[r][b] for b < complete[r] where complete[r] > top: work-item (r, b) over rows x stride.
[[loom::kernel(per_item)]]
void oracle_scores(unsigned n, const float* q, const __bf16* pooled, const int* complete, float* scores,
                   unsigned stride, int top) {
  const unsigned i = item();
  if (i >= n) return;
  const unsigned r = i / stride, b = i % stride;
  if (complete[r] <= top || (int)b >= complete[r]) return;
  float total = 0.0f;
  for (unsigned h = 0; h < kIdxHeads; h++) {
    float s[32];
    for (unsigned l = 0; l < 32; l++) {
      float x = 0.0f;
      for (unsigned c = 0; c < kIdx / 32; c++)
        x = x + q[(r * kIdxHeads + h) * kIdx + l + 32 * c] * bf(pooled[b * kIdx + l + 32 * c]);
      s[l] = x;
    }
    for (unsigned w = 16; w > 0; w /= 2)
      for (unsigned l = 0; l < w; l++) s[l] = s[l] + s[l + w];
    total = total + loom::scalar::maxnumf(s[0], 0.0f);
  }
  scores[i] = total / __builtin_bit_cast(float, 0x413504f3u);
}

LOOM_INLINE unsigned order_key(float v) {
  const unsigned u = __builtin_bit_cast(unsigned, v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// taken[r][b]: block b is among row r's `top` best (ties: lower id); rows at or below top take nothing.
[[loom::kernel(per_item)]]
void oracle_take(unsigned n, const float* scores, const int* complete, int* taken, unsigned stride, int top) {
  const unsigned i = item();
  if (i >= n) return;
  const unsigned r = i / stride, b = i % stride;
  const int nb = complete[r];
  if (nb <= top || (int)b >= nb) {
    taken[i] = 0;
    return;
  }
  const unsigned kb = order_key(scores[i]);
  int rank = 0;
  for (int c = 0; c < nb; c++) {
    const unsigned kc = order_key(scores[r * stride + c]);
    rank += (kc > kb || (kc == kb && c < (int)b)) ? 1 : 0;
  }
  taken[i] = rank < top ? 1 : 0;
}

// Work-item (r, b): a taken block's keys at its place in block order; work-item (r, x) past the blocks: the tail.
[[loom::kernel(per_item)]]
void oracle_place(unsigned n, const int* taken, const int* complete, const int* ends, int* keys, int* count,
                  int* sparse, unsigned stride, unsigned key_stride, int top, int ratio) {
  const unsigned i = item();
  if (i >= n) return;
  const unsigned r = i / stride, b = i % stride;
  const int nb = complete[r], end = ends[r];
  if (b == 0) {
    count[r] = nb > top ? top * ratio + end - nb * ratio : end;
    sparse[r] = nb > top ? 1 : 0;
  }
  if (nb <= top) return;
  int* row = keys + r * key_stride;
  if (taken[i]) {
    int before = 0;
    for (unsigned c = 0; c < b; c++) before += taken[r * stride + c];
    for (int j = 0; j < ratio; j++) row[before * ratio + j] = (int)b * ratio + j;
  }
  if ((int)b < end - nb * ratio) row[top * ratio + (int)b] = nb * ratio + (int)b;
}

// Score of query head h of row r (q scaled as the reference stages it) against key `key` of KV head g.
LOOM_INLINE float score(const float* q, const __bf16* k, unsigned r, unsigned h, unsigned g, int key, long long cap,
                        float scale) {
  float s = 0.0f;
  for (unsigned d = 0; d < kHd; d++)
    s = s + (q[(r * 2 * kGroup + h) * kHd + d] * scale) * bf(k[((long long)g * cap + key) * kHd + d]);
  return s;
}

// out[r][h][d]: work-item (r, h, d) replays every part, tile, softmax step and the merge of the reference.
[[loom::kernel(per_item)]]
void oracle_attend(unsigned n, const float* q, const __bf16* k, const __bf16* v, const int* ids, const int* count,
                   const int* sparse, float* out, long long id_stride, long long cap, float scale) {
  const unsigned i = item();
  if (i >= n) return;
  const unsigned r = i / (2 * kGroup * kHd), h = (i / kHd) % (2 * kGroup), d = i % kHd, g = h / kGroup;
  const int cnt = count[r];
  const bool listed = sparse[r] != 0;
  float po[kParts], pm[kParts], pl[kParts];
  for (unsigned p = 0; p < kParts; p++) {
    const int lo = (int)((long long)p * cnt / kParts), hi = (int)((long long)(p + 1) * cnt / kParts);
    float m = -sa::bits(0x7f800000u), l = 0.0f, o = 0.0f;
    for (int t0 = lo; t0 < hi; t0 += (int)kTile) {
      const int tn = hi - t0 < (int)kTile ? hi - t0 : (int)kTile;
      float sc[32], e[32];
      int key[32];
      float top = -sa::bits(0x7f800000u);
      for (int j = 0; j < tn; j++) {
        key[j] = listed ? ids[r * id_stride + t0 + j] : t0 + j;
        sc[j] = score(q, k, r, h, g, key[j], cap, scale);
        top = loom::scalar::maxnumf(top, sc[j]);
      }
      top = loom::scalar::maxnumf(top, m);
      for (int j = 0; j < 32; j++) e[j] = j < tn ? sa::exp_exact(sc[j] - top) : 0.0f;
      float s[32];
      for (unsigned j = 0; j < 32; j++) s[j] = e[j];
      for (unsigned w = 16; w > 0; w /= 2)
        for (unsigned j = 0; j < w; j++) s[j] = s[j] + s[j + w];
      const float sum = s[0];
      const float f = sa::exp_exact(m - top);
      l = l * f + sum;
      m = top;
      o = o * f;
      for (int j = 0; j < tn; j++) o = o + e[j] * bf(v[((long long)g * cap + key[j]) * kHd + d]);
    }
    po[p] = o;
    pm[p] = m;
    pl[p] = l;
  }
  float top = -sa::bits(0x7f800000u);
  for (unsigned p = 0; p < kParts; p++) top = loom::scalar::maxnumf(top, pm[p]);
  float total = 0.0f, acc = 0.0f;
  for (unsigned p = 0; p < kParts; p++) {
    const float w = pm[p] == -sa::bits(0x7f800000u) ? 0.0f : sa::exp_exact(pm[p] - top);
    total = total + pl[p] * w;
    acc = acc + po[p] * w;
  }
  out[i] = total > 0.0f ? acc / total : 0.0f;
}

}  // namespace sao

#endif  // SA_ORACLE_H_
