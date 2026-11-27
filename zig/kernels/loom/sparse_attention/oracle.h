// SPDX-License-Identifier: MIT

// Reference arithmetic for checks.cxx: serial replicas of the HIP indexer (README: Sum order), fp64 attention.
#ifndef SA_ORACLE_H_
#define SA_ORACLE_H_

#include <loomcxx/kernel.h>
#include <loomcxx/scalar.h>

#include "exp_exact.h"

namespace sao {

constexpr unsigned kHd = 256, kGroup = 12, kIdx = 128, kIdxHeads = 4;

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

// e^x in fp64 for x <= 0: x = k ln2 + r with |r| <= ln2 / 2, a degree-13 Taylor polynomial in r, then 2^k (0 below -700).
LOOM_INLINE double exp64(double x) {
  const double kd = (double)(long long)(x * 1.4426950408889634 + (x < 0.0 ? -0.5 : 0.5));
  const double r = x - kd * 0.6931471805599453;
  double p = 1.0 / 6227020800.0;
  const double c[13] = {1.0 / 479001600.0, 1.0 / 39916800.0, 1.0 / 3628800.0, 1.0 / 362880.0, 1.0 / 40320.0,
                        1.0 / 5040.0, 1.0 / 720.0, 1.0 / 120.0, 1.0 / 24.0, 1.0 / 6.0, 0.5, 1.0, 1.0};
  for (unsigned i = 0; i < 13; i++) p = p * r + c[i];
  const long long e = (long long)kd + 1023;
  const double two_k = __builtin_bit_cast(double, (unsigned long long)(e > 0 ? e : 0) << 52);
  return x < -700.0 ? 0.0 : p * two_k;
}

// fp64 score of query head h of row r (q times scale in fp32, as the kernels stage it) against key `key` of KV head g.
LOOM_INLINE double score64(const float* q, const __bf16* k, unsigned r, unsigned h, unsigned g, int key, long long cap,
                           float scale) {
  double s = 0.0;
  for (unsigned d = 0; d < kHd; d++)
    s = s + (double)(q[(r * 2 * kGroup + h) * kHd + d] * scale) * (double)bf(k[((long long)g * cap + key) * kHd + d]);
  return s;
}

// out[r][h][*]: work-item (r, h), the softmax attention in fp64 (three passes over the keys), rounded to fp32.
[[loom::kernel(per_item)]]
void oracle_attend(unsigned n, const float* q, const __bf16* k, const __bf16* v, const int* ids, const int* count,
                   const int* sparse, float* out, long long id_stride, long long cap, float scale) {
  const unsigned i = item();
  if (i >= n) return;
  const unsigned r = i / (2 * kGroup), h = i % (2 * kGroup), g = h / kGroup;
  const int cnt = count[r];
  const bool listed = sparse[r] != 0;
  double top = -1e300, total = 0.0;
  for (int j = 0; j < cnt; j++) {
    const double s = score64(q, k, r, h, g, listed ? ids[r * id_stride + j] : j, cap, scale);
    top = s > top ? s : top;
  }
  for (int j = 0; j < cnt; j++) total = total + exp64(score64(q, k, r, h, g, listed ? ids[r * id_stride + j] : j, cap, scale) - top);
  double acc[kHd];
  for (unsigned d = 0; d < kHd; d++) acc[d] = 0.0;
  for (int j = 0; j < cnt; j++) {
    const int key = listed ? ids[r * id_stride + j] : j;
    const double p = exp64(score64(q, k, r, h, g, key, cap, scale) - top) / total;
    for (unsigned d = 0; d < kHd; d++) acc[d] = acc[d] + p * (double)bf(v[((long long)g * cap + key) * kHd + d]);
  }
  for (unsigned d = 0; d < kHd; d++) out[(r * 2 * kGroup + h) * kHd + d] = cnt > 0 ? (float)acc[d] : 0.0f;
}

}  // namespace sao

#endif  // SA_ORACLE_H_
