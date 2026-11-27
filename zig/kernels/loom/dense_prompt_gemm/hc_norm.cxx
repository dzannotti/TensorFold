// SPDX-License-Identifier: MIT

// A hyper-connection's writeback and norm (README.md): h_s += bf16(branch inject[s]) when `add`, in place; then
// x = bf16((h rinv) scale), rinv = rsqrt(sum of squares / dims + eps) in the HIP norm's sum order. bf16 h, x [m][4 dims].
#include <loomcxx/kernel.h>

#include "hc_exact.h"
#include "hc_tiles.h"

#ifndef K5_HC_DIMS
#define K5_HC_DIMS 2560
#endif

namespace k5n {

// A stream's 64 work-items; the HIP norm's 1024 threads (each the squares of dims t, t + 1024, ... in order) as
// virtual threads, its 32 waves' xor trees and their sum in wave order. kDims: a stream's width, compiled in.
constexpr unsigned kPer = 64, kThreads = kPer * k5::kStreams, kVirtual = 1024, kWaves = kVirtual / 32;
constexpr unsigned kDims = K5_HC_DIMS, kChunks = kDims / 512;
static_assert(kDims % 512 == 0 && kChunks <= 8, "a stream is 512 to 4096 wide, a multiple of 512");

static loom::kernel::configuration configure_norm(unsigned m) { return {{m, 1, 1}, {kThreads, 1, 1}}; }

LOOM_INLINE void lds_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

// Two packed bf16 streams h + bf16(b inj): bf16 of the sum.
LOOM_INLINE unsigned add_pair(unsigned h, unsigned b, float inj) {
  const float lo = k5::bits(h << 16) + k5::bf16_round(k5::bits(b << 16) * inj);
  const float hi = k5::bits(h & 0xffff0000u) + k5::bf16_round(k5::bits(b & 0xffff0000u) * inj);
  return k5::bf16_bits(lo) | (k5::bf16_bits(hi) << 16);
}

}  // namespace k5n

// Block r: row r's streams; dims must be kDims. part, waves: LDS for the virtual threads' and waves' sums.
LOOM_INLINE void hc_norm_rows(__bf16* h, const __bf16* branch, const __bf16* inject, const float* scale, __bf16* x,
                              int m, int dims, float eps, int add, float* part, float* waves) {
  const unsigned tid = loom::kernel::workitem::id.x, r = loom::kernel::workgroup::id.x;
  const unsigned ud = k5n::kDims, s = tid / k5n::kPer, q = tid % k5n::kPer;
  if (r >= (unsigned)m || dims != (int)k5n::kDims) return;
  k5::u32x4* const hs = (k5::u32x4*)h + (r * k5::kStreams + s) * (ud / 8);
  const k5::u32x4* const bs = (const k5::u32x4*)branch + r * (ud / 8);
  const float inj = add != 0 ? k5::bits((unsigned)((const unsigned short*)inject)[r * k5::kStreams + s] << 16) : 0.0f;
  unsigned kept[4 * k5n::kChunks];
  float pa[8], pb[8];
  [[loom::unroll]] for (unsigned e = 0; e < 8; e++) pa[e] = pb[e] = 0.0f;
  [[loom::unroll]] for (unsigned i = 0; i < k5n::kChunks; i++) {
    {
      const unsigned c = q + k5n::kPer * i;
      const k5::u32x4 h0 = hs[c];
      k5::u32x4 v = h0;
      if (add != 0) {
        const k5::u32x4 b = bs[c];
        const k5::u32x4 sum = {k5n::add_pair(h0[0], b[0], inj), k5n::add_pair(h0[1], b[1], inj),
                               k5n::add_pair(h0[2], b[2], inj), k5n::add_pair(h0[3], b[3], inj)};
        hs[c] = sum;
        v = sum;
      }
      [[loom::unroll]] for (unsigned j = 0; j < 4; j++) kept[4 * i + j] = v[j];
      [[loom::unroll]] for (unsigned j = 0; j < 4; j++) {
        const float lo = k5::bits(v[j] << 16), hi = k5::bits(v[j] & 0xffff0000u);
        if (i % 2 == 0) {
          pa[2 * j] += lo * lo;
          pa[2 * j + 1] += hi * hi;
        } else {
          pb[2 * j] += lo * lo;
          pb[2 * j + 1] += hi * hi;
        }
      }
    }
  }
  float* const ps = part + s * k5n::kVirtual;
  [[loom::unroll]] for (unsigned e = 0; e < 8; e++) {
    ps[8 * q + e] = pa[e];
    ps[8 * (q + k5n::kPer) + e] = pb[e];
  }
  k5n::lds_barrier();
  if (tid < k5::kStreams * k5n::kWaves) {
    const float* const p = part + (tid / k5n::kWaves) * k5n::kVirtual + (tid % k5n::kWaves) * 32;
    float a[32];
    [[loom::unroll]] for (unsigned i = 0; i < 32; i++) a[i] = p[i];
    [[loom::unroll]] for (unsigned level = 0; level < 5; level++) {
      const unsigned mask = 16u >> level;
      [[loom::unroll]] for (unsigned i = 0; i < 16; i++)
        if (i < mask) a[i] = a[i] + a[i + mask];
    }
    waves[tid] = a[0];
  }
  k5n::lds_barrier();
  float total = 0.0f;
  [[loom::unroll]] for (unsigned w = 0; w < k5n::kWaves; w++) total += waves[s * k5n::kWaves + w];
  const float rinv = k5::rsqrt_exact(total / (float)k5n::kDims + eps);
  k5::u32x4* const xs = (k5::u32x4*)x + (r * k5::kStreams + s) * (ud / 8);
  const float* const ss = scale + s * ud;
  [[loom::unroll]] for (unsigned i = 0; i < k5n::kChunks; i++) {
    {
      const unsigned c = q + k5n::kPer * i;
      const k5::f32x4 sa = *(const k5::f32x4*)(ss + 8 * c), sb = *(const k5::f32x4*)(ss + 8 * c + 4);
      const k5::u32x4 out = {k5::norm_pair(kept[4 * i], rinv, sa[0], sa[1]), k5::norm_pair(kept[4 * i + 1], rinv, sa[2], sa[3]),
                             k5::norm_pair(kept[4 * i + 2], rinv, sb[0], sb[1]), k5::norm_pair(kept[4 * i + 3], rinv, sb[2], sb[3])};
      xs[c] = out;
    }
  }
}

[[loom::kernel(k5n::configure_norm)]]
void hc_norm(__bf16* h, const __bf16* branch, const __bf16* inject, const float* scale, __bf16* x, int m, int dims,
             float eps, int add) {
  LOOM_WORKGROUP float part[k5::kStreams * k5n::kVirtual];
  LOOM_WORKGROUP float waves[k5::kStreams * k5n::kWaves];
  hc_norm_rows(h, branch, inject, scale, x, m, dims, eps, add, part, waves);
}
