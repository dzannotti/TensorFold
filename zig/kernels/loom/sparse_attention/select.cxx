// SPDX-License-Identifier: MIT

// Indexer block selection: past `top` complete blocks, the top best (ties: lower id) in block order, then the tail.
#include <loomcxx/atomic.h>
#include <loomcxx/kernel.h>

constexpr unsigned kSelThreads = 256, kSelWaves = kSelThreads / 32;

static loom::kernel::configuration configure_select(unsigned rows) { return {{rows, 1, 1}, {kSelThreads, 1, 1}}; }

namespace sa {

LOOM_INLINE void sel_barrier() {
  using loom::atomic::ordering, loom::atomic::scope;
  loom::kernel::barrier<loom::memory_space::workgroup, scope::workgroup, ordering::acq_rel>();
}

// Larger floats, larger keys (the reference's fn_order).
LOOM_INLINE unsigned order_key(float v) {
  const unsigned u = __builtin_bit_cast(unsigned, v);
  return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// Exclusive prefix sum of one count a work-item over the workgroup; `waves` is kSelWaves ints of LDS.
LOOM_INLINE int block_scan(int v, int* waves, int* total) {
  const unsigned tid = loom::kernel::workitem::id.x, lane = tid & 31, wave = tid >> 5;
  int x = v;
  [[loom::unroll]] for (unsigned s = 0; s < 5; s++) {
    const int y = loom::kernel::subgroup::shuffle_up(x, 1u << s, 32).value;
    x = lane >= (1u << s) ? x + y : x;
  }
  sel_barrier();
  if (lane == 31) waves[wave] = x;
  sel_barrier();
  int before = 0, all = 0;
  [[loom::unroll]] for (unsigned w = 0; w < kSelWaves; w++) {
    const int c = waves[w];
    before += w < wave ? c : 0;
    all += c;
  }
  *total = all;
  return x - v + before;
}

}  // namespace sa

[[loom::kernel(configure_select)]]
void idx_select(const float* scores, const int* complete, const int* ends, int* keys, int* count, int* sparse,
                long long stride, long long key_stride, int top, int ratio) {
  LOOM_WORKGROUP int hist[256];
  LOOM_WORKGROUP int waves[kSelWaves];
  LOOM_WORKGROUP unsigned cut_s[2];
  const unsigned r = loom::kernel::workgroup::id.x, t = loom::kernel::workitem::id.x;
  const int nb = complete[r], end = ends[r];
  if (nb <= top) {
    if (t == 0) {
      count[r] = end;
      sparse[r] = 0;
    }
    return;
  }
  const float* sc = scores + (long long)r * stride;
  unsigned prefix = 0, mask = 0;
  int need = top;
  for (int shift = 24; shift >= 0; shift -= 8) {
    hist[t] = 0;
    sa::sel_barrier();
    for (int b = (int)t; b < nb; b += (int)kSelThreads) {
      const unsigned k = sa::order_key(sc[b]);
      if ((k & mask) == prefix)
        loom::view::atomic::reduce<loom::atomic::kind::addi, loom::atomic::ordering::relaxed,
                                   loom::atomic::scope::workgroup>(1, &hist[(k >> shift) & 255u]);
    }
    sa::sel_barrier();
    if (t == 0) {
      int above = 0, bin = 255;
      while (above + hist[bin] < need) above += hist[bin--];
      cut_s[0] = prefix | ((unsigned)bin << shift);
      cut_s[1] = (unsigned)(need - above);
    }
    sa::sel_barrier();
    prefix = cut_s[0];
    need = (int)cut_s[1];
    mask |= 255u << shift;
    sa::sel_barrier();
  }
  const int chunk = (nb + (int)kSelThreads - 1) / (int)kSelThreads;
  const int lo = nb < (int)t * chunk ? nb : (int)t * chunk, hi = nb < lo + chunk ? nb : lo + chunk;
  int above = 0, equal = 0;
  for (int b = lo; b < hi; b++) {
    const unsigned k = sa::order_key(sc[b]);
    above += k > prefix ? 1 : 0;
    equal += k == prefix ? 1 : 0;
  }
  int above_total = 0, equal_total = 0;
  const int pa = sa::block_scan(above, waves, &above_total);
  int pe = sa::block_scan(equal, waves, &equal_total);
  int out = pa + (pe < need ? pe : need);
  int* row = keys + (long long)r * key_stride;
  for (int b = lo; b < hi; b++) {
    const unsigned k = sa::order_key(sc[b]);
    bool take = k > prefix;
    if (k == prefix) {
      take = pe < need;
      pe++;
    }
    if (take) {
      for (int j = 0; j < ratio; j++) row[out * ratio + j] = b * ratio + j;
      out++;
    }
  }
  for (int k = nb * ratio + (int)t; k < end; k += (int)kSelThreads) row[top * ratio + k - nb * ratio] = k;
  if (t == 0) {
    count[r] = top * ratio + end - nb * ratio;
    sparse[r] = 1;
  }
}
