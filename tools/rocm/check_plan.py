"""experts.cu's routing plan on gfx1151 (HIP code object) against a host reference: exact integers, run twice.

Launches as cuda_kernels.zig Ops.plan does: pairs <= 1024 one 1024-thread block of plan_kernel; else plan_rank
(ceil(P / 1024) blocks), plan_offsets (one block), plan_scatter (256-thread blocks). Run in the rocm-dev container:

    python tools/rocm/check_plan.py --out <dir>
"""

import argparse
import ctypes
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hipmod import Module, genco, ptr, symbols  # noqa: E402

I = ctypes.c_int


def reference(picks, experts, tile):
    """members by expert then pair (stable), items (expert, first, count) of <= tile pairs, counts (items, used)."""
    order = np.argsort(picks, kind='stable')
    cnt = np.bincount(picks, minlength=experts)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    items = [(e, off[e] + tile * j, min(tile, cnt[e] - tile * j)) for e in range(experts) for j in
             range((cnt[e] + tile - 1) // tile)]
    return order.astype(np.int32), np.array(items, dtype=np.int32).reshape(-1, 3), (len(items), int((cnt > 0).sum()))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    co = genco('experts.cu', a.out)
    names = {s.split('tf_experts')[1]: s for s in symbols(co) if 'tf_experts' in s}
    sym = {k: next(v for n, v in names.items() if k in n) for k in ('plan_kernel', 'plan_rank', 'plan_offsets',
                                                                     'plan_scatter')}
    print('symbols', sym)
    m = Module(co)
    f = {k: m.function(v) for k, v in sym.items()}
    rng = np.random.default_rng(7)
    bad = cells = 0

    def run(picks, experts, tile):
        p = len(picks)
        dev = torch.from_numpy(picks).cuda()
        cap = min(p, experts) + p // tile
        members = torch.full((p,), -1, dtype=torch.int32, device='cuda')
        items = torch.full((cap, 3), -1, dtype=torch.int32, device='cuda')
        counts = torch.full((2,), -1, dtype=torch.int32, device='cuda')
        nblk = (p + 1023) // 1024
        rank = torch.empty((p,), dtype=torch.int32, device='cuda')
        hist = torch.empty((nblk * experts,), dtype=torch.int32, device='cuda')
        if p <= 1024:
            m.launch(f['plan_kernel'], (1,), (1024,), [ptr(dev), I(p), I(experts), I(tile), ptr(members), ptr(items),
                                                       ptr(counts)])
        else:
            m.launch(f['plan_rank'], (nblk,), (1024,), [ptr(dev), I(p), I(experts), ptr(rank), ptr(hist)])
            m.launch(f['plan_offsets'], (1,), (1024,), [I(nblk), I(experts), I(tile), ptr(hist), ptr(items), ptr(counts)])
            m.launch(f['plan_scatter'], ((p + 255) // 256,), (256,), [ptr(dev), I(p), I(experts), ptr(rank), ptr(hist),
                                                                      ptr(members)])
        torch.cuda.synchronize()
        n = int(counts[0])
        return members.cpu().numpy(), items[:n].cpu().numpy(), tuple(counts.cpu().tolist())

    for experts in (8, 129, 512, 513, 1024):
        for tile in (16, 64):
            for pairs in (1, 7, 100, 1000, 1024, 1025, 3000, 8 * 1024, 16 * 1024 + 5):
                for kind in ('uniform', 'skewed', 'one'):
                    if kind == 'uniform':
                        picks = rng.integers(0, experts, pairs)
                    elif kind == 'skewed':
                        picks = np.minimum(rng.geometric(0.05, pairs) - 1, experts - 1)
                    else:
                        picks = np.full(pairs, experts - 1)
                    picks = picks.astype(np.int32)
                    want = reference(picks, experts, tile)
                    got = run(picks, experts, tile)
                    again = run(picks, experts, tile)
                    ok = all(np.array_equal(g, w) for g, w in zip(got[:2], want[:2])) and got[2] == want[2] and \
                        all(np.array_equal(x, y) for x, y in zip(got[:2], again[:2]))
                    cells += 1
                    bad += not ok
                    if not ok:
                        print(f'DIFFER plan e{experts} t{tile} p{pairs} {kind}', flush=True)
    print(f"{'PASS' if bad == 0 else 'FAIL'} plan: {cells - bad} of {cells} exact and repeatable", flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
