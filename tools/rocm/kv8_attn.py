#!/usr/bin/env python3
"""kv8.attention (``_chunks8`` + ``_merge``) on gfx1151: accuracy against an fp64 reference, row/batch invariance,
determinism, and launch time.

  check: rows M in ROWS at several positions (dense rows, and sparse rows over random block lists up to 262144 keys):
         each output against fp64 attention over the dequantized cache (worst |err| over all, and per head), each
         row's bits against the same row launched alone and inside other windows (row invariance), two runs equal.
  bench: us a launch of _chunks8 (best of N, alternated with --other's source when given) at decode and prefill shapes.

    PYTHONPATH=src python tools/rocm/kv8_attn.py check
    PYTHONPATH=src python tools/rocm/kv8_attn.py bench [--save before.json]
"""

from __future__ import annotations

import argparse
import json
import sys

import torch
import triton
import triton.language as tl

from tensorfold.families.qwen4_exp.cuda import attention as A, kv8

H, HK, D = 24, 2, 256
RATIO, BUDGET = 4, 2048
ROWS = (1, 3, 16, 17, 129, 161, 2049)


def cache(capacity: int, seed: int):
    g = torch.Generator(device="cuda").manual_seed(seed)
    kc = torch.empty((capacity, HK, D + kv8.PAD), dtype=torch.uint8, device="cuda")
    vc = torch.empty((capacity, HK, D), dtype=torch.uint8, device="cuda")
    step = 1 << 15
    for a in range(0, capacity, step):          # in slices: the bf16 rows of a long cache would not fit the budget
        b = min(capacity, a + step)
        k = (torch.randn((b - a, HK, D), generator=g, device="cuda") * 2).to(torch.bfloat16)
        v = torch.randn((b - a, HK, D), generator=g, device="cuda").to(torch.bfloat16)
        kc[a:b], vc[a:b] = kv8.quantize(k, v)
    return kc, vc


def scratch(rows: int, capacity: int, p0: int, seed: int):
    """An AttnScratch with each row's key list as _select would leave it: dense rows their length, sparse rows TOP
    random blocks of their complete ones in block order then the tail."""

    sc = A.AttnScratch(rows, H, D, capacity, "cuda")
    if not sc.qsa:
        return sc
    g = torch.Generator().manual_seed(seed)
    top = BUDGET // RATIO
    ids = torch.zeros((rows, sc.idw), dtype=torch.int32)
    nk = torch.zeros(rows, dtype=torch.int32)
    sp = torch.zeros(rows, dtype=torch.int32)
    for r in range(rows):
        end = p0 + r + 1
        complete = end // RATIO
        if complete <= top:
            nk[r] = end
            continue
        blocks = torch.randperm(complete, generator=g)[:top].sort().values
        keys = (blocks[:, None] * RATIO + torch.arange(RATIO)).flatten()
        tail = torch.arange(RATIO * complete, end)
        ids[r, :keys.numel() + tail.numel()] = torch.cat([keys, tail]).int()
        nk[r] = keys.numel() + tail.numel()
        sp[r] = 1
    sc.ids.copy_(ids)
    sc.nk.copy_(nk)
    sc.sparse.copy_(sp)
    return sc


def run(q, kc, vc, p0: int, sc, rows: int) -> torch.Tensor:
    pos0 = torch.full((1,), p0, dtype=torch.int32, device="cuda")
    out = torch.empty((rows, H, D), dtype=torch.bfloat16, device="cuda")
    kv8.attention(q, kc, vc, pos0, sc, rows, D ** -0.5, out, context=p0 + rows)
    return out


def reference(q, kc, vc, p0: int, sc, rows: int) -> torch.Tensor:
    """fp64 attention of each row over its keys (the dequantized cache, exact)."""

    out = torch.empty((rows, H, D), dtype=torch.float64)
    nk, sp, ids = sc.nk.cpu(), sc.sparse.cpu(), sc.ids.cpu()
    for r in range(rows):
        end = p0 + r + 1
        keys = ids[r, :nk[r]].long().cuda() if sc.qsa and sp[r] else torch.arange(end, device="cuda")
        k, v = kv8.dequantize(kc[keys], vc[keys])                                  # [n, HK, D] fp32
        qq = q[r].double().view(HK, H // HK, D)
        s = torch.einsum("hgd,nhd->hgn", qq, k.double()) * D ** -0.5
        out[r] = torch.einsum("hgn,nhd->hgd", torch.softmax(s, -1), v.double()).reshape(H, D).cpu()
    return out


@triton.jit
def _decode(W, OUT):
    i = tl.arange(0, 64)
    tl.store(OUT + tl.arange(0, 256), tl.reshape(kv8.bf16_words(tl.load(W + i)[None, :]), (256,)))


def codes_check() -> bool:
    """kv8.bf16_words on all 256 codes against torch's e4m3 -> fp32 / 256 (the NaN codes 0x7F / 0xFF excepted)."""
    if not hasattr(kv8, "bf16_words"):            # an older kv8.py (A/B against the previous kernel)
        return True
    codes = torch.arange(256, dtype=torch.uint8)
    out = torch.empty(256, dtype=torch.bfloat16, device="cuda")
    _decode[(1,)](codes.cuda().view(torch.int32).view(torch.uint32), out)
    want = codes.view(torch.float8_e4m3fn).float() / 256
    fin = torch.isfinite(want)
    ok = torch.equal(out.cpu().float()[fin], want[fin])
    print(f"bf16_words: 254 finite codes {'exact' if ok else 'DIFFER'}")
    return ok


def check(args) -> int:
    bad = 0 if codes_check() else 1
    worst = 0.0
    for capacity, p0s in ((4096, (0, 37, 1900)), (262144, (5000, 70000, 262144 - 2049 - 8))):
        kc, vc = cache(capacity, 1)
        for p0 in p0s:
            for rows in ROWS:
                if p0 + rows > capacity:
                    continue
                g = torch.Generator(device="cuda").manual_seed(p0 * 7 + rows)
                q = torch.randn((rows, H, D), generator=g, device="cuda").to(torch.bfloat16)
                sc = scratch(rows, capacity, p0, p0 + rows)
                out = run(q, kc, vc, p0, sc, rows)
                again = run(q, kc, vc, p0, sc, rows)
                det = torch.equal(out.view(torch.int16), again.view(torch.int16))
                # row invariance: the last row alone (its own launch at its own position) and the first row alone
                inv = True
                for r in {0, rows - 1}:
                    sc1 = scratch(1, capacity, p0 + r, 0)
                    sc1.ids.copy_(sc.ids[r:r + 1]), sc1.nk.copy_(sc.nk[r:r + 1]), sc1.sparse.copy_(sc.sparse[r:r + 1])
                    one = run(q[r:r + 1].contiguous(), kc, vc, p0 + r, sc1, 1)
                    inv &= torch.equal(one.view(torch.int16), out[r:r + 1].view(torch.int16))
                rs = list(range(rows)) if rows <= 17 else sorted({0, 1, rows // 2, rows - 2, rows - 1})
                sub = A.AttnScratch(len(rs), H, D, capacity, "cuda")
                ref = reference(q, kc, vc, p0, sc, rows)[rs]
                err = (out[rs].double().cpu() - ref).abs()
                rel = err.amax() / ref.abs().amax()
                per_head = err.amax(dim=(0, 2))
                worst = max(worst, err.amax().item())
                ok = det and inv
                bad += not ok
                print(f"{'ok  ' if ok else 'FAIL'} cap {capacity:6d} p0 {p0:6d} M {rows:4d}: det {det} inv {inv} "
                      f"max|err| {err.amax().item():.3e} (rel {rel.item():.2e}) per-head worst "
                      f"{per_head.max().item():.3e} best {per_head.min().item():.3e} mean|err| {err.mean().item():.4e}", flush=True)
                del sub
        del kc, vc
        torch.cuda.empty_cache()
    print(f"worst |err| {worst:.3e}; {bad} cases failed")
    return 1 if bad else 0


CASES = (("decode M1 ctx50", 1, 4096, 50), ("decode M3 ctx50", 3, 262144, 50), ("decode M16 ctx400", 16, 262144, 400),
         ("decode M3 ctx70k", 3, 262144, 70000), ("prefill M256 dense p1792", 256, 262144, 1792),
         ("prefill M256 sparse p6144", 256, 262144, 6144))


def bench(args) -> int:
    """Device time of each kernel a launch (torch.profiler), the best of --reps repetitions' medians."""

    from torch.profiler import ProfilerActivity, profile

    res = {}
    kv = {}
    for name, rows, capacity, p0 in CASES:
        if capacity not in kv:
            kv.clear()
            torch.cuda.empty_cache()
            kv[capacity] = cache(capacity, 2)
        kc, vc = kv[capacity]
        q = torch.randn((rows, H, D), device="cuda").to(torch.bfloat16)
        sc = scratch(rows, capacity, p0, 3)
        for _ in range(3):
            run(q, kc, vc, p0, sc, rows)
        torch.cuda.synchronize()
        best = {}
        for _ in range(args.reps):
            with profile(activities=[ProfilerActivity.CUDA]) as prof:
                for _ in range(max(3, min(20, 2000 // rows))):
                    run(q, kc, vc, p0, sc, rows)
                torch.cuda.synchronize()
            times = {}
            for e in prof.events():
                if e.name in ("_chunks8", "_merge") and e.device_time > 0:
                    times.setdefault(e.name, []).append(e.device_time)
            for k, v in times.items():
                med = sorted(v)[len(v) // 2]
                best[k] = min(best.get(k, float("inf")), med)
        res[name] = best
        print(f"{name:28s} " + "  ".join(f"{k} {v:9.1f} us" for k, v in sorted(best.items())), flush=True)
    if args.save:
        json.dump(res, open(args.save, "w"), indent=1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("check", "bench"))
    ap.add_argument("--reps", type=int, default=7)
    ap.add_argument("--save")
    args = ap.parse_args()
    return check(args) if args.what == "check" else bench(args)


if __name__ == "__main__":
    sys.exit(main())
