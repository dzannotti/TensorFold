#!/usr/bin/env python3
"""A decode window's indexer and sparse attention (_pool, _scores, _select/_select_tiles, _chunks8, _merge) at long
contexts on gfx1151: device us a launch (torch.profiler, best of --reps medians), as the engine launches them in a
decode graph (keys = the graph's context bucket, Graphs._bucket).

    PYTHONPATH=src python tools/rocm/qsa_bench.py [--ctx 2048,8192,...] [--rows 1,4,16] [--save out.json]
"""

from __future__ import annotations

import argparse
import json

import torch
import triton

from tensorfold.families.qwen4_exp.cuda import attention as A, kv8

H, HK, D = 24, 2, 256
HI, DI, RATIO, BUDGET = 4, 128, 4, 2048
CAPACITY = 262144
NAMES = ("_pool", "_scores", "_select", "_select_tiles", "_chunks8", "_merge")


def bucket(end: int) -> int:
    return min(CAPACITY, max(8192, triton.next_power_of_2(end)))


def setup(ctx: int, rows: int):
    g = torch.Generator(device="cuda").manual_seed(ctx + rows)
    n = ctx + 64
    kc = torch.randint(0, 0x7F, (n, HK, D + kv8.PAD), dtype=torch.uint8, device="cuda", generator=g)
    kc[:, :, D:] = 0
    kc[:, :, D:D + 4] = torch.tensor([0, 0, 128, 59], dtype=torch.uint8, device="cuda")   # s_k ~ 2^-8 as fp32 bytes
    kc[:, :, D + 4:D + 8] = torch.tensor([0, 0, 128, 59], dtype=torch.uint8, device="cuda")
    vc = torch.randint(0, 0x7F, (n, HK, D), dtype=torch.uint8, device="cuda", generator=g)
    ikc = torch.randn((n, DI), device="cuda", generator=g).to(torch.bfloat16)
    pooled = torch.randn((CAPACITY // RATIO, DI), device="cuda", generator=g).to(torch.bfloat16)
    q = torch.randn((rows, H, D), device="cuda", generator=g).to(torch.bfloat16)
    iq = torch.randn((rows, HI, DI), device="cuda", generator=g).to(torch.bfloat16)
    w = torch.ones(DI, device="cuda")
    inv = 1.0 / (10000 ** (torch.arange(0, 32, device="cuda").float() / 32))
    sc = A.AttnScratch(rows, H, D, CAPACITY, "cuda", budget=BUDGET, ratio=RATIO)
    pos0 = torch.full((1,), ctx - rows, dtype=torch.int32, device="cuda")
    return kc, vc, ikc, pooled, q, iq, w, inv, sc, pos0


def step(t, rows: int, keys: int):
    kc, vc, ikc, pooled, q, iq, w, inv, sc, pos0 = t
    A.qsa_select(iq, ikc, pooled, pos0, w, inv, 1e-6, sc, rows, context=keys)
    kv8.attention(q, kc, vc, pos0, sc, rows, D ** -0.5, sc.out, context=keys)


def main() -> int:
    from torch.profiler import ProfilerActivity, profile

    ap = argparse.ArgumentParser()
    ap.add_argument("--ctx", default="2048,8192,16384,32768,65536,131072")
    ap.add_argument("--rows", default="1,4,16")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--save")
    args = ap.parse_args()
    res = {}
    for ctx in map(int, args.ctx.split(",")):
        for rows in map(int, args.rows.split(",")):
            t = setup(ctx, rows)
            keys = bucket(ctx)
            for _ in range(3):
                step(t, rows, keys)
            torch.cuda.synchronize()
            best: dict[str, float] = {}
            for _ in range(args.reps):
                with profile(activities=[ProfilerActivity.CUDA]) as prof:
                    for _ in range(10):
                        step(t, rows, keys)
                    torch.cuda.synchronize()
                times: dict[str, list[float]] = {}
                for e in prof.events():
                    if e.name in NAMES and e.device_time > 0:
                        times.setdefault(e.name, []).append(e.device_time)
                for k, v in times.items():
                    best[k] = min(best.get(k, float("inf")), sorted(v)[len(v) // 2])
            total = sum(best.values())
            res[f"{ctx}x{rows}"] = best
            print(f"ctx {ctx:6d} rows {rows:2d} bucket {keys:6d}  " +
                  "  ".join(f"{k} {v:7.1f}" for k, v in sorted(best.items())) + f"  | sum {total:7.1f} us", flush=True)
            del t
            torch.cuda.empty_cache()
    if args.save:
        json.dump(res, open(args.save, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
