"""fn_qsa_scores.hip (the prompt indexer's scores) against Triton's attention._scores and prompt_mm._scores_rows
JIT-run on ROCm torch: score bytes over random, wide-range and tie-heavy keys, rows past several positions; then timed.
Run in the rocm-dev container: python tools/rocm/mtp_qsa_check.py [--bench]
"""
import os, sys
import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))
import mtp_hip
from tensorfold.families.qwen4_exp.cuda import attention as A
from tensorfold.families.qwen4_exp.cuda import prompt_mm as P

HI, DI, RATIO, TOP = 4, 128, 4, 512
TILES = {"fn_qsa_scores": (2, 4), "fn_qsa_scores_r2b8": (2, 8)}  # r4b4 / r4b8 need 96 KiB of LDS


def smem(tr, tb):
    return (16 * tb + 16 * tr * HI) * DI * 2


def data(kind, rows, nb, g):
    if kind == "normal":
        q, k = torch.randn(rows, HI, DI, generator=g), torch.randn(nb, DI, generator=g)
    elif kind == "wide":
        e = lambda *s: torch.ldexp(torch.rand(*s, generator=g) + 1, torch.randint(-30, 30, s, generator=g))
        q, k = e(rows, HI, DI) * torch.sign(torch.randn(rows, HI, DI, generator=g)), e(nb, DI)
    else:  # ties: few distinct small values
        q = torch.randint(-2, 3, (rows, HI, DI), generator=g).float()
        k = torch.randint(-2, 3, (nb, DI), generator=g).float() * 0.5
    return q.bfloat16().cuda(), k.bfloat16().cuda()


def main():
    mod = mtp_hip.build("fn_qsa_scores")
    g = torch.Generator().manual_seed(7)
    ok = True
    for kind in ("normal", "wide", "ties"):
        for rows, start in ((37, 2048 - 20), (256, 9000), (100, 70001), (16, 400)):
            nb = (start + rows) // RATIO + 3
            iq, pooled = data(kind, rows, nb, g)
            pos0 = torch.tensor([start], dtype=torch.int32, device="cuda")
            blocks = (start + rows + RATIO - 1) // RATIO
            ref = torch.full((rows, nb), float("nan"), device="cuda")
            A._scores[(rows, triton.cdiv(blocks, 64))](iq, pooled, pos0, ref, nb, HI=HI, DI=DI, RATIO=RATIO, TOP=TOP,
                                                       BB=64, num_warps=4)
            outs = {}
            for rt in (4, 16):
                o = torch.full_like(ref, float("nan"))
                P._scores_rows[(triton.cdiv(rows, rt), triton.cdiv(blocks, 64))](iq, pooled, pos0, o, nb, rows, HI=HI,
                    DI=DI, RATIO=RATIO, TOP=TOP, BB=64, RT=rt)
                outs[f"_scores_rows RT {rt}"] = o
            for name, (tr, tb) in TILES.items():
                o = torch.full_like(ref, float("nan"))
                mod.launch(name, (triton.cdiv(rows, 16 * tr), triton.cdiv(blocks, 16 * tb), 1), 256,
                           [iq, pooled, pos0, o, ("i", nb), ("i", rows), ("i", RATIO), ("i", TOP)], smem(tr, tb))
                outs[name] = o
            rb = ref.view(torch.int32)
            stored = int((~torch.isnan(ref)).sum())
            for name, o in outs.items():
                diff = int((o.view(torch.int32) != rb).sum())
                ok &= diff == 0
                if diff or name.startswith("fn_"):
                    print(f"{'EQUAL ' if diff == 0 else 'DIFFER'} {kind:6s} rows {rows:3d} from {start:5d}: {name} "
                          f"vs _scores, {stored} scores" + (f", {diff} words differ" if diff else ""))
    if "--bench" in sys.argv:
        rows, start = 256, 131072 - 256
        nb = (start + rows) // RATIO
        iq, pooled = data("normal", rows, nb, g)
        pos0 = torch.tensor([start], dtype=torch.int32, device="cuda")
        sc = torch.empty(rows, nb, device="cuda")
        blocks = triton.cdiv(start + rows, RATIO)
        t = mtp_hip.best_us(lambda: A._scores[(rows, triton.cdiv(blocks, 64))](iq, pooled, pos0, sc, nb, HI=HI, DI=DI,
                            RATIO=RATIO, TOP=TOP, BB=64, num_warps=4), reps=5)
        print(f"256 rows at 128k (noisy, shared GPU): _scores {t:.0f} us", end="")
        for name, (tr, tb) in TILES.items():
            t = mtp_hip.best_us(lambda: mod.launch(name, (triton.cdiv(rows, 16 * tr), triton.cdiv(blocks, 16 * tb), 1),
                                256, [iq, pooled, pos0, sc, ("i", nb), ("i", rows), ("i", RATIO), ("i", TOP)],
                                smem(tr, tb)), reps=5)
            print(f", {name} {t:.0f} us", end="")
        print()
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
