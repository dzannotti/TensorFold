"""fn_int4.hip speeds on gfx1151 (best of N; the GPU is shared with production, so the minimum is the signal):
python tools/rocm/int4_bench.py [--nt 1,2,4] [--cases decode,head,prompt,stream]"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiprun
import int4_check as c

D, NI, E, TOP = 2560, 640, 512, 5
SLOTS = TOP + 1


def rand_words(n):
    return torch.randint(-2**31, 2**31 - 1, (n,), dtype=torch.int32, device="cuda")


def experts():
    up = rand_words(E * 2 * NI * D // 8)
    up_s = torch.full((E * 2 * NI * (D // 128),), 0.01, dtype=torch.float16, device="cuda")
    dn = rand_words(E * D * NI // 8)
    dn_s = torch.full((E * D * (NI // 128),), 0.01, dtype=torch.float16, device="cuda")
    return up, up_s, dn, dn_s


def run_moe(ex, x, act, y, plan, mi, R, nt_up, nt_dn, mt, part):
    up, up_s, dn, dn_s = ex
    if part in (0, 2):
        c.launch(128, nt_up, mt, 2, 2, x, D, SLOTS, up, up_s, D, NI, plan, 0, act, mi, E)
    if part in (1, 2):
        c.launch(128, nt_dn, mt, 1, 3, act, NI, 0, dn, dn_s, NI, D, plan, 0, y, mi, E)


def decode(ex, nts, rounds):
    print("decode experts (gate/up SwiGLU + down bf16), Eu experts x m pairs each, items of 16 (MT 1):")
    for eu in (5, 20, 40, 74):
        for m in (1, 4, 16):
            R = eu * m // TOP if eu * m % TOP == 0 else None
            if R is None:
                continue
            picks = np.full((R, SLOTS), E, np.int32)
            for t in range(R):
                for j in range(TOP):
                    picks[t, j] = (TOP * t + j) % eu * (E // eu)
            plan, n_items = c.make_plan(picks, E + 1, 16)
            mi = c.max_items(R * SLOTS, E + 1, 16)
            x = torch.randn(R, D, device="cuda").bfloat16()
            act = torch.zeros(R * SLOTS, NI, dtype=torch.bfloat16, device="cuda")
            y = torch.zeros(R * SLOTS, D, dtype=torch.bfloat16, device="cuda")
            wb = eu * 3 * NI * D / 2 + eu * 3 * NI * D / 128 * 2
            res = []
            for nu in nts:
                for nd in nts:
                    if NI % (8 * nu * c.waves(1)) or D % (16 * nd * c.waves(1)):
                        continue
                    tu = hiprun.best_us(lambda: run_moe(ex, x, act, y, plan, mi, R, nu, nd, 1, 0), 20, rounds)
                    td = hiprun.best_us(lambda: run_moe(ex, x, act, y, plan, mi, R, nu, nd, 1, 1), 20, rounds)
                    res.append((tu + td, nu, nd, tu, td))
            best = min(res)
            line = " ".join(f"[{nu},{nd}] {tu:.0f}+{td:.0f}" for _, nu, nd, tu, td in sorted(res, key=lambda r: (r[1], r[2])))
            print(f"  Eu {eu:3d} m {m:2d} ({R * TOP} pairs): best NT up {best[1]} down {best[2]}: gate/up {best[3]:.1f} us "
                  f"+ down {best[4]:.1f} us = {wb / best[0] / 1e3:.0f} GB/s   (NT [up,down] us: {line})")


def head(nts, rounds):
    V = 248320
    w = rand_words(V * D // 8)
    s = torch.full((V * D // 128,), 0.01, dtype=torch.float16, device="cuda")
    x = torch.randn(16, D, device="cuda").bfloat16()
    out = torch.zeros(16, V, dtype=torch.bfloat16, device="cuda")
    wb = V * D / 2 + V * D / 128 * 2
    for rows in (1, 8, 16):
        r = []
        for nt in nts:
            t = hiprun.best_us(lambda: c.launch(128, nt, 1, 1, 3, x, D, 0, w, s, D, V, None, rows, out, (rows + 15) // 16), 10, rounds)
            r.append(f"NT {nt} {t:.0f} us {wb / t / 1e3:.0f} GB/s")
        print(f"  head [{V}, {D}] {rows} rows: " + ", ".join(r))
    del w


def prompt(ex, nts, rounds):
    rng = np.random.default_rng(3)
    for R in (512, 2048):
        picks = np.full((R, SLOTS), E, np.int32)
        for t in range(R):
            picks[t, :TOP] = rng.choice(E, TOP, replace=False)
        x = torch.randn(R, D, device="cuda").bfloat16()
        act = torch.zeros(R * SLOTS, NI, dtype=torch.bfloat16, device="cuda")
        y = torch.zeros(R * SLOTS, D, dtype=torch.bfloat16, device="cuda")
        flops = 2 * R * TOP * 3 * NI * D
        eu = len(np.unique(picks[:, :TOP]))
        wb = eu * 3 * NI * D / 2
        for tile, mt in ((16, 1), (64, 1), (64, 2)):
            plan, _ = c.make_plan(picks, E + 1, tile)
            mi = c.max_items(R * SLOTS, E + 1, tile)
            for nu, nd in ((nt, nt) for nt in nts):
                tu = hiprun.best_us(lambda: run_moe(ex, x, act, y, plan, mi, R, nu, nd, mt, 0), 3, rounds)
                td = hiprun.best_us(lambda: run_moe(ex, x, act, y, plan, mi, R, nu, nd, mt, 1), 3, rounds)
                t = tu + td
                print(f"  prompt {R} rows ({eu} experts): tile {tile} MT {mt} NT {nu}: gate/up {tu:.0f} us + down {td:.0f} "
                      f"us = {flops / t / 1e6:.1f} TFLOPS, {wb / t / 1e3:.0f} GB/s of distinct weights")


def stream(rounds):
    m = hiprun.Module(hiprun.build(f"{os.path.dirname(os.path.abspath(__file__))}/int4_probe.hip", hiprun.cache("int4_probe.co")))
    for mb in (16, 64, 1024):
        n = mb * 2**20 // 16
        buf = torch.empty(n * 4, dtype=torch.int32, device="cuda")
        out = torch.empty(1280 * 256, dtype=torch.int32, device="cuda")
        t = hiprun.best_us(lambda: m.launch("probe_stream", 1280, 256, [buf, ("q", n), out]), 10, rounds)
        print(f"  plain stream {mb} MiB: {t:.1f} us, {n * 16 / t / 1e3:.0f} GB/s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nt", default="1,2")
    ap.add_argument("--cases", default="stream,decode,head,prompt")
    ap.add_argument("--rounds", type=int, default=7)
    a = ap.parse_args()
    c.M = c.load()
    nts = [int(v) for v in a.nt.split(",")]
    cases = a.cases.split(",")
    if "stream" in cases:
        stream(a.rounds)
    if "head" in cases:
        head(nts, a.rounds)
    if "decode" in cases or "prompt" in cases:
        ex = experts()
        if "decode" in cases:
            decode(ex, nts, a.rounds)
        if "prompt" in cases:
            prompt(ex, nts, a.rounds)


if __name__ == "__main__":
    main()
