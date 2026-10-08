"""A/B of fn_int4.hip builds, alternated round by round (prod shares the GPU: the per-variant minimum is the signal):
python tools/rocm/int4_ab.py "-DTF_INT4_ABL=0" "-DTF_INT4_ABL=1" ... [--src other.hip]"""
import argparse, hashlib, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiprun
import int4_check as c
import int4_bench as b


def cases(ex):
    """(name, bytes, fn(module)) at the decode shapes and the head."""
    out = []
    V = 248320 // 5   # a fifth of the head (keeps memory small; still far past the caches; 32 | V / 16)
    hw = b.rand_words(V * b.D // 8)
    hs = torch.full((V * b.D // 128,), 0.01, dtype=torch.float16, device="cuda")
    hx = torch.randn(16, b.D, device="cuda").bfloat16()
    ho = torch.zeros(16, V, dtype=torch.bfloat16, device="cuda")
    for rows in (1, 16):
        out.append((f"head/5 {rows} rows", V * b.D / 2, lambda rows=rows, nt=1: c.launch(128, nt, 1, 1, 3, hx, b.D, 0, hw, hs, b.D, V, None, rows, ho, (rows + 15) // 16)))
    for eu, m in ((5, 1), (40, 4), (40, 16), (75, 1)):
        # rotate through disjoint expert sets so the 32 MiB MALL never holds the next call's weights (a real layer)
        R = eu * m // b.TOP
        sets = []
        for v in range(max(2, min(8, b.E // eu))):
            picks = np.full((R, b.SLOTS), b.E, np.int32)
            for t in range(R):
                for j in range(b.TOP):
                    picks[t, j] = ((b.TOP * t + j) % eu * (b.E // eu) + v) % b.E
            sets.append(c.make_plan(picks, b.E + 1, 16)[0])
        mi = c.max_items(R * b.SLOTS, b.E + 1, 16)
        x = torch.randn(R, b.D, device="cuda").bfloat16()
        act = torch.zeros(R * b.SLOTS, b.NI, dtype=torch.bfloat16, device="cuda")
        y = torch.zeros(R * b.SLOTS, b.D, dtype=torch.bfloat16, device="cuda")
        wb = eu * 3 * b.NI * b.D / 2
        it = iter(range(1 << 60))
        out.append((f"experts {eu} x {m:2d} rows", wb, lambda x=x, act=act, y=y, sets=sets, mi=mi, R=R, it=it: b.run_moe(ex, x, act, y, sets[next(it) % len(sets)], mi, R, 1, 1, 1, 2)))
    probe = hiprun.Module(hiprun.build(f"{c.ROOT}/tools/rocm/int4_probe.hip", hiprun.cache("int4_probe.co")))
    for mb in (12, 80):
        n = mb * 2**20 // 16
        buf = torch.empty(n * 4, dtype=torch.int32, device="cuda")
        sink = torch.empty(40 * 64 * 256, dtype=torch.int32, device="cuda")
        for name, blocks in (("probe_stream", 1280), ("probe_stream4", 640), ("probe_stream8", 320)):
            out.append((f"{name} {mb} MiB", n * 16, lambda buf=buf, n=n, name=name, blocks=blocks: probe.launch(name, blocks, 256, [buf, ("q", n), sink])))
    big = torch.empty(8 * 12 * 2**20 // 4, dtype=torch.int32, device="cuda")
    n = 12 * 2**20 // 16
    rot = iter(range(1 << 60))
    for name, blocks in (("probe_stream", 1280), ("probe_stream4", 640)):
        out.append((f"{name} 12 MiB of 96", n * 16, lambda name=name, blocks=blocks: probe.launch(name, blocks, 256, [big[(next(rot) % 8) * n * 4:], ("q", n), sink])))
    total = 80 * 2**20
    for steps in (2, 20, 160):
        regions = total // (steps * 1024)
        out.append((f"regions of {steps} KiB, 80 MiB", total, lambda steps=steps, regions=regions: probe.launch("probe_regions", 640, 128, [buf, ("i", regions), ("i", steps), sink])))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("variants", nargs="+")
    ap.add_argument("--src", action="append", help="sources, one a variant (default fn_int4.hip)")
    ap.add_argument("--rounds", type=int, default=15)
    a = ap.parse_args()
    srcs = a.src or [f"{c.ROOT}/zig/kernels/hip/fn_int4.hip"] * len(a.variants)
    mods = []
    for v, src in zip(a.variants, srcs):
        tag = hashlib.sha1((v + src).encode()).hexdigest()[:8]
        mods.append(hiprun.Module(hiprun.build(src, hiprun.cache(f"ab_{tag}.co"), v.split())))
    ex = b.experts()
    cs = cases(ex)
    best = np.full((len(cs), len(mods)), np.inf)
    for _ in range(a.rounds):
        for i, (_, _, f) in enumerate(cs):
            for j, m in enumerate(mods):
                c.M, c.OCC = m, {}
                best[i, j] = min(best[i, j], hiprun.best_us(f, 10, 1))
    print("case".ljust(24) + "".join(v[:22].rjust(24) for v in a.variants))
    for i, (name, wb, _) in enumerate(cs):
        print(name.ljust(24) + "".join(f"{best[i, j]:8.1f} us {wb / best[i, j] / 1e3:5.0f} GB/s".rjust(24) for j in range(len(mods))))


if __name__ == "__main__":
    main()
