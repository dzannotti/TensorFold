"""tools/rocm/fp8_dec_bench.py CO_DIR [CO_DIR ...]: A/B decode bench of fn_qmmf code-object dirs, graph-replayed, weights rotated past the MALL, best of N, alternated."""
import argparse, sys, time, torch
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
import fp8_check as F

SH = {"gdn": (16384, 2560, True), "attn": (13312, 2560, True), "out": (2560, 6144, False), "sgu": (2560, 2560, False),
      "sdn": (2560, 1280, False)}
ap = argparse.ArgumentParser()
ap.add_argument("cos", nargs="+")
ap.add_argument("--rows", default="1,4,16,32")
ap.add_argument("--shapes", default="gdn,attn,out,sgu,sdn")
ap.add_argument("--tries", type=int, default=9)
a = ap.parse_args()
hips = [F.Hip(Path(c)) for c in a.cos]
for nm in a.shapes.split(","):
    n, k, ld = SH[nm]
    lins = [F.random_linear(n, k, 7)]
    wb = lins[0].w8.numel() + lins[0].bs.numel()
    for _ in range(max(1, (128 << 20) // wb)):
        lins.append(F.Fp8BlockLinear(lins[0].w8.clone(), lins[0].bs.clone(), n, k, lins[0].npad))
    x = torch.randn((64, k), device="cuda").to(torch.bfloat16)
    y = torch.empty((64, n + 64), dtype=torch.bfloat16, device="cuda")
    out = []
    for m in [int(r) for r in a.rows.split(",")]:
        reps = len(lins) * 2
        gs = []
        for h in hips:
            g = torch.cuda.CUDAGraph()
            h.matmul(lins[0], x[:m], y, ldo=(n + 64) if ld else None) if ld else h.matmul(lins[0], x[:m], y[:, :n])
            torch.cuda.synchronize()
            with torch.cuda.graph(g):
                for i in range(reps):
                    if ld:
                        h.matmul(lins[i % len(lins)], x[:m], y, ldo=n + 64)
                    else:
                        h.matmul(lins[i % len(lins)], x[:m], y[:, :n])
            gs.append(g)
        if not getattr(F, 'warm', False):
            t = time.time()
            while time.time() - t < 3:
                gs[0].replay()
            torch.cuda.synchronize(); F.warm = True
        best = [1e9] * len(hips)
        for _ in range(a.tries):
            for j, g in enumerate(gs):
                e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                e0.record(); g.replay(); e1.record(); e1.synchronize()
                best[j] = min(best[j], e0.elapsed_time(e1) * 1e3 / reps)
        out.append(f"m{m} " + "/".join(f"{b:.1f}" for b in best) + "us " + "/".join(f"{wb / b / 1e3:.0f}" for b in best))
    print(f"{nm} n{n} k{k}: " + "  ".join(out), flush=True)
    del lins, gs
    torch.cuda.empty_cache()
