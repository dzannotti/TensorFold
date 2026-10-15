#!/usr/bin/env python3
"""Quality gate for the Qwen3.8-27B HIP engine: perplexity, KL and top-1 agreement on a fixed 8 x 4,096 set.

Method (after Mia's AI Lab): 8 sequences of 4,096 tokens (4 wikitext-2 test, 4 CPython source,
tools/strix/quality_ids.npy), each one prompt pass through the engine, which steps it as two 2,048-row
chunks (forward.SPAN) so the second chunk reads the first's cached keys and recurrent state. Every
position's fp32 logits are scored against references:

  bf16   the bf16 HF checkpoint (tools/strix/quality_ref.py ref): PPL, mean KL(ref || engine), top-1.
         Fails if KL rises more than --kl-slack above the baseline JSON's, or top-1 falls more than
         --top1-slack points below it (the accepted Q4 floor).
  base   the baseline engine's own logits (--save-base writes them): a speed change must keep
         KL <= --kl (0.005) and top-1 >= --top1 (98.5%) against them, the "intact" bar.

  source ~/.local/opt/tfrocm/env.sh && zig build tf-qwen35-test -Dhip-arch=gfx1151 -Doptimize=ReleaseFast
  tools/strix/quality.py                          # gate the current build (prompt path)
  tools/strix/quality.py --decode                 # the same rows through one-token decode rounds
  tools/strix/quality.py --save-base --save-json tools/strix/baselines/mlx4-hip-gfx1151.json   # record a new floor

Every run's JSON goes to tools/strix/baselines/last[-decode].json (untracked); --save-json names a floor to commit.

Exit 0 when every gate holds, 1 when one fails, 2 on a setup error. Needs numpy only.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
ROWS = 256  # rows scored per block


def log_softmax(x):
    m = x.max(1, keepdims=True)
    return x - (m + np.log(np.exp(x - m).sum(1, keepdims=True)))


def score(cand, ref, ids):
    """Sums over one sequence: candidate and reference NLL, KL(ref || cand) and top-1 matches."""
    s = dict(nll=0.0, ref_nll=0.0, kl=0.0, top1=0, rows=0, targets=0)
    for r0 in range(0, len(ids), ROWS):
        lc = log_softmax(cand[r0:r0 + ROWS].astype(np.float32))
        lr = log_softmax(ref[r0:r0 + ROWS].astype(np.float32))
        s["kl"] += float((np.exp(lr) * (lr - lc)).sum())
        s["top1"] += int((lc.argmax(1) == lr.argmax(1)).sum())
        s["rows"] += len(lc)
        tgt = ids[r0 + 1:r0 + ROWS + 1]
        at = np.arange(len(tgt))
        s["nll"] -= float(lc[at, tgt].sum())
        s["ref_nll"] -= float(lr[at, tgt].sum())
        s["targets"] += len(tgt)
    return s


def summary(parts):
    t = {k: sum(p[k] for p in parts) for k in parts[0]}
    return {"ppl": float(np.exp(t["nll"] / t["targets"])), "ref_ppl": float(np.exp(t["ref_nll"] / t["targets"])),
            "kl": t["kl"] / t["rows"], "top1": 100.0 * t["top1"] / t["rows"]}


def die(msg):
    print(msg, file=sys.stderr)
    sys.exit(2)


def engine_logits(a, ids_row, out):
    with tempfile.NamedTemporaryFile(suffix=".npy", dir=os.path.dirname(out)) as f:
        np.save(f.name, ids_row.astype(np.int32))
        r = subprocess.run([a.bin, "logits", a.model, f.name, out, "--f32"] + (["--decode"] if a.decode else []),
                           capture_output=True, text=True)
    if r.returncode:
        die(f"engine failed ({r.returncode}):\n{r.stdout}{r.stderr}")
    return np.load(out, mmap_mode="r")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default=os.path.expanduser("~/models/q27/Qwen3.8-27B-MLX-4bit"))
    p.add_argument("--bin", default=os.path.join(ROOT, "zig-out", "bin", "tf-qwen35-test"))
    p.add_argument("--ids", default=os.path.join(HERE, "quality_ids.npy"))
    p.add_argument("--ref", default=os.path.expanduser("~/tfq/ref"), help="bf16 reference logits (quality_ref.py)")
    p.add_argument("--base", default=os.path.expanduser("~/tfq/base"), help="baseline engine logits")
    p.add_argument("--baseline", default=os.path.join(HERE, "baselines", "mlx4-hip-gfx1151.json"),
                   help="baseline JSON (the bf16 floor); '' for none")
    p.add_argument("--decode", action="store_true", help="every row through a one-token decode round (the drafting kernels)")
    p.add_argument("--kl", type=float, default=0.005, help="max mean KL vs the baseline engine")
    p.add_argument("--top1", type=float, default=98.5, help="min top-1 %% vs the baseline engine")
    p.add_argument("--kl-slack", type=float, default=0.005, help="max KL rise vs bf16 over the floor")
    p.add_argument("--top1-slack", type=float, default=0.5, help="max top-1 points lost vs bf16 under the floor")
    p.add_argument("--seqs", type=int, default=0, help="score only the first N sequences (a quick look; no gate)")
    p.add_argument("--save-base", action="store_true", help="write this run's logits (fp16) to --base")
    p.add_argument("--save-json", help="also write the result JSON here (a new floor)")
    p.add_argument("--tmp", default="/dev/shm" if os.path.isdir("/dev/shm") else None)
    a = p.parse_args()

    ids = np.load(a.ids)
    n = a.seqs or len(ids)
    for path in [a.bin] + [os.path.join(a.ref, f"ref_{i}.npy") for i in range(n)]:
        if not os.path.exists(path):
            die(f"missing {path}")
    use_base = not a.save_base and all(os.path.exists(os.path.join(a.base, f"ref_{i}.npy")) for i in range(n))
    if a.save_base:
        os.makedirs(a.base, exist_ok=True)

    t0 = time.time()
    vs_ref, vs_base, per_seq, engine_s = [], [], [], 0.0
    work = tempfile.mkdtemp(prefix="tfq-", dir=a.tmp)
    try:
        for i in range(n):
            t = time.time()
            cand = engine_logits(a, ids[i], os.path.join(work, "cand.npy"))
            engine_s += time.time() - t
            ref = np.load(os.path.join(a.ref, f"ref_{i}.npy"), mmap_mode="r")
            if cand.shape[1] != ref.shape[1]:
                if cand.shape[1] < ref.shape[1]:
                    die(f"engine vocab {cand.shape[1]} < reference {ref.shape[1]}")
                cand = cand[:, :ref.shape[1]]  # the head's padded columns
            s = score(cand, ref, ids[i])
            vs_ref.append(s)
            row = {"seq": i, **summary([s])}
            if use_base:
                b = score(cand, np.load(os.path.join(a.base, f"ref_{i}.npy"), mmap_mode="r"), ids[i])
                vs_base.append(b)
                row.update(base_kl=summary([b])["kl"], base_top1=summary([b])["top1"])
            if a.save_base:
                np.save(os.path.join(a.base, f"ref_{i}.npy"), np.asarray(cand, dtype=np.float16))
            per_seq.append(row)
            print(f"seq {i}: ppl {row['ppl']:.4f} (bf16 {row['ref_ppl']:.4f})  kl {row['kl']:.5f}  top1 {row['top1']:.2f}%"
                  + (f"  | vs base kl {row['base_kl']:.6f} top1 {row['base_top1']:.2f}%" if use_base else ""), flush=True)
            del cand
    finally:
        shutil.rmtree(work, ignore_errors=True)

    res = {"bf16": summary(vs_ref), "per_seq": per_seq, "seqs": n, "model": a.model, "path": "decode" if a.decode else "prompt",
           "commit": subprocess.run(["git", "-C", ROOT, "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip(),
           "engine_seconds": round(engine_s, 1), "wall_seconds": round(time.time() - t0, 1)}
    if use_base:
        res["base"] = summary(vs_base)
    r = res["bf16"]
    print(f"\nengine vs bf16: PPL {r['ppl']:.4f}  (bf16 PPL {r['ref_ppl']:.4f})  KL {r['kl']:.5f}  top-1 {r['top1']:.2f}%")
    fails = []
    if use_base:
        b = res["base"]
        print(f"engine vs base: KL {b['kl']:.6f} (<= {a.kl})  top-1 {b['top1']:.2f}% (>= {a.top1})")
        if b["kl"] > a.kl:
            fails.append(f"KL vs base {b['kl']:.6f} > {a.kl}")
        if b["top1"] < a.top1:
            fails.append(f"top-1 vs base {b['top1']:.2f}% < {a.top1}%")
    if a.baseline and not a.seqs and os.path.exists(a.baseline):
        f = json.load(open(a.baseline))["bf16"]
        print(f"floor ({os.path.basename(a.baseline)}): KL {f['kl']:.5f}  top-1 {f['top1']:.2f}%  PPL {f['ppl']:.4f}")
        if r["kl"] > f["kl"] + a.kl_slack:
            fails.append(f"KL vs bf16 {r['kl']:.5f} > floor {f['kl']:.5f} + {a.kl_slack}")
        if r["top1"] < f["top1"] - a.top1_slack:
            fails.append(f"top-1 vs bf16 {r['top1']:.2f}% < floor {f['top1']:.2f}% - {a.top1_slack}")
    print(f"wall {res['wall_seconds']}s (engine {res['engine_seconds']}s)")
    for path in [os.path.join(HERE, "baselines", "last-decode.json" if a.decode else "last.json"), a.save_json]:
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            json.dump(res, open(path, "w"), indent=1)
            print(f"wrote {path}")
    if a.seqs:
        print("partial run: no gate")
        return 0
    print("FAIL: " + "; ".join(fails) if fails else "PASS: quality intact")
    return 1 if fails else 0


if __name__ == "__main__":
    sys.exit(main())
