#!/usr/bin/env python3
"""Launches a decode round from a rocprofv3 kernel trace of a 1-layer view (graph mode is fine), and the full model's
estimate: a round starts at the main forward's ``_embed`` (an MTP forward's is followed by ``_rmsnorm``); the main
forward's one decoder layer is everything from that embed to the read-out's third write-back (the finish's; its
streams copy, if any, is the finish's too). Prints per-round launches by part, the layer's kernels, and
full = round + (layers - 1) layers (36 DeltaNet + 12 attention: pass both views' layer counts).

    python3 tools/rocm/launch_count.py TRACE.csv [--skip 0.3]
"""

import argparse
import collections
import csv
import sys

WB = ("_hc_writeback", "_hc_wbn", "_hc_wb_norm")


def short(name: str) -> str:
    return name.split("(")[0].replace("void ", "")[:48]


def rounds(names):
    """[(main forward kernels, MTP forwards' kernels, the rest)] for each whole round."""
    starts = [i for i, n in enumerate(names) if n == "_embed" and i + 1 < len(names) and names[i + 1] != "_rmsnorm"]
    out = []
    for a, b in zip(starts, starts[1:]):
        seg = names[a:b]
        mtp = [i for i, n in enumerate(seg) if n == "_embed" and i + 1 < len(seg) and seg[i + 1] == "_rmsnorm"]
        main_end = mtp[0] if mtp else len(seg)
        fwds = [seg[x:y] for x, y in zip(mtp, mtp[1:] + [len(seg)])]
        out.append((seg[:main_end], fwds))
    return out


def layer_of(main):
    wbs = [i for i, n in enumerate(main) if n in WB]
    if len(wbs) < 3:
        return None
    end = wbs[2]
    if main[end - 1].startswith("__amd_rocclr_copyBuffer"):
        end -= 1
    return main[1:end]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("trace")
    ap.add_argument("--skip", type=float, default=0.3, help="the first fraction of rounds (captures, warm-up)")
    a = ap.parse_args()
    rows = list(csv.DictReader(open(a.trace)))
    rows.sort(key=lambda r: int(r["Start_Timestamp"]))
    names = [short(r["Kernel_Name"]) for r in rows]
    rs = rounds(names)
    rs = rs[int(len(rs) * a.skip):]
    if not rs:
        print("no whole rounds")
        return 1
    n = len(rs)
    tot = sum(len(m) + sum(len(f) for f in fw) for m, fw in rs) / n
    mainc = sum(len(m) for m, _ in rs) / n
    fw = [f for _, fws in rs for f in fws]
    mtpc = sum(len(f) for f in fw) / n
    per_fwd = sum(len(f) for f in fw) / max(1, len(fw))
    lay = collections.Counter(len(layer_of(m) or []) for m, _ in rs).most_common(1)[0][0]
    ex = next(layer_of(m) for m, _ in rs if len(layer_of(m) or []) == lay)
    print(f"{a.trace}: {n} rounds; launches a round {tot:.1f} = main forward + draws {mainc:.1f} + MTP {mtpc:.1f} "
          f"({len(fw) / n:.2f} MTP forwards a round, {per_fwd:.1f} launches each); the decoder layer {lay}")
    print("  layer: " + ", ".join(ex))
    hist = collections.Counter(x for m, fws in rs for x in m + [y for f in fws for y in f])
    print("  a round: " + ", ".join(f"{k} {v / n:.1f}" for k, v in hist.most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
