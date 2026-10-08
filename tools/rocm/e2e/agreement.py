#!/usr/bin/env python3
"""Teacher-forced top-1 agreement of a server against the CUDA reference replies (ref/thorim.json).

For each reference reply R (prompt ids P): for each position k, POST /v1/completions with prompt P + R[:k],
max_tokens 1, greedy; the drawn token's id comes from its token_sha (sha256 of the id, looked up over the vocabulary).
Reports per prompt the share of positions whose draw equals R[k] and the positions where it does not, plus the
free-running greedy reply's first divergence from R. Exit 0 when overall agreement >= --min-rate, else 1.

usage: agreement.py [--url URL] [--ref ref/thorim.json] [--names a,b] [--max-positions N] [--jobs 4] [--json FILE]
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor

sys.dont_write_bytecode = True
import tfapi  # noqa: E402


def first_diff(a, b):
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return None if len(a) == len(b) else min(len(a), len(b))


def draw(ids):
    s = tfapi.summary(tfapi.complete(ids, 1, 0.0))
    return tfapi.id_of(s["sha"]) if s["sha"] else None


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=tfapi.URL)
    p.add_argument("--ref", default=os.path.join(os.path.dirname(__file__), "ref", "thorim.json"))
    p.add_argument("--names", help="comma-separated subset of the reference prompts")
    p.add_argument("--max-positions", type=int, default=0, help="first N positions a reply (0: all)")
    p.add_argument("--jobs", type=int, default=4, help="requests in flight")
    p.add_argument("--min-rate", type=float, default=0.98)
    p.add_argument("--json")
    a = p.parse_args()
    tfapi.set_url(a.url)
    ref = json.load(open(a.ref))
    items = [i for i in ref["items"] if not a.names or i["name"] in a.names.split(",")]
    tfapi.id_of("0")  # build the sha table once, before the threads
    results, total, agree = [], 0, 0
    print(f"{'prompt':16s} {'pos':>4s} {'agree':>6s}  free-run  divergent positions")
    with ThreadPoolExecutor(a.jobs) as pool:
        for it in items:
            if not it["ids_verified"]:
                print(f"{it['name']:16s} skipped: reference ids unverified")
                continue
            P, R = it["prompt_ids"], it["reply_ids"]
            n = min(len(R), a.max_positions) if a.max_positions else len(R)
            got = list(pool.map(draw, [P + R[:k] for k in range(n)]))
            bad = [k for k in range(n) if got[k] != R[k]]
            free = tfapi.summary(tfapi.complete(P, len(R), 0.0))
            mine = tfapi.reply_ids(free["text"], free["sha"], free["finish"])
            fd = "same" if free["sha"] == it["token_sha"] else (first_diff(mine, R) if mine else "?")
            total, agree = total + n, agree + n - len(bad)
            results.append({"name": it["name"], "positions": n, "agree": n - len(bad), "divergent": bad,
                            "drawn_at_divergent": [got[k] for k in bad], "ref_at_divergent": [R[k] for k in bad],
                            "free_run_sha": free["sha"], "free_run_first_divergence": fd})
            print(f"{it['name']:16s} {n:4d} {(n - len(bad)) / n:6.1%}  {str(fd):>8s}  {bad[:12]}"
                  + (" ..." if len(bad) > 12 else ""), flush=True)
    rate = agree / total if total else 0.0
    exact = sum(r["free_run_first_divergence"] == "same" for r in results)
    print(f"overall: {agree}/{total} positions agree ({rate:.2%}); {exact}/{len(results)} free-run replies identical")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"server": tfapi.URL, "ref": a.ref, "rate": rate, "results": results}, f, indent=1)
    ok = total > 0 and rate >= a.min_rate
    print(f"agreement: {'PASS' if ok else 'FAIL'} (min {a.min_rate:.0%})")
    return 0 if ok else 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except tfapi.ApiError as e:
        sys.exit(f"agreement: {e}")
