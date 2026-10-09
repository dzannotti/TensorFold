#!/usr/bin/env python3
"""Decode speed against context, the way an agent harness loads the server: a fixed ~N-token prefix (deterministic
random prose standing in for a system prompt + tools, sized with /tokenize) is prefilled once by a warm-up request;
each measured request is that prefix plus a short varied user turn, so the prompt cache resumes the prefix and the
request pays a few hundred prompt tokens and then decode. Per context and mode: --reps requests, EOS ignored,
--tokens out, streamed. Decode = server rounds (tensorfold stats) / (last token - first token).
Modes: sampled (temperature 1.0, thinking on) and greedy (temperature 0, thinking off).

usage: longctx.py [--url http://127.0.0.1:10001] [--ctx 2k,8k,20k] [--tokens 300] [--reps 3] [--modes sampled,greedy]
                  [--seed S] [--json FILE]
Each request's n-grams are new to the server only on the first sight of a prompt: a fresh --seed per server session
measures the cold-page-cache case (the n-gram table is host-mapped, notes/longctx.md).
"""
import argparse
import json
import statistics
import sys

sys.dont_write_bytecode = True
import bench  # noqa: E402
import tfapi  # noqa: E402

MODES = {"sampled": (1.0, True), "greedy": (0.0, False)}
TURNS = ("Explain how a hash map works, with collision handling.", "Summarize the text above in three bullet points.",
         "Write a Python function that merges two sorted lists, then explain its complexity.",
         "What are the trade-offs between B-trees and LSM trees?", "Describe how TCP congestion control works.",
         "Write a haiku about the text above, then explain each line.")


def stats_of(r: dict) -> dict:
    for k in ("tensorfold", "speculative"):
        d = r.get(k) or {}
        for v in (d, d.get("speculative") or {}, d.get("spec") or {}):
            if isinstance(v, dict) and "rounds" in v:
                return v
    return {}


def one(prefix: str, turn: str, tokens: int, temperature: float, think: bool) -> dict:
    r = tfapi.stream_chat([{"role": "system", "content": prefix}, {"role": "user", "content": turn}],
                          max_tokens=tokens, temperature=temperature, ignore_eos=True,
                          chat_template_kwargs={"enable_thinking": think})
    u = r["usage"] or {}
    out = u.get("completion_tokens", r["pieces"])
    rounds = int(stats_of(r).get("rounds") or r["pieces"])
    span = r["last"] - r["first"]
    return {"prompt": u.get("prompt_tokens"), "cached": (u.get("prompt_tokens_details") or {}).get("cached_tokens"),
            "ttft_s": r["first"] - r["t0"], "tokens": out, "rounds": rounds, "tok_per_s": (out - 1) / span,
            "rounds_per_s": (rounds - 1) / span, "ms_per_round": 1000 * span / max(1, rounds - 1),
            "tok_per_round": out / max(1, rounds)}


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default="http://127.0.0.1:10001")
    p.add_argument("--ctx", default="2k,8k,20k")
    p.add_argument("--tokens", type=int, default=300)
    p.add_argument("--reps", type=int, default=3)
    p.add_argument("--modes", default="sampled,greedy")
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--json")
    a = p.parse_args()
    tfapi.set_url(a.url)
    res = []
    print(f"{'ctx':>6} {'mode':>7} {'prompt':>7} {'cached':>7} {'ttft s':>6} {'tok/s':>6} {'rounds/s':>8} {'ms/round':>8} "
          f"{'tok/round':>9}")
    for size in a.ctx.split(","):
        n = bench.size_of(size)
        prefix = bench.prose(n, a.seed + n)
        one(prefix, "Reply with one word.", 4, 0.0, False)          # the prefix prefilled and cached once
        for mode in a.modes.split(","):
            temp, think = MODES[mode]
            rows = []
            for i in range(a.reps):
                r = one(prefix, f"[{mode} {i}] {TURNS[i % len(TURNS)]}", a.tokens, temp, think)
                rows.append(r)
                res.append(r | {"ctx": n, "mode": mode, "rep": i})
                print(f"{n:6d} {mode:>7} {r['prompt'] or 0:7d} {r['cached'] or 0:7d} {r['ttft_s']:6.2f} {r['tok_per_s']:6.1f} "
                      f"{r['rounds_per_s']:8.2f} {r['ms_per_round']:8.1f} {r['tok_per_round']:9.2f}", flush=True)
            med = {k: statistics.median(r[k] for r in rows) for k in ("tok_per_s", "rounds_per_s", "ms_per_round", "tok_per_round")}
            print(f"{n:6d} {mode:>7} median{'':17} {med['tok_per_s']:6.1f} {med['rounds_per_s']:8.2f} "
                  f"{med['ms_per_round']:8.1f} {med['tok_per_round']:9.2f}", flush=True)
    if a.json:
        json.dump(res, open(a.json, "w"), indent=1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
