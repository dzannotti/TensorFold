"""Ciru's eight-concurrent serving check (MiniCPM-4.7 CIRU docs/BENCHMARKS.md), text-only: eight requests at once,
thinking on, temperature 1.0, top-p 0.95, top-k 20, repetition penalty 1.05, each finishing on its own.

Aggregate TG = output tokens / union of first-token-to-end intervals; pooled TG/request = output tokens / summed
first-token-to-end durations (Ciru's definitions). Ciru's eight requests were captured vision/tool bodies; these are
eight different text tasks, so only the shape of the load matches.

    python tools/strix/concurrent8.py --base-url http://127.0.0.1:8099/v1 --model m [--penalty 1.05]
"""

import argparse
import json
import threading
import time
import urllib.error
import urllib.request

TASKS = [
    "Write a Python module implementing an LRU cache with tests.",
    "Explain how TCP congestion control works, with the main algorithms compared.",
    "Design a REST API for a library management system: endpoints, schemas, and error handling.",
    "Write a short story about a lighthouse keeper who finds a message in a bottle.",
    "Implement Dijkstra's algorithm in Rust and explain its complexity.",
    "Compare three approaches to database sharding and when to use each.",
    "Write a bash script that backs up a directory with rotation, and explain each part.",
    "Plan a three-day itinerary for Kyoto with reasons for each stop.",
]


def run(url, model, prompt, max_tokens, penalty, out):
    body = {"model": model, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 1.0, "top_p": 0.95, "top_k": 20, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": True}}
    if penalty:
        body["repetition_penalty"] = penalty
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    t0 = time.monotonic()
    first = last = None
    usage = {}
    try:
        with urllib.request.urlopen(req, timeout=3600) as r:
            for line in r:
                if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                    continue
                d = json.loads(line[6:])
                usage = d.get("usage") or usage
                if any(c.get("delta", {}).get("content") or c.get("delta", {}).get("reasoning_content") for c in d.get("choices", [])):
                    last = time.monotonic()
                    first = first or last
    except urllib.error.HTTPError as e:
        out["error"] = e.read().decode()[:200]
        return
    out.update(start=t0, first=first, last=last, tokens=usage.get("completion_tokens", 0), prompt=usage.get("prompt_tokens", 0))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--model", default="m")
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--penalty", type=float, default=1.05, help="repetition_penalty; 0 leaves it out")
    a = ap.parse_args()
    outs = [{} for _ in TASKS]
    t0 = time.monotonic()
    ts = [threading.Thread(target=run, args=(a.base_url, a.model, p, a.max_tokens, a.penalty, o)) for p, o in zip(TASKS, outs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    wall = time.monotonic() - t0
    bad = [o.get("error") for o in outs if "error" in o or not o.get("first")]
    if bad:
        print("errors:", bad[:2])
        return
    tokens = sum(o["tokens"] for o in outs)
    spans = sorted((o["first"], o["last"]) for o in outs)
    union, end = 0.0, None
    for s, e in spans:
        if end is None or s > end:
            union += e - s
            end = e
        elif e > end:
            union += e - end
            end = e
    pooled = tokens / sum(o["last"] - o["first"] for o in outs)
    print(f"8 concurrent: output {tokens} tok, aggregate TG {tokens / union:.2f} tok/s, pooled TG/request {pooled:.2f}, "
          f"wall {wall:.2f} s, tokens/request {[o['tokens'] for o in outs]}")


if __name__ == "__main__":
    main()
