#!/usr/bin/env python3
"""Prefill and decode speed, reported the way MiaAI-Lab's Qwen3.8-Flash-Next README tables are (sparkDash on GB10).

Prefill: one fresh prompt a size (random prose, a new seed each run, so no prompt cache hits; checked), sized with the
  server's /tokenize to ~4k..128k (4096 * 2^k tokens of text), ending "Summarize the text above in one sentence.",
  16 tokens out, greedy, thinking off, streamed. Prefill speed = prompt tokens / time to first token (Mia's table: e.g.
  4,134 tokens / 1.64 s = 2,526 tok/s); the server's own prefill_seconds rate goes to the JSON too.
Decode: sparkDash's prompts (Mia's engine patch 0005, bench-many dash-*): prose = the hash-map explanation, each
  stream's copy suffixed " (stream i/N)" past one stream; code = one of the 32 short-function tasks a stream. 256 tokens,
  end-of-sequence ignored, greedy, thinking off, N requests barrier-started. Per request = (tokens - 1) / (its last
  token - its first token); aggregate = all streams' tokens after their first / (latest last token - earliest first
  token), so one request's aggregate equals its per-request rate as in Mia's tables. Medians over --reps.

usage: bench.py [--url URL] [--label NAME] [--prefill 4k,8k,...] [--clients 1,2,4,8] [--reps 3] [--json FILE]
"""
import argparse
import json
import random
import statistics
import sys
import threading
import time

sys.dont_write_bytecode = True
import tfapi  # noqa: E402

# Mia's tools/bench.py vocabulary and sentence rule (fresh random prose)
WORDS = ("time year people way day man thing woman life child world school state family student group country "
         "problem hand part place case week company system program question work government number night point "
         "home water room mother area money story fact month lot right study book eye job word business issue "
         "side kind head house service friend father power hour game line end member law car city community name "
         "president team minute idea kid body information back parent face others level office door health person "
         "art war history party result change morning reason research girl guy moment air teacher force education "
         "river mountain signal engine garden theory market winter method bridge letter window voice paper field").split()
SUFFIX = "\n\nSummarize the text above in one sentence."

# sparkDash's decode prompts, verbatim from MiaAI-Lab patches/0005-engine.patch (dash_prose, dash_code)
DASH_PROSE = ("Write a detailed step-by-step explanation of how a hash map works, including collision handling, "
              "resizing, and time complexity. Be thorough.")
_TAIL = ("Output only Python source. No comments, no docstrings, no markdown fences. Then add tests and the helpers "
         "this needs. Keep writing code.")
DASH_CODE = [t + "\n" + _TAIL for t in (
    "binary_search\ndef binary_search(nums, target) -> int: index of target in a sorted list, or -1.",
    "merge_sort\ndef merge_sort(nums) -> list: stable sort of a list of ints, returning a new list.",
    "lru_cache\nclass LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used.",
    "token_bucket\nclass TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False.",
    "ring_buffer\nclass RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow.",
    "dijkstra\ndef dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph.",
    "edit_distance\ndef edit_distance(a, b) -> int: Levenshtein distance between two strings.",
    "semver_cmp\ndef semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b.",
    "url_parse\ndef url_parse(url) -> dict: scheme, host, port, path, and query pairs. No extra libraries.",
    "json_pointer\ndef json_pointer(doc, pointer) -> object: follow an RFC 6901 pointer, or None if missing.",
    "glob_match\ndef glob_match(pattern, text) -> bool: * and ? wildcards, no character classes.",
    "csv_parse\ndef csv_parse(text) -> list: rows of fields, honoring double-quoted commas and escaped quotes.",
    "rle\ndef rle_encode(s) -> str and rle_decode(s) -> str: run-length encoding of single-byte runs.",
    "top_k\ndef top_k(nums, k) -> list: the k largest ints, unordered, using a bounded heap.",
    "interval_merge\ndef merge_intervals(spans) -> list: merge overlapping [start, end] pairs.",
    "topo_sort\ndef topo_sort(nodes, edges) -> list: a valid order, or None if the graph has a cycle.",
    "bloom_filter\nclass BloomFilter: add(item) and might_contain(item) with two hash functions over a bit array.",
    "moving_average\nclass MovingAverage: next(x) returns the mean of the last window values.",
    "retry_backoff\ndef backoff_delays(attempts, base_ms, cap_ms) -> list: exponential delays clipped at the cap.",
    "base64_encode\ndef b64_encode(data) -> str and b64_decode(text) -> bytes: standard base64, no libraries.",
    "expr_eval\ndef eval_expr(text) -> int: evaluate non-negative ints with + - * / and parentheses.",
    "histogram_percentile\ndef percentile(samples, p) -> float: nearest-rank percentile of a list of numbers.",
    "redact_secrets\ndef redact(text) -> str: replace AWS-looking keys and password= values with ***. Keep the rest.",
    "chunk_text\ndef chunk_text(text, size) -> list: split into chunks of at most size chars, breaking on spaces when possible.",
    "route_match\ndef route_match(pattern, path) -> dict or None: /users/:id style params.",
    "crc32\ndef crc32(data) -> int: IEEE CRC-32 of a bytes object.",
    "fixed_window\nclass FixedWindow: allow() is True up to limit events per window_s, else False.",
    "diff_lines\ndef diff_lines(a, b) -> list: line diff as equal/delete/insert ops using a simple LCS.",
    "infix_postfix\ndef infix_to_postfix(tokens) -> list: shunting-yard for + - * / and parentheses.",
    "consistent_hash\nclass ConsistentHash: add_node, remove_node, and get_node(key) on a ring of virtual nodes.",
    "utf8_decode\ndef utf8_decode(data) -> str: decode UTF-8 bytes, replacing invalid sequences with U+FFFD.",
    "dependency_closure\ndef closure(root, deps) -> list: packages reachable from root, each name once, in visit order.",
)]


def size_of(text: str) -> int:
    t = text.strip().lower()
    return int(float(t[:-1]) * 1024) if t.endswith("k") else int(t)


def prose(tokens: int, seed: int) -> str:
    """Random prose whose text (suffix included) tokenizes to ~tokens, converged with /tokenize."""
    rng = random.Random(seed)
    words = []

    def grow(n):
        while n > 0:
            s = [rng.choice(WORDS) for _ in range(rng.randint(6, 16))]
            words.extend(s[:-1] + [s[-1] + "."])
            n -= len(s)
    grow(int(tokens * 0.85))
    for _ in range(8):
        have = len(tfapi.tokenize(" ".join(words) + SUFFIX))
        if abs(have - tokens) <= 16:
            break
        if have < tokens:
            grow(int((tokens - have) * 0.85))
        else:
            del words[-max(1, int((have - tokens) * 0.85)):]
    return " ".join(words) + SUFFIX


def decode_prompt(task: str, i: int, n: int) -> str:
    if task == "code":
        return DASH_CODE[i % len(DASH_CODE)]
    return DASH_PROSE if n == 1 else f"{DASH_PROSE} (stream {i + 1}/{n})"


def together(jobs):
    barrier, out = threading.Barrier(len(jobs)), [None] * len(jobs)

    def run(i, kw):
        barrier.wait()
        try:
            out[i] = tfapi.stream_chat(**kw)
        except tfapi.ApiError as e:
            out[i] = e
    ts = [threading.Thread(target=run, args=(i, kw)) for i, kw in enumerate(jobs)]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for r in out:
        if isinstance(r, Exception):
            raise r
    return out


def med(xs):
    return statistics.median(xs) if xs else float("nan")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=tfapi.URL)
    p.add_argument("--label", default="run")
    p.add_argument("--prefill", default="4k,8k,16k,32k,64k,128k", help="sizes, '' to skip")
    p.add_argument("--prefill-runs", type=int, default=1, help="fresh prompts a size (Mia: one request)")
    p.add_argument("--tasks", default="prose,code", help="decode tasks, '' to skip")
    p.add_argument("--clients", default="1,2,4,8")
    p.add_argument("--reps", type=int, default=3, help="runs a concurrency level (median of N)")
    p.add_argument("--max-tokens", type=int, default=256)
    p.add_argument("--json", help="write every record and the summary here")
    a = p.parse_args()
    tfapi.set_url(a.url)
    seed = int(time.time())
    rows, summary = [], {"label": a.label, "url": a.url, "seed": seed, "prefill": [], "decode": []}
    tfapi.stream_chat([{"role": "user", "content": "Say hi."}], 8)  # warm-up
    print(f"## {a.label}\n")

    sizes = [size_of(x) for x in a.prefill.split(",") if x.strip()]
    if sizes:
        print("**Prefill** (one request)\n\n| Prompt | Tokens | Prefill speed | Time to first token |\n"
              "| ---: | ---: | ---: | ---: |")
    for size in sizes:
        recs = []
        for i in range(a.prefill_runs):
            text = prose(size, seed * 1009 + size + i)
            r = tfapi.stream_chat([{"role": "user", "content": text}], 16)
            n, cached = r["usage"].get("prompt_tokens"), (r["usage"].get("prompt_tokens_details") or {}).get("cached_tokens")
            if cached:
                sys.exit(f"prefill {size}: {cached} prompt tokens came from the cache: the prompt is not fresh")
            ttft = r["first"] - r["t0"]
            srv = r["tensorfold"].get("prefill_seconds")
            rec = {"kind": "prefill", "size": size, "prompt_tokens": n, "ttft": ttft, "tok_s": n / ttft,
                   "server_prefill_s": srv, "server_tok_s": n / srv if srv else None}
            rows.append(rec)
            recs.append(rec)
        row = {k: med([r[k] for r in recs]) for k in ("prompt_tokens", "ttft", "tok_s")}
        row["size"] = size
        summary["prefill"].append(row)
        print(f"| {size / 1024:g}k | {row['prompt_tokens']:,.0f} | {row['tok_s']:,.0f} tok/s | {row['ttft']:.2f} s |",
              flush=True)

    for task in [t for t in a.tasks.split(",") if t]:
        print(f"\n**Decode, {task}** (greedy, thinking off, {a.max_tokens} tokens, end-of-sequence ignored, median of "
              f"{a.reps})\n\n| Concurrent requests | Aggregate | Per request | Time to first token |\n"
              "| ---: | ---: | ---: | ---: |")
        for c in [int(x) for x in a.clients.split(",") if x]:
            aggs, pers, ttfts = [], [], []
            for rep in range(a.reps):
                jobs = [{"messages": [{"role": "user", "content": decode_prompt(task, i, c)}],
                         "max_tokens": a.max_tokens, "ignore_eos": True} for i in range(c)]
                batch = together(jobs)
                span = max(r["last"] for r in batch) - min(r["first"] for r in batch)
                after_first = sum(r["usage"]["completion_tokens"] - 1 for r in batch)
                aggs.append(after_first / span)
                for i, r in enumerate(batch):
                    n = r["usage"]["completion_tokens"]
                    per = (n - 1) / (r["last"] - r["first"]) if r["last"] > r["first"] else float("nan")
                    pers.append(per)
                    ttfts.append(r["first"] - r["t0"])
                    rows.append({"kind": "decode", "task": task, "clients": c, "rep": rep, "stream": i, "tokens": n,
                                 "per_request": per, "ttft": r["first"] - r["t0"], "aggregate": aggs[-1],
                                 "pieces": r["pieces"], "server_tok_s": r["tensorfold"].get("tokens_per_second"),
                                 "accepted": (r.get("speculative") or {}).get("accepted"),
                                 "token_sha": r["tensorfold"].get("token_sha")})
            row = {"task": task, "clients": c, "aggregate": med(aggs), "per_request": med(pers), "ttft": med(ttfts)}
            summary["decode"].append(row)
            print(f"| {c} | {row['aggregate']:.1f} tok/s | {row['per_request']:.1f} tok/s | {row['ttft'] * 1000:.0f} ms |",
                  flush=True)

    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": summary, "records": rows}, f, indent=1)
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except tfapi.ApiError as e:
        sys.exit(f"bench: {e}")
