"""Ciru's long-context probe (MiniCPM-4.7 CIRU docs/BENCHMARKS.md), rebuilt: one request, a ~63,870-token prompt,
512 generated tokens, thinking off, temperature 0.6. It runs uncached, then the same prompt again (cached prefix).

PP = prompt tokens / time to first token; TG = (tokens - 1) / (last token - first token), from the client side.
Ciru's PP is server-side prefill time, so this PP reads slightly low (it includes one decode step and transport).

    python tools/strix/longctx.py --base-url http://127.0.0.1:8099/v1 --model m [--tokens 63870] [--seed 0]
"""

import argparse
import json
import random
import time
import urllib.request

WORDS = ("the of and to in is was for on that with as by at from his her an were are which this be or had not but "
         "they have one all their there been has more when will would who so its into time than only other new some "
         "could these two may first then do any like my now over such our man me even most made after also did many "
         "before must through back years where much your way well down should because each just those people").split()


def stream(url, model, messages, max_tokens, temperature):
    body = {"model": model, "messages": messages, "max_tokens": max_tokens, "min_tokens": max_tokens,
            "ignore_eos": True, "temperature": temperature, "stream": True, "stream_options": {"include_usage": True},
            "chat_template_kwargs": {"enable_thinking": False}}
    req = urllib.request.Request(url + "/chat/completions", json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    t0, stamps, usage = time.monotonic(), [], {}
    with urllib.request.urlopen(req, timeout=3600) as r:
        for line in r:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            d = json.loads(line[6:])
            usage = d.get("usage") or usage
            if any(c.get("delta", {}).get("content") for c in d.get("choices", [])):
                stamps.append(time.monotonic() - t0)
    return stamps, usage


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default="http://127.0.0.1:8099/v1")
    ap.add_argument("--model", default="m")
    ap.add_argument("--tokens", type=int, default=63870, help="target prompt tokens (words are ~1 token each)")
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    text = " ".join(rng.choice(WORDS) for _ in range(a.tokens))
    messages = [{"role": "user", "content": f"{rng.getrandbits(64):016x}\n{text}\n\nSummarize the text above."}]
    for state in ("uncached", "cached"):
        stamps, usage = stream(a.base_url, a.model, messages, a.max_tokens, a.temperature)
        prompt = usage.get("prompt_tokens", a.tokens)
        n = len(stamps)
        # a streamed chunk can carry several tokens (a drafted round lands them together): count tokens, not chunks
        toks = usage.get("completion_tokens", n)
        tg = (toks - 1) / (stamps[-1] - stamps[0]) if n > 1 else 0.0
        print(f"{state:9} prompt {prompt:>6} tok  TTFT {stamps[0]:7.2f} s  PP {prompt / stamps[0]:8.1f} tok/s  "
              f"TG {tg:6.2f} tok/s  ({toks} tokens in {n} chunks, wall {stamps[-1]:.2f} s)")


if __name__ == "__main__":
    main()
