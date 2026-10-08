#!/usr/bin/env python3
"""Capture greedy reference replies from the CUDA engine (thorim), one request at a time with a pause between.

Each prompt of prompts.REF is rendered with the server's chat template (thinking off, its tools included) via
/tokenize, then sent as token ids to /v1/completions (greedy, --max-tokens). The reply's token ids are its text
re-tokenized, kept only when they hash to the server's token_sha (ids_verified). Aborts on any error or on a request
slower than --max-seconds. Writes ref/<name>.json.

usage: capture_ref.py --url http://127.0.0.1:18088 [--out ref/thorim.json] [--pause 3] [--max-tokens 200]
"""
import argparse
import datetime
import json
import os
import sys
import time

sys.dont_write_bytecode = True
import prompts  # noqa: E402
import tfapi  # noqa: E402


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--url", default=tfapi.URL)
    p.add_argument("--out", default=os.path.join(os.path.dirname(__file__), "ref", "thorim.json"))
    p.add_argument("--pause", type=float, default=3.0)
    p.add_argument("--max-tokens", type=int, default=200)
    p.add_argument("--max-seconds", type=float, default=60.0)
    a = p.parse_args()
    tfapi.set_url(a.url)
    health = tfapi.get_json("/health")
    items = []
    for name, (messages, tools) in prompts.REF.items():
        ids = tfapi.render(messages, tools)
        t0 = time.time()
        r = tfapi.complete(ids, a.max_tokens, 0.0)
        secs = time.time() - t0
        s = tfapi.summary(r)
        if not s["sha"]:
            sys.exit(f"{name}: the reply has no token_sha")
        reply = tfapi.reply_ids(s["text"], s["sha"], s["finish"])
        items.append({"name": name, "messages": messages, "tools": tools, "prompt_ids": ids, "text": s["text"],
                      "token_sha": s["sha"], "finish": s["finish"], "completion_tokens": s["tokens"],
                      "reply_ids": reply, "ids_verified": reply is not None, "seconds": round(secs, 3)})
        print(f"{name:16s} {len(ids):4d} prompt, {s['tokens']:3d} reply tokens, {s['finish']:6s} sha {s['sha']} "
              f"ids {'ok' if reply else 'UNVERIFIED'}  {secs:.1f} s", flush=True)
        if secs > a.max_seconds:
            sys.exit(f"{name}: {secs:.0f} s, slower than --max-seconds {a.max_seconds}: aborting (server busy?)")
        time.sleep(a.pause)
    out = {"server": tfapi.URL, "model": tfapi.model(), "health": health,
           "captured": datetime.datetime.now(datetime.timezone.utc).isoformat(timespec="seconds"),
           "settings": {"endpoint": "/v1/completions with rendered ids", "temperature": 0, "thinking": False,
                        "max_tokens": a.max_tokens}, "items": items}
    os.makedirs(os.path.dirname(os.path.abspath(a.out)), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, ensure_ascii=False)
    bad = sum(not i["ids_verified"] for i in items)
    print(f"wrote {a.out}: {len(items)} replies, {bad} without verified ids")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except tfapi.ApiError as e:
        sys.exit(f"capture_ref: {e}")
