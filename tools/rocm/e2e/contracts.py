#!/usr/bin/env python3
"""The engine's end-to-end contracts, checked through the API of a running tensorfold-native (token_sha compares
whole replies token for token; thinking off).

  a draft   each prompt (prose, code, math, JSON, tool calls, ~6k-token context), greedy and sampled with a seed:
            the drafted reply == the "draft": false reply (and the server says drafts were on / off)
  b conc    the same requests --concurrency at a time (barrier start) == each one's solo run
  c resume  a ~4.8k-token system prompt with a fresh nonce: a 3-turn conversation and 3 questions sharing it, sent as
            chat (later turns resume kept prompt states: cached_tokens > 0) == the same prompt ids sent fresh to
            /v1/completions first (an id prompt keeps no state, and nothing with this nonce was kept before)
  d chunks  a --long-token prompt (crosses the 2048-row prefill chunks): fresh solo == prefilled beside another long
            prompt == chat (keeps states) == chat again (resumed from its own kept state)

Exit 0 all pass, 1 a reply differs, 2 could not run, 3 a check could not be verified (e.g. no cache hit: UNCHECKED).
usage: contracts.py [--url URL] [--max-tokens 160] [--concurrency 8] [--long 20000] [--only a,b,c,d] [--cases ...]
"""
import argparse
import secrets
import sys
import threading

sys.dont_write_bytecode = True
import prompts  # noqa: E402
import tfapi  # noqa: E402

SAMPLED = {"temperature": 0.8, "top_p": 0.95, "top_k": 20}
results = []          # (status, contract, case, detail)


def report(status, contract, case, detail=""):
    results.append(status)
    print(f"{status:9s} {contract:6s} {case:28s} {detail}", flush=True)


def mode_args(mode, i):
    return {"temperature": 0.0} if mode == "greedy" else {**SAMPLED, "seed": 1234 + i}


def chat(messages, tools, max_tokens, **kw):
    return tfapi.summary(tfapi.chat(messages, max_tokens, **({"tools": tools} if tools else {}), **kw))


def fresh(messages, tools, max_tokens, **kw):
    return tfapi.summary(tfapi.complete(tfapi.render(messages, tools), max_tokens, **kw))


def together(calls):
    out = [None] * len(calls)
    barrier = threading.Barrier(len(calls))

    def run(i):
        barrier.wait()
        try:
            out[i] = calls[i]()
        except tfapi.ApiError as e:
            out[i] = e
    ts = [threading.Thread(target=run, args=(i,)) for i in range(len(calls))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    for r in out:
        if isinstance(r, Exception):
            raise r
    return out


def same(x, y):
    return x["sha"] == y["sha"] and x["finish"] == y["finish"]


def cases(long_words, pick):
    names = ["prose-sky", "prose-hashmap", "code-fib", "code-zig", "math-primes", "json-book", "tool-weather",
             "tool-search"]
    out = [(n, *prompts.REF[n]) for n in names]
    long = prompts.filler(long_words, 7) + "\n\nWhich three nouns appear most often in the text above? Then write a haiku."
    out.append(("long-6k", prompts.user(long), None))
    return [c for c in out if not pick or c[0] in pick]


def check_draft_conc(a, only):
    pick = a.cases.split(",") if a.cases else None
    jobs = [(f"{n}/{m}", msgs, tools, mode_args(m, i)) for i, (n, msgs, tools) in enumerate(cases(4600, pick))
            for m in ("greedy", "sampled")]
    solo = {}
    for case, msgs, tools, kw in jobs:
        d = chat(msgs, tools, a.max_tokens, **kw)
        solo[case] = d
        if "a" not in only:
            continue
        n = chat(msgs, tools, a.max_tokens, draft=False, **kw)
        info = f"sha {d['sha']}/{n['sha']} accepted {d['accepted']} tokens {d['tokens']}"
        if n["drafts"] is True:
            report("FAIL", "draft", case, '"draft": false was served with drafts')
        elif not same(d, n):
            report("FAIL", "draft", case, info)
        elif d["drafts"] is not True or not d["accepted"]:
            report("UNCHECKED", "draft", case, info + " (no draft was accepted / drafts off)")
        else:
            report("PASS", "draft", case, info)
    if "b" not in only:
        return
    for w in range(0, len(jobs), a.concurrency):
        wave = jobs[w:w + a.concurrency]
        got = together([lambda j=j: chat(j[1], j[2], a.max_tokens, **j[3]) for j in wave])
        for (case, *_), g in zip(wave, got):
            s = solo[case]
            report("PASS" if same(g, s) else "FAIL", "conc", case,
                   f"sha {g['sha']} solo {s['sha']} ({len(wave)} at once)")


def check_resume(a):
    nonce = secrets.token_hex(6)
    system = {"role": "system", "content": f"Session {nonce}. You are a careful assistant; the field log below is "
                                           "background.\n\n" + prompts.filler(4000, 11)}
    u = ["Summarize the field log in one sentence.", "Which object is mentioned first in the log?",
         "Write two lines of verse about that object."]
    canned = ["The log describes a long journey past rivers, bridges and markets.", "The first object is a river."]
    convo, convs = [system], []
    for t in range(3):
        convo = convo + ([{"role": "assistant", "content": canned[t - 1]}] if t else []) + [{"role": "user", "content": u[t]}]
        convs.append((f"turn{t + 1}", convo))
    for i, q in enumerate(["Name three tools a carpenter uses.", "What is 17 * 23?", "Translate 'good morning' to German."]):
        convs.append((f"shared-q{i + 1}", [system, {"role": "user", "content": q}]))
    convs.append(("shared-q1-again", convs[3][1]))
    for mode in ("greedy", "sampled"):
        base = {}
        for name, msgs in convs[:-1]:
            base[name] = fresh(msgs, None, a.resume_tokens, **mode_args(mode, 0))
            if base[name]["cached"]:
                report("UNCHECKED", "resume", f"{name}/{mode}", f"the fresh baseline resumed {base[name]['cached']} tokens")
        for name, msgs in convs:
            r = chat(msgs, None, a.resume_tokens, **mode_args(mode, 0))
            f = base[name.removesuffix("-again")]
            info = f"cached {r['cached']} sha {r['sha']} fresh {f['sha']}"
            if not same(r, f):
                report("FAIL", "resume", f"{name}/{mode}", info)
            elif not r["cached"] and name != "turn1":
                report("UNCHECKED", "resume", f"{name}/{mode}", info + " (no cache hit: prompt cache off?)")
            else:
                report("PASS", "resume", f"{name}/{mode}", info)


def check_chunks(a):
    words = int(a.long / 1.3)
    nonce = secrets.token_hex(6)
    p1 = prompts.user(f"[{nonce}] " + prompts.filler(words, 21) + "\n\nList the five rarest words above, then stop.")
    p2 = prompts.user(f"[{nonce}] " + prompts.filler(words, 22) + "\n\nWhat is the last sentence above about?")
    ids1, ids2 = tfapi.render(p1), tfapi.render(p2)
    print(f"          chunks prompts: {len(ids1)} and {len(ids2)} tokens")
    run = lambda ids: lambda: tfapi.summary(tfapi.complete(ids, a.max_tokens))  # noqa: E731
    f1, f2 = run(ids1)(), run(ids2)()
    c1, c2 = together([run(ids1), run(ids2)])
    k1 = chat(p1, None, a.max_tokens)
    k2 = chat(p1, None, a.max_tokens)
    for case, r, f in (("beside-another", c1, f1), ("beside-another-2", c2, f2), ("chat-keeps", k1, f1),
                       ("chat-resumed", k2, f1)):
        info = f"cached {r['cached']} sha {r['sha']} fresh {f['sha']}"
        if not same(r, f):
            report("FAIL", "chunks", case, info)
        elif case == "chat-resumed" and not r["cached"]:
            report("UNCHECKED", "chunks", case, info + " (no cache hit)")
        else:
            report("PASS", "chunks", case, info)


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--url", default=tfapi.URL)
    p.add_argument("--max-tokens", type=int, default=160)
    p.add_argument("--resume-tokens", type=int, default=64)
    p.add_argument("--concurrency", type=int, default=8)
    p.add_argument("--long", type=int, default=20000, help="tokens of the chunk-crossing prompt")
    p.add_argument("--only", default="a,b,c,d")
    p.add_argument("--cases", help="a/b prompts to run, e.g. prose-sky,tool-weather,long-6k (default all)")
    a = p.parse_args()
    tfapi.set_url(a.url)
    only = set(a.only.split(","))
    print(f"server {tfapi.URL}, model {tfapi.model()}")
    if only & {"a", "b"}:
        check_draft_conc(a, only)
    if "c" in only:
        check_resume(a)
    if "d" in only:
        check_chunks(a)
    n = {s: results.count(s) for s in ("PASS", "FAIL", "UNCHECKED")}
    verdict = "FAIL" if n["FAIL"] else ("UNCHECKED" if n["UNCHECKED"] or not results else "PASS")
    print(f"contracts: {verdict} ({n['PASS']} pass, {n['FAIL']} fail, {n['UNCHECKED']} unchecked)")
    return {"PASS": 0, "FAIL": 1, "UNCHECKED": 3}[verdict]


if __name__ == "__main__":
    try:
        sys.exit(main())
    except tfapi.ApiError as e:
        print(f"contracts: {e}", file=sys.stderr)
        sys.exit(2)
