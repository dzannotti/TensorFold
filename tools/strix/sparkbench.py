#!/usr/bin/env python3
"""Stdlib-only port of sparkDash's decode / prefill / quality benches (sparkDash commit 2c005e3).

Sources (sparkDash repo):
  decode   server/collectors/DecodeBench.js, LlmStreaming.js, src/shared/llmPrompts.js
  prefill  server/collectors/PrefillBench.js
  quality  server/collectors/QualityBench.js, qualitySuite.js (+ data/gsm8k-200.json, data/mmlu-285.json)

Lines marked INFERRED are choices the source does not pin down for the published numbers.
"""
import argparse, json, math, statistics, sys, threading, time, urllib.error, urllib.request, uuid
from datetime import date, timedelta
import re

M32 = 0xFFFFFFFF


def jround(x):  # JS Math.round (half up), not Python's banker's rounding
    return math.floor(x + 0.5)


def round2(x):
    return jround(x * 100) / 100


def now_ms():
    return time.perf_counter() * 1000


def est_tokens(text):  # LlmStreaming.estimateTokenCount: ~4 UTF-16 units per token, min 1
    return max(1, jround(len(text.encode("utf-16-le")) / 8)) if text else 0


# ─── prompts (llmPrompts.js, verbatim) ──────────────────────────────────────

DECODE_PROSE_PROMPT = ("Write a detailed step-by-step explanation of how a hash map works, "
                       "including collision handling, resizing, and time complexity. Be thorough.")
DECODE_STRUCTURED_PROMPT = "Count from 1 to 200. Output only the numbers, separated by spaces. No other text."
CODE_TASK_TAIL = ("Output only Python source. No comments, no docstrings, no markdown fences. "
                  "Then add tests and the helpers this needs. Keep writing code.")
CODE_TASKS = [f"{n}\n{s}\n{CODE_TASK_TAIL}" for n, s in [
    ("binary_search", "def binary_search(nums, target) -> int: index of target in a sorted list, or -1."),
    ("merge_sort", "def merge_sort(nums) -> list: stable sort of a list of ints, returning a new list."),
    ("lru_cache", "class LRUCache: get(key) and put(key, value) with a fixed capacity, evicting the least recently used."),
    ("token_bucket", "class TokenBucket: allow(n) consumes n tokens refilled at a fixed rate, else returns False."),
    ("ring_buffer", "class RingBuffer: push and pop over a fixed-capacity array, raising on overflow and underflow."),
    ("dijkstra", "def dijkstra(graph, src) -> dict: shortest path weights from src on a non-negative weighted graph."),
    ("edit_distance", "def edit_distance(a, b) -> int: Levenshtein distance between two strings."),
    ("semver_cmp", "def semver_cmp(a, b) -> int: compare dotted numeric versions, negative if a < b."),
    ("url_parse", "def url_parse(url) -> dict: scheme, host, port, path, and query pairs. No extra libraries."),
    ("json_pointer", "def json_pointer(doc, pointer) -> object: follow an RFC 6901 pointer, or None if missing."),
    ("glob_match", "def glob_match(pattern, text) -> bool: * and ? wildcards, no character classes."),
    ("csv_parse", "def csv_parse(text) -> list: rows of fields, honoring double-quoted commas and escaped quotes."),
    ("rle", "def rle_encode(s) -> str and rle_decode(s) -> str: run-length encoding of single-byte runs."),
    ("top_k", "def top_k(nums, k) -> list: the k largest ints, unordered, using a bounded heap."),
    ("interval_merge", "def merge_intervals(spans) -> list: merge overlapping [start, end] pairs."),
    ("topo_sort", "def topo_sort(nodes, edges) -> list: a valid order, or None if the graph has a cycle."),
    ("bloom_filter", "class BloomFilter: add(item) and might_contain(item) with two hash functions over a bit array."),
    ("moving_average", "class MovingAverage: next(x) returns the mean of the last window values."),
    ("retry_backoff", "def backoff_delays(attempts, base_ms, cap_ms) -> list: exponential delays clipped at the cap."),
    ("base64_encode", "def b64_encode(data) -> str and b64_decode(text) -> bytes: standard base64, no libraries."),
    ("expr_eval", "def eval_expr(text) -> int: evaluate non-negative ints with + - * / and parentheses."),
    ("histogram_percentile", "def percentile(samples, p) -> float: nearest-rank percentile of a list of numbers."),
    ("redact_secrets", "def redact(text) -> str: replace AWS-looking keys and password= values with ***. Keep the rest."),
    ("chunk_text", "def chunk_text(text, size) -> list: split into chunks of at most size chars, breaking on spaces when possible."),
    ("route_match", "def route_match(pattern, path) -> dict or None: /users/:id style params."),
    ("crc32", "def crc32(data) -> int: IEEE CRC-32 of a bytes object."),
    ("fixed_window", "class FixedWindow: allow() is True up to limit events per window_s, else False."),
    ("diff_lines", "def diff_lines(a, b) -> list: line diff as equal/delete/insert ops using a simple LCS."),
    ("infix_postfix", "def infix_to_postfix(tokens) -> list: shunting-yard for + - * / and parentheses."),
    ("consistent_hash", "class ConsistentHash: add_node, remove_node, and get_node(key) on a ring of virtual nodes."),
    ("utf8_decode", "def utf8_decode(data) -> str: decode UTF-8 bytes, replacing invalid sequences with U+FFFD."),
    ("dependency_closure", "def closure(root, deps) -> list: packages reachable from root, each name once, in visit order."),
]]
CODE_WARMUP_PROMPT = f"warmup_noop\ndef warmup_noop(x): return x unchanged.\n{CODE_TASK_TAIL}"


def decode_prompts(n, kind):  # pickDecodeBenchPrompts
    if kind == "code":
        size = len(CODE_TASKS)
        return [CODE_TASKS[i % size] if i < size else f"[stream {i + 1}]\n{CODE_TASKS[i % size]}" for i in range(n)]
    base = {"prose": DECODE_PROSE_PROMPT, "structured": DECODE_STRUCTURED_PROMPT}[kind]
    return [base] if n <= 1 else [f"{base} (stream {i + 1}/{n})" for i in range(n)]


# ─── streaming request (LlmStreaming.runStreamingRequest) ───────────────────

def thinking_kwargs(on):  # applyThinkingFlags
    return {"enable_thinking": on, "thinking": on, "thinking_mode": "enabled" if on else "disabled"}


def chat_body(model, prompt, max_tokens, think=False, **extra):
    body = {"messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0, "top_p": 1, **extra, "stream": True, "stream_options": {"include_usage": True}}
    if model:  # sparkDash sends model: undefined (dropped) when it has no id
        body = {"model": model, **body}
    body["chat_template_kwargs"] = thinking_kwargs(think)
    return body


def stream_once(url, body, timeout_s):
    t0 = now_ms()
    r = dict(t0=t0, tFirst=None, tLast=None, error=None, model=None, finishReason=None,
             usageCompletion=None, usagePrompt=None, cached=0, serverPromptMs=None, serverPromptN=None,
             answer="", reasoning="")
    chunk_tokens = 0
    try:
        req = urllib.request.Request(url, json.dumps(body).encode(), method="POST",
                                     headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
        # ponytail: urllib timeout is per socket read, not a total deadline like sparkDash's AbortController.
        with urllib.request.urlopen(req, timeout=timeout_s) as resp:
            for raw in resp:
                line = raw.decode("utf-8", "replace").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if not data or data == "[DONE]":
                    continue
                try:
                    js = json.loads(data)
                except ValueError:
                    continue
                r["model"] = js.get("model") or r["model"]
                u = js.get("usage")
                if isinstance(u, dict):
                    if u.get("completion_tokens") is not None:
                        r["usageCompletion"] = int(u["completion_tokens"])
                    if u.get("prompt_tokens") is not None:
                        r["usagePrompt"] = int(u["prompt_tokens"])
                    c = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
                    r["cached"] = max(r["cached"], c)
                tm = js.get("timings")
                if isinstance(tm, dict):  # llama.cpp server-side prompt timing
                    if (tm.get("prompt_ms") or 0) > 0 and (tm.get("prompt_n") or 0) > 0:
                        r["serverPromptMs"], r["serverPromptN"] = tm["prompt_ms"], tm["prompt_n"]
                    r["cached"] = max(r["cached"], tm.get("cache_n") or 0)
                ch = (js.get("choices") or [None])[0] or {}
                if ch.get("finish_reason"):
                    r["finishReason"] = ch["finish_reason"]
                d = ch.get("delta") or {}
                ans = d.get("content") if isinstance(d.get("content"), str) else ""
                rea = d.get("reasoning") or d.get("reasoning_content") or ""
                if not ans and not rea and isinstance(ch.get("text"), str):
                    ans = ch["text"]
                n = est_tokens(ans) + est_tokens(rea)
                if n > 0:  # first/last *visible* token (content or reasoning), not stream EOF
                    t = now_ms()
                    r["tFirst"] = r["tFirst"] if r["tFirst"] is not None else t
                    r["tLast"] = t
                    chunk_tokens += n
                    r["answer"] += ans
                    r["reasoning"] += rea
    except urllib.error.HTTPError as e:
        r["error"] = f"HTTP {e.code}: {e.read().decode('utf-8', 'replace')[:200]}"
    except Exception as e:  # noqa: BLE001 - every failure becomes a failed stream, as in sparkDash
        r["error"] = str(e) or type(e).__name__
    r["ttftMs"] = round2(r["tFirst"] - t0) if r["tFirst"] is not None else 0
    comp = r["usageCompletion"] if r["usageCompletion"] else chunk_tokens  # usage preferred, chunk estimate fallback
    r["completionTokens"] = comp
    r["decodeTokens"] = max(0, comp - (1 if comp > 0 else 0))  # first token excluded
    dms = r["tLast"] - r["tFirst"] if r["tFirst"] is not None and r["tLast"] > r["tFirst"] else 0
    r["decodeTps"] = round2(r["decodeTokens"] / dms * 1000) if dms > 0 and r["decodeTokens"] > 0 else 0
    r["prefillTokens"] = r["usagePrompt"] if r["usagePrompt"] else est_tokens(
        "\n".join(m["content"] for m in body["messages"]))
    return r


def stream(url, body, timeout_s, think=False):
    r = stream_once(url, body, timeout_s)
    if r["error"] and r["error"].startswith("HTTP 400") and "chat_template_kwargs" in body:
        b = {k: v for k, v in body.items() if k not in ("chat_template_kwargs", "enable_thinking", "thinking", "thinking_mode")}
        if not think:  # thinkingOffFallbackBody: keep an explicit off switch
            b.update(chat_template_kwargs=thinking_kwargs(False), thinking={"type": "disabled"}, enable_thinking=False)
        r = stream_once(url, b, timeout_s)
    return r


# ─── decode (DecodeBench.js) ────────────────────────────────────────────────

def decode_wave(url, model, c, max_tokens, kind):
    prompts = decode_prompts(c, kind)
    results = [None] * c

    def run(i):
        body = chat_body(model, prompts[i], max_tokens, min_tokens=max_tokens, ignore_eos=True, stop=[])
        r = stream(url, body, 360)
        if r["error"] and r["error"].startswith("HTTP 400"):  # stripFillForceFields retry
            r = stream(url, {k: v for k, v in body.items() if k not in ("min_tokens", "ignore_eos", "stop")}, 360)
        results[i] = r

    threads = [threading.Thread(target=run, args=(i,)) for i in range(c)]  # all N fired at once (Promise.all)
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    ok = [r for r in results if not r["error"] and r["decodeTokens"] > 0]
    agg = 0
    if ok:
        window = max(r["tLast"] for r in ok) - min(r["tFirst"] for r in ok)
        agg = sum(r["decodeTokens"] for r in ok) / window * 1000 if window > 0 else 0
    mean = lambda xs: statistics.fmean(xs) if xs else 0
    return dict(concurrency=c, streamsOk=len(ok), aggregateDecodeTps=round2(agg),
                meanDecodeTps=round2(mean([r["decodeTps"] for r in ok])),
                medianDecodeTps=round2(statistics.median([r["decodeTps"] for r in ok])) if ok else 0,
                meanTtftMs=round2(mean([r["ttftMs"] for r in ok])),
                totalDecodeTokens=sum(r["decodeTokens"] for r in ok),
                errors=[r["error"] for r in results if r["error"]])


def run_decode(url, model, concs, max_tokens, kind):
    # One job = one warmup (32 tokens, no ignore_eos) then each level once, ascending, no repeats.
    stream(url, chat_body(model, CODE_WARMUP_PROMPT if kind == "code" else decode_prompts(1, kind)[0], 32), 360)
    waves = []
    for c in sorted(set(concs)):
        w = decode_wave(url, model, c, max_tokens, kind)
        # UI: "agg" falls back to meanDecodeTps when aggregate is 0 (BenchmarkDialog.tsx:118)
        print(f"decode {kind:5s} C{c:<2d} agg {w['aggregateDecodeTps'] or w['meanDecodeTps']:8.1f}  "
              f"str {w['meanDecodeTps']:7.1f}  TTFT {w['meanTtftMs']:7.0f} ms  ok {w['streamsOk']}/{c}"
              + (f"  err {w['errors'][0]}" if w["errors"] else ""), flush=True)
        waves.append(w)
    return waves


# ─── prefill (PrefillBench.js) ──────────────────────────────────────────────

FILLER_WORDS = ("the of and to in is you that it he was for on are as with his they at be this have from or one had "
                "by word but not what all were we when your can said there use an each which she do how their if will "
                "up other about out many then them these so some her would make like him into time").split(" ")


def filler_text(count):
    state, words = 0x2545F491, []
    for _ in range(count):
        state = (state * 1664525 + 1013904223) & M32
        words.append(FILLER_WORDS[(state >> 8) % len(FILLER_WORDS)])
    return " " + " ".join(words)


def prefill_prompt(target, salt):
    n = max(8, jround(target))
    header = f"[prefill-bench {salt}]\nIgnore the filler below. Reply with the single word OK.\n"
    footer = "\nReply OK."
    return header + filler_text(max(1, n - est_tokens(header + footer))) + footer


def prefill_sample(url, model, target):
    timeout_s = min(2700, max(90, 60 + target * 8 / 1000))
    r = stream(url, chat_body(model, prefill_prompt(target, uuid.uuid4()), 8), timeout_s)  # fresh salt = cache bust
    if not r["error"] and not (r["ttftMs"] > 0 or (r["serverPromptMs"] or 0) > 0):
        r["error"] = "No first token"
    return r


def prefill_rate(s, overhead):  # rateFromSample
    if (s["serverPromptMs"] or 0) > 0 and (s["serverPromptN"] or 0) > 0:
        return round2(s["serverPromptN"] / s["serverPromptMs"] * 1000)
    computed = max(0, s["prefillTokens"] - s["cached"])
    if s["ttftMs"] <= 0 or computed <= 0:
        return 0
    eff = max(s["ttftMs"] - max(0, overhead), s["ttftMs"] * 0.5)  # overhead correction capped at 50%
    return round2(computed / eff * 1000)


def run_prefill(url, model, sizes):
    prefill_sample(url, model, 512)
    prefill_sample(url, model, 4096)
    cal = [r["ttftMs"] for r in (prefill_sample(url, model, 16) for _ in range(3)) if not r["error"] and r["ttftMs"] > 0]
    overhead = round2(statistics.median(cal)) if cal else 0
    rows = []
    for size in sorted(set(sizes)):
        good = []
        for _ in range(3 if size <= 32768 else 2 if size <= 131072 else 1):
            s = prefill_sample(url, model, size)
            tps = prefill_rate(s, overhead) if not s["error"] else 0
            if tps <= 0:
                break  # a failing size stops its repeats
            good.append((tps, s))
        if good:
            good.sort(key=lambda g: g[0])
            tps, s = good[(len(good) - 1) >> 1]  # median rate, lower middle for even counts
            row = dict(targetTokens=size, promptTokens=s["prefillTokens"], prefillTps=tps, ttftMs=s["ttftMs"],
                       cachedTokens=s["cached"], samples=len(good), overheadMs=overhead,
                       lowConfidence=overhead > s["ttftMs"] * 0.3, error=None)
        else:
            row = dict(targetTokens=size, prefillTps=0, samples=0, error=s["error"] or "no prefill rate")
        print(f"prefill {size // 1024:>4d}k  tokens {row.get('promptTokens', 0):>8,d}  "
              f"{row['prefillTps']:>9,.0f} tok/s  overhead {overhead:.0f} ms" + (f"  err {row['error']}" if row["error"] else ""),
              flush=True)
        rows.append(row)
    return rows


# ─── quality (QualityBench.js + qualitySuite.js; default categories only) ───

def imul(a, b):
    return (a * b) & M32


def mulberry32(seed):
    a = seed & M32

    def nxt():
        nonlocal a
        a = (a + 0x6D2B79F5) & M32
        t = imul(a ^ (a >> 15), a | 1)
        t ^= (t + imul(t ^ (t >> 7), t | 61)) & M32
        return ((t ^ (t >> 14)) & M32) / 4294967296
    return nxt


def fnv1a(s):
    h = 0x811C9DC5
    for ch in s:
        h = imul(h ^ ord(ch), 0x01000193)
    return h


def rng_for(key):
    return mulberry32(fnv1a(f"sparkdash-quality-v1:{key}"))


def randint(rng, lo, hi):
    return lo + math.floor(rng() * (hi - lo + 1))


def pick(rng, xs):
    return xs[math.floor(rng() * len(xs))]


def shuffled(rng, xs):
    out = list(xs)
    for i in range(len(out) - 1, 0, -1):
        j = math.floor(rng() * (i + 1))
        out[i], out[j] = out[j], out[i]
    return out


QA_FIXED = [
    ("What is 17 * 23? Answer with the number only.", ["391"], 16),
    ("What is 1234 + 8766? Answer with the number only.", ["10000"], 16),
    ("What is 2 to the power of 20? Answer with the number only.", ["1048576", "1,048,576"], 16),
    ("A train travels 180 km in 2.5 hours. What is its average speed in km/h? Number only.", ["72"], 16),
    ("If x + 2x + 3x = 48, what is x? Number only.", ["8"], 16),
    ("What is the remainder when 1000 is divided by 7? Number only.", ["6"], 16),
    ("How many minutes are in 3.5 hours? Number only.", ["210"], 16),
    ("What is the square root of 1764? Number only.", ["42"], 16),
    ("What is 15% of 240? Number only.", ["36"], 16),
    ("A rectangle is 12 by 7. What is its area? Number only.", ["84"], 16),
    ("Reverse the string 'tensorfold'. Answer with the reversed string only.", ["dlofrosnet"], 16),
    ("How many letters 'r' are in the word 'strawberry'? Number only.", ["3"], 16),
    ("Sort these numbers ascending: 9, 2, 7, 4, 1. Answer as a comma-separated list.", ["1, 2, 4, 7, 9", "1,2,4,7,9"], 24),
    ("What comes next: 2, 6, 12, 20, 30, ? Number only.", ["42"], 16),
    ("All bloops are razzies and all razzies are lazzies. Are all bloops lazzies? Answer yes or no.", ["yes"], 8),
    ("If today is Wednesday, what day is it 10 days from now? One word.", ["saturday"], 8),
    ("What is the capital of Australia? One word.", ["canberra"], 8),
    ("What is the chemical symbol for gold? Symbol only.", ["au"], 8),
    ("Who wrote 'Pride and Prejudice'? Name only.", ["austen"], 12),
    ("What is the boiling point of water at sea level in degrees Celsius? Number only.", ["100"], 8),
    ("How many sides does a hexagon have? Number only.", ["6"], 8),
    ("What is the largest planet in our solar system? One word.", ["jupiter"], 8),
    ("In Python, what does len([1, [2, 3], 4]) return? Number only.", ["3"], 8),
    ("In Python, what is the value of 7 // 2? Number only.", ["3"], 8),
    ("What is the binary representation of 13? Digits only.", ["1101"], 12),
    ("What is 0.1 + 0.2 rounded to one decimal place? Number only.", ["0.3"], 8),
    ("Convert 5 kilometers to meters. Number only.", ["5000", "5,000"], 12),
    ("What is the greatest common divisor of 84 and 36? Number only.", ["12"], 8),
    ("Spell the word 'necessary' backwards. Letters only.", ["yrassecen"], 16),
    ("What is 999 * 999? Number only.", ["998001", "998,001"], 16),
]
COUNT_WORDS = ["mississippi", "bookkeeper", "banana", "committee", "parallel", "assessment", "successful",
               "occurrence", "tattoo", "referee", "balloon", "coffee", "letter", "address", "giraffe"]
REASON_NAMES = ["Alice", "Bruno", "Carmen", "Dmitri", "Elena", "Farid", "Grace", "Hiro",
                "Ines", "Jamal", "Keiko", "Liam", "Maya", "Noah", "Olga", "Pedro"]
REASON_THINGS = ["apples", "marbles", "stickers", "coins", "pencils", "cards", "shells", "stamps", "beads", "buttons"]
TRACK_PEOPLE = ["Ana", "Bo", "Cy", "Di", "Ed"]


def qa_item(iid, prompt, accept, mode=None, max_tokens=64):
    if not mode:
        a = accept[0]
        mode = ("number" if re.fullmatch(r"-?[\d,]*\.?\d+", a) else
                "list" if re.fullmatch(r"\d+\s*,\s*\d+(\s*,\s*\d+)*", a) else "word")
    return dict(id=iid, category="qa", prompt=prompt, accept=list(dict.fromkeys(accept)), mode=mode,
                maxTokens=max(max_tokens, 64), thinking=False)


def gen_qa():
    items = [qa_item(f"qa-fixed-{i + 1:02d}", q, a, None, m) for i, (q, a, m) in enumerate(QA_FIXED)]
    rng = rng_for("qa")
    for i in range(1, 21):
        a, b = randint(rng, 100, 999), randint(rng, 10, 99)
        items.append(qa_item(f"qa-mul-{i:02d}", f"What is {a} * {b}? Answer with the number only.", [str(a * b), f"{a * b:,}"]))
    for i in range(1, 16):
        a, b, c = randint(rng, 1000, 9999), randint(rng, 1000, 9999), randint(rng, 1000, 9999)
        items.append(qa_item(f"qa-addsub-{i:02d}", f"What is {a} + {b} - {c}? Answer with the number only.",
                             [str(a + b - c), f"{a + b - c:,}"]))
    for i in range(1, 16):
        s = "".join(chr(97 + randint(rng, 0, 25)) for _ in range(randint(rng, 6, 10)))
        items.append(qa_item(f"qa-rev-{i:02d}", f"Reverse the string '{s}'. Answer with the reversed string only.", [s[::-1]]))
    for i in range(1, 16):
        w = COUNT_WORDS[i - 1]
        ch = pick(rng, list(w))
        items.append(qa_item(f"qa-count-{i:02d}", f"How many times does the letter '{ch}' appear in the word '{w}'? Number only.",
                             [str(w.count(ch))]))
    for i in range(1, 16):
        nums = []
        while len(nums) < 7:
            n = randint(rng, 1, 99)
            if n not in nums:
                nums.append(n)
        srt = sorted(nums)
        items.append(qa_item(f"qa-sort-{i:02d}", f"Sort these numbers ascending: {', '.join(map(str, nums))}. Answer as a comma-separated list.",
                             [", ".join(map(str, srt)), ",".join(map(str, srt))]))
    used = set()
    for i in range(1, 11):
        n = randint(rng, 20, 255)
        while n in used:
            n = randint(rng, 20, 255)
        used.add(n)
        items.append(qa_item(f"qa-bin-{i:02d}", f"What is the binary representation of {n}? Digits only.", [bin(n)[2:]]))
    base = date(2020, 1, 1)
    span = (date(2030, 12, 31) - base).days
    for i in range(1, 16):
        d0 = base + timedelta(days=randint(rng, 0, span))
        k = randint(rng, 3, 60)
        items.append(qa_item(f"qa-date-{i:02d}", f"What date is {k} days after {d0.isoformat()}? Answer in YYYY-MM-DD format only.",
                             [(d0 + timedelta(days=k)).isoformat()], "date"))
    for i in range(1, 16):
        a, b = randint(rng, 2, 30), randint(rng, 2, 30)
        items.append(qa_item(f"qa-lcm-{i:02d}", f"What is the least common multiple of {a} and {b}? Number only.",
                             [str(a * b // math.gcd(a, b))]))
    return items


def gen_reason():
    rng, items = rng_for("reason"), []
    for i in range(1, 41):
        A, B, C = shuffled(rng, REASON_NAMES)[:3]
        th = pick(rng, REASON_THINGS)
        x, k, g, c = randint(rng, 20, 90), randint(rng, 2, 5), randint(rng, 3, 15), randint(rng, 5, 40)
        h = randint(rng, 2, x // 2)
        p = pick(rng, [2, 3, 4])
        total = x - h + (k * x - g) + 2 * (c + g)
        prompt = (f"{A} has {x} {th}. {B} has {k} times as many {th} as {A}. {C} has {c} {th}. "
                  f"{B} gives {g} {th} to {C}. Then {A} loses {h} {th}, and {C} doubles the number of {th} they have. "
                  f"Then all three put their {th} together and split them as evenly as possible into {p} boxes, "
                  f"putting any leftover {th} in a jar. How many {th} are in the jar, and how many are in each box? "
                  f"End your reply with a line 'Answer: <jar>, <per box>'.")
        items.append(dict(id=f"reason-{i:02d}", category="reason", prompt=prompt,
                          expected=(total % p, total // p), maxTokens=8192, thinking=True))
    return items


def gen_arith():
    rng, items = rng_for("arith"), []
    for i in range(1, 41):
        v0 = v = randint(rng, 100, 999)
        steps = []
        for s in range(1, 11):
            op = pick(rng, ["mul", "add", "sub", "mod", "div"])
            if op == "mul":
                k = randint(rng, 3, 19); v *= k; text = f"multiply it by {k}"
            elif op == "add":
                k = randint(rng, 100, 9999); v += k; text = f"add {k}"
            elif op == "sub":
                k = randint(rng, 10, max(11, v // 2)); v -= k; text = f"subtract {k}"
            elif op == "mod":
                k = randint(rng, 1000, 9999); v = int(math.fmod(v, k)) + 1000  # JS % truncates
                text = f"replace it by its remainder when divided by {k}, then add 1000"
            else:
                k = randint(rng, 2, 9); v = v // k; text = f"divide it by {k}, rounding down"
            steps.append(f"({s}) {text}")
        items.append(dict(id=f"arith-{i:02d}", category="arith", expected=v, maxTokens=12288, thinking=True,
                          prompt=f"Start with {v0}. Then, in order: {'; '.join(steps)}. What is the final number? "
                                 f"End your reply with a line 'Answer: <number>'."))
    return items


def gen_track():
    rng, items = rng_for("track"), []
    for i in range(1, 41):
        have, sent = {}, []
        for p in TRACK_PEOPLE:
            have[p] = randint(rng, 20, 60)
            sent.append(f"{p} starts with {have[p]} tokens.")
        for _ in range(25):
            a = pick(rng, TRACK_PEOPLE)
            b = pick(rng, TRACK_PEOPLE)
            while b == a:
                b = pick(rng, TRACK_PEOPLE)
            r = rng()
            if r < 0.6 and have[a] > 0:
                k = randint(rng, 1, max(1, have[a] // 2)); have[a] -= k; have[b] += k
                sent.append(f"{a} gives {k} to {b}.")
            elif r < 0.8:
                k = randint(rng, 1, 9); have[a] += k
                sent.append(f"{a} finds {k}.")
            else:
                g = have[a] // 2; have[a] -= g; have[b] += g
                sent.append(f"{a} gives half of their tokens (rounded down) to {b}.")
        who = pick(rng, TRACK_PEOPLE)
        items.append(dict(id=f"track-{i:02d}", category="track", expected=have[who], maxTokens=12288, thinking=True,
                          prompt=f"Track the tokens carefully. {' '.join(sent)} How many tokens does {who} have at the end? "
                                 f"End your reply with a line 'Answer: <number>'."))
    return items


def gen_gsm8k(data_dir):
    rows = json.load(open(f"{data_dir}/gsm8k-200.json"))
    return [dict(id=f"gsm8k-{r['n']:04d}", category="gsm8k", expected=r["a"], maxTokens=8192, thinking=True,
                 prompt=f"{r['q']}\n\nSolve it step by step. End your reply with a line 'Answer: <number>' containing only the final number.")
            for r in rows]


def gen_mmlu(data_dir):
    rows = json.load(open(f"{data_dir}/mmlu-285.json"))
    return [dict(id=f"mmlu-{r['n']:05d}", category="mmlu", expected=r["a"], maxTokens=1024, thinking=False,
                 prompt=f"The following is a multiple choice question about {r['s'].replace('_', ' ')}.\n\n{r['q']}\n"
                        + "\n".join(f"{'ABCD'[i]}. {c}" for i, c in enumerate(r["c"]))
                        + "\n\nReply with the letter of the correct answer. End your reply with a line 'Answer: <letter>'.")
            for r in rows]


def visible(text):
    s = re.sub(r"<think>[\s\S]*?</think>", "", text or "", flags=re.I)
    if re.search(r"</think>", s, re.I):
        s = s[s.lower().rfind("</think>") + 8:]
    o = s.lower().find("<think>")
    return (s[:o] if o >= 0 else s).strip()


def numbers_in(s):
    out = []
    for m in re.finditer(r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?", s):
        tok = m.group()
        if tok.startswith("-") and m.start() > 0 and re.match(r"\w", s[m.start() - 1], re.A):
            tok = tok[1:]
        out.append(float(tok.replace(",", "")))
    return out


NEG_BEFORE = {"not", "no", "never", "neither", "nor", "without", "except", "unlike", "than", "isn't", "wasn't", "aren't",
              "don't", "doesn't", "didn't", "can't", "cannot", "won't", "wouldn't", "shouldn't", "hardly"}
NEG_AFTER = {"isn't", "wasn't", "aren't", "wrong", "incorrect", "false", "untrue"}


def negated(tok, i, wants):
    if any(tok[k] in NEG_BEFORE and tok[k] not in wants for k in range(max(0, i - 2), i)):
        return True
    t1 = tok[i + 1] if i + 1 < len(tok) else None
    t2 = tok[i + 2] if i + 2 < len(tok) else None
    return bool(t1 in NEG_AFTER or (t1 in ("is", "was", "are") and t2 and (t2 == "not" or t2 in NEG_AFTER)))


def score_qa(item, reply):
    text, want = visible(reply), item["accept"]
    if item["mode"] == "number":
        nums = numbers_in(text)
        return bool(nums) and nums[-1] in [float(w.replace(",", "")) for w in want]
    if item["mode"] == "date":
        d = re.findall(r"\d{4}-\d{2}-\d{2}", text)
        return bool(d) and d[-1] == want[0]
    if item["mode"] == "list":
        nums = [int(x) for x in re.findall(r"\d+", want[0])]
        got = [int(x) for x in re.findall(r"\d+", text)]
        return len(got) >= len(nums) and got[-len(nums):] == nums
    tok = re.findall(r"[a-z0-9']+", text.lower())
    if not tok:
        return False
    wants = [w.lower() for w in want]
    clean = lambda i: tok[i] in wants and not negated(tok, i, wants)
    hits = [i for i, t in enumerate(tok) if t in wants]
    if len(tok) <= 15 and not re.search(r"\b(?:or|maybe|perhaps|either)\b|\d\s*/\s*\d", text, re.I) and hits and all(map(clean, hits)):
        return True
    bold = re.findall(r"\*\*([^*]+)\*\*", text)
    bt = re.findall(r"[a-z0-9']+", bold[-1].lower()) if bold else []
    return clean(len(tok) - 1) or (len(bt) == 1 and bt[0] in wants and not negated(bt, 0, wants))


WRAP = r"[*\s$€£`_]*"
NUM_TOKEN = r"-?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?"
SINGLE_NUM = r"-?(?:\d{1,3}(?:[,  ]\d{3})+|\d+)(?:\.\d+)?"
SUFFIX_WORD = r"(?!(?:or|and|to|vs|but|then|maybe|perhaps|is|are|was|not|no|wrong|incorrect)\b)[A-Za-z]+"
SINGLE_FULL = re.compile("^" + WRAP + "(" + SINGLE_NUM + ")" + WRAP + r"(?:%|\s*" + SUFFIX_WORD + "(?:[ /-]" + SUFFIX_WORD
                         + "){0,2})?" + WRAP + r"\.?" + WRAP + r"\Z", re.I)
PAIR_FULL = re.compile("^" + WRAP + "(" + NUM_TOKEN + ")" + WRAP + "," + WRAP + "(" + NUM_TOKEN + ")" + WRAP + r"\.?" + WRAP + r"\Z")
MMLU_LINE = re.compile(r"""\banswer\b[*\s]*(?::|is)[*\s"'`_]*(?:(?:option|choice)\s+)?(?:\(([ABCDabcd])\)|([ABCD])(?![A-Za-z0-9]))[*\s"'`_]*(.*)$""", re.I)
MMLU_HEDGE = re.compile(r"^[\s,;/&]*(?:or|and|/|&)\b|^\s*[,;/&]\s*(?:\(?[ABCDabcd]\)?(?![A-Za-z0-9])|but|though|however|also|maybe|perhaps)", re.I)


def answer_rest(text):
    m = re.findall(r"answer[*\s]*:[*\s$€£`]*([^\n]*)", visible(text), re.I)
    return m[-1] if m else None


def to_num(tok):
    return float(re.sub(r"[,  ]", "", tok))


def mmlu_letter(line):
    m = MMLU_LINE.search(line)
    if not m:
        return None
    paren, letter, rest = m.group(1) is not None, m.group(1) or m.group(2), m.group(3) or ""
    if (not paren and letter != letter.upper()) or MMLU_HEDGE.search(rest):
        return None
    if not paren and rest.strip() and not re.match(r"[.):,;\-–—!]", rest.lstrip()):
        return None
    return letter.upper()


def score(item, reply):
    cat = item["category"]
    if cat == "qa":
        return score_qa(item, reply), None
    if cat == "reason":
        r = answer_rest(reply)
        m = PAIR_FULL.search(r) if r is not None else None
        got = (to_num(m.group(1)), to_num(m.group(2))) if m else None
        return got == item["expected"], got
    if cat in ("arith", "track", "gsm8k"):
        r = answer_rest(reply)
        m = SINGLE_FULL.search(r) if r is not None else None
        got = to_num(m.group(1)) if m else None
        return got is not None and got == item["expected"], got
    lines = visible(reply).split("\n")  # mmlu
    lines = [l[:-1] if l.endswith("\r") else l for l in lines]
    got = None
    for l in lines:
        got = mmlu_letter(l) or got
    if not got:
        final = next((l for l in reversed(lines) if l.strip()), None)
        b = re.match(r"^[*\s(]*([ABCDabcd])[*\s).:]*\Z", final) if final else None
        got = b.group(1).upper() if b else None
    return got == item["expected"], got


def run_quality(url, model, data_dir, width=4):
    if not model:  # fetchModelInfo: first /v1/models id
        try:
            model = json.load(urllib.request.urlopen(url.rsplit("/chat/completions", 1)[0] + "/models", timeout=15))["data"][0]["id"]
        except Exception:  # noqa: BLE001
            pass
    gens = [("qa", gen_qa), ("reason", gen_reason), ("arith", gen_arith), ("track", gen_track),
            ("gsm8k", lambda: gen_gsm8k(data_dir)), ("mmlu", lambda: gen_mmlu(data_dir))]
    summary, rows = {}, []
    for cat, gen in gens:
        try:
            items = gen()
        except OSError as e:
            print(f"quality {cat}: skipped ({e}); pass --quality-data", flush=True)
            continue
        out, lock, it = [], threading.Lock(), iter(items)

        def worker():
            while True:
                with lock:
                    item = next(it, None)
                if item is None:
                    return
                body = chat_body(model, item["prompt"], item["maxTokens"], think=item["thinking"], seed=1234)
                r = stream(url, body, 120 + item["maxTokens"] * 0.15, think=item["thinking"])
                ok, parsed = (False, None) if r["error"] else score(item, r["answer"])  # scored on content, not reasoning
                with lock:
                    out.append(dict(id=item["id"], category=cat, ok=ok, parsed=parsed, error=r["error"],
                                    finishReason=r["finishReason"], completionTokens=r["completionTokens"]))

        ts = [threading.Thread(target=worker) for _ in range(min(width, len(items)))]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
        scored = [r for r in out if not r["error"]]  # request errors are excluded, not failures
        passed = sum(r["ok"] for r in scored)
        pct = jround(passed / len(scored) * 1000) / 10 if scored else None
        summary[cat] = dict(passed=passed, scored=len(scored), total=len(out), pct=pct,
                            errors=len(out) - len(scored), hitMaxTokens=sum(r["finishReason"] == "length" for r in out))
        print(f"quality {cat:6s} {passed:3d}/{len(scored):<3d} {pct if pct is not None else '-':>5}%  errors {len(out) - len(scored)}", flush=True)
        rows += sorted(out, key=lambda r: [i["id"] for i in items].index(r["id"]))
    pcts = [c["pct"] for c in summary.values() if c["pct"] is not None]
    overall = jround(statistics.fmean(pcts) * 10) / 10 if pcts else None  # mean of category percentages
    print(f"quality overall {overall}%", flush=True)
    return dict(categories=summary, overallPct=overall, items=rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--base-url", default="http://127.0.0.1:8080/v1")
    ap.add_argument("--model")
    ap.add_argument("--mode", choices=["decode", "prefill", "quality", "all"], default="all")
    ap.add_argument("--concurrency", default="1,2,4,8")
    ap.add_argument("--decode-types", default="prose,code", help="prose,code,structured")
    ap.add_argument("--max-tokens", type=int, default=400)  # INFERRED: the UI default (DEFAULT_MAX_TOKENS) was used
    ap.add_argument("--sizes", default="8192,16384,32768,65536,131072")  # INFERRED from the published 8k..128k rows
    ap.add_argument("--quality-data", default="data", help="dir holding gsm8k-200.json and mmlu-285.json")
    ap.add_argument("--out")
    a = ap.parse_args()
    url = a.base_url.rstrip("/") + "/chat/completions"
    res = {"baseUrl": a.base_url, "model": a.model}
    if a.mode in ("decode", "all"):
        concs = [int(x) for x in a.concurrency.split(",")]
        res["decode"] = {k: run_decode(url, a.model, concs, a.max_tokens, k) for k in a.decode_types.split(",")}
    if a.mode in ("prefill", "all"):
        res["prefill"] = run_prefill(url, a.model, [int(x) for x in a.sizes.split(",")])
    if a.mode in ("quality", "all"):
        res["quality"] = run_quality(url, a.model, a.quality_data)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
