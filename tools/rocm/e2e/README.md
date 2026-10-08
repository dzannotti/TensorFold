# ROCm end-to-end checks and bench

Python 3 stdlib clients for a running `tensorfold-native` (Qwen3.8 Flash Next). Each takes `--url` (or `TENSORFOLD_URL`,
default `http://127.0.0.1:8088`). Thinking is off in every request.

To reach thorim's CUDA server (production: light, sequential use only):

```bash
ssh -f -N -L 18088:localhost:8088 thorim      # thorim's server is now http://127.0.0.1:18088
```

## Files

| File | What it does |
| --- | --- |
| `tfapi.py` | shared client: posts, timed SSE chat, `/tokenize` rendering, `token_sha` (sha256 of the comma-joined ids, 12 hex) and its id lookup |
| `prompts.py` | the fixed prompt set: prose, code, math, JSON/table, counting, French, Chinese, two tool calls |
| `bench.py` | speed, reported like MiaAI-Lab's README tables |
| `contracts.py` | drafted == `"draft": false`, concurrent == solo, resumed == fresh, chunk invariance |
| `capture_ref.py` | greedy reference replies from the CUDA engine into `ref/` |
| `agreement.py` | teacher-forced top-1 agreement against `ref/thorim.json` |
| `ref/thorim.json` | 20 greedy replies from thorim (200 tokens max), prompt and reply token ids, all ids sha-verified |

## Bench

```bash
python3 bench.py --url http://127.0.0.1:8088 --label strix-rocm --json bench-strix.json
# short smoke run (what was run against thorim):
python3 bench.py --url http://127.0.0.1:18088 --prefill 512 --clients 1 --reps 1
```

Defaults: prefill 4k, 8k, 16k, 32k, 64k, 128k (one fresh prompt each, `--prefill-runs`), decode `prose` and `code` at
1, 2, 4, 8 requests, 256 tokens, `--reps 3` (median). Prints Mia's markdown tables; `--json` keeps every request.

The method, and where it comes from:

- Mia's current README tables (sparkDash on spark4): prefill, one request, greedy, thinking off; prefill speed is
  prompt tokens / time to first token (4,134 / 1.64 s = 2,526 tok/s, and so on down the table). Decode: greedy, thinking
  off; aggregate is every stream's tokens over earliest first token to latest last token, so one request's aggregate
  equals its per-request rate.
- Prompts: sparkDash's own, copied verbatim from Mia's `patches/0005-engine.patch` (`dash_prose`, `dash_code`, used by
  `bench-many dash-*`): prose is the hash-map explanation with `" (stream i/N)"` appended past one stream; code is one
  of 32 short-function tasks a stream ("a short Python function, 256 tokens, end-of-sequence ignored").
- Prefill text: Mia's `tools/bench.py` word list and sentence rule, a new seed every run, ending "Summarize the text above
  in one sentence.", 16 tokens out; sized with `/tokenize` like the Strix llama `bench/speed.py`. The bench stops if
  any prompt token came from the cache.
- Per request = (tokens - 1) / (last token - first token) of that stream; aggregate = the streams' tokens after their
  first / (latest last - earliest first). This is TensorFold's `tools/bench_concurrent.py` rule.

Where the sources differ (this bench follows the README tables, not the scripts):

| | Mia `tools/bench.py` | Strix llama `bench/speed.py` | Mia README (sparkDash) / this bench |
| --- | --- | --- | --- |
| prefill sizes | 8k-128k (8,000 ... words-estimated) | 8192-131072, `/tokenize`-sized, 3 runs under 64k | 4k-128k (4096 * 2^k), one run |
| prefill speed | server `prefill_s` | llama `prompt_per_second` | tokens / TTFT (server rate also in JSON) |
| decode prompts | quicksort (greedy), sky-is-blue (sampled) | same two; prose at server default sampling | hash-map prose and 32 code tasks, both greedy |
| end of sequence | stops | stops | ignored (256 tokens each) |
| concurrency | 1 only, median of 5 | barrier-started N; reps 5 at 1, 2 above | barrier-started N, `--reps` each |
| aggregate | - | tokens / (first start .. last end), TTFT included | tokens after first / (first token .. last token) |
| draft acceptance | - | reported | per request in JSON (`accepted`) |

Mia's 4k row is 4,134 prompt tokens; this bench's text is 4,096 tokens plus the 12-token chat frame (~4,110). TensorFold's
own `tools/bench_openai.py` (single stream, `ignore_eos`) and `tools/bench_concurrent.py URL MODEL --levels 1,2,4,8
--alone` give the same per-stream and aggregate rules on its own prompts; `--alone` also checks concurrent token_sha
against solo.

## Contracts

```bash
python3 contracts.py --url http://127.0.0.1:8088                       # a, b, c, d; ~60 requests, 6 x 20k prefill
python3 contracts.py --only a,b --cases prose-sky,tool-weather --concurrency 4 --max-tokens 96   # quick
python3 contracts.py --only d --long 40000
```

Every comparison is the reply's `token_sha` (all reply token ids, end token included) plus `finish_reason`.

- a: each prompt greedy and sampled (T 0.8, top-p 0.95, top-k 20, seed) with drafts vs `"draft": false`. UNCHECKED when
  no draft was accepted.
- b: the same requests `--concurrency` at a time vs their solo runs.
- c: a fresh-nonce ~4.8k-token system prompt; 3 canned turns and 3 questions sharing it. Fresh baselines go to
  `/v1/completions` as token ids first: an id prompt keeps no prompt state (checked on thorim: two id sends of a
  4,170-token prompt both `cached_tokens` 0, a chat send then 0, the next chat 4,163), and a chat reply's token_sha equals
  the id completion's for the same ids (checked on thorim, tool call included). Then the chats resume (cached > 0).
  Against a `--prompt-cache-gib 0` server the resumed cases come out UNCHECKED.
- d: a `--long`-token prompt fresh solo vs prefilled beside a second long prompt, vs chat, vs chat again (resumed).

Exit 0 pass, 1 fail, 2 could not run, 3 unchecked.

## Agreement with the CUDA engine

```bash
python3 capture_ref.py --url http://127.0.0.1:18088 --pause 3          # thorim, sequential; aborts on error or > 60 s
python3 agreement.py --url http://127.0.0.1:8088 --json agreement-strix.json
python3 agreement.py --url http://127.0.0.1:18088 --names json-book --max-positions 8   # smoke (100% on thorim)
```

Capture renders each prompt with `/tokenize` (chat template, tools, thinking off), sends the ids to `/v1/completions`
greedy, and keeps the reply ids (re-tokenized text, plus the end token when it stopped) only when they hash to the
server's `token_sha`. Agreement sends, for each position k, prompt + reference[:k] with `max_tokens` 1 and reads the
drawn id from the one-token `token_sha`; it reports per-prompt agreement, the divergent positions (with both ids) and the
free-running reply's first divergence. `--min-rate` (default 0.98) sets the exit code; fp8 KV alone costs ~1.2% top-1
against bf16 (Mia's README), so read divergences, not just the rate.
