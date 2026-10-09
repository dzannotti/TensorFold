# Long-context decode on gfx1151 (perf-longctx, 2026-10-09)

Question: decode at ~15-20k context ran 12.6 rounds/s (78 ms a round) vs 18.9 on thorim, ~40 ms at short context.

## 1. The indexer and sparse attention kernels are not it (microbench)

`tools/rocm/qsa_bench.py`: one attention layer's decode launches as the engine makes them in a decode graph (keys =
the graph's context bucket, a power of two >= 8192), fp8 KV, torch.profiler device us, best of 3 medians. The GPU was
shared (the user's server and another agent): spikes of 3-6x on single cells are noise.

| ctx (bucket) | rows | _pool | _scores | _select | _chunks8 | _merge | sum |
|---|---:|---:|---:|---:|---:|---:|---:|
| 2k (8192, dense) | 1 / 4 / 16 | 2.6 / 2.9 / 2.5 | 0.7 / 0.8 / 1.2 | 1.0 / 1.0 / 1.1 | 57 / 63 / 144 | 2-4 | 63 / 71 / 153 |
| 16k (16384) | 1 / 4 / 16 | 3-4 | 11 / 8 / 34 | 20 / 11 / 36 | 364* / 69 / 443 | 4-8 | 404* / 94 / 525 |
| 32k (32768) | 1 / 4 / 16 | 3-4 | 6 / 17 / 77 | 17 / 16 / 58 | 58 / 69 / 560 | 2-20 | 86 / 109 / 718 |
| 64k (65536) | 1 / 4 / 16 | 2-3 | 11 / 26 / 88 | 33 / 28 / 43 | 58 / 68 / 505 | 2-9 | 107 / 128 / 649 |
| 128k (131072) | 1 / 4 / 16 | 2-4 | 21 / 105* / 169 | 57 / 195* / 72 | 56 / 364* / 246 | 2-5 | 139 / 672* / 496 |

(* noisy cells.) `_select` (spilling variants included) is <= ~60 us a launch up to 128k; `_scores` grows linearly
(~1.3 us per 1k blocks a row at 16 rows). At 20k (bucket 32768) vs 2k the indexer + attention costs ~+40 us a layer at
4 rows and ~+0.4-0.5 ms at 16 rows (`_chunks8` gathers 16 rows' distinct 2048-key lists): 13 layers (12 + MTP) =
~0.5 ms a round at typical windows, ~5-7 ms at a 16-row window. Not the 38 ms gap; no kernel was changed and the AOT set
is unchanged (aot-final).

## 2. Full model: context is not the factor; the n-gram table's page faults are

Full model, the served flags, PM4 runtime, GPU shared with another agent (relative numbers). `tools/rocm/e2e/longctx.py`
(a fixed ~N-token prefix prefilled once, measured requests = prefix + a short varied turn resuming it from the prompt
cache, 300 tokens, EOS ignored; 3 reps, median):

| ctx | mode | rocm a6b3077: tok/s, rounds/s, ms/round, tok/round | perf-longctx: tok/s, rounds/s, ms/round, tok/round |
|---|---|---|---|
| 2k | sampled T1, thinking | 51.2, 19.98, 50.0, 2.59 | 54.2, 21.57, 46.4, 2.50 |
| 2k | greedy, no thinking | 73.9, 21.17, 47.2, 3.30 (79.3 ms at 5.77 t/r) | 99.5, 18.31, 54.6, 5.36 (60.6 ms at 6.00 t/r) |
| 20k | sampled T1, thinking | 47.3, 19.15, 52.2, 2.46 | 50.4, 21.56, 46.4, 2.29 |
| 20k | greedy, no thinking | 72.1, 15.54, 64.4, 3.41 (70.2 ms at 5.88 t/r) | 91.3, 17.51, 57.1, 5.77 (72.7 ms at 6.52 t/r) |

Two concurrent sampled thinking requests at 20k (perf-longctx): 38-41 tok/s each, 15.5-17 rounds/s, 2.4 t/r.
An earlier fresh-prompt run (old longctx: 256 vs 20k tokens, thinking on): ms/round 55.5 / 53.5 greedy and 55.6 / 60.5
sampled on rocm, 45-49 ms on a repeat of the same prompt. Context length moves a round by a few ms; what moves it is
whether the round's tokens were seen before and how many rows the window has.

Cause: the FP8 n-gram table (48.7 GiB, `ple-table/`) is memory-mapped on the host (one rank) and every decode window
gathers `rows x 16` rows from it on the host, one at a time (`NgramTable.gatherSome`; prompts >= 1024 lookups use
threads). With ~66 GiB of weights in GTT plus prod, the page cache keeps ~3 GiB of it (`fincore`: 2.85 of 48.7 GiB
resident), so a new n-gram is a disk read: measured 2.4-4.5 ms per cold row serially (MADV_RANDOM, as the engine maps
it). Drafted rows bring new n-grams every round, so a 16-row window can stall the host for tens of ms before its
launches; a repeated prompt (warm pages) runs 7 ms/round vs 13.7 on a 4-layer view (`tfdev-longctx-v4`, rows 1),
`--no-drafts` shows no gap. Prefill shows it too: `stage` (host staging incl. the gather) 200-780 ms a 2048-row chunk
cold vs 2-3 ms warm (TF_FLASHNEXT_PROFILE, CLI bench, p20k), ~2.5 s of a 22 s 20k prefill.

Fix (cuda_weights.zig `NgramTable.prefetch`): before a decode window's serial gather, MADV_WILLNEED every row's pages
so the kernel reads them in parallel. Python probe of the same: 16 cold rows 38.8 ms serial -> 7.9 ms; 64 rows 172 ->
14.9 ms. Bits unchanged (a hint; the gather is the same). Measured above: high-acceptance rounds 79 -> 61 ms (2k) and
70 -> 57-73 ms (20k) at ~6 t/r, sampled rounds/s +8-12%. GB10 has the table in a page cache that fits (128 GB, no GTT
weights in RAM besides), so thorim never pays this.

## Left / next
- The gather is still synchronous on the host before the window's launches: a per-row cold read still costs ~1 IO
  latency (~2-8 ms) when anything is cold. Options: gather on threads for decode too, overlap the next round's
  candidate n-grams (drafts are known before verify), or pin the hot part of the table (resident set by frequency).
- The user's ~27 tok/s at ~20k with thinking and sampling vs thorim's ~60 was not reproduced in isolation (50 tok/s
  single, ~40 per stream at 2 streams here): the live server ran concurrent requests with fresh content; re-measure on
  the real harness with this build.
- Sampled drafting accepts fewer tokens a round (2.2-2.6 vs 3.3-5.8 greedy), so sampled decode is round-bound.
