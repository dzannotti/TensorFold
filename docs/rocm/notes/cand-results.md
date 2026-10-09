# rocm-cand on the full model (2026-10-09, prod down, GPU exclusive per phase)

Candidate: `rocm-cand` @ 8eed266 = rocm-next 7ec5235 + perf-attn (`_chunks8` rewrite) + perf-fp8dec (nontemporal fp8
decode loads) + perf-engine (draw batching, argmax, n-gram MADV_RANDOM + parallel gather, TF_FLASHNEXT_VMM).
Triton set `aot-cand` (rebuilt from this tree, both specs, 830 hsaco, both pointer forms: 2,163 range32 params like
aot-hip3; only `_chunks8`'s 8 hashes differ from aot-hip3). Baseline: rocm-next 7ec5235 + aot-hip3.
Serve: `--context 262144 --parallel 8 --kv-dtype fp8 --thinking`, TF_FLASHNEXT_DEPTH=15. Raw logs: `~/tf-window/cand/`.

## Build and static checks
- `zig build -Dgpu=hip native install`: hipcc misses only the expected 8 (qmm_group, prefill_attention,
  qmm_prefill, experts_prefill, fn_qmm, fn_qmm_prefill, fn_qmm_cluster, fn_roce).
- `zig build test -Dgpu=hip -Daot-set=aot-cand`: 21/21 steps. `triton_parity.py hash`: 908/908 equal.

## Correctness
- CLI sky (fp8 KV, 102 tokens): drafted == `--no-drafts`, engine sha de8d7a445fe7, accepted 51 of 69; the same sha
  as rocm-next on the full model (perf-engine notes), i.e. identical tokens. Load 49 s, 65.6 GiB.
- Server: `contracts.py` (a-d): 54 pass, 0 fail, 6 unchecked (as rocm before).
- `agreement.py` vs thorim, same session protocol, same client:

| build | positions agree | free-run identical |
|---|---:|---:|
| rocm-next + aot-hip3 | 3076/3101 (99.19%) | 7/20 |
| rocm-cand + aot-cand | 3071/3101 (99.03%) | 8/20 |

  Divergent positions: 18 in both, 12 only in cand, 7 only in next (two-sided sign test p ~0.36): near-tie flips
  from `_chunks8`'s changed softmax order (1-ulp bf16 differences, attn-kv8.md), not a systematic loss.
- Decode bench tokens differ from rocm-next in 15/18 streams for the same reason; drafts accepted are similar
  (prose x1 155 = 155, code x1 217 vs 218, prose x8 1232 vs 1179, code x8 1630 vs 1623).

## Speed: A/B (bench.py, same client; order cand full -> next quick -> cand quick, each after an 8k warm-up)

| | rocm-next | rocm-cand (full run) | rocm-cand (quick) | x (quick) | thorim (GB10, our harness) |
|---|---:|---:|---:|---:|---:|
| first prefill after load (8k warm-up) | 319 tok/s (25.7 s) | 899 tok/s (9.1 s) | 974 tok/s (8.4 s) | 3.1 | |
| prefill 8k | 544 tok/s | 994 | 1,044 | 1.92 | 635 |
| prefill 32k | 511 tok/s | 1,076 | 1,100 | 2.15 | |
| prose x1 | 51.2 tok/s | 55.4 | 55.4 | 1.08 | 65.2 |
| prose x8 (aggregate) | 155.7 | 188.5 | 178.6 | 1.15 | |
| code x1 | 93.4 | 105.5 | 106.1 | 1.14 | 122.7 |
| code x8 (aggregate) | 202.1 | 241.7 | 246.9 | 1.22 | |

prose x8: cand quick accepted fewer drafts (1179 vs 1232; 874 vs 818 streamed pieces, i.e. more rounds), so its gain is partly hidden.

## Speed: rocm-cand full bench (Mia's method) beside Mia's GB10 README

**Prefill** (one request; tokens / TTFT)

| Prompt | Tokens | rocm-cand | TTFT | rocm (window 1) | Mia GB10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4k | 4,110 | 974 tok/s | 4.22 s | 276 | 2,526 |
| 8k | 8,203 | 994 tok/s | 8.25 s | 464 | 2,606 |
| 16k | 16,397 | 1,064 tok/s | 15.41 s | 479 | 2,643 |
| 32k | 32,780 | 1,076 tok/s | 30.48 s | 475 | 2,630 |
| 64k | 65,555 | 1,062 tok/s | 61.73 s | 443 | 2,564 |
| 128k | 131,091 | 1,032 tok/s | 127.00 s | 460 | 2,415 |

**Decode, prose** (greedy, thinking off, 256 tokens, EOS ignored, median of 3; aggregate tok/s)

| Requests | rocm-cand agg | per request | TTFT | rocm (window 1) | Mia GB10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 55.4 | 55.4 | 105 ms | 48.6 | 64.4 |
| 2 | 81.4 | 41.8 | 213 ms | 75.7 | 89.0 |
| 4 | 139.0 | 36.1 | 372 ms | 111.1 | 140.7 |
| 8 | 188.5 | 24.7 | 491 ms | 157.5 | 200.9 |

**Decode, code**

| Requests | rocm-cand agg | per request | TTFT | rocm (window 1) | Mia GB10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 105.5 | 105.5 | 136 ms | 91.3 | 57.5 |
| 2 | 129.3 | 66.6 | 233 ms | 109.3 | 92.7 |
| 4 | 180.0 | 47.5 | 509 ms | 149.8 | 130.8 |
| 8 | 241.7 | 34.9 | 656 ms | 199.6 | 189.3 |

Thorim calibration (same engine on GB10, our harness): prose 65.2 / dash-code 122.7 tok/s at 1 request, prefill 8k
635 tok/s. Cand is at 85% / 86% of thorim's decode and 1.6x thorim's 8k prefill.

# Re-run on rocm-cand @ 5074bde (+ perf-pool, + perf-hc)

Triton set `aot-final` rebuilt from 5074bde (988 kernels, 2,445 range32 params; hash set equal to perf-hc's aot-hc7).
Previous cand binary (8eed266 + aot-cand) kept in `~/tf-window/cand/oldbin` for the A/B. Logs: `~/tf-window/cand2/`.

- hipcc misses the same 8; `zig build test -Dgpu=hip -Daot-set=aot-final`: 21/21 steps, 104/104 tests;
  `triton_parity.py hash`: 1100/1100 equal.
- CLI sky: drafted == plain, sha de8d7a445fe7 (= 8eed266 = rocm-next), accepted 51 of 69.
- contracts a-d: 54 pass, 0 fail, 6 unchecked. agreement: 3071/3101 (99.03%), 8/20 free-run identical: the same
  count as 8eed266.
- Same bits in the server too: every decode stream's token_sha (54 of 54: prose/code x1/x8, 3 reps) is equal across
  both old runs and all three new ones. Drafts accepted at x8 vary a little run to run in both builds (scheduling).

## A/B quick bench (order: new full, new quick, old quick, new quick; each session after an 8k warm-up)

| | 8eed266 (2 runs) | 5074bde (3 runs) | x (medians) | thorim |
|---|---|---|---:|---:|
| first 8k prefill (warm-up) | 974, 917 | 1,005, 972, 989 | 1.04 | |
| prefill 8k | 1,044, 967 | 1,079, 1,063, 1,082 | 1.07 | 635 |
| prefill 32k | 1,100, 1,048 | 1,187, 1,149, 1,117 | 1.08 | |
| prose x1 | 55.4, 55.4 | 57.1, 57.1, 57.7 | 1.03 | 65.2 |
| prose x8 agg | 178.6, 161.3 | 197.1, 188.4, 177.9 | ~1.1 (noisy) | |
| code x1 | 106.1, 105.6 | 109.4, 108.7, 109.9 | 1.03 | 122.7 |
| code x8 agg | 246.9, 247.8 | 254.0, 259.2, 261.9 | 1.05 | |

The old binary's second prose x8 (161.3) is an outlier with normal acceptance (2,475 pieces vs 2,439-2,580), i.e.
timing noise; prose x8 swings ~10% between sessions in both builds.

## Full bench (Mia's method) on 5074bde

| Prefill | Tokens | 5074bde | TTFT | 8eed266 | Mia GB10 |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 4k | 4,092 | 1,077 tok/s | 3.80 s | 974 | 2,526 |
| 8k | 8,216 | 1,079 tok/s | 7.61 s | 994 | 2,606 |
| 16k | 16,394 | 1,096 tok/s | 14.96 s | 1,064 | 2,643 |
| 32k | 32,776 | 1,187 tok/s | 27.60 s | 1,076 | 2,630 |
| 64k | 65,533 | 1,124 tok/s | 58.32 s | 1,062 | 2,564 |
| 128k | 131,093 | 1,115 tok/s | 117.59 s | 1,032 | 2,415 |

| Requests | prose agg (per req, TTFT) | 8eed266 | Mia | code agg (per req, TTFT) | 8eed266 | Mia |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 57.1 (57.1, 98 ms) | 55.4 | 64.4 | 109.4 (109.4, 128 ms) | 105.5 | 57.5 |
| 2 | 85.6 (42.8, 260 ms) | 81.4 | 89.0 | 130.3 (69.4, 324 ms) | 129.3 | 92.7 |
| 4 | 146.6 (38.0, 329 ms) | 139.0 | 140.7 | 175.4 (50.3, 436 ms) | 180.0 | 130.8 |
| 8 | 197.1 (26.1, 469 ms) | 188.5 | 200.9 | 254.0 (36.7, 662 ms) | 241.7 | 189.3 |

vs thorim (65.2 / 122.7 / 635): prose x1 88%, code x1 89%, prefill 8k 1.7x. Prose matches Mia's GB10 at 4 requests and
is within 2% at 8; code beats it at every level.
