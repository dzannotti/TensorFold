# Window 2 (2026-10-09 06:00-10:05 BST, prod down, GPU exclusive): draft policy, re-profile, PM4 hang

Binary: rocm-cand 5074bde (`/home/dzannotti/tf-cand/zig-out`, same code as rocm 7ed5fc7), Triton set `aot-final`.
Raw logs: `~/tf-window/w2/` (scripts `sweep.sh`, `p2.sh`, `p3.sh`, `tools/`).

## 1. Draft policy sweep (no code changes)

`tensorfold bench-many $M --streams 8 --kind dash-prose,dash-code --rows 1,8 --reps 2 --max-tokens 256 --ignore-eos
--depth D --served --kv-dtype fp8`, one process a setting, rep 1 (rep 0 pays the captures). Aggregate decode tok/s;
in brackets tokens a stream-round and verify + drafting ms a round (for the first column). Default = depth 15,
TF_FLASHNEXT_CONFIDENCE=-0.4 (running product >= 0.4), TF_FLASHNEXT_PRODUCT_STREAMS=2 (above 2 live streams Python's
per-draft rule at the hard-coded `wide_confidence` 0.5). Two values = two runs (the very first run of the window,
d15's 56.8 / 217.6, was slow; its repeat and every other setting agree to <1%).

Sky (CLI `run`, 102 tokens, fp8 KV, `--depth D`): `--no-drafts` sha de8d7a445fe7; every drafted setting checked
(d4, d8, d15, c06, c02, c0, c0d6, p0c05, p0c02) gives the same sha de8d7a445fe7 (accepted/drafted in the last column).
Note: CLI `run` takes `--depth` (default 6), not TF_FLASHNEXT_DEPTH; earlier "accepted 51 of 69" sky runs were depth 6.

| setting | prose x1 | code x1 | prose x8 | code x8 | sky sha (acc/drafted) |
|---|---:|---:|---:|---:|---|
| d15 (default) | 59.0/56.8 (2.55 t/r, 38+5 ms) | 113.0/113.3 (6.71 t/r, 45+14 ms) | 226.4/217.6 (2.72) | 281.7/284.9 (5.28) | de8d7a445fe7 (51/69) |
| d4 | 56.9 (2.43) | 89.7 (4.40) | 226.4 (2.61) | 285.2 (4.10) | de8d7a445fe7 (51/69) |
| d6 | 58.9 (2.55) | 102.4 (5.80) | 225.6 (2.71) | 285.3 (4.68) | |
| d8 | 59.0/59.0 (2.55) | 111.2/111.3 (6.38) | 226.6/226.7 (2.71) | 294.1/294.2 (5.04) | de8d7a445fe7 (51/69) |
| d10 | 58.9 | 111.5 (6.54) | 227.0 | 290.1 (5.10) | |
| d12 | 58.8 | 112.7 (6.71) | 226.5 | 287.4 (5.26) | |
| c06 (product 0.6) | 56.3 (2.38) | 100.1 (5.43) | 226.5 | 281.5 | de8d7a445fe7 (49/66) |
| c05 | 57.4 (2.48) | 107.3 (6.07) | 226.3 | 280.5 | |
| c03 | 63.8 (2.93) | 111.2 (6.89) | 226.4 | 287.5 | |
| c025 | 65.4 (3.04) | 115.7 (7.29) | 226.4 | 286.9 | |
| c02 | 68.0 (3.19, 39+7 ms) | 117.3 (7.50) | 226.2 | 286.2 | de8d7a445fe7 (54/73) |
| **c01** | **71.5 (3.45, 40+8 ms)** | **119.3 (7.97, 48+18 ms)** | 226.2 | 288.8 | |
| c005 | 68.2 (3.49) | 114.4 (7.97) | 227.2 | 288.3 | |
| c0 (every draft, depth 15) | 47.8 (3.98, 55+28 ms) | 100.4 (8.50) | 150.4 (3.87) | 243.1 (7.16) | de8d7a445fe7 (58/592) |
| c0, depth 6 | 67.4 (3.81) | 102.7 (5.93) | 136.9 (3.72) | 309.1 (5.48) | de8d7a445fe7 (58/249) |
| PRODUCT_STREAMS=8 (product 0.4 at x8) | 58.9 | 113.4 | 223.0 (2.66) | 279.5 (5.09) | |
| PRODUCT_STREAMS=8, product 0.2 | 68.1 | 117.3 | 237.7 (3.07) | 290.2 (5.88) | |
| **PRODUCT_STREAMS=8, product 0.1** | **71.5** | **119.4** | **242.3 (3.26, 83+20 ms)** | **303.8 (6.28, 101+30 ms)** | |
| per-draft 0.7 everywhere (PS=0) | 57.1 (2.38) | 106.3 (5.80) | 216.3 (2.44) | 273.4 (4.52) | |
| per-draft 0.5 everywhere | 59.4 (2.63) | 113.3 (6.89) | 226.2 | 288.6 | de8d7a445fe7 (50/68) |
| per-draft 0.3 everywhere | 67.5 (3.27) | 111.1 (7.73) | 228.9 (3.14) | 280.4 (6.20) | |
| per-draft 0.2 everywhere | 66.8 (3.49) | 106.8 (8.23) | 223.6 (3.37) | 285.7 (6.58) | de8d7a445fe7 (55/84) |
| per-draft 0.1 everywhere | 68.0 (3.92) | 108.5 (8.23) | 212.2 (3.68) | 271.2 (6.89) | |

Reading:
- On gfx1151 a verify window's extra rows are nearly free (prose x1 verify 38 ms at 2.55 tokens a round, 40 ms at
  3.45: weights dominate), so the GB10-tuned product threshold 0.4 stops chains too early. Product >= 0.1 is the
  best at one stream: prose x1 59.0 -> 71.5 (+21%, above thorim's 65.2), code x1 113 -> 119.4 (+5.6%). 0.05 is past
  the optimum (drafting cost grows faster than acceptance), every-draft (c0) is far worse.
- At 8 streams the per-draft 0.5 rule (wide) is flat across depth; the running product at 0.1 also wins there:
  prose x8 226 -> 242 (+7%), code x8 282-294 -> 304 (+3-8%). Depth matters only for code (d15 >= d12 > d8 at x1;
  d8 slightly best at code x8 with the old rule), and d15 is best with the product 0.1 rule.
- **Recommended HIP defaults**: depth 15, running product 0.1 at every stream count (served_confidence -0.1 and
  product_streams >= --parallel). Today, as env: `TF_FLASHNEXT_DEPTH=15 TF_FLASHNEXT_CONFIDENCE=-0.1
  TF_FLASHNEXT_PRODUCT_STREAMS=8`. In code: a per-backend served_confidence / product_streams_default in cuda_engine.zig
  (HIP: -0.1 and maxInt). The engine already picks the rule by live stream count (lanes.confidenceNow: product while
  live <= product_streams, else wide_confidence); depth is engine-wide (graph sizes), so a per-stream-count depth
  would be a cap on the chain length in confidenceNow's caller (draft_stops_most), not a depth change. Not needed
  here: with product 0.1 depth 15 wins at x1 and x8. At 2 and 4 streams (bench-many --rows 2,4, rep 1) product 0.1 everywhere also wins: prose x2 87.1 -> 95.8, x4
  151.6 -> 161.1; code x2 145.3 -> 160.7, x4 197.0 -> 216.5 (+6-11%).
- The x8 samples are one run each; the d15/d8 repeats reproduced to 0.1%, so the steady CLI numbers are tight.

## 2. Re-profile (5074bde + aot-final; default draft rule, depth 15, fp8 KV)

`p2.sh`: TF_FLASHNEXT_PROFILE=1 bench-many prose x1 and CLI `bench` on the 8k prompt (perf/p8k.ids, 8,212 tokens), then
the same under `rocprofv3 --kernel-trace` with `--eager`. Eager under rocprofv3: prose x1 57.7 tok/s (graphed 59.0);
8k prefill 1,377 tok/s (CLI, no server; the server measures ~1,080 on the same build: its bench sends fresh
prompts through the prompt-cache/state path, see cand-results.md).

**Decode, prose x1** (part profiler: 46.3 ms a round eager = main 40.8 + MTP 5.6; rocprofv3 window: 44.2 ms a round,
GPU busy 40.2 ms = 91%, 1,629 dispatches a round). Parts (ms a round): gdn_proj 8.53, experts_gate_up 5.87, out_proj
4.07, hc_up 3.61, hc_down 3.60, experts_down 3.17, attn_proj 2.47, gdn_chain 1.52, head 1.40, router 1.03 (was 2.37),
attention 0.66 (was 3.00), MTP head 5.56 (absorb 2.16, levels 1.72/0.87/0.46/...).

| kernel | ms/round | % of kernel time | launches/round | us/launch |
|---|---:|---:|---:|---:|
| `void tf_fn_qmmf::qmmf_kernel<3, 16, 64, 1, 4, 4, false, fals` | 11.45 | 23.6 | 144.0 | 79.5 |
| `void tf_fn_qmmf_ld::qmmf_kernel<3, 16, 64, 1, 4, 4, false, f` | 9.94 | 20.5 | 48.0 | 207.1 |
| `_b16mm` | 6.50 | 13.4 | 166.2 | 39.1 |
| `tf_int4_128_1_1_1_3` | 5.55 | 11.4 | 51.7 | 107.2 |
| `tf_int4_128_1_1_2_2` | 5.39 | 11.1 | 48.0 | 112.3 |
| `_b16mm_sm` | 3.48 | 7.2 | 105.2 | 33.1 |
| `void tf_fn_gdn_tree::tree_kernel<float, 0, 8, 4, true>` | 1.33 | 2.7 | 36.0 | 37.0 |
| `_router` | 0.88 | 1.8 | 50.7 | 17.3 |
| `_chunks8` | 0.55 | 1.1 | 14.7 | 37.6 |
| `_fp4mm` | 0.41 | 0.8 | 5.5 | 75.0 |

- Dense block-FP8 (`qmmf` + `qmmf_ld`) 21.4 ms + bf16 (`_b16mm`/`_b16mm_sm`: hc mixes + MTP projections) 10.0 ms =
  31.4 ms for ~4.6 GiB of dense weights a round: ~157 GB/s average vs the 242 GB/s streaming floor (20.5 ms). This
  is still the biggest lever at x1: ~11 ms a round (25%) if all dense reads ran at the floor. Per kernel: qmmf_ld
  (fused GDN in_proj, ~40 MB) 207 us = ~195 GB/s; qmmf 80 us per launch (small out/attn matrices: launch/tail bound);
  `_b16mm` 39 us for a 6.5 MB hc mix = ~165 GB/s.
- int4 experts 10.9 ms (2.55 rows a round x 8 experts x 48 layers; weight-bound per distinct expert).
- Router (0.88 ms, was 2.29) and attention (`_chunks8` 0.55 ms, was 3.20) are fixed. Glue (`_hc_*`, strided copies,
  `_reduce`): ~1.5 ms; gaps (9% idle) ~4 ms: ~3 us between dependent kernels (engine-perf.md) = launch count.

**Prefill 8k** (part profiler warm rep: 1,477 ms a 2048-row chunk = 1,387 rows/s; rocprofv3 window 1,484 ms a
chunk, busy 99.2%, 2,101 dispatches a chunk). Parts (ms a chunk): experts_gate_up 179, attention 174 (was 1,511-2,940),
gdn_proj 169, experts_down 146, hc_down 112, hc_writeback 106, out_proj 106, gdn_back 97, gdn_chain 84, hc_mix 80,
gdn_front 68, attn_proj 48.

| kernel | ms/chunk | % of kernel time | launches/chunk | us/launch |
|---|---:|---:|---:|---:|
| `void tf_fn_qmmf::qmmw_kernel<128, 64, false>` | 396.60 | 22.2 | 143.6 | 2761.1 |
| `void tf_fn_qmmf_ld::qmmw_kernel<128, 128, false>` | 213.12 | 11.9 | 47.9 | 4451.0 |
| `tf_int4_128_2_2_2_2` | 178.94 | 10.0 | 47.9 | 3737.2 |
| `_chunks8` | 177.97 | 10.0 | 107.2 | 1659.7 |
| `tf_int4_128_2_2_1_3` | 146.30 | 8.2 | 47.9 | 3055.6 |
| `_b16mm_ks_sm` | 113.74 | 6.4 | 97.5 | 1166.5 |
| `_hc_wb_norm` | 103.98 | 5.8 | 97.5 | 1066.4 |
| `void tf_fn_gdn_io::back_kernel<16, 48>` | 95.55 | 5.4 | 35.9 | 2660.7 |
| `void tf_fn_gdn_prefill::chain_kernel<float, 128, 32>` | 84.18 | 4.7 | 35.9 | 2344.2 |
| `_hc_up_mix` | 80.30 | 4.5 | 97.5 | 823.5 |

- Block-FP8 matmuls (`qmmw` WMMA tiles) 610 ms for ~10 TFLOP a chunk = ~16 TF; int4 experts 325 ms for 4.8 TFLOP =
  ~15 TF; the bf16 hc mixes (`_b16mm_ks_sm` + `_hc_wb_norm` + `_hc_up_mix`) 298 ms for 2.6 TFLOP = ~9 TF; attention
  (`_chunks8` + `_merge`) 190 ms (~7 TF); GDN conv/gates/chain 248 ms (memory/latency bound). gfx1151's dense bf16
  WMMA peak is ~59 TF (40 CUs x 512 flop/clk x 2.9 GHz); a realistic 30 TF on every matmul would take a chunk from
  1.48 s to ~0.9 s (~2,300 tok/s): FP8 dequant in the WMMA tiles (no FP8 hardware) and the GDN kernels are what
  is left. Mia's 2.6k is a GB10 FP8-hardware number.
- Only `_chunks8` uses scratch now (116 bytes a lane; was 2,420 with `_router` and `_hc_up_mix` also spilling).

## 3. Retained-PM4 hang: root cause and workaround

Runtime: `LD_LIBRARY_PATH=tf-rt-libs/pm4:tf-rt-libs/core-10.1 DEBUG_HIP_GRAPH_PM4=1 GPU_MAX_HW_QUEUES=1` (pwilkin
rocm-systems `ilintar-experiments`, commits 58580be2b3 "hip: add retained PM4 graph command lists" + a7694b306d; its
design doc is `projects/rocr-runtime/runtime/docs/contribution/retained-pm4-command-lists.rst`). Knobs in the libs:
DEBUG_HIP_GRAPH_PM4, DEBUG_HIP_GRAPH_PM4_UNQUALIFIED (gfx1151 is already qualified), HSA_GRAPH_COMMAND_LIST_DIAGNOSTICS=1
(prints encoder rejections), plus the stock DEBUG_HIP_GRAPH_BATCH_SIZE / _MERGE_COLLAPSED / _SEGMENT_SCHEDULING. HIP
lowers a graph segment's packet batch to one PM4 IB only when every packet is a kernel dispatch without completion
signal; scratch kernels are bound to the queue's main scratch (falls back to AQL if it does not fit). Script:
`~/tf-window/w2/p3.sh LABEL MODEL [VAR=VAL...]` (CLI sky, 32 tokens, plain, AMD_LOG_LEVEL=3 to count `[PM4] retained`).

Bisect (views built by `~/tf-window/w2/tools/nview.py --layers N --mtp`: the first N layers, PLE kept):

| model | result | HIP graph log |
|---|---|---|
| fn1 (1 DeltaNet layer) | OK, PM4 lists of 2-24 dispatches | `collapse-eligible (max_level=2)` |
| fn1a (1 full-attention layer: `_chunks8` uses 116 B scratch) | OK, a 28-dispatch list with private 116 B | collapse-eligible |
| v4, v12 (4 / 12 layers) | **hang** after the first replay's lists (4 dispatches each) | `deep graph (max_level=8>4), skip collapse -> round-robin (13 segs)`, `Creating 1 parallel streams`, `Resolved queue collision` |
| v4 / v24 + `TF_FLASHNEXT_SHARED_SIDE=0` | OK, one segment, lists of 136 / 736 dispatches, same sha as stock | `collapse-eligible (max_level=0) -> 1 segs` |

Root cause: the engine forks the shared expert onto a second stream inside the captured graph (event record/wait
per layer, cuda_forward.zig `f.side`). From 2 layers on the graph is deep enough that HIP keeps it multi-segment
(two streams, cross-segment sync barriers and completion signals); the retained-PM4 path in that runtime does not
handle a segment's PM4 batch feeding a cross-stream dependency (its own design doc lists "dependency shapes" as
fallback-only, but this shape is not detected), so the first replay waits forever. 1-layer views are shallow enough
to collapse to one stream, which is why they worked. Not size, scratch, LDS, kernargs or VMM (all present in the
working 1-layer and side-off runs). The stock-runtime rocprofv3 graph-mode hang is the same shape: `rocprofv3 --kernel-trace` on graphed bench-many runs
fine with `TF_FLASHNEXT_SHARED_SIDE=0` (prose x1 56.2 tok/s under the tracer; graph-mode trace: 44.7 ms a round, GPU
busy 89.8%, 1,604 dispatches a round; `qmmf` is 42.7 us a launch graphed vs 79.5 eager, so eager traces inflate the
small dense kernels).

Workaround: `TF_FLASHNEXT_SHARED_SIDE=0` (shared expert in line; same bits by design, verified below). Full model:

- CLI sky depth 15, PM4 + side off: `--no-drafts` sha de8d7a445fe7, drafted de8d7a445fe7 (51/69): drafted == plain and
  equal to stock.
- bench-many (served rule, steady reps), same session:

| | stock, side on (default) | stock, side off | PM4, side off |
|---|---:|---:|---:|
| prose x1, default rule | 58.9 | 58.2 | **62.8** (verify 35.3 ms vs 37.8) |
| code x1, default rule | 113.3 | 113.2 | **120.7** |
| prose x1, product 0.1 (PS=8) | 71.5 | 70.9 | **75.3** |
| code x1, product 0.1 | 119.4 | 119.4 | **122.5** |
| prose x8, product 0.1 | 242.3 | 240.3 | 240.1 |
| code x8, product 0.1 | 303.8 | 302.4 | 301.0 |

PM4 gains 5-7% at one stream (the ~3 us gap between dependent dispatches) and nothing at 8 streams (longer kernels,
fewer gaps per byte). The side stream itself is worth ~1% (58.9 vs 58.2).

Server (tensorfold-native, Mia's flags, side off, product 0.1 everywhere, `~/tf-window/w2/p3f.sh pm4|stock`):
- contracts.py a-d: PM4 54 pass / 0 fail / 6 unchecked; stock 54 / 0 / 6 (as the candidate). Decode token shas at
  x1 equal the candidate's default-rule run (prose 2880d0cb0fce, code 70ac5c54fa9a): the policy and PM4 change speed only.
- bench.py (1 and 8 requests, median of 3; then a 4-session alternated x1 A/B):

| server | prose x1 | prose x8 | code x1 | code x8 | 8k prefill |
|---|---:|---:|---:|---:|---:|
| candidate, default rule, side on (cand-results.md) | 57.1 | 197.1 | 109.4 | 254.0 | 1,079 |
| stock, product 0.1, side off | 69.1 / 69.3 / 69.3 / 69.4 | 208.1 | 108.6 / 109.0 / 109.0 / 109.1 | 263.3 | 1,013 |
| PM4, product 0.1, side off | 69.0 / 69.2 / 69.2 / 69.1 | 205.2 | 104.2 / 104.8 / 104.6 / 104.5 | 261.2 | 1,024 |

  So in the server PM4 gained nothing (prose) or lost 4% (code), while the CLI gained 6%. Cause: a retained list
  is prepared on a graph exec's first launch (encode ~44k dwords + executable allocation): 12.7 ms a preparation
  (AMD_LOG_LEVEL=3 timestamps, logging included; a plain launch 4.5 us), and the server recaptured its decode graphs
  for every request (TF_FLASHNEXT_GRAPH_LOG: ~25 captures a request over 20 requests). Root cause in the engine:
  `newSeq` pooled a sequence only when prompt + max_tokens + window >= 8192 rows (`limit >= grow_step`), so
  bench.py's 256-token requests never reused a pooled sequence or its graphs. GPU_MAX_HW_QUEUES (1 or 4) is irrelevant.

**Fix: branch `w2-pool` b0c3cdc** (worktree `/home/dzannotti/tf-w2pool`, 3 lines in cuda_engine.zig `newSeq`): every
request is poolable; a poolable sequence maps the whole first growth step (as a pooled one holds). Built with
aot-final; `zig build test -Dgpu=hip -Daot-set=aot-final` 21/21. Server, product 0.1, side off:
- contracts.py a-d: 54 pass / 0 fail / 6 unchecked with PM4 and with stock; decode shas unchanged
  (prose 2880d0cb0fce, code 70ac5c54fa9a); captures stop after the first requests (50 in 12 requests).
- x1, 6 reps, two alternated sessions each (steady reps): stock prose 70.8 / code 113.3 (was 69.3 / 109.0); PM4
  prose **76.1** / code **120.4** (was 69.2 / 104.6). bench.py full (median of 3, incl. a warming rep): PM4 prose
  75.3 / 209.3 (x8), code 114.5 / 266.9, 8k prefill 1,011, 32k 1,147; stock 70.8 / 207.6, 113.3 / 268.7, 1,060, 1,135.

Server vs candidate defaults (cand-results.md 57.1 / 197.1 / 109.4 / 254.0): product 0.1 + pool fix + PM4 = prose x1
76.1 (+33%, thorim 65.2), code x1 120.4 (+10%, thorim 122.7), prose x8 209 (+6%), code x8 267 (+5%). x8 in the
server stays ~13% below the CLI's (240 / 301): not investigated.

## Recommendations
1. HIP served defaults: running product 0.1 at every stream count (`TF_FLASHNEXT_CONFIDENCE=-0.1
   TF_FLASHNEXT_PRODUCT_STREAMS=8`, depth 15); in code a HIP branch of served_confidence / product_streams_default.
2. Merge w2-pool b0c3cdc (short requests pooled). Then retained PM4 with `TF_FLASHNEXT_SHARED_SIDE=0` is a +7%
   single-stream win in the server (same bits, contracts pass, flat at x8); it needs the out-of-tree runtime
   (tf-rt-libs/pm4 + core-10.1 on LD_LIBRARY_PATH, DEBUG_HIP_GRAPH_PM4=1, GPU_MAX_HW_QUEUES=1). Each new graph still
   costs a ~10 ms list preparation on first replay (first request after load, new context buckets). Upstream report for pwilkin: PM4 batches in multi-segment (forked-stream) graphs hang on the first replay;
   repro = any graph that forks a stream deeper than the collapse limit (max_level > 4).
3. With the side stream off, rocprofv3 works in graph mode: use it for gap analysis (GPU idle ~10% a decode round).

## 4. Final full-model run: rocm a2f8e7e (HIP defaults in code, pool fix), aot-final

Defaults now in code (a2f8e7e): HIP served rule = running product 0.1 at every stream count; shared-expert side
stream off when DEBUG_HIP_GRAPH_PM4 is set (env overrides kept). Served with Mia's flags, TF_FLASHNEXT_DEPTH=15, no
other TF_ env. Script `~/tf-window/w2/final.sh`, logs `~/tf-window/w2/final/`. Each server session: an 8k warm-up first.

| check | PM4 runtime (side off by default) | stock runtime (side on) |
|---|---|---|
| CLI sky depth 15: drafted == plain | True, de8d7a445fe7 both, 57/77 accepted | True, de8d7a445fe7 both, 57/77 |
| contracts.py a-d | 54 pass, 0 fail, 6 unchecked | 54 pass, 0 fail, 6 unchecked |
| agreement.py vs thorim, --max-positions 40 | 790/799 (98.87%), 8/20 free-run identical | 790/799 (98.87%), 8/20 |

**Prefill** (bench.py, one request; tok/s)

| Prompt | Tokens | PM4 | TTFT | stock | TTFT | candidate 5074bde | Mia GB10 |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 4k | 4,094 | 1,042 | 3.93 s | 1,200 | 3.41 s | 1,077 | 2,526 |
| 8k | 8,202 | 1,094 | 7.49 s | 1,196 | 6.86 s | 1,079 | 2,606 |
| 16k | 16,403 | 1,165 | 14.08 s | 1,185 | 13.84 s | 1,096 | 2,643 |
| 32k | 32,772 | 1,217 | 26.94 s | 1,163 | 28.17 s | 1,187 | 2,630 |
| 64k | 65,549 | 1,217 | 53.88 s | 1,167 | 56.16 s | 1,124 | 2,564 |
| 128k | 131,081 | 1,148 | 114.19 s | 1,125 | 116.49 s | 1,115 | 2,415 |

(thorim, our harness: 8k 635 tok/s. Prefill is eager, so PM4 does not touch it; the differences are run-to-run.
Stock quick session: 8k 1,180.)

**Decode, prose** (greedy, thinking off, 256 tokens, EOS ignored, median of 3; aggregate tok/s, per request in brackets)

| Requests | PM4 | stock | candidate 5074bde | Mia GB10 | thorim |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **75.2** (91 ms TTFT) | 71.4 | 57.1 | 64.4 | 65.2 |
| 2 | 90.4 (45.9) | 94.9 (47.6) | 85.6 | 89.0 | |
| 4 | 150.0 (38.3) | 146.8 (38.6) | 146.6 | 140.7 | |
| 8 | 201.8 (27.0) | 203.7 (27.7) | 197.1 | 200.9 | |

**Decode, code**

| Requests | PM4 | stock | candidate 5074bde | Mia GB10 | thorim |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | **117.2** (123 ms TTFT) | 113.2 | 109.4 | 57.5 | 122.7 |
| 2 | 135.7 (69.1) | 139.8 (72.4) | 130.3 | 92.7 | |
| 4 | 176.2 (46.9) | 171.6 (46.2) | 175.4 | 130.8 | |
| 8 | 270.7 (37.6) | 271.4 (37.4) | 254.0 | 189.3 | |

(stock quick session: prose 71.5 / 203.5, code 113.1 / 268.6 at 1 / 8.)

Reading: at one request we now beat Mia's GB10 prose (75.2 vs 64.4) and thorim's prose (65.2); code x1 is at 96% of
thorim (117.2 vs 122.7; the steady-rep A/B in section 3 gave 120.4). PM4's gain is single-stream only (+5% prose, +4%
code); at 2-8 requests PM4 and stock are within noise. Server x2-x8 stays well below the CLI (bench-many prose x8
240): the multi-request server path is the next thing to profile. Prefill is ~1.1-1.2k tok/s, 1.8x thorim, 0.45x Mia.
How to serve: docs/rocm/SERVE.md.
