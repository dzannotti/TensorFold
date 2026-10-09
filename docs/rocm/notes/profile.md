# Profile: Flash Next on gfx1151, rocm @ b768ba8 (runtime ROCm 10.0 / HIP 7.15, Triton set aot-hip2)

Window 2026-10-09, prod down, GPU exclusive. Raw logs: `~/tf-window/perf/`. All decode numbers: depth 15, fp8 KV,
context 262144, served stop rule (`--served`: running product -0.4 at <= 2 streams, 0.5 above), greedy, 256 tokens,
EOS ignored, sparkDash prompts (`bench-many --kind dash-prose,dash-code`, the same prompts bench.py sends).

## Tooling notes (read first)
- `rocprofv3 --kernel-trace` HANGS the engine in graph mode (server warm-up and CLI bench-many alike): the first
  replayed forward never completes (`queue_interposition.cpp: Async signal handler still waiting on signal`), 1,520
  dispatches queued with start == end. Not fixed by GPU_MAX_HW_QUEUES=1. `rocprofv3 --attach` fails (ptrace, scope 1).
  Works with `--eager`, so kernel-level numbers below are eager (kernel durations are the same; gaps are not).
- The engine's own part profiler (`TF_FLASHNEXT_PROFILE=1`, eager, events between parts) gives per-part GPU timeline.
- Server and CLI agree: server prose 1-req 49.3 tok/s, 100 rounds, 155/192 accepted; CLI bench-many 49.7 tok/s,
  100 rounds. So the CLI is a faithful stand-in.
- Our env matches Mia's served env except vision: engine defaults are TF_FLASHNEXT_PRODUCT_STREAMS=2 and
  TF_FLASHNEXT_PREFILL_TAIL=512 (cuda_engine.zig), DEPTH=15 set. Differences: TENSORFOLD_MEMORY_RESERVE_GIB 12 (engine
  default) vs Mia 10 (only sizes the pool), no TF_FLASHNEXT_YARN (=0, same).
- Our build warns `no groups-of-32 4-bit kernels in this build: MTP drafts use the full head` (aot-hip2 lacks the
  draft-head slice kernels): drafts read the full 0.31 GiB int4 head, ~1.3 ms a draft level.

## 1a. Single-stream decode, graphed (CLI bench-many, steady reps)

| | prose x1 | code x1 | prose x8 | code x8 |
|---|---:|---:|---:|---:|
| tok/s (aggregate) | 49.7 | 91.7 | 164.6 | 205.8 |
| rounds / 256 tok a stream | 100 | 38 | 109 | 62 |
| tokens a stream-round | 2.55 | 6.71 | 2.65 | 5.37 |
| round wall (ms) | 51.3 | 72.9 | 113.9 | 160.1 |
| verify (ms; host enqueue) | 44.2 (4.7) | 54.7 (34.4) | 93.0 (8.3) | 121.2 (15.3) |
| drafting, MTP (ms) | 6.9 | 18.1 | 19.6 | 37.7 |
| draws+commits (ms) | 0.2 | 0.4 | 1.1 | 1.0 |

Graphs save only ~2.6 ms a round over eager (prose verify 44.2 graphed vs 46.8 eager), i.e. host/launch overhead is
NOT the main cost: the GPU is busy. Thorim (same engine, GB10) prose 65.2 tok/s at 2.55 tok/round => ~39 ms a round;
we are 51 ms. Code thorim 122.7 => ~54.7 ms a round (if same acceptance) vs our 72.9.

## 1b. Where a decode round goes (TF_FLASHNEXT_PROFILE=1, eager, ms a round, steady rep)

Main-model parts (48 layers; 36 GDN + 12 full attention):

| part | prose x1 | code x1 | prose x8 | code x8 | weights read (fp8 unless noted) | bandwidth floor @242 GB/s | achieved x1 |
|---|---:|---:|---:|---:|---|---:|---:|
| gdn_proj (in_proj qkv,z,a,b) | 8.78 | 8.80 | 9.15 | 10.52 | 1.43 GiB | 6.3 ms | 175 GB/s |
| hc_down (2x mix down/layer) | 4.75 | 4.72 | 7.54 | 9.29 | 0.59 GiB | 2.6 ms | 133 GB/s |
| hc_up (2x mix up/layer) | 3.54 | 3.55 | 4.37 | 4.20 | 0.59 GiB | 2.6 ms | 178 GB/s |
| out_proj (GDN out + attn o) | 4.08 | 4.15 | 5.59 | 6.11 | 0.70 GiB | 3.1 ms | 185 GB/s |
| attn_proj (q,k,v,indexer) | 2.58 | 2.60 | 2.58 | 3.33 | 0.43 GiB | 1.9 ms | 180 GB/s |
| router (bf16 gate) | 2.37 | 2.39 | 2.37 | 2.51 | 0.117 GiB | 0.52 ms | **53 GB/s** |
| topk_plan | 0.51 | 0.52 | 0.54 | 0.57 | - | - | |
| experts_gate_up (int4 + shared) | 6.03 | 10.24 | 17.74 | 26.98 | per distinct expert 1.7 MB/layer | | |
| experts_down | 3.38 | 5.56 | 9.31 | 14.42 | | | |
| attention (12 layers + QSA) | 3.00 | 3.91 | **19.29** | **21.86** | KV tiny (prompt ~50 tokens) | ~0 | |
| gdn_chain | 1.51 | 2.03 | 6.18 | 6.93 | state 36 x 48 x 128 x 128 f32 | | |
| head (int4 lm_head) | 1.42 | 1.43 | 2.61 | 3.92 | 0.31 GiB | 1.36 ms | 230 GB/s |
| ple + hc_* glue + reduce + rest | ~4.6 | ~4.7 | ~6.5 | ~8 | | | |
| **main total** | **47.2** | **55.2** | **95.2** | **119.4** | ~4.6 GiB dense + experts | ~27 ms (x1 prose) | |
| MTP head (drafting) | 6.97 | 18.2 | 20.2 | 38.7 | | | |

MTP draft levels (prose x1): absorb 2.66, L1 2.14, L2 1.13, L3 0.56 ... ms a round; each draft level costs ~1.3 ms of
full int4 head (mtp.head 1.29 ms/round prose, 3.34 code) + attn_proj/out_proj (bf16 MTP layer).

Key decode findings (priority order):
1. **Dense fp8 matvecs run at 133-185 GB/s, not 242.** gdn_proj + hc_down + hc_up + out_proj + attn_proj = 23.7 ms a
   prose round vs a 16.5 ms floor. hc_down is worst (133 GB/s). Getting all to ~220 GB/s saves ~6 ms/round (12%).
2. **Router: 2.37 ms for 0.117 GiB = 53 GB/s** (~1.85 ms lost a round). rocm-next's retune claims router 73 -> 11 us
   per launch (48 launches: 3.5 -> 0.5 ms), which matches this gap.
3. **8-stream attention: 19.3 ms a round (3.0 at 1 stream)** with ~50-token contexts: this is launch/occupancy, not
   bytes. Plus gdn_chain 6.2 vs 1.5 and mtp.attention 4.2 vs 0.6. At 8 streams this is ~20% of the round.
4. MTP drafting 7 ms (prose) / 18 ms (code) a round: the full int4 head per draft level (draft-head slice kernels
   missing from aot-hip2), and the bf16 MTP layer's attn/out projections ~2.4 ms a round prose.

(rocprofv3 kernel-level tables follow below when the eager trace finishes.)
