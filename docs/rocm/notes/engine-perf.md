# Engine-level decode/prefill scheduling (branch perf-engine, 2026-10-09)

Base: rocm-next @ 7ec5235. Kernel set aot-hip3. GPU shared with the perf lead's full-model runs: view timings are
noisy (A/B alternated) unless noted. Scripts: `/home/dzannotti/tf-engine-scratch/*.sh`, logs in `.../out/`.

## 1. MTP draft head
- The warning `no groups-of-32 4-bit kernels in this build: MTP drafts use the full head` was stale: it fired on
  every HIP load before the slice decision. Every profiled run also logged `MTP draft head: the int4 lm_head's 79591
  draft columns (105.1 MB)` and drafted over the slice in all paths (CLI run: decode.drafted -> sampleDraft; bench-many
  and server: draftMany): `mtp.head` = 0.46 ms a call (full head 1.43). profile.md's "1.3 ms a draft level" is
  1.29 ms a *round* = ~2.8 head calls. Fixed: warn after the load only when neither draft head exists (e8d915b).
- Slice logits == full head logits at the draft ids: `tools/rocm/drafthead_check.py` on ROCm: bit-equal at 1/2/8/16
  rows, argmax equal; zig test `packColumns: contiguous columns pack as packWords` covers the engine's packing.
- Rest of MTP cost (prose x1 6.9 ms, code x1 18 ms a round): a draft level is one single-row MTP forward, ~2.1-2.4 ms
  (bf16 attn_proj 0.45 + out_proj 0.37 + head 0.46 + attention 0.26 + mtp_in 0.15 + hc 0.25 + experts ~0.2), ~2.8
  levels a prose round, ~7.3 a code round. Bandwidth-bound bf16 reads, no engine waste found besides:
  - draw per level: toF32 + lse (36 us, one 256-thread block a row) + argmax (67 us -> 21 us, ac1a5d9) + pick.
  - absorb (2.7 ms prose / 3.1 code) runs the whole MTP layer on every kept row although only the last row's logits
    are read; rows before it only need their K/V (and indexer keys). At x1 the matvecs are weight-bound, so the
    saving is the extra rows' routed experts/attention (~0.4-0.5 ms prose, more for code). Not done: needs a
    rows-split inside layerForward/attnBlock (attention is being rewritten elsewhere).

## 2. First prefill after load
Instrumented on the full model (host ms inside stageMany, cold table): chunk 0's n-gram gather took 20.9 s (then
3.0, 1.0, 0.5 s; 13 ms warm). It is the 48 GiB host-mapped FP8 table (ple-table/, NVMe): rows are hashed n-grams
spread over the table, and a cold lookup faulted in ~2.3 MiB of read-around (read_ahead_kb 4096; 1.1 ms a touch,
0.12 ms with MADV_RANDOM), one lookup at a time. (The later chunks' 2.4-3.5 s "ids" time is the staged-event wait
on the previous chunk's GPU work: overlapped, harmless.) A background read of the table after load (bf50f56) did not
help (37.3 vs 39.0 s: the load evicts the table and the read competed with the gather) and was reverted (b43781b).
b40f6bd: the shards are MADV_RANDOM and a chunk's lookups run on the loader's threads (256 a job), so faults overlap
on the NVMe queue. Full model, table pages dropped first (POSIX_FADV_DONTNEED), 8,212-token prompt:

| | rep 0 (first after load) | rep 1 | sha |
|---|---:|---:|---|
| before (no warm) | 38.96 s | 14.33 s | a11a30d2c83f |
| background warm (reverted) | 37.28 s | 18.84 s | a11a30d2c83f |
| b40f6bd | **16.04 s** | 14.24 s | a11a30d2c83f |

## 3. GPU idle in the round loop
- profile.md's 9% idle (4.5 ms a prose round) was measured eager. Graph-mode rocprofv3 on the 1-layer MTP view
  (works there, 4,470 dispatches): host round trips (argmax/draft pick -> read -> next stage) are ~27 us each,
  ~0.1 ms a round in all; the idle is ~3.0 us between *every* pair of dependent kernels inside the graphs (13.2 ms
  over 4,382 gaps). At ~1,740 dispatches a prose round that is ~5 ms: the 9%. Only fewer launches (fusing glue:
  `_hc_*`, `_reduce`, strided copies, the draw kernels) removes it; graphs cannot. `DEBUG_HIP_GRAPH_CLASSIC_PATH=1`
  and `HIP_FORCE_DEV_KERNARG=1`: no measurable change on the view.
- 2cbb025 (one rank, TF_FLASHNEXT_DRAW_ONCE=0 restores the old way): a round's greedy windows share one argmax and
  one wait (was one launch + wait + read a window: x8 had 28-44 argmax launches a round); a draft level queues every
  active row's draw kernels into its own slot and waits once (was a wait a row). Nothing changes at one stream.
- ac1a5d9: argmax at 1024 threads a row: 67 -> 21 us a launch (rocprofv3, view), same column (total-order pick).
- rocprofv3 graph hang: not reproduced on the view in graph mode (21 graphs, short). HIP 7.15 has a batched graph
  path (`ScheduleNodesIntoBatches`, `[Graph batch barrier]`); candidate test on the full model:
  `DEBUG_HIP_GRAPH_CLASSIC_PATH=1 rocprofv3 --kernel-trace ...` (pending).

## Verification (1-layer MTP view, TF_FLASHNEXT_INT4AR_FAST=1, depth 15)
- CLI sky drafted == `--no-drafts`: sha 95c2c0d8d8a2 both, base and new, served rule and TF_FLASHNEXT_CONFIDENCE=0.
- `gate-many --streams 8 --against-solo` (drafted, serial, mixed; greedy + sampled): 24/24 PASS, and every stream's
  sha, rounds and accepted identical between base and new (served rule and confidence 0).
- `zig build test -Dgpu=hip -Daot-set=aot-hip3`: 21/21 steps.

## Full-model results (prod down, GPU exclusive per step, fp8 KV, aot-hip3)
- CLI sky drafted == `--no-drafts`: sha de8d7a445fe7 both (accepted 50 of 69 drafts).
- bench-many --served, steady reps (rep 0 pays the per-request captures), base 7ec5235 -> new (2cbb025 + ac1a5d9):
  prose x1 51.3 -> 51.6 tok/s; code x1 93.4 -> 94.2; prose x8 157.4 -> 157-163; code x8 199.7 -> 206.3 (draws+commits
  1.1 -> 0.4 ms a round, drafting 38.1 -> 35.2). Rounds and tokens a stream-round identical (100/2.55, 38/6.71,
  109/2.65, 62/5.37).
- TF_FLASHNEXT_VMM=0 (eda4701; view: same sha, gate-many 24/24): stock runtime prose x1 ~47-48, code x1 ~82 tok/s
  (VMM costs nothing; copy-grown caches do).
- Retained PM4 (tf-rt-libs/pm4, DEBUG_HIP_GRAPH_PM4=1, GPU_MAX_HW_QUEUES=1): full model hangs after load at the
  first replay with VMM on AND off (CLI run, plain and drafted; also with ROC_AQL_QUEUE_SIZE=65536). On the 1-layer
  view it runs, VMM on and off (same sha as stock). So not VMM; something with the full graph's size/content.
- rocprofv3 --kernel-trace graph hang: unchanged with DEBUG_HIP_GRAPH_CLASSIC_PATH=1 and with ROC_AQL_QUEUE_SIZE=65536
  ("Async signal handler still waiting on signal" right after load); view works. Same size-dependent pattern as PM4.
- Code x1 host enqueue (34 of 55 ms a verify): graphs are keyed by sequence (they bake its buffers) and every
  request gets a new sequence (lanes.preparePrefill / generateMany newSeq; freeSeq drops its graphs; warmGraphs only
  warms the engine's own). A request recaptures each window size x DeltaNet parity it meets (TF_FLASHNEXT_GRAPH_LOG:
  ~100 captures over 6 requests); code meets most of the 16 sizes x 2 parities in its 38 rounds, prose few. A
  capture is an eager run, a wait, the capture and an instantiate: host-bound, hence the CPU-load sensitivity. The
  server pays the same per request. Tried: capturing without the wait (overlapping the eager run): slower (prose
  47-49, code 31-83 tok/s), dropped. Fix to do: keep sequences (and their graphs) across requests: a pool of freed
  VMM sequences reserved at max_len, physical pages unmapped back to the first step when pooled.

## Scripts
`/home/dzannotti/tf-engine-scratch/fullcheck.sh [123]` (memwatch, one process at a time, ~12 min):
1 CLI sky drafted == plain; 2 cold-table first prefill, warm on vs off (drops only ple-table pages with
POSIX_FADV_DONTNEED; reports load time and rep 0 / rep 1 prefill); 3 bench-many x1/x8 prose+code base vs new.
