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
Not GPU staging: the `stage` interval is host time before the chunk's first kernel, spent in the n-gram gather on the
48 GiB host-mapped FP8 table (ple-table/, NVMe). Rows are hashed n-grams spread over the table; with
read_ahead_kb 4096 a cold touch faults in ~2.3 MiB (measured: 300 random touches 1.1 ms each, +686 MiB page cache;
with MADV_RANDOM 0.12 ms each, +1 MiB). Chunk 0 paid ~15k cold faults (16 s), chunk 1 2.4 s, then warm (13 ms). A
dummy chunk at load would only warm what its own n-grams touch; reading the table sequentially takes it all in
(~3.5 GB/s measured by page-touch on a partly cached shard, ~15 s for 48 GiB). bf50f56: `NgramTable.warm` reads one
byte a page of every shard on a background thread started at the end of Engine.init (load time unchanged; joined by
deinit; `TF_FLASHNEXT_TABLE_WARM=0` off). Page cache only (reclaimable, counted in MemAvailable). NOT yet measured
on the full model (see "pending").

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

## Pending (full model; ask the coordinator first)
`/home/dzannotti/tf-engine-scratch/fullcheck.sh [123]` (memwatch, one process at a time, ~12 min):
1 CLI sky drafted == plain; 2 cold-table first prefill, warm on vs off (drops only ple-table pages with
POSIX_FADV_DONTNEED; reports load time and rep 0 / rep 1 prefill); 3 bench-many x1/x8 prose+code base vs new.
