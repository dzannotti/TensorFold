# n-gram row cache and read-ahead (perf-ngram, 2026-10-09)

Problem (longctx.md): at TP=1 the FP8 n-gram table (ple-table/, 48.7 GiB, 320M rows of 160 B) is host-mapped and
gathered per stage; beside 66 GiB of weights in GTT the page cache keeps ~3 GiB of it, so new n-grams are NVMe reads
(2.4-4.5 ms each) on the round's critical path and in prefill staging.

"The FP8 n-gram table is on the GPU" (Mia): in this engine that is `weights.Options.ngram_on_gpu`, default
`world > 1`: at TP=2 each rank uploads its 8 heads' rows (~24 GiB) and gathers them with `fn_ngram_gather`
(fn_pack.cu). No env or CLI sets it; at TP=1 the whole 48.7 GiB table would have to be resident, which does not fit
here. There is no subset/device-cache option upstream.

## What changed
- `cuda_ngram_cache.zig` (new): `RowCache`, set-associative (8 ways, LRU by a global use tick inside a set, 1024
  lock stripes), rows stored as on disk (FP8 codes), so a gather's bits are the mapped table's by construction.
  A miss reads the row outside any lock (the fault), then inserts it; misses of one gather get MADV_WILLNEED first.
  `Warmer`: 2 threads reading queued ids into the cache (a hint: full queue drops, the stage reads what is missing).
- Size: `TF_FLASHNEXT_NGRAM_CACHE_GIB` (0 off); default 1 GiB on HIP (~6.2M rows, ~390k tokens' n-grams; the same
  memory as page cache holds ~25x fewer useful rows), 0 on CUDA. Allocated and written at load, before the engine
  measures MemAvailable for the sequence budget, so it is counted there (it is host RAM, not GTT).
- `NgramTable` (cuda_weights.zig): gathers go through the cache when enabled; `warm(ids)` pushes to the warmer
  (without a cache: MADV_WILLNEED on the caller's thread).
- Read-ahead in decode (cuda_engine.zig): the next verify window is [last kept token, drafts...]. After the MTP
  absorb is queued, the kept token's 16 rows are pushed; each draft's 16 rows are pushed as soon as it is drawn
  (draftMany and the single-stream sampleDraft/drawDraft), so their reads overlap the remaining draft levels.
- Prefill overlap (cuda_forward.zig stageMany): the host gather now runs before `staged.synchronize()` into a host
  scratch, then is copied into the pinned buffer. Before, chunk k+1's gather started only after chunk k's main
  compute finished (the MTP stage's event sits after it); now it overlaps chunk k's compute.

## Verified (host only; GPU untouched)
- `zig build test -Dgpu=hip` and `zig build test` (CUDA option): all steps pass, including 4 new tests: gather ==
  mapped bytes cold/warm across shard boundaries and repeats, out-of-range ids refused, LRU eviction in a full set,
  4 threads + the warmer under constant eviction byte-equal and every way's bytes its own id's.
- `tf-flashnext-ngram-bench` smoke (30 tokens on the real table, page cache warm): PASS, rows byte-equal.

## To run when allowed
- Microbench: `tools/rocm/ngram_bench.sh [TOKENS] [--drop]` (windows 1/6/16, direct vs cache vs ahead, two passes).
  `--drop` only in a test window.
- Full model: see the perf-ngram report (checks listed there).
