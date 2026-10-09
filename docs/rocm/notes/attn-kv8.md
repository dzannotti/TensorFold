# fp8-KV sparse attention (`_chunks8`) on gfx1151 (perf-attn)

## What changed (kv8.py `chunk8`, same signature, grid, constexprs and semantics; Zig unchanged)
- **e4m3 -> bf16 a word at a time** (`bf16_words`): a code's magnitude bits placed as an fp16's are its value / 256
  (subnormals included), `v_cvt_f32_f16` widens it, the top half is its bf16 (exact, no rounding). ~3 ops a code;
  Triton's own conversion was ~30 integer ops and ran after the layout change, on replicated dot operands, with
  the bytes moved through LDS one `ds_load_u8` at a time. The 2^-8 keeps every code a bf16 normal (the WMMA flushes
  bf16 subnormals: measured); it goes back exactly on s_k (x 256) and on the stored partial o (x 256).
  All 254 finite codes decode exactly (`kv8_attn.py check`, first line).
- **s^T = k . q^T**: Triton's chained-dot heuristic gives the first dot warpsPerCTA [4, 1]; with q.k^T (M = 16
  head slots) all four warps computed the same 16 x 64 scores (80 WMMA a tile, 4x redundant). With k.q^T the 64
  keys are M: 16 keys a warp, 32 WMMA a tile in all.
- **D in four 64-wide slices** for both dots (q.k^T's K steps in D order, o as four column blocks), q reloaded per
  tile (an always-true mask stops LICM from hoisting all of q into 128 VGPRs).
- **No masked K/V loads**: a tile's keys past n read row n - 1 (a sparse row's ids past nk read its key 0's slot,
  which is in bounds); their scores are -inf and probabilities 0 as before. The padded head slots read head 0's q
  and are never stored. Masked loads were per-row branches and their address registers spilled.
- Spills 595-678 VGPRs (2.3-2.6 KB private a lane) -> 25-34 (88-116 B). 8 KiB LDS (was 16 KiB).

Bits: not equal to the old kernel's (the softmax sums now reduce over a different layout): over 42 cases (M 1, 3,
16, 17, 129, 161, 2049; dense rows at p0 0/37/1900 of 4096, sparse rows at 5000/70000/260087 of 262144) a few
elements a case differ by one bf16 ulp. Against fp64 attention over the dequantized cache the worst |err| is the same
in every case (9.324e-03 overall; bf16 output rounding), see "Accuracy" below. Deterministic, and each row's bits
equal the row launched alone (first and last row of every case).

## Kernel time (torch.profiler device time, best of 5 reps' medians, 3 alternated A/B rounds; GPU shared, quiet)

| shape (24 heads, 2 KV heads, D 256) | rocm-next `_chunks8` | perf-attn `_chunks8` | x | `_merge` before -> after |
|---|---:|---:|---:|---:|
| decode M1, 50 keys | 51.9-52.1 us | 8.4-9.5 us | 5.8 | 1.5 -> 1.6 us |
| decode M3, 50 keys | 61.0-65.0 | 9.7-9.9 | 6.4 | 2.7 -> 1.9 |
| decode M16, 400 keys | 571-616 | 51.1-51.4 | 11.6 | 3.0 -> 2.2 |
| decode M3, sparse at 70000 (2051 keys) | 649.5-650.0 | 62.1-63.6 | 10.4 | 3.6 -> 3.0 |
| prefill block M256 dense at 1792 | 16,374-16,997 | 1,616-1,648 | 10.2 | 191-205 -> 40-42 |
| prefill block M256 sparse at 6144 | 18,118-18,844 | 1,879-1,907 | 9.8 | 251-262 -> 142-149 |

Reproduce: `tools/rocm/kv8_attn.py bench` with PYTHONPATH at either tree's src, alternated.
Projected (not measured end to end): a 2048-row prefill chunk's 104 launches ~2.6 s -> ~0.2 s (~1.9 ms each; the
target was ~150 ms); an 8-stream decode round's ~109 launches at < 400 keys 20 ms -> ~1-2 ms.
~2.3 TFLOPS of useful work at prefill (256 x 24 x 2051 x 256 x 4 / 1.88 ms), 4.5x the old kernel's.

## The VMM / TLB hypothesis (profile.md 3a): not it
`tools/rocm/kv8_vmm.py` maps the same caches as cuda_vmm.zig does (hipMemAddressReserve + hipMemCreate chunks; HIP
granularity: minimum 4 KiB, recommended 2 MiB) and as hipMalloc, all launched with the plain (non buffer-op) binary
Zig uses on VMM caches. us a launch, best of 5 (old kernel best of 3):

| layout | new: prefill p6144 | new: decode p70000 | new: prefill p200000 | old: prefill p6144 | old: prefill p200000 |
|---|---:|---:|---:|---:|---:|
| hipMalloc (torch) | 1876 | 72.0 | 2945 | 18,871 | 24,185 |
| VMM, 8192-row chunks, base +4 KiB off 2 MiB | 1880 | 73.7 | 3018 | 18,980 | 24,311 |
| VMM, 8192-row chunks, aligned (the engine's growth) | 1884 | 70.4 | 2959 | 19,203 | 24,209 |
| VMM, one chunk, +4 KiB | 1876 | 74.2 | 3041 | 19,019 | 24,260 |
| VMM, one chunk, aligned | 1875 | 71.6 | 2947 | 19,009 | 24,212 |
| VMM, 2 MiB granularity, 8192-row chunks | 1885 | 71.9 | 2946 | 19,357 | 24,272 |
| VMM, 2 MiB granularity, one chunk | 1884 | 73.1 | 2944 | 19,286 | 24,156 |

Mapping moves either kernel by <= 3% (a base 4 KiB off a 2 MiB boundary: +2.5% at 200k keys; 2 MiB granularity
buys nothing measurable), so cuda_vmm.zig is left as is. What does move it: where the selected keys lie. The same
launch at position 200000 costs 1.57x (new) / 1.28x (old) the one at 6144: 200k positions of keys and values are
105 MB per layer, past the 32 MiB MALL, against 3.2 MB. The old kernel's run-to-run swings are more likely its
2.6 KB a lane of scratch (spill traffic and the runtime's scratch sizing under other GPU users) than the TLB; the new
kernel has 116 B. Not verified in the engine.

`hip_tune` has no `_chunks8` row: rocm and rocm-next build it with the same options (warps 4, stages 1), and the
plain `_chunks8` hsaco is byte-identical in aot-hip2 and aot-new (5e92bfcf01 etc.); rocm-next only adds the
range32 twins, which VMM caches never take. rocm-next's slower attention in the profile is not a `_chunks8` change.

## Accuracy (`kv8_attn.py check`, fp64 reference over the dequantized cache)
42 cases (above), both kernels on the same inputs: worst |err| 9.324e-03 for both (M 2049 at p0 0 of 4096; the
bf16 rounding of the output), and the worst element and the worst head equal case by case. Mean |err| over the
checked rows: 2.78534e-04 old, 2.78535e-04 new; per case equal to 4-5 digits (new higher by ~2e-8 in 3 cases, lower
in 1). Dense M 1 at p0 0 is exact for both. Determinism and row invariance: all 42 cases.

## Rebuild and checks
```bash
cd /home/dzannotti/tf-attn
# AOT set (CPU only; 830 hsaco; only _chunks8's 8 hashes differ from rocm-next's aot-new)
PYTHONPATH=src python -B tools/zig/flashnext_aot.py build --target hip --spec zig/tests/cuda/flashnext/kernels.json \
  --spec zig/tests/cuda/flashnext/kernels_int4ar.json --out /home/dzannotti/tf-attn-scratch/aot-attn
PYTHONPATH=src python tools/rocm/triton_parity.py hash --aot $K --spec zig/tests/cuda/flashnext/kernels.json \
  --spec zig/tests/cuda/flashnext/kernels_int4ar.json          # 908/908 equal
PYTHONPATH=src python tools/rocm/triton_parity.py run --aot $K --cases kv8   # 204/204 launches equal (40 _chunks8)
PYTHONPATH=src python tools/rocm/kv8_attn.py check            # 42 cases ok, codes exact
PYTHONPATH=src python tools/rocm/kv8_attn.py bench
PYTHONPATH=src python tools/rocm/kv8_vmm.py
zig build test -Dgpu=hip -Daot-set=$K                          # 21/21 steps
```
The triton_parity kv8 cases now run M 1, 3, 16, 17, 129, 161, 2049 at 1024 and 262144 capacity, sparse rows at
200000 (the indexer's own selection), both ranks.

## Left
- Prefill is ~1.9 ms a 256-row block (target ~1.45): 256 VGPRs, 3 workgroups a CU; the V tile still goes to its
  dot operand through LDS 2 bytes at a time (a key-major tile, keys-contiguous operand).
- Long-context decode is serial over a program's 8 tiles (CH 512: 5 chunks a row): a 256-key chunk would halve
  it but changes NCH, the scratch and `_merge` for every cache format (Python CHUNK, Zig AttnGeometry.chunk).
- Zig's shared-round attention (`attentionMulti`) has no fp8 path: an fp8 round still launches `_chunks8` per stream.
- e4m3 NaN codes (0x7F/0xFF) now decode to finite values; `_attn_prep8` never writes them from finite rows.
