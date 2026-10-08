# GPTQ int4 kernels on gfx1151 (branch rocm-int4)

## Entry points (zig/kernels/hip/fn_int4.hip, `extern "C"`, 128 threads a block)

`tf_int4_<GS>_<NT>_<MT>_<MATS>_<EPI>` for (GS, MATS, EPI) in (128,2,2) gate/up SwiGLU, (128,1,0) fp32 out,
(128,1,3) bf16 out, (64,1,0), (64,1,3); NT 1|2 (n16 tiles a wave), MT 1|2 (row tiles of 16 a pass). Arguments are
fn_int4.cu's `int4_kernel` ones, in order:

    (const bf16* X, int x_stride, int slots, const u32* W, const f16* S, int K, int N,
     const int* items, const int* counts, const int* members, int rows, void* out, int out_stride, float limit, int skip)

Grid: a block a unit, unit = (plan item, 16 * NT * 4 / MATS output columns); `grid = min(max_items * N / (16 * NT *
4 / MATS), occupancy * multiprocessor_count)` (cuda_int4.zig `launch`: `unitCols`, `unit_blocks`). No dynamic
shared memory. The prompt down (`downPrompt`) is the same kernel at MT 2 on the 64-pair plan; there is no
`int4_prompt_kernel` on HIP and `Kernels.pdown` stays unset.

## Layout (cuda_int4.zig `packWords` / `packScales`, `hip_layout`)

Words `[N/16/ST][K/GS][ST][GS/32][16 columns][4]`, ST = gcd(N/16, 16) (`superTiles`): each word is GPTQ's qweight
word for 8 consecutive inputs of one column, xor 0x88888888 (codes as two's complement q - 8). Scales fp16
`[N/16/ST][K/GS][ST][16]`. Sizes equal the CUDA layout's (`wordsOf`, `scalesOf`); `n % 16 == 0` required. The
super tiles keep cuda_weights.zig's head packing in column chunks of 2048 valid (each chunk whole super tiles).

## Numerics

W4A16: bf16 activations, int4 weights exact in bf16, fp32 sums. Per group a WMMA chain (k16 steps in order) then
`acc = fmaf(sum, scale, acc)` in group order. gfx11 WMMA is not IEEE inside a 16-wide step (a product next to zero
products can be off by an fp32 ulp) but an output reads only its row and column (checked with neighbour rows of
2^-30..2^30). Error vs fp64: max |err| / sum |x w| 1.3e-8 .. 3.5e-8 on random and checkpoint data.

## Results (gfx1151, prod sharing the GPU: best of 7 rounds, graph-replayed, MALL defeated by rotating experts)

Plain streaming read: 242 GB/s (1 GiB), 217-227 GB/s (16-64 MiB).

| case | gate/up | down | total |
|---|---|---|---|
| head [248320, 2560] 1 / 8 / 16 rows | | | 243 / 240 / 240 GB/s (1.35 ms) |
| 5 experts x 1 row | 39.6 us 207 GB/s | 23.1 us 177 GB/s | 202 GB/s |
| 5 x 16 | 40.6 us | 22.9 us | 200 GB/s |
| 20 x 1 / 4 / 16 | ~150 us | ~81 us | 217-220 GB/s |
| 40 x 1 / 4 / 16 | ~297 us | ~149 us | 226-229 GB/s |
| 74 x 1 / 4 / 16 | ~539 us | ~267 us | 231-234 GB/s |
| prompt 512 rows, 16-pair items MT 1 | 3.67 ms | 1.81 ms | 4.6 TFLOPS, 229 GB/s |
| prompt 2048 rows, 64-pair items MT 2 NT 2 | 4.05-4.68 ms | 2.29-2.47 ms | 14.1-15.9 TFLOPS |
| prompt 2048 rows, 16-pair items MT 1 | 6.55 ms | 3.52 ms | 10.0 TFLOPS |

What mattered (A/B, same conditions): super-tile layout (per-tile 1 KiB runs: head 197 -> 228 GB/s), barriers
without the workgroup fence (it waits vmcnt(0) in WGP mode), no conditional loads or immediate conversions of
prefetched data (each made the waitcnt pass drain the prefetch), a wave a matrix for gate/up (178 -> 131 VGPRs).
Did not help: deeper prefetch (4, 8 groups), waves_per_eu hints, wider super tiles, skipping the dequant entirely
(so decode is not VALU-bound). WMMA bf16 peak measured 55.5 TFLOPS.

## Tools (run inside the dev image)

- `python tools/rocm/int4_check.py [--quick]`: fp64 reference, row/NT/MT invariance (rows permuted, magnitudes
  2^±30), group order, plan == dense, skipped slot, SwiGLU vs host, determinism, checkpoint slices (layer 0
  experts 0/511, layer 47 expert 137, lm_head columns).
- `python tools/rocm/int4_bench.py`, `python tools/rocm/int4_ab.py -- "<flags A>" "<flags B>"` (or `--src`).
- `tools/rocm/int4_regs.sh`: VGPRs / spills per entry point. `tools/rocm/int4_zig.sh`: cuda_int4.zig host tests at
  both layouts plus a type-check of its HIP paths against `int4_cuda_stub.zig`.
- `tools/rocm/int4_probe.{hip,py}`: WMMA layout, stream / access-pattern / WMMA-rate probes.
