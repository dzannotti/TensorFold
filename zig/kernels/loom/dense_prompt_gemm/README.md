# Dense prompt GEMMs (Loom, gfx11 bf16 WMMA)

Prompt-length products `y = x . W^T` with bf16 activations and fp32 sums, written in Loom's C++ frontend
(hrx-loom 0.3.0) for gfx1151, wave32. Every input is a plain buffer; shapes are kernel arguments.

## Files

| file | job |
|---|---|
| `gemm_tile.h` | generic WMMA tile: geometry, LDS staging two stages ahead, the fold order, any chunk decoder |
| `bf16_rows.h` | bf16 weights as the tile's decoder (16-byte chunks staged as they are) |
| `bf16_gemm.cxx` | `bf16_gemm_128x128`, `bf16_gemm_128x256`: fp32 out |
| `checks.cxx` | `LOOM_CHECK_CASE` / `LOOM_CHECK_BENCHMARK`: fp32 oracle, tile and cut byte identity, Flash Next shapes |
| `hc_exact.h` | the read-out's pointwise math by hipcc's gfx11 sequences: bf16 rounding, `expf`, sigmoid, silu, `rsqrtf` |
| `hc_tiles.h` | the up projection's weights as a decoder whose block columns are each stream's same dimensions |
| `hc_norm.cxx` | `hc_norm`: a hyper-connection's writeback and norm (h in place, x the normed streams) |
| `hc_readout.cxx` | `hc_down_wide`, `hc_down_narrow`: the down with the gates; `hc_up_mix`: the up with the gates and mix |
| `hc_checks.cxx` | the read-out's cases: fp32 oracles, tile and cut byte identity, Flash Next shapes |

## ABI

`bf16_gemm_<rows>x<cols>(x, w, out, m, n, k)`:

- `x`: bf16 `[m][k]`, 16-byte aligned rows; `w`: bf16 `[n][k]`; `out`: fp32 `[m][n]`.
- `k` a multiple of 64 (any other `k` writes nothing); any `m`, `n` >= 1.
- Launch: grid `(ceil(n / cols), ceil(m / rows))`; 256 work-items for 128x128, 512 for 128x256.

## Tile geometry

`Geometry<WR, WC, NT>`: WR x WC waves; a wave owns 32 rows by 16 NT columns (2 x NT 16x16 fragments), so a block
is 32 WR rows by 16 NT WC columns. A stage is 32 inputs of every block row of x and of weights, in LDS with rows
40 inputs apart (16-byte reads of 16 rows land on distinct banks), double-buffered. Each work-item stages fixed
16-byte chunks; a chunk is fetched two stages ahead, held in registers, and stored once the stage before it is
read, with one barrier a stage. Rows past `m` and columns past `n` read a live row, and their sums are never
stored. A wave whose row fragments are all past `m` only stages. The weight decoder (`Dec`) turns a fetched chunk
into 8 exact bf16 inputs; a GGUF or other format plugs in there (scales not exact in bf16 belong outside the
tile).

`bf16_gemm_128x128` is `Geometry<4, 2, 4>`, `bf16_gemm_128x256` `Geometry<4, 4, 4>` (the wider block reads x half
as often: skinny products with long K, such as a hyper-connection's down projection, are bound by those reads).

## Sum order (the bits)

Per fold (128 inputs where `k` is a multiple of 128, else 64): `d` = chained `v_wmma_f32_16x16x16_bf16` over the
fold's K16 slices in K order, starting from +0; `acc += d` per fold, in K order. x is the A operand, the weights
the B operand. The order depends on `k` alone, never on `m`, the tile, a row's place in its block or the chunk
cut, so every tile and every cut gives the same bytes; they equal the TensorFold HIP matrix tile's
(`wmma_gemm_tile` over its BF16 fold decoder).

## Hyper-connection read-out

A hyper-connection (4 streams of `dims` = 2,560, `h` bf16 `[m][4 dims]`) reads its block input in three launches:

- `hc_norm(h, branch, inject, scale, x, m, dims, eps, add)`: when `add`, `h_s += bf16(branch inject[s])` in place
  (`branch` bf16 `[m][dims]`, `inject` bf16 `[m][4]`); then `x = bf16((h rinv) scale)`, `rinv = rsqrt(sum h^2 / dims +
  eps)` per row and stream, `scale` fp32 `[4 dims]`. One block of 256 work-items a row; `dims` is compiled
  (`K5_HC_DIMS`, 2,560). The sum of squares keeps a 1,024-thread block's order: thread t's squares of t, t + 1024, ...
  in order, each 32 threads' xor tree (masks 16 to 1), the 32 sums in order from +0.
- `hc_down_wide` / `hc_down_narrow(x, w, act, inject, m, n, k, low)`: `z = x . W^T` (`w` bf16 `[n][k]`, `k` = 4 dims,
  `n` = `low` or `low + 4`) on the tile, then `v = bf16(bf16(z) / 4)`: `act = bf16(silu(v))` `[m][low]`, `inject =
  bf16(2 bf16(sigmoid(v)))` `[m][4]` for the last 4 columns. Wide: grid `(2, ceil(m / 128))`, 512 work-items, columns
  0..255 on a 128 x 256 tile and the rest on a 128 x 128 tile; narrow: grid `(ceil(n / 128), ceil(m / 128))`, 256.
- `hc_up_mix(act, w, x, mixed, m, dims, k)`: `g = act . W^T` (`w` bf16 `[4 dims][k]`), a 128 x 256 block holding
  dimensions 64 b.. of every stream; `mixed[r][d]` = bf16 of the streams' terms `bf16(bf16(sigmoid(bf16(g))) x)`
  summed in stream order from +0, divided by 4. Grid `(4 dims / 256, ceil(m / 128))`, 512 work-items.

The products keep the tile's sum order, so the bytes equal the HIP chain they replace (`tf_fn_hc_norm`, the BF16
matrix tile, `tf_fn_hc_gates`, the tile, `tf_fn_hc_mix`) at any row count and cut. The gates take z and g through
the LDS as bf16 (their first rounding) and run in rolled loops over 8 columns a work-item: unrolled over a wave's
fragments they spilled the instruction cache. Dividing by 4 is multiplying by 0.25 (the same bits).

The norm stays a launch of its own: forming x from h, rinv and scale while the down stages x (the norm in the
GEMM's prologue) costs more on gfx11 than writing x once, because the transform's VALU work and loads sit between
each stage's WMMAs and its barrier (4,096 rows: down 2.1 ms against 1.5 ms, norm 0.9 ms against 1.35 ms).

## Test and benchmark

```shell
loom-import-cxx checks.cxx --output=checks.loom
loom-test checks.loom --device=amdgpu
loom-import-cxx hc_checks.cxx --output=hc_checks.loom
loom-test hc_checks.loom --device=amdgpu
loom-benchmark checks.loom --device=amdgpu --benchmark=@bench_hc_down_4096_rows --measure=dispatch_complete
```

The benchmark cases time their oracle launches too; time the kernels alone with your own launcher.
