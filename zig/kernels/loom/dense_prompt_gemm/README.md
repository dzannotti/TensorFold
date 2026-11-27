# Dense prompt GEMMs (Loom, gfx11 bf16 WMMA)

Prompt-length products `y = x . W^T` with bf16 activations and fp32 sums, written in Loom's C++ frontend
(hrx-loom 0.3.0) for gfx1151, wave32. Every input is a plain buffer; shapes are kernel arguments.

## Files

| file | job |
|---|---|
| `gemm_tile.h` | generic WMMA tile: geometry, LDS staging one stage ahead, the fold order, any chunk decoder |
| `bf16_rows.h` | bf16 weights as the tile's decoder (16-byte chunks staged as they are) |
| `bf16_gemm.cxx` | `bf16_gemm_128x128`, `bf16_gemm_128x256`: fp32 out |
| `checks.cxx` | `LOOM_CHECK_CASE` / `LOOM_CHECK_BENCHMARK`: fp32 oracle, tile and cut byte identity, Flash Next shapes |

## ABI

`bf16_gemm_<rows>x<cols>(x, w, out, m, n, k)`:

- `x`: bf16 `[m][k]`, 16-byte aligned rows; `w`: bf16 `[n][k]`; `out`: fp32 `[m][n]`.
- `k` a multiple of 64 (any other `k` writes nothing); any `m`, `n` >= 1.
- Launch: grid `(ceil(n / cols), ceil(m / rows))`; 256 work-items for 128x128, 512 for 128x256.

## Tile geometry

`Geometry<WR, WC, NT>`: WR x WC waves; a wave owns 32 rows by 16 NT columns (2 x NT 16x16 fragments), so a block
is 32 WR rows by 16 NT WC columns. A stage is 32 inputs of every block row of x and of weights, in LDS with rows
40 inputs apart (16-byte reads of 16 rows land on distinct banks), double-buffered. Each work-item stages fixed
16-byte chunks; a chunk is fetched one stage ahead into registers and stored once the stage before it is
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

## Test and benchmark

```shell
loom-import-cxx checks.cxx --output=checks.loom
loom-test checks.loom --device=amdgpu
loom-benchmark checks.loom --device=amdgpu --benchmark=@bench_hc_down_4096_rows --measure=dispatch_complete
```

The benchmark cases time their oracle launches too; time the kernels alone with your own launcher.
