# Skinny MXFP8 products (Loom, gfx11 bf16 WMMA)

Decode and verify products `y = x . W^T` for 1 to 64 rows of bf16 activations against MXFP8 weights (e4m3 codes,
e8m0 block scales of 32), written in Loom's C++ frontend (hrx-loom 0.3.0) for gfx1151, wave32. Every weight is read
once a launch whatever the row count, and every row's bytes are those of the one-row launch.

## Files

| file | job |
|---|---|
| `mxfp8_cols.h` | MXFP8 decode piece: 16 e4m3 codes to one WMMA fragment of exact bf16, e8m0 to its f32 multiplier |
| `skinny_tile.h` | generic skinny tile: geometry, block placement, the K loop in the sum order below |
| `skinny_out.h` | epilogue: the column lanes' tree, the non-finite repair, bf16 rounding, the four-way sides |
| `skinny_mxfp8.cxx` | the kernels `skinny_mxfp8_k<K>_f<F>_t<NT>_<out>` |
| `checks.cxx` | `LOOM_CHECK_CASE`: bytes against a dot2 replica of the order, row-count identity |

## ABI

`skinny_mxfp8_k<K>_f<F>_t<NT>_{f32,bf16}(m, n, w, e, x, out)` and `..._x4(m, n, c1, c2, c3, w, e, x, o0, o1, o2, o3)`:

- `w`: e4m3 bytes `[n][K]`; `e`: e8m0 bytes `[n][K / 32]`; `x`: bf16 `[m][K]`, 32-byte aligned rows.
- `out`: fp32 or bf16 `[m][n]`; x4: bf16 `[m][c1]`, `[m][c2 - c1]`, `[m][c3 - c2]`, `[m][n - c3]`, cuts multiples of 16.
- `1 <= m <= 16 F`; a launch out of range writes nothing. Codes 0x7f / 0xff (NaN) are refused by the caller.
- Launch: `32 L` work-items, grid = the sum over sides of `ceil(ceil(width / 8) /
  (2 T NT)) T` blocks (L, T below).

## Sum order (the bits)

The order of TensorFold's HIP stream tile `fp4l_stream` (MXFP8) and of the Loom GEMV `mxfp8_gemv`, so the bytes equal
theirs. It depends on K and the column alone, never on m, F, NT, the block or the row's place:

- L = the most lanes, up to 32, that split K's chunks of 32 evenly (16 at K 2560, 32 at 6144, 4 at 640); T = K / 32 / L.
- Column lane l = 0..L-1 owns chunks `q = ((t + rot) mod T) L + l` for t = 0..T-1, `rot = (c' / 8) mod T`, c' the
  column within its side. Per chunk: `d` = the bf16 dot products of its 32 inputs chained pair by pair from +0 (two
  chained `v_wmma_f32_16x16x16_bf16`, each the same chain as eight `v_dot2_f32_bf16`), weights entering as e4m3 x 2^-8;
  then `acc_l = fma(d, 2^(E - 119), acc_l)` from +0.
- Lanes add by the xor tree: for o = L/2, L/4, .., 1: `acc_l = acc_l + acc_(l ^ o)`; the sum is lane 0's.
- A non-finite sum when some x of the row is non-finite is replaced by the f64 sum (w_i x_i in i order, no fma),
  rounded once to f32. bf16 outputs round to nearest even, any NaN to 0x7fc0.

## Geometry

Weights are the WMMA A operand (a lane's 16 codes of one column), x rows the B operand (padding rows enter as +0), so
an accumulator lane holds one row by 8 columns: elements 0-3 are columns 0-7 of the fragment, 4-7 columns 8-15. A block
has one wave a column lane and NT column fragments a wave. Block p of a tile takes the groups of 8 columns p + T j
(j = 0..2 NT - 1), so all its columns share rotation p; fragment nt holds groups j = 2 nt (rows 0-7) and 2 nt + 1
(rows 8-15), and a wave's x fragments of a chunk serve all of them. The block's scales are staged once into LDS as
multipliers; the lane tree runs through LDS one fragment at a time. Weights of chunk t + 1 load while chunk t computes.

## Test and benchmark

```shell
loom-import-cxx --I=. checks.cxx --output=checks.loom
loom-test checks.loom --device=amdgpu
```
