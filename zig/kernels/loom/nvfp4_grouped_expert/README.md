# NVFP4 grouped-expert prompt kernels (Loom, gfx11 bf16 WMMA)

Routed-expert prompt GEMMs for mixture-of-experts layers whose experts are NVFP4: gate|up with SwiGLU, and down.
Written in Loom's C++ frontend (hrx-loom 0.3.0), compiled for gfx1151, wave32. No host framework is assumed:
every input is a plain buffer, the routing plan is two plain int32 arrays, shapes are kernel arguments checked
against the compiled constants.

## Files

| file | job |
|---|---|
| `nvfp4_decode.h` | NVFP4 storage and its exact decode to bf16 pairs (the format plug-in `Nvfp4Rows`) |
| `grouped_tile.h` | generic grouped-expert WMMA tile: LDS staging, fold order, any row decoder |
| `swiglu_exact.h` | `silu(g) * u` to bf16 with an explicit, reproducible `expf` sequence |
| `gate_up.cxx` | `nvfp4_expert_gate_up`: act = bf16(silu(gate) x up) |
| `down.cxx` | `nvfp4_expert_down`: fp32 out |
| `checks.cxx` | `LOOM_CHECK_CASE` / `LOOM_CHECK_BENCHMARK`: oracle, plan-cut invariance, Flash Next shapes |

## ABI

Both kernels take `(x, words, scales, globals, items, members, out, x_div, n, k)`:

- `x`: bf16 rows, `k` wide. Pair `p` reads row `p / x_div` (gate|up: token rows, `x_div` = top-k; down: pair rows, 1).
- `words`: u32 `[E][rows][k / 8]`, `scales`: e4m3 bytes `[E][rows][k / 16]`, `globals`: f32 per expert
  (`[E][2]` gate, up for gate|up; `[E]` for down). gate|up stacks each expert's gate rows, then its up rows (2n).
- `items`: int32 `[count][3]` = expert, first, rows; `members[first .. first + rows)` are that expert's pair ids.
- `out`: gate|up bf16 `[pairs][n]`, down fp32 `[pairs][n]`, written at row `members[i]`.
- Launch: 256 work-items; grid `(n / 64, ceil(max rows / 128), count)` for gate|up, `(n / 128, ..., count)` for down.
  Blocks past an item's rows exit at once.

Compiled shapes (override with `-DK4_K=`, `-DK4_N=`, `-DK4_DK=`, `-DK4_DN=` at import): gate|up K 2560, N 640;
down K 640, N 2560 (Qwen3.8 Flash Next). A launch whose `n`/`k` differ writes nothing.

## NVFP4 storage ("gfx1151 FP4-Local bake")

e2m1 codes, eight to a u32; nibble i holds input `order[i]` of its eight, `order = {0, 2, 4, 6, 1, 3, 5, 7}`, so
`(w >> 4p) & 0x000f000f` gives inputs 2p and 2p + 1. One e4m3 scale per 16 inputs. Decode is exact:
fp16(e2m1 x 2^-14) x fp16(e4m3) by `v_pk_mul_f16` (fp16 denormals on), widened to f32 and truncated to bf16
(exact: at most 5 significant bits). Eight VALU instructions a pair (4 bit placement, 1 multiply, 2 widen, 1 pack)
plus 5 a 16-input scale; the bit placement and pack are gfx11 Low assembly. The epilogue multiplies the sums by
`global x 2^14`.

## Tile geometry

One workgroup (256 work-items, 8 waves) computes up to 128 rows of one plan item against 128 weight rows of its
expert, over K in stages of 32 inputs double-buffered in LDS (80-byte row pitch, 40 KiB a workgroup). Work-item t
decodes half a stage of weight row t % 128 and copies half a stage of x row t / 2 (rows past the item repeat a live
row; their sums are never stored). Wave w owns tile columns 16w..16w+15 for every live 16-row fragment. The fragment
count is the same for all waves, so the K loop is specialized on it (1..8): no dead products, no data-dependent
branches, barriers stay workgroup-uniform. Fetches run D stages ahead (gate|up 3, down 2), clamped at the last stage;
a stage's decode and LDS store follow the previous stage's products, with one barrier a stage. gate|up's up waves
hand their scaled sums to the gate waves through the freed LDS for the SwiGLU epilogue.

## Sum order (the bits)

Per fold of 128 inputs: `d` = chained `v_wmma_f32_16x16x16_bf16` over the fold's eight K16 slices in K order,
starting from +0; `acc += d` per fold, in K order; out = `acc x (global x 2^14)`. The order depends on K and the fold
only, never on the row count, the plan cut, or a row's place in its tile: a row's bytes are the same in any plan.

## SwiGLU

`silu(g) x u` uses hipcc's IEEE `expf` sequence for `exp(-g)` (two-part log2(e) split, round to even, `v_exp_f32`,
ldexp, 0 / +inf past the range), a correctly rounded divide, one product and one round-to-nearest-even to bf16
(NaN to 0x7fc0). With the sum order above this reproduces the ROCm HIP reference kernels byte for byte.

## Other formats

`grouped_tile<Dec, K, D>` takes any decoder with `Weights`, `Raw`, `fetch(weights, row, stage, half)` (one
work-item's 16 inputs of a 32-input stage, `half` 0 or 1) and `decode<Half>(raw)` (those 16 inputs as exact bf16,
8 pairs in K order). A GGUF or MXFP8 decoder plugs in there; scales that are not exact in bf16 belong in the
epilogue, not the decode.

## Test and benchmark

```shell
loom-import-cxx checks.cxx --output=checks.loom
loom-test checks.loom --device=amdgpu
loom-benchmark checks.loom --device=amdgpu --benchmark=@bench_gate_up_4096_tokens --measure=dispatch_complete
```

The benchmark cases time their setup and oracle launches too; time the kernels alone with your own launcher.
