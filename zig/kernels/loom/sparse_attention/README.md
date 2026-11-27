# Sparse prompt attention and its block indexer (Loom, gfx11, fp32 VALU)

The prompt attention core of a DeepSeek-style sparse-attention layer (Qwen3.8 Flash Next's full-attention layers),
in Loom's C++ frontend (hrx-loom 0.3.0), compiled for gfx1151, wave32. Three kernels with a plain buffer ABI:

- `idx_scores`: each row's block scores, the sum over indexer heads of relu(q . pooled key) / sqrt(128).
- `idx_select`: past `top` complete blocks, the `top` best blocks (ties: lower id) in block order, each as `ratio`
  keys, then the row's tail keys; else the row's first `ends` keys, unlisted.
- `sparse_attend`: softmax attention of 12 query heads over one bf16 KV head's listed (or first n) keys, in 16
  parts merged in part order, all in one launch.

Every output is the same bytes as the HIP reference kernels it replaces (TensorFold `tf_fn_idx_scores`,
`tf_fn_idx_select`, `tf_fn_sparse_lanes12_bf16` + `tf_fn_sparse_merge`, hipcc `-ffp-contract=off`); see Sum order.

## Files

| file | job |
|---|---|
| `attend.cxx` | `sparse_attend`: parts, softmax steps and the merge |
| `lanes_asm.h` | gfx11 Low blocks: the score chain (DPP row_share q operands) and the value chain (readlane p) |
| `exp_exact.h` | hipcc's IEEE `expf` instruction sequence |
| `scores.cxx` | `idx_scores` |
| `select.cxx` | `idx_select` |
| `oracle.h` | serial replicas of the reference arithmetic, one work-item an output |
| `checks.cxx` | `LOOM_CHECK_CASE` / `LOOM_CHECK_BENCHMARK`: every kernel bitwise against its oracle |

## ABI

`sparse_attend(q, k, v, ids, count, sparse, out, id_stride, cap, heads, kv_heads, dims, scale, parts)`

- `q`: fp32 `[rows][heads][256]`; `k`, `v`: bf16 `[kv_heads][cap][256]`.
- Row `r` attends `count[r]` keys: `ids[r * id_stride + i]` when `sparse[r] != 0`, else keys `0 .. count[r) - 1`.
- `out`: fp32 `[rows][heads][256]`. `heads = 12 kv_heads`, `dims = 256`, `parts = 16` (compiled constants; a launch
  with other values writes nothing). Launch `(kv_heads, rows)` workgroups of 512 work-items (one wave a part).

`idx_scores(q, pooled, complete, scores, stride, rows, heads, dims, blocks, top)`

- `q`: fp32 `[rows][4][128]`; `pooled`: bf16 `[blocks][128]`; writes `scores[r * stride + b]` (fp32) only where
  `complete[r] > top` and `b < complete[r]`. Launch `(ceil(blocks / 256), ceil(rows / 16))` workgroups of 256.

`idx_select(scores, complete, ends, keys, count, sparse, stride, key_stride, top, ratio)`

- Writes `keys[r * key_stride ..]`, `count[r]`, `sparse[r]` as above. Launch `(rows)` workgroups of 256.

Rows are independent: any row range is a launch on offset pointers, so a prompt cut into chunks or slices gives the
same bytes as one launch.

## Sum order (the bits)

The reference is scalar fp32 with every product and sum rounded (no FMA contraction); the kernels repeat each
operation in the same order, only on a different lane layout:

- Score of head h and key j: `s = 0; s = s + (q_d * scale) * k_d` for d = 0 .. 255 in order.
- A part is keys `[p n / 16, (p + 1) n / 16)`, in tiles of 32 from its start. Per tile and head:
  `top = max(tile max, m)`, `e_j = expf(s_j - top)` (0 past the tile's keys), `sum` = the xor butterfly over masks
  16, 8, 4, 2, 1 of the 32 lanes' `e`, `f = expf(m - top)`, `l = l f + sum`, `m = top`, `o = o f`, then
  `o = o + e_j v_j` for the tile's keys in order.
- Merge: `top = max_p m_p`, `w_p = expf(m_p - top)` (0 for an empty part), `total = sum_p l_p w_p`,
  `acc = sum_p o_p w_p` in part order from +0, `out = total > 0 ? acc / total : 0` (correctly rounded divide).
- `expf` is hipcc's sequence: two-part `x log2(e)`, round to even, `v_exp_f32`, `ldexp`, 0 below -103.97, +inf above
  88.72. Maxima are exact, so their reduction order is free.
- Indexer: lane l of the reference sums d = l, l + 32, l + 64, l + 96 from +0, then the xor butterfly;
  `total = sum_h relu(dot_h)` in head order from +0; `/ 11.3137085` (fp32 sqrt(128)), correctly rounded.
- Selection is integer: the set and its order are fixed by the scores, so any correct radix select is exact.

None of this depends on the row count, the chunk cut, or a row's place in a launch.

## Geometry

`sparse_attend`: a workgroup per (KV head, row), wave p computes part p. Lane = key for the scores: 12 head chains a
lane, 16 dims a step, the next step's key words in flight; q is staged scaled in LDS and fed to `v_mul_f32_dpp`
by row_share (lane i of each 16-lane row holds q_d0+i), so no LDS broadcast per product. Lane = 8 dims for the
values: `v_readlane` gives `e_j` from the key's lane; four V rows in flight. The merge runs in LDS (2 heads a round,
512 work-items = 2 heads x 256 dims). 192 VGPRs, 47.5 KiB LDS, 8 waves a SIMD.

`idx_scores`: a work-item a block (its 128 key values in registers), 16 rows of q in LDS a workgroup.

## Test and benchmark

```shell
loom-import-cxx checks.cxx --output=checks.loom
loom-test checks.loom --device=amdgpu
loom-benchmark checks.loom --device=amdgpu --benchmark=@bench_rows64_31k --measure=dispatch_complete
```

Cases: dense first rows, 700 keys, the 2,048-key switch, 4k and 9k sparse rows with tails; benchmark cases are
TensorFold's 64-row slice at 4k, 16k, 31.5k and 64k (their timings include input generation; time the kernels
alone with your own launcher).

## Limits of this order

The reference's rounded scalar chains rule out WMMA (its accumulation order is the hardware's), so this kernel is
VALU-bound by construction: 2 VALU operations a multiply-add (the DPP multiply cannot dual-issue), about 7 TFLOP/s on
gfx1151, 12% of the bf16 WMMA peak. A WMMA kernel needs a new order adopted by the reference too.
