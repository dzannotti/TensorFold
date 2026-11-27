# Sparse prompt attention and its block indexer (Loom, gfx11)

The prompt attention core of a DeepSeek-style sparse-attention layer (Qwen3.8 Flash Next's full-attention layers),
in Loom's C++ frontend (hrx-loom 0.3.0), compiled for gfx1151, wave32. Three kernels with a plain buffer ABI:

- `idx_scores`: each row's block scores, the sum over indexer heads of relu(q . pooled key) / sqrt(128).
- `idx_select`: past `top` complete blocks, the `top` best blocks (ties: lower id) in block order, each as `ratio`
  keys, then the row's tail keys; else the row's first `ends` keys, unlisted.
- `sparse_attend`: softmax attention of 12 query heads over one bf16 KV head's listed (or first n) keys on bf16 WMMA
  with exact products, in 16 parts; `sparse_merge` combines the parts in part order.

`idx_scores` and `idx_select` give the same bytes as TensorFold's HIP kernels `tf_fn_idx_scores` and
`tf_fn_idx_select` (fp32 VALU, hipcc `-ffp-contract=off`). `sparse_attend` + `sparse_merge` define their own sum order
(below); TensorFold's `tf_fn_sparse_wmma12_bf16` + `tf_fn_sparse_merge` repeat it for decode rows, byte for byte.

## Files

| file | job |
|---|---|
| `attend.cxx` | `sparse_attend`: the tile loop of a part's wave pair, the part's o, m and l out |
| `merge.cxx` | `sparse_merge`: the parts combined in part order |
| `wmma_tile.h` | gfx11 bf16 WMMA pieces: carriers, exact three-part splits, score and value halves, the softmax step |
| `exp_exact.h` | hipcc's IEEE `expf` instruction sequence |
| `scores.cxx` | `idx_scores` |
| `select.cxx` | `idx_select` |
| `oracle.h` | reference arithmetic for the checks: serial replicas of the indexer, fp64 attention |
| `checks.cxx` | `LOOM_CHECK_CASE` / `LOOM_CHECK_BENCHMARK`: indexer bitwise, attention vs fp64 and its invariances |

## ABI

`sparse_attend(q, k, v, ids, count, sparse, parts_o, parts_ms, id_stride, cap, heads, kv_heads, dims, scale, parts)`

- `q`: fp32 `[rows][heads][256]`; `k`, `v`: bf16 `[kv_heads][cap][256]`.
- Row `r` attends `count[r]` keys: `ids[r * id_stride + i]` when `sparse[r] != 0`, else keys `0 .. count[r) - 1`.
- `parts_o`: fp32 `[rows][heads][16][256]`, `parts_ms`: fp32 `[rows][heads][16][2]` (m, l). `heads = 12 kv_heads`,
  `dims = 256`, `parts = 16` (compiled constants; a launch with other values writes nothing). Launch
  `(kv_heads, rows, 4)` workgroups of 256 work-items (4 parts, a wave pair each).

`sparse_merge(parts_o, parts_ms, out, heads, dims, parts)`: `out` fp32 `[rows][heads][dims]`; launch `(heads, rows)`
workgroups of 256.

`idx_scores(q, pooled, complete, scores, stride, rows, heads, dims, blocks, top)`

- `q`: fp32 `[rows][4][128]`; `pooled`: bf16 `[blocks][128]`; writes `scores[r * stride + b]` (fp32) only where
  `complete[r] > top` and `b < complete[r]`. Launch `(ceil(blocks / 256), ceil(rows / 16))` workgroups of 256.

`idx_select(scores, complete, ends, keys, count, sparse, stride, key_stride, top, ratio)`

- Writes `keys[r * key_stride ..]`, `count[r]`, `sparse[r]` as above. Launch `(rows)` workgroups of 256.

Rows are independent: any row range is a launch on offset pointers, so a prompt cut into chunks or slices, or one
decode row, gives the same bytes as one launch.

## Sum order of `sparse_attend` (the bits)

`v_wmma_f32_16x16x16_bf16` (`mma` below) multiplies bf16 operands exactly and adds the 16 products and its fp32
accumulator in the hardware's own way: aligned to the largest addend with about fp32 width, then rounded. It is
deterministic and each output element depends only on its own row, column and accumulator (checked on gfx1151), so
the order below fixes every bit. Every other operation is IEEE fp32, rounded once, no contraction except the stated
`fma`.

- Exact splits: `x = hi + mid + lo`, `hi` = `x` truncated to bf16, `mid` = `(x - hi)` truncated, `lo = x - hi - mid`;
  each part is a bf16 (fp32 has 24 significant bits, a part 8), `+0` where it is zero. `q` is split after `q * scale`.
  Values below about 2^-110 lose their low parts (bf16 subnormal inputs flush), far below the outputs' resolution.
- Parts: part `p` of a row's `n` keys is `[p n / 16, (p + 1) n / 16)`, in tiles of 16 keys from its start.
- Scores of a tile (key j, head h), the 256 dims as chunks c = 0..15 of 16 dims, side 0 = chunks 0..7, side 1 = 8..15:
  - `d_c = mma(k_c, q_c.hi, +0)`; `H_s = d_first + ... + d_last` of the side's chunks, in chunk order;
  - `C_s` = one chain from `+0` over the side's chunks in order: `mma(k_c, q_c.mid, .)` then `mma(k_c, q_c.lo, .)`;
  - `x_s = H_s + C_s`, `s = x_0 + x_1`.
- Softmax step a tile, head h: `top = max(m, max_j s_j)` over the tile's keys; `e_j = expf(s_j - top)` (`+0` past the
  tile's keys); `sum` = keys 0, 2, .., 14 added in order, likewise 1, 3, .., 15, then the two added; `f = expf(m - top)`;
  `l = l f + sum`; `m = top`.
- Values a tile, per 16-dim block: `d = mma(e.lo, v, +0)`, `d = mma(e.mid, v, d)`, `d = mma(e.hi, v, d)` (e split as
  above), then `o = fma(o, f, d)`.
- Merge: `top = max_p m_p`, `w_p = expf(m_p - top)` (`0` for an empty part), `total = sum_p l_p w_p`,
  `acc = sum_p (o_p w_p)` in part order from `+0`, `out = total > 0 ? acc / total : 0` (correctly rounded divide).
- `expf` is hipcc's sequence: two-part `x log2(e)`, round to even, `v_exp_f32`, `ldexp`, 0 below -103.97, +inf above
  88.72. Maxima are exact, so their reduction order is free.
- Padding: keys past a tile's end enter as `+0` K and V rows and `+0` probabilities; heads 12..15 of the 16-row
  operands are `+0` queries and `+0` probabilities.

Error against fp64 (gfx1151, synthetic bf16 rows of std 1, queries of std 3): about half the previous serial fp32
kernel's rms error and below its maximum (TensorFold receipts). Indexer: lane l of the reference sums d = l, l + 32,
l + 64, l + 96 from +0, then the xor butterfly; `total = sum_h relu(dot_h)` in head order from +0; `/ 11.3137085`
(fp32 sqrt(128)), correctly rounded. Selection is integer: the set and its order are fixed by the scores.

None of this depends on the row count, the chunk cut, or a row's place in a launch.

## Geometry

`sparse_attend`: a workgroup per (KV head, row, 4 parts), 8 waves, wave `2i + s` is side `s` of its `i`-th part. The
query's three parts are staged once in LDS (`[3][16][264]` bf16). A tile: each lane loads its key's half row as A
operands (lane = key; chunks 1..7 a pipelined loop), the pair exchanges `x_s` through its LDS slot (one barrier), both
waves run the same softmax step, and each does the values of its 128 dims: lanes of half h stage the 16 keys of a
16-dim block into the slot and read back one dim's 16 keys as the B operand. Every wave runs the longest part's tile
count, so barriers stay workgroup-uniform; a part's extra tiles add `+0` and change nothing. LDS 33.5 KiB, 172 VGPRs,
6 waves a SIMD.

`idx_scores`: a work-item a block (its 128 key values in registers), 16 rows of q in LDS a workgroup.

## Test and benchmark

```shell
loom-import-cxx checks.cxx --output=checks.loom
loom-test checks.loom --device=amdgpu
loom-benchmark checks.loom --device=amdgpu --benchmark=@bench_rows64_31k --measure=dispatch_complete
```

Cases: dense first rows, 700 keys, the 2,048-key switch, 4k and 9k sparse rows with tails; attention within 1e-4
(absolute + relative) of an fp64 oracle, and a row split and a repeat byte-equal. Benchmark cases are TensorFold's 64-row slice at 4k,
16k, 31.5k and 64k (their timings include input generation; time the kernels alone with your own launcher).
