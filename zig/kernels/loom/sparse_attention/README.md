# Sparse prompt attention and its block indexer (Loom, gfx11)

The prompt attention core of a DeepSeek-style sparse-attention layer (Qwen3.8 Flash Next's full-attention layers),
in Loom's C++ frontend (hrx-loom 0.3.0), compiled for gfx1151, wave32. Three kernels with a plain buffer ABI:

- `idx_scores`: each row's block scores, the sum over indexer heads of relu(q . pooled key) / sqrt(128).
- `idx_select`: past `top` complete blocks, the `top` best blocks (ties: lower id) in block order, each as `ratio`
  keys, then the row's tail keys; else the row's first `ends` keys, unlisted.
- `sparse_attend`: softmax attention of 12 query heads over one bf16 KV head's listed (or first n) keys on bf16 WMMA
  with CUDA's arithmetic (TensorFold's `fn_attn_fused`): bf16 q, bf16 p into the values, fp32 sums, `tf_exp`, bf16 out.

`idx_scores` and `idx_select` give the same bytes as TensorFold's HIP kernels `tf_fn_idx_scores` and
`tf_fn_idx_select` (fp32 VALU, hipcc `-ffp-contract=off`). `sparse_attend` defines its own sum order (below);
TensorFold's `tf_fn_sparse_wmma12_bf16` repeats it for decode, verify and staged rows, byte for byte.

## Files

| file | job |
|---|---|
| `attend.cxx` | `sparse_attend`: a wave's part, the parts merged in order, the row's out |
| `wmma_tile.h` | gfx11 bf16 WMMA pieces: carriers, bf16 rounding, `tf_exp`, the 32-key tile, the merge step |
| `exp_exact.h` | hipcc's IEEE `expf` instruction sequence (the indexer's oracle) |
| `scores.cxx` | `idx_scores` |
| `select.cxx` | `idx_select` |
| `oracle.h` | reference arithmetic for the checks: serial replicas of the indexer, fp64 attention |
| `checks.cxx` | `LOOM_CHECK_CASE` / `LOOM_CHECK_BENCHMARK`: indexer bitwise, attention vs fp64 and its invariances |

## ABI

`sparse_attend(q, k, v, ids, count, sparse, out, id_stride, cap, heads, kv_heads, dims, scale, parts)`

- `q`: fp32 `[rows][heads][256]`; `k`, `v`: bf16 `[kv_heads][cap][256]`; `out`: fp32 `[rows][heads][256]`, each value
  a bf16.
- Row `r` attends `count[r]` keys: `ids[r * id_stride + i]` when `sparse[r] != 0`, else keys `0 .. count[r) - 1`.
- `heads = 12 kv_heads`, `dims = 256`, `parts = 4` (compiled constants; a launch with other values writes nothing).
  Launch `(kv_heads, rows)` workgroups of 128 work-items.

`idx_scores(q, pooled, complete, scores, stride, rows, heads, dims, blocks, top)`

- `q`: fp32 `[rows][4][128]`; `pooled`: bf16 `[blocks][128]`; writes `scores[r * stride + b]` (fp32) only where
  `complete[r] > top` and `b < complete[r]`. Launch `(ceil(blocks / 256), ceil(rows / 16))` workgroups of 256.

`idx_select(scores, complete, ends, keys, count, sparse, stride, key_stride, top, ratio)`

- Writes `keys[r * key_stride ..]`, `count[r]`, `sparse[r]` as above. Launch `(rows)` workgroups of 256.

Rows are independent: any row range is a launch on offset pointers, so a prompt cut into chunks or slices, or one
decode row, gives the same bytes as one launch.

## Sum order of `sparse_attend` (the bits)

`v_wmma_f32_16x16x16_bf16` (`mma` below) multiplies bf16 operands exactly and adds the 16 products and its fp32
accumulator in the hardware's own way. It is deterministic and each output element depends only on its own row, column
and accumulator (checked on gfx1151), so the order below fixes every bit. Every other operation is IEEE fp32, rounded
once, no contraction except the stated `fma`. The arithmetic is CUDA's `attn_chunks` (bf16 q, `tf_exp`, bf16 p into
the values, fp32 sums, chunks merged in order, bf16 out); the tiling is gfx11's.

- Parts: part `p` of a row's `n` keys is `[p n / 4, (p + 1) n / 4)`, in tiles of 32 keys from its start (list order).
- Scores of a tile: `q` rounded to bf16 (round to nearest even); key fragment f (keys 16 f .. 16 f + 15 of the tile),
  `s = mma(k_c, q_c, s)` from `+0` over the 16 dim chunks c in order (one chain), then `x = s * scale`; `-inf` past the
  tile's keys.
- Softmax step a tile, head h: `t` = max of the tile's `x`; if `t != -inf`: `next = max(m, t)`,
  `alpha = m == -inf ? 0 : tf_exp(m - next)` (else `next = m`, `alpha = 1`); `p = x == -inf ? 0 : tf_exp(x - next)`;
  `sum` = from `+0` the even keys 0, 2, .., 30 of the head in order, likewise the odd keys, then the two added;
  `l = fma(l, alpha, sum)`, `m = next`.
- Values a tile, per 16-dim block j: `o_j = mma(bf16(p) of keys 16..31, v, mma(bf16(p) of keys 0..15, v, o_j * alpha))`.
- Merge (CUDA's `fn_attn_merge`) in part order from `(mm, ll, oo) = (-inf, 0, +0)`: if `l_p > 0`:
  `next = max(mm, m_p)`, `a = mm == -inf ? 0 : tf_exp(mm - next)`, `b = tf_exp(m_p - next)` (else `a = 1`, `b = 0`);
  `oo = fma(o_p, b, oo * a)`, `ll = fma(l_p, b, ll * a)`, `mm = next`; `out = ll > 0 ? bf16(oo / ll) : 0`
  (correctly rounded divide).
- `tf_exp` is CUDA's: `j = round(x log2 e)` by the 1.5 * 2^23 trick, `f = x - j ln2` in two `fma`s, a degree-7
  Taylor polynomial in `fma`s, `2^j` as two exact factors; 0 below -104, +inf above 104, NaN kept. Maxima are exact, so
  their reduction order is free.
- Padding: heads 12..15 of the 16-column operands are `+0` queries and never stored; a key slot past the tile reads the
  tile's last key, its probability is `+0`.

Error against fp64 (gfx1151, synthetic bf16 rows of std 1, queries of std 3, 1.5k to 128k contexts): the same as a
host replica of CUDA's arithmetic, dominated by the bf16 output rounding. Indexer: lane l of the reference sums
d = l, l + 32, l + 64, l + 96 from +0, then the xor butterfly; `total = sum_h relu(dot_h)` in head order from +0;
`/ 11.3137085` (fp32 sqrt(128)), correctly rounded. Selection is integer: the set and its order are fixed by the scores.

None of this depends on the row count, the chunk cut, or a row's place in a launch.

## Geometry

`sparse_attend`: a workgroup per (KV head, row), 4 waves, wave p runs part p. The row's q is staged once in LDS as bf16
(`[16][264]`). A tile: each lane loads its keys' rows as A operands (keys on M, so each lane holds one head's 8 scores
of a fragment: max and sum in-lane plus one xor-16 swap); p goes to bf16 in registers as the A operand of the values;
per 16-dim block, lanes stage their key's 16 values transposed in the wave's LDS slot (two slots, alternating) and
read one dim's 16 keys back as the B operand. After the tiles, waves 1..3 hand their o, m and l to wave 0 through LDS
in part order and wave 0 writes the row. LDS 16.25 KiB, 256 VGPRs, 5 waves a SIMD.

`idx_scores`: a work-item a block (its 128 key values in registers), 16 rows of q in LDS a workgroup.

## Test and benchmark

```shell
loom-import-cxx checks.cxx --output=checks.loom
loom-test checks.loom --device=amdgpu
loom-benchmark checks.loom --device=amdgpu --benchmark=@bench_rows64_31k --measure=dispatch_complete
```

Cases: dense first rows, 700 keys, the 2,048-key switch, 4k and 9k sparse rows with tails; attention within 8e-2 +
|want| / 64 of an fp64 oracle (CUDA's bf16 arithmetic), and a repeat byte-equal. Benchmark cases are TensorFold's 64-row slice at 4k,
16k, 31.5k and 64k (their timings include input generation; time the kernels alone with your own launcher).
