# GGUF on the HIP engine (qwen35, Strix Halo)

`tensorfold-native serve <dir>` takes a directory holding a llama.cpp GGUF of a qwen35-family model (the 27B so far),
the release's `config.json`, and its tokenizer files. The GGUF is read in place (mmap, no conversion step): the
projections stay ggml blocks on the device and the kernels decode them.

Checkpoint used: [unsloth/Qwen3.8-27B-GGUF](https://huggingface.co/unsloth/Qwen3.8-27B-GGUF)
`Qwen3.8-27B-UD-Q4_K_XL.gguf` (17.56 GB, 866 tensors, MTP block included). Its mix by bytes: q5_K 45.2%, iq4_xs 17.8%,
q4_K 17.4%, q6_K 16.3%, iq4_nl 1.6%, q8_0 0.7%, q3_K 0.7%, iq3_s 0.2%, f32 vectors. The embedding is q4_K, the head
q6_K. Every Unsloth "UD" K-quant of this model mixes the iq types, so all eight have kernels. The tokenizer and chat
template come from the Qwen/Qwen3.8-27B release: same 248077 tokens as the GGUF's `tokenizer.ggml.tokens`, which only
adds [PAD] ids.

## What llama.cpp's format means here (references read as data)

ggml-org/llama.cpp 8bb3cf0d: `ggml/src/gguf.cpp` (container), `ggml/src/ggml-common.h` (block structs, `kvalues_iq4nl`,
`iq3s_grid`), `ggml/src/ggml-quants.c` (`dequantize_row_*`), `conversion/qwen.py` (`Qwen3NextModel`,
`_LinearAttentionVReorderBase`), `src/models/qwen35.cpp`, `gguf-py/gguf/tensor_mapping.py`,
`ggml/src/ggml-cuda/gated_delta_net.cu`.

- Names: `zig/src/families/qwen3_5/weights/gguf_names.zig` maps llama.cpp's to the MLX ones (`blk.N.attn_qkv` ->
  `linear_attn.in_proj_qkv`, `ssm_alpha`/`ssm_beta` -> `in_proj_a`/`in_proj_b`, `blk.64.nextn.*` -> the `mtp.` head, ...).
- Transforms: norms except `linear_attn.norm` are stored as 1 + w (as MLX bakes them); `ssm_a` is -exp(A_log), taken
  back as log(-a) in fp64; conv1d is [C, K].
- Value heads: llama.cpp reorders the linear attention's 48 value heads to a tiled order (head j * 16 + g reads key head
  g; the CUDA kernel takes `h % key_heads`). The out projection's columns cannot be un-permuted inside 256-element
  superblocks, so the model runs that order: `Spec.tiled_heads`, bit 16 of the recurrence kernels' key-head argument.
  Checked against the bf16 release: tiled order correlation 0.999+ on every linear-attention tensor, grouped ~0.05.

## Kernels (`zig/kernels/hip/quant/gguf.hpp`, `gguf.hip`)

Each decoder gives a 16-element piece as unsigned codes u with fp32 s, mn such that ggml's value is exactly s u + mn.

- Product (decode and lane rounds, any row count): WMMA over the exact integer codes (BF16 128 + u where u < 128,
  kvalues as BF16 for iq4 by an LDS table, byte converts for q8_0), one fold per (s, mn) group on precomputed x sums;
  four waves split the superblocks and add in wave order, so a row's bits never depend on the round's rows.
- Prompt: `prompt_dq_tile` with a GGUF hook (weights = ggml's fp32 value rounded once to BF16), 256 x 128 blocks;
  `gguf_chain` gives the same sums for prompts of up to 128 rows, so a prompt's bits never depend on its cut.
- Row gather (embedding): ggml's fp32 values (a zero may carry the other sign), rounded to the activation kind.

## Verification

- Host reference `core/quant/gguf.zig` and an independent numpy one (`tools/strix/gguf_ref.py`) agree bit for bit on
  rows of every type in the real file; the numpy dequant correlates 0.987-0.9998 with the bf16 release's weights.
- `zig build hip-kernel-test -Dhip-test-filter=GGUF`: every type, gather exact, product within 1e-5 of fp64 with each
  row's bits equal at 1-128 rows, prompt within 1e-5 of fp64 of its BF16 weights with its bits equal at 1-300 rows.
- `tf-qwen35-test check --only invariants`: 47 passed (MTP drafting from the GGUF's own head included).
- bench.sh exactness: alone == 4 streams == draft:false on every prompt.
- Quality gate vs ~/tfq/ref32 (8 x 4096, prompt path): PPL 3.0656 (bf16 3.0359), KL 0.0290, top-1 96.67%
  (floors: MLX4 KL 0.112, EXL3 0.116).
- Same weights: `quality_ref.py ref <gguf dir> --fp32` runs transformers' Qwen3_5 layers in fp32 on the GGUF
  dequantized by gguf_ref.py (llama.cpp's conversion undone); the engine against it on 2 x 4096 tokens: KL 0.00040,
  top-1 99.40%.

## Measured (2026-10-18, locked, gfx1151, DFlash2 drafter, `--context 40960`, sparkbench)

| checkpoint | C1 prose / code | C8 prose / code | prefill 8k / 32k |
|---|---|---|---|
| GGUF UD-Q4_K_XL (16.3 GiB), strix/gguf | 41.9 / 97.9 (prose5 37.1) | 125.1 / 157.9 | 663 / 577 |
| GGUF, first correct build | 31.1 / 82.8 | - | 660 / - |
| EXL3 3.5bpw, same session (main 2efa512c / strix/gguf) | 61.6 / 137.5, 62.6 / 137.5 | 204 / 221 (main, reported) | 650 (reported) |
| MLX4, same session (main / strix/gguf) | 43.6 / 116.8, 42.8 / 116.6 | 150 / 159 (reported) | ~570 (reported) |
| DGX Spark | 58.3 / 136.9 | 220.6 / 323.2 | 1,793 |

Kernel microbench (`TF_GGUF_BENCH=1 zig build hip-kernel-test -Dhip-test-filter="GGUF timings"`, 16 rows, weights
from DRAM where they do not fit the caches): q5_K gate/down 185-190 GB/s, q6_K head 194 GB/s, iq4_xs 160-185 GB/s,
iq4_nl 125 GB/s (of 240); prompt tile 2048 rows 34-46 TFLOPS (q5_K/iq4 gate and down 44-46, o/out 34-35; of 57).

The GGUF moves 16.3 GiB a forward against EXL3's 12, so at equal efficiency its C1 is ~0.74x EXL3's; it reaches 0.65x.
On this box EXL3 3.5bpw stays the fastest 27B; the GGUF is the better-quality one (KL 0.029 vs 0.116).

Opt-in: the EXL3 and MLX4 replies (sparkbench prose and code, C1) are byte-identical between main 2efa512c and
strix/gguf; the shared changes are a new format tag and the recurrence's key-head bit, clear for every other model.
The 35B-A3B was not rerun.

## Not done / next

- C1: profile a round (rocprofv3) for what is outside the products; iq4_nl (18-byte blocks) at 125 GB/s; the q4_K/q5_K
  half-waves load the same bytes (split and exchange by permlanex16); zero-copy (hipHostRegister on the mmap) instead
  of the upload copy.
- Rounds (17-128 rows, C8): the product is 45-55 GB/s at 128 rows; a round tile sharing x in LDS across column waves.
- Prompt: o/out shapes at 34 TFLOPS; the 256 x 128 block only.
- Other GGUF types (q2_K, iq2/iq3_xxs, iq1, q4_0/q4_1, mxfp4) are refused by name; other families (A3B MoE experts,
  Flash Next) have no GGUF mapping yet; config.json and tokenizer files must sit beside the .gguf.
