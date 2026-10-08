# Triton AOT for gfx1151: results (rocm-triton)

Build (dev image, no GPU needed): `PYTHONPATH=src python -B tools/zig/flashnext_aot.py build --target hip
--spec zig/tests/cuda/flashnext/kernels.json --spec zig/tests/cuda/flashnext/kernels_int4ar.json --out <dir>`
-> 454 spec entries, 415 distinct hsaco (11 MB), 0 failures.

Checks: `tools/rocm/triton_parity.py hash` (454/454 JIT hashes in aot.json, hsaco byte-equal);
`tools/rocm/triton_parity.py run --bench 1` (175 wrapper launches, all bit-equal; timings below);
`tools/rocm/fp8_check.py` (fp8e4nv encode/decode == torch.float8_e4m3fn).

## Per function (vgpr / spills from the code object metadata)

| fn | variants | warps | max LDS | max vgpr | variants spilling | max spills | max private bytes |
|---|---|---|---|---|---|---|---|
| _add_streams | 1 | [2] | 0 | 16 | 0 | 0 | 0 |
| _attn_gate | 2 | [2] | 32 | 30 | 0 | 0 | 0 |
| _attn_prep | 6 | [2] | 8 | 51 | 0 | 0 | 0 |
| _attn_prep8 | 6 | [2] | 8 | 55 | 0 | 0 | 0 |
| _b16mm | 85 | [4] | 40960 | 256 | 49 | 298 | 960 |
| _b16mm_ks | 22 | [4] | 24576 | 256 | 22 | 211 | 520 |
| _chunks | 4 | [4] | 32768 | 256 | 4 | 571 | 2220 |
| _chunks8 | 4 | [4] | 16384 | 256 | 4 | 678 | 2600 |
| _chunks_multi | 8 | [4] | 32768 | 256 | 8 | 571 | 2224 |
| _embed | 2 | [2] | 0 | 10 | 0 | 0 | 0 |
| _fp4mm | 54 | [4, 8] | 10240 | 213 | 0 | 0 | 0 |
| _fp4mm_ks | 8 | [8] | 6144 | 160 | 0 | 0 | 0 |
| _hc_act | 2 | [4] | 64 | 29 | 0 | 0 | 0 |
| _hc_mix | 1 | [2] | 32 | 89 | 0 | 0 | 0 |
| _hc_normed | 1 | [4] | 64 | 16 | 0 | 0 | 0 |
| _hc_up_mix | 4 | [4] | 16384 | 256 | 4 | 2089 | 4532 |
| _hc_wb_norm | 7 | [2] | 8 | 82 | 0 | 0 | 0 |
| _hc_writeback | 7 | [2] | 8 | 53 | 0 | 0 | 0 |
| _merge | 4 | [4] | 64 | 82 | 0 | 0 | 0 |
| _merge_multi | 4 | [4] | 64 | 83 | 0 | 0 | 0 |
| _moe_partial | 4 | [2] | 0 | 46 | 0 | 0 | 0 |
| _ple_conv | 3 | [4] | 0 | 40 | 0 | 0 | 0 |
| _ple_embed_bf16 | 2 | [1] | 0 | 10 | 0 | 0 | 0 |
| _ple_gate | 1 | [4] | 16 | 29 | 0 | 0 | 0 |
| _pool | 9 | [1] | 0 | 55 | 0 | 0 | 0 |
| _pool_multi | 2 | [1] | 0 | 55 | 0 | 0 | 0 |
| _prep_multi | 4 | [2] | 8 | 55 | 0 | 0 | 0 |
| _reduce | 16 | [4] | 0 | 256 | 2 | 8 | 20 |
| _rmsnorm | 2 | [4, 8] | 2048 | 92 | 0 | 0 | 0 |
| _router | 9 | [4] | 40960 | 256 | 7 | 370 | 1112 |
| _scores | 4 | [4] | 256 | 99 | 0 | 0 | 0 |
| _scores_rows | 8 | [4] | 256 | 174 | 0 | 0 | 0 |
| _select | 96 | [16] | 32768 | 256 | 18 | 895 | 3584 |
| _select_tiles | 12 | [8, 16] | 8192 | 256 | 6 | 83 | 336 |
| _shift_windows | 9 | [4] | 0 | 35 | 0 | 0 | 0 |
| _topk_rows | 2 | [4] | 16 | 37 | 0 | 0 | 0 |

## Timings (us a launch, best of 5 alternated batches of 20, prod sharing the GPU: relative only)

`aot`: the AOT hsaco (no tt.pointer_range, plain global loads); `jit`: the JIT's own binary for the same small
tensors (buffer ops, tt.pointer_range 32), both launched through the same ctypes ABI. ~6 us is the launch floor.

```
  _b16mm           070c707e70 grid (1, 40, 4)       M=16 x_stride=6144           aot    113.4  jit    122.3  spills 0
  _b16mm           78c38b75a2 grid (1, 40, 4)       M=3 x_stride=6144            aot    101.5  jit    111.3  spills 0
  _b16mm           9801cb1b1a grid (1, 40, 4)       M=1 x_stride=6144            aot    100.9  jit    108.1  spills 0
  _router          dcd6749334 grid (1, 17)          M=16 x_stride=2560           aot     76.8  jit     36.3  spills 174
  _router          e4f88bc8c7 grid (1, 17)          M=3 x_stride=2560            aot     73.2  jit     35.8  spills 174
  _router          f6346bd14f grid (1, 17)          M=1 x_stride=2560            aot     73.1  jit     35.6  spills 174
  _fp4mm           45b29bd5ce grid (1, 10, 1)       M=16 x_stride=2560           aot     36.5  jit     43.2  spills 0
  _fp4mm           b4c6e38c08 grid (1, 20, 1)       M=16 x_stride=2560           aot     36.3  jit     43.3  spills 0
  _fp4mm           c29e5d18fb grid (1, 20, 1)       M=1 x_stride=2560            aot     35.4  jit     42.7  spills 0
  _fp4mm           6272409460 grid (1, 20, 1)       M=3 x_stride=2560            aot     34.6  jit     42.5  spills 0
  _fp4mm           08be46d07b grid (1, 10, 1)       M=3 x_stride=2560            aot     34.4  jit     42.3  spills 0
  _fp4mm           e24e28673e grid (1, 10, 1)       M=1 x_stride=2560            aot     34.2  jit     42.1  spills 0
  _b16mm           42fb4e298a grid (1, 40, 4)       M=16 x_stride=3072           aot     32.1  jit     33.4  spills 0
  _b16mm           c530cf0e64 grid (1, 40, 4)       M=3 x_stride=3072            aot     30.7  jit     31.0  spills 0
  _b16mm           3dc4bb2363 grid (1, 40, 4)       M=1 x_stride=3072            aot     30.3  jit     31.0  spills 0
  _shift_windows   28a8fffc71 grid (36, 40)         keep=1 OLD_L=30720 NEW_L=131840 NEW_ROW=16480 aot     25.4  jit     10.4  spills 0
  _ple_gate        50970e07de grid (1,)                                          aot     21.6  jit     22.1  spills 0
  _b16mm           e27b0e21f4 grid (1, 40, 4)       M=16 x_stride=2560           aot     19.5  jit     19.1  spills 0
  _b16mm           7d8bec6309 grid (1, 40, 4)       M=3 x_stride=2560            aot     18.7  jit     17.8  spills 0
  _b16mm           05eea1cfe9 grid (1, 40, 4)       M=1 x_stride=2560            aot     18.6  jit     17.8  spills 0
  _hc_writeback    c0e9618359 grid (1, 10)          RS=2560                      aot     16.5  jit     16.5  spills 0
  _shift_windows   473137d929 grid (36, 40)         keep=3 OLD_L=30720 NEW_L=131840 NEW_ROW=16480 aot     15.9  jit     15.8  spills 0
  _chunks          353a2357be grid (3, 2, 5)                                     aot     15.4  jit     14.1  spills 553
  _chunks          7a61066f3a grid (3, 2, 1)                                     aot     15.1  jit     13.8  spills 569
  _attn_prep       9c530a4cf3 grid (3, 18)          length=0                     aot     15.0  jit     15.0  spills 0
  _hc_writeback    88eb34d1f8 grid (1, 10)          RS=2560                      aot     15.0  jit     15.2  spills 0
  _attn_prep       c214f1f38c grid (3, 31)          length=0                     aot     14.7  jit     14.6  spills 0
  _b16mm           1cb7c3aed3 grid (1, 6, 32)       M=3 x_stride=10240           aot     13.5  jit     11.9  spills 0
  _hc_writeback    7ce8f2e467 grid (1, 10)          RS=2560                      aot     12.9  jit     13.8  spills 0
  _fp4mm           be159285bf grid (1, 40, 8)       M=16 x_stride=7040           aot     11.6  jit     12.9  spills 0
  _shift_windows   c595498989 grid (36, 40)         keep=16 OLD_L=30720 NEW_L=263680 NEW_ROW=16480 aot     11.1  jit      9.8  spills 0
  _ple_conv        7d5663088c grid (1, 20)          R=1                          aot     10.7  jit     10.7  spills 0
  _ple_conv        dbdcd2f284 grid (3, 20)          R=3                          aot     10.6  jit     10.4  spills 0
  _fp4mm           73a8141d87 grid (1, 40, 8)       M=16 x_stride=7040           aot     10.6  jit     12.0  spills 0
  _ple_conv        90646567b7 grid (16, 20)         R=16                         aot     10.5  jit     10.4  spills 0
  _hc_mix          8a30d589b6 grid (1, 10)                                       aot      9.7  jit      9.7  spills 0
  _fp4mm           da3beb6573 grid (1, 40, 8)       M=3 x_stride=7040            aot      9.6  jit     10.2  spills 0
  _shift_windows   4387e66de4 grid (36, 20)         keep=3 OLD_L=15360 NEW_L=65920 NEW_ROW=8240 aot      9.6  jit     10.7  spills 0
  _fp4mm           7514f20076 grid (1, 40, 8)       M=3 x_stride=7040            aot      9.5  jit     10.1  spills 0
  _fp4mm           c8cb10f2cf grid (1, 40, 4)       M=16 x_stride=3520           aot      9.5  jit      9.9  spills 0
  _add_streams     1adb47a1d9 grid (1, 10)                                       aot      9.5  jit      9.5  spills 0
  _fp4mm           3a06de0045 grid (1, 40, 8)       M=1 x_stride=7040            aot      9.5  jit     10.0  spills 0
  _pool            3eaf3a091b grid (2,)             R=3 length=0                 aot      9.5  jit      9.4  spills 0
  _fp4mm           357a35ae4e grid (1, 40, 4)       M=16 x_stride=3520           aot      9.4  jit      9.5  spills 0
  _fp4mm           97fc9c9f5d grid (1, 40, 8)       M=1 x_stride=7040            aot      9.4  jit     10.0  spills 0
  _hc_writeback    3a75486971 grid (1, 10)          RS=2560                      aot      9.2  jit      9.2  spills 0
  _hc_writeback    3a6bdd0f90 grid (1, 10)          RS=2560                      aot      9.2  jit      9.1  spills 0
  _fp4mm           3850aebd97 grid (1, 40, 4)       M=3 x_stride=3520            aot      9.1  jit      9.1  spills 0
  _fp4mm           ab3f0887dd grid (1, 40, 4)       M=3 x_stride=3520            aot      9.1  jit      9.1  spills 0
  _merge           cb49ce986d grid (3, 2)                                        aot      8.9  jit      8.8  spills 0
  _merge           e840f8bf9f grid (3, 2)                                        aot      8.7  jit      8.8  spills 0
  _fp4mm           5992dc794c grid (1, 40, 4)       M=1 x_stride=3520            aot      8.6  jit      8.5  spills 0
  _hc_normed       c85ea12a2b grid (1, 20)                                       aot      8.5  jit      8.1  spills 0
  _fp4mm           f4082d7d95 grid (1, 40, 4)       M=1 x_stride=3520            aot      8.5  jit      8.5  spills 0
  _hc_act          5ebe8e13e7 grid (1,)                                          aot      8.4  jit      9.3  spills 0
  _shift_windows   8d0df1b6fd grid (36, 20)         keep=16 OLD_L=15360 NEW_L=131840 NEW_ROW=8240 aot      8.2  jit      8.2  spills 0
  _select          386ba33630 grid (3,)             NB=65536                     aot      8.1  jit      8.1  spills 0
  _shift_windows   4651c1def3 grid (1, 40)          keep=2048 OLD_L=92160 NEW_L=20971520 NEW_ROW=10240 aot      8.0  jit      7.9  spills 0
  _hc_act          58a16fc8c6 grid (1,)                                          aot      8.0  jit      7.8  spills 0
  _rmsnorm         388b1b84f9 grid (3, 1)           x_stride=10240               aot      7.9  jit      7.9  spills 0
  _shift_windows   c5a6e34dce grid (1, 40)          keep=4 OLD_L=92160 NEW_L=81920 NEW_ROW=10240 aot      7.9  jit      7.9  spills 0
  _rmsnorm         42a2c790a0 grid (3, 1)           x_stride=2560                aot      7.8  jit      7.7  spills 0
  _scores          4dccf6de5e grid (3, 32)          NB=65536                     aot      7.5  jit      7.4  spills 0
  _shift_windows   6fccd38dc7 grid (1, 40)          keep=1 OLD_L=92160 NEW_L=81920 NEW_ROW=10240 aot      7.2  jit      7.2  spills 0
  _shift_windows   0ed41c7242 grid (36, 20)         keep=1 OLD_L=15360 NEW_L=65920 NEW_ROW=8240 aot      7.2  jit      8.6  spills 0
  _ple_embed_bf16  2f8fff4c1d grid (1, 8)                                        aot      7.1  jit      7.0  spills 0
  _attn_gate       03535d05f2 grid (3, 24)                                       aot      6.9  jit      6.9  spills 0
  _ple_embed_bf16  476dd3da4d grid (1, 16)                                       aot      6.9  jit      6.8  spills 0
  _topk_rows       af70ab09d7 grid (1,)                                          aot      6.7  jit      6.7  spills 0
  _attn_gate       c362669fc7 grid (3, 12)                                       aot      6.6  jit      6.6  spills 0
  _moe_partial     eb1057b496 grid (1, 10)                                       aot      6.4  jit      6.4  spills 0
  _reduce          c91b513928 grid (3,)             total=2560                   aot      6.2  jit      6.1  spills 0
  _reduce          b5a35179e8 grid (1,)             total=972                    aot      6.2  jit      6.1  spills 8
  _reduce          cae5e49f8a grid (3,)             total=2560                   aot      6.1  jit      6.0  spills 0
  _moe_partial     243038041f grid (1, 10)                                       aot      6.1  jit      6.0  spills 0
  _reduce          60a471f8f9 grid (3,)             total=2560                   aot      6.1  jit      6.0  spills 0
  _embed           341f5fe393 grid (1, 10)                                       aot      6.0  jit      5.9  spills 0
  _reduce          812f09a3b7 grid (3,)             total=2560                   aot      6.0  jit      5.9  spills 0
  _embed           e79c777d3d grid (1, 10)                                       aot      5.9  jit      5.8  spills 0
  _chunks8         432a6d7d1a grid (3, 2, 5)                                     aot     81.3  jit     63.2  spills 678
  _chunks8         993cec56db grid (3, 2, 2)                                     aot     55.6  jit     76.3  spills 637
  _chunks8         1d3b2707a3 grid (3, 1, 5)                                     aot     55.0  jit     47.1  spills 646
  _chunks8         a5b62e9c93 grid (3, 1, 2)                                     aot     49.9  jit     47.2  spills 640
  _attn_prep8      d3ea893026 grid (3, 31)          length=0                     aot     21.3  jit     21.1  spills 0
  _attn_prep8      cd4624707e grid (3, 18)          length=0                     aot     13.9  jit     13.8  spills 0
  _pool            3eaf3a091b grid (2,)             R=3 length=0                 aot      9.4  jit      9.4  spills 0
  _merge           28f21b1165 grid (3, 1)                                        aot      8.8  jit      8.8  spills 0
  _merge           cb49ce986d grid (3, 2)                                        aot      8.6  jit      8.5  spills 0
  _merge           3001e9d235 grid (3, 1)                                        aot      8.6  jit      8.8  spills 0
  _merge           e840f8bf9f grid (3, 2)                                        aot      8.4  jit      8.4  spills 0
  _select          386ba33630 grid (3,)             NB=65536                     aot      8.1  jit      7.9  spills 0
  _scores          4dccf6de5e grid (3, 32)          NB=65536                     aot      7.5  jit      7.5  spills 0
```
