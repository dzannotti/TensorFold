# Block-FP8 decode tiles (16/32 rows) on gfx1151: branch perf-fp8dec

Change: `zig/kernels/hip/fn_qmmf.hip` qmmf_kernel's 16- and 32-row tiles (the only tiles M <= 32 uses; one row tile,
so no weight reuse across tiles) load weights with `__builtin_nontemporal_load`. Same instructions otherwise, same
bits: `tools/rocm/fp8_check.py` PASS on every case with the worst-case numbers of fp8_check.txt unchanged.
The 64-row tiles, the fused form and the wide prefill tile keep normal loads (their L2 band reuses weights).

## Results (`tools/rocm/fp8_dec_bench.py build/base build/rocm`, graph-replayed, 128 MiB of rotated copies, best of 21,
A/B alternated; 02:37 BST, perf lead's model between runs, GB/s of weight + scale bytes)

| shape (n x k) | m1 | m4 | m8 | m16 | m17 | m32 |
|---|---|---|---|---|---|---|
| gdn qkvz 16384 x 2560 (ld) | 201 -> 226 | 197 -> 228 | 197 -> 226 | 193 -> 223 | 190 -> 224 | 183 -> 220 |
| attn qkv 13312 x 2560 (ld) | 204 -> 220 | 202 -> 218 | 200 -> 220 | 203 -> 221 | 212 -> 225 | 209 -> 221 |
| out_proj 2560 x 6144 (sk 4) | 217 -> 232 | 216 -> 232 | 194 -> 213 | 198 -> 214 | 203 -> 216 | 201 -> 215 |
| shared gu 2560 x 2560 (sk 4) | 209 -> 225 | 207 -> 223 | 204 -> 220 | 202 -> 218 | 198 -> 209 | 196 -> 208 |
| shared down 2560 x 1280 (sk 2) | 191 -> 207 | 191 -> 207 | 191 -> 207 | 190 -> 205 | 180 -> 194 | 180 -> 193 |

A later run (03:29, noisier) gives the new tiles attn 211-229, out 217-228, shared gu 207-225, shared down 193-208,
base 5-10% below; its gdn row (194-208 for both) was hit by the co-tenant.
Ceilings measured with `build/s`-style probes (pure loads of this layout, no math): 227-232 GB/s gdn, ~218 out,
211-213 shared down; a plain contiguous read of 3.3 MB tops at ~205-213 (launch + ramp ~3 us), so shared down
cannot reach 220 as its own launch.

## Tried, not kept (A/B, same conditions)

- k-major weight layout ([K/64][N/64][4096], a group's tiles side by side): +2-3% with normal loads, nothing on top
  of nontemporal loads.
- Nontemporal scale loads: -2-3%.
- Deeper prefetch (ring of 2-3 groups, weights + inputs together) and de-duplicated weight loads (each half-wave
  loads and converts half of a column's bytes, fragments assembled with permlanex16): 10-25% slower; the compiler
  serializes the ring (vmcnt(0) before the refill) and the cross-half selects cost more than the halved conversion.
- Unconditional (clamped) prefetch of the last group: -5%.
- Padding rows (M < 16) reading distinct lines instead of row M-1: +2% gdn, -2% small shapes.
- Inputs staged in LDS by the block (fragment order, one LDS-only barrier a group, 109 VGPRs): mixed (+4% attn m1-16,
  -5% at 17-32 rows and on gdn); would need SK-sized dynamic LDS.
- Occupancy (VGPRs) does not limit: probes at 4 vs 16 blocks a WGP stream the same.
- Removing the e4m3 conversion and the input loads entirely (timing-only builds) gains only ~3-5%: decode is bound
  by the load pattern, not VALU.

## Left

- m17-32 (the 32-row tile) runs 3-5% below the 16-row tile; shared gu/down are launch bound. The shared expert runs
  on the side stream beside the routed experts, so its latency is mostly hidden; fusing gu -> SwiGLU -> down (or the
  SwiGLU into down's input loads, and the slot copy into down's store via matmulLd) would save ~3 launches a layer.
