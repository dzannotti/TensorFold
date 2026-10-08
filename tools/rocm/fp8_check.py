#!/usr/bin/env python3
"""tl.float8e4nv on this GPU (gfx1151 has no FP8 hardware: Triton emulates it) against torch.float8_e4m3fn, the cast
kv8.py's reference quantizer uses: every fp32 value of every e4m3 binade (exact codes, the ties between them, the
points just off each tie, subnormals, signs) encoded, and all 256 codes decoded to fp32 and bf16.

    python tools/rocm/fp8_check.py
"""

import sys

import torch
import triton
import triton.language as tl


@triton.jit
def _encode(X, Y, N, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    ok = i < N
    tl.store(Y + i, tl.load(X + i, mask=ok).to(tl.float8e4nv), mask=ok)


@triton.jit
def _decode(C, F, B, BLOCK: tl.constexpr):
    i = tl.arange(0, BLOCK)
    c = tl.load(C + i)
    tl.store(F + i, c.to(tl.float32))
    tl.store(B + i, c.to(tl.bfloat16))


def main() -> int:
    dev = "cuda"
    codes = torch.arange(256, dtype=torch.uint8)
    vals = codes.view(torch.float8_e4m3fn).float()
    fin = vals[torch.isfinite(vals)].unique()
    pos = fin[fin >= 0].sort().values
    mids = (pos[1:] + pos[:-1]) / 2                       # ties: round half to even
    eps = torch.nextafter(mids, torch.full_like(mids, 1e9)) - mids
    x = torch.cat([pos, mids, mids + eps, mids - eps, pos * 1.0000001, torch.rand(1 << 16) * 448])
    x = torch.cat([x, -x]).contiguous()
    want = x.to(torch.float8_e4m3fn).view(torch.uint8)
    xd = x.to(dev)
    got = torch.empty(x.numel(), dtype=torch.float8_e4m3fn, device=dev)
    _encode[(triton.cdiv(x.numel(), 1024),)](xd, got, x.numel(), BLOCK=1024)
    got = got.view(torch.uint8).cpu()
    bad = (got != want).nonzero().flatten()
    for i in bad[:8].tolist():
        print(f"encode {x[i].item()!r}: kernel {got[i].item():#04x}, torch {want[i].item():#04x}")
    f = torch.empty(256, dtype=torch.float32, device=dev)
    b = torch.empty(256, dtype=torch.bfloat16, device=dev)
    _decode[(1,)](codes.to(dev).view(torch.float8_e4m3fn), f, b, BLOCK=256)
    ok_f = torch.equal(f.cpu().view(torch.int32)[torch.isfinite(vals)], vals.view(torch.int32)[torch.isfinite(vals)])
    ok_b = torch.equal(b.cpu().float()[torch.isfinite(vals)], vals[torch.isfinite(vals)])
    nan_ok = bool(torch.isnan(f.cpu()[~torch.isfinite(vals)]).all())
    print(f"encode: {x.numel()} values, {bad.numel()} differ from torch; decode fp32 {ok_f}, bf16 {ok_b}, NaN codes {nan_ok}")
    return 0 if bad.numel() == 0 and ok_f and ok_b and nan_ok else 1


if __name__ == "__main__":
    sys.exit(main())
