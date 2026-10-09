#!/usr/bin/env python3
"""The gfx1151 hyper-connection kernels against the ones they replace, bit for bit (ROCm JIT, dev image):

  _b16mm_sm on bf16.slice_major(W)      == _b16mm on W (+ _reduce)       decode down rows, every row count
  _b16mm_ks_sm on slice_major(W)        == _b16mm_ks on W                 prompt down rows (both tiles)
  _b16mm_ks at the 64x128 tile          == the 128x64 tile
  _hc_up_mix (one dot over the streams) == the per-stream form (git HEAD~ source, inlined below) and == _b16mm + _hc_mix
  _b16mm at BK 256 (decode, long slices) == BK 64

    PYTHONPATH=src python tools/rocm/hc_bits.py
"""

from __future__ import annotations

import sys

import torch
import triton
import triton.language as tl

from tensorfold.families.qwen4_exp.cuda import bf16, glue, prompt_mm as P
from tensorfold.families.qwen4_exp.cuda.glue import _bsig

ROWS = (1, 3, 16, 17, 129, 161, 2049)


@triton.jit
def _hc_up_mix_ref(ACT, W, NORMED, MIXED, M, D: tl.constexpr, S: tl.constexpr, K: tl.constexpr,
                   BM: tl.constexpr, BD: tl.constexpr, BK: tl.constexpr):
    """The per-stream _hc_up_mix this branch replaced (rocm-next prompt_mm.py)."""

    rm = tl.program_id(0) * BM + tl.arange(0, BM)
    rd = tl.program_id(1) * BD + tl.arange(0, BD)
    rk = tl.arange(0, BK)
    m_ok = rm < M
    total = tl.zeros((BM, BD), dtype=tl.float32)
    for s in tl.static_range(S):
        acc = tl.zeros((BM, BD), dtype=tl.float32)
        for i in range(K // BK):
            x = tl.load(ACT + rm[:, None] * K + (i * BK + rk)[None, :], mask=m_ok[:, None], other=0.0)
            w = tl.load(W + (s * D + rd)[:, None] * K + (i * BK + rk)[None, :])
            acc = tl.dot(x, tl.trans(w), acc)
        u = acc.to(tl.bfloat16).to(tl.float32)
        n = tl.load(NORMED + rm[:, None] * (S * D) + s * D + rd[None, :], mask=m_ok[:, None], other=0.0).to(tl.float32)
        total += (_bsig(u) * n).to(tl.bfloat16).to(tl.float32)
    tl.store(MIXED + rm[:, None] * D + rd[None, :], (total / S).to(tl.bfloat16), mask=m_ok[:, None])


def same(a: torch.Tensor, b: torch.Tensor) -> bool:
    return torch.equal(a.contiguous().view(torch.uint8), b.contiguous().view(torch.uint8))


def rnd(*shape, scale=1.0):
    return (torch.randn(shape, device="cuda") * scale).to(torch.bfloat16)


def b16(x, w, n, k, sk, f32, kernel=bf16._b16mm, bk=64, warps=2):
    m = x.shape[0]
    bm = 128 if m > 128 else 16
    out = torch.empty(m, n, device="cuda", dtype=torch.float32 if f32 else torch.bfloat16)
    part = torch.empty(sk, m, n, device="cuda") if sk > 1 else out
    kernel[(triton.cdiv(m, bm), triton.cdiv(n, 64), sk)](x, w, out, part, m, k, N=n, K=k, SK=sk, BM=bm, BLOCK_N=64,
                                                         BK=bk, F32=f32, num_warps=warps, num_stages=1)
    if sk > 1:
        bf16._reduce[(triton.cdiv(m * n, 1024),)](part, out, m * n, SK=sk, BLOCK=1024, F32=f32, num_warps=4)
    return out


def ks(x, w, n, k, sk, f32, kernel=P._b16mm_ks, bm=128, bn=64, warps=8):
    m = x.shape[0]
    out = torch.empty(m, n, device="cuda", dtype=torch.float32 if f32 else torch.bfloat16)
    kernel[(triton.cdiv(m, bm) * triton.cdiv(n, bn),)](x, w, out, m, k, N=n, K=k, SK=sk, BM=bm, BLOCK_N=bn, BK=64,
                                                       F32=f32, GROUP=8, num_warps=warps, num_stages=1)
    return out


def main() -> int:
    torch.manual_seed(0)
    bad = n_cases = 0

    def check(name, a, b):
        nonlocal bad, n_cases
        n_cases += 1
        if not same(a, b):
            bad += 1
            print("DIFFER", name)

    for n, k in ((324, 10240), (320, 10240)):
        sk = bf16.split_k(n, k)
        w = rnd(n, k)
        wsm = bf16.slice_major(w, sk)
        for m in ROWS:
            x = rnd(m, k)
            ref = b16(x, w, n, k, sk, True)
            check(f"_b16mm_sm N{n} M{m}", ref, b16(x, wsm, n, k, sk, True, kernel=bf16._b16mm_sm))
            if m > 128:
                check(f"_b16mm_ks N{n} M{m}", ref, ks(x, w, n, k, sk, True))
                check(f"_b16mm_ks 64x128 N{n} M{m}", ref, ks(x, w, n, k, sk, True, bm=64, bn=128, warps=4))
                check(f"_b16mm_ks_sm 64x128 N{n} M{m}", ref,
                      ks(x, wsm, n, k, sk, True, kernel=P._b16mm_ks_sm, bm=64, bn=128, warps=4))
    for n, k in ((2560, 2560), (640, 2560), (2560, 6144)):
        sk = bf16.split_k(n, k)
        w = rnd(n, k)
        for m in (129, 2049):
            x = rnd(m, k)
            check(f"_b16mm_ks 64x128 N{n} K{k} M{m}", ks(x, w, n, k, sk, False), ks(x, w, n, k, sk, False, bm=64, bn=128, warps=4))
    for n, k in ((13952, 2560), (2560, 6144)):
        sk = bf16.split_k(n, k)
        w = rnd(n, k)
        for m in (1, 3, 16):
            x = rnd(m, k)
            check(f"_b16mm BK256 N{n} K{k} M{m}", b16(x, w, n, k, sk, False), b16(x, w, n, k, sk, False, bk=256, warps=4))
    D, S, K = 2560, 4, 320
    w = rnd(S * D, K, scale=0.05)
    for m in ROWS:
        act, normed = rnd(m, K), rnd(m, S * D)
        up = b16(act, w, S * D, K, 1, False)
        mixed = torch.empty(m, D, device="cuda", dtype=torch.bfloat16)
        glue.hc_mix(up, normed, mixed, torch.empty(m, D // 32, device="cuda"), S)
        for bm, warps in ((64, 8), (32, 8), (16, 2)):
            ref = torch.empty_like(mixed)
            _hc_up_mix_ref[(triton.cdiv(m, bm), D // 64)](act, w, normed, ref, m, D=D, S=S, K=K, BM=bm, BD=64, BK=64,
                                                         num_warps=4, num_stages=1)
            new = torch.empty_like(mixed)
            P._hc_up_mix[(triton.cdiv(m, bm), D // 64)](act, w, normed, new, m, D=D, S=S, K=K, BM=bm, BD=64, BK=64,
                                                        num_warps=warps, num_stages=1)
            check(f"_hc_up_mix BM{bm} M{m} vs per-stream", ref, new)
            check(f"_hc_up_mix BM{bm} M{m} vs _b16mm + _hc_mix", mixed, new)
    print(f"{n_cases} cases, {bad} differ")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
