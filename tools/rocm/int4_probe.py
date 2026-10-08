"""WMMA layout and streaming bandwidth on gfx1151: python tools/rocm/int4_probe.py"""
import os, sys, torch
sys.path.insert(0, os.path.dirname(__file__))
import hiprun

here = os.path.dirname(os.path.abspath(__file__))
co = hiprun.build(f"{here}/int4_probe.hip", hiprun.cache("int4_probe.co"))
m = hiprun.Module(co)
a = torch.randn(16, 16, device="cuda").bfloat16()
b = torch.randn(16, 16, device="cuda").bfloat16()
d = torch.zeros(32, 8, device="cuda")
m.launch("probe_wmma", 1, 32, [a, b, d])
ref = a.float() @ b.float().T   # [m, n]
lane = torch.arange(32)
# gfx11 layout guess: d[l][i] = D[2i + l/16][l%16]
guess = torch.stack([ref[2 * i + lane // 16, lane % 16] for i in range(8)], 1).cuda()
print("wmma layout d[l][i] = D[2i + l/16][l%16]:", torch.allclose(d, guess, atol=1e-3), (d - guess).abs().max().item())
n = (1 << 30) // 16
buf = torch.empty(n * 4, dtype=torch.int32, device="cuda")
for blocks, threads in [(40 * 8, 256), (40 * 16, 256), (40 * 32, 256), (40 * 64, 256), (40 * 32, 512)]:
    out = torch.empty(blocks * threads, dtype=torch.int32, device="cuda")
    us = hiprun.best_us(lambda: m.launch("probe_stream", blocks, threads, [buf, ("q", n), out]), reps=5, rounds=7)
    print(f"stream 1 GiB, {blocks} x {threads}: {us:.0f} us, {n * 16 / us / 1e3:.0f} GB/s")
