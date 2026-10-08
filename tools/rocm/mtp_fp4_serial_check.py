"""fp4_serial.hip (tf_fn_fp4_serial_kernel) against Triton's nvfp4._fp4mm JIT-run on ROCm torch at one K slice and
bf16-pattern tables (the MTP shared expert's gate/up: N 640 / 1280, K 2560): output bytes over decode-sized rows,
two row strides, bf16 and fp32 out, the MTP layer's real shared expert and random value ranges; then both timed.
Run in the rocm-dev container: python tools/rocm/mtp_fp4_serial_check.py [--bench]
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../src"))
import mtp_hip
from mtp_experts_check import MODEL, E4M3, quantize
from tensorfold.families.qwen4_exp.cuda import nvfp4 as F

K = 2560
ROWS = (1, 2, 7, 9, 15, 16, 17, 36, 48, 64, 65, 113, 128)
E2M1 = torch.tensor([0, 0.5, 1, 1.5, 2, 3, 4, 6])


def tiles(w):
    """[n, k] bf16 -> nvfp4 pattern tables [n/64, k/64, 64 k, 64 n] as int16 (cuda_layouts.zig tileBits)."""
    n, k = w.shape
    return w.view(torch.int16).reshape(n // 64, 64, k // 64, 64).permute(0, 2, 3, 1).contiguous()


def table(kind, n, g):
    """(weights [n, k] bf16 of e2m1 values or not, scales [k/16, n] fp32)."""
    if kind == "real":
        from safetensors import safe_open
        f = safe_open(MODEL, "pt")
        w = torch.cat([f.get_tensor(f"mtp.layers.0.mlp.shared_expert.{p}_proj.weight") for p in ("gate", "up")])[:n]
        codes, s8, gs = quantize(w)
        c = np.stack([codes & 15, codes >> 4], -1).reshape(n, -1)
        vals = torch.from_numpy(np.where(c & 8, -1.0, 1.0) * E2M1.numpy()[c & 7]).bfloat16()
        return vals, torch.from_numpy((E4M3[s8] * gs).astype(np.float32)).t().contiguous()
    sign = torch.where(torch.rand(n, K, generator=g) < 0.5, -1.0, 1.0)
    if kind in ("e2m1", "e2m1-1"):
        w = (sign * E2M1[torch.randint(0, 8, (n, K), generator=g)]).bfloat16()
    else:
        w = (sign * torch.ldexp(1 + torch.rand(n, K, generator=g), torch.randint(-20, 20, (n, K), generator=g))).bfloat16()
    s = torch.ones(K // 16, n) if kind == "e2m1-1" else torch.ldexp(1 + torch.rand(K // 16, n, generator=g),
                                                                  torch.randint(-6, 6, (K // 16, n), generator=g))
    return w, s


def fp4mm(x, fp, out, f32):
    F.matmul(x, fp, out=out, f32=f32)


def serial(mod, x, fp, out, f32, m, n):
    mod.launch("tf_fn_fp4_serial_kernel", ((m + 15) // 16, n // 32, 1), 128,
               [x, fp.weight, fp.scale, out, ("i", m), ("i", n), ("i", K), ("i", x.stride(0)), ("i", int(f32))])


def main():
    mod = mtp_hip.build("fp4_serial")
    g = torch.Generator().manual_seed(11)
    ok = True
    for n in (640, 1280):
        for wk, xk in (("real", "normal"), ("e2m1-1", "normal"), ("e2m1", "wide"), ("normal", "normal"),
                       ("wide", "wide")):
            if wk == "real" and not os.path.exists(MODEL):
                continue
            w, s = table(wk, n, g)
            fp = F.FP4(weight=tiles(w).cuda(), scale=s.cuda(), n=n, k=K)
            big = torch.randn(max(ROWS), K + 64, generator=g)
            if xk == "wide":
                big = big.sign() * torch.ldexp(1 + torch.rand(big.shape, generator=g), torch.randint(-20, 20, big.shape,
                                                                                                    generator=g))
            big = big.bfloat16().cuda()
            bad = []
            for f32 in (False, True):
                for stride in (K, K + 64):
                    for m in ROWS:
                        x = big.view(-1)[: (m - 1) * stride + K].as_strided((m, K), (stride, 1))
                        dt = torch.float32 if f32 else torch.bfloat16
                        o1 = torch.full((m, n), 3.0, dtype=dt, device="cuda")
                        o2 = torch.full((m, n), -5.0, dtype=dt, device="cuda")
                        fp4mm(x, fp, o1, f32)
                        serial(mod, x, fp, o2, f32, m, n)
                        iv = torch.int32 if f32 else torch.int16
                        d = int((o1.view(iv) != o2.view(iv)).sum())
                        if d:
                            bad.append((m, stride, "fp32" if f32 else "bf16", d))
            ok &= not bad
            print(f"{'EQUAL ' if not bad else 'DIFFER'} N {n} w {wk:6s} x {xk:6s}: {len(ROWS) * 4} launches"
                  + (f", differ (rows, stride, out, values): {bad[:5]}" if bad else ""))
    if "--bench" in sys.argv:
        for n in (640, 1280):
            w, s = table("e2m1", n, g)
            fp = F.FP4(weight=tiles(w).cuda(), scale=s.cuda(), n=n, k=K)
            line = []
            for m in (9, 16, 36, 64, 128):
                x = torch.randn(m, K, device="cuda").bfloat16()
                o = torch.empty(m, n, dtype=torch.bfloat16, device="cuda")
                a = mtp_hip.best_us(lambda: fp4mm(x, fp, o, False))
                b = mtp_hip.best_us(lambda: serial(mod, x, fp, o, False, m, n))
                line.append(f"M {m}: {a:.0f} / {b:.0f}")
            print(f"N {n} _fp4mm / K-serial us (noisy, shared GPU): " + ", ".join(line))
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
