"""The HIP NVFP4 routed experts (fn_nvfp4_experts / fn_nvfp4_shape / fn_experts_prompt .hip) on gfx1151.

1. fp64 reference: down (fp32 out) and gate/up (SwiGLU) on the MTP layer's real bf16 experts quantized as
   cuda_layouts.zig quantizeFp4 does, and on random codes and scales; worst error over the fp32 accumulation bound.
2. bytes: every pair's output equal across the three kernels, T 2/3/4, the two-pass gate/up, item sizes 1..64,
   grids, a pair alone vs in a batch, and run to run.
Run in the rocm-dev container: python tools/rocm/mtp_experts_check.py [--bench]
"""
import os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import mtp_hip

MODEL = "/home/dzannotti/models/qwen38fn-int4-autoround/model_extra_tensors.safetensors"
D, NI, TOPK = 2560, 640, 5
E2M1 = np.array([0, 0.5, 1, 1.5, 2, 3, 4, 6, -0.0, -0.5, -1, -1.5, -2, -3, -4, -6])
dev = "cuda"


def e4m3_values():
    b = np.arange(256)
    e, m = (b >> 3) & 15, b & 7
    v = np.where(e == 0, m * 2.0 ** -9, (1 + m / 8) * 2.0 ** (e - 7))
    v = np.where((b & 0x7F) == 0x7F, np.nan, v)
    return np.where(b & 0x80, -v, v)


E4M3 = e4m3_values()


def quantize(w):
    """bf16 [n, k] (torch) -> codes [n, k/2] u8, scales [n, k/16] u8, g (quantizeFp4 / fp4Global)."""
    x = w.float().numpy().astype(np.float32)
    g = np.float32(max(np.abs(x).max() * np.float32(1 / (6 * 448)), 1e-30))
    blocks = x.reshape(x.shape[0], -1, 16)
    amax = np.abs(blocks).max(-1)
    s = torch.from_numpy(np.minimum((amax * np.float32(1 / 6)) / g, 448).astype(np.float32)).to(torch.float8_e4m3fn)
    s8 = s.view(torch.uint8).numpy()
    step = (E4M3[s8] * g).astype(np.float32)[..., None]
    y = np.where(step > 0, blocks / np.maximum(step, 1e-30), 0)
    code = np.abs(np.abs(y)[..., None] - E2M1[:8]).argmin(-1) + np.where(y < 0, 8, 0)
    code = code.reshape(x.shape).astype(np.uint8)
    return code[:, 0::2] | (code[:, 1::2] << 4), s8, g


def random_fp4(rng, n, k):
    codes = rng.integers(0, 256, (n, k // 2), dtype=np.uint8)
    # scales: normals 2^-6..2^6 and a few subnormals, never NaN
    s = rng.integers(0x08, 0x6F, (n, k // 16), dtype=np.uint8)
    s[rng.random(s.shape) < 0.03] = rng.integers(1, 8, dtype=np.uint8)
    return codes, s, np.float32(2.0 ** rng.integers(-8, -2) * (1 + rng.random()))


def pack(codes, scales):
    """cuda_layouts.zig packExpert of one matrix: [n/32, k/32, 144] u32."""
    n, k2 = codes.shape
    nb, kg = n // 32, k2 // 16
    row = codes.reshape(nb, 4, 8, kg, 16).astype(np.uint32)  # [cb, j, gq, g, byte]
    cell = np.zeros((nb, kg, 144), np.uint32)
    for t in range(4):
        b0, b1, b2, b3 = row[..., 2 * t], row[..., 2 * t + 1], row[..., 8 + 2 * t], row[..., 9 + 2 * t]
        lo = (b0 & 15) | (b1 & 15) << 4 | (b2 & 15) << 8 | (b3 & 15) << 12
        hi = (b0 >> 4) | (b1 >> 4) << 4 | (b2 >> 4) << 8 | (b3 >> 4) << 12
        word = (lo | hi << 16).transpose(0, 3, 2, 1)  # [cb, g, gq, j]
        for gq in range(8):
            cell[:, :, (gq * 4 + t) * 4:(gq * 4 + t) * 4 + 4] = word[:, :, gq, :]
    sb = np.zeros((nb, kg, 64), np.uint8)
    sc = scales.reshape(nb, 4, 4, 2, kg, 2)  # [cb, j, t, c, g, h]
    for t in range(4):
        for j in range(4):
            for c in range(2):
                sb[:, :, t * 16 + j * 2 + c] = sc[:, j, t, c, :, 0]
                sb[:, :, t * 16 + 8 + j * 2 + c] = sc[:, j, t, c, :, 1]
    cell[:, :, 128:] = sb.view(np.uint32).reshape(nb, kg, 16)
    return cell


def dequant(codes, scales):
    """fp64 [n, k] of the codes times their e4m3 block scales."""
    c = np.stack([codes & 15, codes >> 4], -1).reshape(codes.shape[0], -1)
    return E2M1[c] * np.repeat(E4M3[scales], 16, axis=1)


class Experts:
    """E experts' gate/up and down: packed device blocks, fp32 g per (expert, matrix), fp64 dequantized weights."""

    def __init__(self, mats):  # mats[e] = [(codes, scales, g) for gate, up, down]
        self.e = len(mats)
        up = np.stack([np.stack([pack(*m[0][:2]), pack(*m[1][:2])], 2) for m in mats])
        dn = np.stack([pack(*m[2][:2])[:, :, None] for m in mats])
        self.up = torch.from_numpy(up.view(np.int32)).to(dev)
        self.down = torch.from_numpy(dn.view(np.int32)).to(dev)
        self.up_scale = torch.tensor([[m[0][2], m[1][2]] for m in mats], dtype=torch.float32, device=dev)
        self.down_scale = torch.tensor([[m[2][2]] for m in mats], dtype=torch.float32, device=dev)
        self.ref = [[dequant(*mm[:2]) * np.float64(mm[2]) for mm in m] for m in mats]


def real_experts(n):
    from safetensors import safe_open
    f = safe_open(MODEL, "pt")
    p = "mtp.layers.0.mlp.experts.{}.{}_proj.weight"
    return Experts([[quantize(f.get_tensor(p.format(i, m))) for m in ("gate", "up", "down")] for i in range(n)])


def random_experts(rng, n):
    return Experts([[random_fp4(rng, NI, D), random_fp4(rng, NI, D), random_fp4(rng, D, NI)] for _ in range(n)])


def plan(experts_of_pair, n_exp, tile):
    """members (pairs grouped by expert, in pair order), items (e, first, count <= tile), counts."""
    members, items = [], []
    for e in range(n_exp):
        ps = [p for p, x in enumerate(experts_of_pair) if x == e]
        for i in range(0, len(ps), tile):
            items.append((e, len(members) + i, len(ps[i:i + tile])))
        members += ps
    t = lambda v: torch.tensor(v, dtype=torch.int32, device=dev)
    return t(members or [0]), t([x for it in items for x in it] or [0]), t([len(items)]), len(items)


NAMES = {
    "nvfp4": {e: f"_ZN19tf_fn_nvfp4_experts19nvfp4_expert_kernelILi{m}ELi{e}ELi4EEEvPK14__hip_bfloat16iiPK15HIP_"
                 "vector_typeIjLj4EEPKfiiPKiSB_SB_Pvifi" for m, e in ((2, 2), (1, 0), (1, 3))},
    "nt": "_ZN17tf_fn_nvfp4_shape16expert_nt_kernelILi2ELi2ELi1ELi2ELi1EEEvPK14__hip_bfloat16iiPK15HIP_vector_type"
          "IjLj4EEPKfiiPKiSB_SB_Pvifi",
}


class Kernels:
    def __init__(self):
        self.ex = mtp_hip.build("fn_nvfp4_experts")
        self.sh = mtp_hip.build("fn_nvfp4_shape")
        self.pr = mtp_hip.build("fn_experts_prompt")

    def run(self, how, epi, x, slots, ex, members, items, counts, n_items, out, limit=0.0, grid=None, gate=None):
        """how: 'nvfp4', 'nt', ('prompt', T) or 'gu2' (gate pass then up pass)."""
        down = epi in (0, 3)
        w, sc = (ex.down, ex.down_scale) if down else (ex.up, ex.up_scale)
        n, kg = (D, NI // 32) if down else (NI, D // 32)
        nb = n // 32
        units = n_items * nb
        args = lambda o: [x, ("i", x.stride(0)), ("i", slots), w, sc, ("i", kg), ("i", nb), items, counts, members, o,
                          ("i", n), ("f", limit), ("i", -1)]
        if how == "nvfp4":
            name = NAMES["nvfp4"][epi]
            self.ex.launch(name, grid or max(1, min((units + 3) // 4, self.ex.occupancy(name, 128) * 40)), 128,
                           args(out))
        elif how == "nt":
            assert epi == 2
            self.sh.launch(NAMES["nt"], grid or max(1, min(units * 4, self.sh.occupancy(NAMES["nt"], 32) * 40)), 32,
                           args(out))
        elif how == "gu2":
            assert epi == 2
            g = torch.empty_like(out)
            self.pr.launch("fn_prompt4_gate_t2o", grid or max(1, min(units, 160)), 128, args(g))
            self.pr.launch("fn_prompt4_up_t2o", grid or max(1, min(units, 160)), 128, args(out) + [g])
        else:
            t = how[1]
            name = f"fn_prompt4_{ {2: 'gu', 0: 'f32', 3: 'b16'}[epi] }_t{t}".replace(" ", "")
            self.pr.launch(name, grid or max(1, min(units, 160)), 128, args(out))


def bf16_np(t):
    return t.view(torch.int16).cpu().numpy().view(np.uint16)


def reference_check(k, ex, rng, label, rows=48):
    """fp64 reference of down (fp32 out, error over the accumulation bound) and gate/up (bf16 SwiGLU ulps)."""
    pairs = rows * TOPK
    xe = [rng.choice(ex.e, TOPK, replace=False) for _ in range(rows)]
    eop = [int(e) for r in xe for e in r]
    members, items, counts, n_items = plan(eop, ex.e, 16)
    x = (torch.randn(rows, D, device=dev) * 0.5).bfloat16()
    act = (torch.randn(pairs, NI, device=dev) * 0.2).bfloat16()
    xd, ad = x.double().cpu().numpy(), act.double().cpu().numpy()
    # down, fp32 out
    out = torch.full((pairs, D), float("nan"), device=dev)
    k.run("nvfp4", 0, act, 0, ex, members, items, counts, n_items, out)
    got = out.double().cpu().numpy()
    ref, bound = np.empty_like(got), np.empty_like(got)
    for p, e in enumerate(eop):
        w = ex.ref[e][2]
        ref[p], bound[p] = w @ ad[p], np.abs(w) @ np.abs(ad[p])
    err = np.abs(got - ref) / np.maximum(bound, 1e-300) / 2.0 ** -24
    col = err.max(0)
    i = np.unravel_index(err.argmax(), err.shape)
    print(f"{label} down fp32 vs fp64: worst {err.max():.2f} x 2^-24 x sum|w x| (pair {i[0]} col {i[1]}: got "
          f"{got[i]:.9g} ref {ref[i]:.9g}); per-column worst: median {np.median(col):.2f} max {col.max():.2f}; "
          f"rel to |ref|: {np.nanmax(np.abs(got - ref) / np.maximum(np.abs(ref), 1e-30)):.3g}")
    ok = err.max() < 41  # 40 groups' fmaf + the WMMA's 16-term sums, with margin; typical is far smaller
    # gate/up SwiGLU (bf16): fp64 gate and up, then the epilogue's roundings in torch fp32
    out = torch.zeros(pairs, NI, dtype=torch.bfloat16, device=dev)
    k.run("nvfp4", 2, x, TOPK, ex, members, items, counts, n_items, out)
    gr = np.stack([ex.ref[e][0] @ xd[p // TOPK] for p, e in enumerate(eop)])
    ur = np.stack([ex.ref[e][1] @ xd[p // TOPK] for p, e in enumerate(eop)])
    g = torch.from_numpy(gr).float().bfloat16().float()
    u = torch.from_numpy(ur).float().bfloat16().float()
    want = ((g / (1 + torch.exp(-g))).bfloat16().float() * u).bfloat16()
    a, b = bf16_np(out.cpu()).astype(np.int32), bf16_np(want).astype(np.int32)
    ulps = np.abs(np.where(a & 0x8000, 0x8000 - a, a) - np.where(b & 0x8000, 0x8000 - b, b))
    # where they differ, the fp64 gate or up must sit within the accumulation bound (4 x 2^-24 x sum|w x|) of a bf16
    # rounding boundary (or SiLU within 2^-20 of one)
    gbnd = np.stack([np.abs(ex.ref[e][0]) @ np.abs(xd[p // TOPK]) for p, e in enumerate(eop)]) * 2.0 ** -22
    ubnd = np.stack([np.abs(ex.ref[e][1]) @ np.abs(xd[p // TOPK]) for p, e in enumerate(eop)]) * 2.0 ** -22
    near = lambda r, tol: (torch.from_numpy(r - tol).float().bfloat16() != torch.from_numpy(r + tol).float().bfloat16()).numpy()
    gb = g.double().numpy()
    with np.errstate(over="ignore"):
        sr = gb / (1 + np.exp(-gb))  # the SiLU of the bf16 gate: expf's last bit may differ from torch's at a tie
    unexplained = int(((ulps != 0) & ~near(gr, gbnd) & ~near(ur, ubnd) & ~near(sr, np.abs(sr) * 2.0 ** -20)).sum())
    print(f"{label} gate/up SwiGLU vs fp64 + torch epilogue: {np.mean(ulps == 0) * 100:.3f}% equal, the "
          f"{int((ulps != 0).sum())} others all at a gate, up or SiLU bf16 rounding tie: "
          f"{'yes' if unexplained == 0 else f'NO ({unexplained})'}")
    return ok and unexplained == 0


def invariance_check(k, ex, rng):
    """Every pair's bytes across kernels, T, item sizes, grids, batch vs alone, and run to run."""
    ok = True
    rows = 70
    eop = [int(e) for _ in range(rows) for e in rng.choice(ex.e, TOPK, replace=False)]
    pairs = len(eop)
    x = (torch.randn(rows, D, device=dev) * 0.5).bfloat16()
    act = (torch.randn(pairs, NI, device=dev) * 0.2).bfloat16()
    for epi in (2, 0, 3):
        n = NI if epi == 2 else D
        dt = torch.float32 if epi == 0 else torch.bfloat16
        xin, slots = (x, TOPK) if epi == 2 else (act, 0)
        outs = {}
        hows = ["nvfp4", ("prompt", 2), ("prompt", 4)] + (["nt", ("prompt", 3), "gu2"] if epi == 2 else [])
        for how in hows:
            for tile in (16, 1, 7, 64) if how == "nvfp4" else (16, 32, 64):
                for grid in (None, 3):
                    mem, it, cn, ni = plan(eop, ex.e, tile)
                    o = torch.full((pairs, n), -7.0, dtype=dt, device=dev)
                    k.run(how, epi, xin, slots, ex, mem, it, cn, ni, o, grid=grid)
                    outs[(how, tile, grid)] = o
        # pair rows alone: one token row (its 5 pairs), and the first 9 rows
        for sub in (1, 9):
            p = sub * TOPK
            mem, it, cn, ni = plan(eop[:p], ex.e, 16)
            o = torch.full((pairs, n), -7.0, dtype=dt, device=dev)
            k.run("nvfp4", epi, xin, slots, ex, mem, it, cn, ni, o)
            outs[("alone", sub)] = o
        base = outs[("nvfp4", 16, None)].view(torch.int16 if dt == torch.bfloat16 else torch.int32)
        again = torch.full_like(outs[("nvfp4", 16, None)], -7.0)
        mem, it, cn, ni = plan(eop, ex.e, 16)
        k.run("nvfp4", epi, xin, slots, ex, mem, it, cn, ni, again)
        outs[("again",)] = again
        bad = []
        for key, o in outs.items():
            v = o.view(base.dtype)
            p = key[1] * TOPK if key[0] == "alone" else pairs
            if not torch.equal(v[:p], base[:p]):
                bad.append((key, int((v[:p] != base[:p]).sum())))
        print(f"epilogue {epi}: {len(outs)} runs, {'all byte-equal' if not bad else 'DIFFER ' + str(bad[:6])}")
        ok = ok and not bad
    return ok


def bench(k, rng):
    ex = random_experts(rng, 96)
    print("timings over 96 experts (noisy: prod shares the GPU; best of 5 x 20):")
    for rows in (1, 8, 32, 128):
        eop = [int(e) for _ in range(rows) for e in rng.choice(ex.e, TOPK, replace=False)]
        mem, it, cn, ni = plan(eop, ex.e, 16)
        x = torch.randn(rows, D, device=dev).bfloat16()
        act = torch.randn(len(eop), NI, device=dev).bfloat16()
        o1 = torch.empty(len(eop), NI, dtype=torch.bfloat16, device=dev)
        o2 = torch.empty(len(eop), D, dtype=torch.float32, device=dev)
        t_gu = mtp_hip.best_us(lambda: k.run("nvfp4", 2, x, TOPK, ex, mem, it, cn, ni, o1))
        t_nt = mtp_hip.best_us(lambda: k.run("nt", 2, x, TOPK, ex, mem, it, cn, ni, o1))
        t_dn = mtp_hip.best_us(lambda: k.run("nvfp4", 0, act, 0, ex, mem, it, cn, ni, o2))
        used = len(set(eop))
        gb = used * (NI * D * 2 * 0.5625) / 1e3
        print(f"  rows {rows:4d} ({used} experts): gate/up {t_gu:7.1f} us ({gb / t_gu:5.1f} GB/s), nt {t_nt:7.1f} us, "
              f"down {t_dn:7.1f} us ({gb / 2 / t_dn:5.1f} GB/s)")


def main():
    rng = np.random.default_rng(1234)
    k = Kernels()
    ok = True
    ex_real = real_experts(16) if os.path.exists(MODEL) else None
    if ex_real is not None:
        ok &= reference_check(k, ex_real, rng, "MTP experts 0-15")
    ex_rand = random_experts(rng, 12)
    ok &= reference_check(k, ex_rand, rng, "random codes")
    ok &= invariance_check(k, ex_real or ex_rand, rng)
    if "--bench" in sys.argv:
        bench(k, rng)
    print("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
