#!/usr/bin/env python3
"""The gfx1151 block-FP8 lane matmul (zig/kernels/hip/fn_qmmf{,_ld}.hip) against a float64 reference, launched as
cuda_fp8.zig's matmulAt launches it (``plan``), on random and checkpoint tensors.

Python's Fp8BlockLinear runs qmmf.cu (inline PTX: cp.async, ldmatrix, mma.sync): there is no ROCm counterpart to be
byte-equal to, so the contract is BRIEF.md's no-counterpart one: fp32 outputs within the documented accumulation of
the float64 sum, bf16 outputs exactly the fp32 outputs rounded once (RNE), a row's bits independent of M (every M of
ROWS against the largest call's rows and against one-row calls), run-to-run determinism, the tiled, in-block-split
and fused forms byte-equal, and matmulLd's rows byte-equal to matmul's with the gaps untouched.

Inside the dev container, with code objects from tools/rocm/build_fp8.sh:
    python -B tools/rocm/fp8_check.py [--real MODEL_DIR] [--bench] [--co build/rocm]
"""

from __future__ import annotations

import argparse
import ctypes
import json
import struct
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from tensorfold.cuda.nvfp4.linear import Fp8BlockLinear  # noqa: E402  (pure torch layouts: from_checkpoint)

ROWS = [1, 2, 3, 7, 8, 15, 16, 17, 31, 32, 33, 63, 64, 65, 100, 127, 128, 255, 256, 300, 512, 2048]
SHAPES = [  # (name, n, k): the INT4-AutoRound checkpoint's block-FP8 projections
    ("gdn_qkvz", 16384, 2560), ("attn_qkv", 13312, 2560), ("out_proj", 2560, 6144), ("shared_gu", 2560, 2560),
    ("shared_down", 2560, 1280), ("odd_n", 200, 384), ("odd_split", 200, 1536)]
FUSED_ROWS, MAX_SLICES, WIDE_ROWS = 256, 4, 33


# ---- cuda_fp8.zig's plan ---------------------------------------------------------------------------------------------

def split_k(n: int, k: int) -> int:
    """qmm.split_k, capped at the 4 slices a 512-thread gfx1151 block holds."""
    tiles, groups, sk = -(-n // 64), k // 64, 1
    while sk < 8 and tiles * sk < 192 and groups % (sk * 2) == 0 and groups // (sk * 2) >= 8:
        sk *= 2
    return min(sk, MAX_SLICES)


def bucket(m: int) -> int:
    return 16 if m <= 16 else 32 if m <= 32 else 64


def l2_group(rows_t: int, bm: int, k: int) -> int:
    return max(1, min(rows_t, (12 << 20) // (bm * k * 2)))


def plan(n: int, k: int, m: int, fused_rows: int = FUSED_ROWS, wide_rows: int = WIDE_ROWS):
    """(bm, fused, sk, grid, block, group): matmulAt on gfx1151; from WIDE_ROWS the wide tile (bm 128 or -64 for 64,
    `fused` its bn: 128, or 64 with K slices)."""
    sk = split_k(n, k)
    if m >= wide_rows:                                     # cuda_fp8.zig wideTile; bm 64 passed as -64
        bm, bn = 64 if m <= 64 else 128, 128 if sk == 1 else 64
        rows_t = -(-m // bm)
        return (bm if bm == 128 else -64), bn, sk, rows_t * -(-n // bn), 2 * bm, l2_group(rows_t, bm, k)
    fused = sk > 1 and m >= fused_rows
    bm = 64 if fused else bucket(m)
    rows_t = -(-m // bm)
    return bm, fused, sk, rows_t * -(-n // 64), 128 if fused else 128 * sk, l2_group(rows_t, bm, k)


# ---- HIP module launches ---------------------------------------------------------------------------------------------

class Hip:
    def __init__(self, co: Path):
        torch.zeros(1, device="cuda")
        self.h = ctypes.CDLL("libamdhip64.so.7")
        self.mods = {}
        for name in ("fn_qmmf", "fn_qmmf_ld"):
            mod = ctypes.c_void_p()
            data = (co / f"{name}.co").read_bytes()
            self._ok(self.h.hipModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(data)), name)
            self.mods[name] = (mod, data)
        self.fns = {}
        self.wide = b"qmmw_kernel" in self.mods["fn_qmmf"][1]  # an older build (A/B against it): no wide tile

    def _ok(self, r, what):
        if r != 0:
            raise RuntimeError(f"HIP error {r} ({what})")

    def fn(self, ld: bool, bm: int, f32: bool, fused):
        ns, extra = ("13tf_fn_qmmf_ld", "i") if ld else ("10tf_fn_qmmf", "")
        if bm in (-64, 128):                                   # the wide tile (bm 64 as -64): `fused` holds its bn
            sym = f"_ZN{ns}11qmmw_kernelILi{abs(bm)}ELi{fused}ELb{int(f32)}EEEvPK14__hip_bfloat16PKhS5_fPvPfiiiiiii{extra}"
        else:
            sym = (f"_ZN{ns}11qmmf_kernelILi3ELi{bm}ELi64ELi1ELi4ELi4ELb{int(f32)}ELb0ELb{int(fused)}EEEv"
                   f"PK14__hip_bfloat16PKhS5_fPvPfiiiiiii{extra}")
        if sym not in self.fns:
            f = ctypes.c_void_p()
            self._ok(self.h.hipModuleGetFunction(ctypes.byref(f), self.mods["fn_qmmf_ld" if ld else "fn_qmmf"][0],
                                                 sym.encode()), sym)
            self.fns[sym] = f
        return self.fns[sym]

    def matmul(self, lin, x, out, f32=False, ldo=None, ldx=None, fused_rows=FUSED_ROWS, wide_rows=WIDE_ROWS,
               force=None):
        """cuda_fp8.zig matmulAt: x (m, K) bf16 rows ldx apart -> out (m, n) rows n (or ldo) apart."""
        m = x.shape[0]
        bm, fused, sk, grid, block, group = force or plan(lin.n, lin.k, m, fused_rows, wide_rows if self.wide else 1 << 30)
        f = self.fn(ldo is not None, bm, f32, fused)
        ints = [m, lin.n, lin.k, sk, lin.npad, ldx or (lin.k if m == 1 else x.stride(0)), group]
        if ldo is not None:
            ints.append(ldo)
        vals = [ctypes.c_void_p(x.data_ptr()), ctypes.c_void_p(lin.w8.data_ptr()), ctypes.c_void_p(lin.bs.data_ptr()),
                ctypes.c_float(1.0), ctypes.c_void_p(out.data_ptr()), ctypes.c_void_p(0)] + [ctypes.c_int(v) for v in ints]
        ptrs = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.pointer(v), ctypes.c_void_p) for v in vals])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self._ok(self.h.hipModuleLaunchKernel(f, grid, 1, 1, block, 1, 1, 0, stream, ptrs, None), "launch")


# ---- checks ----------------------------------------------------------------------------------------------------------

def reference(lin, x):
    """float64 x @ (w * scale).T and sum |x w scale| (the accumulation's scale), on the GPU."""
    w = lin.dense.double()
    xd = x.double()
    return xd @ w.t(), xd.abs() @ w.abs().t()


def check_case(hip, name, lin, x, log, fp32=True):
    n, k = lin.n, lin.k
    mm = x.shape[0]
    y32 = torch.empty((mm, n), dtype=torch.float32, device="cuda")
    ybf = torch.empty((mm, n), dtype=torch.bfloat16, device="cuda")
    hip.matmul(lin, x, y32, f32=True)
    hip.matmul(lin, x, ybf)
    ref, mag = reference(lin, x)
    err = (y32.double() - ref).abs()
    units = err / (mag * 2.0 ** -24).clamp_min(1e-300)
    sk = split_k(n, k)
    bound = 5 * k / 64 + sk                                     # fp32 roundings a row: 4 WMMA + 1 fma a group, 1 add a slice
    worst = int(units.argmax())
    r, c = divmod(worst, n)
    col_worst = units.max(0).values
    rne = torch.equal(ybf, y32.to(torch.bfloat16))
    y64 = y32.double()                                         # bf16 error <= half its ulp + the fp32 error
    half = torch.ldexp(torch.ones_like(ref), torch.frexp(y64.abs().clamp_min(2.0 ** -126))[1] - 9)
    ulps = float(((ybf.double() - ref).abs() / (half + err)).max())
    ok = rne and float(units.max()) <= bound and ulps <= 1.0
    bad = []
    # row invariance: every M's rows equal the largest call's; one-row calls equal their rows; determinism
    for m in ROWS:
        if m > mm:
            continue
        for f32, full in ((False, ybf), (True, y32)):
            y = torch.empty((m, n), dtype=full.dtype, device="cuda")
            hip.matmul(lin, x[:m], y, f32=f32)
            if not torch.equal(y, full[:m]):
                bad.append(f"m {m} {'fp32' if f32 else 'bf16'}")
    for r1 in sorted({0, 1, mm // 2, mm - 1}):
        y = torch.empty((1, n), dtype=torch.bfloat16, device="cuda")
        hip.matmul(lin, x[r1:r1 + 1], y)
        if not torch.equal(y[0], ybf[r1]):
            bad.append(f"row {r1} one-row")
    y = torch.empty_like(ybf)
    hip.matmul(lin, x, y)
    if not torch.equal(y, ybf):
        bad.append("rerun")
    # forms: tiled one-block-slices, fused and wide (same bits by construction), at 256 and at 17 rows
    for m in sorted({min(256, mm), min(17, mm)}):
        forms = {"tiled": dict(fused_rows=1 << 30, wide_rows=1 << 30), "wide": dict(wide_rows=1)}
        if sk > 1:
            forms["fused"] = dict(fused_rows=1, wide_rows=1 << 30)
        ys = {}
        for form, kw in forms.items():
            ys[form] = torch.empty((m, n), dtype=torch.bfloat16, device="cuda")
            hip.matmul(lin, x[:m], ys[form], **kw)
        bad += [f"{f} != tiled m {m}" for f in ys if not torch.equal(ys[f], ys["tiled"])]
    # matmulLd: rows ldo apart, gaps untouched
    ldo = n + 72
    for m in (1, 17, 100, mm):
        yl = torch.full((m, ldo), -7.0, dtype=torch.bfloat16, device="cuda")
        hip.matmul(lin, x[:m], yl, ldo=ldo)
        if not torch.equal(yl[:, :n], ybf[:m]) or not bool((yl[:, n:] == -7.0).all()):
            bad.append(f"ld m {m}")
    # strided inputs (x rows wider than K)
    xs = torch.zeros((mm, k + 64), dtype=torch.bfloat16, device="cuda")
    xs[:, :k] = x
    ys = torch.empty_like(ybf)
    hip.matmul(lin, xs[:, :k], ys, ldx=k + 64)
    if not torch.equal(ys, ybf):
        bad.append("strided x")
    ok = ok and not bad
    log(f"{'PASS' if ok else 'FAIL'} {name}: n {n} k {k} sk {sk} rows {mm}: fp32 worst {float(units.max()):.2f} "
        f"(bound {bound:.0f}) units of 2^-24 sum|xws| at ({r}, {c}) |err| {float(err.view(-1)[worst]):.3e} "
        f"ref {float(ref.view(-1)[worst]):.4e}; per-column worst median {float(col_worst.median()):.2f} max "
        f"{float(col_worst.max()):.2f}; bf16 == RNE(fp32) {rne}, bf16 err / (half ulp + fp32 err) {ulps:.3f}; invariance/determinism/ld/strided "
        f"{'ok' if not bad else bad}")
    return ok


def exact_check(hip, log):
    """Every e4m3 code (but NaN) through one-hot rows: out[m, j] = e4m3(code[j, m]) * 2^-3 exactly. Odd columns hold
    negative codes and meet rows whose zeros are -0: RDNA3 WMMA sums a -0 product as -1 unit of its internal sum
    (a 1-ulp nudge, deterministic; tools/rocm notes), so no product here is -0."""
    n, k = 256, 128
    pos = torch.stack([torch.randperm(128, generator=torch.Generator().manual_seed(i)) for i in range(n)]).to(torch.uint8)
    pos[pos == 0x7F] = 0x00
    codes = pos | (torch.arange(n) % 2 == 1).to(torch.uint8).view(-1, 1) * 0x80
    w = codes.cuda().view(torch.float8_e4m3fn)
    lin = Fp8BlockLinear.from_checkpoint(w, torch.full((2, 1), 0.125, device="cuda"))
    eye = torch.eye(k, dtype=torch.bfloat16, device="cuda")
    ok = True
    for f32 in (False, True):
        want = (w.float() * 0.125).t().to(torch.float32 if f32 else torch.bfloat16)
        for sign, x in ((0, eye), (1, torch.where(eye == 0, -0.0, eye).to(torch.bfloat16))):
            y = torch.empty((k, n), dtype=want.dtype, device="cuda")
            hip.matmul(lin, x, y, f32=f32)
            cols = slice(sign, None, 2)
            ok = ok and torch.equal(y[:, cols], want[:, cols])      # by value: -0 code 0x80 sums to +0
    log(f"{'PASS' if ok else 'FAIL'} exact: all 254 e4m3 codes (subnormals, zeros, signs) one-hot through the kernel")
    return ok


def random_linear(n, k, seed):
    g = torch.Generator().manual_seed(seed)
    codes = torch.randint(0, 254, (n, k), generator=g, dtype=torch.uint8)
    codes[codes >= 0x7F] += 1
    inv = torch.exp(torch.empty((-(-n // 128), k // 128)).uniform_(-9.0, -4.0, generator=g)).float()
    return make(codes.cuda().view(torch.float8_e4m3fn), inv.cuda())


def make(w, inv):
    lin = Fp8BlockLinear.from_checkpoint(w, inv)
    lin.dense = w.float() * lin.column_scales(inv, lin.n, lin.k).repeat_interleave(64, dim=1)
    return lin


def read_tensor(model: Path, name: str):
    idx = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
    path = model / idx[name]
    with open(path, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        h = json.loads(f.read(hl))[name]
        a, b = h["data_offsets"]
        f.seek(8 + hl + a)
        raw = bytearray(f.read(b - a))
    dt = {"F8_E4M3": torch.float8_e4m3fn, "F32": torch.float32, "BF16": torch.bfloat16}[h["dtype"]]
    return torch.frombuffer(raw, dtype=torch.uint8).view(dt).view(h["shape"]).cuda()


REAL = ["model.language_model.layers.0.linear_attn.in_proj_qkv", "model.language_model.layers.0.linear_attn.in_proj_z",
        "model.language_model.layers.0.linear_attn.out_proj", "model.language_model.layers.3.self_attn.q_proj",
        "model.language_model.layers.3.self_attn.k_proj", "model.language_model.layers.3.self_attn.o_proj",
        "model.language_model.layers.3.mlp.shared_expert.gate_proj",
        "model.language_model.layers.3.mlp.shared_expert.down_proj"]


def bench(hips, log, tries=15, only=None, rows=(1, 8, 16, 32, 64, 128, 512, 2048)):
    """Best-of-`tries` mean launch time, weights rotated over copies past the 32 MiB MALL: decode GB/s (weight bytes)
    and prefill TFLOP/s. Production shares the GPU: compare runs A/B, not against peak."""
    for name, n, k in SHAPES[:5]:
        if only and name not in only:
            continue
        lins = [random_linear(n, k, 7)]
        wbytes = lins[0].w8.numel() + lins[0].bs.numel()
        for _ in range(max(1, (96 << 20) // wbytes)):
            lins.append(Fp8BlockLinear(lins[0].w8.clone(), lins[0].bs.clone(), n, k, lins[0].npad))
        x = torch.randn((max(rows), k), device="cuda").to(torch.bfloat16)
        y = torch.empty((max(rows), n), dtype=torch.bfloat16, device="cuda")
        parts = []
        for m in rows:
            reps = len(lins) * (4 if m <= 128 else 1)
            best = [float("inf")] * len(hips)
            for _ in range(tries):                         # A/B alternated: production's load hits both alike
                for h, hip in enumerate(hips):
                    e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                    e0.record()
                    for i in range(reps):
                        hip.matmul(lins[i % len(lins)], x[:m], y)
                    e1.record()
                    e1.synchronize()
                    best[h] = min(best[h], e0.elapsed_time(e1) * 1e3 / reps)
            parts.append(f"m{m} " + "/".join(f"{b:.0f}" for b in best) + "us " + "/".join(
                f"{wbytes / b / 1e3:.0f}" if m <= 128 else f"{2 * m * n * k / b / 1e6:.1f}" for b in best)
                + ("GB/s" if m <= 128 else "TF"))
        log(f"bench {name} n {n} k {k}: " + ", ".join(parts))
        del lins
        torch.cuda.empty_cache()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--co", default=str(Path(__file__).resolve().parents[2] / "build/rocm"))
    ap.add_argument("--real", help="model dir: check its block-FP8 tensors (a few layers)")
    ap.add_argument("--bench", nargs="*", help="benchmark (optionally only these shape names)")
    ap.add_argument("--co-b", help="second code-object dir: --bench times both, alternated (A/B)")
    ap.add_argument("--rows", help="--bench row counts, comma separated")
    ap.add_argument("--quick", action="store_true", help="skip the random shapes")
    a = ap.parse_args()
    hip = Hip(Path(a.co))
    log = lambda s: print(s, flush=True)  # noqa: E731
    ok = exact_check(hip, log)
    if not a.quick and a.bench is None:
        for i, (name, n, k) in enumerate(SHAPES):
            lin = random_linear(n, k, 1000 + i)
            x = torch.randn((2048, k), generator=torch.Generator().manual_seed(i), device="cpu").to(torch.bfloat16).cuda()
            ok = check_case(hip, name, lin, x, log) and ok
            del lin, x
            torch.cuda.empty_cache()
    if a.real:
        for p in REAL:
            lin = make(read_tensor(Path(a.real), p + ".weight"), read_tensor(Path(a.real), p + ".weight_scale_inv"))
            x = (torch.randn((512, lin.k), generator=torch.Generator().manual_seed(5)) * 0.5).to(torch.bfloat16).cuda()
            ok = check_case(hip, p.split("layers.")[1], lin, x, log) and ok
            del lin, x
            torch.cuda.empty_cache()
    if a.bench is not None:
        hips = [hip] + ([Hip(Path(a.co_b))] if a.co_b else [])
        bench(hips, log, only=a.bench, **({"rows": [int(r) for r in a.rows.split(",")]} if a.rows else {}))
    log("PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
