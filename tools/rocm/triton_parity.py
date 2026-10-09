#!/usr/bin/env python3
"""The HIP kernel set (``flashnext_aot.py build --target hip``) against the ROCm Triton JIT, in the dev image.

  hash: every spec entry is warmed up through the JIT (MockTensor pointers: nothing allocated, nothing launched); its
        kernel hash must be in aot.json and its hsaco byte-equal to the AOT one.
  tiles: every row-tiled matmul variant (_b16mm, _b16mm_ks, _router, _hc_up_mix; both pointer forms) on a partial
        last row tile against whole tiles: each row's bits must not depend on M.
  run:  the Python wrappers (tools/zig/flashnext_triton_fixtures.py's calls, small rows) on random GPU tensors; each
        Triton launch runs through the JIT, then again from aot.json's hsaco with the Zig launcher's ABI (variant by
        aot.zig's matching rules, runtime args + two null scratch pointers, 32 * num_warps threads, metadata shared
        bytes) on the restored inputs; every tensor argument must come out bit-equal.

    PYTHONPATH=src python tools/rocm/triton_parity.py hash --aot <dir> --spec zig/tests/cuda/flashnext/kernels.json ...
    PYTHONPATH=src python tools/rocm/triton_parity.py run --aot <dir>
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import struct
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools" / "zig"))
import flashnext_aot as A  # noqa: E402


# ------------------------------------------------------------------------------------------------------------- hash

class Ptr:
    """A MockTensor whose address is 16-aligned or not, as the spec's divisibility says, within 2 GiB (Triton's
    ``is_within_2gb``: the JIT then specializes it to buffer ops) or not."""

    def __init__(self, dtype, aligned: bool, small: bool = False) -> None:
        self.dtype, self.addr, self.small = dtype, 0 if aligned else 8, small

    def data_ptr(self):
        return self.addr

    def ptr_range(self):
        return 1024 if self.small else 1 << 40


def jit_kwargs(k: dict):
    import triton.language as tl

    reg = lambda key: A.resolve(key.replace(":", "."))       # noqa: E731
    kw = {}
    for p in k["params"]:
        t = k["signature"][p]
        div = ["tt.divisibility", 16] in k["attrs"].get(p, [])
        if t == "constexpr":
            kw[p] = A.value(k["constexprs"][p], tl, reg)
        elif t.startswith("*"):
            short = t[1:]
            name = {"i": "int", "u": "uint"}.get(short[0], "") + short[1:] if short[0] in "iu" else short
            kw[p] = Ptr(tl.dtype(name), div, bool(k.get("range32")))
        elif t in ("i32", "i64", "u32", "u64"):
            kw[p] = 32 if div else 7
        elif t == "fp32":
            kw[p] = 1.5
        else:
            raise SystemExit(f"{k['function']}: parameter {p} of type {t}")
    return kw


def hash_check(aot: Path, specs: list[Path]) -> int:
    rows = {r["hash"]: r for r in json.loads((aot / "aot.json").read_text())["kernels"]}
    bad = 0
    n = 0
    for spec in specs:
        for k in (x for e in json.loads(spec.read_text())["kernels"] for x in A.hip_entries(e)):
            fn = A.resolve(k["function"])
            opts = A.hip_options(k)
            ck = fn.warmup(grid=(1,), **jit_kwargs(k), **opts)
            sha = hashlib.sha256(ck.asm["hsaco"]).hexdigest()
            r = rows.get(ck.hash)
            n += 1
            if r is None or r["cubin_sha256"] != sha:
                bad += 1
                print(f"MISMATCH {k['function']} {k.get('rule', k.get('hash', ''))[:60]}: JIT {ck.hash[:12]} "
                      f"{'not in aot.json' if r is None else 'hsaco differs'}")
    print(f"{n} spec entries warmed up through the JIT, {n - bad} equal to the AOT hsaco, {bad} not")
    return 1 if bad else 0


# -------------------------------------------------------------------------------------------------------------- run

def zig_match(row: dict, args: list[tuple], consts: dict, small: bool) -> bool:
    """aot.zig's ``matches``: args are (name, kind, value, type) with kind ptr / i32 / f32; ``small``: every pointer's
    storage within 2 GiB (aot.zig's allSmall, as the JIT decides it)."""

    for n, c in consts.items():
        got = row["consts"].get(n)
        if got is None or any(got.get(t) != c[t] for t in ("int", "f32") if t in c):
            return False
    params = {p["name"]: p for p in row["params"]}
    runtime = 0
    for name, kind, v, ty in args:
        p = params.get(name)
        if kind == "i32":
            if p is None:
                if v != 1 or row["consts"].get(name, {}).get("int") != 1:
                    return False
                continue
            if p["type"] != "i32" or (not p["nospec"] and v == 1):
                return False
            if p["div16"] != (not p["nospec"] and v % 16 == 0):
                return False
        elif kind == "ptr":
            if p is None or p["type"] != ty or p["div16"] != (v % 16 == 0) or p.get("range32", False) != small:
                return False
        elif p is None or p["type"] != "fp32":
            return False
        runtime += 1
    return runtime == len(row["params"])


class Hip:
    def __init__(self, aot: Path) -> None:
        from triton.backends.amd.driver import _get_path_to_hip_runtime_dylib

        self.lib = ctypes.CDLL(_get_path_to_hip_runtime_dylib())
        self.aot = aot
        data = json.loads((aot / "aot.json").read_text())
        self.rows, self.dir, self.ext = data["kernels"], data.get("bin_dir", "hsaco"), data.get("bin_ext", "hsaco")
        self.funcs: dict[str, ctypes.c_void_p] = {}

    def function(self, key: str, name: str, image=None):
        if key not in self.funcs:
            image = image or (self.aot / self.dir / f"{key}.{self.ext}").read_bytes()
            mod, f = ctypes.c_void_p(), ctypes.c_void_p()
            assert self.lib.hipModuleLoadData(ctypes.byref(mod), ctypes.c_char_p(image)) == 0
            assert self.lib.hipModuleGetFunction(ctypes.byref(f), mod, name.encode()) == 0
            self.funcs[key] = (f, image)
        return self.funcs[key][0]

    def launch(self, row: dict, grid, args: list[tuple], stream: int, func=None) -> None:
        """aot.zig's ``run``: runtime args in the variant's order, then null global and profile scratch pointers."""

        by = {a[0]: a for a in args}
        vals = []
        for p in row["params"]:
            _, kind, v, _ = by[p["name"]]
            vals.append(ctypes.c_uint64(v) if kind == "ptr" else ctypes.c_int32(v) if kind == "i32" else ctypes.c_float(v))
        vals += [ctypes.c_uint64(0), ctypes.c_uint64(0)]
        params = (ctypes.c_void_p * len(vals))(*[ctypes.cast(ctypes.byref(v), ctypes.c_void_p) for v in vals])
        g = list(grid) + [1] * (3 - len(grid))
        f = func or self.function(row["hash"], row["name"])
        rc = self.lib.hipModuleLaunchKernel(f, *map(ctypes.c_uint, g), ctypes.c_uint(row.get("warp_size", 32) * row["num_warps"]),
                                            ctypes.c_uint(1), ctypes.c_uint(1), ctypes.c_uint(row["shared"]),
                                            ctypes.c_void_p(stream), params, None)
        assert rc == 0, f"hipModuleLaunchKernel {rc}"


BENCH = {"n": 0, "rows": []}


def bench(hip: Hip, row: dict, ck, g, args) -> None:
    """Best-of-5 alternated batches of 20 launches: the AOT hsaco and the JIT's own (buffer-op) binary, same ABI."""

    import torch

    stream = torch.cuda.current_stream().cuda_stream
    jf = hip.function("jit:" + ck.hash, row["name"], ck.asm["hsaco"])
    best = {"aot": 1e9, "jit": 1e9}
    for _ in range(5):
        for which, f in (("aot", None), ("jit", jf)):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            for _ in range(20):
                hip.launch(row, g, args, stream, f)
            e1.record()
            e1.synchronize()
            best[which] = min(best[which], e0.elapsed_time(e1) * 1000 / 20)
    ints = " ".join(f"{n}={v}" for n, k_, v, _ in args if k_ == "i32")
    BENCH["rows"].append((row["fn"], row["hash"][:10], tuple(g), ints, best["aot"], best["jit"], row.get("n_spills")))


TYPES = None
TYPES = None
STATS = {"launches": 0, "equal": 0, "differ": 0, "novariant": 0, "fns": {}}


def interceptor(hip: Hip, jit):
    import torch

    class Run:
        def __init__(self):
            self.jit = jit

        def __getitem__(self, grid):
            def call(*a, **kw):
                names = jit.arg_names
                bound = dict(zip(names, a))
                bound.update({k: v for k, v in kw.items() if k in names})
                args, consts, tensors = [], {}, []
                for p in jit.params:
                    if p.name not in bound:
                        if p.is_constexpr and p.has_default:
                            consts[p.name] = p.default
                        continue
                    v = bound[p.name]
                    if p.is_constexpr or v is None:
                        consts[p.name] = v
                    elif isinstance(v, torch.Tensor):
                        args.append((p.name, "ptr", v.data_ptr(), TYPES[v.dtype]))
                        tensors.append(v)
                    elif isinstance(v, int):
                        args.append((p.name, "i32", v, "i32"))
                    else:
                        args.append((p.name, "f32", v, "fp32"))
                zc = {n: c for n, c in ((n, A_const(v)) for n, v in consts.items()) if c is not None}
                g = grid(bound) if callable(grid) else grid
                before = [t.clone() for t in tensors]
                ck = jit[grid](*a, **kw)
                after = [t.clone() for t in tensors]
                fn = jit.fn.__name__
                STATS["launches"] += 1
                small = all(t.untyped_storage().size() <= A.MAX_RANGE for t in tensors)
                rows = [r for r in hip.rows if r["fn"] == fn and zig_match(r, args, zc, small)]
                st = STATS["fns"].setdefault(fn, [0, 0, 0])
                if not rows:
                    STATS["novariant"] += 1
                    st[2] += 1
                    return
                for t, b in zip(tensors, before):
                    t.copy_(b)
                hip.launch(rows[0], g, args, torch.cuda.current_stream().cuda_stream)
                torch.cuda.synchronize()
                same = all(torch.equal(t.view(torch.uint8) if t.dtype.itemsize == 1 else t.view(_int(t.dtype)),
                                       x.view(torch.uint8) if x.dtype.itemsize == 1 else x.view(_int(x.dtype)))
                           for t, x in zip(tensors, after))
                STATS["equal" if same else "differ"] += 1
                st[0 if same else 1] += 1
                if BENCH["n"] and rows[0]["hash"][:10] not in {b[1] for b in BENCH["rows"]}:
                    bench(hip, rows[0], ck, g, args)
                if not same:
                    print(f"DIFFER {fn} grid {g} " + ", ".join(f"{n}={v}" for n, k_, v, _ in args if k_ != "ptr"))
            return call

    return Run()


def _int(dtype):
    import torch

    return {2: torch.int16, 4: torch.int32, 8: torch.int64}[dtype.itemsize]


def A_const(v):
    if isinstance(v, bool):
        return {"int": int(v)}
    if isinstance(v, int):
        return {"int": v}
    if isinstance(v, float):
        return {"f32": struct.unpack("<I", struct.pack("<f", v))[0]}
    return None


HIP = None


def run_check(aot: Path, cases: str, n_bench: int, specs: list[Path]) -> int:
    import torch

    import flashnext_triton_fixtures as F

    global TYPES
    TYPES = F.TYPES | {torch.float8_e4m3fn: "*fp8e4nv"}
    global HIP
    hip = HIP = Hip(aot)
    BENCH["n"] = n_bench
    gen = torch.Generator(device="cuda").manual_seed(0)

    def z(shape, dtype=torch.bfloat16):
        if dtype.is_floating_point:
            return (torch.randn(shape, device="cuda", generator=gen) * 0.5).to(dtype)
        return torch.zeros(shape, dtype=dtype, device="cuda")

    F.z = z
    F.e = z                                      # real storage: the kernels read every row
    F.ROWS = ROWS
    real_scratch = F.attn_mod.AttnScratch
    F.attn_mod.AttnScratch = lambda *a, **kw: real_scratch(*a[:4], "cuda", **kw)
    keep = cases.split(",")
    real_case = F.case

    def case(store, name, fn):
        if any(name.startswith(c) for c in keep) and "c1048576" not in name and "c262151" not in name \
                and ("/head/" not in name or name.startswith("b16/head/r1/tp1")):
            real_case(store, name, fn)

    F.case = case
    for mod, names in ((F.glue, ("_hc_writeback", "_hc_normed", "_hc_act", "_hc_mix", "_rmsnorm", "_attn_prep",
                                 "_attn_gate", "_ple_embed_bf16", "_ple_gate", "_ple_conv", "_add_streams",
                                 "_moe_partial")),
                       (F.bf16, ("_b16mm", "_reduce")), (F.nvfp4, ("_fp4mm", "_reduce")), (F.exl3_mm, ("_embed",)),
                       (F.attn_mod, ("_chunks", "_merge", "_pool", "_scores", "_select", "_select_tiles")),
                       (F.forward, ("_shift_windows",)), (F.moe_mod, ("_router", "_topk_rows"))):
        for n in names:
            setattr(mod, n, interceptor(hip, getattr(mod, n)))
    from tensorfold.families.qwen4_exp.cuda import kv8

    for n in ("_attn_prep8", "_chunks8", "_merge"):
        setattr(kv8, n, interceptor(hip, getattr(kv8, n)))
    F.patch = lambda: None
    F.build()
    if "kv8" in keep:
        kv8_cases(F, z)
    if "direct" in keep:
        direct_cases(z, specs)
    print(json.dumps({k: v for k, v in STATS.items() if k != "fns"}))
    for fn, (eq, df, nv) in sorted(STATS["fns"].items()):
        print(f"  {fn:16s} equal {eq:4d} differ {df:3d} no variant {nv:3d}")
    if BENCH["rows"]:
        print("us a launch (best of 5 x 20, prod GPU noise), the AOT hsaco vs the JIT's own buffer-op binary:")
        for fn, h, g, ints, ta, tj, sp in sorted(BENCH["rows"], key=lambda r: -r[4]):
            print(f"  {fn:16s} {h} grid {str(g):16s} {ints:28s} aot {ta:8.1f}  jit {tj:8.1f}  spills {sp}")
    return 1 if STATS["differ"] else 0


def kv8_cases(F, z) -> None:
    """--kv-dtype fp8 (kv8.py): M rows (ROWS) written by _attn_prep8 into uint8 caches, then attended by _chunks8:
    dense rows near the start, and at 262144 keys sparse rows (the indexer's selected blocks and tail) far in."""

    import torch

    from tensorfold.families.qwen4_exp.cuda import kv8

    D, HD, NI, IHD, HALF, EPS = F.D, F.HD, F.NI, F.IHD, F.HALF, F.EPS
    for world, rk in F.RANKS.items():
        heads, kv = rk["heads"], rk["kv"]
        pw = heads * 2 * HD + 2 * kv * HD + (NI + 1) * IHD
        for capacity, p0 in ((1024, 5), (262144, 5), (262144, 200000)):
            kc = torch.zeros((capacity, kv, HD + kv8.PAD), dtype=torch.uint8, device="cuda")
            vc = torch.zeros((capacity, kv, HD), dtype=torch.uint8, device="cuda")
            ikc, pos0 = z((capacity, IHD)), torch.full((1,), p0, dtype=torch.int32, device="cuda")
            if p0 > 5:                      # the positions before p0: written once, untraced
                kc[:p0], vc[:p0] = kv8.quantize(z((p0, kv, HD)), z((p0, kv, HD)))
            for r in ROWS:
                if p0 + r > capacity:
                    continue
                q, iq = z((r, heads, HD)), z((r, NI, IHD))
                kv8.attn_prep(z((r, pw)), pos0, z((HD,), torch.float32), z((HD,), torch.float32),
                              z((IHD,), torch.float32), z((HALF,), torch.float32), q, kc, vc, iq, ikc, EPS,
                              q_heads=heads, kv_heads=kv, head_dim=HD, index_heads=NI, index_dim=IHD)
                sc = F.attn_mod.AttnScratch(r, heads, HD, capacity, "cuda")
                ctx = p0 + r
                if sc.qsa:
                    F.attn_mod.qsa_select(iq, ikc, z((-(-capacity // 4), IHD)), pos0, z((IHD,), torch.float32),
                                          z((HALF,), torch.float32), EPS, sc, r, context=ctx)
                kv8.attention(q, kc, vc, pos0, sc, r, HD ** -0.5, context=ctx)
            del kc, vc, ikc
            torch.cuda.empty_cache()


ROWS = (1, 3, 16, 17, 129, 161, 2049)


def direct_cases(z, specs: list[Path]) -> None:
    """Kernels the Python wrappers never launch (prompt_mm's, _select_tiles): every spec entry's constexprs and warps,
    at each row count whose int form it was built for, on buffers sized from its source."""

    import torch
    import triton

    from tensorfold.families.qwen4_exp.cuda import attention as attn_mod, bf16 as b16_mod, prompt_mm as pm

    names = ("_b16mm_ks", "_b16mm_ks_sm", "_fp4mm_ks", "_hc_up_mix", "_hc_wb_norm", "_scores_rows")
    mods = {n: pm for n in names} | {"_select_tiles": attn_mod, "_b16mm": b16_mod, "_b16mm_sm": b16_mod}
    wrapped = {n: getattr(m, n) if hasattr(getattr(m, n), "jit") else interceptor(HIP, getattr(m, n))
               for n, m in mods.items()}
    dt = {"*bf16": torch.bfloat16, "*fp32": torch.float32, "*u16": torch.uint16, "*u8": torch.uint8,
          "*i32": torch.int32, "*fp16": torch.float16}
    seen = set()
    for spec in specs:
        for k in (x for e in json.loads(spec.read_text())["kernels"] for x in A.hip_entries(e)):
            if k["name"] not in mods or (k["name"] == "_b16mm" and "gfx1151 tile" not in k.get("rule", "")):
                continue                              # _b16mm: the wrappers' launches cover the spec's own
            key = json.dumps([k["name"], k["constexprs"], k["signature"], k["attrs"], k["options"]["num_warps"]])
            if key in seen:
                continue
            seen.add(key)
            c = {n: (v.get("int") if "int" in v else v.get("bool")) for n, v in k["constexprs"].items()}
            sig = k["signature"]
            t = lambda n, shape: z(shape, dt[sig[n]]) if sig[n] != "*u16" else z(shape).view(torch.uint16)  # noqa
            row_int = {"_hc_wb_norm": "RS", "_scores_rows": "ROWS", "_select_tiles": "NB"}.get(k["name"], "M")
            div = bool(k["attrs"].get(row_int))
            for m in ROWS:
                if k["name"] in ("_hc_wb_norm", "_select_tiles", "_scores_rows") and m > 161:
                    continue
                kw = {}
                if k["name"] in ("_b16mm", "_b16mm_sm"):
                    n, kk, sk = c["N"], c["K"], c["SK"]
                    if (m % 16 == 0) != div or n * kk > 1 << 26 or ("M" in c) != (m == 1) or (m > 128) != (c["BM"] == 128):
                        continue
                    out = t("OUT", (m, n))
                    kw = dict(X=t("X", (m, kk)), W=t("W", (n, kk)), OUT=out, PART=z((sk, m, n), torch.float32) if sk > 1 else out,
                              x_stride=kk, M=m)
                    grid = (triton.cdiv(m, c["BM"]), triton.cdiv(n, c["BLOCK_N"]), sk)
                elif k["name"] in ("_b16mm_ks", "_b16mm_ks_sm"):
                    if (m % 16 == 0) != div:
                        continue
                    n, kk = c["N"], c["K"]
                    kw = dict(X=t("X", (m, kk)), W=t("W", (n, kk)), OUT=t("OUT", (m, n)), M=m, x_stride=kk)
                    grid = (triton.cdiv(m, c["BM"]) * triton.cdiv(n, c["BLOCK_N"]),)
                elif k["name"] == "_fp4mm_ks":
                    if (m % 16 == 0) != div:
                        continue
                    n, kk = c["N"], c["K"]
                    kw = dict(X=t("X", (m, kk)), W=t("W", ((n // c["SBN"]) * (kk // 64) * 64 * c["SBN"],)),
                              S=t("S", ((kk // 16) * n,)), S2=t("S2", (n,)), OUT=t("OUT", (m, n)), M=m, x_stride=kk)
                    grid = (triton.cdiv(m, c["BM"]), triton.cdiv(n, c["BLOCK_N"]))
                elif k["name"] == "_hc_up_mix":
                    if (m % 16 == 0) != div:
                        continue
                    d, st, kk = c["D"], c["S"], c["K"]
                    kw = dict(ACT=t("ACT", (m, kk)), W=t("W", (st * d, kk)), NORMED=t("NORMED", (m, st * d)),
                              MIXED=t("MIXED", (m, d)), M=m)
                    grid = (triton.cdiv(m, c["BM"]), d // c["BD"])
                elif k["name"] == "_hc_wb_norm":
                    d, st, sl = c["D"], c["S"], c["SLOTS"]
                    kw = dict(H=t("H", (m, st * d)), HOUT=t("HOUT", (m, st * d)), PSS=t("PSS", (m, d // c["BLOCK"], st)),
                              BR=t("BR", (c["WORLD"], m, d)), INJ=t("INJ", (m, st)), Y=t("Y", (m, sl, d)),
                              WTS=t("WTS", (m, sl)), RS=m * d, SCALE=t("SCALE", (st * d,)),
                              NORMED=t("NORMED", (m, st * d)), eps=1e-6)
                    grid = (m,)
                elif k["name"] == "_scores_rows":
                    nb = 1008 if div else 1001
                    if (m % 16 == 0) != bool(k["attrs"].get("ROWS")) or m == 1:
                        continue
                    pos0 = torch.full((1,), 4 * nb - m - 3, dtype=torch.int32, device="cuda")
                    kw = dict(IQ=t("IQ", (m, c["HI"], c["DI"])), POOLED=t("POOLED", (nb, c["DI"])), POS0=pos0,
                              SC=t("SC", (m, nb)), NB=nb, ROWS=m)
                    if bool(k["attrs"].get("NB")) != (nb % 16 == 0):
                        continue
                    grid = (triton.cdiv(m, c["RT"]), triton.cdiv(nb, c["BB"]))
                else:                                        # _select_tiles: a row's blocks past _select's registers
                    nb = 40000 if div else 40001
                    pos0 = torch.full((1,), 4 * nb - m - 5, dtype=torch.int32, device="cuda")
                    kw = dict(SC=t("SC", (m, nb)), POS0=pos0, IDS=t("IDS", (m, c["IDW"])), NKR=t("NKR", (m,)),
                              SPR=t("SPR", (m,)), NB=nb)
                    grid = (m,)
                consts = {n: A.value(v, __import__("triton.language", fromlist=["x"]), None)
                          for n, v in k["constexprs"].items() if n not in kw}
                wrapped[k["name"]][grid](**kw, **consts, **A.hip_options(k))

# ------------------------------------------------------------------------------------------------------------ tiles

def _tile_args(row: dict, m: int, base: dict | None = None):
    """Inputs for a row-tiled matmul at ``m`` rows (random, or the first ``m`` rows of ``base``'s), zeroed outputs:
    tensors by name, grid, the compared output (its rows axis) and the strides."""

    import torch

    c = {n: v.get("int") for n, v in row["consts"].items()}
    gen = torch.Generator(device="cuda").manual_seed(7)
    bf = lambda *s: (torch.randn(s, device="cuda", generator=gen) * 0.5).to(torch.bfloat16)   # noqa: E731
    zero = lambda *s, dt=torch.float32: torch.zeros(s, device="cuda", dtype=dt)               # noqa: E731
    rows = lambda name, x: base[name][:m].contiguous() if base else x                         # noqa: E731
    fn = row["fn"]
    if fn in ("_b16mm", "_b16mm_ks", "_b16mm_sm", "_b16mm_ks_sm"):
        n, k, sk = c["N"], c["K"], c["SK"]
        out = zero(m, n, dt=torch.float32 if c["F32"] else torch.bfloat16)
        t = {"X": rows("X", bf(m, k)), "W": base["W"] if base else bf(n, k), "OUT": out}
        if fn.startswith("_b16mm_ks"):
            return t, (-(-m // c["BM"]) * -(-n // c["BLOCK_N"]), 1, 1), ("OUT", 0), {"x_stride": k}
        t["PART"] = zero(sk, m, n) if sk > 1 else out
        return t, (-(-m // c["BM"]), -(-n // c["BLOCK_N"]), sk), ("PART", 1) if sk > 1 else ("OUT", 0), {"x_stride": k}
    if fn == "_router":
        d, ne = c["D"], c["NE"]
        t = {"X": rows("X", bf(m, d)), "W": base["W"] if base else bf(ne, d), "OUT": zero(m, ne)}
        return t, (-(-m // c["BM"]), -(-ne // c["BLOCK_E"]), 1), ("OUT", 0), {"x_stride": d}
    d, s_, k = c["D"], c["S"], c["K"]                                   # _hc_up_mix
    t = {"ACT": rows("ACT", bf(m, k)), "W": base["W"] if base else bf(s_ * d, k),
         "NORMED": rows("NORMED", bf(m, s_ * d)), "MIXED": zero(m, d, dt=torch.bfloat16)}
    return t, (-(-m // c["BM"]), d // c["BD"], 1), ("MIXED", 0), {}


def tiles_check(aot: Path) -> int:
    """Every row-tiled matmul variant with M a runtime int (both pointer forms): a launch whose last row tile is
    partial gives each row the bits a launch of whole tiles gives it (the gfx1151 miscompile of heavily spilled
    masked loads broke exactly this), and the two pointer forms of one specialization give the same bits."""

    import torch

    hip = Hip(aot)
    stream = torch.cuda.current_stream().cuda_stream
    bad = n = 0
    forms: dict = {}
    u8 = lambda x: x.contiguous().view(torch.uint8)                     # noqa: E731
    for row in hip.rows:
        if row["fn"] not in ("_b16mm", "_b16mm_ks", "_b16mm_sm", "_b16mm_ks_sm", "_router", "_hc_up_mix"):
            continue
        c = {k: v.get("int") for k, v in row["consts"].items()}
        mp = next((p for p in row["params"] if p["name"] == "M"), None)
        if mp is None or c.get("N", 0) * c.get("K", 0) > 1 << 26:      # one row only, or a head-sized weight
            continue
        bm = c["BM"]
        full = 2 * bm if bm >= 64 else 64
        ms = [x for x in ((bm + 16, full - 16) if mp["div16"] else (bm + 1, full - 3)) if x % bm]
        pr = any(p.get("range32") for p in row["params"])

        def launch(t, grid, ints, m):
            args = [(p["name"], "ptr", t[p["name"]].data_ptr(), p["type"]) if p["type"].startswith("*") else
                    (p["name"], "i32", m if p["name"] == "M" else ints[p["name"]], "i32") for p in row["params"]]
            hip.launch(row, grid, args, stream)
            torch.cuda.synchronize()

        whole, grid, (o, ax), ints = _tile_args(row, full)
        launch(whole, grid, ints, full)
        for m in ms:
            t, grid, _, _ = _tile_args(row, m, whole)
            launch(t, grid, ints, m)
            n += 1
            want = whole[o][:m] if ax == 0 else whole[o][:, :m]
            key = (row["fn"], json.dumps(row["consts"], sort_keys=True), mp["div16"], m)
            prev = forms.setdefault(key, t[o])
            if not torch.equal(u8(want), u8(t[o])) or not torch.equal(u8(prev), u8(t[o])):
                bad += 1
                print(f"DIFFER {row['fn']} {row['hash'][:10]} {c} range32 {pr} M {m}")
    print(f"{n} partial-tile launches against whole tiles and the other pointer form, {bad} differ")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash")
    h.add_argument("--aot", required=True)
    h.add_argument("--spec", action="append", required=True)
    r = sub.add_parser("run")
    r.add_argument("--aot", required=True)
    r.add_argument("--spec", action="append", default=[str(ROOT / "zig/tests/cuda/flashnext/kernels.json"),
                                                        str(ROOT / "zig/tests/cuda/flashnext/kernels_int4ar.json")])
    r.add_argument("--bench", type=int, default=0, help="1: time each variant's first launch (AOT vs JIT binary)")
    r.add_argument("--cases", default="embed/,hc_,rmsnorm/,add_streams,ple_,moe_partial,moe/,attn_prep/,attn_gate/,"
                   "fp4/,b16/,attention/tp1/c1024/r3/p100,attention/tp1/c262144/r3/p3000/x8192,"
                   "attention/tp2/c262144/r16/p3000/x16384,attention_prefill/tp1/c1024/s0/n30,"
                   "attention_prefill/tp1/c262144/s4096/n19,attention_prefill/tp1/c262144/s2048/n974,shift/,kv8,direct")
    t = sub.add_parser("tiles")
    t.add_argument("--aot", required=True)
    a = ap.parse_args()
    if a.cmd == "tiles":
        return tiles_check(Path(a.aot))
    if a.cmd == "hash":
        return hash_check(Path(a.aot), [Path(s) for s in a.spec])
    return run_check(Path(a.aot), a.cases, a.bench, [Path(x) for x in a.spec])


if __name__ == "__main__":
    sys.exit(main())
