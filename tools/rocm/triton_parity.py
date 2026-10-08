#!/usr/bin/env python3
"""The HIP kernel set (``flashnext_aot.py build --target hip``) against the ROCm Triton JIT, in the dev image.

  hash: every spec entry is warmed up through the JIT (MockTensor pointers: nothing allocated, nothing launched); its
        kernel hash must be in aot.json and its hsaco byte-equal to the AOT one.
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
    """A MockTensor whose address is 16-aligned or not, as the spec's divisibility says."""

    def __init__(self, dtype, aligned: bool) -> None:
        self.dtype, self.addr = dtype, 0 if aligned else 8

    def data_ptr(self):
        return self.addr


def jit_kwargs(k: dict):
    import triton.language as tl

    reg = lambda key: A.resolve(key.replace(":", "."))       # noqa: E731
    kw = {}
    for p in k["params"]:
        t = k["signature"][p]
        div = bool(k["attrs"].get(p))
        if t == "constexpr":
            kw[p] = A.value(k["constexprs"][p], tl, reg)
        elif t.startswith("*"):
            short = t[1:]
            name = {"i": "int", "u": "uint"}.get(short[0], "") + short[1:] if short[0] in "iu" else short
            kw[p] = Ptr(tl.dtype(name), div)
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
        for k in json.loads(spec.read_text())["kernels"]:
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

def zig_match(row: dict, args: list[tuple], consts: dict) -> bool:
    """aot.zig's ``matches``: args are (name, kind, value, type) with kind ptr / i32 / f32."""

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
            if p is None or p["type"] != ty or p["div16"] != (v % 16 == 0):
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
                rows = [r for r in hip.rows if r["fn"] == fn and zig_match(r, args, zc)]
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


def run_check(aot: Path, cases: str, n_bench: int) -> int:
    import torch

    import flashnext_triton_fixtures as F

    global TYPES
    TYPES = F.TYPES | {torch.float8_e4m3fn: "*fp8e4nv"}
    hip = Hip(aot)
    BENCH["n"] = n_bench
    gen = torch.Generator(device="cuda").manual_seed(0)

    def z(shape, dtype=torch.bfloat16):
        if dtype.is_floating_point:
            return (torch.randn(shape, device="cuda", generator=gen) * 0.5).to(dtype)
        return torch.zeros(shape, dtype=dtype, device="cuda")

    F.z = z
    F.e = z                                      # real storage: the kernels read every row
    F.ROWS = (1, 3, 16)
    real_scratch = F.attn_mod.AttnScratch
    F.attn_mod.AttnScratch = lambda *a, **kw: real_scratch(*a[:4], "cuda", **kw)
    keep = cases.split(",")
    real_case = F.case

    def case(store, name, fn):
        if any(name.startswith(c) for c in keep) and "c1048576" not in name and "c262151" not in name:
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
    print(json.dumps({k: v for k, v in STATS.items() if k != "fns"}))
    for fn, (eq, df, nv) in sorted(STATS["fns"].items()):
        print(f"  {fn:16s} equal {eq:4d} differ {df:3d} no variant {nv:3d}")
    if BENCH["rows"]:
        print("us a launch (best of 5 x 20, prod GPU noise), the AOT hsaco vs the JIT's own buffer-op binary:")
        for fn, h, g, ints, ta, tj, sp in sorted(BENCH["rows"], key=lambda r: -r[4]):
            print(f"  {fn:16s} {h} grid {str(g):16s} {ints:28s} aot {ta:8.1f}  jit {tj:8.1f}  spills {sp}")
    return 1 if STATS["differ"] else 0


def kv8_cases(F, z) -> None:
    """--kv-dtype fp8 (kv8.py): three rows written by _attn_prep8 into uint8 caches, then attended by _chunks8."""

    import torch

    from tensorfold.families.qwen4_exp.cuda import kv8

    D, HD, NI, IHD, HALF, EPS, r = F.D, F.HD, F.NI, F.IHD, F.HALF, F.EPS, 3
    for world, rk in F.RANKS.items():
        heads, kv = rk["heads"], rk["kv"]
        pw = heads * 2 * HD + 2 * kv * HD + (NI + 1) * IHD
        for capacity in (1024, 262144):
            kc = torch.zeros((capacity, kv, HD + kv8.PAD), dtype=torch.uint8, device="cuda")
            vc = torch.zeros((capacity, kv, HD), dtype=torch.uint8, device="cuda")
            ikc, pos0 = z((capacity, IHD)), torch.full((1,), 5, dtype=torch.int32, device="cuda")
            q, iq = z((r, heads, HD)), z((r, NI, IHD))
            kv8.attn_prep(z((r, pw)), pos0, z((HD,), torch.float32), z((HD,), torch.float32), z((IHD,), torch.float32),
                          z((HALF,), torch.float32), q, kc, vc, iq, ikc, EPS, q_heads=heads, kv_heads=kv, head_dim=HD,
                          index_heads=NI, index_dim=IHD)
            sc = F.attn_mod.AttnScratch(r, heads, HD, capacity, "cuda")
            if sc.qsa:
                F.attn_mod.qsa_select(iq, ikc, z((-(-capacity // 4), IHD)), pos0, z((IHD,), torch.float32),
                                      z((HALF,), torch.float32), EPS, sc, r, context=8192)
            kv8.attention(q, kc, vc, pos0, sc, r, HD ** -0.5, context=8192 if sc.qsa else None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    h = sub.add_parser("hash")
    h.add_argument("--aot", required=True)
    h.add_argument("--spec", action="append", required=True)
    r = sub.add_parser("run")
    r.add_argument("--aot", required=True)
    r.add_argument("--bench", type=int, default=0, help="1: time each variant's first launch (AOT vs JIT binary)")
    r.add_argument("--cases", default="embed/r1/,hc_,rmsnorm/r3,add_streams,ple_,moe_partial,moe/,attn_prep/r3,"
                   "attn_gate/r3,fp4/,b16/hc_down/r3,b16/fc/,b16/o_proj/r16,b16/gdn_out,attention/tp1/c1024/r3/p100,"
                   "attention/tp1/c262144/r3/p3000/x8192,attention_prefill/tp1/c1024/s0/n30,"
                   "attention_prefill/tp1/c262144/s4096/n19,attention/tp1/c262144/r3/p140000/x16384,shift/,kv8")
    a = ap.parse_args()
    if a.cmd == "hash":
        return hash_check(Path(a.aot), [Path(s) for s in a.spec])
    return run_check(Path(a.aot), a.cases, a.bench)


if __name__ == "__main__":
    sys.exit(main())
