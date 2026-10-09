#!/usr/bin/env python3
"""``_chunks8`` time over caches mapped as the Zig engine maps them (cuda_vmm.zig: hipMemAddressReserve, physical
chunks hipMemCreate'd per growth step, 4 KiB granularity) against other mappings and plain hipMalloc (torch), the
same bytes in each: is the sparse key gather TLB-bound by where the pages land?

  layouts: torch (hipMalloc), vmm-<chunk rows>-<offset KiB>: chunks of that many rows (0: one chunk) mapped from an
  address <offset> KiB past a 2 MiB boundary (0: aligned), chunk sizes rounded up to <gran> (4 KiB or 2 MiB).

    PYTHONPATH=src python tools/rocm/kv8_vmm.py [--reps 5]
"""

from __future__ import annotations

import argparse
import ctypes
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(__file__))
import kv8_attn as K  # noqa: E402
from tensorfold.families.qwen4_exp.cuda import kv8  # noqa: E402

MIB2 = 2 << 20


class Prop(ctypes.Structure):          # hipMemAllocationProp (cuda.h's CUmemAllocationProp layout)
    _fields_ = [("type", ctypes.c_int), ("handle_types", ctypes.c_int), ("loc_type", ctypes.c_int),
                ("loc_id", ctypes.c_int), ("win32", ctypes.c_void_p), ("flags", ctypes.c_uint8 * 8)]


class Access(ctypes.Structure):
    _fields_ = [("loc_type", ctypes.c_int), ("loc_id", ctypes.c_int), ("flags", ctypes.c_int)]


def hip():
    """The HIP runtime torch loaded (its bundled one, not /opt/rocm's)."""
    torch.cuda.init()
    torch.zeros(1, device="cuda")
    libs = {line.split()[-1] for line in open("/proc/self/maps") if "libamdhip64" in line}
    return ctypes.CDLL(libs.pop())


H = None


def ok(rc, what):
    if rc != 0:
        raise RuntimeError(f"{what}: hip error {rc}")


def granularity(flag: int) -> int:
    g = ctypes.c_size_t(0)
    p = Prop(1, 0, 1, torch.cuda.current_device())
    ok(H.hipMemGetAllocationGranularity(ctypes.byref(g), ctypes.byref(p), flag), "granularity")
    return g.value


class Mapped:
    """`total` bytes of address space, physical chunks of `chunk` bytes (rounded to `gran`) mapped from `offset`
    bytes past a 2 MiB-aligned base. data_ptr() is what a kernel takes."""

    def __init__(self, total: int, chunk: int, offset: int, gran: int):
        up = lambda x: -(-x // gran) * gran                       # noqa: E731
        self.size = up(total) + MIB2 + up(offset)
        self.base = ctypes.c_void_p(0)
        ok(H.hipMemAddressReserve(ctypes.byref(self.base), ctypes.c_size_t(self.size), ctypes.c_size_t(MIB2),
                                  None, ctypes.c_ulonglong(0)), "reserve")
        self.start = self.base.value + up(offset)
        self.handles, at, p = [], 0, Prop(1, 0, 1, torch.cuda.current_device())
        while at < total:
            n = up(min(chunk or total, total - at)) if chunk else up(total)
            h = ctypes.c_ulonglong(0)
            ok(H.hipMemCreate(ctypes.byref(h), ctypes.c_size_t(n), ctypes.byref(p), ctypes.c_ulonglong(0)), "create")
            ok(H.hipMemMap(ctypes.c_void_p(self.start + at), ctypes.c_size_t(n), ctypes.c_size_t(0), h,
                           ctypes.c_ulonglong(0)), "map")
            a = Access(1, torch.cuda.current_device(), 3)
            ok(H.hipMemSetAccess(ctypes.c_void_p(self.start + at), ctypes.c_size_t(n), ctypes.byref(a),
                                 ctypes.c_size_t(1)), "access")
            self.handles.append((at, n, h))
            at += n

    def data_ptr(self):
        return self.start

    def fill(self, src: torch.Tensor):
        torch.cuda.synchronize()
        ok(H.hipMemcpy(ctypes.c_void_p(self.start), ctypes.c_void_p(src.data_ptr()),
                       ctypes.c_size_t(src.numel() * src.element_size()), 3), "copy")

    def free(self):
        for at, n, h in self.handles:
            H.hipMemUnmap(ctypes.c_void_p(self.start + at), ctypes.c_size_t(n))
            H.hipMemRelease(h)
        H.hipMemAddressFree(self.base, ctypes.c_size_t(self.size))


class Arg:
    """A pointer Triton launches with: data_ptr and dtype (as a torch tensor of that dtype would give)."""

    def __init__(self, ptr: int, dtype):
        self.ptr, self.dtype = ptr, dtype

    def data_ptr(self):
        return self.ptr

    def ptr_range(self):                 # past 2 GiB: the plain (non buffer-op) build, as Zig launches on VMM caches
        return 1 << 40


def launch(q, kq, vq, sc_ptr, p0, sc, rows):
    pos0 = torch.full((1,), p0, dtype=torch.int32, device="cuda")
    _, h, d = q.shape
    keys = min(p0 + rows, (sc.budget // sc.ratio + 1) * sc.ratio - 1) if sc.qsa else p0 + rows
    chunks = min(sc.nch, -(-keys // 512))
    kv8._chunks8[(rows, K.HK, chunks)](q, kq, vq, sc_ptr, pos0, sc.po, sc.pm, sc.pl, sc.ids, sc.nk, sc.sparse,
                                        H=h, HK=K.HK, D=d, G=h // K.HK, CH=512, NCH=sc.nch, SCALE=d ** -0.5,
                                        IDW=sc.idw, QSA=sc.qsa, num_warps=4, num_stages=1)


def timed(fn, reps: int, n: int) -> float:
    best = float("inf")
    for _ in range(reps):
        e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        e0.record()
        for _ in range(n):
            fn()
        e1.record()
        torch.cuda.synchronize()
        best = min(best, e0.elapsed_time(e1) * 1000 / n)
    return best


def main() -> int:
    global H
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    H = hip()
    print(f"granularity minimum {granularity(0) >> 10} KiB, recommended {granularity(1) >> 10} KiB")
    cap = 262144
    kc, vc = K.cache(cap, 2)
    krow, vrow = K.HK * (K.D + kv8.PAD), K.HK * K.D
    cases = (("prefill M256 sparse p6144", 256, 6144), ("decode M3 sparse p70000", 3, 70000),
             ("prefill M256 sparse p200000", 256, 200000))
    layouts = [("torch hipMalloc", None)] + [(f"vmm chunk {c or 'one'} rows, +{o >> 10} KiB, gran {g >> 10} KiB",
                                              (c, o, g)) for c, o, g in ((8192, 4096, 4096), (8192, 0, 4096),
                                                                         (0, 4096, 4096), (0, 0, 4096),
                                                                         (8192, 0, MIB2), (0, 0, MIB2))]
    for name, rows, p0 in cases:
        q = torch.randn((rows, K.H, K.D), device="cuda").to(torch.bfloat16)
        sc = K.scratch(rows, cap, p0, 3)
        ref = None
        for lname, lay in layouts:
            if lay is None:
                kq, vq, sp = (Arg(kc.data_ptr(), torch.float8_e4m3fn), Arg(vc.data_ptr(), torch.float8_e4m3fn),
                              Arg(kc.data_ptr(), torch.float32))
                maps = []
            else:
                c, o, g = lay
                mk, mv = Mapped(cap * krow, c * krow, o, g), Mapped(cap * vrow, c * vrow, o, g)
                mk.fill(kc), mv.fill(vc)
                maps = [mk, mv]
                kq, vq, sp = Arg(mk.start, torch.float8_e4m3fn), Arg(mv.start, torch.float8_e4m3fn), \
                    Arg(mk.start, torch.float32)
            fn = lambda: launch(q, kq, vq, sp, p0, sc, rows)          # noqa: E731
            fn()
            torch.cuda.synchronize()
            got = sc.po.clone()
            same = ref is None or torch.equal(got.view(torch.int32), ref.view(torch.int32))
            ref = got if ref is None else ref
            t = timed(fn, args.reps, max(3, min(30, 3000 // rows)))
            print(f"{name:28s} {lname:44s} {t:9.1f} us{'' if same else '  PARTIALS DIFFER'}", flush=True)
            torch.cuda.synchronize()
            for m in maps:
                m.free()
    return 0


if __name__ == "__main__":
    sys.exit(main())
