"""Load a HIP code object (hipcc --genco) next to torch and launch its kernels on torch tensors (ctypes)."""
import ctypes, os, struct, subprocess, torch

torch.zeros(1, device="cuda")
_hip = ctypes.CDLL("libamdhip64.so.7")


def _ok(r, what):
    if r != 0:
        raise RuntimeError(f"{what}: hip error {r}")


def cache(name):
    d = os.path.expanduser("~/.cache/tf-int4")
    os.makedirs(d, exist_ok=True)
    return f"{d}/{name}"


def build(src, out, extra=()):
    """hipcc --genco for gfx1151 when `out` is older than `src`."""
    if not os.path.exists(out) or os.path.getmtime(out) < os.path.getmtime(src):
        subprocess.run(["hipcc", "--genco", "--offload-arch=gfx1151", "-O3", "-std=c++20", *extra, "-o", out, src],
                       check=True)
    return out


class Module:
    def __init__(self, path):
        self.m = ctypes.c_void_p()
        _ok(_hip.hipModuleLoad(ctypes.byref(self.m), path.encode()), "hipModuleLoad " + path)
        self.fns = {}

    def fn(self, name):
        if name not in self.fns:
            f = ctypes.c_void_p()
            _ok(_hip.hipModuleGetFunction(ctypes.byref(f), self.m, name.encode()), "hipModuleGetFunction " + name)
            self.fns[name] = f
        return self.fns[name]

    def occupancy(self, name, threads, smem=0):
        n = ctypes.c_int()
        _ok(_hip.hipModuleOccupancyMaxActiveBlocksPerMultiprocessor(ctypes.byref(n), self.fn(name), threads, smem),
            "occupancy")
        return n.value

    def launch(self, name, grid, block, args, smem=0):
        """args: tensors (device pointers), ('i', v) int32, ('f', v) float32, ('q', v) uint64."""
        buf = b""
        for a in args:
            if isinstance(a, torch.Tensor):
                fmt, v = "Q", a.data_ptr()
            elif a is None:
                fmt, v = "Q", 0
            else:
                fmt, v = {"i": "i", "f": "f", "q": "Q"}[a[0]], a[1]
            size = struct.calcsize(fmt)
            buf += b"\0" * (-len(buf) % size) + struct.pack(fmt, v)
        buf += b"\0" * (-len(buf) % 8)
        raw = ctypes.create_string_buffer(buf, len(buf))
        size = ctypes.c_size_t(len(buf))
        cfg = (ctypes.c_void_p * 5)(1, ctypes.cast(raw, ctypes.c_void_p), 2, ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), 3)
        s = torch.cuda.current_stream().cuda_stream
        _ok(_hip.hipModuleLaunchKernel(self.fn(name), grid, 1, 1, block, 1, 1, smem, ctypes.c_void_p(s), None, cfg),
            "launch " + name)


def best_us(f, reps=20, rounds=5, graph=True):
    """Best-of-`rounds` mean microseconds of `reps` calls, replayed from one HIP graph (no host launch cost, as the
    engine's graphs run them); prod shares the GPU, so the minimum is the signal."""
    f()
    torch.cuda.synchronize()
    g = None
    if graph:
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(reps):
                f()
    run = g.replay if g else (lambda: [f() for _ in range(reps)])
    run()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        run()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / reps)
    return best
