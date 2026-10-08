"""Build a zig/kernels/hip source for gfx1151 (hipcc --genco) and launch its kernels on torch tensors (ctypes)."""
import ctypes, os, struct, subprocess, torch

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FLAGS = ["-O3", "-std=c++20", "-ffp-contract=off", "-fno-gpu-flush-denormals-to-zero"]
torch.zeros(1, device="cuda")
_hip = ctypes.CDLL("libamdhip64.so.7")


def _ok(r, what):
    if r != 0:
        raise RuntimeError(f"{what}: hip error {r}")


def build(name, defs=()):
    """zig/kernels/hip/<name>.hip (-D defs) -> ~/.cache/tf-mtp/<name>[-defs].hsaco (rebuilt when a hip source is newer)."""
    d = os.path.join(ROOT, "zig/kernels/hip")
    tag = "".join("-" + x for x in defs).replace("=", "")
    src, out = os.path.join(d, name + ".hip"), os.path.expanduser(f"~/.cache/tf-mtp/{name}{tag}.hsaco")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    newest = max(os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d))
    if not os.path.exists(out) or os.path.getmtime(out) < newest:
        subprocess.run(["hipcc", "--genco", "--offload-arch=gfx1151", *FLAGS, *("-D" + x for x in defs), "-o", out, src], check=True)
    return Module(out)


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
        """args: tensors (device pointers), None (null), ('i', v) int32, ('f', v) float32, ('q', v) uint64."""
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
        cfg = (ctypes.c_void_p * 5)(1, ctypes.cast(raw, ctypes.c_void_p), 2,
                                    ctypes.cast(ctypes.pointer(size), ctypes.c_void_p), 3)
        g = grid if isinstance(grid, tuple) else (grid, 1, 1)
        s = torch.cuda.current_stream().cuda_stream
        _ok(_hip.hipModuleLaunchKernel(self.fn(name), *g, block, 1, 1, smem, ctypes.c_void_p(s), None, cfg),
            "launch " + name)


def best_us(f, reps=20, rounds=5):
    """Best-of-`rounds` mean microseconds of `reps` calls (prod shares the GPU: the minimum is the signal)."""
    f()
    torch.cuda.synchronize()
    best = float("inf")
    for _ in range(rounds):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record()
        for _ in range(reps):
            f()
        b.record()
        b.synchronize()
        best = min(best, a.elapsed_time(b) * 1000 / reps)
    return best
