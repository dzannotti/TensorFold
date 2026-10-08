"""Load a --genco code object into torch's HIP runtime and launch its kernels, as the Zig engine does (module load,
function by mangled name, explicit grid, block, dynamic shared memory and argument list) on torch's current stream."""

import ctypes
from pathlib import Path
import subprocess

import torch

ROOT = Path(__file__).resolve().parents[2]
KERNELS = ROOT / 'zig/kernels/cuda'
FLAGS = ('-x', 'hip', '-O3', '-std=c++20', '-ffp-contract=off', '-fno-gpu-flush-denormals-to-zero',
         '--offload-arch=gfx1151', '-include', 'hip_compat.cuh', '-I', str(KERNELS / 'hip'), '-I', str(KERNELS))


def genco(source, out_dir):
    """hipcc --genco of one zig/kernels/cuda source (tools/rocm/hipcc_kernels.sh's flags) -> the code object."""
    out = Path(out_dir) / (Path(source).stem + '.hsaco')
    out.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(['hipcc', *FLAGS, '--genco', str(KERNELS / source), '-o', str(out)], check=True, timeout=900)
    return out


def symbols(code_object):
    """Kernel symbols of a code object (llvm-nm), for finding the HIP mangled names."""
    elf = Path(code_object).with_suffix('.elf')
    subprocess.run(['/opt/rocm/lib/llvm/bin/clang-offload-bundler', '--type=o', '--unbundle', f'--input={code_object}',
                    f'--output={elf}', '--targets=hipv4-amdgcn-amd-amdhsa--gfx1151'], check=True)
    nm = subprocess.run(['/opt/rocm/lib/llvm/bin/llvm-nm', '--defined-only', str(elf)], capture_output=True,
                        text=True, check=True).stdout
    return [line.split()[-1] for line in nm.splitlines() if ' T ' in line]


class Module:
    def __init__(self, code_object):
        torch.zeros(1, device='cuda')                       # torch's runtime and context first
        self.hip = ctypes.CDLL('libamdhip64.so.7')           # the copy torch loaded (same soname)
        self.handle = ctypes.c_void_p()
        self.check(self.hip.hipModuleLoad(ctypes.byref(self.handle), str(code_object).encode()), 'hipModuleLoad')

    def check(self, rc, what):
        if rc:
            raise RuntimeError(f'{what}: hipError {rc}')

    def function(self, name):
        f = ctypes.c_void_p()
        self.check(self.hip.hipModuleGetFunction(ctypes.byref(f), self.handle, name.encode()), name)
        return f

    def launch(self, f, grid, block, args, shared=0):
        """`args`: ctypes values in the kernel's parameter order (c_void_p for pointers, structures by value)."""
        grid = tuple(grid) + (1,) * (3 - len(grid))
        block = tuple(block) + (1,) * (3 - len(block))
        params = (ctypes.c_void_p * len(args))(*[ctypes.cast(ctypes.pointer(a), ctypes.c_void_p) for a in args])
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.check(self.hip.hipModuleLaunchKernel(f, *map(ctypes.c_uint, grid), *map(ctypes.c_uint, block),
                                                  ctypes.c_uint(shared), stream, params, None), 'launch')


def ptr(t):
    return ctypes.c_void_p(t.data_ptr() if t is not None else 0)
