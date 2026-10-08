"""Compile the operator oracle with the serving container's pinned compiler and arithmetic flags."""

import ctypes
import hashlib
from pathlib import Path
import subprocess

FLAGS = ('-O3', '--fmad=false', '--ftz=false', '--expt-relaxed-constexpr', '-std=c++20',
         '-D__CUDA_NO_HALF_OPERATORS__', '-D__CUDA_NO_HALF_CONVERSIONS__',
         '-D__CUDA_NO_BFLOAT16_CONVERSIONS__', '-D__CUDA_NO_HALF2_OPERATORS__',
         '-gencode=arch=compute_121,code=sm_121', '-shared', '-Xcompiler', '-fPIC')


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# ROCm (gfx1151): the same arithmetic contract under hipcc, the CUDA headers mapped by zig/kernels/cuda/hip_compat.cuh
HIP_ROOT = Path(__file__).resolve().parents[2] / 'zig/kernels/cuda'
HIP_FLAGS = ('-x', 'hip', '-O3', '-std=c++20', '-ffp-contract=off', '-fno-gpu-flush-denormals-to-zero',
             '--offload-arch=gfx1151', '-include' + str(HIP_ROOT / 'hip_compat.cuh'), '-I', str(HIP_ROOT / 'hip'),
             '-I', str(HIP_ROOT),
             '-shared', '-fPIC')


def compile_hip(source, out, names):
    source, out = Path(source), Path(out)
    version = subprocess.run(['hipcc', '--version'], capture_output=True, text=True, check=True).stdout
    files = [source / name for name in names]
    target = out / 'operators.so'
    with (out / 'compile.log').open('w') as stream:
        subprocess.run(['hipcc', *HIP_FLAGS, '-I', str(source), *map(str, files), '-o', str(target)], stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=600)
    return ctypes.CDLL(str(target)), {'hipcc': version, 'flags': list(HIP_FLAGS), 'image_sha256': digest(target),
                                    'sources': {p.name: digest(p) for p in files},
                                    'headers': {p.name: digest(p) for p in sorted(source.glob('*.h'))}}


def require_gpu():
    """The packets are qualified on sm_121 (CUDA) and gfx1151 (ROCm) only."""
    import torch

    if torch.version.hip:
        if torch.cuda.get_device_properties(0).gcnArchName.split(':')[0] != 'gfx1151':
            raise RuntimeError('The ROCm packet is qualified for gfx1151 only')
    elif torch.cuda.get_device_capability() != (12, 1):
        raise RuntimeError('This packet is qualified for sm_121 only')


def compile_operators(source, out, names):
    import torch

    if torch.version.hip:
        return compile_hip(source, out, names)
    source, out = Path(source), Path(out)
    version = subprocess.run(['nvcc', '--version'], capture_output=True, text=True, check=True).stdout
    if 'V13.3.73' not in version:
        raise RuntimeError('Operator parity requires the serving container nvcc13.3.73')
    files = [source / name for name in names]
    target = out / 'operators.so'
    with (out / 'compile.log').open('w') as stream:
        subprocess.run(['nvcc', *FLAGS, *map(str, files), '-o', str(target)], stdout=stream,
                       stderr=subprocess.STDOUT, check=True, timeout=180)
    return ctypes.CDLL(str(target)), {'nvcc': version, 'flags': list(FLAGS), 'image_sha256': digest(target),
                                    'sources': {p.name: digest(p) for p in files},
                                    'headers': {p.name: digest(p) for p in sorted(source.glob('*.h'))}}
