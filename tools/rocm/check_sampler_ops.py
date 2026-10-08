"""ROCm: the argmax and top-k checkers (tools/zig/check_argmax_ops.py, check_argmax_f64_ops.py, check_topk_ops.py)
without check_torch_ops.py's other operators. Run in the rocm-dev container:

    python tools/rocm/check_sampler_ops.py --source zig/kernels/cuda/torch_ops --out <dir> --config <config.json>
"""

import argparse
import json
from pathlib import Path
import sys
import time

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'zig'))

from ops_build import compile_operators, require_gpu
from ops_compare import RawCells
from check_argmax_ops import check_argmax_ops
from check_argmax_f64_ops import check_argmax_f64_ops
from check_topk_ops import check_topk_ops


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--source', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--config', type=Path, required=True)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    config = json.loads(a.config.read_text())
    config = config.get('text_config', config)
    receipt = {'oracle': {'torch': torch.__version__, 'hip': torch.version.hip}, 'cells': []}
    lib, compiled = compile_operators(a.source, a.out, ('argmax.cu', 'argmax_f64.cu', 'topk.cu'))
    receipt.update(compiled)
    torch.cuda.set_device(0)
    require_gpu()
    check = RawCells(a.out, receipt)
    check_argmax_ops(lib, config, check)
    check_argmax_f64_ops(lib, config, check)
    check_topk_ops(lib, config, check)
    check.finish(started)


if __name__ == '__main__':
    main()
