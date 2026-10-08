"""fn_pack.cu's n-gram gathers on gfx1151 (HIP code object, cuda_weights.zig Gather's launch: 256-thread blocks over
tokens * heads * width) against torch indexing: raw bytes, out-of-rank ids 0x7FC0. Run in the rocm-dev container:

    python tools/rocm/check_pack.py --out <dir>
"""

import argparse
import ctypes
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hipmod import Module, genco, ptr  # noqa: E402

I, L = ctypes.c_int, ctypes.c_longlong


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    m = Module(genco('fn_pack.cu', a.out))
    fp8, bf16 = m.function('fn_ngram_gather'), m.function('fn_ngram_gather_bf16')
    torch.manual_seed(3)
    bad = cells = 0
    for heads_all, head0, heads, width, rows, base, tokens in ((8, 0, 8, 160, 5000, 0, 7), (8, 4, 4, 160, 2500, 2500, 33)):
        lut = torch.randint(0, 65536, (256,), dtype=torch.int32).to(torch.int16).cuda()
        for table in ('fp8', 'bf16'):
            store = (torch.randint(0, 256, (rows, width), dtype=torch.uint8) if table == 'fp8' else
                     torch.randint(0, 65536, (rows, width), dtype=torch.int32).to(torch.int16)).cuda()
            ids = torch.randint(0, 2 * rows + base, (tokens, heads_all), dtype=torch.int64).cuda()
            out = torch.empty((tokens, heads * width), dtype=torch.int16, device='cuda')
            total = tokens * heads * width
            args = [I(heads_all), I(head0), I(heads), L(base), L(rows), I(width), I(tokens), ptr(out)]
            if table == 'fp8':
                m.launch(fp8, ((total + 255) // 256,), (256,), [ptr(store), ptr(lut), ptr(ids)] + args)
            else:
                m.launch(bf16, ((total + 255) // 256,), (256,), [ptr(store), ptr(ids)] + args)
            local = ids[:, head0:head0 + heads] - base
            ok = (local >= 0) & (local < rows)
            rows_of = store[local.clamp(0, rows - 1)]                                     # tokens, heads, width
            want = (lut[rows_of.long()] if table == 'fp8' else rows_of)
            want = torch.where(ok[..., None], want, torch.tensor(0x7FC0, dtype=torch.int16, device='cuda'))
            torch.cuda.synchronize()
            same = torch.equal(out, want.reshape(tokens, -1))
            cells += 1
            bad += not same
            print(f"{'EQUAL' if same else 'DIFFER'} ngram/{table}/heads{head0}+{heads}/t{tokens}", flush=True)
    print(f"{'PASS' if bad == 0 else 'FAIL'} pack: {cells - bad} of {cells}", flush=True)
    return 1 if bad else 0


if __name__ == '__main__':
    raise SystemExit(main())
