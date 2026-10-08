"""The HIP draft head (cuda_weights.zig draftInt4: the int4 lm_head's draft-vocabulary columns as their own matrix)
against the full head on gfx1151: python tools/rocm/drafthead_check.py [--model DIR] [--rounds N]

Its bf16 logits must be the full head's at the draft ids bit for bit (same kernel, same K order, an output reads only
its row and column), so the argmax over the draft vocabulary is too; then one call of each timed (best of N, A/B
alternated; noisy while prod shares the GPU).
"""
import argparse, os, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import int4_check as ic

ROOT = ic.ROOT
VOCAB_TXT = f"{ROOT}/zig/src/families/flashnext/cuda_draft_vocab.txt"


def draft_ids(vocab):
    """cuda_weights.zig draftIds at one rank: np.unique of the ids below the vocabulary."""
    v = np.array(open(VOCAB_TXT).read().replace(",", " ").split(), dtype=np.int64)
    return np.unique(v[(v >= 0) & (v < vocab)])


def head(x, w, s, k, n, rows, out=None):
    """The engine's call (cuda_int4.zig dense): bf16 out, decode NT (pickNt: two n16 tiles a wave when 128 | n)."""
    if out is None:
        out = torch.zeros(rows, n, dtype=torch.bfloat16, device="cuda")
    ic.launch(128, 2 if n % 128 == 0 else 1, 1, 1, 3, x, k, 0, w, s, k, n, None, rows, out, (rows + 15) // 16)
    return out


def best(fn, reps):
    a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    a.record()
    for _ in range(reps):
        fn()
    b.record()
    b.synchronize()
    return a.elapsed_time(b) / reps


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/dzannotti/models/qwen38fn-int4-autoround")
    ap.add_argument("--rounds", type=int, default=7)
    a = ap.parse_args()
    ic.M = ic.load()
    qw, sc = ic.st_tensor(a.model, "lm_head.qweight"), ic.st_tensor(a.model, "lm_head.scales")
    k, vocab = qw.shape[0] * 8, qw.shape[1]
    ids = draft_ids(vocab)
    n = len(ids)
    npad = (n + 127) // 128 * 128
    cols = np.concatenate([ids, np.full(npad - n, ids[-1])])
    wf, sf = (ic.dev(t) for t in ic.pack(qw, sc, 0, vocab, 0, k, 128))
    wd, sd = (ic.dev(t) for t in ic.pack(np.ascontiguousarray(qw[:, cols]), np.ascontiguousarray(sc[:, cols]), 0, npad, 0, k, 128))
    print(f"draft head: {n} ids, {npad} columns, {(wd.numel() * 4 + sd.numel() * 2) / 1e6:.1f} MB "
          f"(full head {(wf.numel() * 4 + sf.numel() * 2) / 1e6:.1f} MB)")
    rng = np.random.default_rng(7)
    tid = torch.from_numpy(ids).cuda()
    for rows in (1, 2, 8, 16):
        x = ic.bf16_rows(rng, rows, k)
        x[0, :16] *= 64  # a few large inputs
        x = x.cuda()
        full = head(x, wf, sf, k, vocab, rows)
        drf = head(x, wd, sd, k, npad, rows)
        same = torch.equal(drf[:, :n].view(torch.int16), full[:, tid].view(torch.int16))
        am = torch.equal(drf[:, :n].float().argmax(1), full[:, tid].float().argmax(1))
        ic.expect(same, f"{rows} rows: draft logits differ from the full head's at the draft ids")
        ic.expect(am, f"{rows} rows: argmax over the draft vocabulary differs")
        print(f"  {rows:2d} rows: logits at the draft ids bit-equal {same}, draft argmax equal {am}")
    x = ic.bf16_rows(rng, 1, k).cuda()
    of = torch.empty(1, vocab, dtype=torch.bfloat16, device="cuda")
    od = torch.empty(1, npad, dtype=torch.bfloat16, device="cuda")
    tf, td = [], []
    for _ in range(a.rounds):
        tf.append(best(lambda: head(x, wf, sf, k, vocab, 1, of), 20))
        td.append(best(lambda: head(x, wd, sd, k, npad, 1, od), 20))
    print(f"one row, best of {a.rounds} x 20 (A/B alternated, prod on the GPU): full head {min(tf) * 1e3:.0f} us, "
          f"draft head {min(td) * 1e3:.0f} us ({min(tf) / min(td):.2f}x)")
    print("drafthead-check (HIP):", "FAILED " + "; ".join(ic.FAIL) if ic.FAIL else "ok")
    return 1 if ic.FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
