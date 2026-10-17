# decode-after-prompt: KL on the decoded rows (>= P) of int4-prompt vs bf16-prompt engine logits, and each vs ref32
import os, sys, numpy as np
P = int(sys.argv[1]); pairs = sys.argv[2:]  # seq:int4.npy:bf16.npy
def ls(x):
    x = x.astype(np.float32); m = x.max(1, keepdims=True); return x - (m + np.log(np.exp(x - m).sum(1, keepdims=True)))
def kl(a, b):  # KL(a || b), rows
    la, lb = ls(a), ls(b); return (np.exp(la) * (la - lb)).sum(1)
tot = {"i_b": [], "b_r": [], "i_r": [], "top": []}
for pr in pairs:
    s, fi, fb = pr.split(":")
    ref = np.load(fos.path.expanduser(f"~/tfq/ref32/ref_{s}.npy"), mmap_mode="r")
    I = np.load(fi, mmap_mode="r"); B = np.load(fb, mmap_mode="r")
    V = ref.shape[1]
    r = {"i_b": [], "b_r": [], "i_r": [], "top": []}
    for r0 in range(P, I.shape[0], 128):
        i, b, rf = I[r0:r0+128, :V], B[r0:r0+128, :V], ref[r0:r0+128]
        r["i_b"] += list(kl(b, i)); r["b_r"] += list(kl(rf, b)); r["i_r"] += list(kl(rf, i)); r["top"] += list(i.argmax(1) == b.argmax(1))
    # the prompt rows too (vs bf16 prompt) as a sanity anchor
    pk = np.mean(np.concatenate([kl(B[r0:r0+128, :V], I[r0:r0+128, :V]) for r0 in range(0, P, 512)]))
    print(f"seq {s}: decoded rows {len(r['top'])}: KL(bf16-prompt||int4-prompt) {np.mean(r['i_b']):.5f} top1 agree {100*np.mean(r['top']):.2f}% | vs ref32: bf16-prompt {np.mean(r['b_r']):.5f} int4-prompt {np.mean(r['i_r']):.5f} | prompt rows KL {pk:.5f}")
    for k in tot: tot[k] += r[k]
print(f"ALL: KL(bf16||int4) {np.mean(tot['i_b']):.5f} top1 agree {100*np.mean(tot['top']):.2f}% | vs ref32 bf16-prompt {np.mean(tot['b_r']):.5f} int4-prompt {np.mean(tot['i_r']):.5f}")
