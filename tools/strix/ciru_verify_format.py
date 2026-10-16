"""ciru_q8 format check (headers + memmapped bytes only; no model code runs).

W'[n,k] = s[n,k//256] * q[n,k];  W = W' H (H = blockdiag 128-Sylvester / sqrt(128), symmetric, H H = I)
codes [E,N/16,K/32,2,16,16]: q[nt,kt,h,r,c] = q[16nt+r, 32kt+16h+c];  scales [E,N/16,K/256,16]: s[nt,g,r] = s[16nt+r, g]
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ciru_st import Dir, fwht128

C = Dir('/home/dzannotti/models/minicpm/MiniCPM-4.7-Ultra-CIRU-HALO-AGENT/weights')
Q = Dir('/home/dzannotti/models/q36/Qwen3.6-35B-A3B-MLX-4bit-MTP')
pre = 'model.language_model.layers.{}.mlp.'


def deq(L, E, proj):
    c = np.array(C.get(pre.format(L) + f'experts.ultra_{proj}_codes')[E])
    s = np.array(C.get(pre.format(L) + f'experts.ultra_{proj}_scales')[E]).astype(np.float32)
    NT, KT = c.shape[:2]
    N, K = NT * 16, KT * 32
    q = c.transpose(0, 3, 1, 2, 4).reshape(N, K)
    sf = s.transpose(0, 2, 1).reshape(N, K // 256)
    return q, sf, q.astype(np.float32) * np.repeat(sf, 256, 1)


def mlx(L, E, proj):
    p = f'language_model.model.layers.{L}.mlp.switch_mlp.{proj}_proj'
    w = np.array(Q.get(p + '.weight')[E]); s = np.array(Q.get(p + '.scales')[E]); b = np.array(Q.get(p + '.biases')[E])
    q = np.stack([(w >> (4 * i)) & 15 for i in range(8)], -1).reshape(w.shape[0], -1).astype(np.float32)
    return (q.reshape(q.shape[0], -1, 64) * s[..., None] + b[..., None]).reshape(q.shape[0], -1)


def corr(a, b):
    a = a.ravel() - a.mean(); b = b.ravel() - b.mean()
    return float(a @ b / np.sqrt((a @ a) * (b @ b)))


rng = np.random.default_rng(0)
for L, E, proj in [(0, 0, 'gate'), (21, 100, 'up'), (39, 255, 'down')]:
    q, sf, Wr = deq(L, E, proj)
    amax = np.abs(q.reshape(q.shape[0], -1, 256).astype(np.int32)).max(-1)
    live = sf > 2 ** -24
    assert (amax[live] == 127).all() or (amax[live] >= 126).mean() > 0.999, 'a live group without a full-scale code'
    assert (amax[~live] == 0).all(), 'a floor-scale group with codes'
    R = mlx(L, E, proj)  # Qwen3.6-35B-A3B 4-bit: the parent of MiniCPM's fine-tuned text tower
    x = rng.standard_normal((4, q.shape[1])).astype(np.float32)
    y_rot = (fwht128(x) @ Wr.T)      # W' (H x): what the kernels compute
    y_unrot = x @ fwht128(Wr).T      # (W' H) x: the de-rotated weights
    y_ref = x @ R.T
    print(f"L{L} E{E} {proj}: groups live {live.mean():.3f}, corr(W'H, Qwen3.6 4-bit)={corr(fwht128(Wr), R):.4f}, "
          f"corr(W', Qwen3.6)={corr(Wr, R):.4f}; matvec: max|W'H x - W'(H x)|/max|y|={np.abs(y_rot - y_unrot).max() / np.abs(y_rot).max():.2e}, "
          f"corr vs Qwen3.6={corr(y_rot, y_ref):.4f}")
    assert corr(fwht128(Wr), R) > 0.95 and abs(corr(Wr, R)) < 0.05

# the shared expert as the engine stores it: rotated, 8-bit asymmetric group 128 with fp16 tables
for L in (0, 20, 39):
    errs = []
    for name in ('gate_proj', 'up_proj', 'down_proj'):
        W = np.array(C.get(pre.format(L) + f'shared_expert.{name}.weight'))
        Wh = fwht128(W)
        g = Wh.reshape(W.shape[0], -1, 128)
        lo = g.min(-1, keepdims=True).astype(np.float16).astype(np.float32)
        sc = ((g.max(-1, keepdims=True) - lo) / 255).astype(np.float16).astype(np.float32)
        sc[sc == 0] = 1
        qq = np.clip(np.round((g - lo) / sc), 0, 255)
        back = fwht128((qq * sc + lo).reshape(W.shape))
        x = rng.standard_normal((16, W.shape[1])).astype(np.float32)
        errs.append(np.linalg.norm(x @ back.T - x @ W.T) / np.linalg.norm(x @ W.T))
    print(f'L{L} shared expert 8-bit g128 (rotated) matvec rel err gate/up/down: ' + ' '.join(f'{e:.4f}' for e in errs))
print('ok')
