#!/usr/bin/env python3
# Usage: python3 -I tools/strix/dspark_ref.py DUMP DSPARK_DIR TARGET_WEIGHTS_DIR [--holds N]
"""The DSpark drafter's forward in fp64 on a TF_DRAFT_DUMP file's real inputs, against the engine's drafts.

Follows vLLM's qwen3_dflash.py / qwen3_dspark.py and v1/worker/gpu/spec_decode/dspark/speculator.py (and llama.cpp's
src/models/dflash.cpp): context K/V = k_proj/v_proj(hidden_norm(fc(taps))), K normed (k_norm) and rotated at its
position; the block [anchor, mask...] at pos0.. attends non-causally to the context and itself; logits = lm_head(norm(h));
draft i = argmax(logits_i + markov_w2 . markov_w1[prev]), prev the anchor then draft i-1 (full vocabulary: llama.cpp and
vLLM's default; restricted to each row's top-k head logits: vLLM's dspark_draft_topk path, what the engine runs).
The context is the rows the engine's ring keeps (the dump's begin..pos0, each row's last `window` positions)."""
import json, struct, sys
import numpy as np


def tensors(path):
    f = open(path, 'rb')
    n = struct.unpack('<Q', f.read(8))[0]
    head = json.loads(f.read(n))
    return f, 8 + n, head


def load(path, names):
    f, base, head = tensors(path)
    out = {}
    for name in names:
        if name not in head:
            continue
        t = head[name]
        assert t['dtype'] == 'BF16', name
        a, b = t['data_offsets']
        f.seek(base + a)
        raw = np.frombuffer(f.read(b - a), dtype=np.uint16)
        out[name] = (raw.astype(np.uint32) << 16).view(np.float32).astype(np.float64).reshape(t['shape'])
    return out


def bf(u16):
    return (u16.astype(np.uint32) << 16).view(np.float32).astype(np.float64)


def rms(x, w, eps):
    return x / np.sqrt((x * x).mean(-1, keepdims=True) + eps) * w


def rope(x, pos, theta):  # x (..., n, d), rotate-half (neox) over the whole head
    d = x.shape[-1]
    inv = theta ** (-np.arange(0, d // 2, dtype=np.float64) / (d // 2))
    ang = pos[:, None].astype(np.float64) * inv[None, :]
    c, s = np.cos(ang), np.sin(ang)
    c = np.concatenate([c, c], -1)[:, None, :]
    s = np.concatenate([s, s], -1)[:, None, :]
    x1, x2 = x[..., : d // 2], x[..., d // 2:]
    return x * c + np.concatenate([-x2, x1], -1) * s


def records(path):
    f = open(path, 'rb')
    while True:
        tag = f.read(4)
        if not tag:
            return
        tag = struct.unpack('<I', tag)[0]
        if tag == 1:
            n, width = struct.unpack('<II', f.read(8))
            pos = np.frombuffer(f.read(8 * n), dtype=np.int64)
            taps = np.frombuffer(f.read(2 * n * width), dtype=np.uint16).reshape(n, width)
            yield 'absorb', (pos, taps)
        elif tag == 2:
            B, k, nd = struct.unpack('<III', f.read(12))
            pos0, begin = struct.unpack('<qq', f.read(16))
            ids = np.frombuffer(f.read(4 * B), dtype=np.int32)
            top_ids = np.frombuffer(f.read(4 * B * k), dtype=np.int32).reshape(B, k)
            top_vals = bf(np.frombuffer(f.read(2 * B * k), dtype=np.uint16)).reshape(B, k)
            drafts = np.frombuffer(f.read(4 * nd), dtype=np.uint32)
            yield 'hold', (pos0, begin, ids, top_ids, top_vals, drafts)
        else:
            raise SystemExit(f'bad record {tag}')


def main():
    dump, ddir, tdir = sys.argv[1:4]
    holds = int(sys.argv[sys.argv.index('--holds') + 1]) if '--holds' in sys.argv else 8
    window = int(sys.argv[sys.argv.index('--window') + 1]) if '--window' in sys.argv else 2048
    c = json.load(open(f'{ddir}/config.json'))
    L, H, nh, nkv, hd = c['num_hidden_layers'], c['hidden_size'], c['num_attention_heads'], c['num_key_value_heads'], c['head_dim']
    eps, theta = c['rms_norm_eps'], c['rope_parameters']['rope_theta']
    names = ['fc.weight', 'hidden_norm.weight', 'norm.weight', 'markov_head.markov_w1.weight', 'markov_head.markov_w2.weight']
    for i in range(L):
        names += [f'layers.{i}.{n}.weight' for n in ('input_layernorm', 'post_attention_layernorm', 'self_attn.q_proj', 'self_attn.k_proj',
                                                    'self_attn.v_proj', 'self_attn.o_proj', 'self_attn.q_norm', 'self_attn.k_norm',
                                                    'mlp.gate_proj', 'mlp.up_proj', 'mlp.down_proj')]
    w = load(f'{ddir}/model.safetensors', names)
    idx = json.load(open(f'{tdir}/model.safetensors.index.json'))['weight_map']
    emb_name, head_name = 'model.language_model.embed_tokens.weight', 'lm_head.weight'
    w.update(load(f'{tdir}/{idx[emb_name]}', [emb_name]))
    w.update(load(f'{tdir}/{idx[head_name]}', [head_name]))
    emb, head, w1, w2 = w[emb_name], w[head_name], w['markov_head.markov_w1.weight'], w['markov_head.markov_w2.weight']
    ctx = {}  # position -> fused row (hidden_norm(fc(taps)))
    done = 0
    stats = dict(rows=0, top1=0, set_eq=0, val_err=0.0, chain_topk=0, chain_full=0, topk_vs_full=0, n=0)
    for kind, rec in records(dump):
        if kind == 'absorb':
            pos, taps = rec
            fused = rms(bf(taps) @ w['fc.weight'].T, w['hidden_norm.weight'], eps)
            for p, r in zip(pos, fused):
                ctx[int(p)] = r
            continue
        pos0, begin, ids, top_ids, top_vals, drafts = rec
        B = len(ids)
        cpos = np.array([p for p in range(max(begin, pos0 - window + 1 - 0), pos0) if p in ctx], dtype=np.int64)
        if len(cpos) != pos0 - max(begin, pos0 - window + 1):
            continue  # a context the dump lacks rows of
        cx = np.stack([ctx[int(p)] for p in cpos])
        bpos = pos0 + np.arange(B)
        h = emb[ids]
        G = nh // nkv
        for i in range(L):
            p = lambda n: w[f'layers.{i}.{n}.weight']
            x = rms(h, p('input_layernorm'), eps)
            q = rope(rms((x @ p('self_attn.q_proj').T).reshape(B, nh, hd), p('self_attn.q_norm'), eps), bpos, theta)
            kb = rope(rms((x @ p('self_attn.k_proj').T).reshape(B, nkv, hd), p('self_attn.k_norm'), eps), bpos, theta)
            vb = (x @ p('self_attn.v_proj').T).reshape(B, nkv, hd)
            kc = rope(rms((cx @ p('self_attn.k_proj').T).reshape(-1, nkv, hd), p('self_attn.k_norm'), eps), cpos, theta)
            vc = (cx @ p('self_attn.v_proj').T).reshape(-1, nkv, hd)
            K, V = np.concatenate([kc, kb]), np.concatenate([vc, vb])
            kp = np.concatenate([cpos, np.full(B, -1)])
            out = np.zeros((B, nh, hd))
            for m in range(B):
                vis = (kp == -1) | (kp >= pos0 + m - window + 1)  # the engine's ring window (no cut below `window`)
                for hh in range(nh):
                    s = K[vis, hh // G] @ q[m, hh] / np.sqrt(hd)
                    e = np.exp(s - s.max())
                    out[m, hh] = (e / e.sum()) @ V[vis, hh // G]
            h = h + out.reshape(B, nh * hd) @ p('self_attn.o_proj').T
            x = rms(h, p('post_attention_layernorm'), eps)
            g, u = x @ p('mlp.gate_proj').T, x @ p('mlp.up_proj').T
            h = h + (g / (1 + np.exp(-g)) * u) @ p('mlp.down_proj').T
        logits = rms(h, w['norm.weight'], eps) @ head.T
        k = top_ids.shape[1]
        for r in range(B):
            ref = np.argsort(-logits[r], kind='stable')[:k]
            stats['rows'] += 1
            stats['top1'] += int(ref[0] == top_ids[r][np.argmax(top_vals[r])])
            stats['set_eq'] += int(set(ref.tolist()) == set(top_ids[r].tolist()))
            stats['val_err'] = max(stats['val_err'], float(np.abs(logits[r][top_ids[r]] - top_vals[r]).max()))
        prev_f = prev_k = int(ids[0])
        full, topk = [], []
        for r in range(len(drafts)):
            full.append(int(np.argmax(logits[r] + w2 @ w1[prev_f])))
            prev_f = full[-1]
            cand = np.argsort(-logits[r], kind='stable')[:k]
            sc = logits[r][cand] + w2[cand] @ w1[prev_k]
            topk.append(int(cand[np.argmax(sc)]))
            prev_k = topk[-1]
        stats['n'] += 1
        stats['chain_topk'] += int(topk == drafts.tolist())
        stats['chain_full'] += int(full == drafts.tolist())
        stats['topk_vs_full'] += int(full == topk)
        print(f'hold pos0 {pos0} ctx {len(cpos)}: engine {drafts.tolist()} ref-topk {topk} ref-full {full}')
        done += 1
        if done == holds:
            break
    print(json.dumps(stats))


if __name__ == '__main__':
    main()
