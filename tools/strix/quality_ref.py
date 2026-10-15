#!/usr/bin/env python3
"""The quality gate's fixed inputs: the token set and the bf16 reference logits (torch + transformers).

  quality_ref.py ids <tokenizer.json> [--wikitext FILE]   -> tools/strix/quality_ids.npy, int32 (8, 4096)
  quality_ref.py ref <bf16 model dir> --out DIR           -> DIR/ref_{i}.npy, fp16 (4096, vocab) logits per sequence

The set (after Mia's AI Lab's Qwen3.8-27B measurement): 4 windows of 4,096 tokens from the wikitext-2-raw-v1
test split ("\\n\\n".join of its rows, windows 0..3 from the start) and the first 4,096 tokens of 4 CPython
3.13 stdlib modules. No BOS (Qwen has none in pretraining text).

The reference runs the transformers Qwen3_5 decoder layers (GatedDeltaNet, gated attention, partial mrope)
layer-major: one layer's weights on the GPU at a time (~1 GB), every sequence through it as two 2,048-row
chunks, the second reading the first's KV, conv and recurrent state from that layer's cache. That is the
same arithmetic as a token-major chunked forward and keeps GPU memory near 3 GB. The head is computed in
fp32 from the bf16 final-normed rows, stored as fp16.
"""
import argparse
import json
import os
import sys
import sysconfig
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
IDS = os.path.join(HERE, "quality_ids.npy")
SEQ, CHUNK = 4096, 2048
ROWS = 256  # head rows per block
CPYTHON = ["argparse.py", "typing.py", "inspect.py", "dataclasses.py"]


def build_ids(a):
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(a.tokenizer)
    enc = lambda s: tok.encode(s, add_special_tokens=False).ids
    if a.wikitext:
        text = open(a.wikitext).read()
    else:
        from datasets import load_dataset
        text = "\n\n".join(load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")["text"])
    wiki = enc(text)
    rows = [wiki[i * SEQ:(i + 1) * SEQ] for i in range(4)]
    lib = sysconfig.get_paths()["stdlib"]
    rows += [enc(open(os.path.join(lib, f)).read())[:SEQ] for f in CPYTHON]
    assert all(len(r) == SEQ for r in rows)
    np.save(IDS, np.array(rows, dtype=np.int32))
    print(f"wrote {IDS}: wikitext-2 test windows 0-3, CPython {sys.version.split()[0]} {', '.join(CPYTHON)}")


def reference(a):
    import torch
    from safetensors import safe_open
    from transformers import AutoConfig, DynamicCache
    from transformers.models.qwen3_5 import modeling_qwen3_5 as q

    torch.set_grad_enabled(False)
    torch.set_num_threads(16)  # a shared box
    dev = "cuda"
    cfg = AutoConfig.from_pretrained(a.model).text_config
    cfg._attn_implementation = "sdpa"
    wm = json.load(open(os.path.join(a.model, "model.safetensors.index.json")))["weight_map"]
    files = {}

    def raw(name):
        f = wm[name]
        if f not in files:
            files[f] = safe_open(os.path.join(a.model, f), "pt", device="cpu")
        return files[f].get_tensor(name)

    tensor, names = raw, wm
    quant = json.load(open(os.path.join(a.model, "config.json"))).get("quantization")
    if quant:  # an MLX affine checkpoint, dequantized to bf16 under the HF names (a check of the engine's arithmetic)
        hf = lambda k: k.replace("language_model.model.", "model.language_model.").replace("language_model.lm_head", "lm_head")
        names = {hf(k): k for k in wm if k.startswith("language_model.") and not k.endswith((".scales", ".biases"))}
        shifted = ("input_layernorm.weight", "post_attention_layernorm.weight", "q_norm.weight", "k_norm.weight")

        def tensor(name):
            k = names[name]
            t = raw(k)
            if k.endswith(".weight") and k[:-6] + "scales" in wm:
                bits, group = quant["bits"], quant["group_size"]
                q = t.to(dev).view(torch.int32)
                shifts = torch.arange(0, 32, bits, device=dev, dtype=torch.int32)
                w = ((q.unsqueeze(-1) >> shifts) & ((1 << bits) - 1)).reshape(q.shape[0], -1, group).float()
                s, b = raw(k[:-6] + "scales").to(dev).float(), raw(k[:-6] + "biases").to(dev).float()
                return (w * s.unsqueeze(-1) + b.unsqueeze(-1)).reshape(q.shape[0], -1).to(torch.bfloat16).cpu()
            if k.endswith(shifted) or k == "language_model.model.norm.weight":
                return (t.float() - 1).to(t.dtype)  # MLX stores 1 + w; HF's RMSNorm adds the 1
            if k.endswith("conv1d.weight"):
                return t.transpose(1, 2).contiguous()  # MLX (C, K, 1) -> torch (C, 1, K)
            return t

    ids = torch.from_numpy(np.load(a.ids).astype(np.int64))
    n = ids.shape[0]
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    embed = tensor("model.language_model.embed_tokens.weight")
    hidden = embed[ids].to(dev)  # (n, SEQ, hidden) bf16
    del embed
    rope = q.Qwen3_5TextRotaryEmbedding(cfg).to(dev)
    pos = torch.arange(SEQ, device=dev).view(1, 1, -1).expand(3, 1, -1)
    cos, sin = rope(hidden[:1], pos)
    # chunk 2's rows see every chunk-1 key and their own causal prefix
    mask2 = torch.ones(CHUNK, SEQ, dtype=torch.bool, device=dev).tril(CHUNK)[None, None]
    for i in range(cfg.num_hidden_layers):
        pre = f"model.language_model.layers.{i}."
        sd = {k[len(pre):]: tensor(k).to(dev) for k in names if k.startswith(pre)}
        with torch.device("meta"):
            layer = q.Qwen3_5DecoderLayer(cfg, i)
        layer.load_state_dict(sd, assign=True, strict=True)
        layer.eval()
        full = cfg.layer_types[i] == "full_attention"
        for s in range(n):
            cache = DynamicCache(config=cfg)
            for c in range(SEQ // CHUNK):
                r = slice(c * CHUNK, (c + 1) * CHUNK)
                hidden[s:s + 1, r] = layer(hidden[s:s + 1, r], position_embeddings=(cos[:, r], sin[:, r]),
                                           attention_mask=(mask2 if c else None) if full else None,
                                           past_key_values=cache, use_cache=True)
                if not torch.isfinite(hidden[s, r]).all():
                    sys.exit(f"non-finite rows: layer {i} ({cfg.layer_types[i]}) seq {s} chunk {c}")
        del layer, sd, cache
        print(f"layer {i + 1}/{cfg.num_hidden_layers} {time.time() - t0:.0f}s", flush=True)
    norm = q.Qwen3_5RMSNorm(cfg.hidden_size, eps=cfg.rms_norm_eps).to(dev)
    norm.weight.copy_(tensor("model.language_model.norm.weight"))
    hidden = hidden.cpu()
    head = tensor("lm_head.weight").to(dev)  # (vocab, hidden) bf16
    nll, count = 0.0, 0
    for s in range(n):
        h = norm(hidden[s].to(dev)).float()
        out = np.lib.format.open_memmap(os.path.join(a.out, f"ref_{s}.npy"), "w+", np.float16, (SEQ, head.shape[0]))
        for r0 in range(0, SEQ, ROWS):
            logits = torch.cat([h[r0:r0 + ROWS] @ head[v:v + 32768].float().T for v in range(0, head.shape[0], 32768)], 1)
            lp = torch.log_softmax(logits, -1)
            tgt = ids[s, r0 + 1:r0 + ROWS + 1].to(dev)
            nll -= lp[torch.arange(len(tgt), device=dev), tgt].sum().item()
            count += len(tgt)
            out[r0:r0 + ROWS] = logits.half().cpu().numpy()
        out.flush()
        del out
    meta = {"model": os.path.abspath(a.model), "ids": os.path.abspath(a.ids), "ppl": float(np.exp(nll / count)),
            "seconds": round(time.time() - t0, 1), "torch": torch.__version__,
            "transformers": __import__("transformers").__version__}
    json.dump(meta, open(os.path.join(a.out, "ref.json"), "w"), indent=1)
    print(json.dumps(meta))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    i = sub.add_parser("ids")
    i.add_argument("tokenizer")
    i.add_argument("--wikitext", help="the test split already joined, else fetched with datasets")
    r = sub.add_parser("ref")
    r.add_argument("model")
    r.add_argument("--out", required=True)
    r.add_argument("--ids", default=IDS)
    a = p.parse_args()
    build_ids(a) if a.cmd == "ids" else reference(a)


if __name__ == "__main__":
    main()
