#!/usr/bin/env python3
"""GGUF reading and ggml dequantization in numpy, for the quality references (quality_ref.py) and checks.

Own code, written from llama.cpp's formats as data (ggml-org/llama.cpp 8bb3cf0d): the container per ggml/src/gguf.cpp,
the blocks per ggml/src/ggml-common.h and the values per ggml/src/ggml-quants.c dequantize_row_* in float32 as the C
code computes them (bit-equal to TensorFold's core/quant/gguf.zig reference on the Unsloth Qwen3.8-27B UD-Q4_K_XL file).

  gguf_ref.py ls FILE      metadata, tensor count and the quantization mix
"""
import collections
import re
import struct
import sys

import numpy as np

T = {0:'F32',1:'F16',2:'Q4_0',3:'Q4_1',6:'Q5_0',7:'Q5_1',8:'Q8_0',10:'Q2_K',11:'Q3_K',12:'Q4_K',13:'Q5_K',14:'Q6_K',16:'IQ2_XXS',17:'IQ2_XS',18:'IQ3_XXS',19:'IQ1_S',20:'IQ4_NL',21:'IQ3_S',22:'IQ2_S',23:'IQ4_XS',24:'I8',30:'BF16',39:'MXFP4'}
def rd(f, fmt): return struct.unpack('<'+fmt, f.read(struct.calcsize('<'+fmt)))
def rstr(f): (n,) = rd(f,'Q'); return f.read(n).decode('utf-8','replace')
SC = {0:'B',1:'b',2:'H',3:'h',4:'I',5:'i',6:'f',7:'?',10:'Q',11:'q',12:'d'}
def rval(f,t):
    if t == 8: return rstr(f)
    if t == 9:
        (et,n) = rd(f,'IQ'); return [rval(f,et) for _ in range(n)]
    return rd(f, SC[t])[0]
def read(path):
    f = open(path,'rb')
    assert f.read(4) == b'GGUF'
    ver, nt, nkv = rd(f,'IQQ')
    kv = {}
    for _ in range(nkv):
        k = rstr(f); (t,) = rd(f,'I'); kv[k] = rval(f,t)
    ts = []
    for _ in range(nt):
        name = rstr(f); (nd,) = rd(f,'I'); dims = rd(f,'Q'*nd); (ty,off) = rd(f,'IQ')
        ts.append((name, dims, ty, off))
    al = kv.get('general.alignment', 32)
    data = (f.tell() + al - 1)//al*al
    return ver, kv, ts, data
KV = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)
GRID = np.array([
    0x01010101, 0x01010103, 0x01010105, 0x0101010b, 0x0101010f, 0x01010301, 0x01010303, 0x01010305,
    0x01010309, 0x0101030d, 0x01010501, 0x01010503, 0x0101050b, 0x01010707, 0x01010901, 0x01010905,
    0x0101090b, 0x0101090f, 0x01010b03, 0x01010b07, 0x01010d01, 0x01010d05, 0x01010f03, 0x01010f09,
    0x01010f0f, 0x01030101, 0x01030103, 0x01030105, 0x01030109, 0x01030301, 0x01030303, 0x0103030b,
    0x01030501, 0x01030507, 0x0103050f, 0x01030703, 0x0103070b, 0x01030909, 0x01030d03, 0x01030d0b,
    0x01030f05, 0x01050101, 0x01050103, 0x0105010b, 0x0105010f, 0x01050301, 0x01050307, 0x0105030d,
    0x01050503, 0x0105050b, 0x01050701, 0x01050709, 0x01050905, 0x0105090b, 0x0105090f, 0x01050b03,
    0x01050b07, 0x01050f01, 0x01050f07, 0x01070107, 0x01070303, 0x0107030b, 0x01070501, 0x01070505,
    0x01070703, 0x01070707, 0x0107070d, 0x01070909, 0x01070b01, 0x01070b05, 0x01070d0f, 0x01070f03,
    0x01070f0b, 0x01090101, 0x01090307, 0x0109030f, 0x01090503, 0x01090509, 0x01090705, 0x01090901,
    0x01090907, 0x01090b03, 0x01090f01, 0x010b0105, 0x010b0109, 0x010b0501, 0x010b0505, 0x010b050d,
    0x010b0707, 0x010b0903, 0x010b090b, 0x010b090f, 0x010b0d0d, 0x010b0f07, 0x010d010d, 0x010d0303,
    0x010d0307, 0x010d0703, 0x010d0b05, 0x010d0f03, 0x010f0101, 0x010f0105, 0x010f0109, 0x010f0501,
    0x010f0505, 0x010f050d, 0x010f0707, 0x010f0b01, 0x010f0b09, 0x03010101, 0x03010103, 0x03010105,
    0x03010109, 0x03010301, 0x03010303, 0x03010307, 0x0301030b, 0x0301030f, 0x03010501, 0x03010505,
    0x03010703, 0x03010709, 0x0301070d, 0x03010b09, 0x03010b0d, 0x03010d03, 0x03010f05, 0x03030101,
    0x03030103, 0x03030107, 0x0303010d, 0x03030301, 0x03030309, 0x03030503, 0x03030701, 0x03030707,
    0x03030903, 0x03030b01, 0x03030b05, 0x03030f01, 0x03030f0d, 0x03050101, 0x03050305, 0x0305030b,
    0x0305030f, 0x03050501, 0x03050509, 0x03050705, 0x03050901, 0x03050907, 0x03050b0b, 0x03050d01,
    0x03050f05, 0x03070103, 0x03070109, 0x0307010f, 0x03070301, 0x03070307, 0x03070503, 0x0307050f,
    0x03070701, 0x03070709, 0x03070903, 0x03070d05, 0x03070f01, 0x03090107, 0x0309010b, 0x03090305,
    0x03090309, 0x03090703, 0x03090707, 0x03090905, 0x0309090d, 0x03090b01, 0x03090b09, 0x030b0103,
    0x030b0301, 0x030b0307, 0x030b0503, 0x030b0701, 0x030b0705, 0x030b0b03, 0x030d0501, 0x030d0509,
    0x030d050f, 0x030d0909, 0x030d090d, 0x030f0103, 0x030f0107, 0x030f0301, 0x030f0305, 0x030f0503,
    0x030f070b, 0x030f0903, 0x030f0d05, 0x030f0f01, 0x05010101, 0x05010103, 0x05010107, 0x0501010b,
    0x0501010f, 0x05010301, 0x05010305, 0x05010309, 0x0501030d, 0x05010503, 0x05010507, 0x0501050f,
    0x05010701, 0x05010705, 0x05010903, 0x05010907, 0x0501090b, 0x05010b01, 0x05010b05, 0x05010d0f,
    0x05010f01, 0x05010f07, 0x05010f0b, 0x05030101, 0x05030105, 0x05030301, 0x05030307, 0x0503030f,
    0x05030505, 0x0503050b, 0x05030703, 0x05030709, 0x05030905, 0x05030b03, 0x05050103, 0x05050109,
    0x0505010f, 0x05050503, 0x05050507, 0x05050701, 0x0505070f, 0x05050903, 0x05050b07, 0x05050b0f,
    0x05050f03, 0x05050f09, 0x05070101, 0x05070105, 0x0507010b, 0x05070303, 0x05070505, 0x05070509,
    0x05070703, 0x05070707, 0x05070905, 0x05070b01, 0x05070d0d, 0x05090103, 0x0509010f, 0x05090501,
    0x05090507, 0x05090705, 0x0509070b, 0x05090903, 0x05090f05, 0x05090f0b, 0x050b0109, 0x050b0303,
    0x050b0505, 0x050b070f, 0x050b0901, 0x050b0b07, 0x050b0f01, 0x050d0101, 0x050d0105, 0x050d010f,
    0x050d0503, 0x050d0b0b, 0x050d0d03, 0x050f010b, 0x050f0303, 0x050f050d, 0x050f0701, 0x050f0907,
    0x050f0b01, 0x07010105, 0x07010303, 0x07010307, 0x0701030b, 0x0701030f, 0x07010505, 0x07010703,
    0x07010707, 0x0701070b, 0x07010905, 0x07010909, 0x0701090f, 0x07010b03, 0x07010d07, 0x07010f03,
    0x07030103, 0x07030107, 0x0703010b, 0x07030309, 0x07030503, 0x07030507, 0x07030901, 0x07030d01,
    0x07030f05, 0x07030f0d, 0x07050101, 0x07050305, 0x07050501, 0x07050705, 0x07050709, 0x07050b01,
    0x07070103, 0x07070301, 0x07070309, 0x07070503, 0x07070507, 0x0707050f, 0x07070701, 0x07070903,
    0x07070907, 0x0707090f, 0x07070b0b, 0x07070f07, 0x07090107, 0x07090303, 0x0709030d, 0x07090505,
    0x07090703, 0x07090b05, 0x07090d01, 0x07090d09, 0x070b0103, 0x070b0301, 0x070b0305, 0x070b050b,
    0x070b0705, 0x070b0909, 0x070b0b0d, 0x070b0f07, 0x070d030d, 0x070d0903, 0x070f0103, 0x070f0107,
    0x070f0501, 0x070f0505, 0x070f070b, 0x09010101, 0x09010109, 0x09010305, 0x09010501, 0x09010509,
    0x0901050f, 0x09010705, 0x09010903, 0x09010b01, 0x09010f01, 0x09030105, 0x0903010f, 0x09030303,
    0x09030307, 0x09030505, 0x09030701, 0x0903070b, 0x09030907, 0x09030b03, 0x09030b0b, 0x09050103,
    0x09050107, 0x09050301, 0x0905030b, 0x09050503, 0x09050707, 0x09050901, 0x09050b0f, 0x09050d05,
    0x09050f01, 0x09070109, 0x09070303, 0x09070307, 0x09070501, 0x09070505, 0x09070703, 0x0907070b,
    0x09090101, 0x09090105, 0x09090509, 0x0909070f, 0x09090901, 0x09090f03, 0x090b010b, 0x090b010f,
    0x090b0503, 0x090b0d05, 0x090d0307, 0x090d0709, 0x090d0d01, 0x090f0301, 0x090f030b, 0x090f0701,
    0x090f0907, 0x090f0b03, 0x0b010105, 0x0b010301, 0x0b010309, 0x0b010505, 0x0b010901, 0x0b010909,
    0x0b01090f, 0x0b010b05, 0x0b010d0d, 0x0b010f09, 0x0b030103, 0x0b030107, 0x0b03010b, 0x0b030305,
    0x0b030503, 0x0b030705, 0x0b030f05, 0x0b050101, 0x0b050303, 0x0b050507, 0x0b050701, 0x0b05070d,
    0x0b050b07, 0x0b070105, 0x0b07010f, 0x0b070301, 0x0b07050f, 0x0b070909, 0x0b070b03, 0x0b070d0b,
    0x0b070f07, 0x0b090103, 0x0b090109, 0x0b090501, 0x0b090705, 0x0b09090d, 0x0b0b0305, 0x0b0b050d,
    0x0b0b0b03, 0x0b0b0b07, 0x0b0d0905, 0x0b0f0105, 0x0b0f0109, 0x0b0f0505, 0x0d010303, 0x0d010307,
    0x0d01030b, 0x0d010703, 0x0d010707, 0x0d010d01, 0x0d030101, 0x0d030501, 0x0d03050f, 0x0d030d09,
    0x0d050305, 0x0d050709, 0x0d050905, 0x0d050b0b, 0x0d050d05, 0x0d050f01, 0x0d070101, 0x0d070309,
    0x0d070503, 0x0d070901, 0x0d09050b, 0x0d090907, 0x0d090d05, 0x0d0b0101, 0x0d0b0107, 0x0d0b0709,
    0x0d0b0d01, 0x0d0d010b, 0x0d0d0901, 0x0d0f0303, 0x0d0f0307, 0x0f010101, 0x0f010109, 0x0f01010f,
    0x0f010501, 0x0f010505, 0x0f01070d, 0x0f010901, 0x0f010b09, 0x0f010d05, 0x0f030105, 0x0f030303,
    0x0f030509, 0x0f030907, 0x0f03090b, 0x0f050103, 0x0f050109, 0x0f050301, 0x0f05030d, 0x0f050503,
    0x0f050701, 0x0f050b03, 0x0f070105, 0x0f070705, 0x0f07070b, 0x0f070b07, 0x0f090103, 0x0f09010b,
    0x0f090307, 0x0f090501, 0x0f090b01, 0x0f0b0505, 0x0f0b0905, 0x0f0d0105, 0x0f0d0703, 0x0f0f0101,
], np.uint32)  # ggml-common.h iq3s_grid
BLOCK = {0: (1, 4), 1: (1, 2), 30: (1, 2), 8: (32, 34), 12: (256, 144), 13: (256, 176), 14: (256, 210), 11: (256, 110), 23: (256, 136), 20: (32, 18), 21: (256, 110)}
f16 = lambda b: b.view(np.float16).astype(np.float32)[:, 0]

def scale_min_k4(sc):  # sc [nb, 12] -> (d, m) [nb, 8] (get_scale_min_k4)
    d = np.empty(sc.shape[:-1] + (8,), np.uint8); m = np.empty_like(d)
    for j in range(8):
        if j < 4:
            d[..., j] = sc[..., j] & 63; m[..., j] = sc[..., j + 4] & 63
        else:
            d[..., j] = (sc[..., j + 4] & 0xF) | ((sc[..., j - 4] >> 6) << 4)
            m[..., j] = (sc[..., j + 4] >> 4) | ((sc[..., j] >> 6) << 4)
    return d.astype(np.float32), m.astype(np.float32)

def dequant(ty, raw, n_el):
    """raw: uint8 bytes of n_el values -> float32 [n_el]."""
    if ty == 0: return raw.view(np.float32).copy()
    if ty == 1: return raw.view(np.float16).astype(np.float32)
    if ty == 30: return (raw.view(np.uint16).astype(np.uint32) << 16).view(np.float32)
    E, B = BLOCK[ty]; b = raw.reshape(-1, B)
    if ty == 8:
        d = f16(b[:, 0:2].copy()); q = b[:, 2:].view(np.int8).astype(np.float32)
        return (q * d[:, None]).reshape(-1)
    if ty == 20:
        d = f16(b[:, 0:2].copy()); qs = b[:, 2:]
        v = np.concatenate([KV[qs & 0xF], KV[qs >> 4]], 1)
        return (d[:, None] * v).reshape(-1)
    if ty in (12, 13):
        d = f16(b[:, 0:2].copy()); dm = f16(b[:, 2:4].copy()); sc, mn = scale_min_k4(b[:, 4:16])
        qs = b[:, 16:144] if ty == 12 else b[:, 48:176]
        qh = None if ty == 12 else b[:, 16:48]
        out = np.empty((b.shape[0], 256), np.float32)
        for c in range(4):
            q = qs[:, 32 * c:32 * c + 32]
            lo = (q & 0xF).astype(np.float32); hi = (q >> 4).astype(np.float32)
            if qh is not None:
                lo += np.where(qh & (1 << (2 * c)), 16, 0); hi += np.where(qh & (1 << (2 * c + 1)), 16, 0)
            d1 = d * sc[:, 2 * c]; m1 = dm * mn[:, 2 * c]; d2 = d * sc[:, 2 * c + 1]; m2 = dm * mn[:, 2 * c + 1]
            out[:, 64 * c:64 * c + 32] = d1[:, None] * lo - m1[:, None]
            out[:, 64 * c + 32:64 * c + 64] = d2[:, None] * hi - m2[:, None]
        return out.reshape(-1)
    if ty == 14:
        ql = b[:, 0:128]; qh = b[:, 128:192]; sc = b[:, 192:208].view(np.int8).astype(np.float32); d = f16(b[:, 208:210].copy())
        out = np.empty((b.shape[0], 256), np.float32)
        for n in range(2):
            l_ = ql[:, 64 * n:64 * n + 64]; h_ = qh[:, 32 * n:32 * n + 32]
            q1 = ((l_[:, 0:32] & 0xF) | (((h_ >> 0) & 3) << 4)).astype(np.float32) - 32
            q2 = ((l_[:, 32:64] & 0xF) | (((h_ >> 2) & 3) << 4)).astype(np.float32) - 32
            q3 = ((l_[:, 0:32] >> 4) | (((h_ >> 4) & 3) << 4)).astype(np.float32) - 32
            q4 = ((l_[:, 32:64] >> 4) | (((h_ >> 6) & 3) << 4)).astype(np.float32) - 32
            for qi, q in enumerate((q1, q2, q3, q4)):
                for half in range(2):
                    s = d * sc[:, 8 * n + 2 * qi + half]
                    out[:, 128 * n + 32 * qi + 16 * half:128 * n + 32 * qi + 16 * half + 16] = s[:, None] * q[:, 16 * half:16 * half + 16]
        return out.reshape(-1)
    if ty == 23:
        d = f16(b[:, 0:2].copy()); sh = b[:, 2:4].copy().view(np.uint16)[:, 0].astype(np.int32); sl = b[:, 4:8]; qs = b[:, 8:136]
        out = np.empty((b.shape[0], 256), np.float32)
        for ib in range(8):
            ls = ((sl[:, ib // 2] >> (4 * (ib % 2))) & 0xF).astype(np.int32) | (((sh >> (2 * ib)) & 3) << 4)
            dl = d * (ls - 32).astype(np.float32)
            q = qs[:, 16 * ib:16 * ib + 16]
            out[:, 32 * ib:32 * ib + 16] = dl[:, None] * KV[q & 0xF]
            out[:, 32 * ib + 16:32 * ib + 32] = dl[:, None] * KV[q >> 4]
        return out.reshape(-1)
    if ty == 11:
        hm = b[:, 0:32]; qs = b[:, 32:96]; raw_sc = b[:, 96:108].copy().view(np.uint32); d = f16(b[:, 108:110].copy())
        a0, a1, tmp = raw_sc[:, 0], raw_sc[:, 1], raw_sc[:, 2]
        k1, k2 = np.uint32(0x03030303), np.uint32(0x0f0f0f0f)
        aux = np.stack([(a0 & k2) | (((tmp >> 0) & k1) << 4), (a1 & k2) | (((tmp >> 2) & k1) << 4),
                        ((a0 >> 4) & k2) | (((tmp >> 4) & k1) << 4), ((a1 >> 4) & k2) | (((tmp >> 6) & k1) << 4)], 1)
        scales = aux.view(np.int8).astype(np.float32)  # [nb, 16]
        out = np.empty((b.shape[0], 256), np.float32)
        isx = 0
        for n in range(2):
            q = qs[:, 32 * n:32 * n + 32]
            for j in range(4):
                m = 1 << (4 * n + j)
                for half in range(2):
                    l = slice(16 * half, 16 * half + 16)
                    v = ((q[:, l] >> (2 * j)) & 3).astype(np.float32) - np.where(hm[:, l] & m, 0, 4)
                    dl = d * (scales[:, isx] - 32); isx += 1
                    out[:, 128 * n + 32 * j + 16 * half:128 * n + 32 * j + 16 * half + 16] = dl[:, None] * v
        return out.reshape(-1)
    if ty == 21:
        d = f16(b[:, 0:2].copy()); qs = b[:, 2:66]; qh = b[:, 66:74]; signs = b[:, 74:106]; sc = b[:, 106:110]
        out = np.empty((b.shape[0], 256), np.float32)
        for ib in range(8):
            db = d * (1 + 2 * ((sc[:, ib // 2] >> (4 * (ib % 2))) & 0xF)).astype(np.float32)
            for l in range(4):
                for g in range(2):
                    idx = qs[:, 8 * ib + 2 * l + g].astype(np.int32) | ((qh[:, ib].astype(np.int32) << (8 - 2 * l - g)) & 256)
                    vals = GRID[idx].view(np.uint8).reshape(-1, 4).astype(np.float32)
                    for j in range(4):
                        sgn = np.where(signs[:, 4 * ib + l] & (1 << (j + 4 * g)), -1.0, 1.0).astype(np.float32)
                        out[:, 32 * ib + 8 * l + 4 * g + j] = db * vals[:, j] * sgn
        return out.reshape(-1)
    raise ValueError(f'type {ty}')

def tensor(path, info, data_off, rows=None):
    """[N, K] float32 of a 2-D tensor (or 1-D [K]); rows: a slice of rows to read."""
    name, dims, ty, off = info
    E, B = BLOCK[ty]; K = dims[0]; N = int(np.prod(dims[1:])) if len(dims) > 1 else 1
    rb = K // E * B
    r0, r1 = (0, N) if rows is None else (rows.start, rows.stop)
    mm = np.memmap(path, np.uint8, 'r', offset=data_off + off + r0 * rb, shape=((r1 - r0) * rb,))
    return dequant(ty, np.asarray(mm), (r1 - r0) * K).reshape(r1 - r0, K)


# HF suffix -> GGUF suffix of a trunk layer (llama.cpp gguf-py tensor_mapping.py for qwen35)
HF_NAMES = {
    "input_layernorm.weight": "attn_norm.weight", "post_attention_layernorm.weight": "post_attention_norm.weight",
    "linear_attn.in_proj_qkv.weight": "attn_qkv.weight", "linear_attn.in_proj_z.weight": "attn_gate.weight",
    "linear_attn.in_proj_a.weight": "ssm_alpha.weight", "linear_attn.in_proj_b.weight": "ssm_beta.weight",
    "linear_attn.out_proj.weight": "ssm_out.weight", "linear_attn.conv1d.weight": "ssm_conv1d.weight",
    "linear_attn.dt_bias": "ssm_dt.bias", "linear_attn.A_log": "ssm_a", "linear_attn.norm.weight": "ssm_norm.weight",
    "self_attn.q_proj.weight": "attn_q.weight", "self_attn.k_proj.weight": "attn_k.weight",
    "self_attn.v_proj.weight": "attn_v.weight", "self_attn.o_proj.weight": "attn_output.weight",
    "self_attn.q_norm.weight": "attn_q_norm.weight", "self_attn.k_norm.weight": "attn_k_norm.weight",
    "mlp.gate_proj.weight": "ffn_gate.weight", "mlp.up_proj.weight": "ffn_up.weight", "mlp.down_proj.weight": "ffn_down.weight",
}


class Checkpoint:
    """A qwen35 GGUF under the Hugging Face names, float32 numpy, with llama.cpp's conversion (conversion/qwen.py
    Qwen3NextModel and _LinearAttentionVReorderBase.modify_tensors) undone: norms back to w (but linear_attn.norm),
    A_log = log(-ssm_a), conv1d (C, 1, K), and the linear attention's value heads from the tiled order back to grouped."""

    def __init__(self, path, nk, nv, dk, dv):
        self.path = path
        self.dk = dk
        _, self.kv, ts, self.data = read(path)
        self.info = {t[0]: t for t in ts}
        self.nk, self.vpk, self.dv = nk, nv // nk, dv
        self.layers = int(self.kv["qwen35.block_count"]) - int(self.kv.get("qwen35.nextn_predict_layers", 0))
        self.names = ["model.language_model.embed_tokens.weight", "model.language_model.norm.weight", "lm_head.weight"]
        for i in range(self.layers):
            for hf, g in HF_NAMES.items():
                if f"blk.{i}.{g}" in self.info:
                    self.names.append(f"model.language_model.layers.{i}.{hf}")

    def raw(self, name, rows=None):
        return tensor(self.path, self.info[name], self.data, rows)

    def grouped(self, t, axis, head):
        """The tiled value heads (j nk + g) of axis `axis` back in the grouped order (g vpk + j)."""
        t = np.moveaxis(t, axis, 0)
        shape = t.shape
        t = t.reshape(self.vpk, self.nk, head, -1).transpose(1, 0, 2, 3).reshape(shape)
        return np.ascontiguousarray(np.moveaxis(t, 0, axis))

    def embed_rows(self, ids):
        """The embedding rows of `ids` (any shape), each dequantized once."""
        uniq, inv = np.unique(np.asarray(ids).reshape(-1), return_inverse=True)
        rows = np.stack([self.raw("token_embd.weight", slice(int(i), int(i) + 1))[0] for i in uniq])
        return rows[inv].reshape(*np.asarray(ids).shape, -1)

    def get(self, name):
        if name == "model.language_model.embed_tokens.weight":
            return self.raw("token_embd.weight")
        if name == "lm_head.weight":
            return self.raw("output.weight" if "output.weight" in self.info else "token_embd.weight")
        if name == "model.language_model.norm.weight":
            return self.raw("output_norm.weight")[0] - 1
        i, hf = re.match(r"model\.language_model\.layers\.(\d+)\.(.+)", name).groups()
        t = self.raw(f"blk.{i}.{HF_NAMES[hf]}")
        if hf.endswith(("layernorm.weight", "q_norm.weight", "k_norm.weight")):
            return t[0] - 1
        kd = self.nk * self.dk  # q and k ahead of v in qkv and the conv
        if hf == "linear_attn.A_log":
            return np.log(-self.grouped(t[0], 0, 1).astype(np.float64)).astype(np.float32)
        if hf == "linear_attn.dt_bias":
            return self.grouped(t[0], 0, 1)
        if hf in ("linear_attn.norm.weight",):
            return t[0]
        if hf == "linear_attn.conv1d.weight":
            t = np.concatenate([t[:2 * kd], self.grouped(t[2 * kd:], 0, self.dv)])
            return t[:, None, :]
        if hf == "linear_attn.in_proj_qkv.weight":
            return np.concatenate([t[:2 * kd], self.grouped(t[2 * kd:], 0, self.dv)])
        if hf == "linear_attn.in_proj_z.weight":
            return self.grouped(t, 0, self.dv)
        if hf in ("linear_attn.in_proj_a.weight", "linear_attn.in_proj_b.weight"):
            return self.grouped(t, 0, 1)
        if hf == "linear_attn.out_proj.weight":
            return self.grouped(t, 1, self.dv)
        return t


def main():
    if len(sys.argv) < 3 or sys.argv[1] != "ls":
        sys.exit(__doc__)
    ver, kv, ts, data = read(sys.argv[2])
    print("version", ver, "tensors", len(ts), "data at", data)
    for k, v in kv.items():
        if isinstance(v, list) and len(v) > 8:
            v = f"[{len(v)} items]"
        if not k.startswith("tokenizer.chat_template"):
            print(" ", k, "=", v)
    mix = collections.defaultdict(collections.Counter)
    for name, dims, ty, off in ts:
        mix[re.sub(r"blk\.\d+\.", "blk.N.", name)][T.get(ty, ty)] += 1
    for k in sorted(mix):
        print(" ", k, dict(mix[k]))


if __name__ == "__main__":
    main()
