"""fn_int4.hip (zig/kernels/hip) on gfx1151 against an fp64 host reference: python tools/rocm/int4_check.py [--bench]

Mirrors zig/src/families/flashnext/cuda_int4_check.zig: the dense kernel against the documented order (group sums
exact, rounded to fp32, acc = fmaf(sum, scale, acc) in group order) and the full fp64 dot product; row invariance
(each row of an m-row call byte-equal to the 1-row call, rows permuted); NT / MT invariance; the group order with
cancelling groups; the experts plan == dense (gate/up SwiGLU, down fp32 / bf16, groups of 128 and 64, items of 16 and
64), the skipped slot untouched; determinism; real expert tensors from the checkpoint (--model).
"""
import argparse, json, os, struct, sys
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hiprun

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WAVES = 4
NTS = (1, 2, 4)
MTS = (1, 4)


def load(abl=0):
    co = hiprun.build(f"{ROOT}/zig/kernels/hip/fn_int4.hip", hiprun.cache(f"fn_int4_{abl}.co"), [f"-DTF_INT4_ABL={abl}"])
    return hiprun.Module(co)


M = None
OCC = {}


def sym(gs, nt, mt, mats, epi):
    return f"tf_int4_{gs}_{nt}_{mt}_{mats}_{epi}"


# ---- GPTQ data and the packed layout (cuda_int4.zig packWords / packScales) ----------------------------------------

def gen(rng, n, k):
    """GPTQ bytes for [n, k]: qweight int32 [k/8, n], scales fp16 [k/128, n] in [2^-9, 2^-6)."""
    qw = rng.integers(0, 2**32, size=(k // 8, n), dtype=np.uint64).astype(np.uint32)
    sc = np.ldexp(1.0 + rng.random((k // 128, n)), -9 + rng.integers(0, 3, (k // 128, n))).astype(np.float16)
    return qw, sc


def codes(qw):
    """[k, n] codes 0..15 of qweight [k/8, n]."""
    sh = (4 * np.arange(8, dtype=np.uint32))[None, :, None]
    return ((qw[:, None, :] >> sh) & 15).reshape(-1, qw.shape[1]).astype(np.int8)


def pack(qw, sc, n0, n, k0, k, gs):
    """Words [n/16][k/gs][gs/32][16][4] (GPTQ's word for 8 inputs, xor 0x88888888) and scales [n/16][k/gs][16]."""
    w = qw[k0 // 8:(k0 + k) // 8, n0:n0 + n] ^ np.uint32(0x88888888)          # [k/8, n]
    w = w.reshape(k // gs, gs // 32, 4, n // 16, 16).transpose(3, 0, 1, 4, 2)  # [t][g][ch][c][wi]
    rows = (k0 + gs * np.arange(k // gs)) // 128
    s = sc[rows][:, n0:n0 + n].reshape(k // gs, n // 16, 16).transpose(1, 0, 2)
    return np.ascontiguousarray(w).reshape(-1), np.ascontiguousarray(s).reshape(-1)


def dev(a):
    return torch.from_numpy(np.ascontiguousarray(a)).cuda()


def bf16_rows(rng, rows, k):
    return torch.from_numpy((rng.standard_normal((rows, k)) * 0.5).astype(np.float32)).bfloat16()


def f32_to_bf16_bits(x):
    return torch.from_numpy(np.ascontiguousarray(x, dtype=np.float32)).bfloat16().view(torch.int16).numpy()


# ---- host reference ---------------------------------------------------------------------------------------------

def emulate(q, sc, k0, k, gs, x):
    """x [r, k] fp64, q [k_full, n] codes: (documented order fp32, exact fp64, sum |x w|) [r, n]."""
    qq = q[k0:k0 + k].astype(np.float64) - 8
    n = qq.shape[1]
    kg = k // gs
    p = np.einsum("rgk,gkn->rgn", x.reshape(-1, kg, gs), qq.reshape(kg, gs, n))
    s = sc[(k0 + gs * np.arange(kg)) // 128].astype(np.float64)                        # [kg, n]
    acc = np.zeros((x.shape[0], n), np.float32)
    for g in range(kg):
        acc = (p[:, g].astype(np.float32).astype(np.float64) * s[g] + acc).astype(np.float32)
    exact = (p * s[None]).sum(1)
    w = qq.reshape(kg, gs, n) * s[:, None, :]
    mag = np.abs(x)[:, :, None].reshape(-1, kg, gs, 1) * np.abs(w)[None]
    return acc, exact, mag.sum((1, 2))


def ulps(a, b):
    ia = a.view(np.int32).astype(np.int64)
    ib = b.view(np.int32).astype(np.int64)
    fa = np.where(ia < 0, -(ia & 0x7FFFFFFF), ia)
    fb = np.where(ib < 0, -(ib & 0x7FFFFFFF), ib)
    return np.abs(fa - fb)


# ---- launches (cuda_int4.zig launch) ----------------------------------------------------------------------------

def blocks(name):
    if name not in OCC:
        OCC[name] = max(1, M.occupancy(name, WAVES * 32)) * torch.cuda.get_device_properties(0).multi_processor_count
    return OCC[name]


def launch(gs, nt, mt, mats, epi, x, x_stride, slots, w, s, k, n, plan, rows, out, max_items, skip=-1):
    name = sym(gs, nt, mt, mats, epi)
    units = max_items * (n // (16 * nt))
    grid = min((units + WAVES - 1) // WAVES, blocks(name))
    items, counts, members = plan if plan else (None, None, None)
    M.launch(name, grid, WAVES * 32, [x, ("i", x_stride), ("i", slots), w, s, ("i", k), ("i", n), items, counts,
                                      members, ("i", rows), out, ("i", n), ("f", 0.0), ("i", skip)])


def dense(x, w, s, k, n, gs, rows, f32=True, nt=1):
    out = torch.zeros(rows, n, dtype=torch.float32 if f32 else torch.bfloat16, device="cuda")
    launch(gs, nt, 1, 1, 0 if f32 else 3, x, k, 0, w, s, k, n, None, rows, out, (rows + 15) // 16)
    return out


def make_plan(picks, experts, tile):
    """experts.cu plan_kernel: members grouped by expert (pair order kept), items (expert, first, count) of `tile`."""
    picks = np.asarray(picks).reshape(-1)
    members = np.argsort(picks, kind="stable").astype(np.int32)
    cnt = np.bincount(picks, minlength=experts)
    off = np.concatenate([[0], np.cumsum(cnt)[:-1]])
    items = [(e, off[e] + tile * j, min(tile, cnt[e] - tile * j)) for e in range(experts) for j in range(-(-cnt[e] // tile))]
    return (dev(np.array(items, np.int32).reshape(-1)), dev(np.array([len(items), 0], np.int32)), dev(members)), len(items)


def max_items(pairs, experts, tile):
    return min(pairs, experts) + pairs // tile


# ---- checks -----------------------------------------------------------------------------------------------------

FAIL = []


def expect(ok, what):
    if not ok:
        FAIL.append(what)
        print("  FAIL", what)


def check_dense(rng, n, k_full, k0, k, gs, rows_list):
    qw, sc = gen(rng, n, k_full)
    q = codes(qw)
    wp, sp = pack(qw, sc, 0, n, k0, k, gs)
    w, s = dev(wp), dev(sp)
    most = max(rows_list)
    x = bf16_rows(rng, most, k)
    x[1::3] *= 2.0 ** 30   # rows of very different magnitudes share the WMMA tiles
    x[2::5] *= 2.0 ** -30
    xd = x.cuda()
    solo = torch.cat([dense(xd[r:r + 1], w, s, k, n, gs, 1) for r in range(most)]).cpu().numpy()
    ref, exact, mag = emulate(q, sc, k0, k, gs, x.float().double().numpy()[:8])
    u = ulps(solo[:8], ref)
    rel = np.abs(solo[:8].astype(np.float64) - exact) / np.maximum(mag, 1e-30)
    expect((rel <= 1e-4).all(), f"dense n {n} k {k}: {(rel > 1e-4).sum()} outputs past 1e-4 of the fp64 sum")
    worst = np.unravel_index(np.argmax(rel), rel.shape)
    col_worst = rel.max(0)
    for rows in rows_list:
        perm = rng.permutation(most)[:rows]
        xs = xd[torch.from_numpy(perm).cuda()].contiguous()
        for nt in NTS:
            if n % (16 * nt):
                continue
            got = dense(xs, w, s, k, n, gs, rows, nt=nt).cpu().numpy()
            expect(np.array_equal(got.view(np.uint32), solo[perm].view(np.uint32)),
                   f"dense n {n} k {k} rows {rows} NT {nt}: {(got.view(np.uint32) != solo[perm].view(np.uint32)).sum()} outputs differ from 1-row calls")
    hb = dense(xd[:17], w, s, k, n, gs, 17, f32=False).cpu().view(torch.int16).numpy()
    expect(np.array_equal(hb, f32_to_bf16_bits(solo[:17])), "dense bf16 out is not the fp32 out rounded")
    again = torch.cat([dense(xd[:most], w, s, k, n, gs, most) for _ in range(2)]).cpu().numpy().view(np.uint32)
    expect(np.array_equal(again[:most], again[most:]), "dense: two runs differ")
    print(f"  dense n {n} k {k} (from {k0}) gs {gs}: rows {list(rows_list)} (permuted) NT 1/2/4 == 1-row calls, "
          f"deterministic; vs host order max {u.max()} ulps (median {np.median(u):.0f}); max |err|/sum|xw| "
          f"{rel.max():.2e} at row {worst[0]} col {worst[1]}, per-column worst median {np.median(col_worst):.2e}")


def check_order(rng, gs):
    """One nonzero product a group (inputs zero elsewhere: gfx11 WMMA sums are exact only for isolated products),
    cancelling group scales: the kernel must be bit-equal to acc = fmaf(sum, scale, acc) in group order."""
    n, k = 64, 1024
    kg = k // 128
    q = np.full((k, n), 8, np.int8)
    x = np.zeros(k, np.float32)
    sc = np.zeros((kg, n), np.float16)
    for g in range(kg):
        kin = g * 128 + rng.integers(128)
        x[kin] = (-1.0 if g % 3 == 0 else 1.0) * 2.0 ** (g % 7 - 3)
        for col in range(n):
            q[kin, col] = 15 if rng.random() < 0.5 else 1
            sc[g, col] = np.ldexp(1.0 + (col % 7) / 8.0, 12 if g < 2 else -12 + g)
        if g == 1:   # group 1 cancels group 0
            k0 = np.nonzero(x[:128])[0][0]
            x[kin] = -x[k0]
            q[kin] = q[k0]
            sc[1] = sc[0]
    qw = np.zeros((k // 8, n), np.uint32)
    for i in range(8):
        qw |= q[i::8].astype(np.uint32) << np.uint32(4 * i)
    wp, sp = pack(qw, sc, 0, n, 0, k, gs)
    got = dense(dev(x).bfloat16().reshape(1, k).contiguous(), dev(wp), dev(sp), k, n, gs, 1).cpu().numpy()
    ref, _, _ = emulate(q, sc, 0, k, gs, x.astype(np.float64)[None])
    bad = (got.view(np.uint32) != ref.view(np.uint32)).sum()
    expect(bad == 0, f"group order gs {gs}: {bad} columns")
    print(f"  group order (gs {gs}, cancelling groups, exact group sums): {n - bad} of {n} columns bit-equal "
          f"({(ref != 0).sum()} nonzero)")


def swiglu_host(g, u):
    gv = torch.from_numpy(g).bfloat16().float()
    uv = torch.from_numpy(u).bfloat16().float()
    return ((gv / (1 + torch.exp(-gv))).bfloat16().float() * uv).bfloat16().view(torch.int16).numpy()


def check_experts(rng, E, ni, full_ni, lo, D, top, R):
    slots = top + 1
    gs_down = 64 if lo % 128 or ni % 128 else 128
    uw, us, dw, ds = [], [], [], []
    for e in range(E):
        for _ in range(2):
            qw, sc = gen(rng, full_ni, D)
            a, b = pack(qw, sc, lo, ni, 0, D, 128)
            uw.append(a), us.append(b)
        qw, sc = gen(rng, D, full_ni)
        a, b = pack(qw, sc, 0, D, lo, ni, gs_down)
        dw.append(a), ds.append(b)
    up, up_s, dn, dn_s = dev(np.concatenate(uw)), dev(np.concatenate(us)), dev(np.concatenate(dw)), dev(np.concatenate(ds))
    picks = np.zeros((R, slots), np.int32)
    for r in range(R):
        ch = []
        while len(ch) < top:
            e = rng.integers(2) if rng.integers(4) == 0 else rng.integers(E)
            if e not in ch:
                ch.append(e)
        picks[r, :top] = ch
        picks[r, top] = E
    pairs = R * slots
    x = bf16_rows(rng, R, D).cuda()
    res = {}
    for tile, mt in ((16, 1), (64, 4), (64, 1), (16, 4)):
        plan, _ = make_plan(picks, E + 1, tile)
        mi = max_items(pairs, E + 1, tile)
        for nt in NTS:
            act = torch.zeros(pairs, ni, dtype=torch.bfloat16, device="cuda")
            if ni % (16 * nt) == 0:
                launch(128, nt, mt, 2, 2, x, D, slots, up, up_s, D, ni, plan, 0, act, mi, E)
            y = torch.full((pairs, D), float("nan"), device="cuda")
            y.view(torch.int32).fill_(0x5A5A5A5A)
            yb = torch.zeros(pairs, D, dtype=torch.bfloat16, device="cuda")
            a_in = res[(16, 1, 1)][0] if (16, 1, 1) in res else act
            launch(gs_down, nt, mt, 1, 0, a_in, ni, 0, dn, dn_s, ni, D, plan, 0, y, mi, E)
            launch(gs_down, nt, mt, 1, 3, a_in, ni, 0, dn, dn_s, ni, D, plan, 0, yb, mi, E)
            res[(tile, mt, nt)] = (act, y, yb, ni % (16 * nt) == 0)
    ha, hy, hyb, _ = [t.cpu() if isinstance(t, torch.Tensor) else t for t in res[(16, 1, 1)]]
    ours = picks.reshape(-1) < E
    for key, (a, y, yb, did) in res.items():
        if did:
            expect(torch.equal(a.cpu()[ours], ha[ours]), f"experts gate/up {key} differs from tile 16 MT 1 NT 1")
        expect(torch.equal(y.cpu()[ours].view(torch.int32), hy[ours].view(torch.int32)), f"experts down {key} differs")
        expect(torch.equal(yb.cpu()[ours], hyb[ours]), f"experts down bf16 {key} differs")
        expect(bool((y.cpu()[~ours].view(torch.int32) == 0x5A5A5A5A).all()), f"experts {key}: skipped slot written")
    # each pair against the dense kernel on its expert's matrices (a sample) and the host SwiGLU
    upw, ups = int(ni * D // 8), ni * (D // 128)
    dnw, dns = D * ni // 8, D * (ni // gs_down)
    swiglu_ulp = 0
    checked = 0
    for p in range(pairs):
        e, row = picks.reshape(-1)[p], p // slots
        if e == E or (checked >= 48 and p % 7):
            continue
        checked += 1
        g = dense(x[row:row + 1], up[(2 * e) * upw:], up_s[(2 * e) * ups:], D, ni, 128, 1).cpu().numpy()[0]
        u = dense(x[row:row + 1], up[(2 * e + 1) * upw:], up_s[(2 * e + 1) * ups:], D, ni, 128, 1).cpu().numpy()[0]
        want = swiglu_host(g, u)
        got = ha[p].view(torch.int16).numpy()
        d = np.abs(got.astype(np.int32) - want.astype(np.int32))
        expect(d.max() <= 1, f"experts gate/up pair {p}: {d.max()} bf16 ulps from host SwiGLU")
        swiglu_ulp += int((d == 1).sum())
        yd = dense(res[(16, 1, 1)][0][p:p + 1], dn[e * dnw:], dn_s[e * dns:], ni, D, gs_down, 1).cpu()
        expect(torch.equal(yd[0].view(torch.int32), hy[p].view(torch.int32)), f"experts down pair {p} != dense")
    print(f"  experts E {E} width {ni} (of {full_ni}, from {lo}) D {D} top {top} rows {R}: plan == dense on {checked} "
          f"pairs; tile 16/64, MT 1/4, NT 1/2/4 equal; skipped slot untouched; SwiGLU {swiglu_ulp} outputs 1 bf16 ulp "
          f"from host")


# ---- the checkpoint ---------------------------------------------------------------------------------------------

def st_tensor(model, name):
    idx = json.load(open(f"{model}/model.safetensors.index.json"))["weight_map"]
    path = f"{model}/{idx[name]}"
    with open(path, "rb") as f:
        hl = struct.unpack("<Q", f.read(8))[0]
        meta = json.loads(f.read(hl))[name]
    dt = {"I32": np.uint32, "F16": np.float16}[meta["dtype"]]
    a, b = meta["data_offsets"]
    return np.fromfile(path, dtype=dt, count=(b - a) // np.dtype(dt).itemsize, offset=8 + hl + a).reshape(meta["shape"])


def check_model(rng, model, layer, experts):
    for e in experts:
        for proj in ("gate_proj", "up_proj", "down_proj"):
            base = f"model.language_model.layers.{layer}.mlp.experts.{e}.{proj}"
            try:
                qw, sc, qz = (st_tensor(model, f"{base}.{t}") for t in ("qweight", "scales", "qzeros"))
            except KeyError:
                base = base.replace("model.language_model.", "model.")
                qw, sc, qz = (st_tensor(model, f"{base}.{t}") for t in ("qweight", "scales", "qzeros"))
            expect(bool((qz == 0x77777777).all()), f"{base}: zero points other than 8")
            k, n = qw.shape[0] * 8, qw.shape[1]
            q = codes(qw)
            for gs, k0, kk in ((128, 0, k), (64, k // 2, k // 2)) if proj == "down_proj" else ((128, 0, k),):
                wp, sp = pack(qw, sc, 0, n, k0, kk, gs)
                x = bf16_rows(rng, 4, kk)
                got = dense(x.cuda(), dev(wp), dev(sp), kk, n, gs, 4).cpu().numpy()
                ref, exact, mag = emulate(q, sc, k0, kk, gs, x.double().numpy())
                rel = np.abs(got - exact) / np.maximum(mag, 1e-30)
                expect((rel <= 1e-4).all(), f"{base} gs {gs}: past 1e-4")
                print(f"  {base} [{n}, {kk} from {k0}] gs {gs}: max {ulps(got, ref).max()} ulps vs host order, "
                      f"max |err|/sum|xw| {rel.max():.2e}, worst |err| {np.abs(got - exact).max():.3e} "
                      f"(|y| max {np.abs(exact).max():.3f})")


def main():
    global M
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="/home/dzannotti/models/qwen38fn-int4-autoround")
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    M = load()
    rng = np.random.default_rng(1234)
    rows = [1, 2, 3, 7, 8, 15, 16, 17, 31, 33, 64, 100, 128]
    check_dense(rng, 4096, 2560, 0, 2560, 128, rows)
    check_dense(rng, 640, 2560, 0, 2560, 128, rows)
    check_dense(rng, 2560, 640, 0, 640, 128, rows)
    check_dense(rng, 2560, 640, 320, 320, 64, rows)
    check_dense(rng, 2560, 640, 0, 320, 64, rows)
    check_order(rng, 128)
    check_order(rng, 64)
    check_experts(rng, 24, 640, 640, 0, 2560, 5, 37)
    check_experts(rng, 24, 320, 640, 320, 2560, 5, 37)
    check_experts(rng, 8, 640, 640, 0, 2560, 5, 130)
    if not a.quick:
        check_experts(rng, 24, 320, 640, 0, 2560, 5, 3)
        check_experts(rng, 64, 320, 640, 320, 2560, 5, 700)
        check_experts(rng, 512, 640, 640, 0, 2560, 5, 300)
    if os.path.exists(f"{a.model}/model.safetensors.index.json"):
        check_model(rng, a.model, 0, (0, 511))
    print("int4-check (HIP):", "FAILED " + "; ".join(FAIL) if FAIL else "ok")
    return 1 if FAIL else 0


if __name__ == "__main__":
    sys.exit(main())
