# SPDX-License-Identifier: Apache-2.0
"""python3 gen.py [SPEC,...] > mxfp8_gemv.loom: the MXFP8 x bf16 decode GEMV's variants (VARIANTS by default)."""
import sys

from gen_epilogue import repair, row_index, scale_expr, sides, store

SPEC = """// A variant is R:K:L:CB:WAVES[:G[:P]][:sKS][:bf16|:x4] (its name spells each field): R rows at most (m <= R at
// launch), L lanes a column, KS waves split one column's lanes (lane l = ks * (L / KS) + j), CB columns a lane, G rounds
// a loop trip (G = T: straight line), P column blocks a wave with the next block's weights prefetched. bf16 rounds the
// output to bf16 (nearest even, any NaN to 0x7fc0); x4 also splits it at columns c1 <= c2 <= c3 into four bf16 outputs
// [m][c1], [m][c2 - c1], [m][c3 - c2] and [m][n - c3]. Weights e4m3 [n][k] bytes, e8m0 scales [n][k / 32], x bf16 [m][k].
"""

ORDER = """// The sum order of output (row r, column c) follows K and L alone, never R, CB, WAVES, KS, G, P or m:
//   T = K / 32 / L rounds (L divides K / 32). Lane l = 0..L-1 owns chunks q = ((t + rot) mod T) * L + l in order of
//   t = 0..T-1, rot = (c' / 8) mod T with c' the column within its output (x4: within its side; cuts multiples of 8):
//     d = +0; for p = 0..15: d = dot2(w[32q + 2p], w[32q + 2p + 1]; x[r][32q + 2p], x[r][32q + 2p + 1]; d)
//     acc_l = fma(d, 2^(E[c][q] - 119), acc_l), from acc_l = +0
//   lanes add by xor tree: for o = L/2, L/4, ..., 1: acc_l = acc_l + acc_(l ^ o); the sum is lane 0's.
//   A non-finite sum when some x[r][i] is non-finite is replaced by the f64 sum (w_i * x_i in i order, each
//   product and sum rounded to f64, no fma), rounded once to f32. Weights enter dot2 as bf16 e4m3 x 2^-8 (exact).
"""


class B:
    def __init__(self):
        self.lines = []
        self.n = 0
        self.ind = 1

    def t(self, p="v"):
        self.n += 1
        return f"%{p}{self.n}"

    def e(self, s):
        self.lines.append("  " * self.ind + s)

    def ci(self, v, ty="index"):
        r = self.t("ci")
        self.e(f"{r} = index.constant {v} : index" if ty == "index" else f"{r} = scalar.constant {v} : {ty}")
        return r

    def open(self, s):
        self.e(s)
        self.ind += 1

    def close(self):
        self.ind -= 1
        self.e("}")


def decode(b, words):
    """32 bf16 weights (e4m3 x 2^-8) of one chunk from its 8 code words, in code order."""
    e = b.e
    fs = []
    for par in range(2):
        m, s = b.t("m"), b.t("s")
        if par == 0:
            e(f"{m} = vector.shli {words}, %s7v : vector<8xi32>")
            e(f"{s} = vector.shli {words}, %s8v : vector<8xi32>")
        else:
            e(f"{m} = vector.shrui {words}, %s1v : vector<8xi32>")
            s = words
        ma, sa, hb, hf, f, fb = (b.t(p) for p in ("ma", "sa", "hb", "hf", "f", "fb"))
        e(f"{ma} = vector.andi {m}, %mmag : vector<8xi32>")
        e(f"{sa} = vector.andi {s}, %msgn : vector<8xi32>")
        e(f"{hb} = vector.ori {ma}, {sa} : vector<8xi32>")
        e(f"{hf} = vector.bitcast {hb} : vector<8xi32> to vector<16xf16>")
        e(f"{f} = vector.extf {hf} : vector<16xf16> to vector<16xf32>")
        e(f"{fb} = vector.bitcast {f} : vector<16xf32> to vector<32xbf16>")
        fs.append(fb)
    cat = b.t("cat")
    e(f"{cat} = vector.concat<0> {fs[0]}, {fs[1]} : vector<32xbf16>, vector<32xbf16> -> vector<64xbf16>")
    lanes = []
    for p in range(16):
        lanes += [2 * p + 1, 32 + 2 * p + 1]
    lanes += list(range(32, 64))
    sf, wb = b.t("sf"), b.t("wb")
    e(f"{sf} = vector.shuffle<[{', '.join(map(str, lanes))}]> {cat} : vector<64xbf16>")
    e(f"{wb} = vector.slice {sf}[0] : vector<64xbf16> -> vector<32xbf16>")
    return wb


class V:
    """One variant's geometry."""

    def __init__(self, spec):
        p = spec.split(":")
        self.KS, self.OT, self.SIDES = 1, "f32", 1
        while p[-1][0].isalpha():
            t = p.pop()
            if t == "bf16":
                self.OT = "bf16"
            elif t == "x4":
                self.OT, self.SIDES = "bf16", 4
            else:
                self.KS = int(t[1:])
        self.R, self.K, self.L, self.CB, self.WAVES = map(int, p[:5])
        self.NCH = self.K // 32
        self.LPC = self.L // self.KS
        self.T = self.NCH // self.L
        self.G = int(p[5]) if len(p) > 5 and int(p[5]) else self.T
        self.P = int(p[6]) if len(p) > 6 else 1
        assert self.K % 128 == 0 and self.NCH % self.L == 0 and self.L <= 32 and self.L % self.KS == 0
        assert self.WAVES % self.KS == 0 and self.T % self.G == 0 and (self.P == 1 or self.G == self.T)
        self.COLS_WG = (32 // self.LPC) * self.CB * (self.WAVES // self.KS)

    @property
    def name(self):
        v = self
        s = f"mxfp8_gemv_r{v.R}_k{v.K}_l{v.L}_s{v.KS}_c{v.CB}_w{v.WAVES}_g{v.G}_p{v.P}"
        return s + ("_x4" if v.SIDES == 4 else "_bf16" if v.OT == "bf16" else "")


def header(b, v):
    out = [f'kernel.def export("{v.name}") @{v.name}(%cols: index) {{']
    per = v.COLS_WG * v.P
    out.append("  %one = index.constant 1 : index")
    out.append(f"  %per = index.constant {per} : index")
    out.append(f"  %round_up = index.constant {per - 1} : index")
    out.append("  %padded = index.add %cols, %round_up : index")
    out.append("  %groups = index.div %padded, %per : index")
    out.append(f"  %threads = index.constant {32 * v.WAVES} : index")
    out.append("  kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%threads, %one, %one) : index")
    e = b.e
    if v.SIDES == 1:
        out.append("} launch(%m_arg: i32, %n_arg: i32, %w_arg: buffer, %e_arg: buffer, %x_arg: buffer, %o_arg: buffer) {")
        e("%w, %es, %x, %o = buffer.assume.noalias %w_arg, %e_arg, %x_arg, %o_arg : buffer, buffer, buffer, buffer")
    else:
        out.append("} launch(%m_arg: i32, %n_arg: i32, %c1_arg: i32, %c2_arg: i32, %c3_arg: i32, %w_arg: buffer, "
                   "%e_arg: buffer, %x_arg: buffer, %o_arg: buffer, %o1_arg: buffer, %o2_arg: buffer, %o3_arg: buffer) {")
        e("%w, %es, %x, %o, %o1, %o2, %o3 = buffer.assume.noalias %w_arg, %e_arg, %x_arg, %o_arg, %o1_arg, %o2_arg, "
          "%o3_arg : buffer, buffer, buffer, buffer, buffer, buffer, buffer")
    e("%m_raw = index.cast %m_arg : i32 to index")
    e(f"%m = index.assume %m_raw [range(%m_raw, 1, {v.R})] : index")
    e("%n_raw = index.cast %n_arg : i32 to index")
    e("%n = index.assume %n_raw [range(%n_raw, 1, 65536)] : index")
    e("%origin = index.constant 0 : offset")
    e(f"%wv = buffer.view %w[%origin] : buffer -> view<[%n]x{v.K // 4}xi32>")
    e(f"%wbv = buffer.view %w[%origin] : buffer -> view<[%n]x{v.K}xi8>")
    e(f"%ev = buffer.view %es[%origin] : buffer -> view<[%n]x{v.NCH}xi8>")
    e(f"%xv = buffer.view %x[%origin] : buffer -> view<[%m]x{v.K}xbf16>")
    if v.SIDES == 1:
        e(f"%ov = buffer.view %o[%origin] : buffer -> view<[%m]x[%n]x{v.OT}>")
    else:
        sides(b)
    e("%wg = kernel.workgroup.id<x> : index")
    e("%wave = kernel.subgroup.id : index")
    e("%lane = kernel.subgroup.lane.id : index")
    for name, val in (("ks", v.KS), ("cgs", v.WAVES // v.KS), ("lpc", v.LPC), ("slots", 32 // v.LPC),
                      ("cb", v.CB), ("L", v.L), ("nch", v.NCH), ("nchm1", v.NCH - 1)):
        e(f"%c_{name} = index.constant {val} : index")
    e("%ks = index.rem %wave, %c_ks : index")
    e("%cg = index.div %wave, %c_ks : index")
    e("%j = index.rem %lane, %c_lpc : index")
    e("%slot = index.div %lane, %c_lpc : index")
    e("%jj = index.madd %ks, %c_lpc, %j : index")
    e("%c1 = index.constant 1 : index")
    e("%c0 = index.constant 0 : index")
    e("%nm1 = index.sub %n, %c1 : index")
    e("%mm1 = index.sub %m, %c1 : index")
    e("%c8 = index.constant 8 : index")
    e("%c32 = index.constant 32 : index")
    e("%zf = scalar.constant 0.0 : f32")
    e("%zv = vector.splatf 0.0 : vector<1xf32>")
    for name, val, ty in (("s8v", 8, 8), ("s7v", 7, 8), ("s1v", 1, 8), ("mmag", 1065369472, 8), ("msgn", -2147450880, 8)):
        e(f"%{name} = vector.splati {val} : vector<{ty}xi32>")
    e("%c8_32 = scalar.constant 8 : i32")
    e("%c23_32 = scalar.constant 23 : i32")
    e("%lead = index.cmp eq, %jj, %c0 : index")
    return out


def columns(b, col0, CB):
    """(raw, clamped) column per lane column; clamped columns stay loadable past n."""
    e = b.e
    cols = []
    for c in range(CB):
        raw, cm, col = b.t("colraw"), b.t("colmin"), b.t("col")
        e(f"{raw} = index.add {col0}, {b.ci(c)} : index")
        e(f"{cm} = index.min {raw}, %nm1 : index")
        e(f"{col} = index.assume {cm} [range({cm}, 0, %nm1)] : index")
        cols.append((raw, col))
    return cols


def chunk(b, tt, rot):
    """Chunk of round tt for this lane: round tt + rot of the column's rotated order."""
    q = b.t("q")
    if rot is None:
        b.e(f"{q} = index.madd {tt}, %c_L, %jj : index")
        return q
    a, wrap, less, sel, tr = b.t("rt"), b.t("wrap"), b.t("less"), b.t("sel"), b.t("tr")
    b.e(f"{a} = index.add {tt}, {rot} : index")
    b.e(f"{wrap} = index.cmp uge, {a}, %c_T : index")
    b.e(f"{less} = index.sub {a}, %c_T : index")
    b.e(f"{sel} = scf.select {wrap}, {less}, {a} : index")
    b.e(f"{tr} = index.assume {sel} [range({sel}, 0, %c_Tm1)] : index")
    b.e(f"{q} = index.madd {tr}, %c_L, %jj : index")
    return q


def rotation(b, v, col0):
    """(c' / 8) mod T for the lane's first column c' (x4: from its side's first column), or None when T is 1."""
    if v.T == 1:
        return None
    e = b.e
    e(f"%c_T = index.constant {v.T} : index")
    e(f"%c_Tm1 = index.constant {v.T - 1} : index")
    local = col0
    if v.SIDES == 4:
        base = "%c0"
        for s in (1, 2, 3):
            ge, nb = b.t("ge"), b.t("base")
            e(f"{ge} = index.cmp uge, {col0}, %cut{s} : index")
            e(f"{nb} = scf.select {ge}, %cut{s}, {base} : index")
            base = nb
        diff, local = b.t("diff"), b.t("local")
        e(f"{diff} = index.sub {col0}, {base} : index")
        e(f"{local} = index.assume {diff} [range({diff}, 0, 65535)] : index")
    grp, rem, rot = b.t("grp"), b.t("rem"), b.t("rot")
    e(f"{grp} = index.div {local}, %c8 : index")
    e(f"{rem} = index.rem {grp}, %c_T : index")
    e(f"{rot} = index.assume {rem} [range({rem}, 0, %c_Tm1)] : index")
    return rot


def load(b, v, col, q):
    e = b.e
    k0w, words, eb = b.t("k0w"), b.t("words"), b.t("eb")
    e(f"{k0w} = index.mul {q}, %c8 : index")
    e(f"{words} = vector.load %wv[{col}, {k0w}] : view<[%n]x{v.K // 4}xi32> -> vector<8xi32>")
    e(f"{eb} = view.load %ev[{col}, {q}] : view<[%n]x{v.NCH}xi8> -> i8")
    return words, eb


def scale(b, eb):
    sc = b.t("sc")
    expr = scale_expr(b, eb)
    b.e(f"{sc} = {expr}")
    return sc


def x_slices(b, v, q):
    """x[r][32q .. 32q + 31] as 16 bf16 pairs a row."""
    e = b.e
    k0 = b.t("k0")
    e(f"{k0} = index.mul {q}, %c32 : index")
    rows = []
    for r in range(v.R):
        xb = b.t("xb")
        e(f"{xb} = vector.load %xv[{row_index(b, r)}, {k0}] : view<[%m]x{v.K}xbf16> -> vector<32xbf16>")
        xs = []
        for p in range(16):
            xp = b.t("xp")
            e(f"{xp} = vector.slice {xb}[{2 * p}] : vector<32xbf16> -> vector<2xbf16>")
            xs.append(xp)
        rows.append(xs)
    return rows


def contract(b, v, q, raws, cur, xrows=None):
    """One round: decode CB chunks, then a row's 16-dot2 chain a column and acc = fma(d, scale, acc)."""
    e = b.e
    scs = [scale(b, eb) for _, eb in raws]
    wbs = [decode(b, words) for words, _ in raws]
    xrows = xrows or x_slices(b, v, q)
    for r in range(v.R):
        for c in range(v.CB):
            d = "%zv"
            for p in range(16):
                wp, nd = b.t("wp"), b.t("d")
                e(f"{wp} = vector.slice {wbs[c]}[{2 * p}] : vector<32xbf16> -> vector<2xbf16>")
                e(f"{nd} = vector.dot2f {wp}, {xrows[r][p]}, {d} : vector<2xbf16>, vector<2xbf16>, vector<1xf32>")
                d = nd
            ds, na = b.t("ds"), b.t("na")
            e(f"{ds} = vector.extract {d}[0] : vector<1xf32> -> f32")
            i = r * v.CB + c
            e(f"{na} = scalar.fmaf {ds}, {scs[c]}, {cur[i]} : f32")
            cur[i] = na


def xor_tree(b, vals, top):
    e = b.e
    width = b.t("width")
    e(f"{width} = scalar.constant 32 : i32")
    out = []
    for val in vals:
        off = top // 2
        while off >= 1:
            oc, sh, ok, nv = b.t("off"), b.t("sh"), b.t("ok"), b.t("red")
            e(f"{oc} = scalar.constant {off} : i32")
            e(f"{sh}, {ok} = kernel.subgroup.shuffle<xor> {val}, {oc}, {width} : f32, i32, i32")
            e(f"{nv} = scalar.addf {val}, {sh} : f32")
            val = nv
            off //= 2
        out.append(val)
    return out


def lds_alloc(b, v):
    S, CG, NV = 32 // v.LPC, v.WAVES // v.KS, v.R * v.CB
    b.e(f"%lds_bytes = index.constant {CG * S * v.LPC * v.KS * NV * 4} : offset")
    b.e("%lds = buffer.alloca<workgroup> align(16) %lds_bytes : buffer")
    b.e(f"%lv = buffer.view %lds[%origin] : buffer -> view<{CG}x{S}x{v.LPC}x{v.KS}x{NV}xf32>")


def ks_tree(b, v, cur):
    """Every lane's partials to LDS, then wave ks = 0 folds the KS values of its lane j as the xor tree's top levels."""
    e = b.e
    S, CG, NV = 32 // v.LPC, v.WAVES // v.KS, v.R * v.CB
    ty = f"view<{CG}x{S}x{v.LPC}x{v.KS}x{NV}xf32>"
    for i, val in enumerate(cur):
        e(f"view.store {val}, %lv[%cg, %slot, %j, %ks, {b.ci(i)}] : f32, {ty}")
    e("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    out = []
    for i in range(NV):
        ci = b.ci(i)
        part = []
        for s in range(v.KS):
            pv = b.t("pv")
            e(f"{pv} = view.load %lv[%cg, %slot, %j, {b.ci(s)}, {ci}] : {ty} -> f32")
            part.append(pv)
        span = v.KS // 2
        while span >= 1:
            nxt = []
            for s in range(span):
                na = b.t("kt")
                e(f"{na} = scalar.addf {part[s]}, {part[s + span]} : f32")
                nxt.append(na)
            part = nxt
            span //= 2
        out.append(part[0])
    return out


def finish(b, v, cur, cols):
    """Cross-wave levels (KS > 1), then the in-wave xor tree, then lane 0 repairs and stores rows below m."""
    if v.KS > 1:
        cur = ks_tree(b, v, cur)
    fin = xor_tree(b, cur, v.LPC)
    e = b.e
    for c, (raw, col) in enumerate(cols):
        inb, go = b.t("inb"), b.t("go")
        e(f"{inb} = index.cmp ult, {raw}, %n : index")
        e(f"{go} = scf.select %lead, {inb}, %lead : i1")
        b.open(f"scf.if {go} {{")
        for r in range(v.R):
            if r > 0:
                live = b.t("live")
                e(f"{live} = index.cmp ult, {b.ci(r)}, %m : index")
                b.open(f"scf.if {live} {{")
            val = repair(b, v, fin[r * v.CB + c], r, col)
            store(b, v, val, b.ci(r), raw)
            if r > 0:
                b.close()
        b.close()


def col0_of(b, blk):
    e = b.e
    gw, gs, c0 = b.t("gw"), b.t("gs"), b.t("col0")
    e(f"{gw} = index.madd {blk}, %c_cgs, %cg : index")
    e(f"{gs} = index.madd {gw}, %c_slots, %slot : index")
    e(f"{c0} = index.mul {gs}, %c_cb : index")
    return c0


def kernel(spec):
    v = V(spec)
    b = B()
    out = header(b, v)
    if v.KS > 1:
        lds_alloc(b, v)
    if v.P > 1:
        body_p(b, v)
    else:
        body(b, v)
    b.e("kernel.return")
    return "\n".join(out + b.lines + ["}"]) + "\n"


def body(b, v):
    e = b.e
    col0 = col0_of(b, "%wg")
    cols = columns(b, col0, v.CB)
    rot = rotation(b, v, col0)
    flat = [f"%acc_{i}" for i in range(v.R * v.CB)]
    types = ", ".join("f32" for _ in flat)
    loop = v.G < v.T
    if loop:
        res = [f"%fin_{i}" for i in range(len(flat))]
        inits = ", ".join(f"{a} = %zf : f32" for a in flat)
        e(f"%c_trips = index.constant {v.T // v.G} : index")
        b.open(f"{', '.join(res)} = scf.for %trip = [%c0 to %c_trips step %c1]({inits}) -> ({types}) {{")
        cur = list(flat)
    else:
        cur = ["%zf"] * len(flat)
    for g in range(v.G):
        if loop:
            tt = b.t("tt")
            e(f"{tt} = index.madd %trip, {b.ci(v.G)}, {b.ci(g)} : index")
        else:
            tt = b.ci(g)
        q = chunk(b, tt, rot)
        contract(b, v, q, [load(b, v, col, q) for _, col in cols], cur)
    if loop:
        e(f"scf.yield {', '.join(cur)} : {types}")
        b.close()
        cur = res
    finish(b, v, cur, cols)


def body_p(b, v):
    """Each wave runs P column blocks; block i + 1's weights load while block i contracts; x stays in registers."""
    e = b.e
    e(f"%c_p = index.constant {v.P} : index")
    e(f"%c_pm1 = index.constant {v.P - 1} : index")
    e("%wgp = index.mul %wg, %c_p : index")

    def block_col0(i):
        blk = b.t("blk")
        e(f"{blk} = index.add %wgp, {i} : index")
        return col0_of(b, blk)

    assert v.T == 1, "P column blocks prefetch one block's chunks for the next: rotation needs one round"
    qs = [chunk(b, b.ci(t), None) for t in range(v.T)]
    xpre = [x_slices(b, v, qs[t]) for t in range(v.T)]

    def loads(cols):
        return [[load(b, v, col, qs[t]) for _, col in cols] for t in range(v.T)]

    first = loads(columns(b, block_col0("%c0"), v.CB))
    carried = [x for rnd in first for pair in rnd for x in pair]
    ctypes = ["vector<8xi32>", "i8"] * (v.T * v.CB)
    args = [f"%cw_{i}" for i in range(len(carried))]
    inits = ", ".join(f"{a} = {x} : {t}" for a, x, t in zip(args, carried, ctypes))
    res = [f"%cr_{i}" for i in range(len(carried))]
    b.open(f"{', '.join(res)} = scf.for %it = [%c0 to %c_p step %c1]({inits}) -> ({', '.join(ctypes)}) {{")
    nx, nxm = b.t("nx"), b.t("nxm")
    e(f"{nx} = index.add %it, %c1 : index")
    e(f"{nxm} = index.min {nx}, %c_pm1 : index")
    nxt = loads(columns(b, block_col0(nxm), v.CB))
    cols = columns(b, block_col0("%it"), v.CB)
    cur = ["%zf"] * (v.R * v.CB)
    k = 0
    for t in range(v.T):
        raws = []
        for _ in range(v.CB):
            raws.append((args[k], args[k + 1]))
            k += 2
        contract(b, v, qs[t], raws, cur, xpre[t])
    finish(b, v, cur, cols)
    if v.KS > 1:
        e("kernel.barrier<workgroup> scope(workgroup) ordering(acq_rel)")
    flat_next = [x for rnd in nxt for pair in rnd for x in pair]
    e(f"scf.yield {', '.join(flat_next)} : {', '.join(ctypes)}")
    b.close()


# launches/mxfp8_gemv.zig's table: the stitched row groups' four-way output at K 2560, single products' fp32 and bf16
GROUPS = ["1:2560:16:1:4", "2:2560:16:1:4", "4:2560:16:1:4", "8:2560:16:1:4:1"]
SINGLES = ["1:2560:16:1:4:1:1:s2", "2:2560:16:1:4:1:1:s2", "4:2560:16:1:4:1:1:s2", "8:2560:16:1:4:1:1:s2",
           "1:6144:32:1:4:2:1:s2", "2:6144:32:1:4:2:1:s2", "4:6144:32:1:4:2:1:s2", "8:6144:32:1:4:2:1:s2",
           "1:640:4:1:4:1:1:s2", "2:640:4:1:4:1:1:s2", "4:640:4:1:4:1:1:s2", "8:640:4:1:4:1:1:s2"]
VARIANTS = [v + ":x4" for v in GROUPS] + [v + o for v in SINGLES for o in ("", ":bf16")]


def main():
    specs = sys.argv[1].split(",") if len(sys.argv) > 1 else VARIANTS
    src = ["// SPDX-License-Identifier: Apache-2.0",
           "// Generated by gen.py: MXFP8 decode GEMV, e4m3 [n][k] and e8m0 [n][k/32] x bf16 [m][k] -> [m][n].",
           "// Launch (m: i32, n: i32, w, e, x, out): grid ceil(n / columns a workgroup), 32 x WAVES threads; _x4:",
           "// (m: i32, n: i32, c1: i32, c2: i32, c3: i32, w, e, x, out0, out1, out2, out3).",
           SPEC, ORDER]
    for spec in specs:
        src.append(kernel(spec))
    sys.stdout.write("\n".join(src))


if __name__ == "__main__":
    main()
