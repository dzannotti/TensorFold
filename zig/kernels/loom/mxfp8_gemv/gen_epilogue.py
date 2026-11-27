# SPDX-License-Identifier: Apache-2.0
"""gen.py's outputs: lane 0's repair of a non-finite sum, the bf16 rounding, the four-way sides, row clamping."""


def row_index(b, r):
    """Row r of the tile, clamped to m - 1 so rows past m stay loadable."""
    if r == 0:
        return "%c0"
    cm, cr = b.t("rowmin"), b.t("row")
    b.e(f"{cm} = index.min {b.ci(r)}, %mm1 : index")
    b.e(f"{cr} = index.assume {cm} [range({cm}, 0, %mm1)] : index")
    return cr


def repair(b, v, val, r, col):
    """val, or the f64 sum of row r against column col when val is non-finite and some x of the row is."""
    e = b.e
    fin = b.t("fin")
    e(f"{fin} = scalar.isfinitef {val} : f32")
    res = b.t("rep")
    e(f"{res} = scf.if {fin} -> (f32) {{")
    b.ind += 1
    e(f"scf.yield {val} : f32")
    b.ind -= 1
    b.open("} else {")
    row = row_index(b, r)
    k = b.ci(v.K)
    y0, bad0 = b.t("y0"), b.t("bad0")
    e(f"{y0} = scalar.constant 0.0 : f64")
    e(f"{bad0} = scalar.constant false : i1")
    y, bad, i, ya, ba = b.t("y"), b.t("bad"), b.t("i"), b.t("ya"), b.t("ba")
    b.open(f"{y}, {bad} = scf.for {i} = [%c0 to {k} step %c1]({ya} = {y0} : f64, {ba} = {bad0} : i1) -> (f64, i1) {{")
    byte, eb, b32 = b.t("byte"), b.t("ebyte"), b.t("b32")
    e(f"{byte} = view.load %wbv[{col}, {i}] : view<[%n]x{v.K}xi8> -> i8")
    q = b.t("qi")
    e(f"{q} = index.div {i}, %c32 : index")
    e(f"{eb} = view.load %ev[{col}, {q}] : view<[%n]x{v.NCH}xi8> -> i8")
    e(f"{b32} = scalar.extui {byte} : i8 to i32")
    mg, sg, mgs, sgs, hb, h16, hf, wf = (b.t(p) for p in ("mg", "sg", "mgs", "sgs", "hb", "h16", "hf", "wf"))
    e(f"{mg} = scalar.andi {b32}, {b.ci(127, 'i32')} : i32")
    e(f"{mgs} = scalar.shli {mg}, {b.ci(7, 'i32')} : i32")
    e(f"{sg} = scalar.andi {b32}, {b.ci(128, 'i32')} : i32")
    e(f"{sgs} = scalar.shli {sg}, %c8_32 : i32")
    e(f"{hb} = scalar.ori {mgs}, {sgs} : i32")
    e(f"{h16} = scalar.trunci {hb} : i32 to i16")
    e(f"{hf} = scalar.bitcast {h16} : i16 to f16")
    e(f"{wf} = scalar.extf {hf} : f16 to f64")
    sc, scd, wd, xb, xd, pr, ny, fx, nb = (b.t(p) for p in ("sc", "scd", "wd", "xb", "xd", "pr", "ny", "fx", "nb"))
    expr = scale_expr(b, eb)
    e(f"{sc} = {expr}")
    e(f"{scd} = scalar.extf {sc} : f32 to f64")
    e(f"{wd} = scalar.mulf {wf}, {scd} : f64")
    e(f"{xb} = view.load %xv[{row}, {i}] : view<[%m]x{v.K}xbf16> -> bf16")
    e(f"{xd} = scalar.extf {xb} : bf16 to f64")
    e(f"{pr} = scalar.mulf {wd}, {xd} : f64")
    e(f"{ny} = scalar.addf {ya}, {pr} : f64")
    e(f"{fx} = scalar.isfinitef {xd} : f64")
    e(f"{nb} = scf.select {fx}, {ba}, {b.ci('true', 'i1')} : i1")
    e(f"scf.yield {ny}, {nb} : f64, i1")
    b.close()
    yf, out = b.t("yf"), b.t("out")
    e(f"{yf} = scalar.fptrunc {y} : f64 to f32")
    e(f"{out} = scf.select {bad}, {yf}, {val} : f32")
    e(f"scf.yield {out} : f32")
    b.close()
    return res


def scale_expr(b, eb):
    """2^(e - 119) of an e8m0 byte: its operands are emitted, the final bitcast returned for the caller to name."""
    e = b.e
    e32, ep, bits = b.t("e32"), b.t("ep"), b.t("ebits")
    e(f"{e32} = scalar.extui {eb} : i8 to i32")
    e(f"{ep} = scalar.addi {e32}, %c8_32 : i32")
    e(f"{bits} = scalar.shli {ep}, %c23_32 : i32")
    return f"scalar.bitcast {bits} : i32 to f32"


def sides(b):
    """The four outputs' views: side s holds columns [c_s, c_(s+1)) of the product, c0 = 0 and c4 = n."""
    e = b.e
    e("%c0s = index.constant 0 : index")
    cuts = ["%c0s"]
    for s in (1, 2, 3):
        raw = b.t("craw")
        e(f"{raw} = index.cast %c{s}_arg : i32 to index")
        e(f"%cut{s} = index.assume {raw} [range({raw}, {cuts[-1]}, %n)] : index")
        cuts.append(f"%cut{s}")
    cuts.append("%n")
    for s in range(4):
        e(f"%n{s} = index.sub {cuts[s + 1]}, {cuts[s]} : index")
        e(f"%ov{s} = buffer.view %o{s if s else ''}[%origin] : buffer -> view<[%m]x[%n{s}]xbf16>")


def store(b, v, val, row, col):
    e = b.e
    if v.SIDES == 4:
        store_sides(b, v, val, row, col, 0)
        return
    ty = f"view<[%m]x[%n]x{v.OT}>"
    if v.OT == "f32":
        e(f"view.store {val}, %ov[{row}, {col}] : f32, {ty}")
        return
    bf = bf16_of(b, val)
    e(f"view.store {bf}, %ov[{row}, {col}] : bf16, {ty}")


def store_sides(b, v, val, row, col, s):
    """Column col of side s or a later one: each side's view takes its columns from its first."""
    e = b.e
    if s == 3:
        put_side(b, val, row, col, s)
        return
    inside = b.t("inside")
    e(f"{inside} = index.cmp ult, {col}, %cut{s + 1} : index")
    b.open(f"scf.if {inside} {{")
    put_side(b, val, row, col, s)
    b.ind -= 1
    b.open("} else {")
    store_sides(b, v, val, row, col, s + 1)
    b.close()


def put_side(b, val, row, col, s):
    e = b.e
    at = col
    if s > 0:
        rel, at = b.t("rel"), b.t("at")
        e(f"{rel} = index.sub {col}, %cut{s} : index")
        e(f"{at} = index.assume {rel} [range({rel}, 0, %n{s})] : index")
    bf = bf16_of(b, val)
    e(f"view.store {bf}, %ov{s}[{row}, {at}] : bf16, view<[%m]x[%n{s}]xbf16>")


def bf16_of(b, val):
    """val rounded to bf16 as the stream tile's epilogue does: nearest even, any NaN to 0x7fc0."""
    e = b.e
    u, hi, lsb, t1, t2, rn, nan, bits, h, bf = (b.t(p) for p in ("u", "hi", "lsb", "t1", "t2", "rn", "nan", "bits", "h", "bf"))
    c16 = b.ci(16, "i32")
    e(f"{u} = scalar.bitcast {val} : f32 to i32")
    e(f"{hi} = scalar.shrui {u}, {c16} : i32")
    e(f"{lsb} = scalar.andi {hi}, {b.ci(1, 'i32')} : i32")
    e(f"{t1} = scalar.addi {u}, {lsb} : i32")
    e(f"{t2} = scalar.addi {t1}, {b.ci(32767, 'i32')} : i32")
    e(f"{rn} = scalar.shrui {t2}, {c16} : i32")
    e(f"{nan} = scalar.isnanf {val} : f32")
    e(f"{bits} = scf.select {nan}, {b.ci(32704, 'i32')}, {rn} : i32")
    e(f"{h} = scalar.trunci {bits} : i32 to i16")
    e(f"{bf} = scalar.bitcast {h} : i16 to bf16")
    return bf
