# SPDX-License-Identifier: Apache-2.0
"""python3 gen_tests.py > tests.loom: loom-test cases and loom-benchmark records of gen.py's VARIANTS."""

HEADER = """// Run: loom-test tests.loom --library=mxfp8_gemv.loom --device=amdgpu (zig build loom-test does both).
// Each variant at m = R rows of a ragged n: every row bit-equal to the 1-row variant of its K and L on that row alone,
// and the product within rtol of an f64 reference kernel (positive inputs, so no cancellation); bf16 within its
// rounding; the four-way output's sides against the reference's columns. Benchmarks: the served shapes.
"""
import sys

import gen

SERVED = {2560: (16480, 13952, 1280), 6144: (2560,), 640: (2560,)}


def reference(K):
    """One work-item an output: the f64 sum of w * x in k order, rounded once to f32."""
    NCH = K // 32
    name = f"mxfp8_gemv_reference_k{K}"
    return name, f"""kernel.def @{name}(%count: index) {{
  %one = index.constant 1 : index
  %per = index.constant 64 : index
  %round_up = index.constant 63 : index
  %padded = index.add %count, %round_up : index
  %groups = index.div %padded, %per : index
  kernel.launch.config workgroups(%groups, %one, %one) workgroup_size(%per, %one, %one) : index
}} launch(%m_arg: i32, %n_arg: i32, %w_arg: buffer, %e_arg: buffer, %x_arg: buffer, %o_arg: buffer) {{
  %m_raw = index.cast %m_arg : i32 to index
  %m = index.assume %m_raw [range(%m_raw, 1, 8)] : index
  %n_raw = index.cast %n_arg : i32 to index
  %n = index.assume %n_raw [range(%n_raw, 1, 65536)] : index
  %origin = index.constant 0 : offset
  %wv = buffer.view %w_arg[%origin] : buffer -> view<[%n]x{K}xi8>
  %ev = buffer.view %e_arg[%origin] : buffer -> view<[%n]x{NCH}xi8>
  %xv = buffer.view %x_arg[%origin] : buffer -> view<[%m]x{K}xbf16>
  %ov = buffer.view %o_arg[%origin] : buffer -> view<[%m]x[%n]xf32>
  %wg = kernel.workgroup.id<x> : index
  %item = kernel.workitem.id<x> : index
  %c0 = index.constant 0 : index
  %c1 = index.constant 1 : index
  %c32 = index.constant 32 : index
  %ck = index.constant {K} : index
  %c64 = index.constant 64 : index
  %i = index.madd %wg, %c64, %item : index
  %total = index.mul %m, %n : index
  %inb = index.cmp ult, %i, %total : index
  scf.if %inb {{
    %r0 = index.div %i, %n : index
    %nm1 = index.sub %n, %c1 : index
    %mm1 = index.sub %m, %c1 : index
    %r = index.assume %r0 [range(%r0, 0, %mm1)] : index
    %c_raw = index.rem %i, %n : index
    %c = index.assume %c_raw [range(%c_raw, 0, %nm1)] : index
    %y0 = scalar.constant 0.0 : f64
    %y = scf.for %t = [%c0 to %ck step %c1](%acc = %y0 : f64) -> (f64) {{
      %b = view.load %wv[%c, %t] : view<[%n]x{K}xi8> -> i8
      %q = index.div %t, %c32 : index
      %eb = view.load %ev[%c, %q] : view<[%n]x{NCH}xi8> -> i8
      %b32 = scalar.extui %b : i8 to i32
      %m7 = scalar.constant 127 : i32
      %s7 = scalar.constant 7 : i32
      %m8 = scalar.constant 128 : i32
      %s8 = scalar.constant 8 : i32
      %mg = scalar.andi %b32, %m7 : i32
      %mgs = scalar.shli %mg, %s7 : i32
      %sg = scalar.andi %b32, %m8 : i32
      %sgs = scalar.shli %sg, %s8 : i32
      %hb = scalar.ori %mgs, %sgs : i32
      %h16 = scalar.trunci %hb : i32 to i16
      %hf = scalar.bitcast %h16 : i16 to f16
      %wf = scalar.extf %hf : f16 to f64
      %e32 = scalar.extui %eb : i8 to i32
      %ep = scalar.addi %e32, %s8 : i32
      %s23 = scalar.constant 23 : i32
      %bits = scalar.shli %ep, %s23 : i32
      %sc = scalar.bitcast %bits : i32 to f32
      %scd = scalar.extf %sc : f32 to f64
      %wd = scalar.mulf %wf, %scd : f64
      %xb = view.load %xv[%r, %t] : view<[%m]x{K}xbf16> -> bf16
      %xd = scalar.extf %xb : bf16 to f64
      %p = scalar.mulf %wd, %xd : f64
      %next = scalar.addf %acc, %p : f64
      scf.yield %next : f64
    }}
    %yf = scalar.fptrunc %y : f64 to f32
    view.store %yf, %ov[%r, %c] : f32, view<[%m]x[%n]xf32>
  }}
  kernel.return
}}
"""


def decl(name, sides):
    if sides:
        return (f"kernel.decl @{name}(%cols: index) launch(%m: i32, %n: i32, %c1: i32, %c2: i32, %c3: i32, %w: buffer, "
                "%e: buffer, %x: buffer, %o0: buffer, %o1: buffer, %o2: buffer, %o3: buffer)")
    return f"kernel.decl @{name}(%cols: index) launch(%m: i32, %n: i32, %w: buffer, %e: buffer, %x: buffer, %o: buffer)"


def inputs(lines, v, m, n, seed):
    K, NCH = v.K, v.NCH
    lines += [
        f"  %m = check.literal value({m}) : i32",
        f"  %n = check.literal value({n}) : i32",
        f"  %cols = check.literal value({n}) : index",
        f"  %count = check.literal value({m * n}) : index",
        f"  %sw = check.literal value({seed}) : i64",
        f"  %se = check.literal value({seed + 1}) : i64",
        f"  %sx = check.literal value({seed + 2}) : i64",
        f"  %w = check.generate.random.uniform seed(%sw) range(0 to 126) : tensor<{n}x{K}xi8>",
        f"  %e = check.generate.random.uniform seed(%se) range(112 to 127) : tensor<{n}x{NCH}xi8>",
        f"  %x = check.generate.random.uniform seed(%sx) range(0.0 to 2.0) : tensor<{m}x{K}xbf16>",
        f"  %ref = check.generate.fill value(0.0) : tensor<{m}x{n}xf32>",
    ]
    rt = f"tensor<{n}x{K}xi8>, tensor<{n}x{NCH}xi8>, tensor<{m}x{K}xbf16>"
    lines.append(f"  kernel.launch @mxfp8_gemv_reference_k{K}[%count](%m, %n, %w, %e, %x, %ref) : "
                 f"[index](i32, i32, {rt}, tensor<{m}x{n}xf32>)")
    return rt


def case(v, one, n, seed, m):
    """The variant at m rows of a ragged n against the 1-row variant row by row and the f64 reference."""
    K = v.K
    ot = v.OT
    size = 4 if ot == "f32" else 2
    lines = [f"check.case public @{v.name}_case {{"]
    rt = inputs(lines, v, m, n, seed)
    if v.SIDES == 4:
        cuts = (n // 4, n // 2, 3 * n // 4)
        widths = (cuts[0], cuts[1] - cuts[0], cuts[2] - cuts[1], n - cuts[2])
        for s, wd in enumerate(widths):
            lines.append(f"  %o{s} = check.generate.fill value(0.0) : tensor<{m}x{wd}xbf16>")
        lines += [f"  %c{s + 1} = check.literal value({c}) : i32" for s, c in enumerate(cuts)]
        outs = ", ".join(f"tensor<{m}x{wd}xbf16>" for wd in widths)
        lines.append(f"  kernel.launch @{v.name}[%cols](%m, %n, %c1, %c2, %c3, %w, %e, %x, %o0, %o1, %o2, %o3) : "
                     f"[index](i32, i32, i32, i32, i32, {rt}, {outs})")
        assert m == 1
        first = 0
        for s, wd in enumerate(widths):
            lines.append(f"  %want{s} = check.tensor.view %ref offset({first * 4}) : tensor<1x{n}xf32> -> tensor<1x{wd}xf32>")
            lines.append(f"  check.expect.approximate actual(%o{s}) expected(%want{s}) atol(0.0) rtol(1.0e-2) nan(same) : "
                         f"tensor<1x{wd}xbf16>, tensor<1x{wd}xf32>")
            first += wd
        lines += ["  check.return", "}"]
        return "\n".join(lines)
    lines.append(f"  %out = check.generate.fill value(0.0) : tensor<{m}x{n}x{ot}>")
    lines.append(f"  kernel.launch @{v.name}[%cols](%m, %n, %w, %e, %x, %out) : [index](i32, i32, {rt}, tensor<{m}x{n}x{ot}>)")
    tol = "1.0e-4" if ot == "f32" else "1.0e-2"
    lines.append(f"  check.expect.approximate actual(%out) expected(%ref) atol(0.0) rtol({tol}) nan(same) : "
                 f"tensor<{m}x{n}x{ot}>, tensor<{m}x{n}xf32>")
    lines.append("  %one = check.literal value(1) : i32")
    for r in range(m):
        lines += [
            f"  %x{r} = check.tensor.view %x offset({r * K * 2}) : tensor<{m}x{K}xbf16> -> tensor<1x{K}xbf16>",
            f"  %alone{r} = check.generate.fill value(0.0) : tensor<1x{n}x{ot}>",
            f"  kernel.launch @{one.name}[%cols](%one, %n, %w, %e, %x{r}, %alone{r}) : [index](i32, i32, "
            f"tensor<{n}x{K}xi8>, tensor<{n}x{v.NCH}xi8>, tensor<1x{K}xbf16>, tensor<1x{n}x{ot}>)",
            f"  %row{r} = check.tensor.view %out offset({r * n * size}) : tensor<{m}x{n}x{ot}> -> tensor<1x{n}x{ot}>",
            f"  check.expect.bitwise actual(%row{r}) expected(%alone{r}) : tensor<1x{n}x{ot}>",
        ]
    lines += ["  check.return", "}"]
    return "\n".join(lines)


def bench(v, n, seed):
    """A served shape: the launch alone, then its result within rtol of the reference."""
    m, K, ot = v.R, v.K, v.OT
    lines = [f"check.case public @{v.name}_n{n} {{"]
    rt = inputs(lines, v, m, n, seed)
    lines.append(f"  %out = check.generate.fill value(0.0) : tensor<{m}x{n}x{ot}>")
    lines.append(f"  kernel.launch @{v.name}[%cols](%m, %n, %w, %e, %x, %out) : [index](i32, i32, {rt}, tensor<{m}x{n}x{ot}>)")
    lines.append(f"  check.expect.approximate actual(%out) expected(%ref) atol(0.0) rtol(1.0e-4) nan(same) : "
                 f"tensor<{m}x{n}x{ot}>, tensor<{m}x{n}xf32>")
    lines += ["  check.return", "}", f"check.benchmark<@{v.name}_n{n}> @{v.name}_n{n}_bench"]
    return "\n".join(lines)


def main():
    specs = gen.VARIANTS
    vs = [gen.V(s) for s in specs]
    out = ["// SPDX-License-Identifier: Apache-2.0", "// Generated by gen_tests.py: mxfp8_gemv.loom's checks and benchmarks.",
           HEADER]
    names = set()
    for v in vs:
        out.append(decl(v.name, v.SIDES == 4))
        names.add(v.name)
    for K in sorted({v.K for v in vs}):
        out.append(reference(K)[1])
    seed = 1000
    for v in vs:
        spec = f"1:{v.K}:{v.L}:1:4" + (":bf16" if v.OT == "bf16" and v.SIDES == 1 else "")
        one = gen.V(spec)
        if v.SIDES == 1 and one.name not in names:
            out.append(gen.kernel(spec))
            names.add(one.name)
        n = 3 * v.COLS_WG * v.P + 5
        out.append(case(v, one, n, seed, 1 if v.SIDES == 4 else v.R))
        seed += 10
        if v.SIDES == 1 and v.OT == "f32":
            for served in SERVED[v.K]:
                out.append(bench(v, served, seed))
                seed += 10
    sys.stdout.write("\n\n".join(out) + "\n")


if __name__ == "__main__":
    main()
