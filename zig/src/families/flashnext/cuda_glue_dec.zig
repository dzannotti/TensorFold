//! HIP decode glue in fewer launches (src/tensorfold/families/qwen4_exp/cuda/glue_dec.py, docs/rocm/notes/fuse.md):
//! a hyper-connection read-out takes 3 launches instead of 6 (`_hc_wbn`: write-back + norm; `_b16mm_sm_act`: the
//! down projection with `_hc_act_sk` in its last program; `_hc_up_mix` at 16 rows: up + mix). Each gives the separate
//! kernels' bytes; `check` (glue-check) compares them on random and edge inputs at decode rows and times both in graphs.
const std = @import("std");
const cuda = @import("cuda");
const aot = cuda.aot;
const tri = @import("cuda_triton.zig");
const prompt = @import("cuda_prompt.zig");
const tops = @import("cuda_torch_ops.zig");
const kern = @import("cuda_kernels.zig");
const fp8 = @import("cuda_fp8.zig");

const bf16 = "*bf16";
const f32p = "*fp32";

/// The up + mix decode tile's columns a program (glue_dec: 16 or 32; TF_FLASHNEXT_DEC_BD).
pub const bd_default: usize = 16;
/// TICK words: (row tiles of 16) x (column tiles of 64); decode windows reach 128 rows, the down rows 324 columns.
pub const tick_words: usize = 64;

pub fn wbnAvailable(set: *const aot.Set) bool {
    return set.smallestConst("_hc_wbn", "MODE", 0) != null;
}

pub fn actAvailable(set: *const aot.Set) bool {
    return set.smallestConst("_b16mm_sm_act", "SK", 0) != null;
}

pub fn tkAvailable(set: *const aot.Set) bool {
    return set.smallestConst("_b16mm_tk", "SK", 0) != null;
}

pub fn upMixAvailable(set: *const aot.Set) bool {
    return set.smallestConst("_hc_up_mix", "BM", 16) == 16;
}

/// `_b16mm_sm_act`: the read-out's down projection (slice-major weight [SK, n, k / SK]) into its fp32 slices `part`,
/// then `_hc_act_sk`'s act [m, low] and (n = low + streams) inject gates from them, a tile at a time; `tick`
/// [tick_words] i32, zero (a ticket a (row, column) tile).
pub fn downAct(t: tri.Tri, x: u64, x_stride: usize, w: u64, part: u64, act: u64, inject: ?u64, tick: u64, m: usize, n: usize, k: usize, streams: usize, low: usize) !void {
    const sk = tri.b16SplitK(n, k);
    const tl = tri.b16Tile(m, n, k);
    if (tl.bm != 16 or sk < 2 or (m + 15) / 16 * ((n + tl.bn - 1) / tl.bn) > tick_words) return error.Invalid;
    try t.run("_b16mm_sm_act", .{ (m + 15) / 16, (n + tl.bn - 1) / tl.bn, sk }, &.{ aot.ptr("X", bf16, x), aot.ptr("W", bf16, w), aot.ptr("PART", f32p, part), aot.ptr("ACT", bf16, act), aot.ptr("INJ", bf16, inject orelse act), aot.ptr("TICK", "*i32", tick), aot.int("M", @intCast(m)), aot.int("x_stride", @intCast(x_stride)) }, &.{ aot.ci("N", @intCast(n)), aot.ci("K", @intCast(k)), aot.ci("SK", @intCast(sk)), aot.ci("BM", 16), aot.ci("BLOCK_N", @intCast(tl.bn)), aot.ci("BK", @intCast(tl.bk)), aot.ci("S", @intCast(streams)), aot.ci("LOW", @intCast(low)), aot.ci("LOWP", @intCast(std.math.ceilPowerOfTwoAssert(usize, low))), aot.ci("HAS_INJ", @intFromBool(n == low + streams)) });
}

// -- glue-check: the fused kernels against the separate ones -------------------------------------------------------

const D: usize = 2560;
const S: usize = 4;
const LOW: usize = 320;
const NC: usize = D / 256;
const max_m: usize = 128;
const rows_checked = [_]usize{ 1, 2, 3, 4, 5, 8, 15, 16, 17, 31, 32, 33, 48, 64, 100, 128 };

const Bufs = struct {
    list: [24]cuda.DeviceBuffer = undefined,
    n: usize = 0,

    fn get(b: *Bufs, d: *const cuda.Driver, bytes: usize) !cuda.DeviceBuffer {
        b.list[b.n] = try cuda.DeviceBuffer.alloc(d, bytes);
        b.n += 1;
        return b.list[b.n - 1];
    }

    fn free(b: *Bufs) void {
        for (b.list[0..b.n]) |*x| x.free();
    }
};

fn same(gpa: std.mem.Allocator, a: cuda.DeviceBuffer, b: cuda.DeviceBuffer, n: usize) !bool {
    const x = try gpa.alloc(u8, n);
    defer gpa.free(x);
    const y = try gpa.alloc(u8, n);
    defer gpa.free(y);
    try a.download(0, x);
    try b.download(0, y);
    return std.mem.eql(u8, x, y);
}

/// Every decode fusion against its separate kernels (bytes), then both timed in graphs. Returns whether all equal.
pub fn check(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, th: tops.Torch, ops: kern.Ops, k8: *const fp8.Kernels) !bool {
    var bs: Bufs = .{};
    defer bs.free();
    const h0 = try bs.get(d, max_m * S * D * 2);
    const ha = try bs.get(d, max_m * S * D * 2);
    const hb = try bs.get(d, max_m * S * D * 2);
    const pa = try bs.get(d, max_m * NC * S * 4);
    const pb = try bs.get(d, max_m * NC * S * 4);
    const na = try bs.get(d, max_m * S * D * 2);
    const nb = try bs.get(d, max_m * S * D * 2);
    const xs = try bs.get(d, max_m * S * D / 32 * 4);
    const scale = try bs.get(d, S * D * 4);
    const inj = try bs.get(d, max_m * S * 2);
    const br16 = try bs.get(d, max_m * D * 2);
    const yb = try bs.get(d, max_m * 11 * D * 2);
    const yf = try bs.get(d, max_m * 11 * D * 4);
    const wts = try bs.get(d, max_m * 11 * 4);
    const wdn = try bs.get(d, (LOW + S) * S * D * 2); // the down rows, slice-major [32, N, 320]
    const wup = try bs.get(d, S * D * LOW * 2);
    const parta = try bs.get(d, 32 * max_m * (LOW + S) * 4);
    const partb = try bs.get(d, 32 * max_m * (LOW + S) * 4);
    const acta = try bs.get(d, max_m * LOW * 2);
    const actb = try bs.get(d, max_m * LOW * 2);
    const ija = try bs.get(d, max_m * S * 2);
    const ijb = try bs.get(d, max_m * S * 2);
    const up = try bs.get(d, max_m * S * D * 2);
    const tick = try bs.get(d, tick_words * 4);
    try tick.fill8(0, t.s.handle);
    var prng = std.Random.DefaultPrng.init(0x64_65_63);
    const r = prng.random();
    const eps: f32 = 1e-6;
    var all = true;
    for ([_]prompt.Fill{ .normal, .wide, .edge, .special }) |fill| {
        const calm: prompt.Fill = if (fill == .special) .normal else fill;
        try prompt.fillBuf(gpa, h0, max_m * S * D, r, fill);
        try prompt.fillF32(gpa, scale, S * D, r, if (fill == .wide) .normal else calm);
        try prompt.fillBuf(gpa, inj, max_m * S, r, calm);
        try prompt.fillBuf(gpa, br16, max_m * D, r, calm);
        try prompt.fillBuf(gpa, yb, max_m * 11 * D, r, calm);
        try prompt.fillF32(gpa, yf, max_m * 11 * D, r, calm);
        try prompt.fillF32(gpa, wts, max_m * 11, r, calm);
        try prompt.fillBuf(gpa, wdn, (LOW + S) * S * D, r, if (fill == .special) .normal else fill);
        try prompt.fillBuf(gpa, wup, S * D * LOW, r, if (fill == .special) .normal else fill);
        const kinds = [_]struct { name: []const u8, br: tri.Branch }{
            .{ .name = "none", .br = .none },
            .{ .name = "bf16", .br = .{ .bf16 = br16.ptr } },
            .{ .name = "moe bf16", .br = .{ .moe = .{ .y = yb.ptr, .y_f32 = false, .wts = wts.ptr, .slots = 11 } } },
            .{ .name = "moe fp32", .br = .{ .moe = .{ .y = yf.ptr, .y_f32 = true, .wts = wts.ptr, .slots = 11 } } },
            .{ .name = "moe6 fp32", .br = .{ .moe = .{ .y = yf.ptr, .y_f32 = true, .wts = wts.ptr, .slots = 6 } } },
            .{ .name = "moe6 bf16", .br = .{ .moe = .{ .y = yb.ptr, .y_f32 = false, .wts = wts.ptr, .slots = 6 } } },
        };
        for (rows_checked) |m| {
            // write-back + norm: in place, and (a branch) into another buffer as finish does
            for (kinds) |kd| for ([_]bool{ false, true }) |apart| {
                if (apart and kd.br == .none) continue;
                const inject: ?u64 = if (kd.br == .none) null else inj.ptr;
                try ha.copyFrom(0, h0.ptr, m * S * D * 2, t.s.handle);
                try hb.copyFrom(0, h0.ptr, m * S * D * 2, t.s.handle);
                try pa.fill8(0xA5, t.s.handle);
                try pb.fill8(0x5A, t.s.handle);
                try na.fill8(0xA5, t.s.handle);
                try nb.fill8(0x5A, t.s.handle);
                try t.hcWriteback(ha.ptr, ha.ptr, pa.ptr, inject, kd.br, m, D, S);
                try t.hcNormed(ha.ptr, pa.ptr, scale.ptr, na.ptr, xs.ptr, m, D, S, eps);
                const out = if (apart) up.ptr else hb.ptr;
                if (apart) try up.fill8(0x33, t.s.handle);
                try t.hcWritebackNorm(hb.ptr, out, pb.ptr, inject, kd.br, m, D, S, .{ .scale = scale.ptr, .normed = nb.ptr, .eps = eps, .streams = true });
                try t.s.synchronize();
                var ok = try same(gpa, pa, pb, m * NC * S * 4) and try same(gpa, na, nb, m * S * D * 2);
                ok = ok and try same(gpa, ha, if (apart) up else hb, m * S * D * 2);
                if (apart) ok = ok and try same(gpa, h0, hb, m * S * D * 2); // the source rows untouched
                if (!ok) all = false;
                if (!ok or m == 1 or m == 128) std.debug.print("{s} _hc_wbn {s}{s} rows {d} {s}: streams, squared sums, normed bytes\n", .{ if (ok) "EQUAL" else "DIFFER", kd.name, if (apart) " apart" else "", m, @tagName(fill) });
            };
            // down + act, with and without the inject gates; twice, so the second launch finds TICK at 0
            for ([_]usize{ LOW + S, LOW }) |n| for (0..2) |rep| {
                const inject: ?u64 = if (n == LOW + S) ija.ptr else null;
                try acta.fill8(0xA5, t.s.handle);
                try actb.fill8(0x5A, t.s.handle);
                try ija.fill8(0xA5, t.s.handle);
                try prompt.b16Slices("_b16mm_sm", t, h0.ptr, S * D, wdn.ptr, parta.ptr, true, m, n, S * D);
                try t.hcActSk(parta.ptr, acta.ptr, xs.ptr, inject, m, n, S, LOW, 32);
                const ia = try gpa.alloc(u8, m * S * 2);
                defer gpa.free(ia);
                try t.s.synchronize();
                try ija.download(0, ia);
                try ijb.fill8(0x5A, t.s.handle);
                try downAct(t, h0.ptr, S * D, wdn.ptr, partb.ptr, actb.ptr, if (inject != null) ijb.ptr else null, tick.ptr, m, n, S * D, S, LOW);
                try t.s.synchronize();
                var ok = try same(gpa, parta, partb, 32 * m * n * 4) and try same(gpa, acta, actb, m * LOW * 2);
                if (inject != null) {
                    const ib = try gpa.alloc(u8, m * S * 2);
                    defer gpa.free(ib);
                    try ijb.download(0, ib);
                    ok = ok and std.mem.eql(u8, ia, ib);
                }
                var tk: [tick_words]u32 = undefined;
                try tick.download(0, std.mem.sliceAsBytes(&tk));
                for (tk) |v| ok = ok and v == 0;
                if (!ok) all = false;
                if (!ok or (rep == 0 and (m == 1 or m == 128))) std.debug.print("{s} _b16mm_sm_act N {d} rows {d} {s}: slices, act, gates bytes, ticks back at 0\n", .{ if (ok) "EQUAL" else "DIFFER", n, m, @tagName(fill) });
            };
            // up + mix at 16 rows
            for ([_]usize{ 16, 32 }) |bd| {
                try na.fill8(0xA5, t.s.handle);
                try nb.fill8(0x5A, t.s.handle);
                try t.b16mm(acta.ptr, LOW, wup.ptr, up.ptr, false, 0, m, S * D, LOW);
                try t.hcMix(up.ptr, h0.ptr, na.ptr, xs.ptr, m, D, S);
                try prompt.upMixTile(t, acta.ptr, wup.ptr, h0.ptr, nb.ptr, m, D, S, LOW, 16, bd);
                try t.s.synchronize();
                const ok = try same(gpa, na, nb, m * D * 2);
                if (!ok) all = false;
                if (!ok or m == 1 or m == 128) std.debug.print("{s} _hc_up_mix BM 16 BD {d} rows {d} {s}: mixed bytes\n", .{ if (ok) "EQUAL" else "DIFFER", bd, m, @tagName(fill) });
            }
        }
    }
    if (!try reduceCheck(gpa, d, t, th)) all = false;
    if (!try topkCheck(gpa, d, t, th, ops)) all = false;
    if (!try tkCheck(gpa, d, t)) all = false;
    if (!try swigluCheck(gpa, d, t, th, k8)) all = false;
    for ([_]usize{ 1, 4, 16 }) |m| try bench(d, t, m, .{ .h = ha.ptr, .pss = pa.ptr, .inj = inj.ptr, .y = yb.ptr, .wts = wts.ptr, .scale = scale.ptr, .normed = na.ptr, .xs = xs.ptr, .wdn = wdn.ptr, .wup = wup.ptr, .part = parta.ptr, .act = acta.ptr, .ij = ija.ptr, .up = up.ptr, .mixed = nb.ptr, .tick = tick.ptr });
    return all;
}

const Ptrs = struct { h: u64, pss: u64, inj: u64, y: u64, wts: u64, scale: u64, normed: u64, xs: u64, wdn: u64, wup: u64, part: u64, act: u64, ij: u64, up: u64, mixed: u64, tick: u64 };

/// One read-out (MoE branch, inject gates) the separate way (6 launches) or fused (3).
fn readout(t: tri.Tri, m: usize, p: Ptrs, fused: bool) !void {
    const br: tri.Branch = .{ .moe = .{ .y = p.y, .y_f32 = false, .wts = p.wts, .slots = 11 } };
    if (fused) {
        try t.hcWritebackNorm(p.h, p.h, p.pss, p.inj, br, m, D, S, .{ .scale = p.scale, .normed = p.normed, .eps = 1e-6, .streams = true });
        try downAct(t, p.normed, S * D, p.wdn, p.part, p.act, p.ij, p.tick, m, LOW + S, S * D, S, LOW);
        return prompt.upMixTile(t, p.act, p.wup, p.normed, p.mixed, m, D, S, LOW, 16, bd_default);
    }
    try t.hcWriteback(p.h, p.h, p.pss, p.inj, br, m, D, S);
    try t.hcNormed(p.h, p.pss, p.scale, p.normed, p.xs, m, D, S, 1e-6);
    try prompt.b16Slices("_b16mm_sm", t, p.normed, S * D, p.wdn, p.part, true, m, LOW + S, S * D);
    try t.hcActSk(p.part, p.act, p.xs, p.ij, m, LOW + S, S, LOW, 32);
    try t.b16mm(p.act, LOW, p.wup, p.up, false, 0, m, S * D, LOW);
    try t.hcMix(p.up, p.normed, p.mixed, p.xs, m, D, S);
}

/// 96 read-outs (a decode round's) captured in a graph each way, replayed; ms a replay.
fn bench(d: *const cuda.Driver, t: tri.Tri, m: usize, p: Ptrs) !void {
    var ms: [2]f32 = undefined;
    for ([_]bool{ false, true }, 0..) |fused, i| {
        try cuda.graph.beginCapture(t.s, .thread_local);
        for (0..96) |_| try readout(t, m, p, fused);
        var g = try cuda.graph.endCapture(t.s);
        defer g.deinit();
        var x = try g.instantiate();
        defer x.deinit();
        var e0 = try cuda.Event.init(d, true);
        defer e0.deinit();
        var e1 = try cuda.Event.init(d, true);
        defer e1.deinit();
        try x.launchOn(t.s);
        try e0.record(t.s);
        for (0..20) |_| try x.launchOn(t.s);
        try e1.record(t.s);
        try e1.synchronize();
        ms[i] = try cuda.Event.elapsedMs(e0, e1) / 20;
    }
    std.debug.print("bench 96 read-outs in a graph, {d} rows: 6 launches each {d:.3} ms, fused 3 {d:.3} ms (saves {d:.3} ms)\n", .{ m, ms[0], ms[1], ms[0] - ms[1] });
}

/// fn_ops reduce_ld against `_reduce` + the strided copy: a projection's split bf16 columns (DeltaNet b|a 96, the
/// indexer's 640 at K 2560) into their place in wider rows; the whole rows' bytes compared.
fn reduceCheck(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, th: tops.Torch) !bool {
    var bs: Bufs = .{};
    defer bs.free();
    const K: usize = 2560;
    const ldo: usize = 17000;
    const x = try bs.get(d, max_m * K * 2);
    const w = try bs.get(d, 640 * K * 2);
    const part = try bs.get(d, 8 * max_m * 640 * 4);
    const pb = try bs.get(d, max_m * 640 * 2);
    const oa = try bs.get(d, max_m * ldo * 2);
    const ob = try bs.get(d, max_m * ldo * 2);
    var prng = std.Random.DefaultPrng.init(0x72_6c_64);
    const r = prng.random();
    var all = true;
    for ([_]prompt.Fill{ .normal, .wide, .cancel, .edge, .special }) |fill| {
        try prompt.fillBuf(gpa, x, max_m * K, r, fill);
        try prompt.fillBuf(gpa, w, 640 * K, r, if (fill == .special) .normal else fill);
        for ([_]usize{ 96, 640 }) |n| for (rows_checked) |m| {
            const sk = tri.b16SplitK(n, K);
            try oa.fill8(0x11, t.s.handle);
            try ob.fill8(0x11, t.s.handle);
            try t.b16mm(x.ptr, K, w.ptr, pb.ptr, false, part.ptr, m, n, K);
            try th.slotCopy(pb.ptr, n * 2, oa.ptr + 1000 * 2, ldo * 2, n * 2, m);
            try prompt.b16Slices("_b16mm", t, x.ptr, K, w.ptr, part.ptr, false, m, n, K);
            try th.reduceLd(part.ptr, ob.ptr + 1000 * 2, m, n, sk, ldo);
            try t.s.synchronize();
            const ok = try same(gpa, oa, ob, m * ldo * 2);
            if (!ok) all = false;
            if (!ok or m == 1 or m == 128) std.debug.print("{s} reduce_ld N {d} SK {d} rows {d} {s}: rows bytes\n", .{ if (ok) "EQUAL" else "DIFFER", n, sk, m, @tagName(fill) });
        };
    }
    return all;
}

/// fn_ops topk_plan against `_topk_rows` + the one-block plan: picks, weights, members, items and counts bytes, top-k
/// 10 and 5 of 512 routed logits; logits normal, spread wide (exp's small-input path), and tied.
fn topkCheck(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, th: tops.Torch, ops: kern.Ops) !bool {
    var bs: Bufs = .{};
    defer bs.free();
    const E: usize = 512;
    const rows_max: usize = 93;
    const logits = try bs.get(d, rows_max * (E + 1) * 4);
    var out: [2][5]cuda.DeviceBuffer = undefined;
    const sizes = [5]usize{ 1024 * 4, 1024 * 4, 1024 * 4, (1024 + E + 1) * 3 * 4, 64 };
    for (&out) |*o| for (o, sizes) |*b, n| {
        b.* = try bs.get(d, n);
    };
    var prng = std.Random.DefaultPrng.init(0x74_6b_70);
    const r = prng.random();
    const host = try gpa.alloc(f32, rows_max * (E + 1));
    defer gpa.free(host);
    var all = true;
    for ([_][]const u8{ "normal", "wide", "tied" }) |kind| {
        for (host) |*v| v.* = if (std.mem.eql(u8, kind, "normal")) r.floatNorm(f32) * 2 else if (std.mem.eql(u8, kind, "wide")) r.floatNorm(f32) * 60 else @floatFromInt(@as(i32, @intCast(r.uintLessThan(u32, 7))) - 3);
        try logits.upload(0, std.mem.sliceAsBytes(host));
        for ([_]usize{ 10, 5 }) |top| for (rows_checked) |m| {
            if (m * (top + 1) > 1024) continue;
            for (&out) |*o| for (o) |b| try b.fill8(0xA5, t.s.handle);
            const plan_a: kern.Plan = .{ .members = out[0][2].ptr, .items = out[0][3].ptr, .counts = out[0][4].ptr, .rank = 0, .hist = 0 };
            try t.topkRows(logits.ptr, out[0][0].ptr, out[0][1].ptr, m, E, top);
            try ops.plan(out[0][0].ptr, m * (top + 1), E + 1, kern.plan_tile, plan_a);
            try th.topkPlan(logits.ptr, m, E, E + 1, top, out[1][0].ptr, out[1][1].ptr, kern.plan_tile, out[1][2].ptr, out[1][3].ptr, out[1][4].ptr);
            try t.s.synchronize();
            var ok = true;
            const pairs = m * (top + 1);
            const used = [5]usize{ pairs * 4, pairs * 4, pairs * 4, kern.maxItems(pairs, E + 1, kern.plan_tile) * 3 * 4, 8 };
            for (out[0], out[1], used) |a, b, n| ok = ok and try same(gpa, a, b, n);
            if (!ok) all = false;
            if (!ok or m == 1 or m == 64) std.debug.print("{s} topk_plan top {d} rows {d} {s}: picks, weights, members, items, counts bytes\n", .{ if (ok) "EQUAL" else "DIFFER", top, m, kind });
        };
    }
    return all;
}

/// `_b16mm_tk` against `_b16mm` + `_reduce` (+ the strided copy for rows ldo apart): the decode shapes with split K
/// (DeltaNet b|a into its rows, the MTP head's fc_e / o_proj, fast-fp8's inject rows), bf16 and fp32 out, each twice
/// (the second launch finds the tickets at 0).
fn tkCheck(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri) !bool {
    var bs: Bufs = .{};
    defer bs.free();
    const kmax: usize = 10240;
    const x = try bs.get(d, max_m * kmax * 2);
    const w = try bs.get(d, 2560 * 6144 * 2);
    const part = try bs.get(d, 32 * max_m * 2560 * 4);
    const ref = try bs.get(d, max_m * 2560 * 4);
    const oa = try bs.get(d, max_m * 17000 * 4);
    const ob = try bs.get(d, max_m * 17000 * 4);
    const tick = try bs.get(d, tri.Tri.tk_words * 4);
    try tick.fill8(0, t.s.handle);
    var tt = t;
    tt.tick = tick.ptr;
    var prng = std.Random.DefaultPrng.init(0x74_6b_6b);
    const r = prng.random();
    const shapes = [_]struct { n: usize, k: usize, ldo: usize, fp32: bool }{
        .{ .n = 96, .k = 2560, .ldo = 16480, .fp32 = false },
        .{ .n = 640, .k = 2560, .ldo = 13952, .fp32 = false },
        .{ .n = 2560, .k = 2560, .ldo = 2560, .fp32 = false },
        .{ .n = 2560, .k = 6144, .ldo = 2560, .fp32 = false },
        .{ .n = 4, .k = 10240, .ldo = 4, .fp32 = true },
        .{ .n = 320, .k = 10240, .ldo = 320, .fp32 = true },
    };
    var all = true;
    for ([_]prompt.Fill{ .normal, .wide, .cancel, .edge, .special }) |fill| {
        try prompt.fillBuf(gpa, x, max_m * kmax, r, fill);
        try prompt.fillBuf(gpa, w, 2560 * 6144, r, if (fill == .special) .normal else fill);
        for (shapes) |c| for (rows_checked) |m| {
            if (!tt.tickFits("_b16mm", m, c.n, c.k)) continue;
            const es: usize = if (c.fp32) 4 else 2;
            try oa.fill8(0x11, t.s.handle);
            try ob.fill8(0x11, t.s.handle);
            try t.b16mm(x.ptr, c.k, w.ptr, ref.ptr, c.fp32, part.ptr, m, c.n, c.k);
            for (0..m) |row| try oa.copyFrom(row * c.ldo * es, ref.ptr + row * c.n * es, c.n * es, t.s.handle);
            for (0..2) |_| try tt.b16mmTk(x.ptr, c.k, w.ptr, ob.ptr, c.ldo, c.fp32, part.ptr, m, c.n, c.k);
            try t.s.synchronize();
            var ok = try same(gpa, oa, ob, m * c.ldo * es);
            var tk: [tri.Tri.tk_words]u32 = undefined;
            try tick.download(0, std.mem.sliceAsBytes(&tk));
            for (tk) |v| ok = ok and v == 0;
            if (!ok) all = false;
            if (!ok or m == 1 or m == 128) std.debug.print("{s} _b16mm_tk N {d} K {d} ldo {d} {s} rows {d} {s}: rows bytes, ticks back at 0\n", .{ if (ok) "EQUAL" else "DIFFER", c.n, c.k, c.ldo, if (c.fp32) "fp32" else "bf16", m, @tagName(fill) });
        };
    }
    return all;
}

/// The shared expert's gate|up and SwiGLU: fp8.matmulSwiglu (16/32-row tiles) and fp8.matmul + fn_ops' interleaved
/// SwiGLU, both on interleaved rows, against fp8.matmul + tf_fn_shared_swiglu on the plain rows (act bytes); random
/// e4m3 weights and column scales, n 2560 / 1280 (one rank / two), K 2560.
fn swigluCheck(gpa: std.mem.Allocator, d: *const cuda.Driver, t: tri.Tri, th: tops.Torch, k8: *const fp8.Kernels) !bool {
    var bs: Bufs = .{};
    defer bs.free();
    const K: usize = 2560;
    const x = try bs.get(d, max_m * K * 2);
    const w8 = [2]cuda.DeviceBuffer{ try bs.get(d, 2560 * K), try bs.get(d, 2560 * K) };
    const sc = [2]cuda.DeviceBuffer{ try bs.get(d, 2560 * (K / 64) * 4), try bs.get(d, 2560 * (K / 64) * 4) };
    const g = try bs.get(d, max_m * 2560 * 2);
    const acts = [3]cuda.DeviceBuffer{ try bs.get(d, max_m * 1280 * 2), try bs.get(d, max_m * 1280 * 2), try bs.get(d, max_m * 1280 * 2) };
    var prng = std.Random.DefaultPrng.init(0x73_77_67);
    const r = prng.random();
    var all = true;
    for ([_]usize{ 2560, 1280 }) |n| {
        const w = n / 2;
        const codes = try gpa.alloc(u8, n * K);
        defer gpa.free(codes);
        const cols = try gpa.alloc(f32, n * (K / 64));
        defer gpa.free(cols);
        for (codes) |*c| c.* = blk: {
            const v = r.int(u8);
            break :blk if (v & 0x7F == 0x7F) v - 1 else v; // no NaN codes
        };
        for (cols) |*c| c.* = std.math.ldexp(@as(f32, 1.0) + r.float(f32), r.intRangeAtMost(i32, -12, -6));
        const packed_w = try gpa.alloc(u8, n * K);
        defer gpa.free(packed_w);
        const packed_s = try gpa.alloc(f32, n * (K / 64));
        defer gpa.free(packed_s);
        var lin: [2]fp8.Linear = undefined;
        for (0..2) |v| {
            if (v == 1) {
                // runs of 32: gate rows 32 j .. 32 j + 31, then up rows w + 32 j .., as weights.interleave lays them
                const c2 = try gpa.dupe(u8, codes);
                defer gpa.free(c2);
                const s2 = try gpa.dupe(f32, cols);
                defer gpa.free(s2);
                for (0..n) |j| {
                    const src = (j / 64) * 32 + j % 32 + (if (j % 64 >= 32) w else 0);
                    @memcpy(codes[j * K ..][0..K], c2[src * K ..][0..K]);
                    @memcpy(cols[j * (K / 64) ..][0 .. K / 64], s2[src * (K / 64) ..][0 .. K / 64]);
                }
            }
            try fp8.fragmentOrder(codes, n, K, n, packed_w);
            try fp8.tileScales(cols, n, K, n, packed_s);
            try w8[v].upload(0, packed_w);
            try sc[v].upload(0, std.mem.sliceAsBytes(packed_s));
            lin[v] = .{ .w8 = w8[v].ptr, .bs = sc[v].ptr, .n = @intCast(n), .k = @intCast(K), .npad = @intCast(n) };
        }
        for ([_]prompt.Fill{ .normal, .wide, .edge }) |fill| {
            try prompt.fillBuf(gpa, x, max_m * K, r, fill);
            for (rows_checked) |m| {
                for (acts) |a| try a.fill8(0x5A, t.s.handle);
                try fp8.matmul(k8, t.s, x.ptr, K, lin[0], g.ptr, false, m);
                try th.sharedSwiglu(g.ptr, acts[0].ptr, m, w, w);
                try fp8.matmul(k8, t.s, x.ptr, K, lin[1], g.ptr, false, m);
                try th.sharedSwigluIl(g.ptr, acts[1].ptr, m, w, w, 32);
                const fused = fp8.swigluRows(k8, m);
                if (fused) try fp8.matmulSwiglu(k8, t.s, x.ptr, K, lin[1], acts[2].ptr, m);
                try t.s.synchronize();
                var ok = try same(gpa, acts[0], acts[1], m * w * 2);
                if (fused) ok = ok and try same(gpa, acts[0], acts[2], m * w * 2);
                if (!ok) all = false;
                if (!ok or m == 1 or m == 32 or m == 128) std.debug.print("{s} shared SwiGLU n {d} rows {d} {s}: act bytes (interleaved + fn_ops{s})\n", .{ if (ok) "EQUAL" else "DIFFER", n, m, @tagName(fill), if (fused) ", matmulSwiglu" else "" });
            }
        }
    }
    return all;
}
