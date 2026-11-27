//! The Loom MXFP8 GEMV under the HIP contract: byte-equal to fp4l_stream (MXFP8) at the checkpoint's shapes, rows 1-8.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const fp8 = core.quant.fp8;
const fp4 = hip.launches.fp4local;
const gemv = hip.launches.mxfp8_gemv;
const Arg = fp4.Arg;

pub const Reg = struct { name: []const u8, out: gemv.Out, served: []const c.Shape };

/// Rows 1 to 8 of every product the table takes the stream tile's place at with output `out`.
fn served(comptime out: gemv.Out) []const c.Shape {
    @setEvalBranchQuota(100_000);
    comptime var list: []const c.Shape = &.{};
    inline for (gemv.table, 0..) |e, i| {
        const seen = for (gemv.table[0..i]) |f| {
            if (f.n == e.n and f.v.k == e.v.k and f.out == e.out) break true;
        } else false;
        const most = for (gemv.table) |f| {
            if (f.n == e.n and f.v.k == e.v.k and f.out == e.out and f.v.rows == 8) break 8;
        } else 4;
        if (e.out == out and !seen) for (1..most + 1) |m| {
            list = list ++ [_]c.Shape{.{ .rows = m, .n = e.n, .k = e.v.k }};
        };
    }
    return list;
}

/// The stitched row groups' cuts (qkv | z | b | a, q | k | v | index), else four sides cut at multiples of 8.
fn cutsOf(n: usize) [3]usize {
    return switch (n) {
        16480 => .{ 10240, 16384, 16432 },
        13952 => .{ 12288, 12800, 13312 },
        else => .{ n / 32 * 8, n / 16 * 8, 3 * n / 32 * 8 },
    };
}

const name = "loom mxfp8_gemv";
pub const registrations = [_]Reg{
    .{ .name = name ++ " (fp32 out)", .out = .f32, .served = served(.f32) },
    .{ .name = name ++ " (bf16 out)", .out = .bf16, .served = served(.bf16) },
    .{ .name = name ++ " (4 bf16 sides)", .out = .sides, .served = served(.sides) },
};

/// fp4l_stream's depth: 16 dot2s a chunk (3 each: gfx1151's dot2 is not IEEE), a fold a chunk, the lane tree.
fn depth(s: c.Shape) usize {
    return 48 + s.k / 32 + 8;
}

pub const Case = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    reg: Reg,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    /// fp64 weights [n][k].
    w: []f64 = &.{},
    arg: Arg = undefined,
    cuts: [3]usize = undefined,
    outs: [4]u64 = undefined,
    base: [4]u64 = undefined,
    at: usize = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, reg: Reg) Case {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a), .reg = reg };
    }

    pub fn deinit(self: *Case) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        return .{
            .spec = .{
                .name = self.reg.name,
                .accum = .f32,
                .depth = depth,
                .out = if (self.reg.out == .f32) .f32 else .bf16,
                .replaces = "fp4l_stream (MXFP8)",
                .served = self.reg.served,
                .tiles = .{ .{ .size = 8 }, .{ .size = 64 }, .{} },
                .sums = .{ .why = "dot2 chains of 16 pairs" },
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    const vt = g.hooks(Case);

    /// A variant of the table for k, m rows and `out`: the table's own for a served n, else any of that k (edges in n).
    fn entry(n: usize, k: usize, m: usize, out: gemv.Out) ?usize {
        var any: ?usize = null;
        for (gemv.table, 0..) |e, i| {
            if (e.v.k != k or e.out != out or e.v.rows < m) continue;
            if (e.n == n) return i;
            any = any orelse i;
        }
        return any;
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const m = shape.rows;
        const n = shape.n;
        const k = shape.k;
        if (m == 0 or m > 8 or n == 0 or (self.reg.out == .sides and n < 32)) return false;
        self.at = entry(n, k, m, self.reg.out) orelse return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        const d = try contract.dense.Dense.init(a, .{ .rows = m, .n = 1, .k = k }, 1);
        d.fillX(fill, seed, 1);
        self.x = d.x;
        var prng = std.Random.DefaultPrng.init(seed ^ 0xf4);
        const rng = prng.random();
        const stress = fill == .stress;
        const codes = try a.alloc(u8, n * k);
        const es = try a.alloc(u8, n * k / 32);
        self.w = try a.alloc(f64, n * k);
        for (es) |*e| e.* = if (stress) 127 else @intCast(118 + rng.uintLessThan(u8, 14));
        for (codes, 0..) |*b, i| {
            b.* = if (stress) stressByte(rng, i % k, k) else blk: {
                const v = rng.int(u8);
                break :blk if (v & 0x7f == 0x7f) v ^ 1 else v;
            };
            self.w[i] = fp8.e4m3(b.*) * std.math.ldexp(@as(f64, 1), @as(i32, es[i / 32]) - 127);
        }
        self.arg = .{ .x = 0, .words = 0, .scale = .{ .p = 0, .kind = 0 }, .bias = .{ .p = 0, .kind = 0 }, .out = 0, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .bits = 8, .group = 0, .fp16 = 0 };
        self.arg.words = try self.rig.input(codes);
        self.arg.scale.p = try self.rig.input(es);
        self.arg.x = try self.rig.input(std.mem.sliceAsBytes(self.x));
        self.cuts = if (self.reg.out == .sides) cutsOf(n) else .{ n, n, n };
        const elem: usize = if (self.reg.out == .f32) 4 else 2;
        for (0..4) |s| {
            const width = self.side(s);
            self.outs[s] = if (width > 0) try self.rig.output(m * width * elem) else 0;
            self.base[s] = if (width > 0) try self.rig.scratch(m * width * elem) else 0;
        }
        return true;
    }

    /// Columns of side s: [cuts[s - 1], cuts[s]) with the ends at 0 and n.
    fn side(self: *const Case, s: usize) usize {
        const lo = if (s == 0) 0 else self.cuts[s - 1];
        const hi = if (s == 3) self.shape.n else self.cuts[s];
        return hi - lo;
    }

    /// e4m3 bytes of 1 to 7 (exact), negated in the second half of K.
    fn stressByte(rng: std.Random, i: usize, k: usize) u8 {
        const lut = [_]u8{ 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e };
        const b = lut[rng.uintLessThan(usize, lut.len)];
        return if (i < k / 2) b else b | 0x80;
    }

    /// The GEMV on the case's inputs into `out` (or the hipcc stream tile, `tile`), then a sync.
    fn launch(self: *Case, out: [4]u64, tile: bool) !void {
        const f = &self.l.fp4;
        const r = self.l.r;
        const s = self.rig.s.handle;
        const a = self.arg;
        switch (self.reg.out) {
            .f32, .bf16 => {
                var arg = a;
                if (self.reg.out == .f32) arg.out = out[0] else arg.out16 = out[0];
                if (tile) try f.tileGo(r, .mxfp8, arg, .{}, 1, s) else try f.gemv.go(r, self.at, a.m, a.n, a.words, a.scale.p, a.x, out[0], self.reg.out == .bf16, s);
            },
            .sides => {
                var sides: fp4.StreamSides = .{ .count = 4, .out_half = 1 };
                var widest: c_int = 0;
                var row: u64 = 0;
                for (0..4) |i| {
                    sides.words[i] = a.words + row * self.shape.k;
                    sides.scale[i] = a.scale.p + row * (self.shape.k / 32);
                    sides.out[i] = out[i];
                    sides.n[i] = @intCast(self.side(i));
                    widest = @max(widest, sides.n[i]);
                    row += self.side(i);
                }
                if (tile) {
                    var arg = a;
                    arg.n = widest;
                    try f.tileGo(r, .mxfp8, arg, sides, 1, s);
                } else {
                    const cuts: [3]c_int = .{ @intCast(self.cuts[0]), @intCast(self.cuts[1]), @intCast(self.cuts[2]) };
                    try f.gemv.goSides(r, self.at, a.m, a.n, cuts, a.words, a.scale.p, a.x, out, s);
                }
            },
        }
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Case) !void {
        try self.launch(self.outs, false);
    }

    /// fp4l_stream's bytes on the same inputs, its outputs in the order the GEMV's are allocated.
    pub fn replaced(self: *Case) ![]const u8 {
        try self.launch(self.base, true);
        const elem: usize = if (self.reg.out == .f32) 4 else 2;
        const a = self.arena.allocator();
        var all: std.ArrayList(u8) = .empty;
        for (0..4) |s| {
            const width = self.side(s);
            if (width > 0) try all.appendSlice(a, try self.rig.bytes(self.base[s], self.shape.rows * width * elem));
        }
        return all.items;
    }

    /// Outputs in allocation order: side by side, each (rows, width).
    pub fn count(self: *Case) usize {
        return self.shape.rows * self.shape.n;
    }

    /// Output i in allocation order as (row, column of the product).
    fn coordinate(self: *const Case, i: usize) [2]usize {
        var at = i;
        var first: usize = 0;
        for (0..4) |s| {
            const width = self.side(s);
            if (at < self.shape.rows * width) return .{ at / width, first + at % width };
            at -= self.shape.rows * width;
            first += width;
        }
        unreachable;
    }

    fn terms(self: *Case, p: usize, j: usize, t: []f64) void {
        const k = self.shape.k;
        for (t, self.w[j * k ..][0..k], self.x[p * k ..][0..k]) |*o, wv, xv| o.* = c.fromBf16(xv) * wv;
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        const t = try self.arena.allocator().alloc(f64, self.shape.k);
        for (want, mag, 0..) |*w, *mg, i| {
            const at = self.coordinate(i);
            self.terms(at[0], at[1], t);
            w.* = 0;
            mg.* = 0;
            for (t) |v| {
                w.* += v;
                mg.* += @abs(v);
            }
        }
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.arena.allocator().alloc(f64, self.shape.k);
        for (out, 0..) |*o, i| {
            const at = self.coordinate(i);
            self.terms(at[0], at[1], t);
            o.* = rows.sumIn(acc, order, t);
        }
    }
};

/// The rows of a launch of fewer rows equal the widest served launch's bytes, at the first served product.
pub fn identity(a: std.mem.Allocator, x: *Case) !bool {
    const s = x.reg.served[0];
    var full: usize = 0;
    for (x.reg.served) |t| if (t.n == s.n and t.k == s.k) {
        full = @max(full, t.rows);
    };
    if (!try x.prepare(.{ .rows = full, .n = s.n, .k = s.k }, .random, 0x1d)) return false;
    try x.run();
    const elem: usize = if (x.reg.out == .f32) 4 else 2;
    var want: [4][]u8 = undefined;
    for (0..4) |i| want[i] = if (x.side(i) == 0) &.{} else try a.dupe(u8, try x.rig.bytes(x.outs[i], full * x.side(i) * elem));
    defer for (want) |w| a.free(w);
    var bad: ?usize = null;
    for (1..full) |m| {
        x.arg.m = @intCast(m);
        x.at = Case.entry(s.n, s.k, m, x.reg.out).?;
        try x.launch(x.base, false);
        for (0..4) |i| {
            const len = m * x.side(i) * elem;
            if (len > 0 and !std.mem.eql(u8, try x.rig.bytes(x.base[i], len), want[i][0..len])) bad = bad orelse m;
        }
    }
    if (bad) |m| {
        std.debug.print("FAIL identity {s}: row counts 1-{d} of {d}: first differing count {d}\n", .{ x.reg.name, full - 1, full, m });
        return false;
    }
    std.debug.print("PASS identity {s}: row counts 1-{d} of {d} byte-equal at {d}x{d}\n", .{ x.reg.name, full - 1, full, s.n, s.k });
    return true;
}
