//! Flash Next's Loom prompt attention and indexer under the HIP contract: byte-equal to the hipcc kernels they replace.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const Profile = @import("contract_entry.zig").Profile;
const sparse_k = @import("contract_sparse_lanes.zig");
const idx_k = @import("contract_idx_scores.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;

const ratio = 4;
const top = 512;
const key_stride = top * ratio + ratio;

fn noSums(_: c.Shape) usize {
    return 0;
}

/// tf_fn_idx_select's rows over `n` scored blocks: rows past `top` complete blocks select, the first rows stay dense.
pub const Select = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    dev: [3]u64 = @splat(0),
    out: [3]u64 = @splat(0),
    base: [3]u64 = @splat(0),

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher) Select {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a) };
    }

    pub fn deinit(self: *Select) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Select) g.Kernel {
        return .{
            .spec = .{
                .name = "loom idx_select",
                .replaces = "hip tf_fn_idx_select",
                .accum = .none,
                .depth = noSums,
                .out = .raw,
                .nan = .{ .stated = "a NaN score ranks as tf_fn_idx_select ranks it (the bytes are its)" },
                // shape.n: the last row's complete blocks; rows end at n * ratio + 3 and step back one key a row
                .served = &.{
                    .{ .rows = 64, .n = 1024, .k = 1 }, .{ .rows = 256, .n = 7800, .k = 1 },  .{ .rows = 5, .n = 600, .k = 1 },
                    .{ .rows = 200, .n = 520, .k = 1 }, .{ .rows = 129, .n = 16000, .k = 1 },
                },
                .sums = .{ .why = "integer selection" },
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    const vt = g.hooks(Select);

    /// Scores in [-1, 1) with ties every 7th block (stress: all equal); nonfinite: NaN and both infinities planted.
    pub fn prepare(self: *Select, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        const n = shape.rows;
        if (n < 1 or shape.n < 2 or shape.n * ratio < n + 4 or shape.n > 40000) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        const nb = shape.n;
        var prng = std.Random.DefaultPrng.init(seed);
        const rng = prng.random();
        const scores = try a.alloc(f32, n * nb);
        for (scores, 0..) |*x, i| {
            x.* = if (fill == .stress) 1 else if (i % 7 == 0) 0.5 else @floatCast(rng.float(f64) * 2 - 1);
        }
        if (fill == .nonfinite) for (0..n) |r| {
            scores[r * nb + r % nb] = std.math.nan(f32);
            scores[r * nb + (r + 3) % nb] = std.math.inf(f32);
            scores[r * nb + (r + 5) % nb] = -std.math.inf(f32);
        };
        const complete = try a.alloc(i32, n);
        const ends = try a.alloc(i32, n);
        for (complete, ends, 0..) |*cp, *e, r| {
            e.* = @intCast(nb * ratio + 3 - (n - 1 - r));
            cp.* = @divFloor(e.*, ratio);
        }
        self.dev = .{
            try self.rig.input(std.mem.sliceAsBytes(scores)),
            try self.rig.input(std.mem.sliceAsBytes(complete)),
            try self.rig.input(std.mem.sliceAsBytes(ends)),
        };
        const sizes = [3]usize{ n * key_stride * 4, n * 4, n * 4 };
        for (&self.out, &self.base, sizes) |*o, *b, len| {
            o.* = try self.rig.output(len);
            b.* = try self.rig.scratch(len);
        }
        return true;
    }

    fn launch(self: *Select, out: [3]u64, loom: bool) !void {
        const d = self.dev;
        const n: c_int = @intCast(self.shape.rows);
        const nb: i64 = @intCast(self.shape.n);
        const s = self.rig.s.handle;
        if (loom) {
            try self.l.attn.select(self.l.r, @ptrFromInt(d[0]), nb, @ptrFromInt(d[1]), @ptrFromInt(d[2]), @ptrFromInt(out[0]), key_stride, @ptrFromInt(out[1]), @ptrFromInt(out[2]), n, top, ratio, s);
        } else try self.l.fn_idx_select(@ptrFromInt(d[0]), nb, @ptrFromInt(d[1]), @ptrFromInt(d[2]), @ptrFromInt(out[0]), key_stride, @ptrFromInt(out[1]), @ptrFromInt(out[2]), n, top, ratio, s);
        try self.rig.s.synchronize();
    }

    pub fn run(self: *Select) !void {
        try self.launch(self.out, true);
    }

    /// tf_fn_idx_select's keys, counts and flags on the same inputs, its unwritten keys poisoned as the outputs are.
    pub fn replaced(self: *Select) ![]const u8 {
        const a = self.arena.allocator();
        const sizes = [3]usize{ self.shape.rows * key_stride * 4, self.shape.rows * 4, self.shape.rows * 4 };
        for (self.base, sizes) |b, len| {
            const poison = try a.alloc(u8, len);
            @memset(poison, 0xff);
            try self.rig.put(b, poison);
        }
        try self.launch(self.base, false);
        var all: std.ArrayList(u8) = .empty;
        for (self.base, sizes) |b, len| try all.appendSlice(a, try self.rig.bytes(b, len));
        return all.items;
    }

    pub fn count(self: *Select) usize {
        return self.shape.rows * (key_stride + 2) * 4;
    }

    pub fn reference(_: *Select, want: []f64, mag: []f64) !void {
        @memset(want, 0);
        @memset(mag, 0);
    }

    /// Every gate's run against tf_fn_idx_select's bytes.
    pub fn check(self: *Select, _: g.Gate) !?[]const u8 {
        const got = try self.arena.allocator().dupe(u8, try self.rig.outputBytes());
        const want = try self.replaced();
        if (std.mem.eql(u8, got, want)) return null;
        return "its keys, counts or flags differ from tf_fn_idx_select's";
    }
};

/// The three Loom kernels against the hipcc ones, when the build has them (-Dloom-compile).
pub fn run(a: std.mem.Allocator, _: std.Io, _: Profile) anyerror!g.Tally {
    var t: g.Tally = .{};
    var r = try hip.Runtime.open();
    defer r.close();
    var ctx = try hip.Context.init(&r, 0);
    defer ctx.deinit();
    var s = try hip.Stream.init(&r);
    defer s.deinit();
    var l = try hip.Launcher.load(&r, .{}, hip.kernels.images);
    defer l.unload();
    if (l.attn.f == null) {
        std.debug.print("N/A  loom sparse_attend, idx_scores, idx_select: built without -Dloom-compile\n", .{});
        return t;
    }
    var attend = sparse_k.SparseLanes.init(a, &r, s, &l);
    attend.loom = true;
    defer attend.deinit();
    var scores = idx_k.IdxScores.init(a, &r, s, &l, .none);
    scores.loom = true;
    defer scores.deinit();
    var select = Select.init(a, &r, s, &l);
    defer select.deinit();
    const kernels = [_]g.Kernel{ attend.kernel(), scores.kernel(), select.kernel() };
    try g.runAll(a, &kernels, 0x5a77, &t);
    return t;
}
