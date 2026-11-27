//! FP4-Local's products under the HIP contract: NVFP4 (plain, routed), MXFP8, BF16 on the stream and matrix tiles.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const rows = contract.rows;
const fp8 = core.quant.fp8;
const nvfp4 = core.quant.nvfp4;
const fp4 = hip.launches.fp4local;
const Format = fp4.Format;
pub const Arg = fp4.Arg;

pub const Tile = enum { stream, wmma, item, pair };

/// One registration: a format on a tile at the served shapes; routed runs a two-expert plan with per-expert globals.
pub const Reg = struct { name: []const u8, fmt: Format, tile: Tile, routed: bool = false, served: []const c.Shape };

const dn = "hip fp4l_stream";
const pn = "hip fp4l_wmma";
pub const registrations = [_]Reg{
    // MXFP8 decode: DeltaNet qkv / z / a, out; attention q, k, index, o; shared expert gate|up and down
    .{ .name = dn ++ " (MXFP8, fp32 out)", .fmt = .mxfp8, .tile = .stream, .served = &.{
        .{ .rows = 1, .n = 10240, .k = 2560 }, .{ .rows = 8, .n = 12288, .k = 2560 }, .{ .rows = 3, .n = 2560, .k = 6144 },
        .{ .rows = 1, .n = 48, .k = 2560 },    .{ .rows = 5, .n = 640, .k = 2560 },   .{ .rows = 2, .n = 2560, .k = 640 },
    } },
    .{ .name = pn ++ " (MXFP8, fp32 out)", .fmt = .mxfp8, .tile = .wmma, .served = &.{
        .{ .rows = 129, .n = 6144, .k = 2560 }, .{ .rows = 64, .n = 2560, .k = 6144 }, .{ .rows = 300, .n = 512, .k = 2560 },
        .{ .rows = 16, .n = 48, .k = 2560 },    .{ .rows = 9, .n = 2560, .k = 640 },
    } },
    // NVFP4 experts: gate|up 2560 -> 2 x 640, down 640 -> 2560; plain (MTP) and routed over a plan
    .{ .name = dn ++ " (NVFP4, fp32 out)", .fmt = .nvfp4, .tile = .stream, .served = &.{
        .{ .rows = 1, .n = 1280, .k = 2560 }, .{ .rows = 8, .n = 1280, .k = 2560 }, .{ .rows = 4, .n = 2560, .k = 640 },
    } },
    .{ .name = dn ++ " (NVFP4 routed plan, fp32 out)", .fmt = .nvfp4, .tile = .stream, .routed = true, .served = &.{
        .{ .rows = 10, .n = 1280, .k = 2560 }, .{ .rows = 7, .n = 2560, .k = 640 },
    } },
    .{ .name = pn ++ " (NVFP4 routed plan, fp32 out)", .fmt = .nvfp4, .tile = .wmma, .routed = true, .served = &.{
        .{ .rows = 160, .n = 1280, .k = 2560 }, .{ .rows = 90, .n = 2560, .k = 640 },
    } },
    .{ .name = "hip fp4l_item (NVFP4 routed plan, 64 rows, fp32 out)", .fmt = .nvfp4, .tile = .item, .routed = true, .served = &.{
        .{ .rows = 100, .n = 1280, .k = 2560 }, .{ .rows = 40, .n = 2560, .k = 640 },
    } },
    // the (gate | up) prompt tile with silu(gate) x up in its epilogue: routed experts, and the shared expert
    .{ .name = "hip fp4l_pair (NVFP4 routed plan, bf16 act)", .fmt = .nvfp4, .tile = .pair, .routed = true, .served = &.{
        .{ .rows = 160, .n = 1280, .k = 2560 }, .{ .rows = 64, .n = 1280, .k = 2560 },
    } },
    .{ .name = "hip fp4l_pair (MXFP8, bf16 act)", .fmt = .mxfp8, .tile = .pair, .served = &.{
        .{ .rows = 130, .n = 1280, .k = 2560 }, .{ .rows = 7, .n = 1280, .k = 2560 },
    } },
    // BF16: lm_head, HC mixer down / up (K 320 folds by 64), inject, router
    .{ .name = dn ++ " (BF16, fp32 out)", .fmt = .bf16, .tile = .stream, .served = &.{
        .{ .rows = 1, .n = 248320, .k = 2560 }, .{ .rows = 4, .n = 320, .k = 10240 }, .{ .rows = 8, .n = 10240, .k = 320 },
        .{ .rows = 2, .n = 4, .k = 10240 },     .{ .rows = 1, .n = 512, .k = 2560 },
    } },
    .{ .name = pn ++ " (BF16, fp32 out)", .fmt = .bf16, .tile = .wmma, .served = &.{
        .{ .rows = 128, .n = 320, .k = 10240 }, .{ .rows = 65, .n = 10240, .k = 320 }, .{ .rows = 33, .n = 512, .k = 2560 },
        .{ .rows = 20, .n = 4, .k = 10240 },
    } },
};

/// Decode roundings: 16 dot2s a chunk (3 each: gfx1151's dot2 is not IEEE), a fold a chunk, the lane tree, the global.
fn depthStream(s: c.Shape) usize {
    return 48 + s.k / 32 + 8;
}

fn depthPrompt(s: c.Shape) usize {
    return 2 * 128 + s.k / 32 + 4;
}

const experts = 2;

pub const Case = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    arena: std.heap.ArenaAllocator,
    reg: Reg,
    shape: c.Shape = undefined,
    x: []u16 = &.{},
    /// fp64 weights [experts or 1][n][k], the global already in.
    w: []f64 = &.{},
    /// pair i's expert (routed), and the x / out row it is.
    expert_of: []usize = &.{},
    member: []usize = &.{},
    arg: Arg = undefined,
    items: c_int = 1,
    base: u64 = 0,
    /// The pair tile's baseline act, from base's fp32 product and moe_act.
    base16: u64 = 0,

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, reg: Reg) Case {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .arena = .init(a), .reg = reg };
    }

    pub fn deinit(self: *Case) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Case) g.Kernel {
        const prompt = self.reg.tile != .stream;
        const pair = self.reg.tile == .pair;
        return .{
            .spec = .{
                .name = self.reg.name,
                .accum = if (pair) .none else .f32,
                .depth = if (prompt) depthPrompt else depthStream,
                .ops = if (self.reg.fmt == .nvfp4) 1 else 0,
                .out = if (pair) .bf16 else .f32,
                .replaces = if (self.reg.tile == .item) "fp4l_wmma" else if (pair) "fp4l_wmma then moe_act" else null,
                .served = self.reg.served,
                .tiles = if (prompt)
                    .{ .{ .size = if (self.reg.tile == .item) 64 else 128 }, .{ .size = 128 }, .{} }
                else
                    .{ .{ .size = 4 }, .{ .size = 64 }, .{ .size = 1024, .step = 128, .min = 128 } },
                .sums = .{ .why = "dot2 chains of 16 pairs" },
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    const vt = g.hooks(Case);

    fn fold(self: *const Case, k: usize) usize {
        return if (self.reg.fmt == .bf16) @intCast(fp4.bf16Fold(@intCast(k))) else 128;
    }

    pub fn prepare(self: *Case, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.rows == 0 or shape.n == 0 or shape.k == 0 or shape.k % 32 != 0) return false;
        if (self.reg.tile != .stream and shape.k % self.fold(shape.k) != 0) return false;
        if (self.reg.fmt == .nvfp4 and self.reg.routed and shape.n % 2 != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        const m = shape.rows;
        const n = shape.n;
        const k = shape.k;
        const d = try contract.dense.Dense.init(a, .{ .rows = m, .n = 1, .k = k }, 1);
        d.fillX(fill, seed, 1);
        self.x = d.x;
        var prng = std.Random.DefaultPrng.init(seed ^ 0xf4);
        const rng = prng.random();
        const stacks: usize = if (self.reg.routed) experts else 1;
        self.w = try a.alloc(f64, stacks * n * k);
        var arg: Arg = .{ .x = 0, .words = 0, .scale = .{ .p = 0, .kind = 0 }, .bias = .{ .p = 0, .kind = 0 }, .out = 0, .m = @intCast(m), .n = @intCast(n), .k = @intCast(k), .bits = 0, .group = 0, .fp16 = 0 };
        const stress = fill == .stress;
        switch (self.reg.fmt) {
            .nvfp4 => {
                // baked words (pairs in nibbles p and p + 4), e4m3 scales [n][k / 16], globals [stacks][2]
                const words = try a.alloc(u32, stacks * n * k / 8);
                const scales = try a.alloc(u8, stacks * n * k / 16);
                const globals = try a.alloc(f32, stacks * 2);
                for (globals) |*v| v.* = if (stress) 1 else 0x1p-6 * (1 + rng.float(f32));
                for (scales) |*s| s.* = if (stress) 0x38 else @intCast(0x20 + rng.uintLessThan(u8, 0x30));
                @memset(words, 0);
                for (0..stacks * n) |row| for (0..k) |i| {
                    const code: u4 = if (stress) stressCode(rng, i, k) else rng.int(u4);
                    const p = i % 8 / 2;
                    words[row * k / 8 + i / 8] |= @as(u32, code) << @intCast(4 * p + 16 * (i % 2));
                    const e = row / n;
                    const col = row % n;
                    const gl = globals[2 * e + @intFromBool(self.reg.routed and col >= n / 2)];
                    self.w[row * k + i] = nvfp4.e2m1(code) * fp8.e4m3(scales[row * (k / 16) + i / 16]) * @as(f64, gl);
                };
                arg.words = try self.rig.input(std.mem.sliceAsBytes(words));
                arg.scale.p = try self.rig.input(scales);
                arg.global = try self.rig.input(std.mem.sliceAsBytes(globals));
                arg.global_layout = if (self.reg.routed) .expert_pair else .tensor;
                arg.bits = 4;
            },
            .mxfp8 => {
                const codes = try a.alloc(u8, n * k);
                const es = try a.alloc(u8, n * k / 32);
                for (es) |*e| e.* = if (stress) 127 else @intCast(118 + rng.uintLessThan(u8, 14));
                for (codes, 0..) |*b, i| {
                    b.* = if (stress) stressByte(rng, i % k, k) else blk: {
                        const v = rng.int(u8);
                        break :blk if (v & 0x7f == 0x7f) v ^ 1 else v;
                    };
                    self.w[i] = fp8.e4m3(b.*) * std.math.ldexp(@as(f64, 1), @as(i32, es[i / 32]) - 127);
                }
                arg.words = try self.rig.input(codes);
                arg.scale.p = try self.rig.input(es);
                arg.bits = 8;
            },
            .bf16 => {
                const bits = try a.alloc(u16, n * k);
                for (bits, 0..) |*b, i| {
                    b.* = if (stress) c.toBf16(@floatCast(nvfp4.e2m1(stressCode(rng, i % k, k)))) else c.toBf16(rng.float(f32) * 2 - 1);
                    self.w[i] = c.fromBf16(b.*);
                }
                arg.words = try self.rig.input(std.mem.sliceAsBytes(bits));
                arg.bits = 16;
                arg.group = fp4.bf16Fold(@intCast(k));
            },
        }
        self.expert_of = try a.alloc(usize, m);
        self.member = try a.alloc(usize, m);
        for (self.member, 0..) |*v, i| v.* = if (self.reg.routed) m - 1 - i else i;
        self.items = 1;
        if (self.reg.routed) {
            // item 0: expert 1 takes the first half of the positions, item 1: expert 0 the rest
            const half = (m + 1) / 2;
            const items = [_]i32{ 1, 0, @intCast(half), 0, @intCast(half), @intCast(m - half) };
            for (0..m) |i| self.expert_of[self.member[i]] = if (i < half) 1 else 0;
            const members = try a.alloc(i32, m);
            for (members, self.member) |*v, p| v.* = @intCast(p);
            arg.route = .{ .items = try self.rig.input(std.mem.sliceAsBytes(&items)), .members = try self.rig.input(std.mem.sliceAsBytes(members)) };
            arg.m = @intCast(half);
            self.items = 2;
        } else @memset(self.expert_of, 0);
        arg.x = try self.rig.input(std.mem.sliceAsBytes(self.x));
        if (self.reg.tile == .pair) {
            if (n % 2 != 0 or k % 128 != 0) return false;
            arg.out16 = try self.rig.output(m * n);
            self.base16 = try self.rig.scratch(m * n);
        } else arg.out = try self.rig.output(m * n * 4);
        self.base = try self.rig.scratch(m * n * 4);
        self.arg = arg;
        return true;
    }

    /// Gate 4's weights: values 1 to 6 (codes 2 to 7) in the first half of K, the same negated in the second.
    fn stressCode(rng: std.Random, i: usize, k: usize) u4 {
        const v: u4 = @intCast(2 + rng.uintLessThan(u8, 6));
        return if (i < k / 2) v else v | 8;
    }

    /// e4m3 bytes of 1 to 7 (exact), negated in the second half of K.
    fn stressByte(rng: std.Random, i: usize, k: usize) u8 {
        const lut = [_]u8{ 0x38, 0x40, 0x44, 0x48, 0x4a, 0x4c, 0x4e };
        const b = lut[rng.uintLessThan(usize, lut.len)];
        return if (i < k / 2) b else b | 0x80;
    }

    fn launch(self: *Case, tile: Tile, arg: Arg) !void {
        const f = &self.l.fp4;
        const r = self.l.r;
        const s = self.rig.s.handle;
        switch (tile) {
            .stream => try f.tileGo(r, self.reg.fmt, arg, .{}, self.items, s),
            .wmma => try f.promptGo(r, self.reg.fmt, arg, self.items, 1 << 20, s),
            .item => try f.promptGo(r, self.reg.fmt, arg, self.items, 1, s),
            .pair => try f.pairGo(r, self.reg.fmt, arg, self.items, 0, s),
        }
        try self.rig.s.synchronize();
    }

    /// One launch of the registration's tile on `arg` (the identity checks' row counts and cuts).
    pub fn launchWith(self: *Case, arg: Arg) !void {
        try self.launch(self.reg.tile, arg);
    }

    pub fn run(self: *Case) !void {
        try self.launch(self.reg.tile, self.arg);
    }

    /// The 128-row matrix tile's bytes on the same inputs (then moe_act's, for the pair tile).
    pub fn replaced(self: *Case) ![]const u8 {
        var plain = self.arg;
        plain.out = self.base;
        plain.out16 = 0;
        try self.launch(.wmma, plain);
        if (self.reg.tile != .pair) return self.rig.bytes(self.base, self.count() * 4);
        const width = self.shape.n / 2;
        try self.l.tf_moe_act(@ptrFromInt(self.base), @ptrFromInt(self.base16), 2, @intCast(self.shape.rows), @intCast(width), 0, self.rig.s.handle);
        return self.rig.bytes(self.base16, self.count() * 2);
    }

    /// Outputs: (rows, n) sums, or the pair tile's (rows, n / 2) activations.
    pub fn count(self: *Case) usize {
        return self.shape.rows * if (self.reg.tile == .pair) self.shape.n / 2 else self.shape.n;
    }

    /// Bytes of a row's outputs.
    pub fn rowBytes(self: *const Case) usize {
        return if (self.reg.tile == .pair) self.shape.n else self.shape.n * 4;
    }

    /// The output buffer of a launch (fp32 out, or the pair tile's act).
    pub fn outOf(self: *const Case, arg: Arg) u64 {
        return if (self.reg.tile == .pair) arg.out16 else arg.out;
    }

    fn terms(self: *Case, p: usize, j: usize, t: []f64) void {
        const k = self.shape.k;
        const w = self.w[(self.expert_of[p] * self.shape.n + j) * k ..][0..k];
        for (t, w, self.x[p * k ..][0..k]) |*o, wv, xv| o.* = c.fromBf16(xv) * wv;
    }

    pub fn reference(self: *Case, want: []f64, mag: []f64) !void {
        const t = try self.arena.allocator().alloc(f64, self.shape.k);
        if (self.reg.tile == .pair) {
            const width = self.shape.n / 2;
            for (0..self.shape.rows) |p| for (0..width) |f| {
                var gu: [2]f64 = undefined;
                for (&gu, [_]usize{ f, width + f }) |*v, j| {
                    self.terms(p, j, t);
                    v.* = 0;
                    for (t) |x| v.* += x;
                }
                want[p * width + f] = gu[0] / (1 + @exp(-gu[0])) * gu[1];
                mag[p * width + f] = @abs(want[p * width + f]);
            };
            return;
        }
        for (0..self.shape.rows) |p| for (0..self.shape.n) |j| {
            self.terms(p, j, t);
            var sum: f64 = 0;
            var abs: f64 = 0;
            for (t) |v| {
                sum += v;
                abs += @abs(v);
            }
            want[p * self.shape.n + j] = sum;
            mag[p * self.shape.n + j] = abs;
        };
    }

    pub fn simulate(self: *Case, acc: c.Accum, order: g.Order, out: []f64) !void {
        const t = try self.arena.allocator().alloc(f64, self.shape.k);
        for (0..self.shape.rows) |p| for (0..self.shape.n) |j| {
            self.terms(p, j, t);
            out[p * self.shape.n + j] = rows.sumIn(acc, order, t);
        };
    }
};
