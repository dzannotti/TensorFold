//! Flash Next's Loom prompt DeltaNet recurrence under the HIP contract: byte-equal to tf_gated_delta's hipcc route.
const std = @import("std");
const hip = @import("hip");
const core = @import("core");
const rig_ = @import("contract_rig.zig");
const kit = @import("contract_kit.zig");
const gdn_k = @import("contract_gdn.zig");

const contract = core.contract;
const c = contract.spec;
const g = contract.gates;
const Part = gdn_k.Part;

/// Flash Next's 48 value heads from one row to a whole stage seam past 1,024, and two key heads over 97 tokens.
const served = [_]c.Shape{
    .{ .rows = 1, .n = 48, .k = 128 },  .{ .rows = 7, .n = 48, .k = 128 },   .{ .rows = 63, .n = 48, .k = 128 },
    .{ .rows = 64, .n = 48, .k = 128 }, .{ .rows = 300, .n = 48, .k = 128 }, .{ .rows = 1031, .n = 48, .k = 128 },
    .{ .rows = 97, .n = 6, .k = 128 },
};

/// gdn_recur: shape.rows tokens over shape.n value heads, dk = dv = 128, the state updated in place.
pub const Recur = struct {
    rig: rig_.Rig,
    l: *const hip.Launcher,
    part: Part,
    arena: std.heap.ArenaAllocator,
    shape: c.Shape = undefined,
    p: gdn_k.Problem = undefined,
    in: [5]u64 = @splat(0),
    y: u64 = 0,
    st: u64 = 0,
    st_bytes: []u8 = &.{},
    base: [2]u64 = @splat(0), // the hipcc route's y and state

    pub fn init(a: std.mem.Allocator, rt: *const hip.Runtime, s: hip.Stream, l: *const hip.Launcher, part: Part) Recur {
        return .{ .rig = .{ .a = a, .r = rt, .s = s }, .l = l, .part = part, .arena = .init(a) };
    }

    pub fn deinit(self: *Recur) void {
        self.arena.deinit();
        self.rig.deinit();
    }

    pub fn kernel(self: *Recur) g.Kernel {
        return .{
            .spec = .{
                .name = if (self.part == .y) "loom gdn_recur (y)" else "loom gdn_recur (the state)",
                .replaces = "hip tf_gdn_prompt_128 (64 rows and more), gated_delta_tile<128, 8, 16> (fewer)",
                .accum = .f32,
                .depth = gdn_k.depth,
                .out = .f32,
                .tiles = .{ .{ .size = 8 }, .{}, .{} },
                .served = &served,
            },
            .ctx = self,
            .vt = &vt,
        };
    }

    const vt = g.hooks(Recur);

    pub fn prepare(self: *Recur, shape: c.Shape, fill: g.Fill, seed: u64) !bool {
        if (shape.k != 128 or shape.rows == 0 or shape.n == 0 or shape.n % gdn_k.group != 0) return false;
        self.rig.reset();
        _ = self.arena.reset(.retain_capacity);
        const a = self.arena.allocator();
        self.shape = shape;
        self.p = try gdn_k.Problem.make(a, shape.rows, shape.n, 128, fill, seed, .exact);
        self.in = try gdn_k.upload(&self.rig, a, &self.p);
        self.st_bytes = try kit.f32Bytes(a, self.p.state);
        const yb = shape.rows * shape.n * 128 * 4;
        const sb = self.st_bytes.len;
        // the other part's output is scratch: the gates read one part's bytes
        self.y = if (self.part == .y) try self.rig.output(yb) else try self.rig.scratch(yb);
        self.st = if (self.part == .state) try self.rig.output(sb) else try self.rig.scratch(sb);
        self.base = .{ try self.rig.scratch(yb), try self.rig.scratch(sb) };
        return true;
    }

    /// The state back to the prepared one, then one launch: half a value head a block.
    pub fn run(self: *Recur) !void {
        try self.rig.put(self.st, self.st_bytes);
        const in = self.in;
        try self.l.gdn_loom.run(self.l.r, @ptrFromInt(in[0]), @ptrFromInt(in[1]), @ptrFromInt(in[2]), @ptrFromInt(in[3]), @ptrFromInt(in[4]), @ptrFromInt(self.st), @ptrFromInt(self.y), @intCast(self.shape.rows), @intCast(self.p.kh()), @intCast(self.shape.n), self.rig.s.handle);
        try self.rig.s.synchronize();
    }

    /// The hipcc route from the prepared state over the same tokens: tf_gdn_prompt_128 from 64 rows, else the tile.
    pub fn replaced(self: *Recur) ![]const u8 {
        try self.rig.put(self.base[1], self.st_bytes);
        if (self.shape.rows >= 64) {
            var a: hip.launch.Args = .{};
            for ([_]u64{ self.in[0], self.in[1], self.in[2], self.in[3], self.in[4], self.base[1], self.base[0] }) |x| try a.add(x);
            for ([_]usize{ self.shape.rows, self.p.kh(), self.shape.n }) |x| try a.add(@as(c_int, @intCast(x)));
            const grid: hip.launch.Dim3 = .{ .x = 2, .y = @intCast(self.shape.n) };
            try hip.launch.launch(self.l.gdn_prompt, .{ .grid = grid, .block = .{ .x = 32, .y = 4 } }, self.rig.s, &a);
        } else try gdn_k.plain(self.l, &self.rig, self.in, self.base[1], self.base[0], &self.p, self.shape.rows);
        try self.rig.s.synchronize();
        if (self.part == .y) return self.rig.bytes(self.base[0], self.count() * 4);
        return self.rig.bytes(self.base[1], self.st_bytes.len);
    }

    pub fn count(self: *Recur) usize {
        return if (self.part == .y) self.shape.rows * self.shape.n * 128 else self.p.state.len;
    }

    pub fn reference(self: *Recur, want: []f64, mag: []f64) !void {
        return gdn_k.partOf(self.rig.a, &self.p, self.shape.rows, self.part, want, mag, null);
    }

    pub fn simulate(self: *Recur, acc: c.Accum, order: g.Order, out: []f64) !void {
        const mag = try self.rig.a.alloc(f64, out.len);
        defer self.rig.a.free(mag);
        return gdn_k.partOf(self.rig.a, &self.p, self.shape.rows, self.part, out, mag, .{ acc, order });
    }
};
