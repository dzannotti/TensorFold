//! NVFP4 routed experts' borrowed planes read and refused, then packed on threads by a backend's packer.
const std = @import("std");
const core = @import("core");
const slice = core.nvfp4_slice;

pub const max_threads = 32;

/// A projection's borrowed planes: FP4 codes, plain E4M3 scales or the CSF planes, the global and its bytes.
pub const View = struct { codes: []const u8, scales: []const u8 = &.{}, fixed: []const u8 = &.{}, exceptions: []const u8 = &.{}, global: f32, global_bytes: []const u8 = &.{} };

/// The store as slice.stored reads it, every tensor's pages read ahead (this loader copies all of each).
fn Ahead(comptime S: type) type {
    return struct {
        store: S,

        pub fn get(self: @This(), name: []const u8) !core.safetensors.Tensor {
            return self.store.holdAhead(name);
        }

        pub fn has(self: @This(), name: []const u8) bool {
            return self.store.has(name);
        }
    };
}

/// The store as slice.stored reads it, no pages read ahead (a direct load reads the planes past the page cache).
fn Direct(comptime S: type) type {
    return struct {
        store: S,

        pub fn get(self: @This(), name: []const u8) !core.safetensors.Tensor {
            return self.store.hold(name);
        }

        pub fn has(self: @This(), name: []const u8) bool {
            return self.store.has(name);
        }
    };
}

/// A projection read and refused as loadShape's loop reads it; `direct` reads no plane bytes (gathered, then checked).
pub fn viewOf(store: anytype, name: []const u8, n: usize, k: usize, comptime direct: bool) !View {
    var buf: [512]u8 = undefined;
    const w = try store.hold(try std.fmt.bufPrint(&buf, "{s}.weight", .{name}));
    if (!w.is(.u8, &.{ n, k / 2 })) return error.UnexpectedTensor;
    const source = try slice.stored(if (direct) Direct(@TypeOf(store)){ .store = store } else Ahead(@TypeOf(store)){ .store = store }, name);
    // direct: the CSF planes are checked on their gathered bytes, as scalesOf decodes them before packing
    _ = try if (direct) slice.shapes(source) else slice.dimensions(source);
    if (source.global.bytes.len < 4) return error.UnexpectedTensor;
    var v: View = switch (source.scales) {
        .plain => |s| .{ .codes = source.weight.bytes, .scales = s.bytes, .global = 0 },
        .csf => |s| .{ .codes = source.weight.bytes, .fixed = s.fixed.bytes, .exceptions = s.exceptions.bytes, .global = 0 },
    };
    v.global_bytes = source.global.bytes[0..4];
    if (!direct) try globalOf(&v);
    return v;
}

/// The view's global from its bytes, refused unless finite and positive.
pub fn globalOf(v: *View) !void {
    const global: f32 = @bitCast(std.mem.readInt(u32, v.global_bytes[0..4], .little));
    if (!std.math.isFinite(global) or global <= 0) return error.UnsupportedQuantization;
    v.global = global;
}

/// A view's row-major E4M3 scales [n, k/16]: the plain plane, or its CSF planes decoded whole as slice.slice does.
pub fn scalesOf(v: View, n: usize, k: usize, buf: []u8) ![]const u8 {
    if (v.scales.len > 0) return v.scales;
    const plane = try core.nvfp4_csf.Plane.init(n, k / 16, v.fixed, v.exceptions);
    try plane.decode(buf[0 .. n * (k / 16)], 0, n, 0, k / 16);
    return buf[0 .. n * (k / 16)];
}

/// Experts [lo, hi) of `views` (gate, up, down each) packed by `Packer` as loadShape's loop packs them.
fn Work(comptime Packer: type) type {
    return struct {
        views: []const View,
        up: []u32,
        down: []u32,
        width: usize,
        dims: usize,
        lo: usize,
        hi: usize,
        failed: ?(error{OutOfMemory} || core.nvfp4_csf.Error) = null,

        fn run(w: *@This()) void {
            w.pack() catch |err| {
                w.failed = err;
            };
        }

        fn pack(w: *@This()) !void {
            const count = w.views.len / 3;
            const up_words = w.up.len / count;
            const down_words = w.down.len / count;
            const each = @max(w.width * (w.dims / 16), w.dims * (w.width / 16));
            const bufs = try std.heap.page_allocator.alloc(u8, 3 * each);
            defer std.heap.page_allocator.free(bufs);
            for (w.lo..w.hi) |i| {
                const g = w.views[3 * i];
                const u = w.views[3 * i + 1];
                const o = w.views[3 * i + 2];
                const gs = try scalesOf(g, w.width, w.dims, bufs[0..each]);
                const us = try scalesOf(u, w.width, w.dims, bufs[each..][0..each]);
                const os = try scalesOf(o, w.dims, w.width, bufs[2 * each ..][0..each]);
                Packer.packGateUp(w.up[i * up_words ..][0..up_words], .{ g.codes, gs }, .{ u.codes, us }, w.width, w.dims);
                Packer.packOne(w.down[i * down_words ..][0..down_words], o.codes, os, w.dims, w.width);
            }
        }
    };
}

/// Every expert's gate / up and down words by `Packer` on `n` threads; a thread that cannot start packs here.
pub fn pack(comptime Packer: type, views: []const View, up: []u32, down: []u32, width: usize, dims: usize, n: usize) !void {
    const count = views.len / 3;
    if (count == 0 or views.len != 3 * count or up.len % count != 0 or down.len % count != 0) return error.UnexpectedTensor;
    const t_n = @max(1, @min(n, max_threads, count));
    const per = (count + t_n - 1) / t_n;
    var works: [max_threads]Work(Packer) = undefined;
    var handles: [max_threads]?std.Thread = @splat(null);
    for (0..t_n) |t| {
        works[t] = .{ .views = views, .up = up, .down = down, .width = width, .dims = dims, .lo = @min(count, t * per), .hi = @min(count, (t + 1) * per) };
        handles[t] = std.Thread.spawn(.{}, Work(Packer).run, .{&works[t]}) catch blk: {
            works[t].run();
            break :blk null;
        };
    }
    for (handles[0..t_n]) |h| if (h) |th| th.join();
    for (works[0..t_n]) |w| if (w.failed) |err| return err;
}

/// Each word derived from its own expert's plane bytes, so a word written for the wrong expert shows.
const Tagged = struct {
    fn packGateUp(out: []u32, gate: [2][]const u8, up: [2][]const u8, n: usize, k: usize) void {
        for (out, 0..) |*o, i| o.* = @as(u32, gate[0][0]) + up[1][0] + @as(u32, @intCast(i + n + k));
    }

    fn packOne(out: []u32, codes: []const u8, scales: []const u8, n: usize, k: usize) void {
        for (out, 0..) |*o, i| o.* = @as(u32, codes[0]) * 256 + scales[0] + @as(u32, @intCast(i + n + k));
    }
};

test "Flash Next expert loader: every thread count packs each expert into its own words" {
    const count = 33;
    var planes: [3 * count][2]u8 = undefined;
    var views: [3 * count]View = undefined;
    for (&views, &planes, 0..) |*v, *p, j| {
        p.* = .{ @truncate(j), @truncate(j * 7 + 1) };
        v.* = .{ .codes = p[0..1], .scales = p[1..2], .global = 1.0 };
    }
    var want_up: [count * 4]u32 = undefined;
    var want_down: [count * 2]u32 = undefined;
    for (0..count) |i| {
        Tagged.packGateUp(want_up[i * 4 ..][0..4], .{ views[3 * i].codes, views[3 * i].scales }, .{ views[3 * i + 1].codes, views[3 * i + 1].scales }, 64, 64);
        Tagged.packOne(want_down[i * 2 ..][0..2], views[3 * i + 2].codes, views[3 * i + 2].scales, 64, 64);
    }
    for ([_]usize{ 1, 2, 3, 4, 8, 32 }) |n| {
        var up: [count * 4]u32 = @splat(0xdeadbeef);
        var down: [count * 2]u32 = @splat(0xdeadbeef);
        try pack(Tagged, &views, &up, &down, 64, 64, n);
        try std.testing.expectEqualSlices(u32, &want_up, &up);
        try std.testing.expectEqualSlices(u32, &want_down, &down);
    }
}

test "Flash Next expert loader: a pack without whole experts or with uneven buffers is refused" {
    const views: [6]View = @splat(.{ .codes = &.{}, .global = 1.0 });
    var up: [7]u32 = undefined;
    var down: [4]u32 = undefined;
    try std.testing.expectError(error.UnexpectedTensor, pack(Tagged, views[0..0], &up, &down, 64, 64, 2));
    try std.testing.expectError(error.UnexpectedTensor, pack(Tagged, views[0..2], &up, &down, 64, 64, 2));
    try std.testing.expectError(error.UnexpectedTensor, pack(Tagged, views[0..6], &up, &down, 64, 64, 2));
}
