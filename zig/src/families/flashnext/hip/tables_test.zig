//! The MLX n-gram gather against a searched copy: the sizes' fast path, threads, the parked pool and spans.
const std = @import("std");
const tables = @import("tables.zig");
const host_pages = @import("../host_pages.zig");
const pool = @import("../host_pool.zig");
const Rows = tables.Rows;
const Shard = tables.Shard;
const Workers = tables.Workers;
const max_parts = pool.max_parts;

/// The gather before the sizes were read off `starts`: one row at a time, its shard searched for.
fn searchedGather(r: *const Rows, ids: []const u32, words: []u8, scales: []u8, biases: []u8) !void {
    const wb = r.wordBytes();
    const tb = r.tableBytes();
    if (words.len < ids.len * wb or scales.len < ids.len * tb or biases.len < ids.len * tb)
        return error.StagingTooSmall;
    for (ids, 0..) |id, i| {
        const s = std.sort.upperBound(usize, r.starts, @as(usize, id), struct {
            fn order(x: usize, y: usize) std.math.Order {
                return std.math.order(x, y);
            }
        }.order) - 1;
        if (s >= r.shards.len) return error.NgramRowOutOfRange;
        const row = id - r.starts[s];
        const sh = r.shards[s];
        @memcpy(words[i * wb ..][0..wb], sh.words[row * wb ..][0..wb]);
        @memcpy(scales[i * tb ..][0..tb], sh.scales[row * tb ..][0..tb]);
        @memcpy(biases[i * tb ..][0..tb], sh.biases[row * tb ..][0..tb]);
    }
}

/// Synthetic shards of `sizes` rows (`dims` values a row at `bits`, groups of `group`): bytes hashed from their offset.
const Fake = struct {
    shards: []Shard,
    starts: []usize,
    arrays: [3][]u8,
    rows: Rows,
    /// The gather's parked workers (a pool over `io`) and the page advice over `spans`.
    workers: Workers,
    spans: []const []const u8,

    fn init(a: std.mem.Allocator, sizes: []const usize, dims: usize, bits: u8, group: usize) !Fake {
        return initOn(a, sizes, dims, bits, group, null);
    }

    /// The same shards gathered by `threads` parked workers (null: the caller alone, as without a pool).
    fn initOn(a: std.mem.Allocator, sizes: []const usize, dims: usize, bits: u8, group: usize, threads: ?std.Io) !Fake {
        var f: Fake = .{
            .shards = try a.alloc(Shard, sizes.len),
            .starts = try a.alloc(usize, sizes.len + 1),
            .arrays = .{ &.{}, &.{}, &.{} },
            .rows = undefined,
            .workers = undefined,
            .spans = &.{},
        };
        const wb = dims * bits / 8;
        const tb = dims / group * 2;
        f.starts[0] = 0;
        for (sizes, 0..) |n, i| f.starts[i + 1] = f.starts[i] + n;
        const total = f.starts[sizes.len];
        for (&f.arrays, [_]usize{ wb, tb, tb }, [_]u8{ 7, 3, 11 }) |*arr, width, step| {
            arr.* = try a.alloc(u8, total * width);
            for (arr.*, 0..) |*b, i| b.* = @truncate(i *% step +% width);
        }
        for (sizes, 0..) |n, i| {
            const from = f.starts[i];
            f.shards[i] = .{
                .words = f.arrays[0][from * wb ..][0 .. n * wb],
                .scales = f.arrays[1][from * tb ..][0 .. n * tb],
                .biases = f.arrays[2][from * tb ..][0 .. n * tb],
            };
        }
        const split = Rows.split(f.starts);
        f.rows = .{
            .shards = f.shards,
            .starts = f.starts,
            .wide = split.wide,
            .narrow = split.narrow,
            .extra = split.extra,
            .dims = dims,
            .bits = bits,
            .group = group,
        };
        f.spans = try f.rows.spans(a);
        f.workers = .{ .warmer = try host_pages.Warmer.init(a, f.spans) };
        if (threads) |io| f.workers.pool = try pool.Pool.create(a, io);
        return f;
    }

    fn deinit(f: *Fake, a: std.mem.Allocator) void {
        if (f.workers.pool) |p| p.destroy();
        f.workers.warmer.deinit(a);
        for (f.arrays) |arr| a.free(arr);
        a.free(f.spans);
        a.free(f.shards);
        a.free(f.starts);
    }

    fn gathered(f: *Fake, a: std.mem.Allocator, ids: []const u32) !Staging {
        const s: Staging = .{
            .words = try a.alloc(u8, ids.len * f.rows.wordBytes()),
            .scales = try a.alloc(u8, ids.len * f.rows.tableBytes()),
            .biases = try a.alloc(u8, ids.len * f.rows.tableBytes()),
        };
        @memset(s.words, 0);
        @memset(s.scales, 0);
        @memset(s.biases, 0);
        try f.rows.gather(&f.workers, ids, s.words, s.scales, s.biases);
        return s;
    }

    const Staging = struct {
        words: []u8,
        scales: []u8,
        biases: []u8,
        fn deinit(s: *const Staging, a: std.mem.Allocator) void {
            a.free(s.words);
            a.free(s.scales);
            a.free(s.biases);
        }
    };
};

test "a row's shard comes from the sizes alone: every id of a remainder split finds the searched shard" {
    const a = std.testing.allocator;
    // 128 shards of 250 rows with 3 of 251, as a checkpoint's padded row count splits
    var sizes: [128]usize = @splat(250);
    for (sizes[0..3]) |*n| n.* = 251;
    var ids: std.ArrayList(u32) = .empty;
    defer ids.deinit(a);
    var total: usize = 0;
    for (sizes) |n| total += n;
    for (0..total) |id| try ids.append(a, @intCast(id));
    var f = try Fake.init(a, &sizes, 160, 4, 32);
    defer f.deinit(a);
    try std.testing.expectEqual(@as(usize, 251), f.rows.wide);
    try std.testing.expectEqual(@as(usize, 250), f.rows.narrow);
    try std.testing.expectEqual(@as(usize, 3), f.rows.extra);
    var shard: usize = 0;
    for (ids.items) |id| {
        const at = try f.rows.locate(id);
        if (id == f.starts[shard + 1]) shard += 1;
        try std.testing.expectEqual(shard, at.shard);
        try std.testing.expectEqual(@as(usize, id) - f.starts[shard], at.row);
    }
    try std.testing.expectError(error.NgramRowOutOfRange, f.rows.locate(@intCast(f.starts[128])));
}

test "gather: the sizes' fast path copies the same bytes as a shard search for every id" {
    const a = std.testing.allocator;
    const cases = [_][]const usize{
        &.{},
        &.{97},
        &.{ 40, 40, 40, 40 },
        &.{ 41, 41, 40, 40 },
        &.{ 40, 39, 41, 40 },
        &.{ 250, 250, 250 },
    };
    for (cases) |sizes| {
        var ids: [256]u32 = undefined;
        var n: usize = 0;
        var total: usize = 0;
        for (sizes) |s| total += s;
        while (n < ids.len and n < total) : (n += 1) ids[n] = @intCast((n * 37) % total);
        var f = try Fake.init(a, sizes, 160, 4, 32);
        defer f.deinit(a);
        const want = try a.alloc(u8, n * f.rows.wordBytes());
        defer a.free(want);
        const want_scales = try a.alloc(u8, n * f.rows.tableBytes());
        defer a.free(want_scales);
        const want_biases = try a.alloc(u8, n * f.rows.tableBytes());
        defer a.free(want_biases);
        try searchedGather(&f.rows, ids[0..n], want, want_scales, want_biases);
        const got = try f.gathered(a, ids[0..n]);
        defer got.deinit(a);
        try std.testing.expectEqualSlices(u8, want, got.words);
        try std.testing.expectEqualSlices(u8, want_scales, got.scales);
        try std.testing.expectEqualSlices(u8, want_biases, got.biases);
    }
}

test "gather: rows copied by several threads equal one thread's, and a bad row or a short staging is refused" {
    const a = std.testing.allocator;
    var sizes: [64]usize = @splat(300);
    for (sizes[0..8]) |*s| s.* = 301;
    var total: usize = 0;
    for (sizes) |n| total += n;
    var ids: [8192]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast((i * 7919) % total);
    var f = try Fake.init(a, &sizes, 160, 4, 32);
    defer f.deinit(a);
    const want = try a.alloc(u8, ids.len * f.rows.wordBytes());
    defer a.free(want);
    const want_scales = try a.alloc(u8, ids.len * f.rows.tableBytes());
    defer a.free(want_scales);
    const want_biases = try a.alloc(u8, ids.len * f.rows.tableBytes());
    defer a.free(want_biases);
    try searchedGather(&f.rows, &ids, want, want_scales, want_biases);
    const got = try f.gathered(a, &ids);
    defer got.deinit(a);
    try std.testing.expectEqualSlices(u8, want, got.words);
    try std.testing.expectEqualSlices(u8, want_scales, got.scales);
    try std.testing.expectEqualSlices(u8, want_biases, got.biases);
    // an id past the last row, and staging that cannot hold the rows the ids ask for
    var past = ids;
    past[0] = @intCast(f.starts[sizes.len]);
    const small = try a.alloc(u8, ids.len * f.rows.wordBytes());
    defer a.free(small);
    try std.testing.expectError(error.NgramRowOutOfRange, f.rows.gather(&f.workers, &past, small, want_scales, want_biases));
    const short = small[0 .. small.len - 1];
    try std.testing.expectError(error.StagingTooSmall, f.rows.gather(&f.workers, &ids, short, want_scales, want_biases));
}

test "gather: the parked pool's parts copy the same bytes as one thread for every id count" {
    const a = std.testing.allocator;
    var sizes: [48]usize = @splat(120);
    for (sizes[0..4]) |*s| s.* = 121;
    var total: usize = 0;
    for (sizes) |n| total += n;
    var all: [8193]u32 = undefined;
    for (&all, 0..) |*id, i| id.* = @intCast((i * 7919) % total);
    var one = try Fake.init(a, &sizes, 160, 4, 32);
    defer one.deinit(a);
    var many = try Fake.initOn(a, &sizes, 160, 4, 32, std.testing.io);
    defer many.deinit(a);
    for ([_]usize{ 0, 1, 16, 17, 256, 4096, 8192 }) |n| {
        const ids = all[0..n];
        // the pool's copy, with a fresh seen set each call so every row's pages are asked for
        many.workers.warmer.deinit(a);
        many.workers.warmer = try host_pages.Warmer.init(a, many.spans);
        const got = try many.gathered(a, ids);
        defer got.deinit(a);
        // one thread's, on a warmer that never asks
        const want = try a.alloc(u8, n * one.rows.wordBytes());
        defer a.free(want);
        const want_scales = try a.alloc(u8, n * one.rows.tableBytes());
        defer a.free(want_scales);
        const want_biases = try a.alloc(u8, n * one.rows.tableBytes());
        defer a.free(want_biases);
        try searchedGather(&one.rows, ids, want, want_scales, want_biases);
        try std.testing.expectEqualSlices(u8, want, got.words);
        try std.testing.expectEqualSlices(u8, want_scales, got.scales);
        try std.testing.expectEqualSlices(u8, want_biases, got.biases);
    }
    try std.testing.expectEqual(@as(usize, max_parts - 1), many.workers.pool.?.spawned);
}

test "gather: a cold call wakes the pool and the same ids gathered again stay on the caller" {
    const a = std.testing.allocator;
    var sizes: [4]usize = @splat(300);
    var ids: [64]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i * 7919 % 1200);
    var f = try Fake.initOn(a, &sizes, 160, 4, 32, std.testing.io);
    defer f.deinit(a);
    var got: Fake.Staging = .{
        .words = try a.alloc(u8, ids.len * f.rows.wordBytes()),
        .scales = try a.alloc(u8, ids.len * f.rows.tableBytes()),
        .biases = try a.alloc(u8, ids.len * f.rows.tableBytes()),
    };
    defer got.deinit(a);
    const want = try a.alloc(u8, ids.len * f.rows.wordBytes());
    defer a.free(want);
    const want_scales = try a.alloc(u8, ids.len * f.rows.tableBytes());
    defer a.free(want_scales);
    const want_biases = try a.alloc(u8, ids.len * f.rows.tableBytes());
    defer a.free(want_biases);
    try searchedGather(&f.rows, &ids, want, want_scales, want_biases);
    try std.testing.expect(f.workers.cold(&f.rows, &ids));
    try f.rows.gather(&f.workers, &ids, got.words, got.scales, got.biases);
    try std.testing.expect(!f.workers.cold(&f.rows, &ids));
    try std.testing.expectEqualSlices(u8, want, got.words);
    try std.testing.expectEqualSlices(u8, want_scales, got.scales);
    try std.testing.expectEqualSlices(u8, want_biases, got.biases);
    const cold_calls = f.workers.pool.?.calls.load(.monotonic);
    try std.testing.expect(cold_calls >= 1);
    try f.rows.gather(&f.workers, &ids, got.words, got.scales, got.biases);
    try std.testing.expectEqualSlices(u8, want, got.words);
    try std.testing.expectEqualSlices(u8, want_scales, got.scales);
    try std.testing.expectEqualSlices(u8, want_biases, got.biases);
    try std.testing.expectEqual(cold_calls, f.workers.pool.?.calls.load(.monotonic));
}

test "spans: one byte range for each of the three tensors of every shard" {
    const a = std.testing.allocator;
    var ids: [4]u32 = undefined;
    for (&ids, 0..) |*id, i| id.* = @intCast(i);
    var f = try Fake.init(a, &.{ 2, 2, 2 }, 160, 4, 32);
    defer f.deinit(a);
    const ranges = try f.rows.spans(a);
    defer a.free(ranges);
    try std.testing.expectEqual(@as(usize, 9), ranges.len);
    for (ranges, 0..) |r, i| {
        const shard = f.shards[i / 3];
        try std.testing.expectEqual(r.ptr, switch (i % 3) {
            0 => shard.words.ptr,
            1 => shard.scales.ptr,
            else => shard.biases.ptr,
        });
    }
}
