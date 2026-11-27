//! One GPU's NVFP4 routed experts packed on threads from the store's mapped planes, through its pinned staging.
const std = @import("std");
const core = @import("core");
const cuda = @import("cuda");
const experts = @import("cuda_modelopt_experts.zig");
const ex = cuda.experts.Nvfp4Experts;
const nvfp4 = @import("nvfp4_read.zig");

pub const default_threads = nvfp4.default_threads;
pub const max_threads = nvfp4.max_threads;

/// Pinned bytes a load stages at most, a whole number of experts' words at a time.
pub const staging_bytes: usize = 192 << 20;

pub const parseThreads = nvfp4.parseThreads;
pub const configured = nvfp4.configured;

pub const View = nvfp4.View;

/// Readers a direct load gathers each chunk's planes with (the NVMe's O_DIRECT rate holds from 4 streams).
const direct_readers = 4;

/// A store's hook at chunk k's gather start (`gather`) or pack end (a test's; the weight store has none).
fn mark(store: anytype, gather: bool, k: usize) void {
    if (comptime @hasDecl(std.meta.Child(@TypeOf(store)), "chunkMark")) store.chunkMark(gather, k);
}

/// The chunk's plane views as the gather repoints them: codes, scales, fixed and exceptions of each.
fn planes(part: []View, ptrs: []*[]const u8) []*[]const u8 {
    for (part, 0..) |*v, j| {
        ptrs[4 * j] = &v.codes;
        ptrs[4 * j + 1] = &v.scales;
        ptrs[4 * j + 2] = &v.fixed;
        ptrs[4 * j + 3] = &v.exceptions;
    }
    return ptrs[0 .. 4 * part.len];
}

/// A store's chunkMark as the gather thread's hook (chunk k's gather start).
fn Marker(comptime S: type) type {
    return struct {
        store: S,
        k: usize = 0,

        fn run(ctx: *anyopaque) void {
            const self: *@This() = @ptrCast(@alignCast(ctx));
            mark(self.store, true, self.k);
        }
    };
}

/// Every expert's gate / up and down words in the CUDA kernels' layout, split over `n` threads.
pub fn pack(views: []const View, up: []u32, down: []u32, width: usize, dims: usize, n: usize) !void {
    return nvfp4.pack(ex, views, up, down, width, dims, n);
}

/// A layer's planes borrowed with their pages read ahead, then packed on `n` threads a chunk of experts at a time.
pub fn load(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, dims: usize, width: usize, count: usize, n: usize) !experts.Owned {
    return loadChunked(a, d, store, prefix, dims, width, count, n, staging_bytes);
}

pub fn loadChunked(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, dims: usize, width: usize, count: usize, n: usize, stage_cap: usize) !experts.Owned {
    // a store that names its mapped shards reads them past the page cache (core.direct_gather; TF_LOAD_DIRECT=0: off)
    if (comptime @hasDecl(std.meta.Child(@TypeOf(store)), "sources")) {
        if (core.direct_gather.enabled()) return loadMode(a, d, store, prefix, dims, width, count, n, stage_cap, true);
    }
    return loadMode(a, d, store, prefix, dims, width, count, n, stage_cap, false);
}

/// loadChunked with its planes read through the mapping (`direct` false) or gathered past the page cache.
pub fn loadMode(a: std.mem.Allocator, d: *const cuda.Driver, store: anytype, prefix: []const u8, dims: usize, width: usize, count: usize, n: usize, stage_cap: usize, comptime direct: bool) !experts.Owned {
    const up_bytes = try experts.packedBytes(width, dims, 2);
    const down_bytes = try experts.packedBytes(dims, width, 1);
    const up_total = std.math.mul(usize, count, up_bytes) catch return error.UnexpectedTensor;
    const down_total = std.math.mul(usize, count, down_bytes) catch return error.UnexpectedTensor;
    if (count == 0 or up_bytes == 0 or down_bytes == 0) return error.UnexpectedTensor;
    const view_count = std.math.mul(usize, 3, count) catch return error.UnexpectedTensor;
    const views = try a.alloc(View, view_count);
    defer a.free(views);
    defer store.release();
    var buf: [512]u8 = undefined;
    for (0..count) |i| {
        views[3 * i] = try nvfp4.viewOf(store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.gate_proj", .{ prefix, i }), width, dims, direct);
        views[3 * i + 1] = try nvfp4.viewOf(store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.up_proj", .{ prefix, i }), width, dims, direct);
        views[3 * i + 2] = try nvfp4.viewOf(store, try std.fmt.bufPrint(&buf, "{s}.experts.{d}.down_proj", .{ prefix, i }), dims, width, direct);
    }
    // two arenas: chunk k packs from one while chunk k + 1 is gathered into the other
    var pair = core.direct_gather.Pair.init(a, direct_readers);
    defer pair.deinit();
    var held: [core.direct_gather.max_sources]core.direct_gather.Source = undefined;
    // a layer whose shards don't fit one gather reads through the mapping, as a file system without O_DIRECT does
    const sources: ?[]core.direct_gather.Source = if (direct) store.sources(&held) catch |err| switch (err) {
        error.TooManySources => null,
    } else null;
    const ptrs = try a.alloc(*[]const u8, if (direct) 4 * view_count else 0);
    defer a.free(ptrs);
    if (direct) {
        // every global first (a 4-byte read each, joined where they sit together), checked as viewOf checks them
        for (views, 0..) |*v, j| ptrs[j] = &v.global_bytes;
        if (sources) |src| try pair.read(src, ptrs[0..view_count]);
        for (views) |*v| try nvfp4.globalOf(v);
    }
    var up = try cuda.DeviceBuffer.alloc(d, up_total);
    errdefer up.free();
    var down = try cuda.DeviceBuffer.alloc(d, down_total);
    errdefer down.free();
    // the store's pinned staging, kept across a load's layers and freed by its dropStaging or deinit on any path
    const chunk = @max(1, @min(count, stage_cap / (up_bytes + down_bytes)));
    const stage = try store.pinned(d, chunk * (up_bytes + down_bytes));
    if (sources) |src| {
        mark(store, true, 0);
        try pair.read(src, planes(views[0 .. 3 * @min(chunk, count)], ptrs));
    }
    var marker: Marker(@TypeOf(store)) = .{ .store = store };
    var lo: usize = 0;
    var k: usize = 0;
    while (lo < count) : ({
        lo += chunk;
        k += 1;
    }) {
        const m = @min(chunk, count - lo);
        // the next chunk's gather runs while this one packs; the main thread allocates nothing from `a` meanwhile
        defer pair.wait() catch {};
        if (sources != null and lo + chunk < count) {
            marker.k = k + 1;
            const hook: core.direct_gather.Hook = .{ .ctx = &marker, .run = @TypeOf(marker).run };
            const ahead = planes(views[3 * (lo + chunk) ..][0 .. 3 * @min(chunk, count - lo - chunk)], ptrs);
            try pair.start(sources.?, ahead, hook);
        }
        const up_words = std.mem.bytesAsSlice(u32, stage[0 .. m * up_bytes]);
        const down_words: []u32 = @alignCast(std.mem.bytesAsSlice(u32, stage[m * up_bytes ..][0 .. m * down_bytes]));
        try pack(views[3 * lo ..][0 .. 3 * m], up_words, down_words, width, dims, n);
        mark(store, false, k);
        // synchronous from pinned memory, as cuda_pocket.zig's staged: done before return, so the next chunk may refill
        try d.check(d.api.cuMemcpyHtoD_v2(try up.at(lo * up_bytes), stage.ptr, m * up_bytes), "cuMemcpyHtoD pinned experts");
        try d.check(d.api.cuMemcpyHtoD_v2(try down.at(lo * down_bytes), stage.ptr + m * up_bytes, m * down_bytes), "cuMemcpyHtoD pinned experts");
        try pair.wait();
        // the next chunk's gather reuses these views' memory: only their globals are read after this
        for (views[3 * lo ..][0 .. 3 * m]) |*v| v.* = .{ .codes = &.{}, .global = v.global };
    }
    const up_globals = try a.alloc(f32, 2 * count);
    defer a.free(up_globals);
    const down_globals = try a.alloc(f32, count);
    defer a.free(down_globals);
    for (0..count) |i| {
        up_globals[2 * i] = views[3 * i].global;
        up_globals[2 * i + 1] = views[3 * i + 1].global;
        down_globals[i] = views[3 * i + 2].global;
    }
    var us = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(up_globals));
    errdefer us.free();
    const ds = try cuda.DeviceBuffer.fromHost(d, std.mem.sliceAsBytes(down_globals));
    return .{ .up = up, .down = down, .up_scale = us, .down_scale = ds, .width = width, .dims = dims, .count = count };
}

test "Flash Next expert loader: TF_FLASHNEXT_LOAD_THREADS takes 1 to 32 and refuses the rest" {
    try std.testing.expectEqual(@as(usize, default_threads), try parseThreads(null));
    try std.testing.expectEqual(@as(usize, 1), try parseThreads("1"));
    try std.testing.expectEqual(@as(usize, max_threads), try parseThreads("32"));
    for ([_][]const u8{ "0", "33", "-1", "8x", "" }) |bad| try std.testing.expectError(error.BadLoadThreads, parseThreads(bad));
}

test {
    _ = @import("cuda_modelopt_threads_test.zig");
    _ = @import("nvfp4_read.zig");
}
