//! A prompt's next chunk staged while the chunk before it runs: its n-gram gather leaves the gap between chunks.
const std = @import("std");
const hip = @import("hip");
const model = @import("model.zig");
const tables = @import("tables.zig");
const cache = @import("cache.zig");
const forward = @import("forward.zig");

/// The chunk staged ahead: whose caches, the position and n-gram history it starts from, its rows and span.
const Pre = struct { c: *const cache.Caches, pos: usize, history: [2]i64, rows: usize, which: u1, staged: forward.Staged };

/// Two stagings a chunk's inputs take turns in: one the forward in flight reads, the other the next chunk's.
pub const Ahead = struct {
    hosts: [2]forward.Host,
    /// The staged chunk's tokens: a later chunk takes the span only with the same tokens.
    ids: []u32,
    pre: ?Pre = null,
    /// The staging the forward in flight reads (null: neither).
    busy: ?u1 = null,

    /// Stagings for chunks of up to `rows` rows.
    pub fn init(gpa: std.mem.Allocator, d: *const hip.Runtime, m: *const model.Model, t: *tables.Tables, rows: usize) !Ahead {
        const bytes = forward.layout(m, t, rows).len;
        const ids = try gpa.alloc(u32, rows);
        errdefer gpa.free(ids);
        var first = try hip.HostBuffer.alloc(d, bytes);
        errdefer first.free();
        const second = try hip.HostBuffer.alloc(d, bytes);
        return .{
            .hosts = .{
                .{ .gpa = gpa, .d = d, .staging = first, .tables = t },
                .{ .gpa = gpa, .d = d, .staging = second, .tables = t },
            },
            .ids = ids,
        };
    }

    pub fn deinit(a: *Ahead, gpa: std.mem.Allocator) void {
        for (&a.hosts) |*h| h.deinit();
        gpa.free(a.ids);
        a.* = undefined;
    }

    /// The span staged for `tokens` from `c`'s end, if the last `stage` was this chunk; the stream is synchronized.
    pub fn take(a: *Ahead, c: *const cache.Caches, tokens: []const u32) ?forward.Staged {
        a.busy = null;
        const p = a.pre orelse return null;
        a.pre = null;
        if (p.c != c or p.pos != c.len or !std.mem.eql(i64, &p.history, &c.history)) return null;
        if (!std.mem.eql(u32, a.ids[0..p.rows], tokens)) return null;
        a.busy = p.which;
        return p.staged;
    }

    /// `tokens`, the chunk after the forward in flight on `c`, staged where that forward does not read.
    pub fn stage(a: *Ahead, m: *const model.Model, c: *cache.Caches, tokens: []const u32) !void {
        a.pre = null;
        if (tokens.len == 0 or tokens.len > a.ids.len) return;
        const which: u1 = if (a.busy) |b| ~b else 0;
        const h = &a.hosts[which];
        h.used = 0;
        const segs = [_]forward.Seg{.{ .c = c, .tokens = tokens, .pos = c.len }};
        var staged = try forward.stage(m, h, &segs, false);
        try forward.gather(h, &staged);
        @memcpy(a.ids[0..tokens.len], tokens);
        a.pre = .{ .c = c, .pos = c.len, .history = c.history, .rows = tokens.len, .which = which, .staged = staged };
    }
};

test "a chunk takes the span staged ahead only for its caches, position, history and tokens, and stages around it" {
    var ids: [4]u32 = undefined;
    var a: Ahead = .{ .hosts = undefined, .ids = &ids };
    var c: cache.Caches = undefined;
    var other: cache.Caches = undefined;
    c.len = 64;
    c.history = .{ 7, 9 };
    const tokens = [_]u32{ 1, 2, 3 };
    const staged: forward.Staged = .{ .from = 0, .len = 48 };
    const at = struct {
        fn pre(x: *Ahead, cs: *const cache.Caches, t: []const u32, which: u1, s: forward.Staged) void {
            @memcpy(x.ids[0..t.len], t);
            x.pre = .{ .c = cs, .pos = 64, .history = .{ 7, 9 }, .rows = t.len, .which = which, .staged = s };
        }
    }.pre;
    at(&a, &c, &tokens, 1, staged);
    try std.testing.expectEqual(@as(usize, 48), a.take(&c, &tokens).?.len);
    try std.testing.expectEqual(@as(?u1, 1), a.busy);
    try std.testing.expect(a.take(&c, &tokens) == null); // taken once
    try std.testing.expectEqual(@as(?u1, null), a.busy);
    at(&a, &c, &tokens, 0, staged);
    try std.testing.expect(a.take(&other, &tokens) == null);
    at(&a, &c, &tokens, 0, staged);
    try std.testing.expect(a.take(&c, &.{ 1, 2, 4 }) == null);
    at(&a, &c, &tokens, 0, staged);
    try std.testing.expect(a.take(&c, &.{ 1, 2 }) == null);
    at(&a, &c, &tokens, 0, staged);
    c.history = .{ 7, 8 };
    try std.testing.expect(a.take(&c, &tokens) == null);
    c.history = .{ 7, 9 };
    at(&a, &c, &tokens, 0, staged);
    c.len = 65;
    try std.testing.expect(a.take(&c, &tokens) == null);
    // a chunk longer than the stagings, or none, stages nothing and drops what was staged
    at(&a, &c, &tokens, 0, staged);
    try a.stage(undefined, &c, &.{ 1, 2, 3, 4, 5 });
    try std.testing.expect(a.pre == null);
}
