//! A lane round's plan on the device; launches depend only on its `Shape`, so one graph serves any streams and caches.

const std = @import("std");
const hip = @import("hip");
const state = @import("state.zig");

const round_shape = @import("round_shape.zig");

pub const buckets = round_shape.buckets;
pub const min_span = round_shape.min_span;
pub const bucketOf = round_shape.bucketOf;
pub const spanOf = round_shape.spanOf;
pub const Shape = round_shape.Shape;

/// Where each array sits in the plan buffer, in 4-byte words (the 8-byte arrays on even words).
pub const Layout = struct {
    layers: usize,
    tokens: usize,
    pos: usize,
    slot: usize,
    anc: usize,
    first: usize,
    count: usize,
    keep: usize,
    desc: usize,
    rings: usize,
    snaps: usize,
    words: usize,

    pub fn of(rows: usize, slots: usize, layers: usize) Layout {
        var l: Layout = undefined;
        l.layers = layers;
        l.tokens = 0;
        l.pos = rows;
        l.slot = 2 * rows;
        l.anc = 3 * rows;
        l.first = 4 * rows;
        l.count = l.first + slots;
        l.keep = l.count + slots;
        l.desc = std.mem.alignForward(usize, l.keep + slots, 2);
        l.rings = l.desc + 2 * slots;
        l.snaps = l.rings + 2 * slots;
        l.words = l.snaps + 4 * layers;
        return l;
    }
};

fn put64(w: []u32, v: u64) void {
    w[0] = @truncate(v);
    w[1] = @truncate(v >> 32);
}

/// One stream's window of a round: its descriptor, the slot its first row takes, its tokens, each row's parent row
/// (row 0: -1) for a tree (null: a chain), and its drafter's tap ring (0: none).
pub const Window = struct { desc: u64, pos: usize, tokens: []const u32, parents: ?[]const i32 = null, ring: u64 = 0 };

/// Each row's ancestors and itself as bits of the window's rows: a tree's rows see their own path; a chain's are 0.
pub fn ancestors(parents: []const i32, out: []u32) !void {
    if (parents.len == 0 or parents.len > 32 or parents[0] != -1) return error.BadParents;
    for (parents, out[0..parents.len], 0..) |p, *a, r| {
        if (r > 0 and (p < 0 or p >= r)) return error.BadParents;
        a.* = (@as(u32, 1) << @intCast(r)) | if (r == 0) 0 else out[@intCast(p)];
    }
}

/// Whether `parents` names a chain (each row's parent the row before): the rows then run as one.
pub fn isChain(parents: []const i32) bool {
    for (parents, 0..) |p, r| if (p != @as(i32, @intCast(r)) - 1) return false;
    return true;
}

/// The plan buffer of one engine, as large as its biggest round, and the pinned words a round is written in.
pub const Buffer = struct {
    host: hip.HostBuffer,
    dev: hip.DeviceBuffer,
    /// Pages a head of a layer's pool holds: the kernels find a page's keys and values with it.
    pool_pages: u32,

    /// Device bytes a plan of `rows` rows over `layers` layers holds.
    pub fn deviceBytes(rows: usize, layers: usize) usize {
        return 4 * Layout.of(rows, rows + 1, layers).words;
    }

    pub fn init(d: *const hip.Runtime, rows: usize, layers: usize, pool_pages: usize) !Buffer {
        const bytes = deviceBytes(rows, layers);
        var host = try hip.HostBuffer.alloc(d, bytes);
        errdefer host.free();
        return .{ .host = host, .dev = try hip.DeviceBuffer.alloc(d, bytes), .pool_pages = @intCast(pool_pages) };
    }

    pub fn deinit(b: *Buffer) void {
        b.dev.free();
        b.host.free();
    }

    /// The device words of `shape` as launches read them.
    pub fn args(b: *const Buffer, l: Layout) hip.plan_ops.Args {
        const at = b.dev.base();
        return .{
            .pos = at + 4 * l.pos,
            .slot = at + 4 * l.slot,
            .first = at + 4 * l.first,
            .count = at + 4 * l.count,
            .desc = at + 4 * l.desc,
            .snaps = at + 4 * l.snaps,
            .anc = at + 4 * l.anc,
            .pages = @intCast(state.header_words + 2 * l.layers),
            .pool = b.pool_pages,
        };
    }

    /// Writes the round into the pinned words: windows, the scratch slot's `pad` rows, each linear layer's snapshots;
    /// a tree's rows are keyed at their depth past the stream's positions.
    pub fn fill(b: *Buffer, l: Layout, shape: Shape, windows: []const Window, scratch: u64, snaps: []const [2]u64) !void {
        const w = b.host.slice(u32);
        const slots: usize = shape.slots;
        var at: usize = 0;
        for (windows, 0..) |win, s| {
            w[l.first + s] = @intCast(at);
            w[l.count + s] = @intCast(win.tokens.len);
            w[l.keep + s] = std.math.maxInt(u32);
            put64(w[l.desc + 2 * s ..], win.desc);
            put64(w[l.rings + 2 * s ..], win.ring);
            const anc = w[l.anc + at ..][0..win.tokens.len];
            const tree = if (win.parents) |ps| !isChain(ps) else false;
            if (tree) {
                if (win.parents.?.len != win.tokens.len) return error.BadParents;
                try ancestors(win.parents.?, anc);
            } else @memset(anc, 0);
            for (win.tokens, anc, 0..) |t, a, i| {
                w[l.tokens + at + i] = t;
                w[l.pos + at + i] = @intCast(win.pos + if (tree) @popCount(a) - 1 else i);
                w[l.slot + at + i] = @intCast(s);
            }
            at += win.tokens.len;
        }
        // slots past the streams' are empty and the last is the scratch slot, where padding rows run from its start
        for (windows.len..slots - 1) |s| {
            w[l.first + s] = 0;
            w[l.count + s] = 0;
            w[l.keep + s] = std.math.maxInt(u32);
            put64(w[l.desc + 2 * s ..], 0);
            put64(w[l.rings + 2 * s ..], 0);
        }
        const pad = shape.rows - at;
        w[l.first + slots - 1] = @intCast(at);
        w[l.count + slots - 1] = @intCast(pad);
        w[l.keep + slots - 1] = std.math.maxInt(u32);
        put64(w[l.desc + 2 * (slots - 1) ..], scratch);
        put64(w[l.rings + 2 * (slots - 1) ..], 0);
        for (0..pad) |i| {
            w[l.tokens + at + i] = 0;
            w[l.anc + at + i] = 0;
            w[l.pos + at + i] = @intCast(i);
            w[l.slot + at + i] = @intCast(slots - 1);
        }
        for (snaps, 0..) |pair, i| {
            put64(w[l.snaps + 4 * i ..], pair[0]);
            put64(w[l.snaps + 4 * i + 2 ..], pair[1]);
        }
    }

    /// The filled words up to the device in one copy (a graph holds the copy: it runs from the same pinned words).
    pub fn send(b: *const Buffer, stream: hip.abi.Stream, l: Layout) !void {
        try hip.raw.upload(b.dev, 0, b.host.bytes[0 .. 4 * l.words], stream);
    }
};

test "layout keeps the eight-byte arrays on even words" {
    const l = Layout.of(5, 4, 3);
    try std.testing.expect(l.desc % 2 == 0 and l.rings % 2 == 0 and l.snaps % 2 == 0);
    try std.testing.expect(l.words >= l.snaps + 12);
}

test "a tree's rows see their ancestors" {
    var anc: [5]u32 = undefined;
    try ancestors(&.{ -1, 0, 0, 1, 2 }, &anc);
    try std.testing.expectEqualSlices(u32, &.{ 0b1, 0b11, 0b101, 0b1011, 0b10101 }, &anc);
    try std.testing.expect(isChain(&.{ -1, 0, 1 }) and !isChain(&.{ -1, 0, 0 }));
    try std.testing.expectError(error.BadParents, ancestors(&.{ -1, 1 }, &anc));
}
