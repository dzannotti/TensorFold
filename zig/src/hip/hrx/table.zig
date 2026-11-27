//! Device addresses back to their HRX buffers: kernels and HIP callers pass bare u64 addresses, copies need the buffer.
const std = @import("std");

pub fn Table(comptime Buffer: type) type {
    return struct {
        const Self = @This();
        /// `held` entries count toward `bytes`, the allocations this process owns (not module globals).
        pub const Entry = struct { base: u64, len: u64, buffer: Buffer, host: ?[*]u8 = null, held: bool = true };
        pub const Hit = struct { entry: Entry, offset: u64 };

        entries: std.ArrayList(Entry) = .empty,
        lock: std.atomic.Mutex = .unlocked,
        bytes: u64 = 0,

        fn acquire(t: *Self) void {
            while (!t.lock.tryLock()) std.Thread.yield() catch {};
        }

        fn index(t: *const Self, address: u64) usize {
            var lo: usize = 0;
            var hi: usize = t.entries.items.len;
            while (lo < hi) {
                const mid = (lo + hi) / 2;
                if (t.entries.items[mid].base <= address) lo = mid + 1 else hi = mid;
            }
            return lo;
        }

        pub fn insert(t: *Self, a: std.mem.Allocator, e: Entry) !void {
            t.acquire();
            defer t.lock.unlock();
            const at = t.index(e.base);
            if (at > 0) {
                const prev = t.entries.items[at - 1];
                if (prev.base + prev.len > e.base) return error.Overlap;
            }
            if (at < t.entries.items.len and e.base + e.len > t.entries.items[at].base) return error.Overlap;
            try t.entries.insert(a, at, e);
            if (e.held) t.bytes += e.len;
        }

        /// Removes the entry starting exactly at `base`.
        pub fn remove(t: *Self, base: u64) ?Entry {
            t.acquire();
            defer t.lock.unlock();
            const at = t.index(base);
            if (at == 0 or t.entries.items[at - 1].base != base) return null;
            const e = t.entries.orderedRemove(at - 1);
            if (e.held) t.bytes -= e.len;
            return e;
        }

        /// The entry holding `len` bytes at `address`, and the offset of the address in it.
        pub fn find(t: *Self, address: u64, len: u64) ?Hit {
            t.acquire();
            defer t.lock.unlock();
            const at = t.index(address);
            if (at == 0) return null;
            const e = t.entries.items[at - 1];
            const off = address - e.base;
            if (off > e.len or len > e.len - off) return null;
            return .{ .entry = e, .offset = off };
        }

        /// The entry whose host mapping starts at `host`.
        pub fn findHost(t: *Self, host: usize) ?Entry {
            t.acquire();
            defer t.lock.unlock();
            for (t.entries.items) |e| if (e.host) |h| if (@intFromPtr(h) == host) return e;
            return null;
        }

        /// The entry whose host mapping holds `len` bytes at `host`, and the offset of `host` in it.
        pub fn findMapped(t: *Self, host: usize, len: u64) ?Hit {
            t.acquire();
            defer t.lock.unlock();
            for (t.entries.items) |e| if (e.host) |h| {
                const base = @intFromPtr(h);
                if (host >= base and host - base <= e.len and len <= e.len - (host - base)) return .{ .entry = e, .offset = host - base };
            };
            return null;
        }

        pub fn deinit(t: *Self, a: std.mem.Allocator) void {
            t.entries.deinit(a);
        }
    };
}

test "addresses inside an allocation find it; neighbours and overlaps stay apart" {
    var t: Table(u32) = .{};
    defer t.deinit(std.testing.allocator);
    try t.insert(std.testing.allocator, .{ .base = 0x2000, .len = 0x100, .buffer = 2 });
    try t.insert(std.testing.allocator, .{ .base = 0x1000, .len = 0x100, .buffer = 1 });
    try std.testing.expectError(error.Overlap, t.insert(std.testing.allocator, .{ .base = 0x10f0, .len = 0x20, .buffer = 3 }));
    const hit = t.find(0x1010, 0xf0).?;
    try std.testing.expectEqual(@as(u32, 1), hit.entry.buffer);
    try std.testing.expectEqual(@as(u64, 0x10), hit.offset);
    try std.testing.expect(t.find(0x1010, 0xf1) == null);
    try std.testing.expect(t.find(0x1800, 1) == null);
    try std.testing.expectEqual(@as(u64, 0x200), t.bytes);
    try std.testing.expectEqual(@as(u32, 2), t.remove(0x2000).?.buffer);
    try std.testing.expect(t.remove(0x2000) == null);
    try std.testing.expect(t.find(0x2000, 1) == null);
    var pinned: [64]u8 = undefined;
    try t.insert(std.testing.allocator, .{ .base = 0x4000, .len = 64, .buffer = 4, .host = &pinned });
    const inside = t.findMapped(@intFromPtr(&pinned) + 16, 48).?;
    try std.testing.expectEqual(@as(u32, 4), inside.entry.buffer);
    try std.testing.expectEqual(@as(u64, 16), inside.offset);
    try std.testing.expect(t.findMapped(@intFromPtr(&pinned) + 16, 49) == null);
}
