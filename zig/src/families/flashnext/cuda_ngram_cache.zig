//! The n-gram table's hot rows in engine memory (TF_FLASHNEXT_NGRAM_CACHE_GIB) and the threads that read rows into it
//! ahead of a stage. A row is 160 bytes in a 4 KiB page, so the page cache keeps ~25x less of the table than the same
//! memory holding rows; a row the engine looked up once is a memcpy after, whatever else evicts the page cache.
//! Rows are copied as stored (FP8 codes or bf16 bits): the gather's bits are the mapped table's.
const std = @import("std");

/// The mapped table's rows by global id: shards of `bytes`-byte rows, `starts` each shard's first id, then the total.
pub const Rows = struct {
    shards: []const []const u8,
    starts: []const u64,
    bytes: usize,

    pub fn count(r: Rows) u64 {
        return r.starts[r.starts.len - 1];
    }

    pub fn row(r: Rows, id: u64) []const u8 {
        // shards are equal but for the last: a division finds the shard, a step corrects it
        var s: usize = @intCast(@min(id / (r.starts[1] - r.starts[0]), r.shards.len - 1));
        while (r.starts[s] > id) s -= 1;
        while (r.starts[s + 1] <= id) s += 1;
        return r.shards[s][@intCast((id - r.starts[s]) * r.bytes)..][0..r.bytes];
    }

    /// The rows' pages asked of the disk at once (MADV_WILLNEED reads ahead asynchronously), so cold rows read in
    /// parallel rather than one fault at a time.
    pub fn prefetch(r: Rows, ids: []const i64) void {
        const page = std.heap.pageSize();
        for (ids) |id| {
            if (id < 0 or id >= r.count()) continue;
            const src = r.row(@intCast(id));
            const lo = std.mem.alignBackward(usize, @intFromPtr(src.ptr), page);
            const hi = std.mem.alignForward(usize, @intFromPtr(src.ptr) + src.len, page);
            const p: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(lo);
            std.posix.madvise(p, hi - lo, std.posix.MADV.WILLNEED) catch {};
        }
    }
};

/// Rows by id, set-associative (`ways` a set, LRU within the set by a global use tick), every way's bytes held
/// under its set's lock stripe: a reader copies a whole row or misses, never a torn one.
pub const RowCache = struct {
    pub const ways = 8;
    const stripes = 1024;

    io: std.Io,
    gpa: std.mem.Allocator,
    bytes: usize,
    sets: usize,
    /// id + 1 a way (0: empty)
    tags: []u64,
    /// the tick of a way's last use (u32 wraps after ~4G lookups: one set's order goes stale once, bits unaffected)
    stamps: []u32,
    data: []u8,
    locks: []std.Io.Mutex,
    tick: std.atomic.Value(u32) = .init(1),
    hits: std.atomic.Value(u64) = .init(0),
    misses: std.atomic.Value(u64) = .init(0),

    /// A cache of `budget` bytes in all (rows, tags and stamps) for rows of `bytes`; every page written now, so the
    /// memory is taken (and MemAvailable counts it) before the engine sizes its sequences' budget.
    pub fn init(gpa: std.mem.Allocator, io: std.Io, bytes: usize, budget: usize) !RowCache {
        const sets = @max(1, budget / ((bytes + 12) * ways));
        const n = sets * ways;
        const tags = try gpa.alloc(u64, n);
        errdefer gpa.free(tags);
        const stamps = try gpa.alloc(u32, n);
        errdefer gpa.free(stamps);
        const data = try gpa.alloc(u8, n * bytes);
        errdefer gpa.free(data);
        const locks = try gpa.alloc(std.Io.Mutex, @min(stripes, sets));
        @memset(tags, 0);
        @memset(stamps, 0);
        @memset(data, 0);
        @memset(locks, .init);
        return .{ .io = io, .gpa = gpa, .bytes = bytes, .sets = sets, .tags = tags, .stamps = stamps, .data = data, .locks = locks };
    }

    pub fn deinit(c: *RowCache) void {
        c.gpa.free(c.tags);
        c.gpa.free(c.stamps);
        c.gpa.free(c.data);
        c.gpa.free(c.locks);
        c.* = undefined;
    }

    pub fn capacity(c: *const RowCache) usize {
        return c.sets * ways;
    }

    pub fn heldBytes(c: *const RowCache) usize {
        return c.tags.len * 12 + c.data.len;
    }

    fn setOf(c: *const RowCache, id: u64) usize {
        // ids are hashed n-grams plus a head's offset: mixed again so neighbouring ids spread over the sets
        return @intCast(((id *% 0x9E3779B97F4A7C15) >> 17) % c.sets);
    }

    fn lock(c: *RowCache, set: usize) *std.Io.Mutex {
        const m = &c.locks[set % c.locks.len];
        m.lockUncancelable(c.io);
        return m;
    }

    /// Row `id`'s bytes into `out` when held (true), else false.
    pub fn get(c: *RowCache, id: u64, out: []u8) bool {
        const set = c.setOf(id);
        const m = c.lock(set);
        defer m.unlock(c.io);
        for (set * ways..set * ways + ways) |w| if (c.tags[w] == id + 1) {
            @memcpy(out, c.data[w * c.bytes ..][0..c.bytes]);
            c.stamps[w] = c.tick.fetchAdd(1, .monotonic);
            _ = c.hits.fetchAdd(1, .monotonic);
            return true;
        };
        _ = c.misses.fetchAdd(1, .monotonic);
        return false;
    }

    /// Row `id` held as `row` (an empty way, else the set's least recently used); a held id is only touched.
    pub fn put(c: *RowCache, id: u64, row: []const u8) void {
        const set = c.setOf(id);
        const m = c.lock(set);
        defer m.unlock(c.io);
        var victim = set * ways;
        for (set * ways..set * ways + ways) |w| {
            if (c.tags[w] == id + 1) {
                c.stamps[w] = c.tick.fetchAdd(1, .monotonic);
                return;
            }
            if (c.tags[victim] != 0 and (c.tags[w] == 0 or c.stamps[w] -% c.stamps[victim] > std.math.maxInt(u32) / 2)) victim = w;
        }
        c.tags[victim] = id + 1;
        @memcpy(c.data[victim * c.bytes ..][0..c.bytes], row);
        c.stamps[victim] = c.tick.fetchAdd(1, .monotonic);
    }

    /// Rows `ids` -> `out[ids.len][bytes]` as stored: held rows from the cache, the others read from `rows` (their
    /// pages asked for together first) and kept. An id past the table is an error.
    pub fn gather(c: *RowCache, rows: Rows, ids: []const i64, out: []u8) !void {
        if (out.len != ids.len * c.bytes) return error.NgramOutLength;
        var missed: [256]i64 = undefined;
        var at: [256]usize = undefined;
        var i: usize = 0;
        while (i < ids.len) {
            var n: usize = 0;
            while (i < ids.len and n < missed.len) : (i += 1) {
                const id = ids[i];
                if (id < 0 or id >= rows.count()) return error.NgramIdOutOfRange;
                if (c.get(@intCast(id), out[i * c.bytes ..][0..c.bytes])) continue;
                missed[n] = id;
                at[n] = i;
                n += 1;
            }
            if (n > 1) rows.prefetch(missed[0..n]);
            for (missed[0..n], at[0..n]) |id, k| {
                const dst = out[k * c.bytes ..][0..c.bytes];
                @memcpy(dst, rows.row(@intCast(id))); // the fault (a disk read when cold) outside any lock
                c.put(@intCast(id), dst);
            }
        }
    }
};

/// Threads that read rows into the cache as soon as their ids are known (a draft's n-grams while the head drafts
/// on), so the stage that gathers them finds them held. A hint: ids past a full queue are dropped, the stage reads
/// what is not held itself.
pub const Warmer = struct {
    const queue_len = 4096;
    const batch = 64;

    io: std.Io,
    rows: Rows,
    cache: *RowCache,
    mutex: std.Io.Mutex = .init,
    wake: std.Io.Condition = .init,
    queue: [queue_len]i64 = undefined,
    head: usize = 0,
    len: usize = 0,
    closing: bool = false,
    /// threads reading a batch now
    busy: usize = 0,
    threads: [4]?std.Thread = @splat(null),
    /// ids read into the cache, ids dropped at a full queue
    warmed: std.atomic.Value(u64) = .init(0),
    dropped: std.atomic.Value(u64) = .init(0),

    /// `n` (1-4) threads on `w`, which must not move until `stop`.
    pub fn start(w: *Warmer, n: usize) void {
        for (w.threads[0..@min(n, w.threads.len)]) |*t| t.* = std.Thread.spawn(.{}, run, .{w}) catch null;
    }

    pub fn stop(w: *Warmer) void {
        w.mutex.lockUncancelable(w.io);
        w.closing = true;
        w.wake.broadcast(w.io);
        w.mutex.unlock(w.io);
        for (&w.threads) |*t| if (t.*) |th| {
            th.join();
            t.* = null;
        };
    }

    pub fn push(w: *Warmer, ids: []const i64) void {
        if (ids.len == 0) return;
        w.mutex.lockUncancelable(w.io);
        defer w.mutex.unlock(w.io);
        const take = @min(ids.len, queue_len - w.len);
        for (ids[0..take]) |id| {
            w.queue[(w.head + w.len) % queue_len] = id;
            w.len += 1;
        }
        if (take < ids.len) _ = w.dropped.fetchAdd(ids.len - take, .monotonic);
        w.wake.signal(w.io);
    }

    /// Until the queue is empty and nothing is being read (tests and the microbench).
    pub fn drain(w: *Warmer) void {
        while (true) {
            w.mutex.lockUncancelable(w.io);
            const idle = w.len == 0 and w.busy == 0;
            w.mutex.unlock(w.io);
            if (idle) return;
            std.Thread.yield() catch {};
        }
    }

    fn run(w: *Warmer) void {
        var ids: [batch]i64 = undefined;
        var buf: [batch * 320]u8 = undefined; // 160-byte FP8 rows (bf16 tables: half a batch at a time)
        const most = @max(1, @min(batch, buf.len / w.cache.bytes));
        while (true) {
            w.mutex.lockUncancelable(w.io);
            while (w.len == 0 and !w.closing) w.wake.waitUncancelable(w.io, &w.mutex);
            if (w.closing) {
                w.mutex.unlock(w.io);
                return;
            }
            const n = @min(w.len, most);
            for (ids[0..n], 0..) |*d, k| d.* = w.queue[(w.head + k) % queue_len];
            w.head = (w.head + n) % queue_len;
            w.len -= n;
            w.busy += 1;
            if (w.len > 0) w.wake.signal(w.io);
            w.mutex.unlock(w.io);
            // ids are the engine's own (in range); a bad one only skips its batch
            w.cache.gather(w.rows, ids[0..n], buf[0 .. n * w.cache.bytes]) catch {};
            _ = w.warmed.fetchAdd(n, .monotonic);
            w.mutex.lockUncancelable(w.io);
            w.busy -= 1;
            w.mutex.unlock(w.io);
        }
    }
};

/// TF_FLASHNEXT_NGRAM_CACHE_GIB, else `default_gib` (HIP: the table mostly out of the page cache; 0 off).
pub fn cacheBytes(default_gib: f64) usize {
    const gib = if (std.c.getenv("TF_FLASHNEXT_NGRAM_CACHE_GIB")) |v| std.fmt.parseFloat(f64, std.mem.span(v)) catch default_gib else default_gib;
    return if (gib > 0) @intFromFloat(gib * (1 << 30)) else 0;
}

// -- tests ------------------------------------------------------------------------------------------------------

const testing = std.testing;

/// A table of `shards` shards of `per` rows of `bytes` bytes, each byte a function of its row and column.
fn testTable(gpa: std.mem.Allocator, shards: usize, per: usize, last: usize, bytes: usize) !struct { mem: []u8, sl: [][]const u8, starts: []u64 } {
    const total = (shards - 1) * per + last;
    const mem = try gpa.alloc(u8, total * bytes);
    for (0..total) |r| for (0..bytes) |j| {
        mem[r * bytes + j] = @truncate((r *% 2654435761) >> 7 ^ j *% 31);
    };
    const sl = try gpa.alloc([]const u8, shards);
    const starts = try gpa.alloc(u64, shards + 1);
    starts[0] = 0;
    for (0..shards) |s| {
        const n = if (s + 1 == shards) last else per;
        sl[s] = mem[starts[s] * bytes ..][0 .. n * bytes];
        starts[s + 1] = starts[s] + n;
    }
    return .{ .mem = mem, .sl = sl, .starts = starts };
}

test "the cache's gather equals the mapped rows, cold and warm, across shards" {
    const gpa = testing.allocator;
    const t = try testTable(gpa, 3, 1000, 517, 160);
    defer gpa.free(t.mem);
    defer gpa.free(t.sl);
    defer gpa.free(t.starts);
    const rows: Rows = .{ .shards = t.sl, .starts = t.starts, .bytes = 160 };
    var c = try RowCache.init(gpa, testing.io, 160, 64 << 10);
    defer c.deinit();
    var prng = std.Random.DefaultPrng.init(7);
    const ids = try gpa.alloc(i64, 700); // past one 256-id pass
    defer gpa.free(ids);
    for (ids) |*d| d.* = prng.random().intRangeLessThan(i64, 0, 2517);
    ids[0] = 0;
    ids[1] = 999;
    ids[2] = 1000;
    ids[3] = 2516;
    ids[4] = ids[5]; // a repeat in one pass
    const out = try gpa.alloc(u8, ids.len * 160);
    defer gpa.free(out);
    for (0..2) |_| {
        @memset(out, 0xAA);
        try c.gather(rows, ids, out);
        for (ids, 0..) |id, i| try testing.expectEqualSlices(u8, t.mem[@as(usize, @intCast(id)) * 160 ..][0..160], out[i * 160 ..][0..160]);
    }
    try testing.expect(c.hits.load(.monotonic) > 0);
    try testing.expectError(error.NgramIdOutOfRange, c.gather(rows, &.{2517}, out[0..160]));
    try testing.expectError(error.NgramIdOutOfRange, c.gather(rows, &.{-1}, out[0..160]));
}

test "a full set evicts its least recently used row" {
    const gpa = testing.allocator;
    var c = try RowCache.init(gpa, testing.io, 4, 1); // one set of 8 ways
    defer c.deinit();
    try testing.expectEqual(@as(usize, 1), c.sets);
    var buf: [4]u8 = undefined;
    for (0..8) |i| c.put(i, &@as([4]u8, @splat(@intCast(i))));
    try testing.expect(c.get(0, &buf)); // 0 used again: 1 is now the oldest
    c.put(8, &@as([4]u8, @splat(8)));
    try testing.expect(!c.get(1, &buf));
    try testing.expect(c.get(0, &buf));
    try testing.expectEqualSlices(u8, &@as([4]u8, @splat(0)), &buf);
    try testing.expect(c.get(8, &buf));
    try testing.expectEqualSlices(u8, &@as([4]u8, @splat(8)), &buf);
    for (2..8) |i| try testing.expect(c.get(i, &buf));
    c.put(3, &@as([4]u8, @splat(99))); // a held id is only touched, its bytes kept
    try testing.expect(c.get(3, &buf));
    try testing.expectEqualSlices(u8, &@as([4]u8, @splat(3)), &buf);
}

test "concurrent gathers and the warmer return the mapped bytes under eviction" {
    const gpa = testing.allocator;
    const t = try testTable(gpa, 4, 4096, 4096, 160);
    defer gpa.free(t.mem);
    defer gpa.free(t.sl);
    defer gpa.free(t.starts);
    const rows: Rows = .{ .shards = t.sl, .starts = t.starts, .bytes = 160 };
    var c = try RowCache.init(gpa, testing.io, 160, 256 << 10); // ~1.5k rows for 16k: constant eviction
    defer c.deinit();
    var w: Warmer = .{ .io = testing.io, .rows = rows, .cache = &c };
    w.start(2);
    defer w.stop();
    const Ctx = struct {
        fn go(cc: *RowCache, r: Rows, mem: []const u8, wm: *Warmer, seed: u64, bad: *std.atomic.Value(bool)) void {
            var prng = std.Random.DefaultPrng.init(seed);
            var ids: [300]i64 = undefined;
            var out: [300 * 160]u8 = undefined;
            for (0..40) |_| {
                for (&ids) |*d| d.* = prng.random().intRangeLessThan(i64, 0, 4 * 4096);
                wm.push(ids[0..100]);
                cc.gather(r, &ids, &out) catch {
                    bad.store(true, .monotonic);
                    return;
                };
                for (ids, 0..) |id, i| if (!std.mem.eql(u8, mem[@as(usize, @intCast(id)) * 160 ..][0..160], out[i * 160 ..][0..160])) bad.store(true, .monotonic);
            }
        }
    };
    var bad = std.atomic.Value(bool).init(false);
    var th: [4]std.Thread = undefined;
    for (&th, 0..) |*x, i| x.* = try std.Thread.spawn(.{}, Ctx.go, .{ &c, rows, t.mem, &w, i + 1, &bad });
    for (th) |x| x.join();
    w.drain();
    try testing.expect(!bad.load(.monotonic));
    try testing.expect(w.warmed.load(.monotonic) + w.dropped.load(.monotonic) == 4 * 40 * 100);
    // every way still holds its own id's bytes
    for (c.tags, 0..) |tag, k| if (tag != 0) try testing.expectEqualSlices(u8, t.mem[(tag - 1) * 160 ..][0..160], c.data[k * 160 ..][0..160]);
}

test "the warmer's rows are held afterwards" {
    const gpa = testing.allocator;
    const t = try testTable(gpa, 2, 512, 100, 160);
    defer gpa.free(t.mem);
    defer gpa.free(t.sl);
    defer gpa.free(t.starts);
    const rows: Rows = .{ .shards = t.sl, .starts = t.starts, .bytes = 160 };
    var c = try RowCache.init(gpa, testing.io, 160, 1 << 20);
    defer c.deinit();
    var w: Warmer = .{ .io = testing.io, .rows = rows, .cache = &c };
    w.start(1);
    defer w.stop();
    const ids = [_]i64{ 3, 600, 511, 512, 17 };
    w.push(&ids);
    w.drain();
    var buf: [160]u8 = undefined;
    for (ids) |id| {
        try testing.expect(c.get(@intCast(id), &buf));
        try testing.expectEqualSlices(u8, t.mem[@as(usize, @intCast(id)) * 160 ..][0..160], &buf);
    }
}
