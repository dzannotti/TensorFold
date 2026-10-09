//! tf-flashnext-ngram-bench MODEL TOKENS [--window N] [--cache-gib G] [--gap-ms M] [--passes P] [--drop]: the host
//! n-gram gather's stall a decode window, host only (no GPU). A recorded token stream (comma or space separated ids)
//! is cut into windows of N tokens (a verify window: the last kept token and its drafts) and each window's rows are
//! gathered as the engine stages them, in three modes, each pass over the same stream:
//!   direct  the mapped table with MADV_WILLNEED a window (the engine before the row cache)
//!   cache   through the row cache (cold on its first pass, warm on later ones)
//!   ahead   the cache with the warmer: the next window's ids pushed right after a gather, then M ms of "GPU time"
//! Every gathered row is compared with the mapped table's bytes. --drop drops the table's pages (POSIX_FADV_DONTNEED)
//! before each mode, for a cold page cache: only when nothing else on the machine relies on them being cached.
const std = @import("std");
const core = @import("core");
const fln = @import("flashnext");

const nc = fln.ngram_cache;
const st = core.safetensors;

const Stats = struct {
    ms: std.ArrayList(f64) = .empty,

    fn line(s: *Stats, mode: []const u8, pass: usize, c: ?*nc.RowCache) void {
        std.mem.sort(f64, s.ms.items, {}, std.sort.asc(f64));
        var total: f64 = 0;
        for (s.ms.items) |v| total += v;
        const n = s.ms.items.len;
        const q = struct {
            fn f(v: []const f64, p: f64) f64 {
                return v[@min(v.len - 1, @as(usize, @intFromFloat(p * @as(f64, @floatFromInt(v.len)))))];
            }
        }.f;
        std.debug.print("{s:<7} pass {d}: {d} windows, gather ms mean {d:.3} p50 {d:.3} p90 {d:.3} p99 {d:.3} max {d:.2}, total {d:.1} ms", .{ mode, pass, n, total / @as(f64, @floatFromInt(n)), q(s.ms.items, 0.5), q(s.ms.items, 0.9), q(s.ms.items, 0.99), s.ms.items[n - 1], total });
        if (c) |cc| std.debug.print(", cache hits {d} misses {d}", .{ cc.hits.load(.monotonic), cc.misses.load(.monotonic) });
        std.debug.print("\n", .{});
        s.ms.clearRetainingCapacity();
    }
};

fn usage() u8 {
    std.debug.print("usage: tf-flashnext-ngram-bench MODEL TOKENS [--window N] [--cache-gib G] [--gap-ms M] [--passes P] [--drop]\n", .{});
    return 2;
}

pub fn main(init: std.process.Init) !u8 {
    const gpa = std.heap.c_allocator;
    const io = init.io;
    const args = try init.minimal.args.toSlice(init.arena.allocator());
    if (args.len < 3) return usage();
    var window: usize = 6;
    var gib: f64 = 1;
    var gap_ms: f64 = 40;
    var passes: usize = 2;
    var drop = false;
    var i: usize = 3;
    while (i < args.len) : (i += 1) {
        const a = args[i];
        if (std.mem.eql(u8, a, "--drop")) {
            drop = true;
            continue;
        }
        if (i + 1 >= args.len) return usage();
        i += 1;
        if (std.mem.eql(u8, a, "--window")) window = try std.fmt.parseInt(usize, args[i], 10) else if (std.mem.eql(u8, a, "--cache-gib")) gib = try std.fmt.parseFloat(f64, args[i]) else if (std.mem.eql(u8, a, "--gap-ms")) gap_ms = try std.fmt.parseFloat(f64, args[i]) else if (std.mem.eql(u8, a, "--passes")) passes = try std.fmt.parseInt(usize, args[i], 10) else return usage();
    }
    if (window == 0 or window > 32) return usage();

    var c = try fln.config.Config.read(gpa, io, args[1], .{});
    defer c.deinit();
    const g = try fln.ngram.NGram.init(c.ngramOptions(0));

    // the table's shards from MODEL/ple-table/*.safetensors, in shard order, mapped as the engine maps them
    const tdir = try std.fs.path.join(gpa, &.{ args[1], "ple-table" });
    var files: std.ArrayList(st.File) = .empty;
    var paths: std.ArrayList([]const u8) = .empty;
    var d = try std.Io.Dir.cwd().openDir(io, tdir, .{ .iterate = true });
    var it = d.iterate();
    while (try it.next(io)) |e| if (std.mem.endsWith(u8, e.name, ".safetensors")) {
        const p = try std.fs.path.join(gpa, &.{ tdir, e.name });
        try paths.append(gpa, p);
        try files.append(gpa, try st.File.open(gpa, io, p));
    };
    d.close(io);
    var shards: std.ArrayList([]const u8) = .empty;
    var starts: std.ArrayList(u64) = .empty;
    try starts.append(gpa, 0);
    var bytes: usize = 0;
    var k: usize = 0;
    shard: while (true) : (k += 1) {
        var b1: [96]u8 = undefined;
        var b2: [96]u8 = undefined;
        const n1 = try std.fmt.bufPrint(&b1, "ngram_embedding.shard_{d}.weight", .{k});
        const n2 = try std.fmt.bufPrint(&b2, "ngram_embedding.shards.{d}.weight", .{k});
        for (files.items) |*f| for (f.names.keys()) |name| if (std.mem.endsWith(u8, name, n1) or std.mem.endsWith(u8, name, n2)) {
            const t = f.get(name).?;
            const row: usize = t.bytes.len / t.dim(0);
            if (bytes != 0 and row != bytes) return error.NgramShardsDiffer;
            bytes = row;
            try shards.append(gpa, t.bytes);
            try starts.append(gpa, starts.items[starts.items.len - 1] + t.dim(0));
            const lo = std.mem.alignBackward(usize, @intFromPtr(t.bytes.ptr), std.heap.pageSize());
            const hi = std.mem.alignForward(usize, @intFromPtr(t.bytes.ptr) + t.bytes.len, std.heap.pageSize());
            const p: [*]align(std.heap.page_size_min) u8 = @ptrFromInt(lo);
            std.posix.madvise(p, hi - lo, std.posix.MADV.RANDOM) catch {};
            continue :shard;
        };
        break;
    }
    if (shards.items.len == 0) {
        std.debug.print("no n-gram shards under {s}\n", .{tdir});
        return 1;
    }
    const rows: nc.Rows = .{ .shards = shards.items, .starts = starts.items, .bytes = bytes };
    if (rows.count() != g.rows) {
        std.debug.print("the shards hold {d} rows, the config {d}\n", .{ rows.count(), g.rows });
        return 1;
    }

    // the token stream
    const text = try std.Io.Dir.cwd().readFileAlloc(io, args[2], gpa, .limited(1 << 28));
    var toks: std.ArrayList(i64) = .empty;
    var tk = std.mem.tokenizeAny(u8, text, ", \n\r\t[]");
    while (tk.next()) |w| try toks.append(gpa, try std.fmt.parseInt(i64, w, 10));
    const nwin = toks.items.len / window;
    std.debug.print("{d} tokens, {d} windows of {d}, {d} rows a window ({d} heads, {d} bytes a row), table {d} rows in {d} shards\n", .{ toks.items.len, nwin, window, window * g.heads, g.heads, bytes, rows.count(), shards.items.len });

    // every window's ids, as stage computes them from the history its commits leave
    const ids = try gpa.alloc(i64, nwin * window * g.heads);
    var hist: [fln.ngram.max_n]i64 = undefined;
    g.initialHistory(hist[0 .. g.n - 1]);
    for (0..nwin) |w| {
        const tw = toks.items[w * window ..][0..window];
        try g.ids(hist[0 .. g.n - 1], tw, ids[w * window * g.heads ..][0 .. window * g.heads]);
        g.advance(hist[0 .. g.n - 1], tw);
    }
    const per = window * g.heads;
    const out = try gpa.alloc(u8, per * bytes);
    var s: Stats = .{};
    var bad: usize = 0;
    for ([_][]const u8{ "direct", "cache", "ahead" }) |mode| {
        if (drop) for (paths.items) |p| {
            const f = try std.Io.Dir.cwd().openFile(io, p, .{});
            _ = std.os.linux.fadvise(f.handle, 0, 0, std.os.linux.POSIX_FADV.DONTNEED);
            f.close(io);
        };
        var cache: ?nc.RowCache = if (std.mem.eql(u8, mode, "direct")) null else try nc.RowCache.init(gpa, io, bytes, @intFromFloat(gib * (1 << 30)));
        defer if (cache) |*cc| cc.deinit();
        var warmer: nc.Warmer = undefined;
        const ahead = std.mem.eql(u8, mode, "ahead");
        if (ahead) {
            warmer = .{ .io = io, .rows = rows, .cache = &cache.? };
            warmer.start(2);
        }
        defer if (ahead) warmer.stop();
        for (0..passes) |pass| {
            if (cache) |*cc| {
                cc.hits.store(0, .monotonic);
                cc.misses.store(0, .monotonic);
            }
            for (0..nwin) |w| {
                const wid = ids[w * per ..][0..per];
                const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
                if (cache) |*cc| try cc.gather(rows, wid, out) else {
                    rows.prefetch(wid);
                    for (wid, 0..) |id, r| @memcpy(out[r * bytes ..][0..bytes], rows.row(@intCast(id)));
                }
                const t1 = std.Io.Clock.awake.now(io).toNanoseconds();
                try s.ms.append(gpa, @as(f64, @floatFromInt(t1 - t0)) / 1e6);
                for (wid, 0..) |id, r| bad += @intFromBool(!std.mem.eql(u8, out[r * bytes ..][0..bytes], rows.row(@intCast(id))));
                if (ahead) {
                    if (w + 1 < nwin) warmer.push(ids[(w + 1) * per ..][0..per]);
                    // the round's GPU time, the warmer reading meanwhile
                    try io.sleep(.fromNanoseconds(@intFromFloat(gap_ms * 1e6)), .awake);
                }
            }
            s.line(mode, pass, if (cache) |*cc| cc else null);
        }
    }
    std.debug.print("{s}: every gathered row {s} the mapped table's bytes ({d} differ)\n", .{ if (bad == 0) "PASS" else "FAIL", if (bad == 0) "equals" else "differs from", bad });
    return if (bad == 0) 0 else 1;
}
