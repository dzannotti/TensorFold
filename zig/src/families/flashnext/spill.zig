const std = @import("std");
const Allocator = std.mem.Allocator;

const MAGIC: u32 = 0x4c4c5053;
const VERSION: u32 = 1;

pub const Entry = struct { key: u64, tokens: []u32, bytes: u64, used: u64 };

pub const File = struct { fd: c_int, key: u64 };

const Write = struct { at: i64, bytes: u64 };

pub const Spill = struct {
    gpa: Allocator,
    dir: [:0]u8,
    cap: u64,
    min: usize,
    fingerprint: u64,
    entries: std.ArrayList(Entry) = .empty,
    clock: u64 = 0,
    daily: u64 = 0,
    writes: std.ArrayList(Write) = .empty,

    pub fn fromEnv(gpa: Allocator, fingerprint: u64) ?*Spill {
        const dir = std.mem.span(std.c.getenv("TENSORFOLD_SPILL_DIR") orelse return null);
        if (dir.len == 0) return null;
        const gib = number(f64, "TENSORFOLD_SPILL_GIB", 64);
        const min = number(usize, "TENSORFOLD_SPILL_MIN_TOKENS", 4096);
        const day = number(f64, "TENSORFOLD_SPILL_DAILY_GIB", 32);
        const sp = open(gpa, dir, @intFromFloat(gib * (1 << 30)), min, fingerprint) catch |err| {
            std.log.warn("spill: off ({s})", .{@errorName(err)});
            return null;
        };
        sp.daily = @intFromFloat(day * (1 << 30));
        std.log.info("spill: {d} conversation(s) on disk in {s}, {d} MiB of {d} MiB, at most {d} MiB written a day", .{ sp.entries.items.len, sp.dir, sp.total() >> 20, sp.cap >> 20, sp.daily >> 20 });
        return sp;
    }

    fn number(comptime T: type, name: [*:0]const u8, default: T) T {
        const v = std.mem.span(std.c.getenv(name) orelse return default);
        return switch (@typeInfo(T)) {
            .float => std.fmt.parseFloat(T, v) catch default,
            else => std.fmt.parseInt(T, v, 10) catch default,
        };
    }

    pub fn open(gpa: Allocator, dir: []const u8, cap: u64, min: usize, fingerprint: u64) !*Spill {
        const sp = try gpa.create(Spill);
        errdefer gpa.destroy(sp);
        const d = try std.fmt.allocPrintSentinel(gpa, "{s}", .{dir}, 0);
        errdefer gpa.free(d);
        _ = std.c.mkdir(d, 0o700);
        sp.* = .{ .gpa = gpa, .dir = d, .cap = cap, .min = min, .fingerprint = fingerprint };
        try sp.load();
        sp.trim(null);
        return sp;
    }

    pub fn deinit(sp: *Spill) void {
        for (sp.entries.items) |e| sp.gpa.free(e.tokens);
        sp.entries.deinit(sp.gpa);
        sp.writes.deinit(sp.gpa);
        sp.gpa.free(sp.dir);
        sp.gpa.destroy(sp);
    }

    pub fn total(sp: *const Spill) u64 {
        var n: u64 = 0;
        for (sp.entries.items) |e| n += e.bytes;
        return n;
    }

    fn keyOf(tokens: []const u32) u64 {
        return std.hash.Wyhash.hash(0x5911, std.mem.sliceAsBytes(tokens));
    }

    fn path(sp: *const Spill, buf: []u8, key: u64, ext: []const u8) ![:0]u8 {
        return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.{s}", .{ sp.dir, key, ext }, 0);
    }

    fn load(sp: *Spill) !void {
        const d = std.c.opendir(sp.dir) orelse return error.SpillDir;
        defer _ = std.c.closedir(d);
        while (std.c.readdir(d)) |ent| {
            const name = std.mem.sliceTo(&ent.name, 0);
            if (!std.mem.endsWith(u8, name, ".ids") or name.len != 20) continue;
            const key = std.fmt.parseInt(u64, name[0..16], 16) catch continue;
            if (sp.loadOne(key)) |e| {
                sp.clock += 1;
                var x = e;
                x.used = sp.clock;
                sp.entries.append(sp.gpa, x) catch {
                    sp.gpa.free(e.tokens);
                    return error.OutOfMemory;
                };
            } else sp.unlinkKey(key);
        }
    }

    fn loadOne(sp: *Spill, key: u64) ?Entry {
        var buf: [1200]u8 = undefined;
        const fd = std.c.open(sp.path(&buf, key, "ids") catch return null, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (fd < 0) return null;
        defer _ = std.c.close(fd);
        var head: [5]u32 = undefined;
        if (!readAt(fd, std.mem.sliceAsBytes(&head), 0) or head[0] != MAGIC or head[1] != VERSION) return null;
        if ((@as(u64, head[2]) | @as(u64, head[3]) << 32) != sp.fingerprint or head[4] == 0 or head[4] > 1 << 22) return null;
        const tokens = sp.gpa.alloc(u32, head[4]) catch return null;
        if (!readAt(fd, std.mem.sliceAsBytes(tokens), @sizeOf(@TypeOf(head))) or keyOf(tokens) != key) {
            sp.gpa.free(tokens);
            return null;
        }
        const kv = std.c.open(sp.path(&buf, key, "kv") catch return null, .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
        if (kv < 0) {
            sp.gpa.free(tokens);
            return null;
        }
        const end = std.c.lseek(kv, 0, std.c.SEEK.END);
        _ = std.c.close(kv);
        if (end <= 0) {
            sp.gpa.free(tokens);
            return null;
        }
        return .{ .key = key, .tokens = tokens, .bytes = @intCast(end), .used = 0 };
    }

    fn unlinkKey(sp: *const Spill, key: u64) void {
        var buf: [1200]u8 = undefined;
        if (sp.path(&buf, key, "ids")) |p| _ = std.c.unlink(p) else |_| {}
        if (sp.path(&buf, key, "kv")) |p| _ = std.c.unlink(p) else |_| {}
    }

    fn remove(sp: *Spill, i: usize) void {
        const e = sp.entries.orderedRemove(i);
        sp.unlinkKey(e.key);
        sp.gpa.free(e.tokens);
    }

    fn trim(sp: *Spill, keep: ?u64) void {
        while (sp.total() > sp.cap) {
            var oldest: ?usize = null;
            for (sp.entries.items, 0..) |e, i| if ((keep == null or e.key != keep.?) and (oldest == null or e.used < sp.entries.items[oldest.?].used)) {
                oldest = i;
            };
            sp.remove(oldest orelse return);
        }
    }

    pub fn writtenToday(sp: *Spill) u64 {
        const now = seconds();
        while (sp.writes.items.len > 0 and now - sp.writes.items[0].at >= 86400) _ = sp.writes.orderedRemove(0);
        var n: u64 = 0;
        for (sp.writes.items) |w| n += w.bytes;
        return n;
    }

    pub fn has(sp: *const Spill, tokens: []const u32) bool {
        const key = keyOf(tokens);
        for (sp.entries.items) |e| if (e.key == key) return true;
        return false;
    }

    pub fn create(sp: *Spill, tokens: []const u32) !File {
        const key = keyOf(tokens);
        var buf: [1200]u8 = undefined;
        const tmp = try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.kv.part", .{ sp.dir, key }, 0);
        const fd = std.c.open(tmp, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.SpillWrite;
        return .{ .fd = fd, .key = key };
    }

    pub fn finish(sp: *Spill, f: File, tokens: []const u32, ok: bool) !u64 {
        const end = std.c.lseek(f.fd, 0, std.c.SEEK.CUR);
        _ = std.c.close(f.fd);
        var buf: [1200]u8 = undefined;
        var dst: [1200]u8 = undefined;
        const tmp = try std.fmt.bufPrintSentinel(&buf, "{s}/{x:0>16}.kv.part", .{ sp.dir, f.key }, 0);
        if (!ok or end <= 0) {
            _ = std.c.unlink(tmp);
            return error.SpillWrite;
        }
        if (std.c.rename(tmp, try sp.path(&dst, f.key, "kv")) != 0) return error.SpillWrite;
        const ids = try sp.path(&buf, f.key, "ids");
        const fd = std.c.open(ids, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true }, @as(std.c.mode_t, 0o600));
        if (fd < 0) return error.SpillWrite;
        const head = [5]u32{ MAGIC, VERSION, @truncate(sp.fingerprint), @truncate(sp.fingerprint >> 32), @intCast(tokens.len) };
        const wrote = blk: {
            writeAll(fd, std.mem.sliceAsBytes(&head)) catch break :blk false;
            writeAll(fd, std.mem.sliceAsBytes(tokens)) catch break :blk false;
            break :blk true;
        };
        _ = std.c.close(fd);
        if (!wrote) {
            sp.unlinkKey(f.key);
            return error.SpillWrite;
        }
        var i: usize = 0;
        while (i < sp.entries.items.len) {
            const e = sp.entries.items[i];
            if (e.key == f.key or (e.tokens.len < tokens.len and std.mem.eql(u32, e.tokens, tokens[0..e.tokens.len]))) {
                if (e.key == f.key) {
                    sp.gpa.free(e.tokens);
                    _ = sp.entries.orderedRemove(i);
                } else sp.remove(i);
                continue;
            }
            i += 1;
        }
        sp.writes.append(sp.gpa, .{ .at = seconds(), .bytes = @intCast(end) }) catch {};
        const own = try sp.gpa.dupe(u32, tokens);
        errdefer sp.gpa.free(own);
        sp.clock += 1;
        try sp.entries.append(sp.gpa, .{ .key = f.key, .tokens = own, .bytes = @intCast(end), .used = sp.clock });
        sp.trim(f.key);
        return @intCast(end);
    }

    pub fn best(sp: *const Spill, prompt: []const u32, longer_than: u32) u32 {
        var at: u32 = 0;
        for (sp.entries.items) |e| {
            const n = e.tokens.len;
            if (n <= longer_than or n <= at or n >= prompt.len) continue;
            if (prompt[n - 1] != e.tokens[n - 1] or !std.mem.eql(u32, prompt[0..n], e.tokens)) continue;
            at = @intCast(n);
        }
        return at;
    }

    pub fn openFor(sp: *Spill, prompt: []const u32, at: u32) !c_int {
        for (sp.entries.items) |*e| if (e.tokens.len == at and std.mem.eql(u32, prompt[0..at], e.tokens)) {
            var buf: [1200]u8 = undefined;
            const fd = std.c.open(try sp.path(&buf, e.key, "kv"), .{ .ACCMODE = .RDONLY }, @as(std.c.mode_t, 0));
            if (fd < 0) return error.SpillRead;
            sp.clock += 1;
            e.used = sp.clock;
            return fd;
        };
        return error.SpillMissing;
    }

    pub fn forget(sp: *Spill, prompt: []const u32, at: u32) void {
        for (sp.entries.items, 0..) |e, i| if (e.tokens.len == at and std.mem.eql(u32, prompt[0..at], e.tokens)) return sp.remove(i);
    }
};

fn seconds() i64 {
    var ts: std.posix.timespec = undefined;
    _ = std.posix.system.clock_gettime(.MONOTONIC, &ts);
    return @intCast(ts.sec);
}

pub fn writeAll(fd: c_int, bytes: []const u8) !void {
    var done: usize = 0;
    while (done < bytes.len) {
        const n = std.c.write(fd, bytes.ptr + done, bytes.len - done);
        if (n <= 0) return error.SpillWrite;
        done += @intCast(n);
    }
}

pub fn readAll(fd: c_int, dest: []u8) !void {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.read(fd, dest.ptr + done, dest.len - done);
        if (n <= 0) return error.SpillRead;
        done += @intCast(n);
    }
}

fn readAt(fd: c_int, dest: []u8, at: u64) bool {
    var done: usize = 0;
    while (done < dest.len) {
        const n = std.c.pread(fd, dest.ptr + done, dest.len - done, @intCast(at + done));
        if (n <= 0) return false;
        done += @intCast(n);
    }
    return true;
}
