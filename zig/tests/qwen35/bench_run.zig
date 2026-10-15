//! `bench <model dir> [context]`: a lane round's verify time for chains of 1 to 16 rows at `context` cached positions,
//! and for trees of the same rows (a random parent each), graphs on: what a tree costs against a chain.

const std = @import("std");
const qwen35 = @import("qwen3_5");

const Engine = qwen35.engine.Engine;

const warm = 3;
const reps = 20;

fn nowNs() u64 {
    var ts: std.c.timespec = undefined;
    _ = std.c.clock_gettime(.MONOTONIC, &ts);
    return @as(u64, @intCast(ts.sec)) * 1_000_000_000 + @as(u64, @intCast(ts.nsec));
}

/// Mean ms of a verify of `n` rows at `pos` (a tree when `parents` is given), the stream waited on each time.
fn time(e: *Engine, caches: *qwen35.state.Caches, pos: usize, tokens: []const u32, parents: ?[]const i32) !f64 {
    var reqs: [16]qwen35.draw.Request = undefined;
    var out: [16]u32 = undefined;
    for (reqs[0..tokens.len], 0..) |*q, i| q.* = .{ .sampling = null, .position = pos + i + 1 };
    var rows = [_]Engine.Rows{.{ .caches = caches, .pos = pos, .tokens = tokens }};
    if (parents) |ps| {
        // a build before trees times chains alone
        if (comptime !@hasField(Engine.Rows, "parents")) return error.TreesNotBuilt else {
            rows[0].parents = ps;
            var anc: [16]u32 = undefined;
            try qwen35.plan.ancestors(ps, &anc);
            for (reqs[0..tokens.len], 0..) |*q, i| q.position = pos + @popCount(anc[i]);
        }
    }
    var spent: u64 = 0;
    for (0..warm + reps) |i| {
        const t0 = nowNs();
        _ = try e.verify(&rows, reqs[0..tokens.len], out[0..tokens.len]);
        try e.stream.synchronize();
        if (i >= warm) spent += nowNs() - t0;
    }
    return @as(f64, @floatFromInt(spent)) / reps / 1e6;
}

pub fn run(gpa: std.mem.Allocator, io: std.Io, args: []const [:0]const u8) !void {
    if (args.len < 1) return error.MissingArgument;
    const context = if (args.len > 1) try std.fmt.parseInt(usize, args[1], 10) else 2048;
    const ids = try @import("ids_file.zig").synthetic(gpa, context);
    defer gpa.free(ids);
    const e = try Engine.open(gpa, io, args[0], .{ .capacity = context + 64, .prompt_rows = context, .streams = 2, .batch_rows = 64, .policy = (try @import("group.zig").resolve("", 0)).policy });
    defer e.deinit();
    var caches = try e.newCaches(context + 64);
    defer caches.deinit(gpa);
    _ = try e.prefill(&caches, ids, 0, null, .{ .sampling = null, .position = context }, null);
    var prng = std.Random.DefaultPrng.init(7);
    const tokens: [16]u32 = @splat(1234);
    const trees = @hasField(Engine.Rows, "parents");
    std.debug.print("rows chain_ms{s}\n", .{if (trees) " tree_ms" else ""});
    for (1..17) |n| {
        const chain = try time(e, &caches, context, tokens[0..n], null);
        if (!trees or n < 2) {
            std.debug.print("{d} {d:.3}\n", .{ n, chain });
            continue;
        }
        var parents: [16]i32 = undefined;
        parents[0] = -1;
        for (1..n) |r| parents[r] = @intCast(prng.random().uintLessThan(usize, r));
        std.debug.print("{d} {d:.3} {d:.3}\n", .{ n, chain, try time(e, &caches, context, tokens[0..n], parents[0..n]) });
    }
}
