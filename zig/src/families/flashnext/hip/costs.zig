//! Flash Next's windows and head step, timed at load on scratch streams: what the lane core's depth rule weighs.
const std = @import("std");
const lanes = @import("lanes");
const serve = @import("hip_serve");
const cache = @import("cache.zig");
const lanes_mod = @import("lanes.zig");

const be = lanes.backend;
const shared = serve.costs;
const Fx = lanes_mod.Fx;
pub const Costs = shared.Costs;

/// The scratch prompt: ordinary English ids, long enough that every window sits past the first pooled block.
const prompt = [_]u32{ 372, 13369, 198, 220, 17, 13, 13258, 19, 3266, 310, 264, 1283, 12068, 1794, 290, 16147 };
/// A window's drafts: any ids in the draft vocabulary; the timings never keep them.
const drafts: [cache.max_window - 1]u32 = @splat(11);

const most = shared.most_streams;

/// The shared rows timed: every fourth total, each round's rows spread over as many streams as it has.
fn bucketOf(total: usize, batch_rows: usize) usize {
    return @min(batch_rows, std.mem.alignForward(usize, total, 4));
}

const Rig = struct {
    f: *Fx,
    b: be.Backend,
    ss: []lanes.stream.Stream,
    first: [most]u32 = undefined,

    /// Mean ms of a verify of one window a stream of the given widths (the round, its logits and draws), never kept.
    pub fn time(r: *Rig, widths: []const usize) !f64 {
        if (widths.len > r.ss.len) return error.TooManyWindows;
        var positions: [most][cache.max_window]u64 = undefined;
        var windows: [most]be.Window = undefined;
        var sampled: [most][cache.max_window]u32 = undefined;
        var echo: [most][cache.max_window]u32 = undefined;
        var out: [most]be.Verified = undefined;
        for (widths, 0..) |w, k| {
            const s = &r.ss[k];
            for (positions[k][0..w], 0..) |*p, i| p.* = s.prompt_len + 1 + i;
            windows[k] = .{ .stream = s, .pending = r.first[k], .held = 0, .tokens = drafts[0 .. w - 1], .parents = null, .positions = positions[k][0..w] };
            out[k] = .{ .sampled = sampled[k][0..w], .drafts = echo[k][0..w] };
        }
        // every shape warm past its graph capture (a shape is captured the second time it runs), then timed
        var spent: u64 = 0;
        for (0..shared.warm + shared.reps) |i| {
            const t0 = serve.backend.nowNs();
            try r.b.verify(windows[0..widths.len], out[0..widths.len]);
            try r.f.e.stream.synchronize();
            if (i >= shared.warm) spent += serve.backend.nowNs() - t0;
            // the windows stay unkept: the next verify starts from the same cache lengths
            for (r.ss[0..widths.len]) |*s| r.f.lanes.get(s).?.pending = null;
        }
        return @as(f64, @floatFromInt(spent)) / shared.reps / 1e6;
    }

    /// One chained head step: a chain of the deepest depth from the prompt's last row, by its draft count.
    fn step(r: *Rig) !f64 {
        const depth = lanes_mod.max_depth;
        const s = &r.ss[0];
        const req = [_]be.DraftRequest{.{ .stream = s, .follow = &.{}, .first = .{ .value = r.first[0] }, .rows = null, .start = s.prompt_len, .position = s.prompt_len + 1, .depth = depth }};
        var spent: u64 = 0;
        for (0..shared.warm + shared.reps) |i| {
            const t0 = serve.backend.nowNs();
            try r.b.draft(&req);
            try r.f.e.stream.synchronize();
            if (i >= shared.warm) spent += serve.backend.nowNs() - t0;
        }
        return @as(f64, @floatFromInt(spent)) / shared.reps / 1e6 / depth;
    }
};

/// Lone windows of every width, shared rows over the streams and the head step, timed on scratch streams at load.
pub fn measure(gpa: std.mem.Allocator, f: *Fx, out: *Costs) !void {
    const n: usize = @min(most, f.streams);
    var ss: [most]lanes.stream.Stream = undefined;
    var made: usize = 0;
    const b = f.backend();
    defer for (ss[0..made]) |*s| {
        b.release(s);
        s.deinit(gpa);
    };
    var r: Rig = .{ .f = f, .b = b, .ss = ss[0..n] };
    while (made < n) {
        ss[made] = try lanes.stream.Stream.init(gpa, .{ .id = "costs", .prompt = &prompt, .max_new = 16 });
        made += 1;
        const s = &ss[made - 1];
        try b.prefill(s);
        r.first[made - 1] = try b.read(try b.first(s, s.prompt_len));
    }
    try shared.windows(&r, cache.max_window, out);
    try shared.buckets(&r, n, cache.max_window * n, bucketOf, out);
    if (f.e.weights.model.mtp != null and f.drafts > 0) out.step_ms = try r.step();
}
