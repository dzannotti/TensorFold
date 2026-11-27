//! Flash Next's lone windows and head step, timed at load on a scratch stream: what the lane core's depth rule weighs.
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

const Rig = struct {
    f: *Fx,
    b: be.Backend,
    s: *lanes.stream.Stream,
    first: u32,

    /// Mean ms of a verify of one window of `widths[0]` rows (the round, its logits and its draws), never kept.
    pub fn time(r: *Rig, widths: []const usize) !f64 {
        if (widths.len != 1) return error.SharedNotTimed;
        const w = widths[0];
        var positions: [cache.max_window]u64 = undefined;
        for (positions[0..w], 0..) |*p, i| p.* = r.s.prompt_len + 1 + i;
        const window = [_]be.Window{.{ .stream = r.s, .pending = r.first, .held = 0, .tokens = drafts[0 .. w - 1], .parents = null, .positions = positions[0..w] }};
        // every width warm past its graph capture (a shape is captured the second time it runs), then timed
        var spent: u64 = 0;
        for (0..shared.warm + shared.reps) |i| {
            var sampled: [cache.max_window]u32 = undefined;
            var echo: [cache.max_window]u32 = undefined;
            var out = [_]be.Verified{.{ .sampled = sampled[0..w], .drafts = echo[0..w] }};
            const t0 = serve.backend.nowNs();
            try r.b.verify(&window, &out);
            try r.f.e.stream.synchronize();
            if (i >= shared.warm) spent += serve.backend.nowNs() - t0;
            // the window stays unkept: the next verify starts from the same cache length
            r.f.lanes.get(r.s).?.pending = null;
        }
        return @as(f64, @floatFromInt(spent)) / shared.reps / 1e6;
    }

    /// One chained head step: a chain of the deepest depth from the prompt's last row, by its draft count.
    fn step(r: *Rig) !f64 {
        const depth = lanes_mod.max_depth;
        const req = [_]be.DraftRequest{.{ .stream = r.s, .follow = &.{}, .first = .{ .value = r.first }, .rows = null, .start = r.s.prompt_len, .position = r.s.prompt_len + 1, .depth = depth }};
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

/// Lone windows of every width and the head step, timed on a scratch stream of `f` before any request.
pub fn measure(gpa: std.mem.Allocator, f: *Fx, out: *Costs) !void {
    var s = try lanes.stream.Stream.init(gpa, .{ .id = "costs", .prompt = &prompt, .max_new = 16 });
    defer s.deinit(gpa);
    const b = f.backend();
    defer b.release(&s);
    try b.prefill(&s);
    var r: Rig = .{ .f = f, .b = b, .s = &s, .first = try b.read(try b.first(&s, s.prompt_len)) };
    try shared.windows(&r, cache.max_window, out);
    if (f.e.weights.model.mtp != null and f.drafts > 0) out.step_ms = try r.step();
}
