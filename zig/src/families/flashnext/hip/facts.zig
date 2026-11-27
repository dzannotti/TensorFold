//! The lane core's facts of a loaded Flash Next model, pure so a host test drives them without a device.
const std = @import("std");
const lanes = @import("lanes");
const hip = @import("hip");
const cache = @import("cache.zig");

/// MTP drafts the HIP head takes at most: the window's rows less its pending token.
pub const mtp_max_drafts: u32 = cache.max_window - 1;

/// The policy's `--mtp-drafts` capped at the head's.
pub fn capDrafts(asked: u8) u32 {
    return @min(mtp_max_drafts, @as(u32, asked));
}

/// A shared round's copy window and a lone stream's first: the pending row and three copies; lone runs grow past it.
const shared_copy_rows = 4;

/// Most head drafts a stream chains in a shared round; a lone stream chains the policy's drafts.
pub const shared_chain_drafts = 3;

/// The lane core's facts of a loaded model: `drafting` names a head that drafts, `drafts` its capped chains.
fn ltEnv(name: [*:0]const u8, default: u32) u32 {
    const raw = std.c.getenv(name) orelse return default;
    return std.fmt.parseInt(u32, std.mem.span(raw), 10) catch default;
}

pub fn factsFor(exact_width: u32, streams: usize, drafting: bool, drafts: u32) lanes.Model {
    return .{
        .exact_width = exact_width,
        .first_copy_rows = ltEnv("LT_COPY_ROWS", shared_copy_rows),
        .gpu_tokens = false,
        .mtp = drafting,
        .speculate = drafting,
        .speculate_early = false,
        .drafts = drafts,
        .shared_mtp_cap = ltEnv("LT_SHARED_CAP", shared_chain_drafts),
        .hidden_rows = true,
        .batch_rows = @intCast(exact_width * streams),
        .max_streams = @intCast(streams),
        .draft_streams = drafting,
    };
}

/// `m` with its lone windows' and head step's timed costs: a lone stream's depth rule weighs drafts by time.
pub fn timed(m: lanes.Model, windows: []const lanes.config.Cost, step_ms: f64) lanes.Model {
    var out = m;
    out.window_costs = windows;
    // shared rounds are untimed: a lone window's ms would price every stream's extra rows as one stream's
    out.window_costs_lone = true;
    out.mtp_step_ms = step_ms;
    return out;
}

test "timed windows give the depth rule its forward costs; untimed it keeps the fallback by first-draft rate" {
    const a = std.testing.allocator;
    const windows = [_]lanes.config.Cost{ .{ .width = 1, .ms = 40 }, .{ .width = 2, .ms = 45 }, .{ .width = 3, .ms = 50 } };
    var cfg = try lanes.Config.init(a, timed(factsFor(cache.max_window, 1, true, 7), &windows, 1.5), cache.max_window, cache.max_window - 1);
    defer cfg.deinit(a);
    try std.testing.expect(!cfg.family_costs.empty());
    try std.testing.expectEqual(@as(f64, 1.5), cfg.mtp_step_ms);
    var plain = try lanes.Config.init(a, factsFor(cache.max_window, 1, true, 7), cache.max_window, cache.max_window - 1);
    defer plain.deinit(a);
    try std.testing.expect(plain.family_costs.empty());
}

test "the shared round's copy window: the lane core takes three copies, a lone stream the whole window" {
    const a = std.testing.allocator;
    var cfg = try lanes.Config.init(a, factsFor(cache.max_window, 2, true, 3), cache.max_window, cache.max_window - 1);
    defer cfg.deinit(a);
    try std.testing.expectEqual(@as(u32, 3), cfg.first_copy);
    try std.testing.expectEqual(@as(u32, 7), cfg.max_copy);
    try std.testing.expectEqual(@as(u32, 4), cfg.base_width);
}

test "the extracted facts retain the loaded head drafting mode" {
    for ([_]bool{ false, true }) |drafting| {
        const facts = factsFor(cache.max_window, 2, drafting, 3);
        try std.testing.expectEqual(drafting, facts.draft_streams);
    }
}

test "the facts cap shared chains at three, and the default policy asks the lane core for three drafts" {
    const a = std.testing.allocator;
    const asked: u32 = (hip.Policy{}).mtp.drafts;
    var cfg = try lanes.Config.init(a, factsFor(cache.max_window, 1, true, asked), cache.max_window, cache.max_window - 1);
    defer cfg.deinit(a);
    try std.testing.expectEqual(@as(u32, 3), cfg.most_drafts);
    try std.testing.expectEqual(@as(u32, 3), cfg.shared_mtp_cap);
    var deep = try lanes.Config.init(a, factsFor(cache.max_window, 1, true, 7), cache.max_window, cache.max_window - 1);
    defer deep.deinit(a);
    try std.testing.expectEqual(@as(u32, 7), deep.most_drafts);
    try std.testing.expectEqual(@as(u32, 3), deep.shared_mtp_cap);
}
