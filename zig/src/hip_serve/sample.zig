//! A row's token from its logits: argmax for greedy, else the keyed draw over top_k + MARGIN candidates.

const std = @import("std");
const lanes = @import("lanes");
const Allocator = std.mem.Allocator;

/// exact_sampling.MARGIN: candidates past top_k, so ties resolve by id on the host.
pub const MARGIN = 8;

pub const Dtype = enum { f16, bf16 };

pub fn widen(row: []const u16, dtype: Dtype, i: usize) f32 {
    return switch (dtype) {
        .f16 => @floatCast(@as(f16, @bitCast(row[i]))),
        .bf16 => @bitCast(@as(u32, row[i]) << 16),
    };
}

/// torch.argmax over the widened row: the first index of the largest value (a NaN wins, as torch's).
pub fn argmax(row: []const u16, dtype: Dtype) u32 {
    var best: usize = 0;
    var top = widen(row, dtype, 0);
    for (1..row.len) |i| {
        const v = widen(row, dtype, i);
        if (std.math.isNan(top)) break;
        if (std.math.isNan(v) or v > top) {
            best = i;
            top = v;
        }
    }
    return @intCast(best);
}

/// Structured output: every token whose bit is clear in `bits` (32 a word, token i in word i / 32) becomes -inf.
pub fn mask(row: []u16, dtype: Dtype, bits: []const u32) void {
    const minus_inf: u16 = switch (dtype) {
        .f16 => 0xFC00,
        .bf16 => 0xFF80,
    };
    for (row, 0..) |*v, i| {
        const word = i / 32;
        const allowed = word < bits.len and (bits[word] >> @intCast(i % 32)) & 1 == 1;
        if (!allowed) v.* = minus_inf;
    }
}

const Candidate = struct { value: f32, id: u32 };

fn larger(_: void, a: Candidate, b: Candidate) bool {
    return a.value > b.value or (a.value == b.value and a.id < b.id);
}

/// draw(): greedy (no sampling or temperature <= 0) is the argmax; else `choose` at `position` over the candidates.
pub fn draw(gpa: Allocator, row: []const u16, dtype: Dtype, sampling: ?lanes.Sampling, position: u64) !u32 {
    const s = sampling orelse return argmax(row, dtype);
    if (s.temperature <= 0.0) return argmax(row, dtype);
    const width = row.len;
    const count = if (s.top_k != 0) @min(width, @as(usize, s.top_k) + MARGIN) else width;
    const best = try gpa.alloc(Candidate, count);
    defer gpa.free(best);
    var filled: usize = 0;
    // the whole row when top_k is 0 (choose orders it), else the top `count` by value then id in one pass
    if (count == width) {
        for (best, 0..) |*c, i| c.* = .{ .value = widen(row, dtype, i), .id = @intCast(i) };
        filled = width;
    } else for (0..width) |i| {
        const c: Candidate = .{ .value = widen(row, dtype, i), .id = @intCast(i) };
        if (filled == count and !larger({}, c, best[count - 1])) continue;
        var at = if (filled < count) filled else count - 1;
        if (filled < count) filled += 1;
        while (at > 0 and larger({}, c, best[at - 1])) : (at -= 1) best[at] = best[at - 1];
        best[at] = c;
    }
    const values = try gpa.alloc(f64, count);
    defer gpa.free(values);
    const ids = try gpa.alloc(u64, count);
    defer gpa.free(ids);
    for (best, values, ids) |c, *v, *id| {
        v.* = c.value;
        id.* = c.id;
    }
    return @intCast(try lanes.sampling.choose(gpa, values, ids, position, s));
}

/// The softmax probability of `id` in the widened row (the MTP chain's confidence cut), in double.
pub fn probability(row: []const u16, dtype: Dtype, id: u32) f64 {
    var top: f64 = -std.math.inf(f64);
    for (0..row.len) |i| top = @max(top, widen(row, dtype, i));
    var total: f64 = 0;
    for (0..row.len) |i| total += @exp(@as(f64, widen(row, dtype, i)) - top);
    return @exp(@as(f64, widen(row, dtype, id)) - top) / total;
}

/// The logprob row of `pick` in the widened row (the target's raw distribution): its log-softmax, the `k` best
/// (logit descending, ties by the lower id), and with `rank` vLLM's rank (tokens at or above the pick, itself included).
pub fn logprobRow(row: []const u16, dtype: Dtype, pick: u32, k: u8, rank: bool) lanes.LogprobRow {
    var top: f64 = -std.math.inf(f64);
    for (0..row.len) |i| top = @max(top, widen(row, dtype, i));
    var total: f64 = 0;
    for (0..row.len) |i| total += @exp(@as(f64, widen(row, dtype, i)) - top);
    const lse = top + @log(total);
    const n = @min(k, lanes.logprob.max_top);
    var r: lanes.LogprobRow = .{ .token = pick, .logprob = @floatCast(@as(f64, widen(row, dtype, pick)) - lse), .count = n };
    var vals: [lanes.logprob.max_top]f32 = undefined;
    var filled: usize = 0;
    if (n > 0) for (0..row.len) |i| {
        const v = widen(row, dtype, i);
        if (filled == n and !(v > vals[n - 1])) continue; // a later equal value loses to the lower id
        var at = if (filled < n) filled else n - 1;
        if (filled < n) filled += 1;
        while (at > 0 and v > vals[at - 1]) : (at -= 1) {
            vals[at] = vals[at - 1];
            r.ids[at] = r.ids[at - 1];
        }
        vals[at] = v;
        r.ids[at] = @intCast(i);
    };
    for (0..n) |i| r.logprobs[i] = @floatCast(@as(f64, vals[i]) - lse);
    if (rank) {
        const pv = widen(row, dtype, pick);
        for (0..row.len) |i| r.rank += @intFromBool(widen(row, dtype, i) >= pv);
    }
    return r;
}

test "a logprob row is the log-softmax of the raw row, its best tokens by value then id" {
    // bf16 2.0, 1.0, 2.0, 0.0
    const row = [_]u16{ 0x4000, 0x3f80, 0x4000, 0x0000 };
    const r = logprobRow(&row, .bf16, 1, 3, true);
    const lse = @log(2 * @exp(2.0) + @exp(1.0) + 1.0);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(1.0 - lse)), r.logprob, 1e-6);
    try std.testing.expectEqualSlices(u32, &.{ 0, 2, 1 }, r.ids[0..3]);
    try std.testing.expectApproxEqAbs(@as(f32, @floatCast(2.0 - lse)), r.logprobs[0], 1e-6);
    try std.testing.expectEqual(r.logprob, r.logprobs[2]);
    try std.testing.expectEqual(@as(u32, 3), r.rank);
    var p: f64 = 0;
    for (r.logprobs[0..3]) |lp| p += @exp(@as(f64, lp));
    try std.testing.expect(p <= 1.0);
    try std.testing.expectEqual(@as(u8, 0), logprobRow(&row, .bf16, 3, 0, false).count);
}

test "argmax keeps the first of equal values; bf16 and fp16 widen exactly" {
    const one_bf16: u16 = 0x3f80;
    const row = [_]u16{ 0, one_bf16, one_bf16, 0 };
    try std.testing.expectEqual(@as(u32, 1), argmax(&row, .bf16));
    const one_f16: u16 = 0x3c00;
    try std.testing.expectEqual(@as(f32, 1.0), widen(&.{one_f16}, .f16, 0));
}

test "greedy draws the argmax" {
    const row = [_]u16{ 0x3c00, 0x4000, 0x3e00 };
    try std.testing.expectEqual(@as(u32, 1), try draw(std.testing.allocator, &row, .f16, null, 0));
}

test "a masked row draws only allowed tokens, greedy and sampled" {
    const bf: u16 = 0x3f80; // 1.0
    const big: u16 = 0x4000; // 2.0
    var row = [_]u16{ big, bf, 0, bf, big };
    // tokens 1 and 3 allowed: the largest allowed is 1 (first of equals), not 0 or 4
    mask(&row, .bf16, &.{0b01010});
    try std.testing.expectEqual(@as(u32, 1), argmax(&row, .bf16));
    try std.testing.expectEqual(@as(u16, 0xFF80), row[0]);
    const s: lanes.Sampling = .{ .seed = 3, .temperature = 1.0 };
    for (0..32) |p| {
        const t = try draw(std.testing.allocator, &row, .bf16, s, p);
        try std.testing.expect(t == 1 or t == 3);
    }
    // a mask shorter than the row disallows the rest
    var tail = [_]u16{ bf, bf, big };
    mask(&tail, .f16, &.{});
    try std.testing.expectEqual(@as(u16, 0xFC00), tail[2]);
}
