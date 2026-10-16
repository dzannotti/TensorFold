//! OpenAI's presence and frequency penalties, vLLM's repetition penalty and logit_bias, on a host logits row.
//! vLLM's order: the bias, then repetition (prompt and reply tokens), frequency and presence (reply tokens).
const std = @import("std");

pub const Bias = struct { token: u32, value: f32 };

pub const Penalties = struct {
    presence: f32 = 0,
    frequency: f32 = 0,
    repetition: f32 = 1,
    bias: []const Bias = &.{},

    /// Whether a row changes at all.
    pub fn active(p: Penalties) bool {
        return p.presence != 0 or p.frequency != 0 or p.repetition != 1 or p.bias.len > 0;
    }

    /// `row` (one logit a token) after the penalties; the reply so far is `reply` then `more` (a window's rows before).
    pub fn apply(p: Penalties, gpa: std.mem.Allocator, row: []f32, prompt: []const u32, reply: []const u32, more: []const u32) !void {
        for (p.bias) |b| if (b.token < row.len) {
            row[b.token] += b.value;
        };
        if (p.repetition != 1) {
            var seen = try std.DynamicBitSetUnmanaged.initEmpty(gpa, row.len);
            defer seen.deinit(gpa);
            for ([_][]const u32{ prompt, reply, more }) |part| for (part) |t| if (t < row.len and !seen.isSet(t)) {
                seen.set(t);
                row[t] = if (row[t] > 0) row[t] / p.repetition else row[t] * p.repetition;
            };
        }
        if (p.presence == 0 and p.frequency == 0) return;
        const sorted = try std.mem.concat(gpa, u32, &.{ reply, more });
        defer gpa.free(sorted);
        std.mem.sort(u32, sorted, {}, std.sort.asc(u32));
        var i: usize = 0;
        while (i < sorted.len) {
            const t = sorted[i];
            var n: usize = 1;
            while (i + n < sorted.len and sorted[i + n] == t) n += 1;
            i += n;
            if (t < row.len) row[t] -= p.frequency * @as(f32, @floatFromInt(n)) + p.presence;
        }
    }
};

test "penalties follow vLLM: bias, repetition over prompt and reply, frequency and presence over the reply" {
    const gpa = std.testing.allocator;
    var row = [_]f32{ 2, -2, 1, 1, 0 };
    const p: Penalties = .{ .presence = 0.5, .frequency = 0.25, .repetition = 2, .bias = &.{.{ .token = 4, .value = 3 }} };
    try p.apply(gpa, &row, &.{ 0, 1 }, &.{ 2, 2 }, &.{1});
    // 0: prompt only, repeated: 2 / 2; 1: prompt and reply once: -2 * 2 - 0.25 - 0.5; 2: reply twice: 1 / 2 - 0.5 - 0.5
    try std.testing.expectEqualSlices(f32, &.{ 1, -4.75, -0.5, 1, 3 }, &row);
    try std.testing.expect(!(Penalties{}).active());
    try std.testing.expect(p.active());
}
