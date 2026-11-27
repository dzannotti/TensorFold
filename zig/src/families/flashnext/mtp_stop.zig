//! An MTP chain's stop rule over its drafts' probabilities: the CUDA and HIP heads' one rule.
const std = @import("std");

/// c > 0: a later draft only at or above c; c < 0: while p1 * ... * pj >= -c; 0: every draft.
pub const Stop = struct {
    c: f64,
    product: f64 = 1.0,

    /// Whether draft j (0-based) with probability `p` is proposed, and whether the chain may go on.
    pub fn take(s: *Stop, j: usize, p: f64) struct { propose: bool, more: bool } {
        if (s.c > 0) {
            const low = p < s.c;
            if (low and j > 0) return .{ .propose = false, .more = false };
            return .{ .propose = true, .more = !low };
        }
        if (s.c < 0) {
            s.product *= p;
            // below the line it ends the chain: no later product can reach it again
            if (s.product < -s.c) return .{ .propose = j == 0, .more = false };
        }
        return .{ .propose = true, .more = true };
    }
};

test "the stop rule: per draft above zero, the running product below it, every draft at zero" {
    var per: Stop = .{ .c = 0.6 };
    try std.testing.expect(per.take(0, 0.5).propose and !per.take(0, 0.5).more);
    try std.testing.expect(!per.take(1, 0.5).propose);
    var prod: Stop = .{ .c = -0.4 };
    try std.testing.expect(prod.take(0, 0.8).more); // 0.8
    try std.testing.expect(prod.take(1, 0.6).more); // 0.48
    const t = prod.take(2, 0.8); // 0.384: below 0.4
    try std.testing.expect(!t.propose and !t.more);
    var all: Stop = .{ .c = 0 };
    try std.testing.expect(all.take(5, 0.01).propose and all.take(6, 0.01).more);
}
