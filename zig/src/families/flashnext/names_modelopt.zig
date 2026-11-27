//! A requested tensor name as the checkpoint's weight_map stores it: the name itself or its exported spelling.
const std = @import("std");

const aliases = .{
    .{ "language_model.model.", "model.language_model." },
    .{ "language_model.lm_head.", "lm_head." },
    .{ "language_model.mtp.", "mtp." },
};

/// `requested` if `weight_map` holds it, else its exported spelling (in `buf`); error.MissingTensor if neither.
pub fn resolve(weight_map: std.json.ObjectMap, requested: []const u8, buf: []u8) ![]const u8 {
    if (weight_map.contains(requested)) return requested;
    inline for (aliases) |pair| {
        if (std.mem.startsWith(u8, requested, pair[0])) {
            const name = try std.fmt.bufPrint(buf, "{s}{s}", .{ pair[1], requested[pair[0].len..] });
            if (weight_map.contains(name)) return name;
        }
    }
    return error.MissingTensor;
}

/// `requested`'s exported spelling (in `buf`), or `requested` when no alias names its prefix: a stem, unchecked.
pub fn exported(requested: []const u8, buf: []u8) ![]const u8 {
    var dotted: [512]u8 = undefined;
    const stem = try std.fmt.bufPrint(&dotted, "{s}.", .{requested});
    inline for (aliases) |pair| {
        if (std.mem.startsWith(u8, stem, pair[0])) {
            const out = try std.fmt.bufPrint(buf, "{s}{s}", .{ pair[1], stem[pair[0].len..] });
            return out[0 .. out.len - 1];
        }
    }
    return requested;
}

test "canonical checkpoint aliases preserve exact stored namespace and prefer a direct key" {
    const a = std.testing.allocator;
    const parsed = try std.json.parseFromSlice(std.json.Value, a,
        \\{"model.language_model.layers.0.x.weight":"a","lm_head.weight":"b","mtp.x":"c","language_model.model.direct":"d","model.language_model.direct":"e"}
    , .{});
    defer parsed.deinit();
    const wm = parsed.value.object;
    var buf: [512]u8 = undefined;
    try std.testing.expectEqualStrings("model.language_model.layers.0.x.weight", try resolve(wm, "language_model.model.layers.0.x.weight", &buf));
    try std.testing.expectEqualStrings("lm_head.weight", try resolve(wm, "language_model.lm_head.weight", &buf));
    try std.testing.expectEqualStrings("mtp.x", try resolve(wm, "language_model.mtp.x", &buf));
    try std.testing.expectEqualStrings("language_model.model.direct", try resolve(wm, "language_model.model.direct", &buf));
    try std.testing.expectError(error.MissingTensor, resolve(wm, "language_model.model.missing", &buf));
    try std.testing.expectEqualStrings("model.language_model.layers.3", try exported("language_model.model.layers.3", &buf));
    try std.testing.expectEqualStrings("lm_head", try exported("language_model.lm_head", &buf));
    try std.testing.expectEqualStrings("mtp", try exported("mtp", &buf));
}
