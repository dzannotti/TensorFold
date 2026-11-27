//! The model-free kernels' code objects, built by hipcc for -Dhip-arch and embedded, one a source group.
const objects = @import("hip_kernels");

/// The source groups of zig/kernels/hip, in the order the build embeds them.
pub const Group = enum {
    ops,
    act,
    attention,
    gated_delta,
    prefill,
    decode,
    plan,
    affine_tiles,
    affine_dot2,
    flashnext,
    mamba,
    glm_ocr,
    joint_schema,
    lighton_ocr,
    qwen_vision,
    draft,
    exl3,
    plan_rows,
    inkling2,
    dsv41,
    fp4local,
};
pub const group_count = @typeInfo(Group).@"enum".field_names.len;

/// The architecture the objects were built for; the loader takes them only on that device.
pub const arch: []const u8 = objects.arch;

/// The objects in `Group` order; empty in a build without hipcc.
pub const images: [group_count][]align(8) const u8 = objects.images;

/// The Loom kernels (zig/kernels/loom), one code object each, in zig/build/loom.zig's `kernels` order.
pub const Loom = enum { mxfp8_gemv, nvfp4_expert_gate_up, nvfp4_expert_down, sparse_attend, idx_scores, idx_select };
pub const loom_count = @typeInfo(Loom).@"enum".field_names.len;

/// The architecture the Loom objects were built for (-Dloom-target); empty in a build without -Dloom-compile.
pub const loom_arch: []const u8 = objects.loom_arch;

/// The Loom objects in `Loom` order; empty without -Dloom-compile, where hipcc's kernels run in their place.
pub const loom_images: [loom_count][]align(8) const u8 = objects.loom_images;

/// Each Loom object's artifact manifest (loom-compile --artifact-manifest=details): the interfaces launches check.
pub const loom_manifests: [loom_count][]const u8 = objects.loom_manifests;
