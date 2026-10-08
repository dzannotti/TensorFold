const int4 = @import("int4");
test "HIP launch paths type-check" {
    _ = &int4.Kernels.load;
    _ = &int4.gateUp;
    _ = &int4.down;
    _ = &int4.downPrompt;
    _ = &int4.dense;
    _ = &int4.denseNt;
    _ = &int4.expertsNt;
}
