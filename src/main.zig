// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");

pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    uci.run();
}
