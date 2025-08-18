// zig fmt: off

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const tests = @import("tests.zig");
const uci = @import("uci.zig");

pub fn main() !void
{
    lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug)
    {
        try tests.run_silent_debugmode_test();
        try tests.run_testfile(false, 1);
    }

    uci.run();
}
