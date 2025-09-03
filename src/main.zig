// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const tests = @import("tests.zig");
const uci = @import("uci.zig");

pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    //  Debug tests.
    if (comptime lib.is_debug)
    {
        try tests.run_silent_debugmode_tests();
        tests.print_struct_sizes();
    }

    //try @import("tools/lichess.zig").reorder_lichess_puzzles();

    uci.run();
}
