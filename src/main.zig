// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

     const p = @import("position.zig");
      const bb = @import("bitboards.zig");
      const t = @import("types.zig");
      const f = @import("funcs.zig");


pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    //  Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    // try @import("tests/enginetests.zig").lichess_puzzles();

    uci.run();
}
