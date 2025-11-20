// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

     const p = @import("position.zig");
      const bb = @import("bitboards.zig");
      const t = @import("types.zig");
      const f = @import("funcs.zig");
      const tt = @import("tt.zig");


pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    //  Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    //lib.io.print("{} {}\n", .{ @sizeOf(tt.Entry), @bitSizeOf(tt.Entry)});

    //try @import("tests/enginetests.zig").lichess_puzzles();
    //try @import("search.zig").debug_compute_reduction_table();

    //const table = @import("hce.zig").see_values;
   // std.debug.print("{any}", .{table});

    uci.run();
}
