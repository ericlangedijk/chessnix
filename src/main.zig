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

    // Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    // var ar: [2][6][64][6][64]i16 = std.mem.zeroes([2][6][64][6][64]i16);

    // ar[1][5][23][0][0] = 42;
    // ar[1][5][23][0][5] = 43;

    // const entry: *[6][64]i16 = &ar[1][5][23];
    // std.debug.print("{}\n", .{ entry[0][0] });
    // std.debug.print("{}\n", .{ entry[0][5] });

    //std.debug.print("{}\n", .{@sizeOf(p.Position)});
    //std.debug.print("{}\n", .{@sizeOf(p.Layout)});

    uci.run();
}
