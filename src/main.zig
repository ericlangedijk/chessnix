// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

const builtin = @import("builtin");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");
const bitboards = @import("bitboards.zig");

pub fn main() !void
{
    @setFloatMode(.optimized);
    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    //try @import("tests/enginetests.zig").lichess_puzzles();

    // for (types.Square.all) |sq| {
    //     std.debug.print("{t}\n", .{ sq.e});
    //     funcs.print_bitboard(bitboards.backward_pawn_masks_black[sq.u]);
    // }

    
    uci.run();
}

