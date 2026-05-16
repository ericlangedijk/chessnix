// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

pub fn main() !void {
    comptime @setFloatMode(.optimized);

    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
       try @import("tests.zig").run_silent_debugmode_tests();
    }

    // const utils = @import("utils.zig");
    // var rnd: utils.Random = .init_randomized();
    // const r: u64 = rnd.next();


    // const funcs = @import("funcs.zig");
    //const types = @import("types.zig");
    // const position = @import("position.zig");
    // var c: position.Castling = .{};

    // if (r % 2 == 0)
    //     c.white_king_start_file = 4
    //     else c.white_king_start_file = 2;

    // const bb: u64 = c.king_path(types.Color.BLACK, types.CastleType.LONG);
    // //if (bb == 0) return;

    // if (false) funcs.print_bitboard(bb);

    //const bitboards = @import("bitboards.zig");
    //lib.io.debugprint("{} {}\n\n", .{ @sizeOf(types.ExtMove), @bitSizeOf(types.ExtMove) });

    uci.run();
}
