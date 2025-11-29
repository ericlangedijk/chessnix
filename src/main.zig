// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    // const b = @import("bitboards.zig");
    // const f = @import("funcs.zig");
    // const t = @import("types.zig");
    // f.print_bitboard(b.bb_white_squares);
    // f.print_bitboard(b.bb_black_squares);

    // for (t.Square.all) |sq| {
    //     std.debug.print("{} {}\n", .{ sq.e, sq.color().e});
    // }




    // const h = @import("history.zig");

    // var q: h.QuietHistory2 = std.mem.zeroes(h.QuietHistory2);

    // const ptr: *i16 = &q.piece_from_to[0][0][0];

    // const bonus = h.QuietHistory2.get_bonus(10);

    // for (0..10) |_| {
    //     const scaled_bonus = h.QuietHistory2.scale_bonus(ptr.*, bonus);
    //     ptr.* += scaled_bonus;
    //     std.debug.print("bonus {} scaled_bonus {} value {}\n", .{ bonus, scaled_bonus, ptr.*});
    //     //h.ge
    // }

    uci.run();
}
