// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

// debug wip
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const eval = @import("eval.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const perft = @import("perft.zig");

pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    // Error at depth 1, expected 17, found 22

    //  Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    // var p: position.Position = .empty;
    // try p.set(position.fen_kiwipete);
    // p.draw();

    // // const m: types.Move = .create(.E2, .A6, types.Move.capture);
    // // p.do_move(.WHITE, m);
    // // p.draw();

    // // funcs.print_bitboard(p.bitboard_all);

    // for (types.Piece.all) |pc| {
    //     const p1 = pc.piecetype();
    //     const p2 = pc.piecetype_exp();
    //lib.io.debugprint("{}\n", .{@sizeOf(position.Position)});

    // }


    // perft.run(&p, 3);


    uci.run();
}
