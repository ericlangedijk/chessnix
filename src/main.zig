// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

// const t = @import("types.zig");
 const p = @import("position.zig");
// const f = @import("funcs.zig");
// const bitboards = @import("bitboards.zig");

pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    //  Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }


    // const ev: i32 = 128;
    // for (1..101) |d| {
    //     const r: f32 = @floatFromInt(d);
    //     const e: f32 = ev * @max(0.0, 1.0 - (r/100.0) * (r/100.0));
    //     const i: i32 = @intFromFloat(e);
    //     //const e: i32 = @divTrunc(ev * 100, draw);
    //     const a: i32 = @import("search.zig").Searcher.slide_towards_draw(ev, @truncate(d));
    //     lib.io.debugprint("{} {} {} == {}\n", .{ev, d, i, a});
    // }


    // for (1..128) |depth| {
    //     for (1..100) |move| {
    //         const v = @import("search.zig").lmr_depth_reduction_table[depth][move];
    //         if (depth < 32 and move < 24)
    //             lib.io.debugprint("d = {} m = {} r = {}\n", .{depth, move, v});
    //     }
    // }


    //@import("search.zig").init_reductions();

    //const x = p.Layout.get_layout_ptr(.{ 3, 1, 2, 4, 5, 2, 1, 3 });
    //lib.io.debugprint("{any}\n", .{p.Layout.classic});
    //lib.io.debugprint("{any}\n", .{p.Layout.compute_layout(.{ 0, 7, 4 })});

    //lib.io.debugprint("layoutsize {}\n", .{@sizeOf(p.Layout)});
    //try @import("tests/enginetests.zig").lichess_puzzles();

    // var pos: p.Position = .empty;
    // try pos.set("rnbqkbnr/pppppppp/8/8/4P3/3P4/PPP2PPP/RNBQKBNR w KQkq - 0 1");

    // const b: u64 = pos.by_type(t.PieceType.PAWN);
    // f.print_bitboard(b);
    // const c = f.shift_bitboard(b, .east);
    // f.print_bitboard(c);

    // f.print_bitboard(b & c);

    // pos.draw();
    // const ok = pos.pos_ok();
    // lib.io.debugprint("{}\n", .{ok});

    // lib.io.debugprint("{} {} {}\n", .{t.pawn_protection_table[0].mg, t.pawn_protection_table[1].mg, t.pawn_protection_table[2].mg, });
    // lib.io.debugprint("{} {} {}\n", .{t.pawn_protection_table[3].mg, t.pawn_protection_table[4].mg, t.pawn_protection_table[5].mg, });
    // lib.io.debugprint("{} {} {}\n", .{t.pawn_protection_table[6].mg, t.pawn_protection_table[7].mg, t.pawn_protection_table[8].mg, });
    // lib.io.debugprint("{} {} {}\n", .{t.pawn_protection_table[9].mg, t.pawn_protection_table[10].mg, t.pawn_protection_table[11].mg, });

    // lib.io.debugprint("{}\n", .{ 10 - 5 * 2});
    // //lib.io.debugprint("{}\n", .{ 10 - 5 * 2});

    // var bb = bitboards.bitboard(bitboards.bb_border);
    // while (bb.iter()) |sq| {
    //     lib.io.print("{t}, ", .{ sq.e });
    // }

    // lib.io.print("\n", .{});

    uci.run();
}
