// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");


const funcs = @import("funcs.zig");
const types = @import("types.zig");
const hce = @import("hce.zig");
const hcetables = @import("hcetables.zig");
const attacks = @import("attacks.zig");

pub fn main() !void
{
    @setFloatMode(.optimized);
    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
       // try @import("tests.zig").run_silent_debugmode_tests();
    }

    //try @import("tests/enginetests.zig").lichess_puzzles();

    // const types = @import("types.zig");

    // for (types.Piece.all) |p| {
    //     for (0..24 + 1) |ph| {
    //         std.debug.print("ph {} piece {t} value {}\n", .{ph, p.e, types.phased_piece_values[ph][p.u]});
    //     }
    // }

    // var i: u5 = 0;

    // for (0..42) |_| {
    //     i +%= 1;
    //     std.debug.print("{}, ", .{i});
    // }

    //const E = @import("search.zig").ExtMove;
    //std.debug.print("{} {}", .{ @sizeOf(E), @bitSizeOf(E)});

    // for (types.Square.all) |from| {
    //     var bb: u64 = attacks.get_knight_attacks(from);
    //     while (funcs.bitloop(&bb)) |to| {
    //         const v = from_to_score(types.PieceType.KNIGHT, from, to, 24);
    //         std.debug.print("{t} {t} -> {}\n", .{ from.e, to.e, v });
    //     }
    // }


    uci.run();
}


// fn from_to_score(pt: types.PieceType, from: types.Square, to: types.Square, phase: u8) types.Value {

//     //const pt: u4 = ex.piece.piecetype().u;
//     const from_score: types.ScorePair = hcetables.piece_square_table[pt.u][from.u];
//     const to_score: types.ScorePair = hcetables.piece_square_table[pt.u][to.u];
//     // const f: Value = hce.sliding_score(self.pos, from);
//     // const t: Value = hce.sliding_score(self.pos, to);
//     // return if (t >)

//     const r: types.ScorePair = to_score.sub(from_score);
//     return sliding_score(phase, r);
// }


// pub fn sliding_score(ph: u8, score: types.ScorePair) types.Value {
//     const max: u8 = comptime types.max_phase;
//     const phase: u8 = @min(max, ph);
//     const mg: types.Value = score.mg;
//     const eg: types.Value = score.eg;
//     return @divTrunc(mg * phase + eg * (max - phase), max);
// }



