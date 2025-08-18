// zig fmt: off

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const tests = @import("tests.zig");
const uci = @import("uci.zig");

const eval = @import("eval.zig");

pub fn main() !void
{
    lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug)
    {
        try tests.run_silent_debugmode_test();
        try tests.run_testfile(false, 1);
    }

    // var pos: position.Position = .empty;
    // var st: position.StateInfo = .empty;
    // for (tests.testpositions) |str|
    // {
    //     try pos.set(&st, str);
    //     const p = pos.phase();
    //     if (p == .Midgame)
    //     {
    //     try pos.print();
    //     std.debug.print("{s}\n", .{@tagName(p)});
    //     }
    // }

    // const x = eval.PestoTables.get(types.PieceType.PAWN, types.Color.WHITE, types.Square.H7);
    // const y = eval.PestoTables.get(types.PieceType.PAWN, types.Color.BLACK, types.Square.H2);
    // std.debug.print("{} {}", .{x, y});



    uci.run();
}
