// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    //  Debug tests.
    if (comptime lib.is_debug) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    // try @import("tests/enginetests.zig").lichess_puzzles();

    // const ar = combos;
    // for (ar) |combo| {
    //     std.debug.print("{any}\n", .{ combo });
    // }

    // var i: i32 = -217;
    // while (i < 17) : (i += 1) {
    //     const a = i >> 1;
    //     const b = @divTrunc(i, 2);
    //     std.debug.print("{} {} {}\n", .{i, a, b});
    // }


    uci.run();
}

// const Combo = packed struct {
//     king: u3,
//     r1: u3,
//     r2: u3,
// };

// const combos: [168]Combo = compute_combos();

// fn compute_combos() [168]Combo {
//     var list: [168]Combo = undefined;
//     var idx: usize = 0;

//     for (0..8) |k| {
//         for (0..8) |r1| {
//             if (r1 == k) continue;
//             for (r1 + 1..8) |r2| {
//                 if (r2 == k) continue;
//                 list[idx] = .{ .king = @truncate(k), .r1 = @truncate(r1), .r2 = @truncate(r2) };
//                 idx += 1;
//             }
//         }
//     }
//     return list;
// }