// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

pub fn main() !void
{
    @setFloatMode(.optimized);
    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
        //try @import("tests.zig").run_silent_debugmode_tests();
    }

    // try @import("tests/enginetests.zig").lichess_puzzles();

    // const h = @import("history.zig");
    // const calc = h.corr_calc;

    // const depth: i32 = 1;
    // var e: i16 = 0;
    
    // const err: i32 = 20;

    // const b: i32 = std.math.clamp(@divTrunc(err * depth, 8), -16384, 16384);   
    // const bonus: i16 = @intCast(b);

    // calc.apply_bonus(&e, bonus);

    // //const ediv = bonus += 66 * cv / 512;

    // std.debug.print("++ d {} b {} e {} ediv {}\n", .{ depth, bonus, e, @divTrunc(e, 8) });

    // var e: i16 = 0;
    // for (1..32) |d| {
    //     const depth: i32 = @intCast(d);
    //     const bonus = calc.get_bonus(depth);
    //     calc.apply_bonus(&e, bonus);
    //     std.debug.print("++ d {} b {} e {}\n", .{ d, bonus, e });
    // }

    // for (1..32) |d| {
    //     const depth: i32 = @intCast(d);
    //     const bonus = calc.get_bonus(depth);
    //     calc.apply_bonus(&e, -bonus);
    //     std.debug.print("-- d {} b {} e {}\n", .{ d, bonus, e });
    // }

    uci.run();
}

