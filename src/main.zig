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

    // const cap: i16 = 1200;
    // const base: i16 = 16;
    // var score: i16 = 0;

    // for (1..120) |depth| {
    //     const d: i32 = @intCast(depth);
    //     score += @intCast(@divTrunc(d * base * (cap - score), cap));
    //     std.debug.print("INC {} {}\n", .{ d, score });
    // }

    // for (1..120) |depth| {
    //     const d: i32 = @intCast(depth);
    //     score -= @intCast(@divTrunc(d * base * (cap - score), cap));
    //     std.debug.print("DEC {} {}\n", .{ d, score });
    // }


    // // const cap: i16 = 1200;
    // // const base: i16 = 8;
    // var score: i32 = 0;

    // for (1..50) |d| {
    //     const depth: i32 = @intCast(d);
    //     const bonus: i32 = get_bonus(depth);
    //     history_bonus(&score, bonus);
    //     std.debug.print("depth {} bonus {} score {}\n", .{depth, bonus, score});
    // }

    // for (1..50) |d| {
    //     const depth: i32 = @intCast(d);
    //     const bonus: i32 = get_bonus(depth);
    //     history_bonus(&score, -bonus);
    //     std.debug.print("depth {} bonus {} score {}\n", .{depth, -bonus, score});
    // }

    // // for (1..120) |depth| {
    // //     const d: i32 = @intCast(depth);
    // //     // <-- use score, not (cap - score)
    // //     score -= @intCast(@divTrunc(d * base * score, cap));
    // //     std.debug.print("after dec: {}\n", .{score});
    // // }

    uci.run();
}

