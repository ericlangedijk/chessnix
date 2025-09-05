// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const tests = @import("tests.zig");
const uci = @import("uci.zig");

pub fn main() !void
{
    try lib.initialize();
    defer lib.finalize();

    //  Debug tests.
    if (comptime lib.is_debug) {
        try tests.run_silent_debugmode_tests();
        tests.print_struct_sizes();
    }

    // var i: i32 = 0;
    // while (true) {
    //     i = -search(i) orelse break;
    //     lib.io.debugprint("{},", .{i});
    // }

    uci.run();
}

// fn search(v: i32) Error!i32 {
//     if (v > 40) return ;
//     if (v < 40) return null;
//     return v + 2;
// }

// const Error = error { timeout };