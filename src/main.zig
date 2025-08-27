// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const uci = @import("uci.zig");
const perft = @import("perft.zig");
const masks = @import("masks.zig");
const tests = @import("tests.zig");
const eval = @import("eval.zig");
const funcs = @import("funcs.zig");
const squarepairs = @import("squarepairs.zig");

pub fn main() !void
{
    lib.initialize();
    defer lib.finalize();

    //  Debug tests.
    if (comptime lib.is_debug)
    {
        try tests.run_silent_debugmode_test();
        try tests.run_testfile(false, 1);
        try tests.test_flip(false);
    }

    //try @import("tools/kaggle.zig").create_anna_files();

    uci.run();
}
