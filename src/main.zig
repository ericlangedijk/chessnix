// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const perft = @import("perft.zig");
const tests = @import("tests.zig");
const data = @import("data.zig");
const uci = @import("uci.zig");

const Position = position.Position;
const MoveStorage = position.MoveStorage;

const ctx = lib.ctx;

pub fn main() !void
{
    lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (!lib.is_release)
    {
        try tests.run_silent_debugmode_test();
        try tests.run_testfile(false, 1);
    }

    uci.run() catch |err|
    {
        std.debug.print("ERROR: {}.\n\nPress any key to quit\n", .{ err });
        _ = lib.in.readByte() catch {};
    };
}
