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

    uci.run();
}

const Ext = packed struct {
    op: u6,
    d: u1,
    w: u1,
    mod: u2,
    reg: u3,
    rm: u3,
};