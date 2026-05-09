// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

pub fn main(init: std.process.Init) !void {
    comptime @setFloatMode(.optimized);

    try lib.initialize(init.io);
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
       try @import("tests.zig").run_silent_debugmode_tests();
    }

    // const i = 42;
    // const x = @intFromFloat(i);

    switch (lib.program) {
        .uci_engine => {
            uci.run();
        },
        .lichess_tool => {
            try @import("tools/lichess.zig").convert_lichess_dataset();
        },
        .tuner => {
            try @import("tuner.zig").run();
        }
    }
}
