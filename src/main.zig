// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

pub fn main() !void {
    comptime @setFloatMode(.optimized);

    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug and lib.program == .uci_engine) {
        try @import("tests.zig").run_silent_debugmode_tests();
    }

    switch (lib.program) {
        .uci_engine => {
            uci.run();
        },
        .lichess_tool => {
            try @import("tools/lichess.zig").convert_lichess_dataset();
        },
        .tuner => {
            // try @import("tuner.zig").run();
        }
    }
}
