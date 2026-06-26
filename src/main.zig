// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");

pub fn main() !void {
    comptime @setFloatMode(.optimized);

    try lib.initialize();
    defer lib.finalize();

    switch (lib.program) {
        .uci => {
            if (lib.is_debug) {
                try @import("tests.zig").run_silent_debugmode_tests();
            }
            try @import("uci.zig").run();
        },
        .lichess_dataset_conversion => {
            //
        },
        .hcetuner => {
            try @import("hcetuner.zig").run();
        }
    }
}

fn misc() void {
    //
}
