// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");

pub fn main() !void {
    comptime @setFloatMode(.optimized);

    try lib.initialize();
    defer lib.finalize();

    if (lib.is_debug) {
        try @import("debugtests.zig").run();
    }

    switch (lib.program) {
        .uci => {
            try @import("uci.zig").run();
        },
        .hcetuner => {
            try @import("hcetuner.zig").run();
        },
        .lichess_dataset_conversion => {
            try @import("lichess.zig").convert_lichess_dataset_to_viri();
        },
    }
}
