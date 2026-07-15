// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");

pub fn main() !void {
    comptime @setFloatMode(.optimized);

    try lib.initialize();
    defer lib.finalize();
    //try @import("./local/heuristicswriter.zig").run_viri_avg_movecount(); _ = try lib.io.readline(); if (true) return;
    //try @import("./local/heuristicswriter.zig").run_viri_avg_threatcount(); _ = try lib.io.readline(); if (true) return;
    //try @import("./local/misc.zig").test_threats(); if (true) return;
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
