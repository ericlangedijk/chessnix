// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");

/// In ReleaseSafe mode we log the error + stacktrace to a file.
pub const panic =
    if (lib.is_release_safe) std.debug.FullPanic(lib.panic_function) else std.debug.FullPanic(std.debug.defaultPanic);

pub fn main() !void {
    comptime @setFloatMode(.optimized);

    try lib.initialize();
    defer lib.finalize();
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
