// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");

const Color = types.Color;
const ScorePair = types.ScorePair;

pub fn run() !void {
    // Nothing here.
}

/// Evaluator calls this for each used scorepair.
pub fn register_scorepair_usage(sp: *ScorePair, multiply: u8, us: Color, debugargs: anytype) void {
    lib.only_when_tuning();
    _ = sp;
    _ = multiply;
    _ = us;
    _ = debugargs;
}

/// Evaluator calls this just before returning its result.
pub fn register_final_result(hce_pair: ScorePair, eval: i32) void {
    lib.only_when_tuning();
    _ = hce_pair;
    _ = eval;
}

