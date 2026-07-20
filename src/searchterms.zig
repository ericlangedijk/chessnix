// zig fmt: off

//! Search algorithm constants.

const std = @import("std");

pub const spsa: bool = false;

pub const terms = if (spsa) &tunable_terms else &default_terms;
const default_terms: Terms = std.mem.zeroInit(Terms, .{});
var tunable_terms: Terms = if (spsa) default_terms else @compileError("Using tunable_terms");

pub const Terms = struct {

    late_move_reduction_table_computing: struct {
        comptime noisy_base: f32 = -0.24,
        comptime noisy_divider: f32 = 2.60,
        comptime quiet_base: f32 = 0.80,
        comptime quiet_divider: f32 = 2.04,
    },

    history_math: struct {
        /// depth multiplier.
        comptime scale: i16 = 146,
        /// Maximum of 1 bonus.
        comptime max_bonus: i16 = 1282,
        /// Maximum history score of one entry.
        comptime max_score: i16 = 16384,
    },

    correction_history: struct {
        comptime scale: i32 = 256,
        comptime max_entry_bonus: i32 = 32,
        comptime max_applied_correction: i32 = 160,
        comptime is_big_correction_margin: i32 = 120,
    },

    iterative_deepening: struct {
        /// The margin used to establish the average stability of a search score.
        eval_stability_margin: Tunable(i32) = tunable(i32, 10, 10, 10, 0),
        /// The depth divider to give stability some slack.
        comptime eval_stability_slack_depth_divider: u8 = 12,
    },

    internal_iterative_deepening: struct {
        /// Minimum depth for applying.
        min_depth: i32 = 8,
    },

    reversed_futility_pruning: struct {
        /// Maximum depth for applying.
        max_depth: Tunable(i32) = tunable(i32, 6, 4, 8, 1),
        /// Base margin for eval beating beta.
        base_margin: Tunable(i32) = tunable(i32, 21, 10, 40, 10),
        /// Delta for margin when eval is improving:
        improving_margin: Tunable(i32) = tunable(i32, 40, 10, 100, 10),
        /// Delta for margin wher eval is not improving:
        not_improving_margin: Tunable(i32) = tunable(i32, 74, 10, 100, 10),
    },

    razoring: struct {
        /// Base value to add to eval.
        base_margin: Tunable(i32) = tunable(i32, 50, 50, 50, 0),
        /// Depth multiplier.
        mul: Tunable(i32) = tunable(i32, 220, 220, 220, 0),
        /// Depth quadratic multiplier.
        quad: Tunable(i32) = tunable(i32, 96, 96, 96, 0),
    },

    futility_pruning: struct {
        /// The maximum depth for applying.
        max_depth: Tunable(i32) = tunable(i32, 8, 9, 8, 0),
        /// Default margin.
        margin_base: Tunable(i32) = tunable(i32, 196, 196, 196, 0),
        /// Depth multiplier.
        depth_mul: Tunable(i32) = tunable(i32, 96, 96, 96, 0),
    },

    history_pruning: struct {
        /// The maximum depth for applying.
        max_depth: Tunable(i32) = tunable(i32, 5, 5, 5, 0),
        /// Offset for quiet move.
        quiet_offset: Tunable(i32) = tunable(i32, -518, -518, -518, 0),
        /// Depth multiplier for quiet move.
        quiet_mul: Tunable(i32) = tunable(i32, 1830, 1830, 1830, 0),
    },

    see_pruning: struct {
        /// Maximum depth for applying.
        max_depth: Tunable(i32) = tunable(i32, 8, 8, 8, 0),
        /// Depth multiplier for quiet moves.
        quiet_mul: Tunable(i32) = tunable(i32, -60, -60, -60, 0),
        /// Depth multiplier for noisy moves.
        noisy_mul: Tunable(i32) = tunable(i32, -95, -95, -95, 0),
        /// Prune more or less using move history score.
        quiet_history_div: Tunable(i32) = tunable(i32, 2621, 2621, 2621, 0),
        /// Prune more or less using move history score.
        noisy_history_div: Tunable(i32) = tunable(i32, 546, 546, 546, 0),
    },

    late_move_reduction: struct {
        /// The minimum depth for applying lmr.
        min_depth: Tunable(i32) = tunable(i32, 3, 3, 3, 0),
        /// Divider for calculating extra reduction (or negative reduction) based on a move's history score.
        history_quiet_div: Tunable(i32) = tunable(i32, 14035, 14035, 14035, 0), // Currently 4 scores of max 16384
        history_noisy_div: Tunable(i32) = tunable(i32, 14035, 14035, 14035, 0), // 1 score of max 16384
    },

    quiescence_futility_pruning: struct {
        /// The margin added to the initial best score to allow pruning.
        margin: i32 = 101,
    },

    search_historylist_size: struct {
        /// The list sizes for moves that did not beat alpha.
        /// The idea is to not punish very late moves even more. It also saves stack space and processing.
        comptime for_quiets: u8 = 18,
        comptime for_noisies: u8 = 10,
    },

    tt_entry: struct {
        /// Weight of TT entry depth.
        depth_weight: i32 = 1024,
        /// Weight of TT Entry age (penalty).
        age_penalty_weight: i32 = 1024,
    },
};

fn tunable(comptime T: type, comptime value: T, min: T, max: T, step: T) Tunable(T) {
    comptime if (spsa) {
        return .{ .val = value, .min = min, .max = max, .step = step };
    } else {
        return .{ .val = value };
    };
}

pub fn Tunable(T: type) type {
    comptime if (spsa) {
        return struct {
            val: T,
            min: T,
            max: T,
            step: T,
        };
    } else {
        return struct {
            val: T,
        };
    };
}



