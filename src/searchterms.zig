// zig fmt: off

//! Search algorithm constants.

const std = @import("std");
const lib = @import("lib.zig");

const assert = std.debug.assert;

pub const terms: *const Terms = &default_terms;

const default_terms: Terms = std.mem.zeroInit(Terms, .{});

pub const Terms = struct {

    late_move_reduction_table_math: LateMoveReductionTableMath = .{},
    history_math: HistoryMath = .{},
    correction_history_math: CorrectionHistoryMath = .{},
    tt_entry: TTEntry = .{},
    iterative_deepening: IterativeDeepening_ = .{},
    internal_iterative_deepening: InternalIterativeDeepening_ = .{},
    reversed_futility_pruning: ReversedFutilityPruning = .{},
    razoring: Razoring = .{},
    futility_pruning: FutilityPruning = .{},
    history_pruning: HistoryPruning = .{},
    see_pruning: SeePruning = .{},
    late_move_reduction: LateMoveReduction_ = .{},
    quiescence_futility_pruning: QuiescenceFutilityPruning = .{},
    search_historylist_size: SearchHistoryListSize = .{},

    pub const LateMoveReductionTableMath = struct {
        comptime noisy_base: f32 = -0.24,
        comptime noisy_divider: f32 = 2.60,
        comptime quiet_base: f32 = 0.80,
        comptime quiet_divider: f32 = 2.04,
    };

    pub const HistoryMath = struct {
        /// Depth multiplier.
        comptime scale: i16 = 146,
        /// Maximum of 1 bonus.
        comptime max_bonus: i16 = 1282,
        /// Maximum history score of one entry.
        comptime max_score: i16 = 16384,
    };

    pub const CorrectionHistoryMath = struct {
        comptime scale: i32 = 256,
        comptime max_entry_bonus: i32 = 32,
        comptime max_applied_correction: i32 = 160,
        comptime is_big_correction_margin: i32 = 120,
    };

    pub const TTEntry = struct {
        /// Weight of TT entry depth.
        comptime depth_weight: i32 = 1024,
        /// Weight of TT Entry age (penalty).
        comptime age_penalty_weight: i32 = 1024,
    };

    pub const IterativeDeepening_ = struct {
        /// The margin used to establish the average stability of search scores.
        eval_stability_margin: Tunable(i32) = tunable(i32, 10, 8, 24, 4),
        /// The depth divider to give stability some slack.
        comptime eval_stability_slack_depth_divider: u8 = 12,
    };

    pub const InternalIterativeDeepening_ = struct {
        /// Minimum depth for applying.
        min_depth: Tunable(i32) = tunable(i32, 8, 8, 8, 0),
    };

    pub const ReversedFutilityPruning = struct {
        /// Maximum depth for applying.
        max_depth: Tunable(i32) = tunable(i32, 6, 4, 8, 0), // step 1
        /// Base margin for eval beating beta.
        base_margin: Tunable(i32) = tunable(i32, 21, 10, 60, 0), //step 9
        /// Delta for margin when eval is improving:
        improving_margin: Tunable(i32) = tunable(i32, 40, 20, 80, 0), //step 9
        /// Delta for margin wher eval is not improving:
        not_improving_margin: Tunable(i32) = tunable(i32, 74, 40, 100, 0), //step 9
        /// Divider for margin based on the parentnode's history score.
        history_div: Tunable(i32) = tunable(i32, 819, 400, 1600, 97),
    };

    pub const Razoring = struct {
        /// Base value to add to eval.
        base_margin: Tunable(i32) = tunable(i32, 50, 50, 50, 0),
        /// Depth multiplier.
        mul: Tunable(i32) = tunable(i32, 220, 220, 220, 0),
        /// Depth quadratic multiplier.
        quad: Tunable(i32) = tunable(i32, 96, 96, 96, 0),
    };

    pub const FutilityPruning = struct {
        /// The maximum depth for applying.
        max_depth: Tunable(i32) = tunable(i32, 8, 4, 8, 1),
        /// Default margin.
        base_margin: Tunable(i32) = tunable(i32, 196, 196, 196, 0),
        /// Depth multiplier.
        depth_mul: Tunable(i32) = tunable(i32, 96, 96, 96, 0),
    };

    pub const HistoryPruning = struct {
        /// The maximum depth for applying.
        max_depth: Tunable(i32) = tunable(i32, 5, 5, 5, 0),
        /// Offset for pruning a quiet move.
        quiet_offset: Tunable(i32) = tunable(i32, -518, -518, -518, 0),
        /// Depth multiplier for quiet move.
        quiet_mul: Tunable(i32) = tunable(i32, 1830, 1830, 1830, 0),
    };

    pub const SeePruning = struct {
        /// Maximum depth for applying.
        max_depth: Tunable(i32) = tunable(i32, 8, 6, 10, 1), // #testing 6 ipv 8?
        /// Depth multiplier for quiet moves.
        quiet_mul: Tunable(i32) = tunable(i32, 20, 10, 140, 20),
        /// Depth multiplier for noisy moves.
        noisy_mul: Tunable(i32) = tunable(i32, 80, 20, 140, 20),
        /// Prune more or less using move history score.
        quiet_history_div: Tunable(i32) = tunable(i32, 655, 200, 6000, 1000),
        /// Prune more or less using move history score.
        noisy_history_div: Tunable(i32) = tunable(i32, 546, 200, 6000, 1000),
    };

    pub const LateMoveReduction_ = struct {
        /// The minimum depth for applying lmr.
        min_depth: Tunable(i32) = tunable(i32, 3, 3, 3, 0),
        /// Divider for calculating extra reduction (or negative reduction) based on a quiet move history score.
        quiet_history_div: Tunable(i32) = tunable(i32, 3509, 3509, 3509, 0),
        /// Divider for calculating extra reduction (or negative reduction) based on a noisy move history score.
        noisy_history_div: Tunable(i32) = tunable(i32, 14035, 14035, 14035, 0),
    };

    pub const QuiescenceFutilityPruning = struct {
        /// The margin added to the initial best score to allow pruning.
        margin: i32 = 101,
    };

    pub const SearchHistoryListSize = struct {
        /// The list sizes for moves that did not beat alpha.
        /// The idea is to not punish very late moves even more. It also saves stack space and processing.
        comptime for_quiets: u8 = 18,
        comptime for_noisies: u8 = 10,
    };

};

fn tunable(comptime T: type, comptime value: T, min: T, max: T, step: T) Tunable(T) {
    comptime if (lib.is_tuning_search) {
        return Tunable(T).init(value, min, max, step);
    }
    else {
        return Tunable(T).init(value);
    };
}

pub fn Tunable(comptime T: type) type {
    return if (lib.is_tuning_search) TunableEntry(T) else FixedEntry(T);
}

pub fn TunableEntry(comptime T: type) type {
    return struct {
        val: T,
        min: T,
        max: T,
        step: T,

        fn init(comptime value: T, min: T, max: T, step: T) TunableEntry(T) {
            comptime {
               assert(min <= value);
               assert(max >= value);
               assert(step >= 0);
            }
            return .{
                .val = value,
                .min = min,
                .max = max,
                .step = step,
            };
        }
    };
}

pub fn FixedEntry(comptime T: type) type {
    return struct {
        val: T,

        fn init(comptime value: T) FixedEntry(T) {
            return .{
                .val = value,
            };
        }
    };
}
