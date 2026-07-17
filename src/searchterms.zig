// zig fmt: off

//! Search algorithm constants. Will be tunable later on.

const std = @import("std");
const types = @import("types.zig");

pub const terms = struct {

    pub const late_move_reduction_table_computing = struct {
        pub const noisy_base: f32 = -0.24;
        pub const noisy_divider: f32 = 2.60;
        pub const quiet_base: f32 = 0.80;
        pub const quiet_divider: f32 = 2.04;
    };

    pub const history_math = struct {
        /// depth multiplier.
        pub const scale: i16 = 146;
        /// Maximum of 1 bonus.
        pub const max_bonus: i16 = 1282;
        /// Maximum history score of one entry.
        pub const max_score: i16 = 16384;
    };

    pub const correction_history = struct {
        pub const scale: i32 = 256;
        pub const max_entry_bonus: i32 = 32;
        pub const max_applied_correction: i32 = 160;
        pub const is_big_correction_margin: i32 = 120;
    };

    pub const iterative_deepening = struct {
        /// The margin used to establish the average stability of a search score.
        pub const eval_stability_margin: i32 = 10;
        /// The depth divider to give stability some slack.
        pub const eval_stability_slack_depth_divider: u8 = 12;
    };

    pub const reversed_futility_pruning = struct {
        /// Maximum depth for applying.
        pub const max_depth: i32 = 6;
        /// Base margin for eval beating beta.
        pub const min_margin: i32 = 21;
        /// Delta for margin when eval is improving:
        pub const improving_margin: i32 = 40;
        /// Delta for margin wher eval is not improving:
        pub const not_improving_margin: i32 = 74;
    };

    pub const futility_pruning = struct {
        /// The maximum depth for applying.
        pub const max_depth: i32 = 8;
        /// Default margin.
        pub const margin_base: i32 = 196;
        /// Depth multiplier.
        pub const depth_mult: i32 = 96;
    };

    pub const history_pruning = struct {
        /// The maximum depth for applying.
        pub const max_depth: i32 = 5;
        /// Offset for quiet move.
        pub const quiet_offset: i32 = -518;
        /// Depth multiplier for quiet move.
        pub const quiet_mult: i32 = 1830;
    };

    pub const internal_iterative_deepening  = struct {
        /// Minimum depth for applying.
        pub const min_depth: i32 = 8;
    };

    pub const late_move_reduction = struct {
        /// The minimum depth for applying lmr.
        pub const min_depth: i32 = 3;
        /// Divider for calculating extra reduction (or negative reduction) based on a move's history score.
        pub const history_quiet_divider: i32 = 14035; // Currently 4 scores of max 16384
        pub const history_noisy_divider: i32 = 14035; // 1 score of max 16384
    };

    pub const quiescence_futility_pruning = struct {
        /// The margin added to the initial best score to allow pruning.
        pub const margin: i32 = 101;
    };

    pub const razoring = struct {
        /// The maximum depth for applying.
        pub const max_depth: i32 = 4;
        /// The depth multiplier when applying.
        pub const depth_mult: i32 = 550;
        /// Razoring:
        pub const base: i32 = 50;
        /// Razoring:
        pub const mult: i32 = 220;
        /// Razoring:
        pub const quad: i32 = 96;
    };

    /// The list sizes for moves that did not beat alpha.
    /// The idea is to not punish very late moves even more. It also saves stack space and processing.
    pub const search_historylist_size = struct {
        pub const for_quiets: u8 = 18;
        pub const for_noisies: u8 = 10;
    };

    pub const see_pruning = struct {
        /// Maximum depth for applying.
        pub const max_depth: i32 = 8;
        /// Depth multiplier for quiet moves.
        pub const quiet_mult: i32 = -68;
        /// Depth multiplier for noisy moves.
        pub const noisy_mult: i32 = -123;
    };

    pub const tt_entry = struct {
        /// Weight of TT entry depth.
        pub const depth_weight: i32 = 1024;
        /// Weight of TT Entry age (penalty).
        pub const age_penalty_weight: i32 = 1024;
    };
};
