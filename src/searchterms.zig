// zig fmt: off

//! search algorithm constants. Will be tunable later on.

const std = @import("std");
const types = @import("types.zig");

/// Later on we make this flexible for tuning.
pub const tuned = &default_tuned;

/// Default values.
pub const default_tuned: Tuned = .{};

pub const Tuned = struct {

    /// Correction History:
    corr_hist_scale: i32 = 256,
    /// Correction History:
    corr_hist_max_entry_bonus: i32 = 32,
    /// Correction History:
    corr_hist_max_applied_correction: i32 = 160,
    /// Correction History:
    corr_hist_is_big_correction_margin: i32 = 120,

    /// Iterative Deepening: The margin used to establish the average stability of a search score.
    eval_stability_margin: i32 = 10,
    /// Iterative Deepening: The depth divider to give stability some slack.
    eval_stability_slack_depth_divider: u8 = 12,

    /// Futility Pruning: the maximum depth for applying fp.
    fp_max_depth: i32 = 8,
    /// Futility Pruning:
    fp_margin_base: i32 = 196,
    /// Futility Pruning:
    fp_depth_mult: i32 = 96,

    /// History calculation: depth multiplier.
    history_scale: i16 = 146,
    /// History calculation: maximum of 1 bonus.
    history_max_bonus: i16 = 1282,
    /// History calculation: maximum history score.
    history_max_score: i16 = 16384,

    /// History Pruning: the maximum depth for applying.
    hp_max_depth: i32 = 5,
    /// History Pruning: offset for quiet move.
    hp_quiet_offset: i32 = -518,
    /// History Pruning: depth multiplier for quiet move.
    hp_quiet_mult: i32 = 1830,

    /// Internal Iterative Deepening: minimum depth for applying.
    iir_min_depth: i32 = 8,

    /// Late Move Reduction Table: value used for building the table.
    lmr_table_noisy_base: f32 = -0.24,
    /// Late Move Reduction Table: value used for building the table.
    lmr_table_noisy_divider: f32 = 2.60,
    /// Late Move Reduction Table: value used for building the table.
    lmr_table_quiet_base: f32 = 0.80,
    /// Late Move Reduction Table: value used for building the table.
    lmr_table_quiet_divider: f32 = 2.04,
    /// Late Move Reduction: the minimum depth for applying lmr.
    lmr_min_depth: i32 = 3,
    /// Late Move Reduction: divider for calculating extra reduction (or negative reduction) based on a move's history score.
    lmr_history_quiet_divider: i32 = 14035, // 4 scores of max 16384
    lmr_history_noisy_divider: i32 = 14035, // 1 score of max 16384

    /// Quiscence Search Futility Pruning: the margin added to the initial best score to allow fp.
    qs_fp_margin: i32 = 101,

    /// Razoring: the maximum depth for applying razoring.
    razor_max_depth: i32 = 4,
    /// Razoring: the multiplier when applying razoring.
    razor_depth_multiplier: i32 = 550,

    /// Razoring:
    razor_base: i32 = 50,
    /// Razoring:
    razor_mult: i32 = 220,
    /// Razoring:
    razor_quad: i32 = 96,

    /// Reversed Futility Pruning: maximum depth applying rfp.
    rfp_max_depth: i32 = 6,
    /// Reversed Futility Pruning: base margin for eval beating beta.
    rfp_min_margin: i32 = 21,
    /// Reversed Futility Pruning:
    rfp_improving_margin: i32 = 40,
    /// Reversed Futility Pruning:
    rfp_not_improving_margin: i32 = 74,
    /// Reversed Futility Pruning: (not used).
    rfp_complex_margin_delta: i32 = 10,
    /// Reversed Futility Pruning: (not used).
    rfp_cutnode_margin_delta: i32 = 10,

    /// Search: the list size of quiet moves that did not beat alpha.
    /// The idea is to not punish very late moves even more. It also saves a lot of stack space.
    search_quiet_list_size: u8 = 18,
    /// Search: the list size of capture moves that did not beat alpha.
    /// The idea is to not punish very late moves even more. It also saves a lot of stack space.
    search_noisy_list_size: u8 = 10, // #testing original 8

    /// SEE pruning: Maximum depth for applying.
    see_prune_max_depth: i32 = 8,
    /// SEE pruning: Depth multiplier for quiet moves.
    see_prune_quiet_mult: i32 = -68,
    /// SEE pruning: Depth multiplier for noisy moves.
    see_prune_noisy_mult: i32 = -123,

    /// TT Entry value: weight of depth.
    tt_entry_depth_weight: i32 = 1024,
    /// TT Entry value: weight of age penalty.
    tt_entry_age_penalty_weight: i32 = 1024,
};
