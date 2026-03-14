// zig fmt: off

//! search algorithm constants. Will be tunable later on.

const std = @import("std");
const types = @import("types.zig");

/// Later on we make this flexible for tuning.
pub const tuned = &default_tuned;

/// Default values.
pub const default_tuned: Tuned = .{};

pub const Tuned = struct {
    /// The margin used in Iterative Deepening to establish the average stability of a search score.
    eval_stability_margin: i32 = 10,
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
    /// History calculation: maximum of 1 malus.
    history_max_malus: i16 = 1282,
    /// History calculation: maximum history score.
    history_max_score: i16 = 16384,
    /// History Pruning: offset for quiet move.
    hp_quiet_offset: i32 = -518,
    /// History Pruning: depth multiplier for quiet move.
    hp_quiet_mult: i32 = 1830,
    /// Internal Iterative Deepening: minimum depth for applying iir.
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
    /// Late Move Reduction: divider for calculating extra reduction (or negative reduction) based on a (quiet) move's history score.
    lmr_history_divider: i32 = 14035,
    /// Quiscence Search Futility Pruning: the margin added to the initial best score to allow fp.
    qs_fp_margin: i32 = 101,
    /// Razoring: the maximum depth for applying razoring.
    razor_max_depth: i32 = 4,
    /// Razoring: the multiplier when applying razoring.
    razor_depth_multiplier: i32 = 550,
    /// Reversed Futility Pruning: maximum depth applying rfp.
    rfp_max_depth: i32 = 6,
    /// Reversed Futility Pruning: base margin for eval beating beta.
    rfp_easy_margin: i32 = 21,
    /// Reversed Futility Pruning:
    rfp_improving_margin: i32 = 40,
    /// Reversed Futility Pruning:
    rfp_not_improving_margin: i32 = 74,
    /// Reversed Futility Pruning: (not used).
    rfp_complex_margin_delta: i32 = 10,
    /// Reversed Futility Pruning: (not used).
    rfp_cutnode_margin_delta: i32 = 10,
    /// Search: the list size of quiet moves that did not beat alpha.
    search_quiet_list_size: u8 = 16,
    /// Search: the list size of capture moves that did not beat alpha.
    search_capture_list_size: u8 = 8,
    /// TT Entry value: weight of depth. (785)
    tt_entry_depth_weight: i32 = 1024,
    /// TT Entry value: weight of age penalty. (4622)
    tt_entry_age_penalty_weight: i32 = 1024,
    /// TT Entry value: weight of alpha bound. (not used).
    tt_entry_alpha_weight: i32 = 359,
    /// TT Entry value: weight of beta bound. (not used).
    tt_entry_beta_weight: i32 = 218,
    /// TT Entry value: weight of exact bound. (not used).
    tt_entry_exact_weight: i32 = 77,
    /// TT Entry value: weight if there is a move. (not used).
    tt_entry_move_weight: i32 = 2,
    /// TT Entry value: weight if entry is pvs. (not used).
    tt_entry_pvs_weight: i32 = 386,
};
