// zig fmt: off

//! search algorithm constants. Will be tunable later on.

const types = @import("types.zig");

const Score = types.Score;
const SmallScore = types.SmallScore;

pub const tunables = &default_tunables;

const default_tunables: Tunables = .{
    .history_scale = 146,
    .history_max_bonus = 1282,
    .history_max_malus = 1282,
    .history_max_score = 16384,
    .eval_stability_margin = 10,
    .search_quiet_list_size = 16,
    .search_capture_list_size = 8,
    .rfp_max_depth = 6,
    .razor_max_depth = 4,
    .razor_depth_multiplier = 550,
    .lmr_table_noisy_base = -0.24,
    .lmr_table_noisy_divider = 2.60,
    .lmr_table_quiet_base = 0.80,
    .lmr_table_quiet_divider = 2.04,
    .lmr_min_depth = 3,
    .lmr_history_divider = 14035,
    .quiescence_futility_margin = 101,
    .tt_entry_depth_weight = 1024,
    .tt_entry_age_weight = 1024,
};

const Tunables = struct {
    history_scale: SmallScore,
    history_max_bonus: SmallScore,
    history_max_malus: SmallScore,
    history_max_score: SmallScore,
    eval_stability_margin: Score,
    search_quiet_list_size: u8,
    search_capture_list_size: u8,
    rfp_max_depth: i32,
    razor_max_depth: i32,
    razor_depth_multiplier: i32,
    lmr_table_noisy_base: f32,
    lmr_table_noisy_divider: f32,
    lmr_table_quiet_base: f32,
    lmr_table_quiet_divider: f32,
    lmr_min_depth: i32,
    lmr_history_divider: i32,
    quiescence_futility_margin: i32,
    tt_entry_depth_weight: i32,
    tt_entry_age_weight: i32,
};


