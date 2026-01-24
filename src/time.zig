// zig fmt: off

//! Time management for search.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");
const funcs = @import("funcs.zig");
const utils = @import("utils.zig");
const uci = @import("uci.zig");

const Value = types.Value;
const Color = types.Color;
const Move = types.Move;
const Position = position.Position;

const float = funcs.float;

const max_search_depth = types.max_search_depth;

/// Criterium for ending the search.
pub const Termination = enum {
    /// No limit, except the maximum search depth.
    infinite,
    /// Limited by depth.
    depth,
    /// Hard limit by nodes.
    nodes,
    /// Hard limit by movetime.
    movetime,
    /// Limit by wtime / btime / winc / binc / movestogo.
    clock,
};

// TODO: use endtime -> then we do not have to calculate elapsed_ms each time.
pub const TimeManager = struct {
    termination: Termination,
    timer: utils.Timer,
    max_movetime: u64,
    opt_movetime_base: u64,
    opt_movetime: u64,
    max_nodes: u64,
    max_depth: u8,

    pub const empty: TimeManager = .{
        .termination = .infinite,
        .timer = .empty,
        .max_movetime = 0,
        .opt_movetime_base = 0,
        .opt_movetime = 0,
        .max_nodes = 0,
        .max_depth = 0,
    };

    pub fn set(self: *TimeManager, go: *const uci.Go, us: Color) void {
        self.* = .empty;
        self.timer = .start();

        if (go.infinite) {
            self.termination = .infinite;
            return;
        }

        if (go.movetime > 0) {
            self.termination = .movetime;
            self.max_movetime = go.movetime; // TODO: move overhead
            return;
        }

        if (go.depth > 0) {
            self.termination = .depth;
            // Cap the max depth.
            self.max_depth = @min(max_search_depth, @as(u8, @truncate(go.depth)));
            return;
        }

        if (go.nodes > 0) {
            self.termination = .nodes;
            self.max_nodes = go.nodes;
            return;
        }

        // From here we are in a tournament situation and need smart timing.
        self.termination = .clock;

        const time: u64 = go.time[us.u];
        const inc: u64 = go.increment[us.u];

        const half_inc: u64 = inc / 2;
        // If we have a move increment, don't add everything to the time. Keep reserve.
        // TODO: we still have to figure out something for very small increment or time for the whole game.
        //const half_inc: u64 = (inc * 200) / 300; // #testing

        const move_overhead: u64 = @min(25, time / 2);
        const cyclic_timecontrol: bool = go.movestogo > 0;
        //const movestogo: u64 = if (cyclic_timecontrol) @min(go.movestogo, 50) else 50; // still #experimental (original()
        const movestogo: u64 = if (cyclic_timecontrol) @min(go.movestogo, 40) else 40; // still #experimental

        var timeleft = @max(1, time + half_inc * (movestogo - 1));
        const minus: u64 = move_overhead * (2 + movestogo);
        if (minus < timeleft)
            timeleft -= minus
         else
            timeleft = 1; // TODO: debug this. This would be quite a panic

        var optscale: f32 = 0;
        const mtg: f32 = float(movestogo);
        const t: f32 = float(time);
        const tl: f32 = float(timeleft);
        const mo: f32 = float(move_overhead);

        if (cyclic_timecontrol) {
            optscale = @min(0.90 / mtg, 0.88 * t / tl);
        } else {
            optscale = @min(optscale_fixed, optscale_time_left * t / tl);
        }

        const optime: f32 = optscale * tl;
        self.opt_movetime_base = @intFromFloat(optime);
        self.opt_movetime = self.opt_movetime_base;

        // TODO: stabilize?
        const max_factor: f32 = 0.75;
        const maxtime: f32 = max_factor * t - mo;
        self.max_movetime = @intFromFloat(maxtime);
    }

    /// When in clockmode we should call this function after each search iteration.
    pub fn update_optimal_stoptime(self: *TimeManager, nodes_spent_on_move: u64, total_nodes: u64, best_move_stability: u3, eval_stability: u3) void {
        const best_move_nodes_fraction: f32 = float(nodes_spent_on_move) / float(total_nodes);
        const node_scaling_factor: f32 = (node_tm_base - best_move_nodes_fraction) * node_tm_multiplier;
        const best_move_scaling_factor: f32 = bestmove_stability_scales[best_move_stability];
        const eval_scaling_factor: f32 = eval_stability_scales[eval_stability];
        const base: f32 = @floatFromInt(self.opt_movetime_base);
        const opt: f32 = base * node_scaling_factor * best_move_scaling_factor * eval_scaling_factor;
        self.opt_movetime = @intFromFloat(opt);
        self.opt_movetime = @min(self.max_movetime, self.opt_movetime);
    }

    pub fn max_time_reached(self: *TimeManager) bool {
        return self.timer.elapsed_ms() >= self.max_movetime;
    }

    pub fn optimal_time_reached(self: *TimeManager) bool {
        return self.timer.elapsed_ms() >= self.opt_movetime;
    }
};

const node_tm_base: f32 = 1.53;
const node_tm_multiplier: f32 = 1.74;

const optscale_fixed = 0.025;
const optscale_time_left = 0.20;

const bestmove_stability_scales: [5]f32 = .{
    2.38,
    1.29,
    1.07,
    0.91,
    0.71,
};

const eval_stability_scales: [5]f32 = .{
    1.25,
    1.15,
    1.05,
    0.92,
    0.87,
};
