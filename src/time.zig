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

// go wtime 300000 btime 300000 winc 3000 binc 3000 movestogo 40
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
        //self.starttime = self.timer.read() / std.time.ns_per_ms;

        if (go.infinite) {
            self.termination = .infinite;
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

        const time: u64 = go.time[us.u];
        const inc: u64 = go.increment[us.u];

        const move_overhead: u64 = @min(25, time / 2);

        if (go.movetime > 0) {
            self.termination = .movetime;
            self.max_movetime = time - move_overhead; // TODO: this can maybe underflow?
            self.max_movetime = time - move_overhead;
            return;
        }

        // From here we are in a tournament situation and need smart timing.
        self.termination = .clock;

        const cyclic_timecontrol: bool = go.movestogo > 0;
        const movestogo: u64 = if (cyclic_timecontrol) @min(go.movestogo, 50) else 50;

        const timeleft = @max(1, time + inc * (movestogo - 1) - move_overhead * (2 + movestogo));

        var optscale: f32 = 0;
        const m: f32 = float(movestogo);
        const t: f32 = float(time);
        const tl: f32 = float(timeleft);
        const mo: f32 = float(move_overhead);

        if (cyclic_timecontrol) {
            optscale = @min(0.90 / m, 0.88 * t / tl);
        }
        else {
            optscale = @min(optscale_fixed, optscale_time_left * t / tl);
        }

        const optime: f32 =  optscale * tl;
        self.opt_movetime_base = @intFromFloat(optime);
        self.opt_movetime = self.opt_movetime_base;

        // Absolute limit for a move is 75% of the total time.
        const maxtime: f32 = 0.75 * t - mo;
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

/// The amount of time for gui-engine communication and the few milliseconds to start threads after the go command.
//const move_overhead: u32 = 10;

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




