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

const Color = types.Color;
const Move = types.Move;
const Position = position.Position;

const float32 = funcs.float32;
const float64 = funcs.float64;

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

pub const TimeManager = struct {
    termination: Termination,
    timer: utils.Timer,
    /// The starttime in nanoseconds.
    started: u64,
    /// The max endtime in nanoseconds.
    max_endtime: u64,
    /// The optimal endtime in nanoseconds. Recomputed during search.
    opt_endtime: u64,
    /// Optimal base time in milliseconds. Computed once in set.
    opt_movetime_base: u64,
    /// For the uci "go nodes"
    max_nodes: u64,
    /// For the uci "go depth"
    max_depth: u8,
    // For output only.
    mtg: u64,
    // For output only.
    max_ms: u64,

    pub const empty: TimeManager = .{
        .termination = .infinite,
        .timer = .empty,
        .started = 0,
        .max_endtime = 0,
        .opt_endtime = 0,
        .opt_movetime_base = 0,
        .max_nodes = 0,
        .max_depth = 0,
        .mtg = 0,
        .max_ms = 0,
    };

    /// Assumes the go argument is sanitized after parsing (no negative numbers), so we can safely cast ints.
    pub fn set(self: *TimeManager, go: *const uci.Go, pos: *const Position, one_mover: bool) void {
        // TODO: maybe make smarter. we are (like other engines often do as well) often living off the increment in a long game.
        // can we 'gather' some time using the increment?

        self.* = .empty;
        self.timer = .start();
        self.started = self.timer.read();
        const us: Color = pos.stm;

        if (go.infinite != null) {
            self.termination = .infinite;
            return;
        }

        if (go.movetime) |m| {
            self.termination = .movetime;
            const max: u64 = @intCast(m);
            self.max_ms = max;
            self.max_endtime = self.started + max * 1_000_000;
            return;
        }

        if (go.depth) |d| {
            self.termination = .depth;
            // Cap the max depth here.
            const max: i64 = std.math.clamp(d, 1, max_search_depth);
            self.max_depth = @intCast(max);
            return;
        }

        if (go.nodes) |n| {
            self.termination = .nodes;
            self.max_nodes = @intCast(n);
            return;
        }

        if (go.time[us.u] == null) {
            self.termination = .infinite;
            return;
        }

        // From here we are in a clock situation and need smart timing.
        self.termination = .clock;

        const time: u64 = if (go.time[us.u]) |t| @intCast(t) else 1;
        const inc: u64 = if (go.increment[us.u]) |i| @intCast(i) else 0;
        var movestogo: u64 = if (go.movestogo) |m| @intCast(m) else 0;
        const cyclic_timecontrol: bool = movestogo > 0;
        movestogo = if (cyclic_timecontrol) @min(movestogo, 50) else 50;
        self.mtg = movestogo;
        const increment_per_move: u64 = inc;
        const move_overhead: u64 = 20;
        var timeleft = @max(1, time + increment_per_move * (movestogo - 1));
        const total_move_overhead: u64 = move_overhead * (movestogo + 1);
        if (total_move_overhead + 50 < timeleft) {
            timeleft -= total_move_overhead;
        }
        else {
            timeleft = 50; // Could flag in insane cases.
        }

        const f_movestogo: f64 = float64(movestogo);
        const f_time: f64 = float64(time);
        const f_timeleft: f64 = float64(timeleft);
        const f_move_overhead: f64 = float64(move_overhead);

        const optscale: f64 = switch (cyclic_timecontrol) {
            false => @min(optscale_fixed, optscale_time_left * f_time / f_timeleft),
            true  => @min(0.90 / f_movestogo, 0.88 * f_time / f_timeleft),
        };

        // Calculate time caps.
        const optime: f64 = optscale * f_timeleft;
        self.opt_movetime_base = @intFromFloat(optime);
        const max_factor: f64 = 0.60;
        var maxtime: f64 = max_factor * f_time - f_move_overhead;
        // If we only have one move, spend max 500 ms.
        if (one_mover) {
            maxtime = @min(500, maxtime);
        }
        const max_movetime: u64 = @intFromFloat(maxtime);
        self.max_ms = max_movetime;
        self.max_endtime = self.started + max_movetime * 1_000_000;
        self.opt_endtime = self.started + self.opt_movetime_base * 1_000_000;
    }

    /// When in clockmode we call this function after each search iteration.
    pub fn update_optimal_stoptime(self: *TimeManager, nodes_spent_on_move: u64, total_nodes: u64, best_move_stability: u3, eval_stability: u3) void {
        const best_move_nodes_fraction: f64 = float64(nodes_spent_on_move) / float64(total_nodes);
        const node_scaling_factor: f64 = (node_tm_base - best_move_nodes_fraction) * node_tm_multiplier;
        const best_move_scaling_factor: f64 = bestmove_stability_scales[best_move_stability];
        const eval_scaling_factor: f64 = eval_stability_scales[eval_stability];
        const base: f64 = @floatFromInt(self.opt_movetime_base);
        const opt: f64 = base * node_scaling_factor * best_move_scaling_factor * eval_scaling_factor;
        const opt_move_time: u64 = @intFromFloat(opt);
        self.opt_endtime = self.started + opt_move_time * 1_000_000;
    }

    pub fn read(self: *TimeManager) u64 {
        return self.timer.read();
    }

    pub fn max_time_reached(self: *TimeManager, curr_time: u64) bool {
        return curr_time >= self.max_endtime;
    }

    pub fn optimal_time_reached(self: *TimeManager, curr_time: u64) bool {
        return curr_time >= self.opt_endtime;
    }

    // Print some time info at the start of the search.
    pub fn print_info(self: *const TimeManager, us: Color) void {
        const def = comptime "info string";
        const t: Termination = self.termination;
        switch (t) {
            .infinite  => lib.io.print("{s} us {t} termination {t}\n", .{ def, us.e, t }),
            .depth     => lib.io.print("{s} us {t} termination {t} {}\n", .{ def, us.e, t, self.max_depth }),
            .nodes     => lib.io.print("{s} us {t} termination {t} {}\n", .{ def, us.e, t, self.max_nodes }),
            .movetime  => lib.io.print("{s} us {t} termination {t} {}\n", .{ def, us.e, t, self.max_ms }),
            .clock     => lib.io.print("{s} us {t} termination {t} mtg {} max {} opt {}\n", .{ def, us.e, t, self.mtg, self.max_ms, self.opt_movetime_base }),
        }
    }
};

const node_tm_base: f64 = 1.53;
const node_tm_multiplier: f64 = 1.74;
const optscale_fixed = 0.025;
const optscale_time_left = 0.20;

const bestmove_stability_scales: [5]f64 = .{
    2.38,
    1.29,
    1.07,
    0.91,
    0.71,
};

const eval_stability_scales: [5]f64 = .{
    1.25,
    1.15,
    1.05,
    0.92,
    0.87,
};
