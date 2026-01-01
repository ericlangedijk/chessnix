//! Time management for search.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");
const uci = @import("uci.zig");

const Color = types.Color;
const Move = types.Move;

/// The amount of time for gui-engine communication and the few milliseconds to start threads after the go command.
const move_overhead: u32 = 10;

const max_search_depth = types.max_search_depth;

/// Criterium for ending the search.
pub const Termination = enum {
    /// No limit, except max_search_depth.
    infinite,
    /// Hard limit by nodecount.
    nodes,
    /// Hard limit by uci movetime.
    movetime,
    /// Hard limit by depth (iterations).
    depth,
    /// Limit by wtime / btime / winc / binc / movestogo.
    clock,
};

/// TODO: maybe use times in nanoseconds. then we do not have to calculate milliseconds.
pub const TimeMgr = struct {
    /// Indicates how we terminate the search.
    termination: Termination,
    /// Limiting the depth (iterations).
    max_depth: u8,
    /// Restrict search on number of nodes.
    max_nodes: u64,

    // Restrict search on milliseconds, depending on uci input (movetime or wtime / btime).
    max_movetime: u64,

    //hard_time_limit: u64,
    //soft_time_limit: u64,


    // Table to keep track of how many nodes were spent for each move at root search. Indexing: [move.from_to]
    //nodes_spent: [4096]u64,

    // Current best move
    // best_move: Move,
    // Move stability. Count how much times - after each iteration - this has been the best move. Maximum = 4.
    // best_move_stability: u3,

    timer: utils.Timer,

    pub const default: TimeMgr = .{
        .termination = .infinite,
        .max_depth = 0,
        .max_nodes = 0,
        .max_movetime = 0,

        //.hard_time_limit = 0,
        //.soft_time_limit = 0,


        // .nodes_spent = @splat(0),
        // .best_move = .empty,
        // .best_move_stability = 0,
        .timer = .empty,
    };

    /// Create time manager from the uci go params and starts the timer.
    pub fn init(go: *const uci.Go, us: Color) TimeMgr {
        var mgr: TimeMgr = .default;
        mgr.timer = .start();

        // Infinite.
        if (go.infinite) {
            return mgr;
        }

        // Depth.
        if (go.depth > 0) {
            mgr.termination = .depth;
            mgr.max_depth = @truncate(@min(go.depth, max_search_depth));
            return mgr;
        }

        // Nodes.
        if (go.nodes > 0) {
            mgr.termination = .nodes;
            mgr.max_nodes = go.nodes;
            return mgr;
        }

        // Movetime.
        if (go.movetime > 0) {
            mgr.termination = .movetime;
            //mgr.max_movetime = movetime;
            if (go.movetime > move_overhead) {
                mgr.max_movetime = go.movetime - move_overhead;
            }
            else {
                mgr.max_movetime = 1;
            }

            return mgr;
        }

        // Clock.
        if (go.time[0] > 0 or go.time[1] > 0) {
            mgr.termination = .clock;
            const time: u64 = go.time[us.u];
            const increment: u64 = go.increment[us.u];
            const movestogo: u64 = if (go.movestogo > 0) go.movestogo else 50;

            // original
            mgr.max_movetime = (time / movestogo) + if (movestogo > 1) increment else 0;
            //lib.io.debugprint("maxtime 1 {}\n", .{ mgr.max_movetime });

            // Add all increments to the time. #testing 1.3
            //const totaltime: u64 = time + movestogo * increment - increment;
            //mgr.max_movetime = (totaltime / movestogo) + if (movestogo > 1) increment else 0;

            //lib.io.debugprint("maxtime 2 {}\n", .{ mgr.max_movetime });

            if (mgr.max_movetime > move_overhead) {
                mgr.max_movetime -= move_overhead;
            }
            else {
                mgr.max_movetime = 1;
            }

            // const base_time: u64 = mul(time, base_time_scale) + mul(increment, increment_scale) - move_overhead;
            // const maximum_time: u64 = mul(time, percent_limit);
            // const scaled_hard_limit: u64 = @min(mul(base_time, hard_limit_scale), maximum_time);
            // const scaled_soft_limit: u64 = @min(mul(base_time, soft_limit_scale), maximum_time);
            // mgr.hard_time_limit = @max(5, scaled_hard_limit);
            // mgr.soft_time_limit = @max(1, scaled_soft_limit);
            // lib.io.debugprint("h {} s {}\n", .{ mgr.hard_time_limit, mgr.soft_time_limit });
            // Do not flag on the last move before time control.

            return mgr;
        }

        return mgr;


        // reserve = max(1500, remaining_time / 25)
        // usable   = remaining_time − reserve + increment * 70 / 100
        // maximum  = usable * 45 / 100
    }

    // pub fn nodes_spent_ptr(self: *TimeMgr, move: Move) *u64 {
    //     return &self.nodes_spent[move.from_to()];
    // }

    // /// TODO: use the ptr
    // pub fn update_nodes_spent(self: *TimeMgr, move: Move, spent: u64) void {
    //     self.nodes_spent[move.from_to()] += spent;
    // }

    // pub fn check_stop_early(self: *TimeMgr, best_move: Move, depth: i32, nodes: u64) bool {
    //     if (self.termination != .clock) {
    //         return false;
    //     }

    //     if (depth < 7) {
    //         return self.timer.elapsed_ms() >= self.soft_time_limit;
    //     }

    //     if (self.best_move != best_move) {
    //         self.best_move = best_move;
    //         self.best_move_stability = 0;
    //     }
    //     else {
    //         if (self.best_move_stability < 4) {
    //             self.best_move_stability += 1;
    //         }
    //     }

    //     const spent: f32 = @floatFromInt(self.nodes_spent_ptr(best_move).*);
    //     const total: f32 = @floatFromInt(@max(1, nodes));
    //     const percent_searched: f32 = spent / total;
    //     const percent_scale_factor: f32 = (node_fraction_base - percent_searched) * node_fraction_scale;
    //     const stability_scale: f32 = move_stability_scale[self.best_move_stability];
    //     const optimal_limit: u32 = @min(mul(self.soft_time_limit, percent_scale_factor * stability_scale), self.hard_time_limit);

    //     return self.timer.elapsed_ms() >= optimal_limit; // self.max_movetime;
    // }
};


// const base_time_scale: f32 = 0.055;
// const increment_scale: f32 = 0.91;
// const percent_limit: f32 = 0.77;
// const hard_limit_scale: f32 = 3.25;
// const soft_limit_scale: f32 = 0.83;
// const node_fraction_base: f32 = 1.49;
// const node_fraction_scale: f32 = 1.56;
// const move_stability_scale: [5]f32 = .{ 2.32, 1.22, 1.07, 0.79, 0.68 };

// fn mul(u: u64, f: f32) u32 {
//     return @intFromFloat(float(u) * f);
// }

// pub fn float(u: u64) f32 {
//     return @floatFromInt(u);
// }


