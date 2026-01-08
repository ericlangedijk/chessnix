// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const uci = @import("uci.zig");

const builtin = @import("builtin");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");

pub fn main() !void
{
    @setFloatMode(.optimized);
    try lib.initialize();
    defer lib.finalize();

    // Debug tests.
    if (comptime lib.is_debug) {
        // try @import("tests.zig").run_silent_debugmode_tests();
    }

    // var tm = TimeMgr.init(20000, 10, 200);
    // tm.print();

    // tm.update(1, -10, 10);
    // tm.print();

    // tm.update(2, 10, 12);
    // tm.print();

    // tm.update(3, 12, 8);
    // tm.print();

    // tm.update(4, 8, -100);
    // tm.print();

    // tm.update(0, -100, -10);
    // tm.print();

    // tm.update(1, -10, -1);
    // tm.print();

    // tm.update(2, -1, -1);
    // tm.print();

    // tm.update(3, -1, -3);
    // tm.print();

    // tm.update(1, -1, -500);
    // tm.print();

    uci.run();
}

const TimeMgr = struct {
    max: u64,
    avg: u64,
    opt: u64,
    speed: Speed,
    best_move_counter: u3,

    fn init(wtime: u64, winc: u64, movestogo: u64) TimeMgr {
        var mgr: TimeMgr = undefined;
        const total_increment: u64 = if (movestogo > 1) winc * (movestogo - 1) else 0;
        const total_time: u64 = wtime + total_increment;
        std.debug.print("total time {} total increment {}\n", .{ total_time, total_increment});
        mgr.speed = get_speed(total_time);
        mgr.avg = @divTrunc(total_time, movestogo + 1);
        mgr.max = mgr.avg * speed_factor[@intFromEnum(mgr.speed)];
        mgr.opt = mgr.avg;
        mgr.best_move_counter = 0;
        return mgr;
    }

    fn get_speed(totaltime: u64) Speed {
        // 40 / 3m minutes / 2s = 3 * 60 * 1000 + 2 * 2000 = 180.000 + 4000 = 184.000
        if (totaltime < 184000) {
            return .fast;
        }
        // 40 / 15m / 30s = 15 * 60 * 1000 + 30 * 1000 = 900.000 + 30.000 = 930.000
        else if (totaltime < 930000) {
            return .medium;
        }
        else {
            return .slow;
        }
    }

    // fn minutes(ms: u64) u64 {
    //     return ms / 60000;
    // }

    fn update(self: *TimeMgr, best_move_counter: u3, prev_eval: i32, eval: i32) void {
        self.best_move_counter = best_move_counter;
        const scale: f32 = stability_scale[self.best_move_counter] * get_improvement_scale(prev_eval, eval);
        std.debug.print("new scale scale {}\n", .{ scale });
        self.opt = @min(self.max, mul(self.avg, scale));
        //self.print();
    }

    /// Returns
    /// - minimum 0.5 when big improvement
    /// - maximum 2.0 when big failure
    fn get_improvement_scale(prev_eval: i32, eval: i32) f32 {

        // TODO: only when depth > 7
        if (eval >= 300) {
            return 0.5; // max confidence
        }
        else if (eval >= 100) {
            return 0.75; // high confidence
        }
        else if (eval <= -100) {
            return 1.75; // high
        }
        else if (eval <= -300) {
            return 2.0; // max
        }

        const diff: i32 = prev_eval - eval;
        const X: f32 = 100.0;
        const T: f32 = 2.0;
        const S: f32 = funcs.float(diff);
        const result = std.math.pow(f32, T, std.math.clamp(S, -X, X) / X);
        std.debug.print("score diff {} scale {}\n", .{ -diff, result});
        return result;
    }

    fn print(self: *const TimeMgr) void {
        std.debug.print("speed {} max {} avg {} opt {}\n", .{self.speed, self.max, self.avg, self.opt });
    }

};

const stability_scale: [8]f32 = .{
    2.50,
    2.00,
    1.75,
    1.50,
    1.00,
    0.75,
    0.75,
    0.75,
};

const speed_factor: [3]u8 = .{
    3,
    5,
    8,
};

fn mul(i: u64, f: f32) u64{
    //return @intFromFloat(float: anytype)
    const fi: f32 = funcs.float(i);
    return @intFromFloat(fi * f);
}


const Speed = enum {
    fast,
    medium,
    slow,
};

