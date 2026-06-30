// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const movegen = @import("movegen.zig");

const wtf = lib.wtf;
const io = lib.io;

const Color = types.Color;
const Square = types.Square;
const Move = types.Move;
const Position = position.Position;

/// With output.
pub fn run(pos: *const Position, depth: u8) void
{
    var t = utils.Timer.start();
    const nodes: u64 = switch (pos.stm.e) {
        .white => do_run(true, true, .white, depth, pos),
        .black => do_run(true, true, .black, depth, pos),
    };
    const time = t.lap();
    io.print("perft {}: nodes: {}, time {D}, nps {}\n", .{depth, nodes, time, funcs.nps(nodes, time)});
}

/// Only show output when ready.
pub fn qrun(pos: *const Position, depth: u8) void
{
    var t = utils.Timer.start();
    const nodes: u64 = switch (pos.stm.e) {
        .white => do_run(false, true, .white, depth, pos),
        .black => do_run(false, true, .black, depth, pos),
    };
    const time = t.lap();
    io.print("perft {}: nodes: {}, time {D}, nps {}\n", .{depth, nodes, time, funcs.nps(nodes, time)});
}

/// No output. Just return node count.
pub fn run_quick(pos: *const Position, depth: u8) u64 {
    switch (pos.stm.e) {
        .white => return do_run(false, true, .white, depth, pos),
        .black => return do_run(false, true, .black, depth, pos),
    }
}

fn do_run(comptime output: bool, comptime is_root: bool, comptime us: Color, depth: u8, pos: *const Position) u64 {
    const is_leaf: bool = depth == 2;
    const them: Color = comptime us.opp();
    var count: usize = 0;
    var nodes: usize = 0;

    var storage: movegen.MoveStorage = .init();
    movegen.generate_all_moves(pos, us, &storage);
    const moves = storage.slice();

    for (moves) |ex| {
        if (is_root and depth <= 1) {
            count = 1;
            nodes += 1;
        }
        else {
            var next_pos: Position = pos.*;
            next_pos.do_move(us, ex);
            if (is_leaf) {
                var just_count: movegen.JustCount = .init();
                movegen.generate_all_moves(&next_pos, them, &just_count); // just count
                count = just_count.counted;
                nodes += count;
            }
            else {
                count = do_run(output, false, them, depth - 1, &next_pos); // go recursive
                nodes += count;
            }
        }

        if (output and is_root) {
            ex.move.print_buffered(pos.is_960);
            io.print_buffered(": {}\n", .{ count });
            io.flush();
        }
    }
    return nodes;
}


/// Doing 4 positions, measuring speed.
pub fn bench() !void {
    const Test = struct {
        name: []const u8,
        fen: []const u8,
        end_depth: u8,
        end_depth_nodes: u64,
    };

    const testruns: [4]Test =
    .{
        .{.name = "Startpos", .fen = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",                 .end_depth = 7, .end_depth_nodes = 3195901860 },
        .{.name = "Kiwipete", .fen = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",     .end_depth = 6, .end_depth_nodes = 8031647685 },
        .{.name = "Midgame",  .fen = "r4rk1/1pp1qppp/p1np1n2/2b1p1B1/2B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", .end_depth = 6, .end_depth_nodes = 6923051137 },
        .{.name = "Endgame",  .fen = "5nk1/pp3pp1/2p4p/q7/2PPB2P/P5P1/1P5K/3Q4 w - - 1 28",                      .end_depth = 6, .end_depth_nodes = 849167880 },
    };

    var totalnodes: u64 = 0;
    var totaltime: u64 = 0;

    var pos: Position = .empty;

    main_loop: inline for (&testruns) |*testrun| {
        for (1..testrun.end_depth + 1) |depth| {
            try pos.setup(testrun.fen, false);
            var timer = utils.Timer.start();
            const nodes: u64 = run_quick(&pos, @truncate(depth));
            const time = timer.read();
            io.print("Perft {s} {}: {d:<12} {D:<12}  {d:>12.4} Mnodes/s ({})\n", .{ testrun.name, depth, nodes, time, funcs.mnps(nodes, time), funcs.nps(nodes, time) });

            if (depth == testrun.end_depth) {
                totalnodes += nodes;
                totaltime += time;
                if (nodes == testrun.end_depth_nodes) {
                    io.print("OK\n\n", .{});
                }
                else {
                    io.print("ERROR\n\n", .{});
                    break :main_loop;
                }
            }
        }
    }

    io.print("Total nodes: {} {D} {d:.4} Mnodes/s ({})\n", .{ totalnodes, totaltime, funcs.mnps(totalnodes, totaltime), funcs.nps(totalnodes, totaltime) });
}

// pub fn bench_find_move() !void {

//     // var totalnodes: u64 = 0;
//     var fast_time: u64 = 0;
//     var slow_time: u64 = 0;
//     var cnt: u64 = 0;

//     var pos: Position = .empty;

//     inline for (&testruns) |*testrun| {

//         try pos.setup(testrun.fen, false);
//         var movestorage = position.MoveStorage.init();
//         pos.lazy_generate_all_moves(&movestorage);
//         for (0..10000) |_| {
//             for (movestorage.slice()) |ex|{
//                 const str: []const u8 = ex.move.to_string(false).slice();
//                 var slow_timer = utils.Timer.start();
//                 const slow_find_move: types.ExtMove = try pos.parse_move(str);
//                 slow_time += slow_timer.read();
//                 var fast_timer = utils.Timer.start();
//                 const fast_find_move: types.ExtMove = try pos.lazy_find_move(ex.move.from, ex.move.to, ex.move.prom_safe());
//                 fast_time += fast_timer.read();
//                 if (slow_find_move != fast_find_move) {
//                     io.print("ERROR\n", .{});
//                     return;
//                 }
//                 cnt += 1;
//             }
//         }

//         //     var timer = utils.Timer.start();
//         //     const nodes: u64 = run_quick(&pos, @truncate(depth));
//         //     const time = timer.read();
//         //     io.print("Perft {s} {}: {d:<12} {D:<12}  {d:>12.4} Mnodes/s ({})\n", .{ testrun.name, depth, nodes, time, funcs.mnps(nodes, time), funcs.nps(nodes, time) });

//         //     if (depth == testrun.end_depth) {
//         //         totalnodes += nodes;
//         //         totaltime += time;
//         //         if (nodes == testrun.end_depth_nodes) {
//         //             io.print("OK\n\n", .{});
//         //         }
//         //         else {
//         //             io.print("ERROR\n\n", .{});
//         //             break :main_loop;
//         //         }
//         //     }
//         // }
//     }

//     io.print("count {}, Slow time: {D} Fast time {D}\n", .{ cnt, slow_time, fast_time });
// }
