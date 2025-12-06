// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");

const wtf = lib.wtf;
const io = lib.io;

const Color = types.Color;
const Square = types.Square;
const Move = types.Move;
const Position = position.Position;
const Storage = position.MoveStorage;
const JustCount = position.JustCount;

/// With output.
pub fn run(pos: *const Position, depth: u8) void
{
    var t = utils.Timer.start();
    const nodes: u64 = switch (pos.stm.e) {
        .white => do_run(true, true, Color.WHITE, depth, pos),
        .black => do_run(true, true, Color.BLACK, depth, pos),
    };
    const time = t.lap();
    io.print("perft {}: nodes: {}, time {D}, nps {}\n", .{depth, nodes, time, funcs.nps(nodes, time)});
}

/// Only show output when ready.
pub fn qrun(pos: *const Position, depth: u8) void
{
    var t = utils.Timer.start();
    const nodes: u64 = switch (pos.stm.e) {
        .white => do_run(false, true, Color.WHITE, depth, pos),
        .black => do_run(false, true, Color.BLACK, depth, pos),
    };
    const time = t.lap();
    io.print("perft {}: nodes: {}, time {D}, nps {}\n", .{depth, nodes, time, funcs.nps(nodes, time)});
}

/// No output. Just return node count.
pub fn run_quick(pos: *const Position, depth: u8) u64 {
    switch (pos.stm.e) {
        .white => return do_run(false, true, Color.WHITE, depth, pos),
        .black => return do_run(false, true, Color.BLACK, depth, pos),
    }
}

fn do_run(comptime output: bool, comptime is_root: bool, comptime us: Color, depth: u8, pos: *const Position) u64 {
    const is_leaf: bool = depth == 2;
    const them: Color = comptime us.opp();
    var count: usize = 0;
    var nodes: usize = 0;

    var storage: position.MoveStorage = .init();
    pos.generate_all_moves(us, &storage);
    const moves = storage.slice();

    for (moves) |m| {
        if (is_root and depth <= 1) {
            count = 1;
            nodes += 1;
        }
        else {
            var next_pos: Position = pos.*;
            next_pos.do_move(us, m);
            if (is_leaf) {
                var counter: JustCount = .init();
                next_pos.generate_all_moves(them, &counter); // just count
                count = counter.moves;
                nodes += count;
            }
            else {
                count = do_run(output, false, them, depth - 1, &next_pos); // go recursive
                nodes += count;
            }
        }

        if (output and is_root) {
            m.print_buffered(pos.is_960);
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
            try pos.set(testrun.fen, false);
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
