// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const console = @import("console.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const perft = @import("perft.zig");
const tests = @import("tests.zig");

const Position = position.Position;
const MoveStorage = position.MoveStorage;

const ctx = lib.ctx;

pub fn main() !void
{
    lib.initialize();
    defer
    {
        defer lib.finalize();
        if (lib.is_release) read_key();
    }

    // Debug tests.
    if (!lib.is_release)
    {
        try tests.run_silent_debugmode_test();
        try tests.run_testfile(false, 1);
    }

    // Speed tests. In debug mode this will be very very slow.
    if (lib.is_release)
    {
        try run();
    }
}

fn run() !void
{
    const Test = struct
    {
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

    inline for (&testruns) |*testrun|
    {
        var pos = try Position.from_fen(testrun.fen);
        defer pos.deinit();
        for (1..testrun.end_depth + 1) |depth|
        {
            var timer = funcs.start_timer();
            const nodes: u64 = perft.run_quick(&pos, @truncate(depth));
            const time = timer.read();
            console.print("Perft {s} {}: {:<12} {d:<12}  {d:>12.4} Mnodes/s ({})\n", .{ testrun.name, depth, nodes, std.fmt.fmtDuration(time), funcs.mnps(nodes, time), funcs.nps(nodes, time) });
            if (depth == testrun.end_depth)
            {
                totalnodes += nodes;
                totaltime += time;
                if (nodes == testrun.end_depth_nodes) console.print_str("OK\n\n") else console.print_str("ERROR\n\n");
            }
        }
    }

    console.print("Total nodes: {} {} {d:.4} Mnodes/s ({})\n", .{ totalnodes, std.fmt.fmtDuration(totaltime), funcs.mnps(totalnodes, totaltime), funcs.nps(totalnodes, totaltime) });
    console.print("press enter to exit\n", .{});
}

fn read_key() void
{
    const stdin = std.io.getStdIn().reader();
    _ = stdin.readByte() catch {};
}




