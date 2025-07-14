const std = @import("std");
const lib = @import("lib.zig");
const funcs = @import("funcs.zig");
const console = @import("console.zig");
const position = @import("position.zig");
const notation = @import("notation.zig");

const Color = position.Color;
const Move = position.Move;
const Position = position.Position;
const Storage = position.MoveStorage;
const JustCount = position.JustCount;

/// With output.
pub fn run(pos: *Position, depth: u8) void
{
    var t = funcs.start_timer();
    //const nodes: u64 = do_run(true, true, depth, pos);
    const nodes: u64 = switch (pos.to_move.e)
    {
        .white => do_run(true, true, Color.WHITE, depth, pos),
        .black => do_run(true, true, Color.BLACK, depth, pos),
    };
    const time = t.lap();
    console.print("perft {}: nodes: {}, time {}, nps {}\n", .{depth, nodes, std.fmt.fmtDuration(time), funcs.nps(nodes, time)});
}

/// No ouput. Just return node count.
pub fn run_quick(pos: *Position, depth: u8) u64
{
    switch (pos.to_move.e)
    {
        .white => return do_run(false, true, Color.WHITE, depth, pos),
        .black => return do_run(false, true, Color.BLACK, depth, pos),
    }
}

fn do_run(comptime output: bool, comptime is_root: bool, comptime us: Color, depth: u8, pos: *Position) u64
{
    const is_leaf: bool = depth == 2;
    const them: Color = comptime us.opp();
    var count: usize = 0;
    var nodes: usize = 0;

    var storage: position.MoveStorage = .init();
    pos.generate_moves(us, &storage);
    const moves = storage.slice();

    for (moves) |m|
    {
        if (is_root and depth <= 1)
        {
            count = 1;
            nodes += 1;
        }
        else
        {
            pos.make_move(us, m);

            if (is_leaf)
            {
                var counter: JustCount = .init();
                pos.generate_moves(them, &counter); // just count
                count = counter.moves;
                nodes += count;
            }
            else
            {
                count = do_run(output, false, them, depth - 1, pos); // go recursive
                nodes += count;
            }

            pos.unmake_move(us);
        }

        if (output and is_root)
        {
            console.print("{s}: {}\n", .{ m.to_string().slice(), count });
        }
    }
    return nodes;
}
