const std = @import("std");
const lib = @import("lib.zig");
const data = @import("data.zig");
const funcs = @import("funcs.zig");
const console = @import("console.zig");
const position = @import("position.zig");
const notation = @import("notation.zig");

const Color = position.Color;
const Square = position.Square;
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

// WIP EXPERIMENTAL -> it sould be possible to not store the moves first, but directly make the moves in Perft.
pub const TestReceiver = struct
{

    pos: *Position,

    pub fn init(pos: *Position) TestReceiver
    {
        return TestReceiver
        {
            .pos = pos,
        };
    }

    /// Required funnction.
    pub fn reset(self: *TestReceiver) void
    {
        _ = self;
        //self.moves = 0;
    }

    /// Required funnction.
    pub fn store(self: *TestReceiver, move: Move) void
    {
        self.pos.lazy_make_move(move);
        std.debug.print("make move {s}\n", .{move.to_string().slice()});
        //if (self.pos.ply == 1) self.run()

        self.pos.lazy_unmake_move();
    }

    pub fn run(self: *TestReceiver, comptime output: bool, comptime is_root: bool, comptime us: Color, depth: u8) void
    {
        self.pos.generate_moves(us, self);
        _ = output;
        _ = is_root;
        _ = depth;
    }

};
