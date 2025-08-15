// zig fmt: off

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");

const Value = types.Value;
const Color = types.Color;
const Move = types.Move;
const StateInfo = position.StateInfo;
const Position = position.Position;

const ctx = lib.ctx;
const wtf = lib.wtf;

pub var history: [1024]StateInfo = @splat(.empty);
pub var pos: Position = .create();

pub fn initialize() !void
{
    try set_position(position.fen_classic_startpos, null);
}

/// Sets the position from fen + moves.
/// * Example fen_str: `position fen rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1`
/// * Example moves_str:  `moves e2e4 e7e5 g1f3`
pub fn set_position(fen_str: ?[]const u8, moves_str: ?[]const u8) !void
{
    var tt = funcs.start_timer();

    if (fen_str) |fen|
    {
        try pos.set(&history[0], fen);
    }
    else
    {
        try pos.set(&history[0], position.fen_classic_startpos);
        return;
    }

    // If we have any moves, make them. We stop if we encounter an illegal move.
    if (moves_str) |moves|
    {
        var tokenizer = std.mem.tokenizeScalar(u8, moves, ' ');
        var idx: usize = 0;
        while (tokenizer.next()) |m|
        {
            const move: Move = pos.parse_move(m) catch break;
            pos.lazy_make_move(&history[idx], move);
            idx += 1;
        }
    }

    const t = tt.read();
    std.debug.print("time {}\n", .{std.fmt.fmtDuration(t)});
}

