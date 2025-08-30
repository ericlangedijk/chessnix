// zig fmt: off

// !The central engine in here provides the functions for the UCI loop.

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const uci = @import("uci.zig");
const search = @import("search.zig");


const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;

const Value = types.Value;
const Color = types.Color;
const Move = types.Move;
const StateInfo = position.StateInfo;
const Position = position.Position;
const SearchManager = search.SearchManager;
const Search = search.Search;

pub var history: [types.max_game_length]StateInfo = @splat(.empty);
pub var pos: Position = .empty;
var searchmgr: SearchManager = undefined;

/// Initialize the engine.
/// * Position is set to the classic startposition.
pub fn initialize() !void
{
    try set_startpos(null);
    searchmgr = .init();
}

/// Cleanup.
pub fn finalize() void
{
    searchmgr.deinit();
}

/// Sets the classical startposition + optional moves.
/// * After an illegal move we stop without crashing.
pub fn set_startpos(moves: ?[]const u8) !void
{
    pos.set_startpos(&history[0]);

    if (moves) |str|
    {
        parse_moves(str);
    }
}

/// Sets the position from fen + moves.
/// * If fen is illegal we crash.
/// * If fen is null the startpostiion will be set.
/// * After an illegal move we stop without crashing.
pub fn set_position(fen: ?[]const u8, moves: ?[]const u8) !void
{
    if (fen) |str|
    {
        try pos.set(&history[0], str);
        try pos.validate();
    }
    else
    {
        pos.set_startpos(&history[0]);
        return;
    }

    if (moves) |str|
    {
        parse_moves(str);
    }
}

// If we have any moves, make them. We stop if we encounter an illegal move.
fn parse_moves(moves: []const u8) void
{
    var tokenizer = std.mem.tokenizeScalar(u8, moves, ' ');
    var idx: usize = 1;
    while (tokenizer.next()) |m|
    {
        const move: Move = pos.parse_move(m) catch break;
        pos.lazy_make_move(&history[idx], move);
        idx += 1;
    }
}

pub fn start(go: *const uci.Go) !void
{
    try searchmgr.start(&pos, go);
}

pub fn stop() !void
{
    try searchmgr.stop();
}
