// zig fmt: off

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const search = @import("search.zig");

const ctx = lib.ctx;
const wtf = lib.wtf;

const Value = types.Value;
const Color = types.Color;
const Move = types.Move;
const StateInfo = position.StateInfo;
const Position = position.Position;
const SearchManager = search.SearchManager;
const Search = search.Search;

pub var history: [types.max_game_length]StateInfo = @splat(.empty);
pub var pos: Position = .empty; //create();
var searchmgr: SearchManager = undefined;

/// Initialize the engine.
/// * Position is set to the classic startposition.
pub fn initialize() !void
{
    try set_startpos();
    searchmgr = SearchManager.init(1);
}

/// Cleanup.
pub fn finalize() void
{
    searchmgr.deinit();
}

pub fn set_startpos() !void
{
   pos.set_startpos(&history[0]);
   //try set_position(position.fen_classic_startpos, null);
}

/// Sets the position from fen + moves.
/// * If fen is illegal we crash.
/// * If fen is null the startpostion will be set.
/// * Any illegal move in moves will stop the history without crashing.
pub fn set_position(fen: ?[]const u8, moves: ?[]const u8) !void
{
    if (fen) |str|
    {
        try pos.set(&history[0], str);
    }
    else
    {
        try pos.set(&history[0], position.fen_classic_startpos);
        return;
    }

    // If we have any moves, make them. We stop if we encounter an illegal move.
    if (moves) |str|
    {
        var tokenizer = std.mem.tokenizeScalar(u8, str, ' ');
        var idx: usize = 1;
        while (tokenizer.next()) |m|
        {
            const move: Move = pos.parse_move(m) catch break;
            pos.lazy_make_move(&history[idx], move);
            idx += 1;
        }
    }
}

pub fn start() !void
{
    try searchmgr.start();
}

pub fn stop() !void
{
    try searchmgr.stop();
}





