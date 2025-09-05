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

/// ### Debug settings.
pub const using = packed struct {
    /// Transposition Table.
    pub const tt: bool = true;
    /// Principal Variation Search.
    pub const pvs: bool = true;
    /// Late Move Reduction.
    pub const lmr: bool = false;
    /// Interesting moves extensions.
    pub const ext: bool = true;
};

pub var history: [types.max_game_length]StateInfo = @splat(.empty);
pub var pos: Position = .empty;
pub var options: Options = .default;
pub var searchmgr: SearchManager = undefined;

/// Initialize the engine.
/// * Position is set to the classic startposition.
pub fn initialize() !void {
    set_startpos();
    searchmgr = try .init();
}

/// Cleanup.
pub fn finalize() void {
    searchmgr.deinit();
}

/// Sets the startposition.
pub fn set_startpos() void {
    pos.set_startpos(&history[0]);
}

/// After an illegal move we stop without crashing.
pub fn set_startpos_with_optional_moves(moves: ?[]const u8) !void {
    set_startpos();
    if (moves) |str| {
        parse_moves(str);
    }
}

/// Sets the position from fen + moves.
/// * If fen is illegal we crash.
/// * If fen is null the startpostiion will be set.
/// * After an illegal move we stop without crashing.
pub fn set_position(fen: ?[]const u8, moves: ?[]const u8) !void {
    const f = fen orelse {
        set_startpos();
        return;
    };

    try pos.set(&history[0], f);

    if (moves) |m| {
        parse_moves(m);
    }
}

// If we have any moves, make them. We stop if we encounter an illegal move.
fn parse_moves(moves: []const u8) void {
    var tokenizer = std.mem.tokenizeScalar(u8, moves, ' ');
    var idx: usize = 1;
    while (tokenizer.next()) |m| {
        const move: Move = pos.parse_move(m) catch break;
        pos.lazy_make_move(&history[idx], move);
        idx += 1;
    }
}

pub fn start(go: *const uci.Go) !void {
    try searchmgr.start(&pos, go);
}

pub fn stop() !void {
    try searchmgr.stop();
}

pub fn clear_for_new_game() void
{
    // TODO: set startpos
    searchmgr.clear_for_new_game();
}

/// The available options.
pub const Options = struct {
    /// In megabytes.
    hash_size: u64 = default_hash_size,

    pub const default: Options = .{};

    pub const default_hash_size: u64 = 64;
    pub const min_hash_size: u64 = 1;
    pub const max_hash_size: u64 = 1024;

    pub const default_use_hash: bool = true;

    pub fn set_hash_size(self: *Options, value: u64) void {
        self.hash_size = std.math.clamp(value, min_hash_size, max_hash_size);
    }
};

