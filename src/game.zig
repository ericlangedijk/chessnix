// zig fmt: off

//! Position + History

const std = @import("zig");

const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");

const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;

const Value = types.Value;
const Color = types.Color;
const Move = types.Move;
const StateInfo = position.StateInfo;
const Position = position.Position;

/// A convenient position + history.
pub fn Game(comptime history_size: u16) type
{
    if (history_size == 0) @compileError("History size of Game cannot be zero");

    return struct {
        const Self = @This();

        history: [history_size]StateInfo,
        pos: Position,

        pub fn init_empty() Self {
            return
            .{
                .history = @splat(.empty),
                .pos = .empty,
            };
        }

        pub fn set(self: *Self, fen: []const u8) !void {
            try self.pos.set(&self.history[0], fen);
        }

        pub fn flip(self: *Self) void {
            self.pos.flip(&self.history[0]);
        }
    };
}