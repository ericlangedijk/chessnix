// zig fmt: off

///! Some clunky pgn output, just for local usage.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const movegen = @import("movegen.zig");
const san = @import("san.zig");

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const Position = position.Position;


/// Far from ready...
pub const Pgn = struct {
    startpos: *const Position,
    moves: []const ExtMove,

    pub fn init(startpos: *const Position, moves: []const ExtMove) Pgn {
        return .{
            .startpos = startpos,
            .moves = moves,
        };
    }

    pub fn format(self: *const Pgn, writer: *std.io.Writer) !void {
        try writer.print("[Event \"?\"]\n", .{});
        try writer.print("[Site \"?\"]\n", .{});
        try writer.print("[Date \"????.??.??\"]\n", .{});
        try writer.print("[Round \"?\"]\n", .{});
        try writer.print("[White \"?\"]\n", .{});
        try writer.print("[Black \"?\"]\n", .{});
        try writer.print("[Result \"*\"]\n", .{});
        // #experimental TODO: we need to compare.
        try writer.print("[FEN \"{f}\"]\n", .{ self.startpos });
        try writer.print("\n", .{});

        const last_idx: usize = if (self.moves.len > 0) self.moves.len - 1 else 0;
        var pos: Position = self.startpos.*;
        var a: usize = 0;
        var b: usize = 0;
        var linelen: usize = 0;
        for (self.moves, 0..) |ex, idx| {

            // Move nr.
            if ((idx == 0 and pos.stm.e == .black) or (pos.stm.e == .white)) {
                const nr: u16 = funcs.ply_to_movenumber(pos.game_ply, pos.stm);
                a = writer.end;
                if (idx == 0 and pos.stm.e == .black) {
                    try writer.print("{}...", .{ nr });
                }
                else if (pos.stm.e == .white) {
                    try writer.print("{}.", .{ nr });
                }
                b = writer.end;
                linelen += if (b >= a) b - a else b;
                try space_or_next_line(&linelen, writer);
            }

            const sanmove: san.SanMove = .from_extmove(&pos, ex, null);
            pos.lazy_do_move(ex);

            a = writer.end;
            try writer.print("{f}", .{ sanmove });
            b = writer.end;
            linelen += if (b >= a) b - a else b;

            if (idx < last_idx) {
                try space_or_next_line(&linelen, writer);
            }
            else {
                try writer.print(" ", .{});
            }
        } // (moves)

        // Result
        try writer.print("*\n\n", .{});
    }

    fn space_or_next_line(linelen: *usize, writer: *std.io.Writer) !void {
        if (linelen.* >= 80) {
            try writer.print("\n", .{});
            linelen.* = 0;
        }
        else {
            try writer.print(" ", .{});
            linelen.* += 1;
        }
    }
};
