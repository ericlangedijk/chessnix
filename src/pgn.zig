// zig fmt: off

///! Some clunky pgn output, just for debugging purposes.

const std = @import("std");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const position = @import("position.zig");
const movegen = @import("movegen.zig");

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const Position = position.Position;
const Game = position.Game;

pub const Pgn = struct {
    game: *const Game,
    writer: *std.io.Writer,

    pub fn init(game: *const Game, writer: *std.io.Writer) Pgn {
        return .{ .game = game, .writer = writer };
    }

    pub fn print(self: *Pgn) !void {
        try self.writer.print("[Event \"?\"]\n", .{});
        try self.writer.print("[Site \"?\"]\n", .{});
        try self.writer.print("[Date \"????.??.??\"]\n", .{});
        try self.writer.print("[Round \"?\"]\n", .{});
        try self.writer.print("[White \"?\"]\n", .{});
        try self.writer.print("[Black \"?\"]\n", .{});
        try self.writer.print("[Result \"*\"]\n", .{});
        // TODO: for non starting positions
        // try writer.print("[FEN "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"]
        try self.writer.print("\n", .{});

        // Sometimes we need a move storage for disambiguation and/or checkmate detection.
        var storage: movegen.MoveStorage = .init();
        var pos: Position = self.game.startpos;
        var move_nr: usize = 1;
        // TODO: find a less crazy way to keep track of the line length.
        var a: usize = 0;
        var b: usize = 0;
        var linelen: usize = 0;

        // TODO: start with the correct move nr and use ... when it is black first to move.
        for (self.game.moves.items) |ex| {
            // Move nr.
            if (pos.ply_from_root % 2 == 0) {
                a = self.writer.end;
                try self.writer.print("{}.", .{ move_nr });
                b = self.writer.end;
                linelen += if (b > a) b - a else b;
                try self.space_or_next_line(&linelen);
            }

            // Track move and write.
            const m: PgnMove = do_move(&pos, ex, &storage);
            a = self.writer.end;
            try self.write_move(m);
            b = self.writer.end;
            linelen += if (b > a) b - a else b;
            try self.space_or_next_line(&linelen);

            // Keep track of move number.
            if (pos.ply_from_root % 2 == 0) {
                move_nr += 1;
            }
        } // (moves)

        // Game result
        try self.writer.print("*\n\n", .{});
    }

    pub fn print_to(self: *Pgn, writer: *std.io.Writer) !void {
        const old_writer: *std.io.Writer = self.writer;
        self.writer = writer;
        defer self.writer = old_writer;
        try self.print();
    }

    fn space_or_next_line(self: *Pgn, linelen: *usize) !void {
        if (linelen.* >= 80) {
            try self.writer.print("\n", .{});
            linelen.* = 0;
        }
        else {
            try self.writer.print(" ", .{});
            linelen.* += 1;
        }
    }

    fn write_move(self: *Pgn, m: PgnMove) !void {
        const is_pawn: bool = m.piecetype.e == .pawn;
        if (!is_pawn) {
            try self.writer.print("{u}", .{ m.piecetype.to_char() });
        }
        switch (m.disambiguation) {
            .none => {
                if (is_pawn and m.is_capture) {
                    try self.writer.print("{u}", .{ m.from.char_of_file() });
                }
            },
            .file => try self.writer.print("{u}", .{ m.from.char_of_file() }),
            .rank => try self.writer.print("{u}", .{ m.from.char_of_rank() }),
            .both => try self.writer.print("{t}", .{ m.from.e }),
        }
        if (m.is_capture) {
            try self.writer.print("x", .{});
        }
        try self.writer.print("{t}", .{ m.to.e });
        const checkchar: ?u8 = switch (m.check) {
            .none => null,
            .check => '+',
            .mate => '#'
        };
        if (checkchar) |c| {
            try self.writer.print("{u}", .{ c });
        }
    }

    fn do_move(pos: *Position, ex: ExtMove, storage: *movegen.MoveStorage) PgnMove {
        var moves_generated: bool = false;
        var m: PgnMove = undefined;
        m.piecetype = ex.piece.piecetype();
        m.from = ex.move.from;
        m.to = ex.move.to;
        m.is_capture = ex.move.is_capture();
        m.disambiguation = disambiguate(pos, ex, storage, &moves_generated);
        pos.lazy_do_move(ex);
        m.check = blk: {
            if (!pos.in_check()) {
                break :blk .none;
            }
            if (!moves_generated) {
                movegen.lazy_generate_all_moves(pos, storage);
            }
            break :blk if (storage.count > 0) .check else .mate;
        };
        return m;
    }

    fn disambiguate(pos: *const Position, ex: ExtMove, storage: *movegen.MoveStorage, moves_generated: *bool) Disambiguation {
        storage.reset();
        moves_generated.* = false;

        const us: Color = pos.stm;
        const pt: PieceType = ex.piece.piecetype();

        if (pt.e == .king) {
            return .none;
        }

        // Some early exits.
        switch (ex.move.kind) {
            Move.silent => if (pt.e == .pawn) return .none,
            Move.double_push => return .none,
            Move.knight_promotion...Move.queen_promotion => return .none,
            Move.unreachable_10, Move.unreachable_11 => unreachable,
            else => {},
        }

        // Check same pieces on the board.
        const others: u64 = pos.by_type(pt) & pos.by_color(us) & ~ex.move.from.to_bitboard();
        if (bitboards.popcnt(others) == 0) return .none;

        var file_conflict: bool = false;
        var rank_conflict: bool = false;

        movegen.lazy_generate_all_moves(pos, storage);
        moves_generated.* = true;

        const from_file: u3 = ex.move.from.coord.file;
        const from_rank: u3 = ex.move.from.coord.rank;

        for (storage.slice()) |candidate| {
            if (candidate.move == ex.move or candidate.piece.u != ex.piece.u or candidate.move.to.u != ex.move.to.u) continue;
            if (candidate.move.from.coord.file == from_file) file_conflict = true;
            if (candidate.move.from.coord.rank == from_rank) rank_conflict = true;
            if (file_conflict and rank_conflict) break;
        }

        if (!file_conflict and !rank_conflict) return .none;
        if (file_conflict and rank_conflict) return .both;
        if (file_conflict) return .rank;
        return .file;
    }

    const PgnMove = struct {
        piecetype: PieceType,
        from: Square,
        to: Square,
        is_capture: bool,
        disambiguation: Disambiguation,
        check: Check,
    };

    const Check = enum {
        none,
        check,
        mate,
    };

    const Disambiguation = enum {
        none,
        file,
        rank,
        both,
    };

};
