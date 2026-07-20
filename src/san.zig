// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const position = @import("position.zig");
const attacks = @import("attacks.zig");
const movegen = @import("movegen.zig");

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const Position = position.Position;

/// Contains only the info for san printing.
/// But as a present we get a state.
pub const SanMove = struct {
    piecetype: PieceType,
    from: Square,
    to: Square,
    is_capture: bool,
    prom: PieceType,
    disambiguation: Disambiguation,
    state: State,

    pub const State = enum {
        default, check, mate, stalemate,
    };

    pub const Disambiguation = enum {
        none, file, rank, both,
    };

    pub fn from_extmove(pos: *const Position, ex: ExtMove, known_moves_from_this_position: ?[]const ExtMove) SanMove {
        return internal_get(pos, ex, known_moves_from_this_position);
    }

    pub fn format(self: SanMove, writer: *std.io.Writer) !void {
        const is_pawn: bool = self.piecetype.e == .pawn;
        if (!is_pawn) {
            try writer.print("{u}", .{ self.piecetype.to_char() });
        }
        switch (self.disambiguation) {
            .none => {
                if (is_pawn and self.is_capture) {
                    try writer.print("{u}", .{ self.from.char_of_file() });
                }
            },
            .file => try writer.print("{u}", .{ self.from.char_of_file() }),
            .rank => try writer.print("{u}", .{ self.from.char_of_rank() }),
            .both => try writer.print("{t}", .{ self.from.e }),
        }
        if (self.is_capture) {
            try writer.print("x", .{});
        }
        try writer.print("{t}", .{ self.to.e });
        if (self.prom.e != .no_piecetype) {
            try writer.print("={u}", .{ self.prom.to_char() });
        }
        const checkchar: ?u8 = switch (self.state) {
            .default => null,
            .check => '+',
            .mate => '#',
            .stalemate => null,
        };
        if (checkchar) |c| {
            try writer.print("{u}", .{ c });
        }
    }
};

fn internal_get(pos: *const Position, ex: ExtMove, known_moves_from_this_position: ?[]const ExtMove) SanMove {
    var sanmove: SanMove = .{
        .piecetype = ex.piece.piecetype(),
        .from = ex.move.from,
        .to = ex.move.to,
        .is_capture = ex.move.is_capture(),
        .prom = ex.move.prom_safe(),
        .disambiguation = .none,
        .state = .default,
    };
    sanmove.disambiguation = disambiguate(pos, ex, known_moves_from_this_position);

    // Make move.
    var next_pos: Position = pos.*;
    next_pos.lazy_do_move(ex);

    // Determine state.
    sanmove.state = blk_state: {
        var any: movegen.Any = .init();
        movegen.lazy_generate_all_moves(&next_pos, &any);

        if (!next_pos.in_check()) {
            break :blk_state if (any.has_moves) .default else .stalemate;
        }
        else {
            break :blk_state if (any.has_moves) .check else .mate;
        }
    };

    return sanmove;
}

fn disambiguate(pos: *const Position, ex: ExtMove, known_moves_from_this_position: ?[]const ExtMove) SanMove.Disambiguation {
    const pt: PieceType = ex.piece.piecetype();

    // Some early exits.
    if (pt.e == .king) {
        return .none;
    }

    switch (ex.move.kind) {
        Move.silent => if (pt.e == .pawn) return .none,
        Move.double_push => return .none,
        Move.knight_promotion...Move.queen_promotion => return .none,
        Move.unreachable_10, Move.unreachable_11 => unreachable,
        else => {},
    }

    const from_bb: u64 = ex.move.from.to_bitboard();

    // Check same pieces on the board.
    const others: u64 = pos.by_type(pt) & pos.by_color(pos.stm) & ~from_bb;
    if (bitboards.popcnt(others) == 0) {
        return .none;
    }

    // Quick precheck if another piece can reach the to-square.
    const other_attacks: u64 = switch(pt.e) {
        .pawn, .king, .no_piecetype => 0,
        else => attacks.get_piece_attacks(ex.move.to, pt, .white, pos.all()) & ~from_bb,
    };
    if (other_attacks & pos.pieces(pt, pos.stm) == 0) {
        return .none;
    }

    return blk: {
        if (known_moves_from_this_position) |candidates| {
            break :blk scan_candidate_moves(ex, candidates);
        }
        else {
            var storage: movegen.MoveStorage = .init();
            movegen.lazy_generate_all_moves(pos, &storage);
            break :blk scan_candidate_moves(ex, storage.slice());
        }
    };
}

fn scan_candidate_moves(ex: ExtMove, candidates: []const ExtMove) SanMove.Disambiguation {
    var file_conflict: bool = false;
    var rank_conflict: bool = false;
    var found: bool = false;

    for (candidates) |candidate| {
        if (candidate.piece.u != ex.piece.u or candidate.move.from.u == ex.move.from.u or candidate.move.to.u != ex.move.to.u) continue;
        found = true;
        if (candidate.move.from.coord.file == ex.move.from.coord.file) file_conflict = true;
        if (candidate.move.from.coord.rank == ex.move.from.coord.rank) rank_conflict = true;
        if (file_conflict and rank_conflict) break;
    }

    if (!found) return .none;
    if (file_conflict and rank_conflict) return .both;
    if (!file_conflict) return .file;
    return .rank;
}
