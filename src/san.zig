// zig fmt: off

//! SAN notation output.

const std = @import("std");

const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const data = @import("data.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");

const Color = types.Color;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Square = types.Square;
const Move = types.Move;
const MoveType = types.MoveType;
const Position = position.Position;
const StateInfo = position.StateInfo;

const eql = funcs.eql;
const in = funcs.in;

const Disambiguation = enum {
    None,
    /// Use the file (a..h)
    File,
    /// Use the rank (1..8)
    Rank,
    // Use the full square name.
    Both
};

/// TODO: asserts
pub fn write_san_line(input_pos: *const Position, moves: []const Move, writer: *std.io.Writer) !void {
    if (moves.len == 0 ) return;

    var states: [128]StateInfo = undefined;
    var pos: Position = .empty;
    pos.copy_from(input_pos, &states[0]);
    //std.debug.assert(pos.equals(input_pos, true));

    // if (!pos.equals(input_pos, false)) {
    //     //std.debug.hex
    //     try pos.draw();
    //     try input_pos.draw();
    // }

    //if (true) return;

    var movenr = funcs.ply_to_movenumber(pos.game_ply, pos.to_move);

    // Write move nr.
    try writer.print("{}.", .{ movenr });
    if (pos.to_move.e == .black) try writer.print("..", .{});
    try writer.print(" ", .{});

    const last: usize = moves.len - 1;
    var idx: u8 = 1;
    for (moves, 0..) |move, i| {
        const str = get_san(move, &pos);
        try writer.print("{s} ", .{ str.slice() });
        pos.lazy_do_move(&states[idx], move);
        idx += 1;
        if (i < last and pos.to_move.e == .white) {
             movenr += 1;
             try writer.print("{}. ", .{ movenr });
        }
    }
}

/// Get the correct SAN notation.
/// * The move must not yet be on the board.
pub fn get_san(m: Move, pos: *Position) lib.BoundedArray(u8, 10) {
    var string: lib.BoundedArray(u8, 10) = .empty;

    if (m.type == .castle) {
        string.print_assume_capacity("{s}", .{ types.castle_strings[m.info.castletype.u]});
    } else {
        const moved_piece = pos.board[m.from.u];
        const is_capture: bool = (m.type == .enpassant or pos.board[m.to.u].is_piece());

        // Piece character.
        if (!moved_piece.is_pawn()) {
            string.print_assume_capacity("{u}", .{ moved_piece.piecetype.to_char() });
        }

        const dis: Disambiguation = determine_disambiguation(m, pos);

        // Disambiguation.
        switch (dis) {
            .None => {},
            .File => string.print_assume_capacity("{u}", .{ m.from.char_of_file() }),
            .Rank => string.print_assume_capacity("{u}", .{ m.from.char_of_rank() }),
            .Both => string.print_assume_capacity("{t}", .{ m.from.e } ),
        }

        // Capture?
        if (is_capture) {
            string.print_assume_capacity("x", .{});
        }

        // To square.
        string.print_assume_capacity("{t}", .{ m.to.e });

        // Promotion.
        if (m.type == .promotion) {
            const ch = m.info.prom.to_san_char();
            string.print_assume_capacity("={u}", .{ch} );
        }
    }

    // Now determine check or mate.
    var st: StateInfo = undefined;
    pos.lazy_do_move(&st, m);
    defer pos.lazy_undo_move();

    if (st.checkers != 0) {
        var any: position.Any = .init();
        pos.lazy_generate_moves(&any);
        const ch: u8 = if (!any.has_moves) '#' else '+';
        string.print_assume_capacity("{u}", .{ ch });
    }
    return string;
}

fn determine_disambiguation(m: Move, pos: *const Position) Disambiguation {
    const pc: Piece = pos.get(m.from);

    // Pawn.
    if (pc.is_pawn()) {
        // Pawn push.
        if (m.from.file() == m.to.file()) return .None;
        // This is a pawn capture. We return a fake disambiguation: the file char must be printed.
        return .File;
    }

    const us = pos.to_move;
    const from_bb: u64 = m.from.to_bitboard();
    const pt: PieceType = pc.piecetype;

    // No others -> no disambiguation.
    const others = pos.pieces(pt, us) & ~from_bb;
    if (others == 0) return .None;

    // Quick check if there is a same piece which can do the move on an empty board.
    const precheck: u64 = get_pseudo_moves_to(pt, m.to, others);
    if (precheck == 0) return .None;

    // Now we have to generate.
    var finder: DisambiguationFinder = .init(pos, m);
    pos.lazy_generate_moves(&finder);
    return finder.select_disambiguation();
}

fn get_pseudo_moves_to(pt: PieceType, to: Square, our_pieces: u64) u64 {
    return switch(pt.e) {
        .knight => data.get_knight_attacks(to) & our_pieces,
        .bishop => data.get_bishop_attacks(to, 0) & our_pieces,
        .rook   => data.get_rook_attacks(to, 0) & our_pieces,
        .queen  => data.get_queen_attacks(to, 0) & our_pieces,
        else => bitboards.bb_full,
    };
}

const DisambiguationFinder = struct {
    pos: *const Position,
    move: Move,
    moved_piece: Piece,
    any: bool,
    rank_conflict: bool,
    file_conflict: bool,

    fn init(pos: *const Position, move: Move) DisambiguationFinder {
        return .{
            .pos = pos,
            .move = move,
            .moved_piece = pos.board[move.from.u],
            .any = false,
            .rank_conflict = false,
            .file_conflict = false,
        };
    }

    /// Required function.
    pub fn reset(self: *DisambiguationFinder) void {
        self.rank_conflict = false;
        self.file_conflict = false;
    }

    /// Required function.
    pub fn store(self: *DisambiguationFinder, move: Move) ?void {
        if (self.is_candidate(move)) {
            self.any = true;
            self.file_conflict = (self.move.from.file() == move.from.file());
            self.rank_conflict = (self.move.from.rank() == move.from.rank());
            // Stop producing moves if we have the maximum conflict.
            if (self.rank_conflict and self.file_conflict) return null;
        }
    }

    fn is_candidate(self: *DisambiguationFinder, move: Move) bool {
        return
            self.move.from.e != move.from.e and
            self.move.to.e == move.to.e and
            self.moved_piece.e == self.pos.board[move.from.u].e;
    }

    fn select_disambiguation(self: *const DisambiguationFinder) Disambiguation {
        if (!self.any) return .None;
        if (self.file_conflict and self.rank_conflict) return .Both;
        if (self.file_conflict) return .Rank;
        return .File;
    }
};

test "san"
{
    try lib.initialize();
    var st: position.StateInfo = undefined;
    var pos: Position = .empty;
    var move: types.Move = .empty;
    var san: lib.BoundedArray(u8, 10) = .empty;

    // https://lichess.org/editor/R1n1k3/8/2R1K3/8/8/8/8/8_w_-_-_0_1
    try pos.set(&st, "R1n1k3/8/2R1K3/8/8/8/8/8 w - - 0 1");
    move = .create(.C6, .C8);
    san = get_san(move, &pos);
    try std.testing.expectEqualSlices(u8, "Rcxc8#", san.slice());

    // https://lichess.org/editor/q7/R1n1k3/8/2R5/K7/8/8/8_w_-_-_0_1
    try pos.set(&st, "q7/R1n1k3/8/2R5/K7/8/8/8 w - - 0 1");
    try pos.draw();
    move = .create(.C5, .C7);
    san = get_san(move, &pos);
    try std.testing.expectEqualSlices(u8, "Rxc7+", san.slice());
}

