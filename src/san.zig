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

const Disambiguation = enum
{
    None,
    /// Use the file 'a'..'h'.
    File,
    /// Use the rank '1'..'8'.
    Rank,
    // Use the full square name.
    Both
};

/// TODO: asserts
pub fn write_san_line(input_pos: *const Position, moves: []const Move, writer: *std.io.Writer) !void
{
    if (moves.len == 0 ) return;

    var states: [128]StateInfo = undefined;
    var pos: Position = .clone(input_pos, &states[0]);
    var movenr = funcs.ply_to_movenumber(pos.game_ply, pos.to_move);

    // Write move nr.
    try writer.print("{}.", .{ movenr });
    if (pos.to_move.e == .black) try writer.print("..", .{});
    try writer.print(" ", .{});

    const last: usize = moves.len - 1;
    var idx: u8 = 1;
    for (moves, 0..) |move, i|
    {
        const str = get_san(move, &pos);
        try writer.print("{s} ", .{ str.slice() });
        pos.lazy_make_move(&states[idx], move);
        idx += 1;
        if (i < last and pos.to_move.e == .white)
        {
             movenr += 1;
             try writer.print("{}. ", .{ movenr });
        }
    }
}

/// Get the correct SAN notation.
/// * The move must not yet be on the board.
pub fn get_san(m: Move, pos: *Position) lib.BoundedArray(u8, 10)
{
    var string: lib.BoundedArray(u8, 10) = .empty;

    if (m.type == .castle)
    {
        if (m.info.castletype.e == .short)
            string.print_assume_capacity("O-O", .{})
        else
            string.print_assume_capacity("O-O-O", .{});
    }
    else
    {
        const moved_piece = pos.board[m.from.u];
        const is_capture: bool = (m.type == .enpassant or pos.board[m.to.u].is_piece());

        // Piece character.
        if (!moved_piece.is_pawn())
        {
            string.print_assume_capacity("{u}", .{ moved_piece.piecetype.to_char() });
        }

        const dis: Disambiguation = determine_disambiguation(m, pos);

        // Disambiguation.
        switch (dis)
        {
            .None => {},
            .File => string.print_assume_capacity("{u}", .{ m.from.char_of_file() }),
            .Rank => string.print_assume_capacity("{u}", .{ m.from.char_of_rank() }),
            .Both => string.print_assume_capacity("{t}", .{ m.from.e } ),
        }

        if (is_capture)
        {
            string.print_assume_capacity("x", .{});
        }

        string.print_assume_capacity("{t}", .{ m.to.e });

        if (m.type == .promotion)
        {
            const ch = m.info.prom.to_san_char();
            string.print_assume_capacity("={u}", .{ch} );
        }
    }

    // Now determine check or mate.
    var st: StateInfo = undefined;
    pos.lazy_make_move(&st, m);
    defer pos.lazy_unmake_move();

    if (st.checkers != 0)
    {
        var any: position.Any = .init();
        pos.lazy_generate_moves(&any);
        const ch: u8 = if (!any.has_moves) '#' else '+';
        string.print_assume_capacity("{u}", .{ ch });
    }

    return string;
}

fn determine_disambiguation(m: Move, pos: *const Position) Disambiguation
{
    const pc: Piece = pos.get(m.from);

    // Pawn.
    if (pc.is_pawn())
    {
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

pub fn get_pseudo_moves_to(pt: PieceType, to: Square, our_pieces: u64) u64
{
    return switch(pt.e)
    {
        .knight => data.get_knight_attacks(to) & our_pieces,
        .bishop => data.get_bishop_attacks(to, 0) & our_pieces,
        .rook => data.get_rook_attacks(to, 0) & our_pieces,
        .queen => data.get_queen_attacks(to, 0) & our_pieces,
        else => bitboards.bb_full,
    };
}

const DisambiguationFinder = struct
{
    pos: *const Position,
    move: Move,
    moved_piece: Piece,
    any: bool,
    rank_conflict: bool,
    file_conflict: bool,

    pub fn init(pos: *const Position, move: Move) DisambiguationFinder
    {
        return
        .{
            .pos = pos,
            .move = move,
            .moved_piece = pos.board[move.from.u],
            .any = false,
            .rank_conflict = false,
            .file_conflict = false,
        };
    }

    /// Required function.
    pub fn reset(self: *DisambiguationFinder) void
    {
        self.rank_conflict = false;
        self.file_conflict = false;
    }

    /// Required function.
    pub fn store(self: *DisambiguationFinder, move: Move) ?void
    {
        if (self.move.from.e != move.from.e and self.move.to.e == move.to.e and self.moved_piece.e == self.pos.board[move.from.u].e)
        {
            self.any = true;
            self.file_conflict = (self.move.from.file() == move.from.file());
            self.rank_conflict = (self.move.from.rank() == move.from.rank());
            // Stop producing moves if we have the maximum conflict.
            if (self.rank_conflict and self.file_conflict) return null;
        }
    }

    fn select_disambiguation(self: *const DisambiguationFinder) Disambiguation
    {
        if (!self.any) return .None;
        if (self.file_conflict and self.rank_conflict) return .Both;
        if (self.file_conflict) return .Rank;
        return .File;
    }
};

