// zig fmt: off

//! Not to smart but working disambiguation.

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

pub const NotationType = enum
{
    Short,
    Long,
};

pub fn get(comptime notationtype: NotationType, m: Move, pos: *const Position) std.BoundedArray(u8, 16)
{
    // We always need a copy.
    var cloned_pos: Position = pos.clone(false);
    defer cloned_pos.deinit();

    // todo: comptime Color here.
    switch (notationtype)
    {
        .Short => return get_short_algebraic(m, &cloned_pos),
        .Long => return get_long_algebraic(m, &cloned_pos),
    }
}

fn get_suffix(pos: *const Position) ?u8
{
    if (pos.in_check())
    {
        var store: position.Any = .init();
        pos.lazy_generate_moves(&store);
        const is_mate = !store.has_moves;
        return if (is_mate) '#' else '+';
    }
    return null;
}

fn get_long_algebraic(m: Move, pos: *Position) std.BoundedArray(u8, 16)
{
    var s: std.BoundedArray(u8, 16) = .{};

    pos.lazy_make_move(m);
    const st = pos.current_state;
    const pc = st.moved_piece;
    const capt = st.captured_piece;

    if (m.movetype == .castle)
    {
        if (m.castle_type() == .short) s.appendSliceAssumeCapacity("O-O") else s.appendSliceAssumeCapacity("O-O-O");
    }
    else
    {
        if (!pc.is_pawn()) s.appendAssumeCapacity(pc.piecetype.to_char());
        s.appendSliceAssumeCapacity(m.from.to_string());
        if (capt.is_piece()) s.appendAssumeCapacity('x') else s.appendAssumeCapacity('-');
        s.appendSliceAssumeCapacity(m.to.to_string());
        if (m.movetype == .promotion)
        {
            s.appendAssumeCapacity('=');
            s.appendAssumeCapacity(m.prom.to_piecetype().to_char());
        }
    }
    if (get_suffix(pos)) |suffix| s.appendAssumeCapacity(suffix);
    return s;
}

fn get_short_algebraic(m: Move, pos: *Position) std.BoundedArray(u8, 16)
{
    var s: std.BoundedArray(u8, 16) = .{};

    const dis: Disambiguation = determine_disambiguation(m, pos);

    pos.lazy_make_move(m);

    const st = pos.current_state;
    const pc = st.moved_piece;
    const capt = st.captured_piece;

    if (m.movetype == .castle)
    {
        if (m.castle_type() == .short) s.appendSliceAssumeCapacity("O-O") else s.appendSliceAssumeCapacity("O-O-O");
    }
    else
    {
        if (!pc.is_pawn()) s.appendAssumeCapacity(pc.piecetype.to_char());

        if (pc.is_pawn() and dis == .None and capt.is_piece())
        {
            //std.debug.print("wtf, ", .{});
            s.appendAssumeCapacity(m.from.char_of_file());
        }

        switch (dis)
        {
            .None => {},
            .File => s.appendAssumeCapacity(m.from.char_of_file()),
            .Rank => s.appendAssumeCapacity(m.from.char_of_rank()),
            .Both => s.appendSliceAssumeCapacity(m.from.to_string()),
        }

        if (capt.is_piece()) s.appendAssumeCapacity('x');

        s.appendSliceAssumeCapacity(m.to.to_string());

        if (m.movetype == .promotion)
        {
            s.appendAssumeCapacity('=');
            s.appendAssumeCapacity(m.prom.to_piecetype().to_char());
        }
    }
    if (get_suffix(pos)) |suffix| s.appendAssumeCapacity(suffix);
    return s;
}

fn determine_disambiguation(m: Move, pos: *Position) Disambiguation
{
    const us = pos.to_move;
    const from_sq: Square = m.from;
    const from_bb: u64 = from_sq.to_bitboard();
    const to_sq: Square = m.to;
    const pc: Piece = pos.get(from_sq);
    const pt: PieceType = pc.piecetype;

    // Some early exits.
    const others = pos.pieces(pt, us) & ~from_bb;
    // No others
    if (@popCount(others) == 0) return .None;
    // Pawn push.
    if (pt.e == .pawn and from_sq.file() == to_sq.file()) return .None;

    // Now generate filtered moves.
    var store: Store = .init(pos, from_sq, to_sq, pt);
    pos.lazy_generate_moves( &store);

    var file_conflict: bool = false;
    var rank_conflict: bool = false;

    for (store.slice()) |candidate_move|
    {
        if (candidate_move.from.file() == from_sq.file()) file_conflict = true;
        if (candidate_move.from.rank() == from_sq.rank()) rank_conflict = true;
        if (file_conflict and rank_conflict) break;
    }

    if (!file_conflict and !rank_conflict) return .None;
    if (file_conflict and rank_conflict) return .Both;
    if (file_conflict) return .Rank;
    return .File;
}

const Store = struct
{
    pos: *const Position,
    pt: PieceType,
    from_sq_excluded: Square,
    target_sq: Square,

    moves: [types.max_move_count]Move,
    ptr: [*]Move,

    pub fn init(pos: *const Position, from_sq_excluded: Square, target_sq: Square, pt: PieceType) Store
    {
        var result: Store = undefined;
        result.pos = pos;
        result.ptr = &result.moves;
        result.pt = pt;
        result.from_sq_excluded = from_sq_excluded;
        result.target_sq = target_sq;
        return result;
    }

    /// Required funnction.
    pub fn reset(self: *Store) void
    {
        self.ptr = &self.moves;
    }

    /// Required funnction.
    pub fn store(self: *Store, move: Move) void
    {
        //const pt =
        if (self.from_sq_excluded.e != move.from.e and self.target_sq.e == move.to.e and self.pos.get(move.from).piecetype.e == self.pt.e)
        {
            self.ptr[0] = move;
            self.ptr += 1;
        }
    }

    /// Required function.
    pub fn len(self: *const Store) usize
    {
        return self.ptr - &self.moves;
    }

    pub fn slice(self: *const Store) []const Move
    {
        return self.moves[0..self.len()];
    }
};



const Notation = struct
{
    notationtype: NotationType,
    disambiguation: Disambiguation,
    from_sq: Square,
    to_sq: Square,
    movetype: MoveType,
    promotion: Piece,
    moved_piece: Piece,
    captured_piece: Piece,
    check: bool,
    checkmate: bool,

    fn get_suffix(self: Notation) ?u8
    {
        _ = self;
        return 0;
    }
};

