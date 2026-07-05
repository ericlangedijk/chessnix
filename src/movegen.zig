// zig fmt: off

//! Move generation.

const std = @import("std");
const types = @import("types.zig");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const attacks = @import("attacks.zig");
const position = @import("position.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const io = lib.io;
const popcnt = funcs.popcnt;
const pawns_shift = funcs.pawns_shift;
const pawn_from = funcs.pawn_from;

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const Castle = types.Castle;
const Position = position.Position;
const Castling = position.Castling;

const gf_check = position.gf_check;
const gf_pins  = position.gf_pins;
const gf_black = position.gf_black;
const gf_noisy = position.gf_noisy;

pub fn lazy_generate_all_moves(pos: *const Position, noalias storage: anytype) void {
    switch (pos.stm.e) {
        .white => generate_all_moves(pos, .white, storage),
        .black => generate_all_moves(pos, .black, storage),
    }
}

pub fn lazy_generate_quiescence_moves(pos: *const Position, noalias storage: anytype) void {
    switch (pos.stm.e) {
        .white => generate_quiescence_moves(pos, .white, storage),
        .black => generate_quiescence_moves(pos, .black, storage),
    }
}

/// Generate all legal moves.
pub fn generate_all_moves(pos: *const Position, comptime us: Color, noalias storage: anytype) void {
    const color_flag: u4 = comptime if (us.e == .black) gf_black else 0;
    switch(pos.gen_flags) {
        inline else => |gf| gen(pos, gf | color_flag, storage),
    }
}

/// Generate quiescence moves. When not in check: generate captures and queen only promotions. When in check generate all legal moves and queen only promotions.
pub fn generate_quiescence_moves(pos: *const Position, comptime us: Color, noalias storage: anytype) void {
    const color_flag: u4 = comptime if (us.e == .black) gf_black else 0;
    switch(pos.gen_flags) {
        inline else => |gf| gen(pos, gf | color_flag | gf_noisy, storage),
    }
}

/// See `MoveStorage` for the interface of `storage`. Required are the functions `reset() void` and `store(anytype, ExtMove) void`.
fn gen(pos: *const Position, comptime gen_flags: u4, noalias storage: anytype) void {
    storage.reset();

    const us: Color = comptime if (gen_flags & gf_black != 0) .black else .white;
    const them: Color = comptime us.opp();
    const check: bool = comptime gen_flags & gf_check != 0;
    const noisy: bool = comptime gen_flags & gf_noisy != 0;
    const has_pins: bool = comptime gen_flags & gf_pins != 0;
    const do_all_promotions: bool = comptime !noisy;
    const checkers: u64 = pos.checkmask & pos.by_color(them);
    const doublecheck: bool = check and popcnt(checkers) > 1;
    const occ: u64 = pos.all();
    const bb_us: u64 = pos.by_color(us);
    const bb_them: u64 = pos.by_color(them);
    const bb_not_us: u64 = ~bb_us;
    const king_sq: Square = pos.king_square(us);

    //var bb: u64 = undefined;
    var iter: bitboards.BitboardIterator = undefined;

    const pins_diag: u64 = if (has_pins) pos.pins_diag[us.u] else 0;
    const pins_orth: u64 = if (has_pins) pos.pins_orth[us.u] else 0;
    const pins: u64 = if (has_pins) pins_diag | pins_orth else 0;

    // In case of a doublecheck we can only move the king.
    if (!doublecheck) {
        const target: u64 = if (check) pos.checkmask else if (!noisy) bb_not_us else bb_them;
        const our_pawns: u64 = pos.pawns(us);
        const our_knights: u64 = pos.knights(us);
        const our_queens_bishops: u64 = pos.queens_bishops(us);
        const our_queens_rooks: u64 = pos.queens_rooks(us);

        // Pawns.
        if (our_pawns != 0) {
            const pawn_us: Piece = comptime .init(.pawn, us);
            const seventh_rank: u64 = comptime funcs.relative_rank_bb(us, types.rank_7);
            const last_rank: u64 = comptime funcs.relative_rank_bb(us, types.rank_8);
            const enemies: u64 = if (check) checkers else bb_them;
            const dst: PawnDestinations = compute_pawn_destinations(us, our_pawns, ~occ, enemies, target, pins, pins_orth, pins_diag);
            // Pawns normal.
            if (our_pawns & ~seventh_rank != 0) {
                if (check or !noisy) {
                    iter = .init(dst.single & ~last_rank);
                    while (iter.next()) |to| {
                        store(pawn_from(to, us, .up), to, Move.silent, pawn_us, .no_piece, storage);
                    }
                    iter = .init(dst.double);
                    while (iter.next()) |to| {
                        store(if (us.e == .white) to.sub(16) else to.add(16), to, Move.double_push, pawn_us, .no_piece, storage);
                    }
                }
                iter = .init(dst.left & ~last_rank);
                while (iter.next()) |to| {
                    store(pawn_from(to, us, .northwest), to, Move.capture, pawn_us, pos.board[to.u], storage);
                }
                iter = .init(dst.right & ~last_rank);
                while (iter.next()) |to| {
                    store(pawn_from(to, us, .northeast), to, Move.capture, pawn_us, pos.board[to.u], storage);
                }
                if (pos.ep_square.e != .a1) {
                    iter = .init(attacks.get_pawn_attacks(pos.ep_square, them) & our_pawns); // inversion trick.
                    while (iter.next()) |from| {
                        if (is_legal_ep_move(pos, us, king_sq, from, pos.ep_square)) {
                            const pawn_them: Piece = comptime .init(.pawn, them);
                            store(from, pos.ep_square, Move.ep, pawn_us, pawn_them, storage);
                        }
                    }
                }
            }
            // Pawn promotions.
            if (our_pawns & seventh_rank != 0) {
                iter = .init(dst.single & last_rank);
                while (iter.next()) |to| {
                    store_promotions(pos, us, do_all_promotions, false, pawn_from(to, us, .up), to, storage);
                }
                iter = .init(dst.left & last_rank);
                while (iter.next()) |to| {
                    store_promotions(pos, us, do_all_promotions, true, pawn_from(to, us, .northwest), to, storage);
                }
                iter = .init(dst.right & last_rank);
                while (iter.next()) |to| {
                    store_promotions(pos, us, do_all_promotions, true, pawn_from(to, us, .northeast), to, storage);
                }
            }
        } // (pawns)

        // Pieces.
        iter = .init(our_knights & ~pins);
        while (iter.next()) |from| {
            store_many(pos, from, attacks.get_knight_attacks(from) & target, storage);
        }
        iter = .init(our_queens_bishops & ~pins);
        while (iter.next()) |from| {
            store_many(pos, from, attacks.get_bishop_attacks(from, occ) & target, storage);
        }
        iter = .init(our_queens_bishops & pins_diag);
        while (iter.next()) |from| {
            store_many(pos, from, attacks.get_bishop_attacks(from, occ) & target & pins_diag, storage);
        }
        iter = .init(our_queens_rooks & ~pins);
        while (iter.next()) |from| {
            store_many(pos, from, attacks.get_rook_attacks(from, occ) & target, storage);
        }
        iter = .init(our_queens_rooks & pins_orth);
        while (iter.next()) |from| {
            store_many(pos, from, attacks.get_rook_attacks(from, occ) & target & pins_orth, storage);
        }
    } // (not doublecheck)

    // King.
    const king_us: Piece = comptime .init(.king, us);
    const king_target: u64 = if (check or !noisy) bb_not_us else bb_them;
    const to_bb: u64 = attacks.get_king_attacks(king_sq) & king_target;

    // The king is a troublemaker.
    // For now this 'popcount heuristic' gives the best avg speed, using 2 different approaches to check legality.
    if (popcnt(to_bb) > 2) {
        const bb_without_king: u64 = pos.all() ^ pos.kings(us);
        const bb_unsafe: u64 = pos.attacks_by_for_occupation(them, bb_without_king);
        iter = .init(to_bb & ~bb_unsafe);
        // Normal.
        while (iter.next()) |to| {
            const capt: Piece = pos.board[to.u];
            const flag: u4 = if (capt.e == .no_piece) Move.silent else Move.capture;
            store(king_sq, to, flag, king_us, capt, storage);
        }
        // Castling.
        if (!check and !noisy) {
            inline for (Castle.all) |ct| {
                if (is_castling_ok(pos, us, ct, pins_orth, bb_unsafe)) {
                    const flag: u4 = comptime if (ct.e == .short) Move.castle_short else Move.castle_long;
                    const to = pos.layout.rook_start(us, ct); // king takes rook.
                    store(king_sq, to, flag, king_us, .no_piece, storage);
                }
            }
        }
    }
    else {
        const bb_without_king: u64 = occ ^ pos.kings(us);
        iter = .init(to_bb);
        // Normal.
        while (iter.next()) |to| {
            if (is_legal_kingmove(pos, us, bb_without_king, to)) {
                const capt: Piece = pos.board[to.u];
                const movekind: u4 = if (capt.e == .no_piece) Move.silent else Move.capture;
                store(king_sq, to, movekind, king_us, capt, storage);
            }
        }
        // Castling.
        if (!check and !noisy) {
            inline for (Castle.all) |ct| {
                if (is_castling_ok_iterative(pos, us, ct, pins_orth)) {
                    const movekind: u4 = comptime if (ct.e == .short) Move.castle_short else Move.castle_long;
                    const to = pos.layout.rook_start(us, ct); // king takes rook.
                    store(king_sq, to, movekind, king_us, .no_piece, storage);
                }
            }
        }
    }
}

pub fn is_legal(move: Move, pos: *const Position) bool {
    const ex: ExtMove = lazy_find_move(pos, move.from, move.to, move.prom_safe()) catch return false;
    return ex.move == move;
}

pub fn lazy_find_move(pos: *const Position, from_sq: Square, to_sq: Square, prom: PieceType) !ExtMove {
    return switch (pos.stm.e) {
        .white => find_move(pos, .white, from_sq, to_sq, prom),
        .black => find_move(pos, .black, from_sq, to_sq, prom),
    };
}

pub fn find_move(pos: *const Position, comptime us: Color, from_sq: Square, to_sq: Square, prom: PieceType) !ExtMove {
    switch (us.e) {
        .white => {
            switch(pos.gen_flags) {
                inline else => |gf| return find(pos, gf, from_sq, to_sq, prom)
            }
        },
        .black => {
            switch(pos.gen_flags) {
                inline else => |gf| return find(pos, gf | gf_black, from_sq, to_sq, prom)
            }
        },
    }
}

/// Try find the move.
fn find(pos: *const Position, comptime gen_flags: u4, from_sq: Square, to_sq: Square, prom: PieceType) !ExtMove {
    const us: Color = comptime if (gen_flags & gf_black != 0) .black else .white;
    const them: Color = comptime us.opp();
    const check: bool = comptime gen_flags & gf_check != 0;
    const has_pins: bool = comptime gen_flags & gf_pins != 0;
    const piece: Piece = pos.board[from_sq.u];

    if (piece.e == .no_piece or piece.color().e != us.e) {
        return types.ParsingError.IllegalMove;
    }

    const pt: PieceType = piece.piecetype();
    const is_promotion: bool = prom.e != .no_piecetype;
    const bb_from_filter: u64 = from_sq.to_bitboard();
    const bb_to_filter: u64 = to_sq.to_bitboard();
    const captured: Piece = pos.board[to_sq.u];
    const capt_flag: u4 = if (captured.is_empty()) 0 else Move.capture;
    const checkers: u64 = pos.checkmask & pos.by_color(them);
    const doublecheck: bool = check and popcnt(checkers) > 1;
    const occ: u64 = pos.all();
    const bb_us: u64 = pos.by_color(us);
    const bb_them: u64 = pos.by_color(them);
    const bb_not_us: u64 = ~bb_us;
    const king_sq: Square = pos.king_square(us);

    if (doublecheck and pt.e != .king) {
        return types.ParsingError.IllegalMove;
    }

    const pins_diag: u64 = if (has_pins) pos.pins_diag[us.u] else 0;
    const pins_orth: u64 = if (has_pins) pos.pins_orth[us.u] else 0;
    const pins: u64 = if (has_pins) pins_diag | pins_orth else 0;
    const target: u64 = if (check) pos.checkmask else bb_not_us;
    var bb: u64 = undefined;

    sw: switch (pt.e) {
        .pawn => {
            const seventh_rank: u64 = comptime funcs.relative_rank_bb(us, types.rank_7);
            const pawn_us: Piece = comptime .init(.pawn, us);
            const enemies: u64 = if (check) checkers else bb_them;
            const pawn_filter: u64 = if (is_promotion) seventh_rank else ~seventh_rank;
            const our_pawn: u64 = pos.pawns(us) & bb_from_filter & pawn_filter;

            var dst: PawnDestinations = compute_pawn_destinations(us, our_pawn, ~occ, enemies, target, pins, pins_orth, pins_diag);
            dst.combine(bb_to_filter);

            if (!is_promotion) {
                if (dst.single != 0) {
                    return .init(from_sq, to_sq, Move.silent, pawn_us, .no_piece);
                }
                if (dst.double != 0) {
                    return .init(from_sq, to_sq, Move.double_push, pawn_us, .no_piece);
                }
                if (dst.left | dst.right != 0) {
                    return .init(from_sq, to_sq, Move.capture, pawn_us, captured);
                }
                if (pos.ep_square.e == to_sq.e) {
                    bb = attacks.get_pawn_attacks(pos.ep_square, them) & our_pawn; // inversion trick.
                    if (bb != 0) {
                        if (is_legal_ep_move(pos, us, king_sq, from_sq, to_sq)) {
                            const pawn_them: Piece = comptime .init(.pawn, them);
                            return .init(from_sq, to_sq, Move.ep, pawn_us, pawn_them);
                        }
                    }
                }
            }
            else {
                const prom_flag: u4 = prom.to_promotion_move_flag();
                if (dst.single != 0) {
                    return .init(from_sq, to_sq, prom_flag, pawn_us, captured);
                }
                if (dst.left | dst.right != 0) {
                    return .init(from_sq, to_sq, prom_flag | Move.capture_mask, pawn_us, captured);
                }
            }
        },
        .knight => {
            const our_knight: u64 = pos.knights(us) & bb_from_filter;
            if (our_knight & ~pins != 0) {
                if (attacks.get_knight_attacks(from_sq) & target & bb_to_filter != 0) {
                    return .init(from_sq, to_sq, capt_flag, piece, captured);
                }
            }
        },
        .bishop, .rook, .queen => {
            const our_queen_or_bishop: u64 = pos.queens_bishops(us) & bb_from_filter;
            if (our_queen_or_bishop & ~pins != 0) {
                if (attacks.get_bishop_attacks(from_sq, occ) & target & bb_to_filter != 0) {
                    return .init(from_sq, to_sq, capt_flag, piece, captured);
                }
            }
            if (our_queen_or_bishop & pins_diag != 0) {
                if (attacks.get_bishop_attacks(from_sq, occ) & target & pins_diag & bb_to_filter != 0) {
                    return .init(from_sq, to_sq, capt_flag, piece, captured);
                }
            }
            if (pt.e == .bishop) {
                break :sw;
            }
            const our_queen_or_rook: u64 = pos.queens_rooks(us) & bb_from_filter;
            if (our_queen_or_rook & ~pins != 0) {
                if (attacks.get_rook_attacks(from_sq, occ) & target & bb_to_filter != 0) {
                    return .init(from_sq, to_sq, capt_flag, piece, captured);
                }
            }
            if (our_queen_or_rook & pins_orth != 0) {
                if (attacks.get_rook_attacks(from_sq, occ) & target & pins_orth & bb_to_filter != 0) {
                    return .init(from_sq, to_sq, capt_flag, piece, captured);
                }
            }
        },
        .king => {
            const king_us: Piece = comptime .init(.king, us);
            const king_target: u64 = bb_not_us;
            bb = attacks.get_king_attacks(king_sq) & king_target & bb_to_filter;
            const bb_without_king: u64 = occ ^ pos.kings(us);
            const bb_unsafe: u64 = pos.attacks_by_for_occupation(them, bb_without_king);
            bb &= ~bb_unsafe;
            if (bb != 0 and is_legal_kingmove(pos, us, bb_without_king, to_sq)) {
                return .init(king_sq, to_sq, capt_flag, king_us, captured);
            }
            if (!check) {
                inline for (Castle.all) |ct| {
                    if (to_sq.u == pos.layout.rook_start(us, ct).u and is_castling_ok(pos, us, ct, pins_orth, bb_unsafe)) {
                        return .init(king_sq, to_sq, Move.castle(ct), king_us, .no_piece);
                    }
                }
            }
        },
        else => {
            unreachable;
        }
    }

    return types.ParsingError.IllegalMove;
}

inline fn compute_pawn_destinations(comptime us: Color, pawns: u64, empty_squares: u64, enemies: u64, check_interpolation: u64, pins: u64, pins_orth: u64, pins_diag: u64) PawnDestinations {
    const third_rank: u64 = comptime funcs.relative_rank_bb(us, types.rank_3);

    const single: u64 =
        (pawns_shift(pawns & ~pins, us, .up) & empty_squares) |
        (pawns_shift(pawns & pins_orth, us, .up) & empty_squares & pins_orth);

    const double: u64 =
        pawns_shift(single & third_rank, us, .up) & empty_squares;

    const left: u64 =
        (pawns_shift(pawns & ~pins, us, .northwest) & enemies) |
        (pawns_shift(pawns & pins_diag, us, .northwest) & enemies & pins_diag);

    const right: u64 =
        (pawns_shift(pawns & ~pins, us, .northeast) & enemies) |
        (pawns_shift(pawns & pins_diag, us, .northeast) & enemies & pins_diag);

    return .{
        .single = single & check_interpolation,
        .double = double & check_interpolation,
        .left = left,
        .right = right,
    };
}

/// Tricky one. An ep move can uncover a check.
fn is_legal_ep_move(pos: *const Position, comptime us: Color, king_sq: Square, from: Square, to: Square) bool {
    const them: Color = comptime us.opp();
    const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
    const occ: u64 = (pos.all() ^ from.to_bitboard() ^ capt_sq.to_bitboard()) | to.to_bitboard();
    const att: u64 =
        (attacks.get_rook_attacks(king_sq, occ) & pos.queens_rooks(them)) |
        (attacks.get_bishop_attacks(king_sq, occ) & pos.queens_bishops(them));
    return att == 0;
}

fn is_legal_kingmove(pos: *const Position, comptime us: Color, bb_without_king: u64, to: Square) bool {
    const them: Color = comptime us.opp();
    return !pos.is_square_attacked_by_for_occupation(bb_without_king, to, them);
}

/// Castling method 1.
inline fn is_castling_ok(pos: *const Position, comptime us: Color, comptime ct: Castle, pins_orth: u64, bb_unsafe: u64) bool {
    // Frc requires an additional rook pin check. In classic the rooks cannot be pinned.
    return
        pos.has_castlingright(us, ct) and
        pos.layout.empty_path(us, ct) & pos.all() == 0 and
        pins_orth & pos.layout.rook_start(us, ct).to_bitboard() == 0 and
        is_castling_path_safe(pos, us, ct, bb_unsafe);
}

/// Castling method 2.
inline fn is_castling_ok_iterative(pos: *const Position, comptime us: Color, comptime ct: Castle, pins_orth: u64) bool {
    // Frc requires an additional rook pin check. In classic the rooks cannot be pinned.
    return
        pos.has_castlingright(us, ct) and
        pos.layout.empty_path(us, ct) & pos.all() == 0 and
        pins_orth & pos.layout.rook_start(us, ct).to_bitboard() == 0 and
        is_castling_path_safe_iterative(pos, us, ct);
}

fn is_castling_path_safe(pos: *const Position, comptime us: Color, comptime ct: Castle, bb_unsafe: u64) bool {
    return pos.layout.attack_path(us, ct) & bb_unsafe == 0;
}

fn is_castling_path_safe_iterative(pos: *const Position, comptime us: Color, comptime ct: Castle) bool {
    const them: Color = comptime us.opp();
    var iter = bitboards.iterator(pos.layout.attack_path(us, ct));
    while (iter.next()) |sq| {
        if (pos.is_square_attacked_by(sq, them)) return false;
    }
    return true;
}

/// For knights and sliders.
fn store_many(pos: *const Position, from: Square, bb_to: u64, noalias storage: anytype) void {
    const piece: Piece = pos.board[from.u];
    var iter = bitboards.iterator(bb_to);
    while (iter.next()) |to| {
        const captured: Piece = pos.board[to.u];
        const flag: u4 = if (captured.e == .no_piece) Move.silent else Move.capture;
        store(from, to, flag, piece, captured, storage);
    }
}

/// Store all promotions or only queen (for quiescence).
fn store_promotions(pos: *const Position, comptime us: Color, comptime do_all: bool, comptime is_capture: bool, from: Square, to: Square, noalias storage: anytype) void {
    const Q: u4 = if (is_capture) Move.queen_promotion_capture else Move.queen_promotion;
    const R: u4 = if (is_capture) Move.rook_promotion_capture else Move.rook_promotion;
    const B: u4 = if (is_capture) Move.bishop_promotion_capture else Move.bishop_promotion;
    const N: u4 = if (is_capture) Move.knight_promotion_capture else Move.knight_promotion;

    const piece: Piece = .init(.pawn, us);
    const captured: Piece = if (is_capture) pos.board[to.u] else .no_piece;

    store(from, to, Q, piece, captured, storage);
    if (do_all) {
        store(from, to, R, piece, captured, storage);
        store(from, to, B, piece, captured, storage);
        store(from, to, N, piece, captured, storage);
    }
}

/// Store one move to the storage.
fn store(from: Square, to: Square, flags: u4, piece: Piece, captured: Piece, noalias storage: anytype) void {
    storage.store(ExtMove.init(from, to, flags, piece, captured));
}

const PawnDestinations = struct {
    single: u64,
    double: u64,
    left: u64,
    right: u64,

    fn combine(self: *PawnDestinations, mask: u64) void {
        self.single &= mask;
        self.double &= mask;
        self.left &= mask;
        self.right &= mask;
    }
};

/// Basic storage of moves.
pub const MoveStorage = struct {
    moves: [types.max_move_count]ExtMove,
    count: u8,

    pub fn init() MoveStorage {
        return .{
            .moves = undefined,
            .count = 0
        };
    }

    /// Required function.
    pub fn reset(self: *MoveStorage) void {
        self.count = 0;
    }

    /// Required function.
    pub fn store(self: *MoveStorage, extmove: ExtMove) void {
        assert(self.count < types.max_move_count);
        self.moves[self.count] = extmove;
        self.count += 1;
    }

    pub fn slice(self: *const MoveStorage) []const ExtMove {
        return self.moves[0..self.count];
    }
};

/// For only counting moves.
pub const JustCount = struct {
    counted: u8,

    pub fn init() JustCount {
        return .{ .counted = 0 };
    }

    /// Required function.
    pub fn reset(self: *JustCount) void {
        self.counted = 0;
    }

    /// Required function.
    pub fn store(self: *JustCount, _: ExtMove) void {
        self.counted += 1;
    }
};

/// For only checking if there is any move.
pub const Any = struct {
    has_moves: bool,

    pub fn init() Any {
        return .{ .has_moves = false };
    }

    /// Required function.
    pub fn reset(self: *Any) void {
        self.has_moves = false;
    }

    /// Required function.
    pub fn store(self: *Any, _: ExtMove) void {
        self.has_moves = true;
    }
};
