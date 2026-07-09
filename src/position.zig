// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const squarepairs = @import("squarepairs.zig");
const funcs = @import("funcs.zig");
const zobrist = @import("zobrist.zig");
const attacks = @import("attacks.zig");
const movegen = @import("movegen.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;
const popcnt = bitboards.popcnt;
const pawns_shift = funcs.pawns_shift;
const pawn_from = funcs.pawn_from;

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const Castle = types.Castle;
const ScorePair = types.ScorePair;
const SquarePair = squarepairs.SquarePair;

/// See lib.zig.
pub fn finalize() void {
    layout_map.deinit(ctx.gpa);
    layout_map = .empty; // This solves testing problems for now.
}

pub const Position = struct {
    layout: *const Layout,
    board: [Square.count]Piece,
    bitboards_by_type: [PieceType.count]u64,
    bitboards_by_color: [Color.count]u64,
    material: Material,
    phase_by_color: [Color.count]u8,
    stm: Color,
    ply_from_root: u16,
    game_ply: u16,
    nullmove_state: bool,
    rule50: u16,
    ep_square: Square,
    castlingrights: u4,
    key: u64,
    pawnkey: u64,
    nonpawnkeys: [Color.count]u64,
    minorkey: u64,
    majorkey: u64,
    checkmask: u64,
    pins_diag: [Color.count]u64,
    pins_orth: [Color.count]u64,
    is_960: bool,
    gen_flags: u2, // gf_check + gf_pins (for stm only).

    pub const empty: Position = .init_empty();
    pub const classic_startpos: Position = .init_classic_startpos();

    /// Comptime only.
    fn init_empty() Position {
        return .{
            .layout = empty_layout,
            .board = @splat(.no_piece),
            .bitboards_by_type = @splat(0),
            .bitboards_by_color = @splat(0),
            .material = .empty,
            .phase_by_color = @splat(0),
            .stm = .white,
            .ply_from_root = 0,
            .game_ply = 0,
            .nullmove_state = false,
            .rule50 = 0,
            .ep_square = Square.zero,
            .castlingrights = 0,
            .key = 0,
            .pawnkey = 0,
            .nonpawnkeys = @splat(0),
            .minorkey = 0,
            .majorkey = 0,
            .checkmask = 0,
            .pins_diag = @splat(0),
            .pins_orth = @splat(0),
            .is_960 = false,
            .gen_flags = 0,
        };
    }

    /// Comptime only.
    fn init_classic_startpos() Position {
        const b = bitboards;
        var pos: Position = undefined;
        pos.layout = classic_startpos_layout;
        pos.board = create_board_from_backrow(.{ .rook, .knight, .bishop, .queen, .king, .bishop, .knight, .rook });
        pos.bitboards_by_type = .{
            b.bb_rank_2 | b.bb_rank_7,  // pawns
            b.bb_b1 | b.bb_g1 | b.bb_b8 | b.bb_g8, // knights
            b.bb_c1 | b.bb_f1 | b.bb_c8 | b.bb_f8, // bishops
            b.bb_a1 | b.bb_h1 | b.bb_a8 | b.bb_h8, // rooks
            b.bb_d1 | b.bb_d8, // queens
            b.bb_e1 | b.bb_e8  // kings
        };
        pos.bitboards_by_color = .{ b.bb_rank_1 | b.bb_rank_2, b.bb_rank_7 | b.bb_rank_8 };
        pos.material = .default;
        pos.phase_by_color = .{ 12, 12 };
        pos.stm = .white;
        pos.ply_from_root = 0;
        pos.game_ply = 0;
        pos.nullmove_state = false;
        pos.rule50 = 0;
        pos.ep_square = Square.zero;
        pos.castlingrights = 0b1111;
        pos.key = 0;
        pos.pawnkey = 0;
        pos.nonpawnkeys = @splat(0);
        pos.minorkey = 0;
        pos.majorkey = 0;
        pos.checkmask = 0;
        pos.pins_diag = @splat(0);
        pos.pins_orth = @splat(0);
        pos.is_960 = false;
        pos.gen_flags = 0;
        pos.update_hash();
        return pos;
    }

    /// Comptime only.
    fn create_board_from_backrow(backrow: [8]PieceType) [Square.count]Piece {
        var result: [Square.count]Piece = @splat(.no_piece);
        for (backrow, 0..) |pt, i| {
            const sq: Square = .from_int(i);
            result[sq.u] = .init(pt, .white);
            result[sq.u + 8] = .white_pawn;
            result[sq.u + 48] = .black_pawn;
            result[sq.u + 56] = .init(pt, .black);
        }
        return result;
    }

    /// Shorthand direct setup.
    pub fn from_fen(fen_str: []const u8, is_960: bool) !Position {
        var pos: Position = undefined;
        try pos.setup(fen_str, is_960);
        return pos;
    }

    /// Initializes the position from a fen string.
    pub fn setup(self: *Position, fen_str: []const u8, is_960: bool) !void {
        const state_board: u8 = 0;
        const state_color: u8 = 1;
        const state_castle: u8 = 2;
        const state_ep: u8 = 3;
        const state_draw_count: u8 = 4;
        const state_movenumber: u8 = 5;

        // uci is started with "setoption name UCI_Chess960 value true" before ucinewgame.
        self.* = .empty;
        self.is_960 = is_960;

        var layout_key: LayoutKey = .empty;
        var parse_state: u8 = state_board;
        var rank: u3 = types.rank_8;
        var file: u3 = types.file_a;
        var tokenizer = std.mem.tokenizeScalar(u8, fen_str, ' ');

        outer: while (tokenizer.next()) |token| : (parse_state += 1) {
            switch (parse_state) {
                state_board => {
                    for (token) |c| {
                        switch (c) {
                            '1'...'8' => {
                                const empty_squares: u3 = @truncate(c - '0'); // '8' -> 0
                                file +|= empty_squares;
                            },
                            '/' => {
                                rank -|= 1;
                                file = types.file_a;
                            },
                            else => {
                                const pc: Piece = try .from_char(c);
                                const sq: Square = .from_rank_file(rank, file);
                                self.add_piece_and_update_keys(pc, sq);
                                file +|= 1;
                            },
                        }
                    }
                    // Board ready:
                    if (popcnt(self.kings(.white)) != 1 or popcnt(self.kings(.black)) != 1) {
                        return Error.InvalidKings;
                    }
                    layout_key.init_kings(self.king_square(.white), self.king_square(.black));
                },
                state_color => {
                    if (token[0] == 'b') {
                        self.stm = .black;
                        self.key ^= zobrist.btm();
                    }
                },
                state_castle => {
                    for (token) |c| {
                        switch (c) {
                            // Classic.
                            'K' => try self.set_castling_right(&layout_key, .white, 7, true),
                            'Q' => try self.set_castling_right(&layout_key, .white, 0, true),
                            'k' => try self.set_castling_right(&layout_key, .black, 7, true),
                            'q' => try self.set_castling_right(&layout_key, .black, 0, true),
                            'A'...'H' => try self.set_castling_right(&layout_key, .white, @truncate(c - 'A'), false),
                            'a'...'h' => try self.set_castling_right(&layout_key, .black, @truncate(c - 'a'), false),
                            else => {},
                        }
                    }
                    // Castling ready.
                    self.key ^= zobrist.castling(self.castlingrights);
                },
                state_ep => {
                    if (token.len == 2) {
                        const ep: Square = Square.from_string(token);
                        if (self.is_usable_ep_square(ep)) {
                            self.ep_square = ep;
                            self.key ^= zobrist.enpassant(ep);
                        }
                    }
                },
                state_draw_count => {
                    const v: u16 = std.fmt.parseInt(u16, token, 10) catch break :outer;
                    self.rule50 = v;
                },
                state_movenumber => {
                    const v: u16 = std.fmt.parseInt(u16, token, 10) catch break :outer;
                    self.game_ply = funcs.movenumber_to_ply(v, self.stm);
                },
                else => {
                    break :outer;
                }
            }
        }
        self.is_960 |= layout_key.detect_frc();
        //assert(self.detect_frc() == layout_key.detect_frc());
        self.layout = select_layout(layout_key);
        self.lazy_update_state();
    }

    /// Assumes king is on the board. Applies a rook correction for 'KQkq' castling rights, if nessesary.
    pub fn set_castling_right(self: *Position, key: *LayoutKey, us: Color, input_rook_file: u3, comptime rook_correction: bool) !void {
        const first_rank: u3 = types.relative_rank(us, 0);
        const rank_bb: u64 = bitboards.relative_rank_bitboard(us, 0);
        const king_sq: Square = self.king_square(us);
        const rook_sq: Square = .from_rank_file(first_rank, input_rook_file);
        const rook: Piece = .init(.rook, us);
        var rook_file: u3 = input_rook_file;

        if (king_sq.coord.rank != first_rank) {
            return Error.CastlingLogic;
        }

        if (king_sq.coord.file == types.file_a or king_sq.coord.file == types.file_h) {
            return Error.CastlingLogic;
        }

        const is_rook: bool = self.get(rook_sq).e == rook.e;

        if (!rook_correction and !is_rook) {
            return Error.CastlingLogic;
        }

        if (!rook_correction and king_sq.u == rook_sq.u) {
            return Error.CastlingLogic;
        }

        if (rook_file > king_sq.coord.file) {
            if (rook_correction and !is_rook) {
                // Try to find the most right rook, right from the king.
                const bb: u64 = self.rooks(us) & rank_bb;
                const sq: Square = bitboards.last_square_or_null(bb) orelse return Error.CastlingLogic;
                if (sq.u < king_sq.u) return Error.CastlingLogic;
                rook_file = sq.coord.file;
            }

            const cf: u4 = Castling.flag(us,.short);
            key.set_rook_file(cf, rook_file);
            self.castlingrights |= cf;
        }
        else if (rook_file < king_sq.coord.file) {
            if (rook_correction and !is_rook) {
                // Try to find the most left rook, left from the king.
                const bb: u64 = self.rooks(us) & rank_bb;
                const sq: Square = bitboards.first_square_or_null(bb) orelse return Error.CastlingLogic;
                if (sq.u > king_sq.u) return Error.CastlingLogic;
                rook_file = sq.coord.file;
            }

            const cf: u4 = Castling.flag(us,.long);
            key.set_rook_file(cf, rook_file);
            self.castlingrights |= cf;
        }
    }

    pub fn phase(self: *const Position) u8 {
        return self.phase_by_color[0] + self.phase_by_color[1];
    }

    /// Parses a uci-move.
    pub fn parse_move(self: *const Position, str: []const u8) !ExtMove {
        if (str.len < 4 or str.len > 5) {
            return types.ParsingError.IllegalMove;
        }

        const us = self.stm;
        const from: Square = Square.from_string(str[0..2]);
        var to: Square = Square.from_string(str[2..4]);
        var promo: PieceType = .no_piecetype;

        if (str.len == 4) {
            // TODO: code this better.
            // TODO: catch "g1g1" types of castling moves. the last one i do not catch.
            // Repair malformed castling moves in chess960 positions (not encoding 'king takes rook').
            if ((from.e == .e1 or from.e == .e8) and self.get(from).is_king_of_color(us)) {
                if (to.e == .g1 or to.e == .g8) {
                    const new_to: Square = self.layout.rook_start(us, .short);
                    if (self.get(new_to).is_rook_of_color(us)) {
                        to = new_to;
                    }
                }
                else if (to.e == .c1 or to.e == .c8) {
                    const new_to: Square = self.layout.rook_start(us, .long);
                    if (self.get(new_to).is_rook_of_color(us)) {
                        to = new_to;
                    }
                }
            }
        }
        else if (str.len == 5) {
            promo = switch (str[4]) {
                'n' => .knight,
                'b' => .bishop,
                'r' => .rook,
                'q' => .queen,
                else => return types.ParsingError.InvalidPromotionChar,
            };
        }
        return movegen.lazy_find_move(self, from, to, promo);
    }

    /// A convenient way to set the startposition without the need for a fen string.
    pub fn set_startpos(self: *Position) void {
        self.* = classic_startpos;
    }

    fn compute_hashkeys(self: *const Position, key: *u64, pawnkey: *u64, white_nonpawnkey: *u64, black_nonpawnkey: *u64, minorkey: *u64, majorkey: *u64) void {
        key.* = 0;
        pawnkey.* = 0;
        white_nonpawnkey.* = 0;
        black_nonpawnkey.* = 0;
        minorkey.* = 0;
        majorkey.* = 0;
        var iter = bitboards.iterator(self.all());
        while (iter.next()) |sq| {
            const pc = self.get(sq);
            const z_key: u64 = zobrist.piece_square(pc, sq);
            key.* ^= z_key;
            if (pc.is_pawn()) {
                pawnkey.* ^= z_key;
            }
            else {
                if (pc.color().e == .white) white_nonpawnkey.* ^= z_key else black_nonpawnkey.* ^= z_key;
                if (pc.is_minor()) {
                    minorkey.* ^= z_key;
                }
                else if (pc.is_major()) {
                    majorkey.* ^= z_key;
                }
            }
        }
        key.* ^= zobrist.castling(self.castlingrights);
        if (self.ep_square.e != Square.zero.e) {
            key.* ^= zobrist.enpassant(self.ep_square);
        }
        if (self.stm.e == .black) {
            key.* ^= zobrist.btm();
        }
    }

    pub fn get(self: *const Position, sq: Square) Piece {
        return self.board[sq.u];
    }

    pub fn by_type(self: *const Position, pt: PieceType) u64 {
        return self.bitboards_by_type[pt.u];
    }

    pub fn by_color(self: *const Position, us: Color) u64 {
        return self.bitboards_by_color[us.u];
    }

    pub fn all(self: *const Position) u64 {
        return self.bitboards_by_color[0] | self.bitboards_by_color[1];
    }

    pub fn all_pawns(self: *const Position) u64 {
        return self.by_type(.pawn);
    }

    pub fn all_knights(self: *const Position) u64 {
        return self.by_type(.knight);
    }

    pub fn all_bishops(self: *const Position) u64 {
        return self.by_type(.bishop);
    }

    pub fn all_rooks(self: *const Position) u64 {
        return self.by_type(.rook);
    }

    pub fn all_queens(self: *const Position) u64 {
        return self.by_type(.queen);
    }

    pub fn all_kings(self: *const Position) u64 {
        return self.by_type(.king);
    }

    pub fn all_queens_bishops(self: *const Position) u64 {
        return self.by_type(.bishop) | self.by_type(.queen);
    }

    pub fn all_queens_rooks(self: *const Position) u64 {
        return self.by_type(.rook) | self.by_type(.queen);
    }

    pub fn all_minors(self: *const Position) u64 {
        return self.all_knights() | self.all_bishops();
    }

    pub fn all_majors(self: *const Position) u64 {
        return self.all_rooks() | self.all_queens();
    }

    pub fn pawns(self: *const Position, us: Color) u64 {
        return self.by_type(.pawn) & self.by_color(us);
    }

    pub fn knights(self: *const Position, us: Color) u64 {
        return self.by_type(.knight) & self.by_color(us);
    }

    pub fn bishops(self: *const Position, us: Color) u64 {
        return self.by_type(.bishop) & self.by_color(us);
    }

    pub fn rooks(self: *const Position, us: Color) u64 {
        return self.by_type(.rook) & self.by_color(us);
    }

    pub fn queens(self: *const Position, us: Color) u64 {
        return self.by_type(.queen) & self.by_color(us);
    }

    pub fn kings(self: *const Position, us: Color) u64 {
        return self.by_type(.king) & self.by_color(us);
    }

    pub fn pieces(self: *const Position, comptime pt: PieceType, us: Color) u64 {
        return self.bitboards_by_type[pt.u] & self.bitboards_by_color[us.u];
    }

    pub fn queens_bishops(self: *const Position, us: Color) u64 {
        return (self.by_type(.bishop) | self.by_type(.queen)) & self.by_color(us);
    }

    pub fn queens_rooks(self: *const Position, us: Color) u64 {
        return (self.by_type(.rook) | self.by_type(.queen)) & self.by_color(us);
    }

    pub fn minors(self: *const Position, us: Color) u64 {
        return self.all_minors() & self.by_color(us);
    }

    pub fn majors(self: *const Position, us: Color) u64 {
        return self.all_majors() & self.by_color(us);
    }

    /// Assumes there is a pawn. Returns lsb.
    pub fn pawn_square(self: *const Position, us: Color) Square {
        return bitboards.first_square(self.pawns(us));
    }

    /// Assumes there is a bishop. Returns lsb.
    pub fn bishop_square(self: *const Position, us: Color) Square {
        return bitboards.first_square(self.bishops(us));
    }

    /// Assumes there is a king. Returns lsb.
    pub fn king_square(self: *const Position, us: Color) Square {
        return bitboards.first_square(self.kings(us));
    }

    pub fn sliders(self: *const Position, us: Color) u64 {
        return (self.by_type(.bishop) | self.by_type(.rook) | self.by_type(.queen)) & self.by_color(us);
    }

    pub fn pawn_count(self: *const Position, us: Color) u8 {
        return self.material.counts[us.u][PieceType.pawn.u];
    }

    pub fn knight_count(self: *const Position, us: Color) u8 {
        return self.material.counts[us.u][PieceType.knight.u];
    }

    pub fn bishop_count(self: *const Position, us: Color) u8 {
        return self.material.counts[us.u][PieceType.bishop.u];
    }

    pub fn rook_count(self: *const Position, us: Color) u8 {
        return self.material.counts[us.u][PieceType.rook.u];
    }

    pub fn queen_count(self: *const Position, us: Color) u8 {
        return self.material.counts[us.u][PieceType.queen.u];
    }

    pub fn king_count(self: *const Position, us: Color) u8 {
        return self.material.counts[us.u][PieceType.king.u];
    }

    pub fn minor_count(self: *const Position, us: Color) u8 {
        return self.material.counts[us.u][PieceType.knight.u] + self.material.counts[us.u][PieceType.bishop.u];
    }

    pub fn major_count(self: *const Position, us: Color) u8 {
        return self.material.counts[us.u][PieceType.rook.u] + self.material.counts[us.u][PieceType.queen.u];
    }

    pub fn minor_major_count(self: *const Position, us: Color) u8 {
        return self.minor_count(us) + self.major_count(us);
    }

    pub fn count_of(self: *const Position, pt: PieceType, us: Color) u8 {
        return self.material.counts[us.u][pt.u];
    }

    pub fn has_castlingright(self: *const Position, us: Color, ct: Castle) bool {
        return self.castlingrights & Castling.flag(us, ct) != 0;
    }

    pub fn in_check(self: *const Position) bool {
        return self.checkmask != 0;
    }

    /// Returns a bitboard of pin rays.
    pub fn pins(self: *const Position, us: Color) u64 {
        return self.pins_diag[us.u] | self.pins_orth[us.u];
    }

    /// Returns a bitboard of all pin rays.
    pub fn all_pins(self: *const Position) u64 {
        return self.pins_diag[0] | self.pins_diag[1] | self.pins_orth[0] | self.pins_orth[1];
    }

    pub fn is_draw_by_insufficient_material(self: *const Position) bool {
        const us: Color = .white;
        const them: Color = .black;

        // Queens rooks or pawns: no draw
        if (self.all_queens_rooks() | self.all_pawns() != 0) {
            return false;
        }

        // Only kings: draw
        if (self.all_kings() == self.all()) {
            return true;
        }

        const our_minor_count: u8 = self.minor_count(us);
        const their_minor_count: u8 = self.minor_count(them);

        if (our_minor_count > 1 or their_minor_count > 1) {
            return false;
        }

        // Only king on one side and 1 minor piece on the other: draw
        const K: u48 = comptime Material.encode_48(0, 0, 0, 0, 0);
        if (
            (their_minor_count == 1 and self.material.decode_side(us) == K) or
            (our_minor_count == 1 and self.material.decode_side(them) == K)
        ) {
            return true;
        }

        // Kings with bishops on the same color: draw
        if (self.all_knights() == 0) {
            const same_color = ((self.bishops(us) & bitboards.bb_black_squares) == 0) == ((self.bishops(them) & bitboards.bb_black_squares) == 0);
            if (same_color) {
                return true;
            }
        }
        return false;
    }

    /// Only used for initializing a fen.
    fn is_usable_ep_square(self: *const Position, ep: Square) bool {
        const rank: u3 = ep.coord.rank;
        if (rank == types.rank_3) {
            const w_pawn_sq = ep.add(8);
            const requirements: bool = self.board[w_pawn_sq.u].e == .white_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u - 8].e == .no_piece;
            return requirements and (bitboards.adjacent_square_masks[w_pawn_sq.u] & self.pawns(.black) != 0);
        }
        else if (rank == types.rank_6) {
            const b_pawn_sq = ep.sub(8);
            const requirements:  bool = self.board[b_pawn_sq.u].e == .black_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u + 8].e == .no_piece;
            return requirements and (bitboards.adjacent_square_masks[b_pawn_sq.u] & self.pawns(.white) != 0);
        }
        return false;
    }

    /// Also updates keys.
    pub fn add_piece_and_update_keys(self: *Position, pc: Piece, sq: Square) void {
        const color: Color = pc.color();
        switch (color.e) {
            .white => self.add_piece(.white, pc, sq),
            .black => self.add_piece(.black, pc, sq),
        }
        const k: u64 = zobrist.piece_square(pc, sq);
        self.key ^= k;
        if (pc.is_pawn()) {
            self.pawnkey ^= k;
        }
        else {
            self.nonpawnkeys[color.u] ^= k;
            if (pc.is_minor()) {
                self.minorkey ^= k;
            }
            else if (pc.is_major()) {
                self.majorkey ^= k;
            }
        }
    }

    /// Updates board, bitboards, material, phase. Not keys.
    pub fn lazy_add_piece(self: *Position, pc: Piece, sq: Square) void {
        switch (pc.color().e) {
            .white => self.add_piece(.white, pc, sq),
            .black => self.add_piece(.black, pc, sq),
        }
    }

    /// Updates board, bitboards, material, phase. Not keys.
    fn add_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        if (lib.is_paranoid) {
            assert(self.get(sq).is_empty());
            assert(!pc.is_empty());
            assert(pc.color().e == us.e);
        }
        const mask: u64 = sq.to_bitboard();
        self.board[sq.u] = pc;
        const pt: u4 = if (us.e == .white) pc.u else pc.u - 6;
        self.bitboards_by_type[pt] |= mask;
        self.bitboards_by_color[us.u] |= mask;
        self.material.counts[us.u][pt] += 1;
        self.phase_by_color[us.u] += types.phase_table[pt];
    }

    /// Update sboard, bitboards, material, phase. Not keys.
    fn remove_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        if (lib.is_paranoid) {
            assert(self.get(sq).e == pc.e);
            assert(pc.color().e == us.e);
        }
        const not_mask: u64 = ~sq.to_bitboard();
        self.board[sq.u] = .no_piece;
        const pt: u4 = if (us.e == .white) pc.u else pc.u - 6;
        self.bitboards_by_type[pt] &= not_mask;
        self.bitboards_by_color[us.u] &= not_mask;
        self.material.counts[us.u][pt] -= 1;
        self.phase_by_color[us.u] -= types.phase_table[pt];
    }

    /// Updates board, bitboards. Not keys.
    fn move_piece(self: *Position, comptime us: Color, pc: Piece, from: Square, to: Square) void {
        if (lib.is_paranoid) {
            assert(self.get(from).e == pc.e);
            assert(self.get(to).is_empty());
            assert(pc.color().e == us.e);
        }
        const xor_mask: u64 = from.to_bitboard() | to.to_bitboard();
        self.board[from.u] = .no_piece;
        self.board[to.u] = pc;
        if (us.e == .white) self.bitboards_by_type[pc.u] ^= xor_mask else self.bitboards_by_type[pc.u - 6] ^= xor_mask;
        self.bitboards_by_color[us.u] ^= xor_mask;
    }

    /// First creates an ExtMove.
    pub fn lazy_do_raw_move(self: *Position, m: Move) void {
        switch (self.stm.e) {
            .white => self.do_raw_move(.white, m),
            .black => self.do_raw_move(.black, m),
        }
    }

    /// First creates an ExtMove.
    pub fn do_raw_move(self: *Position, comptime us: Color, m: Move) void {
        const them: Color = us.opp();
        const piece: Piece = self.get(m.from);
        const captured: Piece = switch (m.simple_kind()) {
            .default, .promotion => self.get(m.to),
            .ep => .init(.pawn, them),
            .castle => .no_piece,
        };
        self.do_move(us, ExtMove.from_move(m, piece, captured));
    }

    pub fn lazy_do_move(self: *Position, ex: ExtMove) void {
        switch (self.stm.e) {
            .white => self.do_move(.white, ex),
            .black => self.do_move(.black, ex),
        }
    }

    pub fn do_move(self: *Position, comptime us: Color, ex: ExtMove) void {
        if (lib.is_paranoid) {
            assert(us.e == self.stm.e);
            self.assert_pos_ok(ex);
        }
        const predicted_key = if (lib.is_paranoid) self.predict_key(us, ex) else void;

        const them: Color = comptime us.opp();
        const from: Square = ex.move.from;
        const to: Square = ex.move.to;
        const pc: Piece = ex.piece;
        const key_delta = zobrist.piece_square_from_to(pc, from, to);

        // Update key. Clear ep by default. Note that the zobrist for square a1 (invalid ep) is 0 so this xor is safe.
        self.key ^= zobrist.btm() ^ zobrist.enpassant(self.ep_square);

        // Update some stuff.
        self.stm = them;
        self.ply_from_root += 1;
        self.game_ply += 1;
        self.nullmove_state = false;
        self.ep_square = Square.zero;

        // Update the castling rights.
        if (self.castlingrights != 0) {
            const mask: u4 = self.layout.masks[from.u] | self.layout.masks[to.u];
            if (mask != 0) {
                self.key ^= zobrist.castling(self.castlingrights);
                self.castlingrights &= ~mask;
                self.key ^= zobrist.castling(self.castlingrights);
            }
        }

        // The switch is in numerical order.
        sw: switch (ex.move.kind) {
            Move.silent => {
                self.move_piece(us, pc, from, to);
                self.key ^= key_delta;
                if (pc.is_pawn_of_color(us)) {
                    self.rule50 = 0;
                    self.pawnkey ^= key_delta;
                }
                else {
                    self.rule50 += 1;
                    self.nonpawnkeys[us.u] ^= key_delta;
                    if (pc.is_minor_of_color(us)) {
                        self.minorkey ^= key_delta;
                    }
                    else if (pc.is_major_of_color(us)) {
                        self.majorkey ^= key_delta;
                    }
                }
            },
            Move.double_push => {
                self.rule50 = 0;
                self.move_piece(us, pc, from, to);
                self.key ^= key_delta;
                self.pawnkey ^= key_delta;
                // Only set ep if usable.
                if (bitboards.adjacent_square_masks[to.u] & self.pawns(them) != 0) {
                    const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                    self.ep_square = ep;
                    self.key ^= zobrist.enpassant(ep);
                }
            },
            Move.castle_short => {
                self.rule50 += 1;
                const king: Piece = comptime .init(.king, us);
                const rook: Piece = comptime .init(.rook, us);
                const king_to: Square = comptime Castling.king_dest(us, .short);
                const rook_to: Square = comptime Castling.rook_dest(us, .short);
                // Safe for chess960.
                self.remove_piece(us, rook, to);
                self.remove_piece(us, king, from);
                self.add_piece(us, king, king_to);
                self.add_piece(us, rook, rook_to);
                const king_delta: u64 = zobrist.piece_square_from_to(king, from, king_to);
                const rook_delta: u64 = zobrist.piece_square_from_to(rook, to, rook_to);
                const castle_delta: u64 = king_delta ^ rook_delta;
                self.key ^= castle_delta;
                self.nonpawnkeys[us.u] ^= castle_delta;
                self.majorkey ^= rook_delta;
            },
            Move.castle_long => {
                self.rule50 += 1;
                const king: Piece = comptime .init(.king, us);
                const rook: Piece = comptime .init(.rook, us);
                const king_to: Square = comptime Castling.king_dest(us, .long);
                const rook_to: Square = comptime Castling.rook_dest(us, .long);
                // Safe for chess960.
                self.remove_piece(us, rook, to);
                self.remove_piece(us, king, from);
                self.add_piece(us, king, king_to);
                self.add_piece(us, rook, rook_to);
                const king_delta: u64 = zobrist.piece_square_from_to(king, from, king_to);
                const rook_delta: u64 = zobrist.piece_square_from_to(rook, to, rook_to);
                const castle_delta: u64 = king_delta ^ rook_delta;
                self.key ^= castle_delta;
                self.nonpawnkeys[us.u] ^= castle_delta;
                self.majorkey ^= rook_delta;
            },
            Move.knight_promotion => {
                self.rule50 = 0;
                const pawn: Piece = comptime .init(.pawn, us);
                const prom: Piece = comptime .init(.knight, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                self.key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                self.pawnkey ^= zobrist.piece_square(pawn, from);
                self.nonpawnkeys[us.u] ^= zobrist.piece_square(prom, to);
                self.minorkey ^= zobrist.piece_square(prom, to);
            },
            Move.bishop_promotion => {
                self.rule50 = 0;
                const pawn: Piece = comptime .init(.pawn, us);
                const prom: Piece = comptime .init(.bishop, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                self.key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                self.pawnkey ^= zobrist.piece_square(pawn, from);
                self.nonpawnkeys[us.u] ^= zobrist.piece_square(prom, to);
                self.minorkey ^= zobrist.piece_square(prom, to);
            },
            Move.rook_promotion => {
                self.rule50 = 0;
                const pawn: Piece = comptime .init(.pawn, us);
                const prom: Piece = comptime .init(.rook, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                self.key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                self.pawnkey ^= zobrist.piece_square(pawn, from);
                self.nonpawnkeys[us.u] ^= zobrist.piece_square(prom, to);
                self.majorkey ^= zobrist.piece_square(prom, to);
            },
            Move.queen_promotion => {
                self.rule50 = 0;
                const pawn: Piece = comptime .init(.pawn, us);
                const prom: Piece = comptime .init(.queen, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                self.key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                self.pawnkey ^= zobrist.piece_square(pawn, from);
                self.nonpawnkeys[us.u] ^= zobrist.piece_square(prom, to);
                self.majorkey ^= zobrist.piece_square(prom, to);
            },
            Move.capture => {
                const capt: Piece = ex.captured;
                self.rule50 = 0;

                self.remove_piece(them, capt, to);
                self.move_piece(us, pc, from, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);

                // Moved piece keys.
                self.key ^= capt_delta ^ key_delta;
                if (pc.is_pawn_of_color(us)) {
                    self.pawnkey ^= key_delta;
                }
                else {
                    self.nonpawnkeys[us.u] ^= key_delta;
                    if (pc.is_minor_of_color(us)) {
                        self.minorkey ^= key_delta;
                    }
                    else if (pc.is_major_of_color(us)) {
                        self.majorkey ^= key_delta;
                    }
                }

                // Captured piece keys.
                if (capt.is_pawn_of_color(them)) {
                    self.pawnkey ^= capt_delta;
                }
                else {
                    self.nonpawnkeys[them.u] ^= capt_delta;
                    if (capt.is_minor_of_color(them)) {
                        self.minorkey ^= capt_delta;
                    }
                    else if (capt.is_major_of_color(them)) {
                        self.majorkey ^= capt_delta;
                    }
                }
            },
            Move.ep => {
                self.rule50 = 0;
                const pawn_us: Piece = comptime .init(.pawn, us);
                const pawn_them: Piece = comptime .init(.pawn, them);
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.remove_piece(them, pawn_them, capt_sq);
                self.move_piece(us, pawn_us, from, to);
                self.key ^= key_delta ^ zobrist.piece_square(pawn_them, capt_sq);
                self.pawnkey ^= zobrist.piece_square(pawn_us, from) ^ zobrist.piece_square(pawn_us, to) ^ zobrist.piece_square(pawn_them, capt_sq);
            },
            Move.knight_promotion_capture => {
                const capt: Piece = ex.captured;
                self.remove_piece(them, capt, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                self.key ^= capt_delta;
                self.nonpawnkeys[them.u] ^= capt_delta;
                if (capt.is_minor_of_color(them)) {
                    self.minorkey ^= capt_delta;
                }
                else {
                    self.majorkey ^= capt_delta;
                }
                continue :sw Move.knight_promotion;
            },
            Move.bishop_promotion_capture => {
                const capt: Piece = ex.captured;
                self.remove_piece(them, capt, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                self.key ^= capt_delta;
                self.nonpawnkeys[them.u] ^= capt_delta;
                if (capt.is_minor_of_color(them)) {
                    self.minorkey ^= capt_delta;
                }
                else {
                    self.majorkey ^= capt_delta;
                }
                continue :sw Move.bishop_promotion;
            },
            Move.rook_promotion_capture => {
                const capt: Piece = ex.captured;
                self.remove_piece(them, capt, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                self.key ^= capt_delta;
                self.nonpawnkeys[them.u] ^= capt_delta;
                if (capt.is_minor_of_color(them)) {
                    self.minorkey ^= capt_delta;
                }
                else {
                    self.majorkey ^= capt_delta;
                }
                continue :sw Move.rook_promotion;
            },
            Move.queen_promotion_capture => {
                const capt: Piece = ex.captured;
                self.remove_piece(them, capt, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                self.key ^= capt_delta;
                self.nonpawnkeys[them.u] ^= capt_delta;
                if (capt.is_minor_of_color(them)) {
                    self.minorkey ^= capt_delta;
                }
                else {
                    self.majorkey ^= capt_delta;
                }
                continue :sw Move.queen_promotion;
            },
            else => {
                unreachable;
            },
        }

        self.update_state(them);

        if (lib.is_paranoid) {
            assert(predicted_key == self.key);
            self.assert_pos_ok(ex);
        }
    }

    pub fn lazy_do_nullmove(self: *Position) void {
        switch (self.stm.e) {
            .white => self.do_nullmove(.white),
            .black => self.do_nullmove(.black),
        }
    }

    /// Skip a turn.
    pub fn do_nullmove(self: *Position, comptime us: Color) void {
        if (lib.is_paranoid) {
            assert(!self.nullmove_state);
        }
        self.nullmove_state = true;
        const them: Color = comptime us.opp();
        // Clear ep. Note that the ep zobrist for square a1 is 0 so this xor is safe.
        self.key ^= zobrist.btm() ^ zobrist.enpassant(self.ep_square);
        self.stm = them;
        self.ply_from_root += 1;
        self.game_ply += 1;
        // TODO: Rule50 += 1 ???
        self.ep_square = Square.zero;
        self.update_state(them); // TODO: optimize
    }

    /// Computes the key only. Assumes the move is not yet on the board.
    /// Pass an empty move for predicting a nullmove key.
    pub fn predict_key(self: *const Position, comptime us: Color, ex: ExtMove) u64 {
        const them: Color = comptime us.opp();
        const from: Square = ex.move.from;
        const to: Square = ex.move.to;
        const pc: Piece = ex.piece;

        // Clear ep by default. Note that the zobrist for square a1 (invalid ep) is 0 so this xor is safe.
        var key: u64 = self.key ^ zobrist.btm() ^ zobrist.enpassant(self.ep_square);

        // If this is a nullmove we're done.
        if (ex.move.is_empty()) {
            return key;
        }

        const key_delta = zobrist.piece_square_from_to(pc, from, to);

        // Update the castling rights.
        if (self.castlingrights != 0) {
            const mask: u4 = self.layout.masks[from.u] | self.layout.masks[to.u];
            if (mask != 0) {
                key ^= zobrist.castling(self.castlingrights);
                const new_rights: u4 = self.castlingrights & ~mask;
                key ^= zobrist.castling(new_rights);
            }
        }

        // Switch is in numerical order.
        sw: switch (ex.move.kind) {
            Move.silent => {
                key ^= key_delta;
            },
            Move.double_push => {
                key ^= key_delta;
                // Only use ep if usable.
                if (bitboards.adjacent_square_masks[to.u] & self.pawns(them) != 0) {
                    const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                    key ^= zobrist.enpassant(ep);
                }
            },
            Move.castle_short => {
                const king: Piece = comptime Piece.init(.king, us);
                const rook: Piece = comptime Piece.init(.rook, us);
                const king_to: Square = comptime Castling.king_dest(us, .short);
                const rook_to: Square = comptime Castling.rook_dest(us, .short); // king takes rook
                const castle_delta: u64 = zobrist.piece_square_from_to(king, from, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
                key ^= castle_delta;
            },
            Move.castle_long => {
                const king: Piece = comptime Piece.init(.king, us);
                const rook: Piece = comptime Piece.init(.rook, us);
                const king_to: Square = comptime Castling.king_dest(us, .long);
                const rook_to: Square = comptime Castling.rook_dest(us, .long); // king takes rook
                const castle_delta: u64 = zobrist.piece_square_from_to(king, from, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
                key ^= castle_delta;
            },
            Move.knight_promotion => {
                const pawn: Piece = comptime Piece.init(.pawn, us);
                const prom: Piece = comptime Piece.init(.knight, us);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            Move.bishop_promotion => {
                const pawn: Piece = comptime Piece.init(.pawn, us);
                const prom: Piece = comptime Piece.init(.bishop, us);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            Move.rook_promotion => {
                const pawn: Piece = comptime Piece.init(.pawn, us);
                const prom: Piece = comptime Piece.init(.rook, us);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            Move.queen_promotion => {
                const pawn: Piece = comptime Piece.init(.pawn, us);
                const prom: Piece = comptime Piece.init(.queen, us);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            Move.capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta ^ key_delta;
            },
            Move.ep => {
                const pawn_them: Piece = comptime Piece.init(.pawn, them);
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                key ^= key_delta ^ zobrist.piece_square(pawn_them, capt_sq);
            },
            Move.knight_promotion_capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                continue :sw Move.knight_promotion;
            },
            Move.bishop_promotion_capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                continue :sw Move.bishop_promotion;
            },
            Move.rook_promotion_capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                continue :sw Move.rook_promotion;
            },
            Move.queen_promotion_capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                continue :sw Move.queen_promotion;
            },
            else => {
                unreachable;
            },
        }
        return key;
    }

    pub fn lazy_predict_check(self: *const Position, ex: ExtMove) bool {
        return switch(self.stm.e) {
            .white => self.predict_check(.white, ex),
            .black => self.predict_check(.black, ex),
        };
    }

    pub fn predict_check(self: *const Position, comptime us: Color, ex: ExtMove) bool {
        const them: Color = comptime us.opp();
        const king_sq: Square = self.king_square(them);
        const king_bb: u64 = king_sq.to_bitboard();
        const from: Square = ex.move.from;
        const to: Square = ex.move.to;
        const from_bb: u64 = from.to_bitboard();
        const to_bb: u64 = to.to_bitboard();
        var occ: u64 = self.all();

        switch (ex.move.kind) {
            Move.silent, Move.double_push, Move.capture => {
                const pt: PieceType = self.board[from.u].piecetype();
                occ &= ~from_bb;
                occ |= to_bb;
                const delta: u64 = from_bb | to_bb;
                const qr: u64 = if (pt.e == .queen or pt.e == .rook) self.queens_rooks(us) ^ delta else self.queens_rooks(us);
                const qb: u64 = if (pt.e == .queen or pt.e == .bishop) self.queens_bishops(us) ^ delta else self.queens_bishops(us);
                return
                    (attacks.get_piece_attacks(to, pt, us, occ) & king_bb) |
                    (attacks.get_rook_attacks(king_sq, occ) & qr) |
                    (attacks.get_bishop_attacks(king_sq, occ) & qb) != 0;
            },
            Move.ep => {
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                occ &= ~from_bb;
                occ |= to_bb;
                occ &= ~capt_sq.to_bitboard();
                const qr: u64 = self.queens_rooks(us);
                const qb: u64 = self.queens_bishops(us);
                return
                    (attacks.get_pawn_attacks(to, us) & king_bb) |
                    (attacks.get_rook_attacks(king_sq, occ) & qr) |
                    (attacks.get_bishop_attacks(king_sq, occ) & qb) != 0;
            },
            Move.castle_short => {
                const ct: Castle = Castle.short;
                const rook_dest: Square = Castling.rook_dest(us, ct);
                occ &= ~from_bb;
                occ |= Castling.king_dest(us, ct).to_bitboard();
                occ &= ~self.layout.rook_start(us, ct).to_bitboard();
                occ |= rook_dest.to_bitboard();
                return attacks.get_rook_attacks(rook_dest, occ) & king_bb != 0;
            },
            Move.castle_long => {
                const ct: Castle = Castle.long;
                const rook_dest: Square = Castling.rook_dest(us, ct);
                occ &= ~from_bb;
                occ |= Castling.king_dest(us, ct).to_bitboard();
                occ &= ~self.layout.rook_start(us, ct).to_bitboard();
                occ |= rook_dest.to_bitboard();
                return attacks.get_rook_attacks(rook_dest, occ) & king_bb != 0;
            },
            Move.knight_promotion...Move.queen_promotion, Move.knight_promotion_capture...Move.queen_promotion_capture => {
                const pt: PieceType = ex.move.prom();
                occ &= ~from_bb;
                occ |= to_bb;
                const qr: u64 = if (pt.e == .queen or pt.e == .rook) self.queens_rooks(us) | to_bb else self.queens_rooks(us);
                const qb: u64 = if (pt.e == .queen or pt.e == .bishop) self.queens_bishops(us) | to_bb else self.queens_bishops(us);
                return
                    (attacks.get_piece_attacks(to, pt, us, occ) & king_bb) |
                    (attacks.get_rook_attacks(king_sq, occ) & qr) |
                    (attacks.get_bishop_attacks(king_sq, occ) & qb) != 0;
            },
            else => {
                unreachable;
            },
        }
    }

    /// Recompute all hash keys.
    pub fn update_hash(self: *Position) void {
        self.compute_hashkeys(&self.key, &self.pawnkey, &self.nonpawnkeys[0], &self.nonpawnkeys[1], &self.minorkey, &self.majorkey);
    }

    pub fn lazy_update_state(self: *Position) void {
        switch (self.stm.e) {
            .white => self.update_state(.white),
            .black => self.update_state(.black),
        }
    }

    fn update_state(self: *Position, comptime us: Color) void {
        const them: Color = comptime us.opp();
        const bb_all: u64 = self.all();
        const bb_us: u64 = self.by_color(us);
        const bb_them: u64 = self.by_color(them);
        const our_king_sq: Square = self.king_square(us);
        const their_king_sq: Square = self.king_square(them);

        self.pins_orth = @splat(0);
        self.pins_diag = @splat(0);
        self.gen_flags = 0;

        self.checkmask =
            (attacks.get_pawn_attacks(our_king_sq, us) & self.pawns(them)) |
            (attacks.get_knight_attacks(our_king_sq) & self.knights(them));

        // us (checks and pins).
        {
            const bb_occ_without_us: u64 = bb_all ^ self.by_color(us);
            const candidate_slider_attackers: u64 =
                (attacks.get_bishop_attacks(our_king_sq, bb_occ_without_us) & self.queens_bishops(them)) |
                (attacks.get_rook_attacks(our_king_sq, bb_occ_without_us) & self.queens_rooks(them));

            // Our pins and their checks.
            var iter = bitboards.iterator(candidate_slider_attackers);
            while (iter.next()) |attacker_sq| {
                const pair: *const SquarePair = squarepairs.get(our_king_sq, attacker_sq);
                const bb_ray: u64 = pair.ray & bb_us;
                // We have a slider checker when there is nothing in between.
                if (bb_ray == 0) {
                    self.checkmask |= pair.ray;
                }
                // We have a pin when exactly 1 bit is set. There is one piece in between.
                else if (bb_ray & (bb_ray - 1) == 0) {
                    switch (pair.axis) {
                        .orth => self.pins_orth[us.u] |= pair.ray,
                        .diag => self.pins_diag[us.u] |= pair.ray,
                        else => unreachable,
                    }
                    self.gen_flags |= gf_pins; // only for stm!
                }
            }
        }

        // them (pins only).
        {
            const bb_occ_without_them: u64 = bb_all ^ self.by_color(them);
            const candidate_slider_attackers: u64 =
                (attacks.get_bishop_attacks(their_king_sq, bb_occ_without_them) & self.queens_bishops(us)) |
                (attacks.get_rook_attacks(their_king_sq, bb_occ_without_them) & self.queens_rooks(us));

            // Our pins and their checks.
            var iter = bitboards.iterator(candidate_slider_attackers);
            while (iter.next()) |attacker_sq| {
                const pair: *const SquarePair = squarepairs.get(their_king_sq, attacker_sq);
                const bb_ray: u64 = pair.ray & bb_them;
                // We have a pin when exactly 1 bit is set. There is one piece in between.
                if (bb_ray != 0 and bb_ray & (bb_ray - 1) == 0) {
                    switch (pair.axis) {
                        .orth => self.pins_orth[them.u] |= pair.ray,
                        .diag => self.pins_diag[them.u] |= pair.ray,
                        else => unreachable,
                    }
                }
            }
        }

        if (self.checkmask != 0) {
            self.gen_flags |= gf_check;
        }
    }

    /// Returns true if square `to` is attacked by any piece of `attacker`.
    pub fn is_square_attacked_by(self: *const Position, to: Square, comptime attacker: Color) bool {
        // Uses pawn inversion trick.
        const inverted = comptime attacker.opp();
        return
            (attacks.get_knight_attacks(to) & self.knights(attacker)) |
            (attacks.get_king_attacks(to) & self.kings(attacker)) |
            (attacks.get_pawn_attacks(to, inverted) & self.pawns(attacker)) |
            (attacks.get_rook_attacks(to, self.all()) & self.queens_rooks(attacker)) |
            (attacks.get_bishop_attacks(to, self.all()) & self.queens_bishops(attacker)) != 0;
    }

    /// Returns true if square `to` is attacked by any piece of `attacker` for a certain occupation `occ`.
    pub fn is_square_attacked_by_for_occupation(self: *const Position, occ: u64, to: Square, comptime attacker: Color) bool {
        // Uses pawn inversion trick.
        const inverted = comptime attacker.opp();
        return
            (attacks.get_knight_attacks(to) & self.knights(attacker)) |
            (attacks.get_king_attacks(to) & self.kings(attacker)) |
            (attacks.get_pawn_attacks(to, inverted) & self.pawns(attacker)) |
            (attacks.get_rook_attacks(to, occ) & self.queens_rooks(attacker)) |
            (attacks.get_bishop_attacks(to, occ) & self.queens_bishops(attacker)) != 0;
    }

    /// Gives a bitboard of attackers which attack `to` for both colors.
    pub fn get_combined_attackers_to_for_occupation(self: *const Position, occ: u64, to: Square) u64 {
        // Uses pawn inversion trick.
        return
            (attacks.get_knight_attacks(to) & self.all_knights()) |
            (attacks.get_king_attacks(to) & self.all_kings()) |
            (attacks.get_pawn_attacks(to, .black) & self.pawns(.white)) |
            (attacks.get_pawn_attacks(to, .white) & self.pawns(.black)) |
            (attacks.get_rook_attacks(to, occ) & self.all_queens_rooks()) |
            (attacks.get_bishop_attacks(to, occ) & self.all_queens_bishops());
    }

    pub fn attacks_by(self: *const Position, comptime attacker: Color) u64 {
        return self.attacks_by_for_occupation(attacker, self.all());
    }

    pub fn attacks_by_for_occupation(self: *const Position, comptime attacker: Color, occ: u64) u64 {
        var att: u64 = 0;
        var iter: bitboards.BitboardIterator = undefined;

        // Pawns.
        const their_pawns = self.pawns(attacker);
        if (their_pawns > 0) {
            att |= (pawns_shift(their_pawns, attacker, .northeast) | pawns_shift(their_pawns, attacker, .northwest));
        }

        // Knights.
        iter = .init(self.knights(attacker));
        while (iter.next()) |from| {
            att |= attacks.get_knight_attacks(from);
        }

        // Diagonal sliders.
        iter = .init(self.queens_bishops(attacker));
        while (iter.next()) |from| {
            att |= attacks.get_bishop_attacks(from, occ);
        }

        // Orthogonal sliders.
        iter = .init(self.queens_rooks(attacker));
        while (iter.next()) |from| {
            att |= attacks.get_rook_attacks(from, occ);
        }

        // King.
        att |= attacks.get_king_attacks(self.king_square(attacker));

        return att;
    }

    /// Paranoid only.
    pub fn assert_pos_ok(self: *const Position, ex: ExtMove) void {
        lib.not_in_release();

        // Check counts.
        for (Color.all) |color| {
            for (PieceType.all) |piecetype| {
                const cnt: u8 = popcnt(self.bitboards_by_color[color.u] & self.bitboards_by_type[piecetype.u]);
                lib.verify(self.material.counts[color.u][piecetype.u] == cnt, "pos count mismatch", .{});
            }
        }

        if (popcnt(self.kings(.white)) != 1) {
            lib.wtf("pos white king error", .{});
        }

        if (popcnt(self.kings(.black)) != 1) {
            lib.wtf("pos black king error", .{});
            return false;
        }

        if (popcnt(self.all()) > 32) {
            lib.wtf("pos too many pieces", .{});
        }

        if (self.castlingrights & cf_white_short != 0) {
            assert(self.get(self.layout.king_start(.white)).e == .white_king );
            assert(self.get(self.layout.rook_start(.white, .short)).e == .white_rook );
        }

        if (self.castlingrights & cf_white_long != 0) {
            assert(self.get(self.layout.king_start(.white)).e == .white_king );
            assert(self.get(self.layout.rook_start(.white, .long)).e == .white_rook );
        }

        if (self.castlingrights & cf_black_short != 0) {
            assert(self.get(self.layout.king_start(.black)).e == .black_king );
            assert(self.get(self.layout.rook_start(.black, .short)).e == .black_rook );
        }

        if (self.castlingrights & cf_black_long != 0) {
            assert(self.get(self.layout.king_start(.black)).e == .black_king );
            assert(self.get(self.layout.rook_start(.black, .long)).e == .black_rook );
        }

        if (self.castlingrights == 0 and self.ply_from_root == 0) {
             assert(self.layout == empty_layout);
        }

        assert((self.gen_flags & gf_check == 0) == (self.checkmask == 0));
        assert((self.gen_flags & gf_pins == 0) == (self.pins(self.stm) == 0));

        var key: u64 = undefined;
        var pawnkey: u64 = undefined;
        var white_nonpawnkey: u64 = undefined;
        var black_nonpawnkey: u64 = undefined;
        var minorkey: u64 = undefined;
        var majorkey: u64 = undefined;

        self.compute_hashkeys(&key, &pawnkey, &white_nonpawnkey, &black_nonpawnkey, &minorkey, &majorkey);
        if (key != self.key) {
            lib.wtf("pos key", .{});
        }
        if (pawnkey != self.pawnkey) {
            lib.wtf("pos pawnkey", .{});
        }
        if (white_nonpawnkey != self.nonpawnkeys[0]) {
            lib.wtf("pos white nonpawnkey", .{});
        }
        if (black_nonpawnkey != self.nonpawnkeys[1]) {
            lib.wtf("pos black nonpawnkey", .{});
        }
        if (minorkey != self.minorkey) {
            self.draw();
            lib.wtf("pos minorkey {t} {t}", .{ ex.move.from.e, ex.move.to.e });
        }
        if (majorkey != self.majorkey) {
            self.draw();
            lib.wtf("pos majorkey {t} {t}", .{ ex.move.from.e, ex.move.to.e });
        }

        // In check and not to move.
        const king_sq_white: Square = self.king_square(.white);
        const king_sq_black = self.king_square(.black);

        if (self.is_square_attacked_by(king_sq_white, .black) and self.stm.e != .white) {
            lib.wtf("pos white in check and not to move", .{});
        }

        if (self.is_square_attacked_by(king_sq_black, .white) and self.stm.e != .black) {
            lib.wtf("pos black in check and not to move", .{});
        }
    }

    /// A little validation after uci set.
    pub fn is_valid(self: *const Position) bool {
        if (popcnt(self.by_color(.white)) > 16) return false;
        if (popcnt(self.by_color(.black)) > 16) return false;

        for (Color.all) |c| {
            if (count_of(c, .pawn) > 8) return false;
            if (count_of(c, .knight) > 9) return false;
            if (count_of(c, .bishop) > 9) return false;
            if (count_of(c, .rook) > 9) return false;
            if (count_of(c, .queen) > 9) return false;
            if (count_of(c, .king) != 1) return false;
        }

        // In check and not to move.
        const king_sq_white: Square = self.king_square(.white);
        const king_sq_black = self.king_square(.black);

        if (self.is_square_attacked_by(king_sq_white, .black) and self.stm.e != .white) {
            return false;
        }

        if (self.is_square_attacked_by(king_sq_black, .white) and self.stm.e != .black) {
            return false;
        }

        return true;
    }

    /// Zig-format. Writes the FEN string.
    pub fn format(self: *const Position, writer: *std.io.Writer) std.io.Writer.Error!void {
        // Pieces.
        var rank: u3 = 7;
        while (true) {
            var empty_squares: u4 = 0;
            var file: u3 = 0;
            while (true) {
                const sq: Square = .from_rank_file(rank, file);
                const pc: Piece = self.board[sq.u];
                if (pc.is_empty()) {
                    empty_squares += 1;
                }
                else {
                    if (empty_squares > 0) {
                        try writer.print("{}", .{ empty_squares });
                        empty_squares = 0;
                    }
                    try writer.print("{u}", .{ pc.to_char() });
                }
                if (file == 7) {
                    if (empty_squares > 0) {
                        try writer.print("{}", .{ empty_squares });
                    }
                    if (rank > 0) {
                        try writer.print("/", .{});
                    }
                    break;
                }
                file += 1;
            }
            if (rank == 0) break;
            rank -= 1;
        }

        // Color to move.
        switch (self.stm.e) {
            .white => try writer.print(" w", .{}),
            .black => try writer.print(" b", .{}),
        }

        // Castling rights.
        try writer.print(" ", .{});
        if (self.castlingrights == 0) {
            try writer.print("-", .{});
        }
        else {
            if (!self.is_960) {
                if (self.has_castlingright(.white, .short)) try writer.print("K", .{});
                if (self.has_castlingright(.white, .long))  try writer.print("Q", .{});
                if (self.has_castlingright(.black, .short)) try writer.print("k", .{});
                if (self.has_castlingright(.black, .long))  try writer.print("q", .{});
            }
            else {
                if (self.has_castlingright(.white, .short)) try writer.print("{u}", .{ self.layout.rook_start(.white, .short).char_of_file() - 32 }); // uppercase
                if (self.has_castlingright(.white, .long))  try writer.print("{u}", .{ self.layout.rook_start(.white, .long).char_of_file() - 32 }); // uppercase
                if (self.has_castlingright(.black, .short)) try writer.print("{u}", .{ self.layout.rook_start(.black, .short).char_of_file() });
                if (self.has_castlingright(.black, .long))  try writer.print("{u}", .{ self.layout.rook_start(.black, .long).char_of_file() });
            }
        }

        // Enpassant.
        try writer.print(" ", .{});
        if (self.ep_square.u > 0) {
            try writer.print("{t}", .{ self.ep_square.e });
        }
        else {
            try writer.print("-", .{});
        }

        // Draw counter.
        try writer.print(" {}", .{ self.rule50 });

        // Move number.
        const movenr: u16 = funcs.ply_to_movenumber(self.game_ply, self.stm);
        try writer.print(" {}", .{ movenr });
    }

    /// Prints the position diagram + information.
    pub fn draw(self: *const Position) void {
        // Pieces.
        io.print_buffered("\n", .{});
        for (Square.all) |sq| {
            const square = sq.flipped();
            if (square.coord.file == 0) io.print_buffered("{u}   ", .{square.char_of_rank()});
            const pc: Piece = self.get(square);
            const ch: u8 = if (pc.is_empty()) '.' else pc.to_print_char();
            io.print_buffered("{u} ", .{ch});
            if (square.u % 8 == 7) io.print_buffered("\n", .{});
        }
        io.print_buffered("\n    a b c d e f g h\n\n", .{});

        // Info.
        io.print_buffered("fen: {f}\n", .{ self });
        io.print_buffered("checkers: ", .{});
        if (self.checkmask != 0) {
            const bb = self.checkmask & self.by_color(self.stm.opp());
            var iter = bitboards.iterator(bb);
            while (iter.next()) |sq| {
                io.print_buffered("{t} ", .{sq.e});
            }
        }
        io.print_buffered("\n", .{});
        io.flush();
    }
};

/// Piececounts. King is always 1.
pub const Material = struct {
    counts: [Color.count][PieceType.count]u8,

    pub const empty: Material = .{
        .counts = .{
            .{ 0, 0, 0, 0, 0, 0 },
            .{ 0, 0, 0, 0, 0, 0 },
        },
    };

    pub const default: Material = .{
        .counts = .{
            .{ 8, 2, 2, 2, 1, 1 },
            .{ 8, 2, 2, 2, 1, 1 },
        },
    };

    /// Not used.
    pub fn init(w: u48, b: u48) Material {
        const m: u96 = encode_96(w, b);
        return .{ .counts = @bitCast(m) };
    }

    /// Comptime only to construct endgame consts.
    pub inline fn encode_48(p: u8, n: u8, b: u8, r: u8, q: u8) u48 {
        return @as(u48, p) | @as(u48, n) << 8 | @as(u48, b) << 16 | @as(u48, r) << 24 | @as(u48, q) << 32 | @as(u48, 1) << 40;
    }

    /// Comptime only to construct endgame consts.
    pub inline fn encode_96(w: u48, b: u48) u96 {
        return @as(u96, w) | (@as(u96, b) << 48);
    }

    pub fn decode(self: Material) u96 {
        return @bitCast(self.counts);
    }

    pub fn decode_side(self: Material, us: Color) u48 {
        return @bitCast(self.counts[us.u]);
    }

    pub fn decode_both(self: Material, first: Color, second: Color) u96 {
        return encode_96(self.decode_side(first), self.decode_side(second));
    }
};

/// Hard castling constants.
pub const Castling = struct {
    const dest_squares_king: [Color.count][Castle.count]Square = .{ .{ .g1, .c1 }, .{ .g8, .c8 } };
    const dest_squares_rook: [Color.count][Castle.count]Square = .{ .{ .f1, .d1 }, .{ .f8, .d8 } };
    const flags: [Color.count][Castle.count]u4 = .{ .{ cf_white_short, cf_white_long }, .{ cf_black_short, cf_black_long } };

    pub fn king_dest(us: Color, ct: Castle) Square {
        return dest_squares_king[us.u][ct.u];
    }

    pub fn rook_dest(us: Color, ct: Castle) Square {
        return dest_squares_rook[us.u][ct.u];
    }

    pub fn flag(us: Color, ct: Castle) u4 {
        return flags[us.u][ct.u];
    }
};

/// Small key used during setup. King files must always be set.
pub const LayoutKey = packed struct {
    white: StartFiles,
    black: StartFiles,
    rights: u4,

    pub const StartFiles = packed struct {
        king: u3,
        right_rook: u3,
        left_rook: u3,

        pub const classic: StartFiles = .{
            .king = 4, .right_rook = 7, .left_rook = 0,
        };
    };

    pub const empty: LayoutKey = std.mem.zeroes(LayoutKey);
    pub const classic: LayoutKey = .{ .white = .classic, .black = .classic, .rights = cf_all };

    pub fn init_kings(self: *LayoutKey, white_king_sq: Square, black_king_sq: Square) void {
        if (white_king_sq.coord.rank == 0) {
            const file: u3 = white_king_sq.coord.file;
            self.white.king = file;
            self.white.right_rook = file;
            self.white.left_rook = file;
        }
        if (black_king_sq.coord.rank == 7) {
            const file: u3 = black_king_sq.coord.file;
            self.black.king = file;
            self.black.right_rook = file;
            self.black.left_rook = file;
        }
    }

    pub fn set_rook_file(self: *LayoutKey, castling_flag: u4, file: u3) void {
        self.rights |= castling_flag;
        switch (castling_flag) {
            cf_white_short => self.white.right_rook = file,
            cf_white_long => self.white.left_rook = file,
            cf_black_short => self.black.right_rook = file,
            cf_black_long => self.black.left_rook = file,
            else => unreachable,
        }
    }

    fn king_file(self: LayoutKey, us: Color) u3 {
        return if (us.e == .white) self.white.king else self.black.king;
    }

    fn rook_file(self: LayoutKey, us: Color, ct: Castle) u3 {
        switch (us.e) {
            .white => return if (ct.e == .short) self.white.right_rook else self.white.left_rook,
            .black => return if (ct.e == .short) self.black.right_rook else self.black.left_rook,
        }
    }

    fn can_use_classic_layout(self: LayoutKey) bool {
        if (self.rights == 0) return false;
        return
            self == classic or
            blk: {
                const ws: bool = self.is_set(.white, .short);
                const wl: bool = self.is_set(.white, .long);
                const bs: bool = self.is_set(.black, .short);
                const bl: bool = self.is_set(.black, .long);
                break :blk
                    (!ws or (ws and self.white.king == classic.white.king and self.rook_file(.white, .short) == classic.white.right_rook)) and
                    (!wl or (wl and self.white.king == classic.white.king and self.rook_file(.white, .long)  == classic.white.left_rook)) and
                    (!bs or (bs and self.black.king == classic.black.king and self.rook_file(.black, .short) == classic.black.right_rook)) and
                    (!bl or (bl and self.black.king == classic.black.king and self.rook_file(.black, .long)  == classic.black.left_rook));
            };
    }

    pub fn detect_frc(self: LayoutKey) bool {
        return
            (self.is_set(.white, .short) and (self.white.king != classic.white.king or self.white.right_rook != classic.white.right_rook)) or
            (self.is_set(.white, .long)  and (self.white.king != classic.white.king or self.white.left_rook  != classic.white.left_rook)) or
            (self.is_set(.black, .short) and (self.black.king != classic.black.king or self.black.right_rook != classic.black.right_rook)) or
            (self.is_set(.black, .long)  and (self.black.king != classic.black.king or self.black.left_rook  !=  classic.black.left_rook));
    }

    fn is_set(self: LayoutKey, us: Color, ct: Castle) bool {
        return self.rights & Castling.flag(us, ct) != 0;
    }

    fn to_int(self: LayoutKey) u22 {
        return @bitCast(self);
    }
};

/// Intialial castling layout.
pub const Layout = struct {
    king_start_squares: [Color.count]Square,
    rook_start_squares: [Color.count][Castle.count]Square,
    /// Squares that must be empty, when castling.
    empty_paths: [Color.count][Castle.count]u64,
    /// Squares that must be checked against attacks, when castling.
    attack_paths: [Color.count][Castle.count]u64,
    /// Masks for updating castling rights.
    masks: [Square.count]u4,

    const empty: Layout = .{
        .king_start_squares = .{ .zero, .zero },
        .rook_start_squares = .{ .{ .zero, .zero }, .{ .zero, .zero } },
        .empty_paths = .{ .{ 0, 0 }, .{ 0, 0 } },
        .attack_paths = .{ .{ 0, 0 }, .{ 0, 0 } },
        .masks = @splat(0),
    };

    pub fn from_layoutkey(key: LayoutKey) Layout {
        var result: Layout = .empty;

        inline for (Color.all) |us| {
            const rank: u3 = types.relative_rank(us, 0);
            inline for (Castle.all) |ct| {
                const king_file: u3 = key.king_file(us);
                const rook_file: u3 = key.rook_file(us, ct);
                const king_ok: bool = king_file > 0 and king_file < 7;
                const apply: bool = king_ok and (if (ct.e == .short) rook_file > king_file else rook_file < king_file);
                if (apply and key.is_set(us, ct)) { // TODO: still work to do here.
                    const k: Square = .from_rank_file(rank, king_file);
                    const r: Square = .from_rank_file(rank, rook_file);
                    const all_flags: u4 = Castling.flag(us, .short) | Castling.flag(us, .long);
                    const flag: u4 = Castling.flag(us, ct);
                    result.king_start_squares[us.u] = k;
                    result.masks[k.u] |= all_flags;
                    result.masks[r.u] |= flag;
                    result.rook_start_squares[us.u][ct.u] = r;
                    result.empty_paths[us.u][ct.u] =
                        squarepairs.get(r, Castling.rook_dest(us, ct)).ray & ~k.to_bitboard() |
                        squarepairs.get(k, Castling.king_dest(us, ct)).ray & ~r.to_bitboard();
                    result.attack_paths[us.u][ct.u] = squarepairs.get(k, Castling.king_dest(us, ct)).ray;
                }
            }
        }

        return result;
    }

    pub fn king_start(self: *const Layout, us: Color) Square {
        return self.king_start_squares[us.u];
    }

    pub fn rook_start(self: *const Layout, us: Color, ct: Castle) Square {
        return self.rook_start_squares[us.u][ct.u];
    }

    pub fn empty_path(self: *const Layout, us: Color, ct: Castle) u64 {
        return self.empty_paths[us.u][ct.u];
    }

    pub fn attack_path(self: *const Layout, us: Color, ct: Castle) u64 {
        return self.attack_paths[us.u][ct.u];
    }
};

/// Easy game store, using global lib allocator.
pub const Game = struct {
    startpos: Position,
    moves: std.ArrayList(ExtMove),

    pub fn init() Game {
        return .{
            .startpos = .empty,
            .moves = .empty,
        };
    }

    pub fn reset(self: *Game) void {
        self.moves.clearAndFree(ctx.gpa);
        self.startpos = .empty;
    }

    pub fn deinit(self: *Game) void {
        self.moves.deinit(ctx.gpa);
    }

    /// Set position, clear moves.
    pub fn reset_with_position(self: *Game, pos: *const Position) void {
        self.moves.items.len = 0;
        self.startpos = pos.*;
    }

    pub fn append_move(self: *Game, extmove: ExtMove) !void {
        try self.moves.append(ctx.gpa, extmove);
    }
};


pub const Error = error {
    /// There are not exactly 2 kings.
    InvalidKings,
    /// There is a mismatch of castlingrights and board.
    CastlingLogic,
    /// Impossible check.
    InCheckAndNotToMove,
};

pub fn select_layout(key: LayoutKey) *const Layout {
    // Do do not use a hash entry if not nessesary.
    if (key.rights == 0 or key == LayoutKey.empty) {
        return empty_layout;
    }
    //if (key == LayoutKey.classic) { // key.can_use_classic_layout()) {
    if (key.can_use_classic_layout()) {
        return classic_startpos_layout;
    }
    // Otherwise store in map.
    const ptr = layout_map.getOrPut(ctx.gpa, key) catch wtf("layout_map", .{});
    if (!ptr.found_existing) {
        ptr.key_ptr.* = key;
        ptr.value_ptr.* = .from_layoutkey(key);
    }
    return ptr.value_ptr;
}

/// A global hash containing the encountered castling layouts during setup of positions.
/// Supports Chess960 and theoretically also Chess324.
pub var layout_map: std.AutoHashMapUnmanaged(LayoutKey, Layout) = .empty;

// Castling flags.
pub const cf_white_short: u4 = 0b0001;
pub const cf_white_long : u4 = 0b0010;
pub const cf_black_short: u4 = 0b0100;
pub const cf_black_long : u4 = 0b1000;
pub const cf_white_all  : u4 = cf_white_short | cf_white_long;
pub const cf_black_all  : u4 = cf_black_short | cf_black_long;
pub const cf_all        : u4 = cf_white_short | cf_white_long | cf_black_short | cf_black_long;

pub const classic_startpos_fen: []const u8 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// Gen Flags for move generation.
pub const gf_check: u2 = 1 << 0; // state_flags + gen
pub const gf_pins : u2 = 1 << 1; // state_flags + gen
pub const gf_black: u4 = 1 << 2; // stm + gen
pub const gf_noisy: u4 = 1 << 3; // gen

const EMPTY_LAYOUT: Layout = Layout.empty;
const empty_layout: *const Layout = &EMPTY_LAYOUT;

const CLASSIC_STARTPOS_LAYOUT: Layout = .from_layoutkey(LayoutKey.classic);
const classic_startpos_layout: *const Layout = &CLASSIC_STARTPOS_LAYOUT;

/// King buckets are for future.
pub const king_bucket_table: [64]u4 = .{
    0,  0,  0,  1,  1,  2,  2,  2,
    0,  0,  0,  1,  1,  2,  2,  2,
    3,  3,  3,  4,  4,  5,  5,  5,
    3,  3,  3,  4,  4,  5,  5,  5,
    6,  6,  6,  7,  7,  8,  8,  8,
    6,  6,  6,  7,  7,  8,  8,  8,
    9,  9,  9, 10, 10, 11, 11, 11,
    9,  9,  9, 10, 10, 11, 11, 11,
};
