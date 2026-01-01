// zig fmt: off

const std = @import("std");
const types = @import("types.zig");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const zobrist = @import("zobrist.zig");
const attacks = @import("attacks.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;
const popcnt = funcs.popcnt;
const pop_square = funcs.pop_square;
const pawns_shift = funcs.pawns_shift;
const pawn_from = funcs.pawn_from;
const bitloop = funcs.bitloop;

const Value = types.Value;
const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const CastleType = types.CastleType;
const GamePhase = types.GamePhase;

// castling flags.
pub const cf_white_short: u4 = 0b0001;
pub const cf_white_long: u4 = 0b0010;
pub const cf_black_short: u4 = 0b0100;
pub const cf_black_long: u4 = 0b1000;
pub const cf_all: u4 = 0b1111;

// gen flags.
const gf_black: u4 = 0b0001;
const gf_check: u4 = 0b0010;
const gf_noisy: u4 = 0b0100;
const gf_pins : u4 = 0b1000;

/// Must be called when the program stops.
pub fn finalize() void {
    alternative_layout_map.deinit(ctx.galloc);
}

const EMPTY_LAYOUT: Layout = Layout.empty;
const empty_layout: *const Layout = &EMPTY_LAYOUT;

const CLASSIC_LAYOUT: Layout = Layout.init(Square.A1.u, Square.H1.u, Square.E1.u, Square.A8.u, Square.H8.u, Square.E8.u);
const classic_layout: *const Layout = &CLASSIC_LAYOUT;

/// A dictionary with load on demand chess960 positions.
pub var alternative_layout_map: std.AutoHashMapUnmanaged(LayoutKey, Layout) = .empty;

/// Hard constant. Indexing [color][castletype].
pub const king_castle_destination_squares: [2][2]Square = .{ .{ .G1, .C1 }, .{ .G8, .C8 } };

/// Hard constant. Indexing [color][castletype].
pub const rook_castle_destination_squares: [2][2]Square = .{ .{ .F1, .D1 }, .{ .F8, .D8 } };

/// Hard constant. Indexing [color][castletype].
pub const castle_flags: [2][2]u4 = .{ .{ cf_white_short, cf_white_long }, .{ cf_black_short, cf_black_long } };

pub const Position = struct {
    pub const empty: Position = .init_empty();
    pub const empty_classic: Position = .init_empty_classic();

    /// The initial layout, supporting Chess960.
    layout: *const Layout,
    /// The pieces on the 64 squares.
    board: [64]Piece,
    /// Bitboards occupation. Indexing by [piecetype].
    bitboards_by_type: [6]u64,
    /// Bitboards occupation. Indexing by [color].
    bitboards_by_color: [2]u64,
    /// All pieces.
    bitboard_all: u64,
    /// Piece phase. Used by eval.
    phase: u8,
    /// Side to move.
    stm: Color,
    /// Depth during search.
    ply: u16,
    /// Used for repetition detection.
    ply_from_root: u16,
    /// The real game ply. (Fen strings specify movenr).
    game_ply: u16,
    /// State indicating we did a nullmove.
    nullmove_state: bool,
    /// Draw counter. After 50 reversible moves (100 ply) it is a draw.
    rule50: u16,
    /// The enpassant square. A1 means no ep square.
    ep_square: Square,
    /// Bitflags for castlingrights: cf_white_short, cf_white_long, cf_black_short, cf_black_long.
    castling_rights: u4,
    /// The position hashkey.
    key: u64,
    /// The hashkey of pawns. Used for correction history.
    pawnkey: u64,
    /// Key for each side for non pawns.  Used for correction history.
    nonpawnkeys: [2]u64,
    /// Bitboard of the pieces that currently give check. TODO: we could contemplate delete this and use masking with checkmask.
    checkers: u64,
    /// The paths from the enemy slider checkers to the king (excluding the king, including the checker). Pawns and knights also included.
    checkmask: u64,
    /// Bitboards with the diagonal pin rays (excluding the king, including the attacker). Indexing by [color].
    pins_diagonal: [2]u64,
    /// Bitboards with the orthogonal pin rays (excluding the king, including the attacker). Indexing by [color].
    pins_orthogonal: [2]u64,
    /// Indicates chess 960.
    is_960: bool,

    fn init_empty() Position {
        return .{
            .layout = classic_layout,
            .board = @splat(Piece.NO_PIECE),
            .bitboards_by_type = @splat(0),
            .bitboards_by_color = @splat(0),
            .bitboard_all = 0,
            .phase = 0,
            .stm = Color.WHITE,
            .ply = 0,
            .ply_from_root = 0,
            .game_ply = 0,
            .nullmove_state = false,
            .rule50 = 0,
            .ep_square = Square.zero,
            .castling_rights = 0,
            .key = 0,
            .pawnkey = 0,
            .nonpawnkeys = @splat(0),
            .checkers = 0,
            .checkmask = 0,
            .pins_diagonal = @splat(0),
            .pins_orthogonal = @splat(0),
            .is_960 = false,
        };
    }

    fn init_empty_classic() Position {
        const b = bitboards;
        return .{
            .layout = classic_layout,
            .board = create_board_from_backrow(.{ PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK }),
            .bitboards_by_type = .{
                b.bb_rank_2 | b.bb_rank_7,  // pawns
                b.bb_b1 | b.bb_g1 | b.bb_b8 | b.bb_g8, // knights
                b.bb_c1 | b.bb_f1 | b.bb_c8 | b.bb_f8, // bishops
                b.bb_a1 | b.bb_h1 | b.bb_a8 | b.bb_h8, // rooks
                b.bb_d1 | b.bb_d8, // queens
                b.bb_e1 | b.bb_e8  // kings
            },
            .bitboards_by_color = .{ b.bb_rank_1 | b.bb_rank_2, b.bb_rank_7 | b.bb_rank_8 },
            .bitboard_all = b.bb_rank_1 | b.bb_rank_2 | b.bb_rank_7 | b.bb_rank_8,
            .phase = types.max_phase,
            .stm = Color.WHITE,
            .ply = 0,
            .ply_from_root = 0,
            .game_ply = 0,
            .nullmove_state = false,
            .rule50 = 0,
            .ep_square = Square.zero,
            .castling_rights = 0b1111,
            .key = 0,
            .pawnkey = 0,
            .nonpawnkeys = @splat(0),
            .checkers = 0,
            .checkmask = 0,
            .pins_diagonal = @splat(0),
            .pins_orthogonal = @splat(0),
            .is_960 = false,
        };
    }

    fn create_board_from_backrow(backrow: [8]PieceType) [64]Piece {
        var result: [64]Piece = @splat(Piece.NO_PIECE);
        for (backrow, 0..) |pt, i| {
            const sq: Square = Square.from_usize(i);
            result[sq.u] = Piece.create(pt, Color.WHITE);
            result[sq.u + 8] = Piece.W_PAWN;
            result[sq.u + 48] = Piece.B_PAWN;
            result[sq.u + 56] = Piece.create(pt, Color.BLACK);
        }
        return result;
    }

    /// Initializes the position from a fen string.
    pub fn set(self: *Position, fen_str: []const u8, is_960: bool) !void {
        const state_board: u8 = 0;
        const state_color: u8 = 1;
        const state_castle: u8 = 2;
        const state_ep: u8 = 3;
        const state_draw_count: u8 = 4;
        const state_movenumber: u8 = 5;

        // uci is started with "setoption name UCI_Chess960 value true" before ucinewgame. (or after???)
        self.* = .empty;
        self.is_960 = is_960;

        var parse_state: u8 = state_board;
        var rank: u3 = bitboards.rank_8;
        var file: u3 = bitboards.file_a;
        var tokenizer = std.mem.tokenizeScalar(u8, fen_str, ' ');

        outer: while (tokenizer.next()) |token| : (parse_state += 1) {
            switch (parse_state) {
                state_board => {
                    for (token) |c| {
                        switch (c) {
                            '1'...'8' => {
                                const empty_squares: u3 = @truncate(c - '0');
                                file +|= empty_squares;
                            },
                            '/' => {
                                rank -|= 1;
                                file = bitboards.file_a;
                            },
                            else => {
                                const pc: Piece = try Piece.from_char(c);
                                const sq: Square = Square.from_rank_file(rank, file);
                                self.add_fen_piece(pc, sq);
                                file +|= 1;
                            },
                        }
                    }
                },
                state_color => {
                    if (token[0] == 'b') {
                        self.stm = Color.BLACK;
                        self.key ^= zobrist.btm();
                    }
                },
                state_castle => {
                    for (token) |c| {
                        switch (c) {
                            // Classic.
                            'K' => self.set_castling_right(Color.WHITE, Square.from_rank_file(0, 7)),
                            'Q' => self.set_castling_right(Color.WHITE, Square.from_rank_file(0, 0)),
                            'k' => self.set_castling_right(Color.BLACK, Square.from_rank_file(7, 7)),
                            'q' => self.set_castling_right(Color.BLACK, Square.from_rank_file(7, 0)),
                            'A'...'H' => {
                                const rook_file: u3 = @truncate(c - 'A');
                                self.set_castling_right(Color.WHITE, Square.from_rank_file(0, rook_file));
                            },
                            'a'...'h' => {
                                const rook_file: u3 = @truncate(c - 'a');
                                self.set_castling_right(Color.BLACK, Square.from_rank_file(7, rook_file));
                            },
                            else => {},
                        }
                    }
                    self.key ^= zobrist.castling(self.castling_rights);
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

        if (is_960) {
            self.select_layout();
        }

        self.lazy_update_state();
    }

    /// Assumes king is on the board.
    fn set_castling_right(self: *Position, us: Color, rook_sq: Square) void {
        const king_sq: Square = self.king_square(us);
        if (rook_sq.u > king_sq.u) {
            self.castling_rights |= if (us.e == .white) cf_white_short else cf_black_short;
        }
        else {
            self.castling_rights |= if (us.e == .white) cf_white_long else cf_black_long;
        }
    }

    fn select_layout(self: *Position) void {
        if (self.castling_rights == 0) {
            self.layout = empty_layout;
        }

        const w_king_sq: u6 = self.king_square(Color.WHITE).u;
        const b_king_sq: u6 = self.king_square(Color.BLACK).u;

        var w_left_rook: ?u6 = null;
        var w_right_rook: ?u6 = null;
        var b_left_rook: ?u6 = null;
        var b_right_rook: ?u6 = null;

        // Find right rook white
        if (self.castling_rights & cf_white_short != 0) {
            for (w_king_sq + 1..8) |u| {
                if (self.board[u].e == .w_rook) {
                    w_right_rook = @intCast(u);
                }
            }
        }

        // Find left rook white
        if (self.castling_rights & cf_white_long != 0) {
            for (0..w_king_sq) |u| {
                if (self.board[u].e == .w_rook) {
                    w_left_rook = @intCast(u);
                }
            }
        }

        // Find right rook black
        if (self.castling_rights & cf_black_short != 0) {
            for (b_king_sq + 1..64) |u| {
                if (self.board[u].e == .b_rook) {
                    b_right_rook = @intCast(u);
                }
            }
        }

        // Find left rook black
        if (self.castling_rights & cf_black_long != 0) {
            for (56..b_king_sq) |u| {
                if (self.board[u].e == .b_rook) {
                    b_left_rook = @intCast(u);
                }
            }
        }
        self.layout = get_layout(w_left_rook, w_right_rook, w_king_sq, b_left_rook, b_right_rook, b_king_sq);
    }

    /// Parses a uci-move.
    pub fn parse_move(self: *const Position, str: []const u8) types.ParsingError!Move {
        if (str.len < 4 or str.len > 5) {
            return types.ParsingError.IllegalMove;
        }

        const us = self.stm;
        const from: Square = Square.from_string(str[0..2]);
        var to: Square = Square.from_string(str[2..4]);
        var prom_flags: u4 = 0;

        // No promotion.
        if (str.len == 4) {
            // Castling. We only need to change target square if classic. In that case uci is different from our internal encoding.
            if (!self.is_960) {
                if (from.u == self.layout.king_start_squares[us.u].u and self.board[from.u].e == Piece.create(PieceType.KING, us).e) {
                    if (to.e == Square.G1.e or to.e == Square.G8.e) {
                        to = self.layout.rook_start_squares[us.u][CastleType.SHORT.u]; // King takes rook.
                    }
                    else if (to.e == Square.C1.e or to.e == Square.C8.e) {
                        to = self.layout.rook_start_squares[us.u][CastleType.LONG.u]; // King takes rook.
                    }
                }
            }
        }
        // Promotion.
        else if (str.len == 5) {
            prom_flags = switch (str[4]) {
                'n' => Move.knight_promotion,
                'b' => Move.bishop_promotion,
                'r' => Move.rook_promotion,
                'q' => Move.queen_promotion,
                else => return types.ParsingError.InvalidPromotionChar,
            };
        }

        var finder: MoveFinder = .init(from, to, prom_flags);
        self.lazy_generate_all_moves(&finder);
        if (finder.found()) {
            return finder.move; // return the exact move found.
        }
        return types.ParsingError.IllegalMove;
    }

    /// A convenient (faster) way to set the startposition without the need for a fen string.
    pub fn set_startpos(self: *Position) void {
        self.* = empty_classic;
        self.init_hash();
    }

    /// Flips the board.
    /// TODO: still bugged.
    pub fn flip(self: *Position) void {
        // NOTE: phase does not change with flipping.
        // TODO: does not work for chess960 because of layout.
        const bk: [64]Piece = self.board;
        self.board = @splat(Piece.NO_PIECE);
        var bb = self.all();
        var key: u64 = 0;
        var pawnkey: u64 = 0;
        var white_nonpawnkey: u64 = 0;
        var black_nonpawnkey: u64 = 0;

        while (bitloop(&bb)) |sq| {
            const pc: Piece = bk[sq.u];
            const new_sq: Square = sq.flipped();
            const new_pc: Piece = pc.opp();
            self.board[new_sq.u] = new_pc;
            const hash_delta: u64 = zobrist.piece_square(new_pc, new_sq);
            key ^= hash_delta;
            if (new_pc.is_pawn()) {
                pawnkey ^= hash_delta;
            }
            else {
                if (new_pc.color().e == .white) {
                    white_nonpawnkey ^= hash_delta;
                }
                else  {
                    black_nonpawnkey ^= hash_delta;
                }
            }
        }

        inline for (&self.bitboards_by_type) |*b| {
            b.* = funcs.mirror_vertically(b.*);
        }
        inline for (&self.bitboards_by_color) |*b| {
            b.* = funcs.mirror_vertically(b.*);
        }

        self.bitboard_all = funcs.mirror_vertically(self.bitboard_all);

        self.checkers = funcs.mirror_vertically(self.checkers);
        self.checkmask = funcs.mirror_vertically(self.checkmask);
        self.pins_diagonal = funcs.mirror_vertically(self.pins_diagonal);
        self.pins_orthogonal = funcs.mirror_vertically(self.pins_orthogonal);
        self.pins = funcs.mirror_vertically(self.pins);

        self.stm = self.stm.opp();
        if (self.stm.e == .black) {
            key ^= zobrist.btm();
        }

        self.castling_rights = (self.castling_rights >> 2) | (self.castling_rights << 2);
        key ^= zobrist.castling(self.castling_rights);

        if (self.ep_square.u > 0) {
            self.ep_square = self.ep_square.flipped();
            key ^= zobrist.enpassant(self.ep_square);
        }

        self.key = key;
        self.pawnkey = pawnkey;
        self.nonpawnkeys = .{ white_nonpawnkey, black_nonpawnkey };

        //self.lazy_update_state();

        if (comptime lib.is_paranoid) assert(self.pos_ok());
    }

    fn init_hash(self: *Position) void {
        self.compute_hashkeys(&self.key, &self.pawnkey, &self.nonpawnkeys[0], &self.nonpawnkeys[1]);
    }

    fn compute_hashkeys(self: *const Position, key: *u64, pawnkey: *u64, white_nonpawnkey: *u64, black_nonpawnkey: *u64)void {
        key.* = 0;
        pawnkey.* = 0;
        white_nonpawnkey.* = 0;
        black_nonpawnkey.* = 0;
        // Loop through occupied squares.
        var occ: u64 = self.all();
        while (bitloop(&occ)) |sq| {
            const pc = self.get(sq);
            const z_key: u64 = zobrist.piece_square(pc, sq);
            key.* ^= z_key;
            if (pc.is_pawn()) {
                pawnkey.* ^= z_key;
            }
            else {
                if (pc.color().e == .white)
                    white_nonpawnkey.* ^= z_key
                else
                    black_nonpawnkey.* ^= z_key;
            }
        }
        key.* ^= zobrist.castling(self.castling_rights);
        if (self.ep_square.u > 0) key.* ^= zobrist.enpassant(self.ep_square);
        if (self.stm.e == .black) key.* ^= zobrist.btm();
    }

    pub fn get(self: *const Position, sq: Square) Piece {
        return self.board[sq.u];
    }

    /// Returns bitboard of 1 colored piece.
    pub fn pieces(self: *const Position, pt: PieceType, us: Color) u64 {
        return self.bitboards_by_type[pt.u] & self.bitboards_by_color[us.u];
    }

    pub fn by_type(self: *const Position, pt: PieceType) u64 {
        return self.bitboards_by_type[pt.u];
    }

    pub fn by_color(self: *const Position, us: Color) u64 {
        return self.bitboards_by_color[us.u];
    }

    pub fn all(self: *const Position) u64 {
        return self.bitboard_all;
    }

    pub fn all_pawns(self: *const Position) u64 {
        return self.by_type(PieceType.PAWN);
    }

    pub fn all_knights(self: *const Position) u64 {
        return self.by_type(PieceType.KNIGHT);
    }

    pub fn all_bishops(self: *const Position) u64 {
        return self.by_type(PieceType.BISHOP);
    }

    pub fn all_rooks(self: *const Position) u64 {
        return self.by_type(PieceType.ROOK);
    }

    pub fn all_queens(self: *const Position) u64 {
        return self.by_type(PieceType.QUEEN);
    }

    pub fn all_kings(self: *const Position) u64 {
        return self.by_type(PieceType.KING);
    }

    pub fn all_queens_bishops(self: *const Position) u64 {
        return (self.by_type(PieceType.BISHOP) | self.by_type(PieceType.QUEEN));
    }

    pub fn all_queens_rooks(self: *const Position) u64 {
        return (self.by_type(PieceType.ROOK) | self.by_type(PieceType.QUEEN));
    }

    pub fn pieces_except_pawns_and_kings(self: *const Position, us: Color) u64 {
        return self.by_color(us) & ~self.by_type(PieceType.PAWN) & ~self.all_kings();
    }

    pub fn pawns(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.PAWN) & self.by_color(us);
    }

    pub fn knights(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.KNIGHT) & self.by_color(us);
    }

    pub fn bishops(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.BISHOP) & self.by_color(us);
    }

    pub fn rooks(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.ROOK) & self.by_color(us);
    }

    pub fn queens(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.QUEEN) & self.by_color(us);
    }

    pub fn kings(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.KING) & self.by_color(us);
    }

    pub fn queens_bishops(self: *const Position, us: Color) u64 {
        return (self.by_type(PieceType.BISHOP) | self.by_type(PieceType.QUEEN)) & self.by_color(us);
    }

    pub fn queens_rooks(self: *const Position, us: Color) u64 {
        return (self.by_type(PieceType.ROOK) | self.by_type(PieceType.QUEEN)) & self.by_color(us);
    }

    pub fn king_square(self: *const Position, us: Color) Square {
        return funcs.first_square(self.kings(us));
    }

    pub fn sliders(self: *const Position, us: Color) u64 {
        return (self.by_type(PieceType.BISHOP) | self.by_type(PieceType.ROOK) | self.by_type(PieceType.QUEEN)) & self.by_color(us);
    }

    pub fn is_check(self: *const Position) bool {
        return self.state.checkers > 0;
    }

    /// Returns a bitboard of our pinned pieces.
    pub fn our_pins(self: *const Position, us: Color) u64 {
        return self.pins_diagonal[us.u] | self.pins_orthogonal[us.u];
    }

    /// Returns a bitboard of all pinned pieces.
    pub fn all_pins(self: *const Position) u64 {
        return self.pins_diagonal[0] | self.pins_orthogonal[0] | self.pins_diagonal[1] | self.pins_orthogonal[1];
    }

    /// TODO: strange case: many 1 colored bishops against 1 king is draw.
    pub fn is_draw_by_insufficient_material(self: *const Position) bool {
        const us: Color = Color.WHITE;//self.stm;
        const them: Color = Color.BLACK; //us.opp();

        // Queens rooks or pawns: no draw
        if (self.all_queens_rooks() | self.all_pawns() != 0) {
            return false;
        }

        // Only kings: draw
        if (self.all_kings() == self.all()) {
            return true;
        }

        const our_knights: u64 = self.knights(us);
        const their_knights: u64 = self.knights(them);

        const our_bishops: u64 = self.bishops(us);
        const their_bishops: u64 = self.bishops(them);

        const their_minors: u64 = their_knights | their_bishops;
        const our_minors: u64 = our_knights | our_bishops;

        // More than 1 minor piece on either side: no draw
        if (popcnt(our_minors) > 1 or popcnt(their_minors) > 1) {
            return false;
        }

        // Only king on one side and 1 minor piece on the other: draw
        if (
            (their_minors != 0 and self.by_color(us) & ~self.kings(us) == 0) or
            (our_minors != 0 and self.by_color(them) & ~self.kings(them) == 0)
        ) {
            return true;
        }

        // King + bishop vs king + bishop on same color: draw
        if (popcnt(our_bishops) == 1 and popcnt(their_bishops) == 1) {
            const same_color = ((our_bishops & bitboards.bb_black_squares) == 0) == ((their_bishops & bitboards.bb_black_squares) == 0);
            if (same_color) return true;
        }

        return false;
    }

    pub fn is_castling_allowed(self: *const Position, comptime us: Color, comptime castletype: CastleType) bool {
        const flag = comptime castle_flags[us.u][castletype.u];
        return self.castling_rights & flag != 0;
    }

    /// Only used for initializing a fen.
    fn is_usable_ep_square(self: *const Position, ep: Square) bool {
        const rank: u3 = ep.rank();
        if (rank == bitboards.rank_3) {
            const w_pawn_sq = ep.add(8);
            const requirements: bool = self.board[w_pawn_sq.u].e == .w_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u - 8].e == .no_piece;
            return requirements and (bitboards.ep_masks[w_pawn_sq.u] & self.pawns(Color.BLACK) != 0);
        }
        else if (rank == bitboards.rank_6) {
            const b_pawn_sq = ep.sub(8);
            const requirements:  bool = self.board[b_pawn_sq.u].e == .b_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u + 8].e == .no_piece;
            return requirements and (bitboards.ep_masks[b_pawn_sq.u] & self.pawns(Color.WHITE) != 0);
        }
        return false;
    }

    /// Update board, bitboards, values and keys.
    fn add_fen_piece(self: *Position, pc: Piece, sq: Square) void {
        switch (pc.color().e) {
            .white => self.add_piece(Color.WHITE, pc, sq),
            .black => self.add_piece(Color.BLACK, pc, sq),
        }
        const k: u64 = zobrist.piece_square(pc, sq);
        self.key ^= k;
        if (pc.is_pawn()) {
            self.pawnkey ^= k;
        }
        else {
            if (pc.is_white()) {
                self.nonpawnkeys[0] ^= k;
            }
            else {
                self.nonpawnkeys[1] ^= k;
            }
        }
    }

    /// Update board, bitboards, values, phase. Not keys.
    fn add_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        if (comptime lib.is_paranoid) {
            assert(self.get(sq).is_empty());
            assert(pc.is_piece());
            assert(pc.color().e == us.e);
        }
        const mask: u64 = sq.to_bitboard();
        self.board[sq.u] = pc;
        if (us.e == .white) self.bitboards_by_type[pc.u] |= mask else self.bitboards_by_type[pc.u - 6] |= mask;
        self.bitboards_by_color[us.u] |= mask;
        self.bitboard_all |= mask;
        self.phase += types.phase_table[pc.u];
    }

    /// Update board, bitboards, values, phase. Not keys.
    fn remove_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        if (comptime lib.is_paranoid) {
            assert(self.get(sq).e == pc.e);
            assert(pc.color().e == us.e);
        }
        const not_mask: u64 = ~sq.to_bitboard();
        self.board[sq.u] = Piece.NO_PIECE;
        if (us.e == .white) self.bitboards_by_type[pc.u] &= not_mask else self.bitboards_by_type[pc.u - 6] &= not_mask;
        self.bitboards_by_color[us.u] &= not_mask;
        self.bitboard_all &= not_mask;
        self.phase -= types.phase_table[pc.u];
    }

    /// Update board, bitboards, values, phase. Not keys.
    fn move_piece(self: *Position, comptime us: Color, pc: Piece, from: Square, to: Square) void {
        if (comptime lib.is_paranoid) {
            assert(self.get(from).e == pc.e);
            assert(self.get(to).is_empty());
            assert(pc.color().e == us.e);
        }
        const xor_mask: u64 = from.to_bitboard() | to.to_bitboard();
        self.board[from.u] = Piece.NO_PIECE;
        self.board[to.u] = pc;
        if (us.e == .white) self.bitboards_by_type[pc.u] ^= xor_mask else self.bitboards_by_type[pc.u - 6] ^= xor_mask;
        self.bitboards_by_color[us.u] ^= xor_mask;
        self.bitboard_all ^= xor_mask;
    }

    /// Makes the move on the board.`us` is comptime for performance reasons and must be the `stm`.
    pub fn do_move(self: *Position, comptime us: Color, m: Move) void {
        if (comptime lib.is_paranoid) {
            assert(us.e == self.stm.e);
            assert(self.pos_ok());
        }

        const them: Color = comptime us.opp();
        const from: Square = m.from;
        const to: Square = m.to;
        const pc: Piece = self.board[from.u];
        const key_delta = zobrist.piece_square(pc, from) ^ zobrist.piece_square(pc, to);

        // Local keys for updating.
        // Clear ep by default. Note that the zobrist for square a1 is 0 so this xor is safe.
        var key: u64 = self.key ^ zobrist.btm() ^ zobrist.enpassant(self.ep_square);
        var pawnkey: u64 = self.pawnkey;
        var nonpawnkeys: [2]u64 = self.nonpawnkeys;

        // Update some stuff.
        self.stm = them;
        self.ply += 1;
        self.ply_from_root += 1;
        self.game_ply += 1;
        self.nullmove_state = false;
        self.ep_square = Square.zero;

        // Update the castling rights.
        if (self.castling_rights != 0) {
            const mask: u4 = self.layout.castling_masks[from.u] | self.layout.castling_masks[to.u];
            if (mask != 0) {
                key ^= zobrist.castling(self.castling_rights);
                self.castling_rights &= ~mask;
                key ^= zobrist.castling(self.castling_rights);
            }
        }

        // Switch is in numerical order.
        sw: switch (m.flags) {
            Move.silent => {
                self.move_piece(us, pc, from, to);
                key ^= key_delta;
                if (pc.is_pawn_of_color(us)) {
                    self.rule50 = 0;
                    pawnkey ^= key_delta;
                }
                else {
                    self.rule50 += 1;
                    nonpawnkeys[us.u] ^= key_delta;
                }
            },
            Move.double_push => {
                self.rule50 = 0;
                self.move_piece(us, pc, from, to);
                key ^= key_delta;
                pawnkey ^= key_delta;
                // Only set ep if usable.
                if (bitboards.ep_masks[to.u] & self.pawns(them) != 0) {
                    const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                    self.ep_square = ep;
                    key ^= zobrist.enpassant(ep);
                }
            },
            Move.castle_short => {
                self.rule50 += 1;
                const king: Piece = comptime Piece.create(PieceType.KING, us);
                const rook: Piece = comptime Piece.create(PieceType.ROOK, us);
                const king_to: Square = comptime king_castle_destination_squares[us.u][CastleType.SHORT.u];
                const rook_to: Square = comptime rook_castle_destination_squares[us.u][CastleType.SHORT.u]; // king takes rook
                if (self.layout != classic_layout) {
                    self.remove_piece(us, rook, to);
                    self.remove_piece(us, king, from);
                    self.add_piece(us, king, king_to);
                    self.add_piece(us, rook, rook_to);
                }
                else {
                    self.move_piece(us, king, from, king_to);
                    self.move_piece(us, rook, to, rook_to);
                }
                const castle_delta: u64 = zobrist.piece_square(king, from) ^ zobrist.piece_square(king, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
                key ^= castle_delta;
                nonpawnkeys[us.u] ^= castle_delta;
            },
            Move.castle_long => {
                self.rule50 += 1;
                const king: Piece = comptime Piece.create(PieceType.KING, us);
                const rook: Piece = comptime Piece.create(PieceType.ROOK, us);
                const king_to: Square = comptime king_castle_destination_squares[us.u][CastleType.LONG.u];
                const rook_to: Square = comptime rook_castle_destination_squares[us.u][CastleType.LONG.u]; // king takes rook
                if (self.layout != classic_layout) {
                    self.remove_piece(us, rook, to);
                    self.remove_piece(us, king, from);
                    self.add_piece(us, king, king_to);
                    self.add_piece(us, rook, rook_to);
                }
                else {
                    self.move_piece(us, king, from, king_to);
                    self.move_piece(us, rook, to, rook_to);
                }
                const castle_delta: u64 = zobrist.piece_square(king, from) ^ zobrist.piece_square(king, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
                key ^= castle_delta;
                nonpawnkeys[us.u] ^= castle_delta;
            },
            Move.knight_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.create(PieceType.KNIGHT, us);
                const pawn: Piece = comptime Piece.create(PieceType.PAWN, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                pawnkey ^= zobrist.piece_square(pawn, from);
                nonpawnkeys[us.u] ^= zobrist.piece_square(prom, to);
            },
            Move.bishop_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.create(PieceType.BISHOP, us);
                const pawn: Piece = comptime Piece.create(PieceType.PAWN, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                pawnkey ^= zobrist.piece_square(pawn, from);
                nonpawnkeys[us.u] ^= zobrist.piece_square(prom, to);
            },
            Move.rook_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.create(PieceType.ROOK, us);
                const pawn: Piece = comptime Piece.create(PieceType.PAWN, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                pawnkey ^= zobrist.piece_square(pawn, from);
                nonpawnkeys[us.u] ^= zobrist.piece_square(prom, to);
            },
            Move.queen_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.create(PieceType.QUEEN, us);
                const pawn: Piece = comptime Piece.create(PieceType.PAWN, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                pawnkey ^= zobrist.piece_square(pawn, from);
                nonpawnkeys[us.u] ^= zobrist.piece_square(prom, to);
            },
            Move.capture => {
                self.rule50 = 0;
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                self.move_piece(us, pc, from, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);

                // Moved piece key
                key ^= capt_delta ^ key_delta;
                if (pc.is_pawn_of_color(us)) {
                    pawnkey ^= key_delta;
                }
                else {
                    nonpawnkeys[us.u] ^= key_delta;
                }

                // Captured piece key.
                if (capt.is_pawn_of_color(them)) {
                    pawnkey ^= capt_delta;
                }
                else {
                    nonpawnkeys[them.u] ^= capt_delta;
                }
            },
            Move.ep => {
                self.rule50 = 0;
                const pawn_us: Piece = comptime Piece.create(PieceType.PAWN, us);
                const pawn_them: Piece = comptime Piece.create(PieceType.PAWN, them);
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.remove_piece(them, pawn_them, capt_sq);
                self.move_piece(us, pawn_us, from, to);
                key ^= key_delta ^ zobrist.piece_square(pawn_them, capt_sq);
                pawnkey ^= zobrist.piece_square(pawn_us, from) ^ zobrist.piece_square(pawn_us, to) ^ zobrist.piece_square(pawn_them, capt_sq);
            },
            Move.knight_promotion_capture => {
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                nonpawnkeys[them.u] ^= capt_delta;
                continue :sw Move.knight_promotion;
            },
            Move.bishop_promotion_capture => {
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                nonpawnkeys[them.u] ^= capt_delta;
                continue :sw Move.bishop_promotion;
            },
            Move.rook_promotion_capture => {
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                nonpawnkeys[them.u] ^= capt_delta;
                continue :sw Move.rook_promotion;
            },
            Move.queen_promotion_capture => {
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                nonpawnkeys[them.u] ^= capt_delta;
                continue :sw Move.queen_promotion;
            },
            else => {
                unreachable;
            },
        }

        self.key = key;
        self.pawnkey = pawnkey;
        self.nonpawnkeys = nonpawnkeys;

        self.update_state(them);

        if (comptime lib.is_paranoid) {
            assert(self.pos_ok());
        }
    }

    /// Skip a turn.
    pub fn do_nullmove(self: *Position, comptime us: Color) void {
        if (comptime lib.is_paranoid) {
            assert(!self.nullmove_state);
        }
        self.nullmove_state = true;
        const them: Color = comptime us.opp();
        // Clear ep. Note that the ep zobrist for square a1 is 0 so this xor is safe.
        self.key ^= zobrist.btm() ^ zobrist.enpassant(self.ep_square);
        self.stm = them;
        self.ply += 1;
        self.ply_from_root += 1;
        self.game_ply += 1;
        self.ep_square = Square.zero;
        self.update_state(them);
    }

    /// Update checks and pins. `us` must be the side to move.
    fn update_state(self: *Position, comptime us: Color) void {
        const them: Color = comptime us.opp();
        const bb_all: u64 = self.all();
        const bb_us: u64 = self.by_color(us);

        self.pins_orthogonal = @splat(0);
        self.pins_diagonal = @splat(0);

        self.checkmask =
            (attacks.get_pawn_attacks(self.king_square(us), us) & self.pawns(them)) |
            (attacks.get_knight_attacks(self.king_square(us)) & self.knights(them));

        const our_king_sq: Square = self.king_square(us);
        const bb_occ_without_us: u64 = bb_all ^ self.by_color(us);
        var candidate_slider_attackers: u64 =
            (attacks.get_bishop_attacks(our_king_sq, bb_occ_without_us) & self.queens_bishops(them)) |
            (attacks.get_rook_attacks(our_king_sq, bb_occ_without_us) & self.queens_rooks(them));

        // Our pins and their checks.
        while (bitloop(&candidate_slider_attackers)) |attacker_sq| {
            const pair: *const bitboards.SquarePair = bitboards.get_squarepair(our_king_sq, attacker_sq);
            const bb_ray: u64 = pair.ray & bb_us;
            // We have a slider checker when there is nothing in between.
            if (bb_ray == 0) {
                self.checkmask |= pair.ray;
            }
            // We have a pin when exactly 1 bit is set. There is one piece in between.
            else if (bb_ray & (bb_ray - 1) == 0) {
                switch (pair.axis) {
                    .orth => self.pins_orthogonal[us.u] |= pair.ray,
                    .diag => self.pins_diagonal[us.u] |= pair.ray,
                    else => unreachable,
                }
            }
        }
        self.checkers = self.checkmask & bb_all;

        // Their pins.
        const their_king_sq: Square = self.king_square(them);
        const bb_occ_without_them: u64 = bb_all ^ self.by_color(them);
        candidate_slider_attackers =
            (attacks.get_bishop_attacks(their_king_sq, bb_occ_without_them) & self.queens_bishops(us)) |
            (attacks.get_rook_attacks(their_king_sq, bb_occ_without_them) & self.queens_rooks(us));

        while (bitloop(&candidate_slider_attackers)) |attacker_sq| {
            const pair: *const bitboards.SquarePair = bitboards.get_squarepair(their_king_sq, attacker_sq);
            const bb_ray: u64 = pair.ray & bb_us;
            assert(bb_ray != 0); // This should never happen.
            if (bb_ray & (bb_ray - 1) == 0) {
                switch (pair.axis) {
                    .orth => self.pins_orthogonal[them.u] |= pair.ray,
                    .diag => self.pins_diagonal[them.u] |= pair.ray,
                    else => unreachable,
                }
            }
        }
    }

    pub fn lazy_do_move(self: *Position, move: Move) void {
        switch (self.stm.e) {
            .white => self.do_move(Color.WHITE, move),
            .black => self.do_move(Color.BLACK, move),
        }
    }

    pub fn lazy_do_nullmove(self: *Position) void {
        switch (self.stm.e) {
            .white => self.do_nullmove(Color.WHITE),
            .black => self.do_nullmove(Color.BLACK),
        }
    }

    fn lazy_update_state(self: *Position) void {
        switch (self.stm.e) {
            .white => self.update_state(Color.WHITE),
            .black => self.update_state(Color.BLACK),
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
    pub fn get_combined_attacks_to_for_occupation(self: *const Position, occ: u64, to: Square) u64 {
        // Uses pawn inversion trick.
        return
            (attacks.get_knight_attacks(to) & self.all_knights()) |
            (attacks.get_king_attacks(to) & self.all_kings()) |
            (attacks.get_pawn_attacks(to, Color.BLACK) & self.pawns(Color.WHITE)) |
            (attacks.get_pawn_attacks(to, Color.WHITE) & self.pawns(Color.BLACK)) |
            (attacks.get_rook_attacks(to, occ) & self.all_queens_rooks()) |
            (attacks.get_bishop_attacks(to, occ) & self.all_queens_bishops());
    }

    pub fn attacks_by(self: *const Position, comptime attacker: Color) u64 {
        return self.attacks_by_for_occupation(attacker, self.all());
    }

    pub fn attacks_by_for_occupation(self: *const Position, comptime attacker: Color, occ: u64) u64 {
        var att: u64 = 0;

        // Pawns.
        const their_pawns = self.pawns(attacker);
        if (their_pawns > 0) {
            att |= (pawns_shift(their_pawns, attacker, .northeast) | pawns_shift(their_pawns, attacker, .northwest));
        }

        // Knights.
        var their_knights = self.knights(attacker);
        while (bitloop(&their_knights)) |from|{
            att |= attacks.get_knight_attacks(from);
        }

        // Diagonal sliders.
        var their_diag_sliders = self.queens_bishops(attacker);
        while (bitloop(&their_diag_sliders)) |from|{
            att |= attacks.get_bishop_attacks(from, occ);
        }

        // Orthogonal sliders.
        var their_orth_sliders = self.queens_rooks(attacker);
        while (bitloop(&their_orth_sliders)) |from|{
            att |= attacks.get_rook_attacks(from, occ);
        }

        // King.
        att |= attacks.get_king_attacks(self.king_square(attacker));

        return att;
    }

    /// Returns a bitboard with unsafe squares, x-raying the king.
    pub fn get_unsafe_squares_for_king(self: *const Position, comptime us: Color) u64 {
        return self.attacks_by_for_occupation(us.opp(), self.all() ^ self.kings(us));
    }

    fn is_castlingpath_empty(self: *const Position, comptime us: Color, comptime castletype: CastleType, ) bool {
        // Because of Chess960 support we have to check both paths.
        const path: u64 = self.layout.rook_paths[us.u][castletype.u] | self.layout.king_paths[us.u][castletype.u];
        return path & self.all() == 0;
    }

    pub fn lazy_generate_all_moves(self: *const Position, noalias storage: anytype) void {
        switch (self.stm.e) {
            .white => self.generate_all_moves(Color.WHITE, storage),
            .black => self.generate_all_moves(Color.BLACK, storage),
        }
    }

    pub fn lazy_generate_quiescence_moves(self: *const Position, noalias storage: anytype) void {
        switch (self.stm.e) {
            .white => self.generate_quiescence_moves(Color.WHITE, storage),
            .black => self.generate_quiescence_moves(Color.BLACK, storage),
        }
    }

    // Generate all legal moves.
    pub fn generate_all_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void {
        const color_flag: u4 = comptime if (us.e == Color.BLACK.e) gf_black else 0;
        const check: bool = self.checkers != 0;
        const pins: bool = self.our_pins(us) != 0;
        switch (check) {
            false => switch (pins) {
                false => self.gen(color_flag, storage),
                true  => self.gen(color_flag | gf_pins, storage),
            },
            true => switch (pins) {
                false => self.gen(color_flag | gf_check, storage),
                true  => self.gen(color_flag | gf_check | gf_pins, storage),
            },
        }
    }

    /// Generate quiescence moves.
    /// - When not in check: Generate captures and queen promotions.
    /// - When in check: generate all legal moves and queen promotions.
    pub fn generate_quiescence_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void {
        const color_flag: u4 = comptime if (us.e == Color.BLACK.e) gf_black else 0;
        const check: bool = self.checkers != 0;
        const pins: bool = self.our_pins(us) != 0;

        switch (check) {
            false => switch (pins) {
                false => self.gen(color_flag | gf_noisy, storage),
                true  => self.gen(color_flag | gf_noisy | gf_pins, storage),
            },
            true => switch (pins) {
                false => self.gen(color_flag | gf_check | gf_noisy, storage),
                true  => self.gen(color_flag | gf_check | gf_noisy | gf_pins, storage),
            },
        }
    }

    /// See `MoveStorage` for the interface of `storage`: required are the functions `reset()` and `store()`.
    fn gen(self: *const Position, comptime flags: u4, noalias storage: anytype) void {

        // Comptimes
        const us: Color = comptime Color.from_bool(flags & gf_black != 0);
        const them = comptime us.opp();
        const check: bool = comptime flags & gf_check != 0;
        const noisy: bool = comptime flags & gf_noisy != 0;
        const has_pins: bool = comptime flags & gf_pins != 0;
        const do_all_promotions: bool = comptime !noisy;

        if (comptime lib.is_paranoid) {
            assert(self.stm.e == us.e);
            assert((self.checkers == 0 and !check) or (self.checkers != 0 and check));
        }

        // In fact we do not need the reset function, because we always use the storage only once.
        // However: when not using it, move generation is much slower! Some unknown magical compiler optimization?
        storage.reset();

        const doublecheck: bool = check and popcnt(self.checkers) > 1;
        const bb_all: u64 = self.all();
        const bb_us: u64 = self.by_color(us);
        const bb_them: u64 = self.by_color(them);
        const bb_not_us: u64 = ~bb_us;
        const king_sq: Square = self.king_square(us);
        const pins_diagonal: u64 = self.pins_diagonal[us.u];
        const pins_orthogonal: u64 = self.pins_orthogonal[us.u];
        const pins: u64 = pins_diagonal | pins_orthogonal;

        var bb: u64 = undefined;

        // In case of a doublecheck we can only move the king.
        if (!doublecheck) {
            const our_pawns = self.pawns(us);
            const our_knights = self.knights(us);
            const our_queens_bishops = self.queens_bishops(us);
            const our_queens_rooks = self.queens_rooks(us);

            const target = if (check) self.checkmask else if (!noisy) bb_not_us else bb_them;

            // Pawns.
            if (our_pawns != 0) {
                const third_rank: u64 = comptime funcs.relative_rank_3_bitboard(us);
                const last_rank: u64 = comptime funcs.relative_rank_8_bitboard(us);
                const empty_squares: u64 = ~bb_all;
                const enemies: u64 = if (check) self.checkers else bb_them;

                // Generate all 4 types of pawnmoves: push, double push, capture left, capture right.
                var bb_single = switch(has_pins) {
                    false => (pawns_shift(our_pawns, us, .up) & empty_squares),
                    true  => (pawns_shift(our_pawns & ~pins, us, .up) & empty_squares) |
                             (pawns_shift(our_pawns & pins_diagonal, us, .up) & empty_squares & pins_diagonal) |
                             (pawns_shift(our_pawns & pins_orthogonal, us, .up) & empty_squares & pins_orthogonal)
                };

                var bb_double = pawns_shift(bb_single & third_rank, us, .up) & empty_squares;

                // Pawn push check interpolation.
                bb_single &= target;
                bb_double &= target;

                const bb_northwest: u64 = switch (has_pins) {
                    false => (pawns_shift(our_pawns, us, .northwest) & enemies),
                    true  => (pawns_shift(our_pawns & ~pins, us, .northwest) & enemies) |
                             (pawns_shift(our_pawns & pins_diagonal, us, .northwest) & enemies & pins_diagonal),
                };

                const bb_northeast: u64 = switch (has_pins) {
                    false => (pawns_shift(our_pawns, us, .northeast) & enemies),
                    true  => (pawns_shift(our_pawns & ~pins, us, .northeast) & enemies) |
                             (pawns_shift(our_pawns & pins_diagonal, us, .northeast) & enemies & pins_diagonal),
                };

                // Pawn pushes.
                if (check or !noisy) {
                    // Single push normal.
                    bb = bb_single & ~last_rank;
                    while (bitloop(&bb)) |to| {
                        store(pawn_from(to, us, .up), to, Move.silent, storage) orelse return;
                    }
                    // Double push.
                    bb = bb_double;
                    while (bitloop(&bb)) |to| {
                        store(if (us.e == .white) to.sub(16) else to.add(16), to, Move.double_push, storage) orelse return;
                    }
                }

                // left capture promotions.
                bb = bb_northwest & last_rank;
                while (bitloop(&bb)) |to| {
                    store_promotions(do_all_promotions, pawn_from(to, us, .northwest), to, true, storage) orelse return;
                }

                // right capture promotions.
                bb = bb_northeast & last_rank;
                while (bitloop(&bb)) |to| {
                    store_promotions(do_all_promotions, pawn_from(to, us, .northeast), to, true, storage) orelse return;
                }

                // push promotions.
                bb = bb_single & last_rank;
                while (bitloop(&bb)) |to| {
                    store_promotions(do_all_promotions, pawn_from(to, us, .up), to, false, storage) orelse return;
                }

                // left normal captures.
                bb =  bb_northwest & ~last_rank;
                while (bitloop(&bb)) |to| {
                    store(pawn_from(to, us, .northwest), to, Move.capture, storage) orelse return;
                }

                // right normal captures.
                bb = bb_northeast & ~last_rank;
                while (bitloop(&bb)) |to| {
                    store(pawn_from(to, us, .northeast), to, Move.capture, storage) orelse return;
                }

                const ep: Square = self.ep_square;
                // Enpassant.
                if (ep.u > 0) {
                    bb = attacks.get_pawn_attacks(ep, them) & our_pawns; // inversion trick.
                    while (bitloop(&bb)) |from| {
                        if (self.is_legal_enpassant(us, king_sq, from, ep)) {
                            store(from, ep, Move.ep, storage) orelse return;
                        }
                    }
                }
            } // (pawns)

            // Knights.
            bb = if (!has_pins) our_knights else our_knights & ~pins; // A knight can never escape a pin.
            while (bitloop(&bb)) |from| {
                self.store_many(from, attacks.get_knight_attacks(from) & target, storage) orelse return;
            }

            // Diagonal sliders.
            if (!has_pins) {
                bb = our_queens_bishops;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_bishop_attacks(from, bb_all) & target, storage) orelse return;
                }
            }
            else {
                bb = our_queens_bishops & ~pins;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_bishop_attacks(from, bb_all) & target, storage) orelse return;
                }
                bb = our_queens_bishops & pins_diagonal;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_bishop_attacks(from, bb_all) & target & pins_diagonal, storage) orelse return;
                }
            }

            // Orthogonal sliders.
            if (!has_pins) {
                bb = our_queens_rooks;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_rook_attacks(from, bb_all) & target, storage) orelse return;
                }
            }
            else {
                bb = our_queens_rooks & ~pins;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_rook_attacks(from, bb_all) & target, storage) orelse return;
                }
                bb = our_queens_rooks & pins_orthogonal;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_rook_attacks(from, bb_all) & target & pins_orthogonal, storage) orelse return;
                }
            }
        } // (not doublecheck)

        // King.
        const king_target = if (check or !noisy) bb_not_us else bb_them;
        bb = attacks.get_king_attacks(king_sq) & king_target;

        // The king is a troublemaker. For now this 'popcount heuristic' gives the best avg speed, using 2 different approaches to check legality.
        if (popcnt(bb) > 2) {
            const bb_unsafe: u64 = self.get_unsafe_squares_for_king(us);
            bb &= ~bb_unsafe;
            // Normal.
            while (bitloop(&bb)) |to| {
                const flag: u4 = if (self.board[to.u].e == .no_piece) Move.silent else Move.capture;
                store(king_sq, to, flag, storage) orelse return;
            }
            // Castling.
            if (!check and !noisy) {
                inline for (CastleType.all) |ct| {
                    if (self.is_castling_allowed(us, ct) and self.is_castlingpath_empty(us, ct) and self.is_legal_castle(us, ct, bb_unsafe)) {
                        const castle_flag: u4 = comptime if (ct.e == .short) Move.castle_short else Move.castle_long;
                        const to = self.layout.rook_start_squares[us.u][ct.u]; // king takes rook.
                        store(king_sq, to, castle_flag, storage) orelse return;
                    }
                }
            }
        }
        else {
            const bb_without_king: u64 = bb_all ^ self.kings(us);
            // Normal.
            while (bitloop(&bb)) |to| {
                const flag: u4 = if (self.board[to.u].e == .no_piece) Move.silent else Move.capture;
                if (self.is_legal_kingmove(us, bb_without_king, to)) {
                    store(king_sq, to, flag, storage) orelse return;
                }
            }
            // Castling.
            if (!check and !noisy) {
                inline for (CastleType.all) |ct| {
                    if (self.is_castling_allowed(us, ct) and self.is_castlingpath_empty(us, ct) and self.is_legal_castle_check_attacks(us, ct)) {
                        const castle_flag: u4 = comptime if (ct.e == .short) Move.castle_short else Move.castle_long;
                        const to = self.layout.rook_start_squares[us.u][ct.u]; // king takes rook.
                        store(king_sq, to, castle_flag, storage) orelse return;
                    }
                }
            }
        }
    }

    fn store_many(self: *const Position, from: Square, bb_to: u64, noalias storage: anytype) ?void {
        var bb: u64 = bb_to;
        while (bitloop(&bb)) |to| {
            const flag: u4 = if (self.board[to.u].e == .no_piece) Move.silent else Move.capture;
            store(from, to, flag, storage) orelse return null;
        }
    }

    fn store(from: Square, to: Square, flags: u4, noalias storage: anytype) ?void {
        return storage.store(Move.create(from, to, flags));
    }

    fn store_promotions(comptime do_all: bool, from: Square, to: Square, comptime is_capture: bool, noalias storage: anytype) ?void {
        const flags: u4 = if (is_capture) 0b1000 else 0b000;
        storage.store(Move.create(from, to, flags | Move.queen_promotion)) orelse return null;
        if (do_all) {
            storage.store(Move.create(from, to, flags | Move.rook_promotion)) orelse return null;
            storage.store(Move.create(from, to, flags | Move.bishop_promotion)) orelse return null;
            storage.store(Move.create(from, to, flags | Move.knight_promotion)) orelse return null; // TODO: maybe default as second option.
        }
    }

    /// Tricky one. An ep move can uncover a check.
    fn is_legal_enpassant(self: *const Position, comptime us: Color, king_sq: Square, from: Square, to: Square) bool {
        const them: Color = comptime us.opp();
        const capt_sq = if (us.e == .white) to.sub(8) else to.add(8);
        const occ: u64 = (self.all() ^ from.to_bitboard() ^ capt_sq.to_bitboard()) | to.to_bitboard();
        const att: u64 =
            (attacks.get_rook_attacks(king_sq, occ) & self.queens_rooks(them)) |
            (attacks.get_bishop_attacks(king_sq, occ) & self.queens_bishops(them));
        return att == 0;
    }

    fn is_legal_kingmove(self: *const Position, comptime us: Color, bb_without_king: u64, to: Square) bool {
        const them = comptime us.opp();
        return !self.is_square_attacked_by_for_occupation(bb_without_king, to, them);
    }

    /// Compares the kings path with unsafe squares.
    fn is_legal_castle(self: *const Position, comptime us: Color, comptime castletype: CastleType, bb_unsafe: u64) bool {
        // Chess960 requires this additional pin check. In classic the rooks cannot be pinned.
        if (self.pins_orthogonal[us.u] & self.layout.rook_start_squares[us.u][castletype.u].to_bitboard() != 0) return false;
        const path: u64 = self.layout.attack_paths[us.u][castletype.u];
        return path & bb_unsafe == 0;
    }

    fn is_legal_castle_check_attacks(self: *const Position, comptime us: Color, comptime castletype: CastleType) bool {
        const them: Color = comptime us.opp();
        // Chess960 requires this additional pin check. In classic the rooks cannot be pinned.
        if (self.pins_orthogonal[us.u] & self.layout.rook_start_squares[us.u][castletype.u].to_bitboard() != 0) return false;
        var path: u64 = self.layout.attack_paths[us.u][castletype.u];
        while (bitloop(&path)) |sq| {
            if (self.is_square_attacked_by(sq, them)) return false;
        }
        return true;
    }

    /// Meant to be a validation after uci position command. Not used yet.
    pub fn is_valid(self: *const Position) bool {
        if (popcnt(self.all() > 32)) {
            return false;
        }

        const wk: u8 = popcnt(self.kings(Color.WHITE));
        const bk: u8 = popcnt(self.kings(Color.BLACK));

        if (wk + bk != 2) return false;

        const wk_sq: Square = self.king_square(Color.WHITE);
        if (self.stm.e == .black and self.is_square_attacked_by(wk_sq, Color.BLACK)) {
            return false;
        }

        const bk_sq: Square = self.king_square(Color.BLACK);
        if (self.stm.e == .white and self.is_square_attacked_by(bk_sq, Color.WHITE)) {
            return false;
        }
        return true;
    }

    /// ### Debug only.
    pub fn pos_ok(self: *const Position) bool {
        lib.not_in_release();

        if (self.phase == 0 and (self.pieces_except_pawns_and_kings(Color.WHITE) | self.pieces_except_pawns_and_kings(Color.BLACK)) != 0) {
            lib.io.debugprint("PHASE ERROR", .{});
            self.draw();
            return false;
        }

        if (popcnt(self.kings(Color.WHITE)) != 1) {
            lib.io.debugprint("WHITE KING ERROR", .{});
            return false;
        }

        if (popcnt(self.kings(Color.BLACK)) != 1) {
            lib.io.debugprint("BLACK KING ERROR", .{});
            return false;
        }

        if (popcnt(self.all()) > 32) {
            lib.io.debugprint("TOO MANY PIECES", .{});
            return false;
        }

        var key: u64 = undefined;
        var pawnkey: u64 = undefined;
        var white_nonpawnkey: u64 = undefined;
        var black_nonpawnkey: u64 = undefined;

        self.compute_hashkeys(&key, &pawnkey, &white_nonpawnkey, &black_nonpawnkey);
        if (key != self.key) {
            lib.io.debugprint("KEY {} <> {}\n", .{ self.key, key });
            self.draw();
            return false;
        }
        if (pawnkey != self.pawnkey) {
            lib.io.debugprint("PAWNKEY {} <> {}\n", .{ self.pawnkey, pawnkey });
            self.draw();
            return false;
        }
        if (white_nonpawnkey != self.nonpawnkeys[0]) {
            lib.io.debugprint("WHITE NONPAWNKEY {} <> {}\n", .{ self.nonpawnkeys[0], white_nonpawnkey });
            self.draw();
            return false;
        }
        if (black_nonpawnkey != self.nonpawnkeys[1]) {
            lib.io.debugprint("BLACK NONPAWNKEY {} <> {}\n", .{ self.nonpawnkeys[1], black_nonpawnkey });
            self.draw();
            return false;
        }

        // In check and not to move.
        const king_sq_white: Square = self.king_square(Color.WHITE);
        if (self.is_square_attacked_by(king_sq_white, Color.BLACK) and self.stm.e != .white) {
            lib.io.debugprint("WHITE IN CHECK AND NOT TO MOVE\n", .{});
            self.draw();
            return false;
        }

        const king_sq_black = self.king_square(Color.BLACK);
        if (self.is_square_attacked_by(king_sq_black, Color.WHITE) and self.stm.e != .black) {
            lib.io.debugprint("BLACK IN CHECK AND NOT TO MOVE\n", .{});
            self.draw();
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
        if (self.stm.e == .white) {
            try writer.print(" w", .{});
        }
        else {
            try writer.print(" b", .{});
        }

        // Castling rights.
        try writer.print(" ", .{});
        if (self.castling_rights == 0) {
            try writer.print("-", .{});
        }
        else {
            if (!self.is_960) {
                if (self.castling_rights & cf_white_short != 0) try writer.print("K", .{});
                if (self.castling_rights & cf_white_long != 0)  try writer.print("Q", .{});
                if (self.castling_rights & cf_black_short != 0) try writer.print("k", .{});
                if (self.castling_rights & cf_black_long != 0)  try writer.print("q", .{});
            }
            else {
                if (self.castling_rights & cf_white_short != 0) try writer.print("{u}", .{ self.layout.rook_start_squares[0][0].char_of_file() - 32 });
                if (self.castling_rights & cf_white_long != 0)  try writer.print("{u}", .{ self.layout.rook_start_squares[0][1].char_of_file() - 32 });
                if (self.castling_rights & cf_black_short != 0) try writer.print("{u}", .{ self.layout.rook_start_squares[1][0].char_of_file() });
                if (self.castling_rights & cf_black_long != 0)  try writer.print("{u}", .{ self.layout.rook_start_squares[1][1].char_of_file() });
            }
        }

        // Enpassant.
        try writer.print(" ", .{});
        if (self.ep_square.u > 0) {
            try writer.print("{t}", .{self.ep_square.e});
        }
        else {
            try writer.print("-", .{});
        }

        // Draw counter.
        try writer.print(" {}", .{self.rule50});

        // Move number.
        const movenr: u16 = funcs.ply_to_movenumber(self.game_ply, self.stm);
        try writer.print(" {}", .{movenr});
    }

    /// Prints the position diagram + information.
    pub fn draw(self: *const Position) void {
        // Pieces.
        io.print_buffered("\n", .{});
        for (Square.all_for_printing) |square| {
            if (square.coord.file == 0) io.print_buffered("{u}   ", .{square.char_of_rank()});
            const pc: Piece = self.get(square);
            const ch: u8 = if (pc.is_empty()) '.' else pc.to_print_char();
            io.print_buffered("{u} ", .{ch});
            if (square.u % 8 == 7) io.print_buffered("\n", .{});
        }
        io.print_buffered("\n    a b c d e f g h\n\n", .{});

        // Info.
        io.print_buffered("fen: {f}\n", .{ self });
        io.print_buffered("key: 0x{x:0>16} pawnkey: 0x{x:0>16} white_nonpawnkey: {x:0>16} black nonpawnkey: {x:0>16}\n", .{ self.key, self.pawnkey, self.nonpawnkeys[0], self.nonpawnkeys[1] });
        io.print_buffered("rule50: {}\n", .{ self.rule50 });
        io.print_buffered("ply: {}\n", .{ self.ply });
        io.print_buffered("phase: {}\n", .{ self.phase });
        io.print_buffered("checkers: ", .{});
        if (self.checkers != 0) {
            var bb: u64 = self.checkers;
            while (bitloop(&bb)) |sq| {
                io.print_buffered("{t} ", .{sq.e});
            }
        }
        io.print_buffered("\n", .{});
        io.flush();
    }

    /// ### Debug only.
    /// * If `check_moves` then we also check if the generated moves are the same.
    pub fn equals(self: *const Position, other: *const Position, comptime check_moves: bool) bool {
        lib.not_in_release();

        // Not directly binary comparable with Zig std.
        inline for (0..64) |i| if (self.board[i].u != other.board[i].u) return false;

        const eql: bool =
            self.layout == other.layout and
            std.meta.eql(self.bitboards_by_type, other.bitboards_by_type) and
            std.meta.eql(self.bitboards_by_color, other.bitboards_by_color) and
            self.bitboard_all == other.bitboard_all and
            self.phase == other.phase and
            self.ply == other.ply and
            self.ply_from_root == other.ply_from_root and
            self.game_ply == other.game_ply and
            self.nullmove_state == other.nullmove_state and
            self.rule50 == other.rule50 and
            self.ep_square.u == other.ep_square.u and
            self.castling_rights == other.castling_rights and
            self.key == other.key and
            self.pawnkey == other.pawnkey and
            self.nonpawnkeys[0] == other.nonpawnkeys[0] and
            self.nonpawnkeys[1] == other.nonpawnkeys[1] and
            self.checkers == other.checkers and
            self.checkmask == other.checkmask and
            self.pins_diagonal == other.pins_diagonal and
            self.pins_orthogonal == other.pins_orthogonal and
            self.pins == other.pins;

        if (!eql) return false;

        if (check_moves) {
            var store1: MoveStorage = .init();
            var store2: MoveStorage = .init();
            self.lazy_generate_all_moves(&store1);
            self.lazy_generate_all_moves(&store2);
            const ok: bool = std.mem.eql(Move, store1.slice(), store2.slice());
            return ok;
        }
        else {
            return true;
        }
    }
};

/// Basic storage of moves.
pub const MoveStorage = struct {
    moves: [types.max_move_count]Move,
    count: u8,

    pub fn init() MoveStorage {
        return .{ .moves = undefined, .count = 0 };
    }

    /// Required function.
    pub fn reset(self: *MoveStorage) void {
        self.count = 0;
        // assert(self.count == 0); // TODO: is there any explanation...
    }

    /// Required function.
    pub fn store(self: *MoveStorage, move: Move) ?void {
        assert(self.count < 224);
        self.moves[self.count] = move;
        self.count += 1;
    }

    pub fn slice(self: *const MoveStorage) []const Move {
        return self.moves[0..self.count];
    }
};

pub const JustCount = struct {
    moves: u8,

    pub fn init() JustCount {
        return .{ .moves = 0 };
    }

    /// Required function.
    pub fn reset(self: *JustCount) void {
        self.moves = 0;
    }

    /// Required function.
    pub fn store(self: *JustCount, _: Move) ?void {
        self.moves += 1;
    }
};

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
    pub fn store(self: *Any, _: Move) ?void {
        self.has_moves = true;
        return null;
    }
};

pub const MoveFinder = struct {
    /// The required from square.
    from: Square,
    /// The required to square.
    to: Square,
    /// The required promotion piece **without** capture flag.
    prom_flags: u4,
    /// The move, if found.
    move: Move,

    pub fn init(from: Square, to: Square, prom_flags: u4) MoveFinder {
        return .{ .from = from, .to = to, .prom_flags = prom_flags, .move = .empty };
    }

    /// Required function.
    pub fn reset(self: *MoveFinder) void {
        self.move = .empty;
    }

    /// Required function.
    pub fn store(self: *MoveFinder, move: Move) ?void {
        if (self.from.u == move.from.u and self.to.u == move.to.u and (self.prom_flags == 0 or self.prom_flags == move.flags & ~Move.capture)) {
            self.move = move;
            return null;
        }
    }

    pub fn found(self: *const MoveFinder) bool {
        return !self.move.is_empty();
    }
};

pub const Error = error {
    MissingKing,
    TooManyKings,
    InCheckAndNotToMove,
};

/// Get the layout for this configuration. Load on demand from the layout_map.
fn get_layout(w_left_rook: ?u6, w_right_rook: ?u6, w_king: ?u6, b_left_rook: ?u6, b_right_rook: ?u6, b_king: ?u6) *const Layout {
    const key: LayoutKey = .{
        .w_left_rook = w_left_rook,
        .w_right_rook = w_right_rook,
        .w_king = w_king,
        .b_left_rook = b_left_rook,
        .b_right_rook = b_right_rook,
        .b_king = b_king,
    };
    const ptr = alternative_layout_map.getOrPut(ctx.galloc, key) catch wtf();
    if (!ptr.found_existing) {
        ptr.key_ptr.* = key;
        ptr.value_ptr.* = Layout.from_key(key);
    }
    return ptr.value_ptr;
}

/// Squares are simple u6 for easy auto hash mapping (lazy me avoiding writing a hash function)
pub const LayoutKey = struct {
    w_left_rook: ?u6,
    w_right_rook: ?u6,
    w_king: ?u6,
    b_left_rook: ?u6,
    b_right_rook: ?u6,
    b_king: ?u6,
};

/// Layout stuff for castling.
pub const Layout = struct {
    /// Indexing: [color][castletype].
    rook_start_squares: [2][2]Square,
    /// Indexing: [color].
    king_start_squares: [2]Square,
    /// Paths without the king. Indexing: [color][castletype].
    rook_paths: [2][2]u64,
    /// Paths without the rook. Indexing: [color][castletype].
    king_paths: [2][2]u64,
    /// Paths of the king without the king we should check for enemy attacks. Indexing: [color][castletype].
    attack_paths: [2][2]u64,
    /// Masks of castling rights. Indexing: [square].
    castling_masks: [64]u4,

    const empty: Layout = .{
        .rook_start_squares = .{ .{ .zero, .zero }, .{ .zero, .zero } },
        .king_start_squares = .{ .zero, .zero },
        .rook_paths = .{ .{ 0, 0 }, .{ 0, 0 } },
        .king_paths = .{ .{ 0, 0 }, .{ 0, 0 } },
        .attack_paths = .{ .{ 0, 0 }, .{ 0, 0 } },
        .castling_masks = @splat(0),
    };

    pub fn init(w_left_rook: ?u6, w_right_rook: ?u6, w_king: ?u6, b_left_rook: ?u6, b_right_rook: ?u6, b_king: ?u6) Layout {
        const key: LayoutKey = .{
            .w_left_rook = w_left_rook,
            .w_right_rook = w_right_rook,
            .w_king = w_king,
            .b_left_rook = b_left_rook,
            .b_right_rook = b_right_rook,
            .b_king = b_king,
        };
        return from_key(key);
    }

    pub fn from_key(key: LayoutKey) Layout {
        var result: Layout = .empty;

        // white
        if (key.w_king) |wk_| {
            const wk: Square = .from(wk_);
            result.king_start_squares[0] = wk;
            result.castling_masks[wk.u] |= cf_white_short | cf_white_long;
            // white short castle
            if (key.w_right_rook) |wr_| {
                const wr: Square = .from(wr_);
                result.castling_masks[wr.u] |= cf_white_short;
                result.rook_start_squares[0][0] = wr;
                result.rook_paths[0][0] = bitboards.get_squarepair(wr, rook_castle_destination_squares[0][0]).ray & ~wk.to_bitboard();
                result.king_paths[0][0] = bitboards.get_squarepair(wk, king_castle_destination_squares[0][0]).ray & ~wr.to_bitboard();
                result.attack_paths[0][0] = bitboards.get_squarepair(wk, king_castle_destination_squares[0][0]).ray;
            }
            // white long castle
            if (key.w_left_rook) |wr_| {
                const wr: Square = .from(wr_);
                result.castling_masks[wr.u] |= cf_white_long;
                result.rook_start_squares[0][1] = wr;
                result.rook_paths[0][1] = bitboards.get_squarepair(wr, rook_castle_destination_squares[0][1]).ray & ~wk.to_bitboard();
                result.king_paths[0][1] = bitboards.get_squarepair(wk, king_castle_destination_squares[0][1]).ray & ~wr.to_bitboard();
                result.attack_paths[0][1] = bitboards.get_squarepair(wk, king_castle_destination_squares[0][1]).ray;
            }
        }

        // black
        if (key.b_king) |bk_| {
            const bk: Square = .from(bk_);
            result.king_start_squares[1] = bk;
            result.castling_masks[bk.u] |= cf_black_short | cf_black_long;
            // black short castle
            if (key.b_right_rook) |br_| {
                const br: Square = .from(br_);
                result.castling_masks[br.u] |= cf_black_short;
                result.rook_start_squares[1][0] = br;
                result.rook_paths[1][0] = bitboards.get_squarepair(br, rook_castle_destination_squares[1][0]).ray & ~bk.to_bitboard();
                result.king_paths[1][0] = bitboards.get_squarepair(bk, king_castle_destination_squares[1][0]).ray & ~br.to_bitboard();
                result.attack_paths[1][0] = bitboards.get_squarepair(bk, king_castle_destination_squares[1][0]).ray;
            }
            // black long castle
            if (key.b_left_rook) |br_| {
                const br: Square = .from(br_);
                result.castling_masks[br.u] |= cf_black_long;
                result.rook_start_squares[1][1] = br;
                result.rook_paths[1][1] = bitboards.get_squarepair(br, rook_castle_destination_squares[1][1]).ray & ~bk.to_bitboard();
                result.king_paths[1][1] = bitboards.get_squarepair(bk, king_castle_destination_squares[1][1]).ray & ~br.to_bitboard();
                result.attack_paths[1][1] = bitboards.get_squarepair(bk, king_castle_destination_squares[1][1]).ray;
            }
        }
        return result;
    }
};
