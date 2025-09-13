// zig fmt: off

const std = @import("std");
const types = @import("types.zig");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const funcs = @import("funcs.zig");
const zobrist = @import("zobrist.zig");
const squarepairs = @import("squarepairs.zig");
const masks = @import("masks.zig");
const data = @import("data.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;
const popcnt = funcs.popcnt;
const pop_square = funcs.pop_square;
const pawns_shift = funcs.pawns_shift;
const pawn_from = funcs.pawn_from;

const Value = types.Value;
const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const MoveType = types.MoveType;
const MoveInfo = types.MoveInfo;
const CastleType = types.CastleType;
const GamePhase = types.GamePhase;

const P = types.P;
const N = types.N;
const B = types.B;
const R = types.R;
const Q = types.Q;
const K = types.K;

pub const fen_classic_startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

// castling flags.
pub const cf_white_short: u4 = 0b0001;
pub const cf_white_long: u4 = 0b0010;
pub const cf_black_short: u4 = 0b0100;
pub const cf_black_long: u4 = 0b1000;
pub const cf_all: u4 = 0b1111;

/// Indexing [color][castletype].
pub const king_castle_destination_squares: [2][2]Square = .{ .{ .G1, .C1 }, .{ .G8, .C8 } };

/// Indexing [color][castletype].
pub const rook_castle_destination_squares: [2][2]Square = .{ .{ .F1, .D1 }, .{ .F8, .D8 } };

/// Indexing [color][castletype].
pub const castle_flags: [2][2]u4 = .{ .{ cf_white_short, cf_white_long }, .{ cf_black_short, cf_black_long } };

pub const StateInfo = struct {

    pub const empty: StateInfo = .{};

    /// (copied) Draw counter. After 50 reversible moves (100 ply) it is a draw.
    rule50: u16 = 0,
    /// (copied) The enpassant square of this state.
    ep_square: Square = Square.zero,
    /// (copied) Bitflags for castlingrights: cf_white_short, cf_white_long, cf_black_short, cf_black_long.
    castling_rights: u4 = 0,
    /// The move that was played to reach the current position.
    last_move: Move = .empty,
    /// The piece that did the last_move.
    moved_piece: Piece = Piece.NO_PIECE,
    /// The piece that was captured with last_move.
    captured_piece: Piece = Piece.NO_PIECE,
    /// The board hashkey.
    key: u64 = 0,
    /// Bitboard of the pieces that currently give check.
    checkers: u64 = 0,
    /// The paths from the enemy slider checkers to the king (excluding the king, including the checker).
    /// * Pawns and knights included.
    checkmask: u64 = 0,
    /// Bitboards with the diagonal pin rays (excluding the king, including the attacker).
    pins_diagonal: [2]u64 = .{ 0, 0 },
    /// Bitboards with the orthogonal pin rays (excluding the king, including the attacker).
    pins_orthogonal: [2]u64 = .{ 0, 0 },
    /// All pins.
    pins: [2]u64 = .{ 0, 0 },
    /// Pointer to the previous state. For the engine a fixed history array is used. For search the chain only exists in the recursive callstack.
    prev: ?*StateInfo = null,
    /// Pointer to the next state.
    next: ?*StateInfo = null,

    pub fn is_castling_allowed(self: *const StateInfo, comptime us: Color, comptime castletype: CastleType) bool {
        const flag = comptime castle_flags[us.u][castletype.u];
        return self.castling_rights & flag != 0;
    }

    /// ### Debug only.
    /// Compares everything except`prev`
    pub fn equals(self: *const StateInfo, other: *const StateInfo) bool {
        lib.not_in_release();
        return
            self.rule50 == other.rule50 and
            self.ep_square.u == other.ep_square.u and
            self.castling_rights == other.castling_rights and
            self.last_move == other.last_move and
            self.moved_piece.u == other.moved_piece.u and
            self.captured_piece.u == other.captured_piece.u and
            self.key == other.key and
            self.checkers == other.checkers and
            self.checkmask == other.checkmask and
            std.meta.eql(self.pins_diagonal, other.pins_diagonal) and
            std.meta.eql(self.pins_orthogonal, other.pins_orthogonal) and
            std.meta.eql(self.pins, other.pins);
    }
};

/// The initial layout for `Position`, supporting Chess960.
/// * Never modified during play.
pub const Layout = struct {
    /// The startfiles of [0] left rook, [1] right rook, [2] king file.
    /// * This field *must* be filled before we can initialize the next fields.
    start_files: [3]u3,
    /// Deduced from start_files [0] white king, [1] black king. Initialized in constructors.
    king_start_squares: [2]Square,
    /// Deduced from start_files. Indexing: [color][castletype].
    rook_start_squares: [2][2]Square,
    /// Deduced from `start_files`. Indexing: [color][castletype].
    castling_between_bitboards: [2][2]u64,
    /// The king 'walk' when castling, deduced from `start_files`. Indexing: [color][castletype]
    /// * The king start-square is *not* included.
    castling_king_paths: [2][2]u64,
    /// Deduced from `start_files`. Indexing: [square]
    /// * Used for quick updates of castling rights during make move.
    castling_masks: [64]u4,
};

pub const Position = struct {
    pub const empty: Position = .init_empty();
    pub const empty_classic: Position = .init_empty_classic();

    /// The initial layout, supporting Chess960.
    layout: Layout,
    /// The pieces on the 64 squares.
    board: [64]Piece,
    /// Bitboards occupation indexed by PieceType. [0] full occupation, [1...6] piecetypes.
    bb_by_type: [7]u64,
    /// Bitboards occupation indexed by color: [0] white pieces, [1] black pieces.
    bb_by_color: [2]u64,
    /// Piece values sum. [0] white, [1] black
    values: [2]Value,
    /// Material values sum. [0] white, [1] black
    materials: [2]Value,
    /// The current side to move.
    to_move: Color,
    /// Depth during search. Must match state chain.
    ply: u16,
    /// The real game ply.
    game_ply: u16,
    /// Is this chess960?
    is_960: bool,
    /// State indicating we did a nullmove somewhere in the search.
    nullmove_state: bool = false,
    /// The current state.
    state: *StateInfo,

    fn init_empty() Position {
        return .{
            .layout = .{
                .start_files = @splat(0),
                .king_start_squares = @splat(Square.zero),
                .rook_start_squares = .{ .{ Square.zero, Square.zero }, .{ Square.zero, Square.zero } },
                .castling_between_bitboards = std.mem.zeroes([2][2]u64),
                .castling_king_paths = std.mem.zeroes([2][2]u64),
                .castling_masks = @splat(0),
            },
            .board = @splat(Piece.NO_PIECE),
            .bb_by_type = @splat(0),
            .bb_by_color = @splat(0),
            .values = @splat(0),
            .materials = @splat(0),
            .to_move = Color.WHITE,
            .ply = 0,
            .game_ply = 0,
            .is_960 = false,
            .nullmove_state = false,
            .state = undefined,
        };
    }

    fn init_empty_classic() Position {
        const b = bitboards;
        const v_sum: Value = (P.value() * 8) + (N.value() * 2) + (B.value() * 2) + (R.value() * 2) + Q.value() + K.value();
        const m_sum: Value = (P.material() * 8) + (N.material() * 2) + (B.material() * 2) + (R.material() * 2) + Q.material() + K.material();

        return .{
            .layout = .{
                .start_files = .{ b.file_a, b.file_h, b.file_e },
                .king_start_squares = .{ Square.E1, Square.E8 },
                .rook_start_squares = .{ .{ .H1, .A1 }, .{ .H8, .A8 } },
                .castling_between_bitboards = .{ .{ b.bb_f1 | b.bb_g1, b.bb_d1 | b.bb_c1 | b.bb_b1 }, .{ b.bb_f8 | b.bb_g8, b.bb_d8 | b.bb_c8 | b.bb_b8 } },
                .castling_king_paths = .{ .{ b.bb_f1 | b.bb_g1, b.bb_d1 | b.bb_c1 }, .{ b.bb_f8 | b.bb_g8, b.bb_d8 | b.bb_c8 } },
                .castling_masks = .{ 2, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 12, 0, 0, 4 },
            },
            .board = create_board_from_backrow(.{ R, N, B, Q, K, B, N, R}),
            .bb_by_type = .{
                b.bb_rank_1 | b.bb_rank_2 | b.bb_rank_7 | b.bb_rank_8, // all pieces
                b.bb_rank_2 | b.bb_rank_7,  // pawns
                b.bb_b1 | b.bb_g1 | b.bb_b8 | b.bb_g8, // knights
                b.bb_c1 | b.bb_f1 | b.bb_c8 | b.bb_f8, // bishops
                b.bb_a1 | b.bb_h1 | b.bb_a8 | b.bb_h8, // rooks
                b.bb_d1 | b.bb_d8, // queens
                b.bb_e1 | b.bb_e8  // kings
            },
            .bb_by_color = .{ b.bb_rank_1 | b.bb_rank_2, b.bb_rank_7 | b.bb_rank_8 },
            .values = .{ v_sum, v_sum },
            .materials = .{ m_sum, m_sum },
            .to_move = Color.WHITE,
            .ply = 0,
            .game_ply = 0,
            .is_960 = false,
            .nullmove_state = false,
            .state = undefined,
        };
    }

    fn create_board_from_backrow(backrow: [8]PieceType) [64]Piece {
        var result: [64]Piece = @splat(Piece.NO_PIECE);
        for (backrow, 0..) |pt, i| {
            const sq: Square = Square.from_usize(i);
            result[sq.u] = Piece.make(pt, Color.WHITE);
            result[sq.u + 8] = Piece.W_PAWN;
            result[sq.u + 48] = Piece.B_PAWN;
            result[sq.u + 56] = Piece.make(pt, Color.BLACK);
        }
        return result;
    }

    /// Clears the whole board, clears `st` and set the `self.state` pointer to `st`.
    fn clear(self: *Position, st: *StateInfo) void {
        st.* = .empty;
        self.* = .empty;
        self.state = st;
    }

    /// Initializes the position from a fen string.
    pub fn set(self: *Position, st: *StateInfo, fen_str: []const u8) !void {
        const state_board: u8 = 0;
        const state_color: u8 = 1;
        const state_castle: u8 = 2;
        const state_ep: u8 = 3;
        const state_draw_count: u8 = 4;
        const state_movenumber: u8 = 5;

        self.clear(st);
        self.layout.start_files = .{ bitboards.file_a, bitboards.file_h, bitboards.file_e };

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
                                file = file +| empty_squares;
                            },
                            '/' => {
                                rank -|= 1;
                                file = bitboards.file_a;
                            },
                            else => {
                                const pc: Piece = try Piece.from_fen_char(c);
                                const sq: Square = Square.from_rank_file(rank, file);
                                self.lazy_add_piece(sq, pc);
                                file +|= 1;
                            },
                        }
                    }
                },
                state_color => {
                    if (token[0] == 'b') self.to_move = Color.BLACK;
                },
                state_castle => {
                    for (token) |c| {
                        switch (c) {
                            'K' => st.castling_rights |= cf_white_short,
                            'Q' => st.castling_rights |= cf_white_long,
                            'k' => st.castling_rights |= cf_black_short,
                            'q' => st.castling_rights |= cf_black_long,
                            else => {},
                        }
                    }
                },
                state_ep => {
                    if (token.len == 2) {
                        const ep: Square = Square.from_string(token);
                        if (self.is_usable_ep_square(ep)) st.ep_square = ep;
                    }
                },
                state_draw_count => {
                    const v: u16 = std.fmt.parseInt(u16, token, 10) catch break :outer;
                    st.rule50 = v;
                },
                state_movenumber => {
                    const v: u16 = std.fmt.parseInt(u16, token, 10) catch break :outer;
                    self.game_ply = funcs.movenumber_to_ply(v, self.to_move);
                },
                else => {
                    break :outer;
                }
            }
        }

        self.init_layout();
        self.init_hash();
        self.lazy_update_state();
    }

    /// Parses a uci-move.
    pub fn parse_move(self: *const Position, str: []const u8) types.ParsingError!Move {
        if (str.len < 4 or str.len > 5) return types.ParsingError.IllegalMove;

        // Exact move is produced here for the move finder.
        const us = self.to_move;
        var m: Move = .empty;
        m.from = Square.from_string(str[0..2]);
        m.to = Square.from_string(str[2..4]);

        // No promotion.
        if (str.len == 4) {
            // Enpassant.
            if (self.board[m.from.u].piecetype.e == .pawn and m.from.file() != m.to.file() and self.board[m.to.u].e == .no_piece) {
                m.type = .enpassant;
            }
            // Castling.
            else if (m.from.u == self.layout.king_start_squares[self.to_move.u].u and self.board[m.from.u].e == Piece.make(K, us).e and (m.to.e == Square.G1.e or m.to.e == Square.G8.e or m.to.e == Square.C1.e or m.to.e == Square.C8.e)) {
                m.type = .castle;
                m.info.castletype = if (m.to.u > m.from.u) CastleType.SHORT else CastleType.LONG;
                m.to = self.layout.rook_start_squares[us.u][m.info.castletype.u]; // "King takes rook"
            }
        }
        // Promotion.
        else {
            m.type = .promotion;
            m.info.prom = try MoveInfo.Prom.from_char(str[4]);
        }

        var finder: MoveFinder = .init(m);
        self.lazy_generate_moves(&finder);
        if (finder.found) return m;
        return types.ParsingError.IllegalMove;
    }

    /// A convenient (faster) way to set the startposition without the need for a fen string.
    pub fn set_startpos(self: *Position, new_state: *StateInfo) void {
        new_state.* = .empty;
        new_state.castling_rights = cf_all;
        self.* = empty_classic;
        self.state = new_state;
        self.init_hash();
        // Update state not needed.
    }

    /// Clones a position.
    pub fn clone(src: *const Position, new_state: *StateInfo) Position {
        var pos: Position = src.*;
        new_state.* = src.state.*;
        new_state.prev = null;
        new_state.next = null;
        pos.state = new_state;
        pos.ply = 0;
        return pos;
    }

    /// Clones a position.
    pub fn copy_from(self: *Position, other: *const Position, new_state: *StateInfo) void {
        assert(self != other);
        self.* = other.*;
        new_state.* = other.state.*;
        new_state.prev = null;
        new_state.next = null;
        self.state = new_state;
        self.ply = 0;
    }

    /// Flips the board. Color, pieces etc.
    pub fn flip(self: *Position, new_state: *StateInfo) void {
        var bb: u64 = self.all();
        const bk_board: [64]Piece = self.board;
        const bk_layout = self.layout;
        const bk_state: StateInfo = self.state.*;
        const bk_to_move = self.to_move;
        const bk_game_ply = self.game_ply;
        const bk_is_960 = self.is_960;

        self.clear(new_state);

        self.layout = bk_layout;
        self.to_move = bk_to_move.opp();
        self.game_ply = bk_game_ply;
        self.is_960 = bk_is_960;

        while (bb != 0) {
            const org_sq: Square = pop_square(&bb);
            const org_piece = bk_board[org_sq.u];
            self.lazy_add_piece(org_sq.flipped(), org_piece.opp());
        }

        new_state.rule50 = bk_state.rule50;
        if (bk_state.ep_square.u > 0) {
            new_state.ep_square = bk_state.ep_square.flipped();
        }

        new_state.last_move = bk_state.last_move.flipped();
        new_state.moved_piece = bk_state.moved_piece.opp();
        new_state.captured_piece = bk_state.captured_piece.opp();
        new_state.castling_rights = (bk_state.castling_rights >> 2) | (bk_state.castling_rights << 2);

        self.init_hash();
        self.lazy_update_state();
    }

    /// Initialize layout.
    /// * TODO: if possible give the start_files as parameter here!
    fn init_layout(self: *Position) void {

        const WHITE: u1 = comptime Color.WHITE.u;
        const BLACK: u1 = comptime Color.BLACK.u;
        const O_O: u1 = comptime CastleType.SHORT.u;
        const O_O_O: u1 = comptime CastleType.LONG.u;

        // White.
        var rook_o_o_o: Square = .from_rank_file(bitboards.rank_1, self.layout.start_files[0]);
        var rook_o_o: Square = .from_rank_file(bitboards.rank_1, self.layout.start_files[1]);
        var king: Square = .from_rank_file(bitboards.rank_1, self.layout.start_files[2]);

        self.layout.king_start_squares[0] = king;
        self.layout.rook_start_squares[WHITE][O_O] = rook_o_o;
        self.layout.rook_start_squares[WHITE][O_O_O] = rook_o_o_o;
        self.layout.castling_between_bitboards[WHITE][O_O] = squarepairs.in_between_bitboard(king, rook_o_o);
        self.layout.castling_between_bitboards[WHITE][O_O_O] = squarepairs.in_between_bitboard(king, rook_o_o_o);
        self.layout.castling_king_paths[WHITE][O_O] = determine_king_path(king, Square.G1);
        self.layout.castling_king_paths[WHITE][O_O_O] = determine_king_path(king, Square.C1);
        self.layout.castling_masks[rook_o_o.u] = cf_white_short;
        self.layout.castling_masks[rook_o_o_o.u] = cf_white_long;
        self.layout.castling_masks[king.u] = cf_white_short | cf_white_long;

        // Black.
        rook_o_o_o = .from_rank_file(bitboards.rank_8, self.layout.start_files[0]);
        rook_o_o = .from_rank_file(bitboards.rank_8, self.layout.start_files[1]);
        king = .from_rank_file(bitboards.rank_8, self.layout.start_files[2]);

        self.layout.king_start_squares[1] = king;
        self.layout.rook_start_squares[BLACK][O_O] = rook_o_o;
        self.layout.rook_start_squares[BLACK][O_O_O] = rook_o_o_o;
        self.layout.castling_between_bitboards[BLACK][O_O] = squarepairs.in_between_bitboard(king, rook_o_o);
        self.layout.castling_between_bitboards[BLACK][O_O_O] = squarepairs.in_between_bitboard(king, rook_o_o_o);
        self.layout.castling_king_paths[BLACK][O_O] = determine_king_path(king, Square.G8);
        self.layout.castling_king_paths[BLACK][O_O_O] = determine_king_path(king, Square.C8);
        self.layout.castling_masks[rook_o_o.u] = cf_black_short;
        self.layout.castling_masks[rook_o_o_o.u] = cf_black_long;
        self.layout.castling_masks[king.u] = cf_black_short | cf_black_long;
    }

    fn determine_king_path(from: Square, to: Square) u64 {
        // Excluded the king square from the path. Castle moves are not generated when in check and so it saves validations when generating moves.
        var path: u64 = 0;
        var k: Square = from;
        // Short castle.
        if (k.u < to.u) {
            while (k.u != to.u) {
                k.u += 1;
                path |= k.to_bitboard();
            }
        }
        // Long castle.
        else if (k.u > to.u) {
            while (k.u != to.u) {
                k.u -= 1;
                path |= k.to_bitboard();
            }
        }
        return path;
    }

    fn init_hash(self: *Position) void {
        self.state.key = self.compute_hashkey();
    }

    fn compute_hashkey(self: *const Position) u64 {
        const st = self.state;
        var k: u64 = 0;
        // Loop through occupied squares.
        var occ: u64 = self.all();
        while (occ != 0) {
            const sq = pop_square(&occ);
            const pc = self.get(sq);
            k ^= zobrist.piece_square(pc, sq);
        }
        k ^= zobrist.castling(st.castling_rights);
        if (st.ep_square.u > 0) k ^= zobrist.enpassant(st.ep_square.file());
        if (self.to_move.e == .black) k ^= zobrist.btm();
        return k;
    }

    pub fn get(self: *const Position, sq: Square) Piece {
        return self.board[sq.u];
    }

    pub fn is_check(self: *const Position) bool {
        return self.state.checkers > 0;
    }

    pub fn phase_of(material_without_pawns: Value) GamePhase {
        assert(material_without_pawns >= 0);
        // TODO: lots of promotions becomes opening :(
        return if (material_without_pawns > types.midgame_threshold) .Opening
        else if (material_without_pawns > types.endgame_threshold) .Midgame
        else .Endgame;
    }

    pub fn phase(self: *const Position) GamePhase {
        return phase_of(self.non_pawn_material());
    }

    /// Non-comptime getter for the outside world.
    pub fn pieces(self: *const Position, pt: PieceType, us: Color) u64 {
        return self.bb_by_type[pt.u] & self.bb_by_color[us.u];
    }

    pub fn by_type(self: *const Position, pt: PieceType) u64 {
        return self.bb_by_type[pt.u];
    }

    pub fn by_color(self: *const Position, us: Color) u64 {
        return self.bb_by_color[us.u];
    }

    pub fn all(self: *const Position) u64 {
        return self.bb_by_type[0];
    }

    pub fn all_pawns(self: *const Position) u64 {
        return self.by_type(P);
    }

    pub fn all_knights(self: *const Position) u64 {
        return self.by_type(N);
    }

    pub fn all_queens_bishops(self: *const Position) u64 {
        return (self.by_type(B) | self.by_type(Q));
    }

    pub fn all_queens_rooks(self: *const Position) u64 {
        return (self.by_type(R) | self.by_type(Q));
    }

    pub fn all_kings(self: *const Position) u64 {
        return self.by_type(K);
    }

    pub fn pawns(self: *const Position, us: Color) u64 {
        return self.by_type(P) & self.by_color(us);
    }

    /// All our pieces except pawns.
    pub fn non_pawns(self: *const Position, us: Color) u64 {
        return ~self.by_type(P) & self.by_color(us);
    }

    pub fn knights(self: *const Position, us: Color) u64 {
        return self.by_type(N) & self.by_color(us);
    }

    pub fn bishops(self: *const Position, us: Color) u64 {
        return self.by_type(B) & self.by_color(us);
    }

    pub fn rooks(self: *const Position, us: Color) u64 {
        return self.by_type(R) & self.by_color(us);
    }

    pub fn queens(self: *const Position, us: Color) u64 {
        return self.by_type(Q) & self.by_color(us);
    }

    pub fn kings(self: *const Position, us: Color) u64 {
        return self.by_type(K) & self.by_color(us);
    }

    pub fn queens_bishops(self: *const Position, us: Color) u64 {
        return (self.by_type(B) | self.by_type(Q)) & self.by_color(us);
    }

    pub fn queens_rooks(self: *const Position, us: Color) u64 {
        return (self.by_type(R) | self.by_type(Q)) & self.by_color(us);
    }

    pub fn king_square(self: *const Position, us: Color) Square {
        return funcs.first_square(self.kings(us));
    }

    pub fn sliders(self: *const Position, us: Color) u64 {
        return (self.by_type(B) | self.by_type(R) | self.by_type(Q)) & self.by_color(us);
    }

    /// Returns the sum of the white + black materials.
    pub fn material(self: *const Position) Value {
        return self.materials[0] + self.materials[1];
    }

    pub fn non_pawn_material(self: *const Position) Value {
        return (self.materials[0] + self.materials[1]) - (P.material() * popcnt(self.all_pawns()));
    }

    fn is_usable_ep_square(self: *const Position, ep: Square) bool {
        const rank: u3 = ep.rank();
        if (rank == bitboards.rank_3) {
            const w_pawn_sq = ep.add(8);
            const requirements: bool = self.board[w_pawn_sq.u].e == .w_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u - 8].e == .no_piece;
            return requirements and (masks.get_ep_mask(w_pawn_sq) & self.pawns(Color.BLACK) != 0);
        }
        else if (rank == bitboards.rank_6) {
            const b_pawn_sq = ep.sub(8);
            const requirements:  bool = self.board[b_pawn_sq.u].e == .b_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u + 8].e == .no_piece;
            return requirements and (masks.get_ep_mask(b_pawn_sq) & self.pawns(Color.WHITE) != 0);
        }
        return false;
    }

    pub fn lazy_add_piece(self: *Position, sq: Square, pc: Piece) void {
        switch (pc.color().e) {
            .white => self.add_piece(Color.WHITE, pc, sq),
            .black => self.add_piece(Color.BLACK, pc, sq),
        }
    }

    fn add_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        if (comptime lib.is_paranoid) {
            assert(self.get(sq).is_empty());
            assert(pc.is_piece());
            assert(pc.color().e == us.e);
        }
        const mask: u64 = sq.to_bitboard();
        self.board[sq.u] = pc;
        self.bb_by_type[0] |= mask;
        self.bb_by_type[pc.piecetype.u] |= mask;
        self.bb_by_color[us.u] |= mask;
        self.values[us.u] += pc.value();
        self.materials[us.u] += pc.material();
    }

    fn remove_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        if (comptime lib.is_paranoid) {
            assert(self.get(sq).e == pc.e);
            assert(pc.color().e == us.e);
        }
        const not_mask: u64 = ~sq.to_bitboard();
        self.board[sq.u] = Piece.NO_PIECE;
        self.bb_by_type[0] &= not_mask;
        self.bb_by_type[pc.piecetype.u] &= not_mask;
        self.bb_by_color[us.u] &= not_mask;
        self.values[us.u] -= pc.value();
        self.materials[us.u] -= pc.material();
    }

    fn move_piece(self: *Position, comptime us: Color, pc: Piece, from: Square, to: Square) void {
        if (comptime lib.is_paranoid) {
            assert(self.get(from).e == pc.e);
            assert(self.get(to).is_empty());
            assert(pc.color().e == us.e);
        }
        const xor_mask: u64 = from.to_bitboard() | to.to_bitboard();
        self.board[from.u] = Piece.NO_PIECE;
        self.board[to.u] = pc;
        self.bb_by_type[0] ^= xor_mask;
        self.bb_by_type[pc.piecetype.u] ^= xor_mask;
        self.bb_by_color[us.u] ^= xor_mask;
    }

    pub fn is_threefold_repetition(self: *const Position) bool {
        const st: *const StateInfo = self.state;
        if (st.rule50 < 4) return false;

        const end: u16 = @min(st.rule50, self.ply);
        if (end < 4) return false;

        var count: u8 = 0;
        var i: u16 = 4;
        var run: *const StateInfo = st;
        while (i <= end) : (i += 2) {
            run = run.prev.?.prev.?;
            if (run.key == st.key) {
                count += 1;
                if (count >= 2) return true;
            }
        }
        return false;
    }

    /// Makes the move on the board.
    /// * `us` is comptime for performance reasons and must be the stm.
    /// * `st` will become the new state and will be fully updated.
    pub fn do_move(self: *Position, comptime us: Color, st: *StateInfo, m: Move) void {
        assert(us.e == self.to_move.e);
        if (comptime lib.is_paranoid) assert(self.pos_ok());

        const them: Color = comptime us.opp();
        const from: Square = m.from;
        const to: Square = m.to;
        const movetype: MoveType = m.type;
        const pc: Piece = self.board[from.u];
        const capt: Piece =
            switch (movetype) {
                .normal, .promotion => self.board[to.u],
                .enpassant => Piece.create_pawn(them),
                .castle => Piece.NO_PIECE,
            };

        const is_pawnmove: bool = pc.is_pawn();
        const is_capture: bool = capt.is_piece();
        const hash_delta = zobrist.piece_square(pc, from) ^ zobrist.piece_square(pc, to);

        var key: u64 = self.state.key ^ zobrist.btm();

        // Copy some state stuff. The rest is done down here and in update_state().
        st.rule50 = self.state.rule50 + 1;
        st.ep_square = self.state.ep_square;
        st.castling_rights = self.state.castling_rights;
        self.state.next = st;
        st.prev = self.state;
        self.state = st;

        st.last_move = m;
        st.moved_piece = pc;
        st.captured_piece = capt;

        self.to_move = them;
        self.ply += 1;
        self.game_ply += 1;

        // Reset drawcounter by default.
        if (is_pawnmove or is_capture) st.rule50 = 0;

        // Clear ep by default if it is set.
        if (st.ep_square.u > 0) {
            key ^= zobrist.enpassant(st.ep_square.file());
            st.ep_square = Square.zero;
        }

        // Update the castling rights.
        if (st.castling_rights != 0) {
            const mask: u4 = self.layout.castling_masks[from.u] | self.layout.castling_masks[to.u];
            if (mask != 0) {
                key ^= zobrist.castling(st.castling_rights);
                st.castling_rights &= ~mask;
                key ^= zobrist.castling(st.castling_rights);
            }
        }

        switch (movetype) {
            .normal => {
                if (is_capture) {
                    self.remove_piece(them, capt, to);
                    key ^= zobrist.piece_square(capt, to);
                }
                self.move_piece(us, pc, from, to);
                key ^= hash_delta;
                if (is_pawnmove) {
                    // Double pawn push: only set the ep-square if it is actually possible to do an ep-capture.
                    if (from.u ^ to.u == 16 and masks.get_ep_mask(to) & self.pawns(them) != 0) {
                        const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                        st.ep_square = ep;
                        key ^= zobrist.enpassant(ep.file());
                    }
                }
            },
            .promotion => {
                const prom: Piece = m.info.prom.to_piece(us);
                const pawn: Piece = comptime Piece.create_pawn(us);
                if (is_capture) {
                    self.remove_piece(them, capt, to);
                    key ^= zobrist.piece_square(capt, to);
                }
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            .enpassant => {
                const pawn_us: Piece = comptime Piece.create_pawn(us);
                const pawn_them: Piece = comptime Piece.create_pawn(them);
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.remove_piece(them, pawn_them, capt_sq);
                self.move_piece(us, pawn_us, from, to);
                key ^= hash_delta ^ zobrist.piece_square(pawn_them, capt_sq);
            },
            .castle => {
                // Castling is encoded as "king takes rook".
                const king: Piece = comptime Piece.create_king(us);
                const rook: Piece = comptime Piece.create_rook(us);
                const castletype: CastleType = m.info.castletype;
                const king_to: Square = king_castle_destination_squares[us.u][castletype.u];
                const rook_to: Square = rook_castle_destination_squares[us.u][castletype.u];
                self.move_piece(us, king, from, king_to);
                self.move_piece(us, rook, to, rook_to);
                key ^= zobrist.piece_square(king, from) ^ zobrist.piece_square(king, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
            },
        }

        st.key = key;
        self.update_state(them);

        if (comptime lib.is_paranoid) assert(self.pos_ok());
    }

    /// This must be called with `us` being the color that moved on the previous ply!
    pub fn undo_move(self: *Position, comptime us: Color) void {
        assert(self.to_move.e != us.e);
        if (comptime lib.is_paranoid) assert(self.pos_ok());

        const st: *const StateInfo = self.state;

        self.to_move = us;
        self.ply -= 1;
        self.game_ply -= 1;

        const them: Color = comptime us.opp();
        const m: Move = st.last_move;
        const capt: Piece = st.captured_piece;
        const is_capture: bool = capt.is_piece();
        const from: Square = m.from;
        const to: Square = m.to;

        switch (m.type) {
            .normal => {
                self.move_piece(us, st.moved_piece, to, from);
                if (is_capture) self.add_piece(them, capt, to);
            },
            .promotion => {
                const pawn: Piece = comptime Piece.create_pawn(us);
                const prom: Piece = m.info.prom.to_piece(us);
                self.remove_piece(us, prom, to);
                self.add_piece(us, pawn, from);
                if (is_capture) self.add_piece(them, capt, to);
            },
            .enpassant => {
                const pawn_us: Piece = comptime Piece.create_pawn(us);
                const pawn_them: Piece = comptime Piece.create_pawn(them);
                self.move_piece(us, pawn_us, to, from);
                const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.add_piece(them, pawn_them, ep);
            },
            .castle => {
                // Castling is "king takes rook".
                const king: Piece = comptime Piece.create_king(us);
                const rook: Piece = comptime Piece.create_rook(us);
                const castletype: CastleType = m.info.castletype;
                const rook_to: Square = rook_castle_destination_squares[us.u][castletype.u];
                const king_to: Square = king_castle_destination_squares[us.u][castletype.u];
                self.move_piece(us, rook, rook_to, to);
                self.move_piece(us, king, king_to, from);
            },
        }

        self.state = self.state.prev.?;
        self.state.next = null;

        if (comptime lib.is_paranoid) assert(self.pos_ok());
    }

    /// Skip a turn.
    pub fn do_nullmove(self: *Position, comptime us: Color, st: *StateInfo) void {
        if (comptime lib.is_paranoid) {
            assert(!self.nullmove_state);
        }
        self.nullmove_state = true;
        const them: Color = comptime us.opp();

        var key: u64 = self.state.key ^ zobrist.btm();

        // Copy some state stuff. The rest is done down here and in update_state().
        st.rule50 = self.state.rule50;
        st.ep_square = self.state.ep_square;
        st.castling_rights = self.state.castling_rights;
        self.state.next = st;
        st.prev = self.state;
        self.state = st;

        st.last_move = Move.nullmove;
        st.moved_piece = Piece.NO_PIECE;
        st.captured_piece = Piece.NO_PIECE;

        self.to_move = them;
        self.ply += 1;
        self.game_ply += 1;

        // Clear ep by default if it is set.
        if (st.ep_square.u > 0) {
            key ^= zobrist.enpassant(st.ep_square.file());
            st.ep_square = Square.zero;
        }

        st.key = key;
        self.update_state(them);
    }

    pub fn undo_nullmove(self: *Position, comptime us: Color) void {
        if (comptime lib.is_paranoid) assert(self.nullmove_state);
        self.nullmove_state = false;
        self.to_move = us;
        self.ply -= 1;
        self.game_ply -= 1;
        self.state = self.state.prev.?;
        self.state.next = null;
    }

    /// Update checks and pins for both sides.
    fn update_state(self: *Position, comptime us: Color) void {
        const st: *StateInfo = self.state;
        const bb_all: u64 = self.all();

        st.checkmask = 0;
        st.pins_orthogonal = .{ 0, 0 };
        st.pins_diagonal = .{ 0, 0 };

        st.checkmask =
            (data.get_pawn_attacks(self.king_square(us), us) & self.pawns(us.opp())) |
            (data.get_knight_attacks(self.king_square(us)) & self.knights(us.opp()));

        inline for (Color.all) |color| {
            const our_king_sq: Square = self.king_square(color);
            const bb_occ_without_us: u64 = bb_all ^ self.by_color(color);
            var candidate_attackers: u64 =
                (data.get_bishop_attacks(our_king_sq, bb_occ_without_us) & self.queens_bishops(color.opp())) |
                (data.get_rook_attacks(our_king_sq, bb_occ_without_us) & self.queens_rooks(color.opp()));

            // Use candidate attackers for both checkers and pins.
            while (candidate_attackers != 0) {
                const attacker_sq: Square = pop_square(&candidate_attackers);
                const attacker_square_bitboard: u64 = attacker_sq.to_bitboard();
                const pair = squarepairs.get(self.king_square(color), attacker_sq);
                const bb_ray: u64 = pair.in_between_bitboard & self.by_color(color);
                // We have a slider checker when there is nothing in between.
                if (us.e == color.e and bb_ray == 0) {
                    st.checkmask |= pair.in_between_bitboard | attacker_square_bitboard;
                }
                // We have a pin when exactly 1 bit is set. There is one piece in between.
                else if (popcnt(bb_ray) == 1) {
                    switch (pair.mask) {
                        0b100 => st.pins_orthogonal[color.u] |= pair.in_between_bitboard | attacker_square_bitboard,
                        0b001 => st.pins_diagonal[color.u] |= pair.in_between_bitboard | attacker_square_bitboard,
                        else => unreachable,
                    }
                }
            }
        }

        st.checkers = st.checkmask & bb_all;
        st.pins[0] = st.pins_diagonal[0] | st.pins_orthogonal[0];
        st.pins[1] = st.pins_diagonal[1] | st.pins_orthogonal[1];
    }

    pub fn lazy_do_move(self: *Position, st: *StateInfo, move: Move) void {
        switch (self.to_move.e) {
            .white => self.do_move(Color.WHITE, st, move),
            .black => self.do_move(Color.BLACK, st, move),
        }
    }

    pub fn lazy_undo_move(self: *Position) void {
        switch (self.to_move.e) {
            .white => self.undo_move(Color.BLACK),
            .black => self.undo_move(Color.WHITE),
        }
    }

    pub fn lazy_do_nullmove(self: *Position, st: *StateInfo) void {
        switch (self.to_move.e) {
            .white => self.do_nullmove(Color.WHITE, st),
            .black => self.do_nullmove(Color.BLACK, st),
        }
    }

    pub fn lazy_undo_nullmove(self: *Position) void {
        switch (self.to_move.e) {
            .white => self.undo_nullmove(Color.BLACK),
            .black => self.undo_nullmove(Color.WHITE),
        }
    }

    fn lazy_update_state(self: *Position) void {
        switch (self.to_move.e) {
            .white => self.update_state(Color.WHITE),
            .black => self.update_state(Color.BLACK),
        }
    }

    /// Returns true if square `to` is attacked by any piece of `attacker`.
    pub fn is_square_attacked_by(self: *const Position, to: Square, comptime attacker: Color) bool {
        const inverted = comptime attacker.opp();
        return
            (data.get_knight_attacks(to) & self.knights(attacker)) |
            (data.get_king_attacks(to) & self.kings(attacker)) |
            (data.get_pawn_attacks(to, inverted) & self.pawns(attacker)) |
            (data.get_rook_attacks(to, self.all()) & self.queens_rooks(attacker)) |
            (data.get_bishop_attacks(to, self.all()) & self.queens_bishops(attacker)) != 0;
    }

    /// Returns true if square `to` is attacked by any piece of `attacker` for a certain occupation `occ`.
    pub fn is_square_attacked_by_for_occupation(self: *const Position, occ: u64, to: Square, comptime attacker: Color) bool {
        const inverted = comptime attacker.opp();
        return
            (data.get_knight_attacks(to) & self.knights(attacker)) |
            (data.get_king_attacks(to) & self.kings(attacker)) |
            (data.get_pawn_attacks(to, inverted) & self.pawns(attacker)) |
            (data.get_rook_attacks(to, occ) & self.queens_rooks(attacker)) |
            (data.get_bishop_attacks(to, occ) & self.queens_bishops(attacker)) != 0;
    }

    /// Gives a bitboard of attackers which attack `to` for both colors.
    pub fn get_all_attacks_to_for_occupation(self: *const Position, occ: u64, to: Square) u64 {
        return
            (data.get_knight_attacks(to) & self.all_knights()) |
            (data.get_king_attacks(to) & self.all_kings()) |
            (data.get_pawn_attacks(to, .WHITE) & self.all_pawns()) |
            (data.get_pawn_attacks(to, .BLACK) & self.all_pawns()) |
            (data.get_rook_attacks(to, occ) & self.all_queens_rooks()) |
            (data.get_bishop_attacks(to, occ) & self.all_queens_bishops());
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
        while (their_knights != 0) {
            const from: Square = pop_square(&their_knights);
            att |= data.get_knight_attacks(from);
        }

        // Diagonal sliders.
        var their_diag_sliders = self.queens_bishops(attacker);
        while (their_diag_sliders != 0) {
            const from: Square = pop_square(&their_diag_sliders);
            att |= data.get_bishop_attacks(from, occ);
        }

        // Orthogonal sliders.
        var their_orth_sliders = self.queens_rooks(attacker);
        while (their_orth_sliders != 0) {
            const from: Square = pop_square(&their_orth_sliders);
            att |= data.get_rook_attacks(from, occ);
        }

        // King.
        att |= data.get_king_attacks(self.king_square(attacker));

        return att;
    }

    pub fn get_unsafe_squares_for_king(self: *const Position, comptime us: Color) u64 {
        return self.attacks_by_for_occupation(us.opp(), self.all() & ~self.kings(us));
    }

    fn is_castlingpath_empty(self: *const Position, comptime us: Color, comptime castletype: CastleType, ) bool {
        const path: u64 = self.layout.castling_between_bitboards[us.u][castletype.u];
        return path & self.all() == 0;
    }

    pub fn lazy_generate_moves(self: *const Position, noalias storage: anytype) void {
        switch (self.to_move.e) {
            .white => self.generate_moves(Color.WHITE, storage),
            .black => self.generate_moves(Color.BLACK, storage),
        }
    }

    pub fn lazy_generate_captures(self: *const Position, noalias storage: anytype) void {
        switch (self.to_move.e) {
            .white => self.generate_captures(Color.WHITE, storage),
            .black => self.generate_captures(Color.BLACK, storage),
        }
    }

    pub  fn generate_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void {
        if (comptime lib.is_paranoid) assert(self.to_move.e == us.e);
        storage.reset();
        const check: bool = self.state.checkers != 0;
        const pins: bool = self.state.pins[us.u] != 0;
        switch (check) {
            false => switch (pins) {
                false => self.gen(Params.create(false, us, false, false), storage),
                true  => self.gen(Params.create(false, us, false, true), storage),
            },
            true => switch (pins) {
                false => self.gen(Params.create(false, us, true, false), storage),
                true  => self.gen(Params.create(false, us, true, true), storage),
            },
        }
    }

    /// Generate captures only, but if in check generate all.
    pub  fn generate_captures(noalias self: *const Position, comptime us: Color, noalias storage: anytype) void {
        if (comptime lib.is_paranoid) assert(self.to_move.e == us.e);
        storage.reset();
        const check: bool = self.state.checkers != 0;
        const pins: bool = self.state.pins[us.u] != 0;
        switch (check) {
            false => switch (pins) {
                false => self.gen(Params.create(true, us, false, false), storage),
                true  => self.gen(Params.create(true, us, false, true), storage),
            },
            true => switch (pins) {
                false => self.gen(Params.create(true, us, true, false), storage),
                true  => self.gen(Params.create(true, us, true, true), storage),
            },
        }
    }

    /// See `MoveStorage` for the interface of `storage`: required are the functions `reset()` and `store()`.
    fn gen(self: *const Position, comptime ctp: Params, noalias storage: anytype) void {
        // Comptimes.
        const us = comptime ctp.us;
        const them = comptime us.opp();
        const do_all_promotions: bool = comptime !ctp.captures;

        //const st: *const StateInfo = self.state; // In doubt what is faster: ref or copy.
        const st: StateInfo = self.state.*; // Up until now this seems a bit faster.

        const doublecheck: bool = ctp.check and popcnt(st.checkers) > 1;
        const bb_all: u64 = self.all();
        const bb_us: u64 = self.by_color(us);
        const bb_them: u64 = self.by_color(them);
        const bb_not_us: u64 = ~bb_us;
        const king_sq: Square = self.king_square(us);

        // In case of a doublecheck we can only move the king.
        if (!doublecheck) {
            const our_pawns = self.pawns(us);
            const our_knights = self.knights(us);
            const our_queens_bishops = self.queens_bishops(us);
            const our_queens_rooks = self.queens_rooks(us);

            const target = if (ctp.check) st.checkmask else if (!ctp.captures) bb_not_us else bb_them;

            // Pawns.
            if (our_pawns != 0) {
                const third_rank: u64 = comptime funcs.relative_rank_3_bitboard(us);
                const last_rank: u64 = comptime funcs.relative_rank_8_bitboard(us);
                const empty_squares: u64 = ~bb_all;
                const enemies: u64 = if (ctp.check) st.checkers else bb_them;

                // Generate all 4 types of pawnmoves: push, push double, capture left, capture right.
                var bb_single = switch(ctp.pins) {
                    false => (pawns_shift(our_pawns, us, .up) & empty_squares),
                    true  => (pawns_shift(our_pawns & ~st.pins[us.u], us, .up) & empty_squares) |
                             (pawns_shift(our_pawns & st.pins_diagonal[us.u], us, .up) & empty_squares & st.pins_diagonal[us.u]) |
                             (pawns_shift(our_pawns & st.pins_orthogonal[us.u], us, .up) & empty_squares & st.pins_orthogonal[us.u])
                };

                var bb_double = pawns_shift(bb_single & third_rank, us, .up) & empty_squares;

                // Pawn push check interpolation.
                bb_single &= target;
                bb_double &= target;

                const bb_northwest: u64 = switch (ctp.pins) {
                    false => (pawns_shift(our_pawns, us, .northwest) & enemies),
                    true  => (pawns_shift(our_pawns & ~st.pins[us.u], us, .northwest) & enemies) |
                             (pawns_shift(our_pawns & st.pins_diagonal[us.u], us, .northwest) & enemies & st.pins_diagonal[us.u]),
                };

                const bb_northeast: u64 = switch (ctp.pins) {
                    false => (pawns_shift(our_pawns, us, .northeast) & enemies),
                    true  => (pawns_shift(our_pawns & ~st.pins[us.u], us, .northeast) & enemies) |
                             (pawns_shift(our_pawns & st.pins_diagonal[us.u], us, .northeast) & enemies & st.pins_diagonal[us.u]),
                };

                // Pawn pushes.
                if (ctp.check or !ctp.captures) {
                    // Single push normal
                    var bb_single_push: u64 = bb_single & ~last_rank;
                    while (bb_single_push != 0) {
                        const to: Square = pop_square(&bb_single_push);
                        const from: Square = pawn_from(to, us, .up);
                        store(from, to, storage) orelse return;
                    }
                    // Double.push
                    var bb_double_push: u64 = bb_double;
                    while (bb_double_push != 0) {
                        const to: Square = pop_square(&bb_double_push);
                        const from: Square = if (us.e == .white) to.sub(16) else to.add(16);
                        store(from, to, storage) orelse return;
                    }
                }

                // left capture promotions
                var bb_northwest_promotions = bb_northwest & last_rank;
                while (bb_northwest_promotions != 0) {
                    const to: Square = pop_square(&bb_northwest_promotions);
                    const from: Square = pawn_from(to, us, .northwest);
                    store_promotions(do_all_promotions, from, to, storage) orelse return;
                }

                // right capture promotions
                var bb_northeast_promotions = bb_northeast & last_rank;
                while (bb_northeast_promotions != 0) {
                    const to: Square = pop_square(&bb_northeast_promotions);
                    const from: Square = pawn_from(to, us, .northeast);
                    store_promotions(do_all_promotions, from, to, storage) orelse return;
                }

                // push promotions
                var bb_push_promotions: u64 = bb_single & last_rank;
                while (bb_push_promotions != 0) {
                    const to: Square = pop_square(&bb_push_promotions);
                    const from: Square =  pawn_from(to, us, .up);
                    store_promotions(do_all_promotions, from, to, storage) orelse return;
                }

                // left normal captures,
                var bb_northwest_normal =  bb_northwest & ~last_rank;
                while (bb_northwest_normal != 0) {
                    const to: Square = pop_square(&bb_northwest_normal);
                    const from: Square = pawn_from(to, us, .northwest);
                    store(from, to, storage) orelse return;
                }

                // right normal captures,
                var bb_northeast_normal =  bb_northeast & ~last_rank;
                while (bb_northeast_normal != 0) {
                    const to: Square = pop_square(&bb_northeast_normal);
                    const from: Square = pawn_from(to, us, .northeast);
                    store(from, to, storage) orelse return;
                }

                const ep: Square = st.ep_square;
                // Enpassant.
                if (ep.u > 0) {
                    var bb_enpassant: u64 = data.get_pawn_attacks(ep, them) & our_pawns; // inversion trick.
                    inline for (0..2) |_| {
                        if (bb_enpassant == 0) break;
                        const from: Square = pop_square(&bb_enpassant);
                        if (self.is_legal_enpassant(us, king_sq, from, ep))
                            store_enpassant(from, ep, storage) orelse return;
                    }
                }
            } // (pawns)

            // Knights.
            var bb_knights: u64 = if (!ctp.pins) our_knights else our_knights & ~st.pins[us.u]; // A knight can never escape a pin.
            while (bb_knights != 0) {
                const from: Square = pop_square(&bb_knights);
                var bb_to: u64 = data.get_knight_attacks(from) & target;
                inline for (0..8) |_| {
                    if (bb_to == 0) break;
                    store(from, pop_square(&bb_to), storage) orelse return;
                }
            }

            // Diagonal sliders.
            if (!ctp.pins) {
                var our_sliders: u64 = our_queens_bishops;
                while (our_sliders != 0) {
                    const from: Square = pop_square(&our_sliders);
                    var bb_to: u64 = data.get_bishop_attacks(from, bb_all) & target;
                    while (bb_to != 0) {
                        store(from, pop_square(&bb_to), storage) orelse return;
                    }
                }
            } else {
                var non_pinned_sliders: u64 = our_queens_bishops & ~st.pins[us.u];
                while (non_pinned_sliders != 0) {
                    const from: Square = pop_square(&non_pinned_sliders);
                    var bb_to: u64 = data.get_bishop_attacks(from, bb_all) & target;
                    while (bb_to != 0) {
                        store(from, pop_square(&bb_to), storage) orelse return;
                    }
                }
                var pinned_sliders: u64 = our_queens_bishops & st.pins_diagonal[us.u];
                while (pinned_sliders != 0) {
                    const from: Square = pop_square(&pinned_sliders);
                    var bb_to: u64 = data.get_bishop_attacks(from, bb_all) & target & st.pins_diagonal[us.u];
                    while (bb_to != 0) {
                        store(from, pop_square(&bb_to), storage) orelse return;
                    }
                }
            }

            // Orthogonal sliders.
            if (!ctp.pins) {
                var our_sliders: u64 = our_queens_rooks;
                while (our_sliders != 0) {
                    const from: Square = pop_square(&our_sliders);
                    var bb_to: u64 = data.get_rook_attacks(from, bb_all) & target;
                    while (bb_to != 0) {
                        store(from, pop_square(&bb_to), storage) orelse return;
                    }
                }
            } else {
                var non_pinned_sliders: u64 = our_queens_rooks & ~st.pins[us.u];
                while (non_pinned_sliders != 0) {
                    const from: Square = pop_square(&non_pinned_sliders);
                    var bb_to: u64 = data.get_rook_attacks(from, bb_all) & target;
                    while (bb_to != 0) {
                        store(from, pop_square(&bb_to), storage) orelse return;
                    }
                }

                var pinned_sliders: u64 = our_queens_rooks & st.pins_orthogonal[us.u];
                while (pinned_sliders != 0) {
                    const from: Square = pop_square(&pinned_sliders);
                    var bb_to: u64 = data.get_rook_attacks(from, bb_all) & target & st.pins_orthogonal[us.u];
                    while (bb_to != 0) {
                        store(from, pop_square(&bb_to), storage) orelse return;
                    }
                }
            }

        } // (not doublecheck)

        // King.
        const king_target = if (ctp.check or !ctp.captures) bb_not_us else bb_them;
        var bb_to = data.get_king_attacks(king_sq) & king_target;

        // The king is a troublemaker. For now this 'popcount heuristic' gives the best avg speed, using 2 different approaches to check legality.
        if (popcnt(bb_to) > 2) {
            const bb_unsafe: u64 = self.get_unsafe_squares_for_king(us);
            bb_to &= ~bb_unsafe;
            // Normal.
            while (bb_to != 0) {
                store(king_sq, pop_square(&bb_to), storage) orelse return;
            }
            // Castling.
            if (!ctp.check and !ctp.captures and st.castling_rights != 0) {
                inline for (CastleType.all) |ct| {
                    if (st.is_castling_allowed(us, ct) and self.is_castlingpath_empty(us, ct) and self.is_legal_castle(us, ct, bb_unsafe)) {
                        const to: Square = self.layout.rook_start_squares[us.u][ct.u]; // Castling is "king takes rook".
                        store_castle(king_sq, to, ct, storage) orelse return;
                    }
                }
            }
        } else {
            const bb_without_king: u64 = bb_all ^ self.kings(us);
            // Normal.
            while (bb_to != 0) {
                const to: Square = pop_square(&bb_to);
                if (self.is_legal_kingmove(us, bb_without_king, to)) {
                    store(king_sq, to, storage) orelse return;
                }
            }
            // Castling.
            if (!ctp.check and !ctp.captures and st.castling_rights != 0) {
                inline for (CastleType.all) |ct| {
                    if (st.is_castling_allowed(us, ct) and self.is_castlingpath_empty(us, ct) and self.is_legal_castle_check_attacks(us, ct)) {
                        const to: Square = self.layout.rook_start_squares[us.u][ct.u]; // Castling is "king takes rook".
                        store_castle(king_sq, to, ct, storage) orelse return;
                    }
                }
            }
        }
    }

    fn store(from: Square, to: Square, noalias storage: anytype) ?void {
        return storage.store(Move.create(from, to));
    }

    fn store_enpassant(from: Square, to: Square, noalias storage: anytype) ?void {
        return storage.store(Move.create_enpassant(from, to));
    }

    fn store_promotions(comptime do_all: bool, from: Square, to: Square, noalias storage: anytype) ?void {
        storage.store(Move.create_promotion(from, to, .queen)) orelse return null;

        if (do_all) {
            storage.store(Move.create_promotion(from, to, .rook)) orelse return null;
            storage.store(Move.create_promotion(from, to, .bishop)) orelse return null;
            storage.store(Move.create_promotion(from, to, .knight)) orelse return null;
        }
    }

    fn store_castle(from: Square, to: Square, comptime castletype: CastleType, noalias storage: anytype) ?void {
        return storage.store(Move.create_castle(from, to, castletype));
    }

    /// Tricky one. An ep move can uncover a check.
    fn is_legal_enpassant(self: *const Position, comptime us: Color, king_sq: Square, from: Square, to: Square) bool {
        const them: Color = comptime us.opp();
        const capt_sq = if (us.e == .white) to.sub(8) else to.add(8);
        const occ: u64 = (self.all() ^ from.to_bitboard() ^ capt_sq.to_bitboard()) | to.to_bitboard();
        const att: u64 =
            (data.get_rook_attacks(king_sq, occ) & self.queens_rooks(them)) |
            (data.get_bishop_attacks(king_sq, occ) & self.queens_bishops(them));
        return att == 0;
    }

    fn is_legal_kingmove(self: *const Position, comptime us: Color, bb_without_king: u64, to: Square) bool {
        const them = comptime us.opp();
        return !self.is_square_attacked_by_for_occupation(bb_without_king, to, them);
    }

    /// Compares the kings path with unsafe squares.
    fn is_legal_castle(self: *const Position, comptime us: Color, comptime castletype: CastleType, bb_unsafe: u64) bool {
        const path: u64 = self.layout.castling_king_paths[us.u][castletype.u];
        return path & bb_unsafe == 0;
    }

    /// Checks for each square on the kings path if it is attacked.
    fn is_legal_castle_check_attacks(self: *const Position, comptime us: Color, comptime castletype: CastleType) bool {
        //_ = king_sq;
        const them: Color = comptime us.opp();
        var path: u64 = self.layout.castling_king_paths[us.u][castletype.u];
        while (path != 0) {
            const sq = pop_square(&path);
            if (self.is_square_attacked_by(sq, them)) return false;
        }
        return true;
    }

    /// Meant to be a validation after UCI position command.
    pub fn validate(self: *const Position) Error!void {

        const wk: u8 = popcnt(self.kings(Color.WHITE));
        const bk: u8 = popcnt(self.kings(Color.BLACK));

        if (wk == 0 or bk == 0) return Error.MissingKing;
        if (wk > 1 or bk > 1) return Error.TooManyKings;

        const wk_sq: Square = self.king_square(Color.WHITE);
        if (self.to_move.e == .black and self.is_square_attacked_by(wk_sq, Color.BLACK)) {
            return Error.InCheckAndNotToMove;
        }

        const bk_sq: Square = self.king_square(Color.BLACK);
        if (self.to_move.e == .white and self.is_square_attacked_by(bk_sq, Color.WHITE)) {
            return Error.InCheckAndNotToMove;
        }
        // For the rest everything is assumed to be ok.
    }

    /// ### Debug only.
    pub fn pos_ok(self: *const Position) bool {
        lib.not_in_release();

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

        const a = self.compute_hashkey();
        if (a != self.state.key) {
            lib.io.debugprint("KEY {} <> {} lastmove {s} {s}\n", .{ self.state.key, a, self.state.last_move.to_string().slice(), @tagName(self.state.last_move.type) });
            lib.io.debugprint("KEY\n", .{});
            return false;
        }

        // In check and not to move.
        const king_sq_white: Square = self.king_square(Color.WHITE);
        if (self.is_square_attacked_by(king_sq_white, Color.BLACK) and self.to_move.e != .white) {
            lib.io.debugprint("CHECK\n", .{});
            self.draw() catch wtf();
            return false;
        }

        const king_sq_black = self.king_square(Color.BLACK);
        if (self.is_square_attacked_by(king_sq_black, Color.WHITE) and self.to_move.e != .black) {
            lib.io.debugprint("CHECK\n", .{});
            self.draw() catch wtf();
            return false;
        }
        return true;
    }

    /// Zig-format. Writes the FEN string.
    pub fn format(self: *const Position, writer: *std.io.Writer) std.io.Writer.Error!void {
        const st = self.state;

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
                } else {
                    if (empty_squares > 0) {
                        try writer.print("{}", .{ empty_squares });
                        empty_squares = 0;
                    }
                    try writer.print("{u}", .{ pc.to_fen_char() });
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
        if (self.to_move.e == .white) {
            try writer.print(" w", .{});
        } else {
            try writer.print(" b", .{});
        }

        // Castling rights.
        try writer.print(" ", .{});
        if (st.castling_rights == 0) {
            try writer.print("-", .{});
        } else {
            if (st.castling_rights & cf_white_short != 0) try writer.print("K", .{});
            if (st.castling_rights & cf_white_long != 0)  try writer.print("Q", .{});
            if (st.castling_rights & cf_black_short != 0) try writer.print("k", .{});
            if (st.castling_rights & cf_black_long != 0)  try writer.print("q", .{});
        }

        // Enpassant.
        try writer.print(" ", .{});
        if (st.ep_square.u > 0) {
            try writer.print("{t}", .{st.ep_square.e});
        } else {
            try writer.print("-", .{});
        }

        // Draw counter.
        try writer.print(" {}", .{st.rule50});

        // Move number.
        const movenr: u16 = funcs.ply_to_movenumber(self.game_ply, self.to_move);
        try writer.print(" {}", .{movenr});
    }

    /// Prints the position diagram + information to the io.
    pub fn draw(self: *const Position) !void {
        // Pieces.
        try io.print_buffered("\n", .{});
        for (Square.all_for_printing) |square| {
            if (square.coord.file == 0) try io.print_buffered("{u}   ", .{square.char_of_rank()});
            const pc: Piece = self.get(square);
            const ch: u8 = if (pc.is_empty()) '.' else pc.to_print_char();
            try io.print_buffered("{u} ", .{ch});
            if (square.u % 8 == 7) try io.print_buffered("\n", .{});
        }
        try io.print_buffered("\n    a b c d e f g h\n\n", .{});

        // Info.
        //const move_str: []const u8 = if (self.state.last_move.is_empty()) "" else self.state.last_move.to_string().slice();
        try io.print_buffered("Fen: {f}\n", .{self});
        try io.print_buffered("Key: {x:0>16}\n", .{self.state.key});
        //try io.print_buffered("Last move: {s}\n", .{move_str});
        try io.print_buffered("Checkers: ", .{});
        if (self.state.checkers != 0) {
            var bb: u64 = self.state.checkers;
            while (bb != 0) {
                const sq: Square = pop_square(&bb);
                try io.print_buffered("{t} ", .{sq.e});
            }
        }
        try io.print_buffered("\n", .{});
        try io.flush();
    }

    pub fn print_history(self: *const Position) !void {
        var reversed_stateinfo_list: std.ArrayList(*const StateInfo) = .empty;
        defer reversed_stateinfo_list.deinit(ctx.galloc);

        var movenr: u16 = self.game_ply;
        var stm: Color = self.to_move;
        var curr_state: *const StateInfo = self.state;
        while (true) {
            if (curr_state.last_move.is_empty()) break;
            stm = stm.opp();
            movenr -|= 1;
            reversed_stateinfo_list.append(ctx.galloc, curr_state) catch wtf();
            curr_state = curr_state.prev orelse break;
        }

        // TODO: print "..." when black to move
        var iter = std.mem.reverseIterator(reversed_stateinfo_list.items);
        var i: usize = 0;
        movenr = funcs.ply_to_movenumber(movenr, stm);
        while (iter.next()) |st| {
            const m = st.last_move;
            if (stm.e == .white) {
                try io.print_buffered("{}. ", .{ movenr});
                movenr += 1;
            }
            try io.print_buffered("{s} ", .{ m.to_string().slice()});
            stm = stm.opp();
            i += 1;
        }
        try io.print_buffered("\n", .{});
        try io.flush();
    }

    pub fn print_history2(self: *const Position) !void {

        var iter: PositionStateIterator = .init(self);
        while (iter.next()) |st| {
            try io.print_buffered("{f} ", .{ st.last_move });
        }
        try io.flush();

        // var st: *const StateInfo = self.state;

        // while (st.prev) |p| {
        //     st = p;
        // }

        // while (st.next) |n| {
        //     try io.print_buffered("{f} ", .{ n.last_move });
        //     st = n;
        // }
        // try io.flush();
    }

    /// ### Debug only.
    /// Compares everything except `state`, `ply` and `game_ply`.
    /// * The inner contents of the state *are* compared.
    /// * If `check_moves` then we also check if the generated moves are the same.
    pub fn equals(self: *const Position, other: *const Position, comptime check_moves: bool) bool {
        lib.not_in_release();

        // Types not directly binary comparable with Zig std.
        inline for (0..64) |i| if (self.board[i].u != other.board[i].u) return false;
        inline for (0..2) |i| if (self.layout.king_start_squares[i].e != other.layout.king_start_squares[i].e) return false;
        inline for (0..2, 0..2) |i, j| if (self.layout.rook_start_squares[i][j].e != other.layout.rook_start_squares[i][j].e) return false;

        const eql: bool =
            std.meta.eql(self.bb_by_type, other.bb_by_type) and
            std.meta.eql(self.bb_by_color, other.bb_by_color) and
            std.meta.eql(self.layout.castling_between_bitboards, other.layout.castling_between_bitboards) and
            std.meta.eql(self.layout.castling_king_paths, other.layout.castling_king_paths) and
            std.meta.eql(self.layout.castling_masks, other.layout.castling_masks) and
            std.meta.eql(self.values, other.values) and
            std.meta.eql(self.materials, other.materials) and
            //self.ply == other.ply and
            self.is_960 == other.is_960 and
            self.nullmove_state == other.nullmove_state and
            self.state.equals(other.state);

        if (!eql) return false;

        if (check_moves) {
            var store1: MoveStorage = .init();
            var store2: MoveStorage = .init();
            self.lazy_generate_moves(&store1);
            self.lazy_generate_moves(&store2);
            const ok: bool = std.mem.eql(Move, store1.slice(), store2.slice());
            return ok;
        }
        else return true;
    }
};

/// 4 bits comptime struct for generating moves.
pub const Params = packed struct {
    /// Only generate captures and promotions. Used in quiet search.
    /// * When in check this is mostly ignored. In that case we generate all moves (evasions).
    /// * Whether in check or not: when captures is true we only generate queen promotions.
    captures: bool = false,
    /// The color for which we are generating.
    us: Color = Color.WHITE,
    /// Are we in check?
    check: bool = false,
    /// There are pins. If not we can comptime skip all pin checks.
    pins: bool = false,

    fn create(comptime captures: bool, comptime us: Color, comptime check: bool, comptime pins: bool) Params {
        return Params{ .captures = captures, .us = us, .check = check, .pins = pins };
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
    /// The (exactly defined) move to find.
    to_find: Move,
    found: bool,

    pub fn init(m: Move) MoveFinder {
        return .{ .to_find = m, .found = false };
    }

    /// Required function.
    pub fn reset(self: *MoveFinder) void {
        self.found = false;
    }

    /// Required function.
    pub fn store(self: *MoveFinder, move: Move) ?void {
        if (self.to_find == move) {
            self.found = true;
            return null;
        }
    }
};

pub const PositionStateIterator = struct {
    root: *const StateInfo,
    st: ?*const StateInfo,

    pub fn init(pos: *const Position) PositionStateIterator {
        const root: *const StateInfo = get_root(pos);
        return .{
            .root = root,
            .st = root,
        };
    }

    pub fn next(self: *PositionStateIterator) ?*const StateInfo {
        if (self.st == null) return null;
        self.st = self.st.?.next;
        return self.st;
    }

    pub fn peek(self: *PositionStateIterator) bool {
        return self.st != null and self.st.?.next != null;
    }

    pub fn reset(self: *PositionStateIterator) void {
        self.st = self.root;
    }

    fn get_root(pos: *const Position) *const StateInfo {
        var root: *const StateInfo = pos.state;
        while (root.prev) |p| {
            root = p;
        }
        return root;
    }
};

pub const Error = error {
    MissingKing,
    TooManyKings,
    InCheckAndNotToMove,
};