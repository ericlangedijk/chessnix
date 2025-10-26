// zig fmt: off

const std = @import("std");
const types = @import("types.zig");
const lib = @import("lib.zig");
const chess960 = @import("chess960.zig");
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

const P = types.P;
const N = types.N;
const B = types.B;
const R = types.R;
const Q = types.Q;
const K = types.K;

pub const fen_startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
pub const fen_kiwipete = "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1";

// castling flags.
pub const cf_white_short: u4 = 0b0001;
pub const cf_white_long: u4 = 0b0010;
pub const cf_black_short: u4 = 0b0100;
pub const cf_black_long: u4 = 0b1000;
pub const cf_all: u4 = 0b1111;

// gen flags.
const GF_CASTLE   : u4 =  0b0001; // updated
const GF_PINS     : u4 =  0b0010; // updated
const GF_EP       : u4 =  0b0100; // updated
const GF_CHECK    : u4 =  0b1000; // updated
const GF_CAPTURES : u5 = 0b10000;

/// Indexing [color][castletype].
pub const king_castle_destination_squares: [2][2]Square = .{ .{ .G1, .C1 }, .{ .G8, .C8 } };

/// Indexing [color][castletype].
pub const rook_castle_destination_squares: [2][2]Square = .{ .{ .F1, .D1 }, .{ .F8, .D8 } };

/// Indexing [color][castletype].
pub const castle_flags: [2][2]u4 = .{ .{ cf_white_short, cf_white_long }, .{ cf_black_short, cf_black_long } };

/// The initial layout for `Position`, supporting Chess960.
pub const Layout = struct {
    // 0...959
    //nr: u16,
    /// The startfiles of [0] left rook, [1] right rook, [2] king file.
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

    const bb = bitboards;
    pub const classic: Layout = .{
        //.nr = 518,
        .start_files = .{ bb.file_a, bb.file_h, bb.file_e },
        .king_start_squares = .{ Square.E1, Square.E8 },
        .rook_start_squares = .{ .{ .H1, .A1 }, .{ .H8, .A8 } },
        .castling_between_bitboards = .{ .{ bb.bb_f1 | bb.bb_g1, bb.bb_d1 | bb.bb_c1 | bb.bb_b1 }, .{ bb.bb_f8 | bb.bb_g8, bb.bb_d8 | bb.bb_c8 | bb.bb_b8 } },
        .castling_king_paths = .{ .{ bb.bb_f1 | bb.bb_g1, bb.bb_d1 | bb.bb_c1 }, .{ bb.bb_f8 | bb.bb_g8, bb.bb_d8 | bb.bb_c8 } },
        .castling_masks = .{ 2, 0, 0, 0, 3, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 8, 0, 0, 0, 12, 0, 0, 4 },
    };

    /// All chess 960 layouts. Key = backrow, Value = chess960 nr.
    //var backrow_mapping: std.hash_map.AutoHashMapUnmanaged([8]u4, u16) = .empty;
    var start_file_mapping: std.hash_map.AutoHashMapUnmanaged([3]u3, Layout) = .empty;

    /// Must be called on startup.
    pub fn initialize() void {
        start_file_mapping.put(ctx.galloc, classic.start_files, classic) catch wtf();
        //all = .empty; //init(ctx.galloc); // std.hash_map.AutoHashMapUnmanaged([3]u3, Layout);
        // for (chess960.all_backrows, 0..) |backrow, nr| {
        //     backrow_mapping.put(ctx.galloc, backrow, @truncate(nr)) catch wtf();
        //     if (nr == 518) {
        //         layout_mapping.put(ctx.galloc, backrow, classic) catch wtf();
        //     }
        // }
        //
    }

    /// Must be called on exit.
    pub fn finalize() void {
        // backrow_mapping.deinit(ctx.galloc);
        start_file_mapping.deinit(ctx.galloc);
    }

    /// Currently Load on demand.
    // pub fn get_layout_ptr(backrow: [8]u4) *const Layout {
    //     //const r: [8]PieceType = @bitCast(backrow);
    //     //_ = r;
    //     const result = layout_mapping.getOrPut(ctx.galloc, backrow) catch wtf();
    //     // Not found: intialize layout
    //     if (!result.found_existing) {
    //         // const nr: u16 = backrow_mapping.get(backrow);
    //     }
    //     return result.value_ptr;
    // }

    // pub fn get_layout_ptr2(files: [3]u3) *const Layout {
    //     //const r: [8]PieceType = @bitCast(backrow);
    //     //_ = r;
    //     const result = layout_mapping.getOrPut(ctx.galloc, backrow) catch wtf();
    //     // Not found: intialize layout
    //     if (!result.found_existing) {
    //         // const nr: u16 = backrow_mapping.get(backrow);
    //     }
    //     return result.value_ptr;
    // }


    pub fn compute_layout(files: [3]u3) Layout {

        // TODO: assert files ok.

        const WHITE = Color.WHITE.u;
        const BLACK = Color.BLACK.u;
        const O_O = CastleType.SHORT.u;
        const O_O_O = CastleType.LONG.u;

        var l: Layout = undefined;
        var rook: Square = undefined;
        var king: Square = undefined;

        l.start_files = files;

        // King start.
        l.king_start_squares[WHITE] = .from_rank_file(bitboards.rank_1, files[2]);
        l.king_start_squares[BLACK] = .from_rank_file(bitboards.rank_8, files[2]);

        // Rook start.
        l.rook_start_squares[WHITE][O_O]   = .from_rank_file(bitboards.rank_1, files[1]);
        l.rook_start_squares[WHITE][O_O_O] = .from_rank_file(bitboards.rank_1, files[0]);
        l.rook_start_squares[BLACK][O_O]   = .from_rank_file(bitboards.rank_8, files[1]);
        l.rook_start_squares[BLACK][O_O_O] = .from_rank_file(bitboards.rank_8, files[0]);

        // White short.
        rook = l.rook_start_squares[WHITE][O_O];
        king = l.king_start_squares[WHITE];
        l.castling_between_bitboards[WHITE][O_O] = bitboards.get_squarepair(king, rook).ray & ~rook.to_bitboard();
        l.castling_king_paths[WHITE][O_O]        = bitboards.get_squarepair(king, king_castle_destination_squares[WHITE][O_O]).ray;

        // White long.
        rook = l.rook_start_squares[WHITE][O_O_O];
        king = l.king_start_squares[WHITE];
        l.castling_between_bitboards[WHITE][O_O_O] = bitboards.get_squarepair(king, rook).ray & ~rook.to_bitboard();
        l.castling_king_paths[WHITE][O_O_O]        = bitboards.get_squarepair(king, king_castle_destination_squares[WHITE][O_O_O]).ray;


        // Black short.
        rook = l.rook_start_squares[BLACK][O_O];
        king = l.king_start_squares[BLACK];
        l.castling_between_bitboards[BLACK][O_O] = bitboards.get_squarepair(king, rook).ray & ~rook.to_bitboard();
        l.castling_king_paths[BLACK][O_O]        = bitboards.get_squarepair(king, king_castle_destination_squares[BLACK][O_O]).ray;

        // Black long.
        rook = l.rook_start_squares[BLACK][O_O_O];
        king = l.king_start_squares[BLACK];
        l.castling_between_bitboards[BLACK][O_O_O] = bitboards.get_squarepair(king, rook).ray & ~rook.to_bitboard();
        l.castling_king_paths[BLACK][O_O_O]        = bitboards.get_squarepair(king, king_castle_destination_squares[BLACK][O_O_O]).ray;

        // Flags.
        l.castling_masks = @splat(0);
        // White.
        l.castling_masks[l.king_start_squares[WHITE].u] = cf_white_short | cf_white_long;
        l.castling_masks[l.rook_start_squares[WHITE][O_O_O].u] = cf_white_long;
        l.castling_masks[l.rook_start_squares[WHITE][O_O].u] = cf_white_short;
        // Black.
        l.castling_masks[l.king_start_squares[BLACK].u] = cf_black_short | cf_black_long;
        l.castling_masks[l.rook_start_squares[BLACK][O_O_O].u] = cf_black_long;
        l.castling_masks[l.rook_start_squares[BLACK][O_O].u] = cf_black_short;

        return l;
    }
};

pub const Position = struct {
    pub const empty: Position = .init_empty();
    pub const empty_classic: Position = .init_empty_classic();

    /// The initial layout, supporting Chess960.
    layout: *const Layout,
    /// The pieces on the 64 squares.
    board: [64]Piece,
    /// Bitboards occupation indexed by PieceType. [0...5] piecetypes.
    bitboards_by_type: [6]u64,
    /// Bitboards occupation indexed by color: [0] white pieces, [1] black pieces.
    bitboards_by_color: [2]u64,
    /// All pieces.
    bitboard_all: u64,
    /// Piece values sum. [0] white, [1] black
    /// TODO: replace by scorepairs.
    values: [2]Value,
    /// Material values sum. [0] white, [1] black
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
    /// The enpassant square.
    ep_square: Square,
    /// Bitflags for castlingrights: cf_white_short, cf_white_long, cf_black_short, cf_black_long.
    castling_rights: u4,
    /// The move that was played to reach the current position.
    last_move: Move,
    /// The board hashkey.
    key: u64,
    /// The hashkey of pawns.
    pawnkey: u64,
    /// Bitboard of the pieces that currently give check.
    checkers: u64,
    /// The paths from the enemy slider checkers to the king (excluding the king, including the checker). Pawns and knights included.
    checkmask: u64,
    /// Bitboards with the diagonal pin rays (excluding the king, including the attacker).
    pins_diagonal: u64,
    /// Bitboards with the orthogonal pin rays (excluding the king, including the attacker).
    pins_orthogonal: u64,
    /// All pins.
    pins: u64,

    fn init_empty() Position {
        return .{
            .layout = &Layout.classic,
            .board = @splat(Piece.NO_PIECE),
            .bitboards_by_type = @splat(0),
            .bitboards_by_color = @splat(0),
            .bitboard_all = 0,
            .values = @splat(0),
            .phase = 0,
            .stm = Color.WHITE,
            .ply = 0,
            .ply_from_root = 0,
            .game_ply = 0,
            .nullmove_state = false,
            .rule50 = 0,
            .ep_square = Square.zero,
            .castling_rights = 0,
            .last_move = .empty,
            .key = 0,
            .pawnkey = 0,
            .checkers = 0,
            .checkmask = 0,
            .pins_diagonal = 0,
            .pins_orthogonal = 0,
            .pins = 0,
        };
    }

    fn init_empty_classic() Position {
        const b = bitboards;
        const v_sum: Value = (PieceType.PAWN.value() * 8) + (PieceType.KNIGHT.value() * 2) + (PieceType.BISHOP.value() * 2) + (PieceType.ROOK.value() * 2) + PieceType.QUEEN.value() + PieceType.KING.value();

        return .{
            .layout = &Layout.classic,
            .board = create_board_from_backrow(.{ R, N, B, Q, K, B, N, R }),
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
            .values = .{ v_sum, v_sum },
            .phase = types.max_phase,
            .stm = Color.WHITE,
            .ply = 0,
            .ply_from_root = 0,
            .game_ply = 0,
            .nullmove_state = false,
            .rule50 = 0,
            .ep_square = Square.zero,
            .castling_rights = 0b1111,
            .last_move = .empty,
            .key = 0,
            .pawnkey = 0,
            .checkers = 0,
            .checkmask = 0,
            .pins_diagonal = 0,
            .pins_orthogonal = 0,
            .pins = 0,
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
    pub fn set(self: *Position, fen_str: []const u8) !void {
        const state_board: u8 = 0;
        const state_color: u8 = 1;
        const state_castle: u8 = 2;
        const state_ep: u8 = 3;
        const state_draw_count: u8 = 4;
        const state_movenumber: u8 = 5;

        // uci is started with "setoption name UCI_Chess960 value true" before ucinewgame.
        self.* = .empty;

        var parse_state: u8 = state_board;
        var rank: u3 = bitboards.rank_8;
        var file: u3 = bitboards.file_a;

        // // Assume classic
        // var is_classic: bool = true;

        // // Chess 960 optional.
        // var startfiles: [3]u4 = .{ 0, 0, 0 }; // left rook, right rook, king

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
                            'K' => self.castling_rights |= cf_white_short,
                            'Q' => self.castling_rights |= cf_white_long,
                            'k' => self.castling_rights |= cf_black_short,
                            'q' => self.castling_rights |= cf_black_long,
                            // 'A'...'H' => {
                            //     if (self.castling_rights & cf_white_short == 0) {
                            //         self.castling_rights |= cf_white_short;
                            //     }
                            //     else {
                            //         self.castling_rights |= cf_white_long;
                            //     }
                            // },
                            // 'a'...'h' => {
                            //     if (self.castling_rights & cf_black_short == 0) {
                            //         self.castling_rights |= cf_black_short;
                            //     }
                            //     else {
                            //         self.castling_rights |= cf_black_long;
                            //     }
                            // },
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
        self.lazy_update_state();
    }

    /// Parses a uci-move.
    pub fn parse_move(self: *const Position, str: []const u8) types.ParsingError!Move {
        if (str.len < 4 or str.len > 5) return types.ParsingError.IllegalMove;

        // TODO: in the case of chess960 castling is like we encode it in here: king takes rook.
        const us = self.stm;
        const from: Square = Square.from_string(str[0..2]);
        var to: Square = Square.from_string(str[2..4]);
        var prom_flags: u4 = 0;

        // No promotion.
        if (str.len == 4) {
            // Castling.
            if (from.u == self.layout.king_start_squares[us.u].u and self.board[from.u].e == Piece.create(K, us).e) {
                if (to.e == Square.G1.e or to.e == Square.G8.e) {
                    to = self.layout.rook_start_squares[us.u][CastleType.SHORT.u]; // King takes rook.
                }
                else if (to.e == Square.C1.e or to.e == Square.C8.e) {
                    to = self.layout.rook_start_squares[us.u][CastleType.LONG.u]; // King takes rook.
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
        self.lazy_generate_moves(&finder);
        if (finder.found()) return finder.move; // return the exact move found.
        return types.ParsingError.IllegalMove;
    }

    /// A convenient (faster) way to set the startposition without the need for a fen string.
    pub fn set_startpos(self: *Position) void {
        self.* = empty_classic;
        self.init_hash();
    }

    /// Flips the board.
    pub fn flip(self: *Position) void {
        const bk: [64]Piece = self.board;
        self.board = @splat(Piece.NO_PIECE);
        var bb = self.all();
        var k: u64 = 0;
        var p: u64 = 0;
        while (bb != 0) {
            const sq: Square = pop_square(&bb);
            const pc: Piece = bk[sq.u];
            const new_sq: Square = sq.flipped();
            const new_pc: Piece = pc.opp();
            self.board[new_sq.u] = new_pc;
            k ^= zobrist.piece_square(new_pc, new_sq);
            if (new_pc.is_pawn()) {
                p ^= zobrist.piece_square(new_pc, new_sq);
            }
        }

        std.mem.swap(Value, &self.values[0], &self.values[1]);
        //std.mem.swap(Value, &self.materials[0], &self.materials[1]);
        // NB: phase does not change with flipping.

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
            k ^= zobrist.btm();
        }

        self.castling_rights = (self.castling_rights >> 2) | (self.castling_rights << 2);
        k ^= zobrist.castling(self.castling_rights);

        if (self.ep_square.u > 0) {
            self.ep_square = self.ep_square.flipped();
            k ^= zobrist.enpassant(self.ep_square);
        }

        self.key = k;
        self.pawnkey = p;
        self.last_move = self.last_move.flipped();
    }

    fn init_hash(self: *Position) void {
        self.compute_hashkeys(&self.key, &self.pawnkey);
    }

    fn compute_hashkeys(self: *const Position, k: *u64, p: *u64)void {
        k.* = 0;
        p.* = 0;
        // Loop through occupied squares.
        var occ: u64 = self.all();
        while (occ != 0) {
            const sq = pop_square(&occ);
            const pc = self.get(sq);
            const z_key: u64 = zobrist.piece_square(pc, sq);
            k.* ^= z_key;
            if (pc.is_pawn()) {
                p.* ^= z_key;
            }
        }
        k.* ^= zobrist.castling(self.castling_rights);
        if (self.ep_square.u > 0) k.* ^= zobrist.enpassant(self.ep_square);
        if (self.stm.e == .black) k.* ^= zobrist.btm();
    }

    pub fn get(self: *const Position, sq: Square) Piece {
        return self.board[sq.u];
    }

    pub fn is_check(self: *const Position) bool {
        return self.state.checkers > 0;
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


    /// All pieces except pawns and kings.
    pub fn all_except_pawns_and_kings(self: *const Position) u64 {
        return self.all() & ~self.by_type(PieceType.PAWN) & ~self.all_kings();
    }

    pub fn pawns(self: *const Position, us: Color) u64 {
        return self.by_type(P) & self.by_color(us);
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

    /// Also updates hashkey.
    fn add_fen_piece(self: *Position, pc: Piece, sq: Square) void {
        switch (pc.color().e) {
            .white => self.add_piece(Color.WHITE, pc, sq),
            .black => self.add_piece(Color.BLACK, pc, sq),
        }
        self.key ^= zobrist.piece_square(pc, sq);
        if (pc.is_pawn()) {
            self.pawnkey ^= zobrist.piece_square(pc, sq);
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
        if (us.e == .white) self.bitboards_by_type[pc.u] |= mask else self.bitboards_by_type[pc.u - 6] |= mask;
        self.bitboards_by_color[us.u] |= mask;
        self.bitboard_all |= mask;
        self.values[us.u] += pc.value();
        self.phase += types.phase_table[pc.u];
    }

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
        self.values[us.u] -= pc.value();
        self.phase -= types.phase_table[pc.u];
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
        //const is_pawnmove = if (us.e == .white) pc.e == .w_pawn else pc.e == .b_pawn;
        const hash_delta = zobrist.piece_square(pc, from) ^ zobrist.piece_square(pc, to);

        // Local key for updating.
        // Clear ep  by default. Note that the zobrist for square a1 is 0 so this xor is safe.
        var key: u64 = self.key ^ zobrist.btm() ^ zobrist.enpassant(self.ep_square);

        // Update some stuff.
        self.last_move = m;
        self.stm = them;
        self.ply += 1;
        self.ply_from_root += 1;
        self.game_ply += 1;
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

        // Switch is in numerical order
        sw: switch (m.flags) {
            Move.silent => {
                self.move_piece(us, pc, from, to);
                key ^= hash_delta;
                if (pc.is_pawn_of_color(us)) {
                    self.pawnkey ^= hash_delta;
                    self.rule50 = 0;
                }
                else {
                    self.rule50 += 1;
                }
            },
            Move.double_push => {
                self.move_piece(us, pc, from, to);
                key ^= hash_delta;
                self.pawnkey ^= hash_delta;
                self.rule50 = 0;
                // Only set ep if valid.
                if (bitboards.ep_masks[to.u] & self.pawns(them) != 0) {
                    const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                    self.ep_square = ep;
                    key ^= zobrist.enpassant(ep);
                }
            },
            Move.castle_short => {
                const king: Piece = comptime Piece.create(K, us);
                const rook: Piece = comptime Piece.create(R, us);
                const king_to: Square = comptime king_castle_destination_squares[us.u][CastleType.SHORT.u];
                const rook_to: Square = comptime rook_castle_destination_squares[us.u][CastleType.SHORT.u]; // king takes rook
                self.move_piece(us, king, from, king_to);
                self.move_piece(us, rook, to, rook_to);
                key ^= zobrist.piece_square(king, from) ^ zobrist.piece_square(king, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
                self.rule50 += 1;
            },
            Move.castle_long => {
                const king: Piece = comptime Piece.create(K, us);
                const rook: Piece = comptime Piece.create(R, us);
                const king_to: Square = comptime king_castle_destination_squares[us.u][CastleType.LONG.u];
                const rook_to: Square = comptime rook_castle_destination_squares[us.u][CastleType.LONG.u]; // king takes rook
                self.move_piece(us, king, from, king_to);
                self.move_piece(us, rook, to, rook_to);
                key ^= zobrist.piece_square(king, from) ^ zobrist.piece_square(king, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
                self.rule50 += 1;
            },
            Move.knight_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.create(N, us);
                const pawn: Piece = comptime Piece.create(P, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                self.pawnkey ^= zobrist.piece_square(pawn, from);
            },
            Move.bishop_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.create(B, us);
                const pawn: Piece = comptime Piece.create(P, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                self.pawnkey ^= zobrist.piece_square(pawn, from);
            },
            Move.rook_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.create(R, us);
                const pawn: Piece = comptime Piece.create(P, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                self.pawnkey ^= zobrist.piece_square(pawn, from);
            },
            Move.queen_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.create(Q, us);
                const pawn: Piece = comptime Piece.create(P, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                self.pawnkey ^= zobrist.piece_square(pawn, from);
            },
            Move.capture => {
                self.rule50 = 0;
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                self.move_piece(us, pc, from, to);
                key ^= zobrist.piece_square(capt, to) ^ hash_delta;
                if (pc.is_pawn_of_color(us)) {
                    self.pawnkey ^= hash_delta;
                }
                if (capt.is_pawn_of_color(them)) {
                    self.pawnkey ^= zobrist.piece_square(capt, to);
                }
            },
            Move.ep => {
                self.rule50 = 0;
                const pawn_us: Piece = comptime Piece.create(P, us);
                const pawn_them: Piece = comptime Piece.create(P, them);
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.remove_piece(them, pawn_them, capt_sq);
                self.move_piece(us, pawn_us, from, to);
                key ^= hash_delta ^ zobrist.piece_square(pawn_them, capt_sq);
                self.pawnkey ^= zobrist.piece_square(pawn_us, from) ^ zobrist.piece_square(pawn_us, to) ^ zobrist.piece_square(pawn_them, capt_sq);
            },
            Move.knight_promotion_capture => {
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                key ^= zobrist.piece_square(capt, to);
                continue :sw Move.knight_promotion;
            },
            Move.bishop_promotion_capture => {
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                key ^= zobrist.piece_square(capt, to);
                continue :sw Move.bishop_promotion;
            },
            Move.rook_promotion_capture => {
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                key ^= zobrist.piece_square(capt, to);
                continue :sw Move.rook_promotion;
            },
            Move.queen_promotion_capture => {
                const capt: Piece = self.board[to.u];
                self.remove_piece(them, capt, to);
                key ^= zobrist.piece_square(capt, to);
                continue :sw Move.queen_promotion;
            },
            else => {
                unreachable;
            },
        }

        self.key = key;
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
        self.last_move = Move.nullmove;
        self.stm = them;
        self.ply += 1;
        self.ply_from_root += 1;
        self.game_ply += 1;
        self.ep_square = Square.zero;
        self.update_state(them);
    }

    /// Update checks and pins for stm.
    fn update_state(self: *Position, comptime us: Color) void {
        const them: Color = comptime us.opp();
        const bb_all: u64 = self.all();
        const bb_us: u64 = self.by_color(us);

        self.pins_orthogonal = 0;
        self.pins_diagonal = 0;

        self.checkmask =
            (attacks.get_pawn_attacks(self.king_square(us), us) & self.pawns(them)) |
            (attacks.get_knight_attacks(self.king_square(us)) & self.knights(them));

        const our_king_sq: Square = self.king_square(us);
        const bb_occ_without_us: u64 = bb_all ^ self.by_color(us);
        var candidate_slider_attackers: u64 =
            (attacks.get_bishop_attacks(our_king_sq, bb_occ_without_us) & self.queens_bishops(them)) |
            (attacks.get_rook_attacks(our_king_sq, bb_occ_without_us) & self.queens_rooks(them));

        // Use candidate attackers for both checkers and pins.
        while (candidate_slider_attackers != 0) {
            const attacker_sq: Square = pop_square(&candidate_slider_attackers);
            const pair: *const bitboards.SquarePair = bitboards.get_squarepair(our_king_sq, attacker_sq);
            const bb_ray: u64 = pair.ray & bb_us;
            // We have a slider checker when there is nothing in between.
            if (bb_ray == 0) {
                self.checkmask |= pair.ray;
            }
            // We have a pin when exactly 1 bit is set. There is one piece in between.
            else if (bb_ray & (bb_ray - 1) == 0) {
                switch (pair.axis) {
                    .orth => self.pins_orthogonal |= pair.ray,
                    .diag => self.pins_diagonal |= pair.ray,
                    else => unreachable,
                }
            }
        }
        self.checkers = self.checkmask & bb_all;
        self.pins = self.pins_diagonal | self.pins_orthogonal;
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
        return
            (attacks.get_knight_attacks(to) & self.all_knights()) |
            (attacks.get_king_attacks(to) & self.all_kings()) |
            (attacks.get_pawn_attacks_combined(to)) |
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
        while (their_knights != 0) {
            const from: Square = pop_square(&their_knights);
            att |= attacks.get_knight_attacks(from);
        }

        // Diagonal sliders.
        var their_diag_sliders = self.queens_bishops(attacker);
        while (their_diag_sliders != 0) {
            const from: Square = pop_square(&their_diag_sliders);
            att |= attacks.get_bishop_attacks(from, occ);
        }

        // Orthogonal sliders.
        var their_orth_sliders = self.queens_rooks(attacker);
        while (their_orth_sliders != 0) {
            const from: Square = pop_square(&their_orth_sliders);
            att |= attacks.get_rook_attacks(from, occ);
        }

        // King.
        att |= attacks.get_king_attacks(self.king_square(attacker));

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
        switch (self.stm.e) {
            .white => self.generate_moves(Color.WHITE, storage),
            .black => self.generate_moves(Color.BLACK, storage),
        }
    }

    pub fn lazy_generate_captures(self: *const Position, noalias storage: anytype) void {
        switch (self.stm.e) {
            .white => self.generate_captures(Color.WHITE, storage),
            .black => self.generate_captures(Color.BLACK, storage),
        }
    }

    pub  fn generate_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void {
        if (comptime lib.is_paranoid) assert(self.stm.e == us.e);
        const check: bool = self.checkers != 0;
        const pins: bool = self.pins != 0;
        switch (check) {
            false => switch (pins) {
                false => self.gen(comptime Params.create(false, us, false, false), storage),
                true  => self.gen(comptime Params.create(false, us, false, true), storage),
            },
            true => switch (pins) {
                false => self.gen(comptime Params.create(false, us, true, false), storage),
                true  => self.gen(comptime Params.create(false, us, true, true), storage),
            },
        }
    }

    /// Generate captures only, but if in check generate all.
    pub  fn generate_captures(noalias self: *const Position, comptime us: Color, noalias storage: anytype) void {
        if (comptime lib.is_paranoid) assert(self.stm.e == us.e);
        const check: bool = self.checkers != 0;
        const pins: bool = self.pins != 0;
        switch (check) {
            false => switch (pins) {
                false => self.gen(comptime Params.create(true, us, false, false), storage),
                true  => self.gen(comptime Params.create(true, us, false, true), storage),
            },
            true => switch (pins) {
                false => self.gen(comptime Params.create(true, us, true, false), storage),
                true  => self.gen(comptime Params.create(true, us, true, true), storage),
            },
        }
    }

    /// See `MoveStorage` for the interface of `storage`: required are the functions `reset()` and `store()`.
    fn gen(self: *const Position, comptime ctp: Params, noalias storage: anytype) void {
        storage.reset();

        const us = comptime ctp.us;
        const has_pins: bool = comptime ctp.pins;
        const check: bool = comptime ctp.check;
        const captures: bool = comptime ctp.captures;

        const them = comptime us.opp();
        const do_all_promotions: bool = comptime !captures;

        const doublecheck: bool = check and popcnt(self.checkers) > 1;
        const bb_all: u64 = self.all();
        const bb_us: u64 = self.by_color(us);
        const bb_them: u64 = self.by_color(them);
        const bb_not_us: u64 = ~bb_us;
        const king_sq: Square = self.king_square(us);

        var bb: u64 = undefined;
        var bb_to: u64 = undefined;
        var from: Square = undefined;
        var to: Square = undefined;
        var flag: u4 = undefined;

        // In case of a doublecheck we can only move the king.
        if (!doublecheck) {
            const our_pawns = self.pawns(us);
            const our_knights = self.knights(us);
            const our_queens_bishops = self.queens_bishops(us);
            const our_queens_rooks = self.queens_rooks(us);

            const target = if (check) self.checkmask else if (!captures) bb_not_us else bb_them;

            // Pawns.
            if (our_pawns != 0) {
                const third_rank: u64 = comptime funcs.relative_rank_3_bitboard(us);
                const last_rank: u64 = comptime funcs.relative_rank_8_bitboard(us);
                const empty_squares: u64 = ~bb_all;
                const enemies: u64 = if (check) self.checkers else bb_them;

                // Generate all 4 types of pawnmoves: push, push double, capture left, capture right.
                var bb_single = switch(has_pins) {
                    false => (pawns_shift(our_pawns, us, .up) & empty_squares),
                    true  => (pawns_shift(our_pawns & ~self.pins, us, .up) & empty_squares) |
                             (pawns_shift(our_pawns & self.pins_diagonal, us, .up) & empty_squares & self.pins_diagonal) |
                             (pawns_shift(our_pawns & self.pins_orthogonal, us, .up) & empty_squares & self.pins_orthogonal)
                };

                var bb_double = pawns_shift(bb_single & third_rank, us, .up) & empty_squares;

                // Pawn push check interpolation.
                bb_single &= target;
                bb_double &= target;

                const bb_northwest: u64 = switch (has_pins) {
                    false => (pawns_shift(our_pawns, us, .northwest) & enemies),
                    true  => (pawns_shift(our_pawns & ~self.pins, us, .northwest) & enemies) |
                             (pawns_shift(our_pawns & self.pins_diagonal, us, .northwest) & enemies & self.pins_diagonal),
                };

                const bb_northeast: u64 = switch (has_pins) {
                    false => (pawns_shift(our_pawns, us, .northeast) & enemies),
                    true  => (pawns_shift(our_pawns & ~self.pins, us, .northeast) & enemies) |
                             (pawns_shift(our_pawns & self.pins_diagonal, us, .northeast) & enemies & self.pins_diagonal),
                };

                // Pawn pushes.
                if (check or !captures) {
                    // Single push normal
                    bb = bb_single & ~last_rank;
                    while (bb != 0) {
                        to = pop_square(&bb);
                        from = pawn_from(to, us, .up);
                        store(from, to, Move.silent, storage) orelse return;
                    }
                    // Double.push
                    bb = bb_double;
                    while (bb != 0) {
                        to = pop_square(&bb);
                        from = if (us.e == .white) to.sub(16) else to.add(16);
                        store(from, to, Move.double_push, storage) orelse return;
                    }
                }

                // left capture promotions
                bb = bb_northwest & last_rank;
                while (bb != 0) {
                    to = pop_square(&bb);
                    from = pawn_from(to, us, .northwest);
                    store_promotions(do_all_promotions, from, to, true, storage) orelse return;
                }

                // right capture promotions
                bb = bb_northeast & last_rank;
                while (bb != 0) {
                    to = pop_square(&bb);
                    from = pawn_from(to, us, .northeast);
                    store_promotions(do_all_promotions, from, to, true, storage) orelse return;
                }

                // push promotions
                bb = bb_single & last_rank;
                while (bb != 0) {
                    to = pop_square(&bb);
                    from =  pawn_from(to, us, .up);
                    store_promotions(do_all_promotions, from, to, false, storage) orelse return;
                }

                // left normal captures,
                bb =  bb_northwest & ~last_rank;
                while (bb != 0) {
                    to = pop_square(&bb);
                    from = pawn_from(to, us, .northwest);
                    store(from, to, Move.capture, storage) orelse return;
                }

                // right normal captures,
                bb = bb_northeast & ~last_rank;
                while (bb != 0) {
                    to = pop_square(&bb);
                    from = pawn_from(to, us, .northeast);
                    store(from, to, Move.capture, storage) orelse return;
                }

                const ep: Square = self.ep_square;
                // Enpassant.
                if (ep.u > 0) {
                    bb = attacks.get_pawn_attacks(ep, them) & our_pawns; // inversion trick.
                    inline for (0..2) |_| {
                        if (bb == 0) break;
                        from = pop_square(&bb);
                        if (self.is_legal_enpassant(us, king_sq, from, ep))
                            store(from, ep, Move.ep, storage) orelse return;
                    }
                }
            } // (pawns)

            // Knights.
            bb = if (!has_pins) our_knights else our_knights & ~self.pins; // A knight can never escape a pin.
            while (bb != 0) {
                from = pop_square(&bb);
                bb_to = attacks.get_knight_attacks(from) & target;
                inline for (0..8) |_| {
                    if (bb_to == 0) break;
                    to = pop_square(&bb_to);
                    flag = if (self.board[to.u].e != .no_piece) Move.capture else Move.silent;
                    store(from, to, flag, storage) orelse return;
                }
            }

            // Diagonal sliders.
            if (!has_pins) {
                bb = our_queens_bishops;
                while (bb != 0) {
                    from = pop_square(&bb);
                    bb_to = attacks.get_bishop_attacks(from, bb_all) & target;
                    while (bb_to != 0) {
                        to = pop_square(&bb_to);
                        flag = if (self.board[to.u].e != .no_piece) Move.capture else Move.silent;
                        store(from, to, flag, storage) orelse return;
                    }
                }
            } else {
                bb = our_queens_bishops & ~self.pins;
                while (bb != 0) {
                    from = pop_square(&bb);
                    bb_to = attacks.get_bishop_attacks(from, bb_all) & target;
                    while (bb_to != 0) {
                        to = pop_square(&bb_to);
                        flag = if (self.board[to.u].e != .no_piece) Move.capture else Move.silent;
                        store(from, to, flag, storage) orelse return;
                    }
                }
                bb = our_queens_bishops & self.pins_diagonal;
                while (bb != 0) {
                    from = pop_square(&bb);
                    bb_to = attacks.get_bishop_attacks(from, bb_all) & target & self.pins_diagonal;
                    while (bb_to != 0) {
                        to = pop_square(&bb_to);
                        flag = if (self.board[to.u].e != .no_piece) Move.capture else Move.silent;
                        store(from, to, flag, storage) orelse return;
                    }
                }
            }

            // Orthogonal sliders.
            if (!has_pins) {
                bb = our_queens_rooks;
                while (bb != 0) {
                    from = pop_square(&bb);
                    bb_to = attacks.get_rook_attacks(from, bb_all) & target;
                    while (bb_to != 0) {
                        to = pop_square(&bb_to);
                        flag = if (self.board[to.u].e != .no_piece) Move.capture else Move.silent;
                        store(from, to, flag, storage) orelse return;
                    }
                }
            } else {
                bb = our_queens_rooks & ~self.pins;
                while (bb != 0) {
                    from = pop_square(&bb);
                    bb_to = attacks.get_rook_attacks(from, bb_all) & target;
                    while (bb_to != 0) {
                        to = pop_square(&bb_to);
                        flag = if (self.board[to.u].e != .no_piece) Move.capture else Move.silent;
                        store(from, to, flag, storage) orelse return;
                    }
                }

                bb = our_queens_rooks & self.pins_orthogonal;
                while (bb != 0) {
                    from = pop_square(&bb);
                    bb_to = attacks.get_rook_attacks(from, bb_all) & target & self.pins_orthogonal;
                    while (bb_to != 0) {
                        to = pop_square(&bb_to);
                        flag = if (self.board[to.u].e != .no_piece) Move.capture else Move.silent;
                        store(from, to, flag, storage) orelse return;
                    }
                }
            }
        } // (not doublecheck)

        // King.
        const king_target = if (check or !captures) bb_not_us else bb_them;
        bb_to = attacks.get_king_attacks(king_sq) & king_target;

        // The king is a troublemaker. For now this 'popcount heuristic' gives the best avg speed, using 2 different approaches to check legality.
        if (popcnt(bb_to) > 2) {
            const bb_unsafe: u64 = self.get_unsafe_squares_for_king(us);
            bb_to &= ~bb_unsafe;
            // Normal.
            while (bb_to != 0) {
                to = pop_square(&bb_to);
                flag = if (self.board[to.u].e != .no_piece) Move.capture else Move.silent;
                store(king_sq, to, flag, storage) orelse return;
            }
            // Castling.
            if (!check and !captures and self.castling_rights != 0) {
                if (self.is_castling_allowed(us, CastleType.SHORT) and self.is_castlingpath_empty(us, CastleType.SHORT) and self.is_legal_castle(us, CastleType.SHORT, bb_unsafe)) {
                    to = self.layout.rook_start_squares[us.u][CastleType.SHORT.u]; // king takes rook.
                    store(king_sq, to, Move.castle_short, storage) orelse return;
                }
                if (self.is_castling_allowed(us, CastleType.LONG) and self.is_castlingpath_empty(us, CastleType.LONG) and self.is_legal_castle(us, CastleType.LONG, bb_unsafe)) {
                    to = self.layout.rook_start_squares[us.u][CastleType.LONG.u]; // king takes rook.
                    store(king_sq, to, Move.castle_long, storage) orelse return;
                }
            }
        } else {
            const bb_without_king: u64 = bb_all ^ self.kings(us);
            // Normal.
            while (bb_to != 0) {
                to = pop_square(&bb_to);
                flag = if (self.board[to.u].e != .no_piece) Move.capture else Move.silent;
                if (self.is_legal_kingmove(us, bb_without_king, to)) {
                    store(king_sq, to, flag, storage) orelse return;
                }
            }
            // Castling.
            if (!ctp.check and !ctp.captures and self.castling_rights != 0) {
                if (self.is_castling_allowed(us, CastleType.SHORT) and self.is_castlingpath_empty(us, CastleType.SHORT) and self.is_legal_castle_check_attacks(us, CastleType.SHORT)) {
                    to = self.layout.rook_start_squares[us.u][CastleType.SHORT.u]; // king takes rook.
                    store(king_sq, to, Move.castle_short, storage) orelse return;
                }
                if (self.is_castling_allowed(us, CastleType.LONG) and self.is_castlingpath_empty(us, CastleType.LONG) and self.is_legal_castle_check_attacks(us, CastleType.LONG)) {
                    to = self.layout.rook_start_squares[us.u][CastleType.LONG.u]; // king takes rook.
                    store(king_sq, to, Move.castle_long, storage) orelse return;
                }
            }
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
            storage.store(Move.create(from, to, flags | Move.knight_promotion)) orelse return null;
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
        const path: u64 = self.layout.castling_king_paths[us.u][castletype.u];
        return path & bb_unsafe == 0;
    }

    /// Checks for each square on the kings path if it is attacked.
    fn is_legal_castle_check_attacks(self: *const Position, comptime us: Color, comptime castletype: CastleType) bool {
        const them: Color = comptime us.opp();
        var path: u64 = self.layout.castling_king_paths[us.u][castletype.u];
        while (path != 0) {
            const sq = pop_square(&path);
            if (self.is_square_attacked_by(sq, them)) return false;
        }
        return true;
    }

    /// Meant to be a validation after uci position command.
    pub fn is_valid(self: *const Position) bool {
        if (popcnt(self.all() > 32)) return false;

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

        var k: u64 = undefined;
        var p: u64 = undefined;
        self.compute_hashkeys(&k, &p);
        if (k != self.key) {
            lib.io.debugprint("KEY {} <> {} lastmove {f}\n", .{ self.key, k, self.last_move });
            self.draw();
            return false;
        }
        if (p != self.pawnkey) {
            lib.io.debugprint("PAWNKEY {} <> {} lastmove {f}\n", .{ self.pawnkey, p, self.last_move });
            self.draw();
            return false;
        }

        // In check and not to move.
        const king_sq_white: Square = self.king_square(Color.WHITE);
        if (self.is_square_attacked_by(king_sq_white, Color.BLACK) and self.stm.e != .white) {
            lib.io.debugprint("CHECK\n", .{});
            self.draw();
            return false;
        }

        const king_sq_black = self.king_square(Color.BLACK);
        if (self.is_square_attacked_by(king_sq_black, Color.WHITE) and self.stm.e != .black) {
            lib.io.debugprint("CHECK\n", .{});
            self.draw();
            return false;
        }
        return true;
    }

    // TODO: write debug code
    // pub fn move_ok(self: *const Position, m: Move) bool {
    //     if (self.board[m.from.u].is_empty()) return false;
    //     switch (m.flags) {
    //         Move.silent => {
    //             if (self.board[m.from.u].is_piece()) return false;
    //         },
    //         Move.double_push => {
    //             if (self.board[m.from.u].is_piece()) return false;
    //             if (!self.board[m.from.u].is_pawn()) return false;
    //         },
    //         Move.castle_short => {
    //         },
    //         Move.castle_long => {
    //         },
    //         Move.knight_promotion or Move.bishop_promotion or Move.rook_promotion or Move.queen_promotion or
    //         Move.capture => {},
    //         Move.ep => {},
    //         Move.knight_promotion_capture or Move.bishop_promotion_capture or Move.rook_promotion_capture or Move.queen_promotion_capture => {
    //         },
    //     }
    // }

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
                } else {
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
        } else {
            try writer.print(" b", .{});
        }

        // Castling rights.
        try writer.print(" ", .{});
        if (self.castling_rights == 0) {
            try writer.print("-", .{});
        } else {
            if (self.castling_rights & cf_white_short != 0) try writer.print("K", .{});
            if (self.castling_rights & cf_white_long != 0)  try writer.print("Q", .{});
            if (self.castling_rights & cf_black_short != 0) try writer.print("k", .{});
            if (self.castling_rights & cf_black_long != 0)  try writer.print("q", .{});
        }

        // Enpassant.
        try writer.print(" ", .{});
        if (self.ep_square.u > 0) {
            try writer.print("{t}", .{self.ep_square.e});
        } else {
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
        //const move_str: []const u8 = if (self.state.last_move.is_empty()) "" else self.state.last_move.to_string().slice();
        io.print_buffered("fen: {f}\n", .{ self });
        io.print_buffered("key: 0x{x:0>16} pawnkey: 0x{x:0>16}\n", .{ self.key, self.pawnkey });
        io.print_buffered("rule50: {}\n", .{ self.rule50 });
        //io.print_buffered("nullmovestate: {}\n", .{ self.nullmove_state });
        io.print_buffered("ply: {}\n", .{ self.ply });
        io.print_buffered("phase: {}\n", .{ self.phase });
        //try io.print_buffered("{} == {} + {}\n", .{ self.non_pawn_material(), self.state.non_pawn_material[0], self.state.non_pawn_material[1] });
        io.print_buffered("Last move: {f}\n", .{self.last_move});
        io.print_buffered("checkers: ", .{});
        if (self.checkers != 0) {
            var bb: u64 = self.checkers;
            while (bb != 0) {
                const sq: Square = pop_square(&bb);
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
            std.meta.eql(self.values, other.values) and
            self.phase == other.phase and
            self.ply == other.ply and
            self.ply_from_root == other.ply_from_root and
            self.game_ply == other.game_ply and
            self.nullmove_state == other.nullmove_state and
            self.rule50 == other.rule50 and
            self.ep_square.u == other.ep_square.u and
            self.castling_rights == other.castling_rights and
            self.last_move == other.last_move and
            self.key == other.key and
            self.pawnkey == other.pawnkey and
            self.checkers == other.checkers and
            self.checkmask == other.checkmask and
            self.pins_diagonal == other.pins_diagonal and
            self.pins_orthogonal == other.pins_orthogonal and
            self.pins == other.pins;

        if (!eql) return false;

        if (check_moves) {
            var store1: MoveStorage = .init();
            var store2: MoveStorage = .init();
            self.lazy_generate_moves(&store1);
            self.lazy_generate_moves(&store2);
            const ok: bool = std.mem.eql(Move, store1.slice(), store2.slice());
            return ok;
        }
        else {
            return true;
        }
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
