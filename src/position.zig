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
const popcnt = bitboards.popcnt;
const bitloop = bitboards.bitloop;
const pawns_shift = funcs.pawns_shift;
const pawn_from = funcs.pawn_from;

const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const File = types.File;
const Rank = types.Rank;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const CastleType = types.CastleType;
const ScorePair = types.ScorePair;

// Castling flags.
pub const cf_white_short: u4 = 0b0001;
pub const cf_white_long: u4 = 0b0010;
pub const cf_black_short: u4 = 0b0100;
pub const cf_black_long: u4 = 0b1000;
pub const cf_white: u4 = 0b0011;
pub const cf_black: u4 = 0b1100;
pub const cf_all: u4 = 0b1111;

pub const classic_startpos_fen: []const u8 = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

const gf_check: u2 = 0b0001; // state_flags + gen
const gf_pins : u2 = 0b0010; // state_flags + gen
const gf_black: u4 = 0b0100; // gen only
const gf_noisy: u4 = 0b1000; // gen only

const gen_color_flags: [Color.count]u4 = .{ 0, gf_black };

pub const Castling = struct {
    /// The current castling rights (bitflags cf_white_short, cf_white_long, cf_black_short, cf_black_long).
    rights: u4,
    /// The initial king files.
    king_start_files: [Color.count]u3,
    /// The initial rook files.
    rook_start_files: [Color.count][CastleType.count]u3,
    /// Byte based paths for checking required empty squares when castling. Because of Chess960 support we have to use this extra path.
    empty_paths: [Color.count][CastleType.count]u8,
    /// Byte based paths for checking attacks when castling.
    attack_paths: [Color.count][CastleType.count]u8,

    const empty: Castling = std.mem.zeroes(Castling);

    /// The values for the classic start position.
    pub const default: Castling = .{
        .rights = cf_all,
        .king_start_files = .{ 4, 4 },
        .rook_start_files = .{ .{ 7, 0 }, .{ 7, 0 }},
        .empty_paths  = .{ .{ 0b00000110, 0b01110000 }, .{ 0b00000110, 0b01110000 } },
        .attack_paths = .{ .{ 0b00000110, 0b01110000 }, .{ 0b00000110, 0b01110000 } },
    };

    /// Hard constants.
    const king_dest_squares: [Color.count][CastleType.count]Square = .{ .{ Square.g1, Square.c1 }, .{ Square.g8, Square.c8 } };
    /// Hard constants.
    const rook_dest_squares: [Color.count][CastleType.count]Square = .{ .{ Square.f1, Square.d1 }, .{ Square.f8, Square.d8 } };
    /// Hard constants.
    const castle_flags: [Color.count][CastleType.count]u4 = .{ .{ cf_white_short, cf_white_long }, .{ cf_black_short, cf_black_long } };

    /// Only used during setup. Updates castling rights, rookfile, kingfile. Assumes `self` is the `castling` field of a Position.
    fn set(self: *Castling, us: Color, rook_file: u3) !void {
        const pos: *const Position = @alignCast(@fieldParentPtr("castling", self));
        const first_rank: u3 = bitboards.relative_rank(us, 0);

        // Validate stuff.
        const king_sq: Square = pos.king_square_or_null(us) orelse return Error.MissingKing;
        if (king_sq.coord.rank != first_rank) return Error.CastlingLogic;
        const rook_sq: Square = .from_rank_file(first_rank, rook_file);
        if (pos.board[rook_sq.u].u != Piece.init(PieceType.rook, us).u) return Error.CastlingLogic;

        // Update files, rights and (byte based) paths.
        const king_file: u3 = king_sq.coord.file;
        const ct: CastleType = if (rook_file > king_file) CastleType.short else CastleType.long;
        self.rights |= castle_flag(us, ct);
        self.king_start_files[us.u] = king_file;
        self.rook_start_files[us.u][ct.u] = rook_file;
        const king_path: u64 = bitboards.get_squarepair(self.king_start(us), king_dest(us, ct)).ray;
        const rook_path: u64 = bitboards.get_squarepair(self.rook_start(us, ct), rook_dest(us, ct)).ray;
        const combined_path: u64 = king_path | rook_path;
        self.empty_paths[us.u][ct.u] = @intCast((combined_path & ~self.king_start(us).to_bitboard() & ~self.rook_start(us, ct).to_bitboard()) >> shift(us));
        self.attack_paths[us.u][ct.u] = @intCast(king_path >> shift(us));
    }

    fn king_start(self: *const Castling, us: Color) Square {
        return .from_rank_file(rank(us), self.king_start_files[us.u]);
    }

    fn rook_start(self: *const Castling, us: Color, ct: CastleType) Square {
        return .from_rank_file(rank(us), self.rook_start_files[us.u][ct.u]);
    }

    fn empty_path(self: *const Castling, us: Color, ct: CastleType) u64 {
        return @as(u64, self.empty_paths[us.u][ct.u]) << shift(us);
    }

    fn attack_path(self: *const Castling, us: Color, ct: CastleType) u64 {
        return @as(u64, self.attack_paths[us.u][ct.u]) << shift(us);
    }

    pub fn king_dest(us: Color, ct: CastleType) Square {
        return king_dest_squares[us.u][ct.u];
    }

    pub fn rook_dest(us: Color, ct: CastleType) Square {
        return rook_dest_squares[us.u][ct.u];
    }

    fn is_set(self: *const Castling, us: Color, ct: CastleType) bool {
        return self.rights & castle_flag(us, ct) != 0;
    }

    fn square_changed(self: *Castling, comptime us: Color, sq: Square) void {
        if (sq.u == self.rook_start(us, CastleType.short).u) self.rights &= ~castle_flag(us, CastleType.short);
        if (sq.u == self.rook_start(us, CastleType.long).u) self.rights &= ~castle_flag(us, CastleType.long);
    }

    fn king_moved(self: *Castling, comptime us: Color) void {
        const mask = comptime (castle_flag(us, CastleType.short) | castle_flag(us, CastleType.long));
        self.rights &= ~mask;
    }

    fn our_rights(self: *const Castling, comptime us: Color) u4 {
        return if (us.e == .white) self.rights & cf_white else self.rights & cf_black;
    }

    pub fn castle_flag(us: Color, ct: CastleType) u4 {
        return castle_flags[us.u][ct.u];
    }

    inline fn rank(us: Color) u3 {
        return if (us.e == .white) 0 else 7;
    }

    inline fn shift(us: Color) u6 {
        return if (us.e == .white) 0 else 56;
    }
};

/// Piececounts. King is always 1.
pub const Material = struct {
    /// The counts of each piece.
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

/// Not used, but we keep it around.
pub const Threats = struct {
    by_pawns: u64 = 0,
    by_knights: u64 = 0,
    by_bishops: u64 = 0,
    by_rooks: u64 = 0,
    by_queens: u64 = 0,
    by_king: u64 = 0,

    pub const empty: Threats = .{};
};

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

pub const Position = struct {
    /// The pieces on the 64 squares.
    board: [Square.count]Piece,
    /// Bitboards occupation. Indexing by [piecetype].
    bitboards_by_type: [PieceType.count]u64,
    /// Bitboards occupation. Indexing by [color].
    bitboards_by_color: [Color.count]u64,
    /// Castling stuff.
    castling: Castling,
    /// All piececounts.
    material: Material,
    /// Piece phase per color. Indexing by [color].
    phase_by_color: [Color.count]u8,
    /// Side to move.
    stm: Color,
    /// Used for repetition detection.
    ply_from_root: u16,
    /// The 'real' game ply. Initialized from fen string.
    game_ply: u16,
    /// State indicating we did a nullmove.
    nullmove_state: bool,
    /// Draw counter. After 50 reversible moves (100 ply) it is a draw.
    rule50: u16,
    /// The enpassant square. A1 means no ep square.
    ep_square: Square,
    /// The position hashkey.
    key: u64,
    /// The hashkey of all pawns. Used for correction history.
    pawnkey: u64,
    /// Key for each side for non pawns. Used for correction history.
    nonpawnkeys: [Color.count]u64,
    /// Key for all minors. Used for correction history.
    minorkey: u64,
    /// Key for all majors. Used for correction history.
    majorkey: u64,
    /// The paths from the enemy slider checkers to the king (**excluded** the king, **included** the checker). Pawns and knights of course included but without path.
    checkmask: u64,
    /// Bitboard with the diagonal pin rays (**excluded** the king, **included** the attacker).
    pins_diagonal: u64,
    /// Bitboard with the orthogonal pin rays (**excluded** the king, **included** the attacker).
    pins_orthogonal: u64,
    /// gf_check + gf_pins.
    state_flags: u2,
    /// Indicates chess 960.
    is_960: bool,

    /// An (invalid) empty position.
    pub const empty: Position = .init_empty();
    /// We must use `set_startpos()` to fill in the hashkeys.
    const empty_classic: Position = .init_empty_classic(); // TODO: make complete.

    fn init_empty() Position {
        return .{
            .board = @splat(Piece.no_piece),
            .bitboards_by_type = @splat(0),
            .bitboards_by_color = @splat(0),
            .castling = .empty,
            .material = .empty,
            .phase_by_color = @splat(0),
            .stm = Color.white,
            .ply_from_root = 0,
            .game_ply = 0,
            .nullmove_state = false,
            .rule50 = 0,
            .ep_square = Square.zero,
            .key = 0,
            .pawnkey = 0,
            .nonpawnkeys = @splat(0),
            .minorkey = 0,
            .majorkey = 0,
            .checkmask = 0,
            .pins_diagonal = 0,
            .pins_orthogonal = 0,
            .state_flags = 0,
            .is_960 = false,
        };
    }

    fn init_empty_classic() Position {
        const b = bitboards;
        return .{
            .board = create_board_from_backrow(.{ PieceType.rook, PieceType.knight, PieceType.bishop, PieceType.queen, PieceType.king, PieceType.bishop, PieceType.knight, PieceType.rook }),
            .bitboards_by_type = .{
                b.bb_rank_2 | b.bb_rank_7,  // pawns
                b.bb_b1 | b.bb_g1 | b.bb_b8 | b.bb_g8, // knights
                b.bb_c1 | b.bb_f1 | b.bb_c8 | b.bb_f8, // bishops
                b.bb_a1 | b.bb_h1 | b.bb_a8 | b.bb_h8, // rooks
                b.bb_d1 | b.bb_d8, // queens
                b.bb_e1 | b.bb_e8  // kings
            },
            .bitboards_by_color = .{ b.bb_rank_1 | b.bb_rank_2, b.bb_rank_7 | b.bb_rank_8 },
            .castling = .default,
            .material = .default,
            .phase_by_color = .{ 12, 12 },
            .stm = Color.white,
            .ply_from_root = 0,
            .game_ply = 0,
            .nullmove_state = false,
            .rule50 = 0,
            .ep_square = Square.zero,
            .key = 0,
            .pawnkey = 0,
            .nonpawnkeys = @splat(0),
            .minorkey = 0,
            .majorkey = 0,
            .checkmask = 0,
            .pins_diagonal = 0,
            .pins_orthogonal = 0,
            .state_flags = 0,
            .is_960 = false,
        };
    }

    inline fn create_board_from_backrow(backrow: [8]PieceType) [64]Piece {
        var result: [64]Piece = @splat(Piece.no_piece);
        for (backrow, 0..) |pt, i| {
            const sq: Square = Square.from_usize(i);
            result[sq.u] = Piece.init(pt, Color.white);
            result[sq.u + 8] = Piece.w_pawn;
            result[sq.u + 48] = Piece.b_pawn;
            result[sq.u + 56] = Piece.init(pt, Color.black);
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

        // uci is started with "setoption name UCI_Chess960 value true" before ucinewgame.
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
                                self.lazy_add_piece(pc, sq);
                                file +|= 1;
                            },
                        }
                    }
                },
                state_color => {
                    if (token[0] == 'b') {
                        self.stm = Color.black;
                        self.key ^= zobrist.btm();
                    }
                },
                state_castle => {
                    for (token) |c| {
                        switch (c) {
                            // Classic.
                            'K' => try self.castling.set(Color.white, 7),
                            'Q' => try self.castling.set(Color.white, 0),
                            'k' => try self.castling.set(Color.black, 7),
                            'q' => try self.castling.set(Color.black, 0),
                            'A'...'H' => try self.castling.set(Color.white, @intCast(c - 'A')),
                            'a'...'h' => try self.castling.set(Color.black, @intCast(c - 'a')),
                            else => {},
                        }
                    }
                    self.key ^= zobrist.castling(self.castling.rights);
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
            } // (switch state)
        } // (while next token)

        self.is_960 |= self.deduce_960();
        self.lazy_update_state();
    }

    /// Return true if we encounter a non-classic castling position.
    fn deduce_960(self: *Position) bool {
        return
            (self.castling.is_set(Color.white, CastleType.short) and (self.get(.e1).u != Piece.w_king.u or self.get(.h1).u != Piece.w_rook.u)) or
            (self.castling.is_set(Color.white, CastleType.long)  and (self.get(.e1).u != Piece.w_king.u or self.get(.a1).u != Piece.w_rook.u)) or
            (self.castling.is_set(Color.black, CastleType.short) and (self.get(.e8).u != Piece.b_king.u or self.get(.h8).u != Piece.b_rook.u)) or
            (self.castling.is_set(Color.black, CastleType.long)  and (self.get(.e8).u != Piece.b_king.u or self.get(.a8).u != Piece.b_rook.u));
    }

    pub fn phase(self: *const Position) u8 {
        return self.phase_by_color[0] + self.phase_by_color[1];
    }

    pub fn king_bucket(self: *const Position, us: Color) u4 {
        return king_bucket_table[self.king_square(us).u];
    }

    /// Parses a uci-move.
    /// - Used after a uci "position" command.
    /// - During a game the time needed for this is not included in the movetime. Timer is only active after the "go" command.
    pub fn parse_move(self: *const Position, str: []const u8) types.ParsingError!ExtMove {
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
            // In chess960 the uci encoding is the same as ours: king takes rook.
            if (!self.is_960) {
                if (from.u == self.castling.king_start(us).u and self.board[from.u].e == Piece.init(PieceType.king, us).e) {
                    if (to.e == Square.g1.e or to.e == Square.g8.e) {
                        to = self.castling.rook_start(us, CastleType.short); // King takes rook.
                    }
                    else if (to.e == Square.c1.e or to.e == Square.c8.e) {
                        to = self.castling.rook_start(us, CastleType.long); // King takes rook.
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
            return finder.extmove; // return the exact move found.
        }
        return types.ParsingError.IllegalMove;
    }

    /// A convenient (faster) way to set the startposition without the need for a fen string.
    pub fn set_startpos(self: *Position) void {
        self.* = empty_classic;
        self.init_hash();
    }

    /// Flips the board inplace.
    pub fn flip(self: *Position) void {
        // Backup stuff
        const bk: [64]Piece = self.board;
        var bb = self.all();

        // Clear stuff
        self.board = @splat(Piece.no_piece);
        self.bitboards_by_type = @splat(0);
        self.bitboards_by_color = @splat(0);

        // Local hash keys.
        var key: u64 = 0;
        var pawnkey: u64 = 0;
        var nonpawnkeys: [2]u64 = @splat(0);
        var minorkey: u64 = 0;
        var majorkey: u64 = 0;

        // Put flipped pieces on flipped squares.
        while (bitloop(&bb)) |sq| {
            const pc: Piece = bk[sq.u];
            const new_sq: Square = sq.flipped();
            const new_pc: Piece = pc.opp();
            const bb_new_sq: u64 = new_sq.to_bitboard();

            self.board[new_sq.u] = new_pc;
            self.bitboards_by_type[new_pc.piecetype().u] |= bb_new_sq;
            self.bitboards_by_color[new_pc.color().u] |= bb_new_sq;

            const hash_delta: u64 = zobrist.piece_square(new_pc, new_sq);
            key ^= hash_delta;
            if (new_pc.is_pawn()) {
                pawnkey ^= hash_delta;
            }
            else {
                const color: Color = new_pc.color();
                nonpawnkeys[color.u] ^= hash_delta;
                if (new_pc.is_minor()) {
                    minorkey ^= hash_delta;
                }
                else if (new_pc.is_major()) {
                    majorkey ^= hash_delta;
                }
            }
        }

        // Flip material.
        std.mem.swap([6]u8, &self.material.counts[0], &self.material.counts[1]);

        // Update state by mirroring the bitboards.
        self.checkmask = bitboards.mirror_vertically(self.checkmask);
        self.pins_diagonal = bitboards.mirror_vertically(self.pins_diagonal);
        self.pins_orthogonal = bitboards.mirror_vertically(self.pins_orthogonal);

        // Complete the hashkeys for stm, castling and ep.
        self.stm = self.stm.opp();
        if (self.stm.e == .black) {
            key ^= zobrist.btm();
        }

        // TODO: flip castling
        self.castling_rights = (self.castling_rights >> 2) | (self.castling_rights << 2);
        key ^= zobrist.castling(self.castling_rights);

        if (self.ep_square.u > 0) {
            self.ep_square = self.ep_square.flipped();
            key ^= zobrist.enpassant(self.ep_square);
        }

        self.key = key;
        self.pawnkey = pawnkey;
        self.nonpawnkeys = nonpawnkeys;
        self.minorkey = minorkey;
        self.majorkey = majorkey;

        if (self.is_960) {
            self.select_layout();
        }

        if (comptime lib.is_paranoid) {
            self.verify_pos(.empty);
        }
    }

    /// Returns a flipped position.
    pub fn flipped(self: *const Position) Position {
        var pos: Position = self.*;
        pos.flip();
        return pos;
    }

    fn init_hash(self: *Position) void {
        self.compute_hashkeys(&self.key, &self.pawnkey, &self.nonpawnkeys[0], &self.nonpawnkeys[1], &self.minorkey, &self.majorkey);
    }

    fn compute_hashkeys(self: *const Position, key: *u64, pawnkey: *u64, white_nonpawnkey: *u64, black_nonpawnkey: *u64, minorkey: *u64, majorkey: *u64) void {
        key.* = 0;
        pawnkey.* = 0;
        white_nonpawnkey.* = 0;
        black_nonpawnkey.* = 0;
        minorkey.* = 0;
        majorkey.* = 0;
        var occ: u64 = self.all();
        while (bitloop(&occ)) |sq| {
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
        key.* ^= zobrist.castling(self.castling.rights);
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
        return self.bitboards_by_color[0] | self.bitboards_by_color[1];
    }

    pub fn all_pawns(self: *const Position) u64 {
        return self.by_type(PieceType.pawn);
    }

    pub fn all_knights(self: *const Position) u64 {
        return self.by_type(PieceType.knight);
    }

    pub fn all_bishops(self: *const Position) u64 {
        return self.by_type(PieceType.bishop);
    }

    pub fn all_rooks(self: *const Position) u64 {
        return self.by_type(PieceType.rook);
    }

    pub fn all_queens(self: *const Position) u64 {
        return self.by_type(PieceType.queen);
    }

    pub fn all_kings(self: *const Position) u64 {
        return self.by_type(PieceType.king);
    }

    pub fn all_queens_bishops(self: *const Position) u64 {
        return self.by_type(PieceType.bishop) | self.by_type(PieceType.queen);
    }

    pub fn all_queens_rooks(self: *const Position) u64 {
        return self.by_type(PieceType.rook) | self.by_type(PieceType.queen);
    }

    pub fn all_minors(self: *const Position) u64 {
        return self.all_knights() | self.all_bishops();
    }

    pub fn all_majors(self: *const Position) u64 {
        return self.all_rooks() | self.all_queens();
    }

    pub fn pawns(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.pawn) & self.by_color(us);
    }

    pub fn knights(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.knight) & self.by_color(us);
    }

    pub fn bishops(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.bishop) & self.by_color(us);
    }

    pub fn rooks(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.rook) & self.by_color(us);
    }

    pub fn queens(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.queen) & self.by_color(us);
    }

    pub fn kings(self: *const Position, us: Color) u64 {
        return self.by_type(PieceType.king) & self.by_color(us);
    }

    pub fn queens_bishops(self: *const Position, us: Color) u64 {
        return (self.by_type(PieceType.bishop) | self.by_type(PieceType.queen)) & self.by_color(us);
    }

    pub fn queens_rooks(self: *const Position, us: Color) u64 {
        return (self.by_type(PieceType.rook) | self.by_type(PieceType.queen)) & self.by_color(us);
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

    /// A ssafe function used during setup.
    pub fn king_square_or_null(self: *const Position, us: Color) ?Square {
        return bitboards.first_square_or_null(self.kings(us));
    }

    pub fn sliders(self: *const Position, us: Color) u64 {
        return (self.by_type(PieceType.bishop) | self.by_type(PieceType.rook) | self.by_type(PieceType.queen)) & self.by_color(us);
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

    /// Returns a bitboard of our pins. Remember these are rays.
    pub fn our_pins(self: *const Position) u64 {
        return self.pins_diagonal | self.pins_orthogonal;
    }

    pub fn is_draw_by_insufficient_material(self: *const Position) bool {
        const us: Color = Color.white;
        const them: Color = Color.black;

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
        if (rank == bitboards.rank_3) {
            const w_pawn_sq = ep.add(8);
            const requirements: bool = self.board[w_pawn_sq.u].e == .w_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u - 8].e == .no_piece;
            return requirements and (bitboards.ep_masks[w_pawn_sq.u] & self.pawns(Color.black) != 0);
        }
        else if (rank == bitboards.rank_6) {
            const b_pawn_sq = ep.sub(8);
            const requirements:  bool = self.board[b_pawn_sq.u].e == .b_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u + 8].e == .no_piece;
            return requirements and (bitboards.ep_masks[b_pawn_sq.u] & self.pawns(Color.white) != 0);
        }
        return false;
    }

    /// Update board, bitboards, phase and keys.
    pub fn lazy_add_piece(self: *Position, pc: Piece, sq: Square) void {
        const color: Color = pc.color();
        switch (color.e) {
            inline else => |e| self.add_piece(.{ .e = e }, pc, sq),
            // .white => self.add_piece(Color.white, pc, sq),
            // .black => self.add_piece(Color.black, pc, sq),
        }
    }

    fn update_keys(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        const delta: u64 = zobrist.piece_square(pc, sq);
        self.key ^= delta;
        if (pc.is_pawn_of_color(us)) {
            self.pawnkey ^= delta;
        }
        else {
            self.nonpawnkeys[us.u] ^= delta;
            if (pc.is_minor_of_color(us)) {
                self.minorkey ^= delta;
            }
            else if (pc.is_major_of_color(us)) {
                self.majorkey ^= delta;
            }
        }
    }

    fn update_keys_from_to(self: *Position, comptime us: Color, pc: Piece, from: Square, to: Square) void {
        const delta: u64 = zobrist.piece_from_to(pc, from, to);
        self.key ^= delta;
        if (pc.is_pawn_of_color(us)) {
            self.pawnkey ^= delta;
        }
        else {
            self.nonpawnkeys[us.u] ^= delta;
            if (pc.is_minor_of_color(us)) {
                self.minorkey ^= delta;
            }
            else if (pc.is_major_of_color(us)) {
                self.majorkey ^= delta;
            }
        }
    }

    /// Updates board, bitboards, material, phase, keys.
    fn add_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        if (comptime lib.is_paranoid) {
            assert(self.get(sq).is_empty());
            assert(pc.is_piece());
            assert(pc.color().e == us.e);
        }
        const mask: u64 = sq.to_bitboard();
        self.board[sq.u] = pc;
        const pt: PieceType = pc.piecetype_for_known_color(us);
        self.bitboards_by_type[pt.u] |= mask;
        self.bitboards_by_color[us.u] |= mask;
        self.material.counts[us.u][pt.u] += 1;
        self.phase_by_color[us.u] += types.phase_table[pt.u];
        self.update_keys(us, pc, sq);
    }

    /// Update sboard, bitboards, material, phase, keys.
    fn remove_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        if (comptime lib.is_paranoid) {
            assert(self.get(sq).e == pc.e);
            assert(pc.color().e == us.e);
        }
        const not_mask: u64 = ~sq.to_bitboard();
        self.board[sq.u] = Piece.no_piece;
        const pt: PieceType = pc.piecetype_for_known_color(us);
        self.bitboards_by_type[pt.u] &= not_mask;
        self.bitboards_by_color[us.u] &= not_mask;
        self.material.counts[us.u][pt.u] -= 1;
        self.phase_by_color[us.u] -= types.phase_table[pt.u];
        self.update_keys(us, pc, sq);
    }

    /// Updates board, bitboards, material, phase, keys.
    fn move_piece(self: *Position, comptime us: Color, pc: Piece, from: Square, to: Square) void {
        if (comptime lib.is_paranoid) {
            assert(self.get(from).e == pc.e);
            assert(self.get(to).is_empty());
            assert(pc.color().e == us.e);
        }
        const xor_mask: u64 = from.to_bitboard() | to.to_bitboard();
        self.board[from.u] = Piece.no_piece;
        self.board[to.u] = pc;
        const pt: PieceType = pc.piecetype_for_known_color(us);
        //if (us.e == .white) self.bitboards_by_type[pc.u] ^= xor_mask else self.bitboards_by_type[pc.u - 6] ^= xor_mask;
        self.bitboards_by_type[pt.u] ^= xor_mask;
        self.bitboards_by_color[us.u] ^= xor_mask;
        self.update_keys_from_to(us, pc, from, to);
    }

    /// Makes the move on the board.`us` is comptime for performance reasons and must be the `stm`.
    pub fn do_move(self: *Position, comptime us: Color, ex: ExtMove) void {
        if (comptime lib.is_paranoid) {
            assert(us.e == self.stm.e);
            self.verify_pos(ex);
        }

        // In paranoid mode ensure the predicted key is ok.
        const predicted_key = if (comptime lib.is_paranoid) self.predict_key(us, ex) else void;

        const them: Color = comptime us.opp();
        const from: Square = ex.move.from;
        const to: Square = ex.move.to;
        const pc: Piece = ex.piece;
        const old_rights: u4 = self.castling.rights;

        // Clear ep by default. Note that the zobrist for square a1 (invalid ep) is 0 so this xor is safe.
        self.key ^= zobrist.btm() ^ zobrist.enpassant(self.ep_square);

        // Update some stuff.
        self.stm = them;
        self.ply_from_root += 1;
        self.game_ply += 1;
        self.nullmove_state = false;
        self.ep_square = Square.zero;

        // The switch is in numerical order.
        sw: switch (ex.move.flags) {
            Move.silent => {
                if (pc.is_pawn_of_color(us)) {
                    self.rule50 = 0;
                }
                else {
                    self.rule50 += 1;
                }
                self.move_piece(us, pc, from, to);
                if (pc.is_king_of_color(us)) {
                    self.castling.king_moved(us);
                }
                else {
                    self.castling.square_changed(us, from);
                    self.castling.square_changed(us, to);
                }
            },
            Move.double_push => {
                self.rule50 = 0;
                self.move_piece(us, pc, from, to);
                // Only set ep if usable.
                if (bitboards.ep_masks[to.u] & self.pawns(them) != 0) {
                    const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                    self.ep_square = ep;
                    self.key ^= zobrist.enpassant(ep);
                }
            },
            Move.castle_short => {
                self.rule50 += 1;
                const king: Piece = comptime Piece.init(PieceType.king, us);
                const rook: Piece = comptime Piece.init(PieceType.rook, us);
                const king_to: Square = comptime Castling.king_dest(us, CastleType.short);
                const rook_to: Square = comptime Castling.rook_dest(us, CastleType.short); // king takes rook
                if (self.is_960) {
                    self.remove_piece(us, rook, to);
                    self.remove_piece(us, king, from);
                    self.add_piece(us, king, king_to);
                    self.add_piece(us, rook, rook_to);
                }
                else {
                    self.move_piece(us, king, from, king_to);
                    self.move_piece(us, rook, to, rook_to);
                }
                self.castling.king_moved(us);
            },
            Move.castle_long => {
                self.rule50 += 1;
                const king: Piece = comptime Piece.init(PieceType.king, us);
                const rook: Piece = comptime Piece.init(PieceType.rook, us);
                const king_to: Square = comptime Castling.king_dest(us, CastleType.long);
                const rook_to: Square = comptime Castling.rook_dest(us, CastleType.long); // king takes rook
                if (self.is_960) {
                    self.remove_piece(us, rook, to);
                    self.remove_piece(us, king, from);
                    self.add_piece(us, king, king_to);
                    self.add_piece(us, rook, rook_to);
                }
                else {
                    self.move_piece(us, king, from, king_to);
                    self.move_piece(us, rook, to, rook_to);
                }
                self.castling.king_moved(us);
            },
            Move.knight_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.init(PieceType.knight, us);
                const pawn: Piece = comptime Piece.init(PieceType.pawn, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
            },
            Move.bishop_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.init(PieceType.bishop, us);
                const pawn: Piece = comptime Piece.init(PieceType.pawn, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
            },
            Move.rook_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.init(PieceType.rook, us);
                const pawn: Piece = comptime Piece.init(PieceType.pawn, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
            },
            Move.queen_promotion => {
                self.rule50 = 0;
                const prom: Piece = comptime Piece.init(PieceType.queen, us);
                const pawn: Piece = comptime Piece.init(PieceType.pawn, us);
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
            },
            Move.capture => {
                const capt: Piece = ex.captured;
                self.rule50 = 0;
                self.remove_piece(them, capt, to);
                self.move_piece(us, pc, from, to);
                if (pc.is_king_of_color(us)) {
                    self.castling.king_moved(us);
                    self.castling.square_changed(them, to);
                }
                else {
                    self.castling.square_changed(us, from);
                    self.castling.square_changed(them, to);
                }
            },
            Move.ep => {
                self.rule50 = 0;
                const pawn_us: Piece = comptime Piece.init(PieceType.pawn, us);
                const pawn_them: Piece = comptime Piece.init(PieceType.pawn, them);
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.remove_piece(them, pawn_them, capt_sq);
                self.move_piece(us, pawn_us, from, to);
            },
            Move.knight_promotion_capture => {
                const capt: Piece = ex.captured;
                self.remove_piece(them, capt, to);
                self.castling.square_changed(them, to);
                continue :sw Move.knight_promotion;
            },
            Move.bishop_promotion_capture => {
                const capt: Piece = ex.captured;
                self.remove_piece(them, capt, to);
                self.castling.square_changed(them, to);
                continue :sw Move.bishop_promotion;
            },
            Move.rook_promotion_capture => {
                const capt: Piece = ex.captured;
                self.remove_piece(them, capt, to);
                self.castling.square_changed(them, to);
                continue :sw Move.rook_promotion;
            },
            Move.queen_promotion_capture => {
                const capt: Piece = ex.captured;
                self.remove_piece(them, capt, to);
                self.castling.square_changed(them, to);
                continue :sw Move.queen_promotion;
            },
            else => {
                unreachable;
            },
        }

        self.key ^= zobrist.castling(old_rights) ^ zobrist.castling(self.castling.rights);
        self.update_state(them);

        if (comptime lib.is_paranoid) {
            assert(predicted_key == self.key);
            self.verify_pos(ex);
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
        self.ply_from_root += 1;
        self.game_ply += 1;
        self.ep_square = Square.zero;
        self.update_state(them);
    }

    /// Computes the key only. Assumes the move is not yet on the board.
    /// Pass an empty move for predicting a nullmove key.
    pub fn predict_key(self: *const Position, comptime us: Color, ex: ExtMove) u64 {
        const them: Color = comptime us.opp();
        const from: Square = ex.move.from;
        const to: Square = ex.move.to;
        const pc: Piece = ex.piece;
        const old_rights: u4 = self.castling.rights;
        var new_castling: Castling = self.castling;

        if (comptime lib.verifications) {
            const ok = ex.move.is_empty() or self.get(from).e == ex.piece.e;
            lib.verify(ok, "predict_key() invalid move", .{});
        }

        // Clear ep by default. Note that the zobrist for square a1 (invalid ep) is 0 so this xor is safe.
        var key: u64 = self.key ^ zobrist.btm() ^ zobrist.enpassant(self.ep_square);

        // If this is a nullmove we're done.
        if (ex.move.is_empty()) {
            return key;
        }

        const key_delta = zobrist.piece_square(pc, from) ^ zobrist.piece_square(pc, to);

        // Switch is in numerical order.
        sw: switch (ex.move.flags) {
            Move.silent => {
                key ^= key_delta;
                if (pc.is_king_of_color(us)) {
                    new_castling.king_moved(us);
                }
                else {
                    new_castling.square_changed(us, from);
                    new_castling.square_changed(us, to);
                }
            },
            Move.double_push => {
                key ^= key_delta;
                // Only use ep if usable.
                if (bitboards.ep_masks[to.u] & self.pawns(them) != 0) {
                    const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                    key ^= zobrist.enpassant(ep);
                }
            },
            Move.castle_short => {
                const king: Piece = comptime Piece.init(PieceType.king, us);
                const rook: Piece = comptime Piece.init(PieceType.rook, us);
                const king_to: Square = comptime Castling.king_dest(us, CastleType.short);
                const rook_to: Square = comptime Castling.rook_dest(us, CastleType.short);
                const castle_delta: u64 = zobrist.piece_square(king, from) ^ zobrist.piece_square(king, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
                key ^= castle_delta;
                new_castling.king_moved(us);
            },
            Move.castle_long => {
                const king: Piece = comptime Piece.init(PieceType.king, us);
                const rook: Piece = comptime Piece.init(PieceType.rook, us);
                const king_to: Square = comptime Castling.king_dest(us, CastleType.long);
                const rook_to: Square = comptime Castling.rook_dest(us, CastleType.long);
                const castle_delta: u64 = zobrist.piece_square(king, from) ^ zobrist.piece_square(king, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
                key ^= castle_delta;
                new_castling.king_moved(us);
            },
            Move.knight_promotion => {
                const prom: Piece = comptime Piece.init(PieceType.knight, us);
                const pawn: Piece = comptime Piece.init(PieceType.pawn, us);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            Move.bishop_promotion => {
                const prom: Piece = comptime Piece.init(PieceType.bishop, us);
                const pawn: Piece = comptime Piece.init(PieceType.pawn, us);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            Move.rook_promotion => {
                const prom: Piece = comptime Piece.init(PieceType.rook, us);
                const pawn: Piece = comptime Piece.init(PieceType.pawn, us);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            Move.queen_promotion => {
                const prom: Piece = comptime Piece.init(PieceType.queen, us);
                const pawn: Piece = comptime Piece.init(PieceType.pawn, us);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            Move.capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta ^ key_delta;
                if (pc.is_king_of_color(us)) {
                    new_castling.king_moved(us);
                    new_castling.square_changed(them, to);
                }
                else {
                    new_castling.square_changed(us, from);
                    new_castling.square_changed(them, to);
                }
            },
            Move.ep => {
                const pawn_them: Piece = comptime Piece.init(PieceType.pawn, them);
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                key ^= key_delta ^ zobrist.piece_square(pawn_them, capt_sq);
            },
            Move.knight_promotion_capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                new_castling.square_changed(them, to);
                continue :sw Move.knight_promotion;
            },
            Move.bishop_promotion_capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                new_castling.square_changed(them, to);
                continue :sw Move.bishop_promotion;
            },
            Move.rook_promotion_capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                new_castling.square_changed(them, to);
                continue :sw Move.rook_promotion;
            },
            Move.queen_promotion_capture => {
                const capt: Piece = ex.captured;
                const capt_delta: u64 = zobrist.piece_square(capt, to);
                key ^= capt_delta;
                new_castling.square_changed(them, to);
                continue :sw Move.queen_promotion;
            },
            else => {
                unreachable;
            },
        }

        key ^= zobrist.castling(old_rights) ^ zobrist.castling(new_castling.rights);
        return key;
    }

    fn update_state(self: *Position, comptime us: Color) void {
        const them: Color = comptime us.opp();
        const bb_all: u64 = self.all();
        const bb_us: u64 = self.by_color(us);
        const king_sq: Square = self.king_square(us);

        self.pins_orthogonal = 0;
        self.pins_diagonal = 0;
        self.state_flags = 0;

        self.checkmask =
            (attacks.get_pawn_attacks(king_sq, us) & self.pawns(them)) |
            (attacks.get_knight_attacks(king_sq) & self.knights(them));

        const bb_occ_without_us: u64 = bb_all ^ self.by_color(us);
        var candidate_slider_attackers: u64 =
            (attacks.get_bishop_attacks(king_sq, bb_occ_without_us) & self.queens_bishops(them)) |
            (attacks.get_rook_attacks(king_sq, bb_occ_without_us) & self.queens_rooks(them));

        // Our pins and their checks.
        while (bitloop(&candidate_slider_attackers)) |attacker_sq| {
            const pair: *const bitboards.SquarePair = bitboards.get_squarepair(king_sq, attacker_sq);
            const bb_blockers: u64 = pair.ray & bb_us;
            // We have a slider checker when there is nothing in between.
            if (bb_blockers == 0) {
                self.checkmask |= pair.ray;
            }
            // We have a pin when exactly 1 bit is set. There is one piece in between.
            else if (bb_blockers & (bb_blockers - 1) == 0) {
                switch (pair.axis) {
                    .orth => self.pins_orthogonal |= pair.ray,
                    .diag => self.pins_diagonal |= pair.ray,
                    else => unreachable,
                }
                self.state_flags |= gf_pins;
            }
        }

        // gf_check is the first bit.
        self.state_flags |= @intFromBool(self.checkmask != 0);
    }

    /// Not used, but we keep it around.
    pub fn get_threats(self: *const Position, comptime attacker: Color) Threats {
        const bb_all = self.all();
        var threats: Threats = .empty;

        threats.by_pawns = funcs.pawns_attacks(self.pawns(attacker), attacker);
        var bb: u64 = self.knights(attacker);
        while (bitloop(&bb))|sq| {
            threats.by_knights |= attacks.get_knight_attacks(sq);
        }

        bb = self.bishops(attacker);
        while (bitloop(&bb))|sq| {
            threats.by_bishops |= attacks.get_bishop_attacks(sq, bb_all);
        }

        bb = self.rooks(attacker);
        while (bitloop(&bb))|sq| {
            threats.by_rooks |= attacks.get_rook_attacks(sq, bb_all);
        }

        bb = self.queens(attacker);
        while (bitloop(&bb))|sq| {
            threats.by_queens |= attacks.get_queen_attacks(sq, bb_all);
        }

        threats.by_king = attacks.get_king_attacks(self.king_square(attacker));

        return threats;
    }

    pub fn lazy_do_move(self: *Position, ex: ExtMove) void {
        switch (self.stm.e) {
            .white => self.do_move(Color.white, ex),
            .black => self.do_move(Color.black, ex),
        }
    }

    pub fn lazy_do_nullmove(self: *Position) void {
        switch (self.stm.e) {
            .white => self.do_nullmove(Color.white),
            .black => self.do_nullmove(Color.black),
        }
    }

    pub fn lazy_update_state(self: *Position) void {
        switch (self.stm.e) {
            .white => self.update_state(Color.white),
            .black => self.update_state(Color.black),
        }
    }

    /// Returns true if square `to` is attacked by any piece of `attacker`.
    pub fn is_square_attacked_by(self: *const Position, to: Square, comptime attacker: Color) bool {
        // Uses pawn inversion trick.
        const inverted = comptime attacker.opp();
        const occ: u64 = self.all();
        return
            (attacks.get_knight_attacks(to) & self.knights(attacker)) |
            (attacks.get_king_attacks(to) & self.kings(attacker)) |
            (attacks.get_pawn_attacks(to, inverted) & self.pawns(attacker)) |
            (attacks.get_rook_attacks(to, occ) & self.queens_rooks(attacker)) |
            (attacks.get_bishop_attacks(to, occ) & self.queens_bishops(attacker)) != 0;
    }

    /// Returns true if any square of the `to_bb` bitboard is attacked by `attacker`.
    pub fn any_square_attacked_by(self: *const Position, to_bb: u64, comptime attacker: Color) bool {
        var bb: u64 = to_bb;
        while (bitloop(&bb)) |sq| {
            if (self.is_square_attacked_by(sq, attacker)) return true;
        }
        return false;
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
            (attacks.get_pawn_attacks(to, Color.black) & self.pawns(Color.white)) |
            (attacks.get_pawn_attacks(to, Color.white) & self.pawns(Color.black)) |
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
            att |= funcs.pawns_attacks(their_pawns, attacker);
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

    pub fn lazy_generate_all_moves(self: *const Position, noalias storage: anytype) void {
        switch (self.stm.e) {
            .white => self.generate_all_moves(Color.white, storage),
            .black => self.generate_all_moves(Color.black, storage),
        }
    }

    /// Generate all legal moves.
    pub fn generate_all_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void {
        switch (self.state_flags) {
            inline else => |sf| {
                self.gen(gen_color_flags[us.u] | sf, storage);
            }
        }
    }

    /// Generate quiescence moves.
    /// - When not in check: Generate captures and queen promotions.
    /// - When in check: generate all legal moves and queen promotions.
    pub fn generate_quiescence_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void {
        switch (self.state_flags) {
            inline else => |sf| {
                self.gen(gen_color_flags[us.u] | sf | gf_noisy, storage);
            }
        }
    }

    /// See `MoveStorage` for the interface of `storage`. Required are the functions `reset() void` and `store(anytype, ExtMove) void`.
    fn gen(self: *const Position, comptime flags: u4, noalias storage: anytype) void {
        storage.reset();

        const us: Color = comptime if (flags & gf_black != 0) Color.black else Color.white;
        const them = comptime us.opp();
        const check: bool = comptime flags & gf_check != 0;
        const noisy: bool = comptime flags & gf_noisy != 0;
        const has_pins: bool = comptime flags & gf_pins != 0;
        const do_all_promotions: bool = comptime !noisy;
        const checkers: u64 = self.checkmask & self.by_color(them);

        const doublecheck: bool = check and popcnt(checkers) > 1;
        const bb_all: u64 = self.all();
        const bb_us: u64 = self.by_color(us);
        const bb_them: u64 = self.by_color(them);
        const bb_not_us: u64 = ~bb_us;
        const king_sq: Square = self.king_square(us);
        const pins_diagonal: u64 = self.pins_diagonal;
        const pins_orthogonal: u64 = self.pins_orthogonal;
        const pins: u64 = pins_diagonal | pins_orthogonal;

        var bb: u64 = undefined;

        // In case of a doublecheck we can only move the king.
        if (!doublecheck) {
            const pawn_us: Piece = comptime Piece.init(PieceType.pawn, us);
            const pawn_them: Piece = comptime Piece.init(PieceType.pawn, them);
            const our_pawns: u64 = self.pawns(us);
            const our_knights: u64 = self.knights(us);
            const our_queens_bishops: u64 = self.queens_bishops(us);
            const our_queens_rooks: u64 = self.queens_rooks(us);

            const target: u64 = if (check) self.checkmask else if (!noisy) bb_not_us else bb_them;

            // Pawns.
            if (our_pawns != 0) {
                const third_rank: u64 = comptime bitboards.relative_rank_3_bitboard(us);
                const last_rank: u64 = comptime bitboards.relative_rank_8_bitboard(us);
                const empty_squares: u64 = ~bb_all;
                const enemies: u64 = if (check) checkers else bb_them;

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
                        store(pawn_from(to, us, .up), to, Move.silent, pawn_us, Piece.no_piece, storage);
                    }
                    // Double push.
                    bb = bb_double;
                    while (bitloop(&bb)) |to| {
                        store(if (us.e == .white) to.sub(16) else to.add(16), to, Move.double_push, pawn_us, Piece.no_piece, storage);
                    }
                }

                // left capture promotions.
                bb = bb_northwest & last_rank;
                while (bitloop(&bb)) |to| {
                    self.store_promotions(us, do_all_promotions, true, pawn_from(to, us, .northwest), to, storage);
                }

                // right capture promotions.
                bb = bb_northeast & last_rank;
                while (bitloop(&bb)) |to| {
                    self.store_promotions(us, do_all_promotions, true, pawn_from(to, us, .northeast), to, storage);
                }

                // push promotions.
                bb = bb_single & last_rank;
                while (bitloop(&bb)) |to| {
                    self.store_promotions(us, do_all_promotions, false, pawn_from(to, us, .up), to, storage);
                }

                // left normal captures.
                bb = bb_northwest & ~last_rank;
                while (bitloop(&bb)) |to| {
                    store(pawn_from(to, us, .northwest), to, Move.capture, pawn_us, self.board[to.u], storage);
                }

                // right normal captures.
                bb = bb_northeast & ~last_rank;
                while (bitloop(&bb)) |to| {
                    store(pawn_from(to, us, .northeast), to, Move.capture, pawn_us, self.board[to.u], storage);
                }

                const ep: Square = self.ep_square;
                // Enpassant.
                if (ep.u > 0) {
                    bb = attacks.get_pawn_attacks(ep, them) & our_pawns; // inversion trick.
                    while (bitloop(&bb)) |from| {
                        if (self.is_legal_enpassant(us, king_sq, from, ep)) {
                            store(from, ep, Move.ep, pawn_us, pawn_them, storage);
                        }
                    }
                }
            } // (pawns)

            // Knights.
            bb = if (!has_pins) our_knights else our_knights & ~pins; // A knight can never escape a pin.
            while (bitloop(&bb)) |from| {
                self.store_many(from, attacks.get_knight_attacks(from) & target, storage);
            }

            // Diagonal sliders.
            if (!has_pins) {
                bb = our_queens_bishops;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_bishop_attacks(from, bb_all) & target, storage);
                }
            }
            else {
                bb = our_queens_bishops & ~pins;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_bishop_attacks(from, bb_all) & target, storage);
                }
                bb = our_queens_bishops & pins_diagonal;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_bishop_attacks(from, bb_all) & target & pins_diagonal, storage);
                }
            }

            // Orthogonal sliders.
            if (!has_pins) {
                bb = our_queens_rooks;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_rook_attacks(from, bb_all) & target, storage);
                }
            }
            else {
                bb = our_queens_rooks & ~pins;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_rook_attacks(from, bb_all) & target, storage);
                }
                bb = our_queens_rooks & pins_orthogonal;
                while (bitloop(&bb)) |from| {
                    self.store_many(from, attacks.get_rook_attacks(from, bb_all) & target & pins_orthogonal, storage);
                }
            }

        } // (not doublecheck)

        // King.
        const king_us: Piece = comptime Piece.init(PieceType.king, us);
        const king_target = if (check or !noisy) bb_not_us else bb_them;
        bb = attacks.get_king_attacks(king_sq) & king_target;

        // The king is a troublemaker. For now this 'popcount heuristic' gives the best avg speed, using 2 different approaches to test legality.
        if (popcnt(bb) > 2) {
            const bb_unsafe: u64 = self.get_unsafe_squares_for_king(us);
            bb &= ~bb_unsafe;
            // Normal.
            while (bitloop(&bb)) |to| {
                const flag: u4 = if (self.board[to.u].e == .no_piece) Move.silent else Move.capture;
                store(king_sq, to, flag, king_us, self.board[to.u], storage);
            }
            // Castling.
            if (!check and !noisy) {
                inline for (CastleType.all) |ct| {
                    if (self.is_castling_allowed(us, ct) and self.is_castlingpath_empty(us, ct) and self.is_legal_castle(us, ct, bb_unsafe)) {
                        const castle_flag: u4 = comptime Move.castles[ct.u];
                        const to = self.castling.rook_start(us, ct); // king takes rook.
                        store(king_sq, to, castle_flag, king_us, Piece.no_piece, storage);
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
                    store(king_sq, to, flag, king_us, self.board[to.u], storage);
                }
            }
            // Castling.
            if (!check and !noisy) {
                inline for (CastleType.all) |ct| {
                    if (self.is_castling_allowed(us, ct) and self.is_castlingpath_empty(us, ct) and self.is_legal_castle_check_attacks(us, ct)) {
                        const castle_flag: u4 = comptime Move.castles[ct.u];
                        const to = self.castling.rook_start(us, ct); // king takes rook.
                        store(king_sq, to, castle_flag, king_us, Piece.no_piece, storage);
                    }
                }
            }
        }
    }

    /// For knights and sliders.
    fn store_many(self: *const Position, from: Square, bb_to: u64, noalias storage: anytype) void {
        const piece: Piece = self.board[from.u];
        var bb: u64 = bb_to;
        while (bitloop(&bb)) |to| {
            const captured: Piece = self.board[to.u];
            const flag: u4 = if (captured.e == .no_piece) Move.silent else Move.capture;
            store(from, to, flag, piece, captured, storage);
        }
    }

    /// Store all promotions or just queen (for quiescence).
    fn store_promotions(self: *const Position, comptime us: Color, comptime do_all: bool, comptime is_capture: bool, from: Square, to: Square, noalias storage: anytype) void {
        const Q: u4 = if (is_capture) Move.queen_promotion_capture else Move.queen_promotion;
        const R: u4 = if (is_capture) Move.rook_promotion_capture else Move.rook_promotion;
        const B: u4 = if (is_capture) Move.bishop_promotion_capture else Move.bishop_promotion;
        const N: u4 = if (is_capture) Move.knight_promotion_capture else Move.knight_promotion;

        const piece: Piece = Piece.init(PieceType.pawn, us);
        const captured: Piece = if (is_capture) self.board[to.u] else Piece.no_piece;

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

    /// Tricky one. An ep move can uncover a check.
    fn is_legal_enpassant(self: *const Position, comptime us: Color, king_sq: Square, from: Square, to: Square) bool {
        const them: Color = comptime us.opp();
        const capt_sq = if (us.e == .white) to.sub(8) else to.add(8);
        const occ: u64 = (self.all() ^ from.to_bitboard() ^ capt_sq.to_bitboard() ^ to.to_bitboard());
        const att: u64 =
            (attacks.get_rook_attacks(king_sq, occ) & self.queens_rooks(them)) |
            (attacks.get_bishop_attacks(king_sq, occ) & self.queens_bishops(them));
        return att == 0;
    }

    fn is_legal_kingmove(self: *const Position, comptime us: Color, bb_without_king: u64, to: Square) bool {
        const them: Color = comptime us.opp();
        return !self.is_square_attacked_by_for_occupation(bb_without_king, to, them);
    }

    pub fn is_castling_allowed(self: *const Position, comptime us: Color, comptime ct: CastleType) bool {
        return self.castling.is_set(us, ct);
    }

    fn is_castlingpath_empty(self: *const Position, comptime us: Color, comptime ct: CastleType, ) bool {
        return self.castling.empty_path(us, ct) & self.all() == 0;
    }

    /// Compares the kings path with unsafe squares. (1st way of castling validation)
    fn is_legal_castle(self: *const Position, comptime us: Color, comptime ct: CastleType, bb_unsafe: u64) bool {
        // Chess960 requires the rook pin check (In classic the rooks cannot be pinned).
        return
            self.pins_orthogonal & self.castling.rook_start(us, ct).to_bitboard() == 0 and
            self.castling.attack_path(us, ct) & bb_unsafe == 0;
    }

    /// Test if any square on the king path is attacked. (2nd way of castling validation)
    fn is_legal_castle_check_attacks(self: *const Position, comptime us: Color, comptime ct: CastleType) bool {
        const them: Color = comptime us.opp();
        // Chess960 requires the rook pin check (In classic the rooks cannot be pinned).
        return
            self.pins_orthogonal & self.castling.rook_start(us, ct).to_bitboard() == 0 and
            !self.any_square_attacked_by(self.castling.attack_path(us, ct), them);
    }

    /// Paranoid only.
    pub fn verify_pos(self: *const Position, ex: ExtMove) void {
        lib.not_in_release();

        // Check counts.
        for (Color.all) |color| {
            for (PieceType.all) |piecetype| {
                const cnt: u8 = popcnt(self.bitboards_by_color[color.u] & self.bitboards_by_type[piecetype.u]);
                lib.verify(self.material.counts[color.u][piecetype.u] == cnt, "pos count mismatch", .{});
            }
        }

        if (popcnt(self.kings(Color.white)) != 1) {
            lib.wtf("pos white king error", .{});
        }

        if (popcnt(self.kings(Color.black)) != 1) {
            lib.wtf("pos black king error", .{});
            return false;
        }

        if (popcnt(self.all()) > 32) {
            lib.wtf("pos too many pieces", .{});
        }

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
        const king_sq_white: Square = self.king_square(Color.white);
        const king_sq_black = self.king_square(Color.black);

        if (self.is_square_attacked_by(king_sq_white, Color.black) and self.stm.e != .white) {
            lib.wtf("pos white in check and not to move", .{});
        }

        if (self.is_square_attacked_by(king_sq_black, Color.white) and self.stm.e != .black) {
            lib.wtf("pos black in check and not to move", .{});
        }

        // TODO: move this to assert_pos_ok().
        lib.verify((self.checkmask == 0) == (self.state_flags & gf_check == 0), "flags check", .{});
        lib.verify((self.our_pins() == 0) == (self.state_flags & gf_pins == 0), "flags pins", .{});
    }

    /// A little validation after uci set.
    pub fn is_valid(self: *const Position) bool {
        // TODO: make airtight and also use it. Also check the material on logic. (for example queencount + pawncount <= 9)

        //if (self.queen_count(Color.white) + self.pawn_count(Color.white) > 9) return false;

        if (popcnt(self.by_color(Color.white)) > 16) return false;
        if (popcnt(self.by_color(Color.black)) > 16) return false;

        for (Color.all) |c| {
            if (count_of(c, PieceType.pawn) > 8) return false;
            if (count_of(c, PieceType.knight) > 9) return false;
            if (count_of(c, PieceType.bishop) > 9) return false;
            if (count_of(c, PieceType.rook) > 9) return false;
            if (count_of(c, PieceType.queen) > 9) return false;
            if (count_of(c, PieceType.king) != 1) return false;
        }

        // In check and not to move.
        const king_sq_white: Square = self.king_square(Color.white);
        const king_sq_black = self.king_square(Color.black);

        if (self.is_square_attacked_by(king_sq_white, Color.black) and self.stm.e != .white) {
            return false;
        }

        if (self.is_square_attacked_by(king_sq_black, Color.white) and self.stm.e != .black) {
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
        if (self.castling.rights == 0) {
            try writer.print("-", .{});
        }
        else {
            if (!self.is_960) {
                if (self.castling.rights & cf_white_short != 0) try writer.print("K", .{});
                if (self.castling.rights & cf_white_long != 0)  try writer.print("Q", .{});
                if (self.castling.rights & cf_black_short != 0) try writer.print("k", .{});
                if (self.castling.rights & cf_black_long != 0)  try writer.print("q", .{});
            }
            else {
                if (self.castling.rights & cf_white_short != 0) try writer.print("{u}", .{ self.castling.rook_start(Color.white, CastleType.short).char_of_file() - 32 });
                if (self.castling.rights & cf_white_long != 0)  try writer.print("{u}", .{ self.castling.rook_start(Color.white, CastleType.long).char_of_file() - 32 });
                if (self.castling.rights & cf_black_short != 0) try writer.print("{u}", .{ self.castling.rook_start(Color.black, CastleType.short).char_of_file() });
                if (self.castling.rights & cf_black_long != 0)  try writer.print("{u}", .{ self.castling.rook_start(Color.black, CastleType.long).char_of_file() });
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
        io.print_buffered("key: 0x{x:0>16} pawnkey: 0x{x:0>16} white_nonpawnkey: {x:0>16} black nonpawnkey: {x:0>16} minorkey: {x:0>16} majorkey: {x:0>16}\n", .{ self.key, self.pawnkey, self.nonpawnkeys[0], self.nonpawnkeys[1], self.minorkey, self.majorkey });
        io.print_buffered("rule50: {}\n", .{ self.rule50 });
        io.print_buffered("checkers: ", .{});
        if (self.checkmask != 0) {
            var bb: u64 = self.checkmask & self.by_color(self.stm.opp());
            while (bitloop(&bb)) |sq| {
                io.print_buffered("{t} ", .{sq.e});
            }
        }
        io.print_buffered("\n", .{});

        if (comptime lib.is_debug) {
            io.print_buffered("phases: {} {}\n", .{ self.phase_by_color[0], self.phase_by_color[1] });
            io.print_buffered("white material code: {x:0>12}\n", .{ self.material.decode_side(Color.white) });
            io.print_buffered("black material code: {x:0>12}\n", .{ self.material.decode_side(Color.black) });
            io.print_buffered("black material cast: {x:0>24}\n", .{ self.material.decode() });

            if (self.state_flags & gf_check != 0) io.print_buffered("state: sf_check\n", .{});
            if (self.state_flags & gf_pins != 0) io.print_buffered("state: sf_pins\n", .{});
            //if (self.state & gf_pins != 0)  io.print_buffered("state: gf_pins\n", .{});
        }

        io.flush();
    }

    /// Debug only.
    pub fn equals(self: *const Position, other: *const Position) bool {
        lib.not_in_release();
        const a: []const u8 = std.mem.asBytes(self);
        const b: []const u8 = std.mem.asBytes(other);
        return std.mem.eql(u8, a, b);
    }
};

/// Basic storage of moves.
pub const MoveStorage = struct {
    moves: [types.max_move_count]ExtMove,
    count: u8,

    pub fn init() MoveStorage {
        return .{ .moves = undefined, .count = 0 };
    }

    /// Required function.
    pub fn reset(self: *MoveStorage) void {
        self.count = 0;
    }

    /// Required function.
    pub fn store(self: *MoveStorage, extmove: ExtMove) void {
        assert(self.count < 224);
        self.moves[self.count] = extmove;
        self.count += 1;
    }

    pub fn slice(self: *const MoveStorage) []const ExtMove {
        return self.moves[0..self.count];
    }
};

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

pub const MoveFinder = struct {
    /// The required from square.
    from: Square,
    /// The required to square.
    to: Square,
    /// The required promotion piece **without** capture flag.
    prom_piece_flags: u4,
    /// The move, if found.
    extmove: ExtMove,

    pub fn init(from: Square, to: Square, prom_piece_flags: u4) MoveFinder {
        if (comptime lib.is_paranoid) {
            assert(prom_piece_flags & Move.capture == 0);
        }
        return .{ .from = from, .to = to, .prom_piece_flags = prom_piece_flags, .extmove = .empty };
    }

    /// Required function.
    pub fn reset(self: *MoveFinder) void {
        self.extmove = .empty;
    }

    /// Required function.
    pub fn store(self: *MoveFinder, extmove: ExtMove) void {
        if (self.from.u == extmove.move.from.u and self.to.u == extmove.move.to.u and (self.prom_piece_flags == 0 or self.prom_piece_flags == extmove.move.flags & ~Move.capture)) {
            self.extmove = extmove;
        }
    }

    pub fn found(self: *const MoveFinder) bool {
        return !self.extmove.move.is_empty();
    }
};

pub const Error = error {
    MissingKing,
    CastlingLogic,
    TooManyKings,
    InCheckAndNotToMove,
};
