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
const pawns_shift = funcs.pawns_shift;
const pawn_from = funcs.pawn_from;
const bitloop = funcs.bitloop;

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const Castle = types.Castle;
const ScorePair = types.ScorePair;

/// Must be called when the program stops.
pub fn finalize() void {
    layout_map.deinit(ctx.galloc);
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
    pins_diagonal: u64,
    pins_orthogonal: u64,
    is_960: bool,
    state_flags: u2,

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
            .pins_diagonal = 0,
            .pins_orthogonal = 0,
            .is_960 = false,
            .state_flags = 0,
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
        pos.pins_diagonal = 0;
        pos.pins_orthogonal = 0;
        pos.is_960 = false;
        pos.state_flags = 0;
        pos.init_hash();
        return pos;
    }

    /// Comptime only.
    fn create_board_from_backrow(backrow: [8]PieceType) [Square.count]Piece {
        var result: [Square.count]Piece = @splat(.no_piece);
        for (backrow, 0..) |pt, i| {
            const sq: Square = .from_usize(i);
            result[sq.u] = .init(pt, .white);
            result[sq.u + 8] = .white_pawn;
            result[sq.u + 48] = .black_pawn;
            result[sq.u + 56] = .init(pt, .black);
        }
        return result;
    }

    /// Shorthand direct setup.
    pub fn init(fen_str: []const u8, is_960: bool) !Position {
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
                                const pc: Piece = try .from_char(c);
                                const sq: Square = .from_rank_file(rank, file);
                                self.add_piece_ex(pc, sq);
                                file +|= 1;
                            },
                        }
                    }
                    // Board ready:
                    if (popcnt(self.kings(.white)) != 1 or popcnt(self.kings(.black)) != 1) {
                        return Error.InvalidKings;
                    }
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

        self.is_960 |= self.detect_frc();
        self.layout = get_layout(layout_key);
        self.lazy_update_state();
    }

    /// Assumes king is on the board. Applies a rook correction for 'KQkq' castling rights, if nessesary.
    fn set_castling_right(self: *Position, key: *LayoutKey, us: Color, input_rook_file: u3, comptime rook_correction: bool) !void {
        const rank: u3 = funcs.relative_rank(us, 0);
        const rank_bb: u64 = funcs.relative_rank_bb(us, 0);
        const king_sq: Square = self.king_square(us);
        const rook_sq: Square = .from_rank_file(rank, input_rook_file);
        const rook: Piece = .init(.rook, us);
        var rook_file: u3 = input_rook_file;

        if (king_sq.coord.rank != rank) {
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
                const sq: Square = funcs.last_square_or_null(bb) orelse return Error.CastlingLogic;
                if (sq.u < king_sq.u) return Error.CastlingLogic;
                rook_file = sq.coord.file;
            }

            const cf: u4 = Castling.flag(us,.short);
            key.set_king(us, king_sq.coord.file);
            key.set_rook_file(cf, rook_file);
            self.castlingrights |= cf;
        }
        else if (rook_file < king_sq.coord.file) {
            if (rook_correction and !is_rook) {
                // Try to find the most left rook, left from the king.
                const bb: u64 = self.rooks(us) & rank_bb;
                const sq: Square = funcs.first_square_or_null(bb) orelse return Error.CastlingLogic;
                if (sq.u > king_sq.u) return Error.CastlingLogic;
                rook_file = sq.coord.file;
            }

            const cf: u4 = Castling.flag(us,.long);
            key.set_king(us, king_sq.coord.file);
            key.set_rook_file(cf, rook_file);
            self.castlingrights |= cf;
        }
    }

    fn detect_frc(self: *const Position) bool {
        return
            (self.is_castling_allowed(.white, .short) and self.get(.h1).e != .white_rook) or
            (self.is_castling_allowed(.white, .long)  and self.get(.a1).e != .white_rook) or
            (self.is_castling_allowed(.black, .short) and self.get(.h8).e != .black_rook) or
            (self.is_castling_allowed(.black, .long)  and self.get(.a8).e != .black_rook);
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
        var prom_flag: u4 = 0;

        // No promotion.
        if (str.len == 4) {
            // Castling. We only need to change target square if classic. In that case uci is different from our internal encoding.
            // In chess960 the uci encoding is the same as ours: king takes rook.
            // TODO: work to do here.

            // ORIGINAL
            // if (!self.is_960) {
            //     if (from.u == self.layout.king_start(us).u and self.get(from).e == .init(PieceType.king, us).e) {
            //         if (to.e == .g1 or to.e == .g8) {
            //             to = self.layout.rook_start(us, .short); // King takes rook.
            //         }
            //         else if (to.e == .c1 or to.e == .c8) {
            //             to = self.layout.rook_start(us, .long); // King takes rook.
            //         }
            //     }
            // }

            // I think this catches all situations. I encountered malformed castling moves in chess960 positions (not encoding 'king takes rook').
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
        // Promotion.
        else if (str.len == 5) {
            prom_flag = switch (str[4]) {
                'n' => Move.knight_promotion,
                'b' => Move.bishop_promotion,
                'r' => Move.rook_promotion,
                'q' => Move.queen_promotion,
                else => return types.ParsingError.InvalidPromotionChar,
            };
        }

        // TODO: replace with .find_move()

        var finder: MoveFinder = .init(from, to, prom_flag);
        self.lazy_generate_all_moves(&finder);

        if (finder.found()) {
            return finder.extmove; // return the exact move found.
        }
        return types.ParsingError.IllegalMove;
    }

    /// A convenient way to set the startposition without the need for a fen string.
    pub fn set_startpos(self: *Position) void {
        self.* = classic_startpos;
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
        return funcs.first_square(self.pawns(us));
    }

    /// Assumes there is a bishop. Returns lsb.
    pub fn bishop_square(self: *const Position, us: Color) Square {
        return funcs.first_square(self.bishops(us));
    }

    /// Assumes there is a king. Returns lsb.
    pub fn king_square(self: *const Position, us: Color) Square {
        return funcs.first_square(self.kings(us));
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

    /// Returns a bitboard of our pins. Remember these are rays.
    pub fn our_pins(self: *const Position) u64 {
        return self.pins_diagonal | self.pins_orthogonal;
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
        if (rank == bitboards.rank_3) {
            const w_pawn_sq = ep.add(8);
            const requirements: bool = self.board[w_pawn_sq.u].e == .white_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u - 8].e == .no_piece;
            return requirements and (bitboards.adjacent_square_masks[w_pawn_sq.u] & self.pawns(.black) != 0);
        }
        else if (rank == bitboards.rank_6) {
            const b_pawn_sq = ep.sub(8);
            const requirements:  bool = self.board[b_pawn_sq.u].e == .black_pawn and self.board[ep.u].e == .no_piece and self.board[ep.u + 8].e == .no_piece;
            return requirements and (bitboards.adjacent_square_masks[b_pawn_sq.u] & self.pawns(.white) != 0);
        }
        return false;
    }

    /// Also updates keys.
    pub fn add_piece_ex(self: *Position, pc: Piece, sq: Square) void {
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
    fn add_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void {
        if (comptime lib.is_paranoid) {
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
        if (comptime lib.is_paranoid) {
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
        if (comptime lib.is_paranoid) {
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

    /// Makes the move on the board.`us` is comptime for performance reasons and must be the `stm`.
    pub fn do_move(self: *Position, comptime us: Color, ex: ExtMove) void {
        if (comptime lib.is_paranoid) {
            assert(us.e == self.stm.e);
            self.assert_pos_ok(ex);
        }
        const predicted_key = if (comptime lib.is_paranoid) self.predict_key(us, ex) else void;

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

        if (comptime lib.is_paranoid) {
            assert(predicted_key == self.key);
            self.assert_pos_ok(ex);
        }
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
                const rook_to: Square = comptime Castling.rook_dest(us, .short); // king takes roshortok
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
                self.state_flags |= gf_pins;
            }
        }

        if (self.checkmask != 0) {
            self.state_flags |= gf_check;
        }
    }

    pub fn lazy_do_move(self: *Position, ex: ExtMove) void {
        switch (self.stm.e) {
            .white => self.do_move(.white, ex),
            .black => self.do_move(.black, ex),
        }
    }

    pub fn lazy_do_nullmove(self: *Position) void {
        switch (self.stm.e) {
            .white => self.do_nullmove(.white),
            .black => self.do_nullmove(.black),
        }
    }

    pub fn lazy_update_state(self: *Position) void {
        switch (self.stm.e) {
            .white => self.update_state(.white),
            .black => self.update_state(.black),
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

    // /// Returns true if square `to` is attacked by any piece of `attacker`.
    // pub fn is_square_attacked_by(self: *const Position, to: Square, comptime attacker: Color) bool {
    //     const inverted = comptime attacker.opp();
    //     return
    //         (attacks.get_knight_attacks(to) & self.knights(attacker)) |
    //         (attacks.get_king_attacks(to) & self.kings(attacker)) |
    //         (attacks.get_pawn_attacks(to, inverted) & self.pawns(attacker)) != 0
    //         or
    //         (attacks.get_rook_attacks(to, self.all()) & self.queens_rooks(attacker)) |
    //         (attacks.get_bishop_attacks(to, self.all()) & self.queens_bishops(attacker)) != 0;
    // }

    // /// Returns true if square `to` is attacked by any piece of `attacker` for a certain occupation `occ`.
    // pub fn is_square_attacked_by_for_occupation(self: *const Position, occ: u64, to: Square, comptime attacker: Color) bool {
    //     // Uses pawn inversion trick.
    //     const inverted = comptime attacker.opp();
    //     return
    //         (attacks.get_knight_attacks(to) & self.knights(attacker)) |
    //         (attacks.get_king_attacks(to) & self.kings(attacker)) |
    //         (attacks.get_pawn_attacks(to, inverted) & self.pawns(attacker)) != 0
    //         or
    //         (attacks.get_rook_attacks(to, occ) & self.queens_rooks(attacker)) |
    //         (attacks.get_bishop_attacks(to, occ) & self.queens_bishops(attacker)) != 0;
    // }

    /// Gives a bitboard of attackers which attack `to` for both colors.
    pub fn get_combined_attacks_to_for_occupation(self: *const Position, occ: u64, to: Square) u64 {
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

    pub fn lazy_generate_all_moves(self: *const Position, noalias storage: anytype) void {
        switch (self.stm.e) {
            .white => self.generate_all_moves(.white, storage),
            .black => self.generate_all_moves(.black, storage),
        }
    }

    pub fn lazy_generate_quiescence_moves(self: *const Position, noalias storage: anytype) void {
        switch (self.stm.e) {
            .white => self.generate_quiescence_moves(.white, storage),
            .black => self.generate_quiescence_moves(.black, storage),
        }
    }

    // Generate all legal moves.
    pub fn generate_all_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void {
        const color_flag: u4 = comptime if (us.e == .black) gf_black else 0;
        switch(self.state_flags) {
            inline else => |sf| self.gen(sf | color_flag, storage),
        }
    }

    /// Generate quiescence moves.
    /// - When not in check: Generate captures and queen promotions.
    /// - When in check: generate all legal moves and queen promotions.
    pub fn generate_quiescence_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void {
        const color_flag: u4 = comptime if (us.e == .black) gf_black else 0;
        switch(self.state_flags) {
            inline else => |sf| self.gen(sf | color_flag | gf_noisy, storage),
        }
    }

    /// See `MoveStorage` for the interface of `storage`. Required are the functions `reset() void` and `store(anytype, ExtMove) void`.
    fn gen(self: *const Position, comptime gen_flags: u4, noalias storage: anytype) void {
        storage.reset();

        const us: Color = comptime if (gen_flags & gf_black != 0) .black else .white;
        const them = comptime us.opp();
        const check: bool = comptime gen_flags & gf_check != 0;
        const noisy: bool = comptime gen_flags & gf_noisy != 0;
        const has_pins: bool = comptime gen_flags & gf_pins != 0;
        const do_all_promotions: bool = comptime !noisy;
        const checkers: u64 = self.checkmask & self.by_color(them);
        const doublecheck: bool = check and popcnt(checkers) > 1;
        const occ: u64 = self.all();
        const bb_us: u64 = self.by_color(us);
        const bb_them: u64 = self.by_color(them);
        const bb_not_us: u64 = ~bb_us;
        const king_sq: Square = self.king_square(us);

        var bb: u64 = undefined;

        // In case of a doublecheck we can only move the king.
        if (!doublecheck) {
            const pins_diagonal: u64 = if (has_pins) self.pins_diagonal else 0;
            const pins_orthogonal: u64 = if (has_pins) self.pins_orthogonal else 0;
            const pins: u64 = if (has_pins) pins_diagonal | pins_orthogonal else 0;
            const target = if (check) self.checkmask else if (!noisy) bb_not_us else bb_them;
            const our_pawns = self.pawns(us);
            const our_knights = self.knights(us);
            const our_queens_bishops = self.queens_bishops(us);
            const our_queens_rooks = self.queens_rooks(us);

            // Pawns.
            if (our_pawns != 0) {
                const pawn_us: Piece = comptime .init(.pawn, us);
                const third_rank: u64 = comptime funcs.relative_rank_3_bitboard(us);
                const seventh_rank: u64 = comptime funcs.relative_rank_7_bitboard(us);
                const last_rank: u64 = comptime funcs.relative_rank_8_bitboard(us);
                const empty_squares: u64 = ~occ;
                const enemies: u64 = if (check) checkers else bb_them;

                // First we generate the 4 types of pawnmoves: push, double push, capture left, capture right.
                const bb_single_push: u64 = (pawns_shift(our_pawns & ~pins, us, .up) & empty_squares) | (pawns_shift(our_pawns & pins_orthogonal, us, .up) & empty_squares & pins_orthogonal);
                const bb_double_push: u64 = pawns_shift(bb_single_push & third_rank, us, .up) & empty_squares;
                const bb_northwest: u64 = (pawns_shift(our_pawns & ~pins, us, .northwest) & enemies) | (pawns_shift(our_pawns & pins_diagonal, us, .northwest) & enemies & pins_diagonal);
                const bb_northeast: u64 = (pawns_shift(our_pawns & ~pins, us, .northeast) & enemies) | (pawns_shift(our_pawns & pins_diagonal, us, .northeast) & enemies & pins_diagonal);
                // Pawn push check interpolation.
                const bb_single: u64 = bb_single_push & target;
                const bb_double: u64 = bb_double_push & target;

                // Pawns normal.
                if (our_pawns & ~seventh_rank != 0) {
                    if (check or !noisy) {
                        bb = bb_single & ~last_rank;
                        while (bitloop(&bb)) |to| {
                            store(pawn_from(to, us, .up), to, Move.silent, pawn_us, .no_piece, storage);
                        }
                        bb = bb_double;
                        while (bitloop(&bb)) |to| {
                            store(if (us.e == .white) to.sub(16) else to.add(16), to, Move.double_push, pawn_us, .no_piece, storage);
                        }
                    }
                    bb = bb_northwest & ~last_rank;
                    while (bitloop(&bb)) |to| {
                        store(pawn_from(to, us, .northwest), to, Move.capture, pawn_us, self.board[to.u], storage);
                    }
                    bb = bb_northeast & ~last_rank;
                    while (bitloop(&bb)) |to| {
                        store(pawn_from(to, us, .northeast), to, Move.capture, pawn_us, self.board[to.u], storage);
                    }
                    if (self.ep_square.e != .a1) {
                        bb = attacks.get_pawn_attacks(self.ep_square, them) & our_pawns; // inversion trick.
                        while (bitloop(&bb)) |from| {
                            if (self.is_legal_enpassant(us, king_sq, from, self.ep_square)) {
                                const pawn_them: Piece = comptime .init(.pawn, them);
                                store(from, self.ep_square, Move.ep, pawn_us, pawn_them, storage);
                            }
                        }
                    }
                }
                // Pawn promotions.
                if (our_pawns & seventh_rank != 0) {
                    bb = bb_single & last_rank;
                    while (bitloop(&bb)) |to| {
                        self.store_promotions(us, do_all_promotions, false, pawn_from(to, us, .up), to, storage);
                    }
                    bb = bb_northwest & last_rank;
                    while (bitloop(&bb)) |to| {
                        self.store_promotions(us, do_all_promotions, true, pawn_from(to, us, .northwest), to, storage);
                    }
                    bb = bb_northeast & last_rank;
                    while (bitloop(&bb)) |to| {
                        self.store_promotions(us, do_all_promotions, true, pawn_from(to, us, .northeast), to, storage);
                    }
                }
            } // (pawns)

            // Pieces.
            bb = our_knights & ~pins;
            while (bitloop(&bb)) |from| {
                self.store_many(from, attacks.get_knight_attacks(from) & target, storage);
            }
            bb = our_queens_bishops & ~pins;
            while (bitloop(&bb)) |from| {
                self.store_many(from, attacks.get_bishop_attacks(from, occ) & target, storage);
            }
            bb = our_queens_bishops & pins_diagonal;
            while (bitloop(&bb)) |from| {
                self.store_many(from, attacks.get_bishop_attacks(from, occ) & target & pins_diagonal, storage);
            }
            bb = our_queens_rooks & ~pins;
            while (bitloop(&bb)) |from| {
                self.store_many(from, attacks.get_rook_attacks(from, occ) & target, storage);
            }
            bb = our_queens_rooks & pins_orthogonal;
            while (bitloop(&bb)) |from| {
                self.store_many(from, attacks.get_rook_attacks(from, occ) & target & pins_orthogonal, storage);
            }
        } // (not doublecheck)

        // King.
        const king_us: Piece = comptime .init(.king, us);
        const king_target = if (check or !noisy) bb_not_us else bb_them;
        bb = attacks.get_king_attacks(king_sq) & king_target;

        // The king is a troublemaker.
        // For now this 'popcount heuristic' gives the best avg speed, using 2 different approaches to check legality.
        if (popcnt(bb) > 2) {
            const bb_unsafe: u64 = self.get_unsafe_squares_for_king(us);
            bb &= ~bb_unsafe;
            // Normal.
            while (bitloop(&bb)) |to| {
                const capt: Piece = self.board[to.u];
                const flag: u4 = if (capt.e == .no_piece) Move.silent else Move.capture;
                store(king_sq, to, flag, king_us, capt, storage);
            }
            // Castling.
            if (!check and !noisy) {
                inline for (Castle.all) |ct| {
                    if (self.is_castling_allowed(us, ct) and self.is_castlingpath_empty(us, ct) and self.is_legal_castle(us, ct, bb_unsafe)) {
                        const flag: u4 = comptime if (ct.e == .short) Move.castle_short else Move.castle_long;
                        const to = self.layout.rook_start(us, ct); // king takes rook.
                        store(king_sq, to, flag, king_us, .no_piece, storage);
                    }
                }
            }
        }
        else {
            const bb_without_king: u64 = occ ^ self.kings(us);
            // Normal.
            while (bitloop(&bb)) |to| {
                if (self.is_legal_kingmove(us, bb_without_king, to)) {
                    const capt: Piece = self.board[to.u];
                    const movekind: u4 = if (capt.e == .no_piece) Move.silent else Move.capture;
                    store(king_sq, to, movekind, king_us, capt, storage);
                }
            }
            // Castling.
            if (!check and !noisy) {
                inline for (Castle.all) |ct| {
                    if (self.is_castling_allowed(us, ct) and self.is_castlingpath_empty(us, ct) and self.is_legal_castle_check_attacks(us, ct)) {
                        const movekind: u4 = comptime if (ct.e == .short) Move.castle_short else Move.castle_long;
                        const to = self.layout.rook_start(us, ct); // king takes rook.
                        store(king_sq, to, movekind, king_us, .no_piece, storage);
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

        const piece: Piece = .init(.pawn, us);
        const captured: Piece = if (is_capture) self.board[to.u] else .no_piece;

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

    /// Returns a bitboard with unsafe squares, x-raying the king.
    pub fn get_unsafe_squares_for_king(self: *const Position, comptime us: Color) u64 {
        return self.attacks_by_for_occupation(us.opp(), self.all() ^ self.kings(us));
    }

    pub fn is_castling_allowed(self: *const Position, comptime us: Color, comptime ct: Castle) bool {
        const flag = comptime Castling.flag(us, ct);
        return self.castlingrights & flag != 0;
    }

    fn is_castlingpath_empty(self: *const Position, comptime us: Color, comptime ct: Castle) bool {
        const path: u64 = self.layout.empty_path(us, ct);
        return path & self.all() == 0;
    }

    /// Compares the kings path with unsafe squares.
    fn is_legal_castle(self: *const Position, comptime us: Color, comptime ct: Castle, bb_unsafe: u64) bool {
        // Frc requires this additional pin check. In classic the rooks cannot be pinned.
        if (self.pins_orthogonal & self.layout.rook_start(us, ct).to_bitboard() != 0) return false;
        const path: u64 = self.layout.attack_path(us, ct);
        return path & bb_unsafe == 0;
    }

    fn is_legal_castle_check_attacks(self: *const Position, comptime us: Color, comptime ct: Castle) bool {
        const them: Color = comptime us.opp();
        // Frc requires this additional pin check. In classic the rooks cannot be pinned.
        if (self.pins_orthogonal & self.layout.rook_start(us, ct).to_bitboard() != 0) return false;
        var path: u64 = self.layout.attack_path(us, ct);
        while (bitloop(&path)) |sq| {
            if (self.is_square_attacked_by(sq, them)) return false;
        }
        return true;
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

        assert((self.state_flags & gf_check == 0 and self.checkmask == 0) or (self.state_flags & gf_check != 0 and self.checkmask != 0));
        assert((self.state_flags & gf_pins == 0 and self.our_pins() == 0) or (self.state_flags & gf_pins != 0 and self.our_pins() != 0));

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
        if (self.stm.e == .white) {
            try writer.print(" w", .{});
        }
        else {
            try writer.print(" b", .{});
        }

        // Castling rights.
        try writer.print(" ", .{});
        if (self.castlingrights == 0) {
            try writer.print("-", .{});
        }
        else {
            if (!self.is_960) {
                if (self.castlingrights & cf_white_short != 0) try writer.print("K", .{});
                if (self.castlingrights & cf_white_long != 0)  try writer.print("Q", .{});
                if (self.castlingrights & cf_black_short != 0) try writer.print("k", .{});
                if (self.castlingrights & cf_black_long != 0)  try writer.print("q", .{});
            }
            else {
                if (self.castlingrights & cf_white_short != 0) try writer.print("{u}", .{ self.layout.rook_start(.white, .short).char_of_file() - 32 }); // uppercase
                if (self.castlingrights & cf_white_long != 0)  try writer.print("{u}", .{ self.layout.rook_start(.white, .long).char_of_file() - 32 }); // uppercase
                if (self.castlingrights & cf_black_short != 0) try writer.print("{u}", .{ self.layout.rook_start(.black, .short).char_of_file() });
                if (self.castlingrights & cf_black_long != 0)  try writer.print("{u}", .{ self.layout.rook_start(.black, .long).char_of_file() });
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
            io.print_buffered("white material code: {x:0>12}\n", .{ self.material.decode_side(.white) });
            io.print_buffered("black material code: {x:0>12}\n", .{ self.material.decode_side(.black) });
            io.print_buffered("black material cast: {x:0>24}\n", .{ self.material.decode() });
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

/// Small key used during setup.
pub const LayoutKey = packed struct {
    white: StartFiles,
    black: StartFiles,

    pub const StartFiles = packed struct {
        king: u3,
        right_rook: u3,
        left_rook: u3,

        pub const classic: StartFiles = .{
            .king = 4, .right_rook = 7, .left_rook = 0,
        };
    };

    pub const empty: LayoutKey = std.mem.zeroes(LayoutKey);
    pub const classic: LayoutKey = .{ .white = .classic, .black = .classic };

    pub fn set_king(self: *LayoutKey, us: Color, file: u3) void {
        switch (us.e) {
            .white => self.white.king = file,
            .black => self.black.king = file,
        }
    }

    pub fn set_rook_file(self: *LayoutKey, castling_flag: u4, file: u3) void {
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

    fn from_layoutkey(key: LayoutKey) Layout {
        var result: Layout = .empty;

        inline for (Color.all) |us| {
            const rank: u3 = funcs.relative_rank(us, 0);
            inline for (Castle.all) |ct| {
                const king_file: u3 = key.king_file(us);
                const rook_file: u3 = key.rook_file(us, ct);
                const apply: bool = if (ct.e == .short) rook_file > king_file else rook_file < king_file;
                if (apply) {
                    const k: Square = .from_rank_file(rank, king_file);
                    const r: Square = .from_rank_file(rank, rook_file);
                    const all_flags: u4 = Castling.flag(us, .short) | Castling.flag(us, .long);
                    const flag: u4 = Castling.flag(us, ct);
                    result.king_start_squares[us.u] = k;
                    result.masks[k.u] |= all_flags;
                    result.masks[r.u] |= flag;
                    result.rook_start_squares[us.u][ct.u] = r;
                    result.empty_paths[us.u][ct.u] =
                        bitboards.get_squarepair(r, Castling.rook_dest(us, ct)).ray & ~k.to_bitboard() |
                        bitboards.get_squarepair(k, Castling.king_dest(us, ct)).ray & ~r.to_bitboard();
                    result.attack_paths[us.u][ct.u] = bitboards.get_squarepair(k, Castling.king_dest(us, ct)).ray;
                }
            }
        }

        return result;
    }

    fn king_start(self: *const Layout, us: Color) Square {
        return self.king_start_squares[us.u];
    }

    fn rook_start(self: *const Layout, us: Color, ct: Castle) Square {
        return self.rook_start_squares[us.u][ct.u];
    }

    fn empty_path(self: *const Layout, us: Color, ct: Castle) u64 {
        return self.empty_paths[us.u][ct.u];
    }

    fn attack_path(self: *const Layout, us: Color, ct: Castle) u64 {
        return self.attack_paths[us.u][ct.u];
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

/// For move parsing.
pub const MoveFinder = struct {
    /// The required from square.
    from: Square,
    /// The required to square.
    to: Square,
    /// The required promotion piece **without** capture flag.
    prom_piece_flag: u4,
    /// The move, if found.
    extmove: ExtMove,

    pub fn init(from: Square, to: Square, prom_piece_flag: u4) MoveFinder {
        if (comptime lib.is_paranoid) {
            assert(prom_piece_flag & Move.capture == 0);
        }
        return .{ .from = from, .to = to, .prom_piece_flag = prom_piece_flag, .extmove = .empty };
    }

    /// Required function.
    pub fn reset(self: *MoveFinder) void {
        self.extmove = .empty;
    }

    /// Required function.
    pub fn store(self: *MoveFinder, extmove: ExtMove) void {
        if (self.from.u == extmove.move.from.u and self.to.u == extmove.move.to.u and (self.prom_piece_flag == 0 or self.prom_piece_flag == extmove.move.kind & ~Move.capture)) {
            self.extmove = extmove;
        }
    }

    pub fn found(self: *const MoveFinder) bool {
        return !self.extmove.move.is_empty();
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

/// A global hash containing the encountered castling layouts during setup of positions.
/// Supports Chess960 and theoretically also Chess324.
var layout_map: std.AutoHashMapUnmanaged(LayoutKey, Layout) = .empty;

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
const gf_check: u2 = 1 << 0; // state_flags + gen
const gf_pins : u2 = 1 << 1; // state_flags + gen
const gf_noisy: u4 = 1 << 2; // gen
const gf_black: u4 = 1 << 3; // gen

const GenOrder = struct {
    const silent: u2 = 0;
    const capture: u2 = 1;
    const promotion: u2 = 2;
    const promotion_capture: u2 = 3;
};

const EMPTY_LAYOUT: Layout = Layout.empty;
const empty_layout: *const Layout = &EMPTY_LAYOUT;

const CLASSIC_STARTPOS_LAYOUT: Layout = .from_layoutkey(LayoutKey.classic);
const classic_startpos_layout: *const Layout = &CLASSIC_STARTPOS_LAYOUT;

fn get_layout(key: LayoutKey) *const Layout {
    // If te key is empty or the startposition, do not use a hash entry, but return one of our global consts.
    if (key == LayoutKey.empty) {
        return empty_layout;
    }
    if (key == LayoutKey.classic) {
        return classic_startpos_layout;
    }
    const ptr = layout_map.getOrPut(ctx.galloc, key) catch wtf("frc_map", .{});
    if (!ptr.found_existing) {
        ptr.key_ptr.* = key;
        ptr.value_ptr.* = .from_layoutkey(key);
    }
    return ptr.value_ptr;
}

/// Debug only.
pub fn layout_map_entries() usize {
    return layout_map.count();
}