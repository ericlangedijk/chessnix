// TODO: checkout if we can use this file as a struct. handy for privates!
// TODO: use classic castle stuff for normal chess and monomorphize.

const std = @import("std");
const types = @import("types.zig");
const lib = @import("lib.zig");
const console = @import("console.zig");
const bitboards = @import("bitboards.zig");
const bits = @import("bits.zig");
const funcs = @import("funcs.zig");
const zobrist = @import("zobrist.zig");
const squarepairs = @import("squarepairs.zig");
const data = @import("data.zig");
const fen = @import("fen.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const wtf = lib.wtf;

pub const Orientation = types.Orientation;
pub const Direction = types.Direction;
pub const Color = types.Color;
pub const PieceType = types.PieceType;
pub const Piece = types.Piece;
pub const Square = types.Square;
pub const Move = types.Move;
pub const MoveType = types.MoveType;
pub const CastleType = types.CastleType;

// castling flags.
pub const cf_white_short: u4 = 0b0001;
pub const cf_white_long: u4 = 0b0010;
pub const cf_black_short: u4 = 0b0100;
pub const cf_black_long: u4 = 0b1000;
//const cf_all =

// the (bit) indexes of the castling flags.
pub const ci_white_short: u4 = 0;
pub const ci_white_long: u4 = 1;
pub const ci_black_short: u4 = 2;
pub const ci_black_long: u4 = 3;

const castling_flags: [4]u4 = .{ cf_white_short, cf_white_long, cf_black_short, cf_black_long };
const king_castle_destination_squares: [4]Square = .{ Square.G1, Square.C1, Square.G8, Square.C8 };
const rook_castle_destination_squares: [4]Square = .{ Square.F1, Square.D1, Square.F8, Square.D8 };

const CastlingInformation = struct
{
    sq_start_rook_left: Square,
    sq_start_rook_right: Square,
    sq_start_king_start: Square,
};

pub const StateList = std.ArrayListUnmanaged(StateInfo);

pub const StateInfo = struct
{
    pub const empty: StateInfo = .{};
    /// The pawn hashkey.
    /// (copied)
    pawnkey: u64 = 0,
    /// Draw counter, known to the players of the game: After 50 non-reversible moves (100 ply) it is a draw. This counter keeps track of the number of reversible moves that have been made.
    /// (copied)
    rule50: u16 = 0,
    /// The enpassant square of this state. A1 or NO_EP means: there is no ep_square.
    /// (copied)
    ep_square: ?Square = null,
    /// Bitflags for castlingrights: cf_white_short, cf_white_long, cf_black_short, cf_black_long.
    /// (copied)
    castling_rights: u4 = 0,
    /// The move that was played to reach the current position.
    /// (copied)
    last_move: Move = .empty,
    /// The piece that did the last_move.
    /// (updated)
    moved_piece: Piece = Piece.NO_PIECE,
    /// The piece that was captured with last_move.
    /// (updated)
    captured_piece: Piece = Piece.NO_PIECE,
    /// The board hashkey. Updated after each move.
    /// (updated)
    key: u64 = 0,
    /// Bitboard of the pieces that currently give check (of them).
    /// (updated)
    checkers: u64 = 0,
    /// Bitboard with the pinned pieces (of us).
    /// (updated)
    pinned: u64 = 0,
    /// Bitboard of the pieces that are pinning (of them).
    /// (updated)
    pinners: u64= 0,
};

pub const Position = struct
{
    /// The pieces on the 64 squares.
    board: [64]Piece,
    /// Bitboards occupation indexed by PieceType.
    /// * (0) is for all pieces combined (the complete occupation).
    /// * (7) is unsued.
    bb_by_type: [8]u64,
    /// Bitboards occupation indexed by color. by_type[piecetype] & by_side[color] will give the bitboard for one piece.
    /// * (0) white pieces.
    /// * (1) black pieces.
    bb_by_side: [2]u64,
    /// Supporting Chess960, the startfiles of:
    /// * (0) a-rook file or 'left' rook
    /// * (1) h-rook file or 'right' rook
    /// * (2) king file
    /// * This field *must* be filled before we can initialize the next ones.
    start_files: [3]u3,
    /// Supporting Chess960, deduced from start_siles.\
    /// * (0) white king
    /// * (1) black king
    /// * Initialized in constructors.
    king_start_squares: [2]Square,
    /// Supporting Chess960, deduced from start_files.
    /// * (0) ci_white_short right-rook white
    /// * (1) ci_white_long left-rook white
    /// * (2) ci_black_short  right-rook black (mirrored from white)
    /// * (3) ci_black_long left-rook black (mirrored from white)
    /// * Initialized in constructors.
    rook_start_squares: [4]Square,
    /// Supporting Chess960, deduced from startFiles.
    /// The castling bitboard *between* king and rook.
    /// * (0) `ci_white_short` white e1-h1 (short)
    /// * (1) `ci_white_long` white e1-a1 (long)
    /// * (2) `ci_black_short` black e8-h8 (short)
    /// * (3) `ci_black_long` black e8-a8 (long)
    /// * Initialized in constructors.
    castling_between_bitboards: [4]u64,
    /// Supporting Chess960, deduced from `start_files`.\
    /// Used for quick updates of castling rights during make move.
    castling_masks: [64]u4,
    /// The current side to move.
    to_move: Color,
    /// Depth during search.
    ply: u16,
    /// The real move number.
    game_ply: u16,
    // Is this chess960?
    is_960: bool,
    /// Moves and state-information history.
    state_list: StateList,
    /// For performance reasons we keep this pointer to the actual current state: the last item in the state_list.
    /// * It is always synchronized.
    /// * Use with care. It is writable even for a *const Position.
    current_state: *StateInfo,

    /// Creates an empty (invalid) board with one empty state.
    fn create() Position
    {
        const list: StateList = create_statelist();
        return Position
        {
            .board = @splat(Piece.NO_PIECE),
            .bb_by_type = @splat(0),
            .bb_by_side = @splat(0),
            .start_files = @splat(0),
            .king_start_squares = @splat(Square.A1),
            .rook_start_squares = @splat(Square.A1),
            .castling_between_bitboards = @splat(0),
            .castling_masks = @splat(0),
            .to_move = Color.WHITE,
            .ply = 0,
            .game_ply = 0,
            .is_960 = false,
            .state_list = list,
            .current_state = &list.items[0],
        };
    }

    /// Creates our statelist with one empty state.
    fn create_statelist() StateList
    {
        var list: StateList = .empty;
        list.append(ctx.galloc, StateInfo.empty) catch wtf();
        return list;
    }

    pub fn deinit(self: *Position) void
    {
        self.state_list.deinit(ctx.galloc);
    }

    /// Returns the classic startposition.
    pub fn new() Position
    {
        const backrow: [8]PieceType = .{ PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK };
        var pos: Position = .create();
        pos.init_pieces_from_backrow(backrow);
        pos.start_files = .{ bitboards.file_a, bitboards.file_h, bitboards.file_e };
        pos.current_state.castling_rights = 0b1111;
        pos.after_construction();
        return pos;
    }

    /// Sets startposition using backrow definition.
    fn init_pieces_from_backrow(self: *Position, backrow: [8]PieceType) void
    {
        for (backrow, 0..) |pt, i|
        {
            const sq: Square = Square.from_usize(i);
            self.add_piece(sq, Piece.make(pt, Color.WHITE));
            self.add_piece(sq.plus(8), Piece.W_PAWN);
            self.add_piece(sq.plus(48), Piece.B_PAWN);
            self.add_piece(sq.plus(56), Piece.make(pt, Color.BLACK));
        }
    }

    pub fn new_960(nr: usize) Position
    {
        const backrow: [8]PieceType = @import("chess960.zig").decode(nr);
        var pos: Position = .create();
        pos.is_960 = true;
        pos.state_list = std.ArrayListUnmanaged(StateInfo).initCapacity(ctx.galloc, 32) catch wtf();
        pos.init_pieces_from_backrow(backrow);
        pos.start_files = .{ bitboards.file_a, bitboards.file_h, bitboards.file_e };
        pos.current_state.st.castling_rights = 0b1111;
        pos.after_construction();
        return pos;
    }

    pub fn from_fen(fen_str: []const u8) !Position
    {
        const fenresult: fen.FenResult = try fen.decode(fen_str);
        var pos: Position = .create();
        for (fenresult.pieces, Square.all) |p, sq|
        {
            if (p.is_piece()) pos.add_piece(sq, p);
        }
        pos.start_files = .{ bitboards.file_a, bitboards.file_h, bitboards.file_e };
        pos.game_ply = fenresult.game_ply;
        pos.to_move = fenresult.to_move;
        pos.current_state.castling_rights = fenresult.castling_rights;
        pos.current_state.rule50 = fenresult.draw_count;
        pos.current_state.ep_square = fenresult.ep;
        pos.after_construction();
        return pos;
    }

    /// Clones a position.
    /// * If `including_history` is false, only the active state is copied and `ply` becomes zero. TODO: Repetitions get lost. we deleted the "stockfish" like repetition counter
    pub fn clone(src: *const Position, comptime including_history: bool) Position
    {
        var result: Position = src.*;
        if (!including_history)
        {
            result.ply = 0;
            result.state_list = create_statelist();
            result.current_state = result.get_actual_state();
            result.current_state.* = src.current_state.*;
        }
        else
        {
            result.state_list = src.state_list.clone(ctx.galloc) catch wtf();
            result.current_state = result.get_actual_state();
        }
        return result;
    }

    fn after_construction(self: *Position) void
    {
        self.init_castle_info();
        self.init_hash();
        self.lazy_update_state();
    }

    /// Initialize static vars for castling.
    fn init_castle_info(self: *Position) void
    {
        // white
        {
            const rook_left: Square = .from_rank_file(bitboards.rank_1, self.start_files[0]);
            const rook_right: Square = .from_rank_file(bitboards.rank_1, self.start_files[1]);
            const king: Square = .from_rank_file(bitboards.rank_1, self.start_files[2]);
            self.king_start_squares[0] = king;
            self.rook_start_squares[ci_white_short] = rook_right;
            self.rook_start_squares[ci_white_long] = rook_left;
            self.castling_between_bitboards[ci_white_short] = squarepairs.in_between_bitboard(king, rook_right);
            self.castling_between_bitboards[ci_white_long] = squarepairs.in_between_bitboard(king, rook_left);
            self.castling_masks[rook_right.u] = cf_white_short;
            self.castling_masks[rook_left.u] = cf_white_long;
            self.castling_masks[king.u] = cf_white_short | cf_white_long;
        }
        // black
        {
            const rook_left: Square = .from_rank_file(bitboards.rank_8, self.start_files[0]);
            const rook_right: Square = .from_rank_file(bitboards.rank_8, self.start_files[1]);
            const king: Square = .from_rank_file(bitboards.rank_8, self.start_files[2]);
            self.king_start_squares[1] = king;
            self.rook_start_squares[ci_black_short] = rook_right;
            self.rook_start_squares[ci_black_long] = rook_left;
            self.castling_between_bitboards[ci_black_short] = squarepairs.in_between_bitboard(king, rook_right);
            self.castling_between_bitboards[ci_black_long] = squarepairs.in_between_bitboard(king, rook_left);
            self.castling_masks[rook_right.u] = cf_black_short;
            self.castling_masks[rook_left.u] = cf_black_long;
            self.castling_masks[king.u] = cf_black_short | cf_black_long;
        }
    }

    fn init_hash(self: *Position) void
    {
        const st = self.current_state;
        st.key, st.pawnkey = self.compute_hashkeys();
    }

    fn compute_hashkeys(self: *const Position) struct {u64, u64}
    {
        const st = self.current_state;
        var k: u64 = 0;
        var p: u64 = 0;
        // Loop through occupied squares.
        var occ: u64 = self.all();
        while (occ != 0)
        {
            const sq = funcs.pop_square(&occ);
            const pc = self.get(sq);
            k ^= zobrist.piece_square(pc, sq);
            if (pc.is_pawn()) p ^= zobrist.piece_square(pc, sq);
        }
        k ^= zobrist.castling(st.castling_rights);
        if (st.ep_square) |ep| k ^= zobrist.enpassant(ep.file());
        if (self.to_move.e == .black) k ^= zobrist.btm();
        return .{ k, p};
    }

    /// Returns a readonly state.
    pub inline fn state(self: *const Position) *const StateInfo
    {
        return self.current_state;
    }

    pub fn get(self: *const Position, sq: Square) Piece
    {
        return self.board[sq.u];
    }

    pub fn in_check(self: *const Position) bool
    {
        return self.current_state.checkers > 0;
    }

    // /// Lazy function.
    // pub fn lazy_is_mate(self: *const Position) bool
    // {
    //     if (self.current_state.checkers == 0) return false;
    //     var any_storage: Any = .init();
    //     self.lazy_generate_moves(&any_storage);
    //     return !any_store.has_moves;
    // }

    pub fn by_type(self: *const Position, pt: PieceType) u64
    {
        return self.bb_by_type[pt.u];
    }

    pub fn by_side(self: *const Position, side: Color) u64
    {
        return self.bb_by_side[side.u];
    }

    pub fn pieces(self: *const Position, pt: PieceType, us: Color) u64
    {
        return self.by_type(pt) & self.by_side(us);
    }

    pub fn all(self: *const Position) u64
    {
        return self.by_type(PieceType.NONE);
    }

    pub fn all_pawns(self: *const Position) u64
    {
        return self.by_type(PieceType.PAWN);
    }

    pub fn all_knights(self: *const Position) u64
    {
        return self.by_type(PieceType.KNIGHT);
    }

    pub fn all_bishops(self: *const Position) u64
    {
        return self.by_type(PieceType.BISHOP);
    }

    pub fn all_rooks(self: *const Position) u64
    {
        return self.by_type(PieceType.ROOK);
    }

    pub fn all_queens(self: *const Position) u64
    {
        return self.by_type(PieceType.QUEEN);
    }

    pub fn all_kings(self: *const Position) u64
    {
        return self.by_type(PieceType.KING);
    }

    pub fn all_queens_bishops(self: *const Position) u64
    {
        return self.by_type(PieceType.QUEEN) | self.by_type(PieceType.BISHOP);
    }

    pub fn all_queens_rooks(self: *const Position) u64
    {
        return self.by_type(PieceType.QUEEN) | self.by_type(PieceType.ROOK);
    }

    pub fn pawns(self: *const Position, us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.PAWN);
    }

    pub fn knights(self: *const Position, us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.KNIGHT);
    }

    pub fn bishops(self: *const Position, us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.BISHOP);
    }

    pub fn rooks(self: *const Position, us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.ROOK);
    }

    pub fn queens(self: *const Position, us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.QUEEN);
    }

    pub fn kings(self: *const Position, us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.KING);
    }

    pub fn queens_bishops(self: *const Position, us: Color) u64
    {
        return (self.by_type(PieceType.QUEEN) | self.by_type(PieceType.BISHOP)) & self.by_side(us);
    }

    pub fn queens_rooks(self: *const Position, us: Color) u64
    {
        return (self.by_type(PieceType.QUEEN) | self.by_type(PieceType.ROOK)) & self.by_side(us);
    }

    pub fn white_pawns(self: *const Position) u64
    {
        return self.pawns(Color.WHITE);
    }

    pub fn black_pawns(self: *const Position) u64
    {
        return self.pawns(Color.BLACK);
    }

    pub fn king_square(self: *const Position, us: Color) Square
    {
        return funcs.first_square(self.kings(us));
    }

    pub fn sliders(self: *const Position, us: Color) u64
    {
        return (self.by_type(PieceType.QUEEN) | self.by_type(PieceType.ROOK) | self.by_type(PieceType.BISHOP)) & self.by_side(us);
    }

    pub fn add_piece(self: *Position, sq: Square, pc: Piece) void
    {
        assert(self.get(sq).is_empty());
        assert(pc.is_piece());
        const mask: u64 = sq.to_bitboard();
        const us: Color = pc.color();
        const pt: PieceType = pc.piecetype;
        self.board[sq.idx()] = pc;
        self.bb_by_type[0] |= mask;
        self.bb_by_type[pt.u] |= mask;
        self.bb_by_side[us.u] |= mask;
    }

    pub fn remove_piece(self: *Position, sq: Square) void
    {
        assert(self.get(sq).is_piece());
        const pc: Piece = self.board[sq.u];
        const pt: PieceType = pc.piecetype;
        const mask: u64 = ~sq.to_bitboard();
        const us: Color = pc.color();
        self.board[sq.u] = Piece.NO_PIECE;
        self.bb_by_type[pt.u] &= mask;
        self.bb_by_type[0] &= mask;
        self.bb_by_side[us.u] &= mask;
    }

    pub fn move_piece(self: *Position, from: Square, to: Square) void
    {
        assert(self.get(from).is_piece());
        assert(self.get(to).is_empty());
        const pc: Piece = self.board[from.u];
        const pt: PieceType = pc.piecetype;
        const mask: u64 = from.to_bitboard() | to.to_bitboard();
        const us: Color = pc.color();
        self.board[from.u] = Piece.NO_PIECE;
        self.board[to.u] = pc;
        self.bb_by_type[pt.u] ^= mask;
        self.bb_by_type[0] ^= mask;
        self.bb_by_side[us.u] ^= mask;
    }

    /// Only used in make move. This must be called *before* incrementing the ply.
    fn push_state(self: *Position) *StateInfo
    {
        assert(self.ply + 1 == self.state_list.items.len);
        const st: *StateInfo = self.state_list.addOne(ctx.galloc) catch wtf();
        const old: *const StateInfo = &self.state_list.items[self.ply];
        // Copy some stuff. The reset is updated in make move.
        // Until then the rest of the new state is undefined.
        st.pawnkey = old.pawnkey;
        st.rule50 = old.rule50;
        st.ep_square = old.ep_square;
        st.castling_rights = old.castling_rights;
        self.current_state = st;
        return st;
    }

    // Only used in ummake move.
    fn pop_state(self: *Position) void
    {
        self.state_list.items.len -= 1;
        self.current_state = funcs.ptr_sub(StateInfo, self.current_state, 1);
    }

    /// Returns the actual pointer the the last item in the statelist.
    fn get_actual_state(self: *Position) *StateInfo
    {
        assert(self.ply + 1 == self.state_list.items.len);
        return &self.state_list.items[self.ply];
    }

    pub fn is_threefold_repetition(self: *const Position) bool
    {
        const st = self.current_state;
        if (st.rule50 < 4) return false;

        const end: u16 = @min(st.rule50, self.ply);
        if (end < 4) return false;

        var count: u2 = 0;
        var i: u16 = 4;
        var run: [*]StateInfo = @ptrCast(@constCast(st));
        run -= 2;
        while (i <= end) : (i += 2)
        {
            run -= 2;
            if (run[0].key == st.key)
            {
                count += 1;
                if (count >= 2) return true;
            }
        }
        return false;
    }

    pub fn lazy_make_move(self: *Position, move: Move) void
    {
        switch (self.to_move.e)
        {
            .white => self.make_move(Color.WHITE, move),
            .black => self.make_move(Color.BLACK, move),
        }
    }

    pub fn lazy_unmake_move(self: *Position) void
    {
        switch (self.to_move.e)
        {
            .white => self.make_move(Color.BLACK),
            .black => self.make_move(Color.WHITE),
        }
    }

    /// * Makes the move on the board.
    /// * Color is comptime for performance reasons and must be the stm.
    pub fn make_move(self: *Position, comptime us: Color, m: Move) void
    {
        if (lib.is_paranoid)
        {
            assert(us.e == self.to_move.e);
            assert(self.pos_ok());
        }

        var key: u64 = self.current_state.key ^ zobrist.btm();

        const st: *StateInfo = self.push_state();
        const them: Color = comptime us.opp();
        const from_sq = m.from;
        const to_sq = m.to;
        const movetype = m.movetype;
        const pc: Piece = self.board[from_sq.u];
        const capt: Piece = if(movetype != .enpassant) self.board[to_sq.u] else pc.opp();
        const is_pawnmove: bool = pc.is_pawn();
        const is_capture: bool = capt.is_piece();
        const hash_delta = zobrist.piece_square(pc, from_sq) ^ zobrist.piece_square(pc, to_sq);

        st.rule50 += 1;
        st.last_move = m;
        st.moved_piece = pc;
        st.captured_piece = capt;

        self.ply += 1;
        self.game_ply += 1;
        self.to_move = them;

        // Reset drawcounter by default.
        if (is_pawnmove and movetype == .normal or is_capture)
        {
            st.rule50 = 0;
        }

        // Clear ep by default if it is set
        if (st.ep_square) |ep|
        {
            key ^= zobrist.enpassant(ep.file());
            st.ep_square = null;
        }

        // Update the castling rights
        if (st.castling_rights != 0)
        {
            const mask: u4 = self.castling_masks[from_sq.u] | self.castling_masks[to_sq.u];
            if (mask != 0)
            {
                key ^= zobrist.castling(st.castling_rights);
                st.castling_rights &= ~mask;
                key ^= zobrist.castling(st.castling_rights);
            }
        }

        switch (movetype)
        {
            .normal =>
            {
                if (is_capture)
                {
                    self.remove_piece(to_sq);
                    key ^= zobrist.piece_square(capt, to_sq);
                    if (capt.is_pawn()) st.pawnkey ^= zobrist.piece_square(capt, to_sq);
                }
                self.move_piece(from_sq, to_sq);
                key ^= hash_delta;
                if (is_pawnmove)
                {
                    st.pawnkey ^= hash_delta;
                    // Double pawn push
                    if (from_sq.u ^ to_sq.u == 16)
                    {
                        const ep: Square = if (us.e == .white) to_sq.minus(8) else to_sq.plus(8);
                        st.ep_square = ep;
                        key ^= zobrist.enpassant(ep.file());
                    }
                }
            },
            .promotion =>
            {
                const prom: Piece = m.prom.to_piece(us);
                if (is_capture)
                {
                    self.remove_piece(to_sq);
                    key ^= zobrist.piece_square(capt, to_sq);
                }
                self.remove_piece(from_sq);
                self.add_piece(to_sq, prom);
                key ^= zobrist.piece_square(pc, from_sq) ^ zobrist.piece_square(prom, to_sq);
                st.pawnkey ^= zobrist.piece_square(pc, from_sq);
            },
            .enpassant =>
            {
                const capt_sq: Square = if (us.e == .white) to_sq.minus(8) else to_sq.plus(8);
                self.remove_piece(capt_sq);
                self.move_piece(from_sq, to_sq);
                key ^= hash_delta ^ zobrist.piece_square(capt, capt_sq);
                st.pawnkey ^= hash_delta ^ zobrist.piece_square(capt, capt_sq);
            },
            .castle =>
            {
                // Be aware of the fact that castle moves are encoded as "king takes rook".
                const castletype: CastleType = m.castle_type();
                const king_to: Square = funcs.king_castle_to_square(us, castletype);
                const rook_to: Square = funcs.rook_castle_to_square(us, castletype);
                self.move_piece(from_sq, king_to);
                self.move_piece(to_sq, rook_to);
                key ^= zobrist.piece_square(pc, from_sq) ^ zobrist.piece_square(capt, to_sq) ^ zobrist.piece_square(pc, king_to) ^ zobrist.piece_square(capt, rook_to);
            },
        }

        st.key = key;
        self.update_state(them);

        if (comptime lib.is_paranoid)
        {
            assert(self.to_move.e == them.e);
            assert(self.pos_ok());
        }
    }

    /// This should be called with us == the color that moved the previous ply!
    pub fn unmake_move(self: *Position, comptime us: Color) void
    {
        if (comptime lib.is_paranoid)
        {
            assert(self.to_move.e != us.e);
            assert(self.pos_ok());
        }

        const st = self.current_state;
        self.to_move = us;
        self.ply -= 1;
        self.game_ply -= 1;

        const m: Move = st.last_move;
        const capt: Piece = st.captured_piece;
        const to_sq: Square = m.to;
        const from_sq: Square = m.from;

        switch(m.movetype)
        {
            .normal =>
            {
                self.move_piece(to_sq, from_sq);
                if (capt.is_piece()) self.add_piece(to_sq, capt);
            },
            .promotion =>
            {
                self.remove_piece(to_sq);
                self.add_piece(from_sq, Piece.make(PieceType.PAWN, us));
                if (capt.is_piece()) self.add_piece(to_sq, capt);
            },
            .enpassant =>
            {
                self.move_piece(to_sq, from_sq);
                const ep: Square = if (us.e == .white) to_sq.minus(8) else to_sq.plus(8);
                self.add_piece(ep, capt);
            },
            .castle =>
            {
                const castletype: CastleType = m.castle_type();
                const rook_to: Square = funcs.rook_castle_to_square(us, castletype);
                const king_to: Square = funcs.king_castle_to_square(us, castletype);
                self.move_piece(rook_to, to_sq);
                self.move_piece(king_to, from_sq);
            },
        }

        self.pop_state();

        if (comptime lib.is_paranoid)
        {
            assert(self.to_move.e == us.e);
            assert(self.pos_ok());
        }
    }

    fn lazy_update_state(self: *Position) void
    {
        switch (self.to_move.e)
        {
            .white => self.update_state(Color.WHITE),
            .black => self.update_state(Color.BLACK),
        }
    }

    /// Update current state: checks, pinned, pinners.
    fn update_state(self: *Position, comptime us: Color) void
    {
        const them: Color = comptime us.opp();
        const st: *StateInfo = self.current_state;
        const king_sq: Square = self.king_square(us);

        // Checkers
        st.checkers = self.get_attackers_to(king_sq) & self.by_side(them);

        // Pins.
        st.pinned = 0;
        st.pinners = 0;
        const bb_us: u64 = self.by_side(us);
        const bb_occupation_without_us: u64 = self.all() & ~bb_us;
        var pseudo_attackers: u64 = self.get_sliding_attackers_to_for_occupation(king_sq, bb_occupation_without_us) & self.sliders(them);
        while (pseudo_attackers != 0)
        {
            const sq: Square = funcs.pop_square(&pseudo_attackers);
            const pair = squarepairs.get(sq, king_sq);
            const bb_test: u64 = pair.in_between_bitboard & bb_us;
            // We can only have a pin when exactly 1 bit is set.
            if (@popCount(bb_test) != 1) continue;
            // Now we have a pinner + pinned piece for sure.
            const pt: PieceType = self.board[sq.u].piecetype;
            st.pinned |= bb_test;
            if (pt.e == .queen or pt.e == .bishop and pair.is_diagonal or pt.e == .rook and pair.is_straight)
            {
                st.pinners |= sq.to_bitboard();
            }
        }
    }

    /// Returns a bitboard of all attackers (both colors) to a square. For pawns we use an inversion trick.
    pub fn get_attackers_to(self: *const Position, sq: Square) u64
    {
        return
            (data.get_pawn_attacks(sq, Color.WHITE) & self.black_pawns()) |
            (data.get_pawn_attacks(sq, Color.BLACK) & self.white_pawns()) |
            (data.get_bishop_attacks(sq, self.all()) & self.all_queens_bishops()) |
            (data.get_rook_attacks(sq, self.all()) & self.all_queens_rooks()) |
            (data.get_knight_attacks(sq) & self.all_knights()) |
            (data.get_king_attacks(sq) & self.all_kings());
    }

    pub fn get_attackers_to_for_occupation(self: *const Position, sq: Square, occ: u64) u64
    {
        return
            (data.get_pawn_attacks(sq, Color.WHITE) & self.black_pawns()) |
            (data.get_pawn_attacks(sq, Color.BLACK) & self.white_pawns()) |
            (data.get_bishop_attacks(sq, occ) & self.all_queens_bishops()) |
            (data.get_rook_attacks(sq, occ) & self.all_queens_rooks()) |
            (data.get_knight_attacks(sq) & self.all_knights()) |
            (data.get_king_attacks(sq) & self.all_kings());
    }

    /// Returns a bitboard with all sliding pieces which are attacking square `sq` for a certain occupation.
    /// * For both colors.
    fn get_sliding_attackers_to_for_occupation(self: *const Position, sq: Square, occ: u64) u64
    {
        return
            (data.get_bishop_attacks(sq, occ) & self.all_queens_bishops()) |
            (data.get_rook_attacks(sq, occ) & self.all_queens_rooks());
    }

    /// Do not call for pawns.
    pub fn get_piece_attacks_from(self: *const Position, pt: PieceType, from_sq: Square) u64
    {
        return switch(pt.e)
        {
            .knight => data.get_knight_attacks(from_sq),
            .bishop => data.get_bishop_attacks(from_sq, self.all()),
            .rook => data.get_rook_attacks(from_sq, self.all()),
            .queen => data.get_queen_attacks(from_sq, self.all()),
            .king => data.get_king_attacks(from_sq),
            else => unreachable,
        };
    }

    pub fn is_square_attacked_by_for_occ(self: *const Position, occ: u64, to_sq: Square, comptime attacker: Color) bool
    {
        const inverted = comptime attacker.opp();
        if (data.get_knight_attacks(to_sq) & self.knights(attacker) != 0) return true;
        if (data.get_king_attacks(to_sq) & self.kings(attacker) != 0) return true;
        if (data.get_pawn_attacks(to_sq, inverted) & self.pawns(attacker) != 0) return true;
        if (data.get_bishop_attacks(to_sq, occ) & self.queens_bishops(attacker) != 0) return true;
        if (data.get_rook_attacks(to_sq, occ) & self.queens_rooks(attacker) != 0) return true;
        return false;
    }

    /// Returns true if square [sq] is attacked by any piece of [attacker].\
    /// We start with the easiest ones, to gain a little bit of speed.\
    /// For pawns we use an inversion trick.
    pub fn is_square_attacked_by(self: *const Position, to_sq: Square, comptime attacker: Color) bool
    {
        const them = comptime attacker.opp();

        if (data.get_knight_attacks(to_sq) & self.knights(attacker) != 0) return true;
        if (data.get_king_attacks(to_sq) & self.kings(attacker) != 0) return true;
        if (data.get_pawn_attacks(to_sq, them) & self.pawns(attacker) != 0) return true;
        if (data.get_bishop_attacks(to_sq, self.all()) & self.queens_bishops(attacker) != 0) return true;
        if (data.get_rook_attacks(to_sq, self.all()) & self.queens_rooks(attacker) != 0) return true;
        return false;
    }

    fn is_castling_allowed(self: *const Position, comptime castletype: CastleType, comptime us: Color) bool
    {
        const bit: u2 = comptime index_of(castletype, us);
        return bits.test_bit_u8(self.current_state.castling_rights, bit);
    }

    fn is_castlingpath_empty(self: *const Position, comptime castletype: CastleType, comptime us: Color) bool
    {
        const bit: u2 = comptime index_of(castletype, us);
        const path: u64 = self.castling_between_bitboards[bit];
        return path & self.all() == 0;
    }

    fn rook_castle_startsquare(self: *const Position, comptime castletype: CastleType, comptime us: Color) Square
    {
        const bit: u2 = comptime index_of(castletype, us);
        return self.rook_start_squares[bit];
    }

    fn index_of(comptime castletype: CastleType, comptime us: Color) u2
    {
        return switch(castletype)
        {
            .short => if (us.e == .white) ci_white_short else ci_black_short,
            .long => if (us.e == .white) ci_white_long else ci_black_long,
        };
    }

    // TODO: move to some debug unit.
    pub fn very_lazy_generate_moves(self: *const Position) std.BoundedArray(Move, 218)
    {
        var moves: std.BoundedArray(Move, 218) = .{};
        var st: MoveStorage = .init();
        self.lazy_generate_moves(&st);
        moves.appendSliceAssumeCapacity(st.slice());
        return moves;
    }

    pub fn lazy_generate_moves(self: *const Position, noalias storage: anytype) void
    {
        switch(self.to_move.e)
        {
            .white => self.generate_moves(Color.WHITE, storage),
            .black => self.generate_moves(Color.BLACK, storage),
        }
    }

    pub fn generate_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void
    {
        if (lib.is_paranoid)
        {
            assert(self.to_move.e == us.e);
        }

        storage.reset();
        switch(self.in_check())
        {
            false => self.gen(Params.create(us, .all), storage),
            true => self.gen(Params.create(us, .evasions), storage),
        }
    }

    fn gen(self: *const Position, comptime cpt: Params, noalias storage: anytype) void
    {
        const st = self.current_state;
        const us = comptime cpt.us;
        const them = comptime us.opp();
        const check: bool = comptime cpt.gentype == .evasions;

        const do_all_promotions = comptime cpt.gentype != .captures;
        const doublecheck: bool = check and @popCount(st.checkers) > 1;
        const bb_us = self.by_side(us);
        const bb_them = self.by_side(them);
        const king_sq: Square = self.king_square(us);
        const bb_not_us: u64 = ~bb_us;

        // We start with a bitboard with all empty squares + them squares.
        var target: u64 = bb_not_us;

        // Some reusable vars.
        var bb_from: u64 = 0;
        var bb_to: u64 = 0;
        var from_sq: Square = Square.zero;
        var to_sq: Square = Square.zero;
        var b1: u64 = 0;
        var b2: u64 = 0;
        var b3: u64 = 0;

        // When doing only captures the target bitboard is adjusted.
        if (cpt.gentype == .captures)
        {
            target &= bb_them;
        }

        // When in check update target again for the pieces: only interpolation or capture checker
        if (check)
        {
            const interpolationmask: u64 = if (doublecheck) 0 else squarepairs.in_between_bitboard(king_sq, funcs.first_square(st.checkers));
            target = st.checkers | interpolationmask;
        }

        // All pieces except the king, if we have a target. In case of a doublecheck we can only move the king.
        if (!doublecheck)
        {
            // Pawns
            const pawns_us = self.pawns(us);
            if (pawns_us != 0)
            {
                const empty_squares: u64 = ~self.all();
                const enemies: u64 = if (cpt.gentype == .evasions) st.checkers else bb_them;
                const pawns_on_seventh: u64 = pawns_us & funcs.relative_rank_7_bitboard(us);
                const pawns_not_on_seventh: u64 = pawns_us & ~pawns_on_seventh;

                // Pawn pushes, no promotions.
                if (cpt.gentype != .captures)
                {
                    b1 = funcs.pawns_shift(pawns_not_on_seventh, us, .up) & empty_squares; // single push
                    b2 = funcs.pawns_shift(b1 & funcs.relative_rank_3_bitboard(us), us, .up) & empty_squares; // double push
                    if (check)
                    {
                        b1 &= target;
                        b2 &= target;
                    }
                    while (b1 != 0)
                    {
                        to_sq = funcs.pop_square(&b1);
                        from_sq = if (us.e == .white) to_sq.minus(8) else to_sq.plus(8);
                        if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store(Move.create(from_sq, to_sq), storage);
                    }
                    while (b2 != 0)
                    {
                        to_sq = funcs.pop_square(&b2);
                        from_sq = if (us.e == .white) to_sq.minus(16) else to_sq.plus(16);
                        if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store(Move.create(from_sq, to_sq), storage);
                    }
                }

                // Promotions (including captures) are always generated. When captures then only queen promotions.
                if (pawns_on_seventh != 0)
                {

                    b2 = funcs.pawns_shift(pawns_on_seventh, us, .northwest) & enemies;
                    b3 = funcs.pawns_shift(pawns_on_seventh, us, .northeast) & enemies;
                    b1 = funcs.pawns_shift(pawns_on_seventh, us, .up) & empty_squares;

                    // Pawn push check interpolation
                    if (check) b1 &= target;

                    // north west
                    while (b2 != 0)
                    {
                        to_sq = funcs.pop_square(&b2);
                        from_sq = if (us.e == .white) to_sq.minus(7) else to_sq.plus(7);
                        if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store_promotions(do_all_promotions, from_sq, to_sq, storage);
                    }

                    // north east
                    while (b3 != 0)
                    {
                        to_sq = funcs.pop_square(&b3);
                        from_sq = if (us.e == .white) to_sq.minus(9) else to_sq.plus(9);
                        if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store_promotions(do_all_promotions, from_sq, to_sq, storage);
                    }

                    // push
                    while (b1 != 0)
                    {
                        to_sq = funcs.pop_square(&b1);
                        from_sq = if (us.e == .white) to_sq.minus(8) else to_sq.plus(8);
                        if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store_promotions(do_all_promotions, from_sq, to_sq, storage);
                    }
                }

                // Captures, no promotions.
                if (cpt.gentype == .captures or cpt.gentype == .all or cpt.gentype == .evasions)
                {
                    b2 = funcs.pawns_shift(pawns_not_on_seventh, us, .northwest) & enemies;
                    b3 = funcs.pawns_shift(pawns_not_on_seventh, us, .northeast) & enemies;

                    // north west
                    while (b2 != 0)
                    {
                        to_sq = funcs.pop_square(&b2);
                        from_sq = if (us.e == .white) to_sq.minus(7) else to_sq.plus(7);
                        if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store(Move.create(from_sq, to_sq), storage);
                    }

                    // north east
                    while (b3 != 0)
                    {
                        to_sq = funcs.pop_square(&b3);
                        from_sq = if (us.e == .white) to_sq.minus(9) else to_sq.plus(9);
                        if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store(Move.create(from_sq, to_sq), storage);
                    }

                    // enpassant
                    if (st.ep_square) |ep|
                    {
                        b1 = data.get_pawn_attacks(ep, them) & pawns_us; // inversion trick.
                        while (b1 != 0)
                        {
                            from_sq = funcs.pop_square(&b1);
                            if (self.is_legal_enpassant(us, king_sq, from_sq, ep)) do_store(Move.create_enpassant(from_sq, ep), storage);
                        }
                    }
                }
            }

            // Knight.
            bb_from = self.knights(us) & ~st.pinned; // a knight can never go out of a pin.
            while (bb_from != 0)
            {
                from_sq = funcs.pop_square(&bb_from);
                bb_to = data.get_knight_attacks(from_sq) & target;
                while (bb_to != 0)
                {
                    to_sq = funcs.pop_square(&bb_to);
                    do_store(Move.create(from_sq, to_sq), storage);
                }
            }

            // Bishop.
            bb_from = self.bishops(us);
            while (bb_from != 0)
            {
                from_sq = funcs.pop_square(&bb_from);
                bb_to = self.get_piece_attacks_from(PieceType.BISHOP, from_sq) & target;
                while (bb_to != 0)
                {
                    to_sq = funcs.pop_square(&bb_to);
                    if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store(Move.create(from_sq, to_sq), storage);
                }
            }

            // Rook.
            bb_from = self.rooks(us);
            while (bb_from != 0)
            {
                from_sq = funcs.pop_square(&bb_from);
                bb_to = self.get_piece_attacks_from(PieceType.ROOK, from_sq) & target;
                while (bb_to != 0)
                {
                    to_sq = funcs.pop_square(&bb_to);
                    if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store(Move.create(from_sq, to_sq), storage);
                }
            }

            // Queen.
            bb_from = self.queens(us);
            while (bb_from != 0)
            {
                from_sq = funcs.pop_square(&bb_from);
                bb_to = self.get_piece_attacks_from(PieceType.QUEEN, from_sq) & target;
                while (bb_to != 0)
                {
                    to_sq = funcs.pop_square(&bb_to);
                    if (self.is_legal_check_pins(king_sq, from_sq, to_sq)) do_store(Move.create(from_sq, to_sq), storage);
                }
            }
        }

        // And now the king.
        if (cpt.gentype == .captures)
        {
            target &= bb_them;
        }
        else
        {
            target |= bb_not_us;
        }

        // Normal king moves.
        bb_to = target & data.get_king_attacks(king_sq);
        while (bb_to != 0)
        {
            to_sq = funcs.pop_square(&bb_to);
            if (is_legal_kingmove(self, us, to_sq)) do_store(Move.create(king_sq, to_sq), storage);
        }

        // And finally castling.
        if (!check and cpt.gentype != .captures and st.castling_rights != 0)
        {
            if (self.is_castling_allowed(.short, us) and self.is_castlingpath_empty(.short, us) and self.is_legal_castle(.short, us, king_sq))
            {
                do_store(Move.create_castle(king_sq, self.rook_castle_startsquare(.short, us)), storage);
            }
            if (self.is_castling_allowed(.long, us) and self.is_castlingpath_empty(.long, us) and self.is_legal_castle(.long, us, king_sq))
            {
                do_store(Move.create_castle(king_sq, self.rook_castle_startsquare(.long, us)), storage);
            }
        }
    }

    inline fn do_store(move: Move, noalias storage: anytype) void
    {
        storage.store(move);
    }

    inline fn do_store_promotions(comptime all_prom: bool, from_sq: Square, to_sq: Square, noalias storage: anytype) void
    {
        do_store(Move.create_promotion(from_sq, to_sq, .queen), storage);
        if (all_prom)
        {
            do_store(Move.create_promotion(from_sq, to_sq, .rook), storage);
            do_store(Move.create_promotion(from_sq, to_sq, .bishop), storage);
            do_store(Move.create_promotion(from_sq, to_sq, .knight), storage);
        }
    }

    // TODO: finetune pins with direction.
    fn is_legal_check_pins(self: *const Position, king_sq: Square, from_sq: Square, to_sq: Square) bool
    {
        // Check if there are pins.
        if (!funcs.contains_square(self.current_state.pinned, from_sq)) return true;
        // If we move in the same direction as the pin-direction it is legal.
        const p1 = squarepairs.get(king_sq, from_sq);
        if (!p1.is_slider) return false;
        const p2 = squarepairs.get(king_sq, to_sq);
        return (p1.orientation == p2.orientation);
    }

    fn is_legal_enpassant(self: *const Position, comptime us: Color, king_sq: Square, from_sq: Square, to_sq: Square) bool
    {
        const them: Color = comptime us.opp();
        const capt_sq = if (us.e == .white) to_sq.minus(8) else to_sq.plus(8);
        const occ: u64 = (self.all() ^ from_sq.to_bitboard() ^ capt_sq.to_bitboard()) | to_sq.to_bitboard();
        const att: u64 =
            (data.get_rook_attacks(king_sq, occ) & self.queens_rooks(them)) |
            (data.get_bishop_attacks(king_sq, occ) & self.queens_bishops(them));
        return att == 0;
    }

    fn is_legal_kingmove(self: *const Position, comptime us: Color, to_sq: Square) bool
    {
        const them = comptime us.opp();
        const occ: u64 = self.all() ^ self.kings(us);
        return !self.is_square_attacked_by_for_occ(occ, to_sq, them);
    }

    fn is_legal_castle(self: *const Position, comptime castletype: CastleType, comptime us: Color, king_sq: Square) bool
    {
        // Check kingpath for attacks when castling. (except the king-square, king in check should be pre-checked).
        // * Tricky place for neverending loops.
        // * Remember that castling moves are internally encoded as "king takes rook".
        const them = comptime us.opp();
        const idx = comptime index_of(castletype, us);
        var sq: Square = king_sq;
        const king_to: Square = king_castle_destination_squares[idx];//  comptime funcs.king_castle_to_square(us, castletype);
        switch (castletype)
        {
            .short =>
            {
                while (sq.u != king_to.u)
                {
                    sq.u += 1;
                    if (self.is_square_attacked_by(sq, them)) return false;
                }
            },
            .long =>
            {
                while (sq.u != king_to.u)
                {
                    sq.u -= 1;
                    if (self.is_square_attacked_by(sq, them)) return false;
                }
            }
        }
        return true;
    }

    // TODO: belongs in eval.
    fn see(self: *const Position, m: Move) bool
    {
        const us: Color = self.to_move;
        var side: Color = us;
        const from_sq = m.from;
        const to_sq = m.to;
        const value_them = self.get(to_sq).value();
        const value_us = self.get(from_sq).value();
        if (value_them - value_us > 100)
        {
            return true;
        }
        var gain: [24]i16 = @splat(0);
        gain[0] = value_them;
        gain[1] = value_us - value_them;
        var depth: usize = 1;
        const queens_or_bishops = self.all_queens_bishops();
        const queens_or_rooks = self.all.queens_rooks();
        var occupation = self.all() ^ to_sq.to_bitboard() ^ from_sq.to_bitboard();
        var attackers: u64 = self.get_attackers_to_for_occupation(to_sq, occupation);
        var bb: u64 = 0;

        while (true)
        {
            attackers &= occupation;
            if (attackers == 0) break;
            side = side.opp();

            // Pawn.
            bb = attackers & self.pawns(side);
            if (bb != 0)
            {
                depth += 1;
                gain[depth] = PieceType.PAWN.value() - gain[depth - 1];
                if (@max(-gain[depth - 1], gain[depth]) < 0) return false; // prune
                funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 pawn
                attackers |= data.get_bishop_attacks(to_sq, occupation) & queens_or_bishops; // reveal next diagonal attacker.
                continue;
            }

            // Rook.
            bb = attackers & self.rooks(side);
            if (bb != 0)
            {
                depth += 1;
                gain[depth] = PieceType.ROOK.value() - gain[depth - 1];
                if (@max(-gain[depth - 1], gain[depth]) < 0) return false; // prune
                funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 pawn
                attackers |= data.get_rook_attacks(to_sq, occupation) & queens_or_rooks; // reveal next diagonal attacker.
                continue;
            }
        }

        // Bubble up the score
        depth -= 1;
        while (depth > 0) : (depth -= 1)
        {
            if (gain[depth] > -gain[depth - 1])
            {
                gain[depth - 1] = -gain[depth];
            }
        }
        return gain[0] >= 0;
    }

    fn pos_ok(self: *const Position) bool
    {
        lib.not_in_release();

        assert(self.current_state == &self.state_list.items[self.ply]);

        // An extra test on the kings bitboard.
        if (@popCount(self.all_kings()) != 2)
        {
            std.debug.print("KINGS", .{});
            return false;
        }

        if (@popCount(self.all()) > 32)
        {
            std.debug.print("TOO MANY PIECES", .{});
            return false;
        }

        const a, const b = self.compute_hashkeys();
        if (a != self.current_state.key)
        {
            std.debug.print("KEY {} <> {} lastmove {s} {s}\n", .{self.current_state.key, a, self.current_state.last_move.to_string().slice(), @tagName(self.current_state.last_move.movetype) });
            return false;
        }

        if (b != self.current_state.pawnkey)
        {
            std.debug.print("PAWNKEY {} <> {} lastmove {s} {s}\n", .{self.current_state.pawnkey, b, self.current_state.last_move.to_string().slice(), @tagName(self.current_state.last_move.movetype) });
            return false;
        }

        // in check and not to move
        const king_sq_white: Square = self.king_square(Color.WHITE);
        if (self.is_square_attacked_by(king_sq_white, Color.BLACK) and self.to_move.e != .white)
        {
            std.debug.print("CHECK\n", .{});
            print_pos(self);
            return false;
        }

        const king_sq_black = self.king_square(Color.BLACK);
        if (self.is_square_attacked_by(king_sq_black, Color.WHITE) and self.to_move.e != .black)
        {
            std.debug.print("CHECK\n", .{});
            print_pos(self);

            return false;
        }
        return true;
    }
};

/// 4 bits comptime struct for generating moves.
pub const Params = packed struct
{
    pub const GenType = enum(u2) { all, captures, quiets, evasions };

    us: Color,
    gentype: GenType,

    inline fn create(comptime us: Color, comptime gentype: GenType) Params
    {
        return Params
        {
            .us = us, .gentype = gentype,
        };
    }
};

/// Basic storage of moves.
pub const MoveStorage = struct
{
    moves: [218]Move,
    ptr: [*]Move,

    pub fn init() MoveStorage
    {
        var result: MoveStorage = undefined;
        result.ptr = &result.moves;
        return result;
    }

    /// Required funnction.
    pub fn reset(self: *MoveStorage) void
    {
        self.ptr = &self.moves;
    }

    /// Required funnction.
    pub fn store(self: *MoveStorage, move: Move) void
    {
        assert(self.len() < 218);
        self.ptr[0] = move;
        self.ptr += 1;
    }

    /// Required function.
    pub fn len(self: *const MoveStorage) usize
    {
        return self.ptr - &self.moves;
    }

    pub fn slice(self: *const MoveStorage) []const Move
    {
        return self.moves[0..self.len()];
    }
};

pub const JustCount = struct
{
    moves: usize,

    pub fn init() JustCount
    {
        return .{ .moves = 0 };
    }

    /// Required funnction.
    pub fn reset(self: *JustCount) void
    {
        self.moves = 0;
    }

    /// Required function.
    pub fn len(self: *const JustCount) usize
    {
        return self.moves;
    }

    /// Required funnction.
    pub fn store(self: *JustCount, move: Move) void
    {
        _ = move;
        self.moves += 1;
    }
};

pub const Any = struct
{
    has_moves: bool,

    pub fn init() Any
    {
        return .{ .has_moves = false };
    }

    /// Required funnction.
    pub fn reset(self: *Any) void
    {
        self.has_moves = false;
    }

    /// Required function.
    pub fn len(self: *const Any) usize
    {
        return @intFromBool(self.has_moves);
    }

    /// Required funnction.
    pub fn store(self: *Any, move: Move) void
    {
        _ = move;
        self.has_moves = true;
    }
};



pub fn print_pos(pos: *const Position) void
{
    var y: u3 = 7;
    while (true)
    {
        var x: u3 = 0;
        while (true)
        {
            const square: Square = .from_rank_file(y, x);
            const pc: Piece = pos.board[square.u];
            if (pc.is_empty()) console.print(". ", .{})
            else
            {
                const ch = pc.to_print_char();
                console.print("{u} ", .{ch});
            }
            if (x == 7) break;
            x += 1;
        }
        console.print("\n", .{});
        if (y == 0) break;
        y -= 1;
    }
    const lastmove = pos.current_state.last_move;
    console.print("\n", .{});
    console.print("incheck {}, last move {s}, key {x:0>16}\n", .{pos.in_check(), if (lastmove == Move.empty) "none" else lastmove.to_string().slice(), pos.current_state.key});
    for (pos.state_list.items[1..]) |*st|
    {
        console.print("{s} {s}, ", .{@tagName(st.last_move.movetype), st.last_move.to_string().slice()});
    }
    console.print("\n", .{});
}

