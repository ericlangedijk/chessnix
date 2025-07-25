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

const Value = types.Value;
const Orientation = types.Orientation;
const Direction = types.Direction;
const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const MoveType = types.MoveType;
const CastleType = types.CastleType;

// castling flags.
pub const cf_white_short: u4 = 0b0001;
pub const cf_white_long: u4 = 0b0010;
pub const cf_black_short: u4 = 0b0100;
pub const cf_black_long: u4 = 0b1000;
pub const cf_all: u4 = 0b1111;

// the (bit) indexes of the castling flags.
pub const ci_white_short: u4 = 0;
pub const ci_white_long: u4 = 1;
pub const ci_black_short: u4 = 2;
pub const ci_black_long: u4 = 3;

const king_castle_destination_squares: [4]Square = .{ Square.G1, Square.C1, Square.G8, Square.C8 };
const rook_castle_destination_squares: [4]Square = .{ Square.F1, Square.D1, Square.F8, Square.D8 };

const pop = funcs.pop;

pub const StateInfoList = std.ArrayListUnmanaged(StateInfo);

pub const StateInfo = struct
{
    pub const empty: StateInfo = .{};
    /// The pawn hashkey.
    /// (copied)
    pawnkey: u64 = 0,
    /// Draw counter, known to the players of the game: After 50 non-reversible moves (100 ply) it is a draw. This counter keeps track of the number of reversible moves that have been made.
    /// (copied)
    rule50: u16 = 0,
    /// The enpassant square of this state.
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
    /// Bitboard of the pieces that currently give check.
    /// (updated)
    checkers: u64 = 0,
    /// Bitboard with the pinned pieces.
    /// (updated)
    pinned: u64 = 0,
    /// Bitboard of the pieces that are pinning.
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
    /// The castling bitboard in between king and rook.
    /// * (0) `ci_white_short` white e1-h1 (short)
    /// * (1) `ci_white_long` white e1-a1 (long)
    /// * (2) `ci_black_short` black e8-h8 (short)
    /// * (3) `ci_black_long` black e8-a8 (long)
    /// * Initialized in constructors.
    castling_between_bitboards: [4]u64,
    /// Supporting Chess960, deduced from `start_files`.\
    /// Used for quick updates of castling rights during make move.
    castling_masks: [64]u4,
    /// Piece values sum.
    values: [2]Value,
    /// Material values sum.
    materials: [2]Value,
    /// The current side to move.
    to_move: Color,
    /// Depth during search.
    ply: u16,
    /// The real move number.
    game_ply: u16,
    // Is this chess960?
    is_960: bool,
    /// Moves and state-information history.
    history: StateInfoList,
    /// For performance reasons we keep this pointer to the actual current state: the last item in the statelist.
    /// * It is always synchronized.
    /// QUESTION: I am not sure if this is actually speeding up things.
    current_state: *StateInfo,

    /// Creates an empty (invalid) board with one empty state.
    fn create() Position
    {
        const list: StateInfoList = create_history();
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
            .values = @splat(0),
            .materials = @splat(0),
            .to_move = Color.WHITE,
            .ply = 0,
            .game_ply = 0,
            .is_960 = false,
            .history = list,
            .current_state = &list.items[0],
        };
    }

    /// Creates our statelist with one empty state.
    fn create_history() StateInfoList
    {
        // TODO: some arbitrary capacity is used here.
        // We could use an empty list or a number or maybe a maximum like 2048.
        // When using a maximum we could also use an array and no allocations are needed. But then there is heavier stack usage.
        var list: StateInfoList = StateInfoList.initCapacity(ctx.galloc, 32) catch wtf();
        list.items.len = 1;
        list.items[0] = StateInfo.empty;
        return list;
        //var list: StateInfoList = .empty;
        //list.append(ctx.galloc, StateInfo.empty) catch wtf();
    }

    pub fn deinit(self: *Position) void
    {
        self.history.deinit(ctx.galloc);
    }

    /// Returns the classic startposition.
    pub fn new() Position
    {
        const backrow: [8]PieceType = .{ PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK };
        var pos: Position = .create();
        pos.init_pieces_from_backrow(backrow);
        pos.start_files = .{ bitboards.file_a, bitboards.file_h, bitboards.file_e };
        pos.current_state.castling_rights = cf_all;
        pos.after_construction();
        return pos;
    }

    /// Sets startposition using backrow definition.
    fn init_pieces_from_backrow(self: *Position, backrow: [8]PieceType) void
    {
        for (backrow, 0..) |pt, i|
        {
            const sq: Square = Square.from_usize(i);
            self.add_piece_ext(Color.WHITE, Piece.make(pt, Color.WHITE), sq);
            self.add_piece_ext(Color.WHITE, Piece.W_PAWN, sq.add(8));
            self.add_piece_ext(Color.BLACK, Piece.B_PAWN, sq.add(48));
            self.add_piece_ext(Color.BLACK, Piece.make(pt, Color.BLACK), sq.add(56));
        }
    }

    pub fn new_960(nr: usize) Position
    {
        const backrow: [8]PieceType = @import("chess960.zig").decode(nr);
        var pos: Position = .create();
        pos.is_960 = true;
        pos.history = std.ArrayListUnmanaged(StateInfo).initCapacity(ctx.galloc, 32) catch wtf();
        pos.init_pieces_from_backrow(backrow);
        pos.start_files = .{ bitboards.file_a, bitboards.file_h, bitboards.file_e };
        pos.current_state.st.castling_rights = cf_all;
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
    /// * If `including_history` is false, only the active state is copied and `ply` becomes zero. TODO: Repetitions get lost.
    pub fn clone(src: *const Position, comptime including_history: bool) Position
    {
        var result: Position = src.*;
        if (!including_history)
        {
            result.ply = 0;
            result.history = create_history();
            result.current_state = result.get_actual_state_ptr();
            result.current_state.* = src.current_state.*;
        }
        else
        {
            result.history = src.history.clone(ctx.galloc) catch wtf();
            result.current_state = result.get_actual_state_ptr();
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
        const st: *StateInfo = self.current_state;
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
    pub fn state(self: *const Position) *const StateInfo
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

    pub fn by_type(self: *const Position, pt: PieceType) u64
    {
        return self.bb_by_type[pt.u];
    }

    pub fn by_side(self: *const Position, side: Color) u64
    {
        return self.bb_by_side[side.u];
    }

    pub fn all(self: *const Position) u64
    {
        return self.by_type(PieceType.NO_PIECETYPE);
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

    pub fn king_square(self: *const Position, us: Color) Square
    {
        return funcs.first_square(self.kings(us));
    }

    pub fn sliders(self: *const Position, comptime us: Color) u64
    {
        return (self.by_type(PieceType.QUEEN) | self.by_type(PieceType.ROOK) | self.by_type(PieceType.BISHOP)) & self.by_side(us);
    }

    /// Returns the sum of the white + black materials.
    pub fn material(self: *const Position) Value
    {
        return self.materials[0] + self.materials[1];
    }

    pub fn non_pawn_material(self: *const Position) Value
    {
        return (self.materials[0] + self.materials[1]) - (PieceType.PAWN.material() * @popCount(self.all_pawns()));
    }

    pub fn add_piece(self: *Position, sq: Square, pc: Piece) void
    {
        assert(self.get(sq).is_empty());
        assert(pc.is_piece());
        const mask: u64 = sq.to_bitboard();
        const us: Color = pc.color();
        const pt: PieceType = pc.piecetype;
        self.board[sq.u] = pc;
        self.bb_by_type[0] |= mask;
        self.bb_by_type[pt.u] |= mask;
        self.bb_by_side[us.u] |= mask;
        self.values[us.u] += pc.value();
        self.materials[us.u] += pc.material();
    }

    /// Fast comptime version.
    fn add_piece_ext(self: *Position, comptime us: Color, pc: Piece, sq: Square) void
    {
        assert(self.get(sq).is_empty());
        assert(pc.is_piece());
        assert(pc.color().e == us.e);
        const mask: u64 = sq.to_bitboard();
        self.board[sq.u] = pc;
        self.bb_by_type[0] |= mask;
        self.bb_by_type[pc.piecetype.u] |= mask;
        self.bb_by_side[us.u] |= mask;
        self.values[us.u] += pc.value();
        self.materials[us.u] += pc.material();
    }

    pub fn remove_piece(self: *Position, sq: Square) void
    {
        assert(self.get(sq).is_piece());
        const pc: Piece = self.board[sq.u];
        const pt: PieceType = pc.piecetype;
        const not_mask: u64 = ~sq.to_bitboard();
        const us: Color = pc.color();
        self.board[sq.u] = Piece.NO_PIECE;
        self.bb_by_type[0] &= not_mask;
        self.bb_by_type[pt.u] &= not_mask;
        self.bb_by_side[us.u] &= not_mask;
        self.values[us.u] -= pc.value();
        self.materials[us.u] -= pc.material();
    }

    /// Fast comptime version.
    fn remove_piece_ext(self: *Position, comptime us: Color, pc: Piece, sq: Square) void
    {
        assert(self.get(sq).e == pc.e);
        assert(pc.color().e == us.e);
        const not_mask: u64 = ~sq.to_bitboard();
        self.board[sq.u] = Piece.NO_PIECE;
        self.bb_by_type[0] &= not_mask;
        self.bb_by_type[pc.piecetype.u] &= not_mask;
        self.bb_by_side[us.u] &= not_mask;
        self.values[us.u] -= pc.value();
        self.materials[us.u] -= pc.material();
    }

    pub fn move_piece(self: *Position, from: Square, to: Square) void
    {
        assert(self.get(from).is_piece());
        assert(self.get(to).is_empty());
        const pc: Piece = self.board[from.u];
        const pt: PieceType = pc.piecetype;
        const xor_mask: u64 = from.to_bitboard() | to.to_bitboard();
        const us: Color = pc.color();
        self.board[from.u] = Piece.NO_PIECE;
        self.board[to.u] = pc;
        self.bb_by_type[0] ^= xor_mask;
        self.bb_by_type[pt.u] ^= xor_mask;
        self.bb_by_side[us.u] ^= xor_mask;
    }

    /// Fast comptime version.
    fn move_piece_ext(self: *Position, comptime us: Color, pc: Piece, from: Square, to: Square) void
    {
        assert(self.get(from).e == pc.e);
        assert(self.get(to).is_empty());
        assert(pc.color().e == us.e);
        const xor_mask: u64 = from.to_bitboard() | to.to_bitboard();
        self.board[from.u] = Piece.NO_PIECE;
        self.board[to.u] = pc;
        self.bb_by_type[0] ^= xor_mask;
        self.bb_by_type[pc.piecetype.u] ^= xor_mask;
        self.bb_by_side[us.u] ^= xor_mask;
    }

    /// Only used in make move. This must be called *before* incrementing the ply.
    fn push_state(self: *Position) *StateInfo
    {
        assert(self.ply + 1 == self.history.items.len);
        const st: *StateInfo = self.history.addOne(ctx.galloc) catch wtf();
        const old: *const StateInfo = &self.history.items[self.ply];
        // Copy some stuff. The reset is updated in make move + update state.
        // Until then the rest of the new state is undefined.
        st.pawnkey = old.pawnkey;
        st.rule50 = old.rule50;
        st.ep_square = old.ep_square;
        st.castling_rights = old.castling_rights;
        self.current_state = st;
        return st;
    }

    /// Only used in ummake move.
    fn pop_state(self: *Position) void
    {
        self.history.items.len -= 1;
        self.current_state = funcs.ptr_sub(StateInfo, self.current_state, 1);
    }

    /// Returns the actual pointer the the last item in the statelist.
    fn get_actual_state_ptr(self: *Position) *StateInfo
    {
        assert(self.ply + 1 == self.history.items.len);
        return &self.history.items[self.ply];
    }

    pub fn is_threefold_repetition(self: *const Position) bool
    {
        const st: *const StateInfo = self.current_state;
        if (st.rule50 < 4) return false;

        const end: u16 = @min(st.rule50, self.ply);
        if (end < 4) return false;

        var count: u2 = 0;
        var i: u16 = 4;
        var run: [*]const StateInfo = @ptrCast(st);
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
            .white => self.unmake_move(Color.BLACK),
            .black => self.unmake_move(Color.WHITE),
        }
    }

    /// * Makes the move on the board.
    /// * Color is comptime for performance reasons and must be the stm.
    pub fn make_move(self: *Position, comptime us: Color, m: Move) void
    {
        if (comptime lib.is_paranoid) assert(us.e == self.to_move.e);
        if (comptime lib.is_paranoid) assert(self.pos_ok());

        var key: u64 = self.current_state.key ^ zobrist.btm();

        const st: *StateInfo = self.push_state();
        const them: Color = comptime us.opp();
        const from: Square = m.from;
        const to: Square = m.to;
        const movetype: MoveType = m.movetype;
        const pc: Piece = self.board[from.u];
        const capt: Piece = if(movetype != .enpassant) self.board[to.u] else comptime Piece.create_pawn(them);
        const is_pawnmove: bool = pc.is_pawn();
        const is_capture: bool = capt.is_piece();
        const hash_delta = zobrist.piece_square(pc, from) ^ zobrist.piece_square(pc, to);

        st.rule50 += 1;
        st.last_move = m;
        st.moved_piece = pc;
        st.captured_piece = capt;

        self.ply += 1;
        self.game_ply += 1;
        self.to_move = them;

        // Reset drawcounter by default.
        if (is_pawnmove or is_capture)
        {
            st.rule50 = 0;
        }

        // Clear ep by default if it is set.
        if (st.ep_square) |ep|
        {
            key ^= zobrist.enpassant(ep.file());
            st.ep_square = null;
        }

        // Update the castling rights.
        if (st.castling_rights != 0)
        {
            const mask: u4 = self.castling_masks[from.u] | self.castling_masks[to.u];
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
                    self.remove_piece_ext(them, capt, to);
                    key ^= zobrist.piece_square(capt, to);
                    if (capt.is_pawn()) st.pawnkey ^= zobrist.piece_square(capt, to);
                }
                self.move_piece_ext(us, pc, from, to);
                key ^= hash_delta;
                if (is_pawnmove)
                {
                    st.pawnkey ^= hash_delta;
                    // Double pawn push.
                    if (from.u ^ to.u == 16)
                    {
                        const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                        st.ep_square = ep;
                        key ^= zobrist.enpassant(ep.file());
                    }
                }
            },
            .promotion =>
            {
                const prom: Piece = m.prom.to_piece(us);
                const pawn: Piece = comptime Piece.create_pawn(us);
                if (is_capture)
                {
                    self.remove_piece_ext(them, capt, to);
                    key ^= zobrist.piece_square(capt, to);
                }
                self.remove_piece_ext(us, pawn, from);
                self.add_piece_ext(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
                st.pawnkey ^= zobrist.piece_square(pc, from);
            },
            .enpassant =>
            {
                const pawn_us: Piece = comptime Piece.create_pawn(us);
                const pawn_them: Piece = comptime Piece.create_pawn(them);
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.remove_piece_ext(them, pawn_them, capt_sq);
                self.move_piece_ext(us, pawn_us, from, to);
                key ^= hash_delta ^ zobrist.piece_square(pawn_them, capt_sq);
                st.pawnkey ^= hash_delta ^ zobrist.piece_square(pawn_them, capt_sq);
            },
            .castle =>
            {
                // Castling is encoded as "king takes rook".
                const king: Piece = comptime Piece.create_king(us);
                const rook: Piece = comptime Piece.create_rook(us);
                const castletype: CastleType = m.castle_type();
                const king_to: Square = funcs.king_castle_to_square(us, castletype);
                const rook_to: Square = funcs.rook_castle_to_square(us, castletype);
                self.move_piece_ext(us, king, from, king_to);
                self.move_piece_ext(us, rook, to, rook_to);
                key ^= zobrist.piece_square(king, from) ^ zobrist.piece_square(king, king_to) ^ zobrist.piece_square(rook, to) ^ zobrist.piece_square(rook, rook_to);
            },
        }

        st.key = key;
        self.update_state(them);

        if (comptime lib.is_paranoid) assert(self.to_move.e == them.e);
        if (comptime lib.is_paranoid) assert(self.pos_ok());
    }

    /// This must be called with `us` being the color that moved on the previous ply!
    pub fn unmake_move(self: *Position, comptime us: Color) void
    {
        if (comptime lib.is_paranoid) assert(self.to_move.e != us.e);
        if (comptime lib.is_paranoid) assert(self.pos_ok());

        const st: *const StateInfo = self.current_state;

        self.to_move = us;
        self.ply -= 1;
        self.game_ply -= 1;

        const them: Color = comptime us.opp();
        const m: Move = st.last_move;
        const pc: Piece = st.moved_piece;
        const capt: Piece = st.captured_piece;
        const is_capture: bool = capt.is_piece();
        const from: Square = m.from;
        const to: Square = m.to;

        switch(m.movetype)
        {
            .normal =>
            {
                self.move_piece_ext(us, pc, to, from);
                if (is_capture) self.add_piece_ext(them, capt, to);
            },
            .promotion =>
            {
                const pawn: Piece = comptime Piece.create_pawn(us);
                const prom: Piece = m.prom.to_piece(us);
                self.remove_piece_ext(us, prom, to);
                self.add_piece_ext(us, pawn, from);
                if (is_capture) self.add_piece_ext(them, capt, to);
            },
            .enpassant =>
            {
                const pawn_us: Piece = comptime Piece.create_pawn(us);
                const pawn_them: Piece = comptime Piece.create_pawn(them);
                self.move_piece_ext(us, pawn_us, to, from);
                const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.add_piece_ext(them, pawn_them, ep);
            },
            .castle =>
            {
                // Castling is encoded as "king takes rook".
                const king: Piece = comptime Piece.create_king(us);
                const rook: Piece = comptime Piece.create_rook(us);
                const castletype: CastleType = m.castle_type();
                const rook_to: Square = funcs.rook_castle_to_square(us, castletype);
                const king_to: Square = funcs.king_castle_to_square(us, castletype);
                self.move_piece_ext(us, rook, rook_to, to);
                self.move_piece_ext(us, king, king_to, from);
            },
        }

        self.pop_state();

        if (comptime lib.is_paranoid) assert(self.to_move.e == us.e);
        if (comptime lib.is_paranoid) assert(self.pos_ok());
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
        const bb_all: u64 = self.all();

        // Checkers.
        st.checkers =
            (data.get_pawn_attacks(king_sq, us) & self.pawns(them)) |
            (data.get_knight_attacks(king_sq) & self.knights(them)) |
            (data.get_bishop_attacks(king_sq, bb_all) & self.queens_bishops(them)) |
            (data.get_rook_attacks(king_sq, bb_all) & self.queens_rooks(them));

        // Pins.
        st.pinned = 0;
        st.pinners = 0;
        const bb_us: u64 = self.bb_by_side[us.u];
        const bb_occupation_without_us: u64 = bb_all & ~bb_us;
        var pseudo_attackers: u64 =
            (data.get_bishop_attacks(king_sq, bb_occupation_without_us) & self.queens_bishops(them)) |
            (data.get_rook_attacks(king_sq, bb_occupation_without_us) & self.queens_rooks(them));

        while (pseudo_attackers != 0)
        {
            const attacker_sq: Square = funcs.pop_square(&pseudo_attackers);
            const pair = squarepairs.get(king_sq, attacker_sq);
            const bb_test: u64 = pair.in_between_bitboard & bb_us;
            // We can only have a pin when exactly 1 bit is set.
            if (@popCount(bb_test) == 1)
            {
                // Now we have a pinner + pinned piece for sure.
                const pt: PieceType = self.board[attacker_sq.u].piecetype;
                st.pinned |= bb_test;
                // This mask check is the same as: pt.e == .queen or pt.e == .bishop and pair.is_diagonal or pt.e == .rook and pair.is_orthogonal
                if (pt.u & pair.mask != 0) st.pinners |= attacker_sq.to_bitboard();
            }
        }
    }

    /// Returns true if square `sq` is attacked by any piece of `attacker`.
    pub fn is_square_attacked_by(self: *const Position, to: Square, comptime attacker: Color) bool
    {
        const inverted = comptime attacker.opp();
        // QUESTION: Why is this slower?
        // return
        //     (data.get_knight_attacks(to) & self.knights(attacker) != 0) or
        //     (data.get_king_attacks(to) & self.kings(attacker) != 0) or
        //     (data.get_pawn_attacks(to, inverted) & self.pawns(attacker) != 0) or
        //     (data.get_bishop_attacks(to, self.all()) & self.queens_bishops(attacker) != 0) or
        //     (data.get_rook_attacks(to, self.all()) & self.queens_rooks(attacker) != 0);
        return
            (data.get_knight_attacks(to) & self.knights(attacker)) |
            (data.get_king_attacks(to) & self.kings(attacker)) |
            (data.get_pawn_attacks(to, inverted) & self.pawns(attacker)) |
            (data.get_bishop_attacks(to, self.all()) & self.queens_bishops(attacker)) |
            (data.get_rook_attacks(to, self.all()) & self.queens_rooks(attacker)) != 0;
    }

    /// Returns true if square `sq` is attacked by any piece of `attacker` for a certain occupation `occ`.
    pub fn is_square_attacked_by_for_occupation(self: *const Position, occ: u64, to: Square, comptime attacker: Color) bool
    {
        const inverted = comptime attacker.opp();
        // QUESTION: Why is this slower?
        // return
        //     (data.get_knight_attacks(to) & self.knights(attacker) != 0) or
        //     (data.get_king_attacks(to) & self.kings(attacker) != 0) or
        //     (data.get_pawn_attacks(to), inverted) & self.pawns(attacker) != 0) or
        //     (data.get_bishop_attacks(to, occ) & self.queens_bishops(attacker) != 0) or
        //     (data.get_rook_attacks(to, occ) & self.queens_rooks(attacker) != 0);
        return
            (data.get_knight_attacks(to) & self.knights(attacker)) |
            (data.get_king_attacks(to) & self.kings(attacker)) |
            (data.get_pawn_attacks(to, inverted) & self.pawns(attacker)) |
            (data.get_bishop_attacks(to, occ) & self.queens_bishops(attacker)) |
            (data.get_rook_attacks(to, occ) & self.queens_rooks(attacker)) != 0;
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

    pub fn lazy_generate_moves(self: *const Position, noalias storage: anytype) void
    {
        switch(self.to_move.e)
        {
            .white => self.generate_moves(Color.WHITE, storage),
            .black => self.generate_moves(Color.BLACK, storage),
        }
    }

    pub fn lazy_generate_captures(self: *const Position, noalias storage: anytype) void
    {
        switch(self.to_move.e)
        {
            .white => self.generate_captures(Color.WHITE, storage),
            .black => self.generate_captures(Color.BLACK, storage),
        }
    }

    pub fn generate_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void
    {
        // NOTE: getting state flags with an inline else and create Params from that is easier but much slower than the switch.

        if (comptime lib.is_paranoid) assert(self.to_move.e == us.e);
        storage.reset();

        const check: bool = self.current_state.checkers != 0;
        const pins: bool = self.current_state.pinners != 0;
        switch (check)
        {
            false =>
            {
                switch (pins)
                {
                    false => self.gen(Params.create(false, false, us, false), storage),
                    true  => self.gen(Params.create(false, false, us, true), storage),
                }
            },
            true =>
                switch (pins)
                {
                    false => self.gen(Params.create(true, false, us, false), storage),
                    true  => self.gen(Params.create(true, false, us, true), storage),
                },
        }
    }

    pub fn generate_captures(self: *const Position, comptime us: Color, noalias storage: anytype) void
    {
        if (comptime lib.is_paranoid)
        {
            assert(self.to_move.e == us.e);
        }

        storage.reset();

        const check: bool = self.current_state.checkers != 0;
        const pins: bool = self.current_state.pinners != 0;
        switch(check)
        {
            false =>
            {
                switch (pins)
                {
                    false => self.gen(Params.create(false, true, us, false), storage),
                    true  => self.gen(Params.create(false, true, us, true), storage),
                }
            },
            true =>
            {
                switch (pins)
                {
                    false => self.gen(Params.create(true, false, us, false), storage),
                    true  => self.gen(Params.create(true, false, us, true), storage),
                }
            },
        }
    }

    /// See `MoveStorage` for the interface of storage.
    fn gen(self: *const Position, comptime ctp: Params, noalias storage: anytype) void
    {
        // Comptimes.
        const us = comptime ctp.us;
        const them = comptime us.opp();
        const skip_pins: bool = comptime !ctp.pins;

        // Some locals.
        const st: StateInfo = self.current_state.*; // QUESTION: copying the struct seems strangely enough somewhat faster.
        const doublecheck: bool = ctp.check() and @popCount(st.checkers) > 1;
        const bb_all = self.all();
        const bb_us = self.by_side(us);
        const bb_them = self.by_side(them);
        const bb_not_us: u64 = ~bb_us;
        const king_sq: Square = self.king_square(us);
        const not_pinned: u64 = if (ctp.pins) ~st.pinned else bitboards.bb_full;

        // We start with a bitboard with all empty squares + them squares.
        var target: u64 = bb_not_us;

        // When doing only captures the target bitboard is adjusted.
        if (ctp.captures())
        {
            target &= bb_them;
        }

        // When in check update target again for the pieces: only interpolation or capture checker.
        if (ctp.check())
        {
            const interpolationmask: u64 = if (doublecheck) 0 else squarepairs.in_between_bitboard(king_sq, funcs.first_square(st.checkers));
            target = st.checkers | interpolationmask;
        }

        // TODO: we could do here one more comptime skip: the target bitboard can be empty, but only in the case of generating captures. Otherwise there is always a target.

        // In case of a doublecheck we can only move the king.
        if (!doublecheck)
        {
            // Pawns.
            const bb_pawns = self.pawns(us);// self.bb_by_type[PieceType.PAWN.u] & bb_us;// typed[PieceType.PAWN.u] & bb_us;
            if (bb_pawns != 0)
            {
                const skip_pawn_pins: bool = skip_pins or (st.pinned & bb_pawns == 0);
                const empty_squares: u64 = ~bb_all;
                const enemies: u64 = if (ctp.check()) st.checkers else bb_them;
                const pawns_on_seventh: u64 = bb_pawns & funcs.relative_rank_7_bitboard(us);
                const pawns_not_on_seventh: u64 = bb_pawns & ~pawns_on_seventh;

                // Pawn pushes, no promotions.
                if (!ctp.captures())
                {
                    var bb_single_push: u64 = funcs.pawns_shift(pawns_not_on_seventh, us, .up) & empty_squares;
                    var bb_double_push: u64 = funcs.pawns_shift(bb_single_push & funcs.relative_rank_3_bitboard(us), us, .up) & empty_squares;
                    // Pawn push check interpolation.
                    if (ctp.check())
                    {
                        bb_single_push &= target;
                        bb_double_push &= target;
                    }
                    // Single.
                    while (bb_single_push != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_single_push);
                        const from: Square = if (us.e == .white) to.sub(8) else to.add(8);
                        if (skip_pawn_pins or self.is_legal_check_pin(.vertical, king_sq, from, to)) store(from, to, .normal, .no_prom, storage);
                    }
                    // Double.
                    while (bb_double_push != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_double_push);
                        const from: Square = if (us.e == .white) to.sub(16) else to.add(16);
                        if (skip_pawn_pins or self.is_legal_check_pin(.vertical, king_sq, from, to)) store(from, to, .normal, .no_prom, storage);
                    }
                }

                // Promotions (including captures) are always generated. When captures then only queen promotions.
                if (pawns_on_seventh != 0)
                {
                    var bb_northwest: u64 = funcs.pawns_shift(pawns_on_seventh, us, .northwest) & enemies;
                    var bb_norhteast: u64 = funcs.pawns_shift(pawns_on_seventh, us, .northeast) & enemies;
                    var bb_push: u64 = funcs.pawns_shift(pawns_on_seventh, us, .up) & empty_squares;
                    // Pawn push check interpolation.
                    if (ctp.check())
                    {
                        bb_push &= target;
                    }
                    // north west
                    while (bb_northwest != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_northwest);
                        const from: Square = if (us.e == .white) to.sub(7) else to.add(7);
                        if (skip_pawn_pins or self.is_legal_check_pin(.diagmain, king_sq, from, to)) store_promotions(!ctp.captures(), from, to, storage);
                    }
                    // north east
                    while (bb_norhteast != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_norhteast);
                        const from: Square = if (us.e == .white) to.sub(9) else to.add(9);
                        if (skip_pawn_pins or self.is_legal_check_pin(.diaganti, king_sq, from, to)) store_promotions(!ctp.captures(), from, to, storage);
                    }
                    // push
                    while (bb_push != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_push);
                        const from: Square = if (us.e == .white) to.sub(8) else to.add(8);
                        if (skip_pawn_pins or self.is_legal_check_pin(.vertical, king_sq, from, to)) store_promotions(!ctp.captures(), from, to, storage);
                    }
                }

                // Captures, no promotions.
                if (ctp.captures() or ctp.all())
                {
                    var bb_northwest: u64 = funcs.pawns_shift(pawns_not_on_seventh, us, .northwest) & enemies;
                    var bb_northeast: u64 = funcs.pawns_shift(pawns_not_on_seventh, us, .northeast) & enemies;
                    // north west
                    while (bb_northwest != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_northwest);
                        const from: Square = if (us.e == .white) to.sub(7) else to.add(7);
                        if (skip_pawn_pins or self.is_legal_check_pin(.diagmain, king_sq, from, to)) store(from, to, .normal, .no_prom, storage);
                    }
                    // north east
                    while (bb_northeast != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_northeast);
                        const from: Square = if (us.e == .white) to.sub(9) else to.add(9);
                        if (skip_pawn_pins or self.is_legal_check_pin(.diaganti, king_sq, from, to)) store(from, to, .normal, .no_prom, storage);
                    }
                    // enpassant
                    if (st.ep_square) |ep|
                    {
                        var bb_enpassant: u64 = data.get_pawn_attacks(ep, them) & bb_pawns; // inversion trick.
                        while (bb_enpassant != 0)
                        {
                            const from: Square = funcs.pop_square(&bb_enpassant);
                            if (self.is_legal_enpassant(us, king_sq, from, ep)) store(from, ep, .enpassant, .no_prom, storage);
                        }
                    }
                }
            }

            // Knights.
            {
                var bb_from: u64 = self.knights(us) & not_pinned; // A knight can never escape a pin.
                while (bb_from != 0)
                {
                    const from: Square = funcs.pop_square(&bb_from);
                    var bb_to: u64 = data.get_knight_attacks(from) & target;
                    // QUESTION: This inline loop is somewhat faster for knights than the default while loop. I don't know why. Inlining the other pieces has no effect or a bit worse.
                    inline for (0..8) |_|
                    {
                        if (bb_to == 0) break;
                        const to: Square = funcs.pop_square(&bb_to);
                        store(from, to, .normal, .no_prom, storage);
                    }
                }
            }

            // Bishops.
            {
                var bb_from: u64 = self.bishops(us);
                while (bb_from != 0)
                {
                    const from: Square = funcs.pop_square(&bb_from);
                    var bb_to: u64 = data.get_bishop_attacks(from, bb_all) & target;
                    while (bb_to != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_to);
                        if (skip_pins or self.is_legal_check_pin(null, king_sq, from, to)) store(from, to, .normal, .no_prom, storage);
                    }
                }
            }

            // Rooks.
            {
                var bb_from: u64 = self.rooks(us);
                while (bb_from != 0)
                {
                    const from: Square = funcs.pop_square(&bb_from);
                    var bb_to: u64 = data.get_rook_attacks(from, bb_all) & target;
                    while (bb_to != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_to);
                        if (skip_pins or self.is_legal_check_pin(null, king_sq, from, to)) store(from, to, .normal, .no_prom, storage);
                    }
                }
            }

            // Queens.
            {
                var bb_from: u64 = self.queens(us);
                while (bb_from != 0)
                {
                    const from: Square = funcs.pop_square(&bb_from);
                    var bb_to: u64 = data.get_queen_attacks(from, bb_all) & target;
                    while (bb_to != 0)
                    {
                        const to: Square = funcs.pop_square(&bb_to);
                        if (skip_pins or self.is_legal_check_pin(null, king_sq, from, to)) store(from, to, .normal, .no_prom, storage);
                    }
                }
            }

        } // <-- if (!doublecheck)

        // And now the king.
        if (ctp.captures())
        {
            target &= bb_them;
        }
        else
        {
            target = bb_not_us;
        }

        // Normal king moves.
        {
            var bb_to = data.get_king_attacks(king_sq) & target;
            while (bb_to != 0)
            {
                const to: Square = funcs.pop_square(&bb_to);
                if (is_legal_kingmove(self, us, to)) store(king_sq, to, .normal, .no_prom, storage);
            }
        }

        // Castling moves.
        if (!ctp.check() and !ctp.captures())
        {
            inline for (CastleType.all) |ct|
            {
                if (self.is_castling_allowed(ct, us) and self.is_castlingpath_empty(ct, us) and self.is_legal_castle(ct, us, king_sq))
                {
                    // Castling is encoded as "king takes rook".
                    const to: Square = self.rook_castle_startsquare(ct, us);
                    store(king_sq, to, .castle, .no_prom, storage);
                }
            }
        }
    }

    fn store(from: Square, to: Square, comptime movetype: MoveType, comptime prom: Move.Prom, noalias storage: anytype) void
    {
        storage.store(Move{ .from = from, .to = to, .prom = prom, .movetype = movetype} );
    }

    fn store_promotions(comptime all_prom: bool, from: Square, to: Square, noalias storage: anytype) void
    {
        store(from, to, .promotion, .queen, storage);
        if (all_prom)
        {
            store(from, to, .promotion, .rook, storage);
            store(from, to, .promotion, .bishop, storage);
            store(from, to, .promotion, .knight, storage);
        }
    }

    /// Checks if the piece is pinned and if we can move it.
    /// * If we move in the same direction as the pin-direction it is legal.
    /// * Orientation known: sometimes (like with pawn moves) we know up-front (hard-coded) in which direction we are moving and use this info.
    /// * Orientation null: it needs to be retrieved. in this case we definitely should have a slider pin or otherwise a bug.
    fn is_legal_check_pin(self: *const Position, comptime orientation: ?Orientation, king_sq: Square, from: Square, to: Square) bool
    {
        if (!funcs.contains_square(self.current_state.pinned, from)) return true;
        if (orientation) |ori|
        {
            return ori == squarepairs.get(king_sq, to).orientation;
        }
        else
        {
            return squarepairs.get(king_sq, from).orientation == squarepairs.get(king_sq, to).orientation;
        }
    }

    /// Tricky one. An ep move can uncover a check.
    fn is_legal_enpassant(self: *const Position, comptime us: Color, king_sq: Square, from: Square, to: Square) bool
    {
        const them: Color = comptime us.opp();
        const capt_sq = if (us.e == .white) to.sub(8) else to.add(8);
        const occ: u64 = (self.all() ^ from.to_bitboard() ^ capt_sq.to_bitboard()) | to.to_bitboard();
        const att: u64 =
            (data.get_rook_attacks(king_sq, occ) & self.queens_rooks(them)) |
            (data.get_bishop_attacks(king_sq, occ) & self.queens_bishops(them));
        return att == 0;
    }

    fn is_legal_kingmove(self: *const Position, comptime us: Color, to: Square) bool
    {
        const them = comptime us.opp();
        const occ: u64 = self.all() ^ self.kings(us);
        return !self.is_square_attacked_by_for_occupation(occ, to, them);
    }

    /// Check king's path for attacks when castling.
    /// * Except the king-square. We do not produce castling moves when in check.
    fn is_legal_castle(self: *const Position, comptime castletype: CastleType, comptime us: Color, king_sq: Square) bool
    {
        const them = comptime us.opp();
        const idx = comptime index_of(castletype, us);
        const king_to: Square = king_castle_destination_squares[idx];
        var sq: Square = king_sq;
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

    /// Debug only.
    pub fn pos_ok(self: *const Position) bool
    {
        lib.not_in_release();

        assert(self.current_state == &self.history.items[self.ply]);

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
    /// The color for which we are generating (state).
    us: Color = Color.WHITE,
    /// Are we in check?  (state)
    in_check: bool = false,
    /// There are pins. If not we can comptime skip all pin checks.  (state)
    pins: bool = false,
    /// Only produce captures. (option for quiet search)
    do_captures: bool = false,

    fn create(comptime in_check: bool, comptime do_captures: bool, comptime us: Color, comptime pins: bool) Params
    {
        return Params
        {
            .in_check = in_check, .do_captures = do_captures, .us = us, .pins = pins
        };
    }

    fn from_flags(comptime flags: u4) Params
    {
        return @bitCast(flags);
    }

    fn all(comptime self: Params) bool
    {
        return self.in_check or !self.do_captures;
    }

    fn captures(comptime self: Params) bool
    {
        return self.do_captures;
    }

    fn check(comptime self: Params) bool
    {
        return self.in_check;
    }
};

/// Basic storage of moves.
pub const MoveStorage = struct
{
    moves: [types.max_move_count]Move,
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
        if (comptime lib.is_paranoid) assert(self.len() < types.max_move_count); // assertion slows down I think.
        self.ptr[0] = move;
        self.ptr += 1;
    }

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

    /// Required funnction.
    pub fn store(self: *JustCount, move: Move) void
    {
        _ = move;
        self.moves += 1;
    }

    pub fn len(self: *const JustCount) usize
    {
        return self.moves;
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

    /// Required funnction.
    pub fn store(self: *Any, move: Move) void
    {
        _ = move;
        self.has_moves = true;
    }

    pub fn len(self: *const Any) usize
    {
        return @intFromBool(self.has_moves);
    }
};

/// Debug only.
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
    for (pos.history.items[1..]) |*st|
    {
        console.print("{s} {s}, ", .{@tagName(st.last_move.movetype), st.last_move.to_string().slice()});
    }
    console.print("\n", .{});
}

