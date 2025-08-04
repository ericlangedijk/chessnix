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
const fen = @import("fen.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const wtf = lib.wtf;

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
const CastleType = types.CastleType;

pub const fen_classic_startpos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";

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

pub const StateInfoList = std.ArrayListUnmanaged(StateInfo);

pub const StateInfo = struct
{
    pub const empty: StateInfo = .{};

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
    last_move: Move = .empty,
    /// The piece that did the last_move.
    moved_piece: Piece = Piece.NO_PIECE,
    /// The piece that was captured with last_move.
    captured_piece: Piece = Piece.NO_PIECE,
    /// The board hashkey. Updated after each move.
    key: u64 = 0,
    /// Bitboard of the pieces that currently give check.
    checkers: u64 = 0,
    /// The path from the enemy checker to the king (excluding the king, including the checker).
    /// * NOTE: Undefined when not in check.
    /// * NOTE: Not usable when in doublecheck.
    interpolationmask: u64 = 0,
    /// Bitboard with the diagonal pin rays (excluding the king, including the attacker).
    pins_diagonal: u64 = 0,
    /// Bitboard with the orthogonal pin rays (excluding the king, including the attacker).
    pins_orthogonal: u64 = 0,
    /// All pin rayss.
    pins: u64 = 0,
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
    /// The king 'walk' when castling.
    castling_king_paths: [4]u64,
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
    /// The real d.
    game_ply: u16,
    // Is this chess960?
    is_960: bool,
    /// Moves and state-information history.
    history: StateInfoList,
    /// For performance reasons we keep this pointer to the actual current state: the last item in the history.\
    /// This field is always synchronized.
    state: *StateInfo,

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
            .castling_king_paths = @splat(0),
            .castling_masks = @splat(0),
            .values = @splat(0),
            .materials = @splat(0),
            .to_move = Color.WHITE,
            .ply = 0,
            .game_ply = 0,
            .is_960 = false,
            .history = list,
            .state = &list.items[0],
        };
    }

    fn clear(self: *Position) void
    {
        self.board = @splat(Piece.NO_PIECE);
        self.bb_by_type = @splat(0);
        self.bb_by_side = @splat(0);
        self.start_files = @splat(0);
        self.king_start_squares = @splat(Square.A1);
        self.rook_start_squares = @splat(Square.A1);
        self.castling_between_bitboards = @splat(0);
        self.castling_king_paths = @splat(0);
        self.castling_masks = @splat(0);
        self.values = @splat(0);
        self.materials = @splat(0);
        self.to_move = Color.WHITE;
        self.ply = 0;
        self.game_ply = 0;
        self.is_960 = false;
        self.history.items.len = 1;
        self.state = self.get_actual_state_ptr();
        self.state.* = .empty;
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
        pos.state.castling_rights = cf_all;
        pos.after_construction();
        return pos;
    }

    /// Fills self with the classic startposition.
    pub fn set_startpos(self: *Position) void
    {
        const backrow: [8]PieceType = .{ PieceType.ROOK, PieceType.KNIGHT, PieceType.BISHOP, PieceType.QUEEN, PieceType.KING, PieceType.BISHOP, PieceType.KNIGHT, PieceType.ROOK };
        self.clear();
        self.init_pieces_from_backrow(backrow);
        self.start_files = .{ bitboards.file_a, bitboards.file_h, bitboards.file_e };
        self.state.castling_rights = cf_all;
        self.after_construction();
    }

    /// Sets startposition using backrow definition.
    fn init_pieces_from_backrow(self: *Position, backrow: [8]PieceType) void
    {
        for (backrow, 0..) |pt, i|
        {
            const sq: Square = Square.from_usize(i);
            self.add_piece(Color.WHITE, Piece.make(pt, Color.WHITE), sq);
            self.add_piece(Color.WHITE, Piece.W_PAWN, sq.add(8));
            self.add_piece(Color.BLACK, Piece.B_PAWN, sq.add(48));
            self.add_piece(Color.BLACK, Piece.make(pt, Color.BLACK), sq.add(56));
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
        pos.state.st.castling_rights = cf_all;
        pos.after_construction();
        return pos;
    }

    pub fn from_fen(fen_str: []const u8) !Position
    {
        const fenresult: fen.FenResult = try fen.decode(fen_str);
        var pos: Position = .create();
        for (fenresult.pieces, Square.all) |p, sq|
        {
            if (p.is_piece()) pos.lazy_add_piece(sq, p);
        }
        pos.start_files = .{ bitboards.file_a, bitboards.file_h, bitboards.file_e };
        pos.game_ply = fenresult.game_ply;
        pos.to_move = fenresult.to_move;
        pos.state.castling_rights = fenresult.castling_rights;
        pos.state.rule50 = fenresult.draw_count;
        // We only set ep if it is actually possible to do an ep-capture.
        if (fenresult.ep) |ep|
        {
            if (pos.is_usable_ep_square(ep)) pos.state.ep_square = ep;
        }
        pos.after_construction();
        return pos;
    }

    pub fn set_fen(self: *Position, fen_str: []const u8) !void
    {
        const fenresult: fen.FenResult = try fen.decode(fen_str);
        self.clear();
        for (fenresult.pieces, Square.all) |p, sq|
        {
            if (p.is_piece()) self.lazy_add_piece(sq, p);
        }
        self.start_files = .{ bitboards.file_a, bitboards.file_h, bitboards.file_e };
        self.game_ply = fenresult.game_ply;
        self.to_move = fenresult.to_move;
        self.state.castling_rights = fenresult.castling_rights;
        self.state.rule50 = fenresult.draw_count;
        // We only set ep if it is actually possible to do an ep-capture.
        if (fenresult.ep) |ep|
        {
            if (self.is_usable_ep_square(ep)) self.state.ep_square = ep;
        }
        self.after_construction();
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
            result.state = result.get_actual_state_ptr();
            result.state.* = src.state.*;
        }
        else
        {
            result.history = src.history.clone(ctx.galloc) catch wtf();
            result.state = result.get_actual_state_ptr();
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
            self.castling_king_paths[ci_white_short] = determine_king_path(king, Square.G1);
            self.castling_king_paths[ci_white_long] = determine_king_path(king, Square.C1);
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
            self.castling_king_paths[ci_black_short] = determine_king_path(king, Square.G8);
            self.castling_king_paths[ci_black_long] = determine_king_path(king, Square.C8);
            self.castling_masks[rook_right.u] = cf_black_short;
            self.castling_masks[rook_left.u] = cf_black_long;
            self.castling_masks[king.u] = cf_black_short | cf_black_long;
        }
    }

    fn determine_king_path(king_from: Square, king_to: Square) u64
    {
        var bb: u64 = king_from.to_bitboard();
        var k: Square = king_from;
        if (k.u < king_to.u)
        {
            while (k.u != king_to.u)
            {
                k.u += 1;
                bb |= k.to_bitboard();
            }
        }
        else if (k.u > king_to.u)
        {
            while (k.u != king_to.u)
            {
                k.u -= 1;
                bb |= k.to_bitboard();
            }
        }
        return bb;
    }

    fn init_hash(self: *Position) void
    {
        self.state.key = self.compute_hashkey();
    }

    fn compute_hashkey(self: *const Position) u64
    {
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
        if (st.ep_square) |ep| k ^= zobrist.enpassant(ep.file());
        if (self.to_move.e == .black) k ^= zobrist.btm();
        return k;
    }

    /// Returns a readonly state.
    pub fn get_state(self: *const Position) *const StateInfo
    {
        return self.state;
    }

    pub fn get(self: *const Position, sq: Square) Piece
    {
        return self.board[sq.u];
    }

    pub fn in_check(self: *const Position) bool
    {
        return self.state.checkers > 0;
    }

    /// Non-comptime getter for the outside world.
    pub fn pieces(self: *const Position, pt: PieceType, us: Color) u64
    {
        return switch(us.e)
        {
            .white => self.bb_by_type[pt.u] & self.by_side(Color.WHITE),
            .black => self.bb_by_type[pt.u] & self.by_side(Color.BLACK),
        };
    }

    pub fn by_type(self: *const Position, comptime pt: PieceType) u64
    {
        return self.bb_by_type[pt.u];
    }

    pub fn by_side(self: *const Position, comptime us: Color) u64
    {
        return self.bb_by_side[us.u];
    }

    pub fn all(self: *const Position) u64
    {
        return self.by_type(PieceType.NO_PIECETYPE);
    }

    pub fn all_pawns(self: *const Position) u64
    {
        return self.by_type(PieceType.PAWN);
    }

    pub fn pawns(self: *const Position, comptime us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.PAWN);
    }

    pub fn knights(self: *const Position, comptime us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.KNIGHT);
    }

    pub fn bishops(self: *const Position, comptime us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.BISHOP);
    }

    pub fn rooks(self: *const Position, comptime us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.ROOK);
    }

    pub fn queens(self: *const Position, comptime us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.QUEEN);
    }

    pub fn kings(self: *const Position, comptime us: Color) u64
    {
        return self.by_side(us) & self.by_type(PieceType.KING);
    }

    pub fn queens_bishops(self: *const Position, comptime us: Color) u64
    {
        return (self.by_type(PieceType.BISHOP) | self.by_type(PieceType.QUEEN)) & self.by_side(us);
    }

    pub fn queens_rooks(self: *const Position, comptime us: Color) u64
    {
        return (self.by_type(PieceType.ROOK) | self.by_type(PieceType.QUEEN)) & self.by_side(us);
    }

    pub fn king_square(self: *const Position, comptime us: Color) Square
    {
        return funcs.first_square(self.kings(us));
    }

    pub fn sliders(self: *const Position, comptime us: Color) u64
    {
        return (self.by_type(PieceType.BISHOP) | self.by_type(PieceType.ROOK) | self.by_type(PieceType.QUEEN)) & self.by_side(us);
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

    fn is_usable_ep_square(self: *const Position, ep: Square) bool
    {
        const rank: u3 = ep.rank();
        if (rank == bitboards.rank_3)
        {
            const to_sq = ep.add(8);
            return self.board[to_sq.u].e == .w_pawn and (masks.get_ep_mask(to_sq) & self.pawns(Color.BLACK) != 0);
        }
        else if (rank == bitboards.rank_6)
        {
            const to_sq = ep.sub(8);
            return self.board[to_sq.u].e == .b_pawn and (masks.get_ep_mask(to_sq) & self.pawns(Color.WHITE) != 0);
        }
        unreachable;
    }

    pub fn lazy_add_piece(self: *Position, sq: Square, pc: Piece) void
    {
        switch (pc.color().e)
        {
            .white => self.add_piece(Color.WHITE, pc, sq),
            .black => self.add_piece(Color.BLACK, pc, sq),
        }
    }

    fn add_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void
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

    fn remove_piece(self: *Position, comptime us: Color, pc: Piece, sq: Square) void
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

    /// Fast comptime version.
    fn move_piece(self: *Position, comptime us: Color, pc: Piece, from: Square, to: Square) void
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
        // Copy some stuff. The rest is updated in make move + update state.
        // Until then the rest of the new state is undefined.
        st.rule50 = old.rule50;
        st.ep_square = old.ep_square;
        st.castling_rights = old.castling_rights;
        self.state = st;
        return st;
    }

    /// Only used in ummake move.
    fn pop_state(self: *Position) void
    {
        self.history.items.len -= 1;
        self.state = funcs.ptr_sub(StateInfo, self.state, 1);
    }

    /// Returns the actual pointer the the last item in the statelist.
    fn get_actual_state_ptr(self: *Position) *StateInfo
    {
        assert(self.ply + 1 == self.history.items.len);
        return &self.history.items[self.ply];
    }

    pub fn is_threefold_repetition(self: *const Position) bool
    {
        const st: *const StateInfo = self.state;
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
        assert(us.e == self.to_move.e);
        if (comptime lib.is_paranoid) assert(self.pos_ok());

        var key: u64 = self.state.key ^ zobrist.btm();

        const st: *StateInfo = self.push_state();
        const them: Color = comptime us.opp();
        const from: Square = m.from;
        const to: Square = m.to;
        const movetype: MoveType = m.movetype;
        const pc: Piece = self.board[from.u];
        const capt: Piece = if (movetype != .enpassant) self.board[to.u] else comptime Piece.create_pawn(them);
        const is_pawnmove: bool = pc.is_pawn();
        const is_capture: bool = capt.is_piece();
        const hash_delta = zobrist.piece_square(pc, from) ^ zobrist.piece_square(pc, to);

        st.rule50 += 1;
        st.last_move = m;
        st.moved_piece = pc;
        st.captured_piece = capt;

        self.to_move = them;
        self.ply += 1;
        self.game_ply += 1;

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
                    self.remove_piece(them, capt, to);
                    key ^= zobrist.piece_square(capt, to);
                }
                self.move_piece(us, pc, from, to);
                key ^= hash_delta;
                if (is_pawnmove)
                {
                    // Double pawn push.
                    if (from.u ^ to.u == 16)
                    {
                        // We only set the ep-square if it is actually possible to do an ep-capture.
                        if (masks.get_ep_mask(to) & self.pawns(them) != 0)
                        {
                            const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                            st.ep_square = ep;
                            key ^= zobrist.enpassant(ep.file());
                        }
                    }
                }
            },
            .promotion =>
            {
                const prom: Piece = m.prom.to_piece(us);
                const pawn: Piece = comptime Piece.create_pawn(us);
                if (is_capture)
                {
                    self.remove_piece(them, capt, to);
                    key ^= zobrist.piece_square(capt, to);
                }
                self.remove_piece(us, pawn, from);
                self.add_piece(us, prom, to);
                key ^= zobrist.piece_square(pawn, from) ^ zobrist.piece_square(prom, to);
            },
            .enpassant =>
            {
                const pawn_us: Piece = comptime Piece.create_pawn(us);
                const pawn_them: Piece = comptime Piece.create_pawn(them);
                const capt_sq: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.remove_piece(them, pawn_them, capt_sq);
                self.move_piece(us, pawn_us, from, to);
                key ^= hash_delta ^ zobrist.piece_square(pawn_them, capt_sq);
            },
            .castle =>
            {
                // Castling is encoded as "king takes rook".
                const king: Piece = comptime Piece.create_king(us);
                const rook: Piece = comptime Piece.create_rook(us);
                const castletype: CastleType = m.castle_type();
                const king_to: Square = funcs.king_castle_to_square(us, castletype);
                const rook_to: Square = funcs.rook_castle_to_square(us, castletype);
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
    pub fn unmake_move(self: *Position, comptime us: Color) void
    {
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

        switch (m.movetype)
        {
            .normal =>
            {
                self.move_piece(us, st.moved_piece, to, from);
                if (is_capture) self.add_piece(them, capt, to);
            },
            .promotion =>
            {
                const pawn: Piece = comptime Piece.create_pawn(us);
                const prom: Piece = m.prom.to_piece(us);
                self.remove_piece(us, prom, to);
                self.add_piece(us, pawn, from);
                if (is_capture) self.add_piece(them, capt, to);
            },
            .enpassant =>
            {
                const pawn_us: Piece = comptime Piece.create_pawn(us);
                const pawn_them: Piece = comptime Piece.create_pawn(them);
                self.move_piece(us, pawn_us, to, from);
                const ep: Square = if (us.e == .white) to.sub(8) else to.add(8);
                self.add_piece(them, pawn_them, ep);
            },
            .castle =>
            {
                // Castling is encoded as "king takes rook".
                const king: Piece = comptime Piece.create_king(us);
                const rook: Piece = comptime Piece.create_rook(us);
                const castletype: CastleType = m.castle_type();
                const rook_to: Square = funcs.rook_castle_to_square(us, castletype);
                const king_to: Square = funcs.king_castle_to_square(us, castletype);
                self.move_piece(us, rook, rook_to, to);
                self.move_piece(us, king, king_to, from);
            },
        }

        self.pop_state();

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

    /// obsolete
    fn slower_original_update_state_(self: *Position, comptime us: Color) void
    {
        const them: Color = comptime us.opp();
        const st: *StateInfo = self.state;
        const king_sq: Square = self.king_square(us);
        const bb_all: u64 = self.all();

        const pawn_and_knight_checkers: u64 =
            (data.get_pawn_attacks(king_sq, us) & self.pawns(them)) |
            (data.get_knight_attacks(king_sq) & self.knights(them));

        const diag_checkers: u64 = data.get_bishop_attacks(king_sq, bb_all) & self.queens_bishops(them);
        const orth_checkers: u64 = data.get_rook_attacks(king_sq, bb_all) & self.queens_rooks(them);

        st.checkers = pawn_and_knight_checkers | diag_checkers | orth_checkers;

        if (@popCount(st.checkers) == 1)
        {
            st.interpolationmask = st.checkers;
            if (diag_checkers != 0)
            {
                const them_sq: Square = funcs.first_square(diag_checkers);
                st.interpolationmask |= squarepairs.in_between_bitboard(king_sq, them_sq);
            }
            else if (orth_checkers != 0)
            {
                const them_sq: Square = funcs.first_square(orth_checkers);
                st.interpolationmask |= squarepairs.in_between_bitboard(king_sq, them_sq);
            }
        }

        // Pins.
        st.pins_orthogonal = 0;
        st.pins_diagonal = 0;

        const bb_us: u64 = self.by_side(us);
        const bb_occupation_without_us: u64 = bb_all & ~bb_us;
        var candidate_pinners: u64 =
            (data.get_bishop_attacks(king_sq, bb_occupation_without_us) & self.queens_bishops(them)) |
            (data.get_rook_attacks(king_sq, bb_occupation_without_us) & self.queens_rooks(them));

        while (candidate_pinners != 0)
        {
            // TODO: can we extract attacker_sq directly as a u64 mask?
            const attacker_sq: Square = pop_square(&candidate_pinners);
            const pair = squarepairs.get(king_sq, attacker_sq);
            const bb_test: u64 = pair.in_between_bitboard & bb_us;
            // We can only have a pin when exactly 1 bit is set.
            if (@popCount(bb_test) == 1)
            {
                const attacker_square_bitboard: u64 = attacker_sq.to_bitboard();
                switch (pair.mask)
                {
                    0b100 => st.pins_orthogonal |= pair.in_between_bitboard | attacker_square_bitboard,
                    0b001 => st.pins_diagonal |= pair.in_between_bitboard | attacker_square_bitboard,
                    else => unreachable,
                }
            }
        }

        st.pins = st.pins_diagonal | st.pins_orthogonal;
    }

    fn update_state(self: *Position, comptime us: Color) void
    {
        const them: Color = comptime us.opp();
        const st: *StateInfo = self.state;
        const king_sq: Square = self.king_square(us);
        const bb_us: u64 = self.by_side(us);
        const bb_all: u64 = self.all();

        st.checkers =
            (data.get_pawn_attacks(king_sq, us) & self.pawns(them)) |
            (data.get_knight_attacks(king_sq) & self.knights(them));

        st.interpolationmask = st.checkers;
        st.pins_orthogonal = 0;
        st.pins_diagonal = 0;

        const bb_occupation_without_us: u64 = bb_all & ~bb_us;
        var candidate_attackers: u64 =
            (data.get_bishop_attacks(king_sq, bb_occupation_without_us) & self.queens_bishops(them)) |
            (data.get_rook_attacks(king_sq, bb_occupation_without_us) & self.queens_rooks(them));

        // Use candidate attackers for both checkers and pins.
        while (candidate_attackers != 0)
        {
            const attacker_sq: Square = pop_square(&candidate_attackers);
            const attacker_square_bitboard: u64 = attacker_sq.to_bitboard();
            const pair = squarepairs.get(king_sq, attacker_sq);
            const bb_ray: u64 = pair.in_between_bitboard & bb_us;
            // We have a slider checker when there is nothing in between.
            if (bb_ray == 0)
            {
                st.checkers |= attacker_square_bitboard;
                st.interpolationmask |= pair.in_between_bitboard | attacker_square_bitboard;
            }
            // We have a pin when exactly 1 bit is set. There is one piece in between.
            else if (@popCount(bb_ray) == 1)
            {
                switch (pair.mask)
                {
                    0b100 => st.pins_orthogonal |= pair.in_between_bitboard | attacker_square_bitboard,
                    0b001 => st.pins_diagonal |= pair.in_between_bitboard | attacker_square_bitboard,
                    else => unreachable,
                }
            }
        }
        st.pins = st.pins_diagonal | st.pins_orthogonal;
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

    pub fn get_unsafe_squares_for_king(self: *const Position, comptime us: Color) u64
    {
        var att: u64 = 0;
        const occ: u64 = self.all() ^ self.kings(us);
        const them = comptime us.opp();

        // Pawns.
        {
            const their_pawns = self.pawns(them);
            att |= (pawns_shift(their_pawns, them, .northeast) | pawns_shift(their_pawns, them, .northwest));
        }

        // Knights.
        {
            var their_knights = self.knights(them);
            while (their_knights != 0)
            {
                const from: Square = pop_square(&their_knights);
                att |= data.get_knight_attacks(from);
            }
        }

        // Diagonal sliders.
        {
            var their_diag_sliders = self.queens_bishops(them);
            while (their_diag_sliders != 0)
            {
                const from: Square = pop_square(&their_diag_sliders);
                att |= data.get_bishop_attacks(from, occ);
            }
        }

        // Orhtogonal sliders.
        {
            var their_orth_sliders = self.queens_rooks(them);
            while (their_orth_sliders != 0)
            {
                const from: Square = pop_square(&their_orth_sliders);
                att |= data.get_rook_attacks(from, occ);
            }
        }

        // King.
        {
            att |= data.get_king_attacks(self.king_square(them));
        }

        return att;
    }

    /// Passing in the state is faster.
    fn is_castling_allowed(self: *const Position, comptime castletype: CastleType, comptime us: Color) bool
    {
        const flag = comptime castle_flag_of(castletype, us);
        return self.state.castling_rights & flag != 0;
        //const bit: u2 = comptime index_of(castletype, us);
        //return funcs.test_bit_u8(st.castling_rights, bit);
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

    fn rook_castle_startsquare_dyn(self: *const Position, castletype: CastleType, us: Color) Square
    {
        const bit: u2 = index_of(castletype, us);
        return self.rook_start_squares[bit];
    }

    fn index_of(castletype: CastleType, us: Color) u2
    {
        return switch (castletype)
        {
            .short => if (us.e == .white) ci_white_short else ci_black_short,
            .long => if (us.e == .white) ci_white_long else ci_black_long,
        };
    }

    fn castle_flag_of(castletype: CastleType, us: Color) u4
    {
        return switch (castletype)
        {
            .short => if (us.e == .white) cf_white_short else cf_black_short,
            .long => if (us.e == .white) cf_white_long else cf_black_long,
        };
    }

    pub fn lazy_generate_moves(self: *const Position, noalias storage: anytype) void
    {
        switch (self.to_move.e)
        {
            .white => self.generate_moves(Color.WHITE, storage),
            .black => self.generate_moves(Color.BLACK, storage),
        }
    }

    pub fn lazy_generate_captures(self: *const Position, noalias storage: anytype) void
    {
        switch (self.to_move.e)
        {
            .white => self.generate_captures(Color.WHITE, storage),
            .black => self.generate_captures(Color.BLACK, storage),
        }
    }

    pub  fn generate_moves(self: *const Position, comptime us: Color, noalias storage: anytype) void
    {
        // NOTE: getting state flags with an inline else and create Params from that is easier but much slower than the switch.

        if (comptime lib.is_paranoid) assert(self.to_move.e == us.e);

        storage.reset();

        const check: bool = self.state.checkers != 0;
        const pins: bool = self.state.pins != 0;

        switch (check)
        {
            false =>
            {
                switch (pins)
                {
                    false => self.gen(Params.create(us, false, false), storage),
                    true  => self.gen(Params.create(us, false, true), storage),
                }
            },
            true => switch (pins)
            {
                false => self.gen(Params.create(us, true, false), storage),
                true  => self.gen(Params.create(us, true, true), storage),
            },
        }
    }

    /// See `MoveStorage` for the interface of storage.
    fn gen(self: *const Position, comptime ctp: Params, noalias storage: anytype) void
    {
        // Comptimes.
        const us = comptime ctp.us;
        const them = comptime us.opp();
        const st: StateInfo = self.state.*; // Copying the struct seems somewhat faster than using the pointer.

        const doublecheck: bool = ctp.check and @popCount(st.checkers) > 1;
        const bb_all: u64 = self.all();
        const bb_us: u64 = self.by_side(us);
        const bb_not_us: u64 = ~bb_us;
        const king_sq: Square = self.king_square(us);

        // In case of a doublecheck we can only move the king.
        if (!doublecheck)
        {
            const our_pawns = self.pawns(us);
            const our_knights = self.knights(us);
            const our_queens_bishops = self.queens_bishops(us);
            const our_queens_rooks = self.queens_rooks(us);

            const target = if (ctp.check) st.interpolationmask else bb_not_us;

            // Pawns.
            if (our_pawns != 0)
            {
                const third_rank: u64 = comptime funcs.relative_rank_3_bitboard(us);
                const last_rank: u64 = comptime funcs.relative_rank_8_bitboard(us);
                const empty_squares: u64 = ~bb_all;
                const enemies: u64 = if (ctp.check) st.checkers else self.by_side(them);

                // Generate all 4 types of pawnmoves: push, push double, capture left, capture right.
                var bb_single = switch(ctp.pins)
                {
                    false => (pawns_shift(our_pawns, us, .up) & empty_squares),
                    true  => (pawns_shift(our_pawns & ~st.pins, us, .up) & empty_squares) |
                             (pawns_shift(our_pawns & st.pins_diagonal, us, .up) & empty_squares & st.pins_diagonal) |
                             (pawns_shift(our_pawns & st.pins_orthogonal, us, .up) & empty_squares & st.pins_orthogonal)
                };

                var bb_double = pawns_shift(bb_single & third_rank, us, .up) & empty_squares;

                // Pawn push check interpolation.
                if (ctp.check)
                {
                    bb_single &= target;
                    bb_double &= target;
                }

                const bb_northwest: u64 = switch (ctp.pins)
                {
                    false => (pawns_shift(our_pawns, us, .northwest) & enemies),
                    true  => (pawns_shift(our_pawns & ~st.pins, us, .northwest) & enemies) |
                             (pawns_shift(our_pawns & st.pins_diagonal, us, .northwest) & enemies & st.pins_diagonal),
                };

                const bb_northeast: u64 = switch (ctp.pins)
                {
                    false => (pawns_shift(our_pawns, us, .northeast) & enemies),
                    true  => (pawns_shift(our_pawns & ~st.pins, us, .northeast) & enemies) |
                             (pawns_shift(our_pawns & st.pins_diagonal, us, .northeast) & enemies & st.pins_diagonal),
                };

                // Single push normal
                var bb_single_push: u64 = bb_single & ~last_rank;
                while (bb_single_push != 0)
                {
                    const to: Square = pop_square(&bb_single_push);
                    const from: Square = pawn_from(to, us, .up);
                    store(from, to, storage);
                }

                // Double.push
                while (bb_double != 0)
                {
                    const to: Square = pop_square(&bb_double);
                    const from: Square = if (us.e == .white) to.sub(16) else to.add(16);
                    store(from, to, storage);
                }

                // left capture promotions
                var bb_northwest_promotions = bb_northwest & last_rank;
                while (bb_northwest_promotions != 0)
                {
                    const to: Square = pop_square(&bb_northwest_promotions);
                    const from: Square = pawn_from(to, us, .northwest);
                    store_promotions(from, to, storage);
                }

                // right capture promotions
                var bb_northeast_promotions = bb_northeast & last_rank;
                while (bb_northeast_promotions != 0)
                {
                    const to: Square = pop_square(&bb_northeast_promotions);
                    const from: Square = pawn_from(to, us, .northeast);
                    store_promotions(from, to, storage);
                }

                // push promotions
                var bb_push_promotions: u64 = bb_single & last_rank;
                while (bb_push_promotions != 0)
                {
                    const to: Square = pop_square(&bb_push_promotions);
                    const from: Square =  pawn_from(to, us, .up);
                    store_promotions(from, to, storage);
                }

                // left normal captures,
                var bb_northwest_normal =  bb_northwest & ~last_rank;
                while (bb_northwest_normal != 0)
                {
                    const to: Square = pop_square(&bb_northwest_normal);
                    const from: Square = pawn_from(to, us, .northwest);
                    store(from, to, storage);
                }

                // right normal captures,
                var bb_northeast_normal =  bb_northeast & ~last_rank;
                while (bb_northeast_normal != 0)
                {
                    const to: Square = pop_square(&bb_northeast_normal);
                    const from: Square = pawn_from(to, us, .northeast);
                    store(from, to, storage);
                }

                // Enpassant.
                if (st.ep_square) |ep|
                {
                    var bb_enpassant: u64 = data.get_pawn_attacks(ep, them) & our_pawns; // inversion trick.
                    inline for (0..2) |_|
                    {
                        if (bb_enpassant == 0) break;
                        const from: Square = pop_square(&bb_enpassant);
                        if (self.is_legal_enpassant(us, king_sq, from, ep)) store_enpassant(from, ep, storage);
                    }
                }

            } // (pawns)

            // Knights.
            {
                // A knight can never escape a pin.
                var bb_from: u64 = if (!ctp.pins) our_knights else our_knights & ~st.pins;
                while (bb_from != 0)
                {
                    const from: Square = pop_square(&bb_from);
                    var bb_to: u64 = data.get_knight_attacks(from) & target;
                    inline for (0..8) |_|
                    {
                        if (bb_to == 0) break;
                        store(from, pop_square(&bb_to), storage);
                    }
                }
            }

            // Diagonal sliders.
            {
                if (!ctp.pins)
                {
                    var our_sliders: u64 = our_queens_bishops;
                    while (our_sliders != 0)
                    {
                        const from: Square = pop_square(&our_sliders);
                        var bb_to: u64 = data.get_bishop_attacks(from, bb_all) & target;
                        while (bb_to != 0)
                        {
                            store(from, pop_square(&bb_to), storage);
                        }
                    }
                }
                else
                {
                    var non_pinned_sliders: u64 = our_queens_bishops & ~st.pins;
                    while (non_pinned_sliders != 0)
                    {
                        const from: Square = pop_square(&non_pinned_sliders);
                        var bb_to: u64 = data.get_bishop_attacks(from, bb_all) & target;
                        while (bb_to != 0)
                        {
                            store(from, pop_square(&bb_to), storage);
                        }
                    }
                    var pinned_sliders: u64 =  our_queens_bishops & st.pins_diagonal;
                    while (pinned_sliders != 0)
                    {
                        const from: Square = pop_square(&pinned_sliders);
                        var bb_to: u64 = data.get_bishop_attacks(from, bb_all) & target & st.pins_diagonal;
                        while (bb_to != 0)
                        {
                            store(from, pop_square(&bb_to), storage);
                        }
                    }
                }
            }

            // Orthogonal sliders.
            {
                if (!ctp.pins)
                {
                    var our_sliders: u64 = our_queens_rooks;
                    while (our_sliders != 0)
                    {
                        const from: Square = pop_square(&our_sliders);
                        var bb_to: u64 = data.get_rook_attacks(from, bb_all) & target;
                        while (bb_to != 0)
                        {
                            store(from, pop_square(&bb_to), storage);
                        }
                    }
                }
                else
                {
                    var non_pinned_sliders: u64 = our_queens_rooks &  ~st.pins;
                    while (non_pinned_sliders != 0)
                    {
                        const from: Square = pop_square(&non_pinned_sliders);
                        var bb_to: u64 = data.get_rook_attacks(from, bb_all) & target;
                        while (bb_to != 0)
                        {
                            store(from, pop_square(&bb_to), storage);
                        }
                    }

                    var pinned_sliders: u64 =  our_queens_rooks & st.pins_orthogonal;
                    while (pinned_sliders != 0)
                    {
                        const from: Square = pop_square(&pinned_sliders);
                        var bb_to: u64 = data.get_rook_attacks(from, bb_all) & target & st.pins_orthogonal;
                        while (bb_to != 0)
                        {
                            store(from, pop_square(&bb_to), storage);
                        }
                    }
                }
            }

        } // (not doublecheck)

        // King. TODO: The king is a troublemaker. For now this 'heuristic' gives the best avg speed, using 2 different approaches to check legality.
        {
            var bb_to = data.get_king_attacks(king_sq) & bb_not_us;
            if (@popCount(bb_to) > 2)
            {
                const bb_unsafe: u64 = self.get_unsafe_squares_for_king(us);
                bb_to &= ~bb_unsafe;
                while (bb_to != 0)
                {
                    store(king_sq, pop_square(&bb_to), storage);
                }

                if (!ctp.check and st.castling_rights != 0)
                {
                    inline for (CastleType.all) |ct|
                    {
                        if (self.is_castling_allowed(ct, us) and self.is_castlingpath_empty(ct, us) and self.is_legal_castle(ct, us, bb_unsafe))
                        {
                            const to: Square = self.rook_castle_startsquare(ct, us);  // Castling is encoded as "king takes rook".
                            store_castle(king_sq, to, storage);
                        }
                    }
                }
            }
            else
            {
                const bb_without_king: u64 = bb_all ^ self.kings(us);
                while (bb_to != 0)
                {
                    const to: Square = pop_square(&bb_to);
                    if (self.is_legal_kingmove(us, bb_without_king, to))
                    {
                        store(king_sq, to, storage);
                    }
                }

                if (!ctp.check and st.castling_rights != 0)
                {
                    inline for (CastleType.all) |ct|
                    {
                        if (self.is_castling_allowed(ct, us) and self.is_castlingpath_empty(ct, us) and self.is_legal_castle_check_attacks(ct, us, king_sq))
                        {
                            const to: Square = self.rook_castle_startsquare(ct, us); // Castling is encoded as "king takes rook".
                            store_castle(king_sq, to, storage);
                        }
                    }
                }
            }
        }
    }

    fn store(from: Square, to: Square, noalias storage: anytype) void
    {
        storage.store(Move{ .from = from, .to = to, .prom = .no_prom, .movetype = .normal });
    }

    fn store_enpassant(from: Square, to: Square, noalias storage: anytype) void
    {
        storage.store(Move{ .from = from, .to = to, .prom = .no_prom, .movetype = .enpassant });
    }

    fn store_promotions(from: Square, to: Square, noalias storage: anytype) void
    {
        storage.store(Move{ .from = from, .to = to, .prom = .queen, .movetype = .promotion });
        storage.store(Move{ .from = from, .to = to, .prom = .rook, .movetype = .promotion });
        storage.store(Move{ .from = from, .to = to, .prom = .bishop, .movetype = .promotion });
        storage.store(Move{ .from = from, .to = to, .prom = .knight, .movetype = .promotion });
    }

    fn store_castle(from: Square, to: Square, noalias storage: anytype) void
    {
        storage.store(Move{ .from = from, .to = to, .prom = .no_prom, .movetype = .castle });
    }

    /// obsolete
    /// Checks if the piece is pinned and if we can move it.
    /// * If we move in the same direction as the pin-direction it is legal.
    /// * Orientation known: sometimes (like with pawn moves) we know up-front (hard-coded) in which direction we are moving and use this info.
    /// * Orientation null: it needs to be retrieved. in this case we definitely should have a slider pin or otherwise a bug.
    fn is_legal_check_pin(comptime orientation: ?Orientation, pinned: u64, king_sq: Square, from: Square, to: Square) bool
    {
        if (orientation) |ori|
        {
            return !funcs.contains_square(pinned, from) or ori == squarepairs.get(king_sq, to).orientation;
        }
        else
        {
            return !funcs.contains_square(pinned, from) or squarepairs.get(king_sq, from).orientation == squarepairs.get(king_sq, to).orientation;
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

    fn is_legal_kingmove(self: *const Position, comptime us: Color, bb_without_king: u64, to: Square) bool
    {
        const them = comptime us.opp();
        return !self.is_square_attacked_by_for_occupation(bb_without_king, to, them);
    }

    /// Compares the kings path with unsafe squares.
    fn is_legal_castle(self: *const Position, comptime castletype: CastleType, comptime us: Color, bb_unsafe: u64) bool
    {
        const idx = comptime index_of(castletype, us);
        const path: u64 = self.castling_king_paths[idx];
        return path & bb_unsafe == 0;
    }

    /// Checks for each square on the kings path if it is attacked.
    fn is_legal_castle_check_attacks(self: *const Position, comptime castletype: CastleType, comptime us: Color, king_sq: Square) bool
    {
        const them: Color = comptime us.opp();
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
            },
        }
        return true;
    }

    pub fn parse_move(self: *const Position, str: []const u8) !Move
    {
        assert(str.len >= 4);
        const us = self.to_move;
        var m: Move = .empty;
        const from: Square = Square.from_string(str[0..2]);
        var to: Square = Square.from_string(str[2..4]);
        // Castling encoding
        if (self.get(from).e == Piece.make(PieceType.KING, us).e and to.e == Square.G1.e or to.e == Square.G8.e or to.e == Square.C1.e or to.e == Square.C8.e)
        {
            const castletype: CastleType = if (to.u > from.u) .short else .long;
            to = self.rook_castle_startsquare_dyn(castletype, us);
        }
        m.from = from;
        m.to = to;
        var finder: MoveFinder = .init(from, to);
        self.lazy_generate_moves(&finder);
        if (finder.found()) return finder.found_move;
        return types.ParseError.IllegalMove;
    }

    /// Debug only.
    pub fn pos_ok(self: *const Position) bool
    {
        lib.not_in_release();

        assert(self.state == &self.history.items[self.ply]);

        // An extra test on the kings bitboard.
        if (@popCount(self.kings(Color.WHITE)) != 1)
        {
            std.debug.print("MISSING WHITE KING", .{});
            return false;
        }

        if (@popCount(self.kings(Color.BLACK)) != 1)
        {
            std.debug.print("MISSING BLACK KING", .{});
            return false;
        }

        if (@popCount(self.all()) > 32)
        {
            std.debug.print("TOO MANY PIECES", .{});
            return false;
        }

        const a = self.compute_hashkey();
        if (a != self.state.key)
        {
            std.debug.print("KEY {} <> {} lastmove {s} {s}\n", .{ self.state.key, a, self.state.last_move.to_string().slice(), @tagName(self.state.last_move.movetype) });
            return false;
        }

        // if (b != self.current_state.pawnkey)
        // {
        //     std.debug.print("PAWNKEY {} <> {} lastmove {s} {s}\n", .{ self.current_state.pawnkey, b, self.current_state.last_move.to_string().slice(), @tagName(self.current_state.last_move.movetype) });
        //     return false;
        // }

        // in check and not to move
        const king_sq_white: Square = self.king_square(Color.WHITE);
        if (self.is_square_attacked_by(king_sq_white, Color.BLACK) and self.to_move.e != .white)
        {
            std.debug.print("CHECK\n", .{});
            print_pos(self) catch wtf();
            return false;
        }

        const king_sq_black = self.king_square(Color.BLACK);
        if (self.is_square_attacked_by(king_sq_black, Color.WHITE) and self.to_move.e != .black)
        {
            std.debug.print("CHECK\n", .{});
            print_pos(self) catch wtf();

            return false;
        }
        return true;
    }

    pub fn to_fen(self: *const Position) std.ArrayListUnmanaged(u8)
    {
        const st = self.get_state();

        var s: std.ArrayListUnmanaged(u8) = std.ArrayListUnmanaged(u8).initCapacity(ctx.galloc, 80) catch wtf();

        // Pieces.
        var rank: u3 = 7;
        while (true)
        {
            var empty: u4 = 0;
            var file: u3 = 0;
            while (true)
            {
                const sq: Square = .from_rank_file(rank, file);
                const pc: Piece = self.board[sq.u];
                if (pc.is_empty())
                {
                    empty += 1;
                }
                else
                {
                    if (empty > 0)
                    {
                        s.appendAssumeCapacity(@as(u8, '0') + empty);
                        empty = 0;
                    }
                    s.appendAssumeCapacity(pc.to_fen_char());
                }
                if (file == 7)
                {
                    if (empty > 0)
                    {
                        s.appendAssumeCapacity(@as(u8, '0') + empty);
                    }
                    if (rank > 0) s.appendAssumeCapacity('/');
                    break;
                }
                file += 1;
            }
            if (rank == 0) break;
            rank -= 1;
        }

        // Color to move.
        if (self.to_move.e == .white)
            s.appendSliceAssumeCapacity(" w")
        else
            s.appendSliceAssumeCapacity(" b");

        // Castling rights.
        s.appendAssumeCapacity(' ');
        if (st.castling_rights == 0)
        {
            s.appendAssumeCapacity('-');
        }
        else
        {
            if (st.castling_rights & cf_white_short != 0) s.appendAssumeCapacity('K');
            if (st.castling_rights & cf_white_long != 0)  s.appendAssumeCapacity('Q');
            if (st.castling_rights & cf_black_short != 0) s.appendAssumeCapacity('k');
            if (st.castling_rights & cf_black_long != 0)  s.appendAssumeCapacity('q');
        }

        // Enpassant.
        s.appendAssumeCapacity(' ');
        if (st.ep_square) |ep|
        {
            s.appendSliceAssumeCapacity(ep.to_string());
        } else
        {
            s.appendAssumeCapacity('-');
        }

        // Draw counter.
        s.fixedWriter().print(" {}", .{st.rule50}) catch wtf();

        // Move number.
        const movenr: u16 = funcs.ply_to_movenumber(self.game_ply, self.to_move);
        s.fixedWriter().print(" {}", .{movenr}) catch wtf();

        return s;
    }
};

/// 4 bits comptime struct for generating moves.
pub const Params = packed struct
{
    /// The color for which we are generating.
    us: Color = Color.WHITE,
    /// We are in check?
    check: bool = false,
    // There are pins. If not we can comptime skip all pin checks.
    pins: bool = false,

    fn create(comptime us: Color, comptime check: bool, comptime pins: bool) Params
    {
        return Params{ .us = us, .check = check, .pins = pins };
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

pub const MoveFinder = struct
{
    from: Square,
    to: Square,
    found_move: Move,

    pub fn init(from: Square, to: Square) MoveFinder
    {
        return .{ .from = from, .to = to, .found_move = .empty };
    }

    /// Required funnction.
    pub fn reset(self: *MoveFinder) void
    {
        self.found_move = .empty;
    }

    /// Required funnction.
    pub fn store(self: *MoveFinder, move: Move) void
    {
        if (move.from.u == self.from.u and move.to.u == self.to.u)
        {
            self.found_move = move;
        }
    }

    pub fn found(self: *const MoveFinder) bool
    {
        return !self.found_move.is_empty();
    }
};


/// Prints the position to the `lib.out`.
pub fn print_pos(pos: *const Position) !void
{
    const format = comptime std.fmt.format;

    var s = std.ArrayListUnmanaged(u8).initCapacity(ctx.galloc, 320) catch wtf();
    defer s.deinit(ctx.galloc);

    var fen_str = pos.to_fen();
    defer fen_str.deinit(ctx.galloc);

    const writer = s.writer(ctx.galloc);

    // Pieces.
    try format(writer, "\n", .{});
    for (Square.all_for_printing) |square|
    {
        const pc: Piece = pos.get(square);
        const ch: u8 = if (pc.is_empty()) '.' else pc.to_print_char();
        try format(writer, "{u} ", .{ch});
        if (square.u % 8 == 7) try format(writer, "\n", .{});
    }
    try format(writer, "\n", .{});

    // Info.
    const move_str: []const u8 = if (pos.state.last_move.is_empty()) "" else pos.state.last_move.to_string().slice();
    try format(writer, "Fen: {s}\n", .{fen_str.items});
    try format(writer, "Key: {x:0>16}\n", .{pos.state.key});
    try format(writer, "Last move: {s}\n", .{move_str});
    try format(writer, "Checkers: ", .{});
    if (pos.state.checkers != 0)
    {
        var bb: u64 = pos.state.checkers;
        while (bb != 0)
        {
            const sq: Square = pop_square(&bb);
            try format(writer, "{s} ", .{sq.to_string()});
        }
    }
    try format(writer, "\n\n", .{});
    try lib.out.print("{s}", .{s.items});
}
