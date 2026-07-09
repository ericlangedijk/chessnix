// zig fmt: off

//! Conversions from and to the viri game format.
//! The game format is quite straightforward. [position + moves + terminator], [position + moves + terminator]
//! When using a viri-file seriously we assume correctness. So the file has to be 'verified' before use.

const std = @import("std");
const lib = @import("lib.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const bitboards = @import("bitboards.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");
const hceterms = @import("hceterms.zig");

const ctx = lib.ctx;

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const WDL = types.WDL;
const Castling = position.Castling;
const LayoutKey = position.LayoutKey;
const Position = position.Position;

const popcnt = bitboards.popcnt;

comptime {
    std.debug.assert(@sizeOf(ViriPosition) == 32);
    std.debug.assert(@bitSizeOf(ViriPosition) == 32 * 8);
    std.debug.assert(@sizeOf(ViriMove) == 2);
    std.debug.assert(@bitSizeOf(ViriMove) == 16);
    std.debug.assert(@sizeOf(ViriMoveExt) == 4);
    std.debug.assert(@bitSizeOf(ViriMoveExt) == 32);
    const u: u16 = 0b11_10_111110_110101;
    const e: ViriMove = .init_promo(.f7, .g8, .rook);
    std.debug.assert(u == @as(u16, @bitCast(e)));
}

pub const ViriPosition = extern struct {
    occupancy: u64,
    pieces: [16]u8,
    stm_ep_square: u8,
    halfmove_clock: u8,
    fullmove_number: u16,
    eval: i16, // Always from white perspective.
    wdl: u8,
    extra: u8,

    const unmoved_rook: u8 = 6;
    pub const empty: ViriPosition = std.mem.zeroes(ViriPosition);

    /// Position -> ViriPosition.
    pub fn from_position(pos: *const Position, eval: i16, wdl: u8) ViriPosition {
        const occ: u64 = pos.all();
        var pieces: [16]u8 = @splat(0);

        // Pieceloop.
        var i: usize = 0;
        var iter = bitboards.iterator(occ);
        while (iter.next()) |sq| : (i += 1) {
            const piece: Piece = pos.get(sq);
            const piecetype: PieceType = piece.piecetype();
            const color: Color = piece.color();
            const colorbit: u8 = if (color.e == .black) 0b1000 else 0;
            var piececode: u8 = piecetype.u;  // Value matches chessnix.
            // Encode castling rook.
            if (piecetype.e == .rook and sq.coord.rank == types.relative_rank(color, types.rank_1)) {
                if (
                    (pos.has_castlingright(color, .short) and sq.u == pos.layout.rook_start(color, .short).u) or
                    (pos.has_castlingright(color, .long) and sq.u == pos.layout.rook_start(color, .long).u)
                ) {
                    piececode = unmoved_rook;
                }
            }
            const val: u8 = piececode | colorbit;
            const shift: u3 = if (i % 2 == 0) 0 else 4;
            pieces[i / 2] |= (val << shift);
        }

        return .{
            .occupancy = pos.all(),
            .pieces = pieces,
            .stm_ep_square = @as(u8, if (pos.stm.e == .black) 0b10000000 else 0) | @as(u8, if (pos.ep_square.e != .a1) pos.ep_square.u else 64),
            .halfmove_clock = @truncate(pos.rule50),
            .fullmove_number = funcs.ply_to_movenumber(pos.game_ply, pos.stm),
            .eval = eval,
            .wdl = wdl,
            .extra = 55, // The value with the least amount of occurences i found :-)
        };
    }

    /// ViriPosition -> Position.
    pub fn to_position(self: *const ViriPosition) !Position {
        var pos: Position = .empty;
        const occ: u64 = self.occupancy;
        if (popcnt(occ) > 32) {
            return Error.too_many_pieces;
        }
        var layoutkey: LayoutKey = .empty;
        var castling_rooks: [Color.count]u64 = @splat(0);

        // Piece loop.
        var i: usize = 0;
        var iter = bitboards.iterator(occ);
        while (iter.next()) |sq| : (i += 1) {
            const shift: u3 = if (i % 2 == 0) 0 else 4;
            const code: u8 = (self.pieces[i / 2] >> shift) & 0b1111;
            const color: Color = if (code & 0b1000 == 0) .white else .black;
            const piececode: u3 = @intCast(code & 0b111);
            const piece: Piece = switch (piececode) {
                0...5  => |u| Piece.init(PieceType.from_int(u), color),
                unmoved_rook => blk: {
                    if (sq.coord.rank != types.relative_rank(color, types.rank_1)) {
                        return Error.invalid_castling_rook_square;
                    }
                    castling_rooks[color.u] |= sq.to_bitboard();
                    break :blk Piece.init(.rook, color);
                },
                7 => {
                    return Error.invalid_piece_code;
                },
            };
            pos.lazy_add_piece(piece, sq);
        }

        if (popcnt(pos.kings(.white)) != 1 or popcnt(pos.kings(.black)) != 1) {
            return Error.kings;
        }
        if (pos.all_pawns() & (bitboards.bb_rank_1 | bitboards.bb_rank_8) != 0) {
            return Error.pawns_on_first_or_last_rank;
        }

        // Castling.
        layoutkey.init_kings(pos.king_square(.white), pos.king_square(.black));
        iter = .init(castling_rooks[Color.white.u]);
        while (iter.next()) |sq| {
            try pos.set_castling_right(&layoutkey, .white, sq.coord.file, false);
        }
        iter = .init(castling_rooks[Color.black.u]);
        while (iter.next()) |sq| {
            try pos.set_castling_right(&layoutkey, .black, sq.coord.file, false);
        }

        // Ep.
        const ep_target = self.stm_ep_square & 0b01111111;
        pos.ep_square = if (ep_target < 64) Square.from_int(ep_target) else .a1;
        pos.stm = if (self.stm_ep_square & 0b10000000 == 0) .white else .black;
        if (pos.ep_square.e != .a1) {
            const proper_rank: u3 = if (pos.stm.e == .white) types.rank_6 else types.rank_3;
            if (pos.ep_square.coord.rank != proper_rank) {
                return Error.invalid_ep_square;
            }
        }

        pos.rule50 = self.halfmove_clock;
        pos.game_ply = funcs.movenumber_to_ply(self.fullmove_number, pos.stm);
        pos.layout = position.select_layout(layoutkey);
        pos.update_hash();
        pos.lazy_update_state();
        return pos;
    }
};

/// 16 bits.
pub const ViriMove = packed struct {
    from: Square,
    to: Square,
    prom: u2,
    flag: u2,

    const ep_flag: u2 = 1;
    const castle_flag: u2 = 2;
    const promotion_flag: u2 = 3;

    pub fn init(from: Square, to: Square) ViriMove {
        return .{ .from = from, .to = to, .flag = 0, .prom = 0 };
    }

    pub fn init_ep(from: Square, to: Square) ViriMove {
        return .{ .from = from, .to = to, .prom = 0, .flag = ep_flag };
    }

    pub fn init_castle(from: Square, to: Square) ViriMove {
        return .{ .from = from, .to = to, .prom = 0, .flag = castle_flag };
    }

    pub fn init_promo(from: Square, to: Square, prom: PieceType) ViriMove {
        return .{ .from = from, .to = to, .prom = @intCast(prom.u - 1), .flag = promotion_flag };
    }

    pub fn is_ep(self: ViriMove) bool {
        return self.flag == ep_flag;
    }

    pub fn is_castle(self: ViriMove) bool {
        return self.flag == castle_flag;
    }

    pub fn is_promotion(self: ViriMove) bool {
        return self.flag == promotion_flag;
    }

    /// Convert from a chessnix move.
    pub fn from_move(move: Move) ViriMove {
        return switch (move.simple_kind()) {
            .default => .init(move.from, move.to),
            .ep => .init_ep(move.from, move.to),
            .castle =>.init_castle(move.from, move.to),
            .promotion => .init_promo(move.from, move.to, move.prom()),
        };
    }

    pub fn to_move(self: ViriMove, pos: *const Position) Move {
        if (self.is_ep()) {
            return .init(self.from, self.to, Move.ep);
        }

        if (self.is_castle()) {
            if (self.to.u > self.from.u) {
                return .init(self.from, self.to, Move.castle_short);
            }
            else {
                return .init(self.from, self.to, Move.castle_long);
            }
        }

        const is_capture: bool = !pos.get(self.to).is_empty();
        const capt_flag: u4 = if (is_capture) Move.capture_mask else Move.silent;

        if (self.is_promotion()) {
            const p: u4 = self.prom;
            const promo_type: PieceType = .from_int(p + 1);
            const prom_flag: u4 = promo_type.to_promotion_move_flag();
            return .init(self.from, self.to, prom_flag | capt_flag);
        }

        if (is_capture) {
            return .init(self.from, self.to, Move.capture);
        }
        else {
            const piece: Piece = pos.get(self.from);
            if (piece.is_pawn() and self.from.u ^ self.to.u == 16) {
                return .init(self.from, self.to, Move.double_push);
            }
            else {
                return .init(self.from, self.to, Move.silent);
            }
        }
    }

    pub fn to_ext_move(self: ViriMove, pos: *const Position) ExtMove {
        const us: Color = pos.stm;
        const them: Color = us.opp();
        const move: Move = self.to_move(pos);
        const piece: Piece = pos.get(move.from);
        const captured: Piece = switch (move.simple_kind()) {
            .default, .promotion => pos.get(move.to),
            .ep => .init(.pawn, them),
            .castle => .no_piece,
        };
        return .from_move(move, piece, captured);
    }
};

/// 32 bits.
pub const ViriMoveExt = packed struct {
    move: ViriMove,
    eval: i16, // Always from white perspective.

    pub fn init(move: ViriMove, eval: i16) ViriMoveExt {
        return .{ .move = move, .eval = eval };
    }

    pub fn to_int(self: ViriMoveExt) u32 {
        return @bitCast(self);
    }

    pub fn is_empty(self: ViriMoveExt) bool {
        const v: u32 = @bitCast(self);
        return v == 0;
    }
};

pub const ViriGame = struct {
    startpos: ViriPosition,
    moves: std.ArrayList(ViriMoveExt),

    pub fn init() ViriGame {
        return .{
            .startpos = .empty,
            .moves = .empty,
        };
    }

    pub fn reset(self: *ViriGame) void {
        self.moves.clearAndFree(ctx.gpa);
        self.startpos = .empty;
    }

    pub fn deinit(self: *ViriGame) void {
        self.moves.deinit(ctx.gpa);
    }

    pub fn reset_with_viri_position(self: *ViriGame, pos: *const ViriPosition) void {
        self.moves.items.len = 0;
        self.startpos = pos.*;
    }

    /// Sets position, clear moves.
    pub fn reset_with_position(self: *ViriGame, pos: *const Position, eval: i16, wdl: u8) !void {
        self.moves.items.len = 0;
        self.startpos = .from_position(pos, eval, wdl);
    }

    pub fn append_viri_move(self: *ViriGame, move: ViriMoveExt) !void {
        try self.moves.append(ctx.gpa, move);
    }

    pub fn append_move(self: *ViriGame, move: Move, eval: i16) !void {
        const vm: ViriMove = .from_move(move);
        try self.moves.append(ctx.gpa, ViriMoveExt.init(vm, eval));
    }

    pub fn to_game(self: *const ViriGame) !position.Game {
        var game: position.Game = .init();
        const start: Position = try self.startpos.to_position();
        game.reset_with_position(&start);
        var pos: Position = game.startpos;
        for (self.moves.items) |vm| {
            const ex: types.ExtMove = vm.move.to_ext_move(&pos);
            try game.append_move(ex);
            pos.lazy_do_move(ex);
        }
        return game;
    }
};

pub const ViriFileReader = struct {
    filereader: utils.FileReader,

    pub fn init(filename: []const u8, buf_size: usize) !ViriFileReader {
        return .{
            .filereader = try .init(filename, buf_size)
        };
    }

    pub fn deinit(self: *ViriFileReader) void {
        self.filereader.deinit();
    }

    /// Read next game from the file.
    pub fn next(self: *ViriFileReader, game: *ViriGame) !bool {
        // A little check.
        const p: u64 = self.filereader.rd.logicalPos();
        if (p % 4 != 0) {
            return Error.non_aligned_file_pos;
        }

        // Read position.
        const g: ViriPosition = self.filereader.rd.interface.takeStruct(ViriPosition, .little) catch |err|
            switch (err) {
                std.io.Reader.Error.EndOfStream => return false,
                std.io.Reader.Error.ReadFailed => return err,
            };

        game.reset_with_viri_position(&g);

        // Read moves.
        while(true) {
            const m: ViriMoveExt = self.filereader.rd.interface.takeStruct(ViriMoveExt, .little) catch |err|
                switch (err) {
                    std.io.Reader.Error.EndOfStream => return false,
                    std.io.Reader.Error.ReadFailed => return err,
                };
            if (m.to_int() == 0) {
                break;
            }
            try game.append_viri_move(m);
        }
        return true;
    }
};

pub const ViriFileWriter = struct {
    filewriter: utils.FileWriter,

    pub fn init(filename: []const u8, buf_size: usize) !ViriFileWriter {
        return .{
            .filewriter = try .init(filename, buf_size),
        };
    }

    pub fn deinit(self: *ViriFileWriter) void {
        self.filewriter.deinit();
    }

    pub fn write_game(self: *ViriFileWriter, game: *const ViriGame) !void {
        try self.write_pos(&game.startpos);
        try self.write_moves(game.moves.items, true);
    }

    pub fn write_pos(self: *ViriFileWriter, pos: *const ViriPosition) !void {
        try self.filewriter.wr.interface.writeStruct(pos.*, .little);
    }

    pub fn write_moves(self: *ViriFileWriter, moves: []const ViriMoveExt, with_moves_terminator: bool) !void {
        if (moves.len == 0) {
            return;
        }
        try self.filewriter.wr.interface.writeSliceEndian(ViriMoveExt, moves, .little);
        if (with_moves_terminator) {
            try self.write_moves_terminator();
        }
    }

    pub fn write_move(self: *ViriFileWriter, move: ViriMoveExt, with_moves_terminator: bool) !void {
        try self.filewriter.wr.interface.writeStruct(move, .little);
        if (with_moves_terminator) {
            try self.write_moves_terminator();
        }
    }

    pub fn write_moves_terminator(self: *ViriFileWriter) !void {
        try self.filewriter.wr.interface.writeInt(u32, 0, .little);
    }

    pub fn flush(self: *ViriFileWriter) !void {
        try self.filewriter.wr.interface.flush();
    }
};

pub const Error = error {
    too_many_pieces,
    invalid_castling_rook_square,
    invalid_piece_code,
    kings,
    pawns_on_first_or_last_rank,
    invalid_ep_square,
    non_aligned_file_pos,
};