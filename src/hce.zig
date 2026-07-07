// zig fmt: off

//! Hand Crafted Evaluation. Evaluation of a position.
//! When we are tuning keep track of each used term.

const std = @import("std");
const assert = std.debug.assert;

const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const bitboards = @import("bitboards.zig");
const squarepairs = @import("squarepairs.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const scoring = @import("scoring.zig");
const utils = @import("utils.zig");
const tt = @import("tt.zig");
const search = @import("search.zig");
const hceterms = @import("hceterms.zig");
const endgame = @import("endgame.zig");

const io = lib.io;
const popcnt = bitboards.popcnt;
const int = funcs.int;

const ScorePair = types.ScorePair;
const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Castle = types.Castle;
const Move = types.Move;
const Position = position.Position;

const terms = hceterms.terms;

/// Mutable pointer when tunning otherwise const.
const ScorePairPtr = if (lib.is_tuning) *ScorePair else *const ScorePair;
const hcetuner = if (lib.is_tuning) @import("hcetuner.zig") else void;

pub const king_areas_white: [Square.count]u64 = compute_king_areas_white();
pub const king_areas_black: [Square.count]u64 = compute_king_areas_black();
pub const king_pawnstorm_areas_white: [Square.count]u64 = compute_king_pawnstorm_areas_white();
pub const king_pawnstorm_areas_black: [Square.count]u64 = compute_king_pawnstorm_areas_black();

pub const Evaluator = struct {
    const Self = @This();
    /// Ref to position, only known at evaluation time.
    pos: *const Position,
    /// Cached pinned pieces.
    pinned: [Color.count]u64,
    /// The squares of the kings.
    king_squares: [Color.count]Square,
    /// The 3x4 squares area around and in front of our king.
    king_areas: [Color.count]u64,
    /// The 3x7 squares in front of our king.
    pawn_storm_areas: [Color.count]u64,
    /// Accessible squares which are not protected by enemy pawns.
    mobility_areas: [Color.count]u64,
    /// All pawn attacks.
    pawn_attacks: [Color.count]u64,
    /// All knight attacks (updated on the fly).
    knight_attacks: [Color.count]u64,
    /// All bishop attacks (updated on the fly).
    bishop_attacks: [Color.count]u64,
    /// All rook attacks (updated on the fly).
    rook_attacks: [Color.count]u64,
    /// All queen attacks (updated on the fly).
    queen_attacks: [Color.count]u64,
    /// All king attacks.
    king_attacks: [Color.count]u64,
    /// All attacks to the enemy king area (updated on the fly).
    attack_power: [Color.count]ScorePair,

    pub const Mode = enum(u1) {
        /// For engine we return a score (stm perspective) with endgame scaling applied.
        scaled = 0,
        /// For tuning we return an absolute score (white's perspective) without endgame scaling.
        unscaled = 1,
    };

    pub fn init() Self {
        return .{
            .pos = undefined,
            .pinned = @splat(0),
            .king_squares = .{ .a1, .a1 },
            .king_areas = .{ 0, 0 },
            .pawn_storm_areas = .{ 0, 0 },
            .mobility_areas = .{ 0, 0 },
            .pawn_attacks = .{ 0, 0 },
            .knight_attacks = .{ 0, 0 },
            .bishop_attacks = .{ 0, 0 },
            .rook_attacks = .{ 0, 0 },
            .queen_attacks = .{ 0, 0 },
            .king_attacks = .{ 0, 0 },
            .attack_power = .{ .empty, .empty },
        };
    }

    pub fn evaluate(self: *Self, pos: *const Position, comptime mode: Mode) i32 {
        // Init stuff.
        self.pos = pos;

        self.pinned = .{
            pos.pins(.white) & pos.by_color(.white),
            pos.pins(.black) & pos.by_color(.black),
        };

        const wk: Square = pos.king_square(.white);
        const bk: Square= pos.king_square(.black);

        self.king_squares = .{ wk, bk };

        self.king_areas[Color.white.u] = king_areas_white[wk.u];
        self.king_areas[Color.black.u] = king_areas_black[bk.u];

        self.pawn_storm_areas[Color.white.u] = king_pawnstorm_areas_white[wk.u];
        self.pawn_storm_areas[Color.black.u] = king_pawnstorm_areas_black[bk.u];

        self.pawn_attacks = .{
            funcs.pawns_attacks(pos.pawns(.white), .white),
            funcs.pawns_attacks(pos.pawns(.black), .black),
        };

        self.mobility_areas = .{
            ~(pos.by_color(.white) | self.pawn_attacks[Color.black.u]),
            ~(pos.by_color(.black) | self.pawn_attacks[Color.white.u]),
        };

        self.knight_attacks = @splat(0);
        self.bishop_attacks = @splat(0);
        self.rook_attacks = @splat(0);
        self.queen_attacks = @splat(0);
        self.king_attacks = .{
            attacks.get_king_attacks(wk),
            attacks.get_king_attacks(bk),
        };
        self.attack_power = @splat(ScorePair.empty);

        // And perform eval.
        var score: ScorePair = eval_material(pos);

        score.inc(self.eval_pawns(.white));
        score.dec(self.eval_pawns(.black));

        score.inc(self.eval_knights(.white));
        score.dec(self.eval_knights(.black));

        score.inc(self.eval_bishops(.white));
        score.dec(self.eval_bishops(.black));

        score.inc(self.eval_rooks(.white));
        score.dec(self.eval_rooks(.black));

        score.inc(self.eval_queens(.white));
        score.dec(self.eval_queens(.black));

        score.inc(self.eval_king(.white));
        score.dec(self.eval_king(.black));

        score.inc(self.eval_threats(.white));
        score.dec(self.eval_threats(.black));

        // First add in the tempo bonus. That is what the data was originally tuned for.
        switch (pos.stm.e) {
            .white => score.inc(terms.tempo_bonus),
            .black => score.dec(terms.tempo_bonus),
        }
        if (lib.is_tuning) register(&terms.tempo_bonus, 1, pos.stm, void);

        score = scoring.restrict_scorepair_before_scaling(score);

        // First interpolate the final score.
        var result: i32 = types.phased_score(pos.phase(), score);

        // Don't scale when tuning. Return absolute result.
        if (mode == .unscaled) {
            if (lib.is_tuning) {
                hcetuner.register_final_result(score, result);
            }
            return result;
        }

        // Scale the endresult (this plays stronger than scaling the eg component only).
        // Note that scaling is done on the absolute value (white's perspective).
        const scale: f32 = endgame.scale(pos, result);
        result = funcs.fmul(result, scale);
        result = scoring.restrict_static_eval(result);// std.math.clamp(result, -scoring.static_eval_threshold, scoring.static_eval_threshold);
        return if (pos.stm.e == .white) result else -result;
    }

    fn eval_material(pos: *const Position) ScorePair {
        var sp: ScorePair = .empty;
        // Skip king.
        inline for (0..5) |pt| {
            const vp: ScorePair = terms.material_table[pt];
            const white_count: u8 = pos.material.counts[0][pt];
            const black_count: u8 = pos.material.counts[1][pt];
            // skip evaluation if equal.
            if (white_count != black_count) {
                sp.mg += vp.mg * white_count;
                sp.eg += vp.eg * white_count;
                sp.mg -= vp.mg * black_count;
                sp.eg -= vp.eg * black_count;
                if (lib.is_tuning) {
                    register(&terms.material_table[pt], white_count, .white, .{ PieceType.from_int(pt) });
                    register(&terms.material_table[pt], black_count, .black, .{ PieceType.from_int(pt) });
                }
            }
        }
        return sp;
    }

    fn eval_pawns(self: *Self, comptime us: Color) ScorePair {
        const pos: *const Position = self.pos;

        if (pos.pawns(us) == 0) {
            return .empty;
        }

        const them: Color = comptime us.opp();
        const our_pawns: u64 = pos.pawns(us);
        const their_pawns: u64 = pos.pawns(them);

        var iter: bitboards.BitboardIterator = undefined;
        var score: ScorePair = .empty;
        var passed_pawns: u64 = 0;

        // Pawn phalanx (horizontally next to eachother).
        const phalanx_pawns: u64 = (bitboards.shift_bitboard(our_pawns, .east)) & our_pawns;
        iter = .init(phalanx_pawns);
        while (iter.next()) |sq| {
            const relative_rank: u3 = types.relative_rank(us, sq.coord.rank);
            score.inc(terms.pawn_phalanx_bonus[relative_rank]);
            if (lib.is_tuning) register(&terms.pawn_phalanx_bonus[relative_rank], 1, us, .{ PieceType.pawn.e, relative_rank });
        }

        // Loop through our pawns. Determine passed pawns.
        iter = .init(our_pawns);
        while (iter.next()) |sq| {
            const sq_bb: u64 = sq.to_bitboard();
            const relative_sq: Square = sq.relative(us);
            const relative_rank: u3 = relative_sq.coord.rank;
            const file: u3 = sq.coord.file;

            // Psqt.
            score.inc(terms.piece_square_table[PieceType.pawn.u][relative_sq.u]);
            if (lib.is_tuning) register(&terms.piece_square_table[PieceType.pawn.u][relative_sq.u], 1, us, .{ PieceType.pawn.e, sq.e });

            // Protected pawn.
            const is_protected: bool = self.pawn_attacks[us.u] & sq_bb != 0;
            if (is_protected) {
                score.inc(terms.protected_pawn_bonus[relative_rank]);
                if (lib.is_tuning) register(&terms.protected_pawn_bonus[relative_rank], 1, us, .{ PieceType.pawn.e, sq.e });
            }

            // Doubled pawn.
            const pawns_ahead_on_file = bitboards.forward_file(us, sq) & our_pawns;
            const is_doubled: bool = pawns_ahead_on_file != 0;
            if (is_doubled) {
                score.inc(terms.doubled_pawn_penalty[file]);
                if (lib.is_tuning) register(&terms.doubled_pawn_penalty[file], 1, us, .{ PieceType.pawn.e, sq.e });
            }

            // Passed pawn.
            const their_pawns_ahead: u64 = bitboards.get_passed_pawn_mask(us, sq) & their_pawns;
            const is_passed: bool = their_pawns_ahead == 0;
            if (is_passed) {
                passed_pawns |= sq_bb; // Update passed pawns.
                score.inc(terms.passed_pawn_bonus[relative_rank]);
                if (lib.is_tuning) register(&terms.passed_pawn_bonus[relative_rank], 1, us, .{ PieceType.pawn.e, sq.e });
            }

            // Isolated pawn.
            const pawns_on_adjacent_files = bitboards.adjacent_file_masks[sq.u] & our_pawns;
            const is_isolated: bool = pawns_on_adjacent_files == 0;
            if (is_isolated) {
                score.inc(terms.isolated_pawn_penalty[file]);
                if (lib.is_tuning) register(&terms.isolated_pawn_penalty[file], 1, us, .{ PieceType.pawn.e, sq.e });
            }
        }

        // Passed pawns with kings.
        const king_sq: Square = self.king_squares[us.u];
        const their_king_sq: Square = self.king_squares[them.u];
        iter = .init(passed_pawns);
        while (iter.next()) |sq| {
            const their_move: u1 = @intFromBool(pos.stm.e == them.e);
            const relative_rank: u3 = types.relative_rank(us, sq.coord.rank);
            const dist_to_king: u3 = funcs.square_distance(sq, king_sq);
            score.inc(terms.king_passed_pawn_distance_table[dist_to_king]);
            if (lib.is_tuning) register(&terms.king_passed_pawn_distance_table[dist_to_king], 1, us, .{ PieceType.pawn.e, sq.e });
            const dist_to_enemy_king: u3 = funcs.square_distance(sq, their_king_sq);
            score.inc(terms.enemy_king_passed_pawn_distance_table[dist_to_enemy_king]);
            if (lib.is_tuning) register(&terms.enemy_king_passed_pawn_distance_table[dist_to_enemy_king], 1, us, .{ PieceType.pawn.e, sq.e });
            // Square rule for pawn race.
            const enemy_non_pawn_king_pieces: u64 = pos.by_color(them) & ~pos.kings(them) & ~pos.pawns(them);
            const dist_to_promotion: u3 = (7 - relative_rank);
            if (enemy_non_pawn_king_pieces == 0 and dist_to_promotion < dist_to_enemy_king - their_move) {
                score.inc(terms.king_cannot_reach_passed_pawn_bonus);
                if (lib.is_tuning) register(&terms.king_cannot_reach_passed_pawn_bonus, 1, us, .{ PieceType.pawn.e, sq.e });
            }
        }
        return score;
    }

    fn eval_knights(self: *Self, comptime us: Color) ScorePair {
        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        var score: ScorePair = .empty;

        var iter = bitboards.iterator(pos.knights(us));
        while (iter.next()) |sq| {
            const relative_sq: Square = sq.relative(us);

            // Psqt.
            score.inc(terms.piece_square_table[PieceType.knight.u][relative_sq.u]);
            if (lib.is_tuning) register(&terms.piece_square_table[PieceType.knight.u][relative_sq.u], 1, us, .{ PieceType.knight.e, sq.e });

            // Mobility.
            const legal_moves: u64 = self.legalize_moves(.knight, us, sq, attacks.get_knight_attacks(sq));
            const mobility: u64 = legal_moves & self.mobility_areas[us.u];
            const mobility_count: u7 = popcnt(mobility);
            score.inc(terms.knight_mobility_table[mobility_count]);
            if (lib.is_tuning) register(&terms.knight_mobility_table[mobility_count], 1, us, .{PieceType.knight.e, sq.e, mobility_count});

            // Update attacks.
            self.knight_attacks[us.u] |= legal_moves;

            // King attack power.
            const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
            if (enemy_king_attacks != 0) {
                const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                self.attack_power[us.u].inc(terms.attack_power[PieceType.knight.u][king_attack_count]);
                if (lib.is_tuning) register(&terms.attack_power[PieceType.knight.u][king_attack_count], 1, us, .{ PieceType.knight.e, sq.e, king_attack_count });
            }

            if (self.is_outpost(sq, us)) {
                const sq_in_front: Square = if (us.e == .white) sq.add(8) else sq.sub(8);
                const is_blocking: u1 = @intFromBool(pos.board[sq_in_front.u].is_pawn_of_color(them));
                score.inc(terms.knight_outpost_table[is_blocking][relative_sq.u - 24]);
                if (lib.is_tuning) register(&terms.knight_outpost_table[is_blocking][relative_sq.u - 24], 1, us, .{ PieceType.knight.e, sq.e, is_blocking });
            }
        }
        return score;
    }

    fn eval_bishops(self: *Self, comptime us: Color) ScorePair {
        const them: Color = comptime us.opp();
        const pos: *const Position = self.pos;
        const mobility_occ: u64 = pos.all() ^ pos.queens(us) ^ pos.bishops(us);

        const our_bishops: u64 = pos.bishops(us);
        var score: ScorePair = .empty;

        // Bishop pair.
        if ((our_bishops & bitboards.bb_black_squares != 0) and (our_bishops & bitboards.bb_white_squares != 0)) {
            score.inc(terms.bishop_pair_bonus);
            if (lib.is_tuning) register(&terms.bishop_pair_bonus, 1, us, .{ PieceType.bishop.e });
        }

        var iter = bitboards.iterator(our_bishops);
        while (iter.next()) |sq| {
            const relative_sq: Square = sq.relative(us);

            // Psqt.
            score.inc(terms.piece_square_table[PieceType.bishop.u][relative_sq.u]);
            if (lib.is_tuning) register(&terms.piece_square_table[PieceType.bishop.u][relative_sq.u], 1, us, .{ PieceType.bishop.e, sq.e });

            // Mobility.
            const moves: u64 = self.legalize_moves(PieceType.bishop, us, sq, attacks.get_bishop_attacks(sq, mobility_occ));
            const mobility: u64 = moves & self.mobility_areas[us.u];
            const mobility_count: u7 = popcnt(mobility);
            score.inc(terms.bishop_mobility_table[mobility_count]);
            if (lib.is_tuning) register(&terms.bishop_mobility_table[mobility_count], 1, us, .{ PieceType.bishop.e, sq.e, mobility_count });

            // Bishop on long diagonal.
            if (popcnt(moves & bitboards.bb_center_4) > 1) {
                score.inc(terms.bishop_on_long_diagonal);
                if (lib.is_tuning) register(&terms.bishop_on_long_diagonal, 1, us, .{ PieceType.bishop.e, sq.e });
            }

            // Update attacks.
            self.bishop_attacks[us.u] |= moves;

            // King attack power.
            const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
            if (enemy_king_attacks != 0) {
                const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                self.attack_power[us.u].inc(terms.attack_power[PieceType.bishop.u][king_attack_count]);
                if (lib.is_tuning) register(&terms.attack_power[PieceType.bishop.u][king_attack_count], 1, us, .{ PieceType.bishop.e, sq.e, king_attack_count });
            }

            // Outpost.
            if (self.is_outpost(sq, us)) {
                score.inc(terms.bishop_outpost_table[relative_sq.u - 24]);
                if (lib.is_tuning) register(&terms.bishop_outpost_table[relative_sq.u - 24], 1, us, .{ PieceType.bishop.e, sq.e });
            }
        }
        return score;
    }

    fn eval_rooks(self: *Self, comptime us: Color) ScorePair {
        const them: Color = comptime us.opp();
        const pos: *const Position = self.pos;
        const our_pawns: u64 = pos.pawns(us);
        const their_pawns: u64 = pos.pawns(them);
        const mobility_occ: u64 = pos.all() ^ pos.queens(us) ^ pos.rooks(us);

        var score: ScorePair = .empty;

        var iter = bitboards.iterator(pos.rooks(us));
        while (iter.next()) |sq| {
            const relative_sq: Square = sq.relative(us);

            // Psqt.
            score.inc(terms.piece_square_table[PieceType.rook.u][relative_sq.u]);
            if (lib.is_tuning) register(&terms.piece_square_table[PieceType.rook.u][relative_sq.u], 1, us, .{ PieceType.rook.e, sq.e });

            // Mobility.
            const legal_moves: u64 = self.legalize_moves(.rook, us, sq, attacks.get_rook_attacks(sq, mobility_occ));
            const mobility: u64 = legal_moves & self.mobility_areas[us.u];
            const mobility_count: u7 = popcnt(mobility);
            score.inc(terms.rook_mobility_table[mobility_count]);
            if (lib.is_tuning) register(&terms.rook_mobility_table[mobility_count], 1, us, .{ PieceType.rook.e, sq.e, mobility_count });

            // Update attacks.
            self.rook_attacks[us.u] |= legal_moves;

            // King attack power.
            const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
            if (enemy_king_attacks != 0) {
                const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                self.attack_power[us.u].inc(terms.attack_power[PieceType.rook.u][king_attack_count]);
                if (lib.is_tuning) register(&terms.attack_power[PieceType.rook.u][king_attack_count], 1, us, .{ PieceType.rook.e, sq.e, king_attack_count });
            }

            // Open file.
            const our_pawns_on_file: u64 = our_pawns & bitboards.file_bitboards[sq.coord.file];
            if (our_pawns_on_file == 0) {
                const their_pawns_on_file: u64 = their_pawns & bitboards.file_bitboards[sq.coord.file];
                const half_open: u1 = @intFromBool(their_pawns_on_file != 0);
                score.inc(terms.rook_on_file_bonus[half_open][sq.coord.file]);
                if (lib.is_tuning) register(&terms.rook_on_file_bonus[half_open][sq.coord.file], 1, us, .{ PieceType.rook.e, sq.e, half_open });
            }
        }
        return score;
    }

    fn eval_queens(self: *Self, comptime us: Color) ScorePair {
        var score: ScorePair = .empty;
        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        const mobility_occ = pos.all() ^ pos.bishops(us) ^ pos.rooks(us); // TODO: verify

        var iter = bitboards.iterator(pos.queens(us));
        while (iter.next()) |sq| {
            const relative_sq: Square = sq.relative(us);

            // Psqt
            score.inc(terms.piece_square_table[PieceType.queen.u][relative_sq.u]);
            if (lib.is_tuning) register(&terms.piece_square_table[PieceType.queen.u][relative_sq.u], 1, us, .{ PieceType.queen, sq.e });

            // Mobility
            const moves: u64 = self.legalize_moves(.queen, us, sq, attacks.get_queen_attacks(sq, mobility_occ));
            const mobility: u64 = moves & self.mobility_areas[us.u];
            const mobility_count: u7 = popcnt(mobility);
            score.inc(terms.queen_mobility_table[mobility_count]);
            if (lib.is_tuning) register(&terms.queen_mobility_table[mobility_count], 1, us, .{PieceType.queen, sq.e});

            // Update
            self.queen_attacks[us.u] |= moves;

            // King attack power.
            const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
            if (enemy_king_attacks != 0) {
                const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                self.attack_power[us.u].inc(terms.attack_power[PieceType.queen.u][king_attack_count]);
                if (lib.is_tuning) register(&terms.attack_power[PieceType.queen.u][king_attack_count], 1, us, .{ PieceType.queen.e, sq.e, king_attack_count });
            }
        }
        return score;
    }

    fn eval_king(self: *Self, comptime us: Color) ScorePair {
        var score: ScorePair = .empty;
        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        const our_pawns: u64 = pos.pawns(us);
        const their_pawns: u64 = pos.pawns(them);
        const our_king_sq: Square = self.king_squares[us.u];
        const their_king_sq: Square = self.king_squares[them.u];
        var iter: bitboards.BitboardIterator = undefined;

        const relative_sq: Square = our_king_sq.relative(us);
        // Psqt.
        score.inc(terms.piece_square_table[PieceType.king.u][relative_sq.u]);
        if (lib.is_tuning) register(&terms.piece_square_table[PieceType.king.u][relative_sq.u], 1, us, .{ PieceType.king.e, our_king_sq.e });

        // Pawn protection
        const protecting_pawns: u64 = our_pawns & self.king_areas[us.u];
        iter = .init(protecting_pawns);
        while (iter.next()) |sq| {
            const sp: ScorePairPtr = get_pawn_protection_scorepair(us, our_king_sq, sq);
            score.inc(sp.*);
            if (lib.is_tuning) register(sp, 1, us, .{ PieceType.king.e, our_king_sq.e, sq.e });
        }

        // Pawn storm to enemy king.
        const storming_pawns: u64 = our_pawns & self.pawn_storm_areas[them.u];
        iter = .init(storming_pawns);
        while (iter.next()) |pawn_sq| {
            const sp: ScorePairPtr = get_pawn_storm_scorepair(us, their_king_sq, pawn_sq);
            score.inc(sp.*);
            if (lib.is_tuning) register(sp, 1, us, .{ PieceType.king, our_king_sq.e, pawn_sq.e });
        }

        // Open files to our king.
        const our_pawns_on_file: u64 = our_pawns & bitboards.file_bitboards[our_king_sq.coord.file];
        if (our_pawns_on_file == 0) {
            const their_pawns_on_file: u64 = their_pawns & bitboards.file_bitboards[our_king_sq.coord.file];
            const half_open: u1 = if (their_pawns_on_file != 0) 1 else 0;
            score.inc(terms.king_on_file_penalty[half_open][our_king_sq.coord.file]);
            if (lib.is_tuning) register(&terms.king_on_file_penalty[half_open][our_king_sq.coord.file], 1, us, .{ PieceType.king.e, half_open, our_king_sq.coord.file });
        }

        // King danger.
        score.dec(self.attack_power[them.u]);

        return score;
    }

    fn eval_threats(self: *Self, comptime us: Color) ScorePair {
        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        const our_pieces: u64 = pos.by_color(us);
        const our_attacks: u64 = self.pawn_attacks[us.u] | self.knight_attacks[us.u] | self.bishop_attacks[us.u] | self.rook_attacks[us.u] | self.queen_attacks[us.u];
        const their_attacks: u64 = self.pawn_attacks[them.u] | self.knight_attacks[them.u] | self.bishop_attacks[them.u] | self.rook_attacks[them.u] | self.queen_attacks[them.u];

        var score: ScorePair = .empty;
        var iter: bitboards.BitboardIterator = undefined;

        // Their pawn threats.
        iter = .init(self.pawn_attacks[them.u] & our_pieces);
        while (iter.next()) |sq| {
            const threatened_piece = pos.board[sq.u].piecetype();
            const is_defended: u1 = @intFromBool(bitboards.contains_square(our_attacks, sq));
            score.inc(terms.threatened_by_pawn_penalty[threatened_piece.u][is_defended]);
            if (lib.is_tuning) register(&terms.threatened_by_pawn_penalty[threatened_piece.u][is_defended], 1, us, .{ PieceType.pawn.e, threatened_piece.e, is_defended });
        }

        // Their knight threats.
        iter = .init(self.knight_attacks[them.u] & our_pieces);
        while (iter.next()) |sq| {
            const threatened_piece = pos.board[sq.u].piecetype();
            const is_defended: u1 = @intFromBool(bitboards.contains_square(our_attacks, sq));
            score.inc(terms.threatened_by_knight_penalty[threatened_piece.u][is_defended]);
            if (lib.is_tuning) register(&terms.threatened_by_knight_penalty[threatened_piece.u][is_defended], 1, us, .{ PieceType.knight.e, threatened_piece.e, is_defended });
        }

        // Their bishop threats.
        iter = .init(self.bishop_attacks[them.u] & our_pieces);
        while (iter.next()) |sq| {
            const threatened_piece = pos.board[sq.u].piecetype();
            const is_defended: u1 = @intFromBool(bitboards.contains_square(our_attacks, sq));
            score.inc(terms.threatened_by_bishop_penalty[threatened_piece.u][is_defended]);
            if (lib.is_tuning) register(&terms.threatened_by_bishop_penalty[threatened_piece.u][is_defended], 1, us, .{ PieceType.bishop.e, threatened_piece.e, is_defended });
        }

        // Their rook threats.
        iter = .init(self.rook_attacks[them.u] & our_pieces);
        while (iter.next()) |sq| {
            const threatened_piece = pos.board[sq.u].piecetype();
            const is_defended: u1 = @intFromBool(bitboards.contains_square(our_attacks, sq));
            score.inc(terms.threatened_by_rook_penalty[threatened_piece.u][is_defended]);
            if (lib.is_tuning) register(&terms.threatened_by_rook_penalty[threatened_piece.u][is_defended], 1, us, .{ PieceType.rook.e, threatened_piece.e, is_defended });
        }

        // Our pawn push threats. // TODO: #future also use the double pawn push.
        // Get the squares defended by the enemy, excluding squares that are defended by our pawns and not attacked by their pawns
        const their_piece_attacks: u64 = self.knight_attacks[them.u] | self.bishop_attacks[them.u] | self.rook_attacks[them.u] | self.queen_attacks[them.u];
        const their_protected_squares: u64 = self.pawn_attacks[them.u] | (their_piece_attacks & ~self.pawn_attacks[us.u]);
        // Get the single pawn pushes.
        const safe_pawn_pushes: u64 = funcs.pawns_shift(pos.pawns(us), us, .up) & ~pos.all() & ~their_protected_squares;
        // Which enemy piece (no pawn) would be attacked if we push our pawn.
        const pawn_push_threats: u64 = funcs.pawns_attacks(safe_pawn_pushes, us) & pos.by_color(them) & ~pos.pawns(them);

        iter = .init(pawn_push_threats);
        while (iter.next()) |sq| {
            const threatened: PieceType = pos.board[sq.u].piecetype();
            if (threatened.e != .no_piecetype) {
                score.inc(terms.pawn_push_threat_table[threatened.u]);
                if (lib.is_tuning) register(&terms.pawn_push_threat_table[threatened.u], 1, us, .{ PieceType.pawn.e, sq.e, threatened.e });
            }
        }

        // Possible checks.
        const occ: u64 = pos.all();
        const their_king_sq: Square = self.king_squares[them.u];
        // Get the checking squares.
        const checking_squares_knight: u64 = attacks.get_knight_attacks(their_king_sq);
        const checking_squares_bishop: u64 = attacks.get_bishop_attacks(their_king_sq, occ);
        const checking_squares_rook: u64 = attacks.get_rook_attacks(their_king_sq, occ);
        const checking_squares_queen: u64 = checking_squares_rook | checking_squares_bishop;
        const unsafe: u64 = (their_attacks | self.king_attacks[them.u]);
        const safe: u64 = ~unsafe;
        const not_pawns: u64 = ~pos.pawns(us);
        // Determine our safe checks.
        const knight_checks: u64 = not_pawns & checking_squares_knight & self.knight_attacks[us.u];
        const bishop_checks: u64 = not_pawns & checking_squares_bishop & self.bishop_attacks[us.u];
        const rook_checks  : u64 = not_pawns & checking_squares_rook & self.rook_attacks[us.u];
        const queen_checks : u64 = not_pawns & checking_squares_queen & self.queen_attacks[us.u];
        const n_count: u7 = popcnt(knight_checks & safe);
        const b_count: u7 = popcnt(bishop_checks & safe);
        const r_count: u7 = popcnt(rook_checks & safe);
        const q_count: u7 = popcnt(queen_checks & safe);
        score.inc(terms.safe_check_bonus[PieceType.knight.u].mul(n_count));
        score.inc(terms.safe_check_bonus[PieceType.bishop.u].mul(b_count));
        score.inc(terms.safe_check_bonus[PieceType.rook.u].mul(r_count));
        score.inc(terms.safe_check_bonus[PieceType.queen.u].mul(q_count));
        if (lib.is_tuning) {
            register(&terms.safe_check_bonus[PieceType.knight.u], n_count, us, .{ PieceType.knight.e, n_count });
            register(&terms.safe_check_bonus[PieceType.bishop.u], b_count, us, .{ PieceType.bishop.e, b_count });
            register(&terms.safe_check_bonus[PieceType.rook.u], r_count, us, .{ PieceType.rook.e, r_count });
            register(&terms.safe_check_bonus[PieceType.queen.u], q_count, us, .{ PieceType.queen.e, q_count });
        }

        return score;
    }

    /// Returns true if the square is not attacked by their pawn and the square is protected by our pawn. Only for rank 4, 5, 6.
    fn is_outpost(self: *Self, sq: Square, comptime us: Color) bool {
        const them = comptime us.opp();
        const ranks456: u64 = comptime bitboards.relative_rank_456_bitboard(us);
        return ranks456 & sq.to_bitboard() & self.pawn_attacks[us.u] & ~self.pawn_attacks[them.u] != 0;
    }

    fn get_pawn_protection_scorepair(comptime us: Color, king: Square, pawn: Square) ScorePairPtr {
        if (comptime lib.verifications) {
            verify_king_pawn_protection_area(us, king, pawn);
        }
        const mul: i16 = comptime if (us.e == .black) -1 else 1;
        const rank_diff: i16 = int(i16, pawn.coord.rank) - king.coord.rank;
        const file_diff: i16 = int(i16, pawn.coord.file) - king.coord.file;
        const i: u16 = @abs(7 - (rank_diff * 3 + file_diff) * mul); // 7 is the king index.
        return &terms.pawn_protection_table[i];
    }

    fn get_pawn_storm_scorepair(comptime attacker: Color, defending_king: Square, attacking_pawn: Square) ScorePairPtr {
        if (lib.verifications) {
            verify_king_pawn_storm_area(attacker, defending_king, attacking_pawn);
        }
        const defender: Color = comptime attacker.opp();
        const mul: i16 = comptime if (defender.e == .black) -1 else 1;
        const rank_diff: i16 = int(i16, attacking_pawn.coord.rank) - defending_king.coord.rank;
        const file_diff: i16 = int(i16, attacking_pawn.coord.file) - defending_king.coord.file;
        const i: u16 = @abs(19 - (rank_diff * 3 + file_diff) * mul); // 19 is the king index.
        return &terms.pawn_storm_table[i];
    }

    fn legalize_moves(self: *Evaluator, comptime pt: PieceType, comptime us: Color, from: Square, bb_moves: u64) u64 {
        const pinned: u64 = self.pinned[us.u] & from.to_bitboard();
        // Piece is not pinned: we regard this as legal.
        if (pinned == 0) {
            return bb_moves;
        }
        // Knights cannot escape pins.
        if (pt.e == .knight) {
            return 0;
        }
        const king_sq: Square = self.king_squares[us.u];
        const dir: types.Direction = squarepairs.get(king_sq, from).direction;
        const bb_same_direction_as_pin: u64 = bitboards.direction_bitboards[@intFromEnum(dir)][king_sq.u];
        return bb_moves & bb_same_direction_as_pin;
    }

    /// Check if the pawn square is inside the 3x4 king area.
    fn verify_king_pawn_protection_area(comptime us: Color, our_king_sq: Square, our_pawn_sq: Square) void {
        const area: u64 = if (us.e == .white) king_areas_white[our_king_sq.u] else king_areas_black[our_king_sq.u];
        const ok: bool = bitboards.contains_square(area, our_pawn_sq);
        if (!ok) {
            lib.verify(ok, "verify_king_pawn_protection_area", .{});
        }
    }

    /// Check if the pawn square is inside the 3x7 king area.
    fn verify_king_pawn_storm_area(comptime attacker: Color, their_king_sq: Square, our_pawn_sq: Square) void {
        const area: u64 = if (attacker.e == .black) king_pawnstorm_areas_white[their_king_sq.u] else king_pawnstorm_areas_black[their_king_sq.u];
        const ok: bool = bitboards.contains_square(area, our_pawn_sq);
        if (!ok) {
            lib.verify(ok, "verify_king_pawn_storm_area (attacker {t}): pawn on {t} is not inside pawnstormarea of {t}", .{ attacker.e, our_pawn_sq.e, their_king_sq.e });
        }
    }

    fn register(sp: *ScorePair, multiply: u8, us: Color, debugargs: anytype) void {
        lib.only_when_tuning();
        hcetuner.register_scorepair_usage(sp, multiply, us, debugargs);
    }
};

fn compute_king_areas_white() [Square.count]u64 {
    var ka: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        ka[sq.u] = sq.to_bitboard() | attacks.get_king_attacks(sq);
        // Go 1 rank further.
        ka[sq.u] |= funcs.pawns_shift(ka[sq.u], .white, .up);
    }
    return ka;
}

fn compute_king_areas_black() [Square.count]u64 {
    var ka: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        ka[sq.u] = sq.to_bitboard() | attacks.get_king_attacks(sq);
        // Go 1 rank further.
        ka[sq.u] |= funcs.pawns_shift(ka[sq.u], .black, .up);
    }
    return ka;
}

fn compute_king_pawnstorm_areas_white() [Square.count]u64 {
    var ps: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        ps[sq.u] = bitboards.passed_pawn_masks_white[sq.u];
        // Include the squares next to the king.
        ps[sq.u] |= funcs.pawns_shift(ps[sq.u], .black, .up);
        ps[sq.u] &= ~bitboards.bb_rank_8; //#testing
        ps[sq.u] &= ~sq.to_bitboard(); //#testing
    }
    return ps;
}

fn compute_king_pawnstorm_areas_black() [Square.count]u64 {
    var ps: [Square.count]u64 = @splat(0);
    for (Square.all) |sq| {
        ps[sq.u] = bitboards.passed_pawn_masks_black[sq.u];
        // Include the squares next to the king.
        ps[sq.u] |= funcs.pawns_shift(ps[sq.u], .white, .up);
        ps[sq.u] &= ~bitboards.bb_rank_1; //#testing
        ps[sq.u] &= ~sq.to_bitboard(); //#testing
    }
    return ps;
}
