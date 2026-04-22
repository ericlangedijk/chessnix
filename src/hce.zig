// zig fmt: off

//! Hand Crafted Evaluation.

const std = @import("std");
const assert = std.debug.assert;

const lib = @import("lib.zig");
const attacks = @import("attacks.zig");
const bitboards = @import("bitboards.zig");
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

const bitloop = funcs.bitloop;
const popcnt = funcs.popcnt;

const ScorePair = types.ScorePair;
const Color = types.Color;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Square = types.Square;
const CastleType = types.CastleType;
const Move = types.Move;
const Position = position.Position;

const terms = hceterms.terms;

pub const Evaluator = struct {
    const Self = @This();
    /// Ref to position, only known at evaluation time.
    pos: *const Position,
    /// Cached pins.
    pins: u64,
    /// The squares of the kings.
    king_squares: [2]Square,
    /// The 3x4 squares area around and in front of our king.
    king_areas: [2]u64,
    /// The 3x7 squares in front of our king.
    pawn_storm_areas: [2]u64,
    /// Accessible squares which are not protected by enemy pawns
    mobility_areas: [2]u64,
    /// All pawn attacks (initialized).
    pawn_attacks: [2]u64,
    /// All knight attacks (updated on the fly).
    knight_attacks: [2]u64,
    /// All bishop attacks (overlapping, updated on the fly).
    bishop_attacks: [2]u64,
    /// All rook attacks (overlapping, updated on the fly).
    rook_attacks: [2]u64,
    /// All queen attacks (overlapping, updated on the fly).
    queen_attacks: [2]u64,
    /// All king attacks. Initialized.
    king_attacks: [2]u64,
    /// The attacks combined (updated on the fly) **except** king attacks.
    all_attacks: [2]u64,
    /// All attacks to the enemy king area (updated on the fly).
    attack_power: [2]ScorePair,

    pub fn init() Self {
        return .{
            .pos = undefined,
            .pins = 0,
            .king_squares = .{ .A1, .A1 },
            .king_areas = .{ 0, 0 },
            .pawn_storm_areas = .{ 0, 0 },
            .mobility_areas = .{ 0, 0 },
            .pawn_attacks = .{ 0, 0 },
            .knight_attacks = .{ 0, 0 },
            .bishop_attacks = .{ 0, 0 },
            .rook_attacks = .{ 0, 0 },
            .queen_attacks = .{ 0, 0 },
            .king_attacks = .{ 0, 0 },
            .all_attacks = .{ 0, 0},
            .attack_power = .{ .empty, .empty },
        };
    }

    pub fn evaluate(self: *Self, pos: *const Position) i32 {
        // Init pos reference.
        self.pos = pos;
        var score: ScorePair = get_material_scorepair(pos);

        // We can only use pinned pieces of the stm. Remember: pins include the enemy pinner.
        self.pins = pos.our_pins() & pos.by_color(pos.stm);

        // Init fields.
        self.king_squares = .{ pos.king_square(Color.WHITE), pos.king_square(Color.BLACK) };

        self.king_areas[0] = bitboards.king_areas_white[self.king_squares[0].u];
        self.king_areas[1] = bitboards.king_areas_black[self.king_squares[1].u];

        self.pawn_storm_areas[0] = bitboards.king_pawnstorm_areas_white[self.king_squares[0].u];
        self.pawn_storm_areas[1] = bitboards.king_pawnstorm_areas_black[self.king_squares[1].u];

        self.pawn_attacks = .{
            funcs.pawns_shift(pos.pawns(Color.WHITE), Color.WHITE, .northwest) | funcs.pawns_shift(pos.pawns(Color.WHITE), Color.WHITE, .northeast),
            funcs.pawns_shift(pos.pawns(Color.BLACK), Color.BLACK, .northwest) | funcs.pawns_shift(pos.pawns(Color.BLACK), Color.BLACK, .northeast),
        };

        self.all_attacks = self.pawn_attacks;

        self.mobility_areas = .{
            ~(pos.by_color(Color.WHITE) | self.pawn_attacks[Color.BLACK.u]),
            ~(pos.by_color(Color.BLACK) | self.pawn_attacks[Color.WHITE.u]),
        };

        self.knight_attacks = @splat(0);
        self.bishop_attacks = @splat(0);
        self.rook_attacks = @splat(0);
        self.queen_attacks = @splat(0);
        self.king_attacks = .{
            attacks.get_king_attacks(self.king_squares[0]),
            attacks.get_king_attacks(self.king_squares[1]),
        };
        self.attack_power = @splat(ScorePair.empty);

        score.inc(self.eval_pawns(Color.WHITE));
        score.dec(self.eval_pawns(Color.BLACK));

        score.inc(self.eval_knights(Color.WHITE));
        score.dec(self.eval_knights(Color.BLACK));

        score.inc(self.eval_bishops(Color.WHITE));
        score.dec(self.eval_bishops(Color.BLACK));

        score.inc(self.eval_rooks(Color.WHITE));
        score.dec(self.eval_rooks(Color.BLACK));

        score.inc(self.eval_queens(Color.WHITE));
        score.dec(self.eval_queens(Color.BLACK));

        score.inc(self.eval_king(Color.WHITE));
        score.dec(self.eval_king(Color.BLACK));

        score.inc(self.eval_threats(Color.WHITE));
        score.dec(self.eval_threats(Color.BLACK));

        // First add in the tempo bonus. That is what the data was originally tuned for.
        switch (pos.stm.e) {
            .white => score.inc(terms.tempo_bonus),
            .black => score.dec(terms.tempo_bonus),
        }

        score.mg = std.math.clamp(score.mg, -scoring.static_eval_before_scaling_threshold, scoring.static_eval_before_scaling_threshold);
        score.eg = std.math.clamp(score.eg, -scoring.static_eval_before_scaling_threshold, scoring.static_eval_before_scaling_threshold);

        // Scale the endresult (this plays stronger than scaling the eg component only).
        var result: i32 = types.phased_score(pos.phase(), score);
        const scale: f32 = endgame.scale(pos, result);
        result = funcs.fmul(result, scale);
        result = std.math.clamp(result, -scoring.static_eval_threshold, scoring.static_eval_threshold);
        if (pos.stm.e == .black) {
            result = -result;
        }
        return result;
    }

    fn get_material_scorepair(pos: *const Position) ScorePair {
        var sp: ScorePair = .empty;
        // Skip king.
        inline for (0..5) |pt| {
            const vp: ScorePair = terms.piece_value_table[pt];
            const white_count: u8 = pos.material.counts[0][pt];
            const black_count: u8 = pos.material.counts[1][pt];
            sp.mg += vp.mg * white_count;
            sp.eg += vp.eg * white_count;
            sp.mg -= vp.mg * black_count;
            sp.eg -= vp.eg * black_count;
        }
        return sp;
    }

    fn eval_pawns(self: *Self, comptime us: Color) ScorePair {
        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        const our_pawns: u64 = pos.pawns(us);
        const their_pawns: u64 = pos.pawns(them);

        var score: ScorePair = .empty;
        var passed_pawns: u64 = 0;

        // Pawn phalanx (horizontally next to eachother).
        var phalanx_pawns: u64 = (funcs.shift_bitboard(our_pawns, .east)) & our_pawns;
        while (bitloop(&phalanx_pawns)) |sq| {
            const relative_rank: u3 = funcs.relative_rank(us, sq.coord.rank);
            score.inc(terms.pawn_phalanx_bonus[relative_rank]);
        }

        // Loop through our pawns. Determine passed pawns.
        var loop_pawns: u64 = our_pawns;
        while (bitloop(&loop_pawns)) |sq| {
            const sq_bb: u64 = sq.to_bitboard();
            const relative_sq: Square = sq.relative(us);
            const relative_rank: u3 = relative_sq.coord.rank;
            const file: u3 = sq.coord.file;

            // Psqt.
            score.inc(terms.piece_square_table[PieceType.PAWN.u][relative_sq.u]);

            // Protected pawn.
            const is_protected: bool = self.pawn_attacks[us.u] & sq_bb != 0;
            if (is_protected) {
                score.inc(terms.protected_pawn_bonus[relative_rank]);
            }

            // Doubled pawn.
            const pawns_ahead_on_file = bitboards.forward_file(us, sq) & our_pawns;
            const is_doubled: bool = pawns_ahead_on_file != 0;
            if (is_doubled) {
                score.inc(terms.doubled_pawn_penalty[file]);
            }

            // Passed pawn.
            const their_pawns_ahead: u64 = bitboards.get_passed_pawn_mask(us, sq) & their_pawns;
            const is_passed: bool = their_pawns_ahead == 0;
            if (is_passed) {
                passed_pawns |= sq_bb; // Update passed pawns.
                score.inc(terms.passed_pawn_bonus[relative_rank]);
            }

            // Isolated pawn.
            const pawns_on_adjacent_files = bitboards.adjacent_file_masks[sq.u] & our_pawns;
            const is_isolated: bool = pawns_on_adjacent_files == 0;
            if (is_isolated) {
                score.inc(terms.isolated_pawn_penalty[file]);
            }
        }

        // Passed pawns with kings.
        const king_sq: Square = self.king_squares[us.u];
        const their_king_sq: Square = self.king_squares[them.u];
        while (bitloop(&passed_pawns)) |sq| {
            const their_move: u1 = @intFromBool(pos.stm.e == them.e);
            const relative_rank: u3 = funcs.relative_rank(us, sq.coord.rank);
            const dist_to_king: u3 = funcs.square_distance(sq, king_sq);
            score.inc(terms.king_passed_pawn_distance_table[dist_to_king]);
            const dist_to_enemy_king: u3 = funcs.square_distance(sq, their_king_sq);
            score.inc(terms.enemy_king_passed_pawn_distance_table[dist_to_enemy_king]);
            // Square rule for pawn race.
            const enemy_non_pawn_king_pieces: u64 = pos.by_color(them) & ~pos.kings(them) & ~pos.pawns(them);
            const dist_to_promotion: u3 = (7 - relative_rank);
            if (enemy_non_pawn_king_pieces == 0 and dist_to_promotion < dist_to_enemy_king - their_move) {
                score.inc(terms.king_cannot_reach_passed_pawn_bonus);
            }
        }
        return score;
    }

    fn eval_knights(self: *Self, comptime us: Color) ScorePair {
        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        var score: ScorePair = .empty;

        var loop_knights: u64 = pos.knights(us);
        while (bitloop(&loop_knights)) |sq| {
            const relative_sq: Square = sq.relative(us);

            // Psqt.
            score.inc(terms.piece_square_table[PieceType.KNIGHT.u][relative_sq.u]);

            // Mobility.
            const legal_moves: u64 = self.legalize_moves(PieceType.KNIGHT, us, sq, attacks.get_knight_attacks(sq));
            const mobility: u64 = legal_moves & self.mobility_areas[us.u];
            const cnt: u7 = popcnt(mobility);
            score.inc(terms.knight_mobility_table[cnt]);

            // Update attacks.
            self.knight_attacks[us.u] |= legal_moves;
            self.all_attacks[us.u] |= legal_moves;

            // Attacks to the enemy king.
            const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
            if (enemy_king_attacks != 0) {
                const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                self.attack_power[us.u].inc(terms.attack_power[PieceType.KNIGHT.u][king_attack_count]);
            }

            // Outpost
            if (self.is_outpost(sq, us)) {
                score.inc(terms.knight_outpost_table[relative_sq.u]);
                // Knight outpost is also blocking an enemy pawn.
                const sq_in_front: Square = if (us.e == .white) sq.add(8) else sq.sub(8);
                const is_blocking: bool = pos.board[sq_in_front.u].is_pawn_of_color(them);
                if (is_blocking) {
                    score.inc(terms.knight_outpost_is_blocking_enemy_pawn);
                }
            }
        }
        return score;
    }

    fn eval_bishops(self: *Self, comptime us: Color) ScorePair {
        const them: Color = comptime us.opp();
        const pos: *const Position = self.pos;
        const occ: u64 = pos.all() ^ pos.queens(us) ^ pos.bishops(us);
        var score: ScorePair = .empty;
        var our_bishops: u64 = pos.bishops(us);

        // Bishop pair.
        if ((our_bishops & bitboards.bb_black_squares != 0) and (our_bishops & bitboards.bb_white_squares != 0)) {
            score.inc(terms.bishop_pair_bonus);
        }

        while (bitloop(&our_bishops)) |sq| {
            const relative_sq: Square = sq.relative(us);

            // Psqt.
            score.inc(terms.piece_square_table[PieceType.BISHOP.u][relative_sq.u]);

            // Mobility.
            const moves: u64 = self.legalize_moves(PieceType.BISHOP, us, sq, attacks.get_bishop_attacks(sq, occ));
            const mobility: u64 = moves & self.mobility_areas[us.u];
            const cnt: u7 = popcnt(mobility);
            score.inc(terms.bishop_mobility_table[cnt]);

            // Bishop on long diagonal.
            if (popcnt(moves & bitboards.bb_center_4) > 1) {
                score.inc(terms.bishop_long_diagonal);
            }

            // Update attacks.
            self.bishop_attacks[us.u] |= moves;
            self.all_attacks[us.u] |= moves;

            // King attack.
            const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];

            if (enemy_king_attacks != 0) {
                const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                self.attack_power[us.u].inc(terms.attack_power[PieceType.BISHOP.u][king_attack_count]);
            }

            // Outpost.
            if (self.is_outpost(sq, us)) {
                score.inc(terms.bishop_outpost_table[relative_sq.u]);
                // bishop outpost is also blocking an enemy pawn.
                const sq_in_front: Square = if (us.e == .white) sq.add(8) else sq.sub(8);
                const is_blocking: bool = pos.board[sq_in_front.u].is_pawn_of_color(them);
                if (is_blocking) {
                    score.inc(terms.bishop_outpost_is_blocking_enemy_pawn);
                }
            }
        }
        return score;
    }

    fn eval_rooks(self: *Self, comptime us: Color) ScorePair {
        const them: Color = comptime us.opp();
        const pos: *const Position = self.pos;
        const our_pawns: u64 = pos.pawns(us);
        const their_pawns: u64 = pos.pawns(them);
        const occ: u64 = pos.all() ^ pos.queens(us) ^ pos.rooks(us);

        var score: ScorePair = .empty;
        var our_rooks: u64 = pos.rooks(us);

        while (bitloop(&our_rooks)) |sq| {
            const relative_sq: Square = sq.relative(us);

            // Psqt.
            score.inc(terms.piece_square_table[PieceType.ROOK.u][relative_sq.u]);

            // Mobility.
            const legal_moves: u64 = self.legalize_moves(PieceType.ROOK, us, sq, attacks.get_rook_attacks(sq, occ));
            const mobility: u64 = legal_moves & self.mobility_areas[us.u];
            const cnt: u7 = popcnt(mobility);
            score.inc(terms.rook_mobility_table[cnt]);

            // Update attacks.
            self.rook_attacks[us.u] |= legal_moves;
            self.all_attacks[us.u] |= legal_moves;

            // King attack.
            const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
            if (enemy_king_attacks != 0) {
                const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                self.attack_power[us.u].inc(terms.attack_power[PieceType.ROOK.u][king_attack_count]);
            }

            // Open file.
            const our_pawns_on_file: u64 = our_pawns & bitboards.file_bitboards[sq.coord.file];
            if (our_pawns_on_file == 0) {
                const their_pawns_on_file: u64 = their_pawns & bitboards.file_bitboards[sq.coord.file];
                const half_open: u1 = @intFromBool(their_pawns_on_file != 0);
                score.inc(terms.rook_on_file_bonus[half_open][sq.coord.file]);
            }
        }
        return score;
    }

    fn eval_queens(self: *Self, comptime us: Color) ScorePair {
        var score: ScorePair = .empty;
        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        //const occupied = pos.all() ^ pos.bishops(us) ^ pos.rooks(us);
        const occupied = pos.all() & ~pos.bishops(us) & ~pos.rooks(us);

        var our_queens: u64 = pos.queens(us);
        while (bitloop(&our_queens)) |sq| {
            const relative_sq: Square = sq.relative(us);

            // Material
            //score.inc(terms.piece_value_table[PieceType.QUEEN.u]);

            // Psqt
            score.inc(terms.piece_square_table[PieceType.QUEEN.u][relative_sq.u]);

            // Mobility
            const moves: u64 = self.legalize_moves(PieceType.QUEEN, us, sq, attacks.get_queen_attacks(sq, occupied));
            const mobility: u64 = moves & self.mobility_areas[us.u];
            const cnt: u7 = popcnt(mobility);
            score.inc(terms.queen_mobility_table[cnt]);

            // Update
            self.queen_attacks[us.u] |= moves;
            self.all_attacks[us.u] |= moves;

            // King attack.
            const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
            if (enemy_king_attacks != 0) {
                const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                self.attack_power[us.u].inc(terms.attack_power[PieceType.QUEEN.u][king_attack_count]);

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

        // Psqt
        const relative_sq: Square = our_king_sq.relative(us);
        score.inc(terms.piece_square_table[PieceType.KING.u][relative_sq.u]);

        // Pawn protection
        var protecting_pawns: u64 = our_pawns & self.king_areas[us.u];
        while (bitloop(&protecting_pawns)) |sq| {
            const sp: *const ScorePair = get_pawn_protection_scorepair(us, our_king_sq, sq);
            score.inc(sp.*);
        }

        // Pawn storm to enemy king.
        var storming_pawns: u64 = our_pawns & self.pawn_storm_areas[them.u];
        while (bitloop(&storming_pawns)) |sq| {
            const sp: *const ScorePair = get_pawn_storm_scorepair(us, their_king_sq, sq);
            score.inc(sp.*);
        }

        // Open files to our king.
        const our_pawns_on_file: u64 = our_pawns & bitboards.file_bitboards[our_king_sq.coord.file];
        if (our_pawns_on_file == 0) {
            const their_pawns_on_file: u64 = their_pawns & bitboards.file_bitboards[our_king_sq.coord.file];
            const half_open: u1 = if (their_pawns_on_file != 0) 1 else 0;
            score.inc(terms.king_on_file_penalty[half_open][our_king_sq.coord.file]);
        }

        // King danger.
        score.dec(self.attack_power[them.u]);

        return score;
    }

    fn eval_threats(self: *Self, comptime us: Color) ScorePair {
        const pos: *const Position = self.pos;
        const them: Color = comptime us.opp();
        const our_pieces: u64 = pos.by_color(us);
        const our_attacks: u64 = self.all_attacks[us.u];
        const their_attacks: u64 = self.all_attacks[them.u];

        var score: ScorePair = .empty;
        var bb: u64 = undefined;

        // Their pawn threats.
        bb = self.pawn_attacks[them.u] & our_pieces;
        while (bitloop(&bb)) |sq| {
            const threatened_piece = pos.board[sq.u].piecetype();
            const is_defended: u1 = @intFromBool(funcs.contains_square(our_attacks, sq));
            score.inc(terms.threatened_by_pawn_penalty[threatened_piece.u][is_defended]);
        }

        // Their knight threats.
        bb = self.knight_attacks[them.u] & our_pieces;
        while (bitloop(&bb)) |sq|{
            const threatened_piece = pos.board[sq.u].piecetype();
            const is_defended: u1 = @intFromBool(funcs.contains_square(our_attacks, sq));
            score.inc(terms.threatened_by_knight_penalty[threatened_piece.u][is_defended]);
        }

        // Their bishop threats.
        bb = self.bishop_attacks[them.u] & our_pieces;
        while (bitloop(&bb)) |sq| {
            const threatened_piece = pos.board[sq.u].piecetype();
            const is_defended: u1 = @intFromBool(funcs.contains_square(our_attacks, sq));
            score.inc(terms.threatened_by_bishop_penalty[threatened_piece.u][is_defended]);
        }

        // Their rook threats.
        bb = self.rook_attacks[them.u] & our_pieces;
        while (bitloop(&bb)) |sq| {
            const threatened_piece = pos.board[sq.u].piecetype();
            const is_defended: u1 = @intFromBool(funcs.contains_square(our_attacks, sq));
            score.inc(terms.threatened_by_rook_penalty[threatened_piece.u][is_defended]);
        }

        // Our pawn push threats.
        // Get the squares defended by the enemy, excluding squares that are defended by our pawns and not attacked by their pawns
        const their_piece_attacks: u64 = self.knight_attacks[them.u] | self.bishop_attacks[them.u] | self.rook_attacks[them.u] | self.queen_attacks[them.u];
        const their_protected_squares: u64 = self.pawn_attacks[them.u] | (their_piece_attacks & ~self.pawn_attacks[us.u]);
        const safe_pawn_pushes: u64 = funcs.pawns_shift(pos.pawns(us), us, .up) & ~pos.all() & ~their_protected_squares; // TODO: this is only singlepush.
        // Which enemy piece would be attacked if we push our pawn.
        var pawn_push_threats: u64 = (funcs.pawns_shift(safe_pawn_pushes, us, .northwest) | funcs.pawns_shift(safe_pawn_pushes, us, .northeast)) & pos.by_color(them) & ~pos.pawns(them);
        while (bitloop(&pawn_push_threats)) |sq| {
            const threatened: Piece = pos.board[sq.u];
            score.inc(terms.pawn_push_threat_table[threatened.u]);
        }

        // #testing pawnbreaks
        // const rank_3: u64 = comptime funcs.relative_rank(us, bitboards.rank_3);
        // var virtual_pawn_pushes: u64 = funcs.pawns_shift(pos.pawns(us), us, .up) & ~pos.all_pawns();
        // virtual_pawn_pushes |= funcs.pawns_shift(virtual_pawn_pushes & bitboards.rank_bitboards[rank_3], us, .up) & ~pos.all_pawns();
        // //funcs.print_bitboard(virtual_pawn_pushes);
        // //var pawn_breaks: u64 = (funcs.pawns_shift(virtual_pawn_pushes, us, .northwest) | funcs.pawns_shift(virtual_pawn_pushes, us, .northeast)) & pos.pawns(them);
        // var breaks: u8 = 0;
        // while (bitloop(&virtual_pawn_pushes)) |sq| {
        //     const target: u64 = attacks.get_pawn_attacks(sq, us) & pos.pawns(them);
        //     if (popcnt(target) == 1) {
        //         //lib.io.debugprint("{t} pawnbreak to {t}\n", .{ us.e, sq.e});
        //         breaks += 1;
        //     }
        // }
        // const sp: ScorePair = comptime types.pair(2, -4);
        // score.inc(sp.mul(breaks));
        // //lib.io.debugprint("{} pawnbreaks {}\n", .{ us.e, breaks });


        // Possible checks.
        // Get the checking squares.
        const occ: u64 = pos.all();
        const their_king_sq: Square = self.king_squares[them.u];

        const checking_squares_knight: u64 = attacks.get_knight_attacks(their_king_sq);
        const checking_squares_bishop: u64 = attacks.get_bishop_attacks(their_king_sq, occ);
        const checking_squares_rook: u64 = attacks.get_rook_attacks(their_king_sq, occ);
        const checking_squares_queen: u64 = checking_squares_rook | checking_squares_bishop;

        // Get the safe squares.
        const unsafe: u64 = (their_attacks | self.king_attacks[them.u]);
        const safe: u64 = ~unsafe;
        const not_pawns: u64 = ~pos.pawns(us);

        // Determine our checks.
        const knight_checks: u64 = not_pawns & checking_squares_knight & self.knight_attacks[us.u];
        const bishop_checks: u64 = not_pawns & checking_squares_bishop & self.bishop_attacks[us.u];
        const rook_checks  : u64 = not_pawns & checking_squares_rook & self.rook_attacks[us.u];
        const queen_checks : u64 = not_pawns & checking_squares_queen & self.queen_attacks[us.u];

        score.inc(terms.safe_check_bonus[PieceType.KNIGHT.u].mul(popcnt(knight_checks & safe)));
        score.inc(terms.safe_check_bonus[PieceType.BISHOP.u].mul(popcnt(bishop_checks & safe)));
        score.inc(terms.safe_check_bonus[PieceType.ROOK.u].mul(popcnt(rook_checks & safe)));
        score.inc(terms.safe_check_bonus[PieceType.QUEEN.u].mul(popcnt(queen_checks & safe)));

        return score;
    }

    /// Returns true if the square is not attacked by their pawn and the square is protected by our pawn.
    fn is_outpost(self: *Self, sq: Square, comptime us: Color) bool {
        const them: Color = comptime us.opp();
        const sq_bb: u64 = sq.to_bitboard();
        const safe: bool = self.pawn_attacks[us.u] & sq_bb != 0 and self.pawn_attacks[them.u] & sq_bb == 0;
        return safe and funcs.outpost(us) & sq_bb != 0;
    }

    fn get_pawn_protection_scorepair(comptime us: Color, our_king_sq: Square, our_pawn_sq: Square) *const ScorePair {
        // TODO: make symmetrical.
        if (comptime lib.verifications) {
            verify_king_pawn_protection_area(us, our_king_sq, our_pawn_sq);
        }

        const king_index: i32 = 7; // index of king in the 3x4 pawn storm area
        const area_width: i32 = 3;
        const pawn_rank: i32 = our_pawn_sq.coord.rank;
        const pawn_file: i32 = our_pawn_sq.coord.file;
        const rank_diff: i32 = pawn_rank - our_king_sq.coord.rank;
        const file_diff: i32 = pawn_file - our_king_sq.coord.file;
        const mul: i32 = comptime if (us.e == .black) -1 else 1;
        const idx: i32 = king_index - (rank_diff * area_width + file_diff) * mul;
        assert(idx >= 0);
        const i: u32 = @abs(idx);
        return &terms.pawn_protection_table[i];
    }

    fn get_pawn_storm_scorepair(comptime us: Color, their_king_sq: Square, our_pawn_sq: Square) *const ScorePair {
        // TODO: make symmetrical.
        if (lib.verifications) {
            verify_king_pawn_storm_area(us, their_king_sq, our_pawn_sq);
        }

        const them: Color = comptime us.opp();
        const king_index: i32 = 19; // index of king in the 3x7 pawn storm area
        const area_width: i32 = 3;
        const pawn_rank: i32 = our_pawn_sq.coord.rank;
        const pawn_file: i32 = our_pawn_sq.coord.file;
        const rank_diff: i32 = pawn_rank - their_king_sq.coord.rank;
        const file_diff: i32 = pawn_file - their_king_sq.coord.file;
        const mul: i32 = comptime if (them.e == .black) -1 else 1;
        const idx: i32 = king_index - (rank_diff * area_width + file_diff) * mul;
        assert(idx >= 0);
        const i: u32 = @abs(idx);
        return &terms.pawn_storm_table[i];
    }

    /// This thing makes chessnix playing worse. I keep it around if I find the cause of this little drama.
    fn legalize_moves(self: *Evaluator, comptime pt: PieceType, comptime us: Color, from: Square, bb_moves: u64) u64 {
        const from_bb: u64 = from.to_bitboard();
        const pins: u64 = self.pins & from_bb;
        if (pins == 0) {
            return bb_moves;
        }

        // We cannot handle these pins, because they are not there.
        // Remember the position pins include the opponents pieces. That is the tricky part.
        if (comptime lib.verifications) {
            lib.verify(us.e == self.pos.stm.e, "Evaluator.legalize_moves()", .{});
        }

        // Pinned knight cannot escape a pin.
        if (pt.e == .knight) {
            return 0;
        }

        const king_sq: Square = self.king_squares[us.u];

        // Just to be safe we use orelse. But direction should never be null here.
        const dir: types.Direction = bitboards.get_squarepair(king_sq, from).direction orelse return bb_moves;
        const bb_same_direction_as_pin: u64 = bitboards.direction_bitboards[@intFromEnum(dir)][king_sq.u];

        return bb_moves & bb_same_direction_as_pin;
    }

    /// Check if the pawn square is inside the 3x4 king area.
    fn verify_king_pawn_protection_area(comptime us: Color, our_king_sq: Square, our_pawn_sq: Square) void {
        const area: u64 = if (us.e == .white) bitboards.king_areas_white[our_king_sq.u] else bitboards.king_areas_black[our_king_sq.u];
        const ok: bool = funcs.contains_square(area, our_pawn_sq);
        if (!ok) {
            lib.verify(ok, "verify_king_pawn_protection_area", .{});
        }
    }

    /// Check if the pawn square is inside the 3x4 king area.
    fn verify_king_pawn_storm_area(comptime attacker: Color, their_king_sq: Square, our_pawn_sq: Square) void {
        const area: u64 = if (attacker.e == .black) bitboards.king_pawnstorm_areas_white[their_king_sq.u] else bitboards.king_pawnstorm_areas_black[their_king_sq.u];
        const ok: bool = funcs.contains_square(area, our_pawn_sq);
        if (!ok) {
            lib.verify(ok, "verify_king_pawn_storm_area", .{});
        }
    }
};

/// Returns true for castle, enpassant and promotions, regardless of the threshold.
pub fn see(pos: *const Position, m: Move, threshold: i32) bool {
    if (m.is_castle() or m.is_ep() or m.is_promotion()) {
        return true;
    }

    const from: Square = m.from;
    const to: Square = m.to;

    // Set the score to captured piece minus how much we are allowed to lose.
    var score: i32 = pos.board[to.u].value() - threshold;

    if (score < 0) {
        return false;
    }

    score = pos.board[from.u].value() - score;

    // Equal or winning.
    if (score <= 0) {
        return true;
    }

    const queens_bishops = pos.all_queens_bishops();
    const queens_rooks = pos.all_queens_rooks();

    // Execute the move on a bitboard.
    var occupied = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();

    // Get the initial attacks from both sides to the to-square.
    var all_attacks: u64 = pos.get_combined_attacks_to_for_occupation(occupied, to);
    var us: Color = pos.stm;
    var winner: Color = pos.stm;

    attackloop: while (true) {
        us = us.opp();
        all_attacks &= occupied;

        // Get our attackers.
        const our_attackers: u64 = all_attacks & pos.by_color(us);

        // No attackers left.
        if (our_attackers == 0) {
            break :attackloop;
        }
        winner = winner.opp();
        var next_attacker_value: i32 = 0;

        // Get the least valuable next piece.
        get_next_attacker: inline for (PieceType.all) |piecetype| {
            const next_attacker: u64 = our_attackers & pos.by_type(piecetype);
            if (next_attacker != 0) {

                // Clear this attacker.
                const sq: Square = funcs.first_square(next_attacker);
                funcs.clear_square(&occupied, sq);
                // Reveal next x-ray attacker on the attacks bitboard.
                switch (piecetype.e) {
                    .pawn   => {
                        all_attacks |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
                    },
                    .knight => {
                        // Do nothing: a knight move cannot reveal a new slider.
                    },
                    .bishop => {
                        all_attacks |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
                    },
                    .rook   => {
                        all_attacks |= (attacks.get_rook_attacks(to, occupied) & queens_rooks);
                    },
                    .queen  => {
                        all_attacks |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
                        all_attacks |= (attacks.get_rook_attacks(to, occupied) & queens_rooks);
                    },
                    .king   => {
                        // We can exit here: if the king captures and the opponent can capture our king we lose othersize we win.
                        return if (all_attacks & pos.by_color(us.opp()) != 0) pos.stm.u != winner.u else return pos.stm.u == winner.u;
                    },
                }

                next_attacker_value = piecetype.value();
                break :get_next_attacker;
            }
        }

        score = -score + 1 + next_attacker_value;

        // Quit if the exchange is lost or equal
        if (score <= 0) {
            break :attackloop;
        }
    }
    return pos.stm.u == winner.u;
}


/// Indexing [relative rank][controlled]
const space_table: [8][8]ScorePair = {

};
