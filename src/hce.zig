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
const utils = @import("utils.zig");
const tt = @import("tt.zig");
const search = @import("search.zig");
const hcetables = @import("hcetables.zig");

const io = lib.io;

const wtf = lib.wtf;
const float = funcs.float;
const int = funcs.int;
const pop = funcs.pop_square;
const popcnt = funcs.popcnt;
const popcnt_v = funcs.popcnt_v;

const SmallValue = types.SmallValue;
const Value = types.Value;
const ScorePair = types.ScorePair;
const Float = types.Float;

const Color = types.Color;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Square = types.Square;
const Move = types.Move;
const GamePhase = types.GamePhase;
const Position = position.Position;

const W = Color.WHITE;
const B = Color.BLACK;

/// By [square]
pub const king_areas_white: [64]u64 = compute_king_areas_white();
/// By [square]
pub const king_areas_black: [64]u64 = compute_king_areas_black();
/// By [square]
pub const king_pawnstorm_areas_white: [64]u64 = compute_king_pawnstorm_areas_white();
/// By [square]
pub const king_pawnstorm_areas_black: [64]u64 = compute_king_pawnstorm_areas_black();

fn compute_king_areas_white() [64]u64 {
    var ka: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        ka[sq.u] = sq.to_bitboard() | attacks.get_king_attacks(sq);
        // Go 1 rank further.
        ka[sq.u] |= funcs.pawns_shift(ka[sq.u], Color.WHITE, .up);
    }
    return ka;
}

fn compute_king_areas_black() [64]u64 {
    var ka: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        ka[sq.u] = sq.to_bitboard() | attacks.get_king_attacks(sq);
        // Go 1 rank further.
        ka[sq.u] |= funcs.pawns_shift(ka[sq.u], Color.BLACK, .up);
    }
    return ka;
}

fn compute_king_pawnstorm_areas_white() [64]u64 {
    var ps: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        ps[sq.u] = bitboards.passed_pawn_masks_white[sq.u];
        // Include the squares next to the king.
        ps[sq.u] |= funcs.pawns_shift(ps[sq.u], Color.BLACK, .up);
    }
    return ps;
}

fn compute_king_pawnstorm_areas_black() [64]u64 {
    var ps: [64]u64 = @splat(0);
    for (Square.all) |sq| {
        ps[sq.u] = bitboards.passed_pawn_masks_black[sq.u];
        // Include the squares next to the king.
        ps[sq.u] |= funcs.pawns_shift(ps[sq.u], Color.WHITE, .up);
    }
    return ps;
}

pub fn Evaluator(comptime debug_print_details: bool) type {

    return struct {
        const Self = @This();
        /// Ref to position, only known at evaluation time.
        pos: *const Position,
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
        /// All bishop attacks (updated on the fly).
        bishop_attacks: [2]u64,
        /// All rook attacks (updated on the fly).
        rook_attacks: [2]u64,
        /// All queen attacks (updated on the fly).
        queen_attacks: [2]u64,
        /// All attacks to the enemy king area (updated on the fly).
        attack_power: [2]ScorePair,

        pub fn init() Self {
            return .{
                .pos = undefined,
                .king_squares = .{ .A1, .A1 },
                .king_areas = .{ 0, 0 },
                .pawn_storm_areas = .{ 0, 0 },
                .mobility_areas = .{ 0, 0 },
                .pawn_attacks = .{ 0, 0 },
                .knight_attacks = .{ 0, 0 },
                .bishop_attacks = .{ 0, 0 },
                .rook_attacks = .{ 0, 0 },
                .queen_attacks = .{ 0, 0 },
                .attack_power = .{ .empty, .empty },
            };
        }

        pub fn evaluate(self: *Self, pos: *const Position, evalhash: ?*tt.EvalTranspositionTable) Value {

            // Eval TT Probe.
            if (evalhash) |hash| {
                if (hash.probe(pos.key)) |ev| {
                    return ev;
                }
            }

            const has_pawns: bool = pos.all_pawns() != 0;
            const score: Value = switch(has_pawns) {
                false => self.internal_evaluate(pos, false),
                true  => self.internal_evaluate(pos, true),
            };

            if (evalhash) |hash| {
                hash.store(pos.key, score);
            }

            return score;
        }

        /// Here and there the comptime has_pawns is there to boost a little speed.
        fn internal_evaluate(self: *Self, pos: *const Position, comptime has_pawns: bool) Value {

            // Init pos reference.
            self.pos = pos;

            // Init fields.
            self.king_squares = .{ pos.king_square(Color.WHITE), pos.king_square(Color.BLACK) };

            self.king_areas[0] = king_areas_white[self.king_squares[0].u];
            self.king_areas[1] = king_areas_black[self.king_squares[1].u];

            self.pawn_storm_areas[0] = if (!has_pawns) 0 else king_pawnstorm_areas_white[self.king_squares[0].u];
            self.pawn_storm_areas[1] = if (!has_pawns) 0 else king_pawnstorm_areas_black[self.king_squares[1].u];

            if (!has_pawns) {
                self.pawn_attacks = .{ 0, 0 };
            }
            else {
                self.pawn_attacks = .{
                    funcs.pawns_shift(pos.pawns(Color.WHITE), Color.WHITE, .northwest) | funcs.pawns_shift(pos.pawns(Color.WHITE), Color.WHITE, .northeast),
                    funcs.pawns_shift(pos.pawns(Color.BLACK), Color.BLACK, .northwest) | funcs.pawns_shift(pos.pawns(Color.BLACK), Color.BLACK, .northeast),
                };
            }

            self.mobility_areas = .{
                ~(pos.by_color(Color.WHITE) | self.pawn_attacks[Color.BLACK.u]),
                ~(pos.by_color(Color.BLACK) | self.pawn_attacks[Color.WHITE.u]),
            };

            self.knight_attacks = @splat(0);
            self.bishop_attacks = @splat(0);
            self.rook_attacks = @splat(0);
            self.queen_attacks = @splat(0);
            self.attack_power = @splat(ScorePair.empty);

            // Evaluate.
            var score: ScorePair = .empty;

            if (has_pawns) {
                score.inc(self.eval_pawns(Color.WHITE));
                score.dec(self.eval_pawns(Color.BLACK));
                if (debug_print_details) io.debugprint("after P {any}\n", .{ score });
            }

            score.inc(self.eval_knights(Color.WHITE));
            score.dec(self.eval_knights(Color.BLACK));
            if (debug_print_details) io.debugprint("after N {any}\n", .{ score });

            score.inc(self.eval_bishops(Color.WHITE));
            score.dec(self.eval_bishops(Color.BLACK));
            if (debug_print_details) io.debugprint("after B {any}\n", .{ score });

            score.inc(self.eval_rooks(Color.WHITE));
            score.dec(self.eval_rooks(Color.BLACK));
            if (debug_print_details) io.debugprint("after R {any}\n", .{ score });

            score.inc(self.eval_queens(Color.WHITE));
            score.dec(self.eval_queens(Color.BLACK));
            if (debug_print_details) io.debugprint("after Q {any}\n", .{ score });

            score.inc(self.eval_king(has_pawns, Color.WHITE));
            score.dec(self.eval_king(has_pawns, Color.BLACK));
            if (debug_print_details) io.debugprint("after K {any}\n", .{ score });

            score.inc(self.eval_threats(has_pawns, Color.WHITE));
            score.dec(self.eval_threats(has_pawns, Color.BLACK));
            if (debug_print_details) io.debugprint("after T {any}\n", .{ score });

            if (self.pos.stm.e == .black) {
                score.mg = -score.mg;
                score.eg = -score.eg;
            }

            score.inc(hcetables.tempo_bonus);
            if (debug_print_details) trace(hcetables.tempo_bonus, self.pos.stm, null, "tempo", .{});

            const result: Value = sliding_score(pos, score);

            if (debug_print_details) {
                io.debugprint("pair {} {}\n", .{ score.mg, score.eg });
                io.debugprint("final result {}\n", .{ result });
            }

            return result;
        }

        fn eval_pawns(self: *Self, comptime us: Color) ScorePair {
            if (comptime lib.is_paranoid) {
                assert(self.pos.all_pawns() != 0);
            }
            const pos: *const Position = self.pos;
            const them: Color = comptime us.opp();
            var score: ScorePair = undefined;
            const our_pawns: u64 = pos.pawns(us);
            const their_pawns: u64 = pos.pawns(them);
            var sp: ScorePair = undefined;
            var passed_pawns: u64 = 0;

            score = .empty;
            // Pawn phalanx (horizontally next to eachother).
            var connected_pawns: u64 = (funcs.shift_bitboard(our_pawns, .east)) & our_pawns;
            while (connected_pawns != 0) {
                const sq: Square = pop(&connected_pawns);
                const relative_rank: u3 = funcs.relative_rank(us, sq.coord.rank);
                sp = hcetables.pawn_phalanx_bonus[relative_rank];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "pawn phalanx", .{});
            }

            // Loop through the pawns.
            var pawns: u64 = our_pawns;
            while (pawns != 0) {
                const sq: Square = pop(&pawns);
                const sq_bb: u64 = sq.to_bitboard();
                const relative_sq: Square = sq.relative(us);
                const relative_rank: u3 = relative_sq.coord.rank;
                const file: u3 = sq.coord.file;

                // Material.
                score.inc(hcetables.piece_value_table[PieceType.PAWN.u]);
                if (debug_print_details) trace(hcetables.piece_value_table[PieceType.PAWN.u], us, sq, "material P", .{});

                // Psqt.
                sp = hcetables.piece_square_table[PieceType.PAWN.u][relative_sq.u];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "psqt P", .{});

                // Protected pawn.
                if (self.pawn_attacks[us.u] & sq_bb != 0) {
                    sp = hcetables.protected_pawn_bonus[relative_rank];
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "protected pawn", .{});
                }

                // Doubled pawn.
                const pawns_ahead_on_file = funcs.forward_file(us, sq) & our_pawns;
                const is_doubled: bool = pawns_ahead_on_file != 0;
                if (is_doubled) {
                    sp = hcetables.doubled_pawn_penalty[file];
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "doubled pawn", .{});
                }

                // Passed pawn.
                const their_pawns_ahead: u64 = bitboards.get_passed_pawn_mask(us, sq) & their_pawns;
                if (their_pawns_ahead == 0) {
                    passed_pawns |= sq_bb;
                    sp = hcetables.passed_pawn_bonus[relative_rank];
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "passed pawn", .{});
                }

                // Isolated pawn.
                const pawns_on_adjacent_files = bitboards.adjacent_file_masks[sq.u] & our_pawns;
                if (pawns_on_adjacent_files == 0) {
                    sp = hcetables.isolated_pawn_penalty[file];
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "isolated pawn", .{});
                }
            }

            // Pawn - king stuff.
            const king_sq: Square = self.king_squares[us.u];
            const their_king_sq: Square = self.king_squares[them.u];
            while (passed_pawns != 0) {
                const sq: Square = pop(&passed_pawns);
                const their_move: u1 = @intFromBool(pos.stm.e == them.e);
                const relative_rank: u3 = funcs.relative_rank(us, sq.coord.rank);
                const dist_to_king: u3 = funcs.square_distance(sq, king_sq);

                sp = hcetables.king_passed_pawn_distance_table[dist_to_king];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "king dist to passed pawn, dist = {}", .{ dist_to_king });

                const dist_to_enemy_king: u3 = funcs.square_distance(sq, their_king_sq);
                sp = hcetables.enemy_king_passed_pawn_distance_table[dist_to_enemy_king];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "enemy king dist to passed pawn dist = {}", .{ dist_to_enemy_king });

                // Square rule for pawn race.
                const enemy_non_pawn_king_pieces: u64 = pos.by_color(them) & ~pos.kings(them) & ~pos.pawns(them);
                const dist_to_promotion: u3 = (7 - relative_rank);
                if (enemy_non_pawn_king_pieces == 0 and dist_to_promotion < dist_to_enemy_king - their_move) {
                    score.inc(hcetables.king_cannot_reach_passed_pawn_bonus);
                    if (debug_print_details) trace(hcetables.king_cannot_reach_passed_pawn_bonus, us, sq, "enemy king cannot reach pawn", .{});
                }
            }
            return score;
        }

        fn eval_knights(self: *Self, comptime us: Color) ScorePair {
            const pos: *const Position = self.pos;
            const them: Color = comptime us.opp();
            var score: ScorePair = .empty;
            var our_knights: u64 = pos.knights(us);
            var sp: ScorePair = undefined;

            while (our_knights != 0) {
                const sq: Square = pop(&our_knights);
                const relative_sq: Square = sq.relative(us);

                // Material
                score.inc(hcetables.piece_value_table[PieceType.KNIGHT.u]);
                if (debug_print_details) trace(hcetables.piece_value_table[PieceType.KNIGHT.u], us, sq, "material N", .{});

                // Psqt.
                sp = hcetables.piece_square_table[PieceType.KNIGHT.u][relative_sq.u];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "psqt N", .{});

                // Mobility.
                const legal_moves: u64 = self.legalize_moves(PieceType.KNIGHT, us, sq, attacks.get_knight_attacks(sq));
                const mobility: u64 = legal_moves & self.mobility_areas[us.u];
                const cnt: u7 = popcnt(mobility);
                sp = hcetables.knight_mobility_table[cnt];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "knight mobility, popcnt = {}", .{ cnt });

                // Update.
                self.knight_attacks[us.u] |= legal_moves;

                // Attacks to the enemy king.
                const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
                if (enemy_king_attacks != 0) {
                    const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                    sp = hcetables.attack_power[PieceType.KNIGHT.u][king_attack_count];
                    self.attack_power[us.u].inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "attackpower kingarea knight, king_attack_count = {}", .{ king_attack_count });
                }

                // Outpost
                if (self.is_outpost(sq, us)) {
                    sp = hcetables.knight_outpost_table[relative_sq.u];
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "knight outpost", .{});
                }
            }
            return score;
        }

        fn eval_bishops(self: *Self, comptime us: Color) ScorePair {
            var score: ScorePair = .empty;
            const them: Color = comptime us.opp();
            const pos: *const Position = self.pos;
            var our_bishops: u64 = pos.bishops(us);
            const occ: u64 = pos.all() ^ pos.queens(us) ^ pos.bishops(us);
            var sp: ScorePair = undefined;

            // Bishop pair.
            if ((our_bishops & bitboards.bb_black_squares != 0) and (our_bishops & bitboards.bb_white_squares != 0)) {
                sp = hcetables.bishop_pair_bonus;
                score.inc(sp);
                if (debug_print_details) trace(sp, us, null, "bishoppair", .{});
            }

            while (our_bishops != 0) {
                const sq: Square = pop(&our_bishops);
                const relative_sq: Square = sq.relative(us);

                // Material
                score.inc(hcetables.piece_value_table[PieceType.BISHOP.u]);
                if (debug_print_details) trace(hcetables.piece_value_table[PieceType.BISHOP.u], us, sq, "material B", .{});

                // Psqt.
                sp = hcetables.piece_square_table[PieceType.BISHOP.u][relative_sq.u];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "psqt B", .{});

                // Mobility
                const moves: u64 = self.legalize_moves(PieceType.BISHOP, us, sq, attacks.get_bishop_attacks(sq, occ));
                const mobility: u64 = moves & self.mobility_areas[us.u];
                const cnt: u7 = popcnt(mobility);
                sp = hcetables.bishop_mobility_table[cnt];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "bishop mobility, popcnt = {}", .{ cnt });

                // Update
                self.bishop_attacks[us.u] |= moves;

                // King attack.
                const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];

                if (enemy_king_attacks != 0) {
                    const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                    sp = hcetables.attack_power[PieceType.BISHOP.u][king_attack_count];
                    self.attack_power[us.u].inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "attackpower kingarea bishop, king_attack_count = {}", .{ king_attack_count });
                }

                // Outpost
                if (self.is_outpost(sq, us)) {
                    sp = hcetables.bishop_outpost_table[relative_sq.u];
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "bishop outpost", .{});
                }
            }
            return score;
        }

        fn eval_rooks(self: *Self, comptime us: Color) ScorePair {
            var score: ScorePair = .empty;
            const them: Color = comptime us.opp();
            const pos: *const Position = self.pos;
            const our_pawns: u64 = pos.pawns(us);
            const their_pawns: u64 = pos.pawns(them);
            const occ: u64 = pos.all() ^ pos.queens(us) ^ pos.rooks(us);
            var sp: ScorePair = undefined;

            var our_rooks: u64 = pos.rooks(us);
            while (our_rooks != 0) {
                const sq: Square = pop(&our_rooks);
                const relative_sq: Square = sq.relative(us);

                // Material
                score.inc(hcetables.piece_value_table[PieceType.ROOK.u]);
                if (debug_print_details) trace(hcetables.piece_value_table[PieceType.ROOK.u], us, sq, "material R", .{});

                // Psqt
                sp = hcetables.piece_square_table[PieceType.ROOK.u][relative_sq.u];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "psqt R", .{});

                // Mobility
                const legal_moves: u64 = self.legalize_moves(PieceType.ROOK, us, sq, attacks.get_rook_attacks(sq, occ));
                const mobility: u64 = legal_moves & self.mobility_areas[us.u];
                const cnt: u7 = popcnt(mobility);
                sp = hcetables.rook_mobility_table[cnt];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "rook mobility, popcnt = {}", .{ cnt });

                // Update
                self.rook_attacks[us.u] |= legal_moves;

                // King attack.
                const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
                if (enemy_king_attacks != 0) {
                    const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                    sp = hcetables.attack_power[PieceType.ROOK.u][king_attack_count];
                    self.attack_power[us.u].inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "attackpower kingarea rook, king_attack_count = {}", .{ king_attack_count });
                }

                // Open file.
                const our_pawns_on_file: u64 = our_pawns & bitboards.file_bitboards[sq.coord.file];
                if (our_pawns_on_file == 0) {
                    const their_pawns_on_file: u64 = their_pawns & bitboards.file_bitboards[sq.coord.file];
                    const half_open: u1 = @intFromBool(their_pawns_on_file != 0);
                    sp = hcetables.rook_on_file_bonus[half_open][sq.coord.file];
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "open file rook file = {} halfopen = {}", .{ sq.coord.file, half_open });
                }
            }
            return score;
        }

        fn eval_queens(self: *Self, comptime us: Color) ScorePair {
            var score: ScorePair = .empty;
            const pos: *const Position = self.pos;
            const them: Color = comptime us.opp();
            const occupied = pos.all() ^ pos.bishops(us) ^ pos.rooks(us);
            var sp: ScorePair = undefined;

            var our_queens: u64 = pos.queens(us);
            while (our_queens != 0) {
                const sq: Square = pop(&our_queens);
                const relative_sq: Square = sq.relative(us);

                // Material
                score.inc(hcetables.piece_value_table[PieceType.QUEEN.u]);
                if (debug_print_details) trace(hcetables.piece_value_table[PieceType.QUEEN.u], us, sq, "material Q", .{});

                // Psqt
                sp = hcetables.piece_square_table[PieceType.QUEEN.u][relative_sq.u];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "psqt Q", .{});

                // Mobility
                const moves: u64 = self.legalize_moves(PieceType.QUEEN, us, sq, attacks.get_queen_attacks(sq, occupied));
                const mobility: u64 = moves & self.mobility_areas[us.u];
                const cnt: u7 = popcnt(mobility);
                sp = hcetables.queen_mobility_table[cnt];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "queen mobility, cnt = {}", .{ cnt });

                // Update
                self.queen_attacks[us.u] |= moves;

                // King attack.
                const enemy_king_attacks: u64 = mobility & self.king_areas[them.u];
                if (enemy_king_attacks != 0) {
                    const king_attack_count: u7 = @min(7, popcnt(enemy_king_attacks));
                    sp = hcetables.attack_power[PieceType.QUEEN.u][king_attack_count];
                    self.attack_power[us.u].inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "attackpower kingarea queen, king_attack_count = {}", .{ king_attack_count });
                }
            }
            return score;
        }

        fn eval_king(self: *Self, comptime has_pawns: bool, comptime us: Color) ScorePair {
            var score: ScorePair = .empty;
            const pos: *const Position = self.pos;
            const them: Color = comptime us.opp();
            const our_pawns: u64 = pos.pawns(us);
            const their_pawns: u64 = pos.pawns(them);
            const our_king_sq: Square = self.king_squares[us.u];
            const their_king_sq: Square = self.king_squares[them.u];
            var sp: ScorePair = undefined;

            // Psqt
            const relative_sq: Square = our_king_sq.relative(us);
            sp = hcetables.piece_square_table[PieceType.KING.u][relative_sq.u];
            score.inc(sp);
            if (debug_print_details) trace(sp, us, our_king_sq, "psqt K", .{});

            if (has_pawns) {
                // Pawn protection
                var pawn_protectors: u64 = our_pawns & self.king_areas[us.u];
                while (pawn_protectors != 0) {
                    const sq: Square = pop(&pawn_protectors);
                    sp = get_pawn_protection_scorepair(us, our_king_sq, sq);
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, our_king_sq, "king pawn protection, pawn = {t}", .{ sq.e });
                }

                // Pawn storm to enemy king.
                var storming_pawns: u64 = our_pawns & self.pawn_storm_areas[them.u];
                while (storming_pawns != 0) {
                    const sq: Square = pop(&storming_pawns);
                    sp = get_pawn_storm_scorepair(us, their_king_sq, sq);
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, their_king_sq, "king pawn storm, pawn = {t}", .{ sq.e });
                }
            }

            // Open files to our king.
            const our_pawns_on_file: u64 = our_pawns & bitboards.file_bitboards[our_king_sq.coord.file];
            if (our_pawns_on_file == 0) {
                const their_pawns_on_file: u64 = their_pawns & bitboards.file_bitboards[our_king_sq.coord.file];
                const half_open: u1 = if (their_pawns_on_file != 0) 1 else 0;
                sp = hcetables.king_on_file_penalty[half_open][our_king_sq.coord.file];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, our_king_sq, "king on open file, halfopen = {}", .{ half_open });
            }

            // King danger.
            score.dec(self.attack_power[them.u]);
            if (debug_print_details) trace(self.attack_power[them.u], us, our_king_sq, "king danger (decrease)", .{});

            return score;
        }

        fn eval_threats(self: *Self, comptime has_pawns: bool, comptime us: Color) ScorePair {
            var score: ScorePair = .empty;
            const pos: *const Position = self.pos;
            const them: Color = comptime us.opp();
            const our_pieces: u64 = pos.by_color(us);
            const our_attacks: u64 = self.pawn_attacks[us.u] | self.knight_attacks[us.u] | self.bishop_attacks[us.u] | self.rook_attacks[us.u] | self.queen_attacks[us.u];
            var sp: ScorePair = undefined;
            var bb: u64 = undefined;

            // Their pawn threats.
            if (has_pawns) {
                bb = self.pawn_attacks[them.u] & our_pieces;
                while (bb != 0) {
                    const sq: Square = pop(&bb);
                    const threatened_piece = pos.board[sq.u].piecetype();
                    const is_defended: u1 = @intFromBool(funcs.contains_square(our_attacks, sq));
                    sp = hcetables.threatened_by_pawn_penalty[threatened_piece.u][is_defended];
                    score.inc(sp);
                    if (debug_print_details) trace(sp, us, sq, "{t} threatened by pawn, is_defended {}", .{ threatened_piece.e, is_defended });
                }
            }

            // Their knight threats.
            bb = self.knight_attacks[them.u] & our_pieces;
            while (bb != 0) {
                const sq: Square = pop(&bb);
                const threatened_piece = pos.board[sq.u].piecetype();
                const is_defended: u1 = @intFromBool(funcs.contains_square(our_attacks, sq));
                sp = hcetables.threatened_by_knight_penalty[threatened_piece.u][is_defended];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "{t} threatened by knight, is_defended {}", .{ threatened_piece.e, is_defended });
            }

            // Their bishop threats.
            bb = self.bishop_attacks[them.u] & our_pieces;
            while (bb != 0) {
                const sq: Square = pop(&bb);
                const threatened_piece = pos.board[sq.u].piecetype();
                const is_defended: u1 = @intFromBool(funcs.contains_square(our_attacks, sq));
                sp = hcetables.threatened_by_bishop_penalty[threatened_piece.u][is_defended];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "{t} threatened by bishop, is_defended {}", .{ threatened_piece.e, is_defended });
            }

            // Their rook threats.
            bb = self.rook_attacks[them.u] & our_pieces;
            while (bb != 0) {
                const sq: Square = pop(&bb);
                const threatened_piece = pos.board[sq.u].piecetype();
                const is_defended: u1 = @intFromBool(funcs.contains_square(our_attacks, sq));
                sp = hcetables.threatened_by_rook_penalty[threatened_piece.u][is_defended];
                score.inc(sp);
                if (debug_print_details) trace(sp, us, sq, "{t} threatened by rook, is_defended {}", .{ threatened_piece.e, is_defended });
            }

            // Pawn push threats.
            // Get the squares defended by the enemy, excluding squares that are defended by our pawns and not attacked by their pawns
            if (has_pawns) {
                const their_piece_attacks: u64 = self.knight_attacks[them.u] | self.bishop_attacks[them.u] | self.rook_attacks[them.u] | self.queen_attacks[them.u];
                const enemy_protected_squares: u64 = self.pawn_attacks[them.u] | (their_piece_attacks & ~self.pawn_attacks[us.u]);
                const safe_pawn_pushes: u64 = funcs.pawns_shift(pos.pawns(us), us, .up) & ~pos.all() & ~enemy_protected_squares;
                // Which enemy piece would be attacked if we push our pawn.
                var pawn_push_threats: u64 = (funcs.pawns_shift(safe_pawn_pushes, us, .northwest) | funcs.pawns_shift(safe_pawn_pushes, us, .northeast)) & pos.by_color(them) & ~pos.pawns(them);
                while (pawn_push_threats != 0) {
                    const sq: Square = pop(&pawn_push_threats);
                    if (comptime lib.is_paranoid) assert(!pos.board[sq.u].is_empty() and pos.board[sq.u].color().e == them.e);
                    const threatened: Piece = pos.board[sq.u];
                    sp = hcetables.pawn_push_threat_table[threatened.u];
                    score.inc(sp);
                    if (debug_print_details and threatened.e != .no_piece) trace(sp, us, sq, "pawn push threat {t}", .{ threatened.e });
                }
            }

            // Get the squares that our pieces can reach to place the enemy king in check
            const occupied: u64 = pos.all();
            const not_us: u64 = ~pos.by_color(us);
            const their_king_sq: Square = self.king_squares[them.u];
            const rook_checks: u64 = attacks.get_rook_attacks(their_king_sq, occupied);
            const bishop_checks: u64 = attacks.get_bishop_attacks(their_king_sq, occupied);
            const safe: u64 = ~(self.pawn_attacks[them.u] | self.knight_attacks[them.u] | self.bishop_attacks[them.u] | self.rook_attacks[them.u] | attacks.get_king_attacks(their_king_sq)); // TODO: use their_piece_attacks.

            const safe_knight_checks: u64 = not_us & safe & self.knight_attacks[us.u] & attacks.get_knight_attacks(their_king_sq);
            const safe_bishop_checks: u64 = not_us & safe & self.bishop_attacks[us.u] & bishop_checks;
            const safe_rook_checks: u64 = not_us & safe & self.rook_attacks[us.u] & rook_checks;
            const safe_queen_checks: u64 = not_us & safe & self.queen_attacks[us.u] & (rook_checks | bishop_checks);

            // This was a bug.
            // const safe_knight_checks: u64 = safe & self.knight_attacks[us.u] & attacks.get_knight_attacks(their_king_sq);
            // const safe_bishop_checks: u64 = safe & self.bishop_attacks[us.u] & bishop_checks;
            // const safe_rook_checks: u64 = safe & self.rook_attacks[us.u] & rook_checks;
            // const safe_queen_checks: u64 = safe & self.queen_attacks[us.u] & (rook_checks | bishop_checks);

            sp = hcetables.safe_check_bonus[PieceType.KNIGHT.u].mul(popcnt(safe_knight_checks));
            score.inc(sp);
            if (debug_print_details) trace(sp, us, null, "safe knight checks popcnt {}", .{popcnt(safe_knight_checks)});

            sp = hcetables.safe_check_bonus[PieceType.BISHOP.u].mul(popcnt(safe_bishop_checks));
            score.inc(sp);
            if (debug_print_details) trace(sp, us, null, "safe bishop checks popcnt {}", .{popcnt(safe_bishop_checks)});

            sp = hcetables.safe_check_bonus[PieceType.ROOK.u].mul(popcnt(safe_rook_checks));
            score.inc(sp);
            if (debug_print_details) trace(sp, us, null, "safe rook checks popcnt {}", .{popcnt(safe_rook_checks)});

            sp = hcetables.safe_check_bonus[PieceType.QUEEN.u].mul(popcnt(safe_queen_checks));
            score.inc(sp);
            if (debug_print_details) trace(sp, us, null, "safe queen checks popcnt {}", .{popcnt(safe_queen_checks)});

            return score;
        }

        /// Returns true if the square is not attacked by their pawn and the square is protected by our pawn.
        fn is_outpost(self: *Self, sq: Square, comptime us: Color) bool {
            const them: Color = comptime us.opp();
            const sq_bb: u64 = sq.to_bitboard();
            const safe: bool = self.pawn_attacks[us.u] & sq_bb != 0 and self.pawn_attacks[them.u] & sq_bb == 0;
            return safe and funcs.outpost(us) & sq_bb != 0;
        }

        fn get_pawn_protection_scorepair(comptime us: Color, our_king_sq: Square, our_pawn_sq: Square) ScorePair {
            // TODO: assert
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
            return hcetables.pawn_protection_table[i];
        }

        fn get_pawn_storm_scorepair(comptime us: Color, their_king_sq: Square, our_pawn_sq: Square) ScorePair {
            // TODO: assert
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
            return hcetables.pawn_storm_table[i];
        }

        /// TODO: Certainly not legal. Only heuristic. We need all pins? Now we use only the pins of the side to move.
        fn legalize_moves1(self: *Self, comptime pt: PieceType, comptime us: Color, from: Square, bb_moves: u64) u64 {

            const from_bb: u64 = from.to_bitboard();
            if (pt.e == .knight and self.pos.pins & from_bb & self.pos.pins != 0) {
                return 0;
            }
            _ = us;
            return bb_moves;
        }


        fn legalize_moves(self: *Self, comptime pt: PieceType, comptime us: Color, from: Square, bb_moves: u64) u64 {
            //_ = us;
            const from_bb: u64 = from.to_bitboard();
            if (self.pos.pins & from_bb == 0) {
                return bb_moves;
            }
            if (pt.e == .knight) {
                return 0;
            }
            // else {
            //     return bb_moves & bitboards.get_squarepair(self.king_squares[us.u], from).ray;
            // }
            _ = us;
            return bb_moves;
        }

        pub fn sliding_score(pos: *const Position, score: ScorePair) Value {
            const max: u8 = comptime types.max_phase;
            const phase: u8 = @min(max, pos.phase);
            const mg: Value = score.mg;
            const eg: Value = score.eg;
            return @divTrunc(mg * phase + eg * (max - phase), max);        }

        fn trace(sp: ScorePair, us: Color, sq: ?Square, comptime msg: []const u8, args: anytype) void {
            //const avg
            if (sq) |q| {
                io.debugprint("{t} {t} mg {d:>4} eg {d:>4} ", .{ us.e, q.e, sp.mg, sp.eg });
            }
            else {
                io.debugprint("{t}    mg {d:>4} eg {d:>4} ", .{ us.e, sp.mg, sp.eg });
            }

            io.debugprint(msg, args);
            io.debugprint("\n", .{});
        }
    };
}

/// Static exchange evaluation. Get score of capture fest on square.
/// TODO: perfect pins?
/// TODO: threshold.
pub fn see_score(pos: *const Position, m: Move) Value {
    //const st: *const StateInfo = pos.state;gi
    if (comptime lib.is_paranoid) {
        assert(m.is_capture());
    }

    // This will hit a no_piece square.
    if (m.is_ep()) {
        return 0;
    }

    const from: Square = m.from;
    const to: Square = m.to;
    const value_them = pos.get(to).value();
    const value_us = pos.get(from).value();

    // if (m.is_promotion()) {
    //     value_them += m.promoted_to().value(); // #testing
    // }

    // good capture: if (value_them - value_us > P.value()) return true;
    var gain: [24]Value = @splat(0);
    gain[0] = value_them;
    gain[1] = value_us - value_them;

    var depth: u8 = 1;
    const queens_bishops = pos.all_queens_bishops();
    const queens_rooks = pos.all_queens_rooks();
    var occupation = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();
    var attackers: u64 = pos.get_combined_attacks_to_for_occupation(occupation, to);
    var side: Color = pos.stm;

    // Reusable vars.
    var bb: u64 = undefined;

    attackloop: while (true) {
        attackers &= occupation;
        if (attackers == 0) break;
        side = side.opp();

        // Pawn.
        bb = attackers & pos.pawns(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = PieceType.PAWN.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 pawn.
            attackers |= (attacks.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            continue :attackloop;
        }

        // Knight.
        bb = attackers & pos.knights(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = PieceType.KNIGHT.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 knight (cannot reveal more sliding attackers).
            continue :attackloop;
        }

        // Bishop.
        bb = attackers & pos.bishops(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = PieceType.BISHOP.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 bishop.
            attackers |= (attacks.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            continue :attackloop;
        }

        // Rook.
        bb = attackers & pos.rooks(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = PieceType.ROOK.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 rook.
            attackers |= (attacks.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            continue :attackloop;
        }

        // Queen.
        bb = attackers & pos.queens(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = PieceType.QUEEN.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 queen.
            attackers |= (attacks.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            attackers |= (attacks.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            continue :attackloop;
        }

        // King.
        bb = attackers & pos.kings(side);
        if (bb != 0) {
            side = side.opp();
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 king.
            attackers |= (attacks.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            attackers |= (attacks.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            attackers &= occupation & pos.by_color(side);
            // King can take the next.
            if (attackers == 0) {
                depth += 1;
                gain[depth] = PieceType.KING.value() - gain[depth - 1];
                break :attackloop;
            }
            break :attackloop;
        }
        break :attackloop;
    }

    // Bubble up the score
    depth -= 1;
    while (depth > 0) : (depth -= 1) {
        if (gain[depth] > -gain[depth - 1]) {
            gain[depth - 1] = -gain[depth];
        }
    }
    return gain[0];
}


pub fn see(pos: *const Position, m: Move, threshold: Value) bool {
    if (m.flags & (Move.castle_short | Move.castle_long | Move.ep) != 0) {
        return 0;
    }

    const from: Square = m.from;
    const to: Square = m.to;

    // Set the score to how much we are allowed to lose.
    var score: Value = pos.board[to.u].value() - threshold;

    // TODO: promotion

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

    // Get all attacks from both sides to square.
    var all_attackers: u64 = pos.get_combined_attacks_to_for_occupation(occupied, to);
    var us: Color = pos.stm;
    var winner: Color = pos.stm;

    attackloop: while (true) {
        us = us.opp();
        all_attackers &= occupied;
        if (all_attackers == 0) {
            break :attackloop;
        }
        winner = winner.opp();

        var next_attacker_value: Value = 0;

        find_next_attacker: inline for (PieceType.all) |piecetype|{
            const next_attacker: u64 = all_attackers & pos.by_type(piecetype) & pos.by_color(us);
            if (next_attacker == 0) {
                break :attackloop;
            }

            // Clear this attacker.
            funcs.clear_square(&occupied, funcs.first_square(next_attacker));
            next_attacker_value = piecetype.value();

            // Find Reveal next x-ray attacker on the bitboard.
            switch (piecetype.e) {
                .pawn => {
                    all_attackers |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
                    break :find_next_attacker;
                },
                .knight => {
                    // Do nothing. A knight cannot reveal new sliders.
                    break :find_next_attacker;
                },
                .bishop => {
                    all_attackers |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
                    break :find_next_attacker;
                },
                .rook => {
                    all_attackers |= (attacks.get_bishop_attacks(to, occupied) & queens_rooks);
                    break :find_next_attacker;
                },
                .queen => {
                    all_attackers |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
                    all_attackers |= (attacks.get_rook_attacks(to, occupied) & queens_rooks);
                    break :find_next_attacker;
                },
                .king => {
                    const them: Color = us.opp();
                    // We can exit here: if the king captures and the opponent can then capture our king we lose othersize we win.
                    return if (all_attackers & pos.by_color(them) != 0) pos.stm.u != winner.u else return pos.stm.u == winner;
                },
            }
        }

        score = -score + next_attacker_value;
        // Quit early if the exchange is lost or neutral
        if (score <= 0) {
            break :attackloop;
        }
    }

    return pos.stm.u == winner.u;
}
