// zig fmt: off

//! --- NOT USED YET ---

const std = @import("std");

const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const squarepairs = @import("squarepairs.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const data = @import("data.zig");
const masks = @import("masks.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");

const assert = std.debug.assert;
const io = lib.io;
const wtf = lib.wtf;
const float = funcs.float;
const int = funcs.int;
const pop_square = funcs.pop_square;
const popcnt = funcs.popcnt;
const popcnt_v = funcs.popcnt_v;

const Value = types.Value;
const Float = types.Float;

const Color = types.Color;
const Piece = types.Piece;
const PieceType = types.PieceType;
const Square = types.Square;
const Move = types.Move;
const GamePhase = types.GamePhase;
const StateInfo = position.StateInfo;
const Position = position.Position;

const P: PieceType = types.P;
const N: PieceType = types.N;
const B: PieceType = types.B;
const R: PieceType = types.R;
const Q: PieceType = types.Q;
const K: PieceType = types.K;

/// Returns the evaluation from white perspective.
pub fn evaluate_abs(pos: *const Position, comptime tracking: bool) Value {
    const v = evaluate(pos, tracking);
    return if (pos.to_move.e == .white) v else -v;
}

pub fn evaluate(pos: *const Position, comptime tracking: bool) Value {
    const phase: GamePhase = pos.phase();
    switch (phase) {
        .Opening => return eval(pos, .Opening, tracking),
        .Midgame => return eval(pos, .Midgame, tracking),
        .Endgame => return eval(pos, .Endgame, tracking),
    }
    return evaluate(pos, tracking);
}

pub fn simple_eval(pos: *const Position, comptime us: Color) Value {
    return switch (us.e) {
        .white => pos.values[Color.WHITE.u] - pos.values[Color.BLACK.u],
        .black => pos.values[Color.BLACK.u] - pos.values[Color.WHITE.u],
    };
}

fn add_to_score(score: *Value, delta: Value, comptime color: Color, comptime pt: PieceType, comptime feature: Feature, comptime track: bool, sq: ?Square) void
{
    if (delta == 0) return;
    score.* += delta;
    if (track) print_add_to_score(delta, color, pt, feature, sq);
}

fn print_add_to_score(delta: Value, comptime color: Color, comptime pt: PieceType, comptime feature: Feature, sq: ?Square) void
{
    const square_str: []const u8 = if (sq) |q| q.to_string() else "  ";
    const piece_str: []const u8 = if (pt.e != .no_piecetype) @tagName(pt.e) else "";
    //lib.io.debugprint("{t}: {s}  {t:<9} {t:<32} {d}\n", .{ piece.color().e, s, piece.e, feature, delta });

    lib.io.debugprint("{t} {s:<9} {s:<3} {t:<32} {d}\n", .{color.e, piece_str, square_str, feature, delta });
}


/// Evaluation from the perspective of the side to move.
fn eval(pos: *const Position, comptime phase: GamePhase, comptime tracking: bool) Value {
    // TODO: we can accumulate the attacks and use for the kings at the end.
    // moves = get moves
    // mobility = moves & ~bb_us
    // attacks = moves & bb_them

    const check: bool = pos.state.checkers != 0;
    const non_pawn_material: Value = pos.non_pawn_material();
    const bb_all = pos.all();
    const negate: bool = pos.to_move.e == .black;

    // EXPERIMENTAL: pawn up in pawn endgame is often desastrous.
    const pawns_multiply: Value = if (non_pawn_material == 0) 2 else 1;

    const material_scores: [2]Value = .{ pos.values[Color.WHITE.u] * pawns_multiply, pos.values[Color.BLACK.u] * pawns_multiply };

    // // TODO: not sure.
    // // Exit early.
    // const simple_score: Value = if (!negate) material_scores[0] - material_scores[1] else material_scores[1] - material_scores[0];
    // if (@abs(simple_score) >= 900) return simple_score;

    // Sliding values of the Pesto tables, which we taper at the end.
    // var pesto_phase: Value = 0;
    var pesto_scores_mg: [2]Value = .{ 0, 0 };
    var pesto_scores_eg: [2]Value = .{ 0, 0 };

    // Scores
    var p_scores: [2]Value = .{ 0, 0 };
    var n_scores: [2]Value = .{ 0, 0 };
    var b_scores: [2]Value = .{ 0, 0 };
    var r_scores: [2]Value = .{ 0, 0 };
    var q_scores: [2]Value = .{ 0, 0 };
    var k_scores: [2]Value = .{ 0, 0 };
    var threats_scores: [2]Value = .{ 0, 0 };
    var king_threat_scores: [2]Value = .{ 0, 0 };
    var space_scores: [2]Value = .{ 0, 0 };
    var mobility_scores: [2]Value = .{ 0, 0 };

    ////////////////////////////////////////////////////////////////
    // Color loop.
    ////////////////////////////////////////////////////////////////
    inline for (Color.all) |us| {
        const add = add_to_score;

        const them = comptime us.opp();
        const bb_us = pos.by_color(us);
        const bb_them = pos.by_color(them);
        const our_king_sq = pos.king_square(us);
        const their_king_sq = pos.king_square(them);
        const our_king_area = masks.get_king_area(our_king_sq);
        const their_king_area = masks.get_king_area(their_king_sq);

        const p_score: *Value = &p_scores[us.u];
        const n_score: *Value = &n_scores[us.u];
        const b_score: *Value = &b_scores[us.u];
        const r_score: *Value = &r_scores[us.u];
        const q_score: *Value = &q_scores[us.u];
        const k_score: *Value = &k_scores[us.u];

        const pesto_score_mg: *Value = &pesto_scores_mg[us.u];
        const pesto_score_eg: *Value = &pesto_scores_eg[us.u];

        const mobility: *Value = &mobility_scores[us.u];
        const space: *Value = &space_scores[us.u];
        const threat: *Value = &threats_scores[us.u];
        const king_threat: *Value = &king_threat_scores[us.u];

        // Naive general space heuristic.
        if (phase != .Endgame) {
            var bits: Value = 0;
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_3));
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_4));
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_5)) * 2;
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_6)) * 3;
            bits += popcnt(bb_us & funcs.relative_rank_bb(us, bitboards.rank_7)) * 2;
            add(space, bits, us, PieceType.NO_PIECETYPE, .space, tracking, null);
        }

        ////////////////////////////////////////////////////////////////
        // Piece loop.
        ////////////////////////////////////////////////////////////////
        inline for (PieceType.all) |piecetype| {
            //const piece: Piece = comptime Piece.make(piecetype, us);
            var is_first_piece: bool = true;
            const bb_current = pos.pieces(piecetype, us);
            var bb_running = bb_current;

            ////////////////////////////////////////////////////////////////
            // Square loop.
            ////////////////////////////////////////////////////////////////
            while (bb_running != 0) : (is_first_piece = false) {
                const sq: Square = funcs.pop_square(&bb_running);
                const bb_this_piece = bb_current & sq.to_bitboard();
                const file: u3 = sq.file();
                const rank: u3 = sq.rank();
                const relative_rank: u3 = funcs.relative_rank(us, rank);

                // Keep track of Pesto table scores.
                const pair = Tables.get_scorepair(us, piecetype, sq);
                add(pesto_score_mg, pair.mg, us, piecetype, .pesto_mg, false, sq);
                add(pesto_score_eg, pair.eg, us, piecetype, .pesto_eg, false, sq);

                // NOTE: Keep this int-based pesto around for comparison
                //pesto_phase += Tables.pesto_game_phase[piecetype.u];

                switch (piecetype.e) {
                    .pawn => {
                        const our_pawns: u64 = bb_current;
                        const pawn_attacks: u64 = data.get_pawn_attacks(sq, us);
                        const pawn_moves: u64 = funcs.pawns_shift(bb_this_piece, us, funcs.PawnShift.up);

                        // Opening piece development
                        if (is_first_piece and phase == .Opening) {
                            const to_develop: u64 = bb_us & ~our_pawns & ~pos.kings(us);
                            const ranks: u64 = funcs.relative_rank_bb(us, bitboards.rank_2) | funcs.relative_rank_bb(us, bitboards.rank_3) | funcs.relative_rank_bb(us, bitboards.rank_4) | funcs.relative_rank_bb(us, bitboards.rank_5);
                            const developed: u64 = to_develop & ranks;
                            add(mobility, popcnt_v(developed) * 2, us, PieceType.NO_PIECETYPE, .experimental, tracking, null);
                        }

                        // Threat.
                        var att: u64 = pawn_attacks & bb_them;
                        while (att != 0 ) {
                            const to: Square = pop_square(&att);
                            const theirs: PieceType = pos.get(to).piecetype;
                            add(threat, get_threat(piecetype, theirs), us, piecetype, .threat, tracking, sq);
                        }

                        // Pawn center
                        if (is_first_piece and phase != .Endgame) {
                            const ctrl: u64 = our_pawns & bitboards.bb_center;// (bitboards.bb_rank_4 | bitboards.bb_rank_5);
                            add(p_score, popcnt_v(ctrl) * 20, us, piecetype, .control, tracking, sq);
                        }

                        // King area threat.
                        const att_king_area: u64 = (pawn_attacks | pawn_moves) & their_king_area;
                        add(king_threat, get_king_threat(piecetype, att_king_area), us, piecetype, .threat, tracking, sq);

                        // Reward passed pawn.
                        if (is_passed_pawn(pos, us, sq)) {
                            const e: Value = Tables.passed_pawn_by_rank_scores[relative_rank];
                            add(p_score, e, us, piecetype, .passed_pawn, tracking, sq);
                        }

                        // Punish doubled pawn.
                        const bb_doubled = bitboards.file_bitboards[file] & our_pawns;
                        if (popcnt(bb_doubled) > 1) {
                            add(p_score, Tables.doubled_pawn, us, piecetype, .doubled_pawn, tracking, sq);
                        }

                        // Isolated pawn.
                        const bb_isolated = masks.get_isolated_pawn_mask(sq) & our_pawns;
                        if (bb_isolated == 0) {
                            add(p_score, Tables.isolated_pawn, us, piecetype, .isolated_pawn, tracking, sq);
                        }
                        // Connected pawn.
                    },
                    .knight => {
                        const knight_attacks: u64 = data.get_knight_attacks(sq);

                        // Threat.
                        var att: u64 = knight_attacks & bb_them;
                        while (att != 0 ) {
                            const to: Square = pop_square(&att);
                            const theirs: PieceType = pos.get(to).piecetype;
                            add(threat, get_threat(piecetype, theirs), us, piecetype, .threat, tracking, sq);
                        }

                        // King area threat.
                        const att_king_area: u64 = knight_attacks & their_king_area;
                        add(king_threat, get_king_threat(piecetype, att_king_area), us, piecetype, .king_threat, tracking, sq);

                        // Reward knight is supported by a pawn.
                        if (is_supported_by_pawn(pos, us, sq)) {
                            add(n_score, Tables.supported_by_pawn, us, piecetype, .supported_by_pawn, tracking, sq);
                        }

                        // Knight mobility
                        if (!is_pinned(pos, us, sq)) {
                            const mob: u64 = knight_attacks & ~bb_us;
                            add(mobility, popcnt(mob), us, piecetype, .mobility, tracking, sq);
                        }
                        else {
                            add(mobility, -5, us, piecetype, .mobility, tracking, sq);
                        }
                    },
                    .bishop => {
                        const our_bishops: u64 = bb_current;
                        const bishop_moves: u64 = data.get_bishop_attacks(sq, bb_all);

                        // Threats.
                        var att: u64 = bishop_moves & bb_them;
                        while (att != 0 ) {
                            const to: Square = pop_square(&att);
                            const theirs: PieceType = pos.get(to).piecetype;
                            add(threat, get_threat(piecetype, theirs), us, piecetype, .threat, tracking, sq);
                        }

                        // King area threats.
                        const att_king_area: u64 = bishop_moves & their_king_area;
                        add(king_threat, get_king_threat(piecetype, att_king_area), us, piecetype, .king_threat, tracking, sq);

                        // Bishop pair.
                        if (is_first_piece and our_bishops & bitboards.bb_black_squares & bitboards.bb_white_squares != 0) {
                            add(b_score, Tables.bishop_pair, us, piecetype, .bishop_pair, tracking, null);
                        }

                        // Bishop mobility.
                        if (!is_pinned(pos, us, sq)) {
                            const mob: u64 = bishop_moves & ~bb_us;
                            add(mobility, popcnt(mob), us, piecetype, .mobility, tracking, sq);
                        }
                        else if (is_pinned_orthogonally(pos, us, sq)) {
                            add(mobility, -10, us, piecetype, .mobility, tracking, sq);
                        }
                    },
                    .rook => {
                        const our_rooks: u64 = bb_current;
                        const rook_moves: u64 = data.get_rook_attacks(sq, bb_all);

                        // Threat.
                        var att: u64 = rook_moves & bb_them;
                        while (att != 0 ) {
                            const to: Square = pop_square(&att);
                            const theirs: PieceType = pos.get(to).piecetype;
                            add(threat, get_threat(piecetype, theirs), us, piecetype, .threat, tracking, sq);
                        }

                        // King area threat.
                        const att_king_area: u64 = rook_moves & their_king_area;
                        add(king_threat, get_king_threat(piecetype, att_king_area), us, piecetype, .king_threat, tracking, sq);

                        // Pigs on 7th rank if there are also enemy pawns or the king is cutoff on the 8th rank.
                        if (rank == funcs.relative_rank_7(us)) {
                            const their_pawns_on_7th = pos.pawns(them) & funcs.relative_rank_7_bitboard(them);
                            if (their_pawns_on_7th != 0) {
                                const e = Tables.rook_on_seventh;
                                add(r_score, e, us, piecetype, .rook_on_seventh, tracking, sq);
                            }
                        }

                        // Rook open or halfopen file
                        if (bitboards.file_bitboards[file] & pos.all_pawns() == 0) {
                            add(r_score, 10, us, piecetype, .rook_on_open_file, tracking, sq);
                        }
                        else if (bitboards.file_bitboards[file] & pos.pawns(us) == 0) {
                            add(r_score, 5, us, piecetype, .rook_on_half_open_file, tracking, sq);
                        }

                        // Rooks connected.
                        if (is_first_piece and popcnt(rook_moves & our_rooks) > 0) {
                            const e = Tables.rooks_connected;
                            add(r_score, e, us, piecetype, .rooks_connected, tracking, null);
                        }

                        // Rook mobility
                        if (!is_pinned(pos, us, sq)) {
                            add(mobility, popcnt(rook_moves & ~bb_us), us, piecetype, .mobility, tracking, sq);
                        }
                        else if (is_pinned_diagonally(pos, us, sq)) {
                            add(mobility, -10, us, piecetype, .mobility, tracking, sq);
                        }
                    },
                    .queen => {
                        //const our_queens: u64 = bb_current;
                        const queen_moves: u64 = data.get_queen_attacks(sq, bb_all);

                        // Threats.
                        var att: u64 = queen_moves & bb_them;
                        while (att != 0 ) {
                            const to: Square = pop_square(&att);
                            const theirs: PieceType = pos.get(to).piecetype;
                            add(threat, get_threat(piecetype, theirs), us, piecetype, .threat, tracking, sq);
                        }

                        // King area threats.
                        const att_king_area: u64 = queen_moves & their_king_area;
                        add(king_threat, get_king_threat(piecetype, att_king_area), us, piecetype, .king_threat, tracking, sq);


                        // TODO:
                        q_score.* += 0;

                        // Queen mobility
                        if (!is_pinned(pos, us, sq)) {
                            add(mobility, popcnt(queen_moves & ~bb_us), us, piecetype, .mobility, tracking, sq);
                        }
                        else if (is_pinned(pos, us, sq)) {
                            add(mobility, -5, us, piecetype, .mobility, tracking, sq);
                        }
                    },
                    .king => {
                        if (check and us.e == pos.to_move.e) {
                            add(k_score, Tables.in_check, us, piecetype, .in_check, tracking, sq);
                        }

                        // Open files to our king
                        if (phase != .Endgame) {
                            const open_files: u64 = data.get_file_attacks(our_king_sq, bb_all) & ~bb_us;
                            add(k_score, -popcnt_v(open_files) * 4, us, piecetype, .open_file_to_our_king, tracking, sq);
                        }

                        // Open diagonals to our king
                        if (phase != .Endgame) {
                            const open_files: u64 = data.get_bishop_attacks(our_king_sq, bb_all) & ~bb_us;
                            add(k_score, -popcnt_v(open_files) * 4, us, piecetype, .open_diagonal_to_our_king, tracking, sq);
                        }

                        // King pawn protection
                        const pawn_protection = our_king_area & pos.pawns(us);
                        add(k_score, popcnt(pawn_protection) * 12, us, piecetype, .king_protection_by_pawns, tracking, sq);

                        // King piece protection.
                        if (phase != .Endgame) {
                            const piece_protection = our_king_area & pos.non_pawns(us);
                            add(k_score, popcnt(piece_protection) * 8, us, piecetype, .king_protection_by_pieces, tracking, sq);
                            const knight_protection = our_king_area & pos.knights(us);
                            add(k_score, popcnt(knight_protection) * 4, us, piecetype, .king_protection_by_pieces, tracking, sq);
                        }

                        // EXPERIMENTAL
                        // if (phase == .Endgame and pos.all_pawns() == 0) {
                        //     const d: Value = @as(Value, 8) - squarepairs.get(our_king_sq, their_king_sq).distance;
                        //     if (pos.values[us.u] > pos.values[them.u])
                        //         add(&king_score, d * 10, piece, .king_pushing_king, tracking, sq)
                        //     else
                        //         add(&king_score, d * -10, piece, .king_pushing_king, tracking, sq);
                        // }
                    },
                    else => {
                        unreachable;
                    },
                }
            }
        }
    }

    // NOTE: Keep this int-based pesto around for comparison. This gives the same result as sliding score.
    // const mg_score = pesto_scores_mg[0] - pesto_scores_mg[1];
    // const eg_score = pesto_scores_eg[0] - pesto_scores_eg[1];
    // var mg_phase = pesto_phase;
    // if (mg_phase > 24) mg_phase = 24; // in case of early promotion
    // const eg_phase = 24 - mg_phase;
    // const tapered: Value = @divTrunc(mg_score * mg_phase + eg_score * eg_phase, 24);

    // Taper sliding values.
    const w_pesto: Value = Tables.sliding_score(non_pawn_material, pesto_scores_mg[0], pesto_scores_eg[0]);
    const b_pesto: Value = Tables.sliding_score(non_pawn_material, pesto_scores_mg[1], pesto_scores_eg[1]);
    //const to_move_score: Value = 12;

   //const tm: Value = if (pos.to_move.e == .white) 12 else 0;

    const w_score: Value =
        material_scores[0] + w_pesto + p_scores[0] + n_scores[0] + b_scores[0] + r_scores[0] + q_scores[0] + k_scores[0] + mobility_scores[0] + space_scores[0] + threats_scores[0] + king_threat_scores[0];

    const b_score: Value =
        material_scores[1] + b_pesto + p_scores[1] + n_scores[1] + b_scores[1] + r_scores[1] + q_scores[1] + k_scores[1] + mobility_scores[1] + space_scores[1] + threats_scores[1] + king_threat_scores[1];

    var result = if (negate) b_score - w_score else w_score - b_score;

    // TODO: not sure.
    result = scale_towards_draw(result, pos.state.rule50);


    if (tracking) {
        lib.io.debugprint(
            \\perspective : {t}
            \\material    : {} {}
            \\pawn        : {} {}
            \\knight      : {} {}
            \\bishop      : {} {}
            \\rook        : {} {}
            \\queen       : {} {}
            \\king        : {} {}
            \\pestos      : {} {}
            \\mobility    : {} {}
            \\space       : {} {}
            \\threat      : {} {}
            \\king_threat : {} {}
            \\result      : {}
            \\
            ,
            .{
                pos.to_move.e,
                material_scores[0], material_scores[1],
                p_scores[0], p_scores[1],
                n_scores[0], n_scores[1],
                b_scores[0], b_scores[1],
                r_scores[0], r_scores[1],
                q_scores[0], q_scores[1],
                k_scores[0], k_scores[1],
                w_pesto, b_pesto,
                mobility_scores[0], mobility_scores[1],
                space_scores[0], space_scores[1],
                threats_scores[0], threats_scores[1],
                king_threat_scores[0], king_threat_scores[1],
                result,
            }
        );
    }
    return result;
}

pub fn scale_towards_draw(score: Value, drawcounter: u16) Value {
    if (drawcounter <= 2) return score;
    // drawcounter ranges 0..100
    const s: f32 = @floatFromInt(score);
    const d: f32 = @floatFromInt(drawcounter);
    const factor: f32 = (100.0 - d) / 100.0;
    return @intFromFloat(s * factor);
}

pub fn is_trivial_win(pos: *const Position) bool {
    _ = pos;
    return false;
}

/// Detect forced draw.
pub fn is_draw_by_insufficient_material(pos: *const Position) bool
{
    // Only kings left.
    if (pos.non_pawn_material() == 0) return true;
    const w: Value = pos.materials[0];
    const b: Value = pos.materials[1];
    if (w + b <= types.material_bishop) return true;
    if (w + b == types.material_knight * 2) return true;
    return false;
}

fn get_threat(ours: PieceType, theirs: PieceType) Value {
    // _ = ours;
    //_ = theirs;
    if (lib.is_paranoid) assert(ours.u > 0 and theirs.u > 0);
    const diff: Value = Tables.threat_values[theirs.u] - Tables.threat_values[ours.u];
    return if (diff < 0) 0 else diff; // #EL 0 was 2
}

fn get_king_threat(attacking_pt: PieceType, bb_squares: u64) Value {
    const mult: Value = switch (attacking_pt.e) {
        .no_piecetype => unreachable,
        .pawn => 4,
        .knight => 4,
        .bishop => 3,
        .rook => 2,
        .queen => 2,
        .king => 0,
    };
    return popcnt_v(bb_squares) * mult;
}

/// Static exchange evaluation. Quickly decide if a (capture) move is good or bad.
/// TODO: pins.
/// TODO: threshold.
pub fn see_score(pos: *const Position, m: Move) Value
{
    const from: Square = m.from;
    const to: Square = m.to;
    const value_them = pos.get(to).value();
    const value_us = pos.get(from).value();
    // This is a good capture. For example pawn takes knight.
    //if (value_them - value_us > P.value()) return true;
    var gain: [24]Value = @splat(0);
    gain[0] = value_them;
    gain[1] = value_us - value_them;

    var depth: u8 = 1;
    const queens_bishops = pos.all_queens_bishops();
    const queens_rooks = pos.all_queens_rooks();
    var occupation = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();
    var attackers: u64 = pos.get_all_attacks_to_for_occupation(occupation, to);
    var bb: u64 = 0;
    var side: Color = pos.to_move;
    attackloop: while (true) {
        attackers &= occupation;
        if (attackers == 0) break;
        side = side.opp();

        // Pawn.
        bb = attackers & pos.pawns(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = P.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 pawn.
            attackers |= (data.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            continue :attackloop;
        }

        // Knight.
        bb = attackers & pos.knights(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = N.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 knight (cannot reveal more sliding attackers).
            continue :attackloop;
        }

        // Bishop.
        bb = attackers & pos.bishops(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = B.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 bishop.
            attackers |= (data.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            continue :attackloop;
        }

        // Rook.
        bb = attackers & pos.rooks(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = R.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 rook.
            attackers |= (data.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            continue :attackloop;
        }

        // Queen.
        bb = attackers & pos.queens(side);
        if (bb != 0) {
            depth += 1;
            gain[depth] = Q.value() - gain[depth - 1];
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 queen.
            attackers |= (data.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            attackers |= (data.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            continue :attackloop;
        }

        bb = attackers & pos.kings(side);
        if (bb != 0) {
            side = side.opp();
            funcs.clear_square(&occupation, funcs.first_square(bb)); // clear 1 king.
            attackers |= (data.get_bishop_attacks(to, occupation) & queens_bishops); // reveal next diagonal attacker.
            attackers |= (data.get_rook_attacks(to, occupation) & queens_rooks); // reveal next straight attacker.
            attackers &= occupation & pos.by_color(side);
            // King can take the next.
            if (attackers == 0) {
                depth += 1;
                gain[depth] = K.value() - gain[depth - 1];
                break :attackloop;
            }
            break :attackloop;
        }
        //unreachable;
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

/// TODO: add threshold.
pub fn see(pos: *const Position, m: Move) bool {
    if (m.type == .promotion) return true;

    const from: Square = m.from;
    const to: Square = m.to;

    const victim: Value = pos.get(to).value();
    const attacker: Value = pos.get(from).value();

    // Fast delta: clearly winning trades (e.g., PxN)
    if (victim - attacker > P.value()) return true;

    var occ = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();
    var atks: u64 = pos.get_all_attacks_to_for_occupation(occ, to);
    const qb = pos.all_queens_bishops();
    const qr = pos.all_queens_rooks();

    var side: Color = pos.to_move;
    var bb: u64 = 0;

    // balance = margin we must preserve to stay >= 0 (threshold = 0)
    var balance: Value = victim - attacker;

    while (true) {
        atks &= occ;
        if (atks == 0) break;
        side = side.opp(); // next capturer

        // Pawn
        bb = atks & pos.pawns(side);
        if (bb != 0) {
            balance = -balance - P.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= data.get_bishop_attacks(to, occ) & qb;
            continue;
        }

        // Knight
        bb = atks & pos.knights(side);
        if (bb != 0) {
            balance = -balance - N.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            continue;
        }

        // Bishop
        bb = atks & pos.bishops(side);
        if (bb != 0) {
            balance = -balance - B.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= data.get_bishop_attacks(to, occ) & qb;
            continue;
        }

        // Rook
        bb = atks & pos.rooks(side);
        if (bb != 0) {
            balance = -balance - R.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= data.get_rook_attacks(to, occ) & qr;
            continue;
        }

        // Queen
        bb = atks & pos.queens(side);
        if (bb != 0) {
            balance = -balance - Q.value();
            if (balance < 0) return (side.e != pos.to_move.e);
            funcs.clear_square(&occ, funcs.first_square(bb));
            atks |= (data.get_bishop_attacks(to, occ) & qb);
            atks |= (data.get_rook_attacks(to, occ) & qr);
            continue;
        }

        // King ends the sequence
        if ((atks & pos.kings(side)) != 0) break; // TODO: BUGGY.
        break;
    }

    // No early fail: if the last side to *have the move* is the opponent, our capture stands.
    return (side.e != pos.to_move.e);
}

pub fn is_supported_by_pawn(pos: *const Position, comptime us: Color, sq: Square) bool {
    const them = comptime us.opp();
    return data.get_pawn_attacks(sq, them) & pos.pawns(us) != 0; // using inversion trick
}

pub fn is_passed_pawn(pos: *const Position, comptime us: Color, sq: Square) bool {
    return switch (us.e) {
        .white => masks.get_passed_pawn_mask(.WHITE, sq) & pos.all_pawns() == 0,
        .black => masks.get_passed_pawn_mask(.BLACK, sq) & pos.all_pawns() == 0,
    };
}

pub fn is_pinned(pos: *const Position, us: Color, sq: Square) bool {
    return pos.state.pins[us.u] & sq.to_bitboard() != 0;
}

pub fn is_pinned_diagonally(pos: *const Position, us: Color, sq: Square) bool {
    return pos.state.pins_diagonal[us.u] & sq.to_bitboard() != 0;
}

pub fn is_pinned_orthogonally(pos: *const Position, us: Color, sq: Square) bool {
    return pos.state.pins_orthogonal[us.u] & sq.to_bitboard() != 0;
}

const Feature = enum {
    // Pawn.
    passed_pawn,
    doubled_pawn,
    isolated_pawn,
    center_pawn,

    // Bishop.
    bishop_pair,

    // Rook.
    rook_on_seventh,
    rooks_connected,
    rook_on_open_file,
    rook_on_half_open_file,

    // King.
    rooks_or_queens_staring_at_king,
    bishops_or_queens_staring_at_king,
    pieces_swarming_around_king,
    attacks_close_to_king,
    king_protection_by_pawns,
    king_protection_by_pieces,
    king_pushing_king,
    open_file_to_our_king,
    open_diagonal_to_our_king,

    threat,
    king_threat,

    // Generic.
    pesto_mg,
    pesto_eg,

    supported_by_pawn,
    space,
    mobility,
    control,
    in_check,

    experimental,
};

pub const Tables = struct {
    //fn get_index(comptime us: Color, sq: Square) u6
    fn get_index(us: Color, sq: Square) u6 {
        return switch(us.e) {
            .white => sq.u ^ 56,
            .black => sq.u,
        };
    }

    pub fn get_move_ordering_score(non_pawn_material: Value, us: Color, m: Move, pt: PieceType) Value {
        const pair_from = get_scorepair(us, pt, m.from);
        const pair_to = get_scorepair(us, pt, m.to);
        const score_from = sliding_score(non_pawn_material, pair_from.mg, pair_from.eg);
        const score_to = sliding_score(non_pawn_material, pair_to.mg, pair_to.eg);
        return score_to - score_from;
    }

    /// Returns the table values for opening and endgame, between which we slide the final score.
    //fn get_scorepair(comptime us: Color, comptime pc: PieceType, sq: Square) struct { mg: Value, eg: Value}
    pub fn get_scorepair(us: Color, pc: PieceType, sq: Square) struct { mg: Value, eg: Value} {

        const mg: [*]const Value = switch (pc.e) {
            .pawn   => &mg_pawn_table,
            .knight => &mg_knight_table,
            .bishop => &mg_bishop_table,
            .rook   => &mg_rook_table,
            .queen  => &mg_queen_table,
            .king   => &mg_king_table,
            else => unreachable,
        };

        const eg: [*]const Value = switch (pc.e) {
            .pawn   => &eg_pawn_table,
            .knight => &eg_knight_table,
            .bishop => &eg_bishop_table,
            .rook   => &eg_rook_table,
            .queen  => &eg_queen_table,
            .king   => &eg_king_table,
            else => unreachable,
        };

        const idx: u6 = get_index(us, sq);
        return .{ .mg = mg[idx], .eg = eg[idx] };
    }

    pub fn sliding_score(non_pawn_material: Value, opening: Value, endgame: Value) Value {
        const max: i32 = comptime types.max_material_without_pawns;
        const phase: i32 = @min(non_pawn_material, max);
        return @truncate(@divTrunc(opening * phase + endgame * (max - phase), max));
    }

    //const int gamephaseInc[12] = {0,0,1,1,1,1,2,2,4,4,0,0}; // p p n n b b r r q q k k
    const pesto_game_phase: [8]Value = .{
        0, // no_piece
        0, // pawn
        1, // knight
        1, // bishop
        2, // rook
        4, // queen
        0, // king
        0, // nothing
    };

    // Pesto tables.
    const mg_pawn_table: [64]Value = .{
          0,   0,   0,   0,   0,   0,  0,   0,
         98, 134,  61,  95,  68, 126, 34, -11,
         -6,   7,  26,  31,  65,  56, 25, -20,
        -14,  13,   6,  21,  23,  12, 17, -23,
        -27,  -2,  -5,  12,  17,   6, 10, -25,
        -26,  -4,  -4, -10,   3,   3, 33, -12,
        -35,  -1, -20, -23, -15,  24, 38, -22,
          0,   0,   0,   0,   0,   0,  0,   0,
    };

    const eg_pawn_table: [64]Value = .{
        0,   0,   0,   0,   0,   0,   0,   0,
        178, 173, 158, 134, 147, 132, 165, 187,
        94, 100,  85,  67,  56,  53,  82,  84,
        32,  24,  13,   5,  -2,   4,  17,  17,
        13,   9,  -3,  -7,  -7,  -8,   3,  -1,
        4,   7,  -6,   1,   0,  -5,  -1,  -8,
        13,   8,   8,  10,  13,   0,   2,  -7,
        0,   0,   0,   0,   0,   0,   0,   0,
    };

    const mg_knight_table: [64]Value = .{
        -167, -89, -34, -49,  61, -97, -15, -107,
        -73,  -41,  72,  36,  23,  62,   7,  -17,
        -47,   60,  37,  65,  84, 129,  73,   44,
        -9,    17,  19,  53,  37,  69,  18,   22,
        -13,    4,  16,  13,  28,  19,  21,   -8,
        -23,   -9,  12,  10,  19,  17,  25,  -16,
        -29,  -53, -12,  -3,  -1,  18, -14,  -19,
        -105, -21, -58, -33, -17, -28, -19,  -23,
    };

    const eg_knight_table: [64]Value = .{
        -58, -38, -13, -28, -31, -27, -63, -99,
        -25,  -8, -25,  -2,  -9, -25, -24, -52,
        -24, -20,  10,   9,  -1,  -9, -19, -41,
        -17,   3,  22,  22,  22,  11,   8, -18,
        -18,  -6,  16,  25,  16,  17,   4, -18,
        -23,  -3,  -1,  15,  10,  -3, -20, -22,
        -42, -20, -10,  -5,  -2, -20, -23, -44,
        -29, -51, -23, -15, -22, -18, -50, -64,
    };

    const mg_bishop_table: [64]Value = .{
        -29,   4, -82, -37, -25, -42,   7,  -8,
        -26,  16, -18, -13,  30,  59,  18, -47,
        -16,  37,  43,  40,  35,  50,  37,  -2,
        -4,   5,  19,  50,  37,  37,   7,  -2,
        -6,  13,  13,  26,  34,  12,  10,   4,
         0,  15,  15,  15,  14,  27,  18,  10,
         4,  15,  16,   0,   7,  21,  33,   1,
        -33,  -3, -14, -21, -13, -12, -39, -21,
    };

    const eg_bishop_table: [64]Value = .{
        -14, -21, -11,  -8, -7,  -9, -17, -24,
        -8,  -4,   7, -12, -3, -13,  -4, -14,
        2,  -8,   0,  -1, -2,   6,   0,   4,
        -3,   9,  12,   9, 14,  10,   3,   2,
        -6,   3,  13,  19,  7,  10,  -3,  -9,
        -12,  -3,   8,  10, 13,   3,  -7, -15,
        -14, -18,  -7,  -1,  4,  -9, -15, -27,
        -23,  -9, -23,  -5, -9, -16,  -5, -17,
    };

    const mg_rook_table: [64]Value = .{
        32,  42,  32,  51, 63,  9,  31,  43,
        27,  32,  58,  62, 80, 67,  26,  44,
        -5,  19,  26,  36, 17, 45,  61,  16,
        -24, -11,   7,  26, 24, 35,  -8, -20,
        -36, -26, -12,  -1,  9, -7,   6, -23,
        -45, -25, -16, -17,  3,  0,  -5, -33,
        -44, -16, -20,  -9, -1, 11,  -6, -71,
        -19, -13,   1,  17, 16,  7, -37, -26,
    };

    const eg_rook_table: [64]Value = .{
        13, 10, 18, 15, 12,  12,   8,   5,
        11, 13, 13, 11, -3,   3,   8,   3,
        7,  7,  7,  5,  4,  -3,  -5,  -3,
        4,  3, 13,  1,  2,   1,  -1,   2,
        3,  5,  8,  4, -5,  -6,  -8, -11,
        -4,  0, -5, -1, -7, -12,  -8, -16,
        -6, -6,  0,  2, -9,  -9, -11,  -3,
        -9,  2,  3, -1, -5, -13,   4, -20,
    };

    const mg_queen_table: [64]Value = .{
        -28,   0,  29,  12,  59,  44,  43,  45,
        -24, -39,  -5,   1, -16,  57,  28,  54,
        -13, -17,   7,   8,  29,  56,  47,  57,
        -27, -27, -16, -16,  -1,  17,  -2,   1,
        -9, -26,  -9, -10,  -2,  -4,   3,  -3,
        -14,   2, -11,  -2,  -5,   2,  14,   5,
        -35,  -8,  11,   2,   8,  15,  -3,   1,
        -1, -18,  -9,  10, -15, -25, -31, -50,
    };

    const eg_queen_table: [64]Value = .{
        -9,  22,  22,  27,  27,  19,  10,  20,
        -17,  20,  32,  41,  58,  25,  30,   0,
        -20,   6,   9,  49,  47,  35,  19,   9,
        3,  22,  24,  45,  57,  40,  57,  36,
        -18,  28,  19,  47,  31,  34,  39,  23,
        -16, -27,  15,   6,   9,  17,  10,   5,
        -22, -23, -30, -16, -16, -23, -36, -32,
        -33, -28, -22, -43,  -5, -32, -20, -41,
    };

    const mg_king_table: [64]Value = .{
        -65,  23,  16, -15, -56, -34,   2,  13,
        29,  -1, -20,  -7,  -8,  -4, -38, -29,
        -9,  24,   2, -16, -20,   6,  22, -22,
        -17, -20, -12, -27, -30, -25, -14, -36,
        -49,  -1, -27, -39, -46, -44, -33, -51,
        -14, -14, -22, -46, -44, -30, -15, -27,
        1,   7,  -8, -64, -43, -16,   9,   8,
        -15,  36,  12, -54,   8, -28,  24,  14,
    };

    const eg_king_table: [64]Value = .{
        -74, -35, -18, -18, -11,  15,   4, -17,
        -12,  17,  14,  17,  17,  38,  23,  11,
        10,  17,  23,  15,  20,  45,  44,  13,
        -8,  22,  24,  27,  26,  33,  26,   3,
        -18,  -4,  21,  24,  27,  23,   9, -11,
        -19,  -3,  11,  21,  23,  16,   7,  -9,
        -27, -11,   4,  13,  14,   4,  -5, -17,
        -53, -34, -21, -11, -28, -14, -24, -43
    };

    // // TODO; or not todo.
    // const threat_table: [7][7]Value = .{
    //     .{ 0,  0,  0,  0,  0,  0,  0 },        // no_piece
    //     .{ 0,  0,  20, 30, 50, 90, 100 },  // pawn
    //     .{ 0,  10, 0,  0,  0,  0,  0 },        // knight
    //     .{ 0,  0,  0,  0,  0,  0,  0 },        // bishop
    //     .{ 0,  0,  0,  0,  0,  0,  0 },        // rook
    //     .{ 0,  0,  0,  0,  0,  0,  0 },        // queen
    //     .{ 0,  0,  0,  0,  0,  0,  0 },        // king
    // };

    pub const threat_pawn   : Value = 10;
    pub const threat_knight : Value = 30;
    pub const threat_bishop : Value = 30;
    pub const threat_rook   : Value = 40;
    pub const threat_queen  : Value = 50;
    pub const threat_king   : Value = 50;

    const threat_values: [8]Value = .{
        0, threat_pawn, threat_knight, threat_bishop, threat_rook, threat_queen, threat_king, 0,
    };

    // Pawn.
    const passed_pawn_by_rank_scores: [8]Value = .{ 0, 15, 15, 30, 50, 80, 120, 0 }; // .{ 0, 2, 4, 6, 8, 20, 40, 0 };
    const doubled_pawn: Value = -5;
    const isolated_pawn: Value = -10;

    // Bishop.
    const bishop_pair: Value = 30;

    // Generic.
    const supported_by_pawn: Value = 10;

    // Rook.
    const rook_on_seventh: Value = 20;
    const rooks_connected: Value = 20;

    // King.
    const in_check: Value = -20;
    const rooks_or_queens_staring_at_king: Value = -5;
    const bishops_or_queens_staring_at_king: Value = -5;

    pub fn mobility(comptime pt: PieceType, comptime phase: GamePhase, mobility_bitboard: u64) Value {
        const m: f32 = switch (phase){
            .Opening => 0.8,
            .Midgame => 1.2,
            .Endgame => 0.2,
        };

        const bitcount: Value = popcnt(mobility_bitboard);
        switch (pt.e) {
            .pawn => {
                return funcs.mul(bitcount, m);
            },
            .knight => {
                return funcs.mul(bitcount, m);
            },
            .bishop => {
                return bitcount;
            },
            .rook => {
                return bitcount;
            },
            .queen => {
                return bitcount;
            },
            .king => {
                return bitcount;
            },
            else => unreachable,
        }
        return 0;
    }

};

/// A little 1 second time eval. around 14 million per second.
pub fn bench(pos: *const Position) void {
    var v: i64 = 0;
    var cnt: u64 = 0;
    var timer = utils.Timer.start();
    while (timer.elapsed_ms() < 1000) {
        v += evaluate(pos, false);
        cnt += 1;
    }
    const t = timer.read();
    lib.io.debugprint("v {} cnt {} {}\n", .{v, cnt, funcs.nps(cnt, t)});
}

test "see"
{
    try lib.initialize();
    var pos: Position = .empty;

    const GOOD = true;
    const BAD = false;

    // Pawn exchange.
    try execute_see_test(&pos, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "d5e6", GOOD);
    // Queen takes covered knight.
    try execute_see_test(&pos, "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", "f3f6", BAD);
    // Pawn takes knight.
    try execute_see_test(&pos, "8/8/5b1k/4n3/K2P4/8/8/8 w - - 0 1", "d4e5", GOOD);
    // Losing queen for pawn.
    try execute_see_test(&pos, "8/kb6/2p5/3p4/4Q3/5B2/6QK/8 w - - 0 1", "e4d5", BAD);
    // Losing rook for knight.
    try execute_see_test(&pos, "3q3k/8/4p3/3n4/3R4/8/8/K2R4 w - - 0 1", "d4d5", BAD);
    // Bishop captures pawn. King involved.
    try execute_see_test(&pos, "8/8/2k5/3p4/4B3/5K2/8/8 w - - 0 1", "e4d5", BAD);
    // Pawn captures bishop. King involved.
    try execute_see_test(&pos, "8/8/2k5/3b4/4P3/5K2/8/8 w - - 0 1", "e4d5", GOOD);
    // Pawn exchange. King involved.
    try execute_see_test(&pos, "8/8/2k5/3p4/4P3/5K2/8/8 w - - 0 1", "e4d5", GOOD);
    // Winning a pawn. King involved.
    try execute_see_test(&pos, "8/8/2k5/3p4/4P3/6K1/6B1/8 w - - 0 1", "e4d5", GOOD);
}

/// TODO: complete with see->bool and expected value.
fn execute_see_test(pos: *Position, fen: []const u8, move: []const u8, expected_good: bool) !void {
    var st: StateInfo = undefined;
    try pos.set(&st, fen);
    const m: Move = try pos.parse_move(move);
    const val: Value = see_score(pos, m);
    if (expected_good)
        try std.testing.expect(val >= 0)
    else
        try std.testing.expect(val < 0);
}