// zig fmt: off

//! Static Exchange Evaluation.

const std = @import("std");
const assert = std.debug.assert;

const lib = @import("lib.zig");
const utils = @import("utils.zig");
const attacks = @import("attacks.zig");
const bitboards = @import("bitboards.zig");
const squarepairs = @import("squarepairs.zig");
const funcs = @import("funcs.zig");
const types = @import("types.zig");
const position = @import("position.zig");

const io = lib.io;

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const Position = position.Position;

pub const Mode = enum {
    default,
    simple,
};

/// Returns the result of a possible piece exchange on the to-square of the move.
/// - Assumes the move is valid.
/// - Positive threshold means: are we winning this amount of material or more?
/// - Negative threshold means: the amount we are allowed to lose.
/// - Zero means: a neutral exchange.
pub fn evaluate(pos: *const Position, m: Move, threshold: i32, comptime mode: Mode) bool {
    const val: *const fn (piece_or_piecetype: u4) i32 = comptime switch (mode) {
        .default => default_value,
        .simple => simple_value,
    };

    // Castling can never win a piece.
    if (m.is_castle()) {
        return threshold <= 0;
    }
    const stm: Color = pos.stm;

    const pawn_value: i32 = comptime val(PieceType.pawn.u);
    const from: Square = m.from;
    const to: Square = m.to;
    const is_ep: bool = m.is_ep();
    const is_promo: bool = m.is_promotion();

    // Set the score to captured piece minus how much we are allowed to lose.
    // When the to-square is empty we capture nothing, so this works for quiet moves too.
    const gain: i32 = if (is_ep) pawn_value else val(pos.get(to).u);
    //var score: i32 = value(pos.board[to.u].u, mode) - threshold;
    var score: i32 = gain - threshold;

    // for a promotion: We gain a piece and lose a pawn.
    if (is_promo) {
        score += val(m.prom().u) - pawn_value;
    }

    // We cannot beat the threshold.
    if (score < 0) {
        return false;
    }

    const lose: i32 = if (is_promo) val(m.prom().u) else val(pos.get(from).u);
    score = lose - score;

    // Equal or winning.
    if (score <= 0) {
        return true;
    }

    const queens_bishops = pos.all_queens_bishops();
    const queens_rooks = pos.all_queens_rooks();

    // Determine pieces that can move.
    // Because we always 'execute' the first move we can get away with this.
    const white_allowed_to_move: u64 = ~pos.pins(.white) & pos.by_color(.white);
    const black_allowed_to_move: u64 = ~pos.pins(.black) & pos.by_color(.black);
    const allowed_to_move: u64 = white_allowed_to_move | black_allowed_to_move;

    // Execute the move on a bitboard.
    var occupied = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();
    if (is_ep)  {
        const sq: Square = if (pos.stm.e == .white) pos.ep_square.sub(8) else pos.ep_square.add(8);
        occupied ^= sq.to_bitboard();
    }

    // Get the initial attacks from both sides to the to-square.
    var all_attackers: u64 = pos.get_combined_attackers_to_for_occupation(occupied, to);
    var us: Color = stm;
    var winner: Color = stm;

    attackloop: while (true) {
        us = us.opp();
        all_attackers &= occupied;

        // Get our attackers.
        const our_attackers: u64 = all_attackers & pos.by_color(us) & allowed_to_move;
        if (our_attackers == 0) {
            break :attackloop;
        }
        winner = winner.opp();

        // Get the least valuable next piece.
        const next_attacker_value: i32 = blk: {
            inline for (PieceType.all) |piecetype| {
                const next_attacker: u64 = our_attackers & pos.by_type(piecetype);
                if (next_attacker != 0) {
                    // Clear this attacker.
                    const sq: Square = bitboards.first_square(next_attacker);
                    bitboards.clear_square(&occupied, sq);
                    // Reveal next x-ray attacker on the attacks bitboard. Knights cannot reveal a new x-ray.
                    // We can stop if we see a king. If the king captures and the opponent can capture our king we lose othersize we win.
                    switch (piecetype.e) {
                        .knight => {},
                        .pawn, .bishop => all_attackers |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops),
                        .rook => all_attackers |= (attacks.get_rook_attacks(to, occupied) & queens_rooks),
                        .queen => all_attackers |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops) | (attacks.get_rook_attacks(to, occupied) & queens_rooks),
                        .king => return if (all_attackers & pos.by_color(us.opp()) != 0) stm.u != winner.u else return stm.u == winner.u,
                        else => unreachable,
                    }
                    break :blk val(piecetype.u);
                }
            }
            unreachable; // We must find a piece.
        };

        score = -score + 1 + next_attacker_value;

        // Quit if the exchange is lost or equal
        if (score <= 0) {
            break :attackloop;
        }
    }
    return stm.u == winner.u;
}

fn default_value(piece_or_piecetype: u4) i32 {
    return types.see_piece_values[piece_or_piecetype]; // TODO: BUG ALERT here we used simple_piece_values. run autoplay again.
}

fn simple_value(piece_or_piecetype: u4) i32 {
    return types.simple_piece_values[piece_or_piecetype];
}

pub fn value(piece_or_piecetype: u4, comptime mode: Mode)  i32 {
    return switch (mode) {
        .default => types.piece_values[piece_or_piecetype],
        .testing => types.simple_piece_values[piece_or_piecetype],
    };
}

