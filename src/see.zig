// zig fmt: off

//! Static Exchange Evaluation.
//! Work In Progress...

const std = @import("std");
const assert = std.debug.assert;

const lib = @import("lib.zig");
const utils = @import("utils.zig");
const attacks = @import("attacks.zig");
const bitboards = @import("bitboards.zig");
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
    testing,
};

// TODO: maybe we can make evaluate comptime colored.
// TODO: wrap the attackloop in a zig-get-labeled-block.
// TODO: try live updating pins.

// position fen 8/1kb2R2/3n4/8/8/3R4/8/5K2 w - - 0 1
// [see d3d6 0] -> TRUE (black cannot recapture)

// more difficult -> live updating the pins?
// position fen 8/1kbr1R2/3n4/8/8/3R4/3R4/5K2 w - - 0 1
// [see d3d6 0] should give TRUE


/// Returns true for castle, ep and promotion, regardless of the threshold.
/// - Assumes the move is valid.
/// - Positive threshold means: are we winning this amount of material or more?
/// - Negative threshold means: the amount we are allowed to lose.
pub fn evaluate(pos: *const Position, m: Move, threshold: i32, comptime mode: Mode) bool {
    if (m.is_castle() or m.is_ep() or m.is_promotion()) {
        return true;
    }
    const from: Square = m.from;
    const to: Square = m.to;

    // Set the score to captured piece minus how much we are allowed to lose.
    var score: i32 = value(pos.board[to.u].u, mode) - threshold;
    if (score < 0) {
        return false;
    }
    score = value(pos.board[from.u].u, mode) - score;

    // Equal or winning.
    if (score <= 0) {
        return true;
    }

    const queens_bishops = pos.all_queens_bishops();
    const queens_rooks = pos.all_queens_rooks();

    // Determine pieces that cannot move.
    const white_allowed_to_move: u64 = ~pos.pins(.white) & pos.by_color(.white);
    const black_allowed_to_move: u64 = ~pos.pins(.black) & pos.by_color(.black);
    const allowed_to_move: u64 = white_allowed_to_move | black_allowed_to_move;

    // Execute the move on a bitboard.
    var occupied = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();

    // Get the initial attacks from both sides to the to-square.
    var all_attackers: u64 = pos.get_combined_attackers_to_for_occupation(occupied, to);
    var us: Color = pos.stm;
    var winner: Color = pos.stm;

    attackloop: while (true) {
        us = us.opp();
        all_attackers &= occupied & allowed_to_move;

        // Get our attackers.
        const our_attackers: u64 = all_attackers & pos.by_color(us);

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
                    .pawn => {
                        all_attackers |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
                    },
                    .knight => {
                        // Do nothing: a knight move cannot reveal a new slider.
                    },
                    .bishop => {
                        all_attackers |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
                    },
                    .rook => {
                        all_attackers |= (attacks.get_rook_attacks(to, occupied) & queens_rooks);
                    },
                    .queen => {
                        all_attackers |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops) | (attacks.get_rook_attacks(to, occupied) & queens_rooks);
                    },
                    .king => {
                        // We can exit here: if the king captures and the opponent can capture our king we lose othersize we win.
                        return if (all_attackers & pos.by_color(us.opp()) != 0) pos.stm.u != winner.u else return pos.stm.u == winner.u;
                    },
                    else => {
                        unreachable;
                    }
                }
                next_attacker_value = value(piecetype.u, mode);
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


fn value(u: u4, comptime mode: Mode)  i32 {
    return switch (mode) {
        .default => types.piece_values[u],
        .testing => types.simple_piece_values[u],
    };
}


// --- SEE original chessnix 1.4 ---

// /// Returns true for castle, enpassant and promotions, regardless of the threshold.
// pub fn see(pos: *const Position, m: Move, threshold: i32) bool {
//     if (m.is_castle() or m.is_ep() or m.is_promotion()) {
//         return true;
//     }

//     const from: Square = m.from;
//     const to: Square = m.to;

//     // Set the score to captured piece minus how much we are allowed to lose.
//     var score: i32 = pos.board[to.u].value() - threshold;

//     if (score < 0) {
//         return false;
//     }

//     score = pos.board[from.u].value() - score;

//     // Equal or winning.
//     if (score <= 0) {
//         return true;
//     }

//     const queens_bishops = pos.all_queens_bishops();
//     const queens_rooks = pos.all_queens_rooks();

//     // Execute the move on a bitboard.
//     var occupied = pos.all() ^ to.to_bitboard() ^ from.to_bitboard();

//     // Get the initial attacks from both sides to the to-square.
//     var all_attacks: u64 = pos.get_combined_attackers_to_for_occupation(occupied, to);
//     var us: Color = pos.stm;
//     var winner: Color = pos.stm;

//     attackloop: while (true) {
//         us = us.opp();
//         all_attacks &= occupied;

//         // Get our attackers.
//         const our_attackers: u64 = all_attacks & pos.by_color(us);

//         // No attackers left.
//         if (our_attackers == 0) {
//             break :attackloop;
//         }
//         winner = winner.opp();
//         var next_attacker_value: i32 = 0;

//         // Get the least valuable next piece.
//         get_next_attacker: inline for (PieceType.all) |piecetype| {
//             const next_attacker: u64 = our_attackers & pos.by_type(piecetype);
//             if (next_attacker != 0) {

//                 // Clear this attacker.
//                 const sq: Square = funcs.first_square(next_attacker);
//                 funcs.clear_square(&occupied, sq);
//                 // Reveal next x-ray attacker on the attacks bitboard.
//                 switch (piecetype.e) {
//                     .pawn => {
//                         all_attacks |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
//                     },
//                     .knight => {
//                         // Do nothing: a knight move cannot reveal a new slider.
//                     },
//                     .bishop => {
//                         all_attacks |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
//                     },
//                     .rook => {
//                         all_attacks |= (attacks.get_rook_attacks(to, occupied) & queens_rooks);
//                     },
//                     .queen => {
//                         all_attacks |= (attacks.get_bishop_attacks(to, occupied) & queens_bishops);
//                         all_attacks |= (attacks.get_rook_attacks(to, occupied) & queens_rooks);
//                     },
//                     .king => {
//                         // We can exit here: if the king captures and the opponent can capture our king we lose othersize we win.
//                         return if (all_attacks & pos.by_color(us.opp()) != 0) pos.stm.u != winner.u else return pos.stm.u == winner.u;
//                     },
//                     else => {
//                         unreachable;
//                     }
//                 }

//                 next_attacker_value = piecetype.value();
//                 break :get_next_attacker;
//             }
//         }

//         score = -score + 1 + next_attacker_value;

//         // Quit if the exchange is lost or equal
//         if (score <= 0) {
//             break :attackloop;
//         }
//     }
//     return pos.stm.u == winner.u;
// }
