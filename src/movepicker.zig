// zig fmt: off

///! Staged move picker.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const search = @import("search.zig");
const history = @import("history.zig");

const Value = types.Value;
const SmallValue = types.SmallValue;
const Color = types.Color;
const Square = types.Square;
const Piece = types.Piece;
const Move = types.Move;
const Position = position.Position;
const ExtMove = search.ExtMove;
const Node = search.Node;
const History = history.History;

pub const GenType = enum {
    Search,
    Quiescence,
};

pub const Stage = enum {
    /// Fetch tt_move. Check probably legal
    TTMove,
    Noisy,
    GenerateNoisy,
    Killer1,
    Killer2,
    GenerateQuiets,
    Quiets,
    BadNoisy,
};

pub const MovePicker = struct {
    gentype: GenType,
    stage: Stage,
    pos: *const Position,
    hist: *const History,
    node: *const Node,
    idx: usize,
    tt_move: Move,
    noisy_moves: [80]ExtMove,
    quiet_moves: [120]ExtMove,
    bad_noisy_moves: [40]ExtMove,

    pub fn init(gentype: GenType, pos: *const Position, hist: *const History, node: *const Node, tt_move: Move) MovePicker {
        return .{
            .gentype = gentype,
            .pos = pos,
            .history = hist,
            .node = node,
            .tt_move = tt_move,
            .noisy_moves = undefined,
            .quiet_moves = undefined,
            .bad_noisy_moves = undefined,
        };
    }

    pub fn next(self: *MovePicker) *ExtMove {
        switch (self.stage) {
            .TTMove => {
                self.stage = .Noisy;
                const ok: bool = self.pos.is_move_legal(self.tt_move);
                if (ok) {
                    return self.tt_move;
                }
            },
           .Noisy => {

            },
            .GenerateNoisy => {

            },
            .Killer1 => {

            },
            .Killer2 => {

            },
            .GenerateQuiets => {

            },
            .Quiets => {

            },
            .BadNoisy => {

            },
        }
    }

    pub fn skip_quiets(self: *MovePicker) void {
        self.stage = .BadNoisy;
    }

    fn extract(self: *MovePicker) *ExtMove {
        _ = self;
    }

    fn store(self: *MovePicker, move: Move) ?void {
        _ = self;
        _ = move;
    }

    fn generate_and_score_moves() void {

    }

    fn score_move() void {
    }
};

fn List(max: usize) type {

    return struct {

        extmoves: [max]ExtMove,
        count: usize,

    };
}