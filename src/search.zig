// zig fmt: off

//! --- NOT USED YET ---

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: before going multithreaded get ONE working single threaded search.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// TODO: maybe staged movegeneration
// * TT move
// * Winning captures + checks
// * Killer moves
// * Quiet moves.
// * Losing captures + bad moves (often last or skipped).
// * substract something from time to prevent time-loss.

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const uci = @import("uci.zig");
const eval = @import("eval.zig");

const max_u64: u64 = std.math.maxInt(u64);

const Value = types.Value;
const Color = types.Color;
const Move = types.Move;
const StateInfo = position.StateInfo;
const Position = position.Position;

const ctx = lib.ctx;
const wtf = lib.wtf;

const max_search_depth: u8 = types.max_search_depth;
const max_threads: u16 = types.max_threads;

const PV = lib.BoundedArray(Move, max_search_depth);
const Nodes = lib.BoundedArray(Node, max_search_depth);

pub const SearchParams = struct
{
    const infinite_search_params: SearchParams =
    .{
        .termination = .infinite,
        .max_depth = types.max_search_depth,
        .max_nodes = max_u64,
        .max_movetime_ms = max_u64,
    };

    termination: Termination,
    max_depth: u8,
    max_nodes: u64,
    max_movetime_ms: u64,

    /// Convert the UCI go params to something usable for the search.
    pub fn init(go: *const uci.Go) SearchParams
    {
        //lib.io.print("{any}\n", .{go}) catch wtf();

        // (1) infinite.
        if (go.infinite != null)
        {
            return .infinite_search_params;
        }

        // (2) depth.
        if (go.depth) |depth|
        {
            return
            .{
                .termination = .depth,
                .max_depth = @truncate(@min(depth, types.max_search_depth)),
                .max_nodes = max_u64,
                .max_movetime_ms = max_u64,
            };
        }

        // (3) nodes.
        if (go.nodes) |nodes|
        {
            return
            .{
                .termination = .nodes,
                .max_depth = types.max_search_depth,
                .max_nodes = nodes,
                .max_movetime_ms = max_u64,
            };
        }

        // (4) movetime.
        if (go.movetime) |movetime|
        {
            return
            .{
                .termination = .movetime,
                .max_depth = types.max_search_depth,
                .max_nodes = max_u64,
                .max_movetime_ms = movetime,
            };
        }

        return .infinite_search_params;
    }
};

/// This is to become the "thread" manager.
pub const SearchManager = struct
{
    num_threads: usize,
    searchparams: SearchParams,
    searcher: Search,

    pub fn init() SearchManager
    {
        var mgr: SearchManager = undefined;

        mgr.num_threads = 1;
        mgr.searchparams = .infinite_search_params;
        mgr.searcher = Search.init();

        return mgr;
    }

    pub fn deinit(self: *SearchManager) void
    {
        //self.searchers.deinit(ctx.galloc);
        self.searcher.deinit();
    }

    pub fn start(self: *SearchManager, pos: *const Position, go: *const uci.Go) !void
    {
        // we should have some atomic value to check if the threads are not already running.
        // then start them.
        // std.Thread.spawn
        // std.Thread.detach
        //_ = self;
        //try lib.io.print("WE START THINKING\n", .{});
        self.searchparams = .init(go);
        //try lib.io.print("{any}\n", .{ self.searchparams });
        self.searcher.start_search(self, pos);
    }

    pub fn stop(self: *SearchManager) !void
    {
        // we should have some atomic value to stop the thread(s)
        _ = self;
        try lib.io.print("WE SHOULD STTOP THINKING\n", .{});
    }
};

const Search = struct
{
    /// Backlink ref. Only valid after start_search. TODO: Bad design still...
    mgr: *const SearchManager,
    /// A private copy of the engine pos.
    pos: Position,
    /// Rootmoves. resorted each iteration.
    rootmoves: MovePicker,
    /// We always keep track of the best move a.s.a.p.
    best_move: Move,
    stack: Stack,
    pv_node: Node,
    nodes: u64,
    /// The max reached depth, including quiet depth.
    seldepth: u16,
    iteration: u16,
    stop: bool,
    timer: utils.Timer,

    fn init() Search
    {
        return Search
        {
            .mgr = undefined,
            .pos = .empty,
            .rootmoves = .init(),
            .best_move = .empty,
            .stack = .init(),
            .pv_node = .init(),
            .nodes = 0,
            .seldepth = 0,
            .iteration = 0,
            .stop = false,
            .timer = .empty,
        };
    }

    fn deinit(self: *Search) void
    {
        _ = self;
    }

    // fn mgr(self: *const Search) *const SearchManager
    // {
    //     const manager_ptr: *const SearchManager = @fieldParentPtr("searcher", self);
    //     return manager_ptr;
    // }

    /// Initialize stuff for a new search and go.
    fn start_search(self: *Search, mgr: *const SearchManager, input_pos: *const Position) void
    {
        self.mgr = mgr;
        self.timer.reset();
        self.rootmoves.reset();
        self.best_move = .empty;
        self.stop = false;
        self.iteration = 0;
        self.nodes = 0;
        self.seldepth = 0;
        self.pv_node.clear();
        // stack: Stack,
        // pv_node: Node,

        // Create our private copy of the position.
        var rootstate: StateInfo = undefined;
        self.pos.copy_from(input_pos, &rootstate);

        // Append our rootstate to the history of the source position.
        rootstate.prev = input_pos.state;

        switch (self.pos.to_move.e)
         {
             .white => self.iterative_deepening(Color.WHITE),
             .black => self.iterative_deepening(Color.BLACK),
         }

         lib.io.print("bestmove {f}\n", .{ self.best_move }) catch wtf();
    }

    fn stop_search(self: *Search) void
    {
        _ = self;
    }

    fn iterative_deepening(self: *Search, comptime us: Color) void
    {
        self.rootmoves.generate_moves(&self.pos, us);

        // Have something a.s.a.p.
        if (self.rootmoves.count > 0)
        {
            self.best_move = self.rootmoves.extmoves[0].move;
        }


        self.iteration = 1;
        var depth: u8 = 1;
        var best_score: Value = -types.infinity;
        while (true)
        {
            const score: Value = self.alpha_beta(true, us, depth, -types.infinity, types.infinity);

            // We found a better and different move.
            if (score > best_score)
            {
                best_score = score;
            }

            // Copy the last finished pv.
            self.pv_node.clone_from(&self.stack.nodes.buffer[2]);
            self.best_move = self.pv_node.best_move();

            // And print.
            print_pv(&self.pv_node, self.iteration, self.seldepth, self.nodes, self.timer.read());

            if (self.check_stop()) break;

            //if (self.iteration == 7) break;

            //if (self.timer.elapsed_ms() > 5000) break;

            self.iteration += 1;
            depth += 1;
        }
    }

    fn alpha_beta(self: *Search, comptime is_root: bool, comptime us: Color, depth: u8, input_alpha: Value, beta: Value) Value
    {
        const them = comptime us.opp();
        const pos: *Position = &self.pos;
        const ply: u16 = pos.ply;

        self.nodes += 1;
        self.seldepth = @max(self.seldepth, ply);

        // TODO: what to return on timeout?

        // Requested depth reached.
        if (depth == 0)
        {
            return self.quiet(is_root, us, 0, input_alpha, beta);
        }

        // Too deep.
        if (ply >= max_search_depth)
        {
            return eval.evaluate(pos, us, false);
        }

        const node: *Node = self.stack.get_node(pos.ply + 2);
        const childnode: *const Node = Stack.get_child_node(node);
        const parentnode: *const Node = Stack.get_parent_node(node);
        _ = parentnode;

        // Clear current node.
        node.clear();

        if (pos.state.rule50 >= 100) return 0;

        const in_check: bool = pos.state.checkers > 0;

        // Generate moves or copy the rootmoves.
        var movepicker = MovePicker.init();
        if (is_root)
        {
            movepicker.copy_from(&self.rootmoves);
        }
        else
        {
            movepicker.generate_moves(pos, us);
        }

        // Is this checkmate or stalemate?
        if (movepicker.count == 0)
        {
            if (in_check) return -types.mate + pos.ply else return types.stalemate;
        }

        var alpha: Value = input_alpha;

        // Go trough the moves.
        for (movepicker.slice()) |extmove|
        {
            var st: StateInfo = undefined;
            const move: Move = extmove.move;

            self.pos.make_move(us, &st, move);
            const score: Value = -self.alpha_beta(false, them, depth - 1, -beta, -alpha);
            self.pos.unmake_move(us);

            // Higher alpha. A new best move is found.
            if (score > alpha)
            {
                node.score = score;
                // Beta cutoff / fail high.
                if (score >= beta)
                {
                    return score;
                }
                alpha = score;
                update_pv(move, score, node, childnode);
            }
        }
        return alpha;
    }

    fn quiet(self: *Search, comptime is_root: bool, comptime us: Color, quiet_depth: u8, input_alpha: Value, beta: Value) Value
    {
         _ = is_root;

        const them = comptime us.opp();
        const pos: *Position = &self.pos;
        const ply: u16 = pos.ply;
        const in_check : bool = pos.state.checkers > 0;
        var alpha = input_alpha;

        self.nodes += 1;
        self.seldepth = @max(self.seldepth, ply);

        const static_score: Value = eval.evaluate(pos, us, false);

        // Too deep.
        if (ply >= max_search_depth)
        {
            return static_score;
        }

        if (static_score >= beta)
        {
            return beta;
        }

        if (static_score > alpha)
        {
            alpha = static_score;
        }

        var movepicker: MovePicker = .init();
        movepicker.generate_captures(pos, us);

        // Mate or stalemate?
        if (movepicker.count == 0)
        {
            //pos.print() catch wtf();
            //lib.io.print("WTF, ", .{}) catch wtf();
            if (in_check and quiet_depth == 0)
            {
                 return -types.mate + ply;
            }
            return alpha;
        }

        for (movepicker.slice()) |extmove|
        {
            // TODO: skip bad captures if not incheck.
            const move: Move = extmove.move;
            var st: StateInfo = undefined;
            pos.make_move(us, &st, move);
            const score: Value = -self.quiet(false, them, quiet_depth + 1, -beta, -alpha);
            pos.unmake_move(us);

            // Better move found.
            if (score > alpha)
            {
                // Beta cutoff.
                if (score >= beta)
                {
                    return score;
                }
                alpha = score;
            }
        }
        return alpha;
    }

    /// Copy line from 'childnode' to 'current node'
    fn update_pv(bestmove: Move, score: Value, node: *Node, childnode: *const Node) void
    {
        node.pv.len = 0;
        node.pv.appendAssumeCapacity(bestmove);
        node.pv.appendSliceAssumeCapacity(childnode.pv.slice());
        node.score = score;
    }

    fn check_stop(self: *Search) bool
    {
        if (self.stop) return true;

        const sp: *const SearchParams = &self.mgr.searchparams;

        switch (sp.termination)
        {
            .infinite =>
            {
                return false;
            },
            .nodes =>
            {
                if (self.nodes >= sp.max_nodes) { self.stop = true; return true; }
            },
            // .time =>
            // {
            //     return false;
            // },
            .movetime =>
            {
                //@panic("WTF");
        //        return false;
                // if (self.nodes & 1023 == 0) lib.io.print("{}, ", .{ self.nodes } ) catch wtf();

                //lib.io.print("{} {},", .{ self.timer.elapsed_ms(), sp.max_movetime_ms}) catch wtf();
                if (self.nodes & 1023 == 0 and self.timer.elapsed_ms() >= sp.max_movetime_ms) { self.stop = true; return true; }
                //if (self.timer.elapsed_ms() >= sp.max_movetime_ms) { self.stop = true; return true; }
            },
            .depth =>
            {
                if (self.iteration >= sp.max_depth) { self.stop = true; return true; }
            },
        }

        return false;
    }
};

/// Each Search has its own stack.
pub const Stack = struct
{
    nodes: Nodes,

    fn init() Stack
    {
        return Stack
        {
            .nodes = .{},
        };
    }

    fn get_node(self: *Stack, index: u16) *Node
    {
        return &self.nodes.buffer[index];
    }

    /// This assumes the node is in the stack.
    fn get_child_node(node: *Node) *const Node
    {
        return funcs.ptr_add(Node, node, 1);
    }

    /// This assumes the node is in the stack.
    fn get_parent_node(node: *Node) *const Node
    {
        return funcs.ptr_sub(Node, node, 1);
    }
};

const Mode = packed struct
{
    is_root: bool,
};

const Node = struct
{
    const empty: Node = .{};

    pv: PV = .{},
    score: Value = 0,
    is_check: bool = false,
    is_mate: bool = false,
    is_stalemate: bool = false,

    fn init() Node
    {
        return .{};
    }

    fn clear(self: *Node) void
    {
        self.* = .empty;
    }

    /// This is just the first move.
    fn best_move(self: *const Node) Move
    {
        return self.pv.buffer[0];
    }

    fn clone_from(self: *Node, other: *const Node) void
    {
        self.pv.len = 0;
        self.pv.appendSliceAssumeCapacity(other.pv.slice());
        self.score = other.score;
    }

    // Zig-format for UCI output.
    pub fn format(self: Node, writer: *std.io.Writer) std.io.Writer.Error!void
    {
        const len: usize = self.pv.len;
        if (len == 0) return;

        for (self.pv.buffer[0..len - 1]) |move|
        {
            try writer.print("{f} ", .{ move });
        }

        const last: Move = self.pv.buffer[len - 1];
        try writer.print("{f}", .{ last });
    }
};

pub const MovePicker = struct
{
    extmoves: [types.max_move_count]ExtMove,
    count: u8,
    current: u8,

    fn init() MovePicker
    {
        return
        .{
            .extmoves = undefined,
            .count = 0,
            .current = 0,
        };
    }

    /// Required function.
    pub fn reset(self: *MovePicker) void
    {
        self.count = 0;
        self.current = 0;
    }

    /// Required function.
    pub fn store(self: *MovePicker, move: Move) ?void
    {
        self.extmoves[self.current] = ExtMove{ .move = move, .score = 0 };
        self.count += 1;
        self.current += 1;
    }

    fn copy_from(self: *MovePicker, other: *const MovePicker) void
    {
        const cnt: u8 = other.count;
        @memcpy(self.extmoves[0..cnt], other.extmoves[0..cnt]);
        self.count = cnt;
        self.current = 0;
    }

    fn slice(self: *const MovePicker) []const ExtMove
    {
        return self.extmoves[0..self.count];
    }

    fn generate_moves(self: *MovePicker, pos: *const Position, comptime us: Color) void
    {
        pos.generate_moves(us, self);
    }

    /// Generate captures only, but if in check generate all.
    fn generate_captures(self: *MovePicker, pos: *const Position, comptime us: Color) void
    {
        pos.generate_captures(us, self);
    }

    /// Give each generated move an initial heuristic score.
    fn score_moves(self: *MovePicker) void
    {
        for (self.slice()) |*e|
        {
            //const capt: PieceType = if (e.move.movetype == .enpassant) P else if (e.move)

            if (e.move.movetype == .promotion)
            {
                //e.score = MoveScoring.promotion + e.move.info.prom.to_piecetype();
            }
            else if (e.move.movetype == .enpassant)
            {
                e.score = 40;
            }
            else if (e.move.movetype == .normal)
            {

            }
            e.score = 42;
        }
    }

    fn extract_next(self: *MovePicker) ?ExtMove
    {
        // TODO: params killer moves? pvmove? tt move?

        if (self.count == 0) return null;

        var best_idx = 0;
        var max_score: i32 = self.extmoves[self.current].score;

        const offset = self.current + 1;
        for (self.extmoves[offset..self.count], offset..) |e, idx|
        {
            const score = e.score;
            if (score > max_score)
            {
                best_idx = idx;
                max_score = score;
            }
        }

        if (best_idx != 0)
        {
            // swap.
            //std.mem.swap(self.extmoves[], a: *T, b: *T)
        }

        // //pub inline fn get_next_best(move_list: *MoveList, score_list: *ScoreList, i: usize) Move {
        //     var best_j = i;
        //     var max_score = score_list.scores[i];

        //     // Start from i+1 and iterate over the remaining elements
        //     for ((i + 1)..score_list.count) |j| {
        //         const score = score_list.scores[j];
        //         if (score > max_score) {
        //             best_j = j;
        //             max_score = score;
        //         }
        //     }

        //     // Swap if a better move is found
        //     if (best_j != i) {
        //         const best_move = move_list.moves[best_j];
        //         const best_score = score_list.scores[best_j];
        //         move_list.moves[best_j] = move_list.moves[i];
        //         score_list.scores[best_j] = score_list.scores[i];
        //         move_list.moves[i] = best_move;
        //         score_list.scores[i] = best_score;
        //     }

        //     return move_list.moves[i];
        // // }

    }
};

/// 64 bits. 2 bytes still available.
pub const ExtMove = packed struct
{
    move: Move,
    score: i32,
};

/// Criterium for ending the search.
pub const Termination = enum
{
    infinite,
    nodes,
    // time,
    movetime,
    depth,
};

pub const TimeControl = struct
{
    ponder: bool = false,
    btime: ?u64 = null,
    wtime: ?u64 = null,
    binc: ?u32 = 0,
    winc: ?u32 = 0,
    depth: ?u32 = null,
    nodes: ?u32 = null,
    mate: ?u32 = null,
    movetime: ?u64 = null,
    movestogo: ?u32 = null,
    infinite: bool = false,

    remaining_time: ?u64 = null,
    remaining_enemy_time: ?u64 = null,
    time_inc: ?u32 = null,
};

const MoveScoring = struct
{
    const promotion: i32 = 1_000_000;
    //const bad_capture
};


fn print_pv(pv_node: *const Node, depth: u16, seldepth: u16, nodes: u64, elapsed_nanoseconds: u64) void
{
    // info depth 1 seldepth 2 multipv 1 score cp 17 nodes 20 nps 20000 hashfull 0 tbhits 0 time 1 pv e2e4

    const ms = elapsed_nanoseconds / std.time.ns_per_ms;
    const nps: u64 = funcs.nps(nodes, elapsed_nanoseconds);

    lib.io.print
    (
        "info depth {} seldepth {} score cp {} nodes {} nps {} hashfull {} tbhits {} time {} pv {f}\n",
        .{ depth, seldepth, pv_node.score, nodes, nps, 0, 0, ms, pv_node.* }
    )
    catch wtf();
}

// info depth 3 seldepth 4 multipv 1 score cp 42 nodes 72 nps 72000 hashfull 0 tbhits 0 time 1 pv e2e4



// Case 1 – Fail Low

    // Child’s score <= alpha.

    // Meaning: The position is at most as good as alpha.

    // Return: score.

    // TT flag: UPPERBOUND.

    // No PV update.

// Case 2 – Window Improvement

    // Child’s alpha < score < beta.

    // Meaning: We found a better move that doesn’t cut off.

    // Return: score.

    // TT flag: EXACT.

    // PV update happens here.

// Case 3 – Fail High / Beta Cutoff

    // Child’s score >= beta.

    // Meaning: Opponent can force better than we’re willing to allow.

    // Return: score (not beta).

    // TT flag: LOWERBOUND.

    // No PV update (cutoff line isn’t guaranteed PV).