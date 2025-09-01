// zig fmt: off

//! --- NOT USED YET ---

////////////////////////////////////////////////////////////////////////////////////////////////////////////////

// TODO: before going multithreaded get ONE working single threaded search.

////////////////////////////////////////////////////////////////////////////////////////////////////////////////


// TODO: when timeout in the middle of an iteration is it possible we STILL found a better line (if at root)?

// TODO: maybe staged movegeneration
// * TT move
// * Winning captures + checks
// * Killer moves
// * Quiet moves.
// * Losing captures + bad moves (often last or skipped).
// * substract something from time to prevent time-loss.

// TODO: extensions (giving check? only 1 move possible? king attack?)

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const uci = @import("uci.zig");
const eval = @import("eval.zig");
const tt = @import("tt.zig");

const max_u64: u64 = std.math.maxInt(u64);

const Value = types.Value;
const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Move = types.Move;
const ExtMove = types.ExtMove;
const StateInfo = position.StateInfo;
const Position = position.Position;

const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;

const max_search_depth: u8 = types.max_search_depth;
const max_threads: u16 = types.max_threads;

const PV = lib.BoundedArray(Move, max_search_depth);
const Nodes = lib.BoundedArray(Node, max_search_depth);

/// Test phase only.
const use_tt: bool = true;

/// This is to become the "thread" manager.
pub const SearchManager = struct
{
    /// Shared table for all threads.
    transpositiontable: tt.TranspositionTable,

    /// The number of threads we use.
    num_threads: usize,

    /// Settings from uci, for timeout etc.
    searchparams: SearchParams,

    /// The actual worker.
    searcher: Search,

    pub fn init() !SearchManager
    {
        var mgr: SearchManager = undefined;

        mgr.transpositiontable = try .init(64);
        mgr.num_threads = 1;
        mgr.searchparams = .infinite_search_params;
        mgr.searcher = Search.init();

        return mgr;
    }

    pub fn deinit(self: *SearchManager) void
    {
        self.transpositiontable.deinit();
        self.searcher.deinit();
    }

    pub fn start(self: *SearchManager, pos: *const Position, go: *const uci.Go) !void
    {

        // Zig forum: https://ziggit.dev/t/im-too-dumb-for-zigs-new-io-interface/11645/56
        // we should have some atomic value to check if the threads are not already running.

        // have a var:
            // canceled: std.atomic.Value(bool),

        // check:
            // if (cw.canceled.load(.acquire)) stop

        // cancel:
            // pub fn cancel(w: *CancelableWriter) void
            // {
            //     w.canceled.store(true, .release);
            // }

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

pub const Search = struct
{
    /// Backlink ref. Only valid after start_search. TODO: Bad design still. Maybe use @fieldParentPtr().
    mgr: *SearchManager,

    /// Copied from mgr, for direct access.
    transpositiontable: *tt.TranspositionTable,

    /// Copied from mgr, for direct access.
    termination: Termination,

    /// A private copy of the engine pos.
    pos: Position,

    /// Rootmoves, resorted each iteration.
    rootmoves: MovePicker,

    /// Our node stack.
    nodes: Nodes,

    /// We always keep track of the best move a.s.a.p.
    best_move: Move,

    /// Updated after each iteration.
    pv_node: Node,

    // Updated inside alpha beta if at root.
    worker_pv_node: Node,

    /// The total amount
    processed_nodes: u64,

    /// The max reached depth, including quiet depth and possible extensions.
    seldepth: u16,

    /// Iterative deepening nr.
    iteration: u16,

    /// Timeout
    stopped: bool,

    /// Tracking search time.
    timer: utils.Timer,

    fn init() Search
    {
        return Search
        {
            .mgr = undefined,
            .transpositiontable = undefined,
            .termination = .infinite,
            .pos = .empty,
            .rootmoves = .init(),
            .nodes = .empty,
            .best_move = .empty,
            .pv_node = .init(),
            .worker_pv_node = .init(),
            .processed_nodes = 0,
            .seldepth = 0,
            .iteration = 0,
            .stopped = false,
            .timer = .empty,
        };
    }

    fn deinit(self: *Search) void
    {
        _ = self;
    }

    fn start_search(self: *Search, mgr: *SearchManager, input_pos: *const Position) void
    {
        // Shared stuff.
        self.mgr = mgr;
        self.termination = mgr.searchparams.termination;
        self.transpositiontable = &mgr.transpositiontable;

        // Reset stuff.
        self.timer.reset();
        self.rootmoves.reset();
        self.nodes = .empty;
        self.best_move = .empty;
        self.stopped = false;
        self.iteration = 0;
        self.processed_nodes = 0;
        self.seldepth = 0;
        self.pv_node.clear();
        self.worker_pv_node.clear();

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

        // TODO: when threading put this out!
        lib.io.print("info string used time {}\n", .{ self.timer.elapsed_ms() }) catch wtf();

        // UCI write bestmove after search.
        lib.io.print("bestmove {f}\n", .{ self.best_move }) catch wtf();
    }

    fn stop_search(self: *Search) void
    {
        _ = self;
    }

    fn iterative_deepening(self: *Search, comptime us: Color) void
    {
        const pos: *Position = &self.pos;

        self.rootmoves.generate_moves(pos, us);

        // Have something a.s.a.p.
        if (self.rootmoves.count > 0)
        {
            self.best_move = self.rootmoves.extmoves[0].move;
        }

        self.rootmoves.score_moves(pos);

        self.iteration = 1;
        var depth: u8 = 1;
        var best_score: Value = -types.infinity;
        while (true)
        {
            const score: Value = self.alpha_beta(true, us, depth, -types.infinity, types.infinity);

            // Discard result.
            if (self.stopped) break;

            // We found a better move.
            if (score > best_score)
            {
                best_score = score;
            }

            // Copy the last finished pv.
            self.pv_node.clone_from(self.get_node(2));
            self.best_move = self.pv_node.first_move();

            // And print.
            print_pv(&self.pv_node, self.iteration, self.seldepth, self.processed_nodes, self.timer.read(), self.transpositiontable.permille());

            if (self.check_stop(.iterative_deepening))
            {
                io.print("BREAK AT {} \n", .{ self.iteration}) catch wtf();
                break;
            }

            self.iteration += 1;
            depth += 1;
        }

        const ms = self.timer.read();
        // Final print.
        lib.io.print("P: ", .{}) catch wtf();
        print_pv(&self.pv_node, self.iteration, self.seldepth, self.processed_nodes, ms, self.transpositiontable.filled);
        lib.io.print("W: ", .{}) catch wtf();
        print_pv(&self.worker_pv_node, self.iteration, self.seldepth, self.processed_nodes, ms, self.transpositiontable.filled);
    }

    fn alpha_beta(self: *Search, comptime is_root: bool, comptime us: Color, depth: u8, input_alpha: Value, beta: Value) Value
    {
        const them = comptime us.opp();
        const pos: *Position = &self.pos;
        const key: u64 = pos.state.key;
        const ply: u16 = pos.ply;

        self.processed_nodes += 1;
        self.seldepth = @max(self.seldepth, ply);

        // Timeout?
        if (self.check_stop(.alpha_beta)) return 0;

        // TT probe.
        if (use_tt and !is_root)
        {
            if (self.transpositiontable.probe(key, depth, ply, input_alpha, beta)) |result|
            {
                return result.eval;
            }
        }

        // Requested depth reached: go quiet.
        if (depth == 0) return self.quiet(is_root, us, 0, input_alpha, beta);

        // Too deep.
        if (ply >= max_search_depth) return eval.evaluate(pos, false);

        const node: *Node = self.get_node(ply + 2);
        const childnode: *const Node = get_child_node(node);
        const parentnode: *const Node = get_parent_node(node);
        _ = parentnode;

        // Clear current node.
        node.clear();

        if (pos.state.rule50 >= 100) return 0;

        const is_check: bool = pos.is_check();

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
            if (is_check) return -types.mate + pos.ply else return types.stalemate;
        }

        var st: StateInfo = undefined;
        var alpha: Value = input_alpha;

        // Go trough the moves.
        for (0..movepicker.count) |move_idx|
        {
            const e: ExtMove = movepicker.extract_next(move_idx);
            const move: Move = e.move;

            pos.make_move(us, &st, move);
            const score: Value = -self.alpha_beta(false, them, depth - 1, -beta, -alpha);
            pos.unmake_move(us);

            // Discard result.
            if (self.stopped) break;

            // Higher alpha. A new best move is found.
            if (score > alpha)
            {
                // Beta cutoff / fail high.
                if (score >= beta)
                {
                    // TT store
                    if (use_tt) self.transpositiontable.store(.Beta, key, depth, ply, move, score);
                    return score;
                }
                node.score = score;
                alpha = score;
                update_local_pv(move, score, node, childnode);
                if (is_root)
                {
                    self.worker_pv_node.clone_from(node);
                }
            }
        }

        if (use_tt)
        {
            // TT store
            const kind: tt.Kind = if (alpha <= input_alpha) .Alpha else .Exact;
            self.transpositiontable.store(kind, key, depth, ply, node.first_move(), alpha);
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

        self.processed_nodes += 1;
        self.seldepth = @max(self.seldepth, pos.ply);

        if (self.check_stop(.quiet)) return 0;

        const static_score: Value = eval.evaluate(pos, false);

        // Fail high.
        if (static_score >= beta) return static_score;

        // Raise alpha.
        if (static_score > alpha) alpha = static_score;

        // Too deep.
        if (ply >= max_search_depth) return static_score;

        var movepicker: MovePicker = .init();
        movepicker.generate_captures(pos, us);

        // Mate or stalemate?
        if (movepicker.count == 0)
        {
            if (in_check) // TODO: only at root?
            {
                 return -types.mate + ply;
            }
            return alpha;
        }

        for (movepicker.slice()) |extmove|
        {
            // TODO: skip bad captures (if not in check?).
            const move: Move = extmove.move;
            var st: StateInfo = undefined;
            pos.make_move(us, &st, move);
            const score: Value = -self.quiet(false, them, quiet_depth + 1, -beta, -alpha);
            pos.unmake_move(us);

            // Discard result.
            if (self.stopped) break;

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

    /// Append the line from 'childnode' to 'current node'
    fn update_local_pv(bestmove: Move, score: Value, node: *Node, childnode: *const Node) void
    {
        node.pv.len = 1;
        node.pv.buffer[0] = bestmove;
        node.pv.append_slice_assume_capacity(childnode.pv.slice());
        node.score = score;
    }

    /// In case of a stop the last search result must be discarded immediately.
    fn check_stop(self: *Search, comptime callsite: CallSite) bool
    {
        if (self.stopped) return true;
        if (self.termination == .infinite) return false;

        // This one is called *after* nodes is incremented.
        if (callsite == .alpha_beta or callsite == .quiet)
        {
            switch (self.termination)
            {
                .nodes =>
                {
                    if (self.processed_nodes >= self.mgr.searchparams.max_nodes)
                    {
                        self.stopped = true;
                        return true;
                    }
                    return false;
                },
                .movetime =>
                {
                    if (self.processed_nodes & 2047 == 0 and self.timer.elapsed_ms() >= self.mgr.searchparams.max_movetime_ms)
                    {
                        self.stopped = true;
                        return true;
                    }
                    return false;
                },
                else =>
                {
                    return false;
                },
            }
        }
        // This one is called *before* the iteration is incremented.
        else if (callsite == .iterative_deepening)
        {
            switch (self.termination)
            {
                .depth =>
                {
                    if (self.iteration >= self.mgr.searchparams.max_depth)
                    {
                        self.stopped = true;
                        return true;
                    }
                    return false;
                },
                .movetime =>
                {
                    if (self.timer.elapsed_ms() + 5 >= self.mgr.searchparams.max_movetime_ms)
                    {
                        self.stopped = true;
                        return true;
                    }
                    return false;
                },
                else =>
                {
                    return false;
                },
            }
        }

        // We should handle everything above.
        unreachable;
    }

    fn get_node(self: *Search, index: u16) *Node
    {
        return &self.nodes.buffer[index];
    }

    /// NOTE: I think we just use `ply + 1` if it is the same speed.
    fn get_child_node(node: *Node) *const Node
    {
        return funcs.ptr_add(Node, node, 1);
    }

    /// NOTE: I think we just use `ply - 1` if it is the same speed.
    fn get_parent_node(node: *Node) *const Node
    {
        return funcs.ptr_sub(Node, node, 1);
    }
};

pub const Node = struct
{
    const empty: Node = .{};

    /// Local PV during search.
    pv: PV = .{},

    /// Current eval.
    score: Value = 0,

    /// Can be set by childnode, when searching.
    will_give_check: bool = false,
    will_give_mate: bool = false,
    will_give_stalemate: bool = false,

    fn init() Node
    {
        return .{};
    }

    fn clear(self: *Node) void
    {
        self.pv.len = 0;
        self.score = 0;
        self.will_give_check = false;
        self.will_give_mate = false;
        self.will_give_stalemate = false;
    }

    fn first_move(self: *const Node) Move
    {
        return self.pv.buffer[0];
    }

    fn clone_from(self: *Node, other: *const Node) void
    {
        self.pv.len = 0;
        self.pv.append_slice_assume_capacity(other.pv.slice());
        self.score = other.score;
    }

    /// Zig-format for UCI output the PV.
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

    fn slice(self: *MovePicker) []ExtMove
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

    /// TODO: params killer moves? pvmove? tt move?
    /// Give each generated move an initial heuristic score in the attempt to get the best or most interesting moves at the beginning.\
    /// Must obviously be called *before* the move is made.
    fn score_moves(self: *MovePicker, pos: *const Position) void
    {
        const promotion    : Value =  1_000_000;
        const capture      : Value =    200_000;
        const good_capture : Value =    500_000;
        const bad_capture  : Value = -1_000_000;
        const castle       : Value =    100_000;

        for (self.slice()) |*e|
        {
            //const capt: PieceType = if (e.move.movetype == .enpassant) P else if (e.move)
            const m: Move = e.move;
            //const moved_piece: Piece = pos.board[m.from.u];

            switch (m.type)
            {
                .normal =>
                {
                    const captured_piece: Piece = pos.board[m.to.u];
                    if (captured_piece.is_piece())
                    {
                        const s = eval.see_score(pos, m);
                        if (s >= 0) e.score = good_capture + (s * 10) else e.score = bad_capture - (s * 10);
                    }
                    else
                    {
                        // e.score = eval.
                        //const dist =
                    }
                },
                .promotion =>
                {
                    //const captured_piece: Piece = self.board[m.to.u];
                    e.score = promotion + (m.promoted().value() * 10);
                    const captured_piece: Piece = pos.board[m.to.u];
                    if (captured_piece.is_piece())
                    {
                        e.score += 10; // WIP. all these numbers.......
                    }
                },
                .enpassant =>
                {
                    //const captured_piece: Piece = Piece.make(PieceType.PAWN, pos.to_move.opp());
                    e.score = capture;
                },
                .castle =>
                {
                    e.score = castle;
                },
            }
        }
    }

    fn extract_next(self: *MovePicker, current_idx: usize) ExtMove
    {
        const ptr: [*]ExtMove = &self.extmoves;

        var best_idx: usize = current_idx;
        var max_score: Value = ptr[current_idx].score;

        for (current_idx + 1..self.count) |idx|
        {
            const score = ptr[idx].score;
            if (score > max_score)
            {
                max_score = score;
                best_idx = idx;
            }
        }

        if (best_idx != current_idx)
        {
            std.mem.swap(ExtMove, &ptr[best_idx], &ptr[current_idx]);
        }

        return ptr[current_idx];
    }
};

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

/// Criterium for ending the search.
pub const Termination = enum
{
    infinite,
    nodes,
    // time,
    movetime,
    depth,
};

const CallSite = enum
{
    iterative_deepening,
    alpha_beta,
    quiet,
};

fn print_pv(pv_node: *const Node, depth: u16, seldepth: u16, nodes: u64, elapsed_nanoseconds: u64, hash_full: usize) void
{
    // info depth 1 seldepth 2 multipv 1 score cp 17 nodes 20 nps 20000 hashfull 0 tbhits 0 time 1 pv e2e4

    const ms = elapsed_nanoseconds / std.time.ns_per_ms;
    const nps: u64 = funcs.nps(nodes, elapsed_nanoseconds);

    lib.io.print
    (
        "info depth {} seldepth {} score cp {} nodes {} nps {} hashfull {} tbhits {} time {} pv {f}\n",
        .{ depth, seldepth, pv_node.score, nodes, nps, hash_full, 0, ms, pv_node.* }
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