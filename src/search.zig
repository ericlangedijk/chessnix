// zig fmt: off

//! The heart of the engine: search.

const std = @import("std");
const lib = @import("lib.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const uci = @import("uci.zig");
const eval = @import("eval.zig");
const tt = @import("tt.zig");

const assert = std.debug.assert;
const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;


const Value = types.Value;
const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const StateInfo = position.StateInfo;
const Position = position.Position;
const Entry = tt.Entry;
const Bound = tt.Bound;

const max_u64: u64 = std.math.maxInt(u64);

const max_search_depth: u8 = types.max_search_depth;
const max_threads: u16 = types.max_threads;

const PV = lib.BoundedArray(Move, max_search_depth);
const Nodes = lib.BoundedArray(Node, max_search_depth);

/// This is to become the "thread" manager.
pub const SearchManager = struct {
    /// Shared table for all threads.
    transpositiontable: tt.TranspositionTable,
    /// The number of threads we use.
    num_threads: usize,
    /// Settings from uci, for timeout etc.
    searchparams: SearchParams,
    /// The actual worker.
    searcher: Search,

    pub fn init() !SearchManager {
        var mgr: SearchManager = undefined;
        mgr.transpositiontable = try .init(64);
        mgr.num_threads = 1;
        mgr.searchparams = .infinite_search_params;
        mgr.searcher = Search.init();
        return mgr;
    }

    pub fn deinit(self: *SearchManager) void {
        self.transpositiontable.deinit();
        self.searcher.deinit();
    }

    pub fn start(self: *SearchManager, pos: *const Position, go: *const uci.Go) !void {

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

    pub fn stop(self: *SearchManager) !void {
        // we should have some atomic value to stop the thread(s)
        _ = self;
        try lib.io.print("WE SHOULD STTOP THINKING\n", .{});
    }
};

pub const Search = struct {
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
    /// For scoring quiet moves. Indexing [Piece][Square].
    quiet_heuristic_history: [16][64]Value,
    /// We always keep track of the best move a.s.a.p.
    best_move: Move,
    /// Updated after each iteration.
    pv_node: Node,
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

    fn init() Search {
        return .{
            .mgr = undefined,
            .transpositiontable = undefined,
            .termination = .infinite,
            .pos = .empty,
            .rootmoves = .init(true),
            .nodes = .empty,
            .quiet_heuristic_history = std.mem.zeroes([16][64]Value),
            .best_move = .empty,
            .pv_node = .init(),
            .processed_nodes = 0,
            .seldepth = 0,
            .iteration = 0,
            .stopped = false,
            .timer = .empty,
        };
    }

    fn deinit(self: *Search) void {
        _ = self;
    }

    fn start_search(self: *Search, mgr: *SearchManager, input_pos: *const Position) void {
        // Shared stuff.
        self.mgr = mgr;
        self.termination = mgr.searchparams.termination;
        self.transpositiontable = &mgr.transpositiontable;
        self.transpositiontable.increase_age();

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
        self.quiet_heuristic_history = std.mem.zeroes([16][64]Value);

        // Create our private copy of the position.
        var rootstate: StateInfo = undefined;
        self.pos.copy_from(input_pos, &rootstate);

        // Append our rootstate to the history of the source position. TODO: not needed?
        // rootstate.prev = input_pos.state;


        //const e1: tt.Entry = self.transpositiontable.get_entry(self.pos.state.key);
        //lib.io.debugprint("before tt: {f} score {} depth {}\n", .{e1.move, e1.score, e1.depth});

        switch (self.pos.to_move.e) {
            .white => self.iterative_deepening(Color.WHITE),
            .black => self.iterative_deepening(Color.BLACK),
        }

        // TODO: when threading put this out!

        // Debug print
        print_san_pv(input_pos, &self.pv_node) catch wtf();
        // UCI write bestmove after search.
        lib.io.print("bestmove {f}\n", .{ self.best_move }) catch wtf();
    }

    fn stop_search(self: *Search) void {
        _ = self;
    }

    fn iterative_deepening(self: *Search, comptime us: Color) void {
        const pos: *Position = &self.pos;
        self.rootmoves.generate_moves(pos, us);
        self.rootmoves.process_and_score_moves(self, Move.empty);
        self.rootmoves.sort();
        // Have something a.s.a.p.
        if (self.rootmoves.count > 0) self.best_move = self.rootmoves.extmoves[0].move;

        // io.debugprint("start: ", .{});
        // for (self.rootmoves.slice(), 0..) |e, i|
        // {
        //     //const e = self.rootmoves.extract_next(idx, .empty);
        //     io.debugprint("{f} {}, ", .{e.move, e.score});
        //     if (i > 10) break;
        // }
        // io.debugprint("\n", .{});

        //if (true) return;

        var depth: u8 = 1;
        var best_score: Value = -types.infinity;
        while (true) {
            self.iteration = depth;
            const score: Value = self.alpha_beta(true, us, depth, -types.infinity, types.infinity);
            // Discard result if timeout.
            if (!self.stopped)
            {
                // We found a better move.
                if (score > best_score) {
                    best_score = score;
                }
                // Copy the last finished pv.
                self.pv_node.copy_from(self.get_node(2));
                self.best_move = self.pv_node.first_move();
            }
            self.rootmoves.sort();
            print_pv(&self.pv_node, self.iteration, self.seldepth, self.processed_nodes, self.timer.read(), self.transpositiontable.permille());
            if (self.check_stop(.iterative_deepening)) break;
            if (depth >= max_search_depth) break;
            depth += 1;
        }
    }

    fn alpha_beta(self: *Search, comptime is_root: bool, comptime us: Color, depth: u8, alpha: Value, beta: Value) Value {
        const them = comptime us.opp();
        const pos: *Position = &self.pos;
        const key: u64 = pos.state.key;
        const ply: u16 = pos.ply;

        self.processed_nodes += 1;
        self.seldepth = @max(self.seldepth, ply);

        // Timeout?
        if (self.check_stop(.alpha_beta)) return 0;

        // Probe.
        var hashmove: Move = .empty;
        if (!is_root) {
            if (self.tt_probe(key, depth, ply, alpha, beta, &hashmove)) |entry| {
                return entry.score;
            }
        }

        // Requested depth reached: plunge into quiescence.
        if (depth == 0) return self.quiet(is_root, us, alpha, beta);

        // Too deep.
        if (ply >= max_search_depth) return eval.evaluate(pos, false);

        const node: *Node = self.get_node(ply + 2);
        const childnode: *const Node = self.get_node(ply + 3);

        // Clear current node.
        node.clear();

        // Draw by rule.
        if (pos.state.rule50 >= 100) return 0;

        const is_check: bool = pos.is_check();

        // Use rootmoves or generate on stack.
        const movepicker: *MovePicker =
            if (is_root) &self.rootmoves
            else local: {
                var mp = MovePicker.init(false);
                mp.generate_moves(pos, us);
                mp.process_and_score_moves(self, hashmove);
                break :local &mp;
            };

        // Is this checkmate or stalemate?
        if (movepicker.count == 0) {
            const score = if (is_check) -types.mate + pos.ply else types.stalemate;
            self.tt_store(.Exact, key, depth, ply, .empty, score);
            return score;
        }

        var best_score: Value = alpha;
        var best: Move = .empty;
        var st: StateInfo = undefined;

        // Go trough the moves.
        for (0..movepicker.count) |move_idx| {
            const extmove_ptr: *ExtMove = movepicker.extract_next(move_idx);
            const e: ExtMove = extmove_ptr.*;
            pos.make_move(us, &st, e.move);
            const score: Value = -self.alpha_beta(false, them, depth - 1, -beta, -best_score);
            pos.unmake_move(us);

            // Adjust score if rootmoves for move ordering in the next iteration.
            if (is_root) {
                extmove_ptr.score = score;
            }

            // Discard result.
            if (self.stopped) break;

            // Better move.
            if (score > best_score) {
                // Fail high.
                if (score >= beta) {
                    if (e.info.is_quiet) self.record_quietmove_beta_cutoff(depth, e);
                    self.tt_store(.Lower, key, depth, ply, e.move, score);
                    return score;
                }
                best = e.move;
                best_score = score;
                node.update_pv(e.move, score, childnode);
            }
        }

        if (!best.is_empty()) {
            const bound: Bound = if (best_score <= alpha) .Upper else .Exact;
            self.tt_store(bound, key, depth, ply, best, best_score);
        }
        return best_score;
    }

    fn quiet(self: *Search, comptime is_root: bool, comptime us: Color, input_alpha: Value, beta: Value) Value {
        assert(is_root == false); // TODO: so remove this nonsense!!
        // TT probing and storing is done here with depth zero.
         //_ = is_root;

        const them = comptime us.opp();
        const pos: *Position = &self.pos;
        //const key: u64 = pos.state.key;
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

        //const hashmove: Move = .empty;
        // // TT probe quiet
        // if (self.tt_probe(key, 0, ply, alpha, beta, &hashmove)) |result|
        // {
        //     return result.score;
        // }

        var movepicker: MovePicker = .init(false);
        movepicker.generate_captures(pos, us);
        movepicker.process_and_score_moves(self, Move.empty);

        // Mate or stalemate?
        if (movepicker.count == 0) {
            if (in_check) {
                return -types.mate + ply;
            }
            return alpha;
        }

        var best: Move = .empty;
        for (0.. movepicker.count) |move_idx| {
            // TODO: skip bad captures (if not in check?).
            const extmove_ptr: *ExtMove = movepicker.extract_next(move_idx);
            const e: ExtMove = extmove_ptr.*;
            var st: StateInfo = undefined;
            pos.make_move(us, &st, e.move);
            const score: Value = -self.quiet(false, them, -beta, -alpha);
            pos.unmake_move(us);

            // Discard result.
            if (self.stopped) break;

            // Better move.
            if (score > alpha) {
                // Fail high.
                if (score >= beta) {
                    return score;
                }
                best = e.move;
                alpha = score;
            }
        }
        return alpha;
    }

    fn record_quietmove_beta_cutoff(self: *Search, depth: Value, e: ExtMove) void {
        self.quiet_heuristic_history[e.info.moved_piece.u][e.move.to.u] += depth * depth;
    }

    fn tt_store(self: *Search, bound: Bound, key: u64, depth: u8, ply: u16, move: Move, score: Value) void {
        const adjusted_score = tt.get_adjusted_score_for_store(score, ply);
        self.transpositiontable.store(bound, key, depth, move, adjusted_score);
    }

    /// Returns a Entry when usable for scoring. The score of the entry is adjusted for the ply when mating distance is there.
    /// Hashmove can be used for move ordering and is always the entry's move except when nothing found. Then it is empty.
    fn tt_probe(self: *Search, key: u64, depth: u8, ply: u16, alpha: Value, beta: Value, hashmove: *Move) ?Entry
    {
        var entry: Entry = self.transpositiontable.probe(key) orelse {
            hashmove.* = .empty;
            return null;
        };

        hashmove.* = entry.move;

        // Only use hashmove in this case for move ordering.
        if (entry.age != self.transpositiontable.age) {
            return null;
        }

        switch (entry.bound) {
            .None => {
                unreachable;
            },
            .Exact => {
                return if (entry.depth >= depth) entry else null;
            },
            .Lower => {
                const adjusted_score: Value = tt.get_adjusted_score_for_probe(entry.score, ply);
                if (adjusted_score >= beta) {
                    entry.score = adjusted_score;
                    return entry;
                }
                return null;
            },
            .Upper =>
            {
                if (entry.depth >= depth) {
                    const adjusted_score: Value = tt.get_adjusted_score_for_probe(entry.score, ply);
                    if (adjusted_score <= alpha) {
                        entry.score = adjusted_score;
                        return entry;
                    }
                }
                return null;
            },
        }
    }

    /// In case of a stop the last search result must be discarded immediately.
    fn check_stop(self: *Search, comptime callsite: CallSite) bool
    {
        if (self.stopped) return true;
        if (self.termination == .infinite) return false;

        // This one is called *after* nodes is incremented.
        if (callsite == .alpha_beta or callsite == .quiet) {
            switch (self.termination)
            {
                .nodes => {
                    if (self.processed_nodes >= self.mgr.searchparams.max_nodes) {
                        self.stopped = true;
                        return true;
                    }
                    return false;
                },
                .movetime => {
                    if (self.processed_nodes & 2047 == 0 and self.timer.elapsed_ms() >= self.mgr.searchparams.max_movetime_ms) {
                        self.stopped = true;
                        return true;
                    }
                    return false;
                },
                else => {
                    return false;
                },
            }
        }
        // This one is called *before* the iteration is incremented.
        else if (callsite == .iterative_deepening) {
            switch (self.termination) {
                .depth => {
                    if (self.iteration >= self.mgr.searchparams.max_depth) {
                        self.stopped = true;
                        return true;
                    }
                    return false;
                },
                .movetime => {
                    if (self.timer.elapsed_ms() + 5 >= self.mgr.searchparams.max_movetime_ms) {
                        self.stopped = true;
                        return true;
                    }
                    return false;
                },
                else => {
                    return false;
                },
            }
        }
        // We should handle everything above.
        unreachable;
    }

    fn get_node(self: *Search, index: u16) *Node {
        return &self.nodes.buffer[index];
    }

    /// NOTE: I think we just use `ply + 1` if it is the same speed.
    fn get_child_node(node: *Node) *const Node {
        return funcs.ptr_add(Node, node, 1);
    }

    /// NOTE: I think we just use `ply - 1` if it is the same speed.
    fn get_parent_node(node: *Node) *const Node {
        return funcs.ptr_sub(Node, node, 1);
    }
};

pub const Node = struct {
    const empty: Node = .{};
    /// Local PV during search.
    pv: PV = .{},
    /// Current eval.
    score: Value = 0,

    fn init() Node {
        return .{};
    }

    fn clear(self: *Node) void {
        self.pv.len = 0;
        self.score = 0;
    }

    fn first_move(self: *const Node) Move {
        return self.pv.buffer[0];
    }

    /// Sets `pv` to `bestmove + childnode.pv` and sets `self.score` to `score`.
    fn update_pv(self: *Node, bestmove: Move, score: Value, childnode: *const Node) void {
        self.pv.len = 1;
        self.pv.buffer[0] = bestmove;
        self.pv.append_slice_assume_capacity(childnode.pv.slice());
        self.score = score;
    }

    /// Makes a raw copy.
    fn copy_from(self: *Node, other: *const Node) void {
        self.pv.len = 0;
        self.pv.append_slice_assume_capacity(other.pv.slice());
        self.score = other.score;
    }

    /// Zig-format for UCI output the PV.
    pub fn format(self: Node, writer: *std.io.Writer) std.io.Writer.Error!void {
        const len: usize = self.pv.len;
        if (len == 0) return;
        for (self.pv.buffer[0..len - 1]) |move| {
            try writer.print("{f} ", .{ move });
        }
        const last: Move = self.pv.buffer[len - 1];
        try writer.print("{f}", .{ last });
    }
};

const MovePicker = struct {
    extmoves: [types.max_move_count]ExtMove,
    count: u8,
    current: u8,
    is_root: bool,
    is_sorted: bool,

    fn init(is_root: bool) MovePicker {
        return .{
            .extmoves = undefined,
            .count = 0,
            .current = 0,
            .is_root = is_root,
            .is_sorted = false,
        };
    }

    /// Required function.
    pub fn reset(self: *MovePicker) void {
        self.count = 0;
        self.current = 0;
        self.is_sorted = false;
    }

    /// Required function.
    pub fn store(self: *MovePicker, move: Move) ?void {
        self.extmoves[self.current] = ExtMove{ .move = move, .score = 0, .info = .empty };
        self.count += 1;
        self.current += 1;
    }

    fn copy_from(self: *MovePicker, other: *const MovePicker) void {
        const cnt: u8 = other.count;
        @memcpy(self.extmoves[0..cnt], other.extmoves[0..cnt]);
        self.count = cnt;
        self.current = 0;
    }

    fn generate_moves(self: *MovePicker, pos: *const Position, comptime us: Color) void {
        pos.generate_moves(us, self);
    }

    /// Generate captures only, but if in check generate all.
    fn generate_captures(self: *MovePicker, pos: *const Position, comptime us: Color) void {
        pos.generate_captures(us, self);
    }

    /// Fill ExtMove info and give each generated move an initial heuristic score. Crucial for the search algorithm.
    fn process_and_score_moves(self: *MovePicker, search: *const Search, hashmove: Move) void {
        const pos: *const Position = &search.pos;

        // 1. TT move
        // 2. Winning captures (MVV-LVA or SEE)
        // 3. Promotions
        // 4. Killer 1
        // 5. Killer 2
        // 6. Countermove
        // 7. Quiet moves sorted by history
        // 8. Losing captures

        const promotion    : Value =  1_000_000;
        const capture      : Value =    200_000;
        const good_capture : Value =    500_000;
        const bad_capture  : Value = -1_000_000;
        const castle       : Value =    100_000;

        for (self.slice()) |*e| {
            e.info.is_processed = true;
            const m: Move = e.move;
            e.info.moved_piece = pos.board[m.from.u];

            switch (m.type) {
                .normal => {
                    e.info.captured_piece = pos.board[m.to.u];
                    if (e.info.captured_piece.is_piece()) {
                        const see = eval.see_score(pos, m);
                        if (see >= 0) e.score = good_capture + (see * 10) else e.score = bad_capture + (see * 10);
                    }
                    else {
                        e.info.is_quiet = true;
                        e.score = search.quiet_heuristic_history[e.info.moved_piece.u][m.to.u];
                    }
                },
                .promotion => {
                    e.info.captured_piece = pos.board[m.to.u];
                    e.score = promotion + (m.promoted().value() * 10);
                    if (e.info.captured_piece.is_piece()) {
                        e.score += e.info.captured_piece.value(); // WIP. all these numbers.......
                    }
                },
                .enpassant => {
                    e.info.captured_piece = Piece.create_pawn(pos.to_move.opp());
                    e.score = capture;
                },
                .castle => {
                    e.info.is_quiet = true;
                    e.score = castle;
                    // TODO: e.score = search.quiet_heuristic_history[m.info.movedpiece.u][m.to];
                },
            }
            if (e.move == hashmove) e.score += 1000000;
        }
    }

    fn extract_next(self: *MovePicker, current_idx: usize) *ExtMove {
        const ptr: [*]ExtMove = &self.extmoves;
        if (!self.is_sorted) {
            var best_idx: usize = current_idx;
            var max_score: Value = ptr[current_idx].score;
            for (current_idx + 1..self.count) |idx| {
                const e = ptr[idx];
                if (e.score > max_score) {
                    max_score = e.score;
                    best_idx = idx;
                }
            }
            if (best_idx != current_idx) {
                std.mem.swap(ExtMove, &ptr[best_idx], &ptr[current_idx]);
            }
        }
        return &ptr[current_idx];
    }

    fn slice(self: *MovePicker) []ExtMove {
        return self.extmoves[0..self.count];
    }

    fn sort(self: *MovePicker) void {
        std.mem.sort(ExtMove, self.slice(), {}, less_then);
        self.is_sorted = true;
    }

    fn less_then(_: void, a: ExtMove, b: ExtMove) bool {
        return a.score > b.score;
    }
};

pub const SearchParams = struct
{
    const infinite_search_params: SearchParams = .{
        .termination = .infinite,
        .max_depth = types.max_search_depth,
        .max_nodes = max_u64,
        .max_movetime_ms = max_u64,
    };

    termination: Termination,
    max_depth: u8,
    max_nodes: u64,
    max_movetime_ms: u64,

    /// Convert the UCI go params to something usable for our search.
    pub fn init(go: *const uci.Go) SearchParams {
        // (1) infinite.
        if (go.infinite != null) {
            return .infinite_search_params;
        }
        // (2) depth.
        if (go.depth) |depth| {
            return .{
                .termination = .depth,
                .max_depth = @truncate(@min(depth, types.max_search_depth)),
                .max_nodes = max_u64,
                .max_movetime_ms = max_u64,
            };
        }
        // (3) nodes.
        if (go.nodes) |nodes| {
            return .{
                .termination = .nodes,
                .max_depth = types.max_search_depth,
                .max_nodes = nodes,
                .max_movetime_ms = max_u64,
            };
        }
        // (4) movetime.
        if (go.movetime) |movetime| {
            return .{
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
pub const Termination = enum { infinite, nodes, time, movetime, depth };
const CallSite = enum { iterative_deepening, alpha_beta, quiet };

fn print_pv(pv_node: *const Node, depth: u16, seldepth: u16, nodes: u64, elapsed_nanoseconds: u64, hash_full: usize) void {
    const ms = elapsed_nanoseconds / std.time.ns_per_ms;
    const nps: u64 = funcs.nps(nodes, elapsed_nanoseconds);
    lib.io.print (
        "info depth {} seldepth {} score cp {} nodes {} nps {} hashfull {} tbhits {} time {} pv {f}\n",
        .{ depth, seldepth, pv_node.score, nodes, nps, hash_full, 0, ms, pv_node.* }
    )
    catch wtf();
}

fn print_san_pv(input_pos: *const Position, pv_node: *const Node) !void {
    if (pv_node.pv.len == 0 ) return;
    const san = @import("san.zig");
    const ev = if (input_pos.to_move.e == .white) pv_node.score else -pv_node.score;
    try lib.io.print("eval {} ", .{ ev });
    try san.write_san_line(input_pos, pv_node.pv.slice(), lib.io.out);
    lib.io.print("\n", .{}) catch wtf();
}

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