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
const StateInfo = position.StateInfo;
const Position = position.Position;
const Entry = tt.Entry;
const Bound = tt.Bound;

const P: PieceType = types.P;
const N: PieceType = types.N;
const B: PieceType = types.B;
const R: PieceType = types.R;
const Q: PieceType = types.Q;
const K: PieceType = types.K;

const max_u64: u64 = std.math.maxInt(u64);
const max_u32: u64 = std.math.maxInt(u32);
const max_search_depth: u8 = types.max_search_depth;
const max_threads: u16 = types.max_threads;
const infinity = types.infinity;
const mate = types.mate;
const stalemate = types.stalemate;

const PV = lib.BoundedArray(Move, max_search_depth + 2);
const Nodes = lib.BoundedArray(Node, max_search_depth + 2);

/// This is to become the "thread" manager.
pub const Engine = struct {
    /// Shared tt for all threads.
    transpositiontable: tt.TranspositionTable,
    /// Threading.
    cancelled: std.atomic.Value(bool),
    /// The number of threads we use.
    num_threads: usize,
    /// The controller thread, managing the search threads.
    controller_thread: ?std.Thread,
    /// The working thread. Later this will be more threads. We start with one.
    search_thread: ?std.Thread,
    /// The global state list for our position.
    history: [types.max_game_length]StateInfo,
    /// Our source position.
    pos: Position,
    /// UCI options.
    options: Options,
    /// Settings like timeout etc.
    searchparams: SearchParams,
    /// The actual worker.
    searcher: Searcher,
    /// No output if true.
    muted: bool,
    //tracker: Tracker,

    /// Engine is created on the heap.
    pub fn create() !*Engine {
        var engine: *Engine = try ctx.galloc.create(Engine);
        engine.transpositiontable = try .init(Options.default_hash_size);
        engine.cancelled = std.atomic.Value(bool).init(false);
        engine.num_threads = 1;
        engine.controller_thread = null;
        engine.search_thread = null;
        engine.history = @splat(StateInfo.empty);
        engine.pos = .empty;
        engine.pos.set_startpos(&engine.history[0]);
        engine.options = Options.default;
        engine.searchparams = SearchParams.default;
        engine.searcher = Searcher.init();
        //engine.tracker = try .init();
        return engine;
    }

    pub fn destroy(self: *Engine) void {
        self.transpositiontable.deinit();
        //self.tracker.deinit();
        ctx.galloc.destroy(self);
    }

    pub fn ucinewgame(self: *Engine) !void {
        self.set_startpos();
        self.clear_for_new_game();
    }

    pub fn is_running(self: *const Engine) bool {
        return (self.controller_thread != null);
    }

    fn is_cancelled(self: *const Engine) bool {
        return self.cancelled.load(.acquire);
    }

    /// Start threaded search.
    pub fn start(self: *Engine, go_params: *const uci.Go) !void {
        if (self.is_running()) return;
        self.transpositiontable.clear();
        self.searchparams = .init(go_params, self.pos.to_move);
        self.cancelled.store(false, .release);
        self.transpositiontable.inc_age();
        self.controller_thread = try std.Thread.spawn(.{}, run_controller, .{ self });
    }

    /// Stop threaded search.
    pub fn stop(self: *Engine) !void {
        if (!self.is_running()) return;
        // Force the searcher to stop. Only the engine may do a atomic store to the searchers.
        @atomicStore(bool, &self.searcher.stopped, true, .release);
        self.cancelled.store(true, .release);
    }

    fn run_controller(self: *Engine) !void {
        defer self.controller_thread = null;
        self.search_thread = try std.Thread.spawn(.{}, Searcher.start, .{ &self.searcher, self });
        if (self.search_thread) |th| {
            th.join();
            self.search_thread = null;
        }
    }

    pub fn clear_for_new_game(self: *Engine) void {
        if (self.is_running()) return;
        self.transpositiontable.clear();
    }

    pub fn set_startpos(self: *Engine) void {
        if (self.is_running()) return;
        self.pos.set_startpos(&self.history[0]);
    }

    /// After an illegal move we stop without crashing.
    pub fn set_startpos_with_optional_moves(self: *Engine, moves: ?[]const u8) !void {
        if (self.is_running()) return;
        self.set_startpos();
        if (moves) |str| {
            self.parse_moves(str);
        }
    }

    /// Sets the position from fen + moves.
    /// * If fen is illegal we crash.
    /// * If fen is null the startpostiion will be set.
    /// * After an illegal move we stop without crashing.
    pub fn set_position(self: *Engine, fen: ?[]const u8, moves: ?[]const u8) !void {
        if (self.is_running()) return;
        const f = fen orelse {
            self.set_startpos();
            return;
        };
        try self.pos.set(&self.history[0], f);
        if (moves) |m| {
            self.parse_moves(m);
        }
    }

    // If we have any moves, make them. We stop if we encounter an illegal move.
    fn parse_moves(self: *Engine, moves: []const u8) void {
        var tokenizer = std.mem.tokenizeScalar(u8, moves, ' ');
        var idx: usize = 1;
        while (tokenizer.next()) |m| {
            const move: Move = self.pos.parse_move(m) catch break;
            self.pos.lazy_do_move(&self.history[idx], move);
            idx += 1;
        }
    }

    // /// Engine test function.
    // pub fn give_best_move(self: *Engine, movetime: u32) Move {
    //     self.searchparams = .infinite_search_params;
    //     self.searchparams.termination = .movetime;
    //     self.searchparams.max_movetime = movetime;
    //     self.muted = true;
    //     self.searcher.start(self);
    //     return self.searcher.best_node.first_move();
    // }
};

pub const Stats = struct {
};

pub const Searcher = struct {
    /// Backlink ref. Only valid after start_search.
    engine: *Engine,
    /// Copied from mgr, for direct access.
    transpositiontable: *tt.TranspositionTable,
    /// Copied from mgr, for direct access.
    termination: Termination,
    /// A private copy of the engine pos.
    pos: Position,
    /// Our node stack.
    nodes: Nodes,
    /// Our best node. Updated after each iteration.
    best_node: Node,
    /// For scoring quiet moves on alpha cutoffs. Indexing [Piece][Square].
    alpha_history: [16][64]Value,
    /// For scoring quiet moves on beta cutoffs. Indexing [Piece][Square].
    beta_history: [16][64]Value,
    /// The max reached depth, including quiet depth and possible extensions.
    seldepth: u16,
    /// Iterative deepening nr.
    iteration: u8,
    /// Timeout
    stopped: bool,
    /// Tracking search time.
    timer: utils.Timer,
    /// The total amount of nodes.
    processed_nodes: u64,
    /// The total amount of nodes done in quiescence_search.
    processed_quiescence_nodes: u64,

    //stats: if (lib.is_release) void else Stats,

    fn init() Searcher {
        return .{
            .engine = undefined,
            .transpositiontable = undefined,
            .termination = .infinite,
            .pos = .empty,
            .nodes = .empty,
            .best_node = .empty,
            .alpha_history = std.mem.zeroes([16][64]Value),
            .beta_history = std.mem.zeroes([16][64]Value),
            .seldepth = 0,
            .iteration = 0,
            .stopped = false,
            .timer = .empty,
            .processed_nodes = 0,
            .processed_quiescence_nodes = 0,
        };
    }

    fn start(self: *Searcher, engine: *Engine) void {
        // Shared stuff.
        self.engine = engine;
        self.termination = engine.searchparams.termination;
        self.transpositiontable = &engine.transpositiontable;
        //self.transpositiontable.increase_age();

        // Reset stuff.
        self.timer.reset();
        self.nodes = .empty;
        self.best_node = .empty;
        self.stopped = false;
        self.iteration = 0;
        self.processed_nodes = 0;
        self.seldepth = 0;
        self.alpha_history = std.mem.zeroes([16][64]Value);
        self.beta_history = std.mem.zeroes([16][64]Value);
        self.processed_nodes = 0;
        self.processed_quiescence_nodes = 0;

        // Create our private copy of the position.
        var rootstate: StateInfo = undefined;
        self.pos.copy_from(&engine.pos, &rootstate);

        // Append our rootstate to the history of the source position. TODO: needed?
        // rootstate.prev = input_pos.state;

        switch (self.pos.to_move.e) {
            .white => self.iterative_deepening(Color.WHITE),
            .black => self.iterative_deepening(Color.BLACK),
        }

        //print_san_pv(&engine.pos, &self.best_node) catch wtf();
        lib.io.print("bestmove {f}\n", .{ self.best_node.first_move() }) catch wtf();
    }

    fn stop(self: *Searcher) void {
        _ = self;
    }

    fn iterative_deepening(self: *Searcher, comptime us: Color) void {
        var debug_key: u64 = 0;
        var depth: u8 = 1;
        var best_score: Value = -infinity;
        var score: Value = 0;

        const ss: SearchState = comptime .new(us);

        // Search with increasing depth.
        iterationloop: while (true) {
            self.iteration = depth;

            //self.engine.tracker.track_iteration("#{} {s}", .{ self.iteration, "HELLO"});
            debug_key = self.pos.state.key;

            self.seldepth = 0;
            score = self.search(ss, depth, -infinity, infinity);

            // Discard result if timeout.
            if (self.stopped) break :iterationloop;

            // We found a better move.
            if (score > best_score) {
                best_score = score;
            }

            // Copy the last finished pv.
            self.best_node.copy_from(self.get_node(1));

            //const entry = self.transpositiontable.get(self.pos.state.key);
            //io.debugprint("PV PROBE {f}\n", .{entry.move});
            //self.get_pv_from_tt();

            if (comptime lib.is_paranoid) assert(debug_key == self.pos.state.key);

            if (!self.engine.muted) {
                print_pv(&self.best_node, self.iteration, self.seldepth, self.processed_nodes, self.timer.read(), self.transpositiontable.permille(), self.transpositiontable.hits);
            }

            depth += 1;
            if (self.check_stop()) break :iterationloop;
            if (depth >= max_search_depth) break :iterationloop;
        }

        if (!self.engine.muted) {
            print_pv(&self.best_node, self.iteration, self.seldepth, self.processed_nodes, self.timer.read(), self.transpositiontable.permille(), self.transpositiontable.hits);
        }
    }

    fn search(self: *Searcher, comptime ss: SearchState, depth: u8, input_alpha: Value, input_beta: Value) Value {

        if (comptime lib.is_paranoid) {
            assert(input_beta > input_alpha);
        }

        const us: Color = comptime ss.us;
        const is_root = comptime ss.is_root;
        const iter: u8 = self.iteration;
        const allow_prune: bool = iter >= 3;
        const pos: *Position = &self.pos;
        const key: u64 = pos.state.key;
        const is_check: bool = pos.state.checkers > 0;
        const ply: u16 = pos.ply;
        var alpha: Value = input_alpha;
        var beta: Value = input_beta;

        //if (key == 0xbbf58b9c762b4269) io.debugprint("yes,", .{}); // rnbq1rk1/pp1n1pp1/2p5/4P1NQ/1b1P4/2N5/1P3PPP/R1B2RK1 b - - 3 9

        // The rootnode is always at index 1.
        const node: *Node = self.get_node(ply + 1);
        node.clear();

        // Update counters.
        self.processed_nodes += 1;
        self.seldepth = @max(self.seldepth, ply);

        if (depth > 0) {
            if (pos.state.rule50 >= 100) return 0;

            //This trick works but needs finetuning.
            alpha = @max(alpha, -mate + ply);
			beta = @min(beta, mate - ply);
            if (alpha >= beta) {
                return alpha;
            }
        }

        // Timeout?
        if (self.check_stop()) return eval.evaluate(pos, false);

        // Too deep.
        if (ply >= max_search_depth) return eval.evaluate(pos, false);

        // TT Probe. Keeping the stored tt_move for move ordering.
        var tt_move: Move = .empty;
        if (!is_root) {
            if (self.tt_probe(key, depth, ply, alpha, beta, &tt_move)) |tt_score| {
                return tt_score;
            }
        }

        // We need this twice.
        const simple_eval: Value = eval.simple_eval(pos, us);

        // RFP (Reversed Futility Pruning).
        if (allow_prune and !is_check and depth == 1 and ply > 0 and simple_eval + 300 <= alpha) {
            return self.quiescence_search(us, alpha, beta);
        }

        // Requested depth reached: plunge into quiescence.
        if (depth == 0) {
            return self.quiescence_search(us, alpha, beta);
        }

        // Null move pruning.
        if (!is_check and !pos.nullmove_state and depth >= 4) {
            var st: StateInfo = undefined;
            pos.do_nullmove(us, &st);
            defer pos.undo_nullmove(us);
            const score: Value = -self.search(ss.flip(), depth - 4, -beta, -beta + 1);
            if (score >= beta) return beta;
            if (self.stopped) return score;
        }

        const childnode: *const Node = self.get_node(ply + 2);
        const parentnode: *const Node = self.get_node(ply);

        //const pv_move: Move = if (parentnode.pv.len > 1) parentnode.pv.buffer[1] else .empty;

        // Gen moves.
        var movepicker: MovePicker = .empty;
        movepicker.generate_moves(pos, us);
        movepicker.process_and_score_moves(self, tt_move, node.killers);

        // if (is_root) {
        //     movepicker.sort();
        //     movepicker.debugprint(iter);
        // }

        // Is this checkmate or stalemate?
        if (movepicker.count == 0) {
            const score = if (is_check) -mate + ply else stalemate;
            return score;
        }

        // Have at least a move ready when we have a ridiculous timecontrol.
        if (is_root and depth == 1) {
            self.best_node.pv.append_assume_capacity(movepicker.extract_next(0).move);
        }

        var bound: Bound = .Alpha;
        var best_score: Value = -infinity; // We still want to store something in the TT
        var best_move: Move = .empty;
        var score: Value = 0;

        // Move loop
        for (0..movepicker.count) |move_idx| {
            const e: ExtMove = movepicker.extract_next(move_idx);
            // This move block.
            {
                var st: StateInfo = undefined;
                pos.do_move(us, &st, e.move);
                defer pos.undo_move(us);

                // Futility Pruning.
                if (!is_root and allow_prune and !is_check and depth < 3 and e.is_quiet and move_idx > 5 and simple_eval + (depth * 24) < alpha) continue;

                // Interesting extension.
                var extension: u8 = 0;
                node.extension = parentnode.extension;
                if (parentnode.extension < 5) {
                    if ((is_check or movepicker.count < 3) and depth <= 12) extension = 1;
                    if (e.move.type == .promotion) extension = 1;
                }

                var need_full_search: bool = true;

                // Reduced depth. Depends on good move ordering. Never on early move or at early iterations.
                if (extension == 0 and !is_check and !e.is_capture and depth > 2 and move_idx > 1) {
                    const reduction = 1;
                    score = -self.search(ss.flip(), depth - 1 - reduction, -alpha - 1, -alpha);
                    need_full_search = score > alpha;// and score < beta;
                }

                // Full search.
                if (need_full_search) {
                    score = -self.search(ss.flip(), depth - 1 + extension, -beta, -alpha);
                }
            } // (this_move)


            // Better move found.
            if (score > best_score) {
                best_score = score;
                best_move = e.move;
                // alpha
                if (score > alpha) {
                    if (e.is_quiet) {
                        self.record_alpha_improvement(depth, e);
                    }
                    // beta
                    if (score >= beta) {
                        if (e.is_quiet) {
                            self.record_beta_cutoff(node, depth, e);
                        }
                        self.tt_store(.Beta, key, depth, ply, e.move, beta);
                        return beta;
                    }
                    alpha = score;
                    bound = .Exact;
                    node.update_pv(best_move, best_score, childnode);
                }
            }

            if (self.stopped) break;
        } // (moveloop)

        if (lib.is_paranoid) {
            assert(alpha >= input_alpha);
            assert(!best_move.is_empty());
        }

        if(alpha != input_alpha) {
            self.tt_store(.Exact, key, depth, ply, best_move, best_score);
        } else {
            self.tt_store(.Alpha, key, depth, ply, best_move, alpha);
        }

        return alpha;
    }

    fn quiescence_search(self: *Searcher, comptime us: Color, input_alpha: Value, beta: Value) Value {
        if (comptime lib.is_paranoid) {
            assert(beta > input_alpha);
            assert(self.pos.ply > 0);
        }

        const them = comptime us.opp();
        const pos: *Position = &self.pos;
        const key: u64 = pos.state.key;
        const ply: u16 = pos.ply;
        const is_check : bool = pos.state.checkers > 0;
        var alpha = input_alpha;
        var score: Value = 0;

        // The rootnode is always at index 1.
        const node: *Node = self.get_node(ply + 1);
        node.clear();

        // Update counters.
        self.processed_nodes += 1;
        self.processed_quiescence_nodes += 1;
        self.seldepth = @max(self.seldepth, pos.ply);

        // Static eval.
        score = eval.evaluate(pos, false);

        // Fail high.
        if (score >= beta) return beta;

        // Raise alpha.
        if (score > alpha) alpha = score;

        // Too deep.
        if (ply >= max_search_depth) return score;

        // Timeout: prevent calling timeout twice.
        if (self.check_stop()) return score;

        var tt_move: Move = .empty;
        // TT probe quiet
        if (self.tt_probe(key, 0, ply, alpha, beta, &tt_move)) |tt_score| {
            return tt_score;
        }

        //const childnode: *const Node = self.get_node(ply + 2);
        //const parentnode: *const Node = self.get_node(ply);

        // const node: *Node = self.get_node(ply + 1);
        // const childnode: *const Node = self.get_node(ply + 2);

        var movepicker: MovePicker = .empty;
        movepicker.generate_captures(pos, us);
        movepicker.process_and_score_moves(self, tt_move, .{ Move.empty, Move.empty});

        // Mate or stalemate?
        if (movepicker.count == 0) {
            if (is_check) {
                return -types.mate + ply;
            }
            return alpha;
        }

        var best_move: Move = .empty;
        var bound: Bound = .Alpha;

        // Move loop.
        moveloop: for (0.. movepicker.count) |move_idx| {
            const e: ExtMove = movepicker.extract_next(move_idx);

            // This move block.
            {
                var st: StateInfo = undefined;
                pos.do_move(us, &st, e.move);
                defer pos.undo_move(us);
                score = -self.quiescence_search(them, -beta, -alpha);
                //if (self.stopped) break :moveloop;
            }

            // Beta cutoff.
            if (score >= beta) {
                self.tt_store(.Beta, key, 0, ply, e.move, beta);
                return beta;
            }

            // Better move.
            if (score > alpha) {
                alpha = score;
                best_move = e.move;
                //node.update_pv(best_move, alpha, childnode);
                bound = .Exact;
            }

            if (self.stopped) break :moveloop;
        }

        // if (!best_move.is_empty()) {
        //     self.tt_store(bound, key, 0, ply, best_move, alpha);
        // }

        return alpha;
    }

    fn record_alpha_improvement(self: *Searcher, depth: Value, e: ExtMove) void {
        const m: Move = e.move;
        self.alpha_history[e.moved_piece.u][m.to.u] += depth * depth;
    }

    fn record_beta_cutoff(self: *Searcher, node: *Node, depth: Value, e: ExtMove) void {
        const m: Move = e.move;
        // Update piece square history.
        self.beta_history[e.moved_piece.u][m.to.u] += (depth * depth) * 2;
        // Update killers.
        if (node.killers[0] != m) {
            node.killers[1] = node.killers[0];
            node.killers[0] = m;
        }
    }

    fn tt_store(self: *Searcher, bound: Bound, key: u64, depth: u8, ply: u16, move: Move, score: Value) void {
        self.transpositiontable.store(bound, key, depth, ply, move, score);
    }

    /// Hashmove can be used for move ordering when not empty.
    fn tt_probe(self: *Searcher, key: u64, depth: u8, ply: u16, alpha: Value, beta: Value, tt_move: *Move) ?Value {
        if (lib.is_paranoid) assert(tt_move.* == Move.empty);
        return self.transpositiontable.probe(key, depth, ply, alpha, beta, tt_move);
    }

    fn check_stop(self: *Searcher) bool {
        if (self.stopped) return true;

        const node_interval: bool = self.processed_nodes & 2047 == 0;

        if (node_interval) {
            if (self.engine.is_cancelled()) {
                self.stopped = true;
                return true;
            }
            if (self.termination == .depth and self.iteration > self.engine.searchparams.max_depth) {
                self.stopped = true;
                return true;
            }
        }

        switch (self.termination) {
            .nodes => {
                if (self.processed_nodes >= self.engine.searchparams.max_nodes) {
                    self.stopped = true;
                }
            },
            .movetime, .clock => {
                if (node_interval and self.timer.elapsed_ms() + 10 >= self.engine.searchparams.max_movetime) {
                    self.stopped = true;
                }
            },
            else => {
                return false;
            },
        }
        return self.stopped;
    }

    /// NOTE: The pvnode at ply 0 is stored at nodes[1] NOTE: nope it is not (yet).
    fn get_node(self: *Searcher, index: u16) *Node {
        return &self.nodes.buffer[index];
    }

    fn get_pv_from_tt(self: *Searcher) void {
        var hist: [128]StateInfo = undefined;
        var pv: PV = .empty;
        var i: u8 = 0;
        while (true) {
            //var st: StateInfo = undefined;
            //ar move: Move = .empty;
            //_ = self.tt_probe(self.pos.state.key, 0, i, -infinity, infinity, &move) orelse break;

            const entry = self.transpositiontable.get(self.pos.state.key);
            if (entry.key != self.pos.state.key or entry.move.is_empty()) break;
            const move = entry.move;
            //_ = self.transpositiontable.
            io.debugprint("pvmove {f} ", .{move});
            var any: position.MoveFinder = .init(move);
            self.pos.lazy_generate_captures(&any);
            if (!any.found) break;
            self.pos.lazy_do_move(&hist[i], move);
            pv.append_assume_capacity(move);
            i += 1;
            if (i > 16) break;
        }
        for (pv.slice()) |_| {
            self.pos.lazy_undo_move();
        }
        io.debugprint("\n", .{});
    }


};

pub const Node = struct {
    /// Local PV during search.
    pv: PV = .{},
    /// Current eval.
    score: Value = 0,
    /// Extensions done (sum of chain).
    extension: u8 = 0,
    /// Killer move heuristic.
    killers: [2]Move = .{ .empty, .empty },

    const empty: Node = .{};

    fn init() Node {
        return .{};
    }

    fn clear(self: *Node) void {
        self.pv.len = 0;
        //self.pv.buffer[0] = Move.Empty; // debug safety.
        self.score = 0;
        self.extension = 0;
    }

    fn first_move(self: *const Node) Move {
        return if (self.pv.len > 0) self.pv.buffer[0] else .empty;
    }

    /// Sets `pv` to `bestmove + childnode.pv` and sets `self.score` to `score`.
    fn update_pv(self: *Node, bestmove: Move, score: Value, childnode: *const Node) void {
        self.pv.len = 0;
        //self.pv.buffer[0] = bestmove;
        self.pv.append_assume_capacity(bestmove);
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

/// 64 bits Move with score and info, used during search.
pub const ExtMove = packed struct {
    /// The generated move.
    move: Move,
    /// The score for move ordering only.
    score: i32,
    /// Set during processing.
    moved_piece: Piece,
    /// Set during processing.
    captured_piece: Piece,
    /// Set during processing.
    is_quiet: bool,
    /// Set during processing.
    is_capture: bool,

    pub const empty: ExtMove = .{
        .move = .empty,
        .score = 0,
        .moved_piece = Piece.NO_PIECE,
        .captured_piece = Piece.NO_PIECE,
        .is_quiet = false,
        .is_capture = false
    };
};

const MovePicker = struct {
    extmoves: [types.max_move_count]ExtMove,
    count: u8,

    const empty: MovePicker = .{ .extmoves = undefined, .count = 0 };

    /// Required function.
    pub fn reset(self: *MovePicker) void {
        //self.count = 0;
        assert(self.count == 0);
    }

    /// Required function.
    pub fn store(self: *MovePicker, move: Move) ?void {
        self.extmoves[self.count] = ExtMove{ .move = move, .score = 0, .moved_piece = Piece.NO_PIECE, .captured_piece = Piece.NO_PIECE, .is_quiet = false, .is_capture = false };
        self.count += 1;
    }

    fn copy_from(self: *MovePicker, other: *const MovePicker) void {
        const cnt: u8 = other.count;
        @memcpy(self.extmoves[0..cnt], other.extmoves[0..cnt]);
        self.count = cnt;
    }

    fn generate_moves(self: *MovePicker, pos: *const Position, comptime us: Color) void {
        pos.generate_moves(us, self);
    }

    /// Generate captures only, but if in check generate all.
    fn generate_captures(self: *MovePicker, pos: *const Position, comptime us: Color) void {
        pos.generate_captures(us, self);
    }

    /// Fill ExtMove info and give each generated move an initial heuristic score. Crucial for the search algorithm.
    fn process_and_score_moves(self: *MovePicker, searcher: *const Searcher, tt_move: Move, killers: [2]Move) void {
        const pos: *const Position = &searcher.pos;

        const hash         : Value = 2_000_000;
        const promotion    : Value = 1_000_000;
        const capture      : Value = 500_000;
        const killer1      : Value = 400_000;
        const killer2      : Value = 300_000;
        const bad_capture  : Value = -500_000;

        for (self.slice()) |*e| {
            const m: Move = e.move;
            e.moved_piece = pos.board[m.from.u];

            switch (m.type) {
                .normal => {
                    e.captured_piece = pos.board[m.to.u];
                    if (e.captured_piece.is_piece()) {
                        e.is_capture = true;
                        const see = eval.see_score(pos, m);
                        if (see >= 0)
                            e.score = capture + see
                        else
                            e.score = bad_capture + see; // TODO: how what...
                    }
                    else {
                        e.is_quiet = true;
                        e.score = searcher.beta_history[e.moved_piece.u][m.to.u] + searcher.alpha_history[e.moved_piece.u][m.to.u];
                    }
                },
                .promotion => {
                    e.captured_piece = pos.board[m.to.u];
                    e.score = promotion + (m.promoted().value() * 10);
                    if (e.captured_piece.is_piece()) {
                        e.is_capture = true;
                        e.score += e.captured_piece.value();
                    }
                },
                .enpassant => {
                    e.is_capture = true;
                    e.captured_piece = Piece.create_pawn(pos.to_move.opp());
                    e.score = capture;
                },
                .castle => {
                    e.is_quiet = true;
                    e.score = 0;//castle;
                },
            }

            // Hash move.
            if (m == tt_move) {
                e.score += hash;
            }
            // Killers only affect quiet moves
            else if (e.is_quiet) {
                if (m == killers[0]) {
                    e.score += killer1;
                }
                else if (m == killers[1]) {
                    e.score += killer2;
                }
            }
        }
    }

    fn extract_next(self: *MovePicker, current_idx: usize) ExtMove {
        const ptr: [*]ExtMove = &self.extmoves;
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
        return ptr[current_idx];
    }

    fn slice(self: *MovePicker) []ExtMove {
        return self.extmoves[0..self.count];
    }

    fn sort(self: *MovePicker) void {
        std.mem.sort(ExtMove, self.slice(), {}, less_than);
    }

    fn less_than(_: void, a: ExtMove, b: ExtMove) bool {
        return a.score > b.score;
    }

    fn debugprint(self: *const MovePicker, iteration: u8) void {
        io.debugprint("\n#{}: ", .{ iteration });
        for (self.extmoves[0..self.count], 0..) |e, i| {
           io.debugprint("{f} {}; ", .{e.move, e.score});
           _ = i;
        }
        io.debugprint("\n\n", .{});
    }
};

/// The available options.
pub const Options = struct {
    /// In megabytes.
    hash_size: u64 = default_hash_size,

    pub const default: Options = .{};

    pub const default_hash_size: u64 = 16;
    pub const min_hash_size: u64 = 1;
    pub const max_hash_size: u64 = 1024;

    pub const default_use_hash: bool = true;

    pub fn set_hash_size(self: *Options, value: u64) void {
        self.hash_size = std.math.clamp(value, min_hash_size, max_hash_size);
    }
};

/// TODO: rename to settings?
pub const SearchParams = struct {
    termination: Termination,
    max_depth: u8,
    max_nodes: u64,
    /// In ms: either
    /// * The explicit `uci.go.movetime` if provided.
    /// * Calculated if `uci.go.wtime` or `uci.go.btime` provided.
    /// * zero.
    max_movetime: u32,
    time_left: [2]u32,
    increment: [2]u32,
    movestogo: u32,

    //go wtime 600000 btime 600000 winc 30000 binc 30000 movestogo 40

    const default: SearchParams = .{
        .termination = .infinite,
        .max_depth = 0,
        .max_nodes = 0,
        .max_movetime = 0,
        .time_left = .{ 0, 0 },
        .increment = .{ 0, 0 },
        .movestogo = 0,
    };

    /// Convert the UCI go params to something usable for our search.
    pub fn init(go_params: *const uci.Go, to_move: Color) SearchParams {
        var result: SearchParams = .default;

        // Infinite.
        if (go_params.infinite != null) {
            return result;
        }

        // Clock.
        if (go_params.wtime != null or go_params.btime != null) {
            result.termination = .clock;
            result.time_left[Color.WHITE.u] = go_params.wtime orelse 0;
            result.time_left[Color.BLACK.u] = go_params.btime orelse 0;
            result.increment[Color.WHITE.u] = go_params.winc orelse 0;
            result.increment[Color.BLACK.u] = go_params.binc orelse 0;
            result.movestogo = go_params.movestogo orelse 0;

            const moves: u32 = if (result.movestogo > 0) result.movestogo else 30;
            const time: u32 = if (to_move.e == .white) result.time_left[Color.WHITE.u] else result.time_left[Color.BLACK.u];
            const inc: u32 = if (to_move.e == .white) result.increment[Color.WHITE.u] else result.increment[Color.BLACK.u];

            result.max_movetime = (time / moves) + (inc / 2);
            return result;
        }

        // Depth.
        if (go_params.depth) |depth| {
            result.termination = .depth;
            result.max_depth = @truncate(@min(depth, types.max_search_depth));
            return result;
        }

        // Nodes.
        if (go_params.nodes) |nodes| {
            result.termination = .nodes;
            result.max_nodes = nodes;
            return result;
        }

        // Movetime.
        if (go_params.movetime) |movetime| {
            result.termination = .movetime;
            result.max_movetime = movetime;
            return result;
        }

        return result;
    }
};

/// Criterium for ending the search.
pub const Termination = enum {
    infinite,
    nodes,
    movetime,
    depth,
    clock,
};
const CallSite = enum { iterate, search, quiescence };

const SearchState = struct {
    us: Color,
    is_root: bool,

    /// The state for iterative deepening.
    fn new(comptime us: Color) SearchState {
        return .{ .us = us, .is_root = true };
    }

    /// Flip for recursion.
    fn flip(comptime ss: SearchState) SearchState {
        return .{ .us = ss.us.opp(), .is_root = false };
    }
};

fn print_pv(pv_node: *const Node, depth: u16, seldepth: u16, nodes: u64, elapsed_nanoseconds: u64, hash_full: usize, tt_hits: u64) void {
    const ms = elapsed_nanoseconds / std.time.ns_per_ms;
    const nps: u64 = funcs.nps(nodes, elapsed_nanoseconds);
    lib.io.print (
        "info depth {} seldepth {} score cp {} nodes {} nps {} hashfull {} tthits {} time {} pv {f}\n",
        .{ depth, seldepth, pv_node.score, nodes, nps, hash_full, tt_hits, ms, pv_node.* }
    )
    catch wtf();
}

fn print_san_pv(input_pos: *const Position, pv_node: *const Node) !void {
    if (pv_node.pv.len == 0 ) return;
    const san = @import("san.zig");
    const ev = if (input_pos.to_move.e == .white) pv_node.score else -pv_node.score;
    try lib.io.print("info string score cp {} line: ", .{ ev });
    try san.write_san_line(input_pos, pv_node.pv.slice(), lib.io.out);
    lib.io.print("\n", .{}) catch wtf();
}

const Tracker = struct {

    text: utils.TextFileWriter,

    const What = enum {
        loop,
        alpha,
        beta
    };

    fn init() !Tracker {
        return .{ .text = try .init("C:/Tmp/chessnix.log", ctx.galloc, 4096) };
    }

    fn deinit(self: *Tracker) void {
       self.text.deinit();
    }

    fn track_iteration(self: *Tracker, comptime str: []const u8, args: anytype) void {
        self.text.writeline(str, args) catch wtf();
        self.text.writer.interface.flush() catch wtf();
    }

    fn track_search(self: *Tracker, callsite: CallSite, what: What, pos: *const Position, iteration: u8, depth: u8, current_move: Move, evaluation: ?Value) void {

        // Indent.
        for (1..pos.ply + 1) |_| {
            self.text.write("    ", .{}) catch wtf();
        }
//        self.text.write("d{} ", .{ depth }) catch wtf();
        self.text.write("{}.{}.{} ", .{ iteration, depth, pos.ply }) catch wtf();

        const c: u8 = @tagName(callsite)[0];
        const w: u8 = @tagName(what)[0];
        // What happened.
        self.text.write("{u}.{u} ", .{ c, w }) catch wtf();

        // Line.
        self.text.write("[", .{}) catch wtf();
        var iter: position.PositionStateIterator = .init(pos);
        while (iter.next()) |st| {
            self.text.write("{f} ", .{ st.last_move }) catch wtf();
        }
        self.text.write("]", .{}) catch wtf();

        self.text.write("+ {f} ", .{ current_move }) catch wtf();

        if (evaluation) |e| {
            self.text.write("({}) ", .{ e }) catch wtf();
        }

        self.text.writeline("", .{}) catch wtf();
    }
};

const TimeOut = error { Occurred };

const Error = error {
    EngineIsStillRunning,
};
