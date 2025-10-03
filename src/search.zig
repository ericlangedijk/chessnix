// zig fmt: off

//! Engine + Search.

const std = @import("std");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
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
const max_game_length: u16 = types.max_game_length;
const max_move_count: u8 = types.max_move_count;
const max_search_depth: u8 = types.max_search_depth;
const max_threads: u16 = types.max_threads;
const infinity = types.infinity;
const mate = types.mate;
const mate_threshold = types.mate_threshold;
const stalemate = types.stalemate;

const PV = lib.BoundedArray(Move, max_search_depth + 2); // just to be safe.

pub const Use = packed struct {
    aspiration_window: bool,
    tt: bool,
    null_move_pruning: bool,
    beta_pruning: bool,
    alpha_pruning: bool,
    reversed_futility_pruning: bool,
    futility_pruning: bool,
    late_move_reduction: bool,
    search_extensions: bool,
};

pub const use: Use = .{
    .aspiration_window = true,
    .tt = true,
    .null_move_pruning = true,
    .beta_pruning = true,
    .alpha_pruning = true,
    .reversed_futility_pruning = true,
    .futility_pruning = true,
    .late_move_reduction = true,
    .search_extensions = true,
};

pub const Engine = struct {
    /// Shared tt for all threads.
    transpositiontable: tt.TranspositionTable,
    /// Threading.
    busy: std.atomic.Value(bool),
    /// The number of threads we use.
    num_threads: usize,
    /// The controller thread, managing the search threads.
    controller_thread: ?std.Thread,
    /// The working thread. Later this will be more threads. We start with one.
    search_thread: ?std.Thread,
    /// The global state list for our position.
    history: [max_game_length]StateInfo,
    /// Our source position.
    pos: Position,
    /// UCI Engine options like hashsize etc.
    options: Options,
    /// Settings like timeout etc.
    time_params: TimeParams,
    /// The actual worker.
    searcher: Searcher,
    /// No output if true.
    muted: bool,
    /// Increments with each go command. Resets after 7.
    cycle: u3,

    /// Engine is created on the heap.
    pub fn create() !*Engine {
        var engine: *Engine = try ctx.galloc.create(Engine);
        engine.transpositiontable = try .init(Options.default_hash_size);
        engine.busy = std.atomic.Value(bool).init(false);
        engine.num_threads = 1;
        engine.controller_thread = null;
        engine.search_thread = null;
        engine.history = @splat(StateInfo.empty);
        engine.pos = .empty;
        engine.pos.set_startpos(&engine.history[0]);
        engine.options = Options.default;
        engine.time_params = TimeParams.default;
        engine.searcher = Searcher.init();
        engine.muted = false;
        engine.cycle = 0;
        return engine;
    }

    pub fn destroy(self: *Engine) void {
        self.transpositiontable.deinit();
        ctx.galloc.destroy(self);
    }

    pub fn ucinewgame(self: *Engine) !void {
        if (self.is_busy()) return;
        self.set_startpos();
        self.transpositiontable.clear();
        self.cycle = 0;
    }

    pub fn is_busy(self: *const Engine) bool {
        return self.busy.load(.monotonic); // acquire?
    }

    /// Start threaded search.
    pub fn start(self: *Engine, go_params: *const uci.Go) !void {
        if (self.is_busy()) return;
        self.time_params = .init(go_params, self.pos.to_move);
        self.busy.store(true, .monotonic); // release?
        self.controller_thread = try std.Thread.spawn(.{}, run_controller_thread, .{ self });
    }

    /// Stop threaded search.
    pub fn stop(self: *Engine) !void {
        if (!self.is_busy()) return;
        // Force the searcher to stop. Only the engine may do this atomic store to the searchers. He is the boss.
        @atomicStore(bool, &self.searcher.stopped, true, .monotonic);
        self.busy.store(false, .monotonic);
    }

    fn run_controller_thread(self: *Engine) !void {
        defer {
            self.controller_thread = null;
            self.cycle +|= 1;
        }
        self.transpositiontable.age();
        self.search_thread = try std.Thread.spawn(.{}, Searcher.start, .{ &self.searcher, self, self.cycle });
        if (self.search_thread) |th| {
            th.join();
            self.search_thread = null;
        }
        self.busy.store(false, .monotonic);
    }

    pub fn set_startpos(self: *Engine) void {
        if (self.is_busy()) return;
        self.pos.set_startpos(&self.history[0]);
    }

    /// After an illegal move we stop without crashing.
    pub fn set_startpos_with_optional_moves(self: *Engine, moves: ?[]const u8) !void {
        if (self.is_busy()) return;
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
        if (self.is_busy()) return;
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

    /// Engine test function.
    pub fn give_best_move(self: *Engine, fen: []const u8, moves: []const u8, movetime: u32) !Move {
        if (self.is_busy()) return .empty;
        //lib.io.debugprint("START\n", .{});
        try self.ucinewgame();
        self.muted = true;
        try self.set_position(fen, moves);
        var go: uci.Go = .empty;
        go.movetime = movetime;
        try self.start(&go);
        while (self.is_busy()) {
            // run thread.
        }
        return self.searcher.nodes[0].first_move();
    }
};

pub const Stats = struct {
    /// /// The total amount of nodes.
    nodes: u64 = 0,
    /// The total amount of nodes done in quiescence_search.
    qnodes : u64 = 0,
    /// Iteration reached.
    depth: u16 = 0,
    /// The max reached depth, including quiescence depth and possible extensions.
    seldepth: u16 = 0,
    /// Used for move ordering statistic.
    beta_cutoffs: u64 = 0,

    const empty: Stats = .{};
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

    // Our rootmoves for the pos.
    rootmoves: MovePicker,

    /// The first state.
    rootstate: *const StateInfo,
    /// Our node stack. Index #0 is our main pv. Indexing [ply + 1] for the current ply.
    nodes: [max_search_depth + 2]Node,
    /// For scoring quiet moves on alpha cutoffs. Indexing [Piece][Square].
    alpha_history: [16][64]Value,
    /// For scoring quiet moves on beta cutoffs. Indexing [Piece][Square].
    beta_history: [16][64]Value,
    /// The current iteration.
    iteration: u8,
    /// Timeout. Any (0) value returned when stopped should be ignored.
    stopped: bool,
    /// Tracking search time.
    timer: utils.Timer,
    /// Statistics.
    stats: Stats,

    fn init() Searcher {
        return .{
            .engine = undefined,
            .transpositiontable = undefined,
            .termination = .infinite,
            .pos = .empty,
            .rootmoves = .empty,
            .rootstate = undefined,
            .nodes = @splat(.empty),
            .alpha_history = std.mem.zeroes([16][64]Value),
            .beta_history = std.mem.zeroes([16][64]Value),
            .iteration = 0,
            .stopped = false,
            .timer = .empty,
            .stats = .empty,
        };
    }

    fn print_pv(self: *Searcher, elapsed_nanos: u64) void {
        const ms = elapsed_nanos / 1_000_000;
        const nps: u64 = funcs.nps(self.stats.nodes, elapsed_nanos);

        const rootnode: *const Node = &self.nodes[0];

        //const score: Value = rootnode.score;

        // var mate_output: ?Value = null;
        // if (score <= -mate_threshold) {
        //     mate_output = -(@divTrunc(score + mate, 2) - 1);
        // }
        // else if (score >= mate_threshold) {
        //     mate_output = @divTrunc(mate - score, 2) + 1;
        // }

        // // I hate the mate in X uci output so I add ctp (raw score)
        // if (mate_output != null) {
        //     io.print (
        //         "info depth {} seldepth {} score mate {} ctp {} nodes {} qnodes {}% nps {} ttprobes {} tthits {} order {}% time {} pv {f}\n",
        //         .{
        //             self.stats.depth,
        //             self.stats.seldepth,
        //             mate_output.?,
        //             rootnode.score,
        //             self.stats.nodes,
        //             funcs.percent(self.stats.nodes, self.stats.qnodes),
        //             nps,
        //             self.transpositiontable.probes,
        //             self.transpositiontable.hits,
        //             funcs.percent(self.stats.nodes, self.stats.beta_cutoffs),
        //             ms,
        //             rootnode
        //         }
        //     ) catch wtf();
        // }
        // // No mate score.
        // else

        io.print (
            "info depth {} seldepth {} score cp {} nodes {} qnodes {}% nps {} ttprobes {} tthits {} order {}% time {} pv {f}\n",
            .{
                self.stats.depth,
                self.stats.seldepth,
                rootnode.score,
                self.stats.nodes,
                funcs.percent(self.stats.nodes, self.stats.qnodes),
                nps,
                self.transpositiontable.probes,
                self.transpositiontable.hits,
                funcs.percent(self.stats.nodes, self.stats.beta_cutoffs),
                ms,
                rootnode
            }
        ) catch wtf();

    }

    fn try_set_pv_from_tt(self: *Searcher, node: *Node) void {

        const debug_key: u64 = if (lib.is_paranoid) self.pos.state.key else 0;

        // TODO: we can probably do some more optimizations here. Build in pondering etc. anyway. We are not there yet...
        var hist: [128]StateInfo = undefined;
        var i: u8 = 0;
        while (true) {
            const entry = self.transpositiontable.get(self.pos.state.key);
            if (entry.key != self.pos.state.key or entry.bound != .Exact) break;
            const move = entry.move;
            var any: position.MoveFinder = .init(move);
            self.pos.lazy_generate_moves(&any);
            if (!any.found) break;
            self.pos.lazy_do_move(&hist[i], move);
            node.pv.append_assume_capacity(move);
            //if (i == 0) node.score = entry.score;
            i += 1;
            if (i > 127) break;
        }

        for (node.pv.slice()) |_| {
            self.pos.lazy_undo_move();
        }

        // Check if we did not mess up.
        if (lib.is_paranoid) {
            assert(self.pos.state.key == debug_key);
        }
    }

    fn start(self: *Searcher, engine: *Engine, cycle: u3) void {
        // Shared stuff.
        self.engine = engine;
        self.termination = engine.time_params.termination;
        self.transpositiontable = &engine.transpositiontable;

        // Reset stuff.
        self.timer.reset();
        self.nodes = @splat(.empty);
        self.rootmoves = .empty;
        self.stopped = false;
        self.iteration = 0;
        self.stats = .empty;

        // Clear stuff. This is a new game.
        if (cycle == 0) {
            self.alpha_history = std.mem.zeroes([16][64]Value);
            self.beta_history = std.mem.zeroes([16][64]Value);
        }
        // Preserve stuff during a game. Aging heuristics and prevent overflowing.
        else {
            for (0..16) |p| {
                for (0..64) |q| {
                    self.alpha_history[p][q] = @divTrunc(self.alpha_history[p][q], 2);
                }
            }
            for (0..16) |p| {
                for (0..64) |q| {
                    self.beta_history[p][q] = @divTrunc(self.beta_history[p][q], 2);
                }
            }
        }

        // Create our private copy of the position.
        // Append our rootstate to the history of the source position.
        // This is quite a dirty trick: this st state 'skips' the last global pos state.
        // It has (for now) be that way.
        var st: StateInfo = undefined;
        self.pos.copy_from(&engine.pos, &st);
        st.prev = engine.pos.state.prev;
        self.rootstate = &st;
        //self.pos.ply_from_root = engine.pos.ply_from_root;

        //io.debugprint("game_ply {} ply {} ply_from_root {}\n", .{ self.pos.game_ply, self.pos.ply, self.pos.ply_from_root});

        const rootnode: *Node = &self.nodes[0];

        // io.debugprint("{any}\n\n", .{ r});
        // if (r.prev) |pr|
        // io.debugprint("{any}\n\n", .{ pr }) else io.debugprint("NOPREV\n\n", .{});

        // If there is only one move in a tournament situation move immediately. TODO: move to search -> exit early.
        //const rootnode: *Node = &self.nodes[0];



        //var store: position.MoveStorage = .init(); //TODO: make rootmoves.?
        //self.pos.lazy_generate_moves(&self.rootmoves);

        switch (self.pos.to_move.e) {
            .white => self.rootmoves.generate_moves(&self.pos, Color.WHITE),
            .black => self.rootmoves.generate_moves(&self.pos, Color.BLACK),
        }

        if (self.rootmoves.count == 1 and self.termination == .clock) {
            const bm: Move = self.rootmoves.extmoves[0].move;
            rootnode.score = 0;
            rootnode.pv.append_assume_capacity(bm);
            if (!engine.muted)
                lib.io.print("bestmove {f}\n", .{ bm } ) catch wtf();
            return;
        }

        switch (self.pos.to_move.e) {
            .white => self.iterate(Color.WHITE),
            .black => self.iterate(Color.BLACK),
        }

        if (!engine.muted) {
            //if (rootnode.pv.len > 0) {
                lib.io.print("bestmove {f}\n", .{ rootnode.first_move() }) catch wtf();
            //}
            //else if (self.rootmoves.count > 0) {
              //  lib.io.print("bestmove {f}\n", .{ self.rootmoves.extmoves[0].move }) catch wtf();
            //}
        }
    }

    fn iterate_ORIGINAL(self: *Searcher, comptime us: Color) void {
        var delta: Value = 25;
        var debug_key: u64 = 0;
        var depth: u8 = 1;
        var score: Value = 0;
        var alpha: Value = -infinity;
        var beta: Value = infinity;

        const rootnode: *Node = &self.nodes[0];
        const iteration_node: *Node = &self.nodes[1];

        self.try_set_pv_from_tt(rootnode);
        if (rootnode.pv.len > 0) {
            self.print_pv(0);
        }

        // Search with increasing depth.
        iterationloop: while (true) {

            self.iteration = depth;
            self.stats.depth = depth;
            self.stats.seldepth = 0;

            // Paranoid check.
            if (lib.is_paranoid) {
                debug_key = self.pos.state.key;
            }

            io.debugprint("    #{} {} {}\n", .{ depth, alpha, beta});
            score = self.search(depth, alpha, beta, us, true);

            // Paranoid check.
            if (lib.is_paranoid) {
                assert(debug_key == self.pos.state.key);
            }

            // Discard result if timeout.
            if (self.stopped) {
                break :iterationloop;
            }

            if (comptime use.aspiration_window) {
                // Aspiration Window fail low.
                if (score <= alpha) {
                    alpha -= delta;
                    if (alpha < -infinity) alpha = -infinity;
                    delta *= 2;
                    continue :iterationloop;
                }
                // Aspiration Window fail high.
                else if (score >= beta) {
                    beta += delta;
                    if (beta > infinity) beta = infinity;
                    delta *= 2;
                    continue :iterationloop;
                }
                // Aspiration Window succes.
                else {
                    alpha = score - delta;
                    beta = score + delta;
                    delta = 25;
                }
            }

            // Copy the last finished pv.
            rootnode.copy_from(iteration_node);

            if (!self.engine.muted) {
                self.print_pv(self.timer.read());
            }

            depth += 1;
            if (depth >= max_search_depth or self.check_stop()) break :iterationloop;
        }
            // if (!self.engine.muted) {
            //     self.print_pv(self.timer.read());
            // }
            // for (0..16) |p| {
            //     for (0..64) |q| {
            //         const b = self.beta_history[p][q];
            //         if (b > 0)
            //             io.debugprint("beta {t} {t} {}\n", .{ Piece.from_usize(p).e, Square.from_usize(q).e, b });
            //     }
            // }

    }

    fn iterate(self: *Searcher, comptime us: Color) void {
        var debug_key: u64 = 0;

        const rootnode: *Node = &self.nodes[0];
        const iteration_node: *Node = &self.nodes[1];

        self.try_set_pv_from_tt(rootnode);
        if (rootnode.pv.len > 0) {
            self.print_pv(0);
        }

        var depth: u8 = 1;
        var score: Value = 0;
        var best_score: Value = -infinity;

        // Search with increasing depth.
        iterationloop: while(true) {
            self.iteration = depth;
            self.stats.depth = depth;
            self.stats.seldepth = 0;

            var alpha: Value = -infinity;
            var beta: Value = infinity;
            var delta: Value = 20;

            // The first few iterations no aspiration search.
            if (comptime use.aspiration_window) {
                if (depth >= 4) {
                    alpha = score - delta;
                    beta = score + delta;
                }
            }

            aspiration_loop: while (true) {
                if (lib.is_paranoid) {
                    debug_key = self.pos.state.key;
                }

                // io.debugprint("    #{} {} {}\n", .{ depth, alpha, beta});
                score = self.search(depth, alpha, beta, us, true);

                if (lib.is_paranoid) {
                    assert(debug_key == self.pos.state.key);
                }

                // Discard result if timeout.
                if (self.stopped) {
                    break :iterationloop;
                }

                if (comptime use.aspiration_window) {
                    // Alpha.
                    if (score <= alpha) {
                        alpha = score - delta;
                    }
                    // Beta.
                    else if (score >= beta) {
                        beta = score + delta;
                    }
                    // Success.
                    else {
                        break :aspiration_loop;
                    }
                }
                delta += delta;
            }

            // Copy the last finished pv.
            rootnode.copy_from(iteration_node);

            if (!self.engine.muted) {
                self.print_pv(self.timer.read());
            }

            // if we found the same matescore as the previous iteration then stop.
            if (score >= best_score) {
                if (self.termination == .clock and score >= mate_threshold and best_score >= mate_threshold and score == best_score) {
                    break: iterationloop;
                }
                best_score = score;
            }

            depth += 1;
            if (depth >= max_search_depth or self.check_stop()) break :iterationloop;
        }
    }

    fn search(self: *Searcher, depth: u8, input_alpha: Value, input_beta: Value, comptime us: Color, comptime is_root: bool) Value {

        // Paranoid checks.
        if (comptime lib.is_paranoid) {
            assert(input_beta > input_alpha);
            assert(!is_root or depth > 0);
        }

        const pos: *Position = &self.pos;
        const ply: u16 = pos.ply;

        // Update counters.
        self.stats.nodes += 1;
        self.stats.seldepth = @max(self.stats.seldepth, ply);

        // Timeout.
        if (self.check_stop()) {
            return 0;
        }

        if (!is_root) {
            // Too deep.
            if (ply >= max_search_depth) {
                return eval.evaluate(pos);
            }

            // Draw. TODO: optimize and check if correct.
            if (pos.state.rule50 >= 100 or eval.is_draw_by_insufficient_material(pos) or pos.is_threefold_repetition()) {
                return 0;
            }

            // // Repetition. TODO: optimize and check if correct.
            if (pos.is_upcoming_repetition()) {
                 return 0;
            }
        }

        const them: Color = comptime us.opp();
        const is_pvs: bool = input_beta - input_alpha != 1;
        const key: u64 = pos.state.key;
        const is_check: bool = pos.state.checkers > 0;
        var alpha: Value = input_alpha;
        var beta: Value = input_beta;
        var e: u8 = 0; // extension
        var r: u8 = 0; // reduction

        const node: *Node = &self.nodes[ply + 1];
        const childnode: *const Node = &self.nodes[ply + 2];
        const parentnode: *const Node = &self.nodes[ply];
        node.clear();

        // TT Probe. Keeping the stored tt_move for move ordering.
        var tt_move: Move = .empty;
        if (self.tt_probe(key, depth, ply, alpha, beta, &tt_move)) |tt_score| {
            if (!is_pvs) return tt_score;
        }

        // TODO: already better.... but not at all clear how and what and why. maybe it is complete crap and i should really test if we are following the best line here.
        var pv_move: Move = .empty;
        if (parentnode.pv.len > ply) {
             pv_move = parentnode.pv.buffer[ply];
        }

        // Evaluation.
        const ev: Value = eval.evaluate(pos);
        const improvement: Value = if (ply < 2) 0 else if (ev > self.nodes[ply - 1].score and ev > self.nodes[ply - 2].score) 1 else 0;

        if (!is_root) {
            alpha = @max(alpha, -mate + ply);
			beta = @min(beta, mate - ply);
            if (alpha >= beta) return alpha;
        }

        // // Beta pruning.
        // if (use.beta_pruning) {
        //     if (!is_pvs and !is_check and depth <= 8 and (ev - @as(i32, depth) * 100) >= (beta - improvement * 50)) {
        //         return ev;
        //     }
        // }

        // Reversed Futility Pruning.
        if (comptime use.reversed_futility_pruning) {
            if (!is_pvs and !is_check and depth == 1 and ev + 300 <= alpha) {
                return self.qsearch(0, alpha, beta, us);
            }
        }

        // Requested depth reached: plunge into quiescence.
        if (!is_root and depth == 0 and !is_check) {
            return self.qsearch(0, alpha, beta, us);
        }

        // Beta pruning. (MOVED HERE)
        if (use.beta_pruning) {
            if (!is_pvs and !is_check and depth <= 8 and (ev - @as(i32, depth) * 100) >= (beta - improvement * 50)) {
                return ev;
            }
        }

        // Null move pruning.
        if (comptime use.null_move_pruning) {
            if (!is_pvs and !is_check and !pos.nullmove_state and depth > 2 and ev > beta and pos.non_pawn_material() > 0) {
                r = 3;
                var st: StateInfo = undefined;
                pos.do_nullmove(us, &st);
                defer pos.undo_nullmove(us);
                if (depth <= 3) r = depth - 1;
                const score: Value = -self.search(depth - 1 - r, -beta, -beta + 1, them, false);
                if (self.stopped) {
                    return 0;
                }
                if (score >= beta) {
                    return if (score > mate_threshold) beta else score;
                }
            }
        }

        // Alpha pruning.
        if (use.alpha_pruning) {
            if (!is_root and !is_check and depth <= 4 and alpha > -mate_threshold and ev + 500 * @as(i32, depth) <= alpha) {
                const score: Value = self.qsearch(1, alpha, alpha + 1, us);
                if (self.stopped) {
                    return 0;
                }
                if (score <= alpha) {
                    return score;
                }
            }
        }

        // Gen moves.
        var movepicker: MovePicker = .empty;
        if (is_root) {
            movepicker.copy_from(&self.rootmoves);
        }
        else {
            movepicker.generate_moves(pos, us);
        }
        movepicker.process_and_score_moves(self, pv_move, tt_move, node.killers);

        // Checkmate or stalemate.
        if (movepicker.count == 0) {
            const score: Value = if (is_check) -mate + ply else 0;
            return score;
        }

        var bound: Bound = .Alpha;
        var best_score: Value = -infinity; // We still want to store something in the TT?
        var best_move: Move = .empty;

        ////////////////////////////////////////////////////////////////
        // Move loop
        ////////////////////////////////////////////////////////////////
        moveloop: for (0..movepicker.count) |move_idx| {
            const em: ExtMove = movepicker.extract_next(move_idx);
            var st: StateInfo = undefined;

            // if (!is_pvs and !is_check and alpha > -mate_threshold and lut(lut_prune, move)) {
            //     if (num_legal_moves > depth and (ev + depth * depth * 50 + 100) < alpha) {
            //         continue;
            //     }
            // }

            // Futility Pruning.
            if (comptime use.futility_pruning) {
                if (!is_root and !is_pvs and !is_check and depth < 3 and em.is_quiet and move_idx > 3 and ev + (depth * 24) < alpha) {
                    continue :moveloop;
                }
            }

            // Do move
            pos.do_move(us, &st, em.move);

            var need_full_search: bool = true;
            e = 0;
            r = 0;
            var score: Value = -infinity;

            // Extension.
            if (comptime use.search_extensions) {
                if (is_check or em.move.type == .promotion)
                    e = 1;
            }

            // Reduction.
            if (comptime use.late_move_reduction) {
                if (!is_check and e == 0 and !em.is_capture and depth >= 3 and move_idx > 0) {
                    r = 1;
                    score = -self.search(depth - 1 - r, -alpha - 1, -alpha, them, false);
                    need_full_search = score > alpha and score < beta;
                }
            }

            if (self.stopped) {
                pos.undo_move(us);
                return 0;
            }

            if (need_full_search) {
                score = -self.search(depth + e - 1, -beta, -alpha, them, false);
            }

            pos.undo_move(us);

            if (self.stopped) {
                return 0;
            }

            // Better move found.
            if (score > best_score) {
                best_score = score;
                best_move = em.move;
                // Alpha.
                if (score > alpha) {
                    if (is_pvs){
                        node.update_pv(best_move, best_score, childnode);
                    }
                    if (em.is_quiet) {
                        self.record_alpha_improvement(depth, em, score - alpha);
                    }
                    // Beta.
                    if (score >= beta) {
                        self.stats.beta_cutoffs += 1;
                        if (em.is_quiet) {
                            self.record_beta_cutoff(node, depth, em);
                        }
                        self.tt_store(.Beta, key, depth, ply, em.move, beta);
                        return beta;
                    }
                    alpha = score;
                    bound = .Exact;
                }
            }
        } // (moveloop)

        if (lib.is_paranoid) {
            assert(alpha >= input_alpha);
            assert(!best_move.is_empty());
        }

        if(alpha > input_alpha) {
            self.tt_store(.Exact, key, depth, ply, best_move, best_score);
        } else {
            self.tt_store(.Alpha, key, depth, ply, best_move, best_score); // alpha or best_score?
        }

        return best_score;
    }

    fn qsearch(self: *Searcher, qdepth: u8, input_alpha: Value, input_beta: Value, comptime us: Color) Value {
        if (comptime lib.is_paranoid) {
            assert(self.pos.to_move.e == us.e);
            assert(self.pos.ply > 0);
           // assert(input_beta > input_alpha); // TODO: goes wrong with mating scores. no clue why.
        }


        const pos: *Position = &self.pos;
        const ply: u16 = pos.ply;

        // Update stats.
        self.stats.nodes += 1;
        self.stats.qnodes += 1;
        self.stats.seldepth = @max(self.stats.seldepth, pos.ply);

        if (self.check_stop()) {
            return 0;
        }

        const node: *Node = &self.nodes[pos.ply + 1];
        node.clear();

        if (ply >= max_search_depth) {
            return eval.evaluate(pos);
        }


        const them = comptime us.opp();
        const key: u64 = pos.state.key;
        const is_check : bool = pos.state.checkers > 0;
        var alpha = input_alpha;
        var beta = input_beta;
        var score: Value = 0;

        alpha = @max(alpha, -mate + ply);
        beta = @min(beta, mate - ply);

        if (!is_check) {
            // Static eval.
            score = eval.evaluate(pos);

            // Fail high.
            if (score >= beta) return score;

            // Raise alpha.
            if (score > alpha) alpha = score;
        }

        // TT probe quiet
        var tt_move: Move = .empty;
        if (self.tt_probe(key, 0, ply, alpha, beta, &tt_move)) |tt_score| {
            return tt_score;
        }

        var movepicker: MovePicker = .empty;
        movepicker.generate_captures(pos, us);
        movepicker.process_and_score_moves(self, .empty, tt_move, .{ Move.empty, Move.empty});

        // Mate or stalemate?
        if (movepicker.count == 0) {
            if (is_check) {
                return -mate + ply;
            }
            return alpha;
        }

        // Move loop.
        for (0.. movepicker.count) |move_idx| {
            const e: ExtMove = movepicker.extract_next(move_idx);

            if (!is_check and e.is_bad_capture) {
                continue; // pre-move
            }

            // Do move
            var st: StateInfo = undefined;
            pos.do_move(us, &st, e.move);

            // if (!is_check and e.is_bad_capture) {
            //     pos.undo_move(us);
            //     continue; // post-move
            // }

            score = -self.qsearch(qdepth + 1, -beta, -alpha, them);

            pos.undo_move(us);

            if (self.stopped) {
                return 0;
            }

            // Better move.
            if (score > alpha) {
                alpha = score;
                if (score >= beta) {
                    self.stats.beta_cutoffs += 1;
                    return score;
                }
            }
        }
        return alpha;
    }

    fn record_alpha_improvement(self: *Searcher, depth: Value, e: ExtMove, improvement: Value) void {
        _ = self; _ = depth; _ = e; _ = improvement;
       // self.alpha_history[e.moved_piece.u][e.move.to.u] += @min(improvement, 1000) + improvement + ((depth * depth) * 2);
    }

    fn record_beta_cutoff(self: *Searcher, node: *Node, depth: Value, e: ExtMove) void {
        const m: Move = e.move;
        self.beta_history[e.moved_piece.u][m.to.u] += ((depth * depth) * 3);
        if (node.killers[0] != m) {
            node.killers[1] = node.killers[0];
            node.killers[0] = m;
        }
    }

    fn tt_store(self: *Searcher, bound: Bound, key: u64, depth: u8, ply: u16, move: Move, score: Value) void {
        if (!comptime use.tt) return;
        if (lib.is_paranoid) {
            assert(!self.stopped);
        }
        self.transpositiontable.store(bound, key, depth, ply, move, score);
    }

    /// Hashmove can be used for move ordering when not empty.
    fn tt_probe(self: *Searcher, key: u64, depth: u8, ply: u16, alpha: Value, beta: Value, tt_move: *Move) ?Value {
        if (!comptime use.tt) return null;
        if (lib.is_paranoid) assert(tt_move.* == Move.empty);
        return self.transpositiontable.probe(key, depth, ply, alpha, beta, tt_move);
    }

    fn check_stop(self: *Searcher) bool {
        if (self.stopped) return true;

        const node_interval: bool = self.stats.nodes & 2047 == 0;

        if (node_interval) {
            if (!self.engine.is_busy()) {
                self.stopped = true;
                return true;
            }
        }

        switch (self.termination) {
            .nodes => {
                if (self.stats.nodes >= self.engine.time_params.max_nodes) {
                    self.stopped = true;
                }
            },
            .depth => {
                if (self.iteration > self.engine.time_params.max_depth) {
                    self.stopped = true;
                }
            },
            .movetime, .clock => {
                if (node_interval and self.timer.elapsed_ms() + 5 >= self.engine.time_params.max_movetime) {
                    //io.debugprint("STOP nodes {} time {} used {}\n", .{ self.stats.nodes, self.engine.time_params.max_movetime, self.timer.elapsed_ms() });
                    self.stopped = true;
                }
            },
            else => {
                return false;
            },
        }
        return self.stopped;
    }
};

pub const Node = struct {
    /// Local PV during search.
    pv: PV = .{},
    /// Current result.
    score: Value = 0,
    /// Extensions done (sum of chain). TODO: not used anymore.
    extension: u8 = 0,
    /// Killer move heuristic.
    killers: [2]Move = .{ .empty, .empty },

    const empty: Node = .{};

    fn init() Node {
        return .{};
    }

    /// Clear PV. Does not clear the killers.
    fn clear(self: *Node) void {
        self.pv.len = 0;
        self.score = 0;
        self.extension = 0;
    }

    fn first_move(self: *const Node) Move {
        return if (self.pv.len > 0) self.pv.buffer[0] else .empty;
    }

    /// Sets `pv` to `bestmove + childnode.pv` and sets `self.score` to `score`.
    fn update_pv(self: *Node, bestmove: Move, score: Value, childnode: *const Node) void {
        self.pv.len = 0;
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

    /// Zig-format for UCI pv output.
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
    /// Set during processing.
    is_bad_capture: bool,

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
    extmoves: [max_move_count]ExtMove,
    count: u8,

    const empty: MovePicker = .{ .extmoves = undefined, .count = 0 };

    /// Required function.
    pub fn reset(self: *MovePicker) void {
        // We only are allowed to use this one once.
        if (comptime lib.is_paranoid) assert(self.count == 0);
    }

    /// Required function.
    pub fn store(self: *MovePicker, move: Move) ?void {
        self.extmoves[self.count] = ExtMove{ .move = move, .score = 0, .moved_piece = Piece.NO_PIECE, .captured_piece = Piece.NO_PIECE, .is_quiet = false, .is_capture = false, .is_bad_capture = false };
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

    /// For quiescence search. Generate captures only, but if in check generate all evasions.
    fn generate_captures(self: *MovePicker, pos: *const Position, comptime us: Color) void {
        pos.generate_captures(us, self);
    }

    /// Fill ExtMove info and give each generated move an initial heuristic score. Crucial for the search algorithm.
    fn process_and_score_moves(self: *MovePicker, searcher: *const Searcher, pv_move: Move, tt_move: Move, killers: [2]Move) void {
        const pos: *const Position = &searcher.pos;

        const pv           : Value = 3_000_000;
        const hash         : Value = 2_000_000;
        const promotion    : Value = 1_000_000;
        const capture      : Value = 500_000;
        const killer1      : Value = 400_000;
        const killer2      : Value = 300_000;
        //const bad_capture  : Value = -500_000;

        for (self.slice()) |*e| {
            const m: Move = e.move;
            e.moved_piece = pos.board[m.from.u];

            switch (m.type) {
                .normal => {
                    e.captured_piece = pos.board[m.to.u];
                    if (e.captured_piece.is_piece()) {
                        e.is_capture = true;
                        const see = eval.see_score(pos, m);
                        e.is_bad_capture = see < 0;
                        //e.score = capture + see;
                        if (see >= 0)
                            e.score = capture + see
                        else
                            e.score = capture + see; // bad_capture + see; // bad_capture
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
                    e.score = 0; // TODO: not sure;
                },
            }

            // PV move
            if (m == pv_move)
                e.score += pv;

            // Hash move.
            if (m == tt_move)
                e.score += hash;

            // Killers only for quiet moves
            if (e.is_quiet) {
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
            //io.debugprint("[swap {} {}]", .{ best_idx, current_idx});
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

    pub const default_hash_size: u64 = 64;
    pub const min_hash_size: u64 = 1;
    pub const max_hash_size: u64 = 1024;

    pub const default_use_hash: bool = true;

    pub fn set_hash_size(self: *Options, value: u64) void {
        self.hash_size = std.math.clamp(value, min_hash_size, max_hash_size);
    }
};

pub const TimeParams = struct {
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


    const default: TimeParams = .{
        .termination = .infinite,
        .max_depth = 0,
        .max_nodes = 0,
        .max_movetime = 0,
        .time_left = .{ 0, 0 },
        .increment = .{ 0, 0 },
        .movestogo = 0,
    };

    /// Convert the UCI go params to something usable for our search.
    pub fn init(go_params: *const uci.Go, to_move: Color) TimeParams {
        var result: TimeParams = .default;

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

            result.max_movetime = (time / moves) + if (moves > 1 and inc > 0) (inc / 2) else 0;
            //if (result.movestogo == 1 and inc > 0) result.max_movetime -= (inc / 2);
            //io.debugprint("moves {} time {} inc {} -> max_movetime {}\n", .{ moves, time, inc, result.max_movetime});
            return result;
        }

        // Depth.
        if (go_params.depth) |depth| {
            result.termination = .depth;
            result.max_depth = @truncate(@min(depth, max_search_depth));
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

/// Not used (yet).
const Error = error {
    EngineIsStillRunning,
};
