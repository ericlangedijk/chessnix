// zig fmt: off

//! Engine + Search.

//! This chessnix 1.3 version contains borrowed stuff, adjusted to engine's needs.
//! Most alphabeta engines share a plethora of established ideas, but I want to mention these:
//! Evaluation (which I adjusted and enhanced) from Integral v3 to get me going. Starting with chessnix 2.0 other data will be used.
//! Time management ideas from Alexandria.
//! Alpha Raise Reduction idea from Pawnocchio.

const std = @import("std");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const uci = @import("uci.zig");
const hce = @import("hce.zig");
const tt = @import("tt.zig");
const searchterms = @import("searchterms.zig");
const history = @import("history.zig");
const movepick = @import("movepick.zig");
const time = @import("time.zig");
const scoring = @import("scoring.zig");

const assert = std.debug.assert;
const clamp = std.math.clamp;

const ctx = lib.ctx;
const io = lib.io;

const wtf = lib.wtf;

const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const ExtMove = types.ExtMove;
const ExtMoveList = types.ExtMoveList;
const Position = position.Position;
const Entry = tt.Entry;
const Bound = tt.Bound;
const Evaluator = hce.Evaluator;
const History = history.History;
const MovePicker = movepick.MovePicker;
const TimeManager = time.TimeManager;

const tuned = searchterms.tuned;
const max_game_length: u16 = types.max_game_length;
const max_move_count: u8 = types.max_move_count;
const max_search_depth: u8 = types.max_search_depth;

const null_score = scoring.null_score;
const infinity = scoring.infinity;
const mate = scoring.mate;
const mate_threshold = scoring.mate_threshold;
const draw = scoring.draw;
const stalemate = scoring.stalemate;
const is_matescore = scoring.is_matescore;
const is_drawscore = scoring.is_drawscore;

/// Principal Variation (+ 2 to be safe).
const PV = utils.BoundedArray(Move, max_search_depth + 2);

pub const Engine = struct {
    /// Shared tt for all threads.
    transpositiontable: tt.TranspositionTable,

    //pawntable: tt.PawnTable,

    /// Threading.
    busy: std.atomic.Value(bool),
    /// The number of threads we use. Currently single threaded only.
    num_threads: usize,
    /// The controller thread, managing the search threads. Currently single threaded only.
    controller_thread: ?std.Thread,
    /// The working thread. Later this will be more threads. Currently single threaded only.
    search_thread: ?std.Thread,
    /// Our source position.
    pos: Position,
    /// Array with hash keys.
    repetition_table: [max_game_length]u64,
    /// The available moves at the root. Filled after both ucinewgame and position.
    rootmoves: position.MoveStorage,
    /// The uci engine options like hashsize etc.
    options: Options,
    /// Time management.
    tm: TimeManager,
    /// The actual worker. Currently single threaded only.
    searcher: Searcher,
    /// No output if true.
    mute: bool,

    /// Engine is created on the heap.
    /// The engine will be ready for a 'go' directly after (1) creation, (2) ucinewgame command and (3) position command.
    pub fn create(mute: bool) !*Engine {
        const tt_size: usize = tt.compute_tt_size_in_bytes(Options.default_hash_size);
        var engine: *Engine = try ctx.galloc.create(Engine);
        engine.transpositiontable = try .init(tt_size);
        //engine.pawntable = try .init();
        engine.busy = std.atomic.Value(bool).init(false);
        engine.num_threads = 1;
        engine.controller_thread = null;
        engine.search_thread = null;
        engine.repetition_table = @splat(0);
        engine.pos = .empty;
        engine.pos.set_startpos();
        engine.repetition_table[0] = engine.pos.key;
        engine.pos.lazy_generate_all_moves(&engine.rootmoves);
        engine.options = Options.default;
        engine.tm = TimeManager.empty;
        engine.searcher = Searcher.init();
        engine.mute = mute;
        return engine;
    }

    pub fn destroy(self: *Engine) void {
        self.transpositiontable.deinit();
        //self.pawntable.deinit();
        ctx.galloc.destroy(self);
    }

    /// After the options are determined. Bit of a mess...
    pub fn apply_hash_size(self: *Engine) !void {
        if (self.is_busy()) {
            return;
        }
        const tt_size: usize = tt.compute_tt_size_in_bytes(self.options.hash_size);
        try self.transpositiontable.resize(tt_size);
    }

    pub fn ucinewgame(self: *Engine) !void {
        if (self.is_busy()) {
            return;
        }
        self.pos.set_startpos();
        self.repetition_table[0] = self.pos.key;
        self.transpositiontable.clear();
        //self.pawntable.clear();
        self.pos.lazy_generate_all_moves(&self.rootmoves);
        self.searcher.new_game();
    }

    /// Sets the position from fen + moves. If fen is illegal we crash. If fen is null the startpostion will be set. After an illegal move we stop without crashing.
    pub fn set_position(self: *Engine, fen: []const u8, moves: ?[]const u8) !void {
        if (self.is_busy()) {
            return;
        }
        try self.pos.set(fen, self.options.is_960);
        self.repetition_table[0] = self.pos.key;
        if (moves) |m| {
            self.parse_moves(m);
        }
        self.pos.lazy_generate_all_moves(&self.rootmoves);
    }

    /// If we have any moves, make them. We stop if we encounter an illegal move without crashing.
    fn parse_moves(self: *Engine, moves: []const u8) void {
        var tokenizer = std.mem.tokenizeScalar(u8, moves, ' ');
        var idx: usize = 1;
        while (tokenizer.next()) |m| {
            const ex: ExtMove = self.pos.parse_move(m) catch break;
            self.pos.lazy_do_move(ex);
            self.repetition_table[idx] = self.pos.key;
            idx += 1;
        }
    }

    pub fn is_busy(self: *const Engine) bool {
        return self.busy.load(.acquire);
    }

    /// Start threaded search.
    pub fn go(self: *Engine, go_params: *const uci.Go) !void {
        if (self.is_busy()) {
            return;
        }
        // Set our busy flag.
        self.busy.store(true, .release);

        self.transpositiontable.decay();

        // Set and start time manager.
        self.tm.set(go_params, &self.pos, self.rootmoves.count == 1);

        // Spawn thread.
        self.controller_thread = try std.Thread.spawn(.{}, run_controller_thread, .{ self });
    }

    /// Stop threaded search.
    pub fn stop(self: *Engine) !void {
        if (!self.is_busy()) {
            return;
        }
        // Force the searcher to stop. Only this boss engine may do this atomic store to the searchers.
        @atomicStore(bool, &self.searcher.stopped, true, .release);
        self.busy.store(false, .release);
    }

    /// Stop the program. TODO: searcher still outputs bestmove.
    pub fn quit(self: *Engine) !void {
        if (!self.is_busy()) {
            return;
        }
        // Force the searcher to stop. Only this boss engine may do this atomic store to the searchers.
        @atomicStore(bool, &self.searcher.stopped, true, .release);
        self.busy.store(false, .release);
    }

    fn run_controller_thread(self: *Engine) !void {
        defer {
            self.controller_thread = null;
        }
        self.search_thread = try std.Thread.spawn(.{}, Searcher.go, .{ &self.searcher });
        if (self.search_thread) |th| {
            th.join();
            self.search_thread = null;
        }
        self.busy.store(false, .release);
    }

    pub fn see(self: *Engine, move: []const u8, threshold: i32) !bool {
        if (self.is_busy()) {
            return false;
        }
        const ex: ExtMove = try self.pos.parse_move(move);
        return hce.see(&self.pos, ex.move, threshold);
    }

    // /// For position testing without output. TODO: rewrite this.
    // pub fn give_best_move(self: *Engine, fen: []const u8, moves: []const u8, movetime: u32) !Move {
    //     if (self.is_busy()) return .empty;
    //     try self.ucinewgame();
    //     self.mute = true;
    //     try self.set_position(fen, moves);
    //     var uci_go: uci.Go = .empty;
    //     uci_go.movetime = movetime;
    //     try self.go(&go);
    //     // Run thread.
    //     while (self.is_busy()) {
    //     }
    //     return self.searcher.selected_move;
    // }
};

pub const Stats = struct {
    /// Calls to search or quiescence search
    search_calls: u64 = 0,
    /// The total amount of nodes.
    nodes: u64 = 0,
    /// The total amount of quiescent search nodes.
    qnodes: u64 = 0,
    /// Iteration reached.
    rootdepth: u16 = 0,
    /// The max reached depth.
    seldepth: u16 = 0,
    /// For move ordering stats
    non_terminal_nodes: u64 = 0,
    /// For move ordering stats
    beta_cutoffs: u64 = 0,

    const empty: Stats = .{};
};

/// Indicates if we are searching for principal variation.
const SearchMode = enum {
    pv,
    not_pv,
};

/// Indicates if we are in the first iteration. If so then ply must be 0.
const RootMode = enum {
    root,
    not_root,
};

pub const Nodes = [max_search_depth + 2]Node;

pub const Searcher = struct {
    /// Backlink ref. Only valid after go. Retrieved through ownership.
    engine: *Engine,
    /// Copied from engine for direct access. Only valid after go().
    transpositiontable: *tt.TranspositionTable,
    /// Copied from engine for direct access. Only valid after go().
    tm: *TimeManager,
    /// The evaluator
    evaluator: hce.Evaluator,
    /// For now this is a blunt copy from the Engine.
    repetition_table: [max_game_length]u64,
    /// Our node stack. Index 0 is our main pv.
    nodes: Nodes,

    /// Nodes spent on move for time heuristics.
    nodes_spent: [64 * 64]u64,

    //nodes_spent: [max_move_count]u64,


    /// History heuristics for beta cutoffs.
    hist: History,
    /// Timeout. Any search value (0) returned - when stopped - should be discarded directly.
    stopped: bool,

    /// Statistics.
    stats: Stats,

    fn init() Searcher {
        return .{
            .engine = undefined,
            .transpositiontable = undefined,
            .tm = undefined,
            .evaluator = .init(),
            .repetition_table = @splat(0),
            .nodes = create_nodes(),
            .nodes_spent = @splat(0),
            .hist = .init(),
            .stopped = false,
            .stats = .empty,
        };
    }

    /// Create nodes. Set all their 'readonly' ply numbers.
    fn create_nodes() Nodes {
        var nodes: Nodes = @splat(Node.empty);
        for (&nodes, 0..) |*node, idx| {
            node.ply = @intCast(idx);
        }
        return nodes;
    }

    /// To save time we do this in ucinewgame.
    fn new_game(self: *Searcher) void {
        self.hist.clear();
    }

    fn go(self: *Searcher) void {
        // Backlink stuff from engine.
        self.engine = @fieldParentPtr("searcher", self);
        self.transpositiontable = &self.engine.transpositiontable;
        self.tm = &self.engine.tm;

        // Copy stuff.
        const len: u16 = self.engine.pos.ply_from_root + 1;
        @memcpy(self.repetition_table[0..len], self.engine.repetition_table[0..len]);

        // Reset stuff.
        self.stopped = false;
        self.stats = .empty;
        self.nodes_spent = @splat(0);

        // Make our local copy of the position.
        var pos: Position = self.engine.pos;

        switch (pos.stm.e) {
            .white => self.iterate(Color.WHITE, &pos),
            .black => self.iterate(Color.BLACK, &pos),
        }
    }

    fn iterate(self: *Searcher, comptime us: Color, pos: *const Position) void {
        // Print some time info once (#experimental)
        // TODO: print some more info (non-debug just standard).
        // self.tm.print_info(pos.stm);

        // const only_one_move: bool = self.engine.rootmoves.count == 1;
        const rootnode: *Node = &self.nodes[0];
        const max_depth: u8 = if (self.tm.termination == .depth) self.tm.max_depth else max_search_depth;
        const min_clock_depth: u8 = if (self.tm.termination == .clock) 7 else 255;

        var best_move: Move = .empty;
        var previous_best_move: Move = .empty;
        var previous_score: i32 = 0;
        var average_score: i32 = null_score;
        var best_move_stability: u3 = 0;
        var eval_stability: u3 = 0;
        var iteration_timer: utils.Timer = .start();
        //var last_iteration_times: [4]u64 = @splat(0);
        //var last_prediction: u64 = 0;
        //var prediction_correction: f64 = 1.0;

        // Statistics stuff.
        self.stats.non_terminal_nodes = 0;
        self.stats.beta_cutoffs = 0;

        rootnode.best_move = .empty;

        // Search with increasing depth.
        iterationloop: for (1..max_depth + 1) |d| {
            const depth: u8 = @intCast(d);
            self.stats.rootdepth = depth;
            self.stats.seldepth = 0;
            iteration_timer.reset();

            // Give the eval stability some slack at greater depths. This influences time management.
            // The idea is that at a greater depth we care more about move stability than eval stability.
            const slack: i32 = depth / tuned.eval_stability_slack_depth_divider;

            const start_non_terminal: u64 = self.stats.non_terminal_nodes;
            const start_beta_cutoffs: u64 = self.stats.beta_cutoffs;

            const avg: i32 = if (average_score == null_score) 0 else average_score;
            const score: i32 = self.aspiration_search(us, pos, avg, depth);

            if (!rootnode.best_move.is_empty()) {
                best_move = rootnode.best_move;
            }

            // Discard if timeout.
            if (self.stopped) {
                break :iterationloop;
            }

            // Time management: Keep track of the average score.
            average_score = if (average_score == null_score) score else @divFloor(average_score + score, 2);

            // Time management: Keep track of how many times in a row the best move stays the same.
            if (best_move == previous_best_move) {
                best_move_stability = @min(4, best_move_stability + 1);
            }
            else {
                best_move_stability = 0;
            }

            // Time management: Keep track of the evaluation stability.
            if (score > average_score - tuned.eval_stability_margin - slack and score < average_score + tuned.eval_stability_margin + slack) {
                eval_stability = @min(4, eval_stability + 1);
            }
            else {
                eval_stability = 0;
            }

            // Save previous stuff.
            previous_best_move = best_move;
            previous_score = score;

            // And print the info and pv.
            const formatted_score: []const u8 = scoring.format_score(rootnode.score).slice();
            //const cp: i32 = if (!scoring.is_drawscore(rootnode.score)) rootnode.score else 0;
            const non_terminal_used: u64 = self.stats.non_terminal_nodes - start_non_terminal;
            const beta_cutoffs_used: u64 = self.stats.beta_cutoffs - start_beta_cutoffs;
            const qnodes: usize = funcs.percent(self.stats.nodes, self.stats.qnodes);
            const search_efficiency = funcs.percent(non_terminal_used, beta_cutoffs_used);
            const elapsed_nanos = self.tm.timer.read();
            const ms: u64 = elapsed_nanos / 1_000_000;
            const nps: u64 = funcs.nps(self.stats.nodes, elapsed_nanos);
            const cps: u64 = funcs.nps(self.stats.search_calls, elapsed_nanos);

            if (!self.engine.mute) {
                // TODO: because of the "Crazy Mate Scores Bug or Feature" we display no mate in X (yet).
                io.print_buffered(
                    // "info depth {} seldepth {} score cp {} nodes {} qnodes {}% time {} nps {} cps {} eff {}% pv",
                    // .{ self.stats.rootdepth, self.stats.seldepth, cp, self.stats.nodes, qnodes, ms, nps, cps, search_efficiency }

                    "info depth {} seldepth {} score {s} nodes {} qnodes {}% time {} nps {} cps {} eff {}% pv",
                    .{ self.stats.rootdepth, self.stats.seldepth, formatted_score, self.stats.nodes, qnodes, ms, nps, cps, search_efficiency }
                );

                for (rootnode.pv.slice()) |move| {
                    io.print_buffered(" ", .{});
                    move.print_buffered(pos.is_960);
                }
                io.print("\n", .{}); // also flushes.
                //io.print_buffered("\n", .{});
                //io.flush();
            }

            // Keep track of iteration times.
            const iter_time: u64 = iteration_timer.read();
            // inline for (1..4) |i| {
            //     last_iteration_times[i - 1] = last_iteration_times[i];
            // }
            // last_iteration_times[3] = iter_time;

            // When clockmode update the optimal time to spend and check if we can stop.
            // Addionally check if there is time for a next iteration.
            if (depth >= min_clock_depth) {

                const nodes_spent_on_move: u64 = self.nodes_spent[best_move.from_to()];
                self.tm.update_optimal_stoptime(nodes_spent_on_move, self.stats.nodes, best_move_stability, eval_stability);
                if (self.tm.optimal_time_reached(elapsed_nanos)) {
                    self.stopped = true;
                    break :iterationloop;
                }

                // Try to estimate if we have time left for another iteration.
                // This is very unpredictable (the times are very spiky) but at least it does something.
                const next_cost: u64 = funcs.fmul(iter_time, 1.20);
                if (elapsed_nanos + next_cost >= self.tm.max_endtime) {
                    self.stopped = true;
                    break :iterationloop;
                }

                // inline for (0..4) |i| {
                //     io.print_buffered("[{}] ", .{ last_iteration_times[i] / 1_000_000});
                // }
                // io.print("\n", .{});
                // const ema: u64 =
                //     last_iteration_times[0] * 1 / 10 +
                //     last_iteration_times[1] * 2 / 10 +
                //     last_iteration_times[2] * 3 / 10 +
                //     last_iteration_times[3] * 4 / 10;
                // const remaining: u64 = self.tm.max_endtime - elapsed_nanos;
                // //const predicted = @max(ema * 2, iter_time * 2);
                // //const next_cost: u64 = @min(predicted, remaining / 2);
                // const next_cost: u64 = @min(ema * 2, remaining / 2);
                // //prediction_correction = funcs.float64(iter_time) / funcs.float64(last_prediction);
                // if (last_prediction >= iter_time) {
                //     io.print("err +{}\n", .{ (last_prediction - iter_time) / 1_000_000 });
                // }
                // else {
                //     io.print("err -{}\n", .{ (iter_time - last_prediction) / 1_000_000 });
                // }

                // last_prediction = next_cost;

                // io.print("max {} opt {} rem {} nextcost {}\n", .{ self.tm.max_ms, (self.tm.opt_endtime - self.tm.started) / 1_000_000,  remaining / 1_000_000, next_cost / 1_000_000 });

                // if (elapsed_nanos + next_cost >= self.tm.max_endtime) {
                //     io.print("no time for next it", .{});
                //     self.stopped = true;
                //     break :iterationloop;
                // }
            }
        } // (iterationloop)

        // Finally print the best move.
        if (!self.engine.mute) {
            io.print_buffered("bestmove ", .{} );
            best_move.print_buffered(pos.is_960);
            io.print_buffered("\n", .{});
            io.flush();
        }
    }

    fn aspiration_search(self: *Searcher, comptime us: Color, pos: *const Position, previous_score: i32, depth: i32) i32 {
        var delta: i32 = 9;
        var alpha: i32 = -infinity;
        var beta: i32 = infinity;

        const rootnode: *Node = &self.nodes[0];

        if (depth >= 6) {
            alpha = @max(-infinity, previous_score - delta);
            beta = @min(infinity, previous_score + delta);
        }
        var score: i32 = 0;
        var searchdepth: i32 = depth;

        while (true) {
            rootnode.clear(.clear_fields);

            // Search with aspiration window.
            score = self.search(us, .pv, .root, false, pos, rootnode, searchdepth, alpha, beta);

            // Discard result if timeout.
            if (self.stopped) {
                return 0;
            }

            // Alpha: fail low.
            if (score <= alpha) {
                beta = @divFloor(alpha + beta, 2);
                alpha = @max(-infinity, score - delta);
                searchdepth = depth;
            }
            // Beta: fail high.
            else if (score >= beta) {
                beta = @min(infinity, score + delta);
                if (alpha < 2000) {
                    searchdepth = @max(1, depth - 1);
                }
            }
            // Success.
            else {
                break;
            }
            delta = @divTrunc(delta * 155, 100);
        }
        return score;
    }

    fn search(self: *Searcher, comptime us: Color, comptime mode: SearchMode, comptime rootmode: RootMode, comptime cutnode: bool, pos: *const Position, node: *Node, input_depth: i32, input_alpha: i32, input_beta: i32) i32 {
        const them: Color = comptime us.opp();
        const is_pvs: bool = comptime mode == .pv;
        const is_root: bool = comptime rootmode == .root;
        const ply: u16 = node.ply;

        if (comptime lib.verifications) {
            lib.verify(input_depth >= 0, "search() invalid depth", .{});
            lib.verify(!is_root or is_pvs, "search() root must be pvs", .{});
            lib.verify(input_beta > input_alpha, "search() invalid alphabeta", .{});
        }

        assert(input_depth >= 0);
        assert(!is_root or is_pvs);
        assert(input_beta > input_alpha);

        self.stats.search_calls += 1;

        const parentnode: ?*const Node = if (!is_root) &self.nodes[ply - 1] else null;
        const childnode: *Node = &self.nodes[ply + 1];
        node.clear(.clear_pv);
        childnode.clear(.clear_fields);

        // Too deep.
        if (ply >= max_search_depth) {
            return self.evaluate(pos);
        }

        // Update max reached depth.
        self.stats.seldepth = @max(self.stats.seldepth, ply);

        var depth: i32 = input_depth;
        var alpha: i32 = input_alpha;
        var beta: i32 = input_beta;

        // When reaching depth limit dive into quiescent search.
        if (input_depth == 0) {
            return self.quiescence_search(us, mode, pos, node, alpha, beta);
        }

        // Adjust window.
        if (!is_root) {
            if (pos.is_draw_by_insufficient_material() or self.is_draw_by_repetition_or_rule50(pos)) {
                return scoring.drawscore(self.stats.nodes);
            }

            const worst_possible: i32 = -mate + ply;
            const best_possible: i32 = mate - ply - 1;
            alpha = @max(alpha, worst_possible);
            beta = @min(beta, best_possible);

            // A beta cutoff can occur after adjusting the window.
            if (alpha >= beta) {
                return alpha;
            }
        }

        // Check timeout at the start only.
        if (self.must_stop()) {
            return 0;
        }

        const is_singular_extension: bool = !node.excluded_tt_move.is_empty();

        // TT Probe.
        const tt_entry: tt.Entry = self.transpositiontable.probe(pos.key, ply); // TODO: use the if check and if !is_singular_extension smarter
        // Remember that tt_hit can be true just hitting a raw static eval. Don't trust it.
        const tt_hit: bool = !is_singular_extension and !tt_entry.is_empty();
        const tt_move: Move = if (tt_hit) tt_entry.move else .empty;

        if (!is_pvs and tt_hit and tt_entry.is_score_usable_for_depth(alpha, beta, depth)) {
            return tt_entry.score;
        }

        const original_alpha = alpha;
        const is_check: bool = pos.checkmask != 0;
        var extension: i32 = 0;
        var reduction: i32 = 0;
        var raw_static_eval: i32 = null_score;
        var corrected_raw_static_eval: i32 = null_score;
        var is_tt_eval: bool = false;

        node.static_eval = null_score;
        node.eval = null_score;

        // Static Evaluation.
        if (!is_check and !is_singular_extension) {

            // Get the static eval from the TT if it is there, otherwise compute and store it.
            if (tt_hit and tt_entry.is_raw_static_eval_usable()) {
                raw_static_eval = tt_entry.raw_static_eval;
            }
            else {
                raw_static_eval = self.evaluate(pos);
                self.transpositiontable.store_raw_static_eval(pos.key, raw_static_eval);
            }

            corrected_raw_static_eval = self.correct_raw_static_eval(us, pos, raw_static_eval);
            node.static_eval = apply_drawcounter(pos, corrected_raw_static_eval);

            // Use TT score as a better eval.
            // #CrazyMateScores
            if (tt_hit and tt_entry.is_score_usable_as_eval(node.static_eval)) {
                is_tt_eval = true;
                node.eval = tt_entry.score;
            }
            else {
                node.eval = node.static_eval;
            }
        }

        // Tricky stuff.
        if (comptime lib.verifications) {
            if (!is_check and !is_singular_extension and (raw_static_eval == null_score or corrected_raw_static_eval == null_score or node.static_eval == null_score or node.eval == null_score)) {
                lib.wtf("search() eval not filled", .{});
            }
            if (is_singular_extension and !tt_move.is_empty()) {
                lib.wtf("search() invalid singular extension", .{});
            }
        }

        // Check improvement compared to a previous node with the same side to move. Improvement is 0 or 1.
        var is_improving: bool = false;
        var is_opponent_worsening: bool = false;
        var is_complex: bool = false;

        if (!is_check and node.static_eval != null_score) {
            const prevnode: ?*const Node =
                if (ply >= 2 and self.nodes[ply - 2].static_eval != null_score) &self.nodes[ply - 2]
                else if (ply >= 4 and self.nodes[ply - 4].static_eval != null_score) &self.nodes[ply - 4]
                else null;

            is_complex = @abs(corrected_raw_static_eval - raw_static_eval) >= tuned.corr_hist_is_complex_margin;

            if (prevnode != null) {
                is_improving = node.static_eval > prevnode.?.static_eval;
                is_opponent_worsening = parentnode.?.static_eval != null_score and node.static_eval + parentnode.?.static_eval > 1;
            }
        }

        // Pruning before processing any moves (the whole node is pruned).
        if (!is_pvs and !is_check and !is_singular_extension and !is_matescore(alpha) and !is_matescore(beta)) {

            // Reversed Futility Pruning (rfp): beta pruning.
            if (depth <= tuned.rfp_max_depth and node.eval < mate_threshold and node.eval - tuned.rfp_min_margin >= beta) {
                // If we are improving, the margin to beat beta can be smaller.
                var mult: i32 = if (is_improving) tuned.rfp_improving_margin else tuned.rfp_not_improving_margin;
                if (is_opponent_worsening) mult -= 8;
                const rfp_margin: i32 = (depth * mult);
                if (node.eval - rfp_margin >= beta) {
                    return node.eval;
                }
            }

            // Razoring: alpha pruning.
            const depth_3 = @max(0, depth - 3);
            if (node.static_eval + tuned.razor_base + depth * tuned.razor_mult + depth_3 * depth_3 * tuned.razor_quad <= alpha) {
                const r_score = if (is_tt_eval) node.eval else self.quiescence_search(us, mode, pos, node, alpha, alpha + 1);
                if (self.stopped) {
                    return 0;
                }
                if (r_score <= alpha) {
                    return r_score;
                }
            }

            // TODO: use verification.
            // Null Move Pruning (nmp). Are we still good if we let the opponent play another move?
            //if (!pos.nullmove_state and node.eval >= beta and node.static_eval >= beta + 170 - depth * 24 and pos.phase() > 0 and pos.pieces_except_pawns_and_kings(us) != 0) { // TODO: use count from material?
            if (
                // depth >= 2 and // #testing
                !pos.nullmove_state and
                node.eval >= beta and
                node.static_eval >= beta + 170 - depth * 24 and
                pos.minor_major_count(us) != 0
            ) {
                const eval_reduction: i32 = @min(2, @divTrunc(node.eval - beta, 202));
                const r: i32 = clamp(@divTrunc(depth, 4) + 3 + eval_reduction, 0, depth);
                const next_pos: Position = self.do_nullmove(pos, us);
                node.clear(.undo);
                const nmp_score: i32 = -self.search(them, .not_pv, .not_root, !cutnode, &next_pos, childnode, depth - r, -beta, -beta + 1);
                if (self.stopped) {
                    return 0;
                }
                if (nmp_score >= beta) {
                    // Do not trust matescores.
                    return if (nmp_score >= mate_threshold) beta else nmp_score;
                }
            }
        }

        // Internal Iterative Reduction (iir): If we have no tt move, move ordering is not optimal.
        if ((is_pvs or cutnode) and !is_singular_extension and depth >= tuned.iir_min_depth and tt_move.is_empty()) {
            depth -= 1;
        }

        // Move ordering statistics.
        self.stats.non_terminal_nodes += 1;
        node.double_extensions = if (!is_root) parentnode.?.double_extensions else 0;

        // Some locals for our move loop.
        var quiet_list: ExtMoveList(tuned.search_quiet_list_size) = .init();
        var capture_list: ExtMoveList(tuned.search_capture_list_size) = .init();
        var movepicker: MovePicker(.search, us) = .init(pos, self, node, tt_move);
        var best_move: Move = .empty;
        var best_score: i32 = -infinity;
        var moves_seen: u8 = 0;
        var alpha_raises: u8 = 0;

        // Move loop.
        moveloop: while (movepicker.next()) |ex| {
            self.prefetch(us, pos, ex);

            // Skip this move if we are inside singular extensions.
            if (ex.move == node.excluded_tt_move) {
                continue :moveloop;
            }

            const is_quiet_move: bool = ex.move.is_quiet();
            const is_capture_move: bool = ex.move.is_capture();
            const quiet_history_score: i32 = if (is_quiet_move) self.hist.get_quiet_score(ex, ply, &self.nodes) else 0;

            // Prunings before we actually execute the move.
            if (!is_root and best_score > -mate_threshold) {

                // Late Move Pruning (lmp).
                if (is_quiet_move) {
                    const lmp_threshold: i32 = @divTrunc(5 + (depth * depth), @as(i32, 3) - @intFromBool(is_improving));
                    if (moves_seen >= lmp_threshold) {
                        movepicker.skip_quiets();
                        continue :moveloop;
                    }
                }

                // Futility Pruning (fp).
                const fp_margin: i32 = tuned.fp_margin_base + (depth * tuned.fp_depth_mult);
                if (!is_check and is_quiet_move and depth <= tuned.fp_max_depth and node.eval != null_score and node.eval + fp_margin < alpha) {
                    movepicker.skip_quiets();
                    continue :moveloop;
                }

                // SEE pruning.
                const see_threshold: i32 = if (is_quiet_move) depth * tuned.see_prune_quiet_mult else depth * tuned.see_prune_noisy_mult;
                if (depth <= tuned.see_prune_max_depth and moves_seen >= 1 and !hce.see(pos, ex.move, -see_threshold)) {
                    continue :moveloop;
                }

                // History Pruning (hp).
                if (is_quiet_move and depth <= tuned.hp_max_depth) {
                    if (quiet_history_score < tuned.hp_quiet_offset - (depth * tuned.hp_quiet_mult)) {
                        movepicker.skip_quiets();
                        continue :moveloop;
                    }
                }
            }

            // Determine extension + reduction.
            extension = 0;
            reduction = 0;

            // Singular Extensions (se).
            if (!is_root and !is_singular_extension and depth >= 8 and ex.is_tt_move) {
                const accurate_tt_score: bool = tt_entry.flags.bound != .alpha and tt_entry.depth + 4 >= depth and !is_matescore(tt_entry.score);
                if (accurate_tt_score) {
                    node.excluded_tt_move = ex.move;
                    const reduced_depth: i32 = @divTrunc(depth - 1, 2);
                    const s_beta = tt_entry.score - (depth * 2);
                    const s_score = self.search(us, .not_pv, .not_root, cutnode, pos, node, reduced_depth, s_beta - 1, s_beta);
                    node.excluded_tt_move = .empty;
                    if (self.stopped) {
                        return 0;
                    }

                    // No move beats the tt_move, so extend its search.
                    if (s_score < s_beta) {
                        // Double extend if the tt move is singular by some margin
                        if (!is_pvs and s_score < s_beta - 28 and node.double_extensions <= 8) {
                            extension = 2;
                            node.double_extensions += 1;
                        }
                        else {
                            extension = 1;
                        }
                    }

                    // Singular search beta cutoff, indicating that the tt_move was not singular. So we prune if the same score would cause a cutoff based with this search window
                    else if (s_beta >= beta) {
                        return s_beta;
                    }
                    // Negative Extensions: Search less since the tt_move was not singular and it might cause a beta cutoff again.
                    else if (tt_entry.score >= beta) {
                        extension = -1;
                    }
                }
            }

            if (is_check) {
                extension += 1;
            }

            // Do move
            const next_pos: Position = self.do_move(pos, us, ex);
            defer node.clear(.undo);
            const gives_check: bool = next_pos.checkmask != 0;
            node.current_move = ex;
            node.continuation_entry = self.hist.continuation.get_continuation_entry_for_node(ex);
            self.stats.nodes += 1;

            // Keep track of nodes spent on this move for time management.
            const nodes_before_search: u64 = if (is_root) self.stats.nodes else 0;

            var need_full_search: bool = false;
            var score: i32 = -infinity;
            var new_depth: i32 = depth + extension - 1;
            // We clamp this one.
            new_depth = clamp(new_depth, 0, max_search_depth);

            // Late Move Reduction (lmr). First try a reduced search on late moves.
            const lmr_move_threshold: i32 = if (is_root) 3 else 1;
            if (depth >= tuned.lmr_min_depth and moves_seen >= lmr_move_threshold) {

                // Retrieve default reduction from the lmr table.
                reduction = get_lmr(is_quiet_move, depth, moves_seen);

                // Reduce more if cutnode.
                if (cutnode) {
                    reduction += 1;
                }

                // Alpha Raise Reduction
                reduction += alpha_raises;

                // Reduce more if not pvs.
                if (!is_pvs) {
                    reduction += 1;
                }

                // History reduction.
                if (is_quiet_move) {
                    reduction -= @divFloor(quiet_history_score, tuned.lmr_history_divider);
                }

                // And reduce less in these cases.
                if (gives_check or is_complex) {
                    reduction -= 1;
                }

                // Reduction must not result in a negative depth.
                reduction = clamp(reduction, 0, new_depth - 1);

                // Search nullwindow here.
                score = -self.search(them, .not_pv, .not_root, true, &next_pos, childnode, new_depth - reduction, -alpha - 1, -alpha);
                if (self.stopped) {
                    return 0;
                }

                need_full_search = score > alpha and reduction != 0;

                // // #testing
                // if (need_full_search) {
                //     const go_deeper: bool = new_depth < max_search_depth and score > (best_score + 35 + 2 * new_depth);
                //     const go_shallower = new_depth > 0 and score < best_score + 8;
                //     new_depth += @intFromBool(go_deeper);
                //     new_depth -= @intFromBool(go_shallower);
                // }


            }
            else {
                // No lmr done.
                need_full_search = !is_pvs or moves_seen >= 1;
            }

            // The move (1) scored well from the reduced search above or (2) is probably not a pv move.
            // In these cases We first search with a null window.
            if (need_full_search) {
                score = -self.search(them, .not_pv, .not_root, !cutnode, &next_pos, childnode, new_depth, -alpha - 1, -alpha);
                if (self.stopped) {
                    return 0;
                }
            }

            // Full window search here.
            if (is_pvs and (score > alpha or moves_seen == 0)) {
                score = -self.search(them, .pv, .not_root, false, &next_pos, childnode, new_depth, -beta, -alpha);
                if (self.stopped) {
                    return 0;
                }
            }

            // We only count searched moves. Not the pruned ones.
            moves_seen += 1;

            // Keep track of how many nodes we spent on this root move. Only in tournament situation.
            if (is_root) {
                self.nodes_spent[ex.move.from_to()] += self.stats.nodes - nodes_before_search;
                //self.update_nodes_spent(ex.move, self.stats.nodes - nodes_before_search);
            }

            // Better move found.
            if (score > best_score) {
                best_score = score;
                // Alpha.
                if (score > alpha) {
                    best_move = ex.move;
                    alpha = score;
                    node.best_move = ex.move;

                    if (is_pvs) {
                        node.update_pv(best_move, best_score, childnode);
                    }

                    // Beta.
                    if (score >= beta) {
                        self.stats.beta_cutoffs += 1;
                        self.hist.record_beta_cutoff(depth, ply, ex, &self.nodes, quiet_list.slice());
                        break :moveloop;
                    }
                    // Alpha Raise Reduction.
                    if (depth >= tuned.lmr_min_depth) {
                        alpha_raises += 1;
                    }
                }
            }

            // Keep track of the moves that did not raise alpha.
            if (ex.move != best_move) {
                if (is_quiet_move) {
                    quiet_list.try_add(ex);
                }
                else if (is_capture_move) {
                    capture_list.try_add(ex);
                }
            }

        } // (moveloop)

        if (movepicker.move_count == 0) {
            return if (is_check) -mate + ply else stalemate;
        }

        // Punish captures that did not raise alpha.
        if (!best_move.is_empty()) {
            self.hist.capture.punish(depth, capture_list.slice());
        }

        if (!is_singular_extension) {
            // TT store.
            const bound: tt.Bound = if (best_score >= beta) .beta else if (best_score <= original_alpha) .alpha else .exact;

            // Keep tt move if possible.
            if (bound == .alpha and best_move.is_empty() and !tt_move.is_empty()) {
                best_move = tt_move;
            }
            self.transpositiontable.store(ply, pos.key, best_score, raw_static_eval, best_move, depth, bound);

            // Register correction: the difference between the search score and the static evaluation.
            // Always re-feed with the corrected value to avoid correction explosion.
            if (
                !is_check
                and !best_move.is_noisy()
                and !(bound == .alpha and best_score > corrected_raw_static_eval)
                and !(bound == .beta and best_score < corrected_raw_static_eval)
            ) {
                self.hist.correction.update(us, pos, depth, best_score, corrected_raw_static_eval);
            }
        }

        return best_score;
    }

    fn quiescence_search(self: *Searcher, comptime us: Color, comptime mode: SearchMode, pos: *const Position, node: *Node, input_alpha: i32, input_beta: i32) i32 {
        // Comptimes.
        const them = comptime us.opp();
        const is_pvs: bool = comptime mode == .pv;
        // TODO: do we need to clamp alpha beta like in search?

        const ply: u16 = node.ply;
        self.stats.search_calls +%= 1;

        const childnode: *Node = &self.nodes[ply + 1];
        node.clear(.clear_pv);
        childnode.clear(.clear_fields);

        // Too deep.
        if (ply >= max_search_depth) {
            return self.evaluate(pos);
        }

        // Update stats.
        self.stats.seldepth = @max(self.stats.seldepth, ply);

        // Forced draw?
        if (pos.is_draw_by_insufficient_material() or self.is_draw_by_repetition_or_rule50(pos)) {
            return scoring.drawscore(self.stats.nodes);
        }

        // Check timeout at the start only. At the callsites just read the stopped value.
        if (self.must_stop()) {
            return 0;
        }

        const is_check: bool = pos.checkmask > 0;
        const tt_depth: i32 = @intFromBool(is_check);

        var alpha: i32 = input_alpha;
        const beta: i32 = input_beta;

        // Probe TT.
        const tt_entry: tt.Entry = self.transpositiontable.probe(pos.key, ply);
        const tt_hit: bool = !tt_entry.is_empty();
        const tt_move: Move = if (tt_hit) tt_entry.move else .empty;

        // TT cutoff.
        if (!is_pvs and tt_hit and tt_entry.is_score_usable_for_depth(alpha, beta, tt_depth)) {
            return tt_entry.score;
        }

        var best_score: i32 = -infinity;
        var raw_static_eval: i32 = null_score;

        // Static Evaluation.
        if (!is_check) {
            // Get the static eval from the transposition table or compute and store it.
            if (tt_hit and tt_entry.is_raw_static_eval_usable()) {
                raw_static_eval = tt_entry.raw_static_eval;
            }
            else {
                raw_static_eval = self.evaluate(pos);
                self.transpositiontable.store_raw_static_eval(pos.key, raw_static_eval);
            }

            const corrected_raw_static_eval: i32 = self.correct_raw_static_eval(us, pos, raw_static_eval);
            best_score = apply_drawcounter(pos, corrected_raw_static_eval);

            // Use TT score as a better eval.
            // #CrazyMateScores
            if (tt_hit and tt_entry.is_score_usable_as_eval(best_score)) {
                best_score = tt_entry.score;
            }

            if (best_score >= beta) {
                return best_score;
            }

            // Adjust alpha.
            alpha = @max(alpha, best_score);
        }

        var movepicker: MovePicker(.quiescence, us) = .init(pos, self, node, tt_move);
        var moves_seen: u8 = 0;
        const qs_futility_score: i32 = best_score + tuned.qs_fp_margin;

        // Move loop.
        moveloop: while (movepicker.next()) |ex| {
            self.prefetch(them, pos, ex);

            // Skip bad noisies if we have seen a move already.
            if (moves_seen > 0 and ex.is_bad_capture) {
                break :moveloop;
            }

            // Quiescence Futility Pruning (qs_fp). Prune capture moves that do not win material if the static eval is behind alpha by some margin.
            if (!is_check and ex.move.is_capture() and qs_futility_score <= alpha and !hce.see(pos, ex.move, 1)) {
                best_score = @max(best_score, qs_futility_score);
                continue :moveloop;
            }

            // Do move.
            const next_pos: Position = self.do_move(pos, us, ex);
            defer node.clear(.undo);
            //self.prefetch2(pos.key);
            self.stats.nodes += 1;
            self.stats.qnodes += 1;

            const score: i32 = -self.quiescence_search(them, mode, &next_pos, childnode, -beta, -alpha);
            if (self.stopped) {
                return 0;
            }

            moves_seen += 1;

            // Better move.
            if (score > best_score) {
                best_score = score;
                // Alpha.
                if (score > alpha) {
                    // Beta.
                    if (score >= beta) {
                        break :moveloop;
                    }
                    alpha = score;
                }
            }
        }

        if (movepicker.move_count == 0 and is_check) {
            return -mate + ply;
        }

        // TT store. We do not store a best move because the quiescence search is incomplete by definition. Also we do not store exact.
        const bound: tt.Bound = if (best_score >= beta) .beta else .alpha;
        self.transpositiontable.store(ply, pos.key, best_score, raw_static_eval, .empty, tt_depth, bound);
        return best_score;
    }

    // fn index_of_rootmove(self: *const Searcher, move: Move) usize {
    //     for (self.engine.rootmoves.slice(), 0..) |ex, idx| {
    //         if (ex.move == move) {
    //             return idx;
    //         }
    //     }
    //     unreachable;
    // }

    // fn update_nodes_spent(self: *Searcher, move: Move, spent: u64) void {
    //     const idx: usize = self.index_of_rootmove(move);
    //     self.nodes_spent[idx] += spent;
    // }

    // fn get_nodes_spent(self: *Searcher, move: Move) u64 {
    //     const idx: usize = self.index_of_rootmove(move);
    //     return self.nodes_spent[idx];
    // }

    /// For prefetching the TT entry of the position to come. Memory access speedup.
    /// Assumes the move is not yet done on the board.
    /// For a null move pass an empty move.
    /// Apparently the prefetch is best done early in the move-loop.
    /// TODO: find out if prefetching after actually doing a move is better / faster.
    fn prefetch(self: *const Searcher, comptime us: Color, pos: *const Position, ex: ExtMove) void {
        const k: u64 = pos.predict_key(us, ex);
        @prefetch(self.transpositiontable.hash.get(k), .{});
    }

    // fn prefetch2(self: *const Searcher, key: u64) void {
    //     @prefetch(self.transpositiontable.hash.get(key), .{});
    // }

    /// Only call evaluate when not in check (or the max search depth has been reached).
    fn evaluate(self: *Searcher, pos: *const Position) i32 {
        //return self.evaluator.evaluate(pos, &self.engine.pawntable);
        return self.evaluator.evaluate(pos);
    }

    /// Apply correction history.
    fn correct_raw_static_eval(self: *const Searcher, comptime us: Color, pos: *const Position, raw_static_eval: i32) i32 {
        return self.hist.correction.apply(us, pos, raw_static_eval);
    }

    /// Slide the evaluation slightly towards draw.
    fn apply_drawcounter(pos: *const Position, eval: i32) i32 {
        // if (true) return eval;
        const r: u16 = @min(100, pos.rule50);
        return @divFloor(eval * (220 - r), 220);
        // const r: u16 = @min(100, pos.rule50);
        // return @divFloor(eval * (400 - r), 400);
    }

    fn do_move(self: *Searcher, pos: *const Position, comptime us: Color, ex: ExtMove) Position {
        var next_pos: Position = pos.*;
        next_pos.do_move(us, ex);
        self.repetition_table[next_pos.ply_from_root] = next_pos.key;
        return next_pos;
    }

    fn do_nullmove(self: *Searcher, pos: *const Position, comptime us: Color) Position {
        var next_pos: Position = pos.*;
        next_pos.do_nullmove(us);
        self.repetition_table[next_pos.ply_from_root] = next_pos.key;
        return next_pos;
    }

    fn is_draw_by_repetition_or_rule50(self: *Searcher, pos: *const Position) bool {
        if (pos.ply_from_root < 3 or pos.rule50 < 4) {
            return false;
        }
        if (pos.rule50 >= 100) {
            return true;
        }
        const end: u16 = @min(pos.rule50, pos.ply_from_root);
        if (end < 3) {
            return false;
        }

        var count: u8 = 0;
        var i: u16 = 4;
        var run: [*]u64 = &self.repetition_table;
        run += pos.ply_from_root;
        while (i <= end) : (i += 2) {
            run -= 2;
            if (run[0] == pos.key) {
                count += 1;
                if (count >= 1) {
                    return true;
                }
            }
        }
        return false;
    }

    /// Check for a hard time out or must stop.
    fn must_stop(self: *Searcher) bool {
        // We already stopped.
        if (self.stopped) {
            return true;
        }

        self.stats.search_calls += 1;

        // Don't check too often: the search is running with at least 1 million nodes per second.
        const interval: bool = self.stats.search_calls % 1024 == 0;

        if (!interval) {
            return false;
        }

        // Check if the engine received a stop or quit uci command.
        if (!self.engine.is_busy()) {
            self.stopped = true;
            return true;
        }

        switch (self.tm.termination) {
            .nodes => {
                if (self.stats.nodes >= self.tm.max_nodes) {
                    self.stopped = true;
                }
            },
            .movetime, .clock => {
                const t: u64 = self.tm.read();
                //if (self.tm.max_time_reached()) {
                if (self.tm.max_time_reached(t)) {
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

    /// What to clear this node. (ply is never cleared).
    pub const ClearMode = enum {
        /// Only reset pv. The rest of the fields have to be assigned inside search.
        clear_pv,
        /// Only clear the fields.
        clear_fields,
        /// Undo move.
        undo,
    };

    /// Local PV during search.
    pv: PV = .{},
    /// Set in search on better move found.
    best_move: Move = .empty,
    /// Current pv score.
    score: i32 = null_score,
    /// The search ply. Initialized once.
    ply: u16 = 0,
    /// The current move being searched in the tree.
    current_move: ExtMove = .empty,
    /// Skipped move during singular extensions.
    excluded_tt_move: Move = .empty,
    /// The number of double extensions done in singular extensions.
    double_extensions: u8 = 0,
    /// Static eval with applied correction (correction history and rule50).
    static_eval: i32 = null_score,
    /// Static or TT
    eval: i32 = null_score,
    /// Pointer to continuation history for - hopefully - faster access.
    continuation_entry: ?history.ContinuationEntry = null,

    const empty: Node = .{};

    fn clear(self: *Node, comptime mode: ClearMode) void {
        switch (mode) {
            .clear_pv => {
                self.pv.len = 0;
                self.score = null_score;
            },
            .clear_fields => {
                self.score = null_score;
                self.current_move = .empty;
                self.excluded_tt_move = .empty;
                self.double_extensions = 0;
                self.static_eval = null_score;
                self.eval = null_score;
                self.continuation_entry = null;
            },
            .undo => {
                self.current_move = .empty;
                self.continuation_entry = null;
            }
        }
    }

    /// Sets pv to bestmove + childnode.pv and score.
    fn update_pv(self: *Node, bestmove: Move, score: i32, childnode: *const Node) void {
        self.pv.len = 0;
        self.pv.append_assume_capacity(bestmove);
        self.pv.append_slice_assume_capacity(childnode.pv.slice());
        self.score = score;
    }
};

/// The available engine options.
pub const Options = struct {
    /// In megabytes.
    hash_size: u64 = default_hash_size,
    /// Chess960 support.
    is_960: bool = false,

    pub const default: Options = .{};

    pub const default_hash_size: u64 = 64;
    pub const min_hash_size: u64 = 16;
    pub const max_hash_size: u64 = 1024;

    pub fn set_hash_size(self: *Options, value: u64) void {
        self.hash_size = std.math.clamp(value, min_hash_size, max_hash_size);
    }

    pub fn set_chess_960(self: *Options, value: bool) void {
        self.is_960 = value;
    }
};

/// Retrieve default reduction from the lmr table. Access is safely restricted.
/// - The max reduction for noisy is 9.
/// - The max reduction for quiet is 13.
fn get_lmr(is_quiet: bool, depth: i32, moves_seen: u8) u8 {
    const q: u1 = @intFromBool(is_quiet);
    const d: u8 = @intCast(@min(depth, max_search_depth));
    return lmr_table[q][d][moves_seen];
}

/// Contains late move depth reductions. Indexing: [quiet][depth][moves_seen]
pub const LmrTable = [2][max_search_depth + 2][max_move_count]u8;

pub const lmr_table: LmrTable = compute_lmr_table();

fn compute_lmr(depth: usize, moves: usize, base: f32, divisor: f32) u8 {
    const d: f32 = @floatFromInt(depth);
    const m: f32 = @floatFromInt(moves);
    const ln_depth: f32 = @log(d);
    const ln_moves: f32 = @log(m);
    return @intFromFloat(base + ln_depth * ln_moves / divisor);
}

fn compute_lmr_table() LmrTable {
    @setEvalBranchQuota(128000);
    var result: LmrTable = std.mem.zeroes(LmrTable);
    for (1..max_search_depth + 2) |depth| {
        for (1..max_move_count) |move| {
            result[0][depth][move] = compute_lmr(depth, move, tuned.lmr_table_noisy_base, tuned.lmr_table_noisy_divider); // noisy
            result[1][depth][move] = compute_lmr(depth, move, tuned.lmr_table_quiet_base, tuned.lmr_table_quiet_divider); // quiet
        }
    }
    return result;
}

/// Not used (yet).
const Error = error {
    EngineIsStillRunning,
};
