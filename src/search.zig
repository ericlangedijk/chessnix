// zig fmt: off

//! Engine + Search.

// This chessnix 1.3 version contains borrowed stuff, adjusted to chessnix's needs.
// Most alphabeta engines share a lot of established ideas, but I want to mention these:
// Evaluation and some history math from an old Integral engine (v3) to get me going. Starting with version 2.0 other data will be used.
// Time management math from the Alexandria engine.

const std = @import("std");
const lib = @import("lib.zig");
const bitboards = @import("bitboards.zig");
const types = @import("types.zig");
const utils = @import("utils.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const uci = @import("uci.zig");
const hce = @import("hce.zig");
const hcetables = @import("hcetables.zig");
const tt = @import("tt.zig");
const history = @import("history.zig");
const movepick = @import("movepick.zig");
const time = @import("time.zig");

const assert = std.debug.assert;
const clamp = std.math.clamp;
const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;
const mul = funcs.mul;
const div = funcs.div;
const float = funcs.float;

const SmallValue = types.SmallValue;
const Value = types.Value;
const Color = types.Color;
const PieceType = types.PieceType;
const Piece = types.Piece;
const Square = types.Square;
const Move = types.Move;
const Position = position.Position;
const Entry = tt.Entry;
const Bound = tt.Bound;
const Evaluator = hce.Evaluator;
const History = history.History;
const MovePicker = movepick.MovePicker;
const TimeManager = time.TimeManager;

const max_u64: u64 = std.math.maxInt(u64);
const max_u32: u64 = std.math.maxInt(u32);
const max_game_length: u16 = types.max_game_length;
const max_move_count: u8 = types.max_move_count;
const max_search_depth: u8 = types.max_search_depth;
const max_threads: u16 = types.max_threads;

const no_score = types.no_score;
const infinity = types.infinity;
const mate = types.mate;
const mate_threshold = types.mate_threshold;
const stalemate = types.stalemate;
const invalid_score = types.invalid_score;

/// Principal Variation (+ 2 to be safe).
const PV = lib.BoundedArray(Move, max_search_depth + 2);

pub const Engine = struct {
    /// Shared tt for all threads.
    transpositiontable: tt.TranspositionTable,
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
    /// Set on ucinewgame.
    is_newgame: bool,

    /// Engine is created on the heap.
    /// The engine will be ready for a 'go' directly after (1) creation, (2) ucinewgame command and (3) position command.
    pub fn create(mute: bool) !*Engine {
        const tt_size: usize = tt.compute_tt_size(Options.default_hash_size);
        var engine: *Engine = try ctx.galloc.create(Engine);
        engine.transpositiontable = try .init(tt_size);
        engine.busy = std.atomic.Value(bool).init(false);
        engine.num_threads = 1;
        engine.controller_thread = null;
        engine.search_thread = null;
        engine.repetition_table = @splat(0);
        engine.pos = .empty;
        engine.pos.set_startpos();
        engine.repetition_table[0] = engine.pos.key;
        //engine.rootmoves = .init(); // TODO: INIT
        engine.pos.lazy_generate_all_moves(&engine.rootmoves);
        engine.options = Options.default;
        engine.tm = TimeManager.empty;
        engine.searcher = Searcher.init();
        engine.mute = mute;
        engine.is_newgame = true;
        if (!mute) {
            engine.print_hash_sizes();
        }
        return engine;
    }

    pub fn destroy(self: *Engine) void {
        self.transpositiontable.deinit();
        ctx.galloc.destroy(self);
    }

    fn print_hash_sizes(self: *const Engine) void {
        io.print("info string hash {} MB, entries {}\n", .{ self.options.hash_size, self.transpositiontable.hash.data.len });
    }

    /// After the options are determined. Bit of a mess...
    pub fn apply_hash_size(self: *Engine) !void {
        if (self.is_busy()) {
            return;
        }
        const tt_size: usize = tt.compute_tt_size(self.options.hash_size);
        try self.transpositiontable.resize(tt_size);
        if (!self.mute) {
            self.print_hash_sizes();
        }
    }

    pub fn ucinewgame(self: *Engine) !void {
        if (self.is_busy()) {
            return;
        }
        self.pos.set_startpos();
        self.repetition_table[0] = self.pos.key;
        self.transpositiontable.clear();
        self.pos.lazy_generate_all_moves(&self.rootmoves);
        // self.searcher.newgame(); TODO
        self.is_newgame = true;
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
        // TODO: prepare searcher here: (not new game)
    }

    /// If we have any moves, make them. We stop if we encounter an illegal move without crashing.
    fn parse_moves(self: *Engine, moves: []const u8) void {
        var tokenizer = std.mem.tokenizeScalar(u8, moves, ' ');
        var idx: usize = 1;
        while (tokenizer.next()) |m| {
            const move: Move = self.pos.parse_move(m) catch break;
            self.pos.lazy_do_move(move);
            self.repetition_table[idx] = self.pos.key;
            idx += 1;
        }
    }

    pub fn is_busy(self: *const Engine) bool {
        return self.busy.load(.monotonic);
    }

    /// Start threaded search.
    pub fn start(self: *Engine, go_params: *const uci.Go) !void {
        if (self.is_busy()) {
            return;
        }
        // Set our busy flag.
        self.busy.store(true, .monotonic);

        // Set and start time manager.
        self.tm.set(go_params, self.pos.stm);

        // To prevent messing up
        const new: bool = self.is_newgame;
        self.is_newgame = false;

        // Spawn thread.
        self.controller_thread = try std.Thread.spawn(.{}, run_controller_thread, .{ self, new });
    }

    /// Stop threaded search.
    pub fn stop(self: *Engine) !void {
        if (!self.is_busy()) return;
        // Force the searcher to stop. Only this boss engine may do this atomic store to the searchers.
        @atomicStore(bool, &self.searcher.stopped, true, .monotonic);
        self.busy.store(false, .monotonic);
    }

    fn run_controller_thread(self: *Engine, new: bool) !void {
        defer {
            self.controller_thread = null;
        }
        self.search_thread = try std.Thread.spawn(.{}, Searcher.start, .{ &self.searcher, self, new });
        if (self.search_thread) |th| {
            th.join();
            self.search_thread = null;
        }
        self.busy.store(false, .monotonic);
    }

    /// For position testing without output.
    pub fn give_best_move(self: *Engine, fen: []const u8, moves: []const u8, movetime: u32) !Move {
        if (self.is_busy()) return .empty;
        try self.ucinewgame();
        self.mute = true;
        try self.set_position(fen, moves);
        var go: uci.Go = .empty;
        go.movetime = movetime;
        try self.start(&go);
        // Run thread.
        while (self.is_busy()) {
        }
        return self.searcher.selected_move;
    }
};

pub const Stats = struct {
    /// The total amount of nodes.
    nodes: u64 = 0,
    /// The total amount of quiescent search nodes.
    qnodes: u64 = 0,
    /// Iteration reached.
    depth: u16 = 0,
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
    /// Backlink ref. Only valid after start_search.
    engine: *Engine,
    /// Copied from engine for direct access.
    transpositiontable: *tt.TranspositionTable,
    /// Copied from engine for direct access.
    tm: *TimeManager,
    /// The evaluator
    evaluator: hce.Evaluator,
    /// For now this is a blunt copy from the Engine.
    repetition_table: [max_game_length]u64,
    /// Our node stack. Index #0 is our main pv.
    nodes: Nodes,
    /// Nodes spent on move for time heuristics.
    nodes_spent: [64 * 64]u64,
    /// History heuristics for beta cutoffs.
    hist: History,
    /// Timeout. Any search value (0) returned - when stopped - should be discarded directly.
    stopped: bool,
    /// Statistics.
    stats: Stats,
    /// This is the final move we output.
    selected_move: Move,

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
            .selected_move = .empty,
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

    fn start(self: *Searcher, engine: *Engine, new: bool) void {
        // Shared stuff.
        self.engine = engine;
        self.transpositiontable = &engine.transpositiontable;
        self.tm = &engine.tm;

        // Copy stuff.
        const len: u16 = self.engine.pos.ply_from_root + 1;
        @memcpy(self.repetition_table[0..len], engine.repetition_table[0..len]);

        // Reset stuff.
        self.stopped = false;
        self.stats = .empty;
        self.selected_move = .empty;
        self.nodes_spent = @splat(0);

        // Clear history heuristics.
        if (new) {
            self.hist.clear();
        }

        // Make our local copy of the position.
        var pos: Position = engine.pos;

        // Have at least one move ready for crazy cases.
        self.selected_move = engine.rootmoves.moves[0];

        // If we only have 1 legal move just make it without pv output and return bestmove. No GUI pv viewing pleasure here.
        const return_immediate: bool = engine.rootmoves.count == 1 and self.tm.termination == .clock;

        if (!return_immediate) {
            switch (pos.stm.e) {
                .white => self.iterate(Color.WHITE, &pos),
                .black => self.iterate(Color.BLACK, &pos),
            }
        }

        if (!engine.mute) {
            // #experimental fake output for viewing pleasure
            // if (return_immediate) {
            //     const cp: Value = self.evaluator.evaluate(&pos);
            //     io.print_buffered(
            //         "info depth {} seldepth {} score cp {} nodes {} qnodes {}% time {} nps {} eff {}% pv\n",
            //         .{ 1, 1, cp, 1, 0, 1, 1, 0 }
            //     );
            // }
            io.print_buffered("bestmove ", .{} );
            self.selected_move.print_buffered(pos.is_960);
            io.print_buffered("\n", .{});
            io.flush();
        }
    }

    fn iterate(self: *Searcher, comptime us: Color, pos: *const Position) void {
        const rootnode: *Node = &self.nodes[0]; // TODO: assert the rootnode is initialized to zero stuff.
        const max_depth: u8 = if (self.tm.termination == .depth) self.tm.max_depth else max_search_depth;

        var previous_best_move: Move = .empty;
        var previous_score: Value = 0;
        var average_score: Value = no_score;
        var best_move_stability: u3 = 0;
        var eval_stability: u3 = 0;

        // Statistics stuff.
        self.stats.non_terminal_nodes = 0;
        self.stats.beta_cutoffs = 0;

        // Search with increasing depth.
        iterationloop: for (1..max_depth + 1) |d| {
            const depth: u8 = @intCast(d);
            self.stats.depth = depth;
            self.stats.seldepth = 0;

            const start_non_terminal: u64 = self.stats.non_terminal_nodes;
            const start_beta_cutoffs: u64 = self.stats.beta_cutoffs;

            const avg: Value = if (average_score == no_score) 0 else average_score;
            const score: Value = self.search_aspiration_window(us, pos, avg, depth);

            // Discard if timeout.
            if (self.stopped) {
                return;
            }

            self.selected_move = rootnode.pv.buffer[0]; // TODO: can the move be empty?

            // TM: Keep track of the average score.
            average_score = if (average_score == no_score) score else @divTrunc(average_score + score, 2);

            // TM: Keep track of how many times in a row the best move stays the same.
            if (self.selected_move == previous_best_move) {
                best_move_stability = @min(4, best_move_stability + 1);
            }
            else {
                best_move_stability = 0;
            }

            // TM: Keep track of the evaluation stability.
            if (score > average_score - 10 and score < average_score + 10) {
                eval_stability = @min(4, eval_stability + 1);
            }
            else {
                eval_stability = 0;
            }

            // TM: save previous stuff.
            previous_best_move = self.selected_move;
            previous_score = score;

            // And print the pv, including some stats.
            if (!self.engine.mute) {
                const non_terminal_used: u64 = self.stats.non_terminal_nodes - start_non_terminal;
                const beta_cutoffs_used: u64 = self.stats.beta_cutoffs - start_beta_cutoffs;
                const qnodes: usize = funcs.percent(self.stats.nodes, self.stats.qnodes);
                const search_efficiency = funcs.percent(non_terminal_used, beta_cutoffs_used);
                const elapsed_nanos = self.tm.timer.read();
                const ms = elapsed_nanos / 1_000_000;
                const nps: u64 = funcs.nps(self.stats.nodes, elapsed_nanos);

                io.print_buffered(
                    "info depth {} seldepth {} score cp {} nodes {} qnodes {}% time {} nps {} eff {}% pv",
                    .{ self.stats.depth, self.stats.seldepth, rootnode.score, self.stats.nodes, qnodes, ms, nps, search_efficiency }
                );

                for (rootnode.pv.slice()) |move| {
                    io.print_buffered(" ", .{});
                    move.print_buffered(pos.is_960);
                }
                io.print_buffered("\n", .{});
                io.flush();
            }

            // Now update the optimal time to spend and check if we can stop.
            if (depth > 7 and self.tm.termination == .clock) {
                const nodes_spent_on_move: u64 = self.nodes_spent[self.selected_move.from_to()];
                self.tm.update_optimal_stoptime(nodes_spent_on_move, self.stats.nodes, best_move_stability, eval_stability);
                if (self.tm.optimal_time_reached()) {
                    self.stopped = true;
                    break :iterationloop;
                }
            }
        } // (iteration_loop)
    }

    fn search_aspiration_window(self: *Searcher, comptime us: Color, pos: *const Position, previous_score: Value, depth: i32) Value {
        var delta: Value = 9; //9, 12; #testing
        var alpha: Value = -infinity;
        var beta: Value = infinity;

        const rootnode: *Node = &self.nodes[0];

        if (depth >= 4) { // #testing 4 (original 3)
            alpha = @max(-infinity, previous_score - delta);
            beta = @min(infinity, previous_score + delta);
        }
        var score: Value = 0;

        // NOTE: Chessnix is maybe not strong or precise enough to reduce the searchdepth on a fail high here.
        const searchdepth: i32 = depth;

        while (true) {
            rootnode.clear(.clear_fields);

            // Search aspiration window.
            score = self.search(us, .pv, .root, false, pos, rootnode, searchdepth, alpha, beta);

            // Discard result if timeout.
            if (self.stopped) {
                return 0;
            }

            // Alpha: fail low
            if (score <= alpha) {
                beta = @divTrunc(alpha + beta, 2);
                alpha = @max(-infinity, score - delta);
                //searchdepth = depth;
            }
            // Beta: fail high
            else if (score >= beta) {
                beta = @min(infinity, score + delta);
                // if (alpha < 2000) {
                //     searchdepth = @max(1, depth - 1);
                // }
            }
            // Success.
            else {
                break;
            }
            // delta = funcs.mul(delta, 1.44);
            delta = funcs.mul(delta, 1.55);
        }
        return score;
    }

    fn search(self: *Searcher, comptime us: Color, comptime mode: SearchMode, comptime rootmode: RootMode, comptime cutnode: bool, pos: *const Position, node: *Node, input_depth: i32, input_alpha: Value, input_beta: Value) Value {
        // Comptimes.
        const them: Color = comptime us.opp();
        const is_pvs: bool = comptime mode == .pv;
        const is_root: bool = comptime rootmode == .root;

        assert(input_depth >= 0);
        assert(!is_root or is_pvs);
        assert(input_beta > input_alpha);

        // if (comptime lib.is_guarded) {
        //     lib.guard(input_depth >= 0, "search inputdepth < 0", .{}); // TODO: use assert?
        //     lib.guard(input_beta > input_alpha, "search wrong alpha beta", .{});
        //     //lib.guard(is_root or pos.ply > 0, "search wrong is_root vs ply", .{});
        //     lib.guard(!is_root or input_depth > 0, "search wrong is_root vs input_depth", .{});
        //     //lib.guard(!is_root or is_root == is_pvs, "search wrong is_root vs is_pvs", .{});
        //     lib.guard(is_pvs or input_beta - input_alpha == 1, "search wrong is_pvs vs alphabeta", .{});
        // }

        const childnode: *Node = &self.nodes[node.ply + 1];
        node.clear(.clear_pv);
        childnode.clear(.clear_fields);

        // Too deep.
        if (node.ply >= max_search_depth) {
            return self.evaluate(pos);
        }

        // Update max reached depth.
        self.stats.seldepth = @max(self.stats.seldepth, node.ply);

        var depth: i32 = input_depth;
        var alpha: Value = input_alpha;
        var beta: Value = input_beta;

        // Adjust distance to mate window. We do this here, so we don't have to do this again in quiscence search.
        if (!is_root) {
            alpha = @max(alpha, -mate + node.ply);
            beta = @min(beta, mate - node.ply);
            // A beta cutoff may occur after adjusting the window.
            if (alpha >= beta) {
                return alpha;
            }
        }

        // When reaching limit dive into quiescent search.
        if (input_depth == 0) {
            return self.quiescence_search(us, mode, pos, node, alpha, beta);
        }

        // Locals.
        const parentnode: ?*const Node = if (!is_root) &self.nodes[node.ply - 1] else null; // TODO: only used now for double extensions inheritance.
        const is_singular_extension: bool = !node.excluded_tt_move.is_empty();
        const is_check: bool = pos.checkers != 0;

        var extension: i32 = 0;
        var reduction: i32 = 0;
        var raw_static_eval: Value = no_score;

        // Check timeout at the start only.
        if (self.must_stop()) {
            return 0;
        }

        if (!is_root) {
            // Forced draw.
            if (pos.is_draw_by_insufficient_material() or self.is_draw_by_repetition_or_rule50(pos)) {
                return 0;
            }
        }

        // TT Probe.
        var tt_pv: bool = is_pvs;
        const tt_entry: ?*const tt.Entry = self.transpositiontable.probe(pos.key);
        const tt_hit: bool = !is_singular_extension and tt_entry != null;
        var tt_move: Move = if (tt_hit) tt_entry.?.move else .empty;

        if (tt_hit) {
            tt_pv |= tt_entry.?.was_pv;
            const tt_is_usable_score: bool = tt_entry.?.is_score_usable_for_depth(alpha, beta, depth);
            if (!is_pvs and tt_is_usable_score) {
                return tt.get_adjusted_score_for_tt_probe(tt_entry.?.score, node.ply);
            }
        }

        // Static Evaluation.
        if (is_check) {
            node.static_eval = no_score;
            node.eval = no_score;
        }
        else if (!is_singular_extension) {
            // Get the static eval from the TT if it is there, otherwise compute and store it.
            if (tt_hit and tt_entry.?.is_raw_static_eval_usable()) {
                raw_static_eval = tt_entry.?.raw_static_eval;
            }
            else {
                raw_static_eval = self.evaluate(pos);
                if (pos.rule50 < 60) // #testing
                self.transpositiontable.store_static_eval(pos.key, raw_static_eval);
            }
            // Also apply a little drawcounter adjustment.
            node.static_eval = self.apply_correction(pos, raw_static_eval, us);

            // Get the score from the TT if it is there, otherwise use the static eval.
            if (tt_hit and tt_entry.?.is_score_usable(node.static_eval, node.static_eval)) {
                node.eval = tt.get_adjusted_score_for_tt_probe(tt_entry.?.score, node.ply);
            }
            else {
                node.eval = node.static_eval;
            }
        }

        // Check improvement compared to a previous node with the same side to move.
        var improvement: u1 = 0;

        if (!is_check) {
            // Get a previous node to compare with, if any.
            // const prevnode: ?*const Node =
            //     if (node.ply >= 4 and self.nodes[node.ply - 4].static_eval != no_score) &self.nodes[node.ply - 4]
            //     else if (node.ply >= 2 and self.nodes[node.ply - 2].static_eval != no_score) &self.nodes[node.ply - 2]
            //     else null;

            const prevnode: ?*const Node = // #testing
                if (node.ply >= 2 and self.nodes[node.ply - 2].static_eval != no_score) &self.nodes[node.ply - 2]
                else if (node.ply >= 4 and self.nodes[node.ply - 4].static_eval != no_score) &self.nodes[node.ply - 4]
                else null;

            if (prevnode != null) {
                const threshold: Value = 0;
                improvement = if (node.static_eval > prevnode.?.static_eval + threshold) 1 else 0;
            }
        }

        // Pruning before processing any moves.
        if (!is_root and !is_pvs and !is_check and !is_singular_extension) {

            // Reversed Static Futility Pruning.
            if (depth <= 4 and node.eval < mate_threshold) {
                const m: Value = if (improvement > 0) 40 else 74;
                const futility_margin: Value = depth * m;
                if (node.eval - futility_margin >= beta) {
                    return node.eval;
                }
            }

            // Razoring.
            if (depth <= 4 and node.static_eval + 450 * depth < alpha) {
                const r_score = self.quiescence_search(us, mode, pos, node, alpha, alpha + 1);
                if (self.stopped) {
                    return 0;
                }
                if (r_score <= alpha) return r_score;
            }

            // Null Move Pruning. Are we still good if we let them play another move?
            //if (!pos.nullmove_state and depth >= 2 and node.eval >= beta and node.static_eval >= (beta + 170 - 24 * depth) and pos.phase > 0 and pos.pieces_except_pawns_and_kings(us) != 0) {
            if (!pos.nullmove_state and depth >= 2 and node.eval >= beta and node.static_eval >= (beta + 170 - 24 * depth) and pos.phase > 0 and pos.pieces_except_pawns_and_kings(us) != 0) {// #testing
            //if (!pos.nullmove_state and node.eval >= beta and node.static_eval >= (beta + 170 - 24 * depth) and pos.phase > 0 and pos.pieces_except_pawns_and_kings(us) != 0) {
                const eval_reduction: i32 = @min(2, div(node.eval - beta, 202));
                const r: i32 = clamp(div(depth, 4) + 3 + eval_reduction, 0, depth);
                const next_pos: Position = self.do_nullmove(pos, us);
                node.current_move = .empty;
                node.continuation_entry = null;
                const n_score: Value = -self.search(them, .not_pv, .not_root, !cutnode, &next_pos, childnode, depth - r, -beta, -beta + 1);
                if (self.stopped) {
                    return 0;
                }
                if (n_score >= beta) {
                    return if (n_score > mate_threshold) beta else n_score;
                }
            }
        }

        // Internal Iterative Reduction: If we have no tt move move ordering is not optimal.
        if ((is_pvs or cutnode) and depth >= 8 and tt_move.is_empty() and !is_singular_extension) {
            depth -= 1;
        }

        // For move ordering statistics.
        self.stats.non_terminal_nodes += 1;
        node.double_extensions = if (!is_root) parentnode.?.double_extensions else 0;

        // Some locals for our move loop.
        var quiet_list: ExtMoveList(224) = .init();
        var capture_list: ExtMoveList(80) = .init();
        var movepicker: MovePicker(.search, us) = .init(pos, self, node, tt_move);
        var best_move: Move = .empty;
        var best_score: Value = -infinity;
        var moves_seen: u8 = 0;
        //var quiets_seen: u8 = 0;
        var tt_store_bound: tt.Bound = .alpha;


        // Move loop.
        moveloop: while (movepicker.next()) |ex| {

            //node.current_move = .empty;
            //assert(ex.is_ok(pos));

            // Skip this move if we are inside singular extensions.
            if (ex.move == node.excluded_tt_move) {
                continue :moveloop;
            }

            //const is_quiet: bool = ex.move.is_quiet();
            //const is_capture: bool = ex.move.is_capture();

            // if (is_capture and is_root) {
            //     io.debugprint("  {t} {t} {} bad {}\n", .{ ex.move.from.e, ex.move.to.e, self.hist.capture.get_score(ex), ex.is_bad_capture});
            // }

            const quiet_history_score: Value = if (ex.is_quiet) self.hist.get_quiet_score(ex, node.ply, &self.nodes) else 0; // #testing

            // Prunings before we execute the move.
            if (!is_root and !is_check and !ex.is_tt_move and best_score > -mate_threshold) {

                // Late Move Pruning.
                if (ex.is_quiet) {
                    const lmp_threshold: i32 = @divTrunc(5 + depth * depth, @as(i32, 3) - improvement);
                    if (moves_seen >= lmp_threshold) {
                        movepicker.skip_quiets();
                        continue :moveloop;
                    }
                }

                // Futility Pruning.
                const futility_margin = 196 + 96 * depth;
                if (ex.is_quiet and depth <= 8 and node.eval + futility_margin < alpha) {
                    movepicker.skip_quiets();
                    continue :moveloop;
                }

                // SEE pruning. #testing (more combination testing needed) is_capture seems best for now...
                const see_threshold: Value = if (ex.is_quiet) -64 * depth else -119 * depth;
                //if (@intFromEnum(movepicker.stage) > @intFromEnum(movepick.Stage.counter) and depth <= 8 and !ex.is_countermove and !ex.move.is_promotion() and moves_seen >= 1 and !hce.see(pos, ex.move, see_threshold)) {
                if (ex.is_capture and depth <= 8 and !ex.move.is_promotion() and moves_seen >= 3 and !hce.see(pos, ex.move, see_threshold)) { // TRICKY
                // if (depth <= 8 and !ex.move.is_promotion() and moves_seen >= 3 and !hce.see(pos, ex.move, see_threshold)) { original
                    continue :moveloop;
                }

                // History Pruning. #experimental
                if (ex.is_quiet and depth <= 5) {
                    if (quiet_history_score < -480 - (1695 * depth)) {
                        //movepicker.skip_quiets(); // #testing. did i compile with this on or off?
                        continue :moveloop;
                    }
                }
            }

            // Determine extension + reduction.
            extension = 0;
            reduction = 0;

            // Singular Extensions.
            if (!is_root and tt_hit and depth >= 8 and ex.is_tt_move) {
                const entry: *const tt.Entry = tt_entry.?;
                const accurate_tt_score: bool = entry.bound != .none and entry.bound != .alpha and entry.depth + 4 >= depth and @abs(entry.score) < mate_threshold;
                if (accurate_tt_score) {
                    node.excluded_tt_move = ex.move;
                    const reduced_depth: i32 = div(depth - 1, 2);
                    const s_beta = entry.score - (depth * 2);
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
                    else if (s_beta >= beta and @abs(s_score) < mate_threshold) {
                        return s_beta;
                    }
                    // Negative Extensions: Search less since the tt_move was not singular and it might cause a beta cutoff again.
                    else if (entry.score >= beta) {
                        extension = -1;
                    }
                }
            }

            if (is_check) {
                extension += 1;
            }

            // Do move
            const next_pos: Position = self.do_move(pos, us, ex.move);
            const gives_check: bool = next_pos.checkers != 0;
            node.current_move = ex;
            node.continuation_entry = self.hist.continuation.get_node_entry(ex);
            self.stats.nodes += 1;

            // Keep track of nodes spent on this move for time management heuristic. Only in tournament situation.
            const nodes_before_search: u64 = if (is_root and self.tm.termination == .clock) self.stats.nodes else 0;

            var need_full_search: bool = false;
            var score: Value = -infinity;
            const new_depth: i32 = depth + extension - 1;


            // Late Move Reduction. Try a reduced search on late moves. Very sensitive.
            const lmr_move_threshold: i32 = if (is_root) 3 else 1;
            if (depth >= 3 and moves_seen >= lmr_move_threshold) {
                const depth_idx: usize = @intCast(depth);
                const quiet_idx: usize = @intFromBool(ex.is_quiet);
                reduction = lmr_table[quiet_idx][depth_idx][moves_seen];
                // reduction = lmr_table[quiet_idx][@min(depth_idx, 64)][@min(moves_seen, 64)]; // #testing

                // Reduce more if cutnode.
                if (cutnode) {
                    reduction += 1;
                }

                // if (improvement == 0) { // #testing improvement
                //      reduction += 1;
                // }

                // Reduce more if not pvs.
                if (!is_pvs) {
                    reduction += 1;
                }

                // History reduction. Calculate more or less reduction from history score. TODO: use complete history score? if so, then use a different divider!
                if (ex.is_quiet) {
                    // const hist_score: Value = self.hist.quiet.get_score(ex);
                    // reduction -= @divTrunc(hist_score, 6000);
                    reduction -= @divTrunc(quiet_history_score, 11000); // #testing 11000
                }

                // And reduce less in these cases.
                //if (is_check or gives_check or ex.is_countermove) {
                if (is_check or gives_check) {
                    reduction -= 1;
                }

                // Reduction must not be below 0.
                reduction = std.math.clamp(reduction, 0, new_depth - 1);

                // Search nullwindow here.
                score = -self.search(them, .not_pv, .not_root, true, &next_pos, childnode, new_depth - reduction, -alpha - 1, -alpha);
                if (self.stopped) {
                    return 0;
                }
                need_full_search = score > alpha and reduction != 0;
            }
            else {
                need_full_search = !is_pvs or moves_seen >= 1;
            }

            // The move (1) scored well from the reduced search above or (2) is probably not a pv move.
            // In these cases We first search it with a null window.
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

            // We only count searched moves (not the pruned ones).
            moves_seen += 1;

            // Keep track of how many nodes we spent on this root move. Only in tournament situation.
            if (is_root and self.tm.termination == .clock) {
                self.nodes_spent[ex.move.from_to()] += self.stats.nodes - nodes_before_search;
            }

            // Better move found.
            if (score > best_score) {
                best_score = score;
                // Alpha.
                if (score > alpha) {
                    best_move = ex.move;
                    node.update_pv(best_move, best_score, childnode);
                    // Beta.
                    if (score >= beta) {
                        self.stats.beta_cutoffs += 1;
                        tt_store_bound = .beta;
                        self.hist.record_beta_cutoff(depth, node.ply, ex, &self.nodes, quiet_list.slice());
                        break :moveloop;
                    }
                    alpha = score; // #testing (moved down here)
                    tt_store_bound = .exact;
                }
                // else if (depth > 4 and depth < 10 and beta < mate_threshold and alpha > -mate_threshold) { // #testing
                //     depth -= 1;
                // }
            }

            // Keep track of the moves that did not raise alpha
            if (ex.move != best_move) {
                if (ex.is_quiet) {
                    quiet_list.add(ex);
                }
                else if (ex.is_capture) {
                    capture_list.add(ex); // TODO: test include all nosies? do we have here the weakness of chessnix underestimating passed pawns / promotions ??
                }
            }

        } // (moveloop)

        if (movepicker.move_count == 0) {
            return if (is_check) -mate + node.ply else 0;
        }

        if (!best_move.is_empty()) {
            self.hist.capture.punish(depth, capture_list.slice());
        }

        if (!is_singular_extension) {
            if (pos.rule50 < 60) // #testing
            self.transpositiontable.store(tt_store_bound, pos.key, depth, node.ply, best_move, best_score, tt_pv, raw_static_eval);

            // #discarded correction history for now.
            // if (!is_check and (best_move.is_empty() or best_move.is_quietnoisy()) and !(tt_store_bound == .alpha and best_score <= raw_static_eval) and !(tt_store_bound == .beta and best_score >= raw_static_eval)) {
            //     self.hist.correction.update(us, pos, raw_static_eval, best_score, depth);
            // }
        }

        return best_score;
    }

    fn quiescence_search(self: *Searcher, comptime us: Color, comptime mode: SearchMode, pos: *const Position, node: *Node, input_alpha: Value, input_beta: Value, ) Value {
        // Comptimes.
        const them = comptime us.opp();
        const is_pvs: bool = comptime mode == .pv;

        if (comptime lib.is_paranoid) {
            assert(pos.stm.e == us.e);
        }

        const childnode: *Node = &self.nodes[node.ply + 1];
        node.clear(.clear_pv);
        childnode.clear(.clear_fields);

        // Too deep.
        if (node.ply >= max_search_depth) {
            return self.evaluate(pos);
        }

        // Update stats.
        self.stats.seldepth = @max(self.stats.seldepth, node.ply);

        if (pos.is_draw_by_insufficient_material() or self.is_draw_by_repetition_or_rule50(pos)) {
            return 0;
        }

        // Check timeout at the start only. At the callsites just read the stopped value.
        if (self.must_stop()) {
            return 0;
        }

        const is_check : bool = pos.checkers > 0;
        const tt_depth: Value = @intFromBool(is_check);
        var alpha = input_alpha;
        const beta: Value = input_beta;
        var score: Value = 0;
        var tt_pv = is_pvs;
        var best_score = -infinity;
        var raw_static_eval: Value = no_score;
        var tt_store_bound: tt.Bound = .alpha;

        const tt_entry: ?*const tt.Entry = self.transpositiontable.probe(pos.key);
        const tt_hit: bool = tt_entry != null;
        const tt_move: Move = if (tt_hit) tt_entry.?.move else .empty;

        if (tt_hit) {
            tt_pv |= tt_entry.?.was_pv;
            const tt_is_usable_score: bool = tt_entry.?.is_score_usable_for_depth(alpha, beta, tt_depth);
            if (!is_pvs and tt_is_usable_score) {
                return tt.get_adjusted_score_for_tt_probe(tt_entry.?.score, node.ply);
            }
        }

        // Static Evaluation.
        if (!is_check) {
            // Get the static eval from TT, otherwise compute it.
            if (tt_hit and tt_entry.?.is_raw_static_eval_usable()) {
                raw_static_eval = tt_entry.?.raw_static_eval;
            }
            else {
                raw_static_eval = self.evaluate(pos);
                self.transpositiontable.store_static_eval(pos.key, raw_static_eval);
            }

            // Apply history correction.
            var eval: Value = self.apply_correction(pos, raw_static_eval, us);
            if (tt_hit and tt_entry.?.is_score_usable(score, score)) {
                eval = tt.get_adjusted_score_for_tt_probe(tt_entry.?.score, node.ply);
            }
            best_score = eval;
            // Fail high.
            if (best_score >= beta) {
                return best_score;
            }
            // Adjust alpha.
            if (best_score > alpha) {
                alpha = best_score;
            }
        }

        var movepicker: MovePicker(.quiescence, us) = .init(pos, self, node, tt_move);
        var capture_list: ExtMoveList(80) = .init();
        var best_move: Move = .empty;
        var moves_seen: u8 = 0;
        const futility_score: Value = best_score + 101;

        // Move loop.
        moveloop: while (movepicker.next()) |ex| {
            if (comptime lib.is_paranoid) {
                assert(ex.is_ok(pos));
            }

            const is_capture: bool = ex.move.is_capture();

            // Skip bad noisies. TODO: is this correct?
            //if (moves_seen > 0 and @intFromEnum(movepicker.stage) > @intFromEnum(movepick.Stage.noisy)) {
            if (moves_seen > 0 and ex.is_bad_capture) { // #testing
                break :moveloop;
            }

            // Quiescence Futility Pruning. Prune capture moves that don't win material if the static eval is behind alpha by some margin
            if (!is_check and ex.move.is_capture() and futility_score <= alpha and !hce.see(pos, ex.move, 1)) {
                best_score = @max(best_score, futility_score);
                continue :moveloop;
            }

            // Do move
            const next_pos: Position = self.do_move(pos, us, ex.move);
            self.stats.nodes += 1;
            self.stats.qnodes += 1;

            score = -self.quiescence_search(them, mode, &next_pos, childnode, -beta, -alpha);
            if (self.stopped) {
                return 0;
            }

            moves_seen += 1;

            // Better move.
            if (score > best_score) {
                best_score = score;
                // Alpha.
                if (score > alpha) {
                    // alpha = score;
                    best_move = ex.move;
                    node.update_pv(ex.move, best_score, childnode);
                    // Beta.
                    if (score >= beta) { // #testing changed 'alpha' to score
                        tt_store_bound = .beta;
                        // Slight boost for this capture.
                        if (is_capture) {
                            self.hist.capture.update(1, ex);
                        }
                        break :moveloop;
                    }
                    alpha = score;
                }
            }

            if (ex.move != best_move and is_capture) {
                capture_list.add(ex);
            }
        }

        if (movepicker.move_count == 0 and is_check) {
            return -mate + node.ply;
        }

        // Slight punishment for captures that did not raise alpha.
        if (!best_move.is_empty()) {
            self.hist.capture.punish(1, capture_list.slice());
        }

        // TT store. We do not store a best move because the quiscence search is incomplete by definition.
        if (pos.rule50 < 60) // #testing
        self.transpositiontable.store(tt_store_bound, pos.key, tt_depth, node.ply, Move.empty, best_score, tt_pv, raw_static_eval);
        return best_score;
    }

    /// Assumes not in check.
    fn evaluate(self: *Searcher, pos: *const Position) Value {
        if (comptime lib.is_paranoid) {
            assert(pos.checkers == 0);
        }
        return self.evaluator.evaluate(pos);
    }

    /// Apply the history correction to a raw static evaluation.
    fn apply_correction(self: *Searcher, pos: *const Position, raw_static_eval: Value, comptime us: Color) Value {
        _ = self;
        _ = us;
        var e: Value = raw_static_eval;// + self.hist.correction.get_correction(us, pos); // #testing
        // e = @divTrunc(e * (220 - pos.rule50), 220);
        e = @divTrunc(e * (400 - pos.rule50), 400); // #testing. TODO: don't pass from positive below zero or the other way around (?).
        return std.math.clamp(e, -mate_threshold, mate_threshold);
    }

    fn do_move(self: *Searcher, pos: *const Position, comptime us: Color, m: Move) Position {
        var next_pos: Position = pos.*;
        next_pos.do_move(us, m);
        self.repetition_table[next_pos.ply_from_root] = next_pos.key;
        return next_pos;
    }

    fn do_nullmove(self: *Searcher, pos: *const Position, comptime us: Color) Position {
        var next_pos: Position = pos.*;
        next_pos.do_nullmove(us);
        self.repetition_table[next_pos.ply_from_root] = next_pos.key;
        return next_pos;
    }

    /// Not used.
    // fn adjust_for_drawcounter(score: Value, drawcounter: i32) Value {
    //     if (drawcounter < 7 or score == 0 or score <= -mate_threshold or score >= mate_threshold) {
    //         return score;
    //     }
    //     // Max penalty around 25 centipawns.
    //     const penalty: i32 = div(drawcounter * drawcounter, 400);

    //     // Reduce evaluation.
    //     if (score > 0) {
    //         return @max(0, score - penalty);
    //     }
    //     else {
    //         return @min(0, score + penalty);
    //     }
    // }

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
                if (count >= 1) { // #testing 2????
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

        // Don't check too often: the search is running with at least 1 million nodes per second.
        const node_interval: bool = self.stats.nodes % 1024 == 0;

        if (!node_interval) {
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
                if (self.tm.max_time_reached()) {
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
        clear_fields
    };

    /// Local PV during search.
    pv: PV = .{},
    /// Current pv score.
    score: Value = no_score,
    /// The search ply. Initialized once.
    ply: u16 = 0,
    /// The current move being searched in the tree.
    current_move: ExtMove = .empty,
    /// Skipped move during singular extensions.
    excluded_tt_move: Move = .empty,
    /// The number of double extensions done in singular extensions.
    double_extensions: u8 = 0,
    /// Static eval with applied correction (correction history and rule50).
    static_eval: Value = no_score,
    /// Static or TT
    eval: Value = no_score,
    /// Pointer to continuation history for - hopefully - faster access.
    continuation_entry: ?*[12][64]SmallValue = null,

    // #discarded for now: this is just a pointer festival but it could contribute to speed.
    // Pointer into the continuation history. Updated after making a move.
    // continuation_history_entry: *[12][64]SmallValue,

    /// Note the undefined pointer which has to be initialized
    const empty: Node = .{
//        .continuation_history_entry = undefined,
    };

    fn clear(self: *Node, comptime mode: ClearMode) void {
        switch (mode) {
            // .clear_all => {
            //     self.pv.len = 0;
            //     self.score = no_score;
            //     continue :sw.clear_fields;
            // },
            .clear_pv => {
                self.pv.len = 0;
                self.score = no_score;
            },
            .clear_fields => {
                self.score = no_score;
                self.current_move = .empty;
                self.excluded_tt_move = .empty;
                self.double_extensions = 0;
                self.static_eval = no_score;
                self.eval = no_score;
                self.continuation_entry = null;
            },
        }
    }

    /// Only reset pv. The rest of the fields have to be assigned inside search.
    // fn clear_pv(self: *Node) void {
    //     self.pv.len = 0;
    //     self.score = no_score;
    // }

    // fn clear_fields(self: *Node) void {
    //     self.score = no_score;
    //     self.current_move = .empty;
    //     self.excluded_tt_move = .empty;
    //     self.double_extensions = 0;
    //     self.static_eval = no_score;
    //     self.eval = no_score;
    // }

    /// Sets pv to bestmove + childnode.pv and score.
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
};

/// 64 bits Move with score and info.
pub const ExtMove = packed struct {
    /// The generated move. -> 16 bits
    move: Move = .empty,
    /// Set by movepicker.
    score: i32 = types.invalid_movescore,
    /// Set on init.
    piece: Piece = Piece.NO_PIECE,
    /// Set on init.
    captured: Piece = Piece.NO_PIECE,
    // Set on init.
    is_quiet: bool = false,
    // Set on init.
    is_capture: bool = false,
    // Set on init.
    is_promotion: bool = false,
    /// Set by movepicker.
    is_tt_move: bool = false,
    /// Set by movepicker.
    is_bad_capture: bool = false,

    pub const empty: ExtMove = .{};

    /// Create ExtMove and copy some flags for convenience.
    pub fn init(move: Move) ExtMove {
        return .{ .move = move, .is_quiet = move.is_quiet(), .is_capture = move.is_capture(), .is_promotion = move.is_promotion() };
    }

    /// Debug only. TODO: use guard.
    pub fn is_ok(self: ExtMove, pos: *const Position) bool {
        lib.not_in_release();
        if (self.move.is_empty()) {
            io.debugprint("empty move {any}\n", .{ self });
            return false;
        }
        if (self.piece.is_empty()) {
            io.debugprint("empty piece {any}\n", .{ self });
            return false;
        }
        if (self.move.is_capture() and self.captured.is_empty()) {
            io.debugprint("empty capture {any}\n", .{ self });
            return false;
        }
        if (self.move.is_capture() and !self.move.is_ep()) {
            if (self.captured.e != pos.board[self.move.to.u].e) {
                io.debugprint("wrong capture {any}\n", .{ self });
                return false;
            }
        }
        if (self.move.is_capture() != self.is_capture or self.move.is_promotion() != self.is_promotion or self.move.is_quiet() != self.is_quiet) {
            io.debugprint("wrong flagcapture {any}\n", .{ self });
            return false;
        }
        // if (self.score < types.invalid_movescore + 10_000_000) {
        //     io.debugprint("no score {any}\n", .{ self });
        //     return false;
        // }
        return true;
    }
};

/// Simple storage for search.
pub fn ExtMoveList(max: u8) type {
    return struct {
        const Self = @This();

        extmoves: [max]ExtMove,
        count: u8,

        pub fn init() Self {
            return .{ .extmoves = undefined, .count = 0 };
        }

        pub fn add(self: *Self, ex: ExtMove) void {
            self.extmoves[self.count] = ex;
            self.count += 1;
        }

        pub fn slice(self: *Self) []ExtMove {
            return self.extmoves[0..self.count];
        }
    };
}

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

/// Not used (yet).
const Error = error {
    EngineIsStillRunning,
};

/// LMR. Contains depth reductions. Indexing: [quiet][depth][moves_seen]
pub const lmr_table: [2][max_search_depth][max_move_count]u8 = compute_lmr_table();
// TODO: make this smaller. reductions after 200 moves or depth 100 are insane i think.

fn compute_lmr(depth: usize, moves: usize, base: f32, divisor: f32) u8 {
    const d: f32 = @floatFromInt(depth);
    const m: f32 = @floatFromInt(moves);
    const ln_depth: f32 = @log(d);
    const ln_moves: f32 = @log(m);
    return @intFromFloat(base + ln_depth * ln_moves / divisor);
}

fn compute_lmr_table() [2][max_search_depth][max_move_count]u8 {
    @setEvalBranchQuota(128000);
    var result: [2][max_search_depth][max_move_count]u8 = std.mem.zeroes([2][max_search_depth][max_move_count]u8);
    for (1..max_search_depth) |depth| {
        for (1..max_move_count) |move| {
            result[0][depth][move] = compute_lmr(depth, move, -0.24, 2.60); // noisy
            //result[1][depth][move] = compute_lmr(depth, move, 0.80, 2.04); // quiet (original)
            result[1][depth][move] = compute_lmr(depth, move, 0.75, 2.04); // quiet (i think most stable and ok)
            //result[1][depth][move] = compute_lmr(depth, move, 1.00, 2.00); // quiet (WTF is this MUCH better?????)
        }
    }
    return result;
}
