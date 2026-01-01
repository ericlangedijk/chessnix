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
const TimeMgr = time.TimeMgr;

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
const invalid_score = types.invalid_score;

/// Principal Variation (+ 2 to be safe).
const PV = lib.BoundedArray(Move, max_search_depth + 2);

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
    /// The global board list for our position.
    board_history: [max_game_length]Position,
    /// Array with hash keys.
    repetition_table: [max_game_length]u64,
    /// Our source position.
    pos: Position,
    /// UCI Engine options like hashsize etc.
    options: Options,
    /// Settings like timeout etc.
    time_mgr: TimeMgr,
    /// The actual worker.
    searcher: Searcher,
    /// No output if true.
    mute: bool,
    /// Set on ucinewgame.
    is_newgame: bool,

    /// Engine is created on the heap.
    pub fn create(mute: bool) !*Engine {
        const tt_size: usize = tt.compute_tt_size(Options.default_hash_size);
        var engine: *Engine = try ctx.galloc.create(Engine);
        engine.transpositiontable = try .init(tt_size);
        engine.busy = std.atomic.Value(bool).init(false);
        engine.num_threads = 1;
        engine.controller_thread = null;
        engine.search_thread = null;
        engine.board_history = @splat(.empty);
        engine.repetition_table = @splat(0);
        engine.pos = .empty;
        engine.pos.set_startpos();
        engine.repetition_table[0] = engine.pos.key;
        engine.options = Options.default;
        engine.time_mgr = TimeMgr.default;
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

    /// After the options are determined.
    pub fn apply_hash_size(self: *Engine) !void {
        if (self.is_busy()) return;
        const tt_size: usize = tt.compute_tt_size(self.options.hash_size);
        try self.transpositiontable.resize(tt_size);
        if (!self.mute) {
            self.print_hash_sizes();
        }
    }

    pub fn ucinewgame(self: *Engine) !void {
        if (self.is_busy()) return;
        self.set_startpos();
        self.transpositiontable.clear();
        self.is_newgame = true;
    }

    pub fn set_startpos(self: *Engine) void {
        if (self.is_busy()) return;
        self.pos.set_startpos();
        self.repetition_table[0] = self.pos.key;
    }

    /// After an illegal move we stop without crashing.
    pub fn set_startpos_with_optional_moves(self: *Engine, moves: ?[]const u8) !void {
        if (self.is_busy()) @panic("WTF");
        self.set_startpos();
        if (moves) |str| {
            self.parse_moves(str);
        }
    }

    /// Sets the position from fen + moves.
    /// - If fen is illegal we crash.
    /// - If fen is null the startpostion will be set.
    /// - After an illegal move we stop without crashing.
    pub fn set_position(self: *Engine, fen: ?[]const u8, moves: ?[]const u8) !void {
        if (self.is_busy()) return;
        const f = fen orelse {
            self.set_startpos();
            self.repetition_table[0] = self.pos.key;
            return;
        };
        try self.pos.set(f, self.options.is_960);
        self.repetition_table[0] = self.pos.key;
        if (moves) |m| {
            self.parse_moves(m);
        }
    }

    /// If we have any moves, make them. We stop if we encounter an illegal move.
    fn parse_moves(self: *Engine, moves: []const u8) void {
        var tokenizer = std.mem.tokenizeScalar(u8, moves, ' ');
        var idx: usize = 1;
        while (tokenizer.next()) |m| {
            const move: Move = self.pos.parse_move(m) catch break;
            self.pos.lazy_do_move(move);
            self.board_history[idx] = self.pos;
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
        self.busy.store(true, .monotonic);

        // Start our timer asap. There is quite some time spent on starting the whole thing.
        self.time_mgr = .init(go_params, self.pos.stm);

        self.controller_thread = try std.Thread.spawn(.{}, run_controller_thread, .{ self });
    }

    /// Stop threaded search.
    pub fn stop(self: *Engine) !void {
        if (!self.is_busy()) return;
        // Force the searcher to stop. Only this boss engine may do this atomic store to the searchers.
        @atomicStore(bool, &self.searcher.stopped, true, .monotonic);
        self.busy.store(false, .monotonic);
    }

    fn run_controller_thread(self: *Engine) !void {
        defer {
            self.controller_thread = null;
            self.is_newgame = false;
        }
        self.search_thread = try std.Thread.spawn(.{}, Searcher.start, .{ &self.searcher, self, self.is_newgame });
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

pub const Searcher = struct {
    /// Backlink ref. Only valid after start_search.
    engine: *Engine,
    /// Copied from mgr for direct access.
    transpositiontable: *tt.TranspositionTable,
    /// Copied from engine for direct access.
    time_mgr: *TimeMgr,
    /// The evaluator
    evaluator: hce.Evaluator,
    // Copied from mgr, for direct access.
    //termination: Termination,
    /// Position stack during search.
    position_stack: [max_search_depth + 2]Position,
    /// For now this is a blunt copy from the Engine.
    repetition_table: [max_game_length]u64,
    /// The one and only pv, updated after each iteration.
    pvnode: Node,
    /// Our node stack. Index #0 is our main pv. Indexing [ply + 1] for the current node.
    nodes: [max_search_depth + 2]Node,
    // History heuristics for beta cutoffs.
    hist: History,
    /// The current iteration.
    iteration: u8,
    /// Timeout. Any search value (0) returned - when stopped - should be discarded directly.
    stopped: bool,

    // Tracking search time.
    // timer: utils.Timer,

    /// Statistics.
    stats: Stats,
    /// This is the final move we output.
    selected_move: Move,

    fn init() Searcher {
        return .{
            .engine = undefined,
            .transpositiontable = undefined,
            .time_mgr = undefined,
            .evaluator = .init(),
            // .termination = .infinite,
            .position_stack = @splat(.empty),
            .repetition_table = @splat(0),
            .pvnode = .empty,
            .nodes = @splat(.empty),
            .hist = .init(),
            .iteration = 0,
            .stopped = false,
            // .timer = .empty,
            .stats = .empty,
            .selected_move = .empty,
        };
    }

    fn start(self: *Searcher, engine: *Engine, is_newgame: bool) void {
        // Shared stuff.
        self.engine = engine;
        //self.termination = engine.time_params.termination;
        self.transpositiontable = &engine.transpositiontable;
        self.time_mgr = &engine.time_mgr;

        // Copy stuff.
        const len: u16 = self.engine.pos.ply_from_root + 1;
        @memcpy(self.repetition_table[0..len], engine.repetition_table[0..len]);

        // Reset stuff.
        // self.timer.reset();
        self.pvnode = .empty;
        self.nodes = @splat(.empty);
        self.stopped = false;
        self.iteration = 0;
        self.stats = .empty;
        self.selected_move = .empty;

        if (is_newgame) {
            // Clear history heuristics.
            self.hist.clear();
        }

        // Make our private copy of the position.
        // Important to set the ply to zero.
        const pos: *Position = &self.position_stack[0];
        pos.* = engine.pos;
        pos.ply = 0;

        var rootmoves: position.MoveStorage = .init();
        pos.lazy_generate_all_moves(&rootmoves);

        // If we only have 1 legal move just make it without pv output and return bestmove.
        if (rootmoves.count == 1 and self.time_mgr.termination == .clock) {
            self.selected_move = rootmoves.moves[0];
            if (!engine.mute) {
                io.print_buffered("bestmove ", .{} );
                self.selected_move.print_buffered(pos.is_960);
                io.print_buffered("\n", .{});
                io.flush();
            }
            return;
        }

        // Have at least one move ready for crazy cases.
        self.selected_move = rootmoves.moves[0];

        //io.print("{} ns {} ms\n", .{ self.time_mgr.timer.read(), self.time_mgr.timer.elapsed_ms() });

        switch (pos.stm.e) {
            .white => self.iterate(Color.WHITE),
            .black => self.iterate(Color.BLACK),
        }
        if (!engine.mute) {
            io.print_buffered("bestmove ", .{} );
            self.selected_move.print_buffered(pos.is_960);
            io.print_buffered("\n", .{});
            io.flush();
        }
    }

    fn iterate(self: *Searcher, comptime us: Color) void {
        const pvnode: *Node = &self.pvnode;
        const iteration_node: *Node = &self.nodes[0];
        const pos: *const Position = &self.position_stack[0];

        // Statistics stuff.
        self.stats.non_terminal_nodes = 0;
        self.stats.beta_cutoffs = 0;

        var depth: u8 = 1;
        var score: Value = 0;

        // Search with increasing depth.
        iterationloop: while(true) {
            self.iteration = depth;
            self.stats.depth = depth;
            self.stats.seldepth = 0;
            iteration_node.clear();

            const start_non_terminal: u64 = self.stats.non_terminal_nodes;
            const start_beta_cutoffs: u64 = self.stats.beta_cutoffs;

            var alpha: Value = -infinity;
            var beta: Value = infinity;
            var delta: Value = 20;
            var fails: Value = 0;

            // The first few iterations no aspiration search.
            if (depth >= 4) {
                alpha = @max(score - delta, -infinity);
                beta = @min(score + delta, infinity);
            }

            aspiration_loop: while (true) {
                // Search with aspiration window.
                score = self.search(pos, depth - fails, alpha, beta, .pv, us, true, false);

                // Discard result if timeout.
                if (self.stopped) {
                    break :iterationloop;
                }

                // Alpha.
                if (score <= alpha) {
                    beta = div(alpha + beta, 2);
                    alpha = score - delta;
                }
                // Beta.
                else if (score >= beta) {
                    beta = score + delta;
                    if (alpha < 2000 and fails < 2) {
                        fails += 1;
                    }
                }
                // Success.
                else {
                    break :aspiration_loop;
                }

                delta = funcs.mul(delta, 1.55);
            }

            self.selected_move = iteration_node.pv.buffer[0];

            // Copy the last finished pv.
            pvnode.copy_from(iteration_node);

            // And print it, including some move ordering stats.
            if (!self.engine.mute) {
                const non_terminal_used: u64 = self.stats.non_terminal_nodes - start_non_terminal;
                const beta_cutoffs_used: u64 = self.stats.beta_cutoffs - start_beta_cutoffs;
                const qnodes: usize = funcs.percent(self.stats.nodes, self.stats.qnodes);
                const search_efficiency = funcs.percent(non_terminal_used, beta_cutoffs_used);
                const elapsed_nanos = self.time_mgr.timer.read();
                const ms = elapsed_nanos / 1_000_000;
                const nps: u64 = funcs.nps(self.stats.nodes, elapsed_nanos);

                io.print_buffered(
                    "info depth {} seldepth {} score cp {} nodes {} qnodes {}% time {} nps {} eff {}% pv",
                    .{ self.stats.depth, self.stats.seldepth, pvnode.score, self.stats.nodes, qnodes, ms, nps, search_efficiency }
                );

                for (pvnode.pv.slice()) |move| {
                    io.print_buffered(" ", .{});
                    move.print_buffered(pos.is_960);
                }
                io.print_buffered("\n", .{});
                io.flush();
            }

            // self.stopped = self.time_mgr.check_stop_early(self.selected_move, depth, self.stats.nodes);
            // if (self.stopped) {
            //     // io.debugprint("smart stop after {}\n", .{self.time_mgr.timer.elapsed_ms()});
            //     break :iterationloop;
            // }
                //io.debugprint("nodes spent {s} now {} total {}\n", .{ ex.move.to_string(false).slice(), self.stats.nodes - previous_nodes_spent, self.time.get_nodes_spent_ptr(ex.move).*});

            depth += 1;
            if (depth >= max_search_depth or self.check_stop()) {
                break :iterationloop;
            }
        }
    }

    fn search(self: *Searcher, pos: *const Position, input_depth: i32, input_alpha: Value, input_beta: Value, comptime mode: SearchMode, comptime us: Color, comptime is_root: bool, comptime cutnode: bool) Value {
        // Comptimes.
        const them: Color = comptime us.opp();
        const is_pvs: bool = comptime mode != .nonpv;

        if (comptime lib.is_guarded) {
            lib.guard(input_depth >= 0, "search inputdepth < 0", .{}); // TODO: use assert?
            lib.guard(input_beta > input_alpha, "search wrong alpha beta", .{});
            lib.guard(is_root or pos.ply > 0, "search wrong is_root vs ply", .{});
            lib.guard(!is_root or input_depth > 0, "search wrong is_root vs input_depth", .{});
            lib.guard(!is_root or is_root == is_pvs, "search wrong is_root vs is_pvs", .{});
            lib.guard(is_pvs or input_beta - input_alpha == 1, "search wrong is_pvs vs alphabeta", .{});
        }

        const node: *Node = &self.nodes[pos.ply];
        const childnode: *Node = &self.nodes[pos.ply + 1];
        node.clear_pv();
        childnode.clear();

        // Too deep.
        if (pos.ply >= max_search_depth) {
            return self.evaluate(pos);
        }

        // Update max reached depth.
        self.stats.seldepth = @max(self.stats.seldepth, pos.ply);

        // When reaching limit dive into quiescent search.
        if (input_depth <= 0) {
            return self.quiescence_search(pos, input_alpha, input_beta, mode, us);
        }

        // Locals.
        const parentnode: ?*const Node = if (!is_root) &self.nodes[pos.ply - 1] else null;
        const is_singular_extension: bool = !node.excluded_tt_move.is_empty();
        const is_check: bool = pos.checkers != 0;

        var depth: i32 = @max(0, input_depth);
        var alpha: Value = input_alpha;
        var beta: Value = input_beta;
        var extension: i32 = 0;
        var reduction: i32 = 0;
        var raw_static_eval: ?Value = null;

        // Check timeout at the start only.
        if (self.check_stop()) {
            return 0;
        }

        if (!is_root) {
            // Forced draw.
            if (pos.is_draw_by_insufficient_material() or self.is_draw_by_repetition_or_rule50(pos)) {
                return 0;
            }
            // Adjust distance to mate window.
            alpha = @max(alpha, -mate + pos.ply);
            beta = @min(beta, mate - pos.ply);

            // A beta cutoff may occur after adjusting the window.
            if (alpha >= beta) {
                return alpha;
            }
        }

        // TT Probe.
        var tt_pv: bool = is_pvs;
        const tt_entry: ?*const tt.Entry = self.transpositiontable.probe(pos.key);
        const tt_hit: bool = !is_singular_extension and tt_entry != null; // #testing 1.3 var
        const tt_move: Move = if (tt_hit) tt_entry.?.move else .empty;

        if (tt_hit) {
            tt_pv |= tt_entry.?.was_pv;
            const tt_is_usable_score: bool = tt_entry.?.is_score_usable_for_depth(alpha, beta, depth);
            if (!is_pvs and tt_is_usable_score) {
                return tt.get_adjusted_score_for_tt_probe(tt_entry.?.score, pos.ply);
            }
        }

        // Static Evaluation.
        if (is_check) {
            node.static_eval = null;
            node.eval = null;
        }
        else if (!is_singular_extension) {
            // Get the static eval from the TT if it is there, otherwise compute it.
            if (tt_hit and tt_entry.?.is_raw_static_eval_usable()) {
                raw_static_eval = tt_entry.?.raw_static_eval;
            }
            else {
                raw_static_eval = self.evaluate(pos);
                // Store the computed raw static eval in the TT.
                self.transpositiontable.store(.none, pos.key, 0, 0, .empty, -infinity, false, raw_static_eval);
            }
            // Apply our history correction on the node's static_eval.
            node.static_eval = self.apply_correction(pos, raw_static_eval.?, us);
            // Get the score from the TT if it is there, otherwise use the static eval.
            if (tt_hit and tt_entry.?.is_score_usable(node.static_eval.?, node.static_eval.?)) {
                node.eval = tt.get_adjusted_score_for_tt_probe(tt_entry.?.score, pos.ply);
            }
            else {
                node.eval = node.static_eval;
            }
        }

        // Check improvement compared to a previous node with same stm.
        var improvement: u1 = 0;
        node.improvementrate = 0.0;

        if (!is_check) {
            const prevnode: ?*const Node =
                if (pos.ply >= 2 and self.nodes[pos.ply - 2].static_eval != null) &self.nodes[pos.ply - 2]
                else if (pos.ply >= 4 and self.nodes[pos.ply - 4].static_eval != null) &self.nodes[pos.ply - 4]
                else null;

            if (prevnode != null) {
                improvement = if (node.static_eval.? > prevnode.?.static_eval.?) 1 else 0;
                node.improvementrate = calculate_improvementrate(prevnode.?.improvementrate, prevnode.?.static_eval.?, node.static_eval.?);
            }
        }

        // Pruning before processing any moves.
        if (!is_pvs and !is_check and !is_singular_extension) {

            // Reversed Static Futility Pruning.
            if (depth <= 4 and node.eval.? < mate_threshold) {
                const m: Value = if (improvement > 0) 40 else 74;
                const futility_margin: Value = depth * m;
                if (node.eval.? - futility_margin >= beta) {
                    return node.eval.?;
                }
            }

            // Razoring.
            if (depth <= 4 and node.static_eval.? + 450 * depth < alpha) {
                const r_score = self.quiescence_search(pos, alpha, alpha + 1, mode, us);
                if (self.stopped) {
                    return 0;
                }
                if (r_score <= alpha) return r_score;
            }

            // Null Move Pruning. Are we still good if we let them play another move?
            // Avoid doing this in endgame positions, which have a high probability of zugzwang.
            // if (!pos.nullmove_state and node.eval.? >= beta and node.static_eval.? >= (beta + 170 - 24 * depth) and pos.phase > 0 and pos.pieces_except_pawns_and_kings(us) != 0) {
            //     reduction = 3 + div(depth, 4);
            //     const n_depth: i32 = @max(0, depth - 1 - reduction);
            //     const next_pos: *const Position = self.do_nullmove(pos, us);
            //     const n_score: Value = -self.search(next_pos, n_depth, -beta, -beta + 1, .nonpv, them, false, !cutnode);
            //     if (self.stopped) {
            //         return 0;
            //     }
            //     if (n_score >= beta) {
            //         return if (n_score > mate_threshold) beta else n_score;
            //     }
            // }


            // #testing 1.3 nmp based on integral 3
            // Null Move Pruning. Are we still good if we let them play another move.
            // if (!pos.nullmove_state and node.eval.? >= beta and pos.phase > 0) {
            // //if (!pos.nullmove_state and node.eval.? >= beta and node.static_eval.? >= (beta + 170 - 24 * depth) and pos.phase > 0 and pos.pieces_except_pawns_and_kings(us) != 0) {
            //     const eval_reduction: i32 =  @min(2, div(node.eval.? - beta, 202));
            //     const r: i32 = clamp(div(depth, 4) + 3 + eval_reduction, 0, depth);
            //     const next_pos: *const Position = self.do_nullmove(pos, us);
            //     const n_score: Value = -self.search(next_pos, depth - r, -beta, -beta + 1, .nonpv, them, false, !cutnode);
            //     if (self.stopped) {
            //         return 0;
            //     }
            //     if (n_score >= beta) {
            //         return if (n_score > mate_threshold) beta else n_score;
            //     }
            // }


            // #testing 1.3 nmp based on integral 3 and 4
            // Null Move Pruning. Are we still good if we let them play another move?
            if (!pos.nullmove_state and node.eval.? >= beta and node.static_eval.? >= (beta + 170 - 24 * depth) and pos.phase > 0 and pos.pieces_except_pawns_and_kings(us) != 0) {
                const eval_reduction: i32 = @min(2, div(node.eval.? - beta, tuned.nmp_re));
                const r: i32 = clamp(div(depth, tuned.nmp_rf) + tuned.nmp_rb + eval_reduction, 0, depth);
                const next_pos: *const Position = self.do_nullmove(pos, us);
                const n_score: Value = -self.search(next_pos, depth - r, -beta, -beta + 1, .nonpv, them, false, !cutnode);
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

        // Easy access const.
        const depth_idx: usize = @intCast(depth);
        // For move ordering statistics.
        self.stats.non_terminal_nodes += 1;
        // Inherited node stuff.
        node.double_extensions = if (!is_root) parentnode.?.double_extensions else 0;

        // Now analyze the moves.
        var quiet_list: ExtMoveList(224) = .init();
        var capture_list: ExtMoveList(80) = .init();
        var movepicker: MovePicker(.search, us) = .init(pos, self, node, tt_move);
        var best_move: Move = .empty;
        var best_score: Value = -infinity;
        var moves_seen: u8 = 0;
        var quiets_seen: u8 = 0;

        // Move loop.
        moveloop: while (movepicker.next()) |ex| {

            if (comptime lib.is_paranoid) {
                assert(ex.is_ok(pos));
            }

            // Skip this move if we are inside singular extensions.
            if (ex.move == node.excluded_tt_move) {
                continue :moveloop;
            }

            const is_quiet: bool = ex.move.is_quiet();
            const is_capture: bool = ex.move.is_capture();

            // Pruning before we execute the move.
            if (!is_root and !is_check and best_score > -mate_threshold) {
                // Late Move Pruning.
                if (is_quiet and !is_check) {
                    if (moves_seen >= calculate_lmp_threshold(node.improvementrate, depth)) {
                        movepicker.skip_quiets();
                        continue :moveloop;
                    }
                }

                // Futility Pruning.
                const futility_margin = 196 + 96 * depth;
                if (is_quiet and depth <= 8 and node.eval.? + futility_margin < alpha) {
                    movepicker.skip_quiets();
                    continue :moveloop;
                }

                // SEE pruning.
                const see_threshold: Value = if (is_quiet) -64 * depth else -119 * depth;
                if (depth <= 8 and moves_seen >= 1 and !hce.see(pos, ex.move, see_threshold)) {
                    continue :moveloop;
                }

                // #testing 1.3.
                // History Pruning. Note: history pruning is quite sensitive and tricky.
                if (is_quiet and depth <= 5 and moves_seen >= 3) {
                    if (self.hist.quiet.get_score(ex) < -5000) {
                        //movepicker.skip_quiets();
                        continue :moveloop;
                    }
                }
            }

            // Determine extension + reduction.
            extension = 0;
            reduction = 0;

            // Singular Extensions.
            if (!is_root and tt_hit and depth >= 8 and ex.move == tt_move) {
                const entry: *const tt.Entry = tt_entry.?;
                const accurate_tt_score: bool = entry.bound != .alpha and entry.depth + 4 >= depth and @abs(entry.score) < mate_threshold;
                if (accurate_tt_score) {
                    node.excluded_tt_move = ex.move;
                    const reduced_depth: i32 = div(depth - 1, 2);
                    const s_beta = entry.score - (depth * 2);
                    const s_score = self.search(pos, reduced_depth, s_beta - 1, s_beta, .nonpv, us, false, cutnode);
                    node.excluded_tt_move = Move.empty;
                    if (self.stopped) {
                        return 0;
                    }

                    // No move beats the tt_move, so extend the tt_move's search.
                    if (s_score < s_beta) {
                        // Double extend if the tt move is singular by a big margin
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
                    else if (entry.score >= beta) {
                        extension = -1;
                    }
                }
            }

            if (is_check) {
                extension += 1;
            }

            // Do move
            const next_pos: *const Position = self.do_move(pos, us, ex.move);
            self.stats.nodes += 1;
            node.current_move = ex;
            //const previous_nodes_spent: u64 = if (is_root) self.stats.nodes else 0;

            var need_full_search: bool = true;
            var score: Value = -infinity;
            const new_depth: i32 = depth + extension - 1;

            // Late Move Reduction.
            const lmr_move_threshold: i32 = if (is_root) 3 else 1;
            if (!is_check and depth >= 3 and moves_seen >= lmr_move_threshold) {
                const is_quiet_idx: usize = @intFromBool(is_quiet);
                reduction = lmr_depth_reduction_table[is_quiet_idx][depth_idx][moves_seen];
                if (cutnode) reduction += 1;
                if (is_pvs) reduction -= 1;
                if (is_quiet and self.hist.quiet.get_score(ex) > 1000) reduction -= 1; // #testing 1.3
                if (node.improvementrate < -0.9) reduction += 1; // #testing 1.3
                reduction = std.math.clamp(reduction, 0, new_depth - 1);
                // Search nullwindow here.
                score = -self.search(next_pos, new_depth - reduction, -alpha - 1, -alpha, .nonpv, them, false, true);
                if (self.stopped) {
                    return 0;
                }
                need_full_search = score > alpha and reduction > 0;
            }
            else {
                need_full_search = !is_pvs or moves_seen >= 1;
            }

            // Either the move has potential from a reduced depth search or it's not expected to be a pv move, therefore we search it with a null window.
            if (need_full_search) {
                score = -self.search(next_pos, new_depth, -alpha - 1, -alpha, .nonpv, them, false, !cutnode);
                if (self.stopped) {
                    return 0;
                }
            }

            // Full window search on this move if this is probably a good move.
            if (is_pvs and (score > alpha or moves_seen == 0)) {
                score = -self.search(next_pos, new_depth, -beta, -alpha, .pv, them, false, false);
                if (self.stopped) {
                    return 0;
                }
            }

            moves_seen += 1;
            if (is_quiet) {
                quiets_seen += 1;
            }

            // Keep track of the nodes used for this move.
            // if (is_root) {
            //     self.time_mgr.update_nodes_spent(ex.move, self.stats.nodes - previous_nodes_spent);
            // }

            // Better move found.
            if (score > best_score) {
                best_score = score;
                // Alpha.
                if (score > alpha) {
                    best_move = ex.move;
                    node.update_pv(best_move, best_score, childnode);
                    alpha = score;
                    // Beta.
                    if (score >= beta) {
                        self.stats.beta_cutoffs += 1;
                        self.hist.record_beta_cutoff(parentnode, node, depth, quiet_list.slice(), capture_list.slice());
                        break :moveloop;
                    }
                }
            }

            if (ex.move != best_move) {
                if (is_quiet) {
                    quiet_list.add(ex);
                }
                else if (is_capture) {
                    capture_list.add(ex);
                }
                self.hist.capture.punish(depth, capture_list.slice());
            }

        } // (moveloop)

        if (moves_seen == 0) {
            return if (is_check) -mate + pos.ply else 0;
        }

        if (!is_singular_extension) {

            // TT store
            var tt_bound: Bound = .exact;
            if (alpha >= beta) {
                tt_bound = .beta;
            }
            else if (alpha <= input_alpha) {
                tt_bound = .alpha;
            }

            self.transpositiontable.store(tt_bound, pos.key, depth, pos.ply, best_move, best_score, tt_pv, raw_static_eval);

            // Update correction history.
            if (!is_check and !best_move.is_empty() and best_move.is_quiet() and !(tt_bound == .alpha and best_score <= node.static_eval.?) and !(tt_bound == .beta and best_score >= node.static_eval.?)) {
                self.hist.correction.update(pos, node.static_eval.?, best_score, depth, us);
            }
        }

        return best_score;
    }

    fn quiescence_search(self: *Searcher, pos: *const Position, input_alpha: Value, input_beta: Value, comptime mode: SearchMode, comptime us: Color) Value {
        // Comptimes.
        const is_pvs: bool = comptime mode != .nonpv;
        const them = comptime us.opp();

        if (comptime lib.is_paranoid) {
            assert(pos.stm.e == us.e);
            assert(pos.ply > 0);
        }

        const node: *Node = &self.nodes[pos.ply];
        const childnode: *Node = &self.nodes[pos.ply + 1];
        node.clear_pv();
        childnode.clear();

        // Too deep.
        if (pos.ply >= max_search_depth) {
            return self.evaluate(pos);
        }

        // Update stats.
        self.stats.seldepth = @max(self.stats.seldepth, pos.ply);

        if (pos.is_draw_by_insufficient_material() or self.is_draw_by_repetition_or_rule50(pos)) {
            return 0;
        }

        // Check timeout at the start only. At the callsites just read the stopped value.
        if (self.check_stop()) {
            return 0;
        }

        const is_check : bool = pos.checkers > 0;
        const tt_depth: Value = @intFromBool(is_check);
        var alpha = input_alpha;
        var beta: Value = input_beta;
        var score: Value = 0;
        var tt_pv = is_pvs;
        var best_score = -infinity;
        var raw_static_eval: ?Value = null;

        // Limit dtm. TODO: is this needed in quiescence?
        alpha = @max(alpha, -mate + pos.ply);
        beta = @min(beta, mate - pos.ply);

        const tt_entry: ?*const tt.Entry = self.transpositiontable.probe(pos.key);
        const tt_hit: bool = tt_entry != null;
        const tt_move: Move = if (tt_hit) tt_entry.?.move else .empty;

        if (tt_hit) {
            tt_pv |= tt_entry.?.was_pv;
            const tt_is_usable_score: bool = tt_entry.?.is_score_usable_for_depth(alpha, beta, tt_depth);
            if (!is_pvs and tt_is_usable_score) {
                return tt.get_adjusted_score_for_tt_probe(tt_entry.?.score, pos.ply);
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
            }

            // Apply history correction.
            var eval: Value = self.apply_correction(pos, raw_static_eval.?, us);
            if (tt_hit and tt_entry.?.is_score_usable(score, score)) {
                eval = tt.get_adjusted_score_for_tt_probe(tt_entry.?.score, pos.ply);
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
        var moves_seen: u8 = 0;
        const qs_futility_score: Value = best_score + 101;

        // Move loop.
        moveloop: while (movepicker.next()) |ex| {
            if (comptime lib.is_paranoid) {
                assert(ex.is_ok(pos));
            }

            // Skip bad noisies.
            if (moves_seen > 0 and @intFromEnum(movepicker.stage) > @intFromEnum(movepick.Stage.noisy)) {
                break :moveloop;
            }

            // Quiescence Futility Pruning. Prune capture moves that don't win material if the static eval is behind alpha by some margin
            if (!is_check and ex.move.is_capture() and qs_futility_score <= alpha and !hce.see(pos, ex.move, 1)) {
                best_score = @max(best_score, qs_futility_score);
                continue :moveloop;
            }

            // Do move
            const next_pos: *const Position = self.do_move(pos, us, ex.move);
            self.stats.nodes += 1;
            self.stats.qnodes += 1;

            score = -self.quiescence_search(next_pos, -beta, -alpha, mode, them);
            if (self.stopped) {
                return 0;
            }

            moves_seen += 1;

            // Better move.
            if (score > best_score) {
                best_score = score;
                // Alpha.
                if (score > alpha) {
                    alpha = score;
                    node.update_pv(ex.move, best_score, childnode);
                    // Beta.
                    if (alpha >= beta) {
                        break :moveloop;
                    }
                }
            }
        }

        if (moves_seen == 0 and is_check) {
            return -mate + pos.ply;
        }

        // TT store. We should not store a best move because the quiscence search is incomplete by definition.
        const bound: tt.Bound = if (alpha >= beta) .beta else .alpha;
        self.transpositiontable.store(bound, pos.key, tt_depth, pos.ply, Move.empty, best_score, tt_pv, raw_static_eval);
        return best_score;
    }

    /// Assumes not in check.
    fn evaluate(self: *Searcher, pos: *const Position) Value {
        if (comptime lib.is_paranoid) {
            assert(pos.checkers == 0);
        }
        return self.evaluator.evaluate(pos);
    }

    fn apply_correction(self: *Searcher, pos: *const Position, raw_static_eval: Value, comptime us: Color) Value {
        const e: Value = raw_static_eval + self.hist.correction.get_correction(pos, us);
        return std.math.clamp(e, -mate_threshold, mate_threshold);
    }

    // Assumes not in check.
    // fn evaluate_with_correction(self: *Searcher, pos: *const Position, comptime us: Color) Value {
    //     const e: Value = self.evaluate(pos) + self.hist.correction.get_correction(pos, us);
    //     return std.math.clamp(e, -mate_threshold, mate_threshold);
    // }

    /// Not used. It messes up the static eval and correction history. Maybe later usable.
    fn adjust_for_drawcounter(score: Value, drawcounter: i32) Value {
        if (score == 0 or score <= -mate_threshold or score >= mate_threshold) {
            return score;
        }
        // Max penalty around 25 centipawns.
        const penalty: i32 = div(drawcounter * drawcounter, 400);

        // Reduce evaluation.
        if (score > 0) {
            return @max(0, score - penalty);
        }
        else {
            return @min(0, score + penalty);
        }
    }

    fn do_move(self: *Searcher, pos: *const Position, comptime us: Color, m: Move) *const Position {
        const next_pos: *Position = &self.position_stack[pos.ply + 1];
        next_pos.* = pos.*;
        next_pos.do_move(us, m);
        self.repetition_table[next_pos.ply_from_root] = next_pos.key;
        return next_pos;
    }

    fn do_nullmove(self: *Searcher, pos: *const Position, comptime us: Color) *const Position {
        const next_pos: *Position = &self.position_stack[pos.ply + 1];
        next_pos.* = pos.*;
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

    fn check_stop(self: *Searcher) bool {
        // We already stopped.
        if (self.stopped) {
            return true;
        }

        // Once in a while check if the engine told us to stop.
        const node_interval: bool = self.stats.nodes % 1024 == 0;
        if (node_interval and !self.engine.is_busy()) {
            self.stopped = true;
            return true;
        }

        switch (self.time_mgr.termination) {
            .nodes => {
                if (self.stats.nodes >= self.time_mgr.max_nodes) {
                    self.stopped = true;
                }
            },
            .depth => {
                if (self.iteration > self.time_mgr.max_depth) {
                    self.stopped = true;
                }
            },
            .movetime, .clock => {
                if (node_interval and self.time_mgr.timer.elapsed_ms() >= self.time_mgr.max_movetime) {
                    self.stopped = true;
                }
            },
            else => {
                return false;
            },
        }
        return self.stopped;
    }

    /// Not used.
    fn try_set_pv_from_tt(self: *Searcher, node: *Node) void {
        var pos: Position = self.position_stack[0]; // temp copy
        var i: u8 = 0;
        while (true) {
            const entry = self.transpositiontable.get(pos.key);
            if (entry.key != pos.key or entry.bound != .Exact) break;
            const move = entry.move;
            // Quick check.
            if (pos.board[move.from.u].is_empty()) break;
            var any: position.MoveFinder = .init(move.from, move.to, move.flags);
            pos.lazy_generate_all_moves(&any);
            if (!any.found()) break;
            pos.lazy_do_move(any.move);
            node.pv.append_assume_capacity(any.move);
            i += 1;
            if (i > 127) break;
        }
    }
};

pub const Node = struct {
    /// Local PV during search.
    pv: PV = .{},
    /// Current pv score.
    score: Value = -infinity,
    /// The current move being searched in the tree.
    current_move: ExtMove = .empty,
    /// Skipped move during singular extensions.
    excluded_tt_move: Move = .empty,
    /// The number of double extensions done in singular extensions.
    double_extensions: u8 = 0,
    /// Static eval including correction.
    static_eval: ?Value = null,
    /// Most appropriate eval.
    eval: ?Value = null,
    /// Used for lmp.
    improvementrate: f32 = 0.0,
    /// Killer move heuristic.
    killers: [2]Move = .{ .empty, .empty },

    const empty: Node = .{};

    /// Clear completely.
    fn clear(self: *Node) void {
        self.* = empty;
    }

    /// Only reset pv. The rest of the fields have to be assigned inside search.
    fn clear_pv(self: *Node) void {
        self.pv.len = 0;
        self.score = -infinity;
    }

    /// Sets pv to bestmove + childnode.pv and sets self.score.
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

/// 64 bits Move with score and info, used during search.
pub const ExtMove = packed struct {
    /// The generated move.
    move: Move = .empty,
    /// Set by movepicker.
    score: i32 = types.invalid_movescore,
    /// Set by movepicker.
    piece: Piece = Piece.NO_PIECE,
    /// Set by movepicker.
    captured: Piece = Piece.NO_PIECE,
    /// Set by movepicker.
    is_tt_move: bool = false,
    /// Set by movepicker.
    is_killer: bool = false,
    /// Set by movepicker.
    is_bad_capture: bool = false,

    pub const empty: ExtMove = .{};

    pub fn init(move: Move) ExtMove {
        return .{ .move = move };
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
        if (self.score < types.invalid_movescore + 10_000_000) {
            io.debugprint("no score {any}\n", .{ self });
            return false;
        }
        return true;
    }
};

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


const SearchMode = enum {
    pv,
    nonpv,
};

/// Not used (yet).
const Error = error {
    EngineIsStillRunning,
};

/// LMP. Returns a new improvement rate based on the previous one.
pub fn calculate_improvementrate(prev_improvementrate: f32, prev_eval: Value, curr_eval: Value) f32 {
    const diff: Value = curr_eval - prev_eval;
    const r: f32 = prev_improvementrate + float(diff) / 30.0;
    return clamp(r, -1.0, 1.0);
}

/// LMP. Returns a moves_seen threshold from improvementrate and depth.
pub fn calculate_lmp_threshold(improvementrate: f32, depth: i32) u8 {
    const r: f32 = (5.0 + float(depth * depth)) / (3.0 - @max(0.0, improvementrate));
    return if (r >= max_move_count) max_move_count else @intCast(funcs.int(@abs(r)));
}

/// LMR. Indexing: [quiet][depth][moves_seen]
const lmr_depth_reduction_table: [2][max_search_depth][max_move_count]u8 = compute_reduction_table();

fn compute_lmr(depth: usize, moves: usize, base: f32, divisor: f32) u8 {
        const d: f32 = @floatFromInt(depth);
        const m: f32 = @floatFromInt(moves);
        const ln_depth: f32 = @log(d);
        const ln_moves: f32 = @log(m);
        return @intFromFloat(base + ln_depth * ln_moves / divisor);
}

fn compute_reduction_table() [2][max_search_depth][max_move_count]u8 {
    @setEvalBranchQuota(128000);
    var result: [2][max_search_depth][max_move_count]u8 = std.mem.zeroes([2][max_search_depth][max_move_count]u8);
    for (1..max_search_depth) |depth| {
        for (1..max_move_count) |move| {
            result[0][depth][move] = compute_lmr(depth, move, -0.24, 2.60); // noisy
            result[1][depth][move] = compute_lmr(depth, move, 1.00, 2.00); // quiet
        }
    }
    return result;
}

const tuned = struct {
    const nmp_rb: u16 = 3; // 3 or 4 (3 seems best)
    const nmp_rf: u16 = 4;
    const nmp_re: u16 = 202;
};
