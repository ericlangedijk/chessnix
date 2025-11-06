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

const assert = std.debug.assert;
const ctx = lib.ctx;
const io = lib.io;
const wtf = lib.wtf;

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

/// Principal Variation (+ 2 to be safe).
const PV = lib.BoundedArray(Move, max_search_depth + 2);

pub const Engine = struct {
    /// Shared tt for all threads.
    transpositiontable: tt.TranspositionTable,
    /// Shared tt for all threads.
    evaltranspositiontable: tt.EvalTranspositionTable,
    /// Shared tt for all threads.
    pawntranspositiontable: tt.PawnTranspositionTable,
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
    time_params: TimeParams,
    /// The actual worker.
    searcher: Searcher,
    /// No output if true.
    mute: bool,
    /// Set on ucinewgame.
    is_newgame: bool,

    /// Engine is created on the heap.
    pub fn create(mute: bool) !*Engine {
        const sizes: tt.TTSizes = tt.compute_tt_sizes(Options.default_hash_size);
        var engine: *Engine = try ctx.galloc.create(Engine);
        engine.transpositiontable = try .init(sizes.tt);
        engine.evaltranspositiontable = try .init(sizes.eval);
        engine.pawntranspositiontable = try .init(sizes.pawneval);
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
        engine.time_params = TimeParams.default;
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
        self.evaltranspositiontable.deinit();
        self.pawntranspositiontable.deinit();
        ctx.galloc.destroy(self);
    }

    fn print_hash_sizes(self: *const Engine) void {
        io.print("info string hash {} MB entries tt {} entries eval {} entries pawneval {}\n", .{ self.options.hash_size, self.transpositiontable.data.len, self.evaltranspositiontable.data.len, self.pawntranspositiontable.data.len });
    }

    pub fn apply_hash_size(self: *Engine) !void {
        if (self.is_busy()) return;
        const sizes: tt.TTSizes = tt.compute_tt_sizes(self.options.hash_size);
        try self.transpositiontable.resize(sizes.tt);
        try self.evaltranspositiontable.resize(sizes.eval);
        try self.pawntranspositiontable.resize(sizes.pawneval);
        if (!self.mute) {
            self.print_hash_sizes();
        }
    }

    pub fn ucinewgame(self: *Engine) !void {
        if (self.is_busy()) return;
        self.set_startpos();
        self.transpositiontable.clear();
        self.evaltranspositiontable.clear();
        self.pawntranspositiontable.clear();
        self.is_newgame = true;
    }

    pub fn set_startpos(self: *Engine) void {
        if (self.is_busy()) return;
        self.pos.set_startpos();
        self.repetition_table[0] = self.pos.key;
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
    /// - If fen is illegal we crash.
    /// - If fen is null the startpostiion will be set.
    /// - After an illegal move we stop without crashing.
    pub fn set_position(self: *Engine, fen: ?[]const u8, moves: ?[]const u8) !void {
        if (self.is_busy()) return;
        const f = fen orelse {
            self.set_startpos();
            self.repetition_table[0] = self.pos.key;
            return;
        };
        try self.pos.set(f);
        self.repetition_table[0] = self.pos.key;
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
        if (self.is_busy()) return;
        self.busy.store(true, .monotonic);
        self.time_params = .init(go_params, self.pos.stm);
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
    /// /// The total amount of nodes.
    nodes: u64 = 0,
    /// The total amount of nodes done in quiescence_search.
    qnodes : u64 = 0,
    /// Iteration reached.
    depth: u16 = 0,
    /// The max reached depth, including quiescence depth and extensions.
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
    /// Copied from mgr, for direct access.
    transpositiontable: *tt.TranspositionTable,
    /// Copied from mgr, for direct access.
    evaltranspositiontable: *tt.EvalTranspositionTable,
    /// Copied from mgr, for direct access.
    pawntranspositiontable: *tt.PawnTranspositionTable,
    /// The evaluator
    evaluator: hce.Evaluator(false),
    /// Copied from mgr, for direct access.
    termination: Termination,
    /// Position stack during search.
    position_stack: [max_search_depth + 2]Position,
    /// For now this is a blunt copy from the Engine.
    repetition_table: [max_game_length]u64,
    /// Our rootmoves for the pos (stay unsorted).
    rootmoves: MovePicker,
    /// Our node stack. Index #0 is our main pv. Indexing [ply + 1] for the current node.
    nodes: [max_search_depth + 2]Node,
    // Heuristics for beta cutoffs.
    history_heuristics: History,
    /// The current iteration.
    iteration: u8,
    /// Timeout. Any search value (0) returned - when stopped - should be discarded directly.
    stopped: bool,
    /// Tracking search time.
    timer: utils.Timer,
    /// Statistics.
    stats: Stats,
    /// the global best move.
    selected_move: Move,

    fn init() Searcher {
        return .{
            .engine = undefined,
            .transpositiontable = undefined,
            .evaltranspositiontable = undefined,
            .pawntranspositiontable = undefined,
            .evaluator = .init(),
            .termination = .infinite,
            .position_stack = @splat(.empty),
            .repetition_table = @splat(0),
            .rootmoves = .empty,
            .nodes = @splat(.empty),
            .history_heuristics = .init(),
            .iteration = 0,
            .stopped = false,
            .timer = .empty,
            .stats = .empty,
            .selected_move = .empty,
        };
    }

    fn start(self: *Searcher, engine: *Engine, is_newgame: bool) void {
        // Shared stuff.
        self.engine = engine;
        self.termination = engine.time_params.termination;
        self.transpositiontable = &engine.transpositiontable;
        self.evaltranspositiontable = &engine.evaltranspositiontable;
        self.pawntranspositiontable = &engine.pawntranspositiontable;

        // Copy stuff.
        const len: u16 = self.engine.pos.ply_from_root + 1;
        @memcpy(self.repetition_table[0..len], engine.repetition_table[0..len]);

        // Reset stuff.
        self.timer.reset();
        self.nodes = @splat(.empty);
        self.rootmoves = .empty;
        self.stopped = false;
        self.iteration = 0;
        self.stats = .empty;
        self.selected_move = .empty;

        // Clear or decay beta_history.
        if (is_newgame) {
            self.history_heuristics.clear();
        }
        else {
            self.history_heuristics.decay();
        }

        // Make our private copy of the position.
        // Important to set the ply to zero.
        const pos: *Position = &self.position_stack[0];
        pos.* = engine.pos;
        pos.ply = 0;

        // Generate the rootmoves.
        switch (pos.stm.e) {
            .white => self.rootmoves.generate_moves(pos, Color.WHITE),
            .black => self.rootmoves.generate_moves(pos, Color.BLACK),
        }

        // If we only have 1 legal move just make it without pv output and return.
        if (self.rootmoves.count == 1 and self.termination == .clock) {
            self.selected_move = self.rootmoves.extmoves[0].move;
            if (!engine.mute) {
                io.print("bestmove {f}\n", .{ self.selected_move } );
            }
            return;
        }

        switch (pos.stm.e) {
            .white => self.iterate(Color.WHITE),
            .black => self.iterate(Color.BLACK),
        }
        if (!engine.mute) {
            io.print("bestmove {f}\n", .{ self.selected_move });
        }
    }

    fn iterate(self: *Searcher, comptime us: Color) void {
        const rootnode: *Node = &self.nodes[0];
        const iteration_node: *Node = &self.nodes[1];
        const pos: *const Position = &self.position_stack[0];

        // Maybe usable later...
        // self.try_set_pv_from_tt(rootnode);
        // if (!mute and rootnode.pv.len > 0) {
        //     self.print_pv(0);
        // }

        self.stats.non_terminal_nodes = 0;
        self.stats.beta_cutoffs = 0;

        var depth: u8 = 1;
        var score: Value = 0;
        //var prev_score: Value = -infinity;

        // Search with increasing depth.
        iterationloop: while(true) {
            self.iteration = depth;
            self.stats.depth = depth;
            self.stats.seldepth = 0;

            const start_nodes: u64 = self.stats.nodes;
            const start_non_terminal: u64 = self.stats.non_terminal_nodes;
            const start_beta_cutoffs: u64 = self.stats.beta_cutoffs;

            var alpha: Value = -infinity;
            var beta: Value = infinity;
            var delta: Value = 25;

            // The first few iterations no aspiration search.
            if (depth >= 4) {
                alpha = @max(score - delta, -infinity);
                beta = @min(score + delta, infinity);
            }

            aspiration_loop: while (true) {
                // Search with aspiration window.
                score = self.search(pos, depth, alpha, beta, .pv, us, true);

                // Discard result if timeout.
                if (self.stopped) {
                    break :iterationloop;
                }

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

                delta += delta;
            }

            // Copy the last finished pv.
            rootnode.copy_from(iteration_node);

            // And print it, including some move ordering stats.
            if (!self.engine.mute) {

                const nodes_used: u64 = self.stats.nodes - start_nodes;
                const non_terminal_used: u64 = self.stats.non_terminal_nodes - start_non_terminal;
                const beta_cutoffs_used: u64 = self.stats.beta_cutoffs - start_beta_cutoffs;

                const mbf: f32 = if (non_terminal_used > 0)
                        @as(f32, @floatFromInt(nodes_used)) / @as(f32, @floatFromInt(non_terminal_used))
                    else
                        0.0;

                const search_eff: f32 = if (non_terminal_used > 0)
                    (@as(f32, @floatFromInt(beta_cutoffs_used)) / @as(f32, @floatFromInt(non_terminal_used))) * 100.0
                else
                    0.0;

                const elapsed_nanos = self.timer.read();
                const ms = elapsed_nanos / 1000000;
                const nps: u64 = funcs.nps(self.stats.nodes, elapsed_nanos);

                io.print(
                    "info depth {} seldepth {} score cp {} nodes {} nps {} mbf {d:.2} bc {d:.2} time {} pv {f}\n",
                    .{ self.stats.depth, self.stats.seldepth, rootnode.score, self.stats.nodes, nps, mbf, search_eff, ms, rootnode }
                );
            }

            depth += 1;
            if (depth >= max_search_depth or self.check_stop()) {
                break :iterationloop;
            }
        }
    }

    fn search(self: *Searcher, pos: *const Position, input_depth: i32, input_alpha: Value, input_beta: Value, comptime mode: SearchMode, comptime us: Color, comptime is_root: bool) Value {
        // Comptimes.
        const them: Color = comptime us.opp();
        const is_pvs: bool = comptime mode != .nonpv;

        if (comptime lib.is_paranoid) {
            assert(input_beta > input_alpha);
            assert(!is_root or input_depth > 0);
            assert(!is_root or is_root == is_pvs);
        }

        // Consts
        const ply: u16 = pos.ply;
        const this_node_idx: u16 = ply + 1;
        const depth: i32 = @max(0, input_depth);
        const key: u64 = pos.key;
        const is_check: bool = pos.checkers > 0;

        // Vars.
        var alpha: Value = input_alpha;
        var beta: Value = input_beta;
        var extension: i32 = 0;
        var reduction: i32 = 0;

        // Update counters.
        self.stats.nodes += 1;
        self.stats.seldepth = @max(self.stats.seldepth, ply);

        // Check timeout at the start only. At the callsites just read the stopped value.
        if (self.check_stop()) {
            return 0;
        }

        if (!is_root) {
            // Too deep.
            if (ply >= max_search_depth) {
                return self.evaluate(pos);
            }

            if (pos.is_draw_by_insufficient_material() or self.is_draw_by_repetition_or_rule50(pos)) {
                const random: u2 = @truncate(self.stats.nodes & 2);
                return @as(i32, 1) - random;
            }

            // Adjust dtm window.
            alpha = @max(alpha, -mate + ply);
            beta = @min(beta, mate - ply);

             // A beta cutoff may occur after adjusting the window.
            if (alpha >= beta) {
                return alpha;
            }
        }

        const node: *Node = &self.nodes[this_node_idx];
        const childnode: *Node = &self.nodes[this_node_idx + 1];
        node.clear();

        // Clear killers for the next ply.
        childnode.killers = @splat(Move.empty);

        // TODO: use tt alpha beta info: adjust window??
        // TT Probe. Keeping the stored tt_move for move ordering.
        var tt_move: Move = .empty;
        if (self.tt_probe(key, depth, ply, alpha, beta, &tt_move)) |tt_score| {
            if (!is_pvs) {
                // Boost this move.
                if (tt_move.is_quiet()) {
                    self.history_heuristics.update_quiet_score(pos, tt_move, depth);
                }
                return tt_score;
            }
        }

        // Evaluation.
        const static_eval: Value = self.evaluate(pos) + self.history_heuristics.get_correction(pos);

        // Any improvement compared to previous search (same stm)?
        const improvement: Value = if (ply < 2 or static_eval <= self.nodes[this_node_idx - 2].score) 0 else 1;

        // Requested depth reached: plunge into quiescence.
        if (depth <= 0) {
            return self.quiescence_search(pos, alpha, beta, mode, us);
        }

        // Reversed Futility Pruning
        if (!is_pvs and !is_check and depth <= 4 and static_eval < mate_threshold) {
            const mul: Value =  if (improvement > 0) 40 else 74;
            const futility_margin: Value = depth * mul;
            if (static_eval - futility_margin >= beta) {
                return static_eval;
            }
        }

        // Null Move Pruning.
        if (!is_pvs and !is_check and !pos.nullmove_state and depth >= 3 and static_eval > beta and pos.all_except_pawns_and_kings() != 0) {
            reduction = 2 + @divTrunc(depth, 4) + @min(3, @divTrunc(static_eval - beta, 200));
            const next_pos: *const Position = self.do_nullmove(pos, us);
            const score: Value = -self.search(next_pos, depth - 1 - reduction, -beta, -beta + 1, .nonpv, them, false);
            if (self.stopped) {
                return 0; // discard
            }
            if (score >= beta) {
                return if (score > mate_threshold) beta else score;
            }
        }

        // Generate moves. if at root copy the rootmoves (has no scores) otherwise generate.
        var movepicker: MovePicker = .empty;
        if (is_root) {
            movepicker.copy_from(&self.rootmoves);
        }
        else {
            movepicker.generate_moves(pos, us);
        }
        movepicker.process_and_score_moves(self, pos, tt_move, node.killers);

        // Checkmate or stalemate.
        if (movepicker.count == 0) {
            return if (is_check) -mate + ply else 0;
        }

        // For move ordering statistics.
        self.stats.non_terminal_nodes += 1;

        var best_score: Value = -infinity;
        var best_move: Move = .empty;

        // Move loop.
        moveloop: for (0..movepicker.count) |move_idx| {
            const ex: ExtMove = movepicker.extract_next(move_idx);
            const is_quiet: bool = ex.move.is_quiet();

            if (!is_root and !is_pvs and !is_check and is_quiet) {
                // Futility Pruning.
                if (depth < 3 and move_idx > 3 and static_eval + (depth * 24) < alpha) {
                    continue :moveloop;
                }
                // Late Move Pruning.
                const lmp_threshold: [5]u8 = .{ 0, 3, 6, 12, 24 };
                if (self.iteration > 3 and depth <= 4 and is_quiet and move_idx > lmp_threshold[@abs(depth)]) {
                    continue : moveloop;
                }
            }

            // Do move
            const next_pos: *const Position = self.do_move(pos, us, ex.move);

            // Determine extension or reduction.
            var need_full_search: bool = true;
            extension = 0;
            reduction = 0;
            var score: Value = alpha;

            // Extension.
            if (is_check and (is_pvs or depth >= 4)) {
                extension = 1;
            }

            // Late Move Reduction.
            if (!is_check and depth >= 3 and move_idx > 0 and !ex.is_killer and is_quiet) {
                reduction = lmr_depth_reduction_table[@abs(depth)][move_idx + 1];
                if (is_pvs) reduction -= 1;
                if (improvement == 0) reduction += 1;
                if (reduction < 0) reduction = 0;
            }

            const null_window: bool = reduction != 0;

            // Search nullwindow here.
            if (null_window) {
                score = -self.search(next_pos, depth - 1 - reduction, -alpha - 1, -alpha, .nonpv, them, false);
                need_full_search = score > alpha;
            }

            if (self.stopped) {
                return 0; // discard
            }

            // Either the move has potential from a reduced depth search or it's not expected to be a PV move, therefore we search it with a null window.
            // Borrowed logic from Integral 3. This finally gave depth to the search.
            if (need_full_search) {
                score = -self.search(next_pos, depth + extension - 1, -alpha - 1, -alpha, .nonpv, them, false);
            }

            if (self.stopped) {
                return 0; // discard
            }

            // Full window search on this move if it is probably good.
            // Borrowed logic from Integral 3. This finally gave depth to the search.
            if (is_pvs and (score > alpha or move_idx == 0)) {
                score = -self.search(next_pos, depth + extension - 1, -beta, -alpha, .pv, them, false);
            }

            if (self.stopped) {
                return 0; // discard
            }

            // Better move found.
            if (score > best_score) {
                best_score = score;

                // Alpha.
                if (score > alpha) {
                    best_move = ex.move;
                    if (is_root) {
                        self.selected_move = ex.move;
                    }

                    if (is_pvs){
                        node.update_pv(best_move, best_score, childnode);
                    }

                    alpha = score;

                    // Beta.
                    if (score >= beta) {
                        self.stats.beta_cutoffs += 1;
                        if (is_quiet) {
                            self.history_heuristics.record_quiet_beta_cutoff(node, ex, depth, &movepicker, move_idx);
                        }
                        break :moveloop;
                    }
                }
            }
        } // (moveloop)

        // Nothing to store.
        if (best_move.is_empty()) {
            return best_score;
        }

        // TT store
        var tt_bound: Bound = .Exact;
        if (alpha >= beta) {
            tt_bound = .Beta;
        }
        else if (alpha <= input_alpha) {
            tt_bound = .Alpha;
        }
        self.tt_store(tt_bound, key, depth, ply, best_move, best_score);


        // Wiki: An update to correction history happens when the following conditions are satisfied:
        // - Side-to-move is not in check.
        // - Best move either does not exist or is quiet.
        // - If score type is Lower bound, then score should not be below static evaluation.
        // - If score type is Upper bound, then score should not be above static evaluation.
        if (!is_check and best_move != tt_move and best_move.is_quiet() and !(tt_bound == .Alpha and best_score <= static_eval) and !(tt_bound == .Beta and best_score >= static_eval)) {
             self.history_heuristics.update_correction_history(pos, static_eval, best_score, depth);
        }

        return best_score;
    }

    fn quiescence_search(self: *Searcher, pos: *const Position, input_alpha: Value, input_beta: Value, comptime mode: SearchMode, comptime us: Color) Value {
        if (comptime lib.is_paranoid) {
            assert(pos.stm.e == us.e);
            assert(pos.ply > 0);
        }

        const is_pvs: bool = comptime mode != .nonpv;
        const ply: u16 = pos.ply;

        // Update stats.
        self.stats.nodes += 1;
        self.stats.qnodes += 1;
        self.stats.seldepth = @max(self.stats.seldepth, pos.ply);

        // Check timeout at the start only. At the callsites just read the stopped value.
        if (self.check_stop()) {
            return 0;
        }

        const node: *Node = &self.nodes[pos.ply + 1];
        node.clear();

        if (ply >= max_search_depth) {
            return self.evaluate(pos);
        }

        const them = comptime us.opp();
        const key: u64 = pos.key;
        const is_check : bool = pos.checkers > 0;
        var alpha = input_alpha;
        var beta: Value = input_beta;
        var score: Value = 0;

        // Limit.
        alpha = @max(alpha, -mate + ply);
        beta = @min(beta, mate - ply);

        if (!is_check) {
            // Static eval.
            score = self.evaluate(pos);
            // Fail high.
            if (score >= beta) return score;
            // Raise alpha.
            if (score > alpha) alpha = score;
        }

        // TT probe
        var tt_move: Move = .empty;
        if (self.tt_probe(key, 0, ply, alpha, beta, &tt_move)) |tt_score| {
            if (!is_pvs)
                return tt_score;
        }

        var movepicker: MovePicker = .empty;
        movepicker.generate_captures(pos, us);
        movepicker.process_and_score_moves(self, pos, tt_move, .{ Move.empty, Move.empty });

        // Mate or stalemate?
        if (movepicker.count == 0) {
            if (is_check) {
                return -mate + ply;
            }
            return alpha;
        }

        var best_move: Move = .empty;

        // Move loop.
        moveloop: for (0..movepicker.count) |move_idx| {
            const ex: ExtMove = movepicker.extract_next(move_idx);

            // Skip bad captures.
            if (!is_pvs and !is_check and ex.is_bad_capture and move_idx > 0) {
                continue :moveloop;
            }

            // Do move
            const next_pos: *const Position = self.do_move(pos, us, ex.move);

            score = -self.quiescence_search(next_pos, -beta, -alpha, mode, them);

            if (self.stopped) {
                return 0; // discard
            }

            // Better move.
            if (score > alpha) {
                alpha = score;
                best_move = ex.move;
                if (score >= beta) {
                    self.tt_store(.Beta, key, 0, ply, ex.move, score);
                    return score;
                }
            }
        }

        if (!best_move.is_empty()) {
            self.tt_store(.Alpha, key, 0, ply, best_move, alpha);
        }
        return alpha;
    }

    fn evaluate(self: *Searcher, pos: *const Position) Value {
        return self.evaluator.evaluate(pos, self.evaltranspositiontable, self.pawntranspositiontable);
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
        return next_pos;
    }

    fn tt_store(self: *Searcher, bound: Bound, key: u64, depth: i32, ply: u16, move: Move, score: Value) void {
        if (comptime lib.is_paranoid) {
            assert(depth >= 0);
            assert(!self.stopped);
        }
        self.transpositiontable.store(bound, key, @intCast(depth), ply, move, score);
    }

    /// Returned hashmove can be used for move ordering when not empty. tt_move must be empty on call.
    fn tt_probe(self: *Searcher, key: u64, depth: i32, ply: u16, alpha: Value, beta: Value, tt_move: *Move) ?Value {
        if (comptime lib.is_paranoid) {
            assert(depth >= 0);
            assert(tt_move.* == Move.empty);
        }
        return self.transpositiontable.probe(key, @intCast(depth), ply, alpha, beta, tt_move);
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

    /// Not used.
    fn try_set_pv_from_tt(self: *Searcher, node: *Node) void {
        // TODO: we can probably do some more optimizations here. Build in pondering etc. anyway. We are not there yet...
        // TODO: can we actually do something with the previous PV during a game?
        var pos: Position = self.position_stack[0]; // temp copy
        var i: u8 = 0;
        while (true) {
            const entry = self.transpositiontable.get(pos.key);
            if (entry.key != pos.key or entry.bound != .Exact) break;
            const move = entry.move;
            // Quick check.
            if (pos.board[move.from.u].is_empty()) break;
            var any: position.MoveFinder = .init(move.from, move.to, move.flags);
            pos.lazy_generate_moves(&any);
            if (!any.found()) break;
            pos.lazy_do_move(any.move);
            node.pv.append_assume_capacity(any.move);
            i += 1;
            if (i > 127) break;
        }
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
                    self.stopped = true;
                }
            },
            else => {
                return false;
            },
        }
        return self.stopped;
    }

    /// Buggy shit, this mate in X.
    fn mate_in(score: Value) Value {
        return if (score > 0) @divFloor(mate - score + 1, 2) else @divFloor(-mate - score, 2);
        //     static int MateIn(int evaluation) {
        //   if (evaluation > 0 && evaluation <= kMateScore) {  // Mate in favor
        //     return (kMateScore - evaluation + 1) / 2;
        //   } else if (evaluation < 0 && evaluation >= -kMateScore) {  // Mate against
        //     return -(kMateScore + evaluation) / 2;
        //   }
        //   // Not a mate score
        //   return evaluation;
        // }
    }
};

pub const Node = struct {
    /// Local PV during search.
    pv: PV = .{},
    /// Current result.
    score: Value = 0,
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
    /// The generated move. 16 bits.
    move: Move,
    /// The score for move ordering only. 32 bits.
    score: i32,
    /// Set during processing. 4 bits.
    moved_piece: Piece,
    /// Set during processing. 1 bit.
    is_bad_capture: bool,
    /// Set during processing. 1 bit.
    is_killer: bool,

    pub const empty: ExtMove = .{
        .move = .empty,
        .score = 0,
        .moved_piece = Piece.NO_PIECE,
        .is_capture = false,
        .is_killer = false,
    };
};

const MovePicker = struct {
    extmoves: [max_move_count]ExtMove,
    count: u8,

    const empty: MovePicker = .{ .extmoves = undefined, .count = 0 };

    /// Required function.
    pub fn reset(self: *MovePicker) void {
        // We only are allowed to use this once.
        if (comptime lib.is_paranoid) assert(self.count == 0);
    }

    /// Required function.
    pub fn store(self: *MovePicker, move: Move) ?void {
        self.extmoves[self.count] = ExtMove{ .move = move, .score = 0, .moved_piece = Piece.NO_PIECE, .is_bad_capture = false, .is_killer = false };
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
    fn process_and_score_moves(self: *MovePicker, searcher: *const Searcher, pos: *const Position, tt_move: Move, killers: [2]Move) void {



        const hash         : Value = 8_000_000;
        const promotion    : Value = 4_000_000;
        const capture      : Value = 2_000_000;
        const killer1      : Value = 1_000_000;
        const killer2      : Value =   900_000;

        var score: Value = undefined;

        for (self.slice()) |*ex| {
            score = 0;
            const m: Move = ex.move;
            const pc: Piece = pos.board[m.from.u];
            ex.moved_piece = pc;

            switch (m.flags) {
                Move.silent, Move.double_push, Move.castle_short, Move.castle_long => {
                    score += searcher.history_heuristics.piece_to[pc.u][m.to.u];
                    if (m == killers[0]) {
                        score += killer1;
                        ex.is_killer = true;
                    }
                    else if (m == killers[1]) {
                        score += killer2;
                        ex.is_killer = true;
                    }
                },
                Move.capture => {
                    const see = hce.see_score(pos, m);
                    score += capture + see;
                    ex.is_bad_capture = see < 100;
                },
                Move.ep => {
                    score += capture;
                },
                Move.knight_promotion, Move.bishop_promotion, Move.rook_promotion, Move.queen_promotion => {
                    score += promotion + (m.promoted_to().value() * 10);
                },
                Move.knight_promotion_capture, Move.bishop_promotion_capture, Move.rook_promotion_capture, Move.queen_promotion_capture => {
                    const capt: Piece = pos.board[m.to.u];
                    score += promotion + (m.promoted_to().value() * 10);
                    score += capt.value();
                },
                else => {
                    unreachable;
                },
            }

            // Hash move.
            if (m == tt_move) {
                score += hash;
            }
            ex.score = score;
        }
    }

    /// Get the next best scoring move, putting it at current_idx.
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

    /// The slice of all the moves.
    fn slice(self: *MovePicker) []ExtMove {
        return self.extmoves[0..self.count];
    }

    /// Not used.
    fn sort(self: *MovePicker) void {
        // std.mem.sort(ExtMove, self.slice(), {}, less_than);
        std.mem.sortUnstable(ExtMove, self.extmoves[0..self.count], {}, less_than);
    }

    fn less_than(_: void, a: ExtMove, b: ExtMove) bool {
        return a.score > b.score;
    }

    // Not in release mode.
    fn debugprint(self: *const MovePicker, iteration: u8) void {
        io.debugprint("\n#{}: ", .{ iteration });
        for (self.extmoves[0..self.count], 0..) |e, i| {
           io.debugprint("{f} {}; ", .{ e.move, e.score });
           _ = i;
        }
        io.debugprint("\n\n", .{});
    }
};

const History = struct {
    // Modern programs use a more sophisticated formula for history updates, namely the history gravity formula:
    // void update(Color side2move, Square from, Square to, int bonus)
    // {
    //     int clampedBonus = clamp(bonus, -MAX_HISTORY, MAX_HISTORY);
    //     history[side2move][from][to]
    //         += clampedBonus - history[side2move][from][to] * abs(clampedBonus) / MAX_HISTORY;
    // }

    const CORRECTION_HISTORY_SIZE: usize = 16384;
    const MAX_CORRECTION_HISTORY: Value = 16384;
    const CORRECTION_HISTORY_GRAIN: Value = 256;
    const CORRECTION_HISTORY_WEIGHT_SCALE: Value = 1024;

    const CorrectionTable = [2][CORRECTION_HISTORY_SIZE]Value;

    /// Used for quiet moves. Indexing: [piece][to-square]
    piece_to: [12][64]Value,
    // Used for capture moves. Indexing: [piecetype][captured-piecetype][to-square]
    //capture_to: [6][6][64]Value,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][pawnkey-index]
    pawn_correction: CorrectionTable,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][minorkey-index]
    minor_correction: CorrectionTable,
    /// Small hash. Used for correction of static evaluation. Indexing: [color][majorkey-index]
    major_correction: CorrectionTable,

    fn init() History {
        return .{
           .piece_to = std.mem.zeroes([12][64]Value),
           .pawn_correction = std.mem.zeroes(CorrectionTable),
           .minor_correction = std.mem.zeroes(CorrectionTable),
           .major_correction = std.mem.zeroes(CorrectionTable),
        };
    }

    fn clear(self: *History) void {
        self.piece_to = std.mem.zeroes([12][64]Value);
        self.pawn_correction = std.mem.zeroes(CorrectionTable);
        self.minor_correction = std.mem.zeroes(CorrectionTable);
        self.major_correction = std.mem.zeroes(CorrectionTable);
    }

    fn decay(self: *History) void {
        const to_hist: []Value = @ptrCast(&self.piece_to);
        for (to_hist) |*v| {
            //v.* = @divTrunc(v.*, 2);
            v.* >>= 1;
        }

        self.pawn_correction = std.mem.zeroes(CorrectionTable);
        self.minor_correction = std.mem.zeroes(CorrectionTable);
        self.major_correction = std.mem.zeroes(CorrectionTable);
    }

    fn update_quiet_score(self: *History, pos: *const Position, move: Move, depth: i32) void {
        const bonus: i32 = depth * depth;
        // In an extreme case we could have an invalid move.
        const pc: Piece = pos.board[move.from.u];
        if (pc.is_empty()) return;
        const v: *Value = &self.piece_to[pc.u][move.to.u];
        v.* = std.math.clamp(v.* + bonus, -16384, 16384);
    }

    fn record_quiet_beta_cutoff(self: *History, node: *Node, ex: ExtMove, depth: i32, movepicker: *const MovePicker, move_idx: usize) void {
        if (comptime lib.is_paranoid) {
            assert(ex.move.is_quiet());
        }

        const bonus: i32 = depth * depth;
        const move: Move = ex.move;

        // Increase bonus for this move.
        const v: *Value = &self.piece_to[ex.moved_piece.u][move.to.u];
        v.* = std.math.clamp(v.* + bonus, -16384, 16384);

        // Update killers of this node.
        if (node.killers[0] != move) {
            node.killers[1] = node.killers[0];
            node.killers[0] = move;
        }

        // Decrease bonus of previous quiet moves. These did not cause a beta cutoff.
        if (move_idx > 0) {
            for (movepicker.extmoves[0..move_idx]) |prev| {
                if (comptime lib.is_paranoid) {
                    assert(prev.move != ex.move);
                }
                if (prev.move.is_quiet()) {
                    const p: *Value = &self.piece_to[prev.moved_piece.u][prev.move.to.u];
                    p.* = std.math.clamp(p.* - bonus, -16384, 16384); // ???
                }
            }
        }
    }

    fn update_correction_history(self: *History, pos: *const Position, static_eval: Value, score: Value, depth: Value) void {
        const u: u1 = pos.stm.u;
        const err : Value = (score - static_eval) * CORRECTION_HISTORY_GRAIN;
        const weight: Value = @min(depth * depth + 2 * depth + 1, 128);
        set_correction(&self.pawn_correction[u][pos.pawnkey % CORRECTION_HISTORY_SIZE], err, weight);
        set_correction(&self.minor_correction[u][pos.minorkey % CORRECTION_HISTORY_SIZE], err, weight);
        set_correction(&self.major_correction[u][pos.majorkey % CORRECTION_HISTORY_SIZE], err, weight);
    }

    fn set_correction(entry: *Value, err: Value, weight: Value) void {
        const interpolated = (entry.* * (CORRECTION_HISTORY_WEIGHT_SCALE - weight) + err * weight) >> 10;
        const clamped = std.math.clamp(interpolated, -MAX_CORRECTION_HISTORY, MAX_CORRECTION_HISTORY);
        entry.* = clamped;
    }

    fn get_correction(self: *History, pos: *const Position) Value {
        const u: u1 = pos.stm.u;
        var corr_eval: Value = 0;
        corr_eval += self.pawn_correction[u][pos.pawnkey % CORRECTION_HISTORY_SIZE] * 2;
        corr_eval += self.minor_correction[u][pos.minorkey % CORRECTION_HISTORY_SIZE] * 2;
        corr_eval += self.major_correction[u][pos.majorkey % CORRECTION_HISTORY_SIZE] * 2;
        corr_eval >>= 9;
        // High tech callibrated with one single position :) finding Rxc3
        // position fen 2rqk2r/1p2bppp/pn1pbn2/4p3/4P3/P1N1B1Q1/1PPNBPPP/1KR1R3 b k - 11 17
        corr_eval = std.math.clamp(corr_eval, -60, 60);
        return corr_eval;
    }
};

/// The available options.
pub const Options = struct {
    /// In megabytes.
    hash_size: u64 = default_hash_size,

    pub const default: Options = .{};

    pub const default_hash_size: u64 = 64;
    pub const min_hash_size: u64 = 16;
    pub const max_hash_size: u64 = 1024;

    //pub const default_use_hash: bool = true;

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
    pub fn init(go_params: *const uci.Go, us: Color) TimeParams {
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
            const time: u32 = if (us.e == .white) result.time_left[Color.WHITE.u] else result.time_left[Color.BLACK.u];
            const inc: u32 = if (us.e == .white) result.increment[Color.WHITE.u] else result.increment[Color.BLACK.u];

            // Do not flag on the last move before time control.
            result.max_movetime = (time / moves) + if (moves > 1 and inc > 0) (inc / 2) else 0;
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

const SearchMode = enum {
    pv,
    nonpv,
};

/// Not used (yet).
const Error = error {
    EngineIsStillRunning,
};

/// Bigger depth, further in movelist -> bigger search reduction.
pub const lmr_depth_reduction_table: [max_search_depth][max_move_count]Value = compute_reduction_table();

fn compute_reduction_table() [max_search_depth][max_move_count]Value {
    @setEvalBranchQuota(32000);
    var result: [max_search_depth][max_move_count]Value = std.mem.zeroes([max_search_depth][max_move_count]Value);
    for (1..max_search_depth) |depth| {
        for (1..max_move_count) |move| {
            const d: f32 = @floatFromInt(depth);
            const m: f32 = @floatFromInt(move);
            const v = 1.0 + @log(d) * @log(m) * 0.5;
            result[depth][move] = @intFromFloat(@floor(v));
        }
    }
    return result;
}







// EXPERIMENTAL
// fn allow_late_move_pruning(depth: i32, iteration: i32, move_idx: usize, is_quiet: bool, is_root: bool, is_check: bool) bool {

//     const lmp_threshold_table = [_]u8{ 0, 2, 4, 8, 16, 32, 48, 64 }; // move index limits per depth

//     if (is_root or is_check or !is_quiet) return false;   // never prune these
//     if (depth < 4) return false;   // 3                        // depth too shallow
//     if (iteration < 5) return false; // 4                      // move ordering unstable
//     const capped_depth = @min(@abs(depth), lmp_threshold_table.len - 1);
//     return move_idx > lmp_threshold_table[capped_depth];  // prune late quiets
// }
