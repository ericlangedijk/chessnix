// zig fmt: off

//! --- NOT USED YET ---

const std = @import("std");

const lib = @import("lib.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");

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

pub const SearchManager = struct
{
    num_threads: usize,
    searchers: std.ArrayListUnmanaged(Search),
    //threads: std.ArrayListUnmanaged(std.Thread),

    pub fn init(num_threads: usize) SearchManager
    {
        return
        .{
            .num_threads = num_threads,
            .searchers = create_searchers(num_threads),
        };
    }

    pub fn deinit(self: *SearchManager) void
    {
        self.searchers.deinit(ctx.galloc);
    }

    fn create_searchers(num_threads: usize) std.ArrayListUnmanaged(Search)
    {
        var list: std.ArrayListUnmanaged(Search) = std.ArrayListUnmanaged(Search).initCapacity(ctx.galloc, num_threads) catch wtf();
        for (0..num_threads) |_|
        {
            list.appendAssumeCapacity(Search.init());
        }
        return list;
    }

    pub fn start(self: *SearchManager) !void
    {
        // we should have some atomic value to check if the threads are not already running.
        // then start them.
        // std.Thread.spawn
        // std.Thread.detach
        _ = self;
        try lib.io.print("WE SHOULD START THINKING\n", .{});
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
    pos: Position,
    stack: Stack,
    pv_node: Node,
    nodes: u64,
    iteration: u8,

    pub fn init() Search
    {
        return Search
        {
            .pos = .empty, //.create(),
            .stack = .init(),
            .pv_node = .init(),
            .nodes = 0,
            .iteration = 0,
        };
    }

    pub fn start(self: *Search) void
    {
       switch (self.pos.to_move.e)
        {
            .white => self.iterative_deepening(Color.WHITE),
            .black => self.iterative_deepening(Color.BLACK),
        }
    }

    pub fn stop(self: *Search) void
    {
        _ = self;
    }

    fn iterative_deepening(self: *Search, comptime us: Color) void
    {
        // TODO: generate rootmoves.

        self.iteration = 1;
        var depth: u8 = 1;
        while (true)
        {
            const mode: Mode = .{ .is_root = true };
            const alpha: Value = self.alpha_beta(mode, us, depth, -types.infinity, types.infinity);
            lib.out.print("alpha {}\n", .{ alpha });
            // TODO: lib.out print info PV here.
            if (self.iteration == 3) break;
            depth += 1;
        }
    }

    fn alpha_beta(self: *Search, comptime mode: Mode, comptime us: Color, depth: u8, input_alpha: Value, beta: Value) Value
    {
        const them = comptime us.opp();
        const pos: *Position = &self.pos;
        var alpha: Value = input_alpha;
        var eval: Value = 0;
        if (depth == 0) return self.quiet(mode, us, depth, input_alpha, beta);
        const node: *Node = self.stack.get_node(pos.ply + 2);
        const childnode: *const Node = Stack.get_child_node(node);
        const parentnode: *const Node = Stack.get_parent_node(node);
        _ = parentnode;
        // Clear current node.
        node.pv.len = 0;
        node.eval = 0;
        if (pos.state().rule50 >= 100) return 0;

        // Generate the moves (using rootmoves if at _root).
        var movepicker = MovePicker.init();
        movepicker.generate_moves(pos, us);
        const any_moves: bool = movepicker.len() > 0;
        if (any_moves)
        {
            for (movepicker.slice()) |extmove|
            {
                var st: StateInfo = undefined;
                const move: Move = extmove.move;
                self.pos.make_move(us, &st, move);
                eval = -self.alpha_beta(mode, them, depth - 1, -beta, -alpha);
                self.pos.unmake_move(us);

                // Higher alpha. A new best move is found.
                if (eval > alpha)
                {
                    alpha = eval;
                    update_pv(move, eval, node, childnode);
                }
            }
        }
        else
        {
            if (pos.in_check()) return -types.mate + pos.ply else return types.stalemate;
        }
        return alpha;
    }

    fn quiet(self: *Search, comptime mode: Mode, comptime us: Color, depth: u8, input_alpha: Value, beta: Value) Value
    {
        _ = self;
        _ = mode;
        _ = us;
        _ = depth;
        _ = input_alpha;
        _ = beta;
        // TODO: generate captures / promotions.
        return 0;
    }

    /// Copy line from 'childnode' to 'current node'
    fn update_pv(bestmove: Move, eval: i16, node: *Node, childnode: *const Node) void
    {
        node.pv.len = 0;
        node.pv.appendAssumeCapacity(bestmove);
        node.pv.appendSliceAssumeCapacity(childnode.pv.slice());
        node.eval = eval;
    }
};

/// Each Search has its own stack.
pub const Stack = struct
{
    movepicker: MovePicker,
    nodes: Nodes,

    fn init() Stack
    {
        return Stack
        {
            .movepicker = .init(),
            .nodes = .{},
        };
    }

    fn get_node(self: *Stack, index: u8) *Node
    {
        return &self.nodes.items[index];
    }

    /// This assumes the node is in the stack.
    fn get_child_node(node: *Node) *const Node
    {
        return funcs.ptr_add(Node, node);
    }

    /// This assumes the node is in the stack.
    fn get_parent_node(node: *Node) *const Node
    {
        return funcs.ptr_sub(Node, node);
    }
};

const Mode = packed struct
{
    is_root: bool,
};

const Node = struct
{
    pv: PV = .{},
    eval: Value = 0,
    is_check: bool = false,
    is_mate: bool = false,
    is_stalemate: bool = false,

    fn init() Node
    {
        return .{};
    }
};

const MovePicker = struct
{
    extmoves: [types.max_move_count]ExtMove,
    store_ptr: [*]ExtMove,
    //current: [*]ExtMove,

    fn init() MovePicker
    {
        var result: MovePicker = undefined;
        result.store_ptr = &result.extmoves;
        return result;
        //result.current_selected = null;
    }

    /// Required funnction.
    pub fn reset(self: *MovePicker) void
    {
        self.store_ptr = &self.extmoves;
        //self.index = null;
    }

    /// Required funnction.
    pub fn store(self: *MovePicker, move: Move) void
    {
        self.ptr[0] = ExtMove{ .move = move, .eval = 0 };
        self.store_ptr += 1;
    }

    pub fn len(self: *const MovePicker) usize
    {
        return self.store_ptr - &self.extmoves;
    }

    fn slice(self: *const MovePicker) []const Move
    {
        return self.extmoves[0..self.len()];
    }

    fn generate_moves(self: *MovePicker, pos: *const Position, comptime us: Color) void
    {
        pos.generate_moves(pos, us, &self);
    }

    fn extract_next(self: *MovePicker) ?ExtMove
    {
        _ = self;
    }
};

const ExtMove = struct
{
    move: Move,
    eval: Value,
};
