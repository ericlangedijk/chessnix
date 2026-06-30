// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const utils = @import("utils.zig");
const types = @import("types.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");
const hceterms = @import("hceterms.zig");
const viri = @import("viri.zig");

const Color = types.Color;
const PieceType = types.PieceType;
const Square = types.Square;
const ScorePair = types.ScorePair;
const Position = position.Position;
const Evaluator = hce.Evaluator;

const src_folder = "C:/Data/chess/lichess/datasets/";

const scorepair_count: usize = @sizeOf(hceterms.Terms) / @sizeOf(ScorePair);
const metadata: MetaData = compute_scorepair_metadata();

/// A (mutable) flat view on the terms.
const terms_view: *[scorepair_count]ScorePair = @constCast(@ptrCast(hceterms.terms));

/// Term usage for the current position.
/// For white the usage is increased, for black decreased. So terms can cancel eachother out.
/// The final values are always very small. Never exceeding the range -64...64 and probably even smaller than that.
var curr_usage: [scorepair_count]i32 = @splat(0);
/// the netto usage count.
var curr_term_balance: i32 = 0;
/// Running sum for the current position.
var curr_sp: ScorePair = .empty;

/// `hce.Evaluator` calls this function for each used scorepair.
pub fn register_scorepair_usage(sp: *ScorePair, multiply: u8, us: Color, debugargs: anytype) void {
    lib.only_when_tuning();
    _ = debugargs;
    if (multiply == 0) {
        return;
    }
    const sp_index: usize = index_of_scorepair(sp);
    //const sp_info: *const MetaData.ScorePairInfo = metadata.get_scorepair_info(sp_index);

    switch (us.e) {
        .white => {
            curr_usage[sp_index] += multiply;
            curr_term_balance += multiply;
            curr_sp.inc(sp.*.mul(multiply));
        },
        .black => {
            curr_usage[sp_index] -= multiply;
            curr_term_balance -= multiply;
            curr_sp.dec(sp.*.mul(multiply));
        },
    }

    // lib.io.debugprint(
    //     "usage sp ({:>4},{:>4}), times:{:>2}, {s} ({}), ({t}), args: {any}, ",
    //     .{ sp.mg, sp.eg, multiply, sp_info.name.slice(), sp_info.array_index, us.e, debugargs }
    // );
    // lib.io.debugprint(
    //     "curr_usage: {} curr_sp {any}\n",
    //     .{ curr_usage[sp_index], curr_sp }
    // );
}

/// Read dataset and tune the terms.
pub fn run() !void {
    lib.only_when_tuning();
    const gpa = lib.ctx.gpa;

    var file_nr: usize = 0;
    var filename: []u8 = compute_viri_filename(file_nr);
    const src = "C:/Data/chess/viri/pawnocchio/mix_1.vf";
    var viri_reader: viri.ViriFileReader = try .init(src, gpa);
    defer(viri_reader).deinit();
    var game: viri.ViriGame = .init();
    defer game.deinit(gpa);

    var pos: Position = .empty;
    var evaluator: Evaluator = .init();

    var games_processed: usize = 0;
    mainloop: while (true) {

        viriloop: while (try viri_reader.next(&game)) {
            pos = try game.startpos.to_position();
            const chessnix_eval: i32 = evaluator.evaluate(&pos);
            const dataset_eval: i32 = game.startpos.eval;
            const me: ?i16 = if (game.moves.items.len > 0) game.moves.items[0].eval else null;
            const wdl: types.WDL = .from_int(game.startpos.wdl);
            lib.io.debugprint("chessnix_eval {} dataset_eval {} wdl {t} firstmove eval {any} termbalance {} sum {any}\n", .{chessnix_eval, dataset_eval, wdl.e, me, curr_term_balance, curr_sp});
            //learn_one(&pos, chessnix_eval, dataset_eval);
            games_processed += 1;
            if (games_processed > 10) {
                break :viriloop;
            }
        }
        if (true) {
            break :mainloop;
        }

        file_nr += 1;
        if (file_nr > 99) {
            break :mainloop;
        }
        filename = compute_viri_filename(file_nr);
        viri_reader.deinit();
        viri_reader = try .init(filename, gpa);
    }
}

fn debug_run() !void {
    // begin_pos();
    // //const pos: Position = .classic_startpos;
    // //const pos: Position = try .init("r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1", false);
    // const pos: Position = try .init("2k5/3p4/4p3/8/8/8/2BP4/2K5 b - - 0 1", false);

    // // const pos: Position = .classic_startpos;
    // var evaluator: Evaluator = .init();
    // const chessnix_eval: i32 = evaluator.evaluate(&pos);
    // // lib.io.debugprint("e {} stm {}\n", .{ e, pos.stm.e });
    // // const chessnix_eval: i32 = if (pos.stm.e == .white) e else -e;
    // const dataset_eval: i32 = 0;
    // lib.io.debugprint("chessnix absolute eval {}\n", .{ chessnix_eval});
    // learn_one(&pos, chessnix_eval, dataset_eval);
}

fn begin_pos() void {
    curr_usage = @splat(0);
    curr_term_balance = 0;
    curr_sp = .empty;
}

fn learn_one(pos: *const Position, chessnix_eval: i32, dataset_eval: i32) void {
    //_ = chessnix_eval;
    _ = dataset_eval;

    // Get the netto usage of each term.

    var checksum: ScorePair = .empty; // assert these are equal.

    lib.io.debugprint("current used features\n", .{});
    lib.io.debugprint("SUM {any}\n", .{ curr_sp });

    for (curr_usage, 0..) |curr, sp_index| {
        if (curr != 0) {
            const sp: *ScorePair = &terms_view[sp_index];

            const multiply: u8 = if (curr > 0) @intCast(curr) else @intCast(-curr);

            if (curr > 0) checksum.inc(sp.*.mul(multiply)) else checksum.dec(sp.*.mul(multiply));

            const sp_info: *const MetaData.ScorePairInfo = metadata.get_scorepair_info(sp_index);
            const us: Color = if (curr < 0) .black else .white;
            lib.io.debugprint("usage sp ({:>4},{:>4}), {s} ({}), ", .{ sp.mg, sp.eg, sp_info.name.slice(), sp_info.array_index });
            lib.io.debugprint("curr_usage: {} ({t})\n", .{ curr, us.e });
        }
    }
    const e: i32 = types.phased_score(pos.phase(), checksum);
    lib.io.debugprint("global sum {any} used sum {any} chessnixeval {} checkit {}\n", .{ curr_sp, checksum, chessnix_eval, e });
}

// --- Helper funcs ---

var str_buf: [256]u8 = undefined;

fn to_str(comptime fmt: []const u8, args: anytype) []u8 {
    return std.fmt.bufPrint(&str_buf, fmt, args) catch lib.wtf("to_str", .{});
}

fn compute_viri_filename(file_nr: usize) []u8 {
    return to_str("{s}{:0>4}.vf", .{ src_folder, file_nr });
}

// --- Metadata ---

/// Comptime only.
fn compute_scorepair_metadata() MetaData {
    @setEvalBranchQuota(32000);
    var m: MetaData = .{ .info = @splat(.{})};
    for (0..scorepair_count) |i| {
        m.info[i] = compute_scorepair_info(i);
    }
    return m;
}

/// Comptime only.
fn compute_scorepair_info(index: usize) MetaData.ScorePairInfo {
    const offset: usize = index * @sizeOf(ScorePair);
    const structinfo = @typeInfo(hceterms.Terms).@"struct";
    inline for (structinfo.fields) |field| {
        const field_offset = @offsetOf(hceterms.Terms, field.name);
        const size = @sizeOf(field.type);
        const len: usize = size / @sizeOf(ScorePair);
        if (offset >= field_offset and offset < field_offset + size) {
            var result: MetaData.ScorePairInfo = .{};
            result.raw_index = index;
            result.array_len = len;
            result.array_index = (offset - field_offset) / @sizeOf(ScorePair);
            result.name.append_slice_assume_capacity(field.name);
            return result;
        }
    }
    unreachable;
}

/// Computes the index of one term.
fn index_of_scorepair(sp: *const ScorePair) usize {
    const base: usize = @intFromPtr(terms_view);
    const addr: usize = @intFromPtr(sp);
    if (addr >= base) {
        const idx: usize = (addr - base) / @sizeOf(ScorePair);
        if (idx < scorepair_count) {
            return idx;
        }
    }
    lib.wtf("scorepair index", .{});
}

/// Just contains info of each term.
const MetaData = struct {
    info: [scorepair_count]ScorePairInfo,

    const ScorePairInfo = struct {
        raw_index: usize = 0,
        array_len: usize = 0,
        array_index: usize = 0,
        name: utils.BoundedArray(u8, 64) = .empty,
    };

    fn get_scorepair_info(self: *const MetaData, idx: usize) *const ScorePairInfo {
        return &self.info[idx];
    }
};
