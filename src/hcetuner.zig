// zig fmt: off

const std = @import("std");
const lib = @import("lib.zig");
const utils = @import("utils.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");
const hceterms = @import("hceterms.zig");
const viri = @import("viri.zig");

const Color = types.Color;
const PieceType = types.PieceType;
const Square = types.Square;
const Move = types.Move;
const ScorePair = types.ScorePair;
const Position = position.Position;
const Evaluator = hce.Evaluator;
const Terms = hceterms.Terms;

const src_folder = "C:/Data/chess/lichess/datasets/";
const scorepair_count: usize = @sizeOf(hceterms.Terms) / @sizeOf(ScorePair);
const meta: Meta = compute_scorepair_metadata();

// begin tuning
//      begin read dataset    -> random circular read
//           begin batch
//              begin pos
//              end pos        -> adjust used terms
//           end batch         -> update active terms
//      end read dataset
// end tuning                  -> select best data

/// One learning term.
const BatchTerm = struct {
    old: ScorePair,
    delta: ScorePair,
};

/// A (mutable) flat view on the hceterms.
const terms_view: *[scorepair_count]ScorePair = @constCast(@ptrCast(hceterms.terms));

var rnd: utils.Random = .init(1);
var curr_pos: Position = .empty;
var curr_best_move: Move = .empty;
/// Running sum for the current position.
var curr_sp: ScorePair = .empty;
/// Term usage count for one eval. Range is small: -64...64 max. White usage ++ black usage --.
var curr_usage: [scorepair_count]i32 = @splat(0);
/// For gathering the active indexes before updating.
var curr_active_indexes: utils.BoundedArray(u16, scorepair_count) = .empty;
/// Batch learning data.
var batch_data: [scorepair_count]BatchTerm = undefined;

const batch_size: usize = 9973;

/// Evaluator calls this for each used scorepair.
pub fn register_scorepair_usage(sp: *ScorePair, multiply: u8, us: Color, debugargs: anytype) void {
    lib.only_when_tuning();
    handle_used_term(sp, multiply, us, debugargs);
}

/// Evaluator calls this just before returning its result.
pub fn register_hce_eval_result(hce_result: ScorePair) void {
    lib.only_when_tuning();
    // Here we can assert that hce_result == curr_sp.
    _ = hce_result;
}

/// Read dataset and tune the terms.
pub fn run() !void {
    lib.only_when_tuning();
    begin_tuning();

    var file_nr: usize = 0; _ = &file_nr;
    var filename = compute_viri_filename(file_nr);
    var reader: viri.ViriFileReader = try .init(filename.slice(), 4096);
    defer(reader).deinit();
    var game: viri.ViriGame = .init();
    defer game.deinit();

    var evaluator: Evaluator = .init();
    var total_games_processed: usize = 0;

    // Read one file.
    game_loop: while (try reader.next(&game)) {
        // Evaluate one position.
        begin_eval();
        curr_pos = try game.startpos.to_position();
        const chessnix_eval: i32 = evaluator.evaluate(&curr_pos);
        const dataset_eval: i32 = game.moves.items[0].eval; // Note: the lichessdataset has just 1 move. the eval of game.startpos equals the eval of the first move.
        curr_best_move = game.moves.items[0].move.to_move(&curr_pos);
        std.debug.assert(dataset_eval == game.startpos.eval);
        end_eval(chessnix_eval, dataset_eval);
        total_games_processed += 1;
        if (total_games_processed >= 10) {
            break :game_loop;
        }
    }

    // // Done?
    // file_nr += 1;
    // if (file_nr > 99) {
    //     break :epoch_loop;
    // }
    // // Open next file.
    // filename = compute_viri_filename(file_nr);
    // reader.deinit();
    // reader = try .init(filename.slice(), 4096);
    //try Printer.print_terms();
    // try end_tuning();
    // lib.io.debugprint("fields {}\n", .{ MetaData.terms_field_count });
    // for (MetaData.terms_field_names) |ar| {
    //     lib.io.debugprint("{s}\n", .{ ar.slice() });
    // }
}

fn begin_tuning() void {
    for (&batch_data, 0..) |*bt, i| {
        bt.old = terms_view[i];
        bt.delta = .empty;
    }
}

fn end_tuning() !void {
    var writer: utils.FileWriter = try .init("C:/Data/Tmp/eval.bin", 4096);
    defer writer.deinit();
    try writer.wr.interface.writeSliceEndian(ScorePair, &terms_view.*, .little);
}

fn begin_batch() void {
    // ...
}

/// Adjust the hceterms for new evals.
fn end_batch() void {
    for (&batch_data, terms_view) |*bt, *tv| {
        const new: ScorePair = bt.old.add(bt.delta);
        tv = new;
        bt = new;
        bt.delta = .empty;
    }
}

/// Clear the current eval data for 1 position.
fn begin_eval() void {
    curr_pos = .empty;
    curr_usage = @splat(0);
    curr_sp = .empty;
}

fn handle_used_term(sp: *ScorePair, multiply: u8, us: Color, debugargs: anytype) void {
    _ = debugargs;

    if (multiply == 0) {
        return;
    }

    const sp_index: usize = index_of_scorepair(sp);

    // Update running sum and usage.
    switch (us.e) {
        .white => {
            curr_usage[sp_index] += multiply;
            curr_sp.inc(sp.*.mul(multiply));
        },
        .black => {
            curr_usage[sp_index] -= multiply;
            curr_sp.dec(sp.*.mul(multiply));
        },
    }

    // const sp_info: *const MetaData.ScorePairInfo = metadata.get_scorepair_info(sp_index);
    // if (funcs.eql(sp_info.name.slice(), "knight_outpost_table")) {
    //    lib.io.debugprint("A OUTPOST KNIGHT {any}\n", .{ debugargs });
    // }

    // lib.io.debugprint(
    //     "usage sp ({:>4},{:>4}), times:{:>2}, {s} ({}), ({t}), args: {any}, ",
    //     .{ sp.mg, sp.eg, multiply, sp_info.name.slice(), sp_info.array_index, us.e, debugargs }
    // );
    // lib.io.debugprint(
    //     "curr_usage: {} curr_sp {any}\n",
    //     .{ curr_usage[sp_index], curr_sp }
    // );
}

/// When current eval for 1 position is ready.
fn end_eval(chessnix_eval: i32, dataset_eval: i32) void {
    const err: i32 = dataset_eval - chessnix_eval;
    if (err == 0) {
        return;
    }

    // Loop through the used features and gather the indexes.
    curr_active_indexes.len = 0;
    for (curr_usage, 0..) |curr, sp_index| {
        if (curr == 0) {
            continue;
        }
        curr_active_indexes.append_assume_capacity(@intCast(sp_index)); // u16
    }

    const used: usize = curr_active_indexes.len;
    const phase: u8 = types.restrict_phase(curr_pos.phase());
    //lib.io.debugprint("pos phase {}, curr_sp {f}, err {}, chessnix_eval {} dataset_eval {}, used terms {} {f}\n", .{ phase, curr_sp, err, chessnix_eval, dataset_eval, used, curr_pos });
    lib.io.debugprint("pos phase {}, curr_sp {f}, err {}, chessnix_eval {} dataset_eval {}, used terms {}\n", .{ phase, curr_sp, err, chessnix_eval, dataset_eval, used });
}

/// Get the correct filename inside the folder.
fn compute_viri_filename(file_nr: usize) utils.BoundedArray(u8, 256) {
    return funcs.get_str("{s}{:0>4}.vf", .{ src_folder, file_nr });
}

// --- Metadata of terms ---

/// Comptime only.
fn compute_scorepair_metadata() Meta {
    @setEvalBranchQuota(32000);
    var m: Meta = .{ .info = @splat(.{})};
    for (0..scorepair_count) |i| {
        m.info[i] = compute_scorepair_info(i);
    }
    return m;
}

/// Comptime only.
fn compute_scorepair_info(index: usize) Meta.ScorePairInfo {
    const offset: usize = index * @sizeOf(ScorePair);
    const structinfo = @typeInfo(hceterms.Terms).@"struct";
    inline for (structinfo.fields) |field| {
        const field_offset = @offsetOf(hceterms.Terms, field.name);
        const size = @sizeOf(field.type);
        const len: usize = size / @sizeOf(ScorePair);
        if (offset >= field_offset and offset < field_offset + size) {
            var result: Meta.ScorePairInfo = .{};
            result.raw_index = index;
            result.raw_array_len = len;
            result.raw_array_index = (offset - field_offset) / @sizeOf(ScorePair);
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

/// OMGF
const Meta = struct {
    info: [scorepair_count]ScorePairInfo,

    const zig_terms_struct_info = @typeInfo(hceterms.Terms).@"struct";
    const sizeof_scorepair: usize = @sizeOf(ScorePair);
    const terms_field_count: usize = zig_terms_struct_info.fields.len;

    const terms_field_names: [terms_field_count]utils.BoundedArray(u8, 48) = blk: {
        var names: [terms_field_count]utils.BoundedArray(u8, 48) = @splat(.empty);
        for (zig_terms_struct_info.fields, 0..) |field, i| {
            names[i].append_slice_assume_capacity(field.name);
        }
        break :blk names;
    };

    //const arrays: [

    const ArrayInfo = struct {
        dimensions: utils.BoundedArray(u8, u32),
    };

    const ScorePairInfo = struct {
        raw_index: usize = 0,
        raw_array_len: usize = 0,
        raw_array_index: usize = 0,
        name: utils.BoundedArray(u8, 64) = .empty, // TODO: this is an insane waste of space.
    };

    fn get_scorepair_info(self: *const Meta, idx: usize) *const ScorePairInfo {
        return &self.info[idx];
    }

    // fn compute_field_names() [terms_field_count]utils.BoundedArray(u8, 64) {
    //     var names: [terms_field_count]utils.BoundedArray(u8, 64) = @splat(.empty);
    //     inline for (zig_terms_struct_info.fields, 0..) |field, i| {
    //         names[i].append_slice_assume_capacity(field.name);
    //     }
    //     return names;
    // }
};

// --- Stupid printing ---

const Printer = struct {

    fn print_terms() !void {
        try print_single_array(0, terms_view, lib.io.out);
    }

    fn print_single_array(level: u8, array: []const ScorePair, writer: *std.io.Writer) std.io.Writer.Error!void {
        _ = level;
        //const len: usize = array.len;
        for (array, 0..) |sp, i| {
            const last_of_line: bool = i > 0 and i % 8 == 0;
            try writer.print("pair({}, {}),{s}", .{ sp.mg, sp.eg, if (last_of_line) "" else " " });
            if (last_of_line) {
                try writer.print("\n", .{});
            }
        }
        try writer.print("\n", .{});
    }
};