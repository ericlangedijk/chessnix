// zig fmt: off

///! Here should come - as a first attempt - a Stochistic Gradient Descent implementation.
///!
///! Work In Progress...
///
/// DON'T LOOK HERE.
/// DON'T LOOK HERE.
/// DON'T LOOK HERE.
///

const std = @import("std");
const lib = @import("lib.zig");
const utils = @import("utils.zig");
const types = @import("types.zig");
const funcs = @import("funcs.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");
const hceterms = @import("hceterms.zig");
const viri = @import("viri.zig");

//const float32 = funcs.float32;
const int = funcs.int;
const float = funcs.float;

const Color = types.Color;
const PieceType = types.PieceType;
const Square = types.Square;
const Move = types.Move;
const ScorePair = types.ScorePair;
const String = utils.String;
const Position = position.Position;
const Evaluator = hce.Evaluator;
const Terms = hceterms.Terms;

const src_folder = "C:/Data/chess/lichess/datasets/";
const scorepair_count: usize = @sizeOf(hceterms.Terms) / @sizeOf(ScorePair);
const meta: Meta = compute_scorepair_metadata();

/// A float version of ScorePair
const FloatScorePair = extern struct {
    mg: f32,
    eg: f32,

    const empty: FloatScorePair = .{ .mg = 0, .eg = 0 };

    fn init(mg: f32, eg: f32) FloatScorePair {
        return .{ .mg = mg, .eg = eg };
    }

    fn from_scorepair(sp: ScorePair) FloatScorePair {
        return .{ .mg = float(f32, sp.mg), .eg = float(f32, sp.eg) };
    }

    fn to_scorepair(self: FloatScorePair) ScorePair {
        return .{ .mg = int(i16, self.mg), .eg = int(i16, self.eg) };
    }

    fn inc(self: *FloatScorePair, delta: FloatScorePair) void {
        self.mg += delta.mg;
        self.eg += delta.eg;
    }

    fn mul(self: *FloatScorePair, m: f32) void {
        self.mg *= m;
        self.eg *= m;
    }

    // fn dec(self: *FloatScorePair, delta: FloatScorePair) void {
    //     self.mg -= delta.mg;
    //     self.eg -= delta.eg;
    // }

    fn add(self: FloatScorePair, delta: FloatScorePair) FloatScorePair {
        return .init(self.mg + delta.mg, self.eg + delta.eg);
    }

    // fn sub(self: FloatScorePair, delta: FloatScorePair) FloatScorePair {
    //     return .init(self.mg - delta.mg, self.eg - delta.eg);
    // }

    // fn mul(self: FloatScorePair, m: u8) FloatScorePair {
    //     return .{ .mg = self.mg * m, .eg = self.eg * m };
    // }

    pub fn format(self: FloatScorePair, writer: *std.io.Writer) std.io.Writer.Error!void {
        try writer.print("({d:.4}, {d:.4})", .{ self.mg, self.eg });
    }
};

const Status = struct {
    positions_seen: u64 = 0,

};

/// One learning term.
const TuningTerm = struct {
    src: FloatScorePair, // hce as float.
    delta: FloatScorePair, // current delta.
    samples: u64, // nr of usages

    //updates_done: u64,

    fn future(self: TuningTerm) ScorePair {
       return self.src.add(self.delta).to_scorepair();
    }
};

/// A (mutable) flat view on the hceterms.
const terms: *[scorepair_count]ScorePair = @constCast(@ptrCast(hceterms.terms));
var tuningterms: [scorepair_count]TuningTerm = undefined;
var dataset_filenames: std.ArrayList(String(16)) = .empty;
var rnd: utils.Random = .init(1);


var status: Status = .{};

var curr_pos: Position = .empty;
var curr_best_move: Move = .empty;
/// Running sum for the current position.
var curr_sp: ScorePair = .empty;
/// Term usage count for one eval. Range is small: -64...64 max. White usage ++ black usage --.
var curr_usage: [scorepair_count]i32 = @splat(0);
/// For gathering the active indexes before updating.
var curr_active_indexes: utils.BoundedArray(u16, scorepair_count) = .empty;

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

// --- Tuning ---

/// Read dataset and tune the terms.
pub fn run() !void {
    lib.only_when_tuning();
    try initialize();
    defer finalize();

    var file_nr: usize = 0; _ = &file_nr;
    var filename = compute_viri_filename(file_nr);
    var reader: viri.ViriFileReader = try .init(filename.slice(), 4096);
    defer(reader).deinit();
    var game: viri.ViriGame = .init();
    defer game.deinit();

    var evaluator: Evaluator = .init();
    //var total_games_processed: usize = 0;

    // Read one file.
    game_loop: while (try reader.next(&game)) {
        // Evaluate one position.
        begin_eval();
        curr_pos = try game.startpos.to_position();
        const chessnix_eval: i32 = evaluator.evaluate(&curr_pos, .unscaled);
        const dataset_eval: i32 = game.startpos.eval;
        //curr_best_move = game.moves.items[0].move.to_move(&curr_pos);
        //std.debug.assert(dataset_eval == game.startpos.eval);
        end_eval(chessnix_eval, dataset_eval);
        status.positions_seen += 1;
        if (status.positions_seen >= 100000) {
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

fn initialize()! void {
    try get_dataset_filenames();

    //terms[0] = .init (300, -200);//empty; // #test
    for (terms) |*sp| sp.* = .empty;

    for (&tuningterms, terms) |*bt, sp| {
        bt.src = .from_scorepair(sp);
        bt.delta = .empty;
        bt.samples = 0;
    }
}

fn finalize() void {
    dataset_filenames.deinit(lib.ctx.gpa);
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

/// When eval for the currentposition is ready, update the tuningterms.
fn end_eval(chessnix_eval: i32, dataset_eval: i32) void {
    const LR: f32 = 0.1;  // learning rate.
    //const ST: u32 = 10; // sample threshold.

    const err: i32 = dataset_eval - chessnix_eval;
    if (err >= -1 and err <= 1) {
        return;
    }

    // Loop through the used features during eval and gather the indexes.
    curr_active_indexes.len = 0;
    for (curr_usage, 0..) |curr, sp_index| {
        if (curr == 0) {
            continue;
        }
        curr_active_indexes.append_assume_capacity(@intCast(sp_index)); // u16
    }

    const used: usize = curr_active_indexes.len;
    //const phase: u8 = types.restrict_phase(curr_pos.phase());
    //lib.io.debugprint("pos phase {}, curr_sp {f}, err {}, chessnix_eval {} dataset_eval {}, used terms {} {f}\n", .{ phase, curr_sp, err, chessnix_eval, dataset_eval, used, curr_pos });
    //lib.io.debugprint("pos phase {}, curr_sp {f}, err {}, chessnix_eval {} dataset_eval {}, used terms {}\n", .{ phase, curr_sp, err, chessnix_eval, dataset_eval, used });

    //const abs_err: u32 = @abs(err);
    // Spread out the error over the terms.
    const err_avg: f32 = float(f32, err) / float(f32, used);
    var delta: FloatScorePair = compute_delta(curr_pos.phase(), err_avg);
    delta.mul(LR);

    if (status.positions_seen % 100 == 0) lib.io.debugprint("pos phase {}, used: {}, err {d:.4}, avg_err {d:.4}, delta {f}\n", .{ curr_pos.phase(), used, err, err_avg, delta });
    // if (status.positions_seen % 100 == 0) {
    //      lib.io.debugprint("{f}\n", .{ &curr_pos });
    // }

    for (curr_active_indexes.slice()) |sp_index| {
        const tt: *TuningTerm = &tuningterms[sp_index];
        tt.delta.inc(delta);
        tt.samples += 1;

        //const inf: *const Meta.ScorePairInfo = meta.get_scorepair_info(sp_index); //_ = inf;
        //if (sp_index == 0)
           //lib.io.debugprint("    samples {}, {s}({}) active {f} delta {f} future {f}\n", .{ tt.samples, inf.name.slice(), inf.raw_array_index, tt.src, tt.delta, tt.src.add(delta)});

        // // Update the hceterm.
//        if (tt.samples >= 1) {
            //const hceterm: *ScorePair = &terms[sp_index];
            //hceterm.inc(.init(0, 0));
            //if ((tt.delta.mg >= 1.0 or tt.delta.mg <= -1.0 or tt.delta.eg >= 1.0 and tt.delta.eg <= -1.0) and tt.samples >= ST) {
            if (tt.delta.mg >= 1.0 or tt.delta.mg <= -1.0 or tt.delta.eg >= 1.0 and tt.delta.eg <= -1.0) {
                var future: ScorePair = tt.future();
                future.mg = std.math.clamp(future.mg, -1000, 1000);
                future.eg = std.math.clamp(future.eg, -1000, 1000);
                const hceterm: *ScorePair = &terms[sp_index];
                //const prev: ScorePair = hceterm.*;
                hceterm.* = future;
                //if (sp_index == 0) lib.io.debugprint("    updated {s}({}) from {f} to {f}\n", .{ inf.name.slice(), inf.raw_array_index, prev, hceterm.* });
                tt.src = .from_scorepair(future);
                tt.delta = .empty;
                tt.samples = 0;
            }
  //      }



    }
}

fn compute_delta(pos_phase: u8, err: f32) FloatScorePair {
    const phase: f32 = float(f32, types.restrict_phase(pos_phase)); // max phase == 24
    const max_phase: f32 = float(f32, types.max_phase);
    const relative_phase = phase / max_phase;
    const mg_delta: f32 = err * relative_phase;
    const eg_delta: f32 = err * (1.0 - relative_phase);
    return .init(std.math.clamp(mg_delta, -100.0, 100.0), std.math.clamp(eg_delta, -100.0, 100.0));
}

// --- Misc ---

/// Get the correct filename inside the folder. TODO: remove. we gather all files in folder.
fn compute_viri_filename(file_nr: usize) utils.BoundedArray(u8, 256) {
    return funcs.get_str("{s}{:0>4}.vf", .{ src_folder, file_nr });
}

pub fn get_dataset_filenames() !void {
    var dir: std.fs.Dir = try std.fs.openDirAbsolute(src_folder, .{ .access_sub_paths = false, .iterate = true, .no_follow = true });
    var walker: std.fs.Dir.Walker = try dir.walk(lib.ctx.gpa);
    defer walker.deinit();
    while (try walker.next()) |entry| {
        const ext: []const u8 = std.fs.path.extension(entry.basename);
        if (funcs.str_eql(ext, ".vf")) {
            var ba: String(16) = .empty;
            try ba.append_slice(entry.basename);
            try dataset_filenames.append(lib.ctx.gpa, ba);
        }
    }
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
    const base: usize = @intFromPtr(terms);
    const addr: usize = @intFromPtr(sp);
    if (addr >= base) {
        const idx: usize = (addr - base) / @sizeOf(ScorePair);
        if (idx < scorepair_count) {
            return idx;
        }
    }
    lib.wtf("scorepair index", .{});
}

fn name_of(sp_index: usize) []const u8 {
    return meta.info[sp_index].name.slice();
}

// TODO: find a way or delete...
const Meta = struct {
    info: [scorepair_count]ScorePairInfo,

    // const zig_terms_struct_info = @typeInfo(hceterms.Terms).@"struct";
    // const sizeof_scorepair: usize = @sizeOf(ScorePair);
    // const terms_field_count: usize = zig_terms_struct_info.fields.len;

    // const terms_field_names: [terms_field_count]utils.BoundedArray(u8, 48) = blk: {
    //     var names: [terms_field_count]utils.BoundedArray(u8, 48) = @splat(.empty);
    //     for (zig_terms_struct_info.fields, 0..) |field, i| {
    //         names[i].append_slice_assume_capacity(field.name);
    //     }
    //     break :blk names;
    // };

    // //const arrays: [

    // const ArrayInfo = struct {
    //     dimensions: utils.BoundedArray(u8, u32),
    // };

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

// TODO: write code.
const Printer = struct {

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

// fn end_tuning() !void {
//     var writer: utils.FileWriter = try .init("C:/Data/Tmp/eval.bin", 4096);
//     defer writer.deinit();
//     try writer.wr.interface.writeSliceEndian(ScorePair, &terms_view.*, .little);
// }
