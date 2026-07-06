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
const scoring = @import("scoring.zig");
const position = @import("position.zig");
const hce = @import("hce.zig");
const hceterms = @import("hceterms.zig");
const viri = @import("viri.zig");

const assert = std.debug.assert;
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

    fn mul(self: FloatScorePair, m: f32) FloatScorePair {
        return .{
            .mg = self.mg * m,
            .eg = self.eg * m,
        };
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

    // fn phased_score(self: FloatScorePair, phase: u8) f32 {
    //     const ph = float(f32, phase);
    //     const max: f32 = float(f32, types.max_phase);
    //     //const relative_phase = ph / max;
    //     return (self.mg * ph + self.eg * (max - ph)) / max;
    // }

    pub fn to_score(self: FloatScorePair, phase: u8) f32 {
        if (lib.is_paranoid) {
            assert(phase <= types.max_phase);
        }
        const ph: f32 = float(f32, phase);
        const max_phase: f32 = float(f32, types.max_phase);
        const mg: f32 = self.mg;
        const eg: f32 = self.eg;
        return (mg * ph + eg * (max_phase - ph)) / max_phase;
    }

    pub fn format(self: FloatScorePair, writer: *std.io.Writer) std.io.Writer.Error!void {
        try writer.print("fpair({d:.4}, {d:.4})", .{ self.mg, self.eg });
    }
};

/// Global status during tuning session.
const Status = struct {
    positions_seen: u64 = 0,
    sum_errors: u64 = 0,

    fn avg_error(self: *const Status) f64 {
        return float(f64, self.sum_errors) / float(f64, self.positions_seen);
    }
};

const TuningTerm = struct {
    /// Float representation of the original scorepair.
    hce_pair: FloatScorePair = .empty,
    /// Adjustment.
    delta: FloatScorePair = .empty,
    /// The total number of times this scorepair has been used.
    samples: u64 = 0,

    const empty: TuningTerm = .{};

    fn new_value(self: *const TuningTerm) ScorePair {
       return self.hce_pair.add(self.delta).to_scorepair();
    }
};

const Current = struct {
    pos: Position = .empty,
    /// The final chessnix eval.
    hce_eval: i32 = 0,
    /// The value we are learning from.
    dataset_eval: i32 = 0,
    /// The final chessnix eval scorepair.
    hce_eval_pair: ScorePair = .empty,
    /// Updated during one eval. Must be equal to hce_eval_pair.
    checksum_pair: ScorePair = .empty,
    /// Which scorepairs were used.
    used_indexes: utils.BoundedArray(u32, scorepair_count) = .empty,

    fn clear(self: *Current) void {
        self.* = .{};
    }
};

var is_running: bool = false;
/// A (mutable) flat view on the hceterms.
const terms: *[scorepair_count]ScorePair = @constCast(@ptrCast(hceterms.terms));
/// The currently learned values.
var tuningterms: [scorepair_count]TuningTerm = @splat(.empty);
/// All viri files in the source folder.
var dataset_filenames: std.ArrayList(String(16)) = .empty;
var status: Status = .{};
var current: Current = .{};


/// Evaluator calls this for each used scorepair.
pub fn register_scorepair_usage(sp: *ScorePair, multiply: u8, us: Color, debugargs: anytype) void {
    lib.only_when_tuning();
    if (!is_running) lib.wtf("tuner is not runing", .{});
    register_term(sp, multiply, us, debugargs);
}

/// Evaluator calls this just before returning its result.
pub fn register_final_result(hce_pair: ScorePair, eval: i32) void {
    lib.only_when_tuning();
    if (!is_running) lib.wtf("tuner is not runing", .{});
    _ = eval;
    // Here we assert that checksum matches hce. Otherwise we have a bug.
    current.hce_pair = hce_pair;
    current.checksum_pair = scoring.restrict_scorepair_before_scaling(current.checksum_eval_scorepair);
    if (!std.meta.eql(hce_pair, current.checksum_pair)) {
        lib.wtf(
            "eval sum mismatch {} {f} != {f}",
            .{ status.positions_seen, hce_pair, current.checksum_pair }
        );
    }
}

/// Run tuning session.
pub fn run() !void {
    lib.only_when_tuning();
    try initialize();
    defer finalize();
    is_running = true;
    defer is_running = false;

    const file_index: usize = 0;
    //var filename = compute_dataset_filename(file_index);
    const filename = try std.mem.concat(lib.ctx.gpa, u8, &.{ src_folder, dataset_filenames.items[file_index].slice() });
    defer lib.ctx.gpa.free(filename);
    var reader: viri.ViriFileReader = try .init(filename, 4096);
    defer(reader).deinit();
    var game: viri.ViriGame = .init();
    defer game.deinit();
    var evaluator: Evaluator = .init();

    // Read one file.
    game_loop: while (try reader.next(&game)) {
        // Evaluate one position.
        begin_current();
        current.pos = try game.startpos.to_position();
        current.hce_eval = evaluator.evaluate(&current.pos, .unscaled);
        current.dataset_eval = game.startpos.eval;
        end_current();
        status.positions_seen += 1;
        // Testing 3 positions.
        if (status.positions_seen >= 3) {
            break :game_loop;
        }
    }
}

fn initialize()! void {
    try get_dataset_filenames();
    current.clear();
    // Copy hce terms into tuningterms.
    for (&tuningterms, terms) |*bt, sp| {
        bt.hce_scorepair = .from_scorepair(sp);
        bt.delta = .empty;
        bt.samples = 0;
    }
}

fn finalize() void {
    dataset_filenames.deinit(lib.ctx.gpa);
}

/// Keep track.
fn register_term(sp: *ScorePair, multiply: u8, us: Color, debugargs: anytype) void {
    _ = debugargs;
    if (multiply == 0) {
        return;
    }
    const sp_index: u32 = index_of_scorepair(sp);
    // Update usage and running sum.
    current.used_indexes.append_assume_capacity(sp_index);
    switch (us.e) {
        .white => {
            current.checksum_eval_scorepair.inc(sp.*.mul(multiply));
        },
        .black => {
            current.checksum_eval_scorepair.dec(sp.*.mul(multiply));
        },
    }
}

fn begin_current() void {
    current.clear();
}

/// When eval for the current position is ready, update stuff.
fn end_current() void {
    const learning_rate: f32 = 1.0;

    const err: i32 = current.dataset_eval - current.hce_eval;
    const used: usize = current.used_indexes.len;
    const phase: u8 = types.restrict_phase(current.pos.phase());

    status.sum_errors += @abs(err);

    lib.io.debugprint(
        "phase: {}, hce_pair: {f}, chessnix_eval: {} dataset_eval: {}, used terms :{}, err: {}\n",
        .{ phase, current.hce_eval_scorepair, current.hce_eval, current.dataset_eval, used, err }
    );

    // Spread out the error over the terms.
    const err_avg: f32 = float(f32, err) / float(f32, used);
    var delta: FloatScorePair = compute_delta(phase, err_avg);
    delta.mul(learning_rate);

    var float_future: FloatScorePair = .from_scorepair(current.hce_eval_scorepair);
    for (current.used_indexes.slice()) |sp_index| {
        const tt: *TuningTerm = &tuningterms[sp_index];
        tt.delta.inc(delta);
        tt.samples += 1;

        const inf: *const Meta.ScorePairInfo = meta.get_scorepair_info(sp_index); //_ = inf;
        if (sp_index == 0)
            lib.io.debugprint("    {s}({}) active {f} delta {f}\n", .{ inf.name.slice(), inf.raw_array_index, tt.src, tt.delta });

        float_future.inc(tt.delta);
        // // Update the hceterm.
//        if (tt.samples >= 1) {
            //const hceterm: *ScorePair = &terms[sp_index];
            //hceterm.inc(.init(0, 0));
            //if ((tt.delta.mg >= 1.0 or tt.delta.mg <= -1.0 or tt.delta.eg >= 1.0 and tt.delta.eg <= -1.0) and tt.samples >= ST) {

            // if (tt.delta.mg >= 1.0 or tt.delta.mg <= -1.0 or tt.delta.eg >= 1.0 and tt.delta.eg <= -1.0) {
            //     var future: ScorePair = tt.future();
            //     future.mg = std.math.clamp(future.mg, -1000, 1000);
            //     future.eg = std.math.clamp(future.eg, -1000, 1000);
            //     const hceterm: *ScorePair = &terms[sp_index];
            //     //const prev: ScorePair = hceterm.*;
            //     hceterm.* = future;
            //     //if (sp_index == 0) lib.io.debugprint("    updated {s}({}) from {f} to {f}\n", .{ inf.name.slice(), inf.raw_array_index, prev, hceterm.* });
            //     tt.src = .from_scorepair(future);
            //     tt.delta = .empty;
            //     tt.samples = 0;
            // }
    }
    lib.io.debugprint("         hce: {f}, future: {f}, e: {}, f: {}\n", .{ current.hce_eval_scorepair, float_future, current.hce_eval, float_future.to_score(phase)});
}

fn compute_delta(pos_phase: u8, err: f32) FloatScorePair {
    const phase: f32 = float(f32, pos_phase);
    const max_phase: f32 = float(f32, types.max_phase);
    const relative_phase = phase / max_phase; // 0.0...1.0
    const mg_delta: f32 = err * relative_phase;
    const eg_delta: f32 = err * (1.0 - relative_phase);
    return .init(mg_delta, eg_delta);
}

// --- Misc ---

/// Get the correct filename inside the folder. TODO: remove. we gather all files in folder.
fn compute_dataset_filename(index: usize) utils.BoundedArray(u8, 256) {
    //std.mem.concat(allocator: Allocator, comptime T: type, slices: []const []const T)
    return funcs.get_str("{s}{s}", .{ src_folder, dataset_filenames.items[index].slice() });
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









// --- Metadata of terms don't look here. It's a mess ---

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
fn index_of_scorepair(sp: *const ScorePair) u32 {
    const base: usize = @intFromPtr(terms);
    const addr: usize = @intFromPtr(sp);
    if (addr >= base) {
        const idx: u32 = @intCast((addr - base) / @sizeOf(ScorePair));
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

// TODO: write stupid code.
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

// TODO: write some math tests.

// test "math" {
//     lib.initialize();
//     defer lib.finalize();

//     const chessnix_eval: i32 = 36;
//     const dataset_eval: i32 = 12;
//     const pair: ScorePair = types.pair(-100, 100);
//     var term: TuningTerm = .{
//         .hce_pair = .from_scorepair(pair),
//         .delta = .empty,
//         .samples = 0,
//     };


// }

// fn end_tuning() !void {
//     var writer: utils.FileWriter = try .init("C:/Data/Tmp/eval.bin", 4096);
//     defer writer.deinit();
//     try writer.wr.interface.writeSliceEndian(ScorePair, &terms_view.*, .little);
// }



// std.mem.concat or std.mem.join (they are different)
// std.fs.path.join
// open source dir, then open file relative to it.