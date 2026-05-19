// Clearance-grid update kernel.
//
// One workgroup per placement (dispatched as `(n_placements, 1, 1)`).
// Each workgroup walks the cells inside the placement's affected
// bounding box and atomically writes `min(stored, |c−p| − r)` into
// the clearance buffer.
//
// We treat the f32 buffer as `atomic<u32>` and use `atomicMin` on
// the f32 bit-pattern. That's correct for non-negative IEEE-754
// floats (sign bit zero, exponent + mantissa lexicographically
// ordered by magnitude), which is the only regime our grid uses
// (`0 = occupied`, `+∞ = free`, otherwise positive distance).

struct GridParams {
    origin: vec3<f32>,
    cell_size: f32,
    dims: vec3<u32>,
    range_max: f32,        // max ingredient radius (per-batch)
};

struct Placement {
    x: f32,
    y: f32,
    z: f32,
    radius: f32,
};

@group(0) @binding(0) var<uniform> params: GridParams;
@group(0) @binding(1) var<storage, read> placements: array<Placement>;
@group(0) @binding(2) var<storage, read_write> clearance: array<atomic<u32>>;

fn cell_index(cx: u32, cy: u32, cz: u32) -> u32 {
    return cx + params.dims.x * (cy + params.dims.y * cz);
}

// One workgroup of 64 threads cooperates on one placement. Threads
// chunk the affected cell range in flat order.
@compute @workgroup_size(64, 1, 1)
fn update_clearance(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_index) lid: u32,
) {
    let p = placements[wg.x];
    let inv_cs = 1.0 / params.cell_size;
    let range = p.radius + params.range_max;

    // Per-placement affected cell bbox, clamped to grid extent.
    let lo_x = max(i32(floor((p.x - range - params.origin.x) * inv_cs)), 0);
    let lo_y = max(i32(floor((p.y - range - params.origin.y) * inv_cs)), 0);
    let lo_z = max(i32(floor((p.z - range - params.origin.z) * inv_cs)), 0);
    let hi_x = min(i32(floor((p.x + range - params.origin.x) * inv_cs)), i32(params.dims.x) - 1);
    let hi_y = min(i32(floor((p.y + range - params.origin.y) * inv_cs)), i32(params.dims.y) - 1);
    let hi_z = min(i32(floor((p.z + range - params.origin.z) * inv_cs)), i32(params.dims.z) - 1);

    if (lo_x > hi_x || lo_y > hi_y || lo_z > hi_z) {
        return;
    }

    let sx = u32(hi_x - lo_x + 1);
    let sy = u32(hi_y - lo_y + 1);
    let sz = u32(hi_z - lo_z + 1);
    let n_cells = sx * sy * sz;

    let r2_outer = range * range;
    let r2_inner = p.radius * p.radius;

    // Each of the 64 threads strides through the cell range.
    var i = lid;
    while (i < n_cells) {
        let ix = i % sx;
        let iy = (i / sx) % sy;
        let iz = i / (sx * sy);

        let cx = u32(lo_x) + ix;
        let cy = u32(lo_y) + iy;
        let cz = u32(lo_z) + iz;

        let wx = params.origin.x + (f32(cx) + 0.5) * params.cell_size;
        let wy = params.origin.y + (f32(cy) + 0.5) * params.cell_size;
        let wz = params.origin.z + (f32(cz) + 0.5) * params.cell_size;

        let dx = wx - p.x;
        let dy = wy - p.y;
        let dz = wz - p.z;
        let d2 = dx * dx + dy * dy + dz * dz;

        if (d2 <= r2_outer) {
            var v: f32;
            if (d2 <= r2_inner) {
                v = 0.0;
            } else {
                v = sqrt(d2) - p.radius;
            }
            let bits = bitcast<u32>(v);
            let idx = cell_index(cx, cy, cz);
            // Atomic min on the bit-pattern works for non-negative
            // floats; our grid never stores negative values.
            atomicMin(&clearance[idx], bits);
        }

        i = i + 64u;
    }
}
