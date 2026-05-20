//! Dense uniform grid of per-cell distance to the nearest placed
//! sphere's surface — the placer's clearance field. cellPACK's
//! `distToClosestSurf`.
//!
//! Each cell stores a single `f32`, in world units:
//!
//! - `f32::INFINITY` — no nearby placement (default; the cell is free
//!   for spheres of any radius).
//! - `0.0` — cell is inside some placed sphere (occupied).
//! - positive — distance from this cell's centre to the nearest placed
//!   sphere's surface.
//!
//! Storing actual float distances (rather than u8-quantized clearance
//! in cell-size units) avoids a quantization-tangent failure that
//! shows up whenever `cell_size` happens to divide ingredient radii
//! evenly: with quantized clearance the grid reports "fits exactly"
//! precisely where strict collision math (`d² < r_sum²`) can flip
//! either way under FP roundoff, so most attempts get rejected even
//! though they geometrically fit. With f32 the comparison
//! `stored >= radius` is exact and the grid is authoritative enough
//! that Interior placements can skip the strict QBVH collision check.
//!
//! Cell size is chosen at construction to keep `dims[i] ≤ MAX_AXIS_CELLS`,
//! bounding memory at ~500 MB worst case (500³ × 4 bytes). For most
//! recipes the requested `target_cell_size` wins and the grid is
//! orders of magnitude smaller.

use nalgebra::Point3;
use parsimony_spatial::Aabb;

pub(crate) struct ClearanceGrid {
    pub clearance: Vec<f32>,
    pub dims: [usize; 3],
    pub cell_size: f32,
    pub inv_cell_size: f32,
    pub origin: Point3<f32>,
}

/// Cap on cells per axis. 500³ × 4 bytes = 500 MB. Beyond this we'd
/// want sparse storage; flagged in the design doc as future work for
/// whole-tissue recipes.
const MAX_AXIS_CELLS: usize = 500;

impl ClearanceGrid {
    pub fn new(bbox: Aabb, target_cell_size: f32) -> Self {
        let bbox_size = [
            bbox.max.x - bbox.min.x,
            bbox.max.y - bbox.min.y,
            bbox.max.z - bbox.min.z,
        ];
        let max_extent = bbox_size.iter().cloned().fold(0.0_f32, f32::max);
        let min_cell = max_extent / MAX_AXIS_CELLS as f32;
        let cell_size = target_cell_size.max(min_cell).max(0.001);
        let inv_cell_size = 1.0 / cell_size;
        let dims = [
            ((bbox_size[0] * inv_cell_size).ceil() as usize).max(1),
            ((bbox_size[1] * inv_cell_size).ceil() as usize).max(1),
            ((bbox_size[2] * inv_cell_size).ceil() as usize).max(1),
        ];
        let n = dims[0] * dims[1] * dims[2];
        Self {
            clearance: vec![f32::INFINITY; n],
            dims,
            cell_size,
            inv_cell_size,
            origin: bbox.min,
        }
    }

    #[inline]
    pub fn point_to_cell(&self, p: Point3<f32>) -> [i32; 3] {
        [
            ((p.x - self.origin.x) * self.inv_cell_size).floor() as i32,
            ((p.y - self.origin.y) * self.inv_cell_size).floor() as i32,
            ((p.z - self.origin.z) * self.inv_cell_size).floor() as i32,
        ]
    }

    #[inline]
    pub fn cell_centre(&self, c: [i32; 3]) -> Point3<f32> {
        Point3::new(
            self.origin.x + (c[0] as f32 + 0.5) * self.cell_size,
            self.origin.y + (c[1] as f32 + 0.5) * self.cell_size,
            self.origin.z + (c[2] as f32 + 0.5) * self.cell_size,
        )
    }

    /// Centre of the cell at flat index `i` (`= cx + dims[0] * cy +
    /// dims[0] * dims[1] * cz`).
    #[inline]
    pub fn cell_centre_flat(&self, i: u32) -> Point3<f32> {
        let i = i as usize;
        let cx = i % self.dims[0];
        let cy = (i / self.dims[0]) % self.dims[1];
        let cz = i / (self.dims[0] * self.dims[1]);
        Point3::new(
            self.origin.x + (cx as f32 + 0.5) * self.cell_size,
            self.origin.y + (cy as f32 + 0.5) * self.cell_size,
            self.origin.z + (cz as f32 + 0.5) * self.cell_size,
        )
    }

    /// Clearance (distance to the nearest placed sphere's surface) at the
    /// cell containing `p`. Points outside the grid read as `0.0` (blocked).
    /// Used by the densify phase to test, per proxy, whether a candidate
    /// instance's actual spheres fit — far tighter than the enclosing-sphere
    /// test the main pass uses.
    #[inline]
    pub fn clearance_at(&self, p: Point3<f32>) -> f32 {
        let c = self.point_to_cell(p);
        if c[0] < 0
            || c[1] < 0
            || c[2] < 0
            || c[0] >= self.dims[0] as i32
            || c[1] >= self.dims[1] as i32
            || c[2] >= self.dims[2] as i32
        {
            return 0.0;
        }
        let i = c[0] as usize + self.dims[0] * (c[1] as usize + self.dims[1] * c[2] as usize);
        self.clearance[i]
    }

    /// Update cells in range of a new placement at `p` with radius `r`,
    /// when the largest ingredient anyone might want to sample for is
    /// `max_required_radius`. Cells inside the sphere become `0.0`;
    /// cells outside take `min(current, dist - r)` (in world units),
    /// where `dist` is from the cell centre to `p`.
    pub fn update_for_placement(&mut self, p: Point3<f32>, r: f32, max_required_radius: f32) {
        let range = r + max_required_radius;
        let lo = self.point_to_cell(Point3::new(p.x - range, p.y - range, p.z - range));
        let hi = self.point_to_cell(Point3::new(p.x + range, p.y + range, p.z + range));
        let r2_outer = range * range;
        let r2_inner = r * r;
        let cs = self.cell_size;
        let origin = self.origin;
        let lo_x = lo[0].max(0);
        let lo_y = lo[1].max(0);
        let lo_z = lo[2].max(0);
        let hi_x = hi[0].min(self.dims[0] as i32 - 1);
        let hi_y = hi[1].min(self.dims[1] as i32 - 1);
        let hi_z = hi[2].min(self.dims[2] as i32 - 1);
        let stride_y = self.dims[0];
        let stride_z = self.dims[0] * self.dims[1];
        for cz in lo_z..=hi_z {
            let world_z = origin.z + (cz as f32 + 0.5) * cs;
            let dz = world_z - p.z;
            let dz2 = dz * dz;
            let row_base_z = cz as usize * stride_z;
            for cy in lo_y..=hi_y {
                let world_y = origin.y + (cy as f32 + 0.5) * cs;
                let dy = world_y - p.y;
                let dy2 = dy * dy;
                let row_base = row_base_z + cy as usize * stride_y;
                for cx in lo_x..=hi_x {
                    let world_x = origin.x + (cx as f32 + 0.5) * cs;
                    let dx = world_x - p.x;
                    let d2 = dx * dx + dy2 + dz2;
                    if d2 > r2_outer {
                        continue;
                    }
                    let v = if d2 <= r2_inner { 0.0 } else { d2.sqrt() - r };
                    let i = row_base + cx as usize;
                    let slot = unsafe { self.clearance.get_unchecked_mut(i) };
                    if v < *slot {
                        *slot = v;
                    }
                }
            }
        }
    }
}
