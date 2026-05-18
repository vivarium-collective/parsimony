//! [`VoxelField`] — sparse hierarchical multiscale voxel grid with
//! OpenVDB-style **tile values** at every level. The "space-side"
//! counterpart to the instance-side [`SpatialIndex`](crate::SpatialIndex).
//!
//! ## Structure
//!
//! Three levels, in the style of OpenVDB:
//!
//! ```text
//! Root: HashMap<RootCoord, RootEntry>           — sparse, hashed
//! L1:   [L1Slot; 4096]                          — dense 16³ slot array
//! L0:   L0Tile { Constant | Dense[Cell; 512] }  — dense 8³ leaf cells
//! ```
//!
//! Each `RootEntry` is either a `Tile` (the entire 128³ region is one
//! value) or a `Child` (allocated `L1Tile` with divergent slots). Each
//! `L1Slot` is either a `Tile` (entire 8³ region is one value) or a
//! `Child` (allocated `L0Tile`). Each `L0Tile` is either `Constant`
//! (all 512 cells equal) or `Dense` (per-cell storage).
//!
//! ## Tile values
//!
//! Filling a 128³ region with one value installs a single
//! `RootEntry::Tile` — **O(1) memory, O(1) writes**. Filling an 8³
//! region installs an `L1Slot::Tile`. Partial coverage descends into
//! the children. This is the OpenVDB *sparse fill* — see
//! [`mark_aabb`](VoxelField::mark_aabb).
//!
//! ## Background
//!
//! Each grid has a configurable [`VoxelField::background`] value. The
//! invariant is that the canonical form *never stores the background
//! explicitly*: missing root entries, `L1Slot::Tile(bg)`, `L0Tile::
//! Constant(bg)`, and `Dense` with `non_bg_count == 0` are all
//! canonicalized to absent. This means [`get`](VoxelField::get) on an
//! unwritten cell returns `background` and uses zero allocations.
//!
//! ## Multi-scale queries
//!
//! Region queries like [`is_region_default`](VoxelField::is_region_default)
//! and [`any_cell_with_flags`](VoxelField::any_cell_with_flags) prune
//! at the highest level where the answer is unambiguous. A 200-nm
//! ribosome cluster never has to scan individual 5-nm cells in regions
//! known to be uniform.
//!
//! ## Coordinate math
//!
//! Cell coordinates are signed `i32`. Negative coords work naturally
//! via arithmetic shift and two's-complement bitmasking:
//!
//! - `cell >> 7` = root coordinate (floor division by 128)
//! - `(cell >> 3) & 0xF` = L1 sub-index (0..=15)
//! - `cell & 0x7` = L0 sub-index (0..=7)

use std::collections::HashMap;

use nalgebra::Point3;
use rand::Rng;

use crate::aabb::Aabb;

// ---------- constants ----------

const L0_BITS: u32 = 3;
const L1_BITS: u32 = 4;
const ROOT_BITS: u32 = L0_BITS + L1_BITS;

const L0_DIM: i32 = 1 << L0_BITS; // 8
const L1_DIM: i32 = 1 << L1_BITS; // 16
const ROOT_DIM: i32 = 1 << ROOT_BITS; // 128

const L0_MASK: i32 = L0_DIM - 1; // 0x07
const L1_MASK: i32 = L1_DIM - 1; // 0x0F

const L0_SIZE: usize = (L0_DIM as usize).pow(3); // 512
const L1_SIZE: usize = (L1_DIM as usize).pow(3); // 4096

// ---------- public types ----------

pub type CompartmentId = u16;
pub type CellFlags = u8;

pub const OCCUPIED: CellFlags = 0x01;
pub const SURFACE: CellFlags = 0x02;
pub const MEMBRANE_INNER: CellFlags = 0x04;
pub const MEMBRANE_OUTER: CellFlags = 0x08;

/// 4-byte voxel value: `compartment: u16`, `flags: u8`, `distance: u8`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Hash)]
pub struct Cell {
    pub compartment: CompartmentId,
    pub flags: CellFlags,
    pub distance: u8,
}

impl Cell {
    pub const DEFAULT: Cell = Cell {
        compartment: 0,
        flags: 0,
        distance: 0,
    };

    pub const fn new(compartment: CompartmentId, flags: CellFlags, distance: u8) -> Self {
        Self {
            compartment,
            flags,
            distance,
        }
    }

    pub const fn has_flag(self, mask: CellFlags) -> bool {
        self.flags & mask != 0
    }

    pub const fn is_occupied(self) -> bool {
        self.has_flag(OCCUPIED)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct CellCoord {
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl CellCoord {
    pub const fn new(x: i32, y: i32, z: i32) -> Self {
        Self { x, y, z }
    }
}

type RootCoord = (i32, i32, i32);

#[derive(Debug, Clone, Default)]
pub struct VoxelFieldStats {
    /// Cells with values different from `background`.
    pub active_cells: usize,
    /// L0 tiles allocated as `Dense` (have explicit per-cell storage).
    pub dense_l0_tiles: usize,
    /// L0 tiles allocated as `Constant(non-bg)` (rare in canonical
    /// form — pruning normally promotes them to L1Slot::Tile).
    pub constant_l0_tiles: usize,
    /// L1 slots holding non-background tile values (`L1Slot::Tile(v)`,
    /// `v != bg`) — these represent uniformly-filled 8³ regions
    /// without any L0 allocation.
    pub tile_l1_slots: usize,
    /// L1 tiles allocated (have a `Vec<L1Slot>`).
    pub l1_tiles: usize,
    /// Root entries where `RootEntry::Tile(non-bg)` — entire 128³ region constant.
    pub constant_root_tiles: usize,
    pub memory_bytes: usize,
}

// ---------- private storage ----------

/// One leaf-level voxel block. `Constant(v)` represents a uniform 8³
/// region cheaply; `Dense` carries per-cell values. The canonical form
/// never has `Constant(background)` — the parent should hold
/// `L1Slot::Tile(background)` instead.
#[derive(Debug)]
enum L0Tile {
    Constant(Cell),
    Dense {
        cells: Box<[Cell; L0_SIZE]>,
        /// Number of cells with values != background.
        non_bg_count: u16,
    },
}

impl L0Tile {
    #[inline]
    fn idx(ox: i32, oy: i32, oz: i32) -> usize {
        (ox as usize) + (L0_DIM as usize) * ((oy as usize) + (L0_DIM as usize) * (oz as usize))
    }

    fn get(&self, ox: i32, oy: i32, oz: i32) -> Cell {
        match self {
            L0Tile::Constant(v) => *v,
            L0Tile::Dense { cells, .. } => cells[Self::idx(ox, oy, oz)],
        }
    }

    fn put(&mut self, ox: i32, oy: i32, oz: i32, cell: Cell, bg: Cell) -> Cell {
        let idx = Self::idx(ox, oy, oz);
        match self {
            L0Tile::Constant(v) => {
                let old = *v;
                if old == cell {
                    return old;
                }
                // Expand to Dense.
                let mut cells: Box<[Cell; L0_SIZE]> = Box::new([old; L0_SIZE]);
                let mut non_bg_count: u16 = if old == bg { 0 } else { L0_SIZE as u16 };
                cells[idx] = cell;
                match (old == bg, cell == bg) {
                    (true, false) => non_bg_count += 1,
                    (false, true) => non_bg_count -= 1,
                    _ => {}
                }
                *self = L0Tile::Dense {
                    cells,
                    non_bg_count,
                };
                old
            }
            L0Tile::Dense {
                cells,
                non_bg_count,
            } => {
                let old = cells[idx];
                if old == cell {
                    return old;
                }
                cells[idx] = cell;
                match (old == bg, cell == bg) {
                    (true, false) => *non_bg_count += 1,
                    (false, true) => *non_bg_count -= 1,
                    _ => {}
                }
                old
            }
        }
    }

    /// True iff the tile is canonically equivalent to "every cell is `bg`".
    fn is_background(&self, bg: Cell) -> bool {
        match self {
            L0Tile::Constant(v) => *v == bg,
            L0Tile::Dense { non_bg_count, .. } => *non_bg_count == 0,
        }
    }

    /// If every cell in `Dense` holds the same value, collapse to
    /// `Constant`. Returns the new uniform value if collapsed.
    fn try_compact(&mut self) -> Option<Cell> {
        let L0Tile::Dense { cells, .. } = self else {
            return None;
        };
        let first = cells[0];
        for &c in cells.iter().skip(1) {
            if c != first {
                return None;
            }
        }
        *self = L0Tile::Constant(first);
        Some(first)
    }
}

/// One slot in an `L1Tile`. Inline `Tile(v)` value or pointer to an
/// allocated `L0Tile`. Canonical form: `L1Slot::Child(l0)` where `l0`
/// is background-equivalent is forbidden — should be `L1Slot::Tile(bg)`.
#[derive(Debug)]
enum L1Slot {
    Tile(Cell),
    Child(Box<L0Tile>),
}

#[derive(Debug)]
struct L1Tile {
    /// 4096 slots, addressed by `L1Tile::idx(lx, ly, lz)`.
    slots: Vec<L1Slot>,
    /// Number of slots that are NOT canonical `Tile(background)`.
    /// (Includes `Tile(non-bg)` and any `Child`.)
    non_bg_slot_count: u16,
}

impl L1Tile {
    #[inline]
    fn idx(lx: i32, ly: i32, lz: i32) -> usize {
        (lx as usize) + (L1_DIM as usize) * ((ly as usize) + (L1_DIM as usize) * (lz as usize))
    }

    /// Make a new L1Tile where every slot is `Tile(value)`.
    fn filled_with(value: Cell, bg: Cell) -> Box<Self> {
        let mut slots = Vec::with_capacity(L1_SIZE);
        slots.resize_with(L1_SIZE, || L1Slot::Tile(value));
        let non_bg_slot_count = if value == bg { 0 } else { L1_SIZE as u16 };
        Box::new(Self {
            slots,
            non_bg_slot_count,
        })
    }

    fn is_background(&self) -> bool {
        self.non_bg_slot_count == 0
    }
}

/// Root-level entry: either a constant tile-value covering the entire
/// 128³ root region, or an allocated `L1Tile` with divergent slots.
/// Canonical form: `RootEntry::Tile(background)` never appears in the
/// hashmap — that's "absent root coord".
#[derive(Debug)]
enum RootEntry {
    Tile(Cell),
    Child(Box<L1Tile>),
}

// ---------- the field ----------

#[derive(Debug)]
pub struct VoxelField {
    tiles: HashMap<RootCoord, RootEntry>,
    cell_size: f32,
    inv_cell_size: f32,
    origin: Point3<f32>,
    background: Cell,
    active_cells: usize,
}

impl VoxelField {
    /// New empty field with background = [`Cell::DEFAULT`].
    pub fn new(cell_size: f32) -> Self {
        Self::with_origin_and_background(cell_size, Point3::origin(), Cell::DEFAULT)
    }

    pub fn with_origin(cell_size: f32, origin: Point3<f32>) -> Self {
        Self::with_origin_and_background(cell_size, origin, Cell::DEFAULT)
    }

    pub fn with_background(cell_size: f32, background: Cell) -> Self {
        Self::with_origin_and_background(cell_size, Point3::origin(), background)
    }

    pub fn with_origin_and_background(
        cell_size: f32,
        origin: Point3<f32>,
        background: Cell,
    ) -> Self {
        assert!(cell_size > 0.0, "cell_size must be positive");
        Self {
            tiles: HashMap::new(),
            cell_size,
            inv_cell_size: 1.0 / cell_size,
            origin,
            background,
            active_cells: 0,
        }
    }

    pub fn cell_size(&self) -> f32 {
        self.cell_size
    }

    pub fn origin(&self) -> Point3<f32> {
        self.origin
    }

    pub fn background(&self) -> Cell {
        self.background
    }

    pub fn active_cells(&self) -> usize {
        self.active_cells
    }

    pub fn is_empty(&self) -> bool {
        self.active_cells == 0
    }

    pub fn clear(&mut self) {
        self.tiles.clear();
        self.active_cells = 0;
    }

    // ---------- coord conversion ----------

    pub fn point_to_cell(&self, p: Point3<f32>) -> CellCoord {
        let rel = p - self.origin;
        CellCoord {
            x: (rel.x * self.inv_cell_size).floor() as i32,
            y: (rel.y * self.inv_cell_size).floor() as i32,
            z: (rel.z * self.inv_cell_size).floor() as i32,
        }
    }

    /// Cell range overlapping `aabb`, using **half-open max** semantics:
    /// a cell `c` covers `[c, c+1)` in cell-coord space; it overlaps
    /// `aabb` iff `c+1 > aabb.min` and `c < aabb.max`. Returns the
    /// inclusive range `(lo, hi)`. For `aabb.max` exactly on a grid
    /// line (e.g. `8.0` at `cell_size = 1`), `hi` is the cell just
    /// *before* that line (`7`), not the cell starting on it. Returns
    /// an empty range (`hi < lo` on some axis) if the aabb has zero or
    /// negative extent on that axis.
    pub fn aabb_to_cell_range(&self, aabb: Aabb) -> (CellCoord, CellCoord) {
        let rel_min = aabb.min - self.origin;
        let rel_max = aabb.max - self.origin;
        let lo = CellCoord {
            x: (rel_min.x * self.inv_cell_size).floor() as i32,
            y: (rel_min.y * self.inv_cell_size).floor() as i32,
            z: (rel_min.z * self.inv_cell_size).floor() as i32,
        };
        let hi = CellCoord {
            x: (rel_max.x * self.inv_cell_size).ceil() as i32 - 1,
            y: (rel_max.y * self.inv_cell_size).ceil() as i32 - 1,
            z: (rel_max.z * self.inv_cell_size).ceil() as i32 - 1,
        };
        (lo, hi)
    }

    pub fn cell_center(&self, c: CellCoord) -> Point3<f32> {
        Point3::new(
            self.origin.x + (c.x as f32 + 0.5) * self.cell_size,
            self.origin.y + (c.y as f32 + 0.5) * self.cell_size,
            self.origin.z + (c.z as f32 + 0.5) * self.cell_size,
        )
    }

    // ---------- point-level reads ----------

    pub fn sample(&self, p: Point3<f32>) -> Cell {
        self.get(self.point_to_cell(p))
    }

    pub fn get(&self, c: CellCoord) -> Cell {
        let root = root_coord(c);
        match self.tiles.get(&root) {
            None => self.background,
            Some(RootEntry::Tile(v)) => *v,
            Some(RootEntry::Child(l1)) => {
                let (lx, ly, lz) = l1_sub_indices(c);
                match &l1.slots[L1Tile::idx(lx, ly, lz)] {
                    L1Slot::Tile(v) => *v,
                    L1Slot::Child(l0) => {
                        let (ox, oy, oz) = l0_sub_indices(c);
                        l0.get(ox, oy, oz)
                    }
                }
            }
        }
    }

    // ---------- point-level writes ----------

    pub fn set(&mut self, p: Point3<f32>, cell: Cell) {
        let c = self.point_to_cell(p);
        self.put(c, cell);
    }

    pub fn put(&mut self, c: CellCoord, cell: Cell) -> Cell {
        let bg = self.background;
        let old = self.get(c);
        if old == cell {
            return old;
        }

        // Update grid-wide active counter (we know old != cell).
        match (old == bg, cell == bg) {
            (true, false) => self.active_cells += 1,
            (false, true) => self.active_cells -= 1,
            _ => {}
        }

        let root = root_coord(c);
        let (lx, ly, lz) = l1_sub_indices(c);
        let l1_idx = L1Tile::idx(lx, ly, lz);
        let (ox, oy, oz) = l0_sub_indices(c);

        // Step 1: ensure the root holds RootEntry::Child(l1), expanding
        // a Tile entry if necessary. If there's no entry and we're
        // writing bg, we're done (but we already returned early above).
        let entry = self
            .tiles
            .entry(root)
            .or_insert(RootEntry::Tile(bg));
        if let RootEntry::Tile(v) = entry {
            let v_val = *v;
            // We need to expand (cell != v_val, otherwise old == cell).
            *entry = RootEntry::Child(L1Tile::filled_with(v_val, bg));
        }
        let RootEntry::Child(l1) = entry else {
            unreachable!()
        };

        // Step 2: get or create the L0Tile under this slot.
        let slot = &mut l1.slots[l1_idx];
        let prev_slot_was_bg = matches!(slot, L1Slot::Tile(v) if *v == bg);
        if let L1Slot::Tile(v) = slot {
            // Expand to Child(L0::Constant(*v)).
            *slot = L1Slot::Child(Box::new(L0Tile::Constant(*v)));
        }
        let L1Slot::Child(l0) = slot else { unreachable!() };

        // Step 3: write the cell into L0.
        let _old_inner = l0.put(ox, oy, oz, cell, bg);

        // Step 4: cleanup. If the L0 is now all-bg, collapse it to
        // L1Slot::Tile(bg) and free the box.
        let l0_bg = l0.is_background(bg);
        if l0_bg {
            *slot = L1Slot::Tile(bg);
            if !prev_slot_was_bg {
                l1.non_bg_slot_count -= 1;
            }
        } else if prev_slot_was_bg {
            // We just turned a bg slot into a non-bg one (Tile->Child).
            l1.non_bg_slot_count += 1;
        }

        // Step 5: cleanup. If the L1 is now all-bg, remove the root entry.
        if l1.is_background() {
            self.tiles.remove(&root);
        }

        old
    }

    // ---------- sparse bulk writes ----------

    /// Write `cell` to every voxel whose centre lies inside `aabb`,
    /// using OpenVDB-style sparse fill: when `aabb` fully covers a tile
    /// region at any level, install a constant tile there directly
    /// instead of walking individual cells. Turns a 128³ fill into one
    /// root entry, an 8³ fill into one slot, etc.
    pub fn mark_aabb(&mut self, aabb: Aabb, cell: Cell) {
        let bg = self.background;
        let (lo, hi) = self.aabb_to_cell_range(aabb);
        if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
            return;
        }
        self.fill_range(lo, hi, cell, bg);
    }

    /// Apply a per-cell update over `aabb`. Cell-by-cell — no sparse
    /// fast path because the value depends on the existing cell. For
    /// uniform fills, use [`Self::mark_aabb`] which is much faster on
    /// large regions.
    pub fn update_aabb<F: FnMut(Cell) -> Cell>(&mut self, aabb: Aabb, mut update: F) {
        let (lo, hi) = self.aabb_to_cell_range(aabb);
        if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
            return;
        }
        for cz in lo.z..=hi.z {
            for cy in lo.y..=hi.y {
                for cx in lo.x..=hi.x {
                    let c = CellCoord::new(cx, cy, cz);
                    let old = self.get(c);
                    let new = update(old);
                    if new != old {
                        self.put(c, new);
                    }
                }
            }
        }
    }

    pub fn mark_occupied(&mut self, aabb: Aabb) {
        self.update_aabb(aabb, |c| Cell {
            flags: c.flags | OCCUPIED,
            ..c
        });
    }

    pub fn clear_occupied(&mut self, aabb: Aabb) {
        self.update_aabb(aabb, |c| Cell {
            flags: c.flags & !OCCUPIED,
            ..c
        });
    }

    pub fn mark_compartment(&mut self, aabb: Aabb, id: CompartmentId) {
        self.update_aabb(aabb, |c| Cell {
            compartment: id,
            ..c
        });
    }

    // ---------- sparse fill machinery ----------

    /// Fill `[lo, hi]` (inclusive cell range) with `cell`. Recurses
    /// down the hierarchy, installing constant tiles where the range
    /// fully covers a tile region, descending where it partially does.
    fn fill_range(&mut self, lo: CellCoord, hi: CellCoord, cell: Cell, bg: Cell) {
        let root_lo = (
            cell_to_root_axis(lo.x),
            cell_to_root_axis(lo.y),
            cell_to_root_axis(lo.z),
        );
        let root_hi = (
            cell_to_root_axis(hi.x),
            cell_to_root_axis(hi.y),
            cell_to_root_axis(hi.z),
        );
        for rz in root_lo.2..=root_hi.2 {
            for ry in root_lo.1..=root_hi.1 {
                for rx in root_lo.0..=root_hi.0 {
                    self.fill_root_tile((rx, ry, rz), lo, hi, cell, bg);
                }
            }
        }
    }

    fn fill_root_tile(
        &mut self,
        root: RootCoord,
        lo: CellCoord,
        hi: CellCoord,
        cell: Cell,
        bg: Cell,
    ) {
        let root_lo = (root.0 * ROOT_DIM, root.1 * ROOT_DIM, root.2 * ROOT_DIM);
        let root_hi = (
            root_lo.0 + ROOT_DIM - 1,
            root_lo.1 + ROOT_DIM - 1,
            root_lo.2 + ROOT_DIM - 1,
        );
        let r_lo = (
            lo.x.max(root_lo.0),
            lo.y.max(root_lo.1),
            lo.z.max(root_lo.2),
        );
        let r_hi = (
            hi.x.min(root_hi.0),
            hi.y.min(root_hi.1),
            hi.z.min(root_hi.2),
        );
        let fully_covers_root = r_lo == root_lo && r_hi == root_hi;

        if fully_covers_root {
            self.replace_root_with_constant(root, cell);
            return;
        }

        // Partial coverage: descend into L1 slots, accumulating the
        // active-cell delta from each fill so we can update the
        // grid-level counter in one shot.
        let l1 = self.materialize_l1_for_partial_fill(root, bg);
        let mut delta: i64 = 0;
        for lz in (r_lo.2 >> L0_BITS)..=(r_hi.2 >> L0_BITS) {
            for ly in (r_lo.1 >> L0_BITS)..=(r_hi.1 >> L0_BITS) {
                for lx in (r_lo.0 >> L0_BITS)..=(r_hi.0 >> L0_BITS) {
                    let l1_axis = (lx & L1_MASK, ly & L1_MASK, lz & L1_MASK);
                    delta += fill_l1_slot(
                        l1,
                        l1_axis,
                        (lx * L0_DIM, ly * L0_DIM, lz * L0_DIM),
                        r_lo,
                        r_hi,
                        cell,
                        bg,
                    );
                }
            }
        }
        let l1_is_bg = l1.is_background();

        if delta >= 0 {
            self.active_cells += delta as usize;
        } else {
            self.active_cells -= (-delta) as usize;
        }

        if l1_is_bg {
            self.tiles.remove(&root);
        }
    }

    /// Install a single constant root tile, updating `active_cells`
    /// based on the size of the region being newly bg or non-bg.
    fn replace_root_with_constant(&mut self, root: RootCoord, cell: Cell) {
        let bg = self.background;
        // Compute the active-cell delta. Count cells inside this root
        // that were non-bg, subtract from active_cells; then add ROOT_DIM³
        // if cell != bg.
        let old_active = self.count_active_cells_in_root(root);
        self.active_cells -= old_active;
        if cell != bg {
            self.active_cells += (ROOT_DIM as usize).pow(3);
        }
        if cell == bg {
            self.tiles.remove(&root);
        } else {
            self.tiles.insert(root, RootEntry::Tile(cell));
        }
    }

    /// Ensure the root holds a `Child(L1Tile)` (not `Tile(...)` and
    /// not absent), suitable for partial-coverage modification. Returns
    /// a mutable reference to the L1.
    fn materialize_l1_for_partial_fill(
        &mut self,
        root: RootCoord,
        bg: Cell,
    ) -> &mut L1Tile {
        let entry = self.tiles.entry(root).or_insert(RootEntry::Tile(bg));
        if let RootEntry::Tile(v) = entry {
            let v_val = *v;
            *entry = RootEntry::Child(L1Tile::filled_with(v_val, bg));
        }
        let RootEntry::Child(l1) = entry else {
            unreachable!()
        };
        l1.as_mut()
    }

    fn count_active_cells_in_root(&self, root: RootCoord) -> usize {
        match self.tiles.get(&root) {
            None => 0,
            Some(RootEntry::Tile(v)) => {
                if *v == self.background {
                    0
                } else {
                    (ROOT_DIM as usize).pow(3)
                }
            }
            Some(RootEntry::Child(l1)) => count_active_in_l1(l1, self.background),
        }
    }

    // ---------- region queries (multi-scale pruning) ----------

    pub fn is_region_default(&self, aabb: Aabb) -> bool {
        let bg = self.background;
        self.region_predicate(aabb, |cell| cell == bg)
    }

    pub fn any_cell_with_flags(&self, aabb: Aabb, mask: CellFlags) -> bool {
        !self.region_predicate(aabb, |cell| cell.flags & mask == 0)
    }

    pub fn is_region_free(&self, aabb: Aabb) -> bool {
        !self.any_cell_with_flags(aabb, OCCUPIED)
    }

    /// Walk `aabb` and return true iff `pred(cell)` holds for every
    /// cell in the region. Prunes at the highest tile level where the
    /// answer is unambiguous.
    fn region_predicate<P: FnMut(Cell) -> bool>(&self, aabb: Aabb, mut pred: P) -> bool {
        let (lo, hi) = self.aabb_to_cell_range(aabb);
        if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
            return true;
        }
        let bg = self.background;
        let root_lo = (
            cell_to_root_axis(lo.x),
            cell_to_root_axis(lo.y),
            cell_to_root_axis(lo.z),
        );
        let root_hi = (
            cell_to_root_axis(hi.x),
            cell_to_root_axis(hi.y),
            cell_to_root_axis(hi.z),
        );

        for rz in root_lo.2..=root_hi.2 {
            for ry in root_lo.1..=root_hi.1 {
                for rx in root_lo.0..=root_hi.0 {
                    let root = (rx, ry, rz);
                    match self.tiles.get(&root) {
                        None => {
                            if !pred(bg) {
                                return false;
                            }
                        }
                        Some(RootEntry::Tile(v)) => {
                            if !pred(*v) {
                                return false;
                            }
                        }
                        Some(RootEntry::Child(l1)) => {
                            if !walk_l1_predicate(l1, root, lo, hi, &mut pred) {
                                return false;
                            }
                        }
                    }
                }
            }
        }
        true
    }

    pub fn is_l0_active(&self, c: CellCoord) -> bool {
        let bg = self.background;
        let root = root_coord(c);
        match self.tiles.get(&root) {
            None => false,
            Some(RootEntry::Tile(v)) => *v != bg,
            Some(RootEntry::Child(l1)) => {
                let (lx, ly, lz) = l1_sub_indices(c);
                match &l1.slots[L1Tile::idx(lx, ly, lz)] {
                    L1Slot::Tile(v) => *v != bg,
                    L1Slot::Child(l0) => !l0.is_background(bg),
                }
            }
        }
    }

    pub fn is_l1_active(&self, c: CellCoord) -> bool {
        let bg = self.background;
        let root = root_coord(c);
        match self.tiles.get(&root) {
            None => false,
            Some(RootEntry::Tile(v)) => *v != bg,
            Some(RootEntry::Child(l1)) => !l1.is_background(),
        }
    }

    // ---------- iteration ----------

    /// Visit every non-background cell that overlaps `aabb`. Prunes
    /// at the L1 / L0 tile level where possible. Constant tiles
    /// (`L0Tile::Constant(v)` and `L1Slot::Tile(v)` for non-bg `v`,
    /// and `RootEntry::Tile(v)` for non-bg `v`) are expanded into
    /// per-cell visits — every cell carrying that value is reported.
    pub fn iter_active_cells_in<F: FnMut(CellCoord, Cell)>(&self, aabb: Aabb, mut visit: F) {
        let (lo, hi) = self.aabb_to_cell_range(aabb);
        if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
            return;
        }
        let bg = self.background;
        let root_lo = (
            cell_to_root_axis(lo.x),
            cell_to_root_axis(lo.y),
            cell_to_root_axis(lo.z),
        );
        let root_hi = (
            cell_to_root_axis(hi.x),
            cell_to_root_axis(hi.y),
            cell_to_root_axis(hi.z),
        );
        for rz in root_lo.2..=root_hi.2 {
            for ry in root_lo.1..=root_hi.1 {
                for rx in root_lo.0..=root_hi.0 {
                    let root = (rx, ry, rz);
                    let Some(entry) = self.tiles.get(&root) else {
                        continue;
                    };
                    walk_root_for_iter(entry, root, lo, hi, bg, &mut visit);
                }
            }
        }
    }

    pub fn iter_active_cells<F: FnMut(CellCoord, Cell)>(&self, mut visit: F) {
        let bg = self.background;
        for (&root, entry) in self.tiles.iter() {
            match entry {
                RootEntry::Tile(v) => {
                    if *v == bg {
                        continue;
                    }
                    // Every cell in the 128³ region carries v.
                    expand_uniform_region(
                        (root.0 * ROOT_DIM, root.1 * ROOT_DIM, root.2 * ROOT_DIM),
                        (
                            root.0 * ROOT_DIM + ROOT_DIM - 1,
                            root.1 * ROOT_DIM + ROOT_DIM - 1,
                            root.2 * ROOT_DIM + ROOT_DIM - 1,
                        ),
                        *v,
                        &mut visit,
                    );
                }
                RootEntry::Child(l1) => {
                    for (slot_idx, slot) in l1.slots.iter().enumerate() {
                        let (lx, ly, lz) = unflatten_l1(slot_idx);
                        let l0_origin = (
                            root.0 * ROOT_DIM + lx * L0_DIM,
                            root.1 * ROOT_DIM + ly * L0_DIM,
                            root.2 * ROOT_DIM + lz * L0_DIM,
                        );
                        let l0_max = (
                            l0_origin.0 + L0_DIM - 1,
                            l0_origin.1 + L0_DIM - 1,
                            l0_origin.2 + L0_DIM - 1,
                        );
                        match slot {
                            L1Slot::Tile(v) => {
                                if *v == bg {
                                    continue;
                                }
                                expand_uniform_region(l0_origin, l0_max, *v, &mut visit);
                            }
                            L1Slot::Child(l0) => match l0.as_ref() {
                                L0Tile::Constant(v) => {
                                    if *v == bg {
                                        continue;
                                    }
                                    expand_uniform_region(l0_origin, l0_max, *v, &mut visit);
                                }
                                L0Tile::Dense { cells, .. } => {
                                    for (idx, &cell) in cells.iter().enumerate() {
                                        if cell == bg {
                                            continue;
                                        }
                                        let (ox, oy, oz) = unflatten_l0(idx);
                                        visit(
                                            CellCoord::new(
                                                l0_origin.0 + ox,
                                                l0_origin.1 + oy,
                                                l0_origin.2 + oz,
                                            ),
                                            cell,
                                        );
                                    }
                                }
                            },
                        }
                    }
                }
            }
        }
    }

    // ---------- sampling ----------

    pub fn sample_predicate<R: Rng, P: Fn(Cell) -> bool>(
        &self,
        near: Point3<f32>,
        radius: f32,
        rng: &mut R,
        attempts: usize,
        predicate: P,
    ) -> Option<Point3<f32>> {
        for _ in 0..attempts {
            let p = sample_in_sphere(near, radius, rng);
            if predicate(self.sample(p)) {
                return Some(p);
            }
        }
        None
    }

    pub fn find_free_point<R: Rng>(
        &self,
        near: Point3<f32>,
        radius: f32,
        rng: &mut R,
        attempts: usize,
    ) -> Option<Point3<f32>> {
        self.sample_predicate(near, radius, rng, attempts, |c| !c.is_occupied())
    }

    pub fn find_point_in_compartment<R: Rng>(
        &self,
        near: Point3<f32>,
        radius: f32,
        rng: &mut R,
        attempts: usize,
        id: CompartmentId,
    ) -> Option<Point3<f32>> {
        self.sample_predicate(near, radius, rng, attempts, |c| c.compartment == id)
    }

    // ---------- bounds ----------

    pub fn bounds(&self) -> Option<Aabb> {
        let mut any = false;
        let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        self.iter_active_cells(|c, _| {
            let centre = self.cell_center(c);
            let half = self.cell_size * 0.5;
            min = Point3::new(
                min.x.min(centre.x - half),
                min.y.min(centre.y - half),
                min.z.min(centre.z - half),
            );
            max = Point3::new(
                max.x.max(centre.x + half),
                max.y.max(centre.y + half),
                max.z.max(centre.z + half),
            );
            any = true;
        });
        if any {
            Some(Aabb::new(min, max))
        } else {
            None
        }
    }

    // ---------- compaction ----------

    /// Walk every allocated L0Tile and, if uniform, collapse to
    /// `Constant`. Walk every L1Slot::Child whose L0 is `Constant(v)`
    /// and collapse to `L1Slot::Tile(v)`. Walk every L1 whose slots
    /// all hold the same `Tile(v)` and collapse the root entry to
    /// `RootEntry::Tile(v)`. Drops `RootEntry::Tile(bg)` and L1s with
    /// `non_bg_slot_count == 0`.
    ///
    /// Run after a sequence of edits when you want to recover canonical
    /// form. The structure stays correct without it, but iteration and
    /// region queries are faster on a pruned tree.
    pub fn prune(&mut self) {
        let bg = self.background;
        let roots: Vec<RootCoord> = self.tiles.keys().copied().collect();
        for root in roots {
            let collapse_to: Option<Cell>;
            match self.tiles.get_mut(&root) {
                Some(RootEntry::Child(l1)) => {
                    // Compact L0s and try-collapse L1 slots.
                    let mut uniform: Option<Cell> = None;
                    let mut all_same = true;
                    for slot in l1.slots.iter_mut() {
                        if let L1Slot::Child(l0) = slot
                            && let Some(v) = l0.try_compact()
                        {
                            *slot = L1Slot::Tile(v);
                        }
                        match slot {
                            L1Slot::Tile(v) => {
                                if let Some(u) = uniform {
                                    if u != *v {
                                        all_same = false;
                                    }
                                } else {
                                    uniform = Some(*v);
                                }
                            }
                            L1Slot::Child(_) => {
                                all_same = false;
                            }
                        }
                    }
                    // Recount non_bg_slot_count.
                    let mut count = 0u16;
                    for slot in l1.slots.iter() {
                        if let L1Slot::Tile(v) = slot {
                            if *v != bg {
                                count += 1;
                            }
                        } else {
                            count += 1;
                        }
                    }
                    l1.non_bg_slot_count = count;
                    collapse_to = if all_same { uniform } else { None };
                }
                _ => collapse_to = None,
            }
            if let Some(v) = collapse_to {
                if v == bg {
                    self.tiles.remove(&root);
                } else {
                    self.tiles.insert(root, RootEntry::Tile(v));
                }
            } else if let Some(RootEntry::Child(l1)) = self.tiles.get(&root)
                && l1.is_background()
            {
                self.tiles.remove(&root);
            }
        }
    }

    // ---------- diagnostics ----------

    pub fn stats(&self) -> VoxelFieldStats {
        let bg = self.background;
        let mut dense_l0 = 0usize;
        let mut const_l0 = 0usize;
        let mut tile_l1 = 0usize;
        let mut const_root = 0usize;
        let mut l1_tiles = 0usize;
        let mut bytes = std::mem::size_of::<Self>();
        bytes += self.tiles.capacity()
            * (std::mem::size_of::<RootCoord>() + std::mem::size_of::<RootEntry>());
        for entry in self.tiles.values() {
            match entry {
                RootEntry::Tile(_) => {
                    const_root += 1;
                }
                RootEntry::Child(l1) => {
                    l1_tiles += 1;
                    bytes += std::mem::size_of::<L1Tile>();
                    bytes += l1.slots.capacity() * std::mem::size_of::<L1Slot>();
                    for slot in l1.slots.iter() {
                        match slot {
                            L1Slot::Tile(v) => {
                                if *v != bg {
                                    tile_l1 += 1;
                                }
                            }
                            L1Slot::Child(l0) => {
                                bytes += std::mem::size_of::<L0Tile>();
                                match l0.as_ref() {
                                    L0Tile::Constant(_) => const_l0 += 1,
                                    L0Tile::Dense { .. } => {
                                        dense_l0 += 1;
                                        bytes += L0_SIZE * std::mem::size_of::<Cell>();
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        VoxelFieldStats {
            active_cells: self.active_cells,
            dense_l0_tiles: dense_l0,
            constant_l0_tiles: const_l0,
            tile_l1_slots: tile_l1,
            l1_tiles,
            constant_root_tiles: const_root,
            memory_bytes: bytes,
        }
    }
}

impl Default for VoxelField {
    fn default() -> Self {
        Self::new(1.0)
    }
}

// ---------- free-standing helpers ----------

#[inline]
fn cell_to_root_axis(c: i32) -> i32 {
    c >> ROOT_BITS
}

#[inline]
fn root_coord(c: CellCoord) -> RootCoord {
    (
        cell_to_root_axis(c.x),
        cell_to_root_axis(c.y),
        cell_to_root_axis(c.z),
    )
}

#[inline]
fn l1_sub_indices(c: CellCoord) -> (i32, i32, i32) {
    (
        (c.x >> L0_BITS) & L1_MASK,
        (c.y >> L0_BITS) & L1_MASK,
        (c.z >> L0_BITS) & L1_MASK,
    )
}

#[inline]
fn l0_sub_indices(c: CellCoord) -> (i32, i32, i32) {
    (c.x & L0_MASK, c.y & L0_MASK, c.z & L0_MASK)
}

#[inline]
fn unflatten_l1(idx: usize) -> (i32, i32, i32) {
    let lx = idx % (L1_DIM as usize);
    let ly = (idx / (L1_DIM as usize)) % (L1_DIM as usize);
    let lz = idx / ((L1_DIM as usize) * (L1_DIM as usize));
    (lx as i32, ly as i32, lz as i32)
}

#[inline]
fn unflatten_l0(idx: usize) -> (i32, i32, i32) {
    let ox = idx % (L0_DIM as usize);
    let oy = (idx / (L0_DIM as usize)) % (L0_DIM as usize);
    let oz = idx / ((L0_DIM as usize) * (L0_DIM as usize));
    (ox as i32, oy as i32, oz as i32)
}

/// Fill a single L1 slot. Returns the active-cell delta (new non-bg
/// cells minus old non-bg cells) over the region [slot_lo, slot_hi].
fn fill_l1_slot(
    l1: &mut L1Tile,
    l1_axis: (i32, i32, i32),
    l0_origin: (i32, i32, i32),
    r_lo: (i32, i32, i32),
    r_hi: (i32, i32, i32),
    cell: Cell,
    bg: Cell,
) -> i64 {
    let l0_max = (l0_origin.0 + L0_DIM - 1, l0_origin.1 + L0_DIM - 1, l0_origin.2 + L0_DIM - 1);
    let slot_lo = (
        r_lo.0.max(l0_origin.0),
        r_lo.1.max(l0_origin.1),
        r_lo.2.max(l0_origin.2),
    );
    let slot_hi = (r_hi.0.min(l0_max.0), r_hi.1.min(l0_max.1), r_hi.2.min(l0_max.2));
    let fully_covers_l0 = slot_lo == l0_origin && slot_hi == l0_max;

    let slot_idx = L1Tile::idx(l1_axis.0, l1_axis.1, l1_axis.2);

    if fully_covers_l0 {
        // Replace the entire slot with Tile(cell). Compute the delta
        // from the prior slot's non-bg-cell contribution.
        let old_non_bg = count_non_bg_in_slot(&l1.slots[slot_idx], bg);
        let prev_was_bg = match &l1.slots[slot_idx] {
            L1Slot::Tile(v) => *v == bg,
            L1Slot::Child(l0) => l0.is_background(bg),
        };
        l1.slots[slot_idx] = L1Slot::Tile(cell);
        let now_bg = cell == bg;
        match (prev_was_bg, now_bg) {
            (true, false) => l1.non_bg_slot_count += 1,
            (false, true) => l1.non_bg_slot_count -= 1,
            _ => {}
        }
        let new_non_bg = if cell == bg { 0 } else { L0_SIZE as i64 };
        return new_non_bg - old_non_bg as i64;
    }

    // Partial coverage: materialize L0 Child if needed.
    let prev_was_bg = match &l1.slots[slot_idx] {
        L1Slot::Tile(v) => *v == bg,
        L1Slot::Child(l0) => l0.is_background(bg),
    };
    if let L1Slot::Tile(v) = &l1.slots[slot_idx] {
        let v_val = *v;
        l1.slots[slot_idx] = L1Slot::Child(Box::new(L0Tile::Constant(v_val)));
    }
    let L1Slot::Child(l0) = &mut l1.slots[slot_idx] else {
        unreachable!()
    };
    // Cell-by-cell write into the L0, accumulating active delta.
    let mut delta: i64 = 0;
    for cz in slot_lo.2..=slot_hi.2 {
        for cy in slot_lo.1..=slot_hi.1 {
            for cx in slot_lo.0..=slot_hi.0 {
                let ox = cx - l0_origin.0;
                let oy = cy - l0_origin.1;
                let oz = cz - l0_origin.2;
                let old = l0.put(ox, oy, oz, cell, bg);
                match (old == bg, cell == bg) {
                    (true, false) => delta += 1,
                    (false, true) => delta -= 1,
                    _ => {}
                }
            }
        }
    }
    let l0_bg = l0.is_background(bg);
    if l0_bg {
        l1.slots[slot_idx] = L1Slot::Tile(bg);
        if !prev_was_bg {
            l1.non_bg_slot_count -= 1;
        }
    } else if prev_was_bg {
        l1.non_bg_slot_count += 1;
    }
    delta
}

fn count_non_bg_in_slot(slot: &L1Slot, bg: Cell) -> usize {
    match slot {
        L1Slot::Tile(v) => {
            if *v == bg {
                0
            } else {
                L0_SIZE
            }
        }
        L1Slot::Child(l0) => match l0.as_ref() {
            L0Tile::Constant(v) => {
                if *v == bg {
                    0
                } else {
                    L0_SIZE
                }
            }
            L0Tile::Dense { non_bg_count, .. } => *non_bg_count as usize,
        },
    }
}

fn walk_l1_predicate<P: FnMut(Cell) -> bool>(
    l1: &L1Tile,
    root: RootCoord,
    lo: CellCoord,
    hi: CellCoord,
    pred: &mut P,
) -> bool {
    let root_lo = (root.0 * ROOT_DIM, root.1 * ROOT_DIM, root.2 * ROOT_DIM);
    let root_hi = (
        root_lo.0 + ROOT_DIM - 1,
        root_lo.1 + ROOT_DIM - 1,
        root_lo.2 + ROOT_DIM - 1,
    );
    let r_lo = (
        lo.x.max(root_lo.0),
        lo.y.max(root_lo.1),
        lo.z.max(root_lo.2),
    );
    let r_hi = (
        hi.x.min(root_hi.0),
        hi.y.min(root_hi.1),
        hi.z.min(root_hi.2),
    );

    let l1_lo = (
        (r_lo.0 >> L0_BITS) & L1_MASK,
        (r_lo.1 >> L0_BITS) & L1_MASK,
        (r_lo.2 >> L0_BITS) & L1_MASK,
    );
    let l1_hi = (
        (r_hi.0 >> L0_BITS) & L1_MASK,
        (r_hi.1 >> L0_BITS) & L1_MASK,
        (r_hi.2 >> L0_BITS) & L1_MASK,
    );

    for lz in l1_lo.2..=l1_hi.2 {
        for ly in l1_lo.1..=l1_hi.1 {
            for lx in l1_lo.0..=l1_hi.0 {
                let slot = &l1.slots[L1Tile::idx(lx, ly, lz)];
                match slot {
                    L1Slot::Tile(v) => {
                        if !pred(*v) {
                            return false;
                        }
                    }
                    L1Slot::Child(l0) => {
                        let l0_origin = (
                            root.0 * ROOT_DIM + lx * L0_DIM,
                            root.1 * ROOT_DIM + ly * L0_DIM,
                            root.2 * ROOT_DIM + lz * L0_DIM,
                        );
                        let l0_max = (
                            l0_origin.0 + L0_DIM - 1,
                            l0_origin.1 + L0_DIM - 1,
                            l0_origin.2 + L0_DIM - 1,
                        );
                        let s_lo = (
                            r_lo.0.max(l0_origin.0) - l0_origin.0,
                            r_lo.1.max(l0_origin.1) - l0_origin.1,
                            r_lo.2.max(l0_origin.2) - l0_origin.2,
                        );
                        let s_hi = (
                            r_hi.0.min(l0_max.0) - l0_origin.0,
                            r_hi.1.min(l0_max.1) - l0_origin.1,
                            r_hi.2.min(l0_max.2) - l0_origin.2,
                        );
                        match l0.as_ref() {
                            L0Tile::Constant(v) => {
                                if !pred(*v) {
                                    return false;
                                }
                            }
                            L0Tile::Dense { cells, .. } => {
                                for oz in s_lo.2..=s_hi.2 {
                                    for oy in s_lo.1..=s_hi.1 {
                                        for ox in s_lo.0..=s_hi.0 {
                                            if !pred(cells[L0Tile::idx(ox, oy, oz)]) {
                                                return false;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    true
}

fn walk_root_for_iter<F: FnMut(CellCoord, Cell)>(
    entry: &RootEntry,
    root: RootCoord,
    lo: CellCoord,
    hi: CellCoord,
    bg: Cell,
    visit: &mut F,
) {
    let root_lo = (root.0 * ROOT_DIM, root.1 * ROOT_DIM, root.2 * ROOT_DIM);
    let root_hi = (
        root_lo.0 + ROOT_DIM - 1,
        root_lo.1 + ROOT_DIM - 1,
        root_lo.2 + ROOT_DIM - 1,
    );
    let r_lo = (
        lo.x.max(root_lo.0),
        lo.y.max(root_lo.1),
        lo.z.max(root_lo.2),
    );
    let r_hi = (
        hi.x.min(root_hi.0),
        hi.y.min(root_hi.1),
        hi.z.min(root_hi.2),
    );
    match entry {
        RootEntry::Tile(v) => {
            if *v == bg {
                return;
            }
            expand_uniform_region(r_lo, r_hi, *v, visit);
        }
        RootEntry::Child(l1) => {
            let l1_lo = (
                (r_lo.0 >> L0_BITS) & L1_MASK,
                (r_lo.1 >> L0_BITS) & L1_MASK,
                (r_lo.2 >> L0_BITS) & L1_MASK,
            );
            let l1_hi = (
                (r_hi.0 >> L0_BITS) & L1_MASK,
                (r_hi.1 >> L0_BITS) & L1_MASK,
                (r_hi.2 >> L0_BITS) & L1_MASK,
            );
            for lz in l1_lo.2..=l1_hi.2 {
                for ly in l1_lo.1..=l1_hi.1 {
                    for lx in l1_lo.0..=l1_hi.0 {
                        let slot = &l1.slots[L1Tile::idx(lx, ly, lz)];
                        let l0_origin = (
                            root.0 * ROOT_DIM + lx * L0_DIM,
                            root.1 * ROOT_DIM + ly * L0_DIM,
                            root.2 * ROOT_DIM + lz * L0_DIM,
                        );
                        let l0_max = (
                            l0_origin.0 + L0_DIM - 1,
                            l0_origin.1 + L0_DIM - 1,
                            l0_origin.2 + L0_DIM - 1,
                        );
                        let s_lo = (
                            r_lo.0.max(l0_origin.0),
                            r_lo.1.max(l0_origin.1),
                            r_lo.2.max(l0_origin.2),
                        );
                        let s_hi = (
                            r_hi.0.min(l0_max.0),
                            r_hi.1.min(l0_max.1),
                            r_hi.2.min(l0_max.2),
                        );
                        match slot {
                            L1Slot::Tile(v) => {
                                if *v == bg {
                                    continue;
                                }
                                expand_uniform_region(s_lo, s_hi, *v, visit);
                            }
                            L1Slot::Child(l0) => match l0.as_ref() {
                                L0Tile::Constant(v) => {
                                    if *v == bg {
                                        continue;
                                    }
                                    expand_uniform_region(s_lo, s_hi, *v, visit);
                                }
                                L0Tile::Dense { cells, .. } => {
                                    for cz in s_lo.2..=s_hi.2 {
                                        for cy in s_lo.1..=s_hi.1 {
                                            for cx in s_lo.0..=s_hi.0 {
                                                let ox = cx - l0_origin.0;
                                                let oy = cy - l0_origin.1;
                                                let oz = cz - l0_origin.2;
                                                let cell = cells[L0Tile::idx(ox, oy, oz)];
                                                if cell == bg {
                                                    continue;
                                                }
                                                visit(CellCoord::new(cx, cy, cz), cell);
                                            }
                                        }
                                    }
                                }
                            },
                        }
                    }
                }
            }
        }
    }
}

/// Emit `(coord, value)` for every cell in the inclusive box `[lo, hi]`.
fn expand_uniform_region<F: FnMut(CellCoord, Cell)>(
    lo: (i32, i32, i32),
    hi: (i32, i32, i32),
    value: Cell,
    visit: &mut F,
) {
    for cz in lo.2..=hi.2 {
        for cy in lo.1..=hi.1 {
            for cx in lo.0..=hi.0 {
                visit(CellCoord::new(cx, cy, cz), value);
            }
        }
    }
}

/// Voxelize any parry3d shape that implements [`PointQuery`] into
/// `field`, marking cells whose centre is inside the shape with
/// `Cell::new(compartment, 0, 0)`. Works for analytical primitives
/// ([`parry3d::shape::Ball`], [`parry3d::shape::Capsule`],
/// [`parry3d::shape::Cuboid`]) directly, and for [`TriMesh`](parry3d::shape::TriMesh)
/// once the mesh has been configured for in/out queries — see
/// [`prepare_trimesh_for_voxelize`] for the recipe.
///
/// Only cells whose centre lies within `bounds` are considered —
/// typically pass `shape.local_aabb()` (with an optional dilation) to
/// skip empty space.
///
/// World-space here is identical to shape-local: the caller is expected
/// to either pre-transform the shape's vertices or transform `bounds`
/// and `field.origin()` consistently.
///
/// [`PointQuery`]: parry3d::query::PointQuery
pub fn voxelize_trimesh<S: parry3d::query::PointQuery + ?Sized>(
    field: &mut VoxelField,
    shape: &S,
    compartment: CompartmentId,
    bounds: Aabb,
) {
    let (lo, hi) = field.aabb_to_cell_range(bounds);
    if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
        return;
    }
    let value = Cell::new(compartment, 0, 0);
    for cz in lo.z..=hi.z {
        for cy in lo.y..=hi.y {
            for cx in lo.x..=hi.x {
                let coord = CellCoord::new(cx, cy, cz);
                let centre = field.cell_center(coord);
                if shape.contains_local_point(&centre) {
                    field.put(coord, value);
                }
            }
        }
    }
}

/// Configure a parry3d [`TriMesh`](parry3d::shape::TriMesh) for in/out
/// queries via [`voxelize_trimesh`]. parry3d's `TriMesh` doesn't
/// support point-in-mesh tests out of the box — it needs **oriented**
/// flag and pseudo-normals. This helper applies the standard flags
/// (`ORIENTED | DELETE_DEGENERATE_TRIANGLES | FIX_INTERNAL_EDGES`).
/// Use this on freshly-built or freshly-loaded triangle meshes before
/// voxelization.
pub fn prepare_trimesh_for_voxelize(
    mesh: &mut parry3d::shape::TriMesh,
) -> Result<(), parry3d::shape::TopologyError> {
    use parry3d::shape::TriMeshFlags;
    mesh.set_flags(
        TriMeshFlags::ORIENTED
            | TriMeshFlags::DELETE_DEGENERATE_TRIANGLES
            | TriMeshFlags::FIX_INTERNAL_EDGES,
    )
}

fn count_active_in_l1(l1: &L1Tile, bg: Cell) -> usize {
    let mut total = 0usize;
    for slot in l1.slots.iter() {
        match slot {
            L1Slot::Tile(v) => {
                if *v != bg {
                    total += L0_SIZE;
                }
            }
            L1Slot::Child(l0) => match l0.as_ref() {
                L0Tile::Constant(v) => {
                    if *v != bg {
                        total += L0_SIZE;
                    }
                }
                L0Tile::Dense { non_bg_count, .. } => {
                    total += *non_bg_count as usize;
                }
            },
        }
    }
    total
}

fn sample_in_sphere<R: Rng>(near: Point3<f32>, radius: f32, rng: &mut R) -> Point3<f32> {
    loop {
        let x: f32 = rng.gen_range(-1.0..=1.0);
        let y: f32 = rng.gen_range(-1.0..=1.0);
        let z: f32 = rng.gen_range(-1.0..=1.0);
        let r2 = x * x + y * y + z * z;
        if r2 <= 1.0 {
            return Point3::new(
                near.x + x * radius,
                near.y + y * radius,
                near.z + z * radius,
            );
        }
    }
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::collections::HashMap;
    use std::collections::HashSet;

    fn p(x: f32, y: f32, z: f32) -> Point3<f32> {
        Point3::new(x, y, z)
    }

    fn aabb(min: (f32, f32, f32), max: (f32, f32, f32)) -> Aabb {
        Aabb::new(p(min.0, min.1, min.2), p(max.0, max.1, max.2))
    }

    fn c(x: i32, y: i32, z: i32) -> CellCoord {
        CellCoord::new(x, y, z)
    }

    // ---------- coord math ----------

    #[test]
    fn root_coord_signs() {
        assert_eq!(cell_to_root_axis(0), 0);
        assert_eq!(cell_to_root_axis(127), 0);
        assert_eq!(cell_to_root_axis(128), 1);
        assert_eq!(cell_to_root_axis(-1), -1);
        assert_eq!(cell_to_root_axis(-128), -1);
        assert_eq!(cell_to_root_axis(-129), -2);
    }

    #[test]
    fn l1_sub_negative_coords() {
        let sub = l1_sub_indices(c(-1, -128, -129));
        assert_eq!(sub, (15, 0, 15));
    }

    #[test]
    fn l0_sub_negative_coords() {
        let sub = l0_sub_indices(c(-1, -8, -9));
        assert_eq!(sub, (7, 0, 7));
    }

    // ---------- basic point read/write ----------

    #[test]
    fn empty_returns_default() {
        let f = VoxelField::new(1.0);
        assert_eq!(f.get(c(0, 0, 0)), Cell::DEFAULT);
        assert_eq!(f.sample(p(123.4, -56.7, 89.0)), Cell::DEFAULT);
        assert_eq!(f.active_cells(), 0);
        assert!(f.is_empty());
    }

    #[test]
    fn put_and_get_round_trip() {
        let mut f = VoxelField::new(1.0);
        let cell = Cell::new(42, OCCUPIED | SURFACE, 17);
        f.put(c(3, 5, 7), cell);
        assert_eq!(f.get(c(3, 5, 7)), cell);
        assert_eq!(f.active_cells(), 1);
    }

    #[test]
    fn set_and_sample_round_trip() {
        let mut f = VoxelField::new(2.0);
        let target = p(5.0, 11.0, -3.0);
        let cell = Cell::new(1, OCCUPIED, 5);
        f.set(target, cell);
        assert_eq!(f.sample(target), cell);
        assert_eq!(f.sample(p(5.1, 11.2, -2.9)), cell);
    }

    #[test]
    fn writing_default_frees_everything() {
        let mut f = VoxelField::new(1.0);
        let cell = Cell::new(1, OCCUPIED, 0);
        f.put(c(0, 0, 0), cell);
        let s = f.stats();
        assert!(s.l1_tiles == 1, "L1 should be allocated");
        assert!(s.dense_l0_tiles == 1, "L0 should be Dense");
        // Overwrite with default
        f.put(c(0, 0, 0), Cell::DEFAULT);
        assert_eq!(f.active_cells(), 0);
        let s = f.stats();
        assert_eq!(s.l1_tiles, 0, "L1 should be removed");
        assert_eq!(s.dense_l0_tiles, 0);
    }

    #[test]
    fn negative_coords_work() {
        let mut f = VoxelField::new(1.0);
        let cell = Cell::new(2, SURFACE, 1);
        f.put(c(-200, -1, -129), cell);
        assert_eq!(f.get(c(-200, -1, -129)), cell);
        assert_eq!(f.get(c(-201, -1, -129)), Cell::DEFAULT);
    }

    #[test]
    fn cell_center_round_trip() {
        let f = VoxelField::with_origin(2.0, p(10.0, 20.0, 30.0));
        let coord = c(3, -5, 7);
        let centre = f.cell_center(coord);
        assert_eq!(f.point_to_cell(centre), coord);
    }

    // ---------- tile values: read paths ----------

    #[test]
    fn root_tile_read_short_circuits() {
        // Sparse-fill a whole root tile, verify reads return the value without
        // an L1 ever being allocated.
        let mut f = VoxelField::new(1.0);
        let cell = Cell::new(5, 0, 0);
        // 0..=127 is exactly root tile (0,0,0).
        f.mark_aabb(aabb((0.0, 0.0, 0.0), (128.0, 128.0, 128.0)), cell);
        let s = f.stats();
        assert_eq!(
            s.constant_root_tiles, 1,
            "fill should install a Tile root entry"
        );
        assert_eq!(s.l1_tiles, 0, "no L1 should be allocated");
        assert_eq!(s.dense_l0_tiles, 0);
        // Read should return cell.
        assert_eq!(f.get(c(0, 0, 0)), cell);
        assert_eq!(f.get(c(64, 64, 64)), cell);
        assert_eq!(f.get(c(127, 127, 127)), cell);
        // Outside the tile is bg.
        assert_eq!(f.get(c(128, 0, 0)), Cell::DEFAULT);
    }

    #[test]
    fn l1_slot_tile_read() {
        // Fill an L0-sized region but not a whole root.
        let mut f = VoxelField::new(1.0);
        let cell = Cell::new(3, 0, 0);
        f.mark_aabb(aabb((0.0, 0.0, 0.0), (8.0, 8.0, 8.0)), cell);
        let s = f.stats();
        assert_eq!(s.l1_tiles, 1, "L1 allocated");
        assert_eq!(
            s.dense_l0_tiles, 0,
            "L0 should not be allocated as Dense — slot is Tile"
        );
        assert_eq!(s.constant_l0_tiles, 0);
        assert_eq!(f.get(c(0, 0, 0)), cell);
        assert_eq!(f.get(c(7, 7, 7)), cell);
        assert_eq!(f.get(c(8, 8, 8)), Cell::DEFAULT);
    }

    // ---------- tile values: write paths ----------

    #[test]
    fn write_expands_root_tile() {
        let mut f = VoxelField::new(1.0);
        let a = Cell::new(1, 0, 0);
        let b = Cell::new(2, 0, 0);
        f.mark_aabb(aabb((0.0, 0.0, 0.0), (128.0, 128.0, 128.0)), a);
        assert_eq!(f.stats().constant_root_tiles, 1);
        // Single divergent put should expand to L1 with mostly Tile(a) slots
        // and one Dense slot.
        f.put(c(50, 50, 50), b);
        let s = f.stats();
        assert_eq!(s.constant_root_tiles, 0);
        assert_eq!(s.l1_tiles, 1);
        assert_eq!(s.dense_l0_tiles, 1, "one L0 with divergent cells");
        assert_eq!(f.get(c(50, 50, 50)), b);
        assert_eq!(f.get(c(0, 0, 0)), a);
        assert_eq!(f.get(c(127, 127, 127)), a);
    }

    #[test]
    fn collapse_back_to_constant_root() {
        // After filling whole root, then writing same value to one cell, then
        // overwriting back to bg-of-whole-root via mark_aabb, structure should
        // collapse.
        let mut f = VoxelField::new(1.0);
        let a = Cell::new(1, 0, 0);
        f.mark_aabb(aabb((0.0, 0.0, 0.0), (128.0, 128.0, 128.0)), a);
        f.put(c(50, 50, 50), Cell::new(2, 0, 0));
        // Now overwrite the whole root with bg via mark_aabb.
        f.mark_aabb(aabb((0.0, 0.0, 0.0), (128.0, 128.0, 128.0)), Cell::DEFAULT);
        let s = f.stats();
        assert_eq!(s.constant_root_tiles, 0);
        assert_eq!(s.l1_tiles, 0);
        assert_eq!(f.active_cells(), 0);
    }

    // ---------- per-grid background ----------

    #[test]
    fn nonzero_background_default_value() {
        let bg = Cell::new(7, 0, 100);
        let f = VoxelField::with_background(1.0, bg);
        assert_eq!(f.get(c(0, 0, 0)), bg);
        assert_eq!(f.get(c(999, -999, 0)), bg);
        assert_eq!(f.active_cells(), 0);
    }

    #[test]
    fn writing_background_is_noop() {
        let bg = Cell::new(7, 0, 0);
        let mut f = VoxelField::with_background(1.0, bg);
        f.put(c(5, 5, 5), bg);
        assert_eq!(f.active_cells(), 0);
        assert_eq!(f.stats().l1_tiles, 0);
    }

    #[test]
    fn write_then_revert_to_background() {
        let bg = Cell::new(7, 0, 0);
        let mut f = VoxelField::with_background(1.0, bg);
        let other = Cell::new(2, OCCUPIED, 0);
        f.put(c(0, 0, 0), other);
        assert_eq!(f.active_cells(), 1);
        assert!(f.stats().dense_l0_tiles > 0);
        f.put(c(0, 0, 0), bg);
        assert_eq!(f.active_cells(), 0);
        assert_eq!(f.stats().l1_tiles, 0);
        assert_eq!(f.get(c(0, 0, 0)), bg);
    }

    // ---------- sparse fill performance ----------

    #[test]
    fn mark_aabb_large_uniform_is_cheap() {
        // Fill a 1000³ box; should only touch a small number of tiles.
        let mut f = VoxelField::new(1.0);
        f.mark_aabb(aabb((0.0, 0.0, 0.0), (1000.0, 1000.0, 1000.0)), Cell::new(1, 0, 0));
        let s = f.stats();
        // 1000 / 128 ≈ 7.8, so 8 root tiles per axis = 512 root tiles touched.
        // Fully-covered roots are RootEntry::Tile (constant_root_tiles).
        // Edges may have L1Tile allocations.
        assert!(
            s.constant_root_tiles + s.l1_tiles <= 512,
            "expected <= 512 tiles touched, got constant_root={}, l1_tiles={}",
            s.constant_root_tiles,
            s.l1_tiles,
        );
        // Most should be constant_root_tiles since the box covers most roots fully.
        assert!(
            s.constant_root_tiles >= 200,
            "expected most roots fully covered, got {}",
            s.constant_root_tiles
        );
    }

    #[test]
    fn mark_aabb_partial_descends_to_l0() {
        // aabb (10..40) at cell_size 1.0 maps to cells 10..=39 (half-open max).
        let mut f = VoxelField::new(1.0);
        let cell = Cell::new(2, OCCUPIED, 0);
        f.mark_aabb(aabb((10.0, 10.0, 10.0), (40.0, 40.0, 40.0)), cell);
        assert_eq!(f.active_cells(), 30 * 30 * 30);
        assert_eq!(f.get(c(10, 10, 10)), cell);
        assert_eq!(f.get(c(39, 39, 39)), cell);
        assert_eq!(f.get(c(40, 40, 40)), Cell::DEFAULT);
        assert_eq!(f.get(c(9, 9, 9)), Cell::DEFAULT);
    }

    // ---------- region queries ----------

    #[test]
    fn is_region_default_prunes_correctly() {
        let mut f = VoxelField::new(1.0);
        assert!(f.is_region_default(aabb((-500.0, -500.0, -500.0), (500.0, 500.0, 500.0))));
        f.put(c(100, 100, 100), Cell::new(1, 0, 0));
        assert!(!f.is_region_default(aabb((0.5, 0.5, 0.5), (200.5, 200.5, 200.5))));
        assert!(f.is_region_default(aabb((-50.0, -50.0, -50.0), (50.0, 50.0, 50.0))));
    }

    #[test]
    fn is_region_default_with_root_tile() {
        let mut f = VoxelField::new(1.0);
        f.mark_aabb(aabb((0.0, 0.0, 0.0), (128.0, 128.0, 128.0)), Cell::new(1, 0, 0));
        assert!(!f.is_region_default(aabb((50.5, 50.5, 50.5), (60.5, 60.5, 60.5))));
        // Region outside the root tile is still default.
        assert!(f.is_region_default(aabb((130.0, 130.0, 130.0), (140.0, 140.0, 140.0))));
    }

    #[test]
    fn any_cell_with_flags_finds_occupied() {
        let mut f = VoxelField::new(1.0);
        f.put(c(50, 50, 50), Cell::new(1, OCCUPIED, 0));
        f.put(c(51, 51, 51), Cell::new(1, SURFACE, 0));
        assert!(f.any_cell_with_flags(aabb((0.0, 0.0, 0.0), (100.0, 100.0, 100.0)), OCCUPIED));
        assert!(f.any_cell_with_flags(aabb((0.0, 0.0, 0.0), (100.0, 100.0, 100.0)), SURFACE));
        assert!(!f.any_cell_with_flags(
            aabb((0.0, 0.0, 0.0), (100.0, 100.0, 100.0)),
            MEMBRANE_INNER
        ));
        assert!(!f.any_cell_with_flags(aabb((0.0, 0.0, 0.0), (10.0, 10.0, 10.0)), OCCUPIED));
    }

    #[test]
    fn is_region_free() {
        let mut f = VoxelField::new(1.0);
        let r = aabb((0.0, 0.0, 0.0), (10.0, 10.0, 10.0));
        assert!(f.is_region_free(r));
        f.put(c(5, 5, 5), Cell::new(1, OCCUPIED, 0));
        assert!(!f.is_region_free(r));
        f.put(c(5, 5, 5), Cell::new(1, SURFACE, 0));
        assert!(f.is_region_free(r));
    }

    // ---------- hierarchical activity ----------

    #[test]
    fn l0_l1_activity_flags() {
        let mut f = VoxelField::new(1.0);
        assert!(!f.is_l0_active(c(0, 0, 0)));
        assert!(!f.is_l1_active(c(0, 0, 0)));
        f.put(c(3, 4, 5), Cell::new(1, 0, 0));
        assert!(f.is_l0_active(c(0, 0, 0)));
        assert!(f.is_l1_active(c(0, 0, 0)));
        assert!(!f.is_l0_active(c(100, 100, 100)));
    }

    // ---------- iteration ----------

    #[test]
    fn iter_active_cells_visits_exactly_writes() {
        let mut f = VoxelField::new(1.0);
        let coords = [c(0, 0, 0), c(10, -3, 5), c(-50, 50, -50), c(200, 200, 200)];
        for &cc in &coords {
            f.put(cc, Cell::new(1, OCCUPIED, 0));
        }
        let mut seen: HashSet<CellCoord> = HashSet::new();
        f.iter_active_cells(|c, _| {
            seen.insert(c);
        });
        let expected: HashSet<CellCoord> = coords.iter().copied().collect();
        assert_eq!(seen, expected);
    }

    #[test]
    fn iter_active_cells_in_filters_by_aabb() {
        let mut f = VoxelField::new(1.0);
        for x in 0..50 {
            f.put(c(x, 0, 0), Cell::new(1, OCCUPIED, 0));
        }
        let mut count = 0usize;
        f.iter_active_cells_in(aabb((9.5, -0.5, -0.5), (20.5, 0.5, 0.5)), |_, _| count += 1);
        assert_eq!(count, 12);
    }

    #[test]
    fn iter_visits_constant_tiles() {
        let mut f = VoxelField::new(1.0);
        let cell = Cell::new(1, 0, 0);
        // Fill a full L0 region.
        f.mark_aabb(aabb((0.0, 0.0, 0.0), (8.0, 8.0, 8.0)), cell);
        let mut count = 0usize;
        f.iter_active_cells(|_, _| count += 1);
        assert_eq!(count, L0_SIZE, "should iterate every cell in the L0 tile");
    }

    // ---------- sampling ----------

    #[test]
    fn find_free_point_returns_free_cell() {
        let mut f = VoxelField::new(1.0);
        f.mark_occupied(aabb((-5.0, -5.0, -5.0), (5.0, 5.0, 5.0)));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        let pt = f.find_free_point(p(0.0, 0.0, 0.0), 50.0, &mut rng, 100);
        let pt = pt.expect("expected a free point");
        assert!(!f.sample(pt).is_occupied());
    }

    #[test]
    fn find_free_point_returns_none_when_all_occupied() {
        let mut f = VoxelField::new(1.0);
        f.mark_occupied(aabb((-100.0, -100.0, -100.0), (100.0, 100.0, 100.0)));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFADE);
        let pt = f.find_free_point(p(0.0, 0.0, 0.0), 10.0, &mut rng, 50);
        assert!(pt.is_none());
    }

    // ---------- prune ----------

    #[test]
    fn prune_collapses_uniform_dense() {
        let mut f = VoxelField::new(1.0);
        let v = Cell::new(1, 0, 0);
        // Set 512 distinct puts that happen to all equal v — forces Dense.
        for z in 0..8 {
            for y in 0..8 {
                for x in 0..8 {
                    f.put(c(x, y, z), v);
                }
            }
        }
        let before = f.stats();
        assert!(before.dense_l0_tiles >= 1);
        f.prune();
        let after = f.stats();
        assert_eq!(after.dense_l0_tiles, 0);
        // All cells should still be v.
        assert_eq!(f.get(c(3, 3, 3)), v);
    }

    // ---------- bounds ----------

    #[test]
    fn bounds_encloses_active_cells() {
        let mut f = VoxelField::new(1.0);
        f.put(c(-5, 3, 10), Cell::new(1, 0, 0));
        f.put(c(7, -2, 0), Cell::new(1, 0, 0));
        let b = f.bounds().expect("non-empty");
        assert!(b.contains_point(p(-4.5, 3.5, 10.5)));
        assert!(b.contains_point(p(7.5, -1.5, 0.5)));
        assert!(b.min.x <= -5.0 && b.max.x >= 8.0);
    }

    // ---------- stats / clear ----------

    #[test]
    fn stats_after_writes() {
        let mut f = VoxelField::new(1.0);
        f.put(c(0, 0, 0), Cell::new(1, 0, 0));
        f.put(c(100, 100, 100), Cell::new(1, 0, 0));
        f.put(c(200, 200, 200), Cell::new(1, 0, 0));
        let s = f.stats();
        assert_eq!(s.active_cells, 3);
        assert_eq!(s.dense_l0_tiles, 3);
        assert_eq!(s.l1_tiles, 2);
        assert!(s.memory_bytes > 0);
    }

    #[test]
    fn clear_empties() {
        let mut f = VoxelField::new(1.0);
        for x in 0..10 {
            f.put(c(x, 0, 0), Cell::new(1, 0, 0));
        }
        assert_eq!(f.active_cells(), 10);
        f.clear();
        assert_eq!(f.active_cells(), 0);
        assert!(f.is_empty());
        assert_eq!(f.stats().l1_tiles, 0);
    }

    // ---------- random stress ----------

    #[test]
    fn random_writes_match_hashmap_oracle() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xACE5);
        let mut f = VoxelField::new(1.0);
        let mut oracle: HashMap<CellCoord, Cell> = HashMap::new();
        for _ in 0..5000 {
            let coord = c(
                rng.gen_range(-200..200),
                rng.gen_range(-200..200),
                rng.gen_range(-200..200),
            );
            let cell = if rng.gen_range(0.0..1.0) < 0.2 {
                Cell::DEFAULT
            } else {
                Cell::new(rng.gen_range(1..10), OCCUPIED, rng.gen_range(0..255))
            };
            f.put(coord, cell);
            if cell == Cell::DEFAULT {
                oracle.remove(&coord);
            } else {
                oracle.insert(coord, cell);
            }
        }
        for (coord, expected) in &oracle {
            assert_eq!(f.get(*coord), *expected, "mismatch at {:?}", coord);
        }
        assert_eq!(f.active_cells(), oracle.len());
    }

    #[test]
    fn random_writes_then_prune_then_reread() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFEED);
        let mut f = VoxelField::new(1.0);
        let mut oracle: HashMap<CellCoord, Cell> = HashMap::new();
        let v = Cell::new(1, OCCUPIED, 0);
        // 1500 random writes of the same value
        for _ in 0..1500 {
            let coord = c(
                rng.gen_range(-50..50),
                rng.gen_range(-50..50),
                rng.gen_range(-50..50),
            );
            f.put(coord, v);
            oracle.insert(coord, v);
        }
        f.prune();
        for (coord, expected) in &oracle {
            assert_eq!(f.get(*coord), *expected);
        }
        assert_eq!(f.active_cells(), oracle.len());
    }

    #[test]
    fn cross_root_tile_writes() {
        let mut f = VoxelField::new(1.0);
        f.mark_aabb(
            aabb((127.5, 127.5, 127.5), (129.5, 129.5, 129.5)),
            Cell::new(1, 0, 0),
        );
        assert_eq!(f.active_cells(), 27);
    }

    // ---------- mesh voxelization ----------

    #[test]
    fn voxelize_ball_interior() {
        // Use the analytical Ball directly — it implements PointQuery and
        // doesn't need the TriMesh orientation dance.
        let mut f = VoxelField::new(0.25);
        let ball = parry3d::shape::Ball::new(1.0);
        let bounds = aabb((-1.5, -1.5, -1.5), (1.5, 1.5, 1.5));
        voxelize_trimesh(&mut f, &ball, 1, bounds);
        assert_eq!(
            f.sample(p(0.0, 0.0, 0.0)).compartment,
            1,
            "centre of ball should be interior"
        );
        assert_eq!(f.sample(p(1.4, 1.4, 1.4)).compartment, 0);
        // (4/3)π r³ ≈ 4.19; cells of size 0.25³ ≈ 0.0156; n ≈ 268.
        let n = f.active_cells();
        assert!(
            (200..350).contains(&n),
            "expected ~268 active cells, got {n}"
        );
    }

    #[test]
    fn voxelize_respects_bounds() {
        let mut f = VoxelField::new(1.0);
        let ball = parry3d::shape::Ball::new(5.0);
        let bounds = aabb((0.0, 0.0, 0.0), (5.0, 5.0, 5.0));
        voxelize_trimesh(&mut f, &ball, 7, bounds);
        assert_eq!(f.sample(p(1.0, 1.0, 1.0)).compartment, 7);
        assert_eq!(f.sample(p(-1.0, -1.0, -1.0)).compartment, 0);
    }

    #[test]
    fn voxelize_trimesh_with_oriented_flag() {
        // A real TriMesh — verify that prepare_trimesh_for_voxelize
        // enables in/out queries.
        let ball = parry3d::shape::Ball::new(2.0);
        let (vertices, indices) = ball.to_trimesh(20, 20);
        let mut mesh = parry3d::shape::TriMesh::new(vertices, indices).expect("trimesh");
        prepare_trimesh_for_voxelize(&mut mesh).expect("orient");

        let mut f = VoxelField::new(0.25);
        voxelize_trimesh(
            &mut f,
            &mesh,
            3,
            aabb((-2.5, -2.5, -2.5), (2.5, 2.5, 2.5)),
        );
        assert_eq!(
            f.sample(p(0.0, 0.0, 0.0)).compartment,
            3,
            "centre of trimesh ball should be interior"
        );
        // A point just outside the radius should be default.
        assert_eq!(f.sample(p(2.4, 0.0, 0.0)).compartment, 0);
    }

    #[test]
    fn voxelized_large_ball_compacts_under_prune() {
        // A large ball voxelized cell-by-cell produces many Dense L0 tiles.
        // After prune(), interior 8³ tiles (uniformly compartment-1) get
        // promoted to L1Slot::Tile, leaving only the boundary band as Dense.
        let mut f = VoxelField::new(1.0);
        let ball = parry3d::shape::Ball::new(30.0);
        voxelize_trimesh(
            &mut f,
            &ball,
            1,
            aabb((-32.0, -32.0, -32.0), (32.0, 32.0, 32.0)),
        );
        let before = f.stats();
        f.prune();
        let after = f.stats();
        // Curved surface → most tiles straddle the boundary. We expect ~30-40%
        // of dense tiles to collapse (fully-interior tiles only).
        assert!(
            after.dense_l0_tiles < before.dense_l0_tiles,
            "prune should reduce dense tile count; before={}, after={}",
            before.dense_l0_tiles,
            after.dense_l0_tiles
        );
        let collapsed = before.dense_l0_tiles - after.dense_l0_tiles;
        assert!(
            after.tile_l1_slots >= collapsed,
            "every collapsed Dense tile should become an L1Slot::Tile"
        );
        // Sanity: queries still work after prune.
        assert!(f.sample(p(0.0, 0.0, 0.0)).compartment == 1);
    }
}
