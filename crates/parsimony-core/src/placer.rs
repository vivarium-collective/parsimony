//! Greedy random placer. For each placement directive, pick a candidate
//! point and attempt to place an instance there if no collision; repeat
//! until every directive is either done or stuck.
//!
//! cellPACK's algorithm, simplified along three axes:
//!
//! - **Per-directive valid-cell lists.** Each Interior directive owns
//!   a `Vec<u32>` of grid-cell indices where this ingredient's
//!   enclosing radius currently fits (cell clearance ≥ required, cell
//!   centre fits the compartment, not inside a child compartment).
//!   Sampling picks a random index from this list; stale entries (cells
//!   whose clearance dropped since the last rebuild) get swap-removed.
//!   When the list empties and rebuilding doesn't refill it, the
//!   directive is stuck. This is cellPACK's `allIngrPts` mechanism.
//!
//! - **Sphere-tree collision** via QBVH broad-phase plus exact
//!   centre-distance vs sum-of-radii in the inner loop. Multi-sphere
//!   ingredients (ribosomes, etc.) carry every proxy sphere through.
//!
//! - **Surface placement** falls back to uniform-random sampling on
//!   the compartment boundary, since cells on a 2D manifold don't map
//!   well onto a 3D clearance grid. A small consecutive-rejection
//!   counter detects when the surface is full.

use indexmap::IndexMap;
use nalgebra::{Point3, Quaternion, UnitQuaternion, Vector3};
use rand::{Rng, SeedableRng};
use serde::{Deserialize, Serialize};
use rand_xoshiro::Xoshiro256PlusPlus;

use parsimony_spatial::{Aabb, QbvhIndex, Sphere, SpatialIndex};

use crate::clearance_grid::ClearanceGrid;
use crate::compartment::{Compartment, align_to_normal};
use crate::octree::OccupancyOctree;
use crate::ingredient::{Ingredient, IngredientShape};
use crate::placement::{Placement, Snapshot};
use crate::recipe::{ChromosomeSpec, PackingMode, PlacementDirective, Recipe, RegionKind};

/// Uniform random rotation on SO(3) via Shoemake's method. Pure 3D
/// uniform — equiprobable orientation, no Euler-angle biasing.
/// Random in-plane roll about a surface normal. Applied to tiled patches (e.g.
/// lipid-membrane tiles) so neighbouring tiles don't share an orientation —
/// the rotation jitter is what breaks up the repeated-tile look.
fn random_roll<R: Rng>(
    rot: UnitQuaternion<f32>,
    n: nalgebra::Vector3<f32>,
    rng: &mut R,
) -> UnitQuaternion<f32> {
    if n.norm() < 1e-6 {
        return rot;
    }
    let ang: f32 = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
    UnitQuaternion::from_axis_angle(&nalgebra::Unit::new_normalize(n), ang) * rot
}

fn random_rotation<R: Rng>(rng: &mut R) -> UnitQuaternion<f32> {
    let u1: f32 = rng.gen_range(0.0..1.0);
    let u2: f32 = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
    let u3: f32 = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
    let s1 = (1.0 - u1).sqrt();
    let s2 = u1.sqrt();
    let q = Quaternion::new(s2 * u3.cos(), s1 * u2.sin(), s1 * u2.cos(), s2 * u3.sin());
    UnitQuaternion::new_normalize(q)
}

/// Partition a cell into `n` sub-regions for laying out multiple chromosomes.
/// A capsule is split into `n` shorter capsules along its long axis; a sphere
/// offsets copies along z so they don't fully overlap. Returns the origin-
/// centred sub-shape and its offset from the cell centre.
fn subdivide(
    shape: crate::fiber::CellShape,
    n: usize,
    k: usize,
) -> (crate::fiber::CellShape, nalgebra::Vector3<f32>) {
    use crate::fiber::CellShape;
    if n <= 1 {
        // Single chromosome: fill the whole cell (a spread-out nucleoid). The
        // raw self-avoiding walk drifts off-centre toward one pole; the caller
        // re-centres each chromosome's centre-of-mass onto this offset (here 0,
        // the cell centre), so the DNA stays spread but is centred — not lopsided.
        return (shape, nalgebra::Vector3::zeros());
    }
    match shape {
        CellShape::Capsule { half_len, radius, axis } => {
            // n daughters: centre each chromosome's COM at its soon-to-be
            // daughter's centre — the midpoint of that half of the cell,
            // ≈ ±reach/2 along the long axis (reach = half_len + radius =
            // pole-to-centre distance). Each chromosome is generated spread across
            // a daughter-sized sub-capsule, then the caller pins its COM to this
            // offset, so each nucleoid reads as centred within its own daughter
            // with a clear midcell gap. (The previous version centred at
            // ±0.65·half_len, biased toward midcell rather than the daughter
            // centres.)
            let reach = half_len + radius;
            let t = -0.5 * reach + reach * (k as f32) / ((n - 1) as f32);
            let sub_half = (0.5 * reach / n as f32).max(radius * 0.3);
            let sub_radius = radius * 0.7;
            (CellShape::Capsule { half_len: sub_half, radius: sub_radius, axis }, axis * t)
        }
        CellShape::Sphere { radius } => {
            let spread = radius * 0.55;
            let z = -spread + 2.0 * spread * (k as f32) / ((n - 1) as f32);
            (CellShape::Sphere { radius: radius * 0.5 }, nalgebra::Vector3::new(0.0, 0.0, z))
        }
    }
}

/// Map a v2ecoli replication `domain_index` to a strand index into `n_strands`.
///
/// Phase-A simplification:
///   - `domain_index == 0`  → strand 0 (main chromosome)
///   - `domain_index  > 0`  → last strand (sister / replicated copy)
///
/// Clamped to `n_strands - 1` so callers may safely index into the slice.
/// Replication-domain topology (left vs right replichore, multi-fork index)
/// is refined in a later phase.
fn domain_index_to_strand(domain_index: i32, n_strands: usize) -> usize {
    if n_strands == 0 {
        return 0; // caller guards empty slice
    }
    if domain_index == 0 {
        0
    } else {
        n_strands - 1
    }
}

/// Map a v2ecoli genomic coordinate (signed bp, oriC = 0) to a 3D point and
/// unit tangent on the rendered chromosome strand selected by `domain_index`.
///
/// Returns `None` when `strands` is empty.  A single-bead strand returns that
/// bead with an `+x` unit tangent (degenerate but safe).
///
/// **Mapping:**
/// ```text
/// frac = (0.5 + coordinate / genome_len_bp).rem_euclid(1.0)
/// ```
/// oriC (coordinate = 0) maps to fraction 0.5, i.e. the midpoint of the
/// rendered strand.  The result is the nearest bead by rounded index.
pub fn strand_point(
    strands: &[Vec<Point3<f32>>],
    domain_index: i32,
    coordinate: i64,
    genome_len_bp: u32,
) -> Option<(Point3<f32>, Vector3<f32>)> {
    if strands.is_empty() {
        return None;
    }
    let si = domain_index_to_strand(domain_index, strands.len());
    let strand = &strands[si];
    if strand.is_empty() {
        return None;
    }
    if strand.len() == 1 {
        return Some((strand[0], Vector3::x()));
    }

    // oriC (coordinate = 0) maps to fraction 0.5 (strand midpoint).
    let frac = (0.5_f64 + coordinate as f64 / genome_len_bp as f64).rem_euclid(1.0) as f32;
    let last = (strand.len() - 1) as f32;
    let idx = (frac * last).round() as usize;
    let idx = idx.min(strand.len() - 1);

    let point = strand[idx];

    // Tangent: forward segment, or backward segment at the very end.
    let tangent = if idx + 1 < strand.len() {
        strand[idx + 1] - strand[idx]
    } else {
        strand[idx] - strand[idx - 1]
    };
    let norm = tangent.norm();
    let tangent = if norm > 1e-9 { tangent / norm } else { Vector3::x() };

    Some((point, tangent))
}

/// Map a genomic coordinate (signed bp, oriC = 0) to a 3D point and unit
/// tangent on the *sister* (replication-bubble) strand.
///
/// The bubble spans `[-fork_bp, +fork_bp]` where
/// `fork_bp = fork_fraction × (genome_len_bp / 2)`.  The coordinate is
/// linearly mapped onto `[0, 1]` over that range and clamped:
///
/// ```text
/// frac = ((coordinate + fork_bp) / (2 · fork_bp)).clamp(0, 1)
/// idx  = round(frac × (sister.len() - 1))
/// ```
///
/// oriC (coordinate = 0) maps to fraction 0.5 (the sister midpoint).
/// `+fork_bp` maps to the far end; `-fork_bp` maps to the near end.
/// Coordinates outside `[-fork_bp, fork_bp]` are clamped (no panic).
///
/// Returns `None` when `sister.len() < 2` or `fork_bp <= 0`.
pub fn bubble_point(
    sister: &[Point3<f32>],
    coordinate: i64,
    fork_fraction: f32,
    genome_len_bp: u32,
) -> Option<(Point3<f32>, Vector3<f32>)> {
    let fork_bp = fork_fraction * (genome_len_bp as f32 / 2.0);
    if sister.len() < 2 || fork_bp <= 0.0 {
        return None;
    }

    let frac = ((coordinate as f32 + fork_bp) / (2.0 * fork_bp)).clamp(0.0, 1.0);
    let last = (sister.len() - 1) as f32;
    let idx = (frac * last).round() as usize;
    let idx = idx.min(sister.len() - 1);

    let point = sister[idx];

    // Tangent: forward segment, or backward segment at the very end.
    let tangent = if idx + 1 < sister.len() {
        sister[idx + 1] - sister[idx]
    } else {
        sister[idx] - sister[idx - 1]
    };
    let norm = tangent.norm();
    let tangent = if norm > 1e-9 { tangent / norm } else { Vector3::x() };

    Some((point, tangent))
}

/// Default E. coli K-12 genome length in base pairs. Used when no genome CSV
/// is configured on the chromosome spec but explicit RNAP coordinates must
/// be mapped onto the strand.
const GENOME_BP_DEFAULT: u32 = 4_641_652;

/// Orientation quaternion seating an ingredient's `+x` reference axis onto
/// `dir` (a strand tangent, flipped for reverse-strand RNAPs). When `dir` is
/// antiparallel to `+x`, `UnitQuaternion::rotation_between` returns `None`
/// (the rotation axis is undefined); identity would then wrongly leave the
/// molecule pointing at `+x`, so the fallback is a real 180° turn about `+y`,
/// mapping `+x` onto `-x`.
fn orient_x_onto(dir: Vector3<f32>) -> UnitQuaternion<f32> {
    UnitQuaternion::rotation_between(&Vector3::x(), &dir).unwrap_or_else(|| {
        UnitQuaternion::from_axis_angle(&Vector3::y_axis(), std::f32::consts::PI)
    })
}

/// Cap on consecutive surface-placement rejections before a Surface
/// directive is declared stuck. Surface placements use uniform random
/// sampling on the compartment boundary (no per-cell filtering), so the
/// cap needs to be generous enough to survive transient crowding.
const SURFACE_REJECTION_CAP: u32 = 500;

/// Consecutive proxy-fit misses before the densify phase gives up on an
/// ingredient (the cell is saturated at proxy density for it).
const DENSIFY_FAIL_CAP: u32 = 2000;

/// Cap on a directive's cached valid-cell list. A whole-cell recipe packs
/// hundreds of interior directives over a fine grid; uncapped, each list would
/// hold the entire (tens-of-millions-of-cells) compartment volume and the lists
/// together reach tens of GB — enough to OOM the machine before a single
/// placement. We instead keep at most this many cells per directive, which is
/// ample to sample placements from and is refilled from the live grid whenever
/// it empties. Across hundreds of directives this bounds the lists to ~1 GB
/// total instead of tens of GB; ordinary recipes hold fewer valid cells than
/// this and are unaffected (and stay bit-for-bit reproducible — see
/// [`build_valid_cells_for`]).
const MAX_VALID_CELLS: usize = 500_000;

/// Try budget for the rejection-sampling fast path in [`build_valid_cells_for`],
/// as a multiple of [`MAX_VALID_CELLS`]. Sampling fills the cap in ~cap/density
/// tries, so this lets it succeed down to ~1/8 valid-cell density before
/// falling back to a full scan (the better tool once the grid is that crowded).
const REJECTION_TRY_BUDGET: usize = 8;

/// Consecutive placement misses (since the last success) before the octree
/// backend abandons a directive — the compartment is saturated for it.
const OCTREE_FAIL_CAP: u32 = 1000;

/// Which interior-placement engine the placer uses.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PlacementBackend {
    /// Dense clearance grid + per-directive valid-cell lists (cellPACK-style).
    /// Volume-scaled; the original engine, kept as the default and as a fallback
    /// for recipes that still want its exact behaviour.
    #[default]
    Legacy,
    /// Sparse occupancy octree, built incrementally and shared across the main
    /// pass and densify (which collapse into one proxy-accurate loop). Cost
    /// scales with placed content, not box volume — the engine for whole-cell
    /// recipes. See [`crate::octree`].
    Octree,
}

#[derive(Debug, Clone, Copy)]
pub struct PlacerConfig {
    /// Hard cap on per-instance placement attempts; overrides the
    /// recipe's `jitter_attempts` when smaller (acts as a global ceiling).
    pub max_attempts_per_instance: u32,
    /// Default `jitter_attempts` for ingredients that don't specify one.
    pub default_jitter_attempts: u32,
    /// Override for the clearance-grid cell size, in world units.
    /// `None` means autodetect from the recipe (largest ingredient
    /// radius divided by 8, clamped to ≥ 0.5).
    pub clearance_cell_size: Option<f32>,
    /// Whether the root compartment (the simulation bounding box) is
    /// a physical container that fully encloses every placement.
    /// Defaults to `true` (biology-correct: spheres entirely inside
    /// the box). Set to `false` for cellPACK-style "centre inside
    /// box, sphere may protrude at the edge" — used by the
    /// compare-with-cellpack bench so the apples-to-apples density
    /// match is recoverable. Named compartments (capsule, sphere,
    /// mesh) are always strict regardless of this flag.
    pub strict_bounds: bool,
    /// After the main (enclosing-sphere) pass, run a proxy-accurate densify
    /// phase that fills the remaining requested instances into the gaps the
    /// conservative enclosing-sphere clearance left — letting non-spherical
    /// meshes nestle until their actual proxy spheres touch. Off by default.
    pub densify: bool,
    /// Deterministic ceiling on total candidate attempts in the densify
    /// phase, summed across every interior directive. Densify is already
    /// bounded per-directive (it abandons an ingredient after
    /// `DENSIFY_FAIL_CAP` consecutive misses), but on a whole-cell recipe
    /// those give-up tails sum to tens of millions of attempts; this is the
    /// hard stop that keeps a `--densify` run from monopolising the machine.
    /// Counts attempts, not wall time, so packing stays bit-for-bit
    /// reproducible. When reached, densify stops and keeps what it placed.
    pub densify_max_attempts: u64,
    /// Which placement engine to use. [`PlacementBackend::Octree`] is
    /// content-scaled (for whole-cell recipes); [`PlacementBackend::Legacy`]
    /// (default) is the original grid+valid_cells engine.
    pub backend: PlacementBackend,
    /// Override the recipe's `chromosome.beads` (genome resolution). `None`
    /// uses the recipe value. More beads = more contour/volume + finer genome,
    /// at a heavier obstacle set for the interior pack.
    pub chromosome_beads: Option<usize>,
}

impl Default for PlacerConfig {
    fn default() -> Self {
        Self {
            max_attempts_per_instance: 200,
            default_jitter_attempts: 20,
            clearance_cell_size: None,
            strict_bounds: true,
            densify: false,
            densify_max_attempts: 20_000_000,
            backend: PlacementBackend::Legacy,
            chromosome_beads: None,
        }
    }
}

/// Result of one placer run.
#[derive(Debug, Clone)]
pub struct PlacerOutcome {
    pub snapshot: Snapshot,
    pub stats: PlacerStats,
}

#[derive(Debug, Clone, Default)]
pub struct PlacerStats {
    /// Total instances actually placed (== `snapshot.placements.len()`).
    pub placed: usize,
    /// Total instances requested across all directives.
    pub requested: usize,
    /// Per-ingredient `(placed, requested, total_attempts)` rows.
    pub per_ingredient: Vec<(String, usize, usize, u64)>,
    /// Total placement attempts (across all instances).
    pub total_attempts: u64,
    /// Total successful placements.
    pub successful_attempts: u64,
}

impl PlacerStats {
    pub fn requested_minus_placed(&self) -> usize {
        self.requested.saturating_sub(self.placed)
    }
}

/// The placer.
pub struct GreedyRandomPlacer<'a> {
    recipe: &'a Recipe,
    config: PlacerConfig,
    ingredient_ids: IndexMap<String, u32>,
    compartment_ids: IndexMap<String, u32>,
}

impl<'a> GreedyRandomPlacer<'a> {
    pub fn new(recipe: &'a Recipe, config: PlacerConfig) -> Self {
        let ingredient_ids: IndexMap<String, u32> = recipe
            .ingredients
            .keys()
            .enumerate()
            .map(|(i, k)| (k.clone(), i as u32))
            .collect();
        let compartment_ids: IndexMap<String, u32> = recipe
            .compartments
            .keys()
            .enumerate()
            .map(|(i, k)| (k.clone(), i as u32))
            .collect();
        Self {
            recipe,
            config,
            ingredient_ids,
            compartment_ids,
        }
    }

    pub fn pack(&self, seed: u64) -> PlacerOutcome {
        self.pack_with_obstacles(seed, &[])
    }

    /// Like [`pack`](Self::pack), but seeds the clearance grid with a set
    /// of pre-existing world-space obstacle spheres before packing — so
    /// this run's interior placements avoid geometry produced by an
    /// earlier stage (e.g. the chromosome). Used by the staged pipeline
    /// to pack the interior *around* a fixed chromosome. Obstacles enter
    /// the clearance grid, which governs Interior candidate cells; tiled
    /// Surface placements (the lipid bilayer) self-avoid and are
    /// unaffected.
    pub fn pack_with_obstacles(
        &self,
        seed: u64,
        obstacles: &[(Point3<f32>, f32)],
    ) -> PlacerOutcome {
        match self.config.backend {
            PlacementBackend::Legacy => self.pack_legacy(seed, obstacles),
            PlacementBackend::Octree => self.pack_octree(seed, obstacles),
        }
    }

    /// The original grid + valid-cells engine ([`PlacementBackend::Legacy`]).
    fn pack_legacy(
        &self,
        seed: u64,
        obstacles: &[(Point3<f32>, f32)],
    ) -> PlacerOutcome {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut snapshot = Snapshot::new(self.recipe.name.clone(), seed);
        let mut index: QbvhIndex = QbvhIndex::new();
        // Per-placement records, parallel to `index` and `snapshot.placements`,
        // indexed by UID. Holds shape reference (borrowed from recipe),
        // center, and rotation — everything collision-checking needs.
        let mut shapes_by_uid: Vec<&IngredientShape> = Vec::new();
        let mut centers_by_uid: Vec<Point3<f32>> = Vec::new();
        let mut rotations_by_uid: Vec<UnitQuaternion<f32>> = Vec::new();
        let mut next_uid: u64 = 0;
        let mut stats = PlacerStats::default();

        let max_required_radius = self
            .recipe
            .ingredients
            .values()
            .map(|i| i.shape.enclosing_radius())
            .fold(0.0_f32, f32::max);
        let cell_size = self.config.clearance_cell_size.unwrap_or_else(|| {
            // Cell size = max ingredient radius / 8. Smallest ingredient
            // sees ~1 cell of clearance, biggest ~8 — enough resolution
            // to filter candidates by size. `ClearanceGrid::new` will
            // raise this if needed to keep memory bounded.
            (max_required_radius / 8.0).max(0.5)
        });
        let mut clearance = ClearanceGrid::new(self.recipe.bounding_box, cell_size);

        // Seed upstream-stage obstacles into the clearance grid so this
        // run's interior candidate cells that overlap them are rejected
        // up front (the grid is authoritative for Interior placement).
        for &(c, r) in obstacles {
            clearance.update_for_placement(c, r, max_required_radius);
        }

        let directives: Vec<&PlacementDirective> = self.recipe.directives.iter().collect();
        let mut remaining: Vec<u32> = directives.iter().map(|d| d.count).collect();
        let mut per_ingredient_attempts: Vec<u64> = vec![0; directives.len()];
        let mut per_ingredient_placed: Vec<usize> = vec![0; directives.len()];
        let mut surface_rejections: Vec<u32> = vec![0; directives.len()];
        let mut stuck: Vec<bool> = vec![false; directives.len()];
        let total_requested: u32 = remaining.iter().sum();
        stats.requested = total_requested as usize;

        // Per-directive valid-cell lists (cellPACK's `allIngrPts`).
        // Initial pass: scan the compartment AABB and keep cells where
        // the ingredient's enclosing sphere fits with `radius`
        // clearance from every forbidden surface. Empty for Surface
        // directives.
        let mut valid_cells: Vec<Vec<u32>> = directives
            .iter()
            .map(|d| self.build_valid_cells(d, &clearance, &mut rng))
            .collect();

        loop {
            // Uniform-random pick over live directives — matches
            // cellPACK's default `pickIngredient`. Weighting by count
            // would starve big ingredients of attempts while filling
            // with abundant small ones; uniform gives every ingredient
            // a fair share throughout the run.
            let live: Vec<usize> = (0..directives.len())
                .filter(|&i| remaining[i] > 0 && !stuck[i])
                .collect();
            if live.is_empty() {
                break;
            }
            let dir_idx = live[rng.gen_range(0..live.len())];

            let directive = directives[dir_idx];
            let ingredient = self.recipe.ingredients.get(&directive.ingredient).unwrap();
            let compartment = self.recipe.compartments.get(&directive.compartment).unwrap();
            let enclosing_radius = ingredient.shape.enclosing_radius();

            per_ingredient_attempts[dir_idx] += 1;
            stats.total_attempts += 1;

            let (pos, rotation) = match directive.region {
                RegionKind::Interior => {
                    let children_of_compartment: Vec<&Compartment> = compartment
                        .children
                        .iter()
                        .filter_map(|&id| {
                            self.recipe.compartments.get_index(id as usize).map(|(_, c)| c)
                        })
                        .collect();
                    let mut pos = sample_from_valid_cells(
                        &mut valid_cells[dir_idx],
                        &clearance,
                        compartment,
                        &children_of_compartment,
                        enclosing_radius,
                        self.config.strict_bounds,
                        &mut rng,
                    );
                    if pos.is_none() {
                        // List empty: rebuild once before giving up.
                        // Lazy stale-removal during sampling keeps the
                        // list pruned across placements, but on
                        // emptiness we do a full pass — catches cells
                        // we never sampled directly.
                        valid_cells[dir_idx] = self.build_valid_cells(directive, &clearance, &mut rng);
                        pos = sample_from_valid_cells(
                            &mut valid_cells[dir_idx],
                            &clearance,
                            compartment,
                            &children_of_compartment,
                            enclosing_radius,
                            self.config.strict_bounds,
                            &mut rng,
                        );
                    }
                    let Some(pos) = pos else {
                        stuck[dir_idx] = true;
                        continue;
                    };
                    let rot = if ingredient.shape.needs_rotation() {
                        random_rotation(&mut rng)
                    } else {
                        UnitQuaternion::identity()
                    };
                    (pos, rot)
                }
                RegionKind::Surface => {
                    // Tiled ingredients (e.g. a lipid bilayer) walk an
                    // even Fibonacci point set instead of random sampling
                    // + rejection — dense and O(count). Each attempt takes
                    // the next point; once the set is exhausted the layer
                    // is laid. Non-tileable compartments fall back to
                    // random sampling.
                    let (p, n) = if matches!(ingredient.packing_mode, PackingMode::Tiled) {
                        let idx = per_ingredient_attempts[dir_idx] - 1; // 0-based this attempt
                        if idx >= directive.count as u64 {
                            stuck[dir_idx] = true;
                            continue;
                        }
                        match compartment.kind.surface_point_fibonacci(
                            idx,
                            directive.count as u64,
                            &mut rng,
                        ) {
                            Some(pn) => pn,
                            None => compartment.kind.sample_surface(&mut rng),
                        }
                    } else {
                        compartment.kind.sample_surface(&mut rng)
                    };
                    let mut rot = align_to_normal(ingredient.principal_vector, n);
                    if matches!(ingredient.packing_mode, PackingMode::Tiled) {
                        rot = random_roll(rot, n, &mut rng);
                    }
                    (p, rot)
                }
            };

            // Surface placements don't go through the clearance grid,
            // so we still need a strict QBVH collision check for
            // those. Interior placements were picked from a cell whose
            // clearance ≥ radius and jittered within the cell's
            // slack — the grid + slack-bounded jitter mathematically
            // guarantee no overlap, so we skip the QBVH check. Tiled
            // surface layers (e.g. a lipid bilayer) self-avoid by
            // construction (even Fibonacci spacing), so they skip the
            // check — and the QBVH insert + clearance update below — to
            // stay O(count) instead of O(count^2) at high density.
            let tiled = matches!(ingredient.packing_mode, PackingMode::Tiled);
            if !tiled
                && matches!(directive.region, RegionKind::Surface)
                && self.collides_with_existing(
                    &ingredient.shape,
                    pos,
                    rotation,
                    &index,
                    &shapes_by_uid,
                    &centers_by_uid,
                    &rotations_by_uid,
                )
            {
                surface_rejections[dir_idx] = surface_rejections[dir_idx].saturating_add(1);
                if surface_rejections[dir_idx] >= SURFACE_REJECTION_CAP {
                    stuck[dir_idx] = true;
                }
                continue;
            }

            // Place it.
            let uid = next_uid;
            next_uid += 1;
            let candidate_aabb = Aabb::from_sphere(pos, enclosing_radius);
            // Tiled layers stay out of the QBVH + clearance grid (they
            // self-avoid and form a thin decorative shell). We still push
            // the per-uid arrays so they stay aligned by uid — those
            // entries simply never get queried since they're not indexed.
            if !tiled {
                index.insert(uid, candidate_aabb).expect("uid collision");
            }
            shapes_by_uid.push(&ingredient.shape);
            centers_by_uid.push(pos);
            rotations_by_uid.push(rotation);

            // One pass marks both occupancy (clearance = 0 inside each
            // sphere) and distance for cells in range. Every proxy
            // sphere of the placed ingredient gets its own update so
            // multi-sphere ingredients are tracked accurately.
            if !tiled {
                for (c, r) in ingredient.shape.world_spheres(pos, rotation) {
                    clearance.update_for_placement(c, r, max_required_radius);
                }
            }
            snapshot.placements.push(Placement {
                instance_uid: uid,
                ingredient_id: self.ingredient_ids[&directive.ingredient],
                variant_id: 0,
                compartment_id: self.compartment_ids[&directive.compartment],
                position: pos,
                rotation,
            });
            remaining[dir_idx] -= 1;
            surface_rejections[dir_idx] = 0;
            per_ingredient_placed[dir_idx] += 1;
            stats.placed += 1;
            stats.successful_attempts += 1;
        }

        // Densify phase: fill the remaining requested instances using
        // proxy-accurate fit — each candidate's *actual* proxy spheres must
        // clear the grid, not its enclosing sphere — so non-spherical meshes
        // nestle until their shapes touch, far tighter than the main pass.
        //
        // Bounded two ways: each directive abandons its ingredient after
        // DENSIFY_FAIL_CAP consecutive misses, and the whole phase stops once
        // it has made `densify_max_attempts` candidates total. On a whole-cell
        // recipe the per-directive give-up tails alone sum to millions of
        // attempts, so the global budget is the real guard against a
        // `--densify` run monopolising the machine. The budget counts attempts
        // (not wall time), so packing stays bit-for-bit reproducible.
        if self.config.densify {
            let margin = clearance.cell_size; // grid-resolution safety margin
            let budget = self.config.densify_max_attempts;
            let mut densify_attempts = 0u64;
            'densify: for dir_idx in 0..directives.len() {
                let directive = directives[dir_idx];
                if !matches!(directive.region, RegionKind::Interior) || remaining[dir_idx] == 0 {
                    continue;
                }
                let ingredient = self.recipe.ingredients.get(&directive.ingredient).unwrap();
                let compartment = self.recipe.compartments.get(&directive.compartment).unwrap();
                let er = ingredient.shape.enclosing_radius();
                let needs_rot = ingredient.shape.needs_rotation();
                let bb = compartment.kind.aabb();
                let children: Vec<&Compartment> = compartment
                    .children
                    .iter()
                    .filter_map(|&id| {
                        self.recipe.compartments.get_index(id as usize).map(|(_, c)| c)
                    })
                    .collect();
                let mut consecutive_fail = 0u32;
                while remaining[dir_idx] > 0 && consecutive_fail < DENSIFY_FAIL_CAP {
                    if densify_attempts >= budget {
                        break 'densify;
                    }
                    densify_attempts += 1;
                    // Sample an interior point whose enclosing sphere is
                    // contained (so all proxies stay inside the compartment).
                    let mut sampled = None;
                    for _ in 0..32 {
                        let p = Point3::new(
                            rng.gen_range(bb.min.x..bb.max.x),
                            rng.gen_range(bb.min.y..bb.max.y),
                            rng.gen_range(bb.min.z..bb.max.z),
                        );
                        if compartment.kind.signed_distance(p) >= er
                            && children.iter().all(|c| -c.kind.signed_distance(p) >= er)
                        {
                            sampled = Some(p);
                            break;
                        }
                    }
                    let Some(pos) = sampled else {
                        consecutive_fail += 1;
                        continue;
                    };
                    let rot = if needs_rot {
                        random_rotation(&mut rng)
                    } else {
                        UnitQuaternion::identity()
                    };
                    per_ingredient_attempts[dir_idx] += 1;
                    stats.total_attempts += 1;
                    // Proxy-accurate fit: every proxy must clear the grid.
                    // Tested lazily over the (lazy) sphere iterator, so a mesh
                    // whose first proxy already clashes never pays to transform
                    // the rest — the dominant cost once the cell is crowded and
                    // most attempts fail.
                    let fits = ingredient
                        .shape
                        .world_spheres(pos, rot)
                        .all(|(c, r)| clearance.clearance_at(c) >= r + margin);
                    if fits {
                        let uid = next_uid;
                        next_uid += 1;
                        for (c, r) in ingredient.shape.world_spheres(pos, rot) {
                            clearance.update_for_placement(c, r, max_required_radius);
                        }
                        snapshot.placements.push(Placement {
                            instance_uid: uid,
                            ingredient_id: self.ingredient_ids[&directive.ingredient],
                            variant_id: 0,
                            compartment_id: self.compartment_ids[&directive.compartment],
                            position: pos,
                            rotation: rot,
                        });
                        remaining[dir_idx] -= 1;
                        per_ingredient_placed[dir_idx] += 1;
                        stats.placed += 1;
                        stats.successful_attempts += 1;
                        consecutive_fail = 0;
                    } else {
                        consecutive_fail += 1;
                    }
                }
            }
        }

        for (i, directive) in directives.iter().enumerate() {
            stats.per_ingredient.push((
                directive.ingredient.clone(),
                per_ingredient_placed[i],
                directive.count as usize,
                per_ingredient_attempts[i],
            ));
        }
        // Chromosome (if any): genome fiber + bound proteins, attached to the
        // snapshot. Shared with the octree backend.
        self.place_chromosome(&mut snapshot, &mut next_uid, &mut rng);

        PlacerOutcome { snapshot, stats }
    }

    /// Seat `positions.len()` copies of the `marker` ingredient (if named and
    /// present in the recipe) at the given `center`-relative positions — used
    /// for the chromosome landmark molecules (replisome, oriC, terC).
    fn seat_markers(
        &self,
        snapshot: &mut Snapshot,
        next_uid: &mut u64,
        marker: &Option<String>,
        positions: &[Point3<f32>],
        center: Point3<f32>,
    ) {
        let Some(name) = marker else { return };
        let Some((idx, _, _)) = self.recipe.ingredients.get_full(name) else {
            return;
        };
        for p in positions {
            snapshot.placements.push(Placement {
                instance_uid: *next_uid,
                ingredient_id: idx as u32,
                variant_id: 0,
                compartment_id: 0,
                position: center + p.coords,
                rotation: UnitQuaternion::identity(),
            });
            *next_uid += 1;
        }
    }

    /// Generate the recipe's chromosome fiber (plain or supercoiled) inside its
    /// cell compartment, bind its DNA-binding proteins along it (avoiding the
    /// interior already placed this run), and attach it to the snapshot. No-op
    /// when the recipe has no chromosome. Shared by both placement backends.
    fn place_chromosome<R: Rng>(
        &self,
        snapshot: &mut Snapshot,
        next_uid: &mut u64,
        rng: &mut R,
    ) {
        let Some(chr) = &self.recipe.chromosome else {
            return;
        };
        let Some((center, shape)) = self.chromosome_cell(chr) else {
            return;
        };
        // Genome resolution: recipe value unless overridden (e.g. `pack
        // --chromosome-beads`). More beads → more DNA contour/volume.
        // `beads` is the bead count *per chromosome*.
        let beads = self.config.chromosome_beads.unwrap_or(chr.beads);
        let n_chrom = chr.n_chromosomes.max(1);
        // All DNA strands + fork/oriC/terC positions, cell-centre-relative.
        let mut strands: Vec<Vec<Point3<f32>>> = Vec::new();
        let mut forks: Vec<Point3<f32>> = Vec::new();
        let mut orics: Vec<Point3<f32>> = Vec::new();
        let mut ters: Vec<Point3<f32>> = Vec::new();
        // Per-chromosome strand groups (each group = one chromosome's strands).
        let mut chrom_groups: Vec<Vec<Vec<Point3<f32>>>> = Vec::new();
        if n_chrom == 1 && chr.fork_fraction <= 0.0 {
            // Unreplicated single chromosome: keep the supercoiled multi-domain
            // (rosette) nucleoid layout.
            let pts = match &chr.supercoil {
                Some(sc) => {
                    // Per-domain bead allocation: transcription-coupled (each
                    // plectoneme domain sized to its gene-cluster bp span) when a
                    // genome is set, else evenly split. `domains <= 1` → single
                    // global plectoneme.
                    let alloc: Vec<usize> = chr
                        .genome
                        .as_ref()
                        .filter(|_| sc.domains > 1)
                        .and_then(|p| crate::genome::Genome::from_csv(p).ok())
                        .map(|g| g.domain_bead_allocation(beads, sc.domains))
                        .unwrap_or_else(|| vec![(beads / sc.domains.max(1)).max(2); sc.domains.max(1)]);
                    crate::fiber::generate_nucleoid(
                        shape, &alloc, chr.spacing, chr.bead_radius, sc.radius, sc.pitch, rng,
                    )
                }
                None => crate::fiber::generate_fiber(shape, beads, chr.spacing, chr.bead_radius, rng),
            };
            chrom_groups.push(vec![pts]);
        } else {
            // One or more replicating chromosomes: lay each as a theta (θ)
            // structure in its own sub-region of the cell.
            let (sc_radius, sc_pitch) = chr
                .supercoil
                .as_ref()
                .map(|s| (s.radius, s.pitch))
                .unwrap_or((0.0, 0.0));
            for k in 0..n_chrom {
                let (sub_shape, sub_off) = subdivide(shape, n_chrom, k);
                let theta = crate::fiber::generate_theta_chromosome(
                    sub_shape, beads, chr.fork_fraction, chr.spacing, chr.bead_radius,
                    sc_radius, sc_pitch, rng,
                );
                // Translate strands to their sub-region (sub_off positions each
                // chromosome at its pole; centering is done inside
                // `generate_theta_chromosome`, so no further shift is needed).
                let group: Vec<Vec<Point3<f32>>> = theta
                    .strands
                    .into_iter()
                    .map(|mut s| {
                        for p in &mut s {
                            *p += sub_off;
                        }
                        s
                    })
                    .collect();
                // Per-chromosome landmark positions (sub_off applied).
                let chrom_forks: Vec<Point3<f32>> = theta
                    .forks
                    .into_iter()
                    .map(|mut fk| { fk += sub_off; fk })
                    .collect();
                let chrom_orics: Vec<Point3<f32>> = theta
                    .oric
                    .into_iter()
                    .map(|mut o| { o += sub_off; o })
                    .collect();
                let chrom_ters: Vec<Point3<f32>> = theta
                    .ter
                    .into_iter()
                    .map(|mut t| { t += sub_off; t })
                    .collect();
                // Centering is now done at the source: `generate_theta_chromosome`
                // recenters the main strand (and derives the sister from the
                // centred main), so adding `sub_off` above already places each
                // chromosome at its intended pole sub-region.  The prior rigid
                // `shift = sub_off − centroid` + `medial()` clamp has been
                // removed: it caused the centerline-collapse artifact (beads
                // projected to the x-axis in cap regions via `medial()`).
                forks.extend(chrom_forks);
                orics.extend(chrom_orics);
                ters.extend(chrom_ters);
                chrom_groups.push(group);
            }
        }
        // Map chromosome k → (main_strand_idx, sister_strand_idx) in the flat list.
        // Built before flattening so both the RNAP loop and RNA loop can share it.
        // For chromosome k, main = its first strand's flat index; sister = the
        // second strand's index if the group has ≥ 2 strands (theta/replicating),
        // else None (unreplicated). This avoids hardcoding "2k" — a chromosome
        // without a sister has only 1 strand in its group.
        let mut chrom_strand_idx: Vec<(usize, Option<usize>)> = Vec::new();
        {
            let mut flat = 0usize;
            for g in &chrom_groups {
                let main = flat;
                let sister = if g.len() >= 2 { Some(flat + 1) } else { None };
                chrom_strand_idx.push((main, sister));
                flat += g.len();
            }
        }
        // Flat list of every strand, for the rendered chromosome.
        for g in &chrom_groups {
            for s in g {
                strands.push(s.clone());
            }
        }
        let pts = strands.first().cloned().unwrap_or_default();
        // Parse the genome annotation once (shared by the fiber-protein packing
        // below and the explicit-RNAP loop further down — avoids a double parse).
        let genome = chr
            .genome
            .as_ref()
            .and_then(|p| crate::genome::Genome::from_csv(p).ok());
        // Pack DNA-binding proteins (RNAP, etc.) PER CHROMOSOME — each chromosome
        // is a full genome, so each gets its share. Packing them on the single
        // concatenated fiber instead piled them all onto whichever chromosome had
        // the longest contour (the others got none).
        if !chr.proteins.is_empty() && !chrom_groups.is_empty() {
            let obstacles: Vec<(Point3<f32>, f32)> = snapshot
                .placements
                .iter()
                .flat_map(|pl| {
                    let ing = self
                        .recipe
                        .ingredients
                        .get_index(pl.ingredient_id as usize)
                        .unwrap()
                        .1;
                    ing.shape.world_spheres(pl.position, pl.rotation)
                })
                .collect();
            let n_groups = chrom_groups.len() as u32;
            // When explicit RNAP placements are provided, exclude the rnap_marker
            // ingredient from the count-based random packing — it is handled
            // separately below by the explicit-RNAP loop.
            let skip_rnap_in_random = !chr.rnaps.is_empty();
            // Split each protein's total count across the chromosomes.
            let per_chrom: Vec<(String, u32)> = chr
                .proteins
                .iter()
                .filter(|(name, _)| {
                    !skip_rnap_in_random
                        || chr.rnap_marker.as_deref() != Some(name.as_str())
                })
                .map(|(name, c)| (name.clone(), (c / n_groups).max(1)))
                .collect();
            for group in &chrom_groups {
                let fiber_world: Vec<Point3<f32>> =
                    group.iter().flatten().map(|p| center + p.coords).collect();
                if fiber_world.len() < 2 {
                    continue;
                }
                // TODO: `shape` is origin-relative while `fiber_world` is
                // world-space (offset by `center`). Exact for the production
                // compartment (centred at the world origin), approximate for an
                // off-centre one. Carry `center` into `CellShape` to make exact.
                // With a genome annotation, seat proteins at real transcription /
                // replication sites; otherwise spread them randomly.
                let binds = match &genome {
                    Some(genome) => {
                        let abundances: Vec<(String, u32)> = self
                            .recipe
                            .directives
                            .iter()
                            .map(|d| (d.ingredient.clone(), d.count))
                            .collect();
                        let sites = genome.binding_sites(&per_chrom, &abundances, rng);
                        let mut at: Vec<(u32, &Ingredient, f32)> = Vec::new();
                        for ((name, _), fracs) in per_chrom.iter().zip(&sites) {
                            if let Some((idx, _, ing)) = self.recipe.ingredients.get_full(name) {
                                for &f in fracs {
                                    at.push((idx as u32, ing, f));
                                }
                            }
                        }
                        crate::fiber_pack::pack_on_fiber_at(&fiber_world, &at, &obstacles, chr.bead_radius, shape, rng)
                    }
                    None => {
                        let proteins: Vec<(u32, &Ingredient, u32)> = self
                            .resolve_fiber_proteins(chr)
                            .into_iter()
                            .filter(|(_, ing, _)| {
                                !skip_rnap_in_random
                                    || chr.rnap_marker.as_deref() != Some(ing.name.as_str())
                            })
                            .map(|(id, ing, c)| (id, ing, (c / n_groups).max(1)))
                            .collect();
                        crate::fiber_pack::pack_on_fiber(&fiber_world, &proteins, &obstacles, chr.bead_radius, shape, rng)
                    }
                };
                for b in binds {
                    snapshot.placements.push(Placement {
                        instance_uid: *next_uid,
                        ingredient_id: b.ingredient_id,
                        variant_id: 0,
                        compartment_id: 0,
                        position: b.position,
                        rotation: b.rotation,
                    });
                    *next_uid += 1;
                }
            }
        }
        // Explicit RNAP placement: seat every RNAP from the recipe at its real
        // genomic locus on the strand, oriented along (or against) the tangent,
        // and confined to the cell envelope. This supersedes the count-based
        // random packing for the rnap_marker ingredient (filtered above) when
        // `chr.rnaps` is non-empty.
        if !chr.rnaps.is_empty() {
            if let Some(rnap_name) = &chr.rnap_marker {
                if let Some((idx, _, ing)) = self.recipe.ingredients.get_full(rnap_name) {
                    // Genome length: CSV-parsed length when available, else E. coli default.
                    let glen = genome
                        .as_ref()
                        .map(|g| g.length_bp)
                        .filter(|&l| l > 0)
                        .unwrap_or(GENOME_BP_DEFAULT);
                    let er = ing.shape.enclosing_radius();
                    let inset = shape.inset(er);
                    for rnap in &chr.rnaps {
                        // BF2-3: route to chromosome_index's OWN main strand, not always 0.
                        // Clamp to valid range (single-chromosome recipes default to 0).
                        // max(0) before the usize cast so a stray negative index
                        // can't wrap to usize::MAX and route to the last chromosome.
                        let g = (rnap.chromosome_index.max(0) as usize)
                            .min(chrom_strand_idx.len().saturating_sub(1));
                        let (main_idx, sister_idx_opt) =
                            chrom_strand_idx.get(g).copied().unwrap_or((0, None));
                        // Map genomic coordinate → cell-centre-relative position + tangent
                        // on chromosome g's main strand.  We pass a 1-element slice so that
                        // `strand_point` (which selects domain 0 → strand[0]) operates on
                        // exactly the desired strand — DRY without changing its signature.
                        let Some(main_strand) = strands.get(main_idx) else {
                            continue;
                        };
                        let Some((strand_pt, tangent)) = strand_point(
                            std::slice::from_ref(main_strand),
                            0,
                            rnap.coordinates,
                            glen,
                        ) else {
                            continue;
                        };
                        // Convert to world space (strands are cell-centre-relative).
                        let world_pt = center + strand_pt.coords;

                        // Shared confine+orient+push logic for both main and bubble
                        // placements.  Captures `inset`, `center`, `idx` by value/ref;
                        // `snapshot` and `next_uid` through mutable reborrow.
                        //
                        // NEVER drop an RNAP: 1:1 true-abundance requires every
                        // entry to be rendered. If confinement fails (degenerate
                        // inset — proxy larger than the whole cell), fall back to
                        // the medial-axis projection of the strand point, which
                        // lies on the centerline and is always inside the inset
                        // (a medial point has distance 0 ≤ cap_radius). For a
                        // fully-degenerate inset that collapses to the cell centre.
                        let is_forward = rnap.is_forward;
                        let mut place_at = |wp: Point3<f32>, tan: Vector3<f32>| {
                            let pos = crate::fiber_pack::confine_center(
                                wp,
                                Vector3::y(),
                                Vector3::z(),
                                0.0,
                                0.0,
                                wp,
                                &inset,
                            )
                            .unwrap_or_else(|| {
                                let m = inset.medial(&wp);
                                if inset.contains(&m) { m } else { center }
                            });
                            // Orientation: rotate +x onto ±tangent depending on strand.
                            // `orient_x_onto` handles the antiparallel (`dir == -x`)
                            // case with a real 180° turn instead of a wrong identity.
                            let dir = if is_forward { tan } else { -tan };
                            let rot = orient_x_onto(dir);
                            snapshot.placements.push(Placement {
                                instance_uid: *next_uid,
                                ingredient_id: idx as u32,
                                variant_id: 0,
                                compartment_id: 0,
                                position: pos,
                                rotation: rot,
                            });
                            *next_uid += 1;
                        };

                        // Main placement (strand 0).
                        place_at(world_pt, tangent);

                        // Bubble overlay: daughter RNAPs also appear on the sister
                        // strand at their bubble-relative position.  BF2-3: the
                        // sister is chromosome g's OWN sister (sister_idx_opt),
                        // not always the last strand.  Backward-compat: honour
                        // domain_index != 0 as a pre-BF2 daughter signal alongside
                        // the new is_daughter flag so existing BF1 tests pass.
                        if rnap.is_daughter || rnap.domain_index != 0 {
                            if let Some(sister_idx) = sister_idx_opt {
                                if let Some(sister) = strands.get(sister_idx) {
                                    if let Some((bub_pt, bub_tan)) = bubble_point(
                                        sister,
                                        rnap.coordinates,
                                        chr.fork_fraction,
                                        glen,
                                    ) {
                                        let bub_world = center + bub_pt.coords;
                                        place_at(bub_world, bub_tan);
                                    }
                                }
                            }
                        }
                    }
                }
            }
        }
        // Nascent-RNA strands (B1-3): grow one confined strand per RnaSpec rooted
        // at its RNAP's genomic locus on the chromosome strand.
        //
        // Frame: chromosome strands are center-relative; `shape` is origin/center-
        // relative (production cell is origin-centered).  We compute the root via
        // `strand_point` (already center-relative) and grow in that same frame —
        // so `rna_strands` entries are center-relative, consistent with
        // `Chromosome::strands`.
        //
        // Constants:
        //   rna_step        = 40.0 Å  — fixed coarse-grain step for nascent RNA beads
        //   rna_bead_radius =  4.0 Å  — thin bead (smaller than DNA to reduce steric load)
        let glen = genome
            .as_ref()
            .map(|g| g.length_bp)
            .filter(|&l| l > 0)
            .unwrap_or(GENOME_BP_DEFAULT);
        // Fixed constants (documented above).
        let rna_step: f32 = 40.0;
        let rna_bead_radius: f32 = 4.0;
        for rna in &chr.rnas {
            // Bead count: extended contour length / step, clamped to ≥ 2.
            let bead_count = ((rna.length_nt as f32 * chr.rna_angstrom_per_nt) / rna_step)
                .round() as usize;
            let bead_count = bead_count.max(2);
            // BF2-3: per-chromosome routing — map chromosome_index to its own
            // main/sister strands (same clamped-index logic as the RNAP loop;
            // max(0) guards a stray negative index from wrapping via usize cast).
            let rna_g = (rna.chromosome_index.max(0) as usize)
                .min(chrom_strand_idx.len().saturating_sub(1));
            let (rna_main_idx, rna_sister_idx_opt) =
                chrom_strand_idx.get(rna_g).copied().unwrap_or((0, None));
            // Root: for nascent (is_free=false), strand_point gives a center-relative
            // point on the chromosome; for free (is_free=true), rejection-sample a
            // random interior point inside shape.inset(rna_bead_radius).
            let root = if rna.is_free {
                // Rejection-sample a uniformly random point inside the inset envelope.
                // We sample within the axis-aligned bounding box of the inset (a cube of
                // side 2*reach) and accept when the point is inside the capsule/sphere.
                // Up to 64 tries; fall back to Point3::origin() (cell centre) if none
                // accepted — this satisfies the 1:1 abundance constraint even in
                // degenerate cells where the inset collapses to zero volume.
                let inset = shape.inset(rna_bead_radius);
                let reach = inset.reach();
                let mut chosen = Point3::origin();
                for _ in 0..64 {
                    let candidate = Point3::new(
                        rng.gen_range(-reach..=reach),
                        rng.gen_range(-reach..=reach),
                        rng.gen_range(-reach..=reach),
                    );
                    if inset.contains(&candidate) {
                        chosen = candidate;
                        break;
                    }
                }
                chosen
            } else {
                // Nascent: root on chromosome rna_g's MAIN genome contour, matching
                // where its RNAP is placed (BF2-3 per-chromosome routing).
                strands.get(rna_main_idx)
                    .and_then(|s| strand_point(std::slice::from_ref(s), 0, rna.root_coordinate, glen))
                    .map(|(p, _)| p)
                    .unwrap_or_else(Point3::origin)
            };
            // Grow a confined self-avoiding walk from the root inside `shape`.
            let points = crate::fiber::generate_rna_strand(
                root,
                bead_count,
                rna_step,
                rna_bead_radius,
                shape,
                rng,
            );
            snapshot.rna_strands.push(crate::placement::RnaStrand {
                points,
                is_mrna: rna.is_mRNA,
                is_free: rna.is_free,
                unique_index: rna.unique_index,
                length_nt: rna.length_nt,
            });
            // BF2-3: daughter-domain nascent RNA also appears on chromosome
            // rna_g's OWN sister (replication bubble) strand.  Free strands and
            // domain-0 entries are NOT overlaid.  Backward-compat: honour
            // root_domain != 0 as a pre-BF2 daughter signal alongside is_daughter.
            if !rna.is_free && (rna.is_daughter || rna.root_domain != 0) {
                if let Some(sister_idx) = rna_sister_idx_opt {
                    if let Some(sister) = strands.get(sister_idx) {
                        if let Some((bubble_root, _)) = bubble_point(
                            sister,
                            rna.root_coordinate,
                            chr.fork_fraction,
                            glen,
                        ) {
                            let points = crate::fiber::generate_rna_strand(
                                bubble_root,
                                bead_count,
                                rna_step,
                                rna_bead_radius,
                                shape,
                                rng,
                            );
                            snapshot.rna_strands.push(crate::placement::RnaStrand {
                                points,
                                is_mrna: rna.is_mRNA,
                                is_free: false,
                                unique_index: rna.unique_index,
                                length_nt: rna.length_nt,
                            });
                        }
                    }
                }
            }
        }
        // Seat the chromosome landmark molecules: the replisome at each fork,
        // oriC at each origin, terC at the terminus — so they read as real
        // machinery/loci on the DNA and are individually selectable.
        self.seat_markers(snapshot, next_uid, &chr.fork_marker, &forks, center);
        self.seat_markers(snapshot, next_uid, &chr.oric_marker, &orics, center);
        self.seat_markers(snapshot, next_uid, &chr.ter_marker, &ters, center);
        snapshot.chromosome = Some(crate::placement::Chromosome {
            center,
            radius: chr.bead_radius,
            color: chr.color,
            points: pts,
            strands,
            forks,
        });
    }

    /// Content-scaled placement on a sparse occupancy octree
    /// ([`PlacementBackend::Octree`]). The enclosing-sphere main pass and the
    /// proxy-accurate densify phase collapse into one loop — every candidate is
    /// checked against the octree at proxy accuracy — so there's no separate
    /// densify stage, clearance grid, or valid-cell list. Earlier-stage
    /// obstacles are inserted up front so this run avoids them.
    fn pack_octree(&self, seed: u64, obstacles: &[(Point3<f32>, f32)]) -> PlacerOutcome {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
        let mut snapshot = Snapshot::new(self.recipe.name.clone(), seed);
        let mut next_uid: u64 = 0;
        let mut stats = PlacerStats::default();

        // Frontier resolution: reuse the clearance cell-size policy (explicit
        // override, else max-radius/8). Only the occupied/free frontier refines
        // down to this; empty bulk stays coarse.
        let max_required_radius = self
            .recipe
            .ingredients
            .values()
            .map(|i| i.shape.enclosing_radius())
            .fold(0.0_f32, f32::max);
        let min_cell = self
            .config
            .clearance_cell_size
            .unwrap_or_else(|| (max_required_radius / 8.0).max(0.5));
        let mut octree = OccupancyOctree::new(self.recipe.bounding_box, min_cell);
        for &(c, r) in obstacles {
            octree.insert_sphere(c, r);
        }

        let directives: Vec<&PlacementDirective> = self.recipe.directives.iter().collect();
        let mut remaining: Vec<u32> = directives.iter().map(|d| d.count).collect();
        let mut per_ingredient_attempts: Vec<u64> = vec![0; directives.len()];
        let mut per_ingredient_placed: Vec<usize> = vec![0; directives.len()];
        let mut consecutive_fail: Vec<u32> = vec![0; directives.len()];
        let mut stuck: Vec<bool> = vec![false; directives.len()];
        stats.requested = remaining.iter().sum::<u32>() as usize;

        loop {
            let live: Vec<usize> = (0..directives.len())
                .filter(|&i| remaining[i] > 0 && !stuck[i])
                .collect();
            if live.is_empty() {
                break;
            }
            let dir_idx = live[rng.gen_range(0..live.len())];
            let directive = directives[dir_idx];
            let ingredient = self.recipe.ingredients.get(&directive.ingredient).unwrap();
            let compartment = self.recipe.compartments.get(&directive.compartment).unwrap();
            let er = ingredient.shape.enclosing_radius();
            let tiled = matches!(ingredient.packing_mode, PackingMode::Tiled);

            per_ingredient_attempts[dir_idx] += 1;
            stats.total_attempts += 1;

            let (pos, rotation) = match directive.region {
                RegionKind::Interior => {
                    // Free-biased point from the octree, kept only if its
                    // enclosing sphere is contained in this compartment. (Always
                    // strict containment — the octree backend targets named
                    // compartments; loose root bounds stay on the legacy path.)
                    let Some(p) = octree.sample_free(&mut rng) else {
                        stuck[dir_idx] = true; // no free space anywhere
                        continue;
                    };
                    let inside = compartment.kind.signed_distance(p) >= er
                        && compartment.children.iter().all(|&id| {
                            self.recipe
                                .compartments
                                .get_index(id as usize)
                                .map(|(_, c)| -c.kind.signed_distance(p) >= er)
                                .unwrap_or(true)
                        });
                    if !inside {
                        consecutive_fail[dir_idx] += 1;
                        if consecutive_fail[dir_idx] >= OCTREE_FAIL_CAP {
                            stuck[dir_idx] = true;
                        }
                        continue;
                    }
                    let rot = if ingredient.shape.needs_rotation() {
                        random_rotation(&mut rng)
                    } else {
                        UnitQuaternion::identity()
                    };
                    (p, rot)
                }
                RegionKind::Surface => {
                    let (p, n) = if tiled {
                        let idx = per_ingredient_attempts[dir_idx] - 1;
                        if idx >= directive.count as u64 {
                            stuck[dir_idx] = true;
                            continue;
                        }
                        match compartment.kind.surface_point_fibonacci(
                            idx,
                            directive.count as u64,
                            &mut rng,
                        ) {
                            Some(pn) => pn,
                            None => compartment.kind.sample_surface(&mut rng),
                        }
                    } else {
                        compartment.kind.sample_surface(&mut rng)
                    };
                    let mut rot = align_to_normal(ingredient.principal_vector, n);
                    if tiled {
                        rot = random_roll(rot, n, &mut rng);
                    }
                    (p, rot)
                }
            };

            // Tiled surface layers self-avoid by construction (even spacing) and
            // form a thin shell kept out of the octree, so the interior packs the
            // cell volume rather than the membrane. Place with no collision test.
            if tiled {
                self.record_placement(&mut snapshot, &mut next_uid, directive, pos, rotation);
                remaining[dir_idx] -= 1;
                per_ingredient_placed[dir_idx] += 1;
                stats.placed += 1;
                stats.successful_attempts += 1;
                continue;
            }

            // Proxy-accurate fit: every proxy must clear the octree. Lazy, so a
            // candidate that clashes on its first proxy doesn't transform the rest.
            let fits = ingredient
                .shape
                .world_spheres(pos, rotation)
                .all(|(c, r)| !octree.overlaps(c, r));
            if fits {
                for (c, r) in ingredient.shape.world_spheres(pos, rotation) {
                    octree.insert_sphere(c, r);
                }
                self.record_placement(&mut snapshot, &mut next_uid, directive, pos, rotation);
                remaining[dir_idx] -= 1;
                per_ingredient_placed[dir_idx] += 1;
                stats.placed += 1;
                stats.successful_attempts += 1;
                consecutive_fail[dir_idx] = 0;
            } else {
                consecutive_fail[dir_idx] += 1;
                if consecutive_fail[dir_idx] >= OCTREE_FAIL_CAP {
                    stuck[dir_idx] = true;
                }
            }
        }

        for (i, directive) in directives.iter().enumerate() {
            stats.per_ingredient.push((
                directive.ingredient.clone(),
                per_ingredient_placed[i],
                directive.count as usize,
                per_ingredient_attempts[i],
            ));
        }

        self.place_chromosome(&mut snapshot, &mut next_uid, &mut rng);
        PlacerOutcome { snapshot, stats }
    }

    /// Push one placement onto the snapshot, advancing the UID counter.
    fn record_placement(
        &self,
        snapshot: &mut Snapshot,
        next_uid: &mut u64,
        directive: &PlacementDirective,
        pos: Point3<f32>,
        rotation: UnitQuaternion<f32>,
    ) {
        let uid = *next_uid;
        *next_uid += 1;
        snapshot.placements.push(Placement {
            instance_uid: uid,
            ingredient_id: self.ingredient_ids[&directive.ingredient],
            variant_id: 0,
            compartment_id: self.compartment_ids[&directive.compartment],
            position: pos,
            rotation,
        });
    }

    /// The cell compartment the chromosome lives in: the named one if the
    /// spec gives a name, else the first sphere compartment. Returns its
    /// centre + radius.
    fn chromosome_cell(
        &self,
        chr: &crate::recipe::ChromosomeSpec,
    ) -> Option<(Point3<f32>, crate::fiber::CellShape)> {
        use crate::compartment::CompartmentKind;
        use crate::fiber::CellShape;
        for (name, comp) in &self.recipe.compartments {
            if let Some(want) = &chr.compartment {
                if name != want {
                    continue;
                }
            }
            let (center, shape) = match &comp.kind {
                CompartmentKind::Sphere { center, radius } => {
                    (*center, CellShape::Sphere { radius: *radius })
                }
                CompartmentKind::Capsule { a, b, radius } => {
                    let axis_v = b - a;
                    let half_len = axis_v.norm() * 0.5;
                    let axis = axis_v
                        .try_normalize(1e-6)
                        .unwrap_or_else(nalgebra::Vector3::x);
                    let center = Point3::from((a.coords + b.coords) * 0.5);
                    (center, CellShape::Capsule { half_len, radius: *radius, axis })
                }
                CompartmentKind::Box(aabb) => {
                    (aabb.center(), CellShape::Sphere { radius: aabb.half_extents().min() })
                }
                CompartmentKind::Mesh(m) => {
                    // Treat an elongated mesh (e.g. a constricted-capsule cell)
                    // as a capsule along its longest axis, so chromosomes still
                    // segregate pole-to-pole instead of collapsing to a sphere.
                    let he = m.aabb.half_extents();
                    let ext = [he.x, he.y, he.z];
                    let long_i = (0..3)
                        .max_by(|&i, &j| ext[i].partial_cmp(&ext[j]).unwrap())
                        .unwrap();
                    let radius = (0..3)
                        .filter(|&i| i != long_i)
                        .map(|i| ext[i])
                        .fold(f32::MAX, f32::min)
                        .max(1.0);
                    let half_len = (ext[long_i] - radius).max(0.0);
                    let axis = match long_i {
                        0 => nalgebra::Vector3::x(),
                        1 => nalgebra::Vector3::y(),
                        _ => nalgebra::Vector3::z(),
                    };
                    (m.aabb.center(), CellShape::Capsule { half_len, radius, axis })
                }
            };
            return Some((center, shape));
        }
        None
    }

    /// Resolve the chromosome's bound-protein specs to `(ingredient_id,
    /// ingredient, count)`, skipping any whose object isn't an ingredient
    /// (the recipe loader already rejects those, so this is belt-and-braces).
    fn resolve_fiber_proteins(&self, chr: &ChromosomeSpec) -> Vec<(u32, &Ingredient, u32)> {
        chr.proteins
            .iter()
            .filter_map(|(name, count)| {
                self.recipe
                    .ingredients
                    .get_full(name)
                    .map(|(idx, _, ing)| (idx as u32, ing, *count))
            })
            .collect()
    }

    /// Tree-vs-tree sphere collision against already-placed instances.
    /// QBVH broad-phase narrows to candidates whose enclosing spheres
    /// could overlap; inside, we walk every proxy-sphere pair across
    /// the candidate and the hit instance and reject on any
    /// centre-distance ≤ sum-of-radii.
    #[allow(clippy::too_many_arguments)]
    fn collides_with_existing(
        &self,
        candidate_shape: &IngredientShape,
        candidate_pos: Point3<f32>,
        candidate_rotation: UnitQuaternion<f32>,
        index: &QbvhIndex,
        shapes: &[&IngredientShape],
        centers: &[Point3<f32>],
        rotations: &[UnitQuaternion<f32>],
    ) -> bool {
        let candidate_r = candidate_shape.enclosing_radius();
        let max_other_r = shapes
            .iter()
            .map(|s| s.enclosing_radius())
            .fold(0.0_f32, f32::max);
        let query = Sphere::new(candidate_pos, candidate_r + max_other_r);

        let candidate_spheres: Vec<(Point3<f32>, f32)> = candidate_shape
            .world_spheres(candidate_pos, candidate_rotation)
            .collect();

        let mut collision = false;
        index.query_sphere(&query, |uid| {
            if collision {
                return;
            }
            let other_idx = uid as usize;
            let other_shape = shapes[other_idx];
            let other_center = centers[other_idx];
            let other_rotation = rotations[other_idx];
            let outer_d2 = (candidate_pos - other_center).norm_squared();
            let outer_r = candidate_r + other_shape.enclosing_radius();
            if outer_d2 > outer_r * outer_r {
                return;
            }
            for (oc, or_) in other_shape.world_spheres(other_center, other_rotation) {
                for (cc, cr) in &candidate_spheres {
                    let dx = cc.x - oc.x;
                    let dy = cc.y - oc.y;
                    let dz = cc.z - oc.z;
                    let d2 = dx * dx + dy * dy + dz * dz;
                    let r_sum = cr + or_;
                    if d2 < r_sum * r_sum {
                        collision = true;
                        return;
                    }
                }
            }
        });
        collision
    }

    /// Build the valid-cell list for one directive: every grid cell
    /// inside the compartment's AABB (inset by the ingredient radius)
    /// whose stored clearance is at least the required cell count AND
    /// whose centre passes the compartment's shape-fit test AND isn't
    /// inside a child compartment. Empty for Surface directives.
    fn build_valid_cells<R: Rng>(
        &self,
        directive: &PlacementDirective,
        grid: &ClearanceGrid,
        rng: &mut R,
    ) -> Vec<u32> {
        if matches!(directive.region, RegionKind::Surface) {
            return Vec::new();
        }
        let ingredient = self.recipe.ingredients.get(&directive.ingredient).unwrap();
        let compartment = self.recipe.compartments.get(&directive.compartment).unwrap();
        build_valid_cells_for(
            grid,
            ingredient,
            compartment,
            &self.recipe.compartments,
            self.config.strict_bounds,
            rng,
        )
    }
}

/// Build the valid-cell list for one Interior directive. A cell is
/// kept iff its centre has at least `radius` clearance to every
/// existing sphere surface (the grid's stored f32 value), at least
/// `compartment_cutoff` signed distance into its host compartment,
/// and at least `radius` outside every child compartment. The
/// `compartment_cutoff` differs between the root simulation domain
/// (cellPACK-style loose semantics when `strict_bounds == false`:
/// cutoff = 0, only the centre must be inside) and named compartments
/// (always strict: cutoff = radius, full sphere fits). The grid is
/// authoritative for the sphere-clearance check — sampling combined
/// with slack-bounded jitter then keeps placements collision-free
/// without a downstream QBVH check.
fn build_valid_cells_for<R: Rng>(
    grid: &ClearanceGrid,
    ingredient: &Ingredient,
    compartment: &Compartment,
    all_compartments: &IndexMap<String, Compartment>,
    strict_bounds: bool,
    rng: &mut R,
) -> Vec<u32> {
    let radius = ingredient.shape.enclosing_radius();
    let is_root_domain = compartment.parent.is_none();
    let compartment_cutoff = if is_root_domain && !strict_bounds {
        0.0
    } else {
        radius
    };

    let bb = compartment.kind.aabb();
    let inset_min = Point3::new(
        bb.min.x + compartment_cutoff,
        bb.min.y + compartment_cutoff,
        bb.min.z + compartment_cutoff,
    );
    let inset_max = Point3::new(
        bb.max.x - compartment_cutoff,
        bb.max.y - compartment_cutoff,
        bb.max.z - compartment_cutoff,
    );
    let lo = grid.point_to_cell(inset_min);
    let hi = grid.point_to_cell(inset_max);
    let lo_x = lo[0].max(0);
    let lo_y = lo[1].max(0);
    let lo_z = lo[2].max(0);
    let hi_x = hi[0].min(grid.dims[0] as i32 - 1);
    let hi_y = hi[1].min(grid.dims[1] as i32 - 1);
    let hi_z = hi[2].min(grid.dims[2] as i32 - 1);
    if lo_x > hi_x || lo_y > hi_y || lo_z > hi_z {
        return Vec::new();
    }

    let children: Vec<&Compartment> = compartment
        .children
        .iter()
        .filter_map(|&id| all_compartments.get_index(id as usize).map(|(_, c)| c))
        .collect();

    let stride_y = grid.dims[0];
    let stride_z = grid.dims[0] * grid.dims[1];

    // Is cell (cx,cy,cz) with flat index `i` a valid placement cell? Clearance
    // ≥ radius, ≥ cutoff inside the host compartment, and ≥ radius outside every
    // child compartment (`signed_distance` is positive inside, so `-sd ≥ radius`
    // means "outside the child by ≥ radius"). Shared by both passes below.
    let cell_valid = |i: usize, cx: i32, cy: i32, cz: i32| -> bool {
        if grid.clearance[i] < radius {
            return false;
        }
        let centre = grid.cell_centre([cx, cy, cz]);
        if compartment.kind.signed_distance(centre) < compartment_cutoff {
            return false;
        }
        !children
            .iter()
            .any(|c| -c.kind.signed_distance(centre) < radius)
    };

    let candidates =
        (hi_x - lo_x + 1) as u64 * (hi_y - lo_y + 1) as u64 * (hi_z - lo_z + 1) as u64;

    // Fast path: when the candidate volume dwarfs the cap, draw random cells and
    // keep the valid ones rather than scanning every cell. The list is first
    // built when the compartment is nearly empty, so almost every cell is valid
    // and this fills the cap in ~cap/density tries — orders of magnitude fewer
    // than the tens of millions of cells a whole-cell grid holds (where the full
    // scan costs tens of billions of signed-distance evals). A try budget bounds
    // the work; if the grid is too crowded to fill the cap by sampling, fall
    // through to the exhaustive scan, which finds sparse valid cells directly.
    if candidates > MAX_VALID_CELLS as u64 {
        let budget = MAX_VALID_CELLS.saturating_mul(REJECTION_TRY_BUDGET);
        let mut list = Vec::with_capacity(MAX_VALID_CELLS);
        let mut tries = 0usize;
        while list.len() < MAX_VALID_CELLS && tries < budget {
            tries += 1;
            let cx = rng.gen_range(lo_x..=hi_x);
            let cy = rng.gen_range(lo_y..=hi_y);
            let cz = rng.gen_range(lo_z..=hi_z);
            let i = cx as usize + stride_y * cy as usize + stride_z * cz as usize;
            if cell_valid(i, cx, cy, cz) {
                list.push(i as u32);
            }
        }
        if list.len() == MAX_VALID_CELLS {
            return list;
        }
        // Too crowded for sampling to pay off — fall through to a full scan.
    }

    // Exhaustive scan with reservoir sampling (Algorithm R): keeps a uniform
    // random subset of at most MAX_VALID_CELLS valid cells in one pass. When the
    // candidate volume is ≤ the cap this keeps every valid cell and draws no
    // randomness, so ordinary recipes stay bit-for-bit reproducible.
    let mut list: Vec<u32> = Vec::new();
    let mut seen: usize = 0;
    for cz in lo_z..=hi_z {
        for cy in lo_y..=hi_y {
            for cx in lo_x..=hi_x {
                let i = cx as usize + stride_y * cy as usize + stride_z * cz as usize;
                if !cell_valid(i, cx, cy, cz) {
                    continue;
                }
                if seen < MAX_VALID_CELLS {
                    list.push(i as u32);
                } else {
                    let j = rng.gen_range(0..=seen);
                    if j < MAX_VALID_CELLS {
                        list[j] = i as u32;
                    }
                }
                seen += 1;
            }
        }
    }
    list
}

/// Pick a random entry from a valid-cell list, return a sub-cell-
/// jittered world point at that cell. Jitter is slack-bounded — its
/// worst-case Euclidean displacement stays within the cell's smallest
/// clearance margin (sphere surfaces, compartment boundary, child
/// boundaries), so the jittered point is provably ≥ `radius` from
/// every forbidden surface. That bound is what makes Interior
/// placements collision-free without a downstream QBVH check. Stale
/// entries (clearance dropped below `radius` since the list was
/// built) get popped lazily. Returns `None` only when the list is
/// empty.
fn sample_from_valid_cells<R: Rng>(
    list: &mut Vec<u32>,
    grid: &ClearanceGrid,
    compartment: &Compartment,
    children: &[&Compartment],
    radius: f32,
    strict_bounds: bool,
    rng: &mut R,
) -> Option<Point3<f32>> {
    let half = grid.cell_size * 0.5;
    let inv_sqrt_3 = 0.577_350_26_f32;
    let is_root_domain = compartment.parent.is_none();
    let compartment_cutoff = if is_root_domain && !strict_bounds {
        0.0
    } else {
        radius
    };
    while !list.is_empty() {
        let idx_in_list = rng.gen_range(0..list.len());
        let cell_idx = list[idx_in_list];
        let cell_clearance = grid.clearance[cell_idx as usize];
        if cell_clearance < radius {
            list.swap_remove(idx_in_list);
            continue;
        }
        let centre = grid.cell_centre_flat(cell_idx);
        let sphere_slack = cell_clearance - radius;
        let compartment_slack = compartment.kind.signed_distance(centre) - compartment_cutoff;
        let mut min_slack = sphere_slack.min(compartment_slack);
        for child in children {
            let child_slack = -child.kind.signed_distance(centre) - radius;
            if child_slack < min_slack {
                min_slack = child_slack;
            }
        }
        let max_per_axis = (min_slack * inv_sqrt_3).min(half);
        if max_per_axis > 1e-6 {
            return Some(Point3::new(
                centre.x + rng.gen_range(-max_per_axis..max_per_axis),
                centre.y + rng.gen_range(-max_per_axis..max_per_axis),
                centre.z + rng.gen_range(-max_per_axis..max_per_axis),
            ));
        }
        return Some(centre);
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::recipe::Recipe;

    const SPHERES_IN_A_BOX_TINY: &str = r#"{
        "bounding_box": [[0,0,0],[100,100,100]],
        "objects": {
            "s10": { "type": "single_sphere", "radius": 10 }
        },
        "composition": {
            "space": { "regions": { "interior": ["A"] } },
            "A": { "object": "s10", "count": 20 }
        }
    }"#;

    #[test]
    fn places_some_into_a_box() {
        let recipe = Recipe::from_json_str(SPHERES_IN_A_BOX_TINY).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(0xC0DE);
        assert!(
            !out.snapshot.placements.is_empty(),
            "expected at least some placements, got 0"
        );
        assert!(out.snapshot.placements.len() <= 20);
        assert_eq!(out.stats.requested, 20);
        assert_eq!(out.stats.placed, out.snapshot.placements.len());
    }

    #[test]
    fn no_overlaps_in_output() {
        let recipe = Recipe::from_json_str(SPHERES_IN_A_BOX_TINY).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(0xFADE);
        let r = 10.0_f32;
        for i in 0..out.snapshot.placements.len() {
            for j in (i + 1)..out.snapshot.placements.len() {
                let a = out.snapshot.placements[i].position;
                let b = out.snapshot.placements[j].position;
                let d2 = (a - b).norm_squared();
                let r_sum = r + r;
                assert!(
                    d2 >= r_sum * r_sum - 1e-3,
                    "instances {i} and {j} overlap (d² = {}, r_sum² = {})",
                    d2,
                    r_sum * r_sum,
                );
            }
        }
    }

    #[test]
    fn all_placements_inside_bounding_box() {
        // Default PlacerConfig has `strict_bounds: true` — spheres
        // must fit fully inside the box (biology-correct semantics).
        // The loose mode is exercised by `loose_bounds_allows_protrusion`.
        let recipe = Recipe::from_json_str(SPHERES_IN_A_BOX_TINY).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(0xACE5);
        let r = 10.0_f32;
        let aabb = recipe.bounding_box;
        for p in &out.snapshot.placements {
            assert!(
                p.position.x - r >= aabb.min.x - 1e-3
                    && p.position.x + r <= aabb.max.x + 1e-3
                    && p.position.y - r >= aabb.min.y - 1e-3
                    && p.position.y + r <= aabb.max.y + 1e-3
                    && p.position.z - r >= aabb.min.z - 1e-3
                    && p.position.z + r <= aabb.max.z + 1e-3,
                "placement {:?} extends outside bounding box {:?}",
                p.position,
                aabb,
            );
        }
    }

    #[test]
    fn loose_bounds_allows_protrusion() {
        // With strict_bounds=false the root compartment uses cellPACK's
        // loose semantics — centres inside the box, sphere may
        // protrude at the edge. Verifying: at least one centre lands
        // within `radius` of an edge (which would fail strict-fit).
        let recipe = Recipe::from_json_str(SPHERES_IN_A_BOX_TINY).unwrap();
        let cfg = PlacerConfig {
            strict_bounds: false,
            ..PlacerConfig::default()
        };
        let placer = GreedyRandomPlacer::new(&recipe, cfg);
        let out = placer.pack(0xC0DE);
        let r = 10.0_f32;
        let aabb = recipe.bounding_box;
        let any_protrusion = out.snapshot.placements.iter().any(|p| {
            p.position.x - r < aabb.min.x
                || p.position.x + r > aabb.max.x
                || p.position.y - r < aabb.min.y
                || p.position.y + r > aabb.max.y
                || p.position.z - r < aabb.min.z
                || p.position.z + r > aabb.max.z
        });
        assert!(
            any_protrusion,
            "loose bounds should allow at least one protrusion in a tight-pack recipe"
        );
        // But centres must still be inside the box.
        for p in &out.snapshot.placements {
            assert!(
                p.position.x >= aabb.min.x - 1e-3
                    && p.position.x <= aabb.max.x + 1e-3
                    && p.position.y >= aabb.min.y - 1e-3
                    && p.position.y <= aabb.max.y + 1e-3
                    && p.position.z >= aabb.min.z - 1e-3
                    && p.position.z <= aabb.max.z + 1e-3,
                "centre {:?} outside box {:?} even with loose bounds",
                p.position,
                aabb,
            );
        }
    }

    const DUMBBELLS_IN_A_BOX: &str = r#"{
        "bounding_box": [[0,0,0],[100,100,100]],
        "objects": {
            "dumbbell": {
                "type": "multi_sphere",
                "positions": [[-10, 0, 0], [10, 0, 0]],
                "radii": [5, 5]
            }
        },
        "composition": {
            "space": { "regions": { "interior": ["A"] } },
            "A": { "object": "dumbbell", "count": 12 }
        }
    }"#;

    #[test]
    fn places_multi_sphere_ingredients() {
        let recipe = Recipe::from_json_str(DUMBBELLS_IN_A_BOX).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(0xC0DE);
        assert!(!out.snapshot.placements.is_empty(), "expected some dumbbells placed");
        let any_rotated = out.snapshot.placements.iter().any(|p| {
            (p.rotation.w - 1.0).abs() > 1e-6
        });
        assert!(any_rotated, "expected random rotations on multi-sphere placements");
    }

    const NESTED_CAPSULE: &str = r#"{
        "bounding_box": [[-100,-100,-100],[100,100,100]],
        "objects": {
            "lipid": { "type": "single_sphere", "radius": 2, "principal_vector": [0, 0, 1] },
            "protein": { "type": "single_sphere", "radius": 5 }
        },
        "composition": {
            "space": { "regions": { "interior": ["cell"] } },
            "cell": {
                "compartment": { "kind": "capsule", "a": [-40, 0, 0], "b": [40, 0, 0], "radius": 25 },
                "regions": {
                    "interior": [{ "object": "protein", "count": 30 }],
                    "surface":  [{ "object": "lipid",   "count": 60 }]
                }
            }
        }
    }"#;

    #[test]
    fn places_into_nested_capsule_with_surface_region() {
        let recipe = Recipe::from_json_str(NESTED_CAPSULE).unwrap();
        assert_eq!(recipe.compartments.len(), 2, "space + cell");
        let cell = &recipe.compartments["cell"];
        assert!(cell.parent.is_some(), "cell compartment should have parent");
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(0xC0DE);
        assert!(out.snapshot.placements.len() > 50, "expected most placements to fit");

        for p in &out.snapshot.placements {
            let comp = recipe.compartments.get_index(p.compartment_id as usize).unwrap().1;
            assert_eq!(comp.name, "cell");
        }

        for p in &out.snapshot.placements {
            let ing = recipe.ingredients.get_index(p.ingredient_id as usize).unwrap().1;
            if ing.name == "lipid" {
                let sd = match &recipe.compartments["cell"].kind {
                    crate::compartment::CompartmentKind::Capsule { a, b, radius } => {
                        let ab = b - a;
                        let ap = p.position - a;
                        let h = (ab.dot(&ap) / ab.norm_squared()).clamp(0.0, 1.0);
                        let closest = a + ab * h;
                        (p.position - closest).norm() - radius
                    }
                    _ => unreachable!(),
                };
                assert!(
                    sd.abs() < 1e-2,
                    "lipid not on capsule surface: signed distance = {sd}"
                );
            }
        }
    }

    #[test]
    fn multi_sphere_no_overlaps() {
        let recipe = Recipe::from_json_str(DUMBBELLS_IN_A_BOX).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(0xFADE);
        let mut all_spheres: Vec<(Point3<f32>, f32)> = Vec::new();
        for p in &out.snapshot.placements {
            let ing = recipe.ingredients.get_index(p.ingredient_id as usize).unwrap().1;
            all_spheres.extend(ing.shape.world_spheres(p.position, p.rotation));
        }
        for i in 0..all_spheres.len() {
            for j in (i + 1)..all_spheres.len() {
                let (ca, ra) = all_spheres[i];
                let (cb, rb) = all_spheres[j];
                // Spheres from the SAME placement (consecutive indices in a
                // dumbbell) overlap naturally — skip same-placement pairs.
                if i / 2 == j / 2 {
                    continue;
                }
                let d2 = (ca - cb).norm_squared();
                let r_sum = ra + rb;
                assert!(
                    d2 + 1e-2 >= r_sum * r_sum,
                    "proxy spheres {i} and {j} overlap (d={:.3}, r_sum={r_sum})",
                    d2.sqrt(),
                );
            }
        }
    }

    #[test]
    fn deterministic_with_same_seed() {
        let recipe = Recipe::from_json_str(SPHERES_IN_A_BOX_TINY).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let a = placer.pack(0xC0DE);
        let b = placer.pack(0xC0DE);
        assert_eq!(a.snapshot.placements.len(), b.snapshot.placements.len());
        for (pa, pb) in a.snapshot.placements.iter().zip(b.snapshot.placements.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }

    #[test]
    fn densify_packs_more_than_the_enclosing_pass() {
        // Thin rods: a big enclosing radius (~44) but a slim proxy footprint
        // (r=4). The enclosing-sphere pass keeps them far apart; the densify
        // phase lets them nestle, fitting many more.
        let src = r#"{
            "bounding_box": [[0,0,0],[200,200,200]],
            "objects": { "rod": { "type": "single_cylinder", "length": 80, "radius": 4 } },
            "composition": {
                "space": { "regions": { "interior": [ { "object": "rod", "count": 400 } ] } }
            }
        }"#;
        let recipe = Recipe::from_json_str(src).unwrap();
        let sparse = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(7);
        let dense = GreedyRandomPlacer::new(
            &recipe,
            PlacerConfig { densify: true, ..PlacerConfig::default() },
        )
        .pack(7);
        assert!(
            dense.snapshot.placements.len() > sparse.snapshot.placements.len(),
            "densify should fit more rods (sparse={}, dense={})",
            sparse.snapshot.placements.len(),
            dense.snapshot.placements.len(),
        );
        // The densify pass keeps the main pass's placements and only adds to
        // them, so it never regresses.
        assert!(dense.snapshot.placements.len() >= sparse.snapshot.placements.len());
    }

    #[test]
    fn densify_budget_caps_the_phase() {
        // Same thin-rod recipe. A zero attempt budget makes the densify phase
        // produce no candidates, so it places exactly the enclosing-sphere
        // pass's count — proof the global guard is honoured (and that a
        // pathological recipe can't spin densify unbounded).
        let src = r#"{
            "bounding_box": [[0,0,0],[200,200,200]],
            "objects": { "rod": { "type": "single_cylinder", "length": 80, "radius": 4 } },
            "composition": {
                "space": { "regions": { "interior": [ { "object": "rod", "count": 400 } ] } }
            }
        }"#;
        let recipe = Recipe::from_json_str(src).unwrap();
        let sparse = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(7);
        let capped = GreedyRandomPlacer::new(
            &recipe,
            PlacerConfig { densify: true, densify_max_attempts: 0, ..PlacerConfig::default() },
        )
        .pack(7);
        assert_eq!(
            capped.snapshot.placements.len(),
            sparse.snapshot.placements.len(),
            "a zero budget must place exactly the enclosing-sphere pass's count",
        );
    }

    #[test]
    fn valid_cell_list_is_capped() {
        use crate::clearance_grid::ClearanceGrid;
        use rand::SeedableRng;
        use rand_xoshiro::Xoshiro256PlusPlus;
        // A fine grid over this box holds ~1M cells, ~900k valid for a small
        // interior sphere — far above MAX_VALID_CELLS. Reservoir sampling must
        // clamp the list to the cap; this is what stops a whole-cell recipe
        // from OOMing while building hundreds of these lists at once.
        let recipe = Recipe::from_json_str(
            r#"{
            "bounding_box": [[0,0,0],[200,200,200]],
            "objects": { "s": { "type": "single_sphere", "radius": 3 } },
            "composition": { "space": { "regions": { "interior": [ { "object": "s", "count": 1 } ] } } }
        }"#,
        )
        .unwrap();
        let grid = ClearanceGrid::new(recipe.bounding_box, 2.0);
        let (_, ing) = recipe.ingredients.get_index(0).unwrap();
        let (_, comp) = recipe.compartments.get_index(0).unwrap();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let cells =
            build_valid_cells_for(&grid, ing, comp, &recipe.compartments, true, &mut rng);
        assert_eq!(cells.len(), MAX_VALID_CELLS, "valid-cell list must clamp to the cap");
    }

    #[test]
    fn octree_backend_packs_without_overlap() {
        let recipe = Recipe::from_json_str(SPHERES_IN_A_BOX_TINY).unwrap();
        let cfg = PlacerConfig {
            backend: PlacementBackend::Octree,
            ..PlacerConfig::default()
        };
        let out = GreedyRandomPlacer::new(&recipe, cfg).pack(0xFADE);
        assert!(!out.snapshot.placements.is_empty(), "octree backend placed nothing");
        let r = 10.0_f32;
        for i in 0..out.snapshot.placements.len() {
            for j in (i + 1)..out.snapshot.placements.len() {
                let d2 = (out.snapshot.placements[i].position
                    - out.snapshot.placements[j].position)
                    .norm_squared();
                assert!(
                    d2 >= (2.0 * r) * (2.0 * r) - 1e-3,
                    "octree placements {i},{j} overlap (d²={d2})"
                );
            }
            let q = out.snapshot.placements[i].position;
            for c in [q.x, q.y, q.z] {
                assert!((-1e-3..=100.0 + 1e-3).contains(&c), "placement outside box");
            }
        }
    }

    #[test]
    fn octree_backend_is_deterministic() {
        let recipe = Recipe::from_json_str(SPHERES_IN_A_BOX_TINY).unwrap();
        let cfg = PlacerConfig {
            backend: PlacementBackend::Octree,
            ..PlacerConfig::default()
        };
        let a = GreedyRandomPlacer::new(&recipe, cfg).pack(99);
        let b = GreedyRandomPlacer::new(&recipe, cfg).pack(99);
        assert_eq!(a.snapshot.placements.len(), b.snapshot.placements.len());
        for (pa, pb) in a.snapshot.placements.iter().zip(b.snapshot.placements.iter()) {
            assert_eq!(pa.position, pb.position);
        }
    }

    #[test]
    fn strand_point_maps_origin_to_midpoint_and_is_on_strand() {
        // A simple straight strand of 101 beads along x.
        let strand: Vec<Point3<f32>> = (0..101)
            .map(|i| Point3::new(-500.0 + i as f32 * 10.0, 0.0, 0.0))
            .collect();
        let strands = vec![strand.clone()];
        let glen = 4_641_652u32;
        // coordinate 0 (oriC) -> fraction 0.5 -> bead 50 -> x = 0.0
        let (p0, t0) = strand_point(&strands, 0, 0, glen).unwrap();
        assert!((p0.x - 0.0).abs() < 1e-3, "oriC not at midpoint: {}", p0.x);
        assert!((t0.norm() - 1.0).abs() < 1e-4);
        // a positive coordinate moves toward the high-index end (downstream of oriC)
        let (p1, _) = strand_point(&strands, 0, (glen / 4) as i64, glen).unwrap();
        assert!(p1.x > p0.x);
        // every mapped point is an actual bead-interpolated point on the strand
        assert!(strand.iter().map(|b| (b - p1).norm()).fold(f32::MAX, f32::min) < 11.0);
    }

    #[test]
    fn bubble_point_maps_forks_to_ends_and_oric_to_middle() {
        // sister of 101 beads along x in [-500, 500]
        let sister: Vec<Point3<f32>> = (0..101).map(|i| Point3::new(-500.0 + i as f32 * 10.0, 0.0, 0.0)).collect();
        let glen = 4_641_652u32;
        let ff = 0.45_f32;
        let fork_bp = (ff * glen as f32 / 2.0) as i64; // bubble half-width in bp
        // oriC (coord 0) → frac 0.5 → middle (x≈0)
        let (mid, _) = bubble_point(&sister, 0, ff, glen).unwrap();
        assert!(mid.x.abs() < 6.0, "oriC should map near the sister middle, got {}", mid.x);
        // +fork → frac 1.0 → last bead (x≈+500)
        let (hi, _) = bubble_point(&sister, fork_bp, ff, glen).unwrap();
        assert!(hi.x > 480.0, "+fork should map near the sister far end, got {}", hi.x);
        // -fork → frac 0.0 → first bead (x≈-500)
        let (lo, _) = bubble_point(&sister, -fork_bp, ff, glen).unwrap();
        assert!(lo.x < -480.0, "-fork should map near the sister near end, got {}", lo.x);
        // coordinate beyond the bubble clamps to an end (does not panic / wrap)
        let (clamped, _) = bubble_point(&sister, glen as i64, ff, glen).unwrap();
        assert!(clamped.x > 480.0);
    }

    // ---- A4 test helpers ------------------------------------------------

    /// Build a capsule recipe with a 1000-bead chromosome and `n` explicit
    /// RNAP placements at evenly-spread genomic coordinates (alternating
    /// forward/reverse strand). The two end entries land at extreme coordinates
    /// (~±genome_len/2, i.e. the strand tips) to exercise the no-drop
    /// confinement fallback. The cell is large enough to hold the strand and
    /// all RNAPs without crowding.
    fn recipe_with_chromosome_and_rnaps(n: usize) -> Recipe {
        let genome_len: i64 = GENOME_BP_DEFAULT as i64;
        let half = genome_len / 2;
        let step = if n > 1 { genome_len / n as i64 } else { 0 };
        let rnap_entries: String = (0..n)
            .map(|i| {
                // Spread coordinates evenly across the full [-half, +half]
                // range. i=0 ≈ -genome_len/2 and i=n-1 ≈ +genome_len/2 are
                // extreme tip coordinates (frac ≈ 0.01 / 0.99) that map to the
                // strand ends near the cell wall — the no-drop path.
                let coord = -half + step / 2 + i as i64 * step;
                let fwd = if i % 2 == 0 { "true" } else { "false" };
                format!(
                    r#"{{"coordinates": {coord}, "domain_index": 0, "is_forward": {fwd}}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            r#"{{
                "bounding_box": [[-500,-500,-500],[500,500,500]],
                "objects": {{
                    "rna_polymerase": {{ "type": "single_sphere", "radius": 20 }}
                }},
                "composition": {{
                    "space": {{ "regions": {{ "interior": ["cell"] }} }},
                    "cell": {{
                        "compartment": {{
                            "kind": "capsule",
                            "a": [-150, 0, 0],
                            "b": [150, 0, 0],
                            "radius": 80
                        }},
                        "regions": {{ "interior": [] }}
                    }}
                }},
                "chromosome": {{
                    "beads": 1000,
                    "spacing": 10.0,
                    "bead_radius": 5.0,
                    "compartment": "cell",
                    "rnap_marker": "rna_polymerase",
                    "rnaps": [{rnap_entries}]
                }}
            }}"#
        );
        Recipe::from_json_str(&json).expect("recipe_with_chromosome_and_rnaps: parse failed")
    }

    /// A *replicating* chromosome (`fork_fraction > 0` → main + sister strands)
    /// with a single RNAP at `(coord, domain)`. Used to verify that an RNAP on a
    /// daughter domain is still placed on the MAIN genome contour by coordinate
    /// (matching the v2ecoli `_draw_chromosome` reference: "rim RNAPs: ALL of
    /// them, regardless of domain").
    fn recipe_replicating_with_one_rnap(coord: i64, domain: i32) -> Recipe {
        let json = format!(
            r#"{{
                "bounding_box": [[-500,-500,-500],[500,500,500]],
                "objects": {{
                    "rna_polymerase": {{ "type": "single_sphere", "radius": 20 }}
                }},
                "composition": {{
                    "space": {{ "regions": {{ "interior": ["cell"] }} }},
                    "cell": {{
                        "compartment": {{
                            "kind": "capsule",
                            "a": [-150, 0, 0],
                            "b": [150, 0, 0],
                            "radius": 80
                        }},
                        "regions": {{ "interior": [] }}
                    }}
                }},
                "chromosome": {{
                    "beads": 1000,
                    "spacing": 10.0,
                    "bead_radius": 5.0,
                    "compartment": "cell",
                    "n_chromosomes": 1,
                    "fork_fraction": 0.45,
                    "rnap_marker": "rna_polymerase",
                    "rnaps": [{{"coordinates": {coord}, "domain_index": {domain}, "is_forward": true}}]
                }}
            }}"#
        );
        Recipe::from_json_str(&json).expect("recipe_replicating_with_one_rnap: parse failed")
    }

    /// A daughter-domain RNAP is placed on BOTH the main chromosome contour AND
    /// the sister (replication bubble) strand.  Two placements are expected: one
    /// near the main locus (`strand_point` on strand 0) and one near the bubble
    /// locus (`bubble_point` on the last strand).
    #[test]
    fn rnap_placed_on_main_contour_regardless_of_domain() {
        let coord = 0_i64; // oriC — mid-bubble, where main and sister diverge
        let recipe = recipe_replicating_with_one_rnap(coord, 2); // daughter domain 2
        let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(3);
        let (center, _shape) = first_capsule_cell(&recipe);
        let rnap_id = recipe.ingredients.get_full("rna_polymerase").unwrap().0 as u32;
        let rnaps: Vec<_> = out
            .snapshot
            .placements
            .iter()
            .filter(|p| p.ingredient_id == rnap_id)
            .collect();
        // Daughter RNAP renders on main contour AND the sister bubble strand.
        assert_eq!(rnaps.len(), 2, "daughter RNAP renders on main + bubble");

        let strands = &out.snapshot.chromosome.as_ref().unwrap().strands;
        assert!(strands.len() >= 2, "replicating chromosome should have a sister strand");
        let ff = 0.45_f32;
        let sister = strands.last().unwrap();
        let main_w = center + strand_point(strands, 0, coord, GENOME_BP_DEFAULT).unwrap().0.coords;
        let bub_w = center + bubble_point(sister, coord, ff, GENOME_BP_DEFAULT).unwrap().0.coords;
        let near_main = rnaps.iter().any(|p| (p.position - main_w).norm() < 30.0);
        let near_bubble = rnaps.iter().any(|p| (p.position - bub_w).norm() < 30.0);
        assert!(near_main && near_bubble, "expected one RNAP near main and one near the bubble");
    }

    /// A daughter-domain RNAP at oriC (coord 0) produces two placements, both
    /// confined inside the inset envelope.
    #[test]
    fn daughter_rnap_overlaid_on_bubble() {
        let recipe = recipe_replicating_with_one_rnap(0, 2); // domain 2, coord 0
        let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(3);
        let (_, shape) = first_capsule_cell(&recipe);
        let rnap_id = recipe.ingredients.get_full("rna_polymerase").unwrap().0 as u32;
        let rnaps: Vec<_> = out
            .snapshot
            .placements
            .iter()
            .filter(|p| p.ingredient_id == rnap_id)
            .collect();
        assert_eq!(rnaps.len(), 2, "daughter RNAP should yield exactly 2 placements (main + bubble)");
        let proxy = 20.0_f32;
        let inset = shape.inset(proxy);
        for p in &rnaps {
            assert!(
                inset.contains(&p.position),
                "daughter RNAP placement outside inset envelope: {:?}",
                p.position
            );
        }
    }

    /// A domain-0 RNAP yields exactly ONE placement — no bubble overlay.
    #[test]
    fn domain_zero_rnap_yields_one_placement() {
        let recipe = recipe_replicating_with_one_rnap(0, 0); // domain 0
        let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(3);
        let rnap_id = recipe.ingredients.get_full("rna_polymerase").unwrap().0 as u32;
        let rnaps: Vec<_> = out
            .snapshot
            .placements
            .iter()
            .filter(|p| p.ingredient_id == rnap_id)
            .collect();
        assert_eq!(rnaps.len(), 1, "domain-0 RNAP must produce exactly one placement (no bubble overlay)");
    }

    /// BF1-3: a domain-2 nascent RNA yields TWO strands — one rooted on the main
    /// chromosome contour (strand 0) and a second on the sister (bubble) strand.
    /// The original single-strand assertion is replaced: we now verify that one
    /// root is ≤ 30 Å from the main locus and one is ≤ 30 Å from the bubble
    /// locus, both computed in world space (center + center-relative coords).
    #[test]
    fn nascent_rna_roots_on_main_contour_like_its_rnap() {
        let coord = 0_i64; // oriC — where main and sister diverge most
        let json = format!(
            r#"{{
                "bounding_box": [[-500,-500,-500],[500,500,500]],
                "objects": {{ "rna_segment": {{ "type": "single_sphere", "radius": 4 }} }},
                "composition": {{
                    "space": {{ "regions": {{ "interior": ["cell"] }} }},
                    "cell": {{
                        "compartment": {{ "kind": "capsule", "a": [-150,0,0], "b": [150,0,0], "radius": 80 }},
                        "regions": {{ "interior": [] }}
                    }}
                }},
                "chromosome": {{
                    "beads": 1000, "spacing": 10.0, "bead_radius": 5.0, "compartment": "cell",
                    "n_chromosomes": 1, "fork_fraction": 0.45,
                    "rna_segment": "rna_segment", "rna_angstrom_per_nt": 2.0,
                    "rnas": [{{"root_coordinate": {coord}, "root_domain": 2, "length_nt": 400, "is_mRNA": true}}]
                }}
            }}"#
        );
        let recipe = Recipe::from_json_str(&json).unwrap();
        let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(3);
        let (center, _shape) = first_capsule_cell(&recipe);
        let rna_strands = &out.snapshot.rna_strands;
        // BF1-3: domain-2 nascent RNA now yields TWO strands (main + bubble).
        assert_eq!(rna_strands.len(), 2, "domain-2 nascent RNA must yield 2 strands (main + bubble overlay)");
        let strands = &out.snapshot.chromosome.as_ref().unwrap().strands;
        let main_world = center + strand_point(strands, 0, coord, GENOME_BP_DEFAULT).unwrap().0.coords;
        let bub_world = center
            + bubble_point(strands.last().unwrap(), coord, 0.45, GENOME_BP_DEFAULT)
                .unwrap()
                .0
                .coords;
        let near_main = rna_strands
            .iter()
            .any(|s| (center + s.points[0].coords - main_world).norm() < 30.0);
        let near_bub = rna_strands
            .iter()
            .any(|s| (center + s.points[0].coords - bub_world).norm() < 30.0);
        assert!(near_main, "one RNA strand must root ≤ 30 Å from the main locus");
        assert!(near_bub, "one RNA strand must root ≤ 30 Å from the bubble locus");
    }

    /// BF1-3: overlay rules — domain-2 nascent → 2 strands; domain-0 nascent
    /// → 1 strand (no overlay); free RNA (`is_free=true`) → 1 strand (no overlay).
    #[test]
    fn daughter_rna_overlaid_on_bubble() {
        let json = r#"{
            "bounding_box": [[-500,-500,-500],[500,500,500]],
            "objects": {"rna_segment": {"type": "single_sphere", "radius": 4}},
            "composition": {
                "space": {"regions": {"interior": ["cell"]}},
                "cell": {
                    "compartment": {"kind": "capsule", "a": [-150,0,0], "b": [150,0,0], "radius": 80},
                    "regions": {"interior": []}
                }
            },
            "chromosome": {
                "beads": 1000, "spacing": 10.0, "bead_radius": 5.0, "compartment": "cell",
                "n_chromosomes": 1, "fork_fraction": 0.45,
                "rna_segment": "rna_segment", "rna_angstrom_per_nt": 2.0,
                "rnas": [
                    {"root_coordinate": 0, "root_domain": 2, "length_nt": 400, "is_mRNA": true},
                    {"root_coordinate": 0, "root_domain": 0, "length_nt": 400, "is_mRNA": true},
                    {"root_coordinate": 0, "root_domain": 2, "length_nt": 400, "is_mRNA": true, "is_free": true}
                ]
            }
        }"#;
        let recipe = Recipe::from_json_str(json).unwrap();
        let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(3);
        // domain-2 nascent = 2 strands (main + bubble)
        // domain-0 nascent = 1 strand (no overlay)
        // free RNA         = 1 strand (no overlay, regardless of domain)
        // Total            = 4
        assert_eq!(
            out.snapshot.rna_strands.len(),
            4,
            "expected 4 rna_strands: 2 from domain-2 nascent + 1 from domain-0 + 1 free; \
             got {}",
            out.snapshot.rna_strands.len()
        );
    }

    /// Return `(center, CellShape)` for the first capsule compartment in the
    /// recipe — mirrors the logic in `GreedyRandomPlacer::chromosome_cell`.
    fn first_capsule_cell(recipe: &Recipe) -> (Point3<f32>, crate::fiber::CellShape) {
        use crate::compartment::CompartmentKind;
        use crate::fiber::CellShape;
        for (_, comp) in &recipe.compartments {
            if let CompartmentKind::Capsule { a, b, radius } = &comp.kind {
                let axis_v = b - a;
                let half_len = axis_v.norm() * 0.5;
                let axis = axis_v
                    .try_normalize(1e-6)
                    .unwrap_or_else(nalgebra::Vector3::x);
                let center = Point3::from((a.coords + b.coords) * 0.5);
                return (center, CellShape::Capsule { half_len, radius: *radius, axis });
            }
        }
        panic!("first_capsule_cell: no capsule compartment found in recipe");
    }

    /// A4: every explicit RNAP is placed (1:1, no drops), inside the inset
    /// envelope, AND near an actual strand bead (not collapsed to the centre).
    #[test]
    fn seats_every_rnap_on_strand_inside_envelope() {
        const N: usize = 50;
        let recipe = recipe_with_chromosome_and_rnaps(N);
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(11);
        let (center, shape) = first_capsule_cell(&recipe);
        let rnap_id = recipe.ingredients.get_full("rna_polymerase").unwrap().0 as u32;
        let rnaps: Vec<_> = out
            .snapshot
            .placements
            .iter()
            .filter(|p| p.ingredient_id == rnap_id)
            .collect();

        // 1:1 true abundance — every recipe RNAP is rendered, none dropped
        // (including the extreme tip-coordinate entries via the no-drop path).
        assert_eq!(rnaps.len(), N, "expected {N} RNAPs, got {}", rnaps.len());

        // Confinement invariant — all inside the inset envelope.
        let proxy = 20.0_f32; // rna_polymerase radius
        let inset = shape.inset(proxy);
        for p in &rnaps {
            assert!(
                inset.contains(&p.position),
                "RNAP outside envelope: {:?} (center={:?})",
                p.position,
                center
            );
        }

        // Genomic-axis fidelity — each RNAP tracks ITS OWN locus, not a shared
        // point. (The strand's first bead is at the origin == cell centre, so a
        // "near ANY bead" check would let an all-at-centre bug pass; recomputing
        // the expected strand point per coordinate closes that gap.) Strand beads
        // are cell-centre-relative; world = center + bead.coords. Placement order
        // matches `chr.rnaps` order. Tolerance is bead-spacing scale plus the proxy
        // radius (confinement may pull a wall-hugging bead slightly inward).
        let chrom = out
            .snapshot
            .chromosome
            .as_ref()
            .expect("chromosome was not attached to snapshot");
        let specs = &recipe.chromosome.as_ref().unwrap().rnaps;
        assert_eq!(specs.len(), N, "test fixture should declare {N} rnaps");
        let spacing = 10.0_f32; // chr.spacing
        let tol = spacing + proxy;
        for (p, spec) in rnaps.iter().zip(specs.iter()) {
            let (expected_rel, _) = strand_point(
                &chrom.strands,
                spec.domain_index,
                spec.coordinates,
                GENOME_BP_DEFAULT,
            )
            .expect("strand_point should map every locus");
            let expected = center + expected_rel.coords;
            let d = (expected - p.position).norm();
            assert!(
                d < tol,
                "RNAP not at its locus: coord {} placed {:?}, expected {:?} (d={d}, tol={tol})",
                spec.coordinates,
                p.position,
                expected
            );
        }
        // Sanity: the loci genuinely differ (coordinate-dependent mapping), so
        // the per-locus check above isn't vacuous — the RNAPs span a real extent.
        // Use the full 3D pairwise extent (the strand walk is not axis-aligned).
        let mut max_pair = 0.0_f32;
        for i in 0..rnaps.len() {
            for j in (i + 1)..rnaps.len() {
                max_pair = max_pair.max((rnaps[i].position - rnaps[j].position).norm());
            }
        }
        assert!(
            max_pair > 50.0,
            "RNAPs should spread along the strand, max pairwise extent={max_pair}"
        );
    }

    // ---- B1-3 test helpers -----------------------------------------------

    /// Build a capsule recipe with a 1000-bead chromosome and N explicit nascent
    /// RNA strands at the given (root_coordinate, length_nt) pairs.  Mirrors
    /// `recipe_with_chromosome_and_rnaps`.
    fn recipe_with_chromosome_and_rnas(specs: &[(i64, i64)]) -> Recipe {
        let rna_entries: String = specs
            .iter()
            .map(|&(coord, len)| {
                format!(
                    r#"{{"root_coordinate": {coord}, "root_domain": 0, "length_nt": {len}, "is_mRNA": true}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            r#"{{
                "bounding_box": [[-500,-500,-500],[500,500,500]],
                "objects": {{}},
                "composition": {{
                    "space": {{ "regions": {{ "interior": ["cell"] }} }},
                    "cell": {{
                        "compartment": {{
                            "kind": "capsule",
                            "a": [-150, 0, 0],
                            "b": [150, 0, 0],
                            "radius": 80
                        }},
                        "regions": {{ "interior": [] }}
                    }}
                }},
                "chromosome": {{
                    "beads": 1000,
                    "spacing": 10.0,
                    "bead_radius": 5.0,
                    "compartment": "cell",
                    "rnas": [{rna_entries}]
                }}
            }}"#
        );
        Recipe::from_json_str(&json).expect("recipe_with_chromosome_and_rnas: parse failed")
    }

    /// Build a capsule recipe with a 1000-bead chromosome and N explicit RNA
    /// strands at the given (root_coordinate, length_nt, is_free) triples.
    /// Mirrors `recipe_with_chromosome_and_rnas` but emits the `is_free` key.
    fn recipe_with_chromosome_and_rnas_freeflag(specs: &[(i64, i64, bool)]) -> Recipe {
        let rna_entries: String = specs
            .iter()
            .map(|&(coord, len, free)| {
                format!(
                    r#"{{"root_coordinate": {coord}, "root_domain": 0, "length_nt": {len}, "is_mRNA": true, "is_free": {free}}}"#
                )
            })
            .collect::<Vec<_>>()
            .join(",");
        let json = format!(
            r#"{{
                "bounding_box": [[-500,-500,-500],[500,500,500]],
                "objects": {{}},
                "composition": {{
                    "space": {{ "regions": {{ "interior": ["cell"] }} }},
                    "cell": {{
                        "compartment": {{
                            "kind": "capsule",
                            "a": [-150, 0, 0],
                            "b": [150, 0, 0],
                            "radius": 80
                        }},
                        "regions": {{ "interior": [] }}
                    }}
                }},
                "chromosome": {{
                    "beads": 1000,
                    "spacing": 10.0,
                    "bead_radius": 5.0,
                    "compartment": "cell",
                    "rnas": [{rna_entries}]
                }}
            }}"#
        );
        Recipe::from_json_str(&json)
            .expect("recipe_with_chromosome_and_rnas_freeflag: parse failed")
    }

    /// B2-1: free RNA (`is_free=true`) seeds at a random interior point, NOT
    /// at the chromosome-rooted strand_point, while still confined.
    #[test]
    fn free_rna_seeds_in_interior_not_at_strand_point() {
        // one nascent (is_free=false) and one free (is_free=true) at the SAME
        // root_coordinate: if `is_free` routing were ignored, both would root at
        // the same chromosome strand_point (norm ≈ 0) and the `> 1.0` assertion
        // below would FAIL — so this gates the actual is_free behavior, not just
        // a coordinate difference.
        let recipe = recipe_with_chromosome_and_rnas_freeflag(
            &[(100000_i64, 600_i64, false), (100000, 600, true)],
        );
        let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(7);
        assert_eq!(out.snapshot.rna_strands.len(), 2);
        let (center, shape) = first_capsule_cell(&recipe);
        let inset = shape.inset(4.0);
        for rs in &out.snapshot.rna_strands {
            for p in &rs.points {
                assert!(
                    inset.contains(&(center + p.coords)) || inset.contains(p),
                    "RNA bead outside envelope"
                );
            }
        }
        // the free strand's root must NOT coincide with the nascent strand's chromosome-rooted start
        let nascent_root = out.snapshot.rna_strands[0].points[0];
        let free_root = out.snapshot.rna_strands[1].points[0];
        assert!(
            (nascent_root - free_root).norm() > 1.0,
            "free strand should not root at the same chromosome point"
        );
    }

    /// B1-3: one confined nascent-RNA strand per RnaSpec, rooted near its
    /// RNAP genomic locus, all beads inside the envelope, and longer
    /// length_nt → more beads.
    #[test]
    fn grows_one_confined_strand_per_rna_rooted_at_rnap() {
        let recipe = recipe_with_chromosome_and_rnas(&[(100_000_i64, 400_i64), (-50_000, 1200)]);
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(9);

        // 1:1 true abundance — exactly one strand per RnaSpec.
        assert_eq!(
            out.snapshot.rna_strands.len(),
            2,
            "expected 2 rna_strands, got {}",
            out.snapshot.rna_strands.len()
        );

        let (center, shape) = first_capsule_cell(&recipe);
        let strands = &out.snapshot.chromosome.as_ref().unwrap().strands;
        // RNA bead radius matches the production value (4.0 Å).
        let rna_bead_radius = 4.0_f32;
        let inset = shape.inset(rna_bead_radius);

        for (rs, &(coord, _len)) in out
            .snapshot
            .rna_strands
            .iter()
            .zip(&[(100_000_i64, 400_i64), (-50_000, 1200)])
        {
            // Root proximity: strand[0] within 30 Å of the strand_point locus.
            // Strands and rna_strands are both center-relative; center=[0,0,0]
            // for this cell so world == center-relative.
            let (root, _t) = strand_point(strands, 0, coord, GENOME_BP_DEFAULT)
                .expect("strand_point should map every coordinate");
            assert!(
                (rs.points[0] - root).norm() < 30.0,
                "strand not rooted at its RNAP: points[0]={:?} root={:?} dist={}",
                rs.points[0],
                root,
                (rs.points[0] - root).norm()
            );

            // Confinement: every bead inside the inset envelope.
            // rna_strands are center-relative; the cell center is the origin,
            // so p and center+p.coords are the same point here — OR-ed for
            // robustness if the frame ever shifts.
            for p in &rs.points {
                assert!(
                    inset.contains(&(center + p.coords)) || inset.contains(p),
                    "RNA bead outside envelope: {:?}",
                    p
                );
            }
        }

        // Relative-length fidelity: longer length_nt → more beads.
        assert!(
            out.snapshot.rna_strands[1].points.len()
                > out.snapshot.rna_strands[0].points.len(),
            "expected longer strand for length_nt=1200 vs 400: got {} vs {}",
            out.snapshot.rna_strands[1].points.len(),
            out.snapshot.rna_strands[0].points.len()
        );
    }

    /// A4: the orientation helper turns the RNAP +x reference axis onto the
    /// strand tangent — including the antiparallel (`dir == -x`) case, where a
    /// naive identity fallback would point it the WRONG way (+x).
    #[test]
    fn rnap_orientation_handles_antiparallel_tangent() {
        // Antiparallel: +x must map to -x (a real 180° turn), NOT stay at +x.
        let r = orient_x_onto(-Vector3::x());
        let mapped = r * Vector3::x();
        assert!(
            (mapped - (-Vector3::x())).norm() < 1e-5,
            "antiparallel: +x should map to -x, got {mapped:?}"
        );
        // Parallel: +x stays +x.
        let r2 = orient_x_onto(Vector3::x());
        assert!(((r2 * Vector3::x()) - Vector3::x()).norm() < 1e-5);
        // Arbitrary direction: +x maps onto the (unit) target.
        let d = Vector3::new(0.3, -0.7, 0.5).normalize();
        let r3 = orient_x_onto(d);
        assert!(((r3 * Vector3::x()) - d).norm() < 1e-5);
    }

    // ---- Chromosome centering tests -------------------------------------

    /// Build a minimal capsule recipe with a replicating chromosome
    /// (fork_fraction = 0.45). Used by centering + segregation tests.
    fn replicating_chromosome_recipe_json(n_chromosomes: usize) -> String {
        format!(
            r#"{{
                "bounding_box": [[-600,-200,-200],[600,200,200]],
                "objects": {{}},
                "composition": {{
                    "space": {{ "regions": {{ "interior": ["cell"] }} }},
                    "cell": {{
                        "compartment": {{
                            "kind": "capsule",
                            "a": [-500, 0, 0],
                            "b": [500, 0, 0],
                            "radius": 80
                        }},
                        "regions": {{ "interior": [] }}
                    }}
                }},
                "chromosome": {{
                    "beads": 500,
                    "spacing": 10.0,
                    "bead_radius": 5.0,
                    "compartment": "cell",
                    "n_chromosomes": {n_chromosomes},
                    "fork_fraction": 0.45
                }}
            }}"#
        )
    }

    /// A single replicating chromosome (theta/supercoiled path) must have its
    /// bead centroid near the cell centre after packing. Before the COM-recenter
    /// fix the SAW backbone drifts from the origin, pushing the centroid to
    /// ≈ ±0.47 × half_len; after the fix it must be < 0.12 × half_len.
    #[test]
    fn replicating_chromosome_is_centered() {
        let recipe =
            Recipe::from_json_str(&replicating_chromosome_recipe_json(1)).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(42);

        let chr = out
            .snapshot
            .chromosome
            .as_ref()
            .expect("chromosome missing from snapshot");
        let half_len = 500.0_f32;

        // Centroid of every strand bead (stored center-relative; cell center = origin).
        let all_beads: Vec<&Point3<f32>> = chr.strands.iter().flatten().collect();
        assert!(!all_beads.is_empty(), "no strand beads in snapshot");
        let n = all_beads.len() as f32;
        let centroid_x = all_beads.iter().map(|p| p.x).sum::<f32>() / n;

        assert!(
            centroid_x.abs() < 0.12 * half_len,
            "replicating chromosome not centered: centroid.x = {:.1} \
             (|{:.3}| × half_len = {:.1}), must be < {:.1}",
            centroid_x,
            centroid_x.abs() / half_len,
            centroid_x.abs() / half_len * half_len,
            0.12 * half_len,
        );

        // Every bead must remain inside the full cell envelope.
        let shape = crate::fiber::CellShape::Capsule {
            half_len,
            radius: 80.0,
            axis: Vector3::x(),
        };
        for p in &all_beads {
            assert!(
                shape.contains(p),
                "bead {:?} escaped the cell envelope after clamping",
                p,
            );
        }
    }

    /// With two replicating chromosomes, COM recentering must move each
    /// chromosome to its OWN pole sub-region — NOT collapse both to the
    /// cell centre. Centroids must be on opposite sides of x = 0 and
    /// separated by at least 30 % of the cell half-length.
    #[test]
    fn two_chromosomes_stay_segregated() {
        let recipe =
            Recipe::from_json_str(&replicating_chromosome_recipe_json(2)).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(42);

        let chr = out
            .snapshot
            .chromosome
            .as_ref()
            .expect("chromosome missing from snapshot");
        let half_len = 500.0_f32;

        // 2 chromosomes × 2 strands each (main + sister) = 4 strands total.
        let n_strands = chr.strands.len();
        assert_eq!(
            n_strands, 4,
            "expected 4 strands (2 chromosomes × 2 each), got {n_strands}"
        );
        let mid = n_strands / 2;
        let group0: Vec<&Point3<f32>> = chr.strands[..mid].iter().flatten().collect();
        let group1: Vec<&Point3<f32>> = chr.strands[mid..].iter().flatten().collect();

        let cx0 = group0.iter().map(|p| p.x).sum::<f32>() / group0.len() as f32;
        let cx1 = group1.iter().map(|p| p.x).sum::<f32>() / group1.len() as f32;

        assert!(
            cx0 * cx1 < 0.0,
            "chromosomes collapsed to the same side: cx0 = {cx0:.1}, cx1 = {cx1:.1}"
        );
        assert!(
            (cx1 - cx0).abs() > 0.3 * half_len,
            "chromosomes not sufficiently separated: cx0 = {cx0:.1}, cx1 = {cx1:.1}, \
             |Δx| = {:.1} (need > {:.1})",
            (cx1 - cx0).abs(),
            0.3 * half_len,
        );
    }

    // ---- BF2-3 tests: per-chromosome RNAP/RNA routing ----------------------

    /// Build a 2-chromosome replicating recipe with two explicit RNAPs:
    /// one on chromosome_index 0 and one on chromosome_index 1.  The cell is
    /// long enough (±500 Å) that the two chromosome groups segregate to
    /// clearly opposite poles — allowing a centroid-proximity assertion.
    fn recipe_two_chromosomes_two_rnaps() -> Recipe {
        let json = r#"{
            "bounding_box": [[-600,-200,-200],[600,200,200]],
            "objects": {
                "rna_polymerase": { "type": "single_sphere", "radius": 20 }
            },
            "composition": {
                "space": { "regions": { "interior": ["cell"] } },
                "cell": {
                    "compartment": {
                        "kind": "capsule",
                        "a": [-500, 0, 0],
                        "b": [500, 0, 0],
                        "radius": 80
                    },
                    "regions": { "interior": [] }
                }
            },
            "chromosome": {
                "beads": 500,
                "spacing": 10.0,
                "bead_radius": 5.0,
                "compartment": "cell",
                "n_chromosomes": 2,
                "fork_fraction": 0.45,
                "rnap_marker": "rna_polymerase",
                "rnaps": [
                    {"coordinates": 0, "domain_index": 0, "is_forward": true, "chromosome_index": 0},
                    {"coordinates": 0, "domain_index": 0, "is_forward": true, "chromosome_index": 1}
                ]
            }
        }"#;
        Recipe::from_json_str(json).expect("recipe_two_chromosomes_two_rnaps: parse failed")
    }

    /// BF2-3: each RNAP must land near its OWN chromosome's strand group, not
    /// always chromosome 0.  Before the fix both RNAPs route to strand 0
    /// (chromosome 0's main), so the chromosome-1 RNAP fails the proximity
    /// check for chromosome 1's centroid.
    #[test]
    fn rnap_routes_to_own_chromosome() {
        let recipe = recipe_two_chromosomes_two_rnaps();
        let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(42);

        let rnap_id = recipe.ingredients.get_full("rna_polymerase").unwrap().0 as u32;
        let rnaps: Vec<_> = out
            .snapshot
            .placements
            .iter()
            .filter(|p| p.ingredient_id == rnap_id)
            .collect();

        // Two non-daughter RNAPs → exactly 2 placements (no bubble overlay).
        assert_eq!(rnaps.len(), 2, "expected 2 RNAP placements, got {}", rnaps.len());

        let chr = out.snapshot.chromosome.as_ref().expect("chromosome missing");
        // 2 replicating chromosomes × 2 strands each = 4 strands total.
        assert_eq!(
            chr.strands.len(),
            4,
            "expected 4 strands (2 chrom × 2 each), got {}",
            chr.strands.len()
        );

        // Per-chromosome strand-group centroids (cell-relative; center = origin).
        let c0_beads: Vec<&Point3<f32>> = chr.strands[0..2].iter().flatten().collect();
        let c1_beads: Vec<&Point3<f32>> = chr.strands[2..4].iter().flatten().collect();
        let centroid = |beads: &[&Point3<f32>]| {
            let n = beads.len() as f32;
            Point3::new(
                beads.iter().map(|p| p.x).sum::<f32>() / n,
                beads.iter().map(|p| p.y).sum::<f32>() / n,
                beads.iter().map(|p| p.z).sum::<f32>() / n,
            )
        };
        let c0 = centroid(&c0_beads);
        let c1 = centroid(&c1_beads);

        // Chromosomes must be on opposite x sides (segregated).
        assert!(
            c0.x * c1.x < 0.0,
            "chromosomes not segregated: c0.x={:.1} c1.x={:.1}",
            c0.x, c1.x
        );

        // Recipe order: rnaps[0] = chromosome_index 0, rnaps[1] = chromosome_index 1.
        // After the fix: rnap 0 closer to c0, rnap 1 closer to c1.
        let d0_c0 = (rnaps[0].position - c0).norm();
        let d0_c1 = (rnaps[0].position - c1).norm();
        let d1_c0 = (rnaps[1].position - c0).norm();
        let d1_c1 = (rnaps[1].position - c1).norm();

        assert!(
            d0_c0 < d0_c1,
            "RNAP 0 (chromosome_index=0) not near c0: d_c0={d0_c0:.1} d_c1={d0_c1:.1}"
        );
        assert!(
            d1_c1 < d1_c0,
            "RNAP 1 (chromosome_index=1) not near c1: d_c1={d1_c1:.1} d_c0={d1_c0:.1}"
        );
    }

    /// A single replicating chromosome must have its bead centroid near the
    /// cell centre AND must show NO centerline-collapse artifact — i.e., no
    /// sheaf of DNA beads projected onto the long-axis centerline in the cap
    /// regions.
    ///
    /// The artifact (commit 1b80cab) was caused by `place_chromosome` rigidly
    /// shifting finished beads by `sub_off − centroid` and then clamping
    /// out-of-bounds beads with `inset.medial(p)`.  `medial()` on a capsule
    /// projects onto the x-axis centerline (y=z=0), creating a visible sheaf
    /// of straight DNA strands converging on the axis (measured: ~1.3 % of
    /// beads at radial < 250 Å in the production E. coli model).
    ///
    /// Collapse threshold: beads with
    ///   radial_distance_from_x_axis < 0.05 × cap_radius  (= 4.0 Å here)
    ///   AND |x| > 0.6 × half_len                         (= 300 Å here)
    /// must be ≤ max(⌈0.3 % × n_beads⌉, 2).
    /// The medial-clamp bug produces radial = 0 for every clamped bead, and
    /// even a small centroid drift pushes several beads past the cap.
    /// A naturally confined fiber has ≈ 0 such beads (the first SAW bead
    /// starts at the origin, on-axis, but at |x|=0 — not in the cap region).
    #[test]
    fn replicating_chromosome_centered_without_centerline_collapse() {
        // Seed 404 was selected diagnostically: it produces 24 clamped beads on
        // the buggy code (vs 0 for seed 42 which happens not to trigger clamping).
        let recipe =
            Recipe::from_json_str(&replicating_chromosome_recipe_json(1)).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(404);

        let chr = out
            .snapshot
            .chromosome
            .as_ref()
            .expect("chromosome missing from snapshot");
        let half_len = 500.0_f32;
        let cap_radius = 80.0_f32;

        let all_beads: Vec<&Point3<f32>> = chr.strands.iter().flatten().collect();
        assert!(!all_beads.is_empty(), "no strand beads in snapshot");
        let n = all_beads.len() as f32;

        // (i) Centroid must be near the cell centre (same as replicating_chromosome_is_centered).
        let cx = all_beads.iter().map(|p| p.x).sum::<f32>() / n;
        assert!(
            cx.abs() < 0.12 * half_len,
            "centroid x = {cx:.1} Å (|{:.3}| × half_len), must be < {:.1} Å",
            cx.abs() / half_len,
            0.12 * half_len,
        );

        // (ii) No centerline-collapse artifact.
        // For a bead at (x, y, z), the radial distance from the x-axis is
        // sqrt(y² + z²).  A bead clamped by `medial()` is placed at (t, 0, 0)
        // giving radial = 0.  The 4 Å threshold catches all such beads while
        // being generous enough to exclude natural near-axis beads: with
        // bead_radius = 5 Å the closest two beads can sit is 7.5 Å apart, so
        // even a bead leaning against the axis bead has radial ≥ 5 Å from the
        // centerline.  cap_region_axial = 300 Å excludes the origin (x=0)
        // where the very first bead starts.
        let cap_r_thresh = 0.05 * cap_radius; // 4.0 Å
        let axial_thresh = 0.6 * half_len;    // 300.0 Å
        let collapse_count: usize = all_beads
            .iter()
            .filter(|p| {
                let r = (p.y * p.y + p.z * p.z).sqrt();
                r < cap_r_thresh && p.x.abs() > axial_thresh
            })
            .count();
        // Allow at most 0.3 % of beads (minimum 2) to avoid brittleness.
        let max_allowed = ((0.003 * n).ceil() as usize).max(2);
        assert!(
            collapse_count <= max_allowed,
            "centerline-collapse artifact: {collapse_count} beads have radial < \
             {cap_r_thresh:.1} Å and |x| > {axial_thresh:.0} Å \
             (threshold = {max_allowed} = max(⌈0.3 %×{n}⌉, 2)); \
             medial-axis clamping is likely still active",
        );
    }

    /// C1-2: RnaStrand carries `unique_index` and `length_nt` from the RnaSpec.
    #[test]
    fn rna_strand_carries_unique_index_and_length_nt() {
        let json = r#"{
            "bounding_box": [[-500,-500,-500],[500,500,500]],
            "objects": {},
            "composition": {
                "space": { "regions": { "interior": ["cell"] } },
                "cell": {
                    "compartment": {"kind": "capsule", "a": [-150, 0, 0], "b": [150, 0, 0], "radius": 80},
                    "regions": { "interior": [] }
                }
            },
            "chromosome": {
                "beads": 1000, "spacing": 10.0, "bead_radius": 5.0, "compartment": "cell",
                "rnas": [{"root_coordinate": 100000, "root_domain": 0, "length_nt": 400,
                           "is_mRNA": true, "unique_index": 20}]
            }
        }"#;
        let recipe = Recipe::from_json_str(json).expect("parse failed");
        let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(7);
        assert_eq!(out.snapshot.rna_strands.len(), 1);
        assert_eq!(out.snapshot.rna_strands[0].unique_index, 20,
            "unique_index not carried onto RnaStrand");
        assert_eq!(out.snapshot.rna_strands[0].length_nt, 400,
            "length_nt not carried onto RnaStrand");
    }
}
