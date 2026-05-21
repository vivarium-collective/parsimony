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
use nalgebra::{Point3, Quaternion, UnitQuaternion};
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
fn random_rotation<R: Rng>(rng: &mut R) -> UnitQuaternion<f32> {
    let u1: f32 = rng.gen_range(0.0..1.0);
    let u2: f32 = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
    let u3: f32 = rng.gen_range(0.0..(2.0 * std::f32::consts::PI));
    let s1 = (1.0 - u1).sqrt();
    let s2 = u1.sqrt();
    let q = Quaternion::new(s2 * u3.cos(), s1 * u2.sin(), s1 * u2.cos(), s2 * u3.sin());
    UnitQuaternion::new_normalize(q)
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
                    let rot = align_to_normal(ingredient.principal_vector, n);
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
        let Some((center, cell_r)) = self.chromosome_cell(chr) else {
            return;
        };
        // Genome resolution: recipe value unless overridden (e.g. `pack
        // --chromosome-beads`). More beads → more DNA contour/volume.
        let beads = self.config.chromosome_beads.unwrap_or(chr.beads);
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
                    cell_r,
                    &alloc,
                    chr.spacing,
                    chr.bead_radius,
                    sc.radius,
                    sc.pitch,
                    rng,
                )
            }
            None => crate::fiber::generate_fiber(cell_r, beads, chr.spacing, chr.bead_radius, rng),
        };
        if !chr.proteins.is_empty() && pts.len() >= 2 {
            let fiber_world: Vec<Point3<f32>> =
                pts.iter().map(|p| center + p.coords).collect();
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
            // With a genome annotation, seat DNA-binding proteins at real
            // transcription / replication sites; otherwise spread them randomly.
            let binds = match chr
                .genome
                .as_ref()
                .and_then(|p| crate::genome::Genome::from_csv(p).ok())
            {
                Some(genome) => {
                    let abundances: Vec<(String, u32)> = self
                        .recipe
                        .directives
                        .iter()
                        .map(|d| (d.ingredient.clone(), d.count))
                        .collect();
                    let sites = genome.binding_sites(&chr.proteins, &abundances, rng);
                    let mut at: Vec<(u32, &Ingredient, f32)> = Vec::new();
                    for ((name, _), fracs) in chr.proteins.iter().zip(&sites) {
                        if let Some((idx, _, ing)) = self.recipe.ingredients.get_full(name) {
                            for &f in fracs {
                                at.push((idx as u32, ing, f));
                            }
                        }
                    }
                    crate::fiber_pack::pack_on_fiber_at(&fiber_world, &at, &obstacles, chr.bead_radius, rng)
                }
                None => {
                    let proteins = self.resolve_fiber_proteins(chr);
                    crate::fiber_pack::pack_on_fiber(&fiber_world, &proteins, &obstacles, chr.bead_radius, rng)
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
        snapshot.chromosome = Some(crate::placement::Chromosome {
            center,
            radius: chr.bead_radius,
            color: chr.color,
            points: pts,
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
                    (p, align_to_normal(ingredient.principal_vector, n))
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
    ) -> Option<(Point3<f32>, f32)> {
        for (name, comp) in &self.recipe.compartments {
            if let Some(want) = &chr.compartment {
                if name != want {
                    continue;
                }
            }
            if let crate::compartment::CompartmentKind::Sphere { center, radius } = &comp.kind {
                return Some((*center, *radius));
            }
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
}
