//! Greedy random placer. For each placement directive, attempt up to
//! `jitter_attempts` random points in the target compartment and
//! place the instance if there's no collision with already-packed
//! instances. The original cellPACK algorithm, simplified.
//!
//! Collision is sphere-vs-sphere: the QBVH broad-phase narrows
//! candidates by AABB, and we tighten with the exact sphere distance
//! test. Phase 2 MVP supports only `SingleSphere` ingredients; once
//! sphere-tree representations land we'll swap in proper tree-vs-tree
//! collision behind the same outer loop.

use indexmap::IndexMap;
use nalgebra::{Point3, Quaternion, UnitQuaternion};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

use parsimony_spatial::{
    Aabb, Cell, CellCoord, QbvhIndex, Sphere, SpatialIndex, VoxelField, OCCUPIED as VOXEL_OCCUPIED,
};

use crate::compartment::align_to_normal;
use crate::ingredient::IngredientShape;
use crate::placement::{Placement, Snapshot};
use crate::recipe::{PlacementDirective, Recipe, RegionKind};

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

#[derive(Debug, Clone, Copy)]
pub struct PlacerConfig {
    /// Hard cap on per-instance placement attempts; overrides the
    /// recipe's `jitter_attempts` when smaller (acts as a global ceiling).
    pub max_attempts_per_instance: u32,
    /// Default `jitter_attempts` for ingredients that don't specify one.
    pub default_jitter_attempts: u32,
    /// Use the [`VoxelField`](parsimony_spatial::VoxelField) for a fast
    /// broad-phase rejection step before the (more expensive) sphere-
    /// tree collision test. The voxel field tracks occupied AABBs at a
    /// coarse resolution; if `is_region_free` says the candidate's
    /// AABB is already touching marked cells, skip the QBVH query.
    pub use_voxel_assist: bool,
    /// Cell size of the voxel-assist field, in world units. `None`
    /// means autodetect from the recipe (smallest ingredient radius
    /// divided by 2, clamped to at least 1.0). Smaller cells give
    /// tighter rejection but cost more memory and update time.
    pub voxel_cell_size: Option<f32>,
}

impl Default for PlacerConfig {
    fn default() -> Self {
        Self {
            max_attempts_per_instance: 200,
            default_jitter_attempts: 20,
            use_voxel_assist: true,
            voxel_cell_size: None,
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

        // Voxel-assist field: a coarse occupancy grid we mark on each
        // placement. Used as a fast `is_region_free` pre-check before
        // the (more expensive) QBVH sphere-tree query. cellPACK does
        // an equivalent thing with its `distToClosestSurf` grid — we
        // get the same hot-path speedup, which lets us afford many
        // more attempts per big-sphere placement and significantly
        // narrows the dense-recipe gap.
        let voxel_cell_size = self.config.voxel_cell_size.unwrap_or_else(|| {
            let min_r = self
                .recipe
                .ingredients
                .values()
                .map(|i| i.shape.enclosing_radius())
                .fold(f32::INFINITY, f32::min);
            (min_r * 0.5).max(1.0)
        });
        let mut voxel: Option<VoxelField> = if self.config.use_voxel_assist {
            Some(VoxelField::new(voxel_cell_size))
        } else {
            None
        };
        let voxel_mark = Cell::new(0, VOXEL_OCCUPIED, 0);

        // Stochastic interleaved placement, matching cellPACK's
        // `pickIngredient`: at each step pick a directive weighted by
        // remaining count. Big ingredients keep getting attempts
        // throughout the run while space is gradually filling, instead
        // of getting drained first or last. Significantly closer to
        // cellPACK's output on dense recipes.
        let directives: Vec<&PlacementDirective> = self.recipe.directives.iter().collect();
        let mut remaining: Vec<u32> = directives.iter().map(|d| d.count).collect();
        let mut consecutive_rejections: Vec<u32> = vec![0; directives.len()];
        let mut per_ingredient_attempts: Vec<u64> = vec![0; directives.len()];
        let mut per_ingredient_placed: Vec<usize> = vec![0; directives.len()];
        let total_requested: u32 = remaining.iter().sum();
        stats.requested = total_requested as usize;

        // Cap on consecutive rejections per directive before we declare
        // it stuck. Counter resets on every successful placement, so a
        // directive with `N` instances effectively gets up to `N * cap`
        // attempts at sparse success rates. With voxel-assist the
        // per-attempt cost is small (one `is_region_free` query, O(8
        // tiles) for typical ingredient sizes), so we can afford big
        // numbers here.
        let has_voxel = voxel.is_some();
        let rejection_threshold = |dir_idx: usize| -> u32 {
            let ingredient = self
                .recipe
                .ingredients
                .get(&directives[dir_idx].ingredient)
                .expect("known ingredient");
            let base = ingredient
                .jitter_attempts
                .max(self.config.default_jitter_attempts)
                .min(self.config.max_attempts_per_instance);
            if has_voxel {
                base.saturating_mul(1_000)
            } else {
                base.saturating_mul(50)
            }
        };

        loop {
            // cellPACK's default `pickIngredient`: uniform random pick
            // over directives that still have something to place (and
            // haven't hit their rejection threshold). This gives big
            // ingredients a fair share of attempts throughout the run
            // — weighting by remaining count would bias toward
            // high-count small ingredients and starve the big ones.
            let live: Vec<usize> = (0..directives.len())
                .filter(|&i| {
                    remaining[i] > 0 && consecutive_rejections[i] < rejection_threshold(i)
                })
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

            // Sample position and rotation depending on the directive's
            // region. Interior: uniform sample in the compartment
            // (rejecting points inside child compartments); rotation
            // is random for multi-sphere ingredients. Surface: sample
            // on the compartment's boundary; rotation aligns the
            // ingredient's `principal_vector` with the outward normal.
            let (pos, rotation) = match directive.region {
                RegionKind::Interior => {
                    let Some(pos) = sample_interior_excluding_children(
                        compartment,
                        &self.recipe.compartments,
                        enclosing_radius,
                        &mut rng,
                        16,
                    ) else {
                        consecutive_rejections[dir_idx] = rejection_threshold(dir_idx);
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
                    let (p, n) = compartment.kind.sample_surface(&mut rng);
                    let rot = align_to_normal(ingredient.principal_vector, n);
                    (p, rot)
                }
            };

            let candidate_aabb = Aabb::from_sphere(pos, enclosing_radius);

            // Voxel pre-check: marks union of placed spheres; if any
            // cell in the candidate's AABB is OCCUPIED, skip.
            if let Some(vf) = &voxel
                && !vf.is_region_free(candidate_aabb)
            {
                consecutive_rejections[dir_idx] =
                    consecutive_rejections[dir_idx].saturating_add(1);
                continue;
            }

            // Precise sphere-tree collision check (handles multi-sphere).
            if self.collides_with_existing(
                &ingredient.shape,
                pos,
                rotation,
                &index,
                &shapes_by_uid,
                &centers_by_uid,
                &rotations_by_uid,
            ) {
                consecutive_rejections[dir_idx] = consecutive_rejections[dir_idx].saturating_add(1);
                continue;
            }

            // Place it.
            let uid = next_uid;
            next_uid += 1;
            let aabb = candidate_aabb;
            index.insert(uid, aabb).expect("uid collision");
            shapes_by_uid.push(&ingredient.shape);
            centers_by_uid.push(pos);
            rotations_by_uid.push(rotation);

            if let Some(vf) = &mut voxel {
                // Mark every proxy sphere of the placed ingredient.
                for (c, r) in ingredient.shape.world_spheres(pos, rotation) {
                    mark_sphere_cells(vf, c, r, voxel_mark);
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
            consecutive_rejections[dir_idx] = 0;
            per_ingredient_placed[dir_idx] += 1;
            stats.placed += 1;
            stats.successful_attempts += 1;
        }

        for (i, directive) in directives.iter().enumerate() {
            stats.per_ingredient.push((
                directive.ingredient.clone(),
                per_ingredient_placed[i],
                directive.count as usize,
                per_ingredient_attempts[i],
            ));
        }
        PlacerOutcome { snapshot, stats }
    }

    /// Tree-vs-tree sphere collision against already-placed instances.
    /// QBVH broad-phase narrows to candidates whose enclosing spheres
    /// could overlap; inside, we walk every proxy-sphere pair across
    /// the candidate and the hit instance and reject on any
    /// center-distance <= sum-of-radii.
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
            // Fast outer cull: enclosing-sphere distance check.
            let outer_d2 = (candidate_pos - other_center).norm_squared();
            let outer_r = candidate_r + other_shape.enclosing_radius();
            if outer_d2 > outer_r * outer_r {
                return;
            }
            // Tree-vs-tree.
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
}

/// Sample a point inside `compartment` that's *not* inside any of its
/// child compartments. Up to `attempts` retries; returns `None` if all
/// samples landed inside children (or the compartment is too small for
/// the required clearance).
fn sample_interior_excluding_children<R: Rng>(
    compartment: &crate::compartment::Compartment,
    all_compartments: &IndexMap<String, crate::compartment::Compartment>,
    radius: f32,
    rng: &mut R,
    attempts: usize,
) -> Option<Point3<f32>> {
    if compartment.children.is_empty() {
        return compartment.kind.sample_interior_for_sphere(radius, rng);
    }
    let child_compartments: Vec<&crate::compartment::Compartment> = compartment
        .children
        .iter()
        .filter_map(|&id| all_compartments.get_index(id as usize).map(|(_, c)| c))
        .collect();
    for _ in 0..attempts {
        let p = compartment.kind.sample_interior_for_sphere(radius, rng)?;
        let inside_child = child_compartments.iter().any(|c| c.kind.contains(p));
        if !inside_child {
            return Some(p);
        }
    }
    None
}

/// Mark cells inside a sphere (not its AABB) — only cells whose centres
/// lie within `radius` of `center`. Avoids the 47% over-mark of the
/// corner region you get from `mark_aabb`, which would otherwise
/// produce false-positive rejections in the placer's pre-check.
fn mark_sphere_cells(voxel: &mut VoxelField, center: Point3<f32>, radius: f32, mark: Cell) {
    let aabb = Aabb::from_sphere(center, radius);
    let (lo, hi) = voxel.aabb_to_cell_range(aabb);
    if lo.x > hi.x || lo.y > hi.y || lo.z > hi.z {
        return;
    }
    let r2 = radius * radius;
    for cz in lo.z..=hi.z {
        for cy in lo.y..=hi.y {
            for cx in lo.x..=hi.x {
                let coord = CellCoord::new(cx, cy, cz);
                let c = voxel.cell_center(coord);
                let dx = c.x - center.x;
                let dy = c.y - center.y;
                let dz = c.z - center.z;
                if dx * dx + dy * dy + dz * dz <= r2 {
                    voxel.put(coord, mark);
                }
            }
        }
    }
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
        use crate::recipe::Recipe;
        let recipe = Recipe::from_json_str(DUMBBELLS_IN_A_BOX).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(0xC0DE);
        assert!(!out.snapshot.placements.is_empty(), "expected some dumbbells placed");
        // Each placement should have a non-identity rotation (since shape needs_rotation()).
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
        use crate::recipe::Recipe;
        let recipe = Recipe::from_json_str(NESTED_CAPSULE).unwrap();
        assert_eq!(recipe.compartments.len(), 2, "space + cell");
        let cell = &recipe.compartments["cell"];
        assert!(cell.parent.is_some(), "cell compartment should have parent");
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(0xC0DE);
        assert!(out.snapshot.placements.len() > 50, "expected most placements to fit");

        // Every placement should be associated with the cell compartment.
        for p in &out.snapshot.placements {
            let comp = recipe.compartments.get_index(p.compartment_id as usize).unwrap().1;
            assert_eq!(comp.name, "cell");
        }

        // Surface placements (the lipid ingredient) should sit on the
        // capsule boundary — signed distance ≈ 0.
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
        use crate::recipe::Recipe;
        let recipe = Recipe::from_json_str(DUMBBELLS_IN_A_BOX).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(0xFADE);
        // Collect every world-space proxy sphere across every placement.
        let mut all_spheres: Vec<(Point3<f32>, f32)> = Vec::new();
        for p in &out.snapshot.placements {
            let ing = recipe.ingredients.get_index(p.ingredient_id as usize).unwrap().1;
            all_spheres.extend(ing.shape.world_spheres(p.position, p.rotation));
        }
        // O(n²) all-pairs distance check.
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
}
