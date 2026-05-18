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
use nalgebra::{Point3, UnitQuaternion};
use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use parsimony_spatial::{Aabb, QbvhIndex, Sphere, SpatialIndex};

use crate::ingredient::IngredientShape;
use crate::placement::{Placement, Snapshot};
use crate::recipe::{PlacementDirective, Recipe};

#[derive(Debug, Clone, Copy)]
pub struct PlacerConfig {
    /// Hard cap on per-instance placement attempts; overrides the
    /// recipe's `jitter_attempts` when smaller (acts as a global ceiling).
    pub max_attempts_per_instance: u32,
    /// Default `jitter_attempts` for ingredients that don't specify one.
    pub default_jitter_attempts: u32,
}

impl Default for PlacerConfig {
    fn default() -> Self {
        Self {
            max_attempts_per_instance: 200,
            default_jitter_attempts: 20,
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
        let mut radii_by_uid: Vec<f32> = Vec::new();
        let mut centers_by_uid: Vec<Point3<f32>> = Vec::new();
        let mut next_uid: u64 = 0;
        let mut stats = PlacerStats::default();

        // Place ingredients in priority-then-size-descending order:
        // higher priority first, then within a tie, biggest ingredients
        // first (so they find space while the compartment is still
        // empty). cellPACK's pickIngredient uses stochastic weighted
        // selection — we'll match that later when faithfulness becomes
        // the bottleneck; for now, biggest-first is the standard
        // packing heuristic and produces good fits on dense recipes.
        let mut directives: Vec<&PlacementDirective> = self.recipe.directives.iter().collect();
        directives.sort_by(|a, b| {
            let a_size = self
                .recipe
                .ingredients
                .get(&a.ingredient)
                .map(|i| i.shape.enclosing_radius())
                .unwrap_or(0.0);
            let b_size = self
                .recipe
                .ingredients
                .get(&b.ingredient)
                .map(|i| i.shape.enclosing_radius())
                .unwrap_or(0.0);
            b.priority
                .partial_cmp(&a.priority)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(
                    b_size
                        .partial_cmp(&a_size)
                        .unwrap_or(std::cmp::Ordering::Equal),
                )
        });

        for directive in directives {
            let ingredient = self
                .recipe
                .ingredients
                .get(&directive.ingredient)
                .unwrap_or_else(|| panic!("directive references unknown ingredient `{}`", directive.ingredient));
            let compartment = self
                .recipe
                .compartments
                .get(&directive.compartment)
                .unwrap_or_else(|| panic!("directive references unknown compartment `{}`", directive.compartment));

            let radius = match ingredient.shape {
                IngredientShape::SingleSphere { radius } => radius,
            };

            let jitter_attempts = ingredient
                .jitter_attempts
                .max(self.config.default_jitter_attempts)
                .min(self.config.max_attempts_per_instance);

            stats.requested += directive.count as usize;
            let mut placed_here = 0usize;
            let mut attempts_here = 0u64;

            for _ in 0..directive.count {
                let mut placed_this_one = false;
                for _attempt in 0..jitter_attempts {
                    attempts_here += 1;
                    stats.total_attempts += 1;

                    let Some(pos) =
                        compartment.kind.sample_interior_for_sphere(radius, &mut rng)
                    else {
                        // Compartment too small for this ingredient — skip
                        // the rest of its count for this directive.
                        break;
                    };

                    if !self.collides_with_existing(
                        pos,
                        radius,
                        &index,
                        &radii_by_uid,
                        &centers_by_uid,
                    ) {
                        // Place it.
                        let uid = next_uid;
                        next_uid += 1;
                        let aabb = Aabb::from_sphere(pos, radius);
                        index.insert(uid, aabb).expect("uid collision");
                        radii_by_uid.push(radius);
                        centers_by_uid.push(pos);
                        snapshot.placements.push(Placement {
                            instance_uid: uid,
                            ingredient_id: self.ingredient_ids[&directive.ingredient],
                            variant_id: 0,
                            compartment_id: self.compartment_ids[&directive.compartment],
                            position: pos,
                            rotation: UnitQuaternion::identity(),
                        });
                        placed_here += 1;
                        stats.placed += 1;
                        stats.successful_attempts += 1;
                        placed_this_one = true;
                        break;
                    }
                }
                if !placed_this_one {
                    // give up on this instance — too many rejections
                }
            }

            stats.per_ingredient.push((
                directive.ingredient.clone(),
                placed_here,
                directive.count as usize,
                attempts_here,
            ));
        }

        let _ = ingredient_check(self.recipe); // silence unused warning if no callers
        PlacerOutcome { snapshot, stats }
    }

    /// Sphere-vs-sphere collision test against already-placed instances.
    /// AABB broad-phase via QBVH; precise sphere distance check inside.
    fn collides_with_existing(
        &self,
        pos: Point3<f32>,
        radius: f32,
        index: &QbvhIndex,
        radii: &[f32],
        centers: &[Point3<f32>],
    ) -> bool {
        // Largest radius we've packed so far (worst case for the query).
        // For phase 2 with up to a few hundred placements we can iterate
        // hits and use each one's exact radius — no need for a global
        // maximum sphere cache.
        let mut collision = false;
        let query = Sphere::new(pos, radius + max_radius(radii));
        index.query_sphere(&query, |uid| {
            if collision {
                return;
            }
            let other_idx = uid as usize;
            let other_radius = radii[other_idx];
            let other_center = centers[other_idx];
            let dx = pos.x - other_center.x;
            let dy = pos.y - other_center.y;
            let dz = pos.z - other_center.z;
            let d2 = dx * dx + dy * dy + dz * dz;
            let r_sum = radius + other_radius;
            if d2 < r_sum * r_sum {
                collision = true;
            }
        });
        collision
    }
}

fn max_radius(radii: &[f32]) -> f32 {
    radii.iter().cloned().fold(0.0_f32, f32::max)
}

fn ingredient_check(_r: &Recipe) -> usize {
    0
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
