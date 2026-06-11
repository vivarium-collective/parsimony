//! Quantitative pack metrics — the "ruler" for validating a packing and
//! for comparing recipes / backends (roadmap A1; design §13).
//!
//! Given a [`Snapshot`] and its [`Recipe`], [`compute`] reports:
//!
//! - **Fill** — placed vs. requested, overall and per ingredient.
//! - **Overlaps** — number of overlapping instance pairs and the
//!   penetration depth, from an *exact* proxy-sphere-vs-proxy-sphere
//!   test. A clean packing has none; this is the correctness check the
//!   integration tests can share instead of hand-rolling an O(n²) loop.
//! - **Nearest-neighbour** distance distribution — centre-to-centre and
//!   surface-gap (negative gap ⇒ overlap).
//! - **Pair-correlation g(r)** — the radial distribution function, the
//!   standard structural fingerprint.
//! - **Free space** — a Monte-Carlo estimate of the occupied fraction
//!   and the void-size (largest-empty-ball radius) distribution.
//!
//! Everything heavier than O(n) is accelerated by a [`QbvhIndex`] over
//! enclosing spheres, so the metrics scale to whole-cell packs the same
//! way the placer does.
//!
//! The geometry metrics are pure functions of geometry + domain
//! ([`geometry_metrics`] over a slice of [`InstanceGeom`]), so they can
//! be unit-tested on hand-built configurations without a recipe.

use nalgebra::Point3;
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;
use serde::Serialize;

use parsimony_spatial::{Aabb, QbvhIndex, SpatialIndex, Sphere};

use crate::ingredient::IngredientId;
use crate::placement::Snapshot;
use crate::recipe::Recipe;

/// Two proxy spheres count as overlapping when they interpenetrate by
/// more than this (Å) — a small tolerance to ignore floating-point
/// "just touching" contacts.
const OVERLAP_EPS: f32 = 1e-3;

/// Geometry of one placed instance, flattened for metrics: its centre,
/// its broad-phase enclosing radius, and its world-space proxy spheres.
#[derive(Debug, Clone)]
pub struct InstanceGeom {
    pub ingredient_id: IngredientId,
    pub center: Point3<f32>,
    pub enclosing_radius: f32,
    /// World-space `(centre, radius)` proxy spheres (exact collision rep).
    pub spheres: Vec<(Point3<f32>, f32)>,
}

/// Knobs for the expensive (sampling / histogram) metrics. Defaults are
/// tuned for interactive use on demo-scale packs.
#[derive(Debug, Clone, Copy)]
pub struct MetricsConfig {
    /// Number of g(r) bins. `0` disables the RDF.
    pub rdf_bins: usize,
    /// g(r) cutoff radius (Å). `None` picks an automatic local cutoff.
    pub rdf_r_max: Option<f32>,
    /// Monte-Carlo samples for the free-space estimate. `0` disables it.
    pub free_space_samples: usize,
    /// Seed for the free-space sampler (determinism).
    pub seed: u64,
}

impl Default for MetricsConfig {
    fn default() -> Self {
        Self {
            rdf_bins: 64,
            rdf_r_max: None,
            free_space_samples: 8192,
            seed: 0x5EED_1234,
        }
    }
}

/// Summary statistics over a sample of scalars.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct Stats {
    pub n: usize,
    pub min: f32,
    pub max: f32,
    pub mean: f32,
    pub median: f32,
    pub stddev: f32,
    pub p10: f32,
    pub p90: f32,
}

impl Stats {
    /// Compute stats from an unsorted scalar sample (sorted in place).
    fn of(values: &mut [f32]) -> Stats {
        let n = values.len();
        if n == 0 {
            return Stats::default();
        }
        values.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let mut sum = 0.0_f64;
        let mut sumsq = 0.0_f64;
        for &v in values.iter() {
            sum += v as f64;
            sumsq += (v as f64) * (v as f64);
        }
        let mean = sum / n as f64;
        let var = (sumsq / n as f64 - mean * mean).max(0.0);
        let pct = |q: f32| values[(((n - 1) as f32 * q).round() as usize).min(n - 1)];
        Stats {
            n,
            min: values[0],
            max: values[n - 1],
            mean: mean as f32,
            median: pct(0.5),
            stddev: var.sqrt() as f32,
            p10: pct(0.10),
            p90: pct(0.90),
        }
    }
}

/// Overlap report from the exact proxy-sphere test.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct OverlapReport {
    /// Number of distinct instance pairs that interpenetrate.
    pub pair_count: usize,
    /// Number of instances involved in at least one overlap.
    pub instance_count: usize,
    /// Deepest interpenetration over all overlapping proxy pairs (Å).
    pub max_depth: f32,
    /// Mean interpenetration over overlapping instance pairs (Å).
    pub mean_depth: f32,
}

/// Nearest-neighbour distance distributions.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct NearestNeighbor {
    /// Centre-to-centre distance to each instance's nearest neighbour.
    pub center: Stats,
    /// Surface gap (centre distance − both enclosing radii). Negative ⇒
    /// the enclosing spheres overlap.
    pub surface_gap: Stats,
}

/// Radial distribution function g(r).
#[derive(Debug, Clone, Default, Serialize)]
pub struct Rdf {
    pub bin_width: f32,
    pub r_max: f32,
    pub number_density: f32,
    /// Bin centre radii (Å).
    pub r: Vec<f32>,
    /// g(r) at each bin centre.
    pub g: Vec<f32>,
    /// Raw ordered-pair counts per bin (each unordered pair counted twice).
    pub counts: Vec<u64>,
}

/// Monte-Carlo free-space / void estimate.
#[derive(Debug, Clone, Copy, Default, Serialize)]
pub struct FreeSpace {
    pub samples: usize,
    /// Fraction of samples that fell inside some instance.
    pub occupied_fraction: f32,
    /// Largest-empty-ball radius at the *free* samples (the void-size
    /// distribution).
    pub clearance: Stats,
}

/// Geometry-only metrics (independent of recipe bookkeeping).
#[derive(Debug, Clone, Default, Serialize)]
pub struct GeometryMetrics {
    pub n: usize,
    pub overlaps: OverlapReport,
    pub nearest_neighbor: NearestNeighbor,
    pub rdf: Option<Rdf>,
    pub free_space: Option<FreeSpace>,
}

/// Placed-vs-requested fill for one ingredient.
#[derive(Debug, Clone, Serialize)]
pub struct IngredientFill {
    pub name: String,
    pub placed: usize,
    pub requested: usize,
}

/// The full metrics report for a packing.
#[derive(Debug, Clone, Serialize)]
pub struct PackMetrics {
    pub recipe_name: String,
    pub seed: u64,
    pub n_placed: usize,
    /// Sum of directive counts (generated instances — DNA/RNA/lipid
    /// segments — are not directive-requested, so `n_placed` can exceed
    /// this on whole-cell recipes).
    pub n_requested: usize,
    pub fraction_placed: f32,
    pub domain_volume: f32,
    pub per_ingredient: Vec<IngredientFill>,
    pub geometry: GeometryMetrics,
}

/// Flatten a [`Snapshot`] into per-instance geometry using each
/// placement's ingredient shape (world proxy spheres at its pose).
pub fn instance_geometry(snapshot: &Snapshot, recipe: &Recipe) -> Vec<InstanceGeom> {
    snapshot
        .placements
        .iter()
        .filter_map(|p| {
            let (_, ing) = recipe.ingredients.get_index(p.ingredient_id as usize)?;
            Some(InstanceGeom {
                ingredient_id: p.ingredient_id,
                center: p.position,
                enclosing_radius: ing.shape.enclosing_radius(),
                spheres: ing.shape.world_spheres(p.position, p.rotation).collect(),
            })
        })
        .collect()
}

/// Full metrics for a packing: fill bookkeeping + geometry.
pub fn compute(snapshot: &Snapshot, recipe: &Recipe, cfg: &MetricsConfig) -> PackMetrics {
    let instances = instance_geometry(snapshot, recipe);
    let domain = recipe.bounding_box;
    let geometry = geometry_metrics(&instances, domain, cfg);

    // Requested per ingredient name (summed across directives).
    let mut requested: std::collections::HashMap<&str, usize> = std::collections::HashMap::new();
    for d in &recipe.directives {
        *requested.entry(d.ingredient.as_str()).or_insert(0) += d.count as usize;
    }
    // Placed per ingredient id.
    let mut placed = vec![0usize; recipe.ingredients.len()];
    for p in &snapshot.placements {
        if let Some(slot) = placed.get_mut(p.ingredient_id as usize) {
            *slot += 1;
        }
    }
    let per_ingredient: Vec<IngredientFill> = recipe
        .ingredients
        .keys()
        .enumerate()
        .filter_map(|(i, name)| {
            let req = requested.get(name.as_str()).copied().unwrap_or(0);
            let pl = placed[i];
            (req > 0 || pl > 0).then(|| IngredientFill {
                name: name.clone(),
                placed: pl,
                requested: req,
            })
        })
        .collect();

    let n_placed = snapshot.placements.len();
    let n_requested: usize = recipe.directives.iter().map(|d| d.count as usize).sum();
    PackMetrics {
        recipe_name: snapshot.recipe_name.clone(),
        seed: snapshot.seed,
        n_placed,
        n_requested,
        fraction_placed: n_placed as f32 / n_requested.max(1) as f32,
        domain_volume: domain.volume(),
        per_ingredient,
        geometry,
    }
}

/// Geometry metrics over a flat instance list. Pure function of the
/// geometry and the reference `domain` (used for number density and
/// free-space sampling) — the unit-testable core.
pub fn geometry_metrics(
    instances: &[InstanceGeom],
    domain: Aabb,
    cfg: &MetricsConfig,
) -> GeometryMetrics {
    let n = instances.len();
    if n == 0 {
        return GeometryMetrics::default();
    }

    // One broad-phase index over enclosing spheres, shared by every pass.
    let mut index = QbvhIndex::new();
    for (i, g) in instances.iter().enumerate() {
        index
            .insert(i as u64, Aabb::from_sphere(g.center, g.enclosing_radius))
            .expect("unique instance uid");
    }
    let max_enclosing = instances
        .iter()
        .map(|g| g.enclosing_radius)
        .fold(0.0_f32, f32::max);

    GeometryMetrics {
        n,
        overlaps: overlap_report(instances, &index),
        nearest_neighbor: nearest_neighbor(instances, &index, &domain),
        rdf: rdf(instances, &index, &domain, cfg),
        free_space: free_space(instances, &index, &domain, max_enclosing, cfg),
    }
}

/// Exact overlap detection. For each instance we query the index for
/// neighbours whose enclosing sphere overlaps ours — querying a sphere
/// of radius `enclosing_radius` around our centre is sufficient: the
/// stored AABB of a neighbour `j` contains its enclosing sphere, so the
/// closest point of `AABB_j` to our centre is ≤ `|d| − r_j ≤ r_i`
/// whenever the enclosing spheres overlap (`|d| ≤ r_i + r_j`). We then
/// run the exact proxy-sphere test on each candidate pair.
fn overlap_report(instances: &[InstanceGeom], index: &QbvhIndex) -> OverlapReport {
    let mut pair_count = 0usize;
    let mut depth_sum = 0.0_f64;
    let mut max_depth = 0.0_f32;
    let mut involved = vec![false; instances.len()];

    for (i, gi) in instances.iter().enumerate() {
        let q = Sphere::new(gi.center, gi.enclosing_radius);
        let mut neighbors: Vec<usize> = Vec::new();
        index.query_sphere(&q, |uid| {
            let j = uid as usize;
            if j > i {
                neighbors.push(j);
            }
        });
        for j in neighbors {
            let gj = &instances[j];
            // Deepest interpenetration over all proxy-sphere pairs.
            let mut depth = 0.0_f32;
            for &(ca, ra) in &gi.spheres {
                for &(cb, rb) in &gj.spheres {
                    let d = (ca - cb).norm();
                    depth = depth.max(ra + rb - d);
                }
            }
            if depth > OVERLAP_EPS {
                pair_count += 1;
                depth_sum += depth as f64;
                max_depth = max_depth.max(depth);
                involved[i] = true;
                involved[j] = true;
            }
        }
    }

    OverlapReport {
        pair_count,
        instance_count: involved.iter().filter(|x| **x).count(),
        max_depth,
        mean_depth: if pair_count == 0 {
            0.0
        } else {
            (depth_sum / pair_count as f64) as f32
        },
    }
}

/// Nearest-neighbour centre distance + surface gap for each instance,
/// via an expanding-radius query (doubling until the best hit lies
/// inside the queried ball, so it is provably the global nearest).
fn nearest_neighbor(
    instances: &[InstanceGeom],
    index: &QbvhIndex,
    domain: &Aabb,
) -> NearestNeighbor {
    let n = instances.len();
    if n < 2 {
        return NearestNeighbor::default();
    }
    let diag = domain.extents().norm().max(1.0);
    let mean_encl =
        instances.iter().map(|g| g.enclosing_radius).sum::<f32>() / n as f32;

    let mut centers = Vec::with_capacity(n);
    let mut gaps = Vec::with_capacity(n);
    for (i, gi) in instances.iter().enumerate() {
        let mut r = (gi.enclosing_radius + mean_encl).max(1e-3) * 2.0;
        let mut best;
        let mut best_j;
        loop {
            best = f32::INFINITY;
            best_j = usize::MAX;
            let q = Sphere::new(gi.center, r);
            index.query_sphere(&q, |uid| {
                let j = uid as usize;
                if j != i {
                    let d = (gi.center - instances[j].center).norm();
                    if d < best {
                        best = d;
                        best_j = j;
                    }
                }
            });
            // A hit at distance ≤ r is the true nearest: every instance
            // whose centre is within r is reported (its centre lies in
            // its own AABB), so nothing closer was missed.
            if best.is_finite() && best <= r {
                break;
            }
            r *= 2.0;
            if r > diag {
                break;
            }
        }
        if best_j != usize::MAX {
            centers.push(best);
            gaps.push(best - gi.enclosing_radius - instances[best_j].enclosing_radius);
        }
    }

    NearestNeighbor {
        center: Stats::of(&mut centers),
        surface_gap: Stats::of(&mut gaps),
    }
}

/// Pair-correlation function g(r) up to a local cutoff. Counts ordered
/// pairs within `r_max` per bin (each unordered pair appears twice, once
/// from each end) and normalises by the ideal-gas expectation
/// `N · ρ · shell_volume`, where `ρ = N / V`.
fn rdf(
    instances: &[InstanceGeom],
    index: &QbvhIndex,
    domain: &Aabb,
    cfg: &MetricsConfig,
) -> Option<Rdf> {
    let n = instances.len();
    if cfg.rdf_bins == 0 || n < 2 {
        return None;
    }
    let volume = domain.volume();
    if volume <= 0.0 {
        return None;
    }
    let min_extent = domain.extents().min();
    let mean_encl =
        instances.iter().map(|g| g.enclosing_radius).sum::<f32>() / n as f32;
    let r_max = cfg
        .rdf_r_max
        .unwrap_or_else(|| (12.0 * mean_encl).min(min_extent * 0.5))
        .max(1e-3);
    let bins = cfg.rdf_bins;
    let dr = r_max / bins as f32;

    let mut counts = vec![0u64; bins];
    for (i, gi) in instances.iter().enumerate() {
        let q = Sphere::new(gi.center, r_max);
        index.query_sphere(&q, |uid| {
            let j = uid as usize;
            if j != i {
                let d = (gi.center - instances[j].center).norm();
                if d < r_max {
                    let b = ((d / dr) as usize).min(bins - 1);
                    counts[b] += 1;
                }
            }
        });
    }

    let number_density = n as f32 / volume;
    let mut r = Vec::with_capacity(bins);
    let mut g = Vec::with_capacity(bins);
    for (k, &count) in counts.iter().enumerate() {
        let r_lo = k as f32 * dr;
        let r_hi = r_lo + dr;
        r.push(r_lo + dr * 0.5);
        let shell = (4.0 / 3.0) * std::f32::consts::PI * (r_hi.powi(3) - r_lo.powi(3));
        let ideal = n as f32 * number_density * shell;
        g.push(if ideal > 0.0 {
            count as f32 / ideal
        } else {
            0.0
        });
    }

    Some(Rdf {
        bin_width: dr,
        r_max,
        number_density,
        r,
        g,
        counts,
    })
}

/// Monte-Carlo free-space estimate. For each uniformly-random point in
/// the domain, find the nearest proxy-sphere *surface* (the radius of
/// the largest empty ball centred there). A non-positive distance means
/// the point is inside an instance (occupied); positive values form the
/// void-size distribution.
fn free_space(
    instances: &[InstanceGeom],
    index: &QbvhIndex,
    domain: &Aabb,
    max_enclosing: f32,
    cfg: &MetricsConfig,
) -> Option<FreeSpace> {
    if cfg.free_space_samples == 0 {
        return None;
    }
    let diag = domain.extents().norm().max(1.0);
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(cfg.seed);
    let mut occupied = 0usize;
    let mut clearances = Vec::with_capacity(cfg.free_space_samples);

    for _ in 0..cfg.free_space_samples {
        let p = Point3::new(
            rng.gen_range(domain.min.x..=domain.max.x),
            rng.gen_range(domain.min.y..=domain.max.y),
            rng.gen_range(domain.min.z..=domain.max.z),
        );
        // Expanding search for the nearest proxy surface. An instance not
        // reported within radius R has all proxies at surface distance
        // > R − max_enclosing, so once `best ≤ R − max_enclosing` no
        // unseen instance can beat it.
        let mut r = max_enclosing.max(1e-3) * 2.0;
        let mut best;
        loop {
            best = f32::INFINITY;
            let q = Sphere::new(p, r);
            index.query_sphere(&q, |uid| {
                for &(c, rad) in &instances[uid as usize].spheres {
                    best = best.min((p - c).norm() - rad);
                }
            });
            if best <= 0.0 || (best.is_finite() && best <= r - max_enclosing) {
                break;
            }
            r *= 2.0;
            if r > diag {
                break;
            }
        }
        if best <= 0.0 {
            occupied += 1;
        } else if best.is_finite() {
            clearances.push(best);
        }
    }

    Some(FreeSpace {
        samples: cfg.free_space_samples,
        occupied_fraction: occupied as f32 / cfg.free_space_samples as f32,
        clearance: Stats::of(&mut clearances),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sphere(x: f32, y: f32, z: f32, r: f32) -> InstanceGeom {
        let c = Point3::new(x, y, z);
        InstanceGeom {
            ingredient_id: 0,
            center: c,
            enclosing_radius: r,
            spheres: vec![(c, r)],
        }
    }

    fn cube_domain(half: f32) -> Aabb {
        Aabb::new(
            Point3::new(-half, -half, -half),
            Point3::new(half, half, half),
        )
    }

    #[test]
    fn no_overlaps_on_separated_spheres() {
        // A 3×3×3 lattice, spacing 10, radius 2 — well separated.
        let mut insts = Vec::new();
        for i in 0..3 {
            for j in 0..3 {
                for k in 0..3 {
                    insts.push(sphere(i as f32 * 10.0, j as f32 * 10.0, k as f32 * 10.0, 2.0));
                }
            }
        }
        let m = geometry_metrics(&insts, cube_domain(50.0), &MetricsConfig::default());
        assert_eq!(m.overlaps.pair_count, 0);
        assert_eq!(m.overlaps.instance_count, 0);
        assert_eq!(m.overlaps.max_depth, 0.0);
    }

    #[test]
    fn detects_a_known_overlap_with_correct_depth() {
        // Two radius-5 spheres 8 apart ⇒ overlap depth = 5 + 5 − 8 = 2.
        let insts = vec![sphere(0.0, 0.0, 0.0, 5.0), sphere(8.0, 0.0, 0.0, 5.0)];
        let m = geometry_metrics(&insts, cube_domain(50.0), &MetricsConfig::default());
        assert_eq!(m.overlaps.pair_count, 1);
        assert_eq!(m.overlaps.instance_count, 2);
        assert!((m.overlaps.max_depth - 2.0).abs() < 1e-3, "depth {}", m.overlaps.max_depth);
        assert!((m.overlaps.mean_depth - 2.0).abs() < 1e-3);
    }

    #[test]
    fn overlap_count_matches_brute_force_oracle() {
        // Random small spheres; cross-check the QBVH overlap pass against
        // an O(n²) brute-force enclosing-sphere overlap count.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut insts = Vec::new();
        for _ in 0..200 {
            insts.push(sphere(
                rng.gen_range(-20.0..20.0),
                rng.gen_range(-20.0..20.0),
                rng.gen_range(-20.0..20.0),
                rng.gen_range(1.0..4.0),
            ));
        }
        let mut brute = 0usize;
        for i in 0..insts.len() {
            for j in (i + 1)..insts.len() {
                let d = (insts[i].center - insts[j].center).norm();
                if insts[i].enclosing_radius + insts[j].enclosing_radius - d > OVERLAP_EPS {
                    brute += 1;
                }
            }
        }
        let m = geometry_metrics(&insts, cube_domain(40.0), &MetricsConfig::default());
        assert_eq!(m.overlaps.pair_count, brute, "qbvh overlaps != brute-force");
    }

    #[test]
    fn nearest_neighbor_recovers_lattice_spacing() {
        // 4×4×4 lattice, spacing 7 ⇒ every nearest-neighbour distance = 7.
        let mut insts = Vec::new();
        for i in 0..4 {
            for j in 0..4 {
                for k in 0..4 {
                    insts.push(sphere(i as f32 * 7.0, j as f32 * 7.0, k as f32 * 7.0, 1.0));
                }
            }
        }
        let m = geometry_metrics(&insts, cube_domain(40.0), &MetricsConfig::default());
        let nn = m.nearest_neighbor.center;
        assert_eq!(nn.n, insts.len());
        assert!((nn.min - 7.0).abs() < 1e-3, "min {}", nn.min);
        assert!((nn.median - 7.0).abs() < 1e-3, "median {}", nn.median);
        // Surface gap = 7 − 1 − 1 = 5.
        assert!((m.nearest_neighbor.surface_gap.median - 5.0).abs() < 1e-3);
    }

    #[test]
    fn rdf_is_empty_near_zero_and_conserves_pair_count() {
        // Well-separated spheres ⇒ no pairs closer than the spacing, so
        // the first bins of g(r) are zero; total counts == 2× #pairs
        // within r_max (ordered).
        let mut insts = Vec::new();
        for i in 0..5 {
            for j in 0..5 {
                insts.push(sphere(i as f32 * 6.0, j as f32 * 6.0, 0.0, 1.0));
            }
        }
        let cfg = MetricsConfig {
            rdf_bins: 30,
            rdf_r_max: Some(15.0),
            free_space_samples: 0,
            ..MetricsConfig::default()
        };
        let m = geometry_metrics(&insts, cube_domain(40.0), &cfg);
        let rdf = m.rdf.expect("rdf");
        // Contact distance is 2 (two radius-1 spheres); nearest centres
        // are 6 apart, so bins below ~5 must be empty.
        let below = (5.0 / rdf.bin_width) as usize;
        for k in 0..below {
            assert_eq!(rdf.counts[k], 0, "bin {k} should be empty");
        }
        // Ordered-pair conservation vs. brute force.
        let mut brute_ordered = 0u64;
        for a in 0..insts.len() {
            for b in 0..insts.len() {
                if a != b {
                    let d = (insts[a].center - insts[b].center).norm();
                    if d < rdf.r_max {
                        brute_ordered += 1;
                    }
                }
            }
        }
        assert_eq!(rdf.counts.iter().sum::<u64>(), brute_ordered);
    }

    #[test]
    fn free_space_occupied_fraction_tracks_volume() {
        // One big sphere (r=20) centred in a 100³ box. Occupied fraction
        // ≈ sphere volume / box volume.
        let insts = vec![sphere(0.0, 0.0, 0.0, 20.0)];
        let cfg = MetricsConfig {
            free_space_samples: 40_000,
            rdf_bins: 0,
            ..MetricsConfig::default()
        };
        let domain = cube_domain(50.0); // 100³
        let m = geometry_metrics(&insts, domain, &cfg);
        let fs = m.free_space.expect("free space");
        let expected = (4.0 / 3.0 * std::f32::consts::PI * 20.0_f32.powi(3)) / domain.volume();
        assert!(
            (fs.occupied_fraction - expected).abs() < 0.01,
            "occupied {} vs expected {expected}",
            fs.occupied_fraction
        );
        // Free samples report a positive largest-empty-ball radius.
        assert!(fs.clearance.n > 0);
        assert!(fs.clearance.max > 0.0);
    }
}
