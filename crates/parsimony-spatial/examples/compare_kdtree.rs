//! `compare_kdtree` — reproduce cellPACK's broad-phase query pattern
//! (rebuild a fresh BruteIndex per query) and run the same workload
//! against our incrementally-maintained `QbvhIndex`. Reports the
//! wall-clock speedup ratio.
//!
//! ## What cellPACK does
//!
//! In `cellpack/autopack/ingredient/Ingredient.py:1249` the collision
//! test rebuilds `scipy.spatial.cKDTree` over every packed center
//! before every query. For a packing of `n` ingredients with one
//! collision check per placement, total cost is O(n² log n).
//!
//! ## What QbvhIndex does
//!
//! Maintain one index, insert after each placement, query before.
//! O(n log n) overall.
//!
//! ## Caveat
//!
//! We use `BruteIndex` as the "rebuild each query" stand-in rather
//! than a real kdtree. `BruteIndex::insert` is O(1) so the rebuild
//! cost here is O(n) per query rather than cKDTree's O(n log n) —
//! making this an *under*-estimate of the actual cellPACK pattern's
//! cost. The QBVH speedup against real cKDTree-rebuild is larger.
//!
//! Run with: `cargo run --release --example compare_kdtree -p parsimony-spatial`

use std::time::Instant;

use nalgebra::Point3;
use parsimony_spatial::{Aabb, BruteIndex, QbvhIndex, Sphere, SpatialIndex};
use rand::{Rng, SeedableRng};
use rand_xoshiro::Xoshiro256PlusPlus;

const SIZES: &[usize] = &[100, 1_000, 10_000, 100_000];
const WORLD: f32 = 1_000.0;
const INSTANCE_RADIUS: f32 = 2.0;
const QUERY_RADIUS: f32 = 5.0;

fn gen_positions(n: usize, seed: u64) -> Vec<Point3<f32>> {
    let mut rng = Xoshiro256PlusPlus::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            Point3::new(
                rng.gen_range(-WORLD..WORLD),
                rng.gen_range(-WORLD..WORLD),
                rng.gen_range(-WORLD..WORLD),
            )
        })
        .collect()
}

fn aabb_around(p: Point3<f32>, r: f32) -> Aabb {
    Aabb::from_sphere(p, r)
}

/// Simulate cellPACK: for each placement, rebuild a fresh index
/// containing all prior placements, then query for collisions.
fn cellpack_pattern(positions: &[Point3<f32>]) -> usize {
    let mut total_hits = 0usize;
    for i in 0..positions.len() {
        // Rebuild from scratch.
        let mut idx = BruteIndex::with_capacity(i);
        for (j, prev) in positions.iter().enumerate().take(i) {
            idx.insert(j as u64, aabb_around(*prev, INSTANCE_RADIUS))
                .unwrap();
        }
        // Query "what's within QUERY_RADIUS of this new placement?"
        let q = Sphere::new(positions[i], QUERY_RADIUS);
        idx.query_sphere(&q, |_| total_hits += 1);
    }
    total_hits
}

/// QBVH pattern: one incremental index. Insert after each query.
fn qbvh_pattern(positions: &[Point3<f32>]) -> usize {
    let mut total_hits = 0usize;
    let mut idx = QbvhIndex::new();
    for (i, p) in positions.iter().enumerate() {
        let q = Sphere::new(*p, QUERY_RADIUS);
        idx.query_sphere(&q, |_| total_hits += 1);
        idx.insert(i as u64, aabb_around(*p, INSTANCE_RADIUS))
            .unwrap();
    }
    total_hits
}

fn main() {
    println!(
        "Workload: simulated packing — n placements, each preceded by a sphere\nquery for collisions against all prior placements."
    );
    println!(
        "Box: ±{:.0} per axis; instance r={INSTANCE_RADIUS}, query r={QUERY_RADIUS}.\n",
        WORLD
    );
    println!("{:>10} | {:>14} | {:>14} | {:>10} | {:>10}", "n", "cellPACK", "QBVH", "speedup", "hits eq?");
    println!("{:->10}-+-{:->14}-+-{:->14}-+-{:->10}-+-{:->10}", "", "", "", "", "");

    for &n in SIZES {
        let positions = gen_positions(n, 0xC0DE);

        // Skip the cellPACK pattern at the largest size — it's O(n²),
        // would take many minutes at n = 100k.
        let cellpack = if n <= 10_000 {
            let t = Instant::now();
            let hits = cellpack_pattern(&positions);
            (Some(t.elapsed()), Some(hits))
        } else {
            (None, None)
        };

        let t = Instant::now();
        let qbvh_hits = qbvh_pattern(&positions);
        let qbvh_time = t.elapsed();

        let cellpack_str = match cellpack.0 {
            Some(d) => format!("{:>13.3?}", d),
            None => "      skipped".to_string(),
        };
        let qbvh_str = format!("{:>13.3?}", qbvh_time);
        let speedup_str = match cellpack.0 {
            Some(d) => format!("{:>9.1}×", d.as_secs_f64() / qbvh_time.as_secs_f64()),
            None => "       —".to_string(),
        };
        let hits_match = match cellpack.1 {
            Some(h) => {
                if h == qbvh_hits {
                    "      yes"
                } else {
                    "    NO ⚠"
                }
            }
            None => "        —",
        };
        println!(
            "{:>10} | {} | {} | {} | {}",
            n, cellpack_str, qbvh_str, speedup_str, hits_match
        );
    }
}
