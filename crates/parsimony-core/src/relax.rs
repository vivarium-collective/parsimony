//! Relaxation pass over a packed assembly.
//!
//! The greedy placer is collision-free *within* a stage, but the staged
//! pipeline packs each stage against approximate obstacles, so the merged
//! assembly could in principle carry residual clashes at stage boundaries.
//! This is cellPACK's post-pack "settle": measure the clashes and nudge the
//! movable instances apart (rigid-body translation), reporting the clash
//! count + deepest penetration before and after.
//!
//! It works at **proxy-sphere accuracy** — the only correct level for
//! sphere-tree and mesh ingredients, whose *enclosing* spheres overlap
//! freely while the molecules themselves don't. As a consequence a greedy
//! pack (which the placer keeps proxy-clean by construction) measures zero
//! clashes and the pass returns immediately without moving anything; the
//! settle iterations only run when real overlaps exist.
//!
//! - **Movable**: every interior instance (any shape), as a rigid body of
//!   its proxy spheres.
//! - **Fixed obstacles**: the chromosome strand and the DNA-binding proteins
//!   (placed on it) — entered as proxy spheres so movables avoid them.
//! - **Ignored**: the tiled membrane (lipids stay on the surface; the
//!   interior is kept off the boundary anyway).

use std::collections::HashSet;

use nalgebra::{Point3, Vector3};

use parsimony_spatial::{Aabb, QbvhIndex, Sphere, SpatialIndex};

use crate::compartment::CompartmentKind;
use crate::placement::Snapshot;
use crate::recipe::{PackingMode, Recipe};

#[derive(Debug, Clone)]
pub struct RelaxStats {
    pub iterations: usize,
    pub movable: usize,
    pub proxy_spheres: usize,
    pub clashes_before: usize,
    pub clashes_after: usize,
    pub max_penetration_before: f32,
    pub max_penetration_after: f32,
}

/// Fraction of each overlap corrected per iteration (soft, to stay stable).
const DAMPING: f32 = 0.5;
/// Sentinel instance id marking a fixed (immovable) proxy.
const FIXED: u32 = u32::MAX;
/// Max proxy spheres kept per instance for the broad-phase. Mesh ingredients
/// carry hundreds; a representative subset keeps a whole-cell pass tractable
/// (a clean pack's full proxies don't overlap, so neither does a subset).
const PROXY_CAP: usize = 16;
/// Per-instance step (Å) below which an instance counts as settled: it isn't
/// moved and doesn't keep the loop alive. Kept an order of magnitude under
/// the clash threshold in [`measure`] (1e-3 Å) so that converging here implies
/// no remaining clashes — the early exit can't leave overlaps behind.
const MOVE_EPS: f32 = 1e-4;
/// Recompact the incrementally-updated index every this-many iterations.
/// Updates keep the tree *correct* but can unbalance it; a periodic
/// `rebuild_if_needed` restores query speed without an O(n) check every step.
const REBUILD_EVERY: usize = 16;

pub fn relax(snapshot: &mut Snapshot, recipe: &Recipe, iterations: usize) -> RelaxStats {
    let bound: HashSet<&str> = recipe
        .chromosome
        .as_ref()
        .map(|c| c.proteins.iter().map(|(n, _)| n.as_str()).collect())
        .unwrap_or_default();

    // Proxy spheres for the whole assembly, each tagged with the movable
    // instance it belongs to (or FIXED). Movable instances also record their
    // centre + compartment + proxy range so we can translate them rigidly.
    let mut ppos: Vec<Point3<f32>> = Vec::new();
    let mut prad: Vec<f32> = Vec::new();
    let mut pinst: Vec<u32> = Vec::new();
    let mut mov_center: Vec<Point3<f32>> = Vec::new();
    let mut mov_comp: Vec<Option<(Point3<f32>, f32)>> = Vec::new();
    let mut mov_pidx: Vec<usize> = Vec::new();
    let mut mov_range: Vec<(usize, usize)> = Vec::new();
    let mut mov_rmax: Vec<f32> = Vec::new();

    for (i, pl) in snapshot.placements.iter().enumerate() {
        let Some((_, ing)) = recipe.ingredients.get_index(pl.ingredient_id as usize) else {
            continue;
        };
        if matches!(ing.packing_mode, PackingMode::Tiled) {
            continue; // membrane stays on the surface
        }
        let mut spheres: Vec<(Point3<f32>, f32)> =
            ing.shape.world_spheres(pl.position, pl.rotation).collect();
        if spheres.len() > PROXY_CAP {
            let stride = (spheres.len() / PROXY_CAP).max(1);
            spheres = spheres.into_iter().step_by(stride).take(PROXY_CAP).collect();
        }
        if bound.contains(ing.name.as_str()) {
            for (c, r) in spheres {
                ppos.push(c);
                prad.push(r);
                pinst.push(FIXED);
            }
            continue;
        }
        let id = mov_center.len() as u32;
        let lo = ppos.len();
        let mut rmax = 0.0f32;
        for (c, r) in spheres {
            ppos.push(c);
            prad.push(r);
            pinst.push(id);
            rmax = rmax.max(r);
        }
        let hi = ppos.len();
        if hi == lo {
            continue; // shape produced no spheres (shouldn't happen)
        }
        mov_center.push(pl.position);
        mov_comp.push(compartment_sphere(recipe, pl.compartment_id));
        mov_pidx.push(i);
        mov_range.push((lo, hi));
        mov_rmax.push(rmax);
    }
    if let Some(c) = &snapshot.chromosome {
        for p in &c.points {
            ppos.push(c.center + p.coords);
            prad.push(c.radius);
            pinst.push(FIXED);
        }
    }

    let m = mov_center.len();
    let bbox = recipe.bounding_box;

    let (clashes_before, max_before) = measure(&ppos, &prad, &pinst);

    let mut iters_run = 0;
    if clashes_before > 0 {
        // Build the broad-phase ONCE and keep it current with incremental
        // `update`s as instances move, rather than rebuilding it every
        // iteration. A clean pack has zero clashes and never reaches here; when
        // overlaps do exist this turns the settle loop from O(iterations ×
        // full-rebuild) into O(build + moved-proxies × log n). `maxr` is the
        // largest proxy radius and never changes (proxies only translate).
        let (mut idx, maxr) = build(&ppos, &prad);
        for _ in 0..iterations {
            iters_run += 1;
            let mut disp = vec![Vector3::zeros(); m];
            for p in 0..ppos.len() {
                let inst = pinst[p];
                if inst == FIXED {
                    continue;
                }
                let (pc, pr) = (ppos[p], prad[p]);
                idx.query_sphere(&Sphere::new(pc, pr + maxr), |uid| {
                    let q = uid as usize;
                    if pinst[q] == inst {
                        return; // same instance — rigid, ignore
                    }
                    let delta = pc - ppos[q];
                    let dist = delta.norm();
                    let target = pr + prad[q];
                    if dist < target && dist > 1e-4 {
                        let share = if pinst[q] == FIXED { 1.0 } else { 0.5 };
                        disp[inst as usize] += delta / dist * (target - dist) * share;
                    }
                });
            }
            // Apply each instance's net displacement (damped + capped to a
            // proxy radius/iter so a many-proxy body can't lurch), translate
            // its proxies, mirror the move into the index, and clamp the centre
            // into its compartment. Instances that barely move are left in
            // place (and out of the index churn).
            let mut max_move = 0.0f32;
            for i in 0..m {
                let mut d = disp[i] * DAMPING;
                let cap = mov_rmax[i].max(1.0);
                if d.norm() > cap {
                    d = d.normalize() * cap;
                }
                let new_center = clamp_into(mov_center[i] + d, mov_rmax[i], mov_comp[i], &bbox);
                let actual = new_center - mov_center[i];
                let amove = actual.norm();
                if amove <= MOVE_EPS {
                    continue; // settled — don't touch its proxies or the index
                }
                max_move = max_move.max(amove);
                mov_center[i] = new_center;
                let (lo, hi) = mov_range[i];
                for p in lo..hi {
                    ppos[p] += actual;
                    idx.update(p as u64, Aabb::from_sphere(ppos[p], prad[p])).ok();
                }
            }
            // Converged: nothing moved more than MOVE_EPS this pass, so any
            // residual overlap is below the clash threshold. Stop early instead
            // of burning the rest of the iteration budget.
            if max_move <= MOVE_EPS {
                break;
            }
            if iters_run % REBUILD_EVERY == 0 {
                idx.rebuild_if_needed();
            }
        }
    }

    let (clashes_after, max_after) = if iters_run > 0 {
        measure(&ppos, &prad, &pinst)
    } else {
        (clashes_before, max_before)
    };

    for i in 0..m {
        snapshot.placements[mov_pidx[i]].position = mov_center[i];
    }

    RelaxStats {
        iterations: iters_run,
        movable: m,
        proxy_spheres: ppos.len(),
        clashes_before,
        clashes_after,
        max_penetration_before: max_before,
        max_penetration_after: max_after,
    }
}

/// Count overlapping proxy pairs from *different* instances (each counted
/// once), and the deepest penetration. Pairs of two fixed proxies are
/// ignored (the chromosome's own beads touch by design).
fn measure(ppos: &[Point3<f32>], prad: &[f32], pinst: &[u32]) -> (usize, f32) {
    let (idx, maxr) = build(ppos, prad);
    let mut clashes = 0usize;
    let mut max_pen = 0.0f32;
    for p in 0..ppos.len() {
        let (pc, pr, pi) = (ppos[p], prad[p], pinst[p]);
        idx.query_sphere(&Sphere::new(pc, pr + maxr), |uid| {
            let q = uid as usize;
            if q <= p {
                return; // count each pair once
            }
            let qi = pinst[q];
            if pi == qi {
                return; // same instance, or both fixed-but-equal (FIXED==FIXED)
            }
            if pi == FIXED && qi == FIXED {
                return;
            }
            let dist = (pc - ppos[q]).norm();
            let target = pr + prad[q];
            if dist < target - 1e-3 {
                clashes += 1;
                let pen = target - dist;
                if pen > max_pen {
                    max_pen = pen;
                }
            }
        });
    }
    (clashes, max_pen)
}

/// Bulk-build a proxy index keyed by proxy slot (`build_from` is tighter and
/// faster than a stream of inserts), returning it alongside the largest proxy
/// radius for broad-phase query inflation.
fn build(ppos: &[Point3<f32>], prad: &[f32]) -> (QbvhIndex, f32) {
    let mut idx = QbvhIndex::new();
    idx.build_from((0..ppos.len()).map(|i| (i as u64, Aabb::from_sphere(ppos[i], prad[i]))))
        .ok();
    let maxr = prad.iter().copied().fold(0.0f32, f32::max);
    (idx, maxr)
}

fn compartment_sphere(recipe: &Recipe, cid: u32) -> Option<(Point3<f32>, f32)> {
    match recipe.compartments.get_index(cid as usize) {
        Some((_, c)) => match &c.kind {
            CompartmentKind::Sphere { center, radius } => Some((*center, *radius)),
            _ => None,
        },
        None => None,
    }
}

/// Keep an instance of (enclosing) radius `r` inside its spherical
/// compartment, else the bounding box.
fn clamp_into(
    p: Point3<f32>,
    r: f32,
    comp: Option<(Point3<f32>, f32)>,
    bbox: &Aabb,
) -> Point3<f32> {
    if let Some((center, radius)) = comp {
        let v = p - center;
        let max = (radius - r).max(0.0);
        if v.norm() > max {
            return center + v.normalize() * max;
        }
        return p;
    }
    Point3::new(
        p.x.clamp(bbox.min.x + r, bbox.max.x - r),
        p.y.clamp(bbox.min.y + r, bbox.max.y - r),
        p.z.clamp(bbox.min.z + r, bbox.max.z - r),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::{GreedyRandomPlacer, PlacerConfig};
    use crate::placement::Placement;
    use nalgebra::UnitQuaternion;

    const RECIPE: &str = r#"{
        "bounding_box": [[-200,-200,-200],[200,200,200]],
        "objects": { "s": { "type": "single_sphere", "radius": 20 } },
        "composition": {
            "space": { "regions": { "interior": [ { "object": "s", "count": 1 } ] } }
        }
    }"#;

    #[test]
    fn separates_overlapping_spheres() {
        let recipe = Recipe::from_json_str(RECIPE).unwrap();
        let mut snap = Snapshot::new("t".into(), 0);
        for (uid, x) in [(0u64, -5.0f32), (1, 5.0)] {
            snap.placements.push(Placement {
                instance_uid: uid,
                ingredient_id: 0,
                variant_id: 0,
                compartment_id: 0,
                position: Point3::new(x, 0.0, 0.0),
                rotation: UnitQuaternion::identity(),
            });
        }
        let stats = relax(&mut snap, &recipe, 100);
        assert_eq!(stats.movable, 2);
        assert!(stats.clashes_before >= 1);
        assert_eq!(stats.clashes_after, 0, "overlap should resolve");
        let d = (snap.placements[0].position - snap.placements[1].position).norm();
        assert!(d + 1e-2 >= 40.0, "still overlapping (d={d})");
    }

    #[test]
    fn keeps_a_clean_pack_clean() {
        let recipe = Recipe::from_json_str(
            r#"{
            "bounding_box": [[0,0,0],[300,300,300]],
            "objects": { "s": { "type": "single_sphere", "radius": 15 } },
            "composition": { "space": { "regions": { "interior": [ { "object": "s", "count": 60 } ] } } }
        }"#,
        )
        .unwrap();
        let out = GreedyRandomPlacer::new(&recipe, PlacerConfig::default()).pack(1);
        let mut snap = out.snapshot;
        let before: Vec<_> = snap.placements.iter().map(|p| p.position).collect();
        let stats = relax(&mut snap, &recipe, 10);
        assert_eq!(stats.clashes_before, 0, "greedy pack must be proxy-clean");
        assert_eq!(stats.iterations, 0, "no work when already clean");
        for (p, b) in snap.placements.iter().zip(before.iter()) {
            assert!((p.position - b).norm() < 1e-6, "clean pack must not move");
        }
    }
}
