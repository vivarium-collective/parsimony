//! Packing ingredients onto a fiber (the chromosome) as a 1-D substrate.
//!
//! DNA-binding proteins — RNA polymerase, the replisome, transcription
//! factors — don't fill a volume; they bind *along* the genome. This packs
//! them onto the chromosome fiber: sample a position by arc length, seat the
//! protein on the DNA surface (radially offset so it touches but doesn't
//! interpenetrate the strand) at a random azimuth, orient its principal
//! axis along the local tangent (the protein rides the DNA), and reject
//! candidates that collide with already-placed proteins or upstream
//! obstacles. The fiber's own beads need no obstacle check — the radial
//! offset guarantees the protein sits outside the strand by construction.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use rand::Rng;

use parsimony_spatial::{Aabb, QbvhIndex, Sphere, SpatialIndex};

use crate::compartment::align_to_normal;
use crate::fiber::CellShape;
use crate::ingredient::{Ingredient, IngredientId};

/// One protein placed on the fiber.
#[derive(Debug, Clone)]
pub struct FiberBinding {
    pub ingredient_id: IngredientId,
    pub position: Point3<f32>,
    pub rotation: UnitQuaternion<f32>,
}

/// Number of (arc-length, azimuth) candidates tried per instance before
/// giving up on it.
const ATTEMPTS_PER_INSTANCE: usize = 24;

/// Small radial gap so a binding rests just *outside* its local DNA bead
/// (clearing it strictly) rather than exactly touching it.
const BIND_GAP: f32 = 1.0;

/// Place proteins along `fiber` (world-space bead centres). Each entry of
/// `proteins` is `(ingredient_id, ingredient, count)`; `obstacles` are
/// world-space spheres to avoid (e.g. the packed interior). `fiber_radius`
/// is the strand's bead radius — proteins are seated `fiber_radius +
/// protein_radius` off the axis so they rest on the DNA surface.
/// Deterministic for a given RNG.
pub fn pack_on_fiber<R: Rng>(
    fiber: &[Point3<f32>],
    proteins: &[(IngredientId, &Ingredient, u32)],
    obstacles: &[(Point3<f32>, f32)],
    fiber_radius: f32,
    shape: CellShape,
    rng: &mut R,
) -> Vec<FiberBinding> {
    if fiber.len() < 2 {
        return Vec::new();
    }

    // Cumulative arc length along the strand.
    let mut cum = vec![0.0_f32; fiber.len()];
    for i in 1..fiber.len() {
        cum[i] = cum[i - 1] + (fiber[i] - fiber[i - 1]).norm();
    }
    let total = cum[fiber.len() - 1];
    if total <= 1e-3 {
        return Vec::new();
    }

    // Position + unit tangent at arc length `s`.
    let sample = |s: f32| -> (Point3<f32>, Vector3<f32>) {
        let s = s.clamp(0.0, total);
        let mut k = 0;
        while k + 1 < fiber.len() && cum[k + 1] <= s {
            k += 1;
        }
        let k1 = (k + 1).min(fiber.len() - 1);
        let seg = (cum[k1] - cum[k]).max(1e-6);
        let t = ((s - cum[k]) / seg).clamp(0.0, 1.0);
        let pos = fiber[k] + (fiber[k1] - fiber[k]) * t;
        let tang = (fiber[k1] - fiber[k])
            .try_normalize(1e-6)
            .unwrap_or_else(|| Vector3::z());
        (pos, tang)
    };

    // Collision index over obstacle + DNA-bead + placed-protein spheres.
    let mut spheres: Vec<(Point3<f32>, f32)> = obstacles.to_vec();
    // The strand's own beads are obstacles too, so a binding can rest just
    // outside its local DNA but won't overlap a *different* coil winding a
    // random azimuth might aim it into.
    spheres.extend(fiber.iter().map(|p| (*p, fiber_radius)));
    let mut index = QbvhIndex::new();
    let mut max_r = 0.0_f32;
    for (i, (c, r)) in spheres.iter().enumerate() {
        index.insert(i as u64, Aabb::from_sphere(*c, *r)).expect("uid");
        max_r = max_r.max(*r);
    }

    // Largest proteins first: they need open surface, and smaller ones then
    // fill the gaps left around them (otherwise abundant small proteins
    // saturate the strand and starve the big ones).
    let mut order: Vec<&(IngredientId, &Ingredient, u32)> = proteins.iter().collect();
    order.sort_by(|a, b| {
        b.1.shape
            .enclosing_radius()
            .partial_cmp(&a.1.shape.enclosing_radius())
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.0.cmp(&b.0))
    });

    let mut out = Vec::new();
    for &&(id, ing, count) in &order {
        let er = ing.shape.enclosing_radius();
        let off = fiber_radius + er + BIND_GAP;
        let inset = shape.inset(er);
        for inst in 0..count {
            // Stratify the first attempt along the strand so instances
            // spread out; later attempts sample uniformly at random.
            let base_s = (inst as f32 + rng.gen_range(0.0..1.0)) / count.max(1) as f32 * total;
            let mut placed = false;
            for attempt in 0..ATTEMPTS_PER_INSTANCE {
                let s = if attempt == 0 { base_s } else { rng.gen_range(0.0..total) };
                let (p, tang) = sample(s);
                let n1 = perp(tang);
                let n2 = tang.cross(&n1);
                let phi = rng.gen_range(0.0..std::f32::consts::TAU);
                let radial = n1 * phi.cos() + n2 * phi.sin();
                let center = confine_center(p + radial * off, n1, n2, phi, off, p, &inset);
                let rot = align_to_normal(ing.principal_vector, tang);
                let cand: Vec<(Point3<f32>, f32)> =
                    ing.shape.world_spheres(center, rot).collect();
                if collides(&index, &spheres, max_r, &cand) {
                    continue;
                }
                for s in &cand {
                    let uid = spheres.len() as u64;
                    index.insert(uid, Aabb::from_sphere(s.0, s.1)).expect("uid");
                    spheres.push(*s);
                    max_r = max_r.max(s.1);
                }
                out.push(FiberBinding {
                    ingredient_id: id,
                    position: center,
                    rotation: rot,
                });
                placed = true;
                break;
            }
            let _ = placed; // unplaced instances are simply skipped
        }
    }
    out
}

/// Like [`pack_on_fiber`], but each instance targets a specific arc *fraction*
/// of the strand (its gene's genomic position) rather than a random spot.
/// `placements` is one entry per instance: `(ingredient_id, ingredient,
/// fraction in [0,1))`. Each is seated at `fraction × contour`, searching
/// azimuths and a small (growing) arc jitter for a clear spot so it stays near
/// its locus. Used to put RNAP/DNAP at real transcription / replication sites.
pub fn pack_on_fiber_at<R: Rng>(
    fiber: &[Point3<f32>],
    placements: &[(IngredientId, &Ingredient, f32)],
    obstacles: &[(Point3<f32>, f32)],
    fiber_radius: f32,
    shape: CellShape,
    rng: &mut R,
) -> Vec<FiberBinding> {
    if fiber.len() < 2 {
        return Vec::new();
    }
    let mut cum = vec![0.0_f32; fiber.len()];
    for i in 1..fiber.len() {
        cum[i] = cum[i - 1] + (fiber[i] - fiber[i - 1]).norm();
    }
    let total = cum[fiber.len() - 1];
    if total <= 1e-3 {
        return Vec::new();
    }
    let sample = |s: f32| -> (Point3<f32>, Vector3<f32>) {
        let s = s.clamp(0.0, total);
        let mut k = 0;
        while k + 1 < fiber.len() && cum[k + 1] <= s {
            k += 1;
        }
        let k1 = (k + 1).min(fiber.len() - 1);
        let seg = (cum[k1] - cum[k]).max(1e-6);
        let t = ((s - cum[k]) / seg).clamp(0.0, 1.0);
        let pos = fiber[k] + (fiber[k1] - fiber[k]) * t;
        let tang = (fiber[k1] - fiber[k])
            .try_normalize(1e-6)
            .unwrap_or_else(|| Vector3::z());
        (pos, tang)
    };

    let mut spheres: Vec<(Point3<f32>, f32)> = obstacles.to_vec();
    spheres.extend(fiber.iter().map(|p| (*p, fiber_radius)));
    let mut index = QbvhIndex::new();
    let mut max_r = 0.0_f32;
    for (i, (c, r)) in spheres.iter().enumerate() {
        index.insert(i as u64, Aabb::from_sphere(*c, *r)).expect("uid");
        max_r = max_r.max(*r);
    }

    // Largest first, as in pack_on_fiber, so big proteins claim surface before
    // small abundant ones crowd the strand.
    let mut order: Vec<usize> = (0..placements.len()).collect();
    order.sort_by(|&a, &b| {
        placements[b]
            .1
            .shape
            .enclosing_radius()
            .partial_cmp(&placements[a].1.shape.enclosing_radius())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut out = Vec::new();
    for &pi in &order {
        let (id, ing, frac) = placements[pi];
        let er = ing.shape.enclosing_radius();
        let off = fiber_radius + er + BIND_GAP;
        let inset = shape.inset(er);
        let base_s = frac.rem_euclid(1.0) * total;
        for attempt in 0..ATTEMPTS_PER_INSTANCE {
            // Attempt 0 hits the gene exactly; later attempts wander a little
            // (up to ~4% of the contour) to find an open azimuth/spot nearby.
            let jitter = if attempt == 0 {
                0.0
            } else {
                rng.gen_range(-1.0_f32..1.0)
                    * (attempt as f32 / ATTEMPTS_PER_INSTANCE as f32)
                    * 0.04
                    * total
            };
            let (p, tang) = sample(base_s + jitter);
            let n1 = perp(tang);
            let n2 = tang.cross(&n1);
            let phi = rng.gen_range(0.0..std::f32::consts::TAU);
            let radial = n1 * phi.cos() + n2 * phi.sin();
            let center = confine_center(p + radial * off, n1, n2, phi, off, p, &inset);
            let rot = align_to_normal(ing.principal_vector, tang);
            let cand: Vec<(Point3<f32>, f32)> = ing.shape.world_spheres(center, rot).collect();
            if collides(&index, &spheres, max_r, &cand) {
                continue;
            }
            for s in &cand {
                let uid = spheres.len() as u64;
                index.insert(uid, Aabb::from_sphere(s.0, s.1)).expect("uid");
                spheres.push(*s);
                max_r = max_r.max(s.1);
            }
            out.push(FiberBinding {
                ingredient_id: id,
                position: center,
                rotation: rot,
            });
            break;
        }
    }
    out
}

/// Confine a candidate protein centre to the inset envelope.
///
/// When `raw` escapes the inset, tries 8 azimuthal rotations around the
/// local strand tangent (no RNG — deterministic from the existing `phi`).
/// Falls back to a linear pull toward the medial axis until contained.
/// The obstacle/overlap check still runs *after* confinement, so a
/// pulled-inward candidate that lands on the strand will be rejected by
/// the caller's collision check and trigger the next attempt.
fn confine_center(
    raw: Point3<f32>,
    n1: Vector3<f32>,
    n2: Vector3<f32>,
    phi: f32,
    off: f32,
    strand_pt: Point3<f32>,
    inset: &CellShape,
) -> Point3<f32> {
    if inset.contains(&raw) {
        return raw;
    }
    // Try 8 evenly-spaced azimuthal rotations (deterministic, no RNG).
    let step = std::f32::consts::TAU / 8.0;
    for k in 1..=8_u32 {
        let p = phi + k as f32 * step;
        let alt = strand_pt + (n1 * p.cos() + n2 * p.sin()) * off;
        if inset.contains(&alt) {
            return alt;
        }
    }
    // Fallback: pull linearly toward the medial axis until contained.
    let inward = inset.inward(&raw);
    let pull = inset.cap_radius() * 0.1_f32;
    let mut pos = raw;
    for _ in 0..20 {
        if inset.contains(&pos) {
            return pos;
        }
        pos += inward * pull;
    }
    pos // degenerate shape — return best effort
}

/// True if any candidate sphere overlaps a stored sphere.
fn collides(
    index: &QbvhIndex,
    spheres: &[(Point3<f32>, f32)],
    max_r: f32,
    cand: &[(Point3<f32>, f32)],
) -> bool {
    for &(c, r) in cand {
        let mut hit = false;
        index.query_sphere(&Sphere::new(c, r + max_r), |uid| {
            if hit {
                return;
            }
            let (oc, or_) = spheres[uid as usize];
            let rs = r + or_;
            if (oc - c).norm_squared() < rs * rs {
                hit = true;
            }
        });
        if hit {
            return true;
        }
    }
    false
}

/// Some unit vector perpendicular to `t`.
fn perp(t: Vector3<f32>) -> Vector3<f32> {
    let a = if t.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
    (a - t * t.dot(&a)).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ingredient::IngredientShape;
    use crate::recipe::PackingMode;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn sphere_ingredient(radius: f32) -> Ingredient {
        Ingredient {
            name: "p".into(),
            shape: IngredientShape::SingleSphere { radius },
            color: [0.5, 0.5, 0.5],
            jitter_attempts: 20,
            packing_mode: PackingMode::Random,
            principal_vector: Vector3::z(),
            mesh_lods: vec![],
            segment: None,
        }
    }

    #[test]
    fn binds_proteins_on_the_fiber_surface_without_overlap() {
        use crate::fiber::CellShape;
        // A straight fiber along +x — use a large sphere so confinement never triggers.
        let fiber: Vec<Point3<f32>> =
            (0..50).map(|i| Point3::new(i as f32 * 10.0, 0.0, 0.0)).collect();
        let ing = sphere_ingredient(8.0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        let fiber_radius = 5.0;
        let shape = CellShape::Sphere { radius: 1000.0 };
        let binds = pack_on_fiber(&fiber, &[(0, &ing, 30)], &[], fiber_radius, shape, &mut rng);

        assert!(binds.len() > 15, "placed only {}", binds.len());
        // Each protein sits one (fiber_radius + protein_radius) off the
        // x-axis — i.e. on the DNA surface.
        for b in &binds {
            let radial = (b.position.y.powi(2) + b.position.z.powi(2)).sqrt();
            assert!(
                (radial - (fiber_radius + 8.0 + BIND_GAP)).abs() < 1e-1,
                "off the fiber surface: radial {radial}"
            );
            assert!(b.position.x >= -1e-1 && b.position.x <= 490.0 + 1e-1);
        }
        // No two bound proteins interpenetrate.
        for i in 0..binds.len() {
            for j in (i + 1)..binds.len() {
                let d = (binds[i].position - binds[j].position).norm();
                assert!(d + 1e-2 >= 16.0, "proteins {i},{j} overlap (d={d})");
            }
        }
    }

    #[test]
    fn bound_proteins_stay_inside_the_cell_envelope() {
        use crate::fiber::CellShape;
        let shape = CellShape::Capsule { half_len: 400.0, radius: 120.0, axis: Vector3::x() };
        // A fiber hugging the wall (y ~ +radius), where a naive outward offset escapes.
        let fiber: Vec<Point3<f32>> = (0..40)
            .map(|i| Point3::new(-300.0 + i as f32 * 15.0, 115.0, 0.0))
            .collect();
        let ing = sphere_ingredient(25.0); // proxy radius 25
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let binds = pack_on_fiber(&fiber, &[(0, &ing, 30)], &[], 12.0, shape, &mut rng);
        assert!(!binds.is_empty());
        let inset = shape.inset(25.0);
        for b in &binds {
            assert!(inset.contains(&b.position), "binding outside envelope: {:?}", b.position);
        }
    }

    #[test]
    fn avoids_obstacles_and_is_deterministic() {
        use crate::fiber::CellShape;
        let fiber: Vec<Point3<f32>> =
            (0..40).map(|i| Point3::new(i as f32 * 10.0, 0.0, 0.0)).collect();
        let ing = sphere_ingredient(8.0);
        // Large sphere — confinement never triggers, so determinism is preserved.
        let shape = CellShape::Sphere { radius: 1000.0 };

        // An obstacle wall blocking the +y side of the fiber's first half.
        let obstacles: Vec<(Point3<f32>, f32)> =
            (0..20).map(|i| (Point3::new(i as f32 * 10.0, 13.0, 0.0), 9.0)).collect();
        let mut a = Xoshiro256PlusPlus::seed_from_u64(7);
        let binds = pack_on_fiber(&fiber, &[(0, &ing, 30)], &obstacles, 5.0, shape, &mut a);
        for b in &binds {
            for o in &obstacles {
                let d = (b.position - o.0).norm();
                assert!(d + 1e-2 >= 8.0 + 9.0, "protein overlaps obstacle (d={d})");
            }
        }
        // Deterministic for a fixed seed.
        let mut b = Xoshiro256PlusPlus::seed_from_u64(7);
        let again = pack_on_fiber(&fiber, &[(0, &ing, 30)], &obstacles, 5.0, shape, &mut b);
        assert_eq!(binds.len(), again.len());
        for (x, y) in binds.iter().zip(again.iter()) {
            assert_eq!(x.position, y.position);
        }
    }
}
