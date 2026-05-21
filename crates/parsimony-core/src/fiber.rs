//! Chromosome / fiber generation.
//!
//! Lays a coarse-grained polymer (the genome) inside a compartment as a
//! constrained self-avoiding random walk: beads spaced ~`step` apart,
//! kept inside the cell and away from self-overlap, with worm-like-chain
//! persistence for stiffness. The result is a polyline of bead centres
//! for an [`IngredientShape::Fiber`](crate::ingredient::IngredientShape::Fiber).
//!
//! This mirrors the spirit of Goodsell/Maritan's lattice nucleoid (a
//! constrained polymer filling the cell). [`generate_supercoiled_fiber`]
//! adds plectonemic supercoiling — an interwound double helix wound along
//! a confined backbone axis; on-fiber DNA-binding proteins come next.

use nalgebra::{Point3, Rotation3, Unit, Vector3};
use rand::Rng;

/// Uniformly random unit vector (rejection-sampled in the unit ball).
fn random_unit<R: Rng>(rng: &mut R) -> Vector3<f32> {
    loop {
        let v: Vector3<f32> = Vector3::new(
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
            rng.gen_range(-1.0..1.0),
        );
        let n2 = v.norm_squared();
        if n2 > 1e-6 && n2 <= 1.0 {
            return v / n2.sqrt();
        }
    }
}

/// Generate a coarse-grained chromosome path inside a sphere of
/// `cell_radius` centred at the origin: a self-avoiding random walk of up
/// to `bead_count` beads spaced `step` apart, kept inside the cell
/// (allowing for `bead_radius`) and ≥ ~1.5·`bead_radius` from any
/// non-adjacent bead, with worm-like-chain persistence. Points are
/// origin-relative (the caller places the fiber at the compartment
/// centre). Returns however many beads it placed — if the walk traps
/// itself in the confined volume it stops early rather than spinning.
pub fn generate_fiber<R: Rng>(
    cell_radius: f32,
    bead_count: usize,
    step: f32,
    bead_radius: f32,
    rng: &mut R,
) -> Vec<Point3<f32>> {
    let max_r = (cell_radius - bead_radius).max(step);
    let min_sep = 1.5 * bead_radius;
    let min_sep2 = min_sep * min_sep;

    let mut pts: Vec<Point3<f32>> = Vec::with_capacity(bead_count);
    pts.push(Point3::origin());
    let mut dir = random_unit(rng);
    let mut stuck_runs = 0usize;

    while pts.len() < bead_count {
        let last = pts[pts.len() - 1];
        let mut placed = false;
        for _ in 0..48 {
            let rnd = random_unit(rng);
            // Worm-like chain: blend persistence (stiffness) with noise.
            let cd = (dir * 0.65 + rnd * 0.35).normalize();
            let next = last + cd * step;
            if next.coords.norm() > max_r {
                continue; // would leave the cell
            }
            // Self-avoidance (the immediate predecessor is allowed to touch).
            let mut ok = true;
            let upto = pts.len().saturating_sub(1);
            for j in 0..upto {
                if (pts[j] - next).norm_squared() < min_sep2 {
                    ok = false;
                    break;
                }
            }
            if ok {
                pts.push(next);
                dir = cd;
                placed = true;
                break;
            }
        }
        if placed {
            stuck_runs = 0;
        } else {
            // Trapped: kink in a fresh direction biased toward the centre
            // to escape a crowded boundary. Give up if it keeps failing
            // (the cell is saturated at this spacing).
            let toward_center = (Point3::origin() - last)
                .try_normalize(1e-6)
                .unwrap_or_else(|| random_unit(rng));
            dir = (random_unit(rng) * 0.5 + toward_center * 0.5).normalize();
            stuck_runs += 1;
            if stuck_runs > bead_count {
                break;
            }
        }
    }
    pts
}

/// Generate a *plectonemically supercoiled* chromosome inside a sphere of
/// `cell_radius`: a coarse-grained backbone axis (a confined self-avoiding
/// walk) with the genome wound around it as an interwound double helix —
/// out along the axis as one superhelix, back as the complementary one
/// (phase-offset by π), the way a bacterial chromosome's negatively
/// supercoiled DNA folds into plectonemes. `sc_radius` is the superhelix
/// radius, `sc_pitch` its axial rise per turn; beads stay `step` apart
/// along the strand. Falls back to a plain [`generate_fiber`] walk when the
/// coil can't fit the cell. Points are origin-relative; deterministic for
/// a given RNG.
pub fn generate_supercoiled_fiber<R: Rng>(
    cell_radius: f32,
    bead_count: usize,
    step: f32,
    bead_radius: f32,
    sc_radius: f32,
    sc_pitch: f32,
    rng: &mut R,
) -> Vec<Point3<f32>> {
    // No coil (or it can't fit): just lay a plain self-avoiding walk.
    if sc_radius <= 1e-3 || cell_radius < sc_radius + 2.0 * bead_radius + step {
        return generate_fiber(cell_radius, bead_count, step, bead_radius, rng);
    }
    let tau = std::f32::consts::TAU;
    let cpt = ((tau * sc_radius).powi(2) + sc_pitch * sc_pitch).sqrt(); // contour / turn
    let da = step * sc_pitch / cpt; // axial advance per bead
    let dphi = step * tau / cpt; // phase advance per bead

    // Backbone axis: a thin self-avoiding walk — its separation is just the
    // bead skin, so the step stays fine for smooth frames. Confining it to
    // `cell_radius - sc_radius` leaves room for the coil + skin so wound
    // beads stay inside the cell. The axis runs roughly a cell-diameter long
    // at these scales, so it doesn't fold tight enough for the coils
    // themselves to interpenetrate. (Passing sc_radius+bead_radius here
    // instead would set the walk's self-avoidance ≫ step and trap it.)
    let n_half = (bead_count / 2).max(1);
    // Oversize the walk: the smoothing pass below shortens it, and we want
    // enough arc length left to seat all `n_half` beads of each strand.
    let axis_len_target = n_half as f32 * da * 1.8;
    let axis_step = (sc_pitch / 6.0).max(2.0 * bead_radius);
    let axis_n = ((axis_len_target / axis_step).ceil() as usize + 2).max(2);
    let mut axis = generate_fiber(cell_radius - sc_radius, axis_n, axis_step, bead_radius, rng);
    if axis.len() < 2 {
        return generate_fiber(cell_radius, bead_count, step, bead_radius, rng);
    }
    // A fat coil amplifies backbone kinks: at a backbone vertex the coil's
    // offset plane tilts by the kink angle, jerking the wound strand by
    // ~sc_radius·angle. The raw self-avoiding walk kinks by tens of degrees,
    // so Laplacian-smooth the axis until its curvature is gentle relative to
    // the coil radius and the helix winds evenly.
    smooth_polyline(&mut axis, 25);

    // Rotation-minimizing frames + cumulative arc length along the axis.
    let frames = frames_along(&axis);
    let mut cum = vec![0.0_f32; axis.len()];
    for i in 1..axis.len() {
        cum[i] = cum[i - 1] + (axis[i] - axis[i - 1]).norm();
    }
    let total = cum[axis.len() - 1];

    // Axis position + (N1, N2) frame at arc length `a`.
    let sample = |a: f32| -> (Point3<f32>, Vector3<f32>, Vector3<f32>) {
        let a = a.clamp(0.0, total);
        let mut k = 0;
        while k + 1 < axis.len() && cum[k + 1] <= a {
            k += 1;
        }
        let k1 = (k + 1).min(axis.len() - 1);
        let seg = (cum[k1] - cum[k]).max(1e-6);
        let t = ((a - cum[k]) / seg).clamp(0.0, 1.0);
        let pos = axis[k] + (axis[k1] - axis[k]) * t;
        // Interpolate the vertex tangents (not the piecewise-constant segment
        // direction) so the offset frame rotates smoothly through vertices.
        let tang = (frames[k].0 * (1.0 - t) + frames[k1].0 * t)
            .try_normalize(1e-6)
            .unwrap_or(frames[k].0);
        let mut n1 = frames[k].1 * (1.0 - t) + frames[k1].1 * t;
        n1 = (n1 - tang * tang.dot(&n1))
            .try_normalize(1e-6)
            .unwrap_or_else(|| perp(tang));
        let n2 = tang.cross(&n1);
        (pos, n1, n2)
    };

    let fit_half = ((total / da).floor() as usize).clamp(1, n_half);
    let mut pts = Vec::with_capacity(2 * fit_half);
    // Outgoing superhelix: a = 0 → L, phase rising.
    for i in 0..fit_half {
        let (c, n1, n2) = sample(i as f32 * da);
        let phi = i as f32 * dphi;
        pts.push(c + n1 * (phi.cos() * sc_radius) + n2 * (phi.sin() * sc_radius));
    }
    // Return superhelix: a = L → 0, phase continues but offset by π so the
    // two strands interwind (opposite sides of the axis at every height).
    let phi_apex = (fit_half - 1) as f32 * dphi;
    for j in 0..fit_half {
        let (c, n1, n2) = sample((fit_half - 1 - j) as f32 * da);
        let phi = phi_apex + std::f32::consts::PI + j as f32 * dphi;
        pts.push(c + n1 * (phi.cos() * sc_radius) + n2 * (phi.sin() * sc_radius));
    }
    pts
}

/// Wind an interwound (plectonemic) double strand of up to `n_beads` along a
/// local `axis` polyline: out along the axis as one superhelix, back as the
/// complementary one (offset by π) so the two strands interwind. Extracted from
/// [`generate_supercoiled_fiber`] so [`generate_nucleoid`] can wind one
/// plectoneme per domain. Returns the bead path (out then back).
fn wind_plectoneme(
    axis: &[Point3<f32>],
    n_beads: usize,
    step: f32,
    sc_radius: f32,
    sc_pitch: f32,
) -> Vec<Point3<f32>> {
    if axis.len() < 2 || n_beads < 2 {
        return axis.to_vec();
    }
    let tau = std::f32::consts::TAU;
    let cpt = ((tau * sc_radius).powi(2) + sc_pitch * sc_pitch).sqrt();
    let da = step * sc_pitch / cpt;
    let dphi = step * tau / cpt;
    let frames = frames_along(axis);
    let mut cum = vec![0.0_f32; axis.len()];
    for i in 1..axis.len() {
        cum[i] = cum[i - 1] + (axis[i] - axis[i - 1]).norm();
    }
    let total = cum[axis.len() - 1];
    if total <= 1e-3 {
        return Vec::new();
    }
    let sample = |a: f32| -> (Point3<f32>, Vector3<f32>, Vector3<f32>) {
        let a = a.clamp(0.0, total);
        let mut k = 0;
        while k + 1 < axis.len() && cum[k + 1] <= a {
            k += 1;
        }
        let k1 = (k + 1).min(axis.len() - 1);
        let seg = (cum[k1] - cum[k]).max(1e-6);
        let t = ((a - cum[k]) / seg).clamp(0.0, 1.0);
        let pos = axis[k] + (axis[k1] - axis[k]) * t;
        let tang = (frames[k].0 * (1.0 - t) + frames[k1].0 * t)
            .try_normalize(1e-6)
            .unwrap_or(frames[k].0);
        let mut n1 = frames[k].1 * (1.0 - t) + frames[k1].1 * t;
        n1 = (n1 - tang * tang.dot(&n1))
            .try_normalize(1e-6)
            .unwrap_or_else(|| perp(tang));
        let n2 = tang.cross(&n1);
        (pos, n1, n2)
    };
    let half = (n_beads / 2).max(1);
    let fit_half = ((total / da).floor() as usize).clamp(1, half);
    let mut pts = Vec::with_capacity(2 * fit_half);
    for i in 0..fit_half {
        let (c, n1, n2) = sample(i as f32 * da);
        let phi = i as f32 * dphi;
        pts.push(c + n1 * (phi.cos() * sc_radius) + n2 * (phi.sin() * sc_radius));
    }
    let phi_apex = (fit_half - 1) as f32 * dphi;
    for j in 0..fit_half {
        let (c, n1, n2) = sample((fit_half - 1 - j) as f32 * da);
        let phi = phi_apex + std::f32::consts::PI + j as f32 * dphi;
        pts.push(c + n1 * (phi.cos() * sc_radius) + n2 * (phi.sin() * sc_radius));
    }
    pts
}

/// Generate a *rosette* nucleoid: the circular genome as a series of
/// plectonemic loop **domains** branching off a central backbone scaffold —
/// Maritan's "unsupercoiled segments punctuated with supercoiled plectonemes".
/// Each of `domains` domains is a local interwound loop (≈ a topological
/// domain) anchored on the scaffold and bulging outward; consecutive loops
/// connect along the scaffold so the whole thing reads as one circular genome.
/// More faithful than one global plectoneme, and the loops compact the genome
/// so more contour fits the cell. `domains <= 1` falls back to a single
/// plectoneme.
#[allow(clippy::too_many_arguments)]
pub fn generate_nucleoid<R: Rng>(
    cell_radius: f32,
    bead_count: usize,
    step: f32,
    bead_radius: f32,
    sc_radius: f32,
    sc_pitch: f32,
    domains: usize,
    rng: &mut R,
) -> Vec<Point3<f32>> {
    if domains <= 1 {
        return generate_supercoiled_fiber(
            cell_radius, bead_count, step, bead_radius, sc_radius, sc_pitch, rng,
        );
    }
    let core_r = (cell_radius - sc_radius - bead_radius).max(step);
    // Scaffold anchor points (one per domain) — a short confined walk in the
    // cell core; loops bulge outward from these.
    let backbone_step = (2.2 * sc_radius).max(core_r * 1.2 / domains as f32);
    let anchors = generate_fiber(core_r * 0.65, domains, backbone_step, bead_radius, rng);
    if anchors.len() < 2 {
        return generate_supercoiled_fiber(
            cell_radius, bead_count, step, bead_radius, sc_radius, sc_pitch, rng,
        );
    }
    let na = anchors.len();
    let per = (bead_count / na).max(2);
    let max_apex = core_r; // keep wound beads (offset sc_radius) inside the cell
    let loop_height = (sc_radius * 3.0).min(core_r * 0.45);

    let mut out: Vec<Point3<f32>> = Vec::with_capacity(bead_count);
    for d in 0..na {
        let a = anchors[d];
        let b = anchors[(d + 1) % na];
        let mid = Point3::from((a.coords + b.coords) * 0.5);
        // Bulge away from the cell centre (fall back to a perpendicular if the
        // midpoint sits at the origin).
        let outward = mid
            .coords
            .try_normalize(1e-3)
            .unwrap_or_else(|| perp((b - a).try_normalize(1e-6).unwrap_or(Vector3::z())));
        let mut apex = mid + outward * loop_height;
        if apex.coords.norm() > max_apex {
            apex = Point3::from(apex.coords.normalize() * max_apex);
        }
        // Local axis a → apex → b, subdivided + smoothed for clean frames.
        let sub = 6;
        let mut axis = Vec::with_capacity(2 * sub);
        for s in 0..=sub {
            let t = s as f32 / sub as f32;
            axis.push(Point3::from(a.coords * (1.0 - t) + apex.coords * t));
        }
        for s in 1..=sub {
            let t = s as f32 / sub as f32;
            axis.push(Point3::from(apex.coords * (1.0 - t) + b.coords * t));
        }
        smooth_polyline(&mut axis, 8);
        out.extend(wind_plectoneme(&axis, per, step, sc_radius, sc_pitch));
        if out.len() >= bead_count {
            break;
        }
    }
    out.truncate(bead_count);
    out
}

/// Per-vertex rotation-minimizing frames `(T, N1, N2)` along a polyline.
/// Each normal is carried forward by the rotation that maps the previous
/// tangent onto the current one, then re-orthogonalised — avoiding the
/// abrupt normal flips a Frenet frame suffers at inflection points, so the
/// wound helix doesn't kink where the backbone straightens.
fn frames_along(axis: &[Point3<f32>]) -> Vec<(Vector3<f32>, Vector3<f32>, Vector3<f32>)> {
    let m = axis.len();
    let tangent = |i: usize| -> Vector3<f32> {
        // Centered difference at interior vertices — a smoother tangent than
        // the one-sided segment direction, so the wound coil's offset plane
        // turns gradually rather than in per-segment steps.
        let t = if i == 0 {
            axis[1] - axis[0]
        } else if i + 1 >= m {
            axis[m - 1] - axis[m - 2]
        } else {
            axis[i + 1] - axis[i - 1]
        };
        t.try_normalize(1e-6).unwrap_or_else(|| Vector3::z())
    };
    let mut frames = Vec::with_capacity(m);
    let t0 = tangent(0);
    let n1 = perp(t0);
    frames.push((t0, n1, t0.cross(&n1)));
    for i in 1..m {
        let tp = frames[i - 1].0;
        let tc = tangent(i);
        let mut n1 = frames[i - 1].1;
        let v = tp.cross(&tc);
        let s = v.norm();
        if s > 1e-6 {
            let angle = s.atan2(tp.dot(&tc));
            n1 = Rotation3::from_axis_angle(&Unit::new_normalize(v), angle) * n1;
        }
        n1 = (n1 - tc * tc.dot(&n1))
            .try_normalize(1e-6)
            .unwrap_or_else(|| perp(tc));
        frames.push((tc, n1, tc.cross(&n1)));
    }
    frames
}

/// In-place Laplacian smoothing of an open polyline (endpoints fixed):
/// each interior vertex moves toward the average of its neighbours. Run a
/// few dozen passes to turn a kinky walk into a gently curving backbone.
fn smooth_polyline(pts: &mut Vec<Point3<f32>>, passes: usize) {
    let n = pts.len();
    if n < 3 {
        return;
    }
    for _ in 0..passes {
        let prev = pts.clone();
        for i in 1..n - 1 {
            pts[i] = Point3::from(
                (prev[i - 1].coords + prev[i].coords * 2.0 + prev[i + 1].coords) * 0.25,
            );
        }
    }
}

/// Some unit vector perpendicular to `t`.
fn perp(t: Vector3<f32>) -> Vector3<f32> {
    let a = if t.x.abs() < 0.9 { Vector3::x() } else { Vector3::y() };
    (a - t * t.dot(&a)).normalize()
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn fiber_is_confined_spaced_and_self_avoiding() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let (cell_radius, step, bead_radius) = (200.0_f32, 10.0_f32, 4.0_f32);
        let pts = generate_fiber(cell_radius, 1500, step, bead_radius, &mut rng);

        // Should fill a good fraction of the cell before (if) trapping.
        assert!(pts.len() > 800, "placed only {} beads", pts.len());

        // Every bead inside the cell.
        for p in &pts {
            assert!(
                p.coords.norm() <= cell_radius - bead_radius + 1e-2,
                "bead outside cell: {}",
                p.coords.norm()
            );
        }
        // Consecutive beads spaced ~step.
        for w in pts.windows(2) {
            let d = (w[1] - w[0]).norm();
            assert!((d - step).abs() < 1e-2, "step {d} != {step}");
        }
        // Non-adjacent beads don't overlap.
        let min_sep = 1.5 * bead_radius;
        for i in 0..pts.len() {
            for j in (i + 2)..pts.len() {
                assert!(
                    (pts[i] - pts[j]).norm() >= min_sep - 1e-2,
                    "beads {i} and {j} overlap"
                );
            }
        }
    }

    #[test]
    fn deterministic_for_seed() {
        let mut a = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(42);
        let pa = generate_fiber(150.0, 300, 8.0, 3.0, &mut a);
        let pb = generate_fiber(150.0, 300, 8.0, 3.0, &mut b);
        assert_eq!(pa, pb);
    }

    #[test]
    fn supercoiled_fiber_is_confined_and_interwound() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let (cell, step, bead) = (2000.0_f32, 22.0_f32, 10.0_f32);
        let (scr, pitch) = (80.0_f32, 100.0_f32);
        let pts = generate_supercoiled_fiber(cell, 1500, step, bead, scr, pitch, &mut rng);
        assert!(pts.len() > 1000, "placed only {} beads", pts.len());

        // Every bead inside the cell.
        for p in &pts {
            assert!(
                p.coords.norm() <= cell - bead + 1e-1,
                "bead outside cell: {}",
                p.coords.norm()
            );
        }

        // The strand doubles back at the midpoint: the apical hairpin spans
        // the coil (~2·sc_radius), proving the interwound structure.
        let mid = pts.len() / 2;
        let apex = (pts[mid] - pts[mid - 1]).norm();
        assert!(apex > 1.5 * scr, "apex hairpin should span the coil, got {apex}");

        // Everywhere else, consecutive beads are spaced ~step along the helix.
        for i in 0..pts.len() - 1 {
            if i + 1 == mid {
                continue;
            }
            let d = (pts[i + 1] - pts[i]).norm();
            assert!(
                d > step * 0.5 && d < step * 1.3,
                "helix spacing {d} off step {step} at bead {i}"
            );
        }
    }

    #[test]
    fn supercoil_is_deterministic_and_falls_back() {
        let mut a = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(5);
        assert_eq!(
            generate_supercoiled_fiber(2000.0, 800, 22.0, 10.0, 80.0, 100.0, &mut a),
            generate_supercoiled_fiber(2000.0, 800, 22.0, 10.0, 80.0, 100.0, &mut b),
        );
        // sc_radius 0 ⇒ identical to the plain self-avoiding walk.
        let mut c = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut d = Xoshiro256PlusPlus::seed_from_u64(5);
        assert_eq!(
            generate_supercoiled_fiber(150.0, 300, 8.0, 3.0, 0.0, 50.0, &mut c),
            generate_fiber(150.0, 300, 8.0, 3.0, &mut d),
        );
    }

    #[test]
    fn nucleoid_rosette_is_confined_multi_domain_and_deterministic() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let (cell, step, bead) = (2000.0_f32, 22.0_f32, 10.0_f32);
        let (scr, pitch) = (80.0_f32, 120.0_f32);
        let pts = generate_nucleoid(cell, 8000, step, bead, scr, pitch, 40, &mut rng);
        assert!(pts.len() > 4000, "placed only {} beads", pts.len());
        for p in &pts {
            assert!(
                p.coords.norm() <= cell - bead + 1.0,
                "bead outside cell: {}",
                p.coords.norm()
            );
        }
        // Multiple domains ⇒ several apical hairpins (big consecutive jumps),
        // unlike a single plectoneme (one hairpin).
        let jumps = pts
            .windows(2)
            .filter(|w| (w[1] - w[0]).norm() > 1.5 * scr)
            .count();
        assert!(jumps >= 5, "expected several domain hairpins, got {jumps}");

        // Deterministic; domains<=1 falls back to a single plectoneme.
        let mut a = Xoshiro256PlusPlus::seed_from_u64(9);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(9);
        assert_eq!(
            generate_nucleoid(cell, 2000, step, bead, scr, pitch, 30, &mut a),
            generate_nucleoid(cell, 2000, step, bead, scr, pitch, 30, &mut b),
        );
        let mut c = Xoshiro256PlusPlus::seed_from_u64(9);
        let mut e = Xoshiro256PlusPlus::seed_from_u64(9);
        assert_eq!(
            generate_nucleoid(cell, 1500, step, bead, scr, pitch, 1, &mut c),
            generate_supercoiled_fiber(cell, 1500, step, bead, scr, pitch, &mut e),
        );
    }
}
