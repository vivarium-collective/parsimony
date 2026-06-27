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

use nalgebra::{Matrix3, Point3, Rotation3, Unit, UnitQuaternion, Vector3};
use rand::Rng;

/// Gaussian constriction (septum) at the midplane of a dividing capsule.
/// `depth` is the fractional radius reduction at axial=0 (e.g. 0.6 → 60%
/// narrower waist); `width` is the Gaussian σ in the same units as the
/// capsule's `half_len`/`radius` (Ångströms). Stored as a fraction so
/// `inset` does not need to rescale it.
#[derive(Debug, Clone, Copy)]
pub struct Septum {
    pub depth: f32,
    pub width: f32,
}

/// The envelope the chromosome is confined to, in the origin-relative frame
/// the fiber generators use (the caller offsets the result by the compartment
/// centre). A sphere is centred at the origin; a capsule's medial segment runs
/// from `-half_len*axis` to `+half_len*axis` with cap radius `radius`.
#[derive(Debug, Clone, Copy)]
pub enum CellShape {
    Sphere { radius: f32 },
    Capsule { half_len: f32, radius: f32, axis: Vector3<f32>, septum: Option<Septum> },
}

impl CellShape {
    /// Farthest interior extent from the origin (used for fallback sizing).
    pub fn reach(&self) -> f32 {
        match *self {
            CellShape::Sphere { radius } => radius,
            CellShape::Capsule { half_len, radius, .. } => half_len + radius,
        }
    }

    /// Shrink the envelope inward by `margin` on every side.
    pub fn inset(&self, margin: f32) -> CellShape {
        match *self {
            CellShape::Sphere { radius } => CellShape::Sphere { radius: (radius - margin).max(0.0) },
            CellShape::Capsule { half_len, radius, axis, septum } => CellShape::Capsule {
                half_len: (half_len - margin).max(0.0),
                radius: (radius - margin).max(0.0),
                axis,
                septum, // depth is a fraction → scales automatically with reduced radius
            },
        }
    }

    /// Nearest point on the medial axis (sphere: the origin). Always inside
    /// the envelope (distance 0 from the medial axis ≤ cap radius), so it is a
    /// safe guaranteed-inside fallback for confinement.
    pub(crate) fn medial(&self, p: &Point3<f32>) -> Point3<f32> {
        match *self {
            CellShape::Sphere { .. } => Point3::origin(),
            CellShape::Capsule { half_len, axis, .. } => {
                let t = p.coords.dot(&axis).clamp(-half_len, half_len);
                Point3::from(axis * t)
            }
        }
    }

    /// Allowed radial distance from the medial axis at point `p`.
    /// For a plain capsule or sphere this is the constant `cap_radius()`.
    /// For a septum-aware capsule the cylinder region is tapered by a
    /// Gaussian dip: `radius * (1 - depth * exp(-(axial/width)²))`;
    /// end-caps are always at the full `radius`.
    fn effective_radius_at(&self, p: &Point3<f32>) -> f32 {
        if let CellShape::Capsule { half_len, radius, axis, septum: Some(sep) } = *self {
            let axial = p.coords.dot(&axis);
            if axial.abs() <= half_len {
                return radius * (1.0 - sep.depth * (-(axial / sep.width).powi(2)).exp());
            }
        }
        self.cap_radius()
    }

    /// Is `p` (origin-relative) inside the envelope?
    pub fn contains(&self, p: &Point3<f32>) -> bool {
        (p - self.medial(p)).norm() <= self.effective_radius_at(p)
    }

    pub(crate) fn cap_radius(&self) -> f32 {
        match *self {
            CellShape::Sphere { radius } => radius,
            CellShape::Capsule { radius, .. } => radius,
        }
    }

    /// Project `p` to the nearest point inside this envelope (no-op if already
    /// inside): pull it radially onto the surface around the nearest medial-axis
    /// point. Used to keep chromosome beads within the cell after a recentring
    /// shift nudges the spread-out nucleoid's leading end past a (narrowing) cap.
    pub(crate) fn clamp_inside(&self, p: Point3<f32>) -> Point3<f32> {
        let m = self.medial(&p);
        let v = p - m;
        let dist = v.norm();
        let maxr = self.cap_radius();
        if dist > maxr && dist > 1e-6 {
            m + v * (maxr / dist)
        } else {
            p
        }
    }

    /// Outward unit direction (radial from the medial axis) at `p`.
    pub fn outward(&self, p: &Point3<f32>) -> Vector3<f32> {
        (p - self.medial(p)).try_normalize(1e-6).unwrap_or_else(|| perp(self.long_axis()))
    }

    /// Direction toward the medial axis/centre at `p` (escape direction).
    pub fn inward(&self, p: &Point3<f32>) -> Vector3<f32> {
        (self.medial(p) - p).try_normalize(1e-6).unwrap_or_else(|| perp(self.long_axis()))
    }

    fn long_axis(&self) -> Vector3<f32> {
        match *self {
            CellShape::Sphere { .. } => Vector3::z(),
            CellShape::Capsule { axis, .. } => axis,
        }
    }
}

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

/// Internal: self-avoiding worm-like-chain walk starting from `start`,
/// placing up to `bead_count` beads spaced `step` apart, kept inside
/// `inset`. `bead_radius` controls the min-separation exclusion zone.
/// Stops early if the walk traps itself. Shared by [`generate_fiber`] and
/// [`generate_rna_strand`].
fn walk_from<R: Rng>(
    start: Point3<f32>,
    inset: CellShape,
    bead_count: usize,
    step: f32,
    bead_radius: f32,
    rng: &mut R,
) -> Vec<Point3<f32>> {
    let min_sep = 1.5 * bead_radius;
    let min_sep2 = min_sep * min_sep;

    let mut pts: Vec<Point3<f32>> = Vec::with_capacity(bead_count);
    pts.push(start);
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
            if !inset.contains(&next) {
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
            // Trapped: kink in a fresh direction biased toward the medial
            // axis to escape a crowded boundary. Give up if it keeps failing
            // (the cell is saturated at this spacing).
            let toward_center = inset.inward(&last);
            dir = (random_unit(rng) * 0.5 + toward_center * 0.5).normalize();
            stuck_runs += 1;
            if stuck_runs > bead_count {
                break;
            }
        }
    }
    pts
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
    shape: CellShape,
    bead_count: usize,
    step: f32,
    bead_radius: f32,
    rng: &mut R,
) -> Vec<Point3<f32>> {
    walk_from(Point3::origin(), shape.inset(bead_radius), bead_count, step, bead_radius, rng)
}

/// Generate a confined self-avoiding RNA strand grown from a given `root`
/// (the RNAP attachment point). Behaves identically to [`generate_fiber`]
/// but seeds the walk at `root` rather than the origin, making it suitable
/// for every nascent transcript in the cell.
///
/// If `root` is marginally outside `shape.inset(bead_radius)` it is
/// surface-pulled just inside along the inward-radial normal (never
/// collapsed to the medial axis). When a step cannot find an in-bounds
/// candidate within the retry budget the walk stops early and returns
/// however many beads were placed (≥ 1). Deterministic for a fixed `rng`
/// seed.
pub fn generate_rna_strand<R: Rng>(
    root: Point3<f32>,
    bead_count: usize,
    step: f32,
    bead_radius: f32,
    shape: CellShape,
    rng: &mut R,
) -> Vec<Point3<f32>> {
    let inset = shape.inset(bead_radius);
    // Clamp root to just inside the inset via surface-pull if needed. We
    // move along the inward-radial direction (outward from medial axis) —
    // never to the medial axis — preserving the azimuthal location of the
    // RNAP attachment point.
    let root_clamped = if inset.contains(&root) {
        root
    } else {
        let m = inset.medial(&root);
        let rad = root.coords - m.coords;
        let r = rad.norm();
        let dir = if r > 1e-6 { rad / r } else { perp(shape.long_axis()) };
        m + dir * (inset.cap_radius() * 0.999)
    };
    walk_from(root_clamped, inset, bead_count, step, bead_radius, rng)
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
    shape: CellShape,
    bead_count: usize,
    step: f32,
    bead_radius: f32,
    sc_radius: f32,
    sc_pitch: f32,
    rng: &mut R,
) -> Vec<Point3<f32>> {
    // No coil (or it can't fit): just lay a plain self-avoiding walk.
    if sc_radius <= 1e-3 || shape.reach() < sc_radius + 2.0 * bead_radius + step {
        return generate_fiber(shape, bead_count, step, bead_radius, rng);
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
    let axis_inset = shape.inset(sc_radius);
    let mut axis = generate_fiber(axis_inset, axis_n, axis_step, bead_radius, rng);
    if axis.len() < 2 {
        return generate_fiber(shape, bead_count, step, bead_radius, rng);
    }
    // A fat coil amplifies backbone kinks: at a backbone vertex the coil's
    // offset plane tilts by the kink angle, jerking the wound strand by
    // ~sc_radius·angle. The raw self-avoiding walk kinks by tens of degrees,
    // so Laplacian-smooth the axis until its curvature is gentle relative to
    // the coil radius and the helix winds evenly.
    smooth_polyline(&mut axis, 25);
    // Recenter the backbone axis to the shape origin.  The SAW starts at the
    // origin and drifts toward one pole; the wound coil inherits that drift,
    // pushing the chromosome centroid away from centre.  Translate the axis
    // centroid to the origin (a deterministic shift, no new RNG draws), then
    // pull any vertex that now lies outside axis_inset back to the inset
    // SURFACE along the inward-radial direction — preserving azimuthal position
    // and never projecting to the centerline, which would cause a
    // collinear-convergence artifact.  Use the bead-radius inset as the
    // confinement boundary so wound beads (axis + sc_radius offset) stay
    // within the cell envelope.
    {
        let n = axis.len() as f32;
        let c = axis.iter().fold(Vector3::zeros(), |acc, p| acc + p.coords) / n;
        for p in &mut axis {
            p.coords -= c;
        }
        // Pull to axis_inset.inset(bead_radius) so the sc_radius winding offset
        // keeps the wound beads inside the cell (cap_radius - bead_radius + sc_radius
        // ≤ cap_radius when bead_radius ≥ sc_radius, which holds for typical params).
        let tight = axis_inset.inset(bead_radius);
        let cr = tight.cap_radius();
        for p in &mut axis {
            if !tight.contains(p) {
                let m = tight.medial(p);
                let rad = p.coords - m.coords;
                let r = rad.norm();
                let dir = if r > 1e-6 { rad / r } else { perp(axis_inset.long_axis()) };
                *p = m + dir * (cr * 0.999);
            }
        }
    }

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

/// A replicating chromosome laid out as a theta (θ) structure.
pub struct ThetaChromosome {
    /// `[main]` when unreplicated; `[main, sister]` when replicating — the
    /// sister strand covers the replicated region (the bubble around oriC).
    pub strands: Vec<Vec<Point3<f32>>>,
    /// The two replication-fork positions (origin-relative), or empty.
    pub forks: Vec<Point3<f32>>,
    /// Origin-of-replication (oriC) positions: one per genome copy — the main
    /// strand's oriC, plus the sister copy's oriC once replication has begun.
    pub oric: Vec<Point3<f32>>,
    /// Replication-terminus (terC) position: the genome locus opposite oriC.
    pub ter: Vec<Point3<f32>>,
}

/// Generate a chromosome as a theta structure. The main genome is a single
/// supercoiled strand; when `fork_fraction > 0` the replicated region around
/// oriC is duplicated into a *sister* strand that bulges away from the main
/// one and pinches back to it at two forks (the classic replicating-loop
/// bubble). `fork_fraction` is how far each fork has travelled along its
/// replichore, 0..1 (0.45 → forks ~45% of the way oriC→terC). Origin-relative;
/// deterministic for a given RNG.
pub fn generate_theta_chromosome<R: Rng>(
    shape: CellShape,
    genome_beads: usize,
    fork_fraction: f32,
    step: f32,
    bead_radius: f32,
    sc_radius: f32,
    sc_pitch: f32,
    rng: &mut R,
) -> ThetaChromosome {
    // `generate_supercoiled_fiber` recenters the backbone axis when sc_radius > 0,
    // but falls back to a plain `generate_fiber` walk for sc_radius = 0.  That
    // walk starts at the origin and drifts; the caller (`place_chromosome`) used
    // to correct this with a rigid centroid shift + `medial()` clamp — which
    // created the centerline-collapse artifact (beads projected onto the cell
    // axis in cap regions).  We fix it here, at the source: translate the main
    // strand so its centroid is at the shape origin, then surface-pull any bead
    // that falls outside the inset (inward-radial direction only, never medial).
    let inset = shape.inset(bead_radius);
    let mut main = generate_supercoiled_fiber(shape, genome_beads, step, bead_radius, sc_radius, sc_pitch, rng);
    if !main.is_empty() {
        let nm = main.len() as f32;
        let c = main.iter().fold(Vector3::zeros(), |acc, p| acc + p.coords) / nm;
        for p in &mut main {
            p.coords -= c;
        }
        // Surface-pull: beads that fell outside the inset after recentering are
        // moved to the inset surface at their own azimuthal angle — NOT to the
        // medial axis (which would create the centerline convergence we are fixing).
        let cr = inset.cap_radius();
        for p in &mut main {
            if !inset.contains(p) {
                let m = inset.medial(p);
                let rad = p.coords - m.coords;
                let r = rad.norm();
                let dir = if r > 1e-6 { rad / r } else { perp(shape.long_axis()) };
                *p = m + dir * (cr * 0.999);
            }
        }
    }
    let n = main.len();
    let f = fork_fraction.clamp(0.0, 0.95);
    // Replicated beads per replichore; the bubble spans 2·r beads around oriC.
    let r = ((f * n as f32 / 2.0).round() as usize).min(n / 2);
    if r < 2 || n < 8 {
        // Unreplicated: one oriC at the strand midpoint (the hairpin fold),
        // terC where the two arms rejoin. Close the loop into a true circle.
        let mid = main[n / 2];
        let end = main[0];
        let first = main[0];
        main.push(first);
        return ThetaChromosome {
            strands: vec![main],
            forks: Vec::new(),
            oric: vec![mid],
            ter: vec![end],
        };
    }
    // Put oriC at the strand midpoint so the bubble is contiguous (no wrap on an
    // open strand); terC then sits near the strand ends, approximating the loop.
    let oric = n / 2;
    let lo = oric - r; // forward fork
    let hi = oric + r; // reverse fork
    // One stable bulge direction for the whole bubble (perpendicular to the
    // fork-to-fork chord, biased outward from the medial axis) → a clean lens.
    let chord = main[hi] - main[lo];
    let chord_dir = chord.try_normalize(1e-6).unwrap_or_else(Vector3::x);
    let mut off = perp(chord_dir);
    let out = shape.outward(&main[oric]);
    if off.dot(&out) < 0.0 {
        off = -off;
    }
    let bulge_max = (shape.cap_radius() * 0.35).max(3.0 * bead_radius);
    let span = (hi - lo) as f32;
    let mut sister = Vec::with_capacity(hi - lo + 1);
    for i in lo..=hi {
        let u = (i - lo) as f32 / span; // 0..1 across the bubble
        let mut bulge = bulge_max * (std::f32::consts::PI * u).sin();
        // Keep the sister inside the cell: shrink the offset until it fits.
        let mut cand = main[i] + off * bulge;
        let mut tries = 0;
        while !inset.contains(&cand) && bulge > bead_radius && tries < 8 {
            bulge *= 0.6;
            cand = main[i] + off * bulge;
            tries += 1;
        }
        sister.push(cand);
    }
    // Fork markers (replisomes) sit at the bubble's two pinch points, main[lo]
    // and main[hi]. On a space-filling self-avoiding walk those two beads can
    // land close together even though they are far apart on the genome, which
    // collapses the two replisome markers into a single visible blob. Guarantee
    // a minimum separation: if the endpoints are nearer than a marker-visible
    // gap, spread them symmetrically about their midpoint along a stable axis
    // (the fork-to-fork chord when well-defined, else the sister-bulge `off`).
    let mut fork_lo = main[lo];
    let mut fork_hi = main[hi];
    let min_fork_sep = (shape.cap_radius() * 0.12).max(250.0);
    if (fork_hi - fork_lo).norm() < min_fork_sep {
        let mid = Point3::from((fork_lo.coords + fork_hi.coords) * 0.5);
        let dir = (fork_hi - fork_lo).try_normalize(1e-6).unwrap_or(off);
        fork_lo = mid - dir * (0.5 * min_fork_sep);
        fork_hi = mid + dir * (0.5 * min_fork_sep);
    }
    let forks = vec![fork_lo, fork_hi];
    // Two oriCs (the bubble has duplicated the origin): one on the main strand,
    // one on the sister, both at the bubble's centre. terC is opposite oriC, at
    // the strand end (the genome was cut there for this open-strand layout).
    let oric = vec![main[oric], sister[sister.len() / 2]];
    let ter = vec![main[0]];
    // Close the genome into a TRUE circle. The supercoiled main strand is a
    // hairpin: it winds out to the far pole (the fold = oriC) and back, so its
    // two ends both sit at the near pole (a=0). Joining them (append the first
    // bead) makes the backbone a continuous closed loop with no free ends —
    // oriC at the fold pole, terC where the two replichore arms rejoin. The
    // replicated bubble (sister) stays an open arc, so the whole structure is a
    // proper θ: one circle with a replication bubble.
    let first = main[0];
    main.push(first);
    ThetaChromosome { strands: vec![main, sister], forks, oric, ter }
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
    let pitch_da = (step * sc_pitch / cpt).max(1e-4); // axial advance at requested pitch
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
    // Tighten the coil so all `half` beads fit this axis (toward full genome
    // length), but never looser than the requested pitch; dphi keeps the
    // on-helix bead spacing ≈ step regardless of how tight da gets.
    let da = (total / half as f32).min(pitch_da).clamp(1e-4, step * 0.999);
    let dphi = ((step * step - da * da).max(0.0)).sqrt() / sc_radius.max(1e-3);
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

/// Generate a *rosette* nucleoid: the circular genome as plectonemic loop
/// **domains** branching off a central backbone scaffold — Maritan's
/// "unsupercoiled segments punctuated with supercoiled plectonemes".
/// `domain_beads[d]` is the bead count of domain `d`; sizing domains to real
/// gene-cluster spans (see [`crate::genome::Genome::domain_bead_allocation`])
/// makes the topology **transcription-coupled** rather than evenly spaced.
/// `domain_beads.len() <= 1` falls back to a single global plectoneme.
#[allow(clippy::too_many_arguments)]
pub fn generate_nucleoid<R: Rng>(
    shape: CellShape,
    domain_beads: &[usize],
    step: f32,
    bead_radius: f32,
    sc_radius: f32,
    sc_pitch: f32,
    rng: &mut R,
) -> Vec<Point3<f32>> {
    let domains = domain_beads.len();
    let total: usize = domain_beads.iter().sum();
    if domains <= 1 {
        return generate_supercoiled_fiber(shape, total, step, bead_radius, sc_radius, sc_pitch, rng);
    }
    // Core envelope that keeps wound beads (offset by sc_radius) inside the cell.
    let core = shape.inset(sc_radius + bead_radius);
    let backbone_step = (2.2 * sc_radius).max(core.reach() * 1.2 / domains as f32);
    // Anchors in a slightly tighter inset so loops have room to bulge.
    let anchor_shape = core.inset(core.cap_radius() * 0.35);
    let anchors = generate_fiber(anchor_shape, domains, backbone_step, bead_radius, rng);
    if anchors.len() < 2 {
        return generate_supercoiled_fiber(shape, total, step, bead_radius, sc_radius, sc_pitch, rng);
    }
    let na = anchors.len();
    let loop_height = (sc_radius * 3.0).min(core.cap_radius() * 0.45);

    let mut out: Vec<Point3<f32>> = Vec::with_capacity(total);
    for d in 0..na {
        let a = anchors[d];
        let b = anchors[(d + 1) % na];
        let mid = Point3::from((a.coords + b.coords) * 0.5);
        let outward = core.outward(&mid);
        let mut apex = mid + outward * loop_height;
        // Clamp the apex back inside the core envelope if the bulge overshoots.
        if !core.contains(&apex) {
            let m = core.medial(&apex);
            apex = m + core.outward(&apex) * core.cap_radius();
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
        out.extend(wind_plectoneme(&axis, domain_beads[d].max(2), step, sc_radius, sc_pitch));
    }
    out
}

/// Per-segment transforms for tiling a DNA-segment mesh along a bead path:
/// sample the path every `seg_step` Å of contour and return `(position,
/// rotation)` where the segment's local +Z (its helix axis) is aligned to the
/// path tangent and rolled about it by the cumulative duplex twist
/// (`twist_rad_per_a` rad per Å of contour) so consecutive segments tile into a
/// continuous, real-scale double helix. Used to render the chromosome as ~tens
/// of thousands of instances of one shared dsDNA mesh (LOD/cel like proteins),
/// instead of a bespoke tube.
pub fn dna_segment_transforms(
    path: &[Point3<f32>],
    seg_step: f32,
    twist_rad_per_a: f32,
) -> Vec<(Point3<f32>, UnitQuaternion<f32>)> {
    if path.len() < 2 || seg_step <= 1e-3 {
        return Vec::new();
    }
    let frames = frames_along(path);
    let mut cum = vec![0.0_f32; path.len()];
    for i in 1..path.len() {
        cum[i] = cum[i - 1] + (path[i] - path[i - 1]).norm();
    }
    let total = cum[path.len() - 1];
    if total <= seg_step {
        return Vec::new();
    }
    let n = (total / seg_step).floor() as usize;
    let mut out = Vec::with_capacity(n + 1);
    let mut i = 0; // monotonic — `s` only increases, so never rescan from 0
    for k in 0..=n {
        let s = (k as f32 * seg_step).min(total);
        while i + 1 < path.len() && cum[i + 1] <= s {
            i += 1;
        }
        let i1 = (i + 1).min(path.len() - 1);
        let seg = (cum[i1] - cum[i]).max(1e-6);
        let t = ((s - cum[i]) / seg).clamp(0.0, 1.0);
        let pos = path[i] + (path[i1] - path[i]) * t;
        let tang = (frames[i].0 * (1.0 - t) + frames[i1].0 * t)
            .try_normalize(1e-6)
            .unwrap_or(frames[i].0);
        let mut n1 = frames[i].1 * (1.0 - t) + frames[i1].1 * t;
        n1 = (n1 - tang * tang.dot(&n1))
            .try_normalize(1e-6)
            .unwrap_or_else(|| perp(tang));
        let n2 = tang.cross(&n1);
        // Roll N1 about the tangent by the cumulative twist, then build the
        // rotation mapping local (X,Y,Z) → (x, tang×x, tang).
        let tw = s * twist_rad_per_a;
        let x = (n1 * tw.cos() + n2 * tw.sin())
            .try_normalize(1e-6)
            .unwrap_or(n1);
        let y = tang.cross(&x);
        let r = Matrix3::from_columns(&[x, y, tang]);
        let q = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(r));
        out.push((pos, q));
    }
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

    /// Convenience: seed an RNG from a u64 (used by multiple tests below).
    fn rng_from(seed: u64) -> Xoshiro256PlusPlus {
        Xoshiro256PlusPlus::seed_from_u64(seed)
    }

    #[test]
    fn rna_strand_roots_at_given_point_and_stays_inside() {
        let shape = CellShape::Capsule { half_len: 400.0, radius: 120.0, axis: Vector3::x(), septum: None };
        let root = Point3::new(-200.0, 50.0, 0.0);
        let mut rng = rng_from(3);
        let strand = generate_rna_strand(root, 60, 18.0, 4.0, shape, &mut rng);
        assert!(strand.len() >= 30, "expected a substantial strand, got {}", strand.len());
        assert!((strand[0] - root).norm() < 8.0, "strand must root at the RNAP point");
        let inset = shape.inset(4.0);
        for p in &strand {
            assert!(inset.contains(p), "RNA bead outside envelope: {:?}", p);
        }
        // longer request → longer strand (monotone in bead_count)
        let longer = generate_rna_strand(root, 120, 18.0, 4.0, shape, &mut rng_from(3));
        assert!(longer.len() >= strand.len());
    }

    #[test]
    fn cellshape_sphere_and_capsule_contain_and_inset() {
        let s = CellShape::Sphere { radius: 100.0 };
        assert!(s.contains(&Point3::new(0.0, 0.0, 99.0)));
        assert!(!s.contains(&Point3::new(0.0, 0.0, 101.0)));
        assert_eq!(s.inset(10.0).reach(), 90.0);

        // Capsule along x: half_len 700, radius 400 (the ecoli_starter cell).
        let c = CellShape::Capsule { half_len: 700.0, radius: 400.0, axis: Vector3::x(), septum: None };
        // On-axis beyond the cylinder but within a cap:
        assert!(c.contains(&Point3::new(700.0, 0.0, 399.0)));
        assert!(!c.contains(&Point3::new(700.0, 0.0, 401.0)));
        // Far past the cap tip is outside:
        assert!(!c.contains(&Point3::new(1101.0, 0.0, 0.0)));
        assert!(c.contains(&Point3::new(1099.0, 0.0, 0.0)));
        // Outward direction is radial from the medial axis (perpendicular to x):
        let o = c.outward(&Point3::new(0.0, 0.0, 50.0));
        assert!((o - Vector3::z()).norm() < 1e-5);
    }

    #[test]
    fn fiber_is_confined_spaced_and_self_avoiding() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let (radius, step, bead_radius) = (200.0_f32, 10.0_f32, 4.0_f32);
        let shape = CellShape::Sphere { radius };
        let pts = generate_fiber(shape, 1500, step, bead_radius, &mut rng);
        assert!(pts.len() > 800, "placed only {} beads", pts.len());
        for p in &pts {
            assert!(p.coords.norm() <= radius - bead_radius + 1e-2, "outside: {}", p.coords.norm());
        }
        for w in pts.windows(2) {
            let d = (w[1] - w[0]).norm();
            assert!((d - step).abs() < 1e-2, "step {d} != {step}");
        }
        let min_sep = 1.5 * bead_radius;
        for i in 0..pts.len() {
            for j in (i + 2)..pts.len() {
                assert!((pts[i] - pts[j]).norm() >= min_sep - 1e-2, "beads {i},{j} overlap");
            }
        }
    }

    #[test]
    fn fiber_confined_to_capsule() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let (half_len, radius, step, bead) = (700.0_f32, 400.0_f32, 22.0_f32, 10.0_f32);
        let shape = CellShape::Capsule { half_len, radius, axis: Vector3::x(), septum: None };
        let pts = generate_fiber(shape, 2000, step, bead, &mut rng);
        assert!(pts.len() > 1000, "placed only {} beads", pts.len());
        let inset = shape.inset(bead);
        for p in &pts {
            assert!(inset.contains(p), "bead outside capsule: {p}");
        }
        // The rod should be longer than it is wide: x-extent > radius.
        let xext = pts.iter().map(|p| p.x).fold(f32::MIN, f32::max)
            - pts.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        assert!(xext > radius, "fiber should extend along the rod, x-extent {xext}");
    }

    #[test]
    fn deterministic_for_seed() {
        let mut a = Xoshiro256PlusPlus::seed_from_u64(42);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(42);
        let s = CellShape::Sphere { radius: 150.0 };
        assert_eq!(
            generate_fiber(s, 300, 8.0, 3.0, &mut a),
            generate_fiber(s, 300, 8.0, 3.0, &mut b)
        );
    }

    #[test]
    fn supercoiled_fiber_is_confined_and_interwound() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(11);
        let (radius, step, bead) = (2000.0_f32, 22.0_f32, 10.0_f32);
        let (scr, pitch) = (80.0_f32, 100.0_f32);
        let shape = CellShape::Sphere { radius };
        let pts = generate_supercoiled_fiber(shape, 1500, step, bead, scr, pitch, &mut rng);
        assert!(pts.len() > 1000, "placed only {} beads", pts.len());
        for p in &pts {
            assert!(p.coords.norm() <= radius - bead + 1e-1, "outside: {}", p.coords.norm());
        }
        let mid = pts.len() / 2;
        let apex = (pts[mid] - pts[mid - 1]).norm();
        assert!(apex > 1.5 * scr, "apex hairpin should span the coil, got {apex}");
        for i in 0..pts.len() - 1 {
            if i + 1 == mid { continue; }
            let d = (pts[i + 1] - pts[i]).norm();
            assert!(d > step * 0.5 && d < step * 1.3, "helix spacing {d} off step at {i}");
        }
    }

    #[test]
    fn theta_chromosome_has_sister_bubble_and_two_forks() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let shape = CellShape::Capsule {
            half_len: 4000.0,
            radius: 2500.0,
            axis: Vector3::x(),
            septum: None,
        };
        // Replicating: forks 40% of the way along each replichore.
        let theta = generate_theta_chromosome(shape, 1200, 0.40, 22.0, 10.0, 80.0, 100.0, &mut rng);
        assert_eq!(theta.strands.len(), 2, "replicating cell → main + sister strand");
        assert_eq!(theta.forks.len(), 2, "a theta bubble has exactly two forks");
        // The sister bubble covers ~2·0.40·(beads/2) = ~0.40·beads beads.
        let main_len = theta.strands[0].len();
        let sister_len = theta.strands[1].len();
        assert!(
            sister_len > main_len / 6 && sister_len < main_len,
            "sister bubble size {sister_len} vs main {main_len}"
        );
        // The genome is a TRUE circle: the main strand is a closed loop, so its
        // last bead coincides with its first (the seam joining the two arms).
        let main_s = &theta.strands[0];
        assert_eq!(
            *main_s.last().unwrap(), main_s[0],
            "main strand must be a closed loop (last bead == first)"
        );
        // The two fork markers (replisomes) must be visibly separated, never
        // collapsed onto each other — the bubble's pinch points can land close
        // together on the space-filling walk, so we enforce a minimum gap.
        let min_fork_sep = (shape.cap_radius() * 0.12).max(250.0);
        let fork_gap = (theta.forks[1] - theta.forks[0]).norm();
        assert!(
            fork_gap >= min_fork_sep - 1.0,
            "the two replisome forks must be at least {min_fork_sep:.0} Å apart, got {fork_gap:.0}"
        );
        // Forks stay near the bubble: within the cell, close to the main strand
        // (on it when not displaced for separation).
        for fk in &theta.forks {
            let d = theta.strands[0].iter().map(|p| (p - fk).norm()).fold(f32::INFINITY, f32::min);
            assert!(d <= min_fork_sep, "fork should stay near the main strand (got {d:.0} Å)");
        }
        // Unreplicated → a single strand, no forks.
        let mut r2 = Xoshiro256PlusPlus::seed_from_u64(7);
        let flat = generate_theta_chromosome(shape, 1200, 0.0, 22.0, 10.0, 80.0, 100.0, &mut r2);
        assert_eq!(flat.strands.len(), 1);
        assert!(flat.forks.is_empty());
        // Unreplicated genome is still a closed circle.
        let fs = &flat.strands[0];
        assert_eq!(*fs.last().unwrap(), fs[0], "unreplicated strand must be a closed loop");
    }

    #[test]
    fn supercoil_is_deterministic_and_falls_back() {
        let s = CellShape::Sphere { radius: 2000.0 };
        let mut a = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(5);
        assert_eq!(
            generate_supercoiled_fiber(s, 800, 22.0, 10.0, 80.0, 100.0, &mut a),
            generate_supercoiled_fiber(s, 800, 22.0, 10.0, 80.0, 100.0, &mut b),
        );
        // sc_radius 0 ⇒ identical to the plain self-avoiding walk.
        let s2 = CellShape::Sphere { radius: 150.0 };
        let mut c = Xoshiro256PlusPlus::seed_from_u64(5);
        let mut d = Xoshiro256PlusPlus::seed_from_u64(5);
        assert_eq!(
            generate_supercoiled_fiber(s2, 300, 8.0, 3.0, 0.0, 50.0, &mut c),
            generate_fiber(s2, 300, 8.0, 3.0, &mut d),
        );
    }

    #[test]
    fn nucleoid_rosette_is_confined_multi_domain_and_deterministic() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        let (radius, step, bead) = (2000.0_f32, 22.0_f32, 10.0_f32);
        let (scr, pitch) = (80.0_f32, 120.0_f32);
        let s = CellShape::Sphere { radius };
        let pts = generate_nucleoid(s, &vec![200; 40], step, bead, scr, pitch, &mut rng);
        assert!(pts.len() > 4000, "placed only {} beads", pts.len());
        for p in &pts {
            assert!(p.coords.norm() <= radius - bead + 1.0, "outside: {}", p.coords.norm());
        }
        let jumps = pts.windows(2).filter(|w| (w[1] - w[0]).norm() > 1.5 * scr).count();
        assert!(jumps >= 5, "expected several domain hairpins, got {jumps}");

        let mut a = Xoshiro256PlusPlus::seed_from_u64(9);
        let mut b = Xoshiro256PlusPlus::seed_from_u64(9);
        assert_eq!(
            generate_nucleoid(s, &vec![66; 30], step, bead, scr, pitch, &mut a),
            generate_nucleoid(s, &vec![66; 30], step, bead, scr, pitch, &mut b),
        );
        let mut c = Xoshiro256PlusPlus::seed_from_u64(9);
        let mut e = Xoshiro256PlusPlus::seed_from_u64(9);
        assert_eq!(
            generate_nucleoid(s, &[1500], step, bead, scr, pitch, &mut c),
            generate_supercoiled_fiber(s, 1500, step, bead, scr, pitch, &mut e),
        );
    }

    #[test]
    fn nucleoid_confined_to_capsule() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(5);
        // E. coli-ish rod: 2 µm × 0.8 µm in Å.
        let shape = CellShape::Capsule { half_len: 7000.0, radius: 4000.0, axis: Vector3::x(), septum: None };
        let (step, bead, scr, pitch) = (22.0_f32, 10.0_f32, 80.0_f32, 120.0_f32);
        let pts = generate_nucleoid(shape, &vec![300; 60], step, bead, scr, pitch, &mut rng);
        assert!(pts.len() > 6000, "placed only {} beads", pts.len());
        let inset = shape.inset(bead);
        let outside = pts.iter().filter(|p| !inset.contains(p)).count();
        assert!(outside == 0, "{outside} nucleoid beads left the capsule");
        // The nucleoid fills the rod: x-extent exceeds the cap radius.
        let xext = pts.iter().map(|p| p.x).fold(f32::MIN, f32::max)
            - pts.iter().map(|p| p.x).fold(f32::MAX, f32::min);
        assert!(xext > shape.cap_radius(), "nucleoid should fill the rod, x-extent {xext}");
    }

    #[test]
    fn capsule_with_septum_tapers_at_midcell() {
        // Capsule along x: half_len 400, radius 120, depth 0.6, width 30.
        // Effective radius at midcell = 120*(1-0.6*exp(0)) = 48.
        let shape = CellShape::Capsule {
            half_len: 400.0,
            radius: 120.0,
            axis: Vector3::x(),
            septum: Some(Septum { depth: 0.6, width: 30.0 }),
        };

        // Midcell (axial=0), radial 80: 80 > 48 → outside.
        let p_mid_80 = Point3::new(0.0, 80.0, 0.0);
        assert!(!shape.contains(&p_mid_80),
            "radial=80 > eff_r=48 at midcell → outside");

        // Midcell, radial 40: 40 < 48 → inside.
        let p_mid_40 = Point3::new(0.0, 40.0, 0.0);
        assert!(shape.contains(&p_mid_40),
            "radial=40 < eff_r=48 at midcell → inside");

        // Cap region (axial=480 > half_len=400), radial 80: caps unaffected → inside.
        let p_cap = Point3::new(480.0, 80.0, 0.0);
        assert!(shape.contains(&p_cap),
            "cap region unaffected by septum: radial=80 < radius=120");

        // Without septum: midcell radial=80 → inside (80 < 120).
        let no_septum = CellShape::Capsule {
            half_len: 400.0, radius: 120.0, axis: Vector3::x(), septum: None,
        };
        assert!(no_septum.contains(&p_mid_80),
            "no septum → full radius, radial=80 inside");

        // inset(10) carries septum: radius→110, depth still 0.6 → eff_r=110*(1-0.6)=44.
        let inset = shape.inset(10.0);
        match inset {
            CellShape::Capsule { radius, septum: Some(s), .. } => {
                assert!((radius - 110.0).abs() < 1e-3, "inset radius should be 110");
                assert!((s.depth - 0.6).abs() < 1e-3, "inset carries septum depth");
            }
            _ => panic!("inset should return Capsule with Some(Septum)"),
        }
        // Confirm inset midcell boundary near eff_r≈44 (110*(1-0.6)=43.999… in f32).
        // Use values clearly inside/outside to avoid floating-point boundary ambiguity.
        let p_mid_43 = Point3::new(0.0, 43.0, 0.0);
        let p_mid_45 = Point3::new(0.0, 45.0, 0.0);
        assert!(inset.contains(&p_mid_43), "inset midcell r=43 inside (eff_r≈44)");
        assert!(!inset.contains(&p_mid_45), "inset midcell r=45 outside (eff_r≈44)");
    }

    #[test]
    fn dna_segment_transforms_tile_and_orient() {
        // Straight path along +x.
        let path: Vec<Point3<f32>> =
            (0..=100).map(|i| Point3::new(i as f32 * 10.0, 0.0, 0.0)).collect();
        let xf = dna_segment_transforms(&path, 40.0, 0.176);
        assert!(xf.len() > 20, "expected many segments, got {}", xf.len());
        // Positions advance ~seg_step along the path and stay on it.
        assert!((xf[1].0.x - 40.0).abs() < 2.0);
        for (p, _) in &xf {
            assert!(p.y.abs() < 1e-3 && p.z.abs() < 1e-3, "off-path {p}");
        }
        // Local +Z (helix axis) maps to the tangent (+x).
        let zmap = xf[0].1 * Vector3::z();
        assert!((zmap - Vector3::x()).norm() < 1e-3, "z→tangent failed: {zmap}");
        // Cumulative twist rolls the cross-section between segments.
        let x0 = xf[0].1 * Vector3::x();
        let x5 = xf[5].1 * Vector3::x();
        assert!((x0 - x5).norm() > 0.1, "twist should roll the cross-section");
    }
}
