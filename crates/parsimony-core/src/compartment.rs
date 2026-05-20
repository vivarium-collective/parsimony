//! Compartments — the bounded volumes ingredients are placed within.
//! v0.2 carries analytical kinds ([`Box`](CompartmentKind::Box),
//! [`Sphere`](CompartmentKind::Sphere),
//! [`Capsule`](CompartmentKind::Capsule)) plus mesh
//! ([`Mesh`](CompartmentKind::Mesh) via parry3d), with parent/children
//! pointers for nested-compartment recipes. See
//! `docs/parsimony-design.md` §7.

use std::f32::consts::PI;

use nalgebra::{Point3, UnitQuaternion, Vector3};
use rand::Rng;
use serde::{Deserialize, Serialize};

use parsimony_spatial::Aabb;

/// Stable handle for a compartment within a [`Recipe`](crate::Recipe).
pub type CompartmentId = u32;

/// Geometric kind of a compartment.
// Mesh variant is much larger than analytical variants (TriMesh carries
// vertex/index/BVH storage). Heap-allocate it via Box so the enum stays
// small.
#[derive(Debug, Clone)]
pub enum CompartmentKind {
    Box(Aabb),
    Sphere {
        center: Point3<f32>,
        radius: f32,
    },
    /// Hemisphere-capped cylinder along the axis `a..b`, with cap
    /// radius `radius`. Matches the rod-shape used for *E. coli*.
    Capsule {
        a: Point3<f32>,
        b: Point3<f32>,
        radius: f32,
    },
    /// Closed triangle mesh. The `parry3d::TriMesh` is expected to
    /// already carry `TriMeshFlags::ORIENTED | DELETE_DEGENERATE` so
    /// `contains_local_point` works (see `parsimony_spatial::prepare_trimesh_for_voxelize`).
    Mesh(Box<MeshCompartment>),
}

#[derive(Debug, Clone)]
pub struct MeshCompartment {
    pub trimesh: parry3d::shape::TriMesh,
    pub aabb: Aabb,
}

/// Implements a manual `Serialize`/`Deserialize` that round-trips the
/// analytical kinds and reports an error for `Mesh` (which carries a
/// non-serializable parry3d structure). Mesh compartments are
/// materialized on recipe load from a file path, not snapshotted.
mod kind_serde {
    use super::*;
    use serde::de::Error as _;

    #[derive(Serialize, Deserialize)]
    #[serde(tag = "kind", rename_all = "snake_case")]
    enum SerdeKind {
        Box {
            min: [f32; 3],
            max: [f32; 3],
        },
        Sphere {
            center: [f32; 3],
            radius: f32,
        },
        Capsule {
            a: [f32; 3],
            b: [f32; 3],
            radius: f32,
        },
    }

    impl Serialize for CompartmentKind {
        fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
            let v = match self {
                CompartmentKind::Box(b) => SerdeKind::Box {
                    min: [b.min.x, b.min.y, b.min.z],
                    max: [b.max.x, b.max.y, b.max.z],
                },
                CompartmentKind::Sphere { center, radius } => SerdeKind::Sphere {
                    center: [center.x, center.y, center.z],
                    radius: *radius,
                },
                CompartmentKind::Capsule { a, b, radius } => SerdeKind::Capsule {
                    a: [a.x, a.y, a.z],
                    b: [b.x, b.y, b.z],
                    radius: *radius,
                },
                CompartmentKind::Mesh(_) => {
                    return Err(serde::ser::Error::custom(
                        "Mesh compartments are not directly serializable; recreate from recipe",
                    ));
                }
            };
            v.serialize(s)
        }
    }

    impl<'de> Deserialize<'de> for CompartmentKind {
        fn deserialize<D: serde::Deserializer<'de>>(d: D) -> Result<Self, D::Error> {
            let v = SerdeKind::deserialize(d)?;
            Ok(match v {
                SerdeKind::Box { min, max } => CompartmentKind::Box(Aabb::new(
                    Point3::new(min[0], min[1], min[2]),
                    Point3::new(max[0], max[1], max[2]),
                )),
                SerdeKind::Sphere { center, radius } => CompartmentKind::Sphere {
                    center: Point3::new(center[0], center[1], center[2]),
                    radius,
                },
                SerdeKind::Capsule { a, b, radius } => {
                    if radius <= 0.0 {
                        return Err(D::Error::custom("capsule radius must be positive"));
                    }
                    CompartmentKind::Capsule {
                        a: Point3::new(a[0], a[1], a[2]),
                        b: Point3::new(b[0], b[1], b[2]),
                        radius,
                    }
                }
            })
        }
    }
}

impl CompartmentKind {
    /// Signed distance from `p` to the compartment boundary. Positive
    /// when `p` is inside (and equal to the distance to the nearest
    /// boundary point), negative when outside, zero on the boundary.
    /// Used both for sphere-fit checks (`fits_sphere`) and to bound
    /// jitter in the placer (jittering by more than this distance
    /// would push the sphere outside).
    pub fn signed_distance(&self, p: Point3<f32>) -> f32 {
        match self {
            CompartmentKind::Box(aabb) => {
                // Per-axis "distance to the nearer face" can be negative
                // when `p` lies outside. Taking the min yields the
                // signed distance for inside points (positive = min
                // face distance) and a sound negative bound for outside
                // points (any axis being out makes the min negative).
                let dx = (p.x - aabb.min.x).min(aabb.max.x - p.x);
                let dy = (p.y - aabb.min.y).min(aabb.max.y - p.y);
                let dz = (p.z - aabb.min.z).min(aabb.max.z - p.z);
                dx.min(dy).min(dz)
            }
            CompartmentKind::Sphere { center, radius } => {
                radius - (p - center).norm()
            }
            CompartmentKind::Capsule { a, b, radius } => {
                -capsule_signed_distance(p, *a, *b, *radius)
            }
            CompartmentKind::Mesh(m) => {
                use parry3d::query::PointQuery;
                let proj = m.trimesh.project_local_point(&p, true);
                let d = (p - proj.point).norm();
                if m.trimesh.contains_local_point(&p) {
                    d
                } else {
                    -d
                }
            }
        }
    }

    /// True iff a sphere of radius `r` centred at `p` is fully inside
    /// the compartment.
    pub fn fits_sphere(&self, p: Point3<f32>, r: f32) -> bool {
        self.signed_distance(p) >= r
    }

    /// True iff `p` is inside the compartment.
    pub fn contains(&self, p: Point3<f32>) -> bool {
        match self {
            CompartmentKind::Box(aabb) => aabb.contains_point(p),
            CompartmentKind::Sphere { center, radius } => {
                (p - center).norm_squared() <= radius * radius
            }
            CompartmentKind::Capsule { a, b, radius } => {
                capsule_signed_distance(p, *a, *b, *radius) <= 0.0
            }
            CompartmentKind::Mesh(m) => {
                use parry3d::query::PointQuery;
                m.trimesh.contains_local_point(&p)
            }
        }
    }

    pub fn sample_interior<R: Rng>(&self, rng: &mut R) -> Point3<f32> {
        match self {
            CompartmentKind::Box(aabb) => Point3::new(
                rng.gen_range(aabb.min.x..aabb.max.x),
                rng.gen_range(aabb.min.y..aabb.max.y),
                rng.gen_range(aabb.min.z..aabb.max.z),
            ),
            CompartmentKind::Sphere { center, radius } => loop {
                let x: f32 = rng.gen_range(-1.0..1.0);
                let y: f32 = rng.gen_range(-1.0..1.0);
                let z: f32 = rng.gen_range(-1.0..1.0);
                if x * x + y * y + z * z <= 1.0 {
                    return Point3::new(
                        center.x + x * radius,
                        center.y + y * radius,
                        center.z + z * radius,
                    );
                }
            },
            CompartmentKind::Capsule { a, b, radius } => sample_capsule_interior(*a, *b, *radius, rng),
            CompartmentKind::Mesh(m) => loop {
                let p = Point3::new(
                    rng.gen_range(m.aabb.min.x..m.aabb.max.x),
                    rng.gen_range(m.aabb.min.y..m.aabb.max.y),
                    rng.gen_range(m.aabb.min.z..m.aabb.max.z),
                );
                if self.contains(p) {
                    return p;
                }
            },
        }
    }

    /// Sample a random point such that a sphere of radius `r` placed
    /// at that point is fully inside the compartment. Returns `None`
    /// if no such point exists.
    pub fn sample_interior_for_sphere<R: Rng>(
        &self,
        r: f32,
        rng: &mut R,
    ) -> Option<Point3<f32>> {
        match self {
            CompartmentKind::Box(aabb) => {
                if aabb.max.x - aabb.min.x <= 2.0 * r
                    || aabb.max.y - aabb.min.y <= 2.0 * r
                    || aabb.max.z - aabb.min.z <= 2.0 * r
                {
                    return None;
                }
                Some(Point3::new(
                    rng.gen_range((aabb.min.x + r)..(aabb.max.x - r)),
                    rng.gen_range((aabb.min.y + r)..(aabb.max.y - r)),
                    rng.gen_range((aabb.min.z + r)..(aabb.max.z - r)),
                ))
            }
            CompartmentKind::Sphere { center, radius } => {
                if r >= *radius {
                    return None;
                }
                let r_inset = radius - r;
                loop {
                    let x: f32 = rng.gen_range(-1.0..1.0);
                    let y: f32 = rng.gen_range(-1.0..1.0);
                    let z: f32 = rng.gen_range(-1.0..1.0);
                    if x * x + y * y + z * z <= 1.0 {
                        return Some(Point3::new(
                            center.x + x * r_inset,
                            center.y + y * r_inset,
                            center.z + z * r_inset,
                        ));
                    }
                }
            }
            CompartmentKind::Capsule { a, b, radius } => {
                if r >= *radius {
                    return None;
                }
                Some(sample_capsule_interior(*a, *b, radius - r, rng))
            }
            CompartmentKind::Mesh(m) => {
                for _ in 0..64 {
                    let p = Point3::new(
                        rng.gen_range((m.aabb.min.x + r)..(m.aabb.max.x - r)),
                        rng.gen_range((m.aabb.min.y + r)..(m.aabb.max.y - r)),
                        rng.gen_range((m.aabb.min.z + r)..(m.aabb.max.z - r)),
                    );
                    if self.contains(p) {
                        return Some(p);
                    }
                }
                None
            }
        }
    }

    /// Sample a point on the compartment's surface with the outward
    /// normal at that point.
    pub fn sample_surface<R: Rng>(&self, rng: &mut R) -> (Point3<f32>, Vector3<f32>) {
        match self {
            CompartmentKind::Box(aabb) => {
                let e = aabb.extents();
                // Area-weighted face pick.
                let face_areas = [
                    e.y * e.z, // x faces (x2)
                    e.x * e.z, // y faces
                    e.x * e.y, // z faces
                ];
                let total = 2.0 * (face_areas[0] + face_areas[1] + face_areas[2]);
                let pick = rng.gen_range(0.0..total);
                let (axis, sign) = pick_face(pick, face_areas);
                let min = [aabb.min.x, aabb.min.y, aabb.min.z];
                let max = [aabb.max.x, aabb.max.y, aabb.max.z];
                let face_coord = if sign > 0.0 { max[axis] } else { min[axis] };
                let other0 = (axis + 1) % 3;
                let other1 = (axis + 2) % 3;
                let p0 = rng.gen_range(min[other0]..max[other0]);
                let p1 = rng.gen_range(min[other1]..max[other1]);
                let mut pt = [0.0; 3];
                pt[axis] = face_coord;
                pt[other0] = p0;
                pt[other1] = p1;
                let mut n = [0.0; 3];
                n[axis] = sign;
                (
                    Point3::new(pt[0], pt[1], pt[2]),
                    Vector3::new(n[0], n[1], n[2]),
                )
            }
            CompartmentKind::Sphere { center, radius } => {
                let n = sample_unit_sphere(rng);
                (center + n * *radius, n)
            }
            CompartmentKind::Capsule { a, b, radius } => sample_capsule_surface(*a, *b, *radius, rng),
            CompartmentKind::Mesh(m) => sample_mesh_surface(&m.trimesh, rng),
        }
    }

    /// Deterministic even ("Fibonacci sphere") surface point + outward
    /// normal: the `i`-th of `n` points spread quasi-uniformly over the
    /// surface. Tiled surface placement (e.g. a lipid bilayer) uses this
    /// instead of random `sample_surface` + collision rejection, which
    /// blows up for a dense regular layer. Returns `None` for kinds
    /// without an analytic even tiling; callers fall back to random.
    pub fn surface_point_fibonacci<R: Rng>(
        &self,
        i: u64,
        n: u64,
        rng: &mut R,
    ) -> Option<(Point3<f32>, Vector3<f32>)> {
        match self {
            CompartmentKind::Sphere { center, radius } => {
                let nf = n.max(1) as f32;
                let ifl = i as f32;
                let y = 1.0 - 2.0 * (ifl + 0.5) / nf; // walk +1 → −1
                let r = (1.0 - y * y).max(0.0).sqrt();
                let golden = std::f32::consts::PI * (3.0 - 5.0_f32.sqrt());
                let theta = golden * ifl;
                let mut dir = Vector3::new(theta.cos() * r, y, theta.sin() * r);
                // Tangential jitter dissolves the regular Fibonacci spiral
                // into an organic (blue-noise-ish) distribution — still
                // even, but without the obviously patterned spiral lines.
                let spacing = 3.54 / nf.sqrt(); // ~arc spacing on unit sphere
                let amp = 0.5 * spacing;
                let helper = if dir.x.abs() < 0.9 {
                    Vector3::x()
                } else {
                    Vector3::y()
                };
                let t1 = (helper - dir * dir.dot(&helper)).normalize();
                let t2 = dir.cross(&t1);
                dir = (dir + t1 * rng.gen_range(-amp..amp) + t2 * rng.gen_range(-amp..amp))
                    .normalize();
                Some((center + dir * *radius, dir))
            }
            _ => None,
        }
    }

    pub fn aabb(&self) -> Aabb {
        match self {
            CompartmentKind::Box(aabb) => *aabb,
            CompartmentKind::Sphere { center, radius } => Aabb::from_sphere(*center, *radius),
            CompartmentKind::Capsule { a, b, radius } => Aabb::new(
                Point3::new(
                    a.x.min(b.x) - radius,
                    a.y.min(b.y) - radius,
                    a.z.min(b.z) - radius,
                ),
                Point3::new(
                    a.x.max(b.x) + radius,
                    a.y.max(b.y) + radius,
                    a.z.max(b.z) + radius,
                ),
            ),
            CompartmentKind::Mesh(m) => m.aabb,
        }
    }

    pub fn volume(&self) -> f32 {
        match self {
            CompartmentKind::Box(aabb) => {
                let e = aabb.extents();
                e.x * e.y * e.z
            }
            CompartmentKind::Sphere { radius, .. } => (4.0 / 3.0) * PI * radius.powi(3),
            CompartmentKind::Capsule { a, b, radius } => {
                let h = (b - a).norm();
                PI * radius.powi(2) * h + (4.0 / 3.0) * PI * radius.powi(3)
            }
            CompartmentKind::Mesh(m) => {
                // Approximation: AABB volume. Phase 3+ can compute mesh volume properly.
                let e = m.aabb.extents();
                e.x * e.y * e.z
            }
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compartment {
    pub name: String,
    pub kind: CompartmentKind,
    /// Parent compartment id (or `None` for the root).
    #[serde(default)]
    pub parent: Option<CompartmentId>,
    /// Child compartment ids — placements with interior region in the
    /// parent must exclude these.
    #[serde(default)]
    pub children: Vec<CompartmentId>,
}

// ---------- analytical helpers ----------

fn pick_face(pick: f32, face_areas: [f32; 3]) -> (usize, f32) {
    let mut acc = 0.0;
    for (axis, area) in face_areas.iter().enumerate() {
        acc += area;
        if pick < acc {
            return (axis, -1.0);
        }
        acc += area;
        if pick < acc {
            return (axis, 1.0);
        }
    }
    (2, 1.0)
}

fn sample_unit_sphere<R: Rng>(rng: &mut R) -> Vector3<f32> {
    loop {
        let x: f32 = rng.gen_range(-1.0..1.0);
        let y: f32 = rng.gen_range(-1.0..1.0);
        let z: f32 = rng.gen_range(-1.0..1.0);
        let r2 = x * x + y * y + z * z;
        if r2 > 0.0 && r2 <= 1.0 {
            let r = r2.sqrt();
            return Vector3::new(x / r, y / r, z / r);
        }
    }
}

/// Signed distance from `p` to capsule (axis `a..b`, cap radius `r`).
/// Negative inside, positive outside.
fn capsule_signed_distance(p: Point3<f32>, a: Point3<f32>, b: Point3<f32>, r: f32) -> f32 {
    let ab = b - a;
    let ap = p - a;
    let h = ab.dot(&ap) / ab.norm_squared();
    let h = h.clamp(0.0, 1.0);
    let closest = a + ab * h;
    (p - closest).norm() - r
}

fn sample_capsule_interior<R: Rng>(
    a: Point3<f32>,
    b: Point3<f32>,
    radius: f32,
    rng: &mut R,
) -> Point3<f32> {
    // Capsule = cylinder body + two hemispherical caps. Sample by
    // volume-weighted region pick, then uniform inside the picked region.
    let h = (b - a).norm();
    let cyl_v = PI * radius * radius * h;
    let cap_v = (4.0 / 3.0) * PI * radius.powi(3);
    let total = cyl_v + cap_v;
    let pick = rng.gen_range(0.0..total);

    let axis = (b - a).normalize();
    // Build an orthonormal basis (axis, e1, e2).
    let arb = if axis.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let e1 = axis.cross(&arb).normalize();
    let e2 = axis.cross(&e1);

    if pick < cyl_v {
        // Uniform in the cylinder. Pick a point in a disk of radius `radius`,
        // and a height along axis.
        let (rx, ry) = sample_disk(radius, rng);
        let t: f32 = rng.gen_range(0.0..h);
        a + axis * t + e1 * rx + e2 * ry
    } else {
        // Uniform in the union of two hemispheres = equivalent to a single
        // full ball of `radius`. Decide which cap by axis-coordinate sign.
        let p_local = sample_unit_ball(rng) * radius;
        let t_axis = axis.dot(&p_local);
        let centre = if t_axis >= 0.0 { b } else { a };
        centre + p_local
    }
}

fn sample_unit_ball<R: Rng>(rng: &mut R) -> Vector3<f32> {
    loop {
        let x: f32 = rng.gen_range(-1.0..1.0);
        let y: f32 = rng.gen_range(-1.0..1.0);
        let z: f32 = rng.gen_range(-1.0..1.0);
        if x * x + y * y + z * z <= 1.0 {
            return Vector3::new(x, y, z);
        }
    }
}

fn sample_disk<R: Rng>(radius: f32, rng: &mut R) -> (f32, f32) {
    loop {
        let x: f32 = rng.gen_range(-1.0..1.0);
        let y: f32 = rng.gen_range(-1.0..1.0);
        if x * x + y * y <= 1.0 {
            return (x * radius, y * radius);
        }
    }
}

fn sample_capsule_surface<R: Rng>(
    a: Point3<f32>,
    b: Point3<f32>,
    radius: f32,
    rng: &mut R,
) -> (Point3<f32>, Vector3<f32>) {
    let h = (b - a).norm();
    let cyl_area = 2.0 * PI * radius * h;
    let cap_area = 4.0 * PI * radius * radius;
    let total = cyl_area + cap_area;
    let pick = rng.gen_range(0.0..total);
    let axis = (b - a).normalize();
    let arb = if axis.x.abs() < 0.9 {
        Vector3::new(1.0, 0.0, 0.0)
    } else {
        Vector3::new(0.0, 1.0, 0.0)
    };
    let e1 = axis.cross(&arb).normalize();
    let e2 = axis.cross(&e1);

    if pick < cyl_area {
        // Sample on cylindrical band.
        let theta = rng.gen_range(0.0..(2.0 * PI));
        let t = rng.gen_range(0.0..h);
        let n = e1 * theta.cos() + e2 * theta.sin();
        let p = a + axis * t + n * radius;
        (p, n)
    } else {
        // Sample on one of the hemispherical caps.
        let n = sample_unit_sphere(rng);
        let centre = if axis.dot(&n) >= 0.0 { b } else { a };
        let p = centre + n * radius;
        (p, n)
    }
}

fn sample_mesh_surface<R: Rng>(
    mesh: &parry3d::shape::TriMesh,
    rng: &mut R,
) -> (Point3<f32>, Vector3<f32>) {
    // Area-weighted triangle pick, then barycentric sample inside.
    let vertices = mesh.vertices();
    let indices = mesh.indices();
    let mut total_area = 0.0;
    let mut prefix: Vec<f32> = Vec::with_capacity(indices.len());
    for tri in indices {
        let v0 = vertices[tri[0] as usize];
        let v1 = vertices[tri[1] as usize];
        let v2 = vertices[tri[2] as usize];
        let e1 = v1 - v0;
        let e2 = v2 - v0;
        let area = e1.cross(&e2).norm() * 0.5;
        total_area += area;
        prefix.push(total_area);
    }
    let pick = rng.gen_range(0.0..total_area);
    let tri_idx = prefix.partition_point(|&p| p < pick);
    let tri_idx = tri_idx.min(indices.len() - 1);
    let tri = indices[tri_idx];
    let v0 = vertices[tri[0] as usize];
    let v1 = vertices[tri[1] as usize];
    let v2 = vertices[tri[2] as usize];
    let e1 = v1 - v0;
    let e2 = v2 - v0;
    // Barycentric uniform sample.
    let r1: f32 = rng.gen_range(0.0..1.0);
    let r2: f32 = rng.gen_range(0.0..1.0);
    let (u, v) = if r1 + r2 > 1.0 {
        (1.0 - r1, 1.0 - r2)
    } else {
        (r1, r2)
    };
    let p = v0 + e1 * u + e2 * v;
    let n = e1.cross(&e2).normalize();
    (p, n)
}

/// Rotation that aligns the ingredient's `principal_vector` with a
/// target surface `normal`. Returns identity when `principal_vector`
/// is already (anti-)aligned (within `1e-6`).
pub fn align_to_normal(
    principal_vector: Vector3<f32>,
    normal: Vector3<f32>,
) -> UnitQuaternion<f32> {
    let from = principal_vector.normalize();
    let to = normal.normalize();
    UnitQuaternion::rotation_between(&from, &to).unwrap_or(UnitQuaternion::identity())
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn box_contains_point() {
        let b = CompartmentKind::Box(Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 10.0),
        ));
        assert!(b.contains(Point3::new(5.0, 5.0, 5.0)));
        assert!(!b.contains(Point3::new(-1.0, 5.0, 5.0)));
    }

    #[test]
    fn sphere_contains_point() {
        let s = CompartmentKind::Sphere {
            center: Point3::origin(),
            radius: 10.0,
        };
        assert!(s.contains(Point3::new(0.0, 0.0, 0.0)));
        assert!(s.contains(Point3::new(5.0, 5.0, 5.0)));
        assert!(!s.contains(Point3::new(11.0, 0.0, 0.0)));
    }

    #[test]
    fn capsule_contains_point() {
        let c = CompartmentKind::Capsule {
            a: Point3::new(-10.0, 0.0, 0.0),
            b: Point3::new(10.0, 0.0, 0.0),
            radius: 5.0,
        };
        // along axis
        assert!(c.contains(Point3::new(0.0, 0.0, 0.0)));
        // inside one cap
        assert!(c.contains(Point3::new(13.0, 1.0, 0.0)));
        // outside cylinder body
        assert!(!c.contains(Point3::new(0.0, 6.0, 0.0)));
        // outside cap
        assert!(!c.contains(Point3::new(16.0, 0.0, 0.0)));
    }

    #[test]
    fn sample_capsule_interior_stays_inside() {
        let c = CompartmentKind::Capsule {
            a: Point3::new(-50.0, 0.0, 0.0),
            b: Point3::new(50.0, 0.0, 0.0),
            radius: 20.0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        for _ in 0..200 {
            let p = c.sample_interior(&mut rng);
            assert!(c.contains(p), "interior sample outside capsule: {:?}", p);
        }
    }

    #[test]
    fn sample_sphere_surface_on_surface() {
        let s = CompartmentKind::Sphere {
            center: Point3::origin(),
            radius: 10.0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xACE5);
        for _ in 0..100 {
            let (p, n) = s.sample_surface(&mut rng);
            let d = p.coords.norm();
            assert!((d - 10.0).abs() < 1e-4, "surface point not on sphere: |p|={d}");
            // Normal should be radially outward.
            let dot = n.dot(&p.coords.normalize());
            assert!(dot > 0.999, "surface normal not radial: dot={dot}");
        }
    }

    #[test]
    fn align_to_normal_aligns() {
        let pv = Vector3::new(0.0, 0.0, 1.0);
        let n = Vector3::new(1.0, 0.0, 0.0);
        let q = align_to_normal(pv, n);
        let rotated = q * pv;
        assert!((rotated - n).norm() < 1e-5);
    }

    #[test]
    fn sample_capsule_surface_on_surface() {
        let c = CompartmentKind::Capsule {
            a: Point3::new(-50.0, 0.0, 0.0),
            b: Point3::new(50.0, 0.0, 0.0),
            radius: 20.0,
        };
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFADE);
        for _ in 0..100 {
            let (p, _n) = c.sample_surface(&mut rng);
            // Signed distance to capsule surface should be ~0.
            let sd = capsule_signed_distance(p, c_a(&c), c_b(&c), c_r(&c));
            assert!(sd.abs() < 1e-3, "surface point not on capsule: sd={sd}");
        }
    }

    fn c_a(c: &CompartmentKind) -> Point3<f32> {
        if let CompartmentKind::Capsule { a, .. } = c {
            *a
        } else {
            panic!()
        }
    }
    fn c_b(c: &CompartmentKind) -> Point3<f32> {
        if let CompartmentKind::Capsule { b, .. } = c {
            *b
        } else {
            panic!()
        }
    }
    fn c_r(c: &CompartmentKind) -> f32 {
        if let CompartmentKind::Capsule { radius, .. } = c {
            *radius
        } else {
            panic!()
        }
    }

    #[test]
    fn volumes_are_correct() {
        let b = CompartmentKind::Box(Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 10.0),
        ));
        assert!((b.volume() - 1000.0).abs() < 1e-3);
        let s = CompartmentKind::Sphere {
            center: Point3::origin(),
            radius: 10.0,
        };
        let expected = (4.0 / 3.0) * std::f32::consts::PI * 1000.0;
        assert!((s.volume() - expected).abs() < 1.0);
    }

    #[test]
    fn sample_interior_for_sphere_stays_inside() {
        let b = CompartmentKind::Box(Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(100.0, 100.0, 100.0),
        ));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        for _ in 0..100 {
            let p = b.sample_interior_for_sphere(10.0, &mut rng).unwrap();
            assert!(p.x >= 10.0 && p.x <= 90.0);
            assert!(p.y >= 10.0 && p.y <= 90.0);
            assert!(p.z >= 10.0 && p.z <= 90.0);
        }
    }

    #[test]
    fn sample_interior_for_sphere_too_big_returns_none() {
        let b = CompartmentKind::Box(Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 10.0),
        ));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        assert!(b.sample_interior_for_sphere(20.0, &mut rng).is_none());
    }
}
