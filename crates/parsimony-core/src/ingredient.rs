//! Ingredients — the molecular species being packed.
//!
//! Shape representations:
//! - `SingleSphere` — one sphere
//! - `MultiSphere` — a sphere-tree (cellPACK's "packing representation")
//! - `Mesh` — a triangle mesh + sphere-tree proxies for fast broad-phase
//!
//! `Cube`, `Cylinder`, and `MultiCylinder` recipe types convert to
//! `MultiSphere` at recipe-load time via the [`shape_helpers`] sphere
//! tree generators — same downstream pipeline, just a different
//! geometric source.

use std::sync::Arc;

use nalgebra::{Matrix3, Point3, Rotation3, SymmetricEigen, UnitQuaternion, Vector3};
use parry3d::shape::TriMesh;
use serde::{Deserialize, Serialize};

use crate::recipe::PackingMode;

/// Stable handle for an ingredient within a [`Recipe`](crate::Recipe).
pub type IngredientId = u32;

/// One sphere of a multi-sphere proxy: an offset (in ingredient-local
/// space, rotated with the ingredient on placement) plus a radius.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProxySphere {
    pub offset: Vector3<f32>,
    pub radius: f32,
}

/// An origin-centred, principal-axis-aligned ellipsoid fitted to an
/// ingredient's geometry. `semi_axes` are the half-extents along the
/// local axes encoded by `rotation` (which maps ellipsoid-local axes
/// into ingredient-local space). Renderers use it as a cheap
/// anisotropic stand-in — a cigar for rods, a disc for plates — where
/// they'd otherwise draw a single enclosing sphere that over-claims
/// badly for non-spherical molecules.
#[derive(Debug, Clone, Copy)]
pub struct PrincipalEllipsoid {
    pub semi_axes: [f32; 3],
    pub rotation: UnitQuaternion<f32>,
}

/// Shape representation used for collision testing. cellPACK calls
/// this the "packing representation" — a small set of spheres that
/// approximate the ingredient's volume for fast tree-vs-tree overlap
/// tests.
#[derive(Debug, Clone)]
pub enum IngredientShape {
    /// A single sphere of given radius.
    SingleSphere { radius: f32 },
    /// A multi-sphere proxy.
    MultiSphere { spheres: Vec<ProxySphere> },
    /// A triangle mesh, with a sphere-tree proxy for fast collision.
    /// The full mesh is retained for later use (exact collision,
    /// rendering, mesh-surface sampling).
    Mesh {
        trimesh: Arc<TriMesh>,
        proxies: Vec<ProxySphere>,
    },
    /// A fiber: a polyline of bead centres (ingredient-local space) with
    /// a uniform bead radius — the chromosome, one long coarse-grained
    /// chain. Collision treats it as a chain of spheres.
    Fiber {
        points: Vec<Point3<f32>>,
        radius: f32,
    },
}

impl IngredientShape {
    /// Maximum world-space distance from the ingredient's centre to
    /// any point on its surface, at any rotation. Used to construct
    /// broad-phase AABBs and bounding-sphere queries.
    pub fn enclosing_radius(&self) -> f32 {
        match self {
            IngredientShape::SingleSphere { radius } => *radius,
            IngredientShape::MultiSphere { spheres } => spheres
                .iter()
                .map(|s| s.offset.norm() + s.radius)
                .fold(0.0_f32, f32::max),
            IngredientShape::Mesh { proxies, trimesh } => {
                // Use the max of (proxy bounding-sphere) and (vertex
                // bounding-sphere). Proxies undershoot the actual
                // mesh in two failure modes: (a) empty when the
                // marching-cubes surface doesn't enclose interior
                // voxels (open / thin meshes), and (b) too-small
                // when the proxy voxelization only captures a
                // sub-region of an elongated molecule. The vertex
                // bounding sphere is correct in both cases — it's
                // derived from the rendered geometry itself.
                let from_proxies = proxies
                    .iter()
                    .map(|s| s.offset.norm() + s.radius)
                    .fold(0.0_f32, f32::max);
                let from_vertices = trimesh
                    .vertices()
                    .iter()
                    .map(|v| v.coords.norm())
                    .fold(0.0_f32, f32::max);
                from_proxies.max(from_vertices)
            }
            IngredientShape::Fiber { points, radius } => points
                .iter()
                .map(|p| p.coords.norm() + radius)
                .fold(0.0_f32, f32::max),
        }
    }

    /// An origin-centred, principal-axis ellipsoid that snugly encloses
    /// this shape — a cheap anisotropic stand-in for renderers to draw
    /// as the placeholder / far-LOD proxy instead of a single enclosing
    /// sphere (which reads as a fat ball for elongated or flat
    /// molecules). `None` when a sphere is already faithful
    /// (`SingleSphere`) or when multi-sphere already renders its own
    /// shape-true union, or there aren't enough points to fit one.
    pub fn principal_ellipsoid(&self) -> Option<PrincipalEllipsoid> {
        match self {
            IngredientShape::SingleSphere { .. } => None,
            IngredientShape::MultiSphere { .. } => None,
            IngredientShape::Mesh { trimesh, .. } => {
                principal_ellipsoid_of(trimesh.vertices().iter().map(|v| v.coords))
            }
            IngredientShape::Fiber { .. } => None,
        }
    }

    /// True iff this ingredient's geometry actually depends on rotation
    /// (a single sphere does not; sphere-trees and meshes do).
    pub fn needs_rotation(&self) -> bool {
        match self {
            IngredientShape::SingleSphere { .. } => false,
            IngredientShape::MultiSphere { spheres } => spheres.len() > 1,
            IngredientShape::Mesh { .. } => true,
            IngredientShape::Fiber { points, .. } => points.len() > 1,
        }
    }

    /// Iterate world-space `(centre, radius)` for every proxy sphere
    /// of this ingredient instance.
    pub fn world_spheres<'a>(
        &'a self,
        position: Point3<f32>,
        rotation: UnitQuaternion<f32>,
    ) -> WorldSphereIter<'a> {
        WorldSphereIter {
            shape: self,
            position,
            rotation,
            index: 0,
        }
    }

    /// Construct a sphere-tree representation of an axis-aligned cube
    /// of given half-extents. Returns a `MultiSphere` with 8 spheres
    /// at the octant centres (±h/2 on each axis), radius = the
    /// smallest half-extent. Conservative coverage — the cube's
    /// corners extend slightly past the proxies, but the inscribed
    /// box of each octant is fully covered.
    pub fn cube(half_extents: Vector3<f32>) -> Self {
        IngredientShape::MultiSphere {
            spheres: shape_helpers::cube_proxies(half_extents),
        }
    }

    /// Construct a sphere-tree for a cylinder of given length and
    /// radius along the local Z axis. Returns a `MultiSphere` with a
    /// chain of overlapping spheres of `radius` spaced along the axis.
    pub fn cylinder(length: f32, radius: f32) -> Self {
        IngredientShape::MultiSphere {
            spheres: shape_helpers::cylinder_proxies(length, radius),
        }
    }

    /// Construct a sphere-tree from a chain of cylinder segments in
    /// the ingredient's local frame.
    pub fn multi_cylinder(segments: &[CylinderSegment]) -> Self {
        IngredientShape::MultiSphere {
            spheres: shape_helpers::multi_cylinder_proxies(segments),
        }
    }

    /// Construct a `Mesh` shape from a triangle mesh, with proxy
    /// spheres generated by voxelising the mesh interior at the given
    /// voxel size.
    pub fn mesh_with_voxelised_proxies(trimesh: Arc<TriMesh>, voxel_size: f32) -> Self {
        let proxies = shape_helpers::voxelise_mesh_to_proxies(&trimesh, voxel_size);
        IngredientShape::Mesh { trimesh, proxies }
    }
}

/// Fit an origin-centred oriented ellipsoid to a point cloud. The axes
/// are the eigenvectors of the centroid-relative covariance (the
/// principal directions); each semi-axis is the 99th-percentile
/// |projection onto that axis| measured from the ORIGIN — so the
/// ellipsoid is centred where the rendered geometry's origin is (the
/// same place the sphere proxy it replaces sits) and shrugs off the
/// stray outlier vertices marching-cubes surfaces sometimes leave far
/// from the body. Returns `None` for clouds too small to fit.
fn principal_ellipsoid_of(
    points: impl Iterator<Item = Vector3<f32>>,
) -> Option<PrincipalEllipsoid> {
    let pts: Vec<Vector3<f32>> = points.collect();
    let n = pts.len();
    if n < 4 {
        return None;
    }
    let inv_n = 1.0 / n as f32;

    let mut centroid = Vector3::zeros();
    for p in &pts {
        centroid += *p;
    }
    centroid *= inv_n;

    let mut cov = Matrix3::zeros();
    for p in &pts {
        let d = *p - centroid;
        cov += d * d.transpose();
    }
    cov *= inv_n;

    // Eigenvectors of a symmetric matrix are orthonormal, so `axes` is
    // orthogonal (det ±1). Flip one column if it came out left-handed
    // so `from_rotation_matrix` reads a rotation, not a reflection.
    let mut axes = SymmetricEigen::new(cov).eigenvectors;
    if axes.determinant() < 0.0 {
        let flipped = -axes.column(2).into_owned();
        axes.set_column(2, &flipped);
    }

    let cut = ((n as f32 * 0.99) as usize).min(n - 1);
    let mut semi_axes = [0.0_f32; 3];
    for k in 0..3 {
        let axis = axes.column(k).into_owned();
        let mut proj: Vec<f32> = pts.iter().map(|p| p.dot(&axis).abs()).collect();
        proj.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        // Clamp away from zero so a perfectly flat axis still yields a
        // (very thin) renderable ellipsoid rather than a degenerate one.
        semi_axes[k] = proj[cut].max(1e-3);
    }

    let rotation = UnitQuaternion::from_rotation_matrix(&Rotation3::from_matrix_unchecked(axes));
    Some(PrincipalEllipsoid { semi_axes, rotation })
}

pub struct WorldSphereIter<'a> {
    shape: &'a IngredientShape,
    position: Point3<f32>,
    rotation: UnitQuaternion<f32>,
    index: usize,
}

impl<'a> Iterator for WorldSphereIter<'a> {
    type Item = (Point3<f32>, f32);
    fn next(&mut self) -> Option<Self::Item> {
        match self.shape {
            IngredientShape::SingleSphere { radius } => {
                if self.index == 0 {
                    self.index = 1;
                    Some((self.position, *radius))
                } else {
                    None
                }
            }
            IngredientShape::MultiSphere { spheres } => {
                if self.index < spheres.len() {
                    let s = &spheres[self.index];
                    self.index += 1;
                    Some((self.position + self.rotation * s.offset, s.radius))
                } else {
                    None
                }
            }
            IngredientShape::Mesh { proxies, .. } => {
                if self.index < proxies.len() {
                    let s = &proxies[self.index];
                    self.index += 1;
                    Some((self.position + self.rotation * s.offset, s.radius))
                } else {
                    None
                }
            }
            IngredientShape::Fiber { points, radius } => {
                if self.index < points.len() {
                    let p = points[self.index];
                    self.index += 1;
                    Some((self.position + self.rotation * p.coords, *radius))
                } else {
                    None
                }
            }
        }
    }
}

/// One segment of a multi-cylinder ingredient: a capsule axis from
/// `start` to `end` (both in ingredient-local space) with a uniform
/// `radius`.
#[derive(Debug, Clone, Copy)]
pub struct CylinderSegment {
    pub start: Point3<f32>,
    pub end: Point3<f32>,
    pub radius: f32,
}

/// One level of detail for a mesh ingredient. `voxel_size` is the
/// nominal world-units resolution the LOD was generated at; the
/// viewer uses it to pick which LOD to fetch given a screen-space
/// pixel budget. Ordered coarse → fine in `Ingredient::mesh_lods`.
#[derive(Debug, Clone)]
pub struct MeshLod {
    pub url: String,
    pub voxel_size: f32,
}

/// An ingredient species.
#[derive(Debug, Clone)]
pub struct Ingredient {
    pub name: String,
    pub shape: IngredientShape,
    /// Display colour (`[r, g, b]`, each in 0..=1).
    pub color: [f32; 3],
    /// Max placement attempts per individual instance before giving up.
    pub jitter_attempts: u32,
    pub packing_mode: PackingMode,
    /// Direction (in ingredient-local space) that should align with
    /// the surface normal when this ingredient is placed in a Surface
    /// region. Default `(0, 0, 1)`. cellPACK convention.
    pub principal_vector: Vector3<f32>,
    /// For `IngredientShape::Mesh` ingredients: the LOD pyramid
    /// (sorted coarse → fine) that downstream renderers can fetch.
    /// Empty for non-mesh shapes. A single-LOD recipe lands here as
    /// a one-element vec.
    pub mesh_lods: Vec<MeshLod>,
    /// Optional ingredient name of a per-bead segment mesh. When set on a
    /// `MultiSphere` ingredient, the pack writer renders each instance as that
    /// mesh tiled along its bead chain (e.g. mRNA → a real RNA strand), the
    /// same idea as the chromosome's `dna_segment`. Pack/collision still use
    /// the multi-sphere proxy; this only affects rendering output.
    pub segment: Option<String>,
}

/// Sphere-tree generators for analytical shape primitives + a mesh
/// voxeliser. All in their own module to keep `IngredientShape`
/// declarations clean.
pub mod shape_helpers {
    use super::*;
    use parry3d::query::PointQuery;

    /// 2×2×2 sphere lattice covering a cube of half-extents `h`.
    /// Spheres sit at the eight octant centres `(±hx/2, ±hy/2, ±hz/2)`
    /// with radius `‖h‖/2` — half the cube's space-diagonal. With
    /// that radius each octant-sphere tangentially reaches both the
    /// origin and the matching cube corner, so the union fully
    /// covers the cube interior. The proxy's enclosing-sphere radius
    /// matches the cube's geometric enclosing radius (= the
    /// half-diagonal), so broad-phase queries don't over-claim
    /// outside the cube either. Per-pair sphere checks downstream
    /// remain cheap (at most 8×8 = 64 pairs between two cubes).
    pub fn cube_proxies(h: Vector3<f32>) -> Vec<ProxySphere> {
        let r = h.norm() * 0.5;
        let mut spheres = Vec::with_capacity(8);
        for sx in [-1.0_f32, 1.0] {
            for sy in [-1.0_f32, 1.0] {
                for sz in [-1.0_f32, 1.0] {
                    spheres.push(ProxySphere {
                        offset: Vector3::new(sx * h.x * 0.5, sy * h.y * 0.5, sz * h.z * 0.5),
                        radius: r,
                    });
                }
            }
        }
        spheres
    }

    /// Chain of overlapping spheres along the local Z axis for a
    /// cylinder of given length and radius. Sphere spacing equals
    /// `radius` so each pair overlaps by 50%.
    pub fn cylinder_proxies(length: f32, radius: f32) -> Vec<ProxySphere> {
        let half = length * 0.5;
        // Number of intervals along the axis; spacing = radius (50%
        // overlap). Always at least 2 spheres (endpoints) so a very
        // short cylinder still has interior coverage.
        let n = ((length / radius).ceil() as usize).max(1);
        let mut spheres = Vec::with_capacity(n + 1);
        for i in 0..=n {
            let t = i as f32 / n as f32;
            let z = -half + length * t;
            spheres.push(ProxySphere {
                offset: Vector3::new(0.0, 0.0, z),
                radius,
            });
        }
        spheres
    }

    /// Concatenated cylinder chain — one sphere-chain per segment.
    pub fn multi_cylinder_proxies(segments: &[CylinderSegment]) -> Vec<ProxySphere> {
        let mut spheres = Vec::new();
        for seg in segments {
            let dir = seg.end - seg.start;
            let length = dir.norm();
            if length < 1e-6 {
                spheres.push(ProxySphere {
                    offset: seg.start.coords,
                    radius: seg.radius,
                });
                continue;
            }
            let n = ((length / seg.radius).ceil() as usize).max(1);
            for i in 0..=n {
                let t = i as f32 / n as f32;
                let p = seg.start + dir * t;
                spheres.push(ProxySphere {
                    offset: p.coords,
                    radius: seg.radius,
                });
            }
        }
        spheres
    }

    /// Voxelise the mesh interior on a uniform grid of `voxel_size`
    /// cells, return one proxy sphere per interior cell (centre =
    /// cell centre, radius = `voxel_size · √3 / 2` so the sphere
    /// covers the cell volume). Grid is the mesh's local AABB.
    /// Requires the mesh to already be configured for in/out queries
    /// (`parsimony_spatial::prepare_trimesh_for_voxelize`).
    pub fn voxelise_mesh_to_proxies(trimesh: &TriMesh, voxel_size: f32) -> Vec<ProxySphere> {
        let aabb = trimesh.local_aabb();
        let min = aabb.mins;
        let max = aabb.maxs;
        let nx = (((max.x - min.x) / voxel_size).ceil() as usize).max(1);
        let ny = (((max.y - min.y) / voxel_size).ceil() as usize).max(1);
        let nz = (((max.z - min.z) / voxel_size).ceil() as usize).max(1);
        // Sphere radius covers the voxel diagonal so the union of
        // proxies covers the voxelised interior with no gaps.
        let r = voxel_size * 0.5 * 3.0_f32.sqrt();
        let mut spheres = Vec::new();
        for iz in 0..nz {
            for iy in 0..ny {
                for ix in 0..nx {
                    let p = Point3::new(
                        min.x + (ix as f32 + 0.5) * voxel_size,
                        min.y + (iy as f32 + 0.5) * voxel_size,
                        min.z + (iz as f32 + 0.5) * voxel_size,
                    );
                    if trimesh.contains_local_point(&p) {
                        spheres.push(ProxySphere {
                            offset: p.coords,
                            radius: r,
                        });
                    }
                }
            }
        }
        spheres
    }
}

/// Minimal Wavefront OBJ loader. Reads `v` (vertex) and `f` (face)
/// lines, ignoring textures/normals/groups. Triangulates polygonal
/// faces with a fan from the first vertex. Negative face indices
/// (relative to current vertex count) are supported per the OBJ spec.
pub mod obj {
    use std::path::Path;

    use nalgebra::Point3;
    use parry3d::shape::TriMesh;

    #[derive(Debug, thiserror::Error)]
    pub enum ObjError {
        #[error("io: {0}")]
        Io(#[from] std::io::Error),
        #[error("trimesh: {0}")]
        TriMesh(String),
        #[error("no triangles parsed from `{0}`")]
        Empty(String),
    }

    pub fn load_trimesh(path: &Path) -> Result<TriMesh, ObjError> {
        let text = std::fs::read_to_string(path)?;
        let mut vertices: Vec<Point3<f32>> = Vec::new();
        let mut indices: Vec<[u32; 3]> = Vec::new();
        for line in text.lines() {
            let line = line.trim();
            if line.starts_with("v ") {
                let parts: Vec<f32> = line[2..]
                    .split_whitespace()
                    .filter_map(|s| s.parse().ok())
                    .collect();
                if parts.len() >= 3 {
                    vertices.push(Point3::new(parts[0], parts[1], parts[2]));
                }
            } else if line.starts_with("f ") {
                let face: Vec<u32> = line[2..]
                    .split_whitespace()
                    .filter_map(|tok| {
                        let v_str = tok.split('/').next()?;
                        let i: i32 = v_str.parse().ok()?;
                        let idx = if i > 0 {
                            (i - 1) as i64
                        } else {
                            // OBJ negative indices reference from the end
                            // of the current vertex list.
                            vertices.len() as i64 + i as i64
                        };
                        if idx >= 0 && (idx as usize) < vertices.len() {
                            Some(idx as u32)
                        } else {
                            None
                        }
                    })
                    .collect();
                if face.len() < 3 {
                    continue;
                }
                // Fan triangulation.
                for i in 1..face.len() - 1 {
                    indices.push([face[0], face[i], face[i + 1]]);
                }
            }
        }
        if indices.is_empty() {
            return Err(ObjError::Empty(path.display().to_string()));
        }
        TriMesh::new(vertices, indices)
            .map_err(|e| ObjError::TriMesh(format!("{e:?}")))
    }
}

#[cfg(test)]
mod ellipsoid_tests {
    use super::*;

    #[test]
    fn fit_is_anisotropic_and_axis_aligned() {
        // A cloud stretched ~10:1 along local X, thin in Y/Z.
        let mut pts = Vec::new();
        for i in 0..400 {
            let t = i as f32 / 399.0;
            pts.push(Vector3::new(
                (t - 0.5) * 20.0,                 // x in [-10, 10]
                ((i % 7) as f32 - 3.0) * 0.2,     // y in ~[-0.6, 0.6]
                ((i % 5) as f32 - 2.0) * 0.2,     // z in ~[-0.4, 0.4]
            ));
        }
        let e = principal_ellipsoid_of(pts.into_iter()).expect("fit");

        let max = e.semi_axes.iter().copied().fold(0.0_f32, f32::max);
        let min = e.semi_axes.iter().copied().fold(f32::INFINITY, f32::min);
        assert!(max > 8.0, "long semi-axis should be ~10, got {max}");
        assert!(min < 2.0, "short semi-axes should be thin, got {min}");

        // The long axis must point along ±X in ingredient-local space.
        let k = (0..3)
            .max_by(|a, b| e.semi_axes[*a].partial_cmp(&e.semi_axes[*b]).unwrap())
            .unwrap();
        let local = Vector3::new((k == 0) as i32 as f32, (k == 1) as i32 as f32, (k == 2) as i32 as f32);
        let world = e.rotation * local;
        assert!(world.x.abs() > 0.98, "long axis should align with ±X, got {world:?}");
    }

    #[test]
    fn none_for_tiny_cloud() {
        let pts = vec![Vector3::new(0.0, 0.0, 0.0), Vector3::new(1.0, 0.0, 0.0)];
        assert!(principal_ellipsoid_of(pts.into_iter()).is_none());
    }
}
