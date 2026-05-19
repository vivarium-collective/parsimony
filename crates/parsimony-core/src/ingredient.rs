//! Ingredients — the molecular species being packed. v0.1 supports
//! `single_sphere` only; `multi_sphere`, `single_cube`, `single_cylinder`,
//! `mesh`, and `grow` (fibers) come later.

use nalgebra::{Point3, UnitQuaternion, Vector3};
use serde::{Deserialize, Serialize};

use crate::recipe::PackingMode;

fn default_principal_vector() -> Vector3<f32> {
    Vector3::new(0.0, 0.0, 1.0)
}

/// Stable handle for an ingredient within a [`Recipe`](crate::Recipe).
/// `u32` rather than `usize` to keep [`Placement`](crate::Placement)
/// compact.
pub type IngredientId = u32;

/// One sphere of a multi-sphere proxy: an offset relative to the
/// ingredient's local origin (transformed by the ingredient's rotation
/// on placement) plus a radius.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct ProxySphere {
    pub offset: Vector3<f32>,
    pub radius: f32,
}

/// Shape representation used for collision testing. cellPACK calls
/// this the "packing representation" — a small set of spheres that
/// approximate the ingredient's volume for fast tree-vs-tree overlap
/// tests. Real-world ingredients use 10–100 spheres; we support
/// arbitrary counts.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IngredientShape {
    /// A single sphere of given radius.
    SingleSphere { radius: f32 },
    /// A multi-sphere proxy — a list of `(offset, radius)` spheres
    /// rigidly attached to the ingredient's local frame.
    MultiSphere { spheres: Vec<ProxySphere> },
}

impl IngredientShape {
    /// Maximum world-space distance from the ingredient's center to
    /// any point on its surface, at any rotation. Used to construct
    /// broad-phase AABBs and bounding-sphere queries.
    pub fn enclosing_radius(&self) -> f32 {
        match self {
            IngredientShape::SingleSphere { radius } => *radius,
            IngredientShape::MultiSphere { spheres } => spheres
                .iter()
                .map(|s| s.offset.norm() + s.radius)
                .fold(0.0_f32, f32::max),
        }
    }

    /// True iff this ingredient's geometry actually depends on rotation
    /// (a single sphere does not; a multi-sphere with more than one
    /// proxy sphere does).
    pub fn needs_rotation(&self) -> bool {
        matches!(self, IngredientShape::MultiSphere { spheres } if spheres.len() > 1)
    }

    /// Iterate world-space `(center, radius)` for every proxy sphere of
    /// this ingredient instance, given the instance's `position` and
    /// `rotation`.
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
        }
    }
}

/// An ingredient species. Variants for conformational cycles
/// (ATP synthase E/T/O) and ligand-bound forms come later — see
/// `docs/parsimony-design.md` §5.2.1.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Ingredient {
    pub name: String,
    pub shape: IngredientShape,
    /// Display color (`[r, g, b]`, each in 0..=1). Passed through to output.
    pub color: [f32; 3],
    /// Max placement attempts per individual instance before giving up.
    pub jitter_attempts: u32,
    pub packing_mode: PackingMode,
    /// Direction (in ingredient-local space) that should align with the
    /// surface normal when this ingredient is placed in a Surface
    /// region. Default `(0, 0, 1)`. cellPACK convention.
    #[serde(default = "default_principal_vector")]
    pub principal_vector: Vector3<f32>,
}
