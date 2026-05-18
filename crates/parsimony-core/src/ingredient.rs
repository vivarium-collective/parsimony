//! Ingredients — the molecular species being packed. v0.1 supports
//! `single_sphere` only; `multi_sphere`, `single_cube`, `single_cylinder`,
//! `mesh`, and `grow` (fibers) come later.

use serde::{Deserialize, Serialize};

use crate::recipe::PackingMode;

/// Stable handle for an ingredient within a [`Recipe`](crate::Recipe).
/// `u32` rather than `usize` to keep [`Placement`](crate::Placement)
/// compact.
pub type IngredientId = u32;

/// Shape representation used for collision testing. Phase 2 MVP carries
/// only a sphere radius; later variants add sphere trees, meshes, and
/// atomic models.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum IngredientShape {
    /// A single sphere of given radius.
    SingleSphere { radius: f32 },
}

impl IngredientShape {
    /// AABB-side enclosing extent of the ingredient at identity rotation.
    /// Used to construct broad-phase AABBs for the spatial index.
    pub fn enclosing_radius(&self) -> f32 {
        match self {
            IngredientShape::SingleSphere { radius } => *radius,
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
}
