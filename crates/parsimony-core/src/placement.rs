//! Placement, snapshot, and incremental-op types — the data model
//! for a packed configuration. See `docs/parsimony-design.md` §§ 5.4,
//! 5.5, 10.

use nalgebra::{Point3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::compartment::CompartmentId;
use crate::ingredient::IngredientId;

/// Stable handle for an ingredient variant within an ingredient family.
/// Default 0 = canonical form (see design doc §5.2.1).
pub type VariantId = u16;

/// One placed instance.
#[derive(Debug, Clone, Copy, Serialize, Deserialize)]
pub struct Placement {
    pub instance_uid: u64,
    pub ingredient_id: IngredientId,
    pub variant_id: VariantId,
    pub compartment_id: CompartmentId,
    pub position: Point3<f32>,
    pub rotation: UnitQuaternion<f32>,
}

/// Snapshot of a packed configuration — the load-bearing serialized
/// form. The spatial index is reconstructable from this (per design
/// doc §5.5).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub recipe_name: String,
    pub seed: u64,
    pub placements: Vec<Placement>,
}

impl Snapshot {
    pub fn new(recipe_name: String, seed: u64) -> Self {
        Self {
            recipe_name,
            seed,
            placements: Vec::new(),
        }
    }
}

// ---------- the state API: incremental ops (design doc §10.1) ----------

/// One atomic edit to a [`Snapshot`]. Applied via `apply_batch`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum Op {
    Insert {
        placement: Placement,
    },
    Remove {
        uid: u64,
    },
    Move {
        uid: u64,
        position: Point3<f32>,
        rotation: UnitQuaternion<f32>,
    },
    Replace {
        uid: u64,
        change: ReplaceChange,
    },
}

/// In-place identity / variant / pose change. See design doc §10.1
/// for the ATP-synthase example.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ReplaceChange {
    Variant {
        variant_id: VariantId,
    },
    Identity {
        ingredient_id: IngredientId,
        variant_id: VariantId,
    },
    Both {
        ingredient_id: IngredientId,
        variant_id: VariantId,
        position: Point3<f32>,
        rotation: UnitQuaternion<f32>,
    },
}
