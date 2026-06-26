//! Placement, snapshot, and incremental-op types — the data model
//! for a packed configuration. See `docs/parsimony-design.md` §§ 5.4,
//! 5.5, 10.

use nalgebra::{Point3, UnitQuaternion};
use serde::{Deserialize, Serialize};

use crate::compartment::CompartmentId;
use crate::ingredient::IngredientId;

/// A single confined nascent-RNA strand produced by `place_chromosome`.
/// Points are in the same center-relative frame as `Chromosome::strands`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct RnaStrand {
    /// Bead positions, center-relative.
    pub points: Vec<Point3<f32>>,
    /// `true` = mRNA; `false` = other RNA class (rRNA, tRNA, …).
    pub is_mrna: bool,
}

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
/// A generated chromosome (genome) fiber: a coarse-grained bead path
/// from the chain generator, carried on the snapshot for the writer to
/// emit as a `fiber` ingredient. Points are relative to `center` (the
/// cell compartment centre it was generated in).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Chromosome {
    pub center: Point3<f32>,
    pub radius: f32,
    pub color: [f32; 3],
    /// Primary strand (back-compat / smooth-tube fallback). When `strands`
    /// is non-empty it holds the same path as `strands[0]`.
    pub points: Vec<Point3<f32>>,
    /// All DNA strands across all chromosome instances, `center`-relative.
    /// A non-replicating cell has one strand per chromosome; a replicating
    /// one adds a sister strand over the replicated (theta) bubble. The
    /// writer tiles dsDNA segments along each strand independently.
    #[serde(default)]
    pub strands: Vec<Vec<Point3<f32>>>,
    /// Replication-fork positions (`center`-relative) — the Y-junctions where
    /// a sister strand rejoins the main genome. Two per replicating chromosome.
    #[serde(default)]
    pub forks: Vec<Point3<f32>>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snapshot {
    pub recipe_name: String,
    pub seed: u64,
    pub placements: Vec<Placement>,
    /// The genome fiber, if the recipe declared a chromosome.
    pub chromosome: Option<Chromosome>,
    /// Nascent-RNA strands grown from `ChromosomeSpec::rnas`.
    /// One entry per `RnaSpec`, in recipe order, center-relative.
    #[serde(default)]
    pub rna_strands: Vec<RnaStrand>,
}

impl Snapshot {
    pub fn new(recipe_name: String, seed: u64) -> Self {
        Self {
            recipe_name,
            seed,
            placements: Vec::new(),
            chromosome: None,
            rna_strands: Vec::new(),
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
