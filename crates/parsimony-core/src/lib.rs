//! Core parsimony: recipe loader, ingredient model, compartments,
//! placement strategies, snapshot serialization.
//!
//! See `docs/parsimony-design.md` §§ 5, 7, 8, 9, 10 for the design.

pub mod compartment;
pub(crate) mod clearance_grid;
pub mod ingredient;
pub mod output;
pub mod placement;
pub mod placer;
pub mod recipe;

pub use compartment::{Compartment, CompartmentId, CompartmentKind};
pub use ingredient::{Ingredient, IngredientId, IngredientShape, ProxySphere};
pub use output::{write_simularium_json, write_transforms_json};
pub use placement::{Op, Placement, ReplaceChange, Snapshot, VariantId};
pub use placer::{GreedyRandomPlacer, PlacerConfig, PlacerOutcome, PlacerStats};
pub use recipe::{
    PackingMode, PlacementDirective, Recipe, RecipeError, RegionKind,
};
