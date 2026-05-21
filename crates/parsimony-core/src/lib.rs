//! Core parsimony: recipe loader, ingredient model, compartments,
//! placement strategies, snapshot serialization.
//!
//! See `docs/parsimony-design.md` §§ 5, 7, 8, 9, 10 for the design.

pub mod compartment;
pub(crate) mod clearance_grid;
pub mod fiber;
pub mod fiber_pack;
pub mod ingredient;
pub(crate) mod octree;
pub mod output;
pub mod pipeline;
pub mod placement;
pub mod placer;
pub mod recipe;
pub mod relax;

pub use compartment::{Compartment, CompartmentId, CompartmentKind};
pub use fiber_pack::{pack_on_fiber, FiberBinding};
pub use ingredient::{Ingredient, IngredientId, IngredientShape, ProxySphere};
pub use output::{write_pack_json, write_simularium_json, write_transforms_json};
pub use pipeline::{
    Pipeline, PipelineError, PipelineRun, Stage, StageKind, StagePlan, StageReport,
};
pub use placement::{Op, Placement, ReplaceChange, Snapshot, VariantId};
pub use placer::{
    GreedyRandomPlacer, PlacementBackend, PlacerConfig, PlacerOutcome, PlacerStats,
};
pub use recipe::{
    PackingMode, PlacementDirective, Recipe, RecipeError, RegionKind,
};
pub use relax::{relax, RelaxStats};
