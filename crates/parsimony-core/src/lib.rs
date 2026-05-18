//! Core parsimony: recipe loader, ingredient model, compartments,
//! placement strategies, snapshot serialization.
//!
//! See `docs/parsimony-design.md` §§ 5, 7, 8, 9, 10 for the design.
//!
//! Phase 2 module plan (not yet implemented):
//!
//! ```ignore
//! pub mod recipe;        // serde over cellPACK v2.1 + parsimony extensions
//! pub mod ingredient;    // Ingredient, Variant, sphere-tree representation
//! pub mod compartment;   // analytical + mesh; in/out classification
//! pub mod placement;     // Placement, Snapshot, Op, ReplaceChange
//! pub mod placer;        // Placer trait + greedy random rejection sampler
//! pub mod collision;     // sphere-tree vs sphere-tree
//! pub mod env;           // Env: holds spatial index + voxel field + recipe
//! pub mod output;        // Simularium + transform-list JSON writers
//! pub mod sphere_tree;   // .sph parser, K-means decomposition
//! ```
