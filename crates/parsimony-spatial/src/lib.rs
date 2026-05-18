//! Sparse hierarchical multiscale spatial structures for parsimony.
//!
//! See `docs/parsimony-design.md` §6 for the architecture. Two
//! complementary primitives:
//!
//! - [`SpatialIndex`] — over placed instances. Drives collision and
//!   neighbor queries. Implementations live in [`brute`] (reference),
//!   `bvh` (Phase 1b), and `hier_grid` (Phase 1c).
//! - `VoxelField` — over space itself. Drives compartment classification
//!   and free-space sampling. Three-level sparse hierarchical grid
//!   (Phase 1d, not yet implemented).
//!
//! Phase 1a status: AABB + query primitives + `SpatialIndex` trait +
//! [`BruteIndex`] reference impl (correctness oracle for later phases).

pub mod aabb;
pub mod query;
pub mod index;
pub mod brute;

pub use aabb::Aabb;
pub use brute::BruteIndex;
pub use index::{SpatialIndex, SpatialIndexExt};
pub use query::{IndexError, IndexStats, Ray, Sphere};
