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
//! Phase 1b′ status: 4-wide SIMD [`QbvhIndex`] with native incremental
//! ops (insert/remove/update all O(log₄ n), no rebuild during steady-
//! state edits). [`BruteIndex`] is retained as the correctness oracle.

pub mod aabb;
pub mod brute;
pub mod index;
pub mod qbvh;
pub mod query;

pub use aabb::Aabb;
pub use brute::BruteIndex;
pub use index::{SpatialIndex, SpatialIndexExt};
pub use qbvh::{QbvhConfig, QbvhIndex};
pub use query::{IndexError, IndexStats, Ray, Sphere};
