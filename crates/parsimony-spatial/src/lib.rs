//! Sparse hierarchical multiscale spatial structures for parsimony.
//!
//! See `docs/parsimony-design.md` §6 for the architecture. Two
//! complementary primitives:
//!
//! - `SpatialIndex` — over placed instances. Drives collision and
//!   neighbor queries. Backed by `BvhIndex` and `HierGridIndex`.
//! - `VoxelField` — over space itself. Drives compartment classification
//!   and free-space sampling. Three-level sparse hierarchical grid
//!   (OpenVDB-inspired 16/8/8 fanout).
//!
//! Phase 1 module plan (not yet implemented):
//!
//! ```ignore
//! pub mod aabb;
//! pub mod index;       // SpatialIndex trait
//! pub mod bvh;         // BvhIndex
//! pub mod hier_grid;   // HierGridIndex
//! pub mod voxel;       // VoxelField
//! pub mod query;       // shared query types
//! ```
