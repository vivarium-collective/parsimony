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
//! Phase 1d status: + [`VoxelField`] — sparse hierarchical multiscale
//! voxel grid for compartment classification and free-space sampling.

pub mod aabb;
pub mod brute;
pub mod index;
pub mod qbvh;
pub mod query;
pub mod voxel;

pub use aabb::Aabb;
pub use brute::BruteIndex;
pub use index::{SpatialIndex, SpatialIndexExt};
pub use qbvh::{QbvhConfig, QbvhIndex};
pub use query::{IndexError, IndexStats, Ray, Sphere};
pub use voxel::{
    prepare_trimesh_for_voxelize, voxelize_trimesh, Cell, CellCoord, CellFlags, CompartmentId,
    VoxelField, VoxelFieldStats, MEMBRANE_INNER, MEMBRANE_OUTER, OCCUPIED, SURFACE,
};
