//! The [`SpatialIndex`] trait — common interface for every spatial
//! index over packed instances. See `docs/parsimony-design.md` §6.3.

use crate::aabb::Aabb;
use crate::query::{IndexError, IndexStats, Sphere};

/// Append-and-query interface over instance UIDs and their AABBs.
///
/// Implementations support insertion, removal, and AABB-update of
/// individual instances by UID, and report the set of UIDs whose AABBs
/// overlap a given query region.
///
/// ## UID semantics
///
/// UIDs are caller-supplied `u64`s, unique per index. Inserting a
/// duplicate is [`IndexError::DuplicateUid`]; operating on an unknown
/// UID is [`IndexError::NotFound`]. UIDs need not be dense or monotonic.
///
/// ## Query semantics
///
/// Queries are *AABB overlap* over the **stored** AABBs. Both
/// `query_aabb` and `query_sphere` may report any UID whose stored AABB
/// touches (`<=`) the query region. False positives at the
/// AABB-vs-real-shape level are the caller's problem to filter; the
/// index's job is the broad phase.
///
/// `query_sphere` does the sphere-vs-AABB test inline, so its results
/// are tight to the sphere shape (not just its enclosing AABB).
///
/// ## Visitor pattern
///
/// Queries take an `FnMut(u64)` callback to avoid allocation. Use the
/// blanket [`SpatialIndexExt`] for `Vec`-returning variants.
pub trait SpatialIndex {
    /// Insert `(uid, aabb)`. Errors if `uid` is already present.
    fn insert(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError>;

    /// Remove an instance by UID. Errors if absent.
    fn remove(&mut self, uid: u64) -> Result<(), IndexError>;

    /// Replace the stored AABB of an existing instance. Errors if absent.
    fn update(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError>;

    /// Visit every UID whose stored AABB overlaps `q`.
    fn query_aabb<F: FnMut(u64)>(&self, q: &Aabb, visit: F);

    /// Visit every UID whose stored AABB intersects `q` (sphere-vs-AABB).
    fn query_sphere<F: FnMut(u64)>(&self, q: &Sphere, visit: F);

    /// Number of instances stored.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Diagnostic snapshot — for benchmarking and structural assertions.
    fn stats(&self) -> IndexStats;
}

/// Ergonomic helpers on top of [`SpatialIndex`]. Blanket-implemented.
pub trait SpatialIndexExt: SpatialIndex {
    /// Collect every UID whose AABB overlaps `q`.
    fn collect_aabb(&self, q: &Aabb) -> Vec<u64> {
        let mut out = Vec::new();
        self.query_aabb(q, |uid| out.push(uid));
        out
    }

    /// Collect every UID whose AABB intersects `q`.
    fn collect_sphere(&self, q: &Sphere) -> Vec<u64> {
        let mut out = Vec::new();
        self.query_sphere(q, |uid| out.push(uid));
        out
    }
}

impl<T: SpatialIndex + ?Sized> SpatialIndexExt for T {}
