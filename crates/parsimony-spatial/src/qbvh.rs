//! [`QbvhIndex`] — 4-wide SIMD bounding volume hierarchy with native
//! incremental insert/remove/update.
//!
//! Each internal node stores 4 child AABBs in SoA (`f32x4` per axis,
//! min and max — 6 lanes total), so traversal tests all 4 children in
//! parallel with a single SIMD compare. Tree depth is `log₄(n)` instead
//! of `log₂(n)`.
//!
//! ## Partial nodes are free
//!
//! Empty slots use a **sentinel AABB** (`min = +∞`, `max = -∞`) that
//! naturally fails every intersection test. So nodes with 1, 2, or 3
//! occupied slots need no special-casing in the SIMD path — the unused
//! lanes are masked out by the standard test.
//!
//! ## Native incremental ops, O(log n)
//!
//! - **Insert**: descend by min-growth (one SIMD test per level),
//!   append to the chosen leaf, refit ancestors. If the leaf exceeds
//!   [`QbvhConfig::leaf_capacity`], split it into up to 4 sub-leaves
//!   under a new internal node (`O(leaf_capacity)` work, replaces the
//!   leaf in its parent's slot).
//! - **Remove**: swap-remove from the leaf, refit. If the leaf becomes
//!   empty its parent's SoA slot is reset to the sentinel; the empty
//!   leaf is kept in storage and reclaimed on the next [`rebuild`](Self::rebuild).
//! - **Update**: replace AABB in place, refit leaf and ancestors.
//!
//! Every edit is O(log₄ n). No O(n log n) rebuild penalty during
//! steady-state dynamics.
//!
//! ## Build
//!
//! [`QbvhIndex::build_from`] does top-down 4-way SAH: at each internal
//! node, binary-split the prim slice via SAH, then sub-binary-split
//! each half (skipped when a half is already leaf-sized), producing up
//! to 4 partitions per internal node.
//!
//! [Self::rebuild]: QbvhIndex::rebuild

use std::collections::HashMap;

use nalgebra::{Point3, Vector3};
use wide::{CmpLe, f32x4};

use crate::aabb::Aabb;
use crate::index::SpatialIndex;
use crate::query::{IndexError, IndexStats, Sphere};

// ---------- config + node types ----------

/// Tunables for [`QbvhIndex`].
#[derive(Debug, Clone, Copy)]
pub struct QbvhConfig {
    /// Max prims per leaf before [`SpatialIndex::insert`] triggers a leaf split.
    pub leaf_capacity: usize,
    /// Bins for the binned-SAH split.
    pub sah_bins: usize,
    /// SAH traversal cost coefficient.
    pub traversal_cost: f32,
    /// SAH intersection cost coefficient.
    pub intersection_cost: f32,
    /// Rebuild trigger threshold: `current_sah_cost / pristine_sah_cost > this`.
    pub rebuild_threshold: f32,
}

impl Default for QbvhConfig {
    fn default() -> Self {
        Self {
            leaf_capacity: 4,
            sah_bins: 16,
            traversal_cost: 1.0,
            intersection_cost: 1.0,
            rebuild_threshold: 1.5,
        }
    }
}

type NodeIdx = u32;
const NULL_NODE: NodeIdx = u32::MAX;

#[derive(Debug, Clone, Copy)]
struct Primitive {
    uid: u64,
    aabb: Aabb,
}

#[derive(Debug)]
enum NodeKind {
    Internal {
        soa: SoaAabb4,
        children: [NodeIdx; 4],
        child_count: u8,
    },
    Leaf {
        aabb: Aabb,
        prims: Vec<Primitive>,
    },
}

#[derive(Debug)]
struct Node {
    parent: NodeIdx,
    parent_slot: u8,
    kind: NodeKind,
}

/// 4-wide SoA AABB storage. Empty slots use `min = +∞`, `max = -∞`
/// which naturally fail SIMD intersection tests.
#[derive(Debug, Clone, Copy)]
struct SoaAabb4 {
    min_x: f32x4,
    min_y: f32x4,
    min_z: f32x4,
    max_x: f32x4,
    max_y: f32x4,
    max_z: f32x4,
}

impl SoaAabb4 {
    fn empty() -> Self {
        let pi = f32x4::splat(f32::INFINITY);
        let ni = f32x4::splat(f32::NEG_INFINITY);
        Self {
            min_x: pi,
            min_y: pi,
            min_z: pi,
            max_x: ni,
            max_y: ni,
            max_z: ni,
        }
    }

    fn set_lane(&mut self, lane: u8, aabb: Aabb) {
        let i = lane as usize;
        debug_assert!(i < 4);
        let mut a = self.min_x.to_array();
        a[i] = aabb.min.x;
        self.min_x = f32x4::new(a);
        let mut a = self.min_y.to_array();
        a[i] = aabb.min.y;
        self.min_y = f32x4::new(a);
        let mut a = self.min_z.to_array();
        a[i] = aabb.min.z;
        self.min_z = f32x4::new(a);
        let mut a = self.max_x.to_array();
        a[i] = aabb.max.x;
        self.max_x = f32x4::new(a);
        let mut a = self.max_y.to_array();
        a[i] = aabb.max.y;
        self.max_y = f32x4::new(a);
        let mut a = self.max_z.to_array();
        a[i] = aabb.max.z;
        self.max_z = f32x4::new(a);
    }

    fn clear_lane(&mut self, lane: u8) {
        self.set_lane(lane, Aabb::empty());
    }

    fn get_lane(&self, lane: u8) -> Aabb {
        let i = lane as usize;
        debug_assert!(i < 4);
        let min_x = self.min_x.to_array()[i];
        let min_y = self.min_y.to_array()[i];
        let min_z = self.min_z.to_array()[i];
        let max_x = self.max_x.to_array()[i];
        let max_y = self.max_y.to_array()[i];
        let max_z = self.max_z.to_array()[i];
        Aabb::new(Point3::new(min_x, min_y, min_z), Point3::new(max_x, max_y, max_z))
    }

    /// Union of the 4 stored AABBs. Empty slots' `+∞`/`-∞` sentinels
    /// don't affect the union — `min(real, +∞) = real`,
    /// `max(real, -∞) = real`.
    fn union(&self) -> Aabb {
        let mn_x = self.min_x.to_array();
        let mn_y = self.min_y.to_array();
        let mn_z = self.min_z.to_array();
        let mx_x = self.max_x.to_array();
        let mx_y = self.max_y.to_array();
        let mx_z = self.max_z.to_array();
        let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
        let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
        for i in 0..4 {
            min.x = min.x.min(mn_x[i]);
            min.y = min.y.min(mn_y[i]);
            min.z = min.z.min(mn_z[i]);
            max.x = max.x.max(mx_x[i]);
            max.y = max.y.max(mx_y[i]);
            max.z = max.z.max(mx_z[i]);
        }
        Aabb::new(min, max)
    }

    /// SIMD intersection test against a query AABB. Returns a 4-bit
    /// mask in bits 0..=3 — bit `i` set means lane `i` overlaps `q`.
    fn intersects_aabb_mask(&self, q: &Aabb) -> u32 {
        let qmin_x = f32x4::splat(q.min.x);
        let qmin_y = f32x4::splat(q.min.y);
        let qmin_z = f32x4::splat(q.min.z);
        let qmax_x = f32x4::splat(q.max.x);
        let qmax_y = f32x4::splat(q.max.y);
        let qmax_z = f32x4::splat(q.max.z);

        let hit_x = self.min_x.cmp_le(qmax_x) & qmin_x.cmp_le(self.max_x);
        let hit_y = self.min_y.cmp_le(qmax_y) & qmin_y.cmp_le(self.max_y);
        let hit_z = self.min_z.cmp_le(qmax_z) & qmin_z.cmp_le(self.max_z);

        let hit = hit_x & hit_y & hit_z;
        (hit.move_mask() as u32) & 0xF
    }

    /// SIMD intersection test against a sphere via nearest-point-on-AABB
    /// distance squared. Returns a 4-bit mask.
    fn intersects_sphere_mask(&self, q: &Sphere) -> u32 {
        let cx = f32x4::splat(q.center.x);
        let cy = f32x4::splat(q.center.y);
        let cz = f32x4::splat(q.center.z);
        let zero = f32x4::splat(0.0);

        // axis-wise distance from sphere center to AABB:
        //   d_axis = max(0, max(min_axis - c_axis, c_axis - max_axis))
        // when c is inside [min, max], both diffs are <= 0 and we clamp to 0
        let dx = (self.min_x - cx).max(cx - self.max_x).max(zero);
        let dy = (self.min_y - cy).max(cy - self.max_y).max(zero);
        let dz = (self.min_z - cz).max(cz - self.max_z).max(zero);

        let d2 = dx * dx + dy * dy + dz * dz;
        let r2 = f32x4::splat(q.radius * q.radius);
        let mask = d2.cmp_le(r2);
        (mask.move_mask() as u32) & 0xF
    }
}

// ---------- the index ----------

/// 4-wide SIMD bounding volume hierarchy.
#[derive(Debug)]
pub struct QbvhIndex {
    nodes: Vec<Option<Node>>,
    free_list: Vec<NodeIdx>,
    root: Option<NodeIdx>,
    by_uid: HashMap<u64, NodeIdx>,
    instance_count: usize,
    pristine_sah_cost: f32,
    config: QbvhConfig,
}

impl Default for QbvhIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl QbvhIndex {
    pub fn new() -> Self {
        Self::with_config(QbvhConfig::default())
    }

    pub fn with_config(config: QbvhConfig) -> Self {
        Self {
            nodes: Vec::new(),
            free_list: Vec::new(),
            root: None,
            by_uid: HashMap::new(),
            instance_count: 0,
            pristine_sah_cost: 0.0,
            config,
        }
    }

    pub fn config(&self) -> &QbvhConfig {
        &self.config
    }

    /// Bulk build from `(uid, aabb)` pairs. Faster and tighter than
    /// many incremental inserts. Returns [`IndexError::DuplicateUid`]
    /// for the first duplicate in `items`.
    pub fn build_from<I>(&mut self, items: I) -> Result<(), IndexError>
    where
        I: IntoIterator<Item = (u64, Aabb)>,
    {
        self.clear();
        let mut prims: Vec<Primitive> = Vec::new();
        let mut seen: HashMap<u64, ()> = HashMap::new();
        for (uid, aabb) in items {
            if seen.insert(uid, ()).is_some() {
                self.clear();
                return Err(IndexError::DuplicateUid(uid));
            }
            prims.push(Primitive { uid, aabb });
        }
        if prims.is_empty() {
            return Ok(());
        }
        let root = self.build_recursive(&mut prims, NULL_NODE, 0);
        self.root = Some(root);
        self.instance_count = self.by_uid.len();
        self.pristine_sah_cost = self.compute_sah_cost();
        Ok(())
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.free_list.clear();
        self.root = None;
        self.by_uid.clear();
        self.instance_count = 0;
        self.pristine_sah_cost = 0.0;
    }

    /// Force a full rebuild from the current set of primitives.
    pub fn rebuild(&mut self) {
        if self.root.is_none() {
            return;
        }
        let mut prims = self.collect_prims();
        self.nodes.clear();
        self.free_list.clear();
        self.root = None;
        self.by_uid.clear();
        if prims.is_empty() {
            self.instance_count = 0;
            return;
        }
        let root = self.build_recursive(&mut prims, NULL_NODE, 0);
        self.root = Some(root);
        self.instance_count = self.by_uid.len();
        self.pristine_sah_cost = self.compute_sah_cost();
    }

    pub fn needs_rebuild(&self) -> bool {
        if self.pristine_sah_cost <= 0.0 {
            return false;
        }
        self.compute_sah_cost() / self.pristine_sah_cost > self.config.rebuild_threshold
    }

    pub fn rebuild_if_needed(&mut self) -> bool {
        if self.needs_rebuild() {
            self.rebuild();
            true
        } else {
            false
        }
    }

    pub fn sah_cost(&self) -> f32 {
        self.compute_sah_cost()
    }

    pub fn quality_ratio(&self) -> f32 {
        if self.pristine_sah_cost <= 0.0 {
            return 1.0;
        }
        self.compute_sah_cost() / self.pristine_sah_cost
    }

    /// Walk the tree asserting every structural invariant. Panics on
    /// any violation; used heavily in tests.
    #[cfg(any(test, debug_assertions))]
    pub fn assert_valid(&self) {
        let Some(root) = self.root else {
            assert_eq!(self.instance_count, 0);
            assert!(self.by_uid.is_empty());
            return;
        };
        assert_eq!(self.nodes[root as usize].as_ref().unwrap().parent, NULL_NODE);
        let mut prim_count = 0usize;
        self.assert_subtree(root, &mut prim_count);
        assert_eq!(prim_count, self.instance_count, "instance_count out of sync");
        assert_eq!(prim_count, self.by_uid.len(), "by_uid size out of sync");
    }

    #[cfg(any(test, debug_assertions))]
    fn assert_subtree(&self, idx: NodeIdx, prim_count: &mut usize) {
        let node = self.nodes[idx as usize].as_ref().expect("dangling node ref");
        match &node.kind {
            NodeKind::Internal {
                soa,
                children,
                child_count,
            } => {
                let mut counted = 0u8;
                for (lane, &child) in children.iter().enumerate() {
                    let slot_aabb = soa.get_lane(lane as u8);
                    if child == NULL_NODE {
                        // empty slot must hold the sentinel
                        assert!(
                            !slot_aabb.is_valid(),
                            "empty slot {} at node {} has finite AABB",
                            lane,
                            idx
                        );
                    } else {
                        counted += 1;
                        let c = self.nodes[child as usize].as_ref().unwrap();
                        assert_eq!(c.parent, idx, "child {} wrong parent", child);
                        assert_eq!(
                            c.parent_slot, lane as u8,
                            "child {} wrong parent_slot",
                            child
                        );
                        let c_aabb = match &c.kind {
                            NodeKind::Leaf { aabb, .. } => *aabb,
                            NodeKind::Internal { soa: s, .. } => s.union(),
                        };
                        assert_eq!(
                            slot_aabb, c_aabb,
                            "slot {} AABB doesn't match child {} AABB",
                            lane, child
                        );
                        self.assert_subtree(child, prim_count);
                    }
                }
                assert_eq!(counted, *child_count, "child_count out of sync at {}", idx);
            }
            NodeKind::Leaf { aabb, prims } => {
                let mut acc = Aabb::empty();
                for p in prims {
                    acc = acc.union(&p.aabb);
                    assert_eq!(
                        self.by_uid.get(&p.uid),
                        Some(&idx),
                        "by_uid mismatch for uid {}",
                        p.uid
                    );
                    *prim_count += 1;
                }
                if prims.is_empty() {
                    assert!(!aabb.is_valid(), "empty leaf must have empty AABB sentinel");
                } else {
                    assert_eq!(*aabb, acc, "leaf {} cached AABB stale", idx);
                }
            }
        }
    }

    // ---------- private helpers ----------

    fn alloc_node(&mut self, node: Node) -> NodeIdx {
        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx as usize] = Some(node);
            idx
        } else {
            let idx = self.nodes.len() as NodeIdx;
            self.nodes.push(Some(node));
            idx
        }
    }

    fn free_node(&mut self, idx: NodeIdx) {
        self.nodes[idx as usize] = None;
        self.free_list.push(idx);
    }

    fn collect_prims(&self) -> Vec<Primitive> {
        let mut out = Vec::with_capacity(self.instance_count);
        for n in self.nodes.iter().flatten() {
            if let NodeKind::Leaf { prims, .. } = &n.kind {
                out.extend_from_slice(prims);
            }
        }
        out
    }

    /// Compute a node's own AABB on demand (cheap from cached leaf
    /// AABB, one SoA reduction for internals).
    fn node_aabb(&self, idx: NodeIdx) -> Aabb {
        let n = self.nodes[idx as usize].as_ref().unwrap();
        match &n.kind {
            NodeKind::Leaf { aabb, .. } => *aabb,
            NodeKind::Internal { soa, .. } => soa.union(),
        }
    }

    /// Recompute a leaf's cached AABB from its prims. Empty leaves get
    /// the sentinel.
    fn refit_leaf_local(&mut self, leaf: NodeIdx) {
        let n = self.nodes[leaf as usize].as_mut().unwrap();
        if let NodeKind::Leaf { prims, aabb } = &mut n.kind {
            if prims.is_empty() {
                *aabb = Aabb::empty();
            } else {
                let mut acc = Aabb::empty();
                for p in prims.iter() {
                    acc = acc.union(&p.aabb);
                }
                *aabb = acc;
            }
        }
    }

    /// Walk from `start` up the parent chain, updating each parent's
    /// SoA slot to reflect the current child's AABB.
    fn refit_path(&mut self, start: NodeIdx) {
        let mut child = start;
        let mut child_aabb = self.node_aabb(child);
        loop {
            let n = self.nodes[child as usize].as_ref().unwrap();
            let parent = n.parent;
            let slot = n.parent_slot;
            if parent == NULL_NODE {
                return;
            }
            // update parent's SoA at this child's slot
            if let Some(p) = self.nodes[parent as usize].as_mut()
                && let NodeKind::Internal { soa, .. } = &mut p.kind
            {
                soa.set_lane(slot, child_aabb);
            }
            child = parent;
            child_aabb = self.node_aabb(child);
        }
    }

    /// Find the leaf into which a new AABB should be inserted, by
    /// minimum surface-area growth at each step. SIMD-friendly in
    /// principle, scalar in this code path (it's only `log₄ n` steps).
    fn find_insertion_leaf(&self, new_aabb: &Aabb) -> NodeIdx {
        let mut current = self.root.expect("non-empty tree");
        loop {
            let node = self.nodes[current as usize].as_ref().unwrap();
            match &node.kind {
                NodeKind::Leaf { .. } => return current,
                NodeKind::Internal { soa, children, .. } => {
                    let mut best = NULL_NODE;
                    let mut best_growth = f32::INFINITY;
                    for (lane, &child) in children.iter().enumerate() {
                        if child == NULL_NODE {
                            continue;
                        }
                        let aabb = soa.get_lane(lane as u8);
                        let merged = aabb.union(new_aabb);
                        let growth = merged.surface_area() - aabb.surface_area();
                        if growth < best_growth {
                            best_growth = growth;
                            best = child;
                        }
                    }
                    current = best;
                }
            }
        }
    }

    /// Split an over-capacity leaf into up to 4 sub-leaves under a new
    /// internal node, occupying the leaf's index. The leaf's parent
    /// slot is unchanged (the new internal node has the same combined
    /// AABB as the old leaf).
    fn split_leaf(&mut self, leaf: NodeIdx) {
        // Take prims out.
        let mut prims = {
            let n = self.nodes[leaf as usize].as_mut().unwrap();
            match &mut n.kind {
                NodeKind::Leaf { prims, .. } => std::mem::take(prims),
                _ => unreachable!("split_leaf on non-leaf"),
            }
        };
        let n = prims.len();
        if n <= self.config.leaf_capacity {
            // shouldn't really happen, but be safe — put them back
            let node = self.nodes[leaf as usize].as_mut().unwrap();
            if let NodeKind::Leaf { prims: p, aabb } = &mut node.kind {
                *aabb = aabb_of_prims(&prims);
                *p = prims;
            }
            return;
        }

        // Plan: 4-way partition via 3 binary SAH splits.
        let parent_aabb = aabb_of_prims(&prims);
        let cuts = partition_4_way(&mut prims, &parent_aabb, &self.config);

        // Build the active partitions list.
        let mut partitions: Vec<(usize, usize)> = Vec::with_capacity(4);
        for w in 0..4 {
            let s = cuts[w];
            let e = cuts[w + 1];
            if s < e {
                partitions.push((s, e));
            }
        }

        if partitions.len() < 2 {
            // Partitioning failed (all centroids coincident, etc.). Put prims back —
            // the leaf will be over capacity but correct. Rebuild eventually fixes it.
            let node = self.nodes[leaf as usize].as_mut().unwrap();
            if let NodeKind::Leaf { prims: p, aabb } = &mut node.kind {
                *aabb = parent_aabb;
                *p = prims;
            }
            return;
        }

        // Allocate child leaves for each partition.
        let mut soa = SoaAabb4::empty();
        let mut children = [NULL_NODE; 4];
        let mut child_count = 0u8;
        for (slot, &(s, e)) in partitions.iter().enumerate() {
            let sub_prims: Vec<Primitive> = prims[s..e].to_vec();
            let sub_aabb = aabb_of_prims(&sub_prims);
            let child_idx = self.alloc_node(Node {
                parent: leaf,
                parent_slot: slot as u8,
                kind: NodeKind::Leaf {
                    aabb: sub_aabb,
                    prims: sub_prims,
                },
            });
            // Update by_uid for prims now in this child.
            if let NodeKind::Leaf { prims: cp, .. } =
                &self.nodes[child_idx as usize].as_ref().unwrap().kind
            {
                for p in cp {
                    self.by_uid.insert(p.uid, child_idx);
                }
            }
            soa.set_lane(slot as u8, sub_aabb);
            children[slot] = child_idx;
            child_count += 1;
        }

        // Convert `leaf` in place into an internal node. Its parent
        // and parent_slot stay the same; its combined AABB equals
        // parent_aabb (same prims, just regrouped), so the parent's
        // SoA at the old leaf's slot remains correct.
        let node = self.nodes[leaf as usize].as_mut().unwrap();
        node.kind = NodeKind::Internal {
            soa,
            children,
            child_count,
        };
    }

    /// Append a primitive to the chosen leaf, updating cache + by_uid.
    fn append_to_leaf(&mut self, leaf: NodeIdx, prim: Primitive) -> usize {
        let n = self.nodes[leaf as usize].as_mut().unwrap();
        if let NodeKind::Leaf { prims, aabb } = &mut n.kind {
            *aabb = if prims.is_empty() {
                prim.aabb
            } else {
                aabb.union(&prim.aabb)
            };
            prims.push(prim);
            self.by_uid.insert(prim.uid, leaf);
            prims.len()
        } else {
            unreachable!()
        }
    }

    /// Top-down 4-way SAH build. `parent` is the parent index (or
    /// [`NULL_NODE`] for the root). `slot` is which child slot we'll
    /// occupy in the parent (or 0 for the root).
    fn build_recursive(
        &mut self,
        prims: &mut [Primitive],
        parent: NodeIdx,
        slot: u8,
    ) -> NodeIdx {
        let aabb = aabb_of_prims(prims);
        if prims.len() <= self.config.leaf_capacity {
            return self.make_leaf(prims, aabb, parent, slot);
        }

        let cuts = partition_4_way(prims, &aabb, &self.config);
        // Collect non-empty partitions.
        let mut partitions: Vec<(usize, usize)> = Vec::with_capacity(4);
        for w in 0..4 {
            let s = cuts[w];
            let e = cuts[w + 1];
            if s < e {
                partitions.push((s, e));
            }
        }
        // Degenerate: only one non-empty partition — make a leaf even if oversized.
        if partitions.len() < 2 {
            return self.make_leaf(prims, aabb, parent, slot);
        }

        // Reserve the internal node so children can record us as parent.
        let internal_idx = self.alloc_node(Node {
            parent,
            parent_slot: slot,
            kind: NodeKind::Internal {
                soa: SoaAabb4::empty(),
                children: [NULL_NODE; 4],
                child_count: 0,
            },
        });

        // Build each partition.
        let mut soa = SoaAabb4::empty();
        let mut children = [NULL_NODE; 4];
        let mut child_count = 0u8;
        for (lane, &(s, e)) in partitions.iter().enumerate() {
            let sub = &mut prims[s..e];
            let sub_aabb = aabb_of_prims(sub);
            let child_idx = self.build_recursive(sub, internal_idx, lane as u8);
            soa.set_lane(lane as u8, sub_aabb);
            children[lane] = child_idx;
            child_count += 1;
        }

        // Fill in the internal node.
        if let Some(n) = self.nodes[internal_idx as usize].as_mut() {
            n.kind = NodeKind::Internal {
                soa,
                children,
                child_count,
            };
        }
        internal_idx
    }

    fn make_leaf(
        &mut self,
        prims: &[Primitive],
        aabb: Aabb,
        parent: NodeIdx,
        slot: u8,
    ) -> NodeIdx {
        let prims_vec = prims.to_vec();
        let idx = self.alloc_node(Node {
            parent,
            parent_slot: slot,
            kind: NodeKind::Leaf {
                aabb,
                prims: prims_vec,
            },
        });
        // Set by_uid mappings.
        if let NodeKind::Leaf { prims, .. } =
            &self.nodes[idx as usize].as_ref().unwrap().kind
        {
            for p in prims {
                self.by_uid.insert(p.uid, idx);
            }
        }
        idx
    }

    fn compute_sah_cost(&self) -> f32 {
        let Some(root) = self.root else {
            return 0.0;
        };
        let sa_root = self.node_aabb(root).surface_area().max(f32::EPSILON);
        let mut cost = 0.0;
        for n in self.nodes.iter().flatten() {
            match &n.kind {
                NodeKind::Internal { soa, .. } => {
                    let sa = soa.union().surface_area();
                    cost += self.config.traversal_cost * sa / sa_root;
                }
                NodeKind::Leaf { aabb, prims } => {
                    if !prims.is_empty() {
                        cost += self.config.intersection_cost
                            * prims.len() as f32
                            * aabb.surface_area()
                            / sa_root;
                    }
                }
            }
        }
        cost
    }
}

impl SpatialIndex for QbvhIndex {
    fn insert(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError> {
        if self.by_uid.contains_key(&uid) {
            return Err(IndexError::DuplicateUid(uid));
        }
        if self.root.is_none() {
            // empty tree — create a leaf root
            let leaf = self.alloc_node(Node {
                parent: NULL_NODE,
                parent_slot: 0,
                kind: NodeKind::Leaf {
                    aabb,
                    prims: vec![Primitive { uid, aabb }],
                },
            });
            self.root = Some(leaf);
            self.by_uid.insert(uid, leaf);
            self.instance_count = 1;
            self.pristine_sah_cost = self.compute_sah_cost();
            return Ok(());
        }

        let leaf = self.find_insertion_leaf(&aabb);
        let new_size = self.append_to_leaf(leaf, Primitive { uid, aabb });
        self.instance_count += 1;
        self.refit_path(leaf);

        if new_size > self.config.leaf_capacity {
            self.split_leaf(leaf);
            // Splitting only restructures within the same combined AABB; no further refit needed.
        }
        Ok(())
    }

    fn remove(&mut self, uid: u64) -> Result<(), IndexError> {
        let leaf = self.by_uid.remove(&uid).ok_or(IndexError::NotFound(uid))?;
        // swap-remove the prim
        {
            let n = self.nodes[leaf as usize].as_mut().unwrap();
            if let NodeKind::Leaf { prims, .. } = &mut n.kind {
                let pos = prims
                    .iter()
                    .position(|p| p.uid == uid)
                    .expect("by_uid pointed to wrong leaf");
                prims.swap_remove(pos);
            } else {
                unreachable!("by_uid pointed to a non-leaf node");
            }
        }
        self.refit_leaf_local(leaf);
        self.instance_count -= 1;

        // If the leaf is now empty, detach it from its parent (the
        // parent's SoA slot has already been set to the empty
        // sentinel by refit_path below — but we also free the leaf
        // node and clear the child slot).
        let leaf_empty = matches!(
            &self.nodes[leaf as usize].as_ref().unwrap().kind,
            NodeKind::Leaf { prims, .. } if prims.is_empty()
        );

        self.refit_path(leaf);

        if leaf_empty {
            self.detach_empty(leaf);
        }
        Ok(())
    }

    fn update(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError> {
        let leaf = *self.by_uid.get(&uid).ok_or(IndexError::NotFound(uid))?;
        {
            let n = self.nodes[leaf as usize].as_mut().unwrap();
            if let NodeKind::Leaf { prims, .. } = &mut n.kind {
                let prim = prims
                    .iter_mut()
                    .find(|p| p.uid == uid)
                    .expect("by_uid pointed to wrong leaf");
                prim.aabb = aabb;
            } else {
                unreachable!()
            }
        }
        self.refit_leaf_local(leaf);
        self.refit_path(leaf);
        Ok(())
    }

    fn query_aabb<F: FnMut(u64)>(&self, q: &Aabb, mut visit: F) {
        let Some(root) = self.root else {
            return;
        };
        let mut stack: Vec<NodeIdx> = Vec::with_capacity(64);
        stack.push(root);
        while let Some(idx) = stack.pop() {
            let node = self.nodes[idx as usize].as_ref().unwrap();
            match &node.kind {
                NodeKind::Internal { soa, children, .. } => {
                    let mask = soa.intersects_aabb_mask(q);
                    let mut m = mask;
                    while m != 0 {
                        let lane = m.trailing_zeros() as usize;
                        m &= m - 1;
                        let child = children[lane];
                        if child != NULL_NODE {
                            stack.push(child);
                        }
                    }
                }
                NodeKind::Leaf { aabb, prims } => {
                    if !aabb.intersects(q) {
                        continue;
                    }
                    for p in prims {
                        if p.aabb.intersects(q) {
                            visit(p.uid);
                        }
                    }
                }
            }
        }
    }

    fn query_sphere<F: FnMut(u64)>(&self, q: &Sphere, mut visit: F) {
        let Some(root) = self.root else {
            return;
        };
        let mut stack: Vec<NodeIdx> = Vec::with_capacity(64);
        stack.push(root);
        while let Some(idx) = stack.pop() {
            let node = self.nodes[idx as usize].as_ref().unwrap();
            match &node.kind {
                NodeKind::Internal { soa, children, .. } => {
                    let mask = soa.intersects_sphere_mask(q);
                    let mut m = mask;
                    while m != 0 {
                        let lane = m.trailing_zeros() as usize;
                        m &= m - 1;
                        let child = children[lane];
                        if child != NULL_NODE {
                            stack.push(child);
                        }
                    }
                }
                NodeKind::Leaf { aabb, prims } => {
                    if !q.intersects_aabb(aabb) {
                        continue;
                    }
                    for p in prims {
                        if q.intersects_aabb(&p.aabb) {
                            visit(p.uid);
                        }
                    }
                }
            }
        }
    }

    fn len(&self) -> usize {
        self.instance_count
    }

    fn stats(&self) -> IndexStats {
        let mut nodes = 0usize;
        let mut max_depth = 0usize;
        let mut depth_sum = 0u64;
        let mut leaf_count = 0u64;
        let mut bytes = std::mem::size_of::<Self>();
        bytes += self.nodes.capacity() * std::mem::size_of::<Option<Node>>();
        bytes += self.free_list.capacity() * std::mem::size_of::<NodeIdx>();
        bytes +=
            self.by_uid.capacity() * (std::mem::size_of::<u64>() + std::mem::size_of::<NodeIdx>());

        if let Some(root) = self.root {
            let mut stack: Vec<(NodeIdx, usize)> = Vec::new();
            stack.push((root, 0));
            while let Some((idx, depth)) = stack.pop() {
                nodes += 1;
                let node = self.nodes[idx as usize].as_ref().unwrap();
                match &node.kind {
                    NodeKind::Internal { children, .. } => {
                        for &c in children {
                            if c != NULL_NODE {
                                stack.push((c, depth + 1));
                            }
                        }
                    }
                    NodeKind::Leaf { prims, .. } => {
                        bytes += prims.capacity() * std::mem::size_of::<Primitive>();
                        if !prims.is_empty() {
                            leaf_count += 1;
                            depth_sum += depth as u64;
                            if depth > max_depth {
                                max_depth = depth;
                            }
                        }
                    }
                }
            }
        }

        let mean_depth = if leaf_count == 0 {
            0.0
        } else {
            depth_sum as f32 / leaf_count as f32
        };
        let root_aabb = self.root.map(|r| self.node_aabb(r));

        IndexStats {
            instances: self.instance_count,
            nodes,
            max_depth,
            mean_depth,
            root_aabb,
            memory_bytes: bytes,
        }
    }
}

impl QbvhIndex {
    /// Free a now-empty node (leaf or internal) and clear its slot in
    /// the parent. Cascades upward: if removing the slot brings the
    /// parent's `child_count` to zero, recursively detach the parent
    /// as well — that way 0-child internal nodes never linger in the
    /// tree (which would otherwise corrupt `find_insertion_leaf`).
    fn detach_empty(&mut self, node: NodeIdx) {
        let (parent, slot) = {
            let n = self.nodes[node as usize].as_ref().unwrap();
            (n.parent, n.parent_slot)
        };
        self.free_node(node);
        if parent == NULL_NODE {
            self.root = None;
            return;
        }
        let parent_empty;
        {
            let p = self.nodes[parent as usize].as_mut().unwrap();
            match &mut p.kind {
                NodeKind::Internal {
                    soa,
                    children,
                    child_count,
                } => {
                    soa.clear_lane(slot);
                    children[slot as usize] = NULL_NODE;
                    *child_count = child_count.saturating_sub(1);
                    parent_empty = *child_count == 0;
                }
                NodeKind::Leaf { .. } => unreachable!("parent must be internal"),
            }
        }
        if parent_empty {
            self.detach_empty(parent);
        }
        // 1-child internals are tolerated; rebuild compacts them lazily.
    }
}

// ---------- free-standing SAH helpers ----------

fn aabb_of_prims(prims: &[Primitive]) -> Aabb {
    let mut acc = Aabb::empty();
    for p in prims {
        acc = acc.union(&p.aabb);
    }
    acc
}

fn centroid_bounds(prims: &[Primitive]) -> (Point3<f32>, Point3<f32>) {
    let mut min = Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut max = Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for p in prims {
        let c = p.aabb.center();
        min = Point3::new(min.x.min(c.x), min.y.min(c.y), min.z.min(c.z));
        max = Point3::new(max.x.max(c.x), max.y.max(c.y), max.z.max(c.z));
    }
    (min, max)
}

fn longest_axis(extent: Vector3<f32>) -> usize {
    if extent.x >= extent.y && extent.x >= extent.z {
        0
    } else if extent.y >= extent.z {
        1
    } else {
        2
    }
}

/// Binary SAH split of `prims` in place. Returns the split index `mid`
/// (so `prims[..mid]` and `prims[mid..]` are the two halves), or
/// `None` if no useful split is possible (all centroids coincident).
fn sah_binary_split(prims: &mut [Primitive], parent_aabb: &Aabb, cfg: &QbvhConfig) -> Option<usize> {
    let n = prims.len();
    if n < 2 {
        return None;
    }
    let (cmin, cmax) = centroid_bounds(prims);
    let cextent = cmax - cmin;
    let axis = longest_axis(cextent);
    let extent = cextent[axis];
    if !extent.is_finite() || extent < f32::EPSILON {
        return None;
    }

    let bins = cfg.sah_bins;
    let inv_extent = bins as f32 / extent;
    let cmin_axis = cmin[axis];

    let mut bin_count = vec![0usize; bins];
    let mut bin_aabb = vec![Aabb::empty(); bins];
    let mut prim_bin = vec![0u8; n];
    for (i, p) in prims.iter().enumerate() {
        let c = p.aabb.center()[axis];
        let mut b = ((c - cmin_axis) * inv_extent) as isize;
        if b < 0 {
            b = 0;
        }
        let mut b = b as usize;
        if b >= bins {
            b = bins - 1;
        }
        bin_count[b] += 1;
        bin_aabb[b] = bin_aabb[b].union(&p.aabb);
        prim_bin[i] = b as u8;
    }

    let mut left_count = vec![0usize; bins];
    let mut left_aabb = vec![Aabb::empty(); bins];
    let mut acc_c = 0usize;
    let mut acc_a = Aabb::empty();
    for i in 0..bins {
        acc_c += bin_count[i];
        acc_a = acc_a.union(&bin_aabb[i]);
        left_count[i] = acc_c;
        left_aabb[i] = acc_a;
    }
    let mut right_count = vec![0usize; bins];
    let mut right_aabb = vec![Aabb::empty(); bins];
    let mut acc_c = 0usize;
    let mut acc_a = Aabb::empty();
    for i in (0..bins).rev() {
        acc_c += bin_count[i];
        acc_a = acc_a.union(&bin_aabb[i]);
        right_count[i] = acc_c;
        right_aabb[i] = acc_a;
    }

    let parent_sa = parent_aabb.surface_area().max(f32::EPSILON);
    let mut best_cost = f32::INFINITY;
    let mut best_bin = 0usize;
    for i in 0..bins - 1 {
        let lc = left_count[i];
        let rc = right_count[i + 1];
        if lc == 0 || rc == 0 {
            continue;
        }
        let cost = cfg.traversal_cost
            + (lc as f32 * left_aabb[i].surface_area()
                + rc as f32 * right_aabb[i + 1].surface_area())
                / parent_sa
                * cfg.intersection_cost;
        if cost < best_cost {
            best_cost = cost;
            best_bin = i;
        }
    }
    if !best_cost.is_finite() {
        // Median fallback when SAH found no valid split.
        prims.sort_by(|a, b| {
            a.aabb.center()[axis]
                .partial_cmp(&b.aabb.center()[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return Some(n / 2);
    }

    // Partition in place: prims[i] goes left if prim_bin[i] <= best_bin.
    let mut left_end = 0usize;
    for i in 0..n {
        if prim_bin[i] as usize <= best_bin {
            prims.swap(i, left_end);
            prim_bin.swap(i, left_end);
            left_end += 1;
        }
    }
    if left_end == 0 || left_end == n {
        // Defensive: shouldn't happen given the lc/rc>0 guard, but bail safely.
        prims.sort_by(|a, b| {
            a.aabb.center()[axis]
                .partial_cmp(&b.aabb.center()[axis])
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        return Some(n / 2);
    }
    Some(left_end)
}

/// Partition `prims` into up to 4 contiguous groups via 3 binary SAH
/// splits: top split, then each half if big enough. Returns 5 cut
/// indices `[0, c1, c2, c3, n]` (some adjacent pairs may be equal,
/// indicating an empty partition).
fn partition_4_way(
    prims: &mut [Primitive],
    parent_aabb: &Aabb,
    cfg: &QbvhConfig,
) -> [usize; 5] {
    let n = prims.len();
    // Top binary split.
    let mid = match sah_binary_split(prims, parent_aabb, cfg) {
        Some(m) => m,
        None => {
            // No useful split — return single partition [0..n].
            return [0, 0, 0, 0, n];
        }
    };
    // Sub-split the left half if it's big enough.
    let left_aabb = aabb_of_prims(&prims[..mid]);
    let left_mid = if mid > cfg.leaf_capacity {
        sah_binary_split(&mut prims[..mid], &left_aabb, cfg).unwrap_or(0)
    } else {
        0
    };
    let right_aabb = aabb_of_prims(&prims[mid..]);
    let right_mid_off = if n - mid > cfg.leaf_capacity {
        sah_binary_split(&mut prims[mid..], &right_aabb, cfg).unwrap_or(0)
    } else {
        0
    };
    let right_mid = mid + right_mid_off;

    // We have up to 4 partitions: [0..left_mid], [left_mid..mid],
    // [mid..right_mid], [right_mid..n]. Some may be empty.
    [0, left_mid, mid, right_mid, n]
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute::BruteIndex;
    use crate::index::SpatialIndexExt;
    use rand::{Rng, SeedableRng};
    use rand_xoshiro::Xoshiro256PlusPlus;
    use std::collections::HashSet;

    fn p(x: f32, y: f32, z: f32) -> Point3<f32> {
        Point3::new(x, y, z)
    }

    fn aabb(min: (f32, f32, f32), max: (f32, f32, f32)) -> Aabb {
        Aabb::new(p(min.0, min.1, min.2), p(max.0, max.1, max.2))
    }

    fn rand_aabb<R: Rng>(rng: &mut R, world: f32, max_size: f32) -> Aabb {
        let cx: f32 = rng.gen_range(-world..world);
        let cy: f32 = rng.gen_range(-world..world);
        let cz: f32 = rng.gen_range(-world..world);
        let hs: f32 = rng.gen_range(0.1..max_size);
        Aabb::new(
            p(cx - hs, cy - hs, cz - hs),
            p(cx + hs, cy + hs, cz + hs),
        )
    }

    // ---------- SoaAabb4 unit tests ----------

    #[test]
    fn soa_empty_intersects_nothing() {
        let soa = SoaAabb4::empty();
        let q = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        assert_eq!(soa.intersects_aabb_mask(&q), 0);
        let s = Sphere::new(p(0.0, 0.0, 0.0), 100.0);
        assert_eq!(soa.intersects_sphere_mask(&s), 0);
    }

    #[test]
    fn soa_set_get_round_trip() {
        let mut soa = SoaAabb4::empty();
        let a = aabb((1.0, 2.0, 3.0), (4.0, 5.0, 6.0));
        soa.set_lane(2, a);
        assert_eq!(soa.get_lane(2), a);
        // other lanes still empty
        assert!(!soa.get_lane(0).is_valid());
        assert!(!soa.get_lane(1).is_valid());
        assert!(!soa.get_lane(3).is_valid());
    }

    #[test]
    fn soa_intersects_aabb_mask_correct() {
        let mut soa = SoaAabb4::empty();
        // lane 0: overlaps q
        soa.set_lane(0, aabb((0.0, 0.0, 0.0), (2.0, 2.0, 2.0)));
        // lane 1: outside on x
        soa.set_lane(1, aabb((10.0, 0.0, 0.0), (12.0, 2.0, 2.0)));
        // lane 2: touches q on face
        soa.set_lane(2, aabb((1.0, 0.0, 0.0), (3.0, 1.0, 1.0)));
        // lane 3: outside on y
        soa.set_lane(3, aabb((0.0, 10.0, 0.0), (1.0, 12.0, 1.0)));

        let q = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        let mask = soa.intersects_aabb_mask(&q);
        assert_eq!(mask & 1, 1, "lane 0 should hit");
        assert_eq!(mask & 2, 0, "lane 1 should miss");
        assert_eq!(mask & 4, 4, "lane 2 should hit (touching)");
        assert_eq!(mask & 8, 0, "lane 3 should miss");
    }

    #[test]
    fn soa_intersects_sphere_mask_correct() {
        let mut soa = SoaAabb4::empty();
        soa.set_lane(0, aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))); // center inside, hit
        soa.set_lane(1, aabb((100.0, 100.0, 100.0), (101.0, 101.0, 101.0))); // far, miss
        soa.set_lane(2, aabb((2.0, 0.0, 0.0), (3.0, 1.0, 1.0))); // nearest dist 1.5
        // lane 3 empty
        let q = Sphere::new(p(0.5, 0.5, 0.5), 2.0);
        let mask = soa.intersects_sphere_mask(&q);
        assert_eq!(mask & 1, 1, "lane 0 should hit");
        assert_eq!(mask & 2, 0, "lane 1 should miss");
        assert_eq!(mask & 4, 4, "lane 2 should hit (d=1.5 < r=2.0)");
        assert_eq!(mask & 8, 0, "lane 3 empty");

        // Tighter sphere — lane 2 should now miss.
        let q2 = Sphere::new(p(0.5, 0.5, 0.5), 1.0);
        let mask2 = soa.intersects_sphere_mask(&q2);
        assert_eq!(mask2 & 4, 0, "lane 2 should miss when radius too small");
    }

    #[test]
    fn soa_union_handles_empty_lanes() {
        let mut soa = SoaAabb4::empty();
        soa.set_lane(1, aabb((-1.0, -2.0, -3.0), (4.0, 5.0, 6.0)));
        let u = soa.union();
        assert_eq!(u, aabb((-1.0, -2.0, -3.0), (4.0, 5.0, 6.0)));
    }

    // ---------- index tests ----------

    #[test]
    fn empty_is_empty() {
        let idx = QbvhIndex::new();
        assert!(idx.is_empty());
        let s = idx.stats();
        assert_eq!(s.instances, 0);
        assert!(s.root_aabb.is_none());
        idx.assert_valid();
    }

    #[test]
    fn single_insert_then_query() {
        let mut idx = QbvhIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        idx.insert(7, a).unwrap();
        idx.assert_valid();
        assert_eq!(idx.collect_aabb(&a), vec![7]);
    }

    #[test]
    fn duplicate_uid_errors() {
        let mut idx = QbvhIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        idx.insert(1, a).unwrap();
        assert_eq!(idx.insert(1, a), Err(IndexError::DuplicateUid(1)));
    }

    #[test]
    fn remove_unknown_errors() {
        let mut idx = QbvhIndex::new();
        assert_eq!(idx.remove(42), Err(IndexError::NotFound(42)));
    }

    #[test]
    fn update_unknown_errors() {
        let mut idx = QbvhIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        assert_eq!(idx.update(42, a), Err(IndexError::NotFound(42)));
    }

    #[test]
    fn remove_only_leaves_empty_tree() {
        let mut idx = QbvhIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        idx.insert(5, a).unwrap();
        idx.remove(5).unwrap();
        idx.assert_valid();
        assert!(idx.is_empty());
        assert!(idx.collect_aabb(&a).is_empty());
    }

    #[test]
    fn update_round_trip() {
        let mut idx = QbvhIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        let b = aabb((100.0, 100.0, 100.0), (101.0, 101.0, 101.0));
        idx.insert(5, a).unwrap();
        idx.assert_valid();
        assert_eq!(idx.collect_aabb(&a), vec![5]);

        idx.update(5, b).unwrap();
        idx.assert_valid();
        assert!(idx.collect_aabb(&a).is_empty());
        assert_eq!(idx.collect_aabb(&b), vec![5]);

        idx.remove(5).unwrap();
        idx.assert_valid();
        assert!(idx.collect_aabb(&b).is_empty());
    }

    #[test]
    fn bulk_build_against_brute() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        let n = 1_000usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        let mut qbvh = QbvhIndex::new();
        qbvh.build_from(prims.iter().copied()).unwrap();
        qbvh.assert_valid();

        let mut brute = BruteIndex::new();
        for (uid, a) in &prims {
            brute.insert(*uid, *a).unwrap();
        }

        for _ in 0..50 {
            let q = rand_aabb(&mut rng, 100.0, 20.0);
            let mut got = qbvh.collect_aabb(&q);
            let mut want = brute.collect_aabb(&q);
            got.sort();
            want.sort();
            assert_eq!(got, want, "QBVH disagrees with brute on aabb query");

            let center = p(
                rng.gen_range(-100.0..100.0),
                rng.gen_range(-100.0..100.0),
                rng.gen_range(-100.0..100.0),
            );
            let s = Sphere::new(center, rng.gen_range(1.0..30.0));
            let mut got = qbvh.collect_sphere(&s);
            let mut want = brute.collect_sphere(&s);
            got.sort();
            want.sort();
            assert_eq!(got, want, "QBVH disagrees with brute on sphere query");
        }
    }

    #[test]
    fn incremental_insert_against_brute() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFADE);
        let mut qbvh = QbvhIndex::new();
        let mut brute = BruteIndex::new();
        let n = 500usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        for (uid, a) in &prims {
            qbvh.insert(*uid, *a).unwrap();
            brute.insert(*uid, *a).unwrap();
        }
        qbvh.assert_valid();
        assert_eq!(qbvh.len(), n);

        for _ in 0..50 {
            let q = rand_aabb(&mut rng, 100.0, 20.0);
            let mut got = qbvh.collect_aabb(&q);
            let mut want = brute.collect_aabb(&q);
            got.sort();
            want.sort();
            assert_eq!(got, want);
        }
    }

    /// The big one: interleave many insert/remove/update ops with
    /// random queries, and verify invariants and oracle agreement
    /// throughout. This is the dynamics workload.
    #[test]
    fn edit_heavy_churn() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF);
        let mut qbvh = QbvhIndex::new();
        let mut brute = BruteIndex::new();
        let mut live: HashSet<u64> = HashSet::new();
        let mut next_uid: u64 = 0;

        for step in 0..3000 {
            let r: f32 = rng.gen_range(0.0..1.0);
            if r < 0.45 || live.is_empty() {
                let uid = next_uid;
                next_uid += 1;
                let a = rand_aabb(&mut rng, 100.0, 5.0);
                qbvh.insert(uid, a).unwrap();
                brute.insert(uid, a).unwrap();
                live.insert(uid);
            } else if r < 0.7 {
                let pick = *live.iter().nth(rng.gen_range(0..live.len())).unwrap();
                qbvh.remove(pick).unwrap();
                brute.remove(pick).unwrap();
                live.remove(&pick);
            } else {
                let pick = *live.iter().nth(rng.gen_range(0..live.len())).unwrap();
                let a = rand_aabb(&mut rng, 100.0, 5.0);
                qbvh.update(pick, a).unwrap();
                brute.update(pick, a).unwrap();
            }

            if step % 200 == 0 {
                qbvh.assert_valid();
                // spot-check queries
                for _ in 0..5 {
                    let q = rand_aabb(&mut rng, 100.0, 20.0);
                    let mut got = qbvh.collect_aabb(&q);
                    let mut want = brute.collect_aabb(&q);
                    got.sort();
                    want.sort();
                    assert_eq!(got, want, "QBVH/brute disagree at step {}", step);
                }
            }
        }
        qbvh.assert_valid();
        assert_eq!(qbvh.len(), brute.len());
    }

    #[test]
    fn leaf_split_triggers_at_capacity() {
        // With default leaf_capacity = 4, inserting 5 prims at the
        // same coordinates should split into an internal node.
        let mut qbvh = QbvhIndex::new();
        // 5 prims at *different* points so SAH can separate them
        for i in 0..5u64 {
            let x = i as f32 * 2.0;
            qbvh.insert(i, aabb((x, 0.0, 0.0), (x + 1.0, 1.0, 1.0))).unwrap();
        }
        qbvh.assert_valid();
        assert_eq!(qbvh.len(), 5);
        // Root should now be Internal (split happened).
        let root = qbvh.root.unwrap();
        let n = qbvh.nodes[root as usize].as_ref().unwrap();
        assert!(
            matches!(&n.kind, NodeKind::Internal { .. }),
            "root should be Internal after 5-th insert"
        );
    }

    #[test]
    fn many_inserts_then_many_removes_back_to_empty() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD00D);
        let mut qbvh = QbvhIndex::new();
        let n = 300usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        for (uid, a) in &prims {
            qbvh.insert(*uid, *a).unwrap();
        }
        qbvh.assert_valid();
        for (uid, _) in &prims {
            qbvh.remove(*uid).unwrap();
        }
        qbvh.assert_valid();
        assert!(qbvh.is_empty());
    }

    #[test]
    fn rebuild_preserves_query_results() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFEED);
        let n = 800usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        let mut qbvh = QbvhIndex::new();
        for (uid, a) in &prims {
            qbvh.insert(*uid, *a).unwrap();
        }
        qbvh.assert_valid();

        let queries: Vec<Aabb> = (0..30)
            .map(|_| rand_aabb(&mut rng, 100.0, 20.0))
            .collect();
        let pre: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| {
                let mut v = qbvh.collect_aabb(q);
                v.sort();
                v
            })
            .collect();

        qbvh.rebuild();
        qbvh.assert_valid();
        let post: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| {
                let mut v = qbvh.collect_aabb(q);
                v.sort();
                v
            })
            .collect();
        assert_eq!(pre, post);
        assert!((qbvh.quality_ratio() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn rebuild_after_removes_compacts() {
        // Insert n, remove half, rebuild — empty leaves should be gone.
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xCAFE);
        let n = 200usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        let mut qbvh = QbvhIndex::new();
        for (uid, a) in &prims {
            qbvh.insert(*uid, *a).unwrap();
        }
        for (uid, _) in prims.iter().step_by(2) {
            qbvh.remove(*uid).unwrap();
        }
        qbvh.assert_valid();
        let stats_before = qbvh.stats();
        qbvh.rebuild();
        qbvh.assert_valid();
        let stats_after = qbvh.stats();
        assert_eq!(stats_after.instances, n / 2);
        // After rebuild, node count typically drops (empty leaves removed).
        assert!(stats_after.nodes <= stats_before.nodes);
    }

    #[test]
    fn coincident_centroids_correctness() {
        let mut qbvh = QbvhIndex::new();
        let mut brute = BruteIndex::new();
        for i in 0..20u64 {
            let s = 0.5 + (i as f32) * 0.1;
            let a = aabb((-s, -s, -s), (s, s, s));
            qbvh.insert(i, a).unwrap();
            brute.insert(i, a).unwrap();
        }
        qbvh.assert_valid();
        let q = aabb((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05));
        let mut got = qbvh.collect_aabb(&q);
        let mut want = brute.collect_aabb(&q);
        got.sort();
        want.sort();
        assert_eq!(got, want);
    }

    #[test]
    fn build_with_duplicate_input_errors() {
        let prims = vec![
            (1u64, aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
            (1u64, aabb((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))),
        ];
        let mut qbvh = QbvhIndex::new();
        assert_eq!(
            qbvh.build_from(prims.into_iter()),
            Err(IndexError::DuplicateUid(1))
        );
        assert!(qbvh.is_empty());
    }

    #[test]
    fn stats_are_consistent() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xABCD);
        let n = 200usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        let mut qbvh = QbvhIndex::new();
        qbvh.build_from(prims.iter().copied()).unwrap();
        let s = qbvh.stats();
        assert_eq!(s.instances, n);
        assert!(s.nodes > 0);
        assert!(s.max_depth >= 1);
        let root = s.root_aabb.expect("non-empty");
        for (_, a) in &prims {
            assert!(root.contains_aabb(a));
        }
    }
}
