//! [`BvhIndex`] — top-down SAH bounding volume hierarchy.
//!
//! Binary BVH (not 4-wide SIMD), self-owning. Each leaf holds up to
//! `leaf_capacity` primitives; internal nodes branch 2-wide. Nodes live
//! in a flat `Vec<Option<Node>>` indexed by `usize`; freed slots
//! recycle through a free list.
//!
//! ## Build
//!
//! [`BvhIndex::build_from`] constructs a tree top-down using binned
//! SAH (Wald 2007, defaults: 16 bins, traversal/intersection cost
//! ratio 1.0/1.0). If no SAH split improves over a leaf and the slice
//! is over the leaf capacity, we fall back to a median split on the
//! longest centroid axis to guarantee termination.
//!
//! ## Incremental ops
//!
//! - [`SpatialIndex::insert`]: descend by minimum-surface-area-growth,
//!   append to the chosen leaf, refit ancestors. Leaves grow without
//!   bound; quality is recovered by [`BvhIndex::rebuild`].
//! - [`SpatialIndex::remove`]: locate the prim via the uid map, swap-
//!   remove it from its leaf, refit ancestors.
//! - [`SpatialIndex::update`]: replace the stored AABB in place, refit
//!   leaf and ancestors.
//!
//! ## Rebuild policy
//!
//! Tree quality after many incremental ops drifts away from the SAH
//! optimum. [`BvhIndex::needs_rebuild`] compares the current SAH cost
//! against the cost measured right after the last full build;
//! [`BvhIndex::rebuild_if_needed`] acts on it. Triggering is *manual* —
//! callers decide when to pay the rebuild.

use std::cmp::Ordering;
use std::collections::HashMap;

use nalgebra::{Point3, Vector3};

use crate::aabb::Aabb;
use crate::index::SpatialIndex;
use crate::query::{IndexError, IndexStats, Sphere};

/// Tunables for [`BvhIndex`].
#[derive(Debug, Clone, Copy)]
pub struct BvhConfig {
    /// Target maximum primitives per leaf at build time. Incremental
    /// inserts may exceed this until the next rebuild.
    pub leaf_capacity: usize,
    /// Number of bins for the binned-SAH split search.
    pub sah_bins: usize,
    /// SAH traversal cost (cost of descending an internal node).
    pub traversal_cost: f32,
    /// SAH intersection cost (cost of testing one primitive).
    pub intersection_cost: f32,
    /// Rebuild trigger: `current_sah_cost / pristine_sah_cost > this`.
    pub rebuild_threshold: f32,
}

impl Default for BvhConfig {
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

type NodeIdx = usize;

#[derive(Debug, Clone, Copy)]
struct Primitive {
    uid: u64,
    aabb: Aabb,
}

#[derive(Debug)]
enum NodeKind {
    Internal { left: NodeIdx, right: NodeIdx },
    Leaf { prims: Vec<Primitive> },
}

#[derive(Debug)]
struct Node {
    aabb: Aabb,
    parent: Option<NodeIdx>,
    kind: NodeKind,
}

/// Top-down SAH bounding volume hierarchy over instance AABBs.
#[derive(Debug)]
pub struct BvhIndex {
    nodes: Vec<Option<Node>>,
    free_list: Vec<NodeIdx>,
    root: Option<NodeIdx>,
    /// `uid -> leaf node index`.
    by_uid: HashMap<u64, NodeIdx>,
    instance_count: usize,
    pristine_sah_cost: f32,
    config: BvhConfig,
}

impl Default for BvhIndex {
    fn default() -> Self {
        Self::new()
    }
}

impl BvhIndex {
    pub fn new() -> Self {
        Self::with_config(BvhConfig::default())
    }

    pub fn with_config(config: BvhConfig) -> Self {
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

    pub fn config(&self) -> &BvhConfig {
        &self.config
    }

    /// Build a fresh BVH from `(uid, aabb)` pairs. Faster and tighter
    /// than inserting one-by-one. Duplicates within the input are
    /// rejected: returns [`IndexError::DuplicateUid`] for the first
    /// repeated UID seen.
    pub fn build_from<I>(&mut self, items: I) -> Result<(), IndexError>
    where
        I: IntoIterator<Item = (u64, Aabb)>,
    {
        self.clear();
        let mut prims: Vec<Primitive> = Vec::new();
        for (uid, aabb) in items {
            if self.by_uid.contains_key(&uid) {
                self.clear();
                return Err(IndexError::DuplicateUid(uid));
            }
            // placeholder index; corrected after build via assign_leaves.
            self.by_uid.insert(uid, usize::MAX);
            prims.push(Primitive { uid, aabb });
        }
        if prims.is_empty() {
            return Ok(());
        }
        let root = self.build_recursive(&mut prims, None);
        self.root = Some(root);
        self.instance_count = self.by_uid.len();
        self.pristine_sah_cost = self.compute_sah_cost();
        Ok(())
    }

    /// Drop all instances, freeing internal storage but keeping
    /// allocated capacity.
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
        // Rebuild from scratch.
        self.nodes.clear();
        self.free_list.clear();
        self.root = None;
        self.by_uid.clear();
        let n = prims.len();
        for p in &prims {
            self.by_uid.insert(p.uid, usize::MAX);
        }
        let root = self.build_recursive(&mut prims, None);
        self.root = Some(root);
        self.instance_count = n;
        self.pristine_sah_cost = self.compute_sah_cost();
    }

    /// True iff the current SAH cost exceeds `pristine * rebuild_threshold`.
    pub fn needs_rebuild(&self) -> bool {
        if self.pristine_sah_cost <= 0.0 {
            return false;
        }
        let current = self.compute_sah_cost();
        current / self.pristine_sah_cost > self.config.rebuild_threshold
    }

    /// Rebuild if [`needs_rebuild`](Self::needs_rebuild) reports degradation.
    /// Returns true if a rebuild occurred.
    pub fn rebuild_if_needed(&mut self) -> bool {
        if self.needs_rebuild() {
            self.rebuild();
            true
        } else {
            false
        }
    }

    /// SAH cost — sum over all live nodes of (cost * SA(node) / SA(root)).
    /// Returns 0 for an empty tree.
    pub fn sah_cost(&self) -> f32 {
        self.compute_sah_cost()
    }

    /// Cost ratio vs. the cost after the last full build. > 1.0 means
    /// the tree has degraded since the last build.
    pub fn quality_ratio(&self) -> f32 {
        if self.pristine_sah_cost <= 0.0 {
            return 1.0;
        }
        self.compute_sah_cost() / self.pristine_sah_cost
    }

    /// Debug helper: walk the tree and assert structural invariants
    /// (parent links match child references, every internal node's AABB
    /// contains both children's AABBs, leaves' AABBs cover their prims,
    /// `by_uid` points to the right leaf, and `instance_count` matches
    /// the actual prim total). Panics on any violation.
    #[cfg(any(test, debug_assertions))]
    pub fn assert_valid(&self) {
        let Some(root) = self.root else {
            assert_eq!(self.instance_count, 0);
            assert!(self.by_uid.is_empty());
            return;
        };
        assert!(self.nodes[root].as_ref().unwrap().parent.is_none());
        let mut count = 0usize;
        self.assert_subtree(root, &mut count);
        assert_eq!(count, self.instance_count, "instance_count out of sync");
        assert_eq!(count, self.by_uid.len(), "by_uid size out of sync");
    }

    #[cfg(any(test, debug_assertions))]
    fn assert_subtree(&self, idx: NodeIdx, count: &mut usize) {
        let node = self.nodes[idx].as_ref().expect("dangling node ref");
        match &node.kind {
            NodeKind::Internal { left, right } => {
                let l = self.nodes[*left].as_ref().unwrap();
                let r = self.nodes[*right].as_ref().unwrap();
                assert_eq!(l.parent, Some(idx), "left child has wrong parent");
                assert_eq!(r.parent, Some(idx), "right child has wrong parent");
                assert!(
                    node.aabb.contains_aabb(&l.aabb) && node.aabb.contains_aabb(&r.aabb),
                    "internal node AABB doesn't enclose children"
                );
                self.assert_subtree(*left, count);
                self.assert_subtree(*right, count);
            }
            NodeKind::Leaf { prims } => {
                for p in prims {
                    assert!(
                        node.aabb.contains_aabb(&p.aabb),
                        "leaf AABB doesn't enclose prim"
                    );
                    assert_eq!(
                        self.by_uid.get(&p.uid),
                        Some(&idx),
                        "by_uid mismatch for uid {}",
                        p.uid
                    );
                    *count += 1;
                }
            }
        }
    }

    // ---------- private helpers ----------

    fn alloc_node(&mut self, node: Node) -> NodeIdx {
        if let Some(idx) = self.free_list.pop() {
            self.nodes[idx] = Some(node);
            idx
        } else {
            let idx = self.nodes.len();
            self.nodes.push(Some(node));
            idx
        }
    }

    fn collect_prims(&self) -> Vec<Primitive> {
        let mut out = Vec::with_capacity(self.instance_count);
        for node in self.nodes.iter().flatten() {
            if let NodeKind::Leaf { prims } = &node.kind {
                out.extend_from_slice(prims);
            }
        }
        out
    }

    /// Build a subtree from `prims` and return its root index. Slot in
    /// `parent` afterwards via `set_parent` if needed.
    fn build_recursive(&mut self, prims: &mut [Primitive], parent: Option<NodeIdx>) -> NodeIdx {
        let aabb = aabb_of_prims(prims);

        if prims.len() <= self.config.leaf_capacity {
            return self.make_leaf(prims, aabb, parent);
        }

        let split = pick_split(prims, &aabb, &self.config);
        let mid = match split {
            SplitChoice::At(m) => m,
            SplitChoice::ForceMedian(axis) => {
                prims.sort_by(|a, b| {
                    a.aabb.center()[axis]
                        .partial_cmp(&b.aabb.center()[axis])
                        .unwrap_or(Ordering::Equal)
                });
                prims.len() / 2
            }
            SplitChoice::MakeLeaf => return self.make_leaf(prims, aabb, parent),
        };

        // Allocate this internal node first to reserve its index, so
        // children can record their parent.
        let idx = self.alloc_node(Node {
            aabb,
            parent,
            kind: NodeKind::Internal {
                left: usize::MAX,
                right: usize::MAX,
            },
        });
        let (left_prims, right_prims) = prims.split_at_mut(mid);
        let left = self.build_recursive(left_prims, Some(idx));
        let right = self.build_recursive(right_prims, Some(idx));
        if let Some(node) = self.nodes[idx].as_mut() {
            node.kind = NodeKind::Internal { left, right };
        }
        idx
    }

    fn make_leaf(&mut self, prims: &[Primitive], aabb: Aabb, parent: Option<NodeIdx>) -> NodeIdx {
        let prims_vec = prims.to_vec();
        let idx = self.alloc_node(Node {
            aabb,
            parent,
            kind: NodeKind::Leaf { prims: prims_vec },
        });
        if let Some(Node {
            kind: NodeKind::Leaf { prims },
            ..
        }) = self.nodes[idx].as_ref()
        {
            for p in prims {
                self.by_uid.insert(p.uid, idx);
            }
        }
        idx
    }

    /// Pick the leaf into which a new AABB should be inserted, by
    /// minimum surface-area growth at each step.
    fn find_insertion_leaf(&self, new_aabb: &Aabb) -> NodeIdx {
        let mut current = self.root.expect("non-empty tree");
        loop {
            let node = self.nodes[current].as_ref().unwrap();
            match &node.kind {
                NodeKind::Leaf { .. } => return current,
                NodeKind::Internal { left, right } => {
                    let l_aabb = self.nodes[*left].as_ref().unwrap().aabb;
                    let r_aabb = self.nodes[*right].as_ref().unwrap().aabb;
                    let l_growth =
                        l_aabb.union(new_aabb).surface_area() - l_aabb.surface_area();
                    let r_growth =
                        r_aabb.union(new_aabb).surface_area() - r_aabb.surface_area();
                    current = if l_growth <= r_growth { *left } else { *right };
                }
            }
        }
    }

    /// Recompute one node's AABB from its current contents.
    fn compute_node_aabb(&self, idx: NodeIdx) -> Aabb {
        match &self.nodes[idx].as_ref().unwrap().kind {
            NodeKind::Internal { left, right } => {
                let l = self.nodes[*left].as_ref().unwrap().aabb;
                let r = self.nodes[*right].as_ref().unwrap().aabb;
                l.union(&r)
            }
            NodeKind::Leaf { prims } => {
                let mut acc = Aabb::empty();
                for p in prims {
                    acc = acc.union(&p.aabb);
                }
                acc
            }
        }
    }

    fn refit_node(&mut self, idx: NodeIdx) {
        let new_aabb = self.compute_node_aabb(idx);
        if let Some(node) = self.nodes[idx].as_mut() {
            node.aabb = new_aabb;
        }
    }

    /// Refit `start` and walk parent pointers refitting every ancestor.
    fn refit_from(&mut self, start: NodeIdx) {
        let mut cur = Some(start);
        while let Some(idx) = cur {
            self.refit_node(idx);
            cur = self.nodes[idx].as_ref().unwrap().parent;
        }
    }

    fn compute_sah_cost(&self) -> f32 {
        let Some(root) = self.root else {
            return 0.0;
        };
        let sa_root = self.nodes[root]
            .as_ref()
            .unwrap()
            .aabb
            .surface_area()
            .max(f32::EPSILON);
        let mut cost = 0.0;
        for node in self.nodes.iter().flatten() {
            let sa = node.aabb.surface_area();
            match &node.kind {
                NodeKind::Internal { .. } => {
                    cost += self.config.traversal_cost * sa / sa_root;
                }
                NodeKind::Leaf { prims } => {
                    cost += self.config.intersection_cost * prims.len() as f32 * sa / sa_root;
                }
            }
        }
        cost
    }
}

impl SpatialIndex for BvhIndex {
    fn insert(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError> {
        if self.by_uid.contains_key(&uid) {
            return Err(IndexError::DuplicateUid(uid));
        }
        let Some(_) = self.root else {
            // first insertion: single-leaf root
            let leaf = self.alloc_node(Node {
                aabb,
                parent: None,
                kind: NodeKind::Leaf {
                    prims: vec![Primitive { uid, aabb }],
                },
            });
            self.root = Some(leaf);
            self.by_uid.insert(uid, leaf);
            self.instance_count = 1;
            self.pristine_sah_cost = self.compute_sah_cost();
            return Ok(());
        };

        let leaf = self.find_insertion_leaf(&aabb);
        if let Some(Node {
            kind: NodeKind::Leaf { prims },
            ..
        }) = self.nodes[leaf].as_mut()
        {
            prims.push(Primitive { uid, aabb });
        }
        self.by_uid.insert(uid, leaf);
        self.instance_count += 1;
        self.refit_from(leaf);
        Ok(())
    }

    fn remove(&mut self, uid: u64) -> Result<(), IndexError> {
        let leaf = self.by_uid.remove(&uid).ok_or(IndexError::NotFound(uid))?;
        if let Some(Node {
            kind: NodeKind::Leaf { prims },
            ..
        }) = self.nodes[leaf].as_mut()
        {
            let pos = prims
                .iter()
                .position(|p| p.uid == uid)
                .expect("by_uid points to wrong leaf");
            prims.swap_remove(pos);
        } else {
            // by_uid pointed to a non-leaf — corruption
            unreachable!("by_uid pointed to a non-leaf node");
        }
        self.instance_count -= 1;
        self.refit_from(leaf);
        Ok(())
    }

    fn update(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError> {
        let leaf = *self.by_uid.get(&uid).ok_or(IndexError::NotFound(uid))?;
        if let Some(Node {
            kind: NodeKind::Leaf { prims },
            ..
        }) = self.nodes[leaf].as_mut()
        {
            let prim = prims
                .iter_mut()
                .find(|p| p.uid == uid)
                .expect("by_uid points to wrong leaf");
            prim.aabb = aabb;
        } else {
            unreachable!("by_uid pointed to a non-leaf node");
        }
        self.refit_from(leaf);
        Ok(())
    }

    fn query_aabb<F: FnMut(u64)>(&self, q: &Aabb, mut visit: F) {
        let Some(root) = self.root else {
            return;
        };
        let mut stack: Vec<NodeIdx> = Vec::with_capacity(64);
        stack.push(root);
        while let Some(idx) = stack.pop() {
            let node = self.nodes[idx].as_ref().unwrap();
            if !node.aabb.intersects(q) {
                continue;
            }
            match &node.kind {
                NodeKind::Internal { left, right } => {
                    stack.push(*left);
                    stack.push(*right);
                }
                NodeKind::Leaf { prims } => {
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
            let node = self.nodes[idx].as_ref().unwrap();
            if !q.intersects_aabb(&node.aabb) {
                continue;
            }
            match &node.kind {
                NodeKind::Internal { left, right } => {
                    stack.push(*left);
                    stack.push(*right);
                }
                NodeKind::Leaf { prims } => {
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
            let mut stack = vec![(root, 0usize)];
            while let Some((idx, depth)) = stack.pop() {
                nodes += 1;
                let node = self.nodes[idx].as_ref().unwrap();
                match &node.kind {
                    NodeKind::Internal { left, right } => {
                        stack.push((*left, depth + 1));
                        stack.push((*right, depth + 1));
                    }
                    NodeKind::Leaf { prims } => {
                        bytes += prims.capacity() * std::mem::size_of::<Primitive>();
                        leaf_count += 1;
                        depth_sum += depth as u64;
                        if depth > max_depth {
                            max_depth = depth;
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
        let root_aabb = self.root.map(|r| self.nodes[r].as_ref().unwrap().aabb);

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

// ---------- SAH helpers (free functions, no `self`) ----------

#[derive(Debug, Clone, Copy)]
enum SplitChoice {
    /// Split at `prims[..mid]` / `prims[mid..]`. Prims are already
    /// partitioned in place.
    At(usize),
    /// SAH found no good split; fall back to median split on this axis.
    ForceMedian(usize),
    /// All centroids effectively coincident; just make a leaf even if
    /// over capacity. (Caller should still respect a hard upper bound.)
    MakeLeaf,
}

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

fn pick_split(prims: &mut [Primitive], parent_aabb: &Aabb, cfg: &BvhConfig) -> SplitChoice {
    let n = prims.len();
    let (cmin, cmax) = centroid_bounds(prims);
    let cextent = cmax - cmin;
    let axis = longest_axis(cextent);
    let extent = cextent[axis];

    if !extent.is_finite() || extent < f32::EPSILON {
        return SplitChoice::MakeLeaf;
    }

    let bins = cfg.sah_bins;
    let inv_extent = bins as f32 / extent;
    let cmin_axis = cmin[axis];

    // Bin each primitive.
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

    // Prefix sums from the left.
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
    // Prefix sums from the right.
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
    let leaf_cost = cfg.intersection_cost * n as f32;

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
        return SplitChoice::ForceMedian(axis);
    }
    // If SAH says "don't split", but we have too many for one leaf, force.
    if best_cost >= leaf_cost && n > cfg.leaf_capacity {
        return SplitChoice::ForceMedian(axis);
    }

    // Partition prims so prim_bin[i] <= best_bin comes first.
    let mut left_end = 0usize;
    for i in 0..n {
        if prim_bin[i] as usize <= best_bin {
            prims.swap(i, left_end);
            prim_bin.swap(i, left_end);
            left_end += 1;
        }
    }
    // Defensive: if partitioning degenerated, fall back to median.
    if left_end == 0 || left_end == n {
        return SplitChoice::ForceMedian(axis);
    }
    SplitChoice::At(left_end)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::brute::BruteIndex;
    use crate::index::SpatialIndexExt;
    use nalgebra::Point3;
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
        Aabb::new(p(cx - hs, cy - hs, cz - hs), p(cx + hs, cy + hs, cz + hs))
    }

    #[test]
    fn empty_is_empty() {
        let idx = BvhIndex::new();
        assert!(idx.is_empty());
        assert_eq!(idx.len(), 0);
        let stats = idx.stats();
        assert!(stats.root_aabb.is_none());
        assert_eq!(stats.nodes, 0);
        idx.assert_valid();
    }

    #[test]
    fn single_insert_then_query() {
        let mut idx = BvhIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        idx.insert(7, a).unwrap();
        idx.assert_valid();
        assert_eq!(idx.collect_aabb(&a), vec![7]);
    }

    #[test]
    fn duplicate_uid_errors() {
        let mut idx = BvhIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        idx.insert(1, a).unwrap();
        assert_eq!(idx.insert(1, a), Err(IndexError::DuplicateUid(1)));
    }

    #[test]
    fn remove_and_update_round_trip() {
        let mut idx = BvhIndex::new();
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
        assert_eq!(idx.len(), 0);
    }

    #[test]
    fn bulk_build_against_brute() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        let n = 1_000usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        let mut bvh = BvhIndex::new();
        bvh.build_from(prims.iter().copied()).unwrap();
        bvh.assert_valid();

        let mut brute = BruteIndex::new();
        for (uid, a) in &prims {
            brute.insert(*uid, *a).unwrap();
        }

        for _ in 0..50 {
            let q = rand_aabb(&mut rng, 100.0, 20.0);
            let mut got: Vec<u64> = bvh.collect_aabb(&q);
            let mut want: Vec<u64> = brute.collect_aabb(&q);
            got.sort();
            want.sort();
            assert_eq!(got, want, "bvh disagrees with brute on aabb query");

            let center = p(
                rng.gen_range(-100.0..100.0),
                rng.gen_range(-100.0..100.0),
                rng.gen_range(-100.0..100.0),
            );
            let s = Sphere::new(center, rng.gen_range(1.0..30.0));
            let mut got: Vec<u64> = bvh.collect_sphere(&s);
            let mut want: Vec<u64> = brute.collect_sphere(&s);
            got.sort();
            want.sort();
            assert_eq!(got, want, "bvh disagrees with brute on sphere query");
        }
    }

    #[test]
    fn incremental_insert_against_brute() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFADE);
        let mut bvh = BvhIndex::new();
        let mut brute = BruteIndex::new();
        let n = 500usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        for (uid, a) in &prims {
            bvh.insert(*uid, *a).unwrap();
            brute.insert(*uid, *a).unwrap();
        }
        bvh.assert_valid();

        for _ in 0..50 {
            let q = rand_aabb(&mut rng, 100.0, 20.0);
            let mut got: Vec<u64> = bvh.collect_aabb(&q);
            let mut want: Vec<u64> = brute.collect_aabb(&q);
            got.sort();
            want.sort();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn churn_keeps_invariants() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF);
        let mut bvh = BvhIndex::new();
        let mut brute = BruteIndex::new();
        let mut live: HashSet<u64> = HashSet::new();
        let mut next_uid: u64 = 0;

        for _ in 0..2000 {
            let r: f32 = rng.gen_range(0.0..1.0);
            if r < 0.5 || live.is_empty() {
                let uid = next_uid;
                next_uid += 1;
                let a = rand_aabb(&mut rng, 100.0, 5.0);
                bvh.insert(uid, a).unwrap();
                brute.insert(uid, a).unwrap();
                live.insert(uid);
            } else if r < 0.75 {
                let pick = *live.iter().nth(rng.gen_range(0..live.len())).unwrap();
                bvh.remove(pick).unwrap();
                brute.remove(pick).unwrap();
                live.remove(&pick);
            } else {
                let pick = *live.iter().nth(rng.gen_range(0..live.len())).unwrap();
                let a = rand_aabb(&mut rng, 100.0, 5.0);
                bvh.update(pick, a).unwrap();
                brute.update(pick, a).unwrap();
            }
            if rng.gen_range(0..200) == 0 {
                bvh.assert_valid();
            }
        }
        bvh.assert_valid();
        assert_eq!(bvh.len(), brute.len());

        // verify queries agree
        for _ in 0..30 {
            let q = rand_aabb(&mut rng, 100.0, 20.0);
            let mut got: Vec<u64> = bvh.collect_aabb(&q);
            let mut want: Vec<u64> = brute.collect_aabb(&q);
            got.sort();
            want.sort();
            assert_eq!(got, want);
        }
    }

    #[test]
    fn rebuild_preserves_query_results() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD00D);
        let n = 800usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        let mut bvh = BvhIndex::new();
        for (uid, a) in &prims {
            bvh.insert(*uid, *a).unwrap();
        }
        bvh.assert_valid();

        // Snapshot queries before rebuild.
        let queries: Vec<Aabb> = (0..30).map(|_| rand_aabb(&mut rng, 100.0, 20.0)).collect();
        let pre: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| {
                let mut v = bvh.collect_aabb(q);
                v.sort();
                v
            })
            .collect();

        bvh.rebuild();
        bvh.assert_valid();
        assert_eq!(bvh.len(), n);

        let post: Vec<Vec<u64>> = queries
            .iter()
            .map(|q| {
                let mut v = bvh.collect_aabb(q);
                v.sort();
                v
            })
            .collect();
        assert_eq!(pre, post, "rebuild changed query results");
        // After full build, quality_ratio == 1.0 exactly.
        assert!((bvh.quality_ratio() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn rebuild_recovers_quality_after_churn() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFEED);
        let n = 600usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        let mut bvh = BvhIndex::new();
        bvh.build_from(prims.iter().copied()).unwrap();
        let baseline = bvh.sah_cost();

        // Many updates with much larger AABBs — should degrade the tree.
        for (uid, _) in &prims {
            let big = rand_aabb(&mut rng, 200.0, 50.0);
            bvh.update(*uid, big).unwrap();
        }
        let after_churn = bvh.sah_cost();
        // Degradation happened: cost should have changed (typically grown).
        assert!(after_churn != baseline);

        bvh.rebuild();
        bvh.assert_valid();
        assert!((bvh.quality_ratio() - 1.0).abs() < 1e-4);
    }

    #[test]
    fn build_with_duplicate_input_errors() {
        let prims = vec![
            (1u64, aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0))),
            (1u64, aabb((2.0, 2.0, 2.0), (3.0, 3.0, 3.0))),
        ];
        let mut bvh = BvhIndex::new();
        assert_eq!(
            bvh.build_from(prims.into_iter()),
            Err(IndexError::DuplicateUid(1))
        );
        assert!(bvh.is_empty());
    }

    #[test]
    fn stats_are_consistent() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xCAFE);
        let n = 100usize;
        let prims: Vec<(u64, Aabb)> = (0..n)
            .map(|i| (i as u64, rand_aabb(&mut rng, 100.0, 5.0)))
            .collect();
        let mut bvh = BvhIndex::new();
        bvh.build_from(prims.iter().copied()).unwrap();
        let s = bvh.stats();
        assert_eq!(s.instances, n);
        assert!(s.nodes > 0);
        assert!(s.max_depth > 0);
        assert!(s.mean_depth > 0.0);
        assert!(s.mean_depth as usize <= s.max_depth);
        let root = s.root_aabb.expect("non-empty");
        for (_, a) in &prims {
            assert!(root.contains_aabb(a));
        }
    }

    /// Coincident centroids — every primitive AABB has the same center,
    /// so binned SAH degenerates. The fallback paths must keep correctness.
    #[test]
    fn coincident_centroids_correctness() {
        let mut bvh = BvhIndex::new();
        let mut brute = BruteIndex::new();
        // 20 prims all centered at origin, varying sizes
        for i in 0..20u64 {
            let s = 0.5 + (i as f32) * 0.1;
            let a = aabb((-s, -s, -s), (s, s, s));
            bvh.insert(i, a).unwrap();
            brute.insert(i, a).unwrap();
        }
        bvh.assert_valid();

        let q = aabb((-0.05, -0.05, -0.05), (0.05, 0.05, 0.05));
        let mut got = bvh.collect_aabb(&q);
        let mut want = brute.collect_aabb(&q);
        got.sort();
        want.sort();
        assert_eq!(got, want);
    }
}
