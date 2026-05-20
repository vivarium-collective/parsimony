//! Sparse occupancy octree — the content-scaled placement engine.
//!
//! The clearance grid ([`crate::clearance_grid`]) answers "where can this land?"
//! by enumerating empty *volume* (a dense cell per `cell_size³` of the box, plus
//! a derived free-cell list per ingredient). At whole-cell scale that's tens of
//! millions of cells and hundreds of derived lists — cost set by empty space,
//! not by what's placed.
//!
//! This octree answers the same question but its cost scales with *content*:
//! it starts as one empty root and subdivides **only near placed geometry**.
//! Big empty regions stay one coarse leaf; the interior of a placed blob is a
//! few coarse `full` leaves; only the frontier between occupied and free space
//! is refined down to `min_cell`. Each node caches the free volume in its
//! subtree, so sampling descends *weighted toward free space* and lands in a
//! known-free leaf — preserving the grid's near-direct placement at high
//! density (where blind rejection would need thousands of tries) without the
//! grid's volume-scaled price.
//!
//! Three jobs, one structure, updated incrementally as we place:
//! - [`OccupancyOctree::insert_sphere`] — mark a proxy occupied.
//! - [`OccupancyOctree::overlaps`] — exact proxy-vs-proxy collision test.
//! - [`OccupancyOctree::sample_free`] — free-volume-weighted point sample.

use nalgebra::Point3;
use rand::Rng;

use parsimony_spatial::Aabb;

/// Sentinel for "no children" — the node is a leaf.
const NO_CHILDREN: u32 = u32::MAX;
/// Free-volume weight given to a partially-occupied min-cell leaf, as a
/// fraction of its volume. Keeps such leaves sampled rarely (they're mostly
/// full) but not never (a free corner may remain); the exact in-leaf rejection
/// + the caller's collision check keep correctness regardless of this estimate.
const PARTIAL_FREE_FRAC: f32 = 0.15;
/// In-leaf sample attempts before a partial leaf reports itself unsamplable.
const PARTIAL_SAMPLE_TRIES: usize = 8;
/// Weighted child picks before an internal node reports its subtree unsamplable
/// (a child's optimistic free estimate can dead-end; retry a few others).
const CHILD_RETRIES: usize = 6;

struct Node {
    bounds: Aabb,
    /// Index of the first of 8 contiguous children, or [`NO_CHILDREN`] if leaf.
    children: u32,
    /// The node's region is entirely covered by inserted spheres.
    full: bool,
    /// Estimated free volume in this node's subtree (drives weighted sampling).
    free: f32,
    /// Inserted spheres overlapping this leaf — only populated on partial
    /// `min_cell` leaves (the occupied/free frontier).
    proxies: Vec<(Point3<f32>, f32)>,
}

/// A sparse occupancy octree over a compartment's bounding box.
pub(crate) struct OccupancyOctree {
    nodes: Vec<Node>,
    /// Start indices of freed 8-child blocks, available for reuse. Children are
    /// always allocated/freed as a contiguous block of 8, so a freed block
    /// re-serves any later subdivision — the arena tracks peak *live* nodes
    /// rather than every node ever allocated (collapses would otherwise leak).
    free_blocks: Vec<u32>,
    root: u32,
    min_cell: f32,
}

impl OccupancyOctree {
    /// Empty octree over `bounds`; the frontier is refined down to `~min_cell`.
    pub fn new(bounds: Aabb, min_cell: f32) -> Self {
        let free = aabb_volume(&bounds);
        let root = Node {
            bounds,
            children: NO_CHILDREN,
            full: false,
            free,
            proxies: Vec::new(),
        };
        Self {
            nodes: vec![root],
            free_blocks: Vec::new(),
            root: 0,
            min_cell: min_cell.max(1e-3),
        }
    }

    /// Fraction of the root volume still free (diagnostic).
    #[allow(dead_code)]
    pub fn free_fraction(&self) -> f32 {
        let root = &self.nodes[self.root as usize];
        let vol = aabb_volume(&root.bounds);
        if vol <= 0.0 {
            0.0
        } else {
            (root.free / vol).clamp(0.0, 1.0)
        }
    }

    /// Number of nodes in the arena (diagnostic — tracks content scaling).
    #[allow(dead_code)]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Mark the region of a sphere occupied.
    pub fn insert_sphere(&mut self, center: Point3<f32>, radius: f32) {
        self.insert(self.root, center, radius);
    }

    /// Insert into the subtree at `idx`; returns the subtree's new free volume.
    fn insert(&mut self, idx: u32, c: Point3<f32>, r: f32) -> f32 {
        let bounds = self.nodes[idx as usize].bounds;
        if self.nodes[idx as usize].full {
            return 0.0;
        }
        if !sphere_intersects_aabb(&bounds, c, r) {
            return self.nodes[idx as usize].free; // untouched
        }
        if aabb_inside_sphere(&bounds, c, r) {
            // Region now entirely covered — collapse to a single full leaf,
            // reclaiming any subtree's slots.
            self.free_children(idx);
            let n = &mut self.nodes[idx as usize];
            n.full = true;
            n.free = 0.0;
            n.proxies = Vec::new();
            return 0.0;
        }
        // Partial overlap. At the resolution floor, become a frontier leaf
        // holding the proxy; otherwise refine and recurse.
        if aabb_min_dim(&bounds) <= self.min_cell {
            let center = bounds.center();
            let n = &mut self.nodes[idx as usize];
            n.proxies.push((c, r));
            let covered = n
                .proxies
                .iter()
                .any(|(pc, pr)| (center - pc).norm() <= *pr);
            n.free = if covered {
                0.0
            } else {
                aabb_volume(&bounds) * PARTIAL_FREE_FRAC
            };
            return n.free;
        }
        if self.nodes[idx as usize].children == NO_CHILDREN {
            self.subdivide(idx);
        }
        let first = self.nodes[idx as usize].children;
        let mut free = 0.0;
        for k in 0..8 {
            free += self.insert(first + k, c, r);
        }
        if free <= 0.0 {
            // All children fully covered — collapse this node too, reclaiming
            // the now-redundant child block.
            self.free_children(idx);
            let n = &mut self.nodes[idx as usize];
            n.full = true;
            n.free = 0.0;
            return 0.0;
        }
        self.nodes[idx as usize].free = free;
        free
    }

    /// Split a leaf into 8 empty octant children, reusing a freed block if one
    /// is available.
    fn subdivide(&mut self, idx: u32) {
        let b = self.nodes[idx as usize].bounds;
        let mid = b.center();
        let first = self.alloc_block();
        for k in 0..8usize {
            let ob = octant(&b, &mid, k);
            let free = aabb_volume(&ob);
            let child = &mut self.nodes[first as usize + k];
            child.bounds = ob;
            child.children = NO_CHILDREN;
            child.full = false;
            child.free = free;
            child.proxies.clear();
        }
        self.nodes[idx as usize].children = first;
    }

    /// Reserve a contiguous block of 8 child slots — a recycled freed block if
    /// one exists, else a fresh extension of the arena.
    fn alloc_block(&mut self) -> u32 {
        if let Some(first) = self.free_blocks.pop() {
            return first;
        }
        let first = self.nodes.len() as u32;
        for _ in 0..8 {
            self.nodes.push(Node {
                bounds: Aabb::empty(),
                children: NO_CHILDREN,
                full: false,
                free: 0.0,
                proxies: Vec::new(),
            });
        }
        first
    }

    /// Recursively free the subtree below `idx` back to the block pool and make
    /// `idx` a leaf. Releases each freed node's proxy heap.
    fn free_children(&mut self, idx: u32) {
        let first = self.nodes[idx as usize].children;
        if first == NO_CHILDREN {
            return;
        }
        for k in 0..8u32 {
            self.free_children(first + k);
            let child = &mut self.nodes[(first + k) as usize];
            child.children = NO_CHILDREN;
            child.full = false;
            child.free = 0.0;
            child.proxies = Vec::new();
        }
        self.free_blocks.push(first);
        self.nodes[idx as usize].children = NO_CHILDREN;
    }

    /// True if `sphere` overlaps any inserted sphere.
    pub fn overlaps(&self, center: Point3<f32>, radius: f32) -> bool {
        self.overlaps_node(self.root, center, radius)
    }

    fn overlaps_node(&self, idx: u32, c: Point3<f32>, r: f32) -> bool {
        let n = &self.nodes[idx as usize];
        if !sphere_intersects_aabb(&n.bounds, c, r) {
            return false;
        }
        if n.full {
            return true;
        }
        if n.children == NO_CHILDREN {
            return n
                .proxies
                .iter()
                .any(|(pc, pr)| (c - pc).norm() <= r + *pr);
        }
        (0..8).any(|k| self.overlaps_node(n.children + k, c, r))
    }

    /// Sample a point biased toward free space: descend choosing children by
    /// free volume, land in a known-free leaf. `None` if no free space remains
    /// (the compartment is saturated). The point is free of inserted geometry;
    /// the caller still applies compartment containment + its own proxy check.
    pub fn sample_free<R: Rng>(&self, rng: &mut R) -> Option<Point3<f32>> {
        self.sample_node(self.root, rng)
    }

    fn sample_node<R: Rng>(&self, idx: u32, rng: &mut R) -> Option<Point3<f32>> {
        let n = &self.nodes[idx as usize];
        if n.full || n.free <= 0.0 {
            return None;
        }
        if n.children == NO_CHILDREN {
            if n.proxies.is_empty() {
                return Some(uniform_in(&n.bounds, rng));
            }
            for _ in 0..PARTIAL_SAMPLE_TRIES {
                let p = uniform_in(&n.bounds, rng);
                if !n.proxies.iter().any(|(pc, pr)| (p - pc).norm() <= *pr) {
                    return Some(p);
                }
            }
            return None;
        }
        let total = n.free;
        for _ in 0..CHILD_RETRIES {
            let mut x = rng.gen_range(0.0..total);
            let mut child = n.children + 7; // fallback (FP slack)
            for k in 0..8 {
                let cf = self.nodes[(n.children + k) as usize].free;
                if x < cf {
                    child = n.children + k;
                    break;
                }
                x -= cf;
            }
            if let Some(p) = self.sample_node(child, rng) {
                return Some(p);
            }
        }
        None
    }
}

// ---------- geometry helpers ----------

fn aabb_volume(b: &Aabb) -> f32 {
    ((b.max.x - b.min.x) * (b.max.y - b.min.y) * (b.max.z - b.min.z)).max(0.0)
}

fn aabb_min_dim(b: &Aabb) -> f32 {
    (b.max.x - b.min.x)
        .min(b.max.y - b.min.y)
        .min(b.max.z - b.min.z)
}

/// The `k`-th octant of `b` split at `mid` (bit 0 = x, 1 = y, 2 = z; 0 = low).
fn octant(b: &Aabb, mid: &Point3<f32>, k: usize) -> Aabb {
    let (xlo, xhi) = if k & 1 == 0 { (b.min.x, mid.x) } else { (mid.x, b.max.x) };
    let (ylo, yhi) = if k & 2 == 0 { (b.min.y, mid.y) } else { (mid.y, b.max.y) };
    let (zlo, zhi) = if k & 4 == 0 { (b.min.z, mid.z) } else { (mid.z, b.max.z) };
    Aabb::new(Point3::new(xlo, ylo, zlo), Point3::new(xhi, yhi, zhi))
}

fn sphere_intersects_aabb(b: &Aabb, c: Point3<f32>, r: f32) -> bool {
    let nx = c.x.clamp(b.min.x, b.max.x);
    let ny = c.y.clamp(b.min.y, b.max.y);
    let nz = c.z.clamp(b.min.z, b.max.z);
    let (dx, dy, dz) = (c.x - nx, c.y - ny, c.z - nz);
    dx * dx + dy * dy + dz * dz <= r * r
}

fn aabb_inside_sphere(b: &Aabb, c: Point3<f32>, r: f32) -> bool {
    // Farthest corner of the box from c is within r.
    let fx = (c.x - b.min.x).abs().max((c.x - b.max.x).abs());
    let fy = (c.y - b.min.y).abs().max((c.y - b.max.y).abs());
    let fz = (c.z - b.min.z).abs().max((c.z - b.max.z).abs());
    fx * fx + fy * fy + fz * fz <= r * r
}

fn uniform_in<R: Rng>(b: &Aabb, rng: &mut R) -> Point3<f32> {
    Point3::new(
        sample_axis(rng, b.min.x, b.max.x),
        sample_axis(rng, b.min.y, b.max.y),
        sample_axis(rng, b.min.z, b.max.z),
    )
}

fn sample_axis<R: Rng>(rng: &mut R, lo: f32, hi: f32) -> f32 {
    if hi > lo {
        rng.gen_range(lo..hi)
    } else {
        lo
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    fn box_(half: f32) -> Aabb {
        Aabb::new(Point3::new(-half, -half, -half), Point3::new(half, half, half))
    }

    #[test]
    fn empty_tree_samples_inside_bounds() {
        let oct = OccupancyOctree::new(box_(100.0), 10.0);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(1);
        assert!((oct.free_fraction() - 1.0).abs() < 1e-4);
        for _ in 0..200 {
            let p = oct.sample_free(&mut rng).expect("free");
            assert!(p.x >= -100.0 && p.x <= 100.0);
            assert!(p.y >= -100.0 && p.y <= 100.0);
            assert!(p.z >= -100.0 && p.z <= 100.0);
        }
    }

    #[test]
    fn insert_reduces_free_and_overlaps_detects() {
        let mut oct = OccupancyOctree::new(box_(100.0), 8.0);
        let before = oct.free_fraction();
        oct.insert_sphere(Point3::new(0.0, 0.0, 0.0), 30.0);
        let after = oct.free_fraction();
        assert!(after < before, "free should drop ({before} -> {after})");
        // Overlap math: a query at the centre overlaps; far away doesn't.
        assert!(oct.overlaps(Point3::new(0.0, 0.0, 0.0), 1.0));
        assert!(oct.overlaps(Point3::new(35.0, 0.0, 0.0), 10.0)); // 35 < 30+10
        assert!(!oct.overlaps(Point3::new(90.0, 90.0, 90.0), 5.0));
    }

    #[test]
    fn sample_free_avoids_occupied_region() {
        // Occupy a corner sphere; sampled points must not land inside it, even
        // though it's a few % of the box volume (uniform sampling would).
        let mut oct = OccupancyOctree::new(box_(100.0), 6.0);
        let (oc, or_) = (Point3::new(-55.0, -55.0, -55.0), 40.0);
        oct.insert_sphere(oc, or_);
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(7);
        let mut inside = 0;
        let n = 2000;
        for _ in 0..n {
            if let Some(p) = oct.sample_free(&mut rng) {
                if (p - oc).norm() <= or_ {
                    inside += 1;
                }
            }
        }
        assert!(inside <= n / 100, "sampled inside occupied sphere {inside}/{n} times");
    }

    #[test]
    fn fully_occupied_has_no_free_space() {
        let mut oct = OccupancyOctree::new(box_(50.0), 8.0);
        // A sphere that encloses the whole box (corner distance = 50*sqrt3 ≈ 86.6).
        oct.insert_sphere(Point3::new(0.0, 0.0, 0.0), 90.0);
        assert!(oct.free_fraction() < 1e-3, "box should be saturated");
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(3);
        assert!(oct.sample_free(&mut rng).is_none(), "no free space to sample");
        // A single full root leaf — collapse kept it from subdividing.
        assert!(oct.node_count() <= 9, "fully-covered box must not refine");
    }

    #[test]
    fn deterministic_for_seed() {
        let mut build = || {
            let mut oct = OccupancyOctree::new(box_(100.0), 8.0);
            oct.insert_sphere(Point3::new(20.0, 0.0, -10.0), 25.0);
            let mut rng = Xoshiro256PlusPlus::seed_from_u64(42);
            (0..50)
                .map(|_| oct.sample_free(&mut rng).unwrap())
                .collect::<Vec<_>>()
        };
        let a = build();
        let b = build();
        assert_eq!(a, b);
    }

    #[test]
    fn node_count_scales_with_content_not_volume() {
        // The same single placement in a tiny box and a 64x-bigger box should
        // refine to a similar node count — cost tracks content, not volume.
        let mut small = OccupancyOctree::new(box_(50.0), 5.0);
        small.insert_sphere(Point3::new(0.0, 0.0, 0.0), 15.0);
        let mut big = OccupancyOctree::new(box_(200.0), 5.0);
        big.insert_sphere(Point3::new(0.0, 0.0, 0.0), 15.0);
        // The big box is 64x the volume; node counts should stay the same order
        // of magnitude (the empty bulk is never subdivided).
        assert!(
            big.node_count() < small.node_count() * 4,
            "node count should not scale with empty volume (small={}, big={})",
            small.node_count(),
            big.node_count()
        );
    }
}
