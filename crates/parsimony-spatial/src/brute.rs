//! `BruteIndex` — the reference [`SpatialIndex`] implementation.
//!
//! Stores `(uid, aabb)` in parallel `Vec`s with a `HashMap<uid, idx>`
//! for O(1) point-lookup. Every query scans the entire instance list.
//! Used as the correctness oracle for the optimized indices in
//! Phases 1b (`BvhIndex`) and 1c (`HierGridIndex`).
//!
//! Asymptotics: O(1) insert / O(1) amortized remove (swap-remove) /
//! O(1) update / O(n) query. Memory is `~32 B` per instance.

use std::collections::HashMap;

use crate::aabb::Aabb;
use crate::index::SpatialIndex;
use crate::query::{IndexError, IndexStats, Sphere};

#[derive(Debug, Default, Clone)]
pub struct BruteIndex {
    aabbs: Vec<Aabb>,
    uids: Vec<u64>,
    by_uid: HashMap<u64, usize>,
}

impl BruteIndex {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn with_capacity(cap: usize) -> Self {
        Self {
            aabbs: Vec::with_capacity(cap),
            uids: Vec::with_capacity(cap),
            by_uid: HashMap::with_capacity(cap),
        }
    }

    /// Borrow the stored AABB for `uid`, mainly for tests.
    pub fn get(&self, uid: u64) -> Option<&Aabb> {
        self.by_uid.get(&uid).map(|&i| &self.aabbs[i])
    }

    /// Iterate every `(uid, aabb)` in insertion-modulo-swap order.
    pub fn iter(&self) -> impl Iterator<Item = (u64, &Aabb)> {
        self.uids.iter().copied().zip(self.aabbs.iter())
    }
}

impl SpatialIndex for BruteIndex {
    fn insert(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError> {
        if self.by_uid.contains_key(&uid) {
            return Err(IndexError::DuplicateUid(uid));
        }
        let idx = self.aabbs.len();
        self.aabbs.push(aabb);
        self.uids.push(uid);
        self.by_uid.insert(uid, idx);
        Ok(())
    }

    fn remove(&mut self, uid: u64) -> Result<(), IndexError> {
        let idx = self.by_uid.remove(&uid).ok_or(IndexError::NotFound(uid))?;
        let last = self.aabbs.len() - 1;
        if idx != last {
            let moved_uid = self.uids[last];
            self.aabbs.swap(idx, last);
            self.uids.swap(idx, last);
            self.by_uid.insert(moved_uid, idx);
        }
        self.aabbs.pop();
        self.uids.pop();
        Ok(())
    }

    fn update(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError> {
        let idx = *self.by_uid.get(&uid).ok_or(IndexError::NotFound(uid))?;
        self.aabbs[idx] = aabb;
        Ok(())
    }

    fn query_aabb<F: FnMut(u64)>(&self, q: &Aabb, mut visit: F) {
        for (i, aabb) in self.aabbs.iter().enumerate() {
            if aabb.intersects(q) {
                visit(self.uids[i]);
            }
        }
    }

    fn query_sphere<F: FnMut(u64)>(&self, q: &Sphere, mut visit: F) {
        for (i, aabb) in self.aabbs.iter().enumerate() {
            if q.intersects_aabb(aabb) {
                visit(self.uids[i]);
            }
        }
    }

    fn len(&self) -> usize {
        self.aabbs.len()
    }

    fn stats(&self) -> IndexStats {
        let root = if self.aabbs.is_empty() {
            None
        } else {
            let mut acc = self.aabbs[0];
            for b in &self.aabbs[1..] {
                acc = acc.union(b);
            }
            Some(acc)
        };
        let aabb_bytes = self.aabbs.capacity() * std::mem::size_of::<Aabb>();
        let uid_bytes = self.uids.capacity() * std::mem::size_of::<u64>();
        let map_bytes =
            self.by_uid.capacity() * (std::mem::size_of::<u64>() + std::mem::size_of::<usize>());
        IndexStats {
            instances: self.aabbs.len(),
            nodes: 0,
            max_depth: 0,
            mean_depth: 0.0,
            root_aabb: root,
            memory_bytes: aabb_bytes + uid_bytes + map_bytes,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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
    fn new_is_empty() {
        let idx = BruteIndex::new();
        assert_eq!(idx.len(), 0);
        assert!(idx.is_empty());
        assert!(idx.stats().root_aabb.is_none());
    }

    #[test]
    fn insert_then_query_finds_self() {
        let mut idx = BruteIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        idx.insert(7, a).unwrap();
        assert_eq!(idx.collect_aabb(&a), vec![7]);
    }

    #[test]
    fn duplicate_uid_errors() {
        let mut idx = BruteIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        idx.insert(1, a).unwrap();
        assert_eq!(idx.insert(1, a), Err(IndexError::DuplicateUid(1)));
        assert_eq!(idx.len(), 1);
    }

    #[test]
    fn remove_unknown_errors() {
        let mut idx = BruteIndex::new();
        assert_eq!(idx.remove(42), Err(IndexError::NotFound(42)));
    }

    #[test]
    fn update_unknown_errors() {
        let mut idx = BruteIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        assert_eq!(idx.update(42, a), Err(IndexError::NotFound(42)));
    }

    #[test]
    fn remove_preserves_other_uids() {
        let mut idx = BruteIndex::new();
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        let n = 200usize;
        let aabbs: Vec<Aabb> = (0..n).map(|_| rand_aabb(&mut rng, 100.0, 10.0)).collect();
        for (uid, a) in aabbs.iter().enumerate() {
            idx.insert(uid as u64, *a).unwrap();
        }
        assert_eq!(idx.len(), n);

        // remove every other uid
        for uid in (0..n).step_by(2) {
            idx.remove(uid as u64).unwrap();
        }
        assert_eq!(idx.len(), n / 2);

        // each survivor is queryable at its own stored AABB
        for uid in (1..n).step_by(2) {
            assert_eq!(idx.get(uid as u64), Some(&aabbs[uid]));
            let hits = idx.collect_aabb(&aabbs[uid]);
            assert!(hits.contains(&(uid as u64)));
        }

        // each removed UID is gone — its old AABB no longer reports it
        for uid in (0..n).step_by(2) {
            assert!(idx.get(uid as u64).is_none());
            let hits = idx.collect_aabb(&aabbs[uid]);
            assert!(!hits.contains(&(uid as u64)));
        }
    }

    #[test]
    fn update_moves_in_query() {
        let mut idx = BruteIndex::new();
        let a = aabb((0.0, 0.0, 0.0), (1.0, 1.0, 1.0));
        let b = aabb((100.0, 100.0, 100.0), (101.0, 101.0, 101.0));
        idx.insert(5, a).unwrap();
        assert_eq!(idx.collect_aabb(&a), vec![5]);
        assert!(idx.collect_aabb(&b).is_empty());
        idx.update(5, b).unwrap();
        assert!(idx.collect_aabb(&a).is_empty());
        assert_eq!(idx.collect_aabb(&b), vec![5]);
    }

    /// Compare `query_aabb` to a hand-written O(n²) reference on randomized
    /// AABBs. Any false positive or false negative fails the test.
    #[test]
    fn query_aabb_agrees_with_naive_oracle() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFADE);
        let mut idx = BruteIndex::new();
        let n = 300usize;
        let aabbs: Vec<Aabb> = (0..n).map(|_| rand_aabb(&mut rng, 100.0, 8.0)).collect();
        for (uid, a) in aabbs.iter().enumerate() {
            idx.insert(uid as u64, *a).unwrap();
        }
        for _ in 0..50 {
            let q = rand_aabb(&mut rng, 100.0, 20.0);
            let mut got: Vec<u64> = idx.collect_aabb(&q);
            got.sort();
            let mut want: Vec<u64> = aabbs
                .iter()
                .enumerate()
                .filter(|(_, a)| a.intersects(&q))
                .map(|(i, _)| i as u64)
                .collect();
            want.sort();
            assert_eq!(got, want);
        }
    }

    /// Same but for sphere queries.
    #[test]
    fn query_sphere_agrees_with_naive_oracle() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xBEEF);
        let mut idx = BruteIndex::new();
        let n = 500usize;
        let aabbs: Vec<Aabb> = (0..n).map(|_| rand_aabb(&mut rng, 100.0, 5.0)).collect();
        for (uid, a) in aabbs.iter().enumerate() {
            idx.insert(uid as u64, *a).unwrap();
        }
        for _ in 0..50 {
            let center = p(
                rng.gen_range(-100.0..100.0),
                rng.gen_range(-100.0..100.0),
                rng.gen_range(-100.0..100.0),
            );
            let radius: f32 = rng.gen_range(1.0..30.0);
            let q = Sphere::new(center, radius);
            let mut got: Vec<u64> = idx.collect_sphere(&q);
            got.sort();
            let mut want: Vec<u64> = aabbs
                .iter()
                .enumerate()
                .filter(|(_, a)| q.intersects_aabb(a))
                .map(|(i, _)| i as u64)
                .collect();
            want.sort();
            assert_eq!(got, want);

            // every reported UID's AABB really does overlap the sphere
            let got_set: HashSet<u64> = got.iter().copied().collect();
            for uid in &got {
                assert!(q.intersects_aabb(&aabbs[*uid as usize]));
            }
            // every unreported UID really doesn't
            for (i, a) in aabbs.iter().enumerate() {
                if !got_set.contains(&(i as u64)) {
                    assert!(!q.intersects_aabb(a));
                }
            }
        }
    }

    #[test]
    fn stats_root_aabb_covers_everything() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xD00D);
        let mut idx = BruteIndex::new();
        let aabbs: Vec<Aabb> = (0..50).map(|_| rand_aabb(&mut rng, 100.0, 5.0)).collect();
        for (uid, a) in aabbs.iter().enumerate() {
            idx.insert(uid as u64, *a).unwrap();
        }
        let root = idx.stats().root_aabb.expect("non-empty");
        for a in &aabbs {
            assert!(root.contains_aabb(a), "root AABB doesn't cover instance");
        }
    }

    /// After many random insert/remove/update calls, the internal
    /// invariants between `aabbs`, `uids`, and `by_uid` must still hold.
    #[test]
    fn invariants_under_churn() {
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xFEED);
        let mut idx = BruteIndex::new();
        let mut live: HashSet<u64> = HashSet::new();
        let mut next_uid: u64 = 0;

        for _ in 0..2000 {
            let r: f32 = rng.gen_range(0.0..1.0);
            if r < 0.5 || live.is_empty() {
                let uid = next_uid;
                next_uid += 1;
                let a = rand_aabb(&mut rng, 100.0, 5.0);
                idx.insert(uid, a).unwrap();
                live.insert(uid);
            } else if r < 0.8 {
                // remove a random live uid
                let pick = *live.iter().nth(rng.gen_range(0..live.len())).unwrap();
                idx.remove(pick).unwrap();
                live.remove(&pick);
            } else {
                // update a random live uid
                let pick = *live.iter().nth(rng.gen_range(0..live.len())).unwrap();
                let a = rand_aabb(&mut rng, 100.0, 5.0);
                idx.update(pick, a).unwrap();
            }
        }

        assert_eq!(idx.len(), live.len());
        // every live uid is queryable at its stored AABB
        for uid in &live {
            let a = *idx.get(*uid).expect("live uid present");
            assert!(idx.collect_aabb(&a).contains(uid));
        }
    }
}
