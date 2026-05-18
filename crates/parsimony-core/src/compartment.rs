//! Compartments — the bounded volumes ingredients can be placed
//! within. v0.1 carries only axis-aligned boxes; sphere, capsule, and
//! mesh kinds come in later phases (see `docs/parsimony-design.md` §7).

use nalgebra::Point3;
use rand::Rng;
use serde::{Deserialize, Serialize};

use parsimony_spatial::Aabb;

/// Stable handle for a compartment within a [`Recipe`](crate::Recipe).
pub type CompartmentId = u32;

/// Geometric kind of a compartment. Each variant supplies its own
/// in/out test and uniform-sampling implementation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum CompartmentKind {
    Box(Aabb),
}

impl CompartmentKind {
    /// True iff `p` is inside the compartment.
    pub fn contains(&self, p: Point3<f32>) -> bool {
        match self {
            CompartmentKind::Box(aabb) => aabb.contains_point(p),
        }
    }

    /// Sample a uniform random point in the interior of the compartment.
    pub fn sample_interior<R: Rng>(&self, rng: &mut R) -> Point3<f32> {
        match self {
            CompartmentKind::Box(aabb) => Point3::new(
                rng.gen_range(aabb.min.x..aabb.max.x),
                rng.gen_range(aabb.min.y..aabb.max.y),
                rng.gen_range(aabb.min.z..aabb.max.z),
            ),
        }
    }

    /// Sample a random point such that a sphere of radius `r` placed
    /// at that point is fully inside the compartment. Returns `None`
    /// if no such point exists.
    pub fn sample_interior_for_sphere<R: Rng>(
        &self,
        r: f32,
        rng: &mut R,
    ) -> Option<Point3<f32>> {
        match self {
            CompartmentKind::Box(aabb) => {
                if aabb.max.x - aabb.min.x <= 2.0 * r
                    || aabb.max.y - aabb.min.y <= 2.0 * r
                    || aabb.max.z - aabb.min.z <= 2.0 * r
                {
                    return None;
                }
                Some(Point3::new(
                    rng.gen_range((aabb.min.x + r)..(aabb.max.x - r)),
                    rng.gen_range((aabb.min.y + r)..(aabb.max.y - r)),
                    rng.gen_range((aabb.min.z + r)..(aabb.max.z - r)),
                ))
            }
        }
    }

    pub fn aabb(&self) -> Aabb {
        match self {
            CompartmentKind::Box(aabb) => *aabb,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Compartment {
    pub name: String,
    pub kind: CompartmentKind,
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;
    use rand_xoshiro::Xoshiro256PlusPlus;

    #[test]
    fn box_contains_point() {
        let b = CompartmentKind::Box(Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 10.0),
        ));
        assert!(b.contains(Point3::new(5.0, 5.0, 5.0)));
        assert!(!b.contains(Point3::new(-1.0, 5.0, 5.0)));
    }

    #[test]
    fn sample_interior_for_sphere_stays_inside() {
        let b = CompartmentKind::Box(Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(100.0, 100.0, 100.0),
        ));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        for _ in 0..100 {
            let p = b.sample_interior_for_sphere(10.0, &mut rng).unwrap();
            assert!(p.x >= 10.0 && p.x <= 90.0);
            assert!(p.y >= 10.0 && p.y <= 90.0);
            assert!(p.z >= 10.0 && p.z <= 90.0);
        }
    }

    #[test]
    fn sample_interior_for_sphere_too_big_returns_none() {
        let b = CompartmentKind::Box(Aabb::new(
            Point3::new(0.0, 0.0, 0.0),
            Point3::new(10.0, 10.0, 10.0),
        ));
        let mut rng = Xoshiro256PlusPlus::seed_from_u64(0xC0DE);
        assert!(b.sample_interior_for_sphere(20.0, &mut rng).is_none());
    }
}
