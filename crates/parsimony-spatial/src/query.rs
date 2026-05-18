//! Query shapes, diagnostics, and errors shared across every
//! [`SpatialIndex`](crate::index::SpatialIndex) implementation.

use nalgebra::{Point3, Vector3};
use thiserror::Error;

use crate::aabb::Aabb;

/// A sphere query — center plus radius.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Sphere {
    pub center: Point3<f32>,
    pub radius: f32,
}

impl Sphere {
    #[inline]
    pub fn new(center: Point3<f32>, radius: f32) -> Self {
        Self { center, radius }
    }

    /// AABB tightly enclosing this sphere.
    #[inline]
    pub fn aabb(&self) -> Aabb {
        Aabb::from_sphere(self.center, self.radius)
    }

    /// Sphere-vs-AABB overlap. Uses the nearest-point-on-AABB-to-center
    /// distance: if the squared distance from the center to its
    /// projection onto the AABB is at most `r²`, they overlap.
    #[inline]
    pub fn intersects_aabb(&self, aabb: &Aabb) -> bool {
        let mut d2 = 0.0;
        // unrolled — three axes, branch-light
        let cx = self.center.x;
        if cx < aabb.min.x {
            let v = aabb.min.x - cx;
            d2 += v * v;
        } else if cx > aabb.max.x {
            let v = cx - aabb.max.x;
            d2 += v * v;
        }
        let cy = self.center.y;
        if cy < aabb.min.y {
            let v = aabb.min.y - cy;
            d2 += v * v;
        } else if cy > aabb.max.y {
            let v = cy - aabb.max.y;
            d2 += v * v;
        }
        let cz = self.center.z;
        if cz < aabb.min.z {
            let v = aabb.min.z - cz;
            d2 += v * v;
        } else if cz > aabb.max.z {
            let v = cz - aabb.max.z;
            d2 += v * v;
        }
        d2 <= self.radius * self.radius
    }
}

/// A ray — origin plus direction. `dir` is assumed unit length; callers
/// should normalize before constructing. Reserved for future ray-AABB
/// queries (visibility, raycast in-out tests); not yet wired into the
/// `SpatialIndex` trait.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Ray {
    pub origin: Point3<f32>,
    pub dir: Vector3<f32>,
}

impl Ray {
    #[inline]
    pub fn new(origin: Point3<f32>, dir: Vector3<f32>) -> Self {
        Self { origin, dir }
    }
}

/// Diagnostics about the state of a [`SpatialIndex`](crate::index::SpatialIndex).
/// Useful for profiling, benchmarking, and asserting structural invariants
/// in tests.
#[derive(Debug, Clone, Default)]
pub struct IndexStats {
    /// Number of instances stored.
    pub instances: usize,
    /// Number of internal nodes (0 for a flat structure).
    pub nodes: usize,
    /// Maximum depth (root-to-leaf), 0 for flat.
    pub max_depth: usize,
    /// Mean leaf depth (informational).
    pub mean_depth: f32,
    /// AABB enclosing every stored instance, or `None` if empty.
    pub root_aabb: Option<Aabb>,
    /// Approximate memory footprint in bytes.
    pub memory_bytes: usize,
}

/// Errors returned by the [`SpatialIndex`](crate::index::SpatialIndex) ops.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum IndexError {
    #[error("uid {0} already exists in the index")]
    DuplicateUid(u64),
    #[error("uid {0} not found in the index")]
    NotFound(u64),
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::Point3;

    fn p(x: f32, y: f32, z: f32) -> Point3<f32> {
        Point3::new(x, y, z)
    }

    #[test]
    fn sphere_aabb_matches_from_sphere() {
        let s = Sphere::new(p(1.0, 2.0, 3.0), 0.5);
        assert_eq!(s.aabb(), Aabb::from_sphere(p(1.0, 2.0, 3.0), 0.5));
    }

    #[test]
    fn sphere_intersects_aabb_center_inside() {
        let s = Sphere::new(p(0.5, 0.5, 0.5), 0.1);
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(1.0, 1.0, 1.0));
        assert!(s.intersects_aabb(&a));
    }

    #[test]
    fn sphere_intersects_aabb_touching() {
        // sphere of radius 1 centered at (2,0,0), AABB [0,1]^3 — closest
        // point on AABB is (1,0,0), distance 1, equal to radius
        let s = Sphere::new(p(2.0, 0.0, 0.0), 1.0);
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(1.0, 1.0, 1.0));
        assert!(s.intersects_aabb(&a));
    }

    #[test]
    fn sphere_misses_aabb_when_too_far() {
        let s = Sphere::new(p(5.0, 5.0, 5.0), 1.0);
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(1.0, 1.0, 1.0));
        assert!(!s.intersects_aabb(&a));
    }

    #[test]
    fn sphere_intersects_corner() {
        // sphere just close enough to touch AABB corner
        let s = Sphere::new(p(2.0, 2.0, 2.0), (3.0f32).sqrt());
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(1.0, 1.0, 1.0));
        assert!(s.intersects_aabb(&a));
        // sphere just barely missing
        let s2 = Sphere::new(p(2.0, 2.0, 2.0), (3.0f32).sqrt() - 0.01);
        assert!(!s2.intersects_aabb(&a));
    }
}
