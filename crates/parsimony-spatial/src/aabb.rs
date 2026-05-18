//! Axis-aligned bounding boxes — the fundamental query and storage
//! shape used by every `SpatialIndex` implementation.

use nalgebra::{Point3, Vector3};

/// An axis-aligned bounding box in 3D.
///
/// Invariant when [`is_valid`](Self::is_valid) is true: `min <= max`
/// componentwise. The "empty" AABB has `min` strictly greater than
/// `max` on every axis (so it intersects nothing); this is the neutral
/// element for [`union`](Self::union).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: Point3<f32>,
    pub max: Point3<f32>,
}

impl Aabb {
    /// AABB from explicit corners. Caller is responsible for the
    /// `min <= max` invariant; use [`from_points`](Self::from_points)
    /// otherwise.
    #[inline]
    pub fn new(min: Point3<f32>, max: Point3<f32>) -> Self {
        Self { min, max }
    }

    /// Smallest AABB enclosing two arbitrary points, regardless of order.
    #[inline]
    pub fn from_points(a: Point3<f32>, b: Point3<f32>) -> Self {
        Self {
            min: Point3::new(a.x.min(b.x), a.y.min(b.y), a.z.min(b.z)),
            max: Point3::new(a.x.max(b.x), a.y.max(b.y), a.z.max(b.z)),
        }
    }

    /// AABB tightly enclosing a sphere.
    #[inline]
    pub fn from_sphere(center: Point3<f32>, radius: f32) -> Self {
        let r = Vector3::new(radius, radius, radius);
        Self {
            min: center - r,
            max: center + r,
        }
    }

    /// The neutral element for [`union`](Self::union): `min = +∞`,
    /// `max = -∞`. Intersects nothing; absorbed by anything.
    #[inline]
    pub fn empty() -> Self {
        Self {
            min: Point3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY),
            max: Point3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY),
        }
    }

    /// True iff `min <= max` componentwise.
    #[inline]
    pub fn is_valid(&self) -> bool {
        self.min.x <= self.max.x
            && self.min.y <= self.max.y
            && self.min.z <= self.max.z
    }

    #[inline]
    pub fn center(&self) -> Point3<f32> {
        Point3::from((self.min.coords + self.max.coords) * 0.5)
    }

    #[inline]
    pub fn extents(&self) -> Vector3<f32> {
        self.max - self.min
    }

    #[inline]
    pub fn half_extents(&self) -> Vector3<f32> {
        self.extents() * 0.5
    }

    #[inline]
    pub fn volume(&self) -> f32 {
        let e = self.extents();
        if e.x < 0.0 || e.y < 0.0 || e.z < 0.0 {
            0.0
        } else {
            e.x * e.y * e.z
        }
    }

    /// Surface area. Used by SAH BVH builds in Phase 1b.
    #[inline]
    pub fn surface_area(&self) -> f32 {
        let e = self.extents();
        if e.x < 0.0 || e.y < 0.0 || e.z < 0.0 {
            0.0
        } else {
            2.0 * (e.x * e.y + e.y * e.z + e.z * e.x)
        }
    }

    /// True iff `p` lies on or inside this AABB.
    #[inline]
    pub fn contains_point(&self, p: Point3<f32>) -> bool {
        self.min.x <= p.x
            && p.x <= self.max.x
            && self.min.y <= p.y
            && p.y <= self.max.y
            && self.min.z <= p.z
            && p.z <= self.max.z
    }

    /// True iff `other` is fully contained within `self`.
    #[inline]
    pub fn contains_aabb(&self, other: &Aabb) -> bool {
        self.min.x <= other.min.x
            && other.max.x <= self.max.x
            && self.min.y <= other.min.y
            && other.max.y <= self.max.y
            && self.min.z <= other.min.z
            && other.max.z <= self.max.z
    }

    /// True iff the two AABBs share any volume (touching counts).
    #[inline]
    pub fn intersects(&self, other: &Aabb) -> bool {
        self.min.x <= other.max.x
            && other.min.x <= self.max.x
            && self.min.y <= other.max.y
            && other.min.y <= self.max.y
            && self.min.z <= other.max.z
            && other.min.z <= self.max.z
    }

    /// Smallest AABB enclosing both.
    #[inline]
    pub fn union(&self, other: &Aabb) -> Aabb {
        Aabb::new(
            Point3::new(
                self.min.x.min(other.min.x),
                self.min.y.min(other.min.y),
                self.min.z.min(other.min.z),
            ),
            Point3::new(
                self.max.x.max(other.max.x),
                self.max.y.max(other.max.y),
                self.max.z.max(other.max.z),
            ),
        )
    }

    /// Smallest AABB enclosing self and `p`.
    #[inline]
    pub fn expand_point(&self, p: Point3<f32>) -> Aabb {
        Aabb::new(
            Point3::new(self.min.x.min(p.x), self.min.y.min(p.y), self.min.z.min(p.z)),
            Point3::new(self.max.x.max(p.x), self.max.y.max(p.y), self.max.z.max(p.z)),
        )
    }

    /// Expand outward by `margin` on every face.
    #[inline]
    pub fn dilate(&self, margin: f32) -> Aabb {
        let m = Vector3::new(margin, margin, margin);
        Aabb::new(self.min - m, self.max + m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn p(x: f32, y: f32, z: f32) -> Point3<f32> {
        Point3::new(x, y, z)
    }

    #[test]
    fn from_points_orders_corners() {
        let a = Aabb::from_points(p(3.0, -1.0, 5.0), p(-2.0, 4.0, 0.0));
        assert_eq!(a.min, p(-2.0, -1.0, 0.0));
        assert_eq!(a.max, p(3.0, 4.0, 5.0));
        assert!(a.is_valid());
    }

    #[test]
    fn from_sphere_is_centered() {
        let a = Aabb::from_sphere(p(1.0, 2.0, 3.0), 0.5);
        assert_eq!(a.min, p(0.5, 1.5, 2.5));
        assert_eq!(a.max, p(1.5, 2.5, 3.5));
    }

    #[test]
    fn empty_is_absorbed_by_union() {
        let e = Aabb::empty();
        assert!(!e.is_valid());
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(1.0, 1.0, 1.0));
        assert_eq!(e.union(&a), a);
        assert_eq!(a.union(&e), a);
    }

    #[test]
    fn contains_point_on_boundary_inclusive() {
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(1.0, 1.0, 1.0));
        assert!(a.contains_point(p(0.0, 0.0, 0.0)));
        assert!(a.contains_point(p(1.0, 1.0, 1.0)));
        assert!(a.contains_point(p(0.5, 0.5, 0.5)));
        assert!(!a.contains_point(p(1.1, 0.5, 0.5)));
        assert!(!a.contains_point(p(-0.1, 0.5, 0.5)));
    }

    #[test]
    fn intersects_cases() {
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(1.0, 1.0, 1.0));
        // overlapping
        assert!(a.intersects(&Aabb::new(p(0.5, 0.5, 0.5), p(2.0, 2.0, 2.0))));
        // touching faces (inclusive)
        assert!(a.intersects(&Aabb::new(p(1.0, 0.0, 0.0), p(2.0, 1.0, 1.0))));
        // fully disjoint
        assert!(!a.intersects(&Aabb::new(p(2.0, 0.0, 0.0), p(3.0, 1.0, 1.0))));
        // disjoint along y only
        assert!(!a.intersects(&Aabb::new(p(0.0, 2.0, 0.0), p(1.0, 3.0, 1.0))));
    }

    #[test]
    fn contains_aabb_strict_and_loose() {
        let big = Aabb::new(p(0.0, 0.0, 0.0), p(10.0, 10.0, 10.0));
        assert!(big.contains_aabb(&Aabb::new(p(1.0, 1.0, 1.0), p(2.0, 2.0, 2.0))));
        assert!(big.contains_aabb(&big));
        assert!(!big.contains_aabb(&Aabb::new(p(-1.0, 0.0, 0.0), p(5.0, 5.0, 5.0))));
    }

    #[test]
    fn union_and_expand_point() {
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(1.0, 1.0, 1.0));
        let b = Aabb::new(p(2.0, 3.0, 4.0), p(5.0, 6.0, 7.0));
        assert_eq!(a.union(&b), Aabb::new(p(0.0, 0.0, 0.0), p(5.0, 6.0, 7.0)));
        assert_eq!(
            a.expand_point(p(-1.0, 2.0, 0.5)),
            Aabb::new(p(-1.0, 0.0, 0.0), p(1.0, 2.0, 1.0))
        );
    }

    #[test]
    fn volume_and_surface_area() {
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(2.0, 3.0, 4.0));
        assert_eq!(a.volume(), 24.0);
        assert_eq!(a.surface_area(), 2.0 * (6.0 + 12.0 + 8.0));
    }

    #[test]
    fn dilate_grows_uniformly() {
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(1.0, 1.0, 1.0));
        let d = a.dilate(0.5);
        assert_eq!(d.min, p(-0.5, -0.5, -0.5));
        assert_eq!(d.max, p(1.5, 1.5, 1.5));
    }

    #[test]
    fn center_and_extents() {
        let a = Aabb::new(p(0.0, 0.0, 0.0), p(2.0, 4.0, 6.0));
        assert_eq!(a.center(), p(1.0, 2.0, 3.0));
        assert_eq!(a.extents(), Vector3::new(2.0, 4.0, 6.0));
        assert_eq!(a.half_extents(), Vector3::new(1.0, 2.0, 3.0));
    }
}
