//! Integration: the E. coli capsule recipe packs a confined rod nucleoid.
use std::path::PathBuf;

use parsimony_core::{GreedyRandomPlacer, PlacerConfig, Recipe};

fn recipe_path() -> Option<PathBuf> {
    let p = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../examples/recipes/ecoli_nucleoid.json");
    p.exists().then_some(p)
}

#[test]
fn ecoli_nucleoid_packs_inside_the_capsule() {
    let Some(path) = recipe_path() else { return };
    // Needs the K-12 gene CSV (Task 7) + the 1BNA/1hqm/1aon meshes.
    let recipe = Recipe::from_file(&path).expect("load recipe");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xE_C0_11);
    let placed = out.snapshot.placements.len();
    assert!(placed > 10_000, "expected a populated cell, got {placed} placements");

    // Capsule from the recipe: a=(-7000,0,0) b=(7000,0,0) r=4000.
    let (a, b, r) = (
        nalgebra::Point3::new(-7000.0_f32, 0.0, 0.0),
        nalgebra::Point3::new(7000.0_f32, 0.0, 0.0),
        4000.0_f32,
    );
    let dist_to_seg = |p: &nalgebra::Point3<f32>| {
        let ab = b - a;
        let t = ((p - a).dot(&ab) / ab.norm_squared()).clamp(0.0, 1.0);
        (p - (a + ab * t)).norm()
    };
    // Every placement is within the capsule (allow one bead radius of slack).
    let outside = out
        .snapshot
        .placements
        .iter()
        .filter(|pl| dist_to_seg(&pl.position) > r + 15.0)
        .count();
    assert!(outside == 0, "{outside} placements left the capsule");
}
