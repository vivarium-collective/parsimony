//! End-to-end integration test: load the vendored `spheres_in_a_box.json`
//! (examples/recipes/), pack it, and validate the result. The recipe
//! lives in-repo, so this no longer depends on a cellPACK checkout.

use std::path::Path;
use std::time::Instant;

use parsimony_core::{
    write_simularium_json, write_transforms_json, GreedyRandomPlacer, PlacerConfig, Recipe,
};

const SPHERES_RECIPE: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../examples/recipes/spheres_in_a_box.json"
);

fn spheres_recipe_path() -> Option<&'static Path> {
    // Vendored in-repo — always present, so the tests below never skip.
    Some(Path::new(SPHERES_RECIPE))
}

#[test]
fn pack_runs_under_time_budget() {
    // Release-build wall time on spheres_in_a_box is ~350 ms; debug
    // build is ~20× slower due to disabled optimisations on the hot
    // clearance-update loop. Bound generously so both pass; release
    // builds get a dedicated perf assertion in the bench.
    let bound_ms = if cfg!(debug_assertions) { 15_000 } else { 2_000 };
    let Some(path) = spheres_recipe_path() else {
        eprintln!("skipping: {SPHERES_RECIPE} not found");
        return;
    };
    let recipe = Recipe::from_file(path).expect("load recipe");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let t = Instant::now();
    let out = placer.pack(0xC0DE);
    let elapsed = t.elapsed();
    assert!(
        elapsed.as_millis() < bound_ms,
        "expected <{}ms, got {:.2?} for {} placements",
        bound_ms,
        elapsed,
        out.snapshot.placements.len()
    );
}

#[test]
fn pack_places_a_reasonable_fraction() {
    let Some(path) = spheres_recipe_path() else {
        return;
    };
    let recipe = Recipe::from_file(path).expect("load recipe");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xC0DE);
    let placed = out.snapshot.placements.len();
    let requested = out.stats.requested;
    let pct = 100.0 * placed as f32 / requested as f32;
    // spheres_in_a_box has ~102% nominal volume fraction; ideal random
    // packing density of equal spheres is ~64%. Mixed sizes do better.
    // We expect somewhere around 80-95%.
    assert!(
        placed > requested * 70 / 100,
        "expected >70% placed (got {placed}/{requested} = {pct:.0}%)"
    );
}

#[test]
fn no_overlaps_in_packing() {
    let Some(path) = spheres_recipe_path() else {
        return;
    };
    let recipe = Recipe::from_file(path).expect("load recipe");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xFADE);

    // Build (radius, position) pairs.
    let mut pairs: Vec<(f32, nalgebra::Point3<f32>)> = Vec::new();
    for p in &out.snapshot.placements {
        let ing = recipe.ingredients.get_index(p.ingredient_id as usize).unwrap().1;
        let r = ing.shape.enclosing_radius();
        pairs.push((r, p.position));
    }
    let n = pairs.len();
    // O(n²) — fine for n<1000.
    for i in 0..n {
        for j in (i + 1)..n {
            let (ra, pa) = pairs[i];
            let (rb, pb) = pairs[j];
            let d2 = (pa - pb).norm_squared();
            let r_sum = ra + rb;
            // Allow a tiny epsilon for fp rounding.
            assert!(
                d2 + 1e-2 >= r_sum * r_sum,
                "instances {i} (r={ra}) and {j} (r={rb}) overlap (d={:.3}, r_sum={r_sum})",
                d2.sqrt(),
            );
        }
    }
}

#[test]
fn all_inside_bounding_box() {
    // Default PlacerConfig has `strict_bounds: true` — sphere fully
    // inside the box. Loose-bounds mode (which would allow protrusions
    // and match cellPACK's `is_point_inside_bb` default) is opt-in via
    // `PlacerConfig::strict_bounds = false`; see the placer's unit
    // tests for that path.
    let Some(path) = spheres_recipe_path() else {
        return;
    };
    let recipe = Recipe::from_file(path).expect("load recipe");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xACE5);
    let bb = recipe.bounding_box;
    for p in &out.snapshot.placements {
        let ing = recipe.ingredients.get_index(p.ingredient_id as usize).unwrap().1;
        let r = ing.shape.enclosing_radius();
        assert!(
            p.position.x - r >= bb.min.x - 1e-3
                && p.position.x + r <= bb.max.x + 1e-3
                && p.position.y - r >= bb.min.y - 1e-3
                && p.position.y + r <= bb.max.y + 1e-3
                && p.position.z - r >= bb.min.z - 1e-3
                && p.position.z + r <= bb.max.z + 1e-3,
            "placement extends outside bounding box: pos={:?}, r={r}, bb=[{:?}..{:?}]",
            p.position,
            bb.min,
            bb.max,
        );
    }
}

#[test]
fn simularium_output_is_well_formed() {
    let Some(path) = spheres_recipe_path() else {
        return;
    };
    let recipe = Recipe::from_file(path).expect("load recipe");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xBEEF);
    let v = write_simularium_json(&out.snapshot, &recipe);
    // Round-trip through serde to ensure validity.
    let s = serde_json::to_string(&v).unwrap();
    let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
    // Schema sanity checks.
    assert!(parsed["trajectoryInfo"].is_object());
    assert!(parsed["spatialData"]["bundleData"].is_array());
    let frame = &parsed["spatialData"]["bundleData"][0];
    let data = frame["data"].as_array().unwrap();
    assert_eq!(data.len(), out.snapshot.placements.len() * 11);
    // Type mapping covers all ingredient ids referenced.
    let tm = parsed["trajectoryInfo"]["typeMapping"]
        .as_object()
        .unwrap();
    let max_type_id = (recipe.ingredients.len() as i64) - 1;
    assert!(tm.contains_key(&max_type_id.to_string()));
}

#[test]
fn transforms_output_is_well_formed() {
    let Some(path) = spheres_recipe_path() else {
        return;
    };
    let recipe = Recipe::from_file(path).expect("load recipe");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xACE5);
    let v = write_transforms_json(&out.snapshot, &recipe);
    let s = serde_json::to_string(&v).unwrap();
    let _: serde_json::Value = serde_json::from_str(&s).unwrap();
    assert_eq!(v["name"], "spheres_in_a_box");
    assert!(!v["placements"].as_array().unwrap().is_empty());
}

#[test]
fn deterministic_same_seed_same_output() {
    let Some(path) = spheres_recipe_path() else {
        return;
    };
    let recipe = Recipe::from_file(path).expect("load recipe");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let a = placer.pack(0xC0DE);
    let b = placer.pack(0xC0DE);
    assert_eq!(a.snapshot.placements.len(), b.snapshot.placements.len());
    for (pa, pb) in a.snapshot.placements.iter().zip(&b.snapshot.placements) {
        assert_eq!(pa.position, pb.position);
        assert_eq!(pa.ingredient_id, pb.ingredient_id);
    }
}
