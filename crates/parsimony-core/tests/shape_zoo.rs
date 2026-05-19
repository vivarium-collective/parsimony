//! Smoke + correctness test for the shape_zoo demo recipe (exercises
//! single_sphere + single_cube + single_cylinder + multi_cylinder +
//! multi_sphere + mesh ingredients packed into a Box compartment).

use std::path::Path;

use parsimony_core::{GreedyRandomPlacer, PlacerConfig, Recipe};

const RECIPE: &str = "/home/pattern/code/parsimony/examples/recipes/shape_zoo.json";

fn recipe_path() -> Option<&'static Path> {
    let p = Path::new(RECIPE);
    if p.exists() { Some(p) } else { None }
}

#[test]
fn shape_zoo_packs_everything() {
    let Some(path) = recipe_path() else { return };
    let recipe = Recipe::from_file(path).expect("load");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xC0DE);
    let placed = out.snapshot.placements.len();
    let requested = out.stats.requested;
    assert!(
        placed >= requested * 90 / 100,
        "expected >=90% placed (got {placed}/{requested})"
    );
}

#[test]
fn shape_zoo_no_overlaps() {
    let Some(path) = recipe_path() else { return };
    let recipe = Recipe::from_file(path).expect("load");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xFADE);

    // Build the full list of every world-space proxy sphere across
    // every placement. uid identifies the source placement so we
    // skip same-placement pairs (multi-sphere proxies share a uid
    // and may legitimately touch internally).
    let mut spheres: Vec<(nalgebra::Point3<f32>, f32, u64)> = Vec::new();
    for p in &out.snapshot.placements {
        let ing = recipe
            .ingredients
            .get_index(p.ingredient_id as usize)
            .unwrap()
            .1;
        for (c, r) in ing.shape.world_spheres(p.position, p.rotation) {
            spheres.push((c, r, p.instance_uid));
        }
    }

    // O(n²) over all proxy pairs (n ~ a few thousand for this recipe).
    for i in 0..spheres.len() {
        for j in (i + 1)..spheres.len() {
            let (ca, ra, ua) = spheres[i];
            let (cb, rb, ub) = spheres[j];
            if ua == ub {
                continue;
            }
            let d2 = (ca - cb).norm_squared();
            let r_sum = ra + rb;
            assert!(
                d2 + 1e-2 >= r_sum * r_sum,
                "proxy spheres of uids {ua} and {ub} overlap (d={:.3}, r_sum={r_sum})",
                d2.sqrt(),
            );
        }
    }
}

#[test]
fn shape_zoo_includes_every_shape_type() {
    let Some(path) = recipe_path() else { return };
    let recipe = Recipe::from_file(path).expect("load");
    use parsimony_core::IngredientShape;
    let mut has_single_sphere = false;
    let mut has_multi_sphere = false;
    let mut has_mesh = false;
    for (_, ing) in &recipe.ingredients {
        match &ing.shape {
            IngredientShape::SingleSphere { .. } => has_single_sphere = true,
            IngredientShape::MultiSphere { .. } => has_multi_sphere = true,
            IngredientShape::Mesh { .. } => has_mesh = true,
        }
    }
    assert!(has_single_sphere, "demo should include a single_sphere");
    assert!(
        has_multi_sphere,
        "demo should include MultiSphere-backed shapes (cube/cylinder/multi-sphere)"
    );
    assert!(has_mesh, "demo should include a mesh ingredient");
}
