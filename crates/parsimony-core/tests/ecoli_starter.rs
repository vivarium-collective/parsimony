//! Integration test: load the hand-authored E. coli starter recipe,
//! pack it, validate placement geometry. Phase 3 acceptance test.

use std::path::Path;

use parsimony_core::{
    CompartmentKind, GreedyRandomPlacer, IngredientShape, PlacerConfig, Recipe,
};

const RECIPE: &str = "/home/pattern/code/parsimony/examples/recipes/ecoli_starter.json";

fn recipe_path() -> Option<&'static Path> {
    let p = Path::new(RECIPE);
    if p.exists() { Some(p) } else { None }
}

#[test]
fn ecoli_packs_everything() {
    let Some(path) = recipe_path() else { return };
    let recipe = Recipe::from_file(path).expect("load");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xC0DE);
    // E. coli starter is sparse — should pack 100% (or very close).
    let placed = out.snapshot.placements.len();
    let requested = out.stats.requested;
    assert!(
        placed >= requested * 95 / 100,
        "expected >=95% placed (got {placed}/{requested})"
    );
}

#[test]
fn ecoli_lipids_on_capsule_surface() {
    let Some(path) = recipe_path() else { return };
    let recipe = Recipe::from_file(path).expect("load");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xFADE);

    // Find the cell compartment's capsule parameters.
    let cell = &recipe.compartments["cell"];
    let CompartmentKind::Capsule { a, b, radius } = &cell.kind else {
        panic!("cell must be a capsule");
    };
    let lipid_ingredient_id = recipe
        .ingredients
        .get_index_of("lipid")
        .expect("lipid ingredient");

    for p in &out.snapshot.placements {
        if p.ingredient_id as usize != lipid_ingredient_id {
            continue;
        }
        // Signed distance to capsule surface should be ~0.
        let ab = b - a;
        let ap = p.position - a;
        let h = (ab.dot(&ap) / ab.norm_squared()).clamp(0.0, 1.0);
        let closest = a + ab * h;
        let sd = (p.position - closest).norm() - radius;
        assert!(
            sd.abs() < 1e-2,
            "lipid not on capsule surface: signed distance = {sd}"
        );
    }
}

#[test]
fn ecoli_interior_proteins_inside_cell() {
    let Some(path) = recipe_path() else { return };
    let recipe = Recipe::from_file(path).expect("load");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xACE5);
    let cell = &recipe.compartments["cell"];
    let lipid_ingredient_id = recipe.ingredients.get_index_of("lipid").unwrap();

    for p in &out.snapshot.placements {
        if p.ingredient_id as usize == lipid_ingredient_id {
            continue; // lipids sit on the surface — skip
        }
        let ing = recipe.ingredients.get_index(p.ingredient_id as usize).unwrap().1;
        let r = ing.shape.enclosing_radius();
        // Every proxy sphere should be inside the cell.
        for (c, _) in ing.shape.world_spheres(p.position, p.rotation) {
            let inside = cell.kind.contains(c);
            assert!(
                inside,
                "ingredient `{}` proxy sphere at {:?} (instance r={r}) lies outside cell",
                ing.name, c
            );
        }
    }
}

#[test]
fn ecoli_no_overlaps_in_interior() {
    let Some(path) = recipe_path() else { return };
    let recipe = Recipe::from_file(path).expect("load");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xBEEF);
    let lipid_ingredient_id = recipe.ingredients.get_index_of("lipid").unwrap();

    // Collect (world_center, radius, instance_uid) for every interior
    // proxy sphere. Exclude lipids (surface; they can sit dense without
    // colliding because they're just markers).
    let mut spheres: Vec<(nalgebra::Point3<f32>, f32, u64)> = Vec::new();
    for p in &out.snapshot.placements {
        if p.ingredient_id as usize == lipid_ingredient_id {
            continue;
        }
        let ing = recipe.ingredients.get_index(p.ingredient_id as usize).unwrap().1;
        for (c, r) in ing.shape.world_spheres(p.position, p.rotation) {
            spheres.push((c, r, p.instance_uid));
        }
    }
    // O(n²). 1360 interior proxy spheres → ~1.8M pairs — fast.
    let n = spheres.len();
    let _ = IngredientShape::SingleSphere { radius: 0.0 }; // silence the unused-warning if any
    for i in 0..n {
        for j in (i + 1)..n {
            let (ca, ra, ua) = spheres[i];
            let (cb, rb, ub) = spheres[j];
            // Spheres on the same placement (multi-sphere ribosome) may
            // touch — skip same-uid pairs.
            if ua == ub {
                continue;
            }
            let d2 = (ca - cb).norm_squared();
            let r_sum = ra + rb;
            assert!(
                d2 + 1e-2 >= r_sum * r_sum,
                "uids {ua} and {ub} proxy spheres overlap (d={:.3}, r_sum={r_sum})",
                d2.sqrt(),
            );
        }
    }
}

#[test]
fn ecoli_ribosomes_have_random_rotations() {
    let Some(path) = recipe_path() else { return };
    let recipe = Recipe::from_file(path).expect("load");
    let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
    let out = placer.pack(0xC0DE);
    let ribo_id = recipe.ingredients.get_index_of("ribosome").unwrap();
    let mut rotations: Vec<nalgebra::UnitQuaternion<f32>> = Vec::new();
    for p in &out.snapshot.placements {
        if p.ingredient_id as usize == ribo_id {
            rotations.push(p.rotation);
        }
    }
    assert!(rotations.len() > 100);
    // At least most rotations should differ from identity.
    let non_identity = rotations
        .iter()
        .filter(|r| (r.w - 1.0).abs() > 1e-4)
        .count();
    assert!(
        non_identity > rotations.len() * 9 / 10,
        "expected >90% of ribosome rotations to differ from identity, got {non_identity}/{}",
        rotations.len()
    );
}
