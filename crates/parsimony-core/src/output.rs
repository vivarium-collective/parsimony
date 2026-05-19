//! Output writers — [`Snapshot`] → JSON in either a debug-friendly
//! transform-list format or the Simularium format consumed by the
//! `cellpack.allencell.org` viewer.

use serde_json::{json, Value};

use crate::placement::Snapshot;
use crate::recipe::Recipe;

/// Debug-friendly format: one entry per placement with explicit fields.
pub fn write_transforms_json(snapshot: &Snapshot, recipe: &Recipe) -> Value {
    let mut type_mapping = serde_json::Map::new();
    for (id, (name, ing)) in recipe.ingredients.iter().enumerate() {
        type_mapping.insert(
            id.to_string(),
            json!({
                "name": name,
                "radius": ing.shape.enclosing_radius(),
                "color": ing.color,
            }),
        );
    }

    let placements: Vec<Value> = snapshot
        .placements
        .iter()
        .map(|p| {
            json!({
                "uid": p.instance_uid,
                "type_id": p.ingredient_id,
                "variant_id": p.variant_id,
                "compartment_id": p.compartment_id,
                "position": [p.position.x, p.position.y, p.position.z],
                "rotation": [
                    p.rotation.i,
                    p.rotation.j,
                    p.rotation.k,
                    p.rotation.w,
                ],
            })
        })
        .collect();

    json!({
        "name": snapshot.recipe_name,
        "seed": snapshot.seed,
        "bounding_box": [
            [recipe.bounding_box.min.x, recipe.bounding_box.min.y, recipe.bounding_box.min.z],
            [recipe.bounding_box.max.x, recipe.bounding_box.max.y, recipe.bounding_box.max.z],
        ],
        "type_mapping": type_mapping,
        "placements": placements,
    })
}

/// Simularium trajectory format (v3 schema). Compatible with the
/// `cellpack.allencell.org` viewer. Emits a single frame at time 0.
///
/// The data array is flat: for each agent we encode 11 floats —
/// `[vis_type, uniqueId, typeId, x, y, z, xrot, yrot, zrot,
/// collisionRadius, nSubPoints]`. `vis_type = 1000.0` is the default
/// per-agent visualization.
pub fn write_simularium_json(snapshot: &Snapshot, recipe: &Recipe) -> Value {
    let mut type_mapping = serde_json::Map::new();
    for (id, (name, ing)) in recipe.ingredients.iter().enumerate() {
        let [r, g, b] = ing.color;
        let hex = format!(
            "#{:02x}{:02x}{:02x}",
            (r.clamp(0.0, 1.0) * 255.0) as u8,
            (g.clamp(0.0, 1.0) * 255.0) as u8,
            (b.clamp(0.0, 1.0) * 255.0) as u8,
        );
        type_mapping.insert(
            id.to_string(),
            json!({
                "name": name,
                "geometry": {
                    "displayType": "SPHERE",
                    "color": hex,
                }
            }),
        );
    }

    let bb = recipe.bounding_box;
    let size = json!({
        "x": bb.max.x - bb.min.x,
        "y": bb.max.y - bb.min.y,
        "z": bb.max.z - bb.min.z,
    });

    // Simularium positions are relative to the box centre.
    let centre_x = (bb.min.x + bb.max.x) * 0.5;
    let centre_y = (bb.min.y + bb.max.y) * 0.5;
    let centre_z = (bb.min.z + bb.max.z) * 0.5;

    let mut data: Vec<f64> = Vec::with_capacity(snapshot.placements.len() * 11);
    for p in &snapshot.placements {
        let (name, ing) = recipe
            .ingredients
            .get_index(p.ingredient_id as usize)
            .expect("ingredient_id out of range");
        let _ = name;
        let radius = ing.shape.enclosing_radius();
        let (roll, pitch, yaw) = p.rotation.euler_angles();
        data.extend_from_slice(&[
            1000.0_f64,                          // visType
            p.instance_uid as f64,               // uniqueId
            p.ingredient_id as f64,              // typeId
            (p.position.x - centre_x) as f64,    // x (centered)
            (p.position.y - centre_y) as f64,    // y
            (p.position.z - centre_z) as f64,    // z
            roll as f64,                         // xrot
            pitch as f64,                        // yrot
            yaw as f64,                          // zrot
            radius as f64,                       // collisionRadius
            0.0,                                 // nSubPoints
        ]);
    }

    json!({
        "trajectoryInfo": {
            "version": 3,
            "timeUnits": { "magnitude": 1.0, "name": "s" },
            "timeStepSize": 0.0,
            "totalSteps": 1,
            "spatialUnits": { "magnitude": 1.0, "name": "nm" },
            "size": size,
            "cameraDefault": {
                "position": { "x": 0.0, "y": 0.0, "z": 120.0 },
                "lookAtPosition": { "x": 0.0, "y": 0.0, "z": 0.0 },
                "upVector": { "x": 0.0, "y": 1.0, "z": 0.0 },
                "fovDegrees": 75.0,
            },
            "typeMapping": type_mapping,
        },
        "spatialData": {
            "version": 1,
            "msgType": 1,
            "bundleStart": 0,
            "bundleSize": 1,
            "bundleData": [
                {
                    "frameNumber": 0,
                    "time": 0.0,
                    "data": data,
                }
            ],
        },
        "plotData": {
            "version": 1,
            "data": [],
        }
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::placer::{GreedyRandomPlacer, PlacerConfig};

    const TINY: &str = r#"{
        "name": "tiny",
        "bounding_box": [[0,0,0],[100,100,100]],
        "objects": {
            "s": { "type": "single_sphere", "radius": 5, "color": [1, 0, 0] }
        },
        "composition": {
            "space": { "regions": { "interior": ["A"] } },
            "A": { "object": "s", "count": 10 }
        }
    }"#;

    fn pack(seed: u64) -> (Recipe, crate::placement::Snapshot) {
        let recipe = Recipe::from_json_str(TINY).unwrap();
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(seed);
        (recipe, out.snapshot)
    }

    #[test]
    fn transforms_json_is_well_formed() {
        let (recipe, snap) = pack(0xC0DE);
        let v = write_transforms_json(&snap, &recipe);
        // Round-trip parse should succeed.
        let s = serde_json::to_string(&v).unwrap();
        let _: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(v["name"], "tiny");
        assert_eq!(v["seed"], 0xC0DE);
        assert!(v["placements"].is_array());
        // Each placement has expected fields.
        for p in v["placements"].as_array().unwrap() {
            assert!(p["uid"].is_number());
            assert!(p["type_id"].is_number());
            assert!(p["position"].as_array().unwrap().len() == 3);
        }
    }

    #[test]
    fn simularium_json_has_expected_shape() {
        let (recipe, snap) = pack(0xFADE);
        let v = write_simularium_json(&snap, &recipe);
        // Top-level keys.
        assert!(v["trajectoryInfo"].is_object());
        assert!(v["spatialData"].is_object());
        assert_eq!(v["trajectoryInfo"]["version"], 3);
        // Type mapping has at least one entry.
        let tm = v["trajectoryInfo"]["typeMapping"].as_object().unwrap();
        assert!(!tm.is_empty());
        assert_eq!(tm["0"]["geometry"]["displayType"], "SPHERE");
        // bundleData has one frame.
        let bd = v["spatialData"]["bundleData"].as_array().unwrap();
        assert_eq!(bd.len(), 1);
        // Frame data has placements × 11 floats.
        let data = bd[0]["data"].as_array().unwrap();
        assert_eq!(data.len(), snap.placements.len() * 11);
    }

    #[test]
    fn simularium_positions_are_centred() {
        let (recipe, snap) = pack(0xACE5);
        let v = write_simularium_json(&snap, &recipe);
        let data = v["spatialData"]["bundleData"][0]["data"]
            .as_array()
            .unwrap();
        // Box centre is (50, 50, 50); cells inside should give relative coords in [-50, 50].
        for chunk in data.chunks(11) {
            let x = chunk[3].as_f64().unwrap();
            let y = chunk[4].as_f64().unwrap();
            let z = chunk[5].as_f64().unwrap();
            assert!(x.abs() <= 50.0 && y.abs() <= 50.0 && z.abs() <= 50.0);
        }
    }
}
