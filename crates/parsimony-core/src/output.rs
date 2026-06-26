//! Output writers — [`Snapshot`] → JSON in one of three formats:
//!
//! - **`parsimony.pack.v1`** ([`write_pack_json`]) is the *primary*
//!   format. Native to parsimony, drives the local three.js viewer.
//!   Carries everything the renderer needs: full compartment
//!   hierarchy, per-ingredient shape descriptors (sphere /
//!   multi-sphere / mesh with mesh URL), per-instance position +
//!   quaternion rotation. Designed to grow with dynamics frames,
//!   instance-specific overrides, etc.
//! - **`transforms`** ([`write_transforms_json`]) is a flat
//!   transform list, kept around as a stable debug-friendly export
//!   for downstream tooling.
//! - **Simularium** ([`write_simularium_json`]) is an *optional*
//!   export for users who want to load packs into
//!   `cellpack.allencell.org`. Lossy by construction (everything
//!   becomes a sphere with `displayType: "SPHERE"`); not the source
//!   of truth.

use serde_json::{json, Value};

use crate::ingredient::IngredientShape;
use crate::placement::Snapshot;
use crate::recipe::Recipe;

/// parsimony's native pack format. Single JSON document; the
/// `format` field is the discriminator so future revs can co-exist.
///
/// Schema:
///
/// ```text
/// {
///   "format": "parsimony.pack.v1",
///   "recipe_name": "...",
///   "seed": <u64>,
///   "bounds": { "min": [x,y,z], "max": [x,y,z] },
///   "compartments": [
///     { "id": 0, "name": "space", "parent": null, "kind": "box",
///       "bounds": { ... } },
///     { "id": 1, "name": "cell",  "parent": 0,    "kind": "sphere",
///       "center": [...], "radius": ... },
///     // ... or kind: "capsule" | "mesh"
///   ],
///   "ingredients": [
///     { "id": 0, "name": "...", "color": [r,g,b],
///       "shape": { "kind": "sphere", "radius": ... } },
///     { "id": 1, "name": "...", "color": [r,g,b],
///       "shape": { "kind": "multi_sphere",
///                  "spheres": [{ "offset": [...], "radius": ... }, ...],
///                  "enclosing_radius": ... } },
///     { "id": 2, "name": "...", "color": [r,g,b],
///       "shape": { "kind": "mesh", "url": "examples/.../x.obj",
///                  "enclosing_radius": ... } },
///   ],
///   "placements": [
///     { "uid": 0, "ingredient": 1, "compartment": 1,
///       "position": [x,y,z],
///       "rotation": [w,x,y,z] }   // unit quaternion, w first
///   ]
/// }
/// ```
pub fn write_pack_json(snapshot: &Snapshot, recipe: &Recipe) -> Value {
    // Compartments — flatten with parent ids so the viewer can
    // rebuild the hierarchy.
    let compartments: Vec<Value> = recipe
        .compartments
        .iter()
        .enumerate()
        .map(|(id, (name, c))| {
            let parent = c
                .parent
                .map(|p| Value::from(p as u64))
                .unwrap_or(Value::Null);
            let kind = match &c.kind {
                crate::compartment::CompartmentKind::Box(bb) => json!({
                    "kind": "box",
                    "bounds": {
                        "min": [bb.min.x, bb.min.y, bb.min.z],
                        "max": [bb.max.x, bb.max.y, bb.max.z],
                    },
                }),
                crate::compartment::CompartmentKind::Sphere { center, radius } => json!({
                    "kind": "sphere",
                    "center": [center.x, center.y, center.z],
                    "radius": radius,
                }),
                crate::compartment::CompartmentKind::Capsule { a, b, radius } => json!({
                    "kind": "capsule",
                    "a": [a.x, a.y, a.z],
                    "b": [b.x, b.y, b.z],
                    "radius": radius,
                }),
                crate::compartment::CompartmentKind::Mesh(m) => json!({
                    "kind": "mesh",
                    "bounds": {
                        "min": [m.aabb.min.x, m.aabb.min.y, m.aabb.min.z],
                        "max": [m.aabb.max.x, m.aabb.max.y, m.aabb.max.z],
                    },
                }),
            };
            // Merge kind block into outer object.
            let mut obj = json!({
                "id": id as u64,
                "name": name,
                "parent": parent,
            });
            if let Value::Object(extra) = kind {
                if let Value::Object(o) = &mut obj {
                    for (k, v) in extra {
                        o.insert(k, v);
                    }
                }
            }
            obj
        })
        .collect();

    let mut ingredients: Vec<Value> = recipe
        .ingredients
        .iter()
        .enumerate()
        .map(|(id, (name, ing))| {
            let shape = match &ing.shape {
                IngredientShape::SingleSphere { radius } => json!({
                    "kind": "sphere",
                    "radius": radius,
                }),
                IngredientShape::MultiSphere { spheres } => {
                    let s: Vec<Value> = spheres
                        .iter()
                        .map(|s| json!({
                            "offset": [s.offset.x, s.offset.y, s.offset.z],
                            "radius": s.radius,
                        }))
                        .collect();
                    json!({
                        "kind": "multi_sphere",
                        "spheres": s,
                        "enclosing_radius": ing.shape.enclosing_radius(),
                    })
                }
                IngredientShape::Mesh { .. } => {
                    let lods: Vec<Value> = ing
                        .mesh_lods
                        .iter()
                        .map(|l| json!({ "url": l.url, "voxel_size": l.voxel_size }))
                        .collect();
                    let mut shape = json!({
                        "kind": "mesh",
                        "lods": lods,
                        "enclosing_radius": ing.shape.enclosing_radius(),
                    });
                    // Anisotropic stand-in for the viewer's placeholder /
                    // far-LOD: a principal-axis ellipsoid so elongated
                    // species read as cigars instead of fat enclosing
                    // balls. Rotation is [w, x, y, z] (pack convention).
                    if let Some(e) = ing.shape.principal_ellipsoid() {
                        shape["ellipsoid"] = json!({
                            "semi_axes": e.semi_axes,
                            "rotation": [e.rotation.w, e.rotation.i, e.rotation.j, e.rotation.k],
                        });
                    }
                    shape
                }
                IngredientShape::Fiber { points, radius } => json!({
                    "kind": "fiber",
                    "points": points.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
                    "radius": radius,
                    "enclosing_radius": ing.shape.enclosing_radius(),
                }),
            };
            json!({
                "id": id as u64,
                "name": name,
                "color": ing.color,
                "shape": shape,
            })
        })
        .collect();

    let mut placements: Vec<Value> = Vec::with_capacity(snapshot.placements.len());
    for p in &snapshot.placements {
        // mRNA-style ingredients: a multi_sphere carrying a `segment` mesh
        // renders as that mesh tiled along the instance's bead chain (a real
        // RNA strand instead of beads), reusing the chromosome's segment tiler.
        // The multi-sphere proxy was still used for packing/collision.
        let seg = recipe
            .ingredients
            .get_index(p.ingredient_id as usize)
            .and_then(|(_, ing)| match (&ing.shape, &ing.segment) {
                (IngredientShape::MultiSphere { spheres }, Some(s)) if spheres.len() >= 2 => {
                    recipe.ingredients.get_index_of(s.as_str()).map(|id| (spheres, id))
                }
                _ => None,
            });
        if let Some((spheres, seg_id)) = seg {
            let path: Vec<_> = spheres.iter().map(|s| p.position + p.rotation * s.offset).collect();
            let step = (path[1] - path[0]).norm().max(1.0); // ~one segment per bead
            for (pos, rot) in crate::fiber::dna_segment_transforms(&path, step, 0.0) {
                placements.push(json!({
                    "uid": placements.len() as u64,
                    "ingredient": seg_id as u64,
                    "compartment": p.compartment_id,
                    "position": [pos.x, pos.y, pos.z],
                    "rotation": [rot.w, rot.i, rot.j, rot.k],
                }));
            }
            continue;
        }
        placements.push(json!({
            "uid": placements.len() as u64,
            "ingredient": p.ingredient_id,
            "compartment": p.compartment_id,
            "position": [p.position.x, p.position.y, p.position.z],
            "rotation": [p.rotation.w, p.rotation.i, p.rotation.j, p.rotation.k],
        }));
    }

    // The genome, if the placer generated one. When the recipe configures a
    // `segment` ingredient (a per-bead dsDNA mesh), render it the molecular way
    // — tile that shared mesh along the path as ~tens of thousands of instances
    // (one per ~12 bp), oriented + twisted into a continuous helix, so it goes
    // through the same InstancedMesh + per-instance LOD + cel/outline path as
    // proteins. Otherwise fall back to a single smooth `fiber` tube.
    if let Some(chr) = &snapshot.chromosome {
        let seg = recipe
            .chromosome
            .as_ref()
            .and_then(|c| c.segment.as_ref())
            .and_then(|name| recipe.ingredients.get_index_of(name));
        if let Some(seg_id) = seg {
            let seg_step = 12.0 * 3.4; // ~12 bp per 1BNA segment, 3.4 Å/bp
            let twist = std::f32::consts::TAU / (10.5 * 3.4); // B-DNA: 10.5 bp/turn
            // Tile each strand independently: the main genome plus, for a
            // replicating chromosome, the sister strand over the theta bubble.
            let strand_list: Vec<&Vec<_>> = if chr.strands.is_empty() {
                vec![&chr.points]
            } else {
                chr.strands.iter().collect()
            };
            for strand in strand_list {
                let world: Vec<_> = strand.iter().map(|p| chr.center + p.coords).collect();
                for (pos, rot) in crate::fiber::dna_segment_transforms(&world, seg_step, twist) {
                    placements.push(json!({
                        "uid": placements.len() as u64,
                        "ingredient": seg_id as u64,
                        "compartment": 0,
                        "position": [pos.x, pos.y, pos.z],
                        "rotation": [rot.w, rot.i, rot.j, rot.k],
                    }));
                }
            }
        } else {
            let id = ingredients.len() as u64;
            let enc = chr
                .points
                .iter()
                .map(|p| p.coords.norm())
                .fold(0.0_f32, f32::max)
                + chr.radius;
            ingredients.push(json!({
                "id": id,
                "name": "chromosome",
                "color": chr.color,
                "shape": {
                    "kind": "fiber",
                    "points": chr.points.iter().map(|p| [p.x, p.y, p.z]).collect::<Vec<_>>(),
                    "radius": chr.radius,
                    "enclosing_radius": enc,
                },
            }));
            placements.push(json!({
                "uid": placements.len() as u64,
                "ingredient": id,
                "compartment": 0,
                "position": [chr.center.x, chr.center.y, chr.center.z],
                "rotation": [1.0, 0.0, 0.0, 0.0],
            }));
        }
    }

    // Nascent-RNA strands: tile the `rna_segment` mesh along each strand,
    // mirroring the dna_segment block above. Points are center-relative (same
    // frame as `Chromosome::strands`); add the chromosome center to recover
    // world coordinates. seg_step = 40.0 Å (one segment per bead, matching the
    // strand bead spacing); twist = 0.0 (ssRNA, no helical twist).
    if !snapshot.rna_strands.is_empty() {
        let chrom = recipe.chromosome.as_ref();
        let resolve =
            |name: Option<&String>| name.and_then(|n| recipe.ingredients.get_index_of(n));
        // Nascent strands tile `rna_segment`; free strands tile `rna_segment_free`
        // (falling back to `rna_segment` when no separate free ingredient is set).
        let nascent_seg = resolve(chrom.and_then(|c| c.rna_segment.as_ref()));
        let free_seg =
            resolve(chrom.and_then(|c| c.rna_segment_free.as_ref())).or(nascent_seg);
        let center = snapshot
            .chromosome
            .as_ref()
            .map(|c| c.center)
            .unwrap_or(nalgebra::Point3::origin());
        let seg_step = 40.0_f32; // one segment per bead (bead spacing = 40 Å)
        for rna in &snapshot.rna_strands {
            let seg_id = if rna.is_free { free_seg } else { nascent_seg };
            let Some(seg_id) = seg_id else { continue };
            let world: Vec<_> = rna.points.iter().map(|p| center + p.coords).collect();
            for (pos, rot) in crate::fiber::dna_segment_transforms(&world, seg_step, 0.0) {
                placements.push(json!({
                    "uid": placements.len() as u64,
                    "ingredient": seg_id as u64,
                    "compartment": 0,
                    "position": [pos.x, pos.y, pos.z],
                    "rotation": [rot.w, rot.i, rot.j, rot.k],
                }));
            }
        }
    }

    // Nascent-peptide coils: tile the `peptide_segment` mesh along each strand,
    // mirroring the rna_segment block above.  Points are center-relative (same
    // frame as rna_strands); add the chromosome center for world coordinates.
    // seg_step = 30.0 Å (one segment per bead, matching the peptide bead spacing);
    // twist = 0.0 (unstructured coil, no helical twist).
    if !snapshot.peptide_strands.is_empty() {
        let chrom = recipe.chromosome.as_ref();
        let pep_seg = chrom
            .and_then(|c| c.peptide_segment.as_ref())
            .and_then(|n| recipe.ingredients.get_index_of(n));
        if let Some(pep_seg_id) = pep_seg {
            let center = snapshot
                .chromosome
                .as_ref()
                .map(|c| c.center)
                .unwrap_or(nalgebra::Point3::origin());
            let seg_step = 30.0_f32;
            for strand in &snapshot.peptide_strands {
                let world: Vec<_> = strand.iter().map(|p| center + p.coords).collect();
                for (pos, rot) in crate::fiber::dna_segment_transforms(&world, seg_step, 0.0) {
                    placements.push(json!({
                        "uid": placements.len() as u64,
                        "ingredient": pep_seg_id as u64,
                        "compartment": 0,
                        "position": [pos.x, pos.y, pos.z],
                        "rotation": [rot.w, rot.i, rot.j, rot.k],
                    }));
                }
            }
        }
    }

    // Cache-busting token for the viewer's IndexedDB mesh cache: a fresh value
    // each write, so regenerating the pack/meshes invalidates cached geometry
    // (which is keyed by URL and would otherwise serve stale meshes).
    let cache_version = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0);
    json!({
        "format": "parsimony.pack.v1",
        "cache_version": cache_version,
        "recipe_name": snapshot.recipe_name,
        "seed": snapshot.seed,
        "bounds": {
            "min": [recipe.bounding_box.min.x, recipe.bounding_box.min.y, recipe.bounding_box.min.z],
            "max": [recipe.bounding_box.max.x, recipe.bounding_box.max.y, recipe.bounding_box.max.z],
        },
        "compartments": compartments,
        "ingredients": ingredients,
        "placements": placements,
    })
}

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
                "geometry": { "displayType": "SPHERE", "color": hex },
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

    /// B1-4: `write_pack_json` tiles the `rna_segment` mesh along every nascent-RNA
    /// strand in `snapshot.rna_strands`, emitting one placement per bead-step
    /// (seg_step = 40 Å, twist = 0).  With two 800-nt strands (40 beads each)
    /// the output must contain many more than 20 `rna_segment` placements.
    #[test]
    fn emits_rna_segment_placements_for_each_strand() {
        let json = r#"{
            "bounding_box": [[-500,-500,-500],[500,500,500]],
            "objects": {
                "rna_segment": { "type": "single_sphere", "radius": 4.0, "color": [0.2, 0.8, 0.4] }
            },
            "composition": {
                "space": { "regions": { "interior": ["cell"] } },
                "cell": {
                    "compartment": {
                        "kind": "capsule",
                        "a": [-150, 0, 0],
                        "b": [150, 0, 0],
                        "radius": 80
                    },
                    "regions": { "interior": [] }
                }
            },
            "chromosome": {
                "beads": 1000,
                "spacing": 10.0,
                "bead_radius": 5.0,
                "compartment": "cell",
                "rna_segment": "rna_segment",
                "rnas": [
                    {"root_coordinate": 100000, "root_domain": 0, "length_nt": 800, "is_mRNA": true},
                    {"root_coordinate": -50000,  "root_domain": 0, "length_nt": 800, "is_mRNA": false}
                ]
            }
        }"#;
        let recipe = crate::recipe::Recipe::from_json_str(json).expect("recipe parse failed");
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(5);
        let pack = write_pack_json(&out.snapshot, &recipe);
        let seg_id = recipe.ingredients.get_index_of("rna_segment").unwrap();
        let n = pack["placements"].as_array().unwrap().iter()
            .filter(|p| p["ingredient"].as_u64() == Some(seg_id as u64))
            .count();
        assert!(n > 20, "expected many tiled rna_segment placements, got {n}");
    }

    /// B1-5: `write_pack_json` routes nascent strands to `rna_segment` and free
    /// strands to `rna_segment_free` when both ingredients are configured.  With
    /// one nascent RNA and one free RNA in the recipe, both ingredients must
    /// appear in the placements list.
    #[test]
    fn free_rna_strands_tile_rna_segment_free() {
        let json = r#"{
            "bounding_box": [[-500,-500,-500],[500,500,500]],
            "objects": {
                "rna_segment":      { "type": "single_sphere", "radius": 4.0, "color": [0.2, 0.8, 0.4] },
                "rna_segment_free": { "type": "single_sphere", "radius": 4.0, "color": [0.8, 0.2, 0.4] }
            },
            "composition": {
                "space": { "regions": { "interior": ["cell"] } },
                "cell": {
                    "compartment": {
                        "kind": "capsule",
                        "a": [-150, 0, 0],
                        "b": [150, 0, 0],
                        "radius": 80
                    },
                    "regions": { "interior": [] }
                }
            },
            "chromosome": {
                "beads": 1000,
                "spacing": 10.0,
                "bead_radius": 5.0,
                "compartment": "cell",
                "rna_segment":      "rna_segment",
                "rna_segment_free": "rna_segment_free",
                "rnas": [
                    {"root_coordinate": 100000, "root_domain": 0, "length_nt": 800, "is_mRNA": true, "is_free": false},
                    {"root_coordinate": -50000,  "root_domain": 0, "length_nt": 800, "is_mRNA": true, "is_free": true}
                ]
            }
        }"#;
        let recipe = crate::recipe::Recipe::from_json_str(json).expect("recipe parse failed");
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(7);
        let pack = write_pack_json(&out.snapshot, &recipe);
        let placements = pack["placements"].as_array().unwrap();

        let nascent_id = recipe.ingredients.get_index_of("rna_segment").unwrap() as u64;
        let free_id = recipe.ingredients.get_index_of("rna_segment_free").unwrap() as u64;

        let nascent_count = placements.iter()
            .filter(|p| p["ingredient"].as_u64() == Some(nascent_id))
            .count();
        let free_count = placements.iter()
            .filter(|p| p["ingredient"].as_u64() == Some(free_id))
            .count();

        assert!(nascent_count > 0,
            "expected rna_segment placements for nascent strand, got {nascent_count}");
        assert!(free_count > 0,
            "expected rna_segment_free placements for free strand, got {free_count}");
    }

    /// B1-5b: when `rna_segment_free` is NOT defined, a free strand falls back
    /// to tiling `rna_segment` (same ingredient as nascent strands).
    #[test]
    fn free_rna_falls_back_to_rna_segment_when_no_free_ingredient() {
        let json = r#"{
            "bounding_box": [[-500,-500,-500],[500,500,500]],
            "objects": {
                "rna_segment": { "type": "single_sphere", "radius": 4.0, "color": [0.2, 0.8, 0.4] }
            },
            "composition": {
                "space": { "regions": { "interior": ["cell"] } },
                "cell": {
                    "compartment": {
                        "kind": "capsule",
                        "a": [-150, 0, 0],
                        "b": [150, 0, 0],
                        "radius": 80
                    },
                    "regions": { "interior": [] }
                }
            },
            "chromosome": {
                "beads": 1000,
                "spacing": 10.0,
                "bead_radius": 5.0,
                "compartment": "cell",
                "rna_segment": "rna_segment",
                "rnas": [
                    {"root_coordinate": 100000, "root_domain": 0, "length_nt": 800, "is_mRNA": true, "is_free": false},
                    {"root_coordinate": -50000,  "root_domain": 0, "length_nt": 800, "is_mRNA": true, "is_free": true}
                ]
            }
        }"#;
        let recipe = crate::recipe::Recipe::from_json_str(json).expect("recipe parse failed");
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(9);
        let pack = write_pack_json(&out.snapshot, &recipe);
        let placements = pack["placements"].as_array().unwrap();

        let seg_id = recipe.ingredients.get_index_of("rna_segment").unwrap() as u64;
        let n = placements.iter()
            .filter(|p| p["ingredient"].as_u64() == Some(seg_id))
            .count();

        // Both nascent and free strands tile rna_segment (no free-specific ingredient)
        assert!(n > 20,
            "expected both strands tiled via rna_segment fallback, got {n} placements");
    }

    /// C2-1: `write_pack_json` tiles `peptide_segment` along each peptide strand
    /// in `snapshot.peptide_strands`, emitting at least one placement per strand.
    #[test]
    fn emits_peptide_segment_placements_for_peptide_strands() {
        let json = r#"{
            "bounding_box": [[-3000,-3000,-3000],[3000,3000,3000]],
            "objects": {
                "70S_ribosome":    { "type": "single_sphere", "radius": 120 },
                "peptide_segment": { "type": "single_sphere", "radius": 3.0, "color": [0.4, 0.8, 0.6] }
            },
            "composition": {
                "space": { "regions": { "interior": ["cell"] } },
                "cell": {
                    "compartment": {"kind": "capsule", "a": [-1500,0,0], "b": [1500,0,0], "radius": 1000},
                    "regions": { "interior": [] }
                }
            },
            "chromosome": {
                "beads": 1000, "spacing": 10.0, "bead_radius": 5.0, "compartment": "cell",
                "ribosome_marker": "70S_ribosome",
                "peptide_segment": "peptide_segment",
                "peptide_angstrom_per_aa": 3.0,
                "rnas": [
                    {"root_coordinate": 0, "root_domain": 0, "length_nt": 600,
                     "is_mRNA": true, "is_free": true, "unique_index": 20}
                ],
                "ribosomes": [
                    {"mRNA_index": 20, "pos_on_mRNA": 300, "peptide_length": 200}
                ]
            }
        }"#;
        let recipe = crate::recipe::Recipe::from_json_str(json).expect("recipe parse failed");
        let placer = GreedyRandomPlacer::new(&recipe, PlacerConfig::default());
        let out = placer.pack(42);
        let pack = write_pack_json(&out.snapshot, &recipe);
        let seg_id = recipe.ingredients.get_index_of("peptide_segment").unwrap();
        let n = pack["placements"].as_array().unwrap().iter()
            .filter(|p| p["ingredient"].as_u64() == Some(seg_id as u64))
            .count();
        assert!(n > 0, "expected peptide_segment placements, got {n}");
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
