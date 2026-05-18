//! Recipe loader. Reads cellPACK v2.1 JSON recipes, applies object
//! inheritance, and resolves the composition tree into a flat list of
//! placement directives.
//!
//! The subset supported in Phase 2 is what `spheres_in_a_box.json`
//! needs: `single_sphere` objects with optional `inherit`,
//! axis-aligned `bounding_box` compartments, and a flat composition
//! tree referenced from `space`. Nested compartments, mesh objects,
//! and gradients are recognized in the schema but rejected until the
//! relevant components land.

use indexmap::IndexMap;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::compartment::{Compartment, CompartmentKind};
use crate::ingredient::{Ingredient, IngredientShape};
use parsimony_spatial::Aabb;

// ---------- raw JSON schema (mirrors cellPACK v2.1) ----------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawRecipe {
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    version: Option<String>,
    #[serde(default)]
    format_version: Option<String>,
    #[serde(default)]
    description: Option<String>,
    bounding_box: [[f32; 3]; 2],
    objects: IndexMap<String, RawObject>,
    composition: IndexMap<String, RawCompositionEntry>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawObject {
    #[serde(rename = "type")]
    #[serde(default)]
    type_name: Option<String>,
    #[serde(default)]
    inherit: Option<String>,
    #[serde(default)]
    color: Option<[f32; 3]>,
    #[serde(default)]
    radius: Option<f32>,
    #[serde(default)]
    jitter_attempts: Option<u32>,
    #[serde(default)]
    packing_mode: Option<String>,
    // (remaining cellPACK fields ignored until needed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawCompositionEntry {
    #[serde(default)]
    object: Option<String>,
    #[serde(default)]
    count: Option<u32>,
    #[serde(default)]
    regions: Option<IndexMap<String, Vec<RawRegionEntry>>>,
    #[serde(default)]
    molarity: Option<f32>,
    #[serde(default)]
    priority: Option<f32>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
enum RawRegionEntry {
    /// A reference to another composition entry by name.
    Ref(String),
    /// Inline placement directive.
    Inline {
        object: String,
        #[serde(default)]
        count: Option<u32>,
        #[serde(default)]
        molarity: Option<f32>,
        #[serde(default)]
        priority: Option<f32>,
    },
}

// ---------- typed output ----------

/// Region within a compartment that ingredients can be placed in.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RegionKind {
    Interior,
    Surface,
}

/// How a placement candidate is picked. Mirrors cellPACK's `packing_mode`
/// field; only `Random` is supported in v0.1.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PackingMode {
    #[default]
    Random,
}

/// One resolved placement directive: "place `count` instances of
/// `ingredient` in `region` of `compartment`". The recipe's composition
/// tree is walked at load time and flattened into a list of these.
#[derive(Debug, Clone)]
pub struct PlacementDirective {
    pub ingredient: String,
    pub compartment: String,
    pub region: RegionKind,
    pub count: u32,
    pub priority: f32,
    pub packing_mode: PackingMode,
}

/// A fully-resolved recipe ready to feed into the placer.
#[derive(Debug, Clone)]
pub struct Recipe {
    pub name: String,
    pub bounding_box: Aabb,
    pub ingredients: IndexMap<String, Ingredient>,
    pub compartments: IndexMap<String, Compartment>,
    pub directives: Vec<PlacementDirective>,
}

impl Recipe {
    pub fn from_json_str(src: &str) -> Result<Self, RecipeError> {
        let raw: RawRecipe = serde_json::from_str(src)?;
        resolve(raw)
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, RecipeError> {
        let src = std::fs::read_to_string(path.as_ref())
            .map_err(|e| RecipeError::Io(e.to_string()))?;
        Self::from_json_str(&src)
    }
}

#[derive(Debug, Error)]
pub enum RecipeError {
    #[error("JSON parse error: {0}")]
    Json(#[from] serde_json::Error),
    #[error("I/O error: {0}")]
    Io(String),
    #[error("object `{0}` not found")]
    UnknownObject(String),
    #[error("composition entry `{0}` not found")]
    UnknownCompositionEntry(String),
    #[error("inheritance cycle: {0}")]
    InheritanceCycle(String),
    #[error("object `{name}`: {message}")]
    InvalidObject { name: String, message: String },
    #[error("composition entry `{name}`: {message}")]
    InvalidComposition { name: String, message: String },
    #[error("unsupported ingredient type `{0}`")]
    UnsupportedIngredient(String),
    #[error("unsupported packing mode `{0}`")]
    UnsupportedPackingMode(String),
    #[error("missing top-level composition entry (need `space` or `bounding_area`)")]
    MissingRoot,
}

// ---------- resolution ----------

fn resolve(raw: RawRecipe) -> Result<Recipe, RecipeError> {
    let bounding_box = Aabb::new(
        nalgebra::Point3::new(raw.bounding_box[0][0], raw.bounding_box[0][1], raw.bounding_box[0][2]),
        nalgebra::Point3::new(raw.bounding_box[1][0], raw.bounding_box[1][1], raw.bounding_box[1][2]),
    );

    // 1. Resolve object inheritance.
    let mut resolved_objects: IndexMap<String, RawObject> = IndexMap::new();
    let mut visiting: Vec<String> = Vec::new();
    for (name, _) in raw.objects.iter() {
        resolve_object(name, &raw.objects, &mut resolved_objects, &mut visiting)?;
    }

    // 2. Build typed ingredients from the resolved objects.
    let mut ingredients: IndexMap<String, Ingredient> = IndexMap::new();
    for (name, obj) in &resolved_objects {
        let typ = obj.type_name.as_deref().unwrap_or("");
        let shape = match typ {
            "single_sphere" => {
                let Some(r) = obj.radius else {
                    // No radius — this is an inheritance template (cellPACK
                    // recipes commonly define a typed `base` for child
                    // objects to inherit from). Don't materialize an
                    // ingredient for it.
                    continue;
                };
                IngredientShape::SingleSphere { radius: r }
            }
            "" => continue,
            _ => {
                // Unsupported types are tolerated at parse time so recipes
                // referencing them for non-placement roles (mesh compartments,
                // fibers, etc.) still load. A directive that actually points
                // to one errors at composition resolution.
                continue;
            }
        };
        ingredients.insert(
            name.clone(),
            Ingredient {
                name: name.clone(),
                shape,
                color: obj.color.unwrap_or([0.5, 0.5, 0.5]),
                jitter_attempts: obj.jitter_attempts.unwrap_or(20),
                packing_mode: parse_packing_mode(obj.packing_mode.as_deref())?,
            },
        );
    }

    // 3. Find the root composition entry (`space` or `bounding_area`).
    let root_name = if raw.composition.contains_key("space") {
        "space"
    } else if raw.composition.contains_key("bounding_area") {
        "bounding_area"
    } else {
        return Err(RecipeError::MissingRoot);
    };

    // 4. Build the root compartment (the bounding box).
    let mut compartments: IndexMap<String, Compartment> = IndexMap::new();
    compartments.insert(
        root_name.to_string(),
        Compartment {
            name: root_name.to_string(),
            kind: CompartmentKind::Box(bounding_box),
        },
    );

    // 5. Walk the composition tree, accumulating directives.
    let mut directives: Vec<PlacementDirective> = Vec::new();
    walk_composition(
        root_name,
        root_name,
        &raw.composition,
        &resolved_objects,
        &ingredients,
        &mut compartments,
        &mut directives,
    )?;

    Ok(Recipe {
        name: raw.name.unwrap_or_else(|| "unnamed".to_string()),
        bounding_box,
        ingredients,
        compartments,
        directives,
    })
}

fn resolve_object(
    name: &str,
    raw: &IndexMap<String, RawObject>,
    out: &mut IndexMap<String, RawObject>,
    visiting: &mut Vec<String>,
) -> Result<(), RecipeError> {
    if out.contains_key(name) {
        return Ok(());
    }
    if visiting.iter().any(|v| v == name) {
        return Err(RecipeError::InheritanceCycle(name.to_string()));
    }
    visiting.push(name.to_string());

    let obj = raw
        .get(name)
        .ok_or_else(|| RecipeError::UnknownObject(name.to_string()))?
        .clone();

    let resolved = if let Some(parent) = obj.inherit.clone() {
        resolve_object(&parent, raw, out, visiting)?;
        let parent_obj = out.get(&parent).unwrap().clone();
        merge_object(parent_obj, obj)
    } else {
        obj
    };

    out.insert(name.to_string(), resolved);
    visiting.pop();
    Ok(())
}

fn merge_object(parent: RawObject, child: RawObject) -> RawObject {
    RawObject {
        type_name: child.type_name.or(parent.type_name),
        inherit: child.inherit.or(parent.inherit),
        color: child.color.or(parent.color),
        radius: child.radius.or(parent.radius),
        jitter_attempts: child.jitter_attempts.or(parent.jitter_attempts),
        packing_mode: child.packing_mode.or(parent.packing_mode),
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_composition(
    entry_name: &str,
    enclosing_compartment: &str,
    raw_comp: &IndexMap<String, RawCompositionEntry>,
    resolved_objects: &IndexMap<String, RawObject>,
    ingredients: &IndexMap<String, Ingredient>,
    _compartments: &mut IndexMap<String, Compartment>,
    directives: &mut Vec<PlacementDirective>,
) -> Result<(), RecipeError> {
    let entry = raw_comp
        .get(entry_name)
        .ok_or_else(|| RecipeError::UnknownCompositionEntry(entry_name.to_string()))?;

    // If this entry references an object, treat it as a placement
    // directive (placed in its parent compartment).
    if let Some(obj_name) = &entry.object {
        let count = match (entry.count, entry.molarity) {
            (Some(c), _) => c,
            (None, Some(_m)) => {
                // molarity = molecules per litre; need volume in litres to convert.
                // For Phase 2 MVP, require explicit count. Note as future work.
                return Err(RecipeError::InvalidComposition {
                    name: entry_name.to_string(),
                    message: "molarity not yet supported; use `count`".into(),
                });
            }
            (None, None) => {
                return Err(RecipeError::InvalidComposition {
                    name: entry_name.to_string(),
                    message: "needs `count` or `molarity`".into(),
                });
            }
        };
        if !ingredients.contains_key(obj_name) {
            // The object exists in the recipe but isn't an ingredient
            // (e.g. mesh compartment). Phase 2 MVP doesn't handle this.
            let typ = resolved_objects
                .get(obj_name)
                .and_then(|o| o.type_name.clone())
                .unwrap_or_else(|| "<unknown>".into());
            return Err(RecipeError::UnsupportedIngredient(format!(
                "{obj_name} (type {typ})"
            )));
        }
        directives.push(PlacementDirective {
            ingredient: obj_name.clone(),
            compartment: enclosing_compartment.to_string(),
            region: RegionKind::Interior,
            count,
            priority: entry.priority.unwrap_or(0.0),
            packing_mode: PackingMode::default(),
        });
    }

    // If this entry has regions, recurse into them with this entry's
    // name as the enclosing compartment (the entry itself is a
    // container, possibly *also* a placed object — but Phase 2 MVP
    // doesn't handle nested compartments).
    if let Some(regions) = &entry.regions {
        let next_compartment = if entry_name == enclosing_compartment {
            // root: stay with the root compartment
            enclosing_compartment
        } else {
            // Nested compartment — phase 2 doesn't support it yet.
            return Err(RecipeError::InvalidComposition {
                name: entry_name.to_string(),
                message: "nested compartments not yet supported".into(),
            });
        };

        for (region_name, entries) in regions {
            let region = match region_name.as_str() {
                "interior" => RegionKind::Interior,
                "surface" => RegionKind::Surface,
                other => {
                    return Err(RecipeError::InvalidComposition {
                        name: entry_name.to_string(),
                        message: format!("unknown region `{other}`"),
                    });
                }
            };
            for re in entries {
                match re {
                    RawRegionEntry::Ref(ref_name) => {
                        // Recurse into the referenced entry; treat its placements as
                        // being in this region.
                        walk_composition_region(
                            ref_name,
                            next_compartment,
                            region,
                            raw_comp,
                            resolved_objects,
                            ingredients,
                            directives,
                        )?;
                    }
                    RawRegionEntry::Inline { object, count, molarity: _, priority } => {
                        let count = count.ok_or(RecipeError::InvalidComposition {
                            name: entry_name.to_string(),
                            message: "inline directive needs `count`".into(),
                        })?;
                        if !ingredients.contains_key(object) {
                            return Err(RecipeError::UnsupportedIngredient(object.clone()));
                        }
                        directives.push(PlacementDirective {
                            ingredient: object.clone(),
                            compartment: next_compartment.to_string(),
                            region,
                            count,
                            priority: priority.unwrap_or(0.0),
                            packing_mode: PackingMode::default(),
                        });
                    }
                }
            }
        }
    }

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn walk_composition_region(
    entry_name: &str,
    enclosing_compartment: &str,
    region: RegionKind,
    raw_comp: &IndexMap<String, RawCompositionEntry>,
    resolved_objects: &IndexMap<String, RawObject>,
    ingredients: &IndexMap<String, Ingredient>,
    directives: &mut Vec<PlacementDirective>,
) -> Result<(), RecipeError> {
    let entry = raw_comp
        .get(entry_name)
        .ok_or_else(|| RecipeError::UnknownCompositionEntry(entry_name.to_string()))?;

    if let Some(obj_name) = &entry.object {
        let count = entry.count.ok_or(RecipeError::InvalidComposition {
            name: entry_name.to_string(),
            message: "needs `count`".into(),
        })?;
        if !ingredients.contains_key(obj_name) {
            let typ = resolved_objects
                .get(obj_name)
                .and_then(|o| o.type_name.clone())
                .unwrap_or_else(|| "<unknown>".into());
            return Err(RecipeError::UnsupportedIngredient(format!(
                "{obj_name} (type {typ})"
            )));
        }
        directives.push(PlacementDirective {
            ingredient: obj_name.clone(),
            compartment: enclosing_compartment.to_string(),
            region,
            count,
            priority: entry.priority.unwrap_or(0.0),
            packing_mode: PackingMode::default(),
        });
    }

    if entry.regions.is_some() {
        return Err(RecipeError::InvalidComposition {
            name: entry_name.to_string(),
            message: "nested compartments not yet supported".into(),
        });
    }

    Ok(())
}

fn parse_packing_mode(s: Option<&str>) -> Result<PackingMode, RecipeError> {
    match s {
        None | Some("random") => Ok(PackingMode::Random),
        Some(other) => Err(RecipeError::UnsupportedPackingMode(other.to_string())),
    }
}

// ---------- tests ----------

#[cfg(test)]
mod tests {
    use super::*;

    const SPHERES_IN_A_BOX: &str = r#"{
        "version": "1.0.0",
        "format_version": "2.0",
        "name": "spheres_in_a_box",
        "bounding_box": [[0,0,0],[1000,1000,1000]],
        "objects": {
            "base": { "type": "single_sphere" },
            "sphere_100": { "type": "single_sphere", "inherit": "base", "radius": 100, "color": [0.5,0.5,0.5] },
            "sphere_200": { "type": "single_sphere", "inherit": "base", "radius": 200 },
            "sphere_50":  { "type": "single_sphere", "inherit": "base", "radius": 50 },
            "sphere_25":  { "type": "single_sphere", "inherit": "base", "radius": 25 }
        },
        "composition": {
            "space": { "regions": { "interior": ["A","B","C","D"] } },
            "A": { "object": "sphere_100", "count": 60 },
            "B": { "object": "sphere_200", "count": 20 },
            "C": { "object": "sphere_50",  "count": 150 },
            "D": { "object": "sphere_25",  "count": 400 }
        }
    }"#;

    #[test]
    fn loads_spheres_in_a_box() {
        let r = Recipe::from_json_str(SPHERES_IN_A_BOX).expect("load");
        assert_eq!(r.name, "spheres_in_a_box");
        assert_eq!(r.bounding_box.min, nalgebra::Point3::new(0.0, 0.0, 0.0));
        assert_eq!(r.bounding_box.max, nalgebra::Point3::new(1000.0, 1000.0, 1000.0));
        // Four sized ingredients (base has no radius, gets dropped).
        assert_eq!(r.ingredients.len(), 4);
        assert!(r.ingredients.contains_key("sphere_100"));
        assert!(r.ingredients.contains_key("sphere_25"));
        // One compartment (`space`).
        assert_eq!(r.compartments.len(), 1);
        assert!(matches!(
            r.compartments["space"].kind,
            CompartmentKind::Box(_)
        ));
        // Four placement directives.
        assert_eq!(r.directives.len(), 4);
        let total: u32 = r.directives.iter().map(|d| d.count).sum();
        assert_eq!(total, 60 + 20 + 150 + 400);
        for d in &r.directives {
            assert_eq!(d.compartment, "space");
            assert_eq!(d.region, RegionKind::Interior);
        }
    }

    #[test]
    fn inheritance_merges_fields() {
        let src = r#"{
            "bounding_box": [[0,0,0],[100,100,100]],
            "objects": {
                "parent": { "type": "single_sphere", "radius": 5, "color": [1,0,0] },
                "child":  { "inherit": "parent", "radius": 10 }
            },
            "composition": {
                "space": { "regions": { "interior": ["A"] } },
                "A": { "object": "child", "count": 1 }
            }
        }"#;
        let r = Recipe::from_json_str(src).unwrap();
        let child = &r.ingredients["child"];
        assert!(matches!(child.shape, IngredientShape::SingleSphere { radius } if radius == 10.0));
        // Color inherited from parent.
        assert_eq!(child.color, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn inheritance_cycle_errors() {
        let src = r#"{
            "bounding_box": [[0,0,0],[1,1,1]],
            "objects": {
                "a": { "type": "single_sphere", "inherit": "b", "radius": 1 },
                "b": { "type": "single_sphere", "inherit": "a", "radius": 1 }
            },
            "composition": {
                "space": { "regions": { "interior": [] } }
            }
        }"#;
        let err = Recipe::from_json_str(src).unwrap_err();
        assert!(matches!(err, RecipeError::InheritanceCycle(_)), "got {err}");
    }

    #[test]
    fn unknown_object_in_directive_errors() {
        let src = r#"{
            "bounding_box": [[0,0,0],[1,1,1]],
            "objects": {
                "sphere": { "type": "single_sphere", "radius": 1 }
            },
            "composition": {
                "space": { "regions": { "interior": ["X"] } },
                "X": { "object": "ghost", "count": 1 }
            }
        }"#;
        let err = Recipe::from_json_str(src).unwrap_err();
        assert!(matches!(err, RecipeError::UnsupportedIngredient(_)), "got {err}");
    }

    #[test]
    fn missing_root_errors() {
        let src = r#"{
            "bounding_box": [[0,0,0],[1,1,1]],
            "objects": {},
            "composition": { "foo": {} }
        }"#;
        let err = Recipe::from_json_str(src).unwrap_err();
        assert!(matches!(err, RecipeError::MissingRoot));
    }

    #[test]
    fn loads_real_spheres_in_a_box_from_cellpack() {
        // Smoke test against the actual file in the cellpack clone, if present.
        let path = std::path::Path::new(
            "/home/pattern/code/cellpack/examples/recipes/v2/spheres_in_a_box.json",
        );
        if !path.exists() {
            return; // skip if cellpack isn't available
        }
        let r = Recipe::from_file(path).expect("load real recipe");
        assert_eq!(r.directives.iter().map(|d| d.count).sum::<u32>(), 630);
    }

    #[test]
    fn inline_directive_supported() {
        let src = r#"{
            "bounding_box": [[0,0,0],[100,100,100]],
            "objects": {
                "s": { "type": "single_sphere", "radius": 1 }
            },
            "composition": {
                "space": { "regions": { "interior": [ { "object": "s", "count": 5 } ] } }
            }
        }"#;
        let r = Recipe::from_json_str(src).unwrap();
        assert_eq!(r.directives.len(), 1);
        assert_eq!(r.directives[0].count, 5);
    }
}
