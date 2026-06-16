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
    #[serde(default)]
    chromosome: Option<RawChromosome>,
}

/// Recipe-level declaration of the genome fiber — a single chromosome
/// the packer generates as a constrained self-avoiding walk in its cell.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawChromosome {
    beads: usize,
    spacing: f32,
    #[serde(default)]
    bead_radius: Option<f32>,
    #[serde(default)]
    color: Option<[f32; 3]>,
    #[serde(default)]
    compartment: Option<String>,
    /// Optional plectonemic supercoiling: wind the genome as an interwound
    /// double helix along a backbone axis instead of a plain walk.
    #[serde(default)]
    supercoil: Option<RawSupercoil>,
    /// DNA-binding proteins to pack along the fiber (RNAP, replisome, …).
    #[serde(default)]
    proteins: Vec<RawFiberProtein>,
    /// Optional gene-annotation CSV (resolved relative to the recipe). When
    /// set, DNA-binding proteins are seated at real transcription sites
    /// (genes, abundance-weighted) instead of uniformly along the fiber.
    #[serde(default)]
    genome: Option<String>,
    /// Ingredient name of a per-bead dsDNA mesh. When set, the chromosome is
    /// rendered as instanced LOD segments of it (molecular, Goodsell-shaded)
    /// tiled along the genome path, instead of a single smooth tube.
    #[serde(default)]
    segment: Option<String>,
    /// Number of chromosome copies to lay out, each in its own sub-region of
    /// the cell (1 = a single nucleoid; 2 = a pre-division cell). `beads` is
    /// the bead count *per chromosome*.
    #[serde(default)]
    n_chromosomes: Option<usize>,
    /// Replication extent: how far each fork has travelled along its replichore
    /// (0..1). 0 = unreplicated; >0 draws a theta bubble + two forks per
    /// chromosome.
    #[serde(default)]
    fork_fraction: Option<f32>,
    /// Ingredient name placed at each replication fork (e.g. the replisome /
    /// DNA polymerase). Optional.
    #[serde(default)]
    fork_marker: Option<String>,
}

/// Superhelix parameters for a plectonemically supercoiled chromosome.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawSupercoil {
    /// Superhelix radius (how fat the coil is).
    radius: f32,
    /// Axial rise per turn (how stretched the coil is).
    pitch: f32,
    /// Number of plectoneme loop domains (rosette nucleoid). `None`/1 = one
    /// global plectoneme; higher = Maritan-style supercoiled domains.
    #[serde(default)]
    domains: Option<usize>,
}

/// A protein species bound along the chromosome. `object` names an entry in
/// `objects` (any ingredient shape); `count` instances are placed on the fiber.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawFiberProtein {
    object: String,
    count: u32,
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
    /// Multi-sphere proxy positions, parallel to `radii`.
    #[serde(default)]
    positions: Option<Vec<[f32; 3]>>,
    /// Multi-sphere proxy radii, parallel to `positions`.
    #[serde(default)]
    radii: Option<Vec<f32>>,
    /// Surface-alignment axis for membrane-anchored ingredients.
    #[serde(default)]
    principal_vector: Option<[f32; 3]>,
    /// `single_cube`: full side lengths along each axis (cellPACK
    /// stores `edges`; we accept `size` for clarity).
    #[serde(default)]
    size: Option<[f32; 3]>,
    /// `single_cylinder`: length of cylinder along local Z axis.
    #[serde(default)]
    length: Option<f32>,
    /// `multi_cylinder`: list of segments, each a `(start, end,
    /// radius)` capsule axis in ingredient-local space.
    #[serde(default)]
    segments: Option<Vec<RawCylinderSegment>>,
    /// `mesh`: path to an OBJ file (resolved relative to the recipe
    /// file's parent directory). Legacy single-LOD form; modern
    /// recipes should use `mesh_lods`.
    #[serde(default)]
    mesh_path: Option<String>,
    /// `mesh`: voxel size for generating the sphere-tree proxy. If
    /// absent, defaults to `mesh_aabb_diagonal / 16` (keeps the proxy
    /// tree under a few hundred spheres for reasonable mesh sizes).
    #[serde(default)]
    proxy_voxel_size: Option<f32>,
    /// `mesh`: multi-LOD pyramid. Each entry has a `path` (OBJ,
    /// resolved relative to the recipe) and the `voxel_size` it was
    /// generated at (drives the renderer's LOD selection). Order
    /// coarse → fine. If both `mesh_path` and `mesh_lods` are set,
    /// `mesh_lods` wins; the collision proxy uses the finest entry.
    #[serde(default)]
    mesh_lods: Option<Vec<RawMeshLod>>,
    /// Optional ingredient name of a per-bead segment mesh. On a `multi_sphere`
    /// ingredient, the pack writer renders each instance as that mesh tiled
    /// along its bead chain (e.g. mRNA → a real RNA strand).
    #[serde(default)]
    segment: Option<String>,
    // (remaining cellPACK fields ignored until needed)
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawMeshLod {
    /// OBJ file path, resolved relative to the recipe's directory.
    path: String,
    /// Voxel size (Å) the mesh was generated at — used by the viewer
    /// to pick which LOD to fetch at a given zoom level.
    voxel_size: f32,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
struct RawCylinderSegment {
    start: [f32; 3],
    end: [f32; 3],
    radius: f32,
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
    /// Extended schema: declare an analytical compartment directly,
    /// without going through a mesh ingredient. Non-cellPACK but
    /// the simplest way to spell capsule / sphere compartments.
    #[serde(default)]
    compartment: Option<RawCompartmentSpec>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
enum RawCompartmentSpec {
    Box { min: [f32; 3], max: [f32; 3] },
    Sphere { center: [f32; 3], radius: f32 },
    Capsule { a: [f32; 3], b: [f32; 3], radius: f32 },
    /// Triangle mesh compartment. `mesh_path` is resolved relative
    /// to the recipe file's parent directory.
    Mesh { mesh_path: String },
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
/// field.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum PackingMode {
    #[default]
    Random,
    /// Even ("Fibonacci") tiling over a compartment surface — for dense
    /// regular layers like a lipid bilayer. Falls back to random on
    /// compartments without an analytic even tiling.
    Tiled,
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
    /// Genome fiber spec (one chromosome the placer generates), if any.
    pub chromosome: Option<ChromosomeSpec>,
}

/// Resolved chromosome spec (from the recipe-level `chromosome` field).
#[derive(Debug, Clone)]
pub struct ChromosomeSpec {
    pub beads: usize,
    pub spacing: f32,
    pub bead_radius: f32,
    pub color: [f32; 3],
    pub compartment: Option<String>,
    /// Plectonemic supercoiling, if requested.
    pub supercoil: Option<SupercoilSpec>,
    /// DNA-binding proteins to pack along the fiber: `(ingredient name, count)`.
    pub proteins: Vec<(String, u32)>,
    /// Resolved path to the gene-annotation CSV, if the recipe set `genome`.
    pub genome: Option<std::path::PathBuf>,
    /// Ingredient name of the per-bead dsDNA mesh (instanced LOD rendering).
    pub segment: Option<String>,
    /// Number of chromosome copies (each in its own cell sub-region). >= 1.
    pub n_chromosomes: usize,
    /// Replication extent (0..1); 0 = unreplicated.
    pub fork_fraction: f32,
    /// Ingredient name placed at each replication fork, if any.
    pub fork_marker: Option<String>,
}

/// Resolved superhelix parameters (see [`RawSupercoil`]).
#[derive(Debug, Clone, Copy)]
pub struct SupercoilSpec {
    pub radius: f32,
    pub pitch: f32,
    /// Plectoneme loop domains (rosette nucleoid); 1 = single global plectoneme.
    pub domains: usize,
}

impl Recipe {
    pub fn from_json_str(src: &str) -> Result<Self, RecipeError> {
        let raw: RawRecipe = serde_json::from_str(src)?;
        resolve(raw, None, None)
    }

    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, RecipeError> {
        Self::from_file_with_proxy_lod(path, None)
    }

    /// Like [`Recipe::from_file`], but builds each mesh ingredient's collision
    /// proxy from a chosen LOD: `proxy_lod` indexes `mesh_lods` (0 = coarsest,
    /// clamped per ingredient); `None` uses the finest. A coarser proxy LOD is
    /// far lighter to load + voxelise at whole-cell scale and barely changes
    /// the result, since the proxy is voxelised at `proxy_voxel_size` regardless.
    pub fn from_file_with_proxy_lod(
        path: impl AsRef<std::path::Path>,
        proxy_lod: Option<usize>,
    ) -> Result<Self, RecipeError> {
        let path = path.as_ref();
        let src = std::fs::read_to_string(path)
            .map_err(|e| RecipeError::Io(e.to_string()))?;
        let raw: RawRecipe = serde_json::from_str(&src)?;
        // Mesh-ingredient paths in the recipe are resolved relative to
        // the recipe file's parent directory.
        let base = path.parent().map(|p| p.to_path_buf());
        resolve(raw, base.as_deref(), proxy_lod)
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

fn resolve(
    raw: RawRecipe,
    recipe_dir: Option<&std::path::Path>,
    proxy_lod: Option<usize>,
) -> Result<Recipe, RecipeError> {
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
        // For Mesh ingredients we remember the LOD pyramid so the
        // viewer can fetch the right resolution per zoom level. URLs
        // are resolved to project-root-relative paths so an http
        // server serving the project root can deliver them.
        let mut mesh_lods: Vec<crate::ingredient::MeshLod> = Vec::new();
        let shape = match typ {
            "single_sphere" => {
                let Some(r) = obj.radius else {
                    continue;
                };
                IngredientShape::SingleSphere { radius: r }
            }
            "multi_sphere" => {
                let positions = obj.positions.as_ref().ok_or_else(|| RecipeError::InvalidObject {
                    name: name.clone(),
                    message: "multi_sphere requires `positions`".into(),
                })?;
                let radii = obj.radii.as_ref().ok_or_else(|| RecipeError::InvalidObject {
                    name: name.clone(),
                    message: "multi_sphere requires `radii`".into(),
                })?;
                if positions.len() != radii.len() {
                    return Err(RecipeError::InvalidObject {
                        name: name.clone(),
                        message: format!(
                            "multi_sphere positions ({}) and radii ({}) length mismatch",
                            positions.len(),
                            radii.len()
                        ),
                    });
                }
                let spheres: Vec<crate::ingredient::ProxySphere> = positions
                    .iter()
                    .zip(radii.iter())
                    .map(|(p, r)| crate::ingredient::ProxySphere {
                        offset: nalgebra::Vector3::new(p[0], p[1], p[2]),
                        radius: *r,
                    })
                    .collect();
                IngredientShape::MultiSphere { spheres }
            }
            "single_cube" => {
                let Some(size) = obj.size else {
                    return Err(RecipeError::InvalidObject {
                        name: name.clone(),
                        message: "single_cube requires `size` (3-element side lengths)".into(),
                    });
                };
                let half = nalgebra::Vector3::new(size[0] * 0.5, size[1] * 0.5, size[2] * 0.5);
                IngredientShape::cube(half)
            }
            "single_cylinder" => {
                let Some(length) = obj.length else {
                    return Err(RecipeError::InvalidObject {
                        name: name.clone(),
                        message: "single_cylinder requires `length`".into(),
                    });
                };
                let Some(radius) = obj.radius else {
                    return Err(RecipeError::InvalidObject {
                        name: name.clone(),
                        message: "single_cylinder requires `radius`".into(),
                    });
                };
                IngredientShape::cylinder(length, radius)
            }
            "multi_cylinder" => {
                let Some(segments) = obj.segments.as_ref() else {
                    return Err(RecipeError::InvalidObject {
                        name: name.clone(),
                        message: "multi_cylinder requires `segments`".into(),
                    });
                };
                let segs: Vec<crate::ingredient::CylinderSegment> = segments
                    .iter()
                    .map(|s| crate::ingredient::CylinderSegment {
                        start: nalgebra::Point3::new(s.start[0], s.start[1], s.start[2]),
                        end: nalgebra::Point3::new(s.end[0], s.end[1], s.end[2]),
                        radius: s.radius,
                    })
                    .collect();
                IngredientShape::multi_cylinder(&segs)
            }
            "mesh" => {
                // Determine LOD set. Prefer `mesh_lods`; fall back
                // to the legacy single `mesh_path` (treated as one
                // LOD at the proxy resolution).
                let resolved_lods: Vec<(std::path::PathBuf, f32, String)> =
                    if let Some(raw_lods) = obj.mesh_lods.as_ref() {
                        if raw_lods.is_empty() {
                            return Err(RecipeError::InvalidObject {
                                name: name.clone(),
                                message: "mesh_lods must contain at least one entry".into(),
                            });
                        }
                        raw_lods
                            .iter()
                            .map(|lod| {
                                let full = match recipe_dir {
                                    Some(dir) => dir.join(&lod.path),
                                    None => std::path::PathBuf::from(&lod.path),
                                };
                                let url = resolve_mesh_url(&full);
                                (full, lod.voxel_size, url)
                            })
                            .collect()
                    } else if let Some(rel_path) = obj.mesh_path.as_ref() {
                        let full = match recipe_dir {
                            Some(dir) => dir.join(rel_path),
                            None => std::path::PathBuf::from(rel_path),
                        };
                        let url = resolve_mesh_url(&full);
                        // Without explicit LODs, the legacy
                        // `proxy_voxel_size` doubles as the LOD
                        // descriptor; if absent we leave it as 1.0
                        // and the viewer falls back to "always use
                        // this one".
                        let voxel_size = obj.proxy_voxel_size.unwrap_or(1.0);
                        vec![(full, voxel_size, url)]
                    } else {
                        return Err(RecipeError::InvalidObject {
                            name: name.clone(),
                            message: "mesh requires `mesh_path` or `mesh_lods`".into(),
                        });
                    };

                // Record LOD URLs (coarse → fine ordering preserved
                // from input).
                mesh_lods = resolved_lods
                    .iter()
                    .map(|(_, vs, url)| crate::ingredient::MeshLod {
                        url: url.clone(),
                        voxel_size: *vs,
                    })
                    .collect();

                // Collision-proxy LOD: `proxy_lod` (0 = coarsest) clamped to
                // the available LODs; default (None) is the finest (last
                // entry). Coarser is much lighter to load + voxelise at
                // whole-cell scale and barely changes the proxy, since it's
                // voxelised at `proxy_voxel_size` either way.
                let lod_idx = match proxy_lod {
                    Some(i) => i.min(resolved_lods.len() - 1),
                    None => resolved_lods.len() - 1,
                };
                let (full, _, _) = &resolved_lods[lod_idx];
                let mut trimesh = crate::ingredient::obj::load_trimesh(full)
                    .map_err(|e| RecipeError::InvalidObject {
                        name: name.clone(),
                        message: format!("failed to load mesh `{}`: {e}", full.display()),
                    })?;
                // Configure for in/out point queries — required by the
                // sphere-tree voxeliser.
                parsimony_spatial::prepare_trimesh_for_voxelize(&mut trimesh)
                    .map_err(|e| RecipeError::InvalidObject {
                        name: name.clone(),
                        message: format!("mesh orient: {e}"),
                    })?;
                let voxel_size = obj.proxy_voxel_size.unwrap_or_else(|| {
                    let a = trimesh.local_aabb();
                    let extents = a.maxs - a.mins;
                    (extents.x.powi(2) + extents.y.powi(2) + extents.z.powi(2)).sqrt() / 16.0
                });
                IngredientShape::mesh_with_voxelised_proxies(
                    std::sync::Arc::new(trimesh),
                    voxel_size,
                )
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
                principal_vector: obj
                    .principal_vector
                    .map(|v| nalgebra::Vector3::new(v[0], v[1], v[2]))
                    .unwrap_or_else(|| nalgebra::Vector3::new(0.0, 0.0, 1.0)),
                mesh_lods,
                segment: obj.segment.clone(),
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
            parent: None,
            children: Vec::new(),
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
        recipe_dir,
    )?;

    // Chromosome-bound protein references must resolve to real ingredients.
    if let Some(chr) = &raw.chromosome {
        for p in &chr.proteins {
            if !ingredients.contains_key(&p.object) {
                return Err(RecipeError::UnsupportedIngredient(format!(
                    "{} (chromosome protein)",
                    p.object
                )));
            }
        }
    }

    Ok(Recipe {
        name: raw.name.unwrap_or_else(|| "unnamed".to_string()),
        bounding_box,
        ingredients,
        compartments,
        directives,
        chromosome: raw.chromosome.map(|c| ChromosomeSpec {
            beads: c.beads,
            spacing: c.spacing,
            bead_radius: c.bead_radius.unwrap_or(8.0),
            color: c.color.unwrap_or([0.9, 0.4, 0.5]),
            compartment: c.compartment,
            supercoil: c
                .supercoil
                .map(|s| SupercoilSpec {
                    radius: s.radius,
                    pitch: s.pitch,
                    domains: s.domains.unwrap_or(1),
                }),
            proteins: c.proteins.iter().map(|p| (p.object.clone(), p.count)).collect(),
            genome: c.genome.as_ref().map(|g| match recipe_dir {
                Some(dir) => dir.join(g),
                None => std::path::PathBuf::from(g),
            }),
            segment: c.segment,
            n_chromosomes: c.n_chromosomes.unwrap_or(1).max(1),
            fork_fraction: c.fork_fraction.unwrap_or(0.0),
            fork_marker: c.fork_marker,
        }),
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

/// Map an OBJ path (relative to the recipe) into a URL suitable for
/// downstream renderers. Prefer a path relative to the current
/// working directory (so an http server rooted there can serve it);
/// fall back to canonicalised absolute.
fn resolve_mesh_url(full: &std::path::Path) -> String {
    let absolute = std::fs::canonicalize(full).unwrap_or_else(|_| full.to_path_buf());
    let cwd = std::env::current_dir().ok();
    let stripped = cwd.and_then(|c| absolute.strip_prefix(&c).ok().map(|p| p.to_path_buf()));
    stripped.unwrap_or(absolute).to_string_lossy().into_owned()
}

fn merge_object(parent: RawObject, child: RawObject) -> RawObject {
    RawObject {
        type_name: child.type_name.or(parent.type_name),
        inherit: child.inherit.or(parent.inherit),
        color: child.color.or(parent.color),
        radius: child.radius.or(parent.radius),
        jitter_attempts: child.jitter_attempts.or(parent.jitter_attempts),
        packing_mode: child.packing_mode.or(parent.packing_mode),
        positions: child.positions.or(parent.positions),
        radii: child.radii.or(parent.radii),
        principal_vector: child.principal_vector.or(parent.principal_vector),
        size: child.size.or(parent.size),
        length: child.length.or(parent.length),
        segments: child.segments.or(parent.segments),
        mesh_path: child.mesh_path.or(parent.mesh_path),
        proxy_voxel_size: child.proxy_voxel_size.or(parent.proxy_voxel_size),
        mesh_lods: child.mesh_lods.or(parent.mesh_lods),
        segment: child.segment.or(parent.segment),
    }
}

#[allow(clippy::too_many_arguments)]
fn walk_composition(
    entry_name: &str,
    enclosing_compartment: &str,
    raw_comp: &IndexMap<String, RawCompositionEntry>,
    resolved_objects: &IndexMap<String, RawObject>,
    ingredients: &IndexMap<String, Ingredient>,
    compartments: &mut IndexMap<String, Compartment>,
    directives: &mut Vec<PlacementDirective>,
    recipe_dir: Option<&std::path::Path>,
) -> Result<(), RecipeError> {
    let entry = raw_comp
        .get(entry_name)
        .ok_or_else(|| RecipeError::UnknownCompositionEntry(entry_name.to_string()))?
        .clone();
    let entry = &entry;

    // If this entry declares its own analytical compartment, register
    // it as a child of `enclosing_compartment`.
    if let Some(spec) = &entry.compartment {
        if compartments.contains_key(entry_name) {
            return Err(RecipeError::InvalidComposition {
                name: entry_name.to_string(),
                message: "duplicate compartment".into(),
            });
        }
        let kind = match spec {
            RawCompartmentSpec::Box { min, max } => CompartmentKind::Box(parsimony_spatial::Aabb::new(
                nalgebra::Point3::new(min[0], min[1], min[2]),
                nalgebra::Point3::new(max[0], max[1], max[2]),
            )),
            RawCompartmentSpec::Sphere { center, radius } => CompartmentKind::Sphere {
                center: nalgebra::Point3::new(center[0], center[1], center[2]),
                radius: *radius,
            },
            RawCompartmentSpec::Capsule { a, b, radius } => CompartmentKind::Capsule {
                a: nalgebra::Point3::new(a[0], a[1], a[2]),
                b: nalgebra::Point3::new(b[0], b[1], b[2]),
                radius: *radius,
            },
            RawCompartmentSpec::Mesh { mesh_path } => {
                let full = match recipe_dir {
                    Some(dir) => dir.join(mesh_path),
                    None => std::path::PathBuf::from(mesh_path),
                };
                let mut trimesh = crate::ingredient::obj::load_trimesh(&full).map_err(|e| {
                    RecipeError::InvalidComposition {
                        name: entry_name.to_string(),
                        message: format!("failed to load mesh `{}`: {e}", full.display()),
                    }
                })?;
                parsimony_spatial::prepare_trimesh_for_voxelize(&mut trimesh).map_err(|e| {
                    RecipeError::InvalidComposition {
                        name: entry_name.to_string(),
                        message: format!("mesh orient: {e}"),
                    }
                })?;
                let aabb_local = trimesh.local_aabb();
                let aabb = parsimony_spatial::Aabb::new(
                    nalgebra::Point3::new(aabb_local.mins.x, aabb_local.mins.y, aabb_local.mins.z),
                    nalgebra::Point3::new(aabb_local.maxs.x, aabb_local.maxs.y, aabb_local.maxs.z),
                );
                CompartmentKind::Mesh(Box::new(crate::compartment::MeshCompartment {
                    trimesh,
                    aabb,
                }))
            }
        };
        compartments.insert(
            entry_name.to_string(),
            Compartment {
                name: entry_name.to_string(),
                kind,
                parent: None,
                children: Vec::new(),
            },
        );
    }
    // What compartment do this entry's directives live in?
    // - If this entry declares its own compartment, it's the entry itself.
    // - Otherwise, placements live in the enclosing compartment.
    let entry_compartment = if entry.compartment.is_some() {
        entry_name
    } else {
        enclosing_compartment
    };

    // If this entry references an object, treat it as a placement
    // directive (placed in the enclosing compartment).
    if let Some(obj_name) = &entry.object {
        let count = match (entry.count, entry.molarity) {
            (Some(c), _) => c,
            (None, Some(m)) => molarity_to_count(m, enclosing_compartment, compartments)?,
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

    // If this entry has regions, recurse into them. The "enclosing"
    // for nested placements is this entry's own compartment if it
    // declared one, else the surrounding enclosing compartment.
    if let Some(regions) = &entry.regions {
        let next_compartment = entry_compartment;
        // Link parent/child if this entry has its own compartment.
        if entry.compartment.is_some() && enclosing_compartment != entry_name {
            let parent_id = compartments
                .get_index_of(enclosing_compartment)
                .map(|i| i as u32);
            let child_id = compartments
                .get_index_of(entry_name)
                .map(|i| i as u32)
                .expect("just inserted");
            if let Some(c) = compartments.get_mut(entry_name) {
                c.parent = parent_id;
            }
            if let Some(parent) = compartments.get_mut(enclosing_compartment) {
                parent.children.push(child_id);
            }
        }

        let regions = regions.clone();
        for (region_name, entries) in &regions {
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
                        walk_composition_region(
                            ref_name,
                            next_compartment,
                            region,
                            raw_comp,
                            resolved_objects,
                            ingredients,
                            compartments,
                            directives,
                            recipe_dir,
                        )?;
                    }
                    RawRegionEntry::Inline { object, count, molarity, priority } => {
                        let count = match (count, molarity) {
                            (Some(c), _) => *c,
                            (None, Some(m)) => {
                                molarity_to_count(*m, next_compartment, compartments)?
                            }
                            (None, None) => {
                                return Err(RecipeError::InvalidComposition {
                                    name: entry_name.to_string(),
                                    message: "inline directive needs `count` or `molarity`".into(),
                                });
                            }
                        };
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
    compartments: &mut IndexMap<String, Compartment>,
    directives: &mut Vec<PlacementDirective>,
    recipe_dir: Option<&std::path::Path>,
) -> Result<(), RecipeError> {
    let entry = raw_comp
        .get(entry_name)
        .ok_or_else(|| RecipeError::UnknownCompositionEntry(entry_name.to_string()))?;

    if let Some(obj_name) = &entry.object {
        let count = match (entry.count, entry.molarity) {
            (Some(c), _) => c,
            (None, Some(m)) => molarity_to_count(m, enclosing_compartment, compartments)?,
            (None, None) => {
                return Err(RecipeError::InvalidComposition {
                    name: entry_name.to_string(),
                    message: "needs `count` or `molarity`".into(),
                });
            }
        };
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

    // If this entry has its own compartment + regions, recurse via
    // the main walker (which handles building the nested compartment
    // and recursing into its regions). Pass the enclosing compartment
    // as the parent of this entry's compartment.
    if entry.compartment.is_some() || entry.regions.is_some() {
        walk_composition(
            entry_name,
            enclosing_compartment,
            raw_comp,
            resolved_objects,
            ingredients,
            compartments,
            directives,
            recipe_dir,
        )?;
    }

    Ok(())
}

fn parse_packing_mode(s: Option<&str>) -> Result<PackingMode, RecipeError> {
    match s {
        None | Some("random") => Ok(PackingMode::Random),
        Some("tiled") => Ok(PackingMode::Tiled),
        Some(other) => Err(RecipeError::UnsupportedPackingMode(other.to_string())),
    }
}

/// Convert a molar concentration to an integer instance count, given
/// the volume (in cubic Ångströms) of the enclosing compartment. The
/// cellPACK convention:
///
/// `count = molarity (M) × Avogadro's_number × volume_in_litres`
///
/// With `1 Å³ = 10⁻²⁷ L` and `Avogadro = 6.022 × 10²³`:
///
/// `count = molarity × 6.022 × 10⁻⁴ × volume_in_Å³`
///
/// This matches `cellpack.autopack.Recipe.setCount` (`Recipe.py:135`).
fn molarity_to_count(
    molarity: f32,
    enclosing_compartment: &str,
    compartments: &IndexMap<String, Compartment>,
) -> Result<u32, RecipeError> {
    let comp = compartments
        .get(enclosing_compartment)
        .ok_or_else(|| RecipeError::InvalidComposition {
            name: enclosing_compartment.to_string(),
            message: "molarity directive refers to an unknown compartment".into(),
        })?;
    let volume = comp.kind.volume();
    let n = molarity * 6.022e-4 * volume;
    if !n.is_finite() || n < 0.0 {
        return Err(RecipeError::InvalidComposition {
            name: enclosing_compartment.to_string(),
            message: format!("molarity {molarity} × volume {volume} produced invalid count {n}"),
        });
    }
    Ok(n.round() as u32)
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
    fn loads_real_spheres_in_a_box() {
        // Smoke test against the vendored recipe (examples/recipes/),
        // resolved relative to this crate so it's machine-independent.
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/recipes/spheres_in_a_box.json"
        );
        let r = Recipe::from_file(std::path::Path::new(path)).expect("load vendored recipe");
        assert_eq!(r.directives.iter().map(|d| d.count).sum::<u32>(), 630);
    }

    #[test]
    fn molarity_converts_to_count() {
        // 1 µM in 1 µm³ = (1e-6 M) × 6.022e-4 × (1e12 Å³) = 0.6022 instances ≈ 1.
        // Recipe uses a 1000 Å³ box with 1 mM, so:
        //   1e-3 × 6.022e-4 × 1e9 = 602.2 → 602.
        let src = r#"{
            "bounding_box": [[0,0,0],[1000,1000,1000]],
            "objects": {
                "s": { "type": "single_sphere", "radius": 5 }
            },
            "composition": {
                "space": { "regions": { "interior": [ { "object": "s", "molarity": 1.0e-3 } ] } }
            }
        }"#;
        // Inline directive with molarity isn't supported yet (Phase 2 MVP
        // only allows molarity on top-level directives via the `space`
        // entry's molarity field on a referenced entry). Verify the
        // top-level molarity path:
        let _ = src;

        let src2 = r#"{
            "bounding_box": [[0,0,0],[1000,1000,1000]],
            "objects": {
                "s": { "type": "single_sphere", "radius": 5 }
            },
            "composition": {
                "space": { "regions": { "interior": ["A"] } },
                "A": { "object": "s", "molarity": 1.0e-3 }
            }
        }"#;
        let r = Recipe::from_json_str(src2).unwrap();
        // 1e-3 M × 6.022e-4 × 1e9 Å³ = 602.2 ≈ 602
        assert_eq!(r.directives.len(), 1);
        assert!(
            (601..=603).contains(&r.directives[0].count),
            "expected ~602 from molarity conversion, got {}",
            r.directives[0].count
        );
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

    #[test]
    fn loads_single_cube() {
        let src = r#"{
            "bounding_box": [[0,0,0],[100,100,100]],
            "objects": {
                "block": { "type": "single_cube", "size": [10, 10, 10] }
            },
            "composition": {
                "space": { "regions": { "interior": [ { "object": "block", "count": 1 } ] } }
            }
        }"#;
        let r = Recipe::from_json_str(src).unwrap();
        let s = &r.ingredients["block"].shape;
        // Cube expands to a MultiSphere with eight octant spheres.
        let n = match s {
            IngredientShape::MultiSphere { spheres } => spheres.len(),
            _ => panic!("expected MultiSphere"),
        };
        assert_eq!(n, 8, "expected 2×2×2 = 8 proxy spheres for cube");
        // Enclosing radius: octant centre at half_extent/2 from origin
        // plus per-sphere radius (= ‖h‖/2); total = ‖h‖ = the cube's
        // half-diagonal. For a 10³ cube that's 5·√3 ≈ 8.66.
        let er = s.enclosing_radius();
        let expected = (75.0_f32).sqrt();
        assert!(
            (er - expected).abs() < 1e-4,
            "got {er}, expected {expected}"
        );
    }

    #[test]
    fn loads_single_cylinder() {
        let src = r#"{
            "bounding_box": [[0,0,0],[100,100,100]],
            "objects": {
                "rod": { "type": "single_cylinder", "length": 20, "radius": 3 }
            },
            "composition": {
                "space": { "regions": { "interior": [ { "object": "rod", "count": 1 } ] } }
            }
        }"#;
        let r = Recipe::from_json_str(src).unwrap();
        let s = &r.ingredients["rod"].shape;
        let n = match s {
            IngredientShape::MultiSphere { spheres } => spheres.len(),
            _ => panic!("expected MultiSphere"),
        };
        // length 20, radius 3 → ceil(20/3) = 7 intervals, 8 spheres.
        assert_eq!(n, 8);
        // Enclosing radius: endpoint at (0,0,10) plus radius 3 = 13.
        let er = s.enclosing_radius();
        assert!((er - 13.0).abs() < 1e-4, "got {er}");
    }

    #[test]
    fn loads_multi_cylinder() {
        let src = r#"{
            "bounding_box": [[0,0,0],[100,100,100]],
            "objects": {
                "chain": {
                    "type": "multi_cylinder",
                    "segments": [
                        { "start": [0, 0, 0], "end": [0, 0, 10], "radius": 2 },
                        { "start": [0, 0, 10], "end": [5, 0, 14], "radius": 2 }
                    ]
                }
            },
            "composition": {
                "space": { "regions": { "interior": [ { "object": "chain", "count": 1 } ] } }
            }
        }"#;
        let r = Recipe::from_json_str(src).unwrap();
        let s = &r.ingredients["chain"].shape;
        let n = match s {
            IngredientShape::MultiSphere { spheres } => spheres.len(),
            _ => panic!("expected MultiSphere"),
        };
        // Segment 1 (len 10, radius 2): 5 intervals → 6 spheres.
        // Segment 2 (len √41 ≈ 6.4, radius 2): 4 intervals → 5 spheres.
        // Total: 11 spheres.
        assert!(n >= 9 && n <= 12, "got {n} spheres");
    }

    #[test]
    fn loads_mesh_compartment_from_local_obj() {
        let mesh_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/meshes/sphere.obj"
        );
        // Quoted JSON values must escape backslashes; the path is
        // POSIX-clean here so we can inline directly.
        let src = format!(
            r#"{{
                "bounding_box": [[-2,-2,-2],[2,2,2]],
                "objects": {{
                    "marker": {{ "type": "single_sphere", "radius": 0.05 }}
                }},
                "composition": {{
                    "space": {{ "regions": {{ "interior": ["cell"] }} }},
                    "cell": {{
                        "compartment": {{ "kind": "mesh", "mesh_path": "{mesh_path}" }},
                        "regions": {{
                            "surface": [ {{ "object": "marker", "count": 5 }} ]
                        }}
                    }}
                }}
            }}"#
        );
        let r = Recipe::from_json_str(&src).expect("load");
        assert_eq!(r.compartments.len(), 2, "space + cell");
        let cell = &r.compartments["cell"];
        assert!(
            matches!(cell.kind, CompartmentKind::Mesh(_)),
            "cell should be a mesh compartment"
        );
        // Verify mesh compartment can sample a surface point and the
        // point is roughly on the mesh (within voxel scale).
        let mut rng = rand::thread_rng();
        for _ in 0..10 {
            let (p, n) = cell.kind.sample_surface(&mut rng);
            assert!(p.coords.norm().is_finite());
            assert!(n.norm() > 0.5, "normal should be near-unit length");
        }
    }

    #[test]
    fn loads_mesh_ingredient_from_local_obj() {
        // Use the repo's tessellated sphere.obj (examples/meshes).
        let mesh_path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../examples/meshes/sphere.obj"
        );
        let src = format!(
            r#"{{
                "bounding_box": [[0,0,0],[100,100,100]],
                "objects": {{
                    "mesh_ball": {{
                        "type": "mesh",
                        "mesh_path": "{mesh_path}",
                        "proxy_voxel_size": 0.2
                    }}
                }},
                "composition": {{
                    "space": {{ "regions": {{ "interior": [ {{ "object": "mesh_ball", "count": 1 }} ] }} }}
                }}
            }}"#
        );
        let r = Recipe::from_json_str(&src).unwrap();
        let s = &r.ingredients["mesh_ball"].shape;
        match s {
            IngredientShape::Mesh { trimesh, proxies } => {
                assert!(!proxies.is_empty(), "mesh proxies should be non-empty");
                assert!(trimesh.indices().len() > 100);
            }
            _ => panic!("expected Mesh shape"),
        }
    }
}
