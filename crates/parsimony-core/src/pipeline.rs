//! Staged packing pipeline — a dependency DAG with a content-addressed
//! cache.
//!
//! A whole-cell pack is expensive and made of loosely-coupled parts: the
//! genome fiber, the membrane, the cytoplasmic interior. They have a
//! natural order — pack the chromosome once, then pack the interior
//! *around* it — and most of the time only one part changes. This module
//! turns that into a small build system:
//!
//! - A [`Pipeline`] is a recipe plus a list of [`Stage`]s, each declaring
//!   what it packs ([`StageKind`]) and which stages it `depends_on`.
//! - Each stage's output (a partial [`Snapshot`]) is **content-addressed**:
//!   its cache key is a hash of everything that determines its result —
//!   the recipe's ingredient/compartment identity, the stage's own
//!   selection + seed, and its dependencies' cache keys. Identical inputs
//!   ⇒ identical key ⇒ the cached snapshot is reused.
//! - Changing one stage's inputs changes only its key (and its
//!   descendants', through the dependency edge), so a re-run regenerates
//!   exactly the stale subtree and reuses the rest. This is the "pack the
//!   chromosome once, then many interior packings" workflow.
//!
//! A dependency edge is also a packing constraint: a stage is packed with
//! its dependencies' geometry seeded as fixed obstacles (via
//! [`GreedyRandomPlacer::pack_with_obstacles`]), so the interior avoids
//! the chromosome it was built on.
//!
//! The global ingredient/compartment identity is folded into *every*
//! stage's key, so a cached snapshot's `ingredient_id`s can never go
//! stale: any add/remove/reorder/reshape of ingredients busts all stages.
//! Counts, seeds, and per-stage selection invalidate only the stages they
//! touch. (Mesh ingredients are keyed by their LOD URLs + triangle counts;
//! editing an OBJ in place without renaming needs `--force`.)

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use nalgebra::Point3;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use rand::SeedableRng;
use rand_xoshiro::Xoshiro256PlusPlus;

use crate::compartment::{Compartment, CompartmentKind};
use crate::ingredient::{Ingredient, IngredientShape};
use crate::placement::{Placement, Snapshot};
use crate::placer::{GreedyRandomPlacer, PlacementBackend, PlacerConfig};
use crate::recipe::{PackingMode, Recipe, RegionKind};

// ───── data model ────────────────────────────────────────────────────

fn default_true() -> bool {
    true
}

/// A staged packing: a recipe plus a DAG of stages.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Pipeline {
    /// Output name (the merged pack is written as `<name>.pack.json`).
    pub name: String,
    /// Recipe all stages draw from, resolved relative to the pipeline file.
    pub recipe: PathBuf,
    /// Base RNG seed; each stage derives an independent seed from it
    /// (unless the stage sets its own `seed`). Changing this re-rolls
    /// the whole pipeline.
    #[serde(default)]
    pub seed: u64,
    /// Whole-sphere containment for the root domain (parsimony default).
    #[serde(default = "default_true")]
    pub strict_bounds: bool,
    /// Placement engine for this pipeline's Pack stages. Default is the legacy
    /// grid+valid_cells engine; set `"backend": "octree"` for the content-scaled
    /// engine (whole-cell recipes).
    #[serde(default)]
    pub backend: PlacementBackend,
    pub stages: Vec<Stage>,
}

/// One node in the pipeline DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Stage {
    /// Unique within the pipeline; names the cache file + the report row.
    pub id: String,
    #[serde(flatten)]
    pub kind: StageKind,
    /// Stage ids this stage packs around (their geometry becomes
    /// obstacles, and their cache keys feed this stage's key).
    #[serde(default)]
    pub depends_on: Vec<String>,
    /// Per-stage seed override. Default: derived from the pipeline seed
    /// and the stage id, so stages are independent and re-rolling one
    /// (by setting this) doesn't disturb the others.
    #[serde(default)]
    pub seed: Option<u64>,
}

/// What a stage packs.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StageKind {
    /// Generate the recipe's chromosome fiber (and nothing else).
    Chromosome,
    /// Pack ingredient directives. `include` (if non-empty) keeps only
    /// those ingredients; `exclude` drops them. Empty/empty = pack all.
    Pack {
        #[serde(default)]
        include: Vec<String>,
        #[serde(default)]
        exclude: Vec<String>,
        /// Run the proxy-accurate densify phase for this stage (packs
        /// non-spherical mesh ingredients tighter than enclosing-sphere
        /// clearance allows).
        #[serde(default)]
        densify: bool,
        /// Clearance-grid cell size (Å). Finer than the autodetected
        /// `max_radius/8` recovers placements the coarse grid misses (and
        /// shrinks the densify margin), at higher memory/time. `None` = auto.
        #[serde(default)]
        clearance_cell_size: Option<f32>,
    },
    /// Bind the chromosome's DNA-binding proteins along the fiber produced
    /// by a `chromosome` dependency. The proteins + counts come from the
    /// recipe's `chromosome.proteins`; list the chromosome (and any stages
    /// whose geometry the proteins should avoid) in `depends_on`.
    FiberPack,
}

impl StageKind {
    fn label(&self) -> String {
        match self {
            StageKind::Chromosome => "chromosome".into(),
            StageKind::Pack { include, exclude, densify, .. } => {
                let sel = if !include.is_empty() {
                    format!("pack include=[{}]", include.join(","))
                } else if !exclude.is_empty() {
                    format!("pack exclude=[{}]", exclude.join(","))
                } else {
                    "pack all".into()
                };
                if *densify {
                    format!("{sel} +dense")
                } else {
                    sel
                }
            }
            StageKind::FiberPack => "fiber proteins".into(),
        }
    }
}

impl Stage {
    /// Effective RNG seed: the explicit override, else the pipeline seed
    /// mixed with this stage's id (so each stage is independent).
    fn effective_seed(&self, pipeline_seed: u64) -> u64 {
        self.seed.unwrap_or_else(|| {
            let mut h = Fnv::new();
            h.u64(pipeline_seed);
            h.string(&self.id);
            h.finish()
        })
    }
}

/// One stage's contribution to a run.
#[derive(Debug, Clone)]
pub struct StageReport {
    pub id: String,
    pub kind: String,
    pub cache_key: String,
    pub from_cache: bool,
    pub placed: usize,
    pub chromosome: bool,
}

/// The result of running a pipeline: the live recipe, the merged
/// snapshot (ready for [`write_pack_json`](crate::write_pack_json)), and
/// a per-stage report.
#[derive(Debug, Clone)]
pub struct PipelineRun {
    pub recipe: Recipe,
    pub merged: Snapshot,
    pub reports: Vec<StageReport>,
}

/// A stage's planned state without running it (for `status`).
#[derive(Debug, Clone)]
pub struct StagePlan {
    pub id: String,
    pub kind: String,
    pub cache_key: String,
    pub cached: bool,
    pub depends_on: Vec<String>,
}

#[derive(Debug, Error)]
pub enum PipelineError {
    #[error("{0}")]
    Io(#[from] std::io::Error),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error(transparent)]
    Recipe(#[from] crate::recipe::RecipeError),
    #[error("duplicate stage id `{0}`")]
    DuplicateStage(String),
    #[error("stage `{stage}` depends on unknown stage `{dep}`")]
    UnknownDep { stage: String, dep: String },
    #[error("stage dependency cycle (unresolved: {0})")]
    Cycle(String),
}

// ───── public API ────────────────────────────────────────────────────

impl Pipeline {
    pub fn load(path: &Path) -> Result<Self, PipelineError> {
        Ok(serde_json::from_str(&std::fs::read_to_string(path)?)?)
    }

    /// Compute each stage's cache key + whether a cached result exists,
    /// without packing anything. Stages are returned in dependency order.
    pub fn plan(
        &self,
        base_dir: &Path,
        cache_dir: &Path,
        proxy_lod: Option<usize>,
    ) -> Result<Vec<StagePlan>, PipelineError> {
        let recipe = Recipe::from_file_with_proxy_lod(base_dir.join(&self.recipe), proxy_lod)?;
        let order = topo_order(&self.stages)?;
        let mut keys: HashMap<&str, String> = HashMap::new();
        let mut plans = Vec::with_capacity(order.len());
        for &i in &order {
            let stage = &self.stages[i];
            let key = self.stage_key(stage, &recipe, &keys);
            let cached = cache_path(cache_dir, &stage.id, &key).exists();
            plans.push(StagePlan {
                id: stage.id.clone(),
                kind: stage.kind.label(),
                cache_key: key.clone(),
                cached,
                depends_on: stage.depends_on.clone(),
            });
            keys.insert(&stage.id, key);
        }
        Ok(plans)
    }

    /// Run the pipeline: pack (or load from cache) each stage in
    /// dependency order, then merge into one snapshot. `force` ignores
    /// the cache and repacks every stage.
    pub fn run(
        &self,
        base_dir: &Path,
        cache_dir: &Path,
        force: bool,
        proxy_lod: Option<usize>,
    ) -> Result<PipelineRun, PipelineError> {
        let recipe = Recipe::from_file_with_proxy_lod(base_dir.join(&self.recipe), proxy_lod)?;
        std::fs::create_dir_all(cache_dir)?;
        let order = topo_order(&self.stages)?;

        let mut keys: HashMap<&str, String> = HashMap::new();
        let mut snaps: HashMap<&str, Snapshot> = HashMap::new();
        let mut reports = Vec::with_capacity(order.len());

        for &i in &order {
            let stage = &self.stages[i];
            let key = self.stage_key(stage, &recipe, &keys);
            let path = cache_path(cache_dir, &stage.id, &key);

            let (snapshot, from_cache) = if !force && path.exists() {
                let snap: Snapshot = serde_json::from_str(&std::fs::read_to_string(&path)?)?;
                (snap, true)
            } else {
                let snapshot = if matches!(stage.kind, StageKind::FiberPack) {
                    run_fiber_pack(stage, &recipe, &snaps, self.seed)
                } else {
                    let dep_snaps: Vec<&Snapshot> =
                        stage.depends_on.iter().map(|d| &snaps[d.as_str()]).collect();
                    let obstacles = obstacles_from(&dep_snaps, &recipe);
                    let sub = sub_recipe(&recipe, &stage.kind);
                    let (densify, cell_size) = match &stage.kind {
                        StageKind::Pack { densify, clearance_cell_size, .. } => {
                            (*densify, *clearance_cell_size)
                        }
                        _ => (false, None),
                    };
                    let cfg = PlacerConfig {
                        strict_bounds: self.strict_bounds,
                        densify,
                        clearance_cell_size: cell_size,
                        backend: self.backend,
                        ..PlacerConfig::default()
                    };
                    GreedyRandomPlacer::new(&sub, cfg)
                        .pack_with_obstacles(stage.effective_seed(self.seed), &obstacles)
                        .snapshot
                };
                std::fs::write(&path, serde_json::to_string(&snapshot)?)?;
                (snapshot, false)
            };

            reports.push(StageReport {
                id: stage.id.clone(),
                kind: stage.kind.label(),
                cache_key: key.clone(),
                from_cache,
                placed: snapshot.placements.len(),
                chromosome: snapshot.chromosome.is_some(),
            });
            keys.insert(&stage.id, key);
            snaps.insert(&stage.id, snapshot);
        }

        // Merge in dependency order: concatenate placements (renumbered to
        // stay globally unique), carry whichever stage made a chromosome, and
        // accumulate nascent-RNA strands (generated by the chromosome stage).
        let mut merged = Snapshot::new(recipe.name.clone(), self.seed);
        let mut uid = 0u64;
        for &i in &order {
            let snap = snaps.remove(self.stages[i].id.as_str()).expect("stage snapshot");
            for mut pl in snap.placements {
                pl.instance_uid = uid;
                uid += 1;
                merged.placements.push(pl);
            }
            if snap.chromosome.is_some() {
                merged.chromosome = snap.chromosome;
            }
            merged.rna_strands.extend(snap.rna_strands);
            merged.peptide_strands.extend(snap.peptide_strands);
        }

        Ok(PipelineRun {
            recipe,
            merged,
            reports,
        })
    }

    /// Cache key for a stage: global recipe identity + stage selection +
    /// effective seed + dependency keys. See the module docs for the
    /// invalidation contract.
    fn stage_key(&self, stage: &Stage, recipe: &Recipe, keys: &HashMap<&str, String>) -> String {
        let mut h = Fnv::new();
        h.string("parsimony.pipeline.v1");

        // Global identity context, hashed for EVERY stage so a cached
        // snapshot's ingredient_ids/compartment_ids can never go stale.
        for (name, ing) in &recipe.ingredients {
            h.string(name);
            hash_ingredient(&mut h, ing);
        }
        for (name, comp) in &recipe.compartments {
            h.string(name);
            hash_compartment(&mut h, comp);
        }
        hash_aabb(&mut h, &recipe.bounding_box);
        h.u8(self.strict_bounds as u8);
        h.u8(self.backend as u8);

        // Stage identity + selection.
        h.string(&stage.id);
        h.u64(stage.effective_seed(self.seed));
        match &stage.kind {
            StageKind::Chromosome => {
                h.string("chromosome");
                match &recipe.chromosome {
                    Some(c) => {
                        h.u64(c.beads as u64);
                        h.f32(c.spacing);
                        h.f32(c.bead_radius);
                        h.string(c.compartment.as_deref().unwrap_or(""));
                        match &c.supercoil {
                            Some(s) => {
                                h.u8(1);
                                h.f32(s.radius);
                                h.f32(s.pitch);
                            }
                            None => h.u8(0),
                        }
                    }
                    None => h.string("<none>"),
                }
            }
            StageKind::Pack { include, exclude, densify, clearance_cell_size } => {
                h.string("pack");
                h.u8(*densify as u8);
                h.f32(clearance_cell_size.unwrap_or(0.0));
                for d in &recipe.directives {
                    if !pack_selects(include, exclude, &d.ingredient) {
                        continue;
                    }
                    h.string(&d.ingredient);
                    h.string(&d.compartment);
                    h.u8(region_tag(d.region));
                    h.u32(d.count);
                    h.u8(matches!(d.packing_mode, PackingMode::Tiled) as u8);
                }
            }
            StageKind::FiberPack => {
                h.string("fiber_pack");
                match &recipe.chromosome {
                    Some(c) => {
                        h.f32(c.bead_radius);
                        for (name, count) in &c.proteins {
                            h.string(name);
                            h.u32(*count);
                        }
                    }
                    None => h.string("<none>"),
                }
            }
        }

        // Dependency keys (sorted for order-independence).
        let mut dep_keys: Vec<&str> = stage.depends_on.iter().map(|d| keys[d.as_str()].as_str()).collect();
        dep_keys.sort_unstable();
        for k in dep_keys {
            h.string(k);
        }

        format!("{:016x}", h.finish())
    }
}

// ───── stage execution helpers ───────────────────────────────────────

/// Build the sub-recipe a stage actually packs: the full ingredient and
/// compartment tables (so ids stay consistent across stages), with the
/// directive list / chromosome narrowed to this stage's job.
fn sub_recipe(recipe: &Recipe, kind: &StageKind) -> Recipe {
    let mut r = recipe.clone();
    match kind {
        StageKind::Chromosome => {
            r.directives.clear();
            // Bound proteins are a separate FiberPack stage; the placer must
            // not also bind them while generating the fiber here.
            if let Some(c) = r.chromosome.as_mut() {
                c.proteins.clear();
            }
        }
        StageKind::Pack { include, exclude, .. } => {
            r.chromosome = None;
            r.directives
                .retain(|d| pack_selects(include, exclude, &d.ingredient));
        }
        // FiberPack is run by `run_fiber_pack`, not the volume placer, so it
        // never reaches here — but the match must stay exhaustive.
        StageKind::FiberPack => {
            r.directives.clear();
            r.chromosome = None;
        }
    }
    r
}

fn pack_selects(include: &[String], exclude: &[String], name: &str) -> bool {
    let included = include.is_empty() || include.iter().any(|s| s == name);
    let excluded = exclude.iter().any(|s| s == name);
    included && !excluded
}

/// World-space obstacle spheres contributed by a stage's dependencies:
/// the chromosome's beads and every placed ingredient's proxy spheres.
fn obstacles_from(deps: &[&Snapshot], recipe: &Recipe) -> Vec<(Point3<f32>, f32)> {
    let mut obs = Vec::new();
    for snap in deps {
        if let Some(chr) = &snap.chromosome {
            for p in &chr.points {
                obs.push((
                    Point3::new(
                        chr.center.x + p.x,
                        chr.center.y + p.y,
                        chr.center.z + p.z,
                    ),
                    chr.radius,
                ));
            }
        }
        for pl in &snap.placements {
            if let Some((_, ing)) = recipe.ingredients.get_index(pl.ingredient_id as usize) {
                obs.extend(ing.shape.world_spheres(pl.position, pl.rotation));
            }
        }
    }
    obs
}

/// Derive the `CellShape` for a chromosome compartment from a recipe.
/// Mirrors the logic in `placer::chromosome_cell`. Returns an effectively
/// unbounded sphere as a fallback so confinement never rejects anything when
/// the compartment cannot be resolved.
///
// TODO: the returned `CellShape` is origin-relative, but the fiber passed to
// `pack_on_fiber*` is world-space (offset by the compartment centre). This is
// exact for the production compartment (centred at the world origin) and only
// approximate for an off-centre compartment. Carry the compartment centre into
// `CellShape` (or translate the fiber to origin-relative) to make it exact.
fn chrom_cell_shape(recipe: &Recipe, compartment: Option<&str>) -> crate::fiber::CellShape {
    use crate::fiber::CellShape;
    for (name, comp) in &recipe.compartments {
        if let Some(want) = compartment {
            if name.as_str() != want {
                continue;
            }
        }
        return match &comp.kind {
            CompartmentKind::Sphere { radius, .. } => CellShape::Sphere { radius: *radius },
            CompartmentKind::Capsule { a, b, radius } => {
                let axis_v = b - a;
                let half_len = axis_v.norm() * 0.5;
                let axis = axis_v
                    .try_normalize(1e-6)
                    .unwrap_or_else(nalgebra::Vector3::x);
                CellShape::Capsule { half_len, radius: *radius, axis }
            }
            _ => CellShape::Sphere { radius: 1e9 },
        };
    }
    CellShape::Sphere { radius: 1e9 }
}

/// Execute a [`StageKind::FiberPack`] stage: bind the chromosome's proteins
/// along the fiber from a `chromosome` dependency, avoiding the
/// dependencies' placed geometry (e.g. the interior). The DNA beads
/// themselves aren't obstacles — bindings are meant to sit on them.
fn run_fiber_pack(
    stage: &Stage,
    recipe: &Recipe,
    snaps: &HashMap<&str, Snapshot>,
    pipeline_seed: u64,
) -> Snapshot {
    let mut snap = Snapshot::new(recipe.name.clone(), stage.effective_seed(pipeline_seed));
    let Some(chr_spec) = &recipe.chromosome else {
        return snap;
    };
    let Some(chrom) = stage
        .depends_on
        .iter()
        .filter_map(|d| snaps.get(d.as_str()))
        .find_map(|s| s.chromosome.as_ref())
    else {
        return snap;
    };

    // Cell shape used for confining fiber-bound proteins inside the envelope.
    let chrom_shape = chrom_cell_shape(recipe, chr_spec.compartment.as_deref());

    // Pack proteins across ALL strands (every chromosome + its sister bubble),
    // not just the first — otherwise all RNAP piles onto one chromosome. Each
    // strand gets a share of every protein proportional to its contour length.
    let strands: Vec<Vec<nalgebra::Point3<f32>>> = if chrom.strands.is_empty() {
        vec![chrom.points.clone()]
    } else {
        chrom.strands.clone()
    };
    let contour = |s: &[nalgebra::Point3<f32>]| -> f32 {
        s.windows(2).map(|w| (w[1] - w[0]).norm()).sum()
    };
    let lengths: Vec<f32> = strands.iter().map(|s| contour(s)).collect();
    let total_len: f32 = lengths.iter().sum::<f32>().max(1e-6);

    let obstacles: Vec<(nalgebra::Point3<f32>, f32)> = stage
        .depends_on
        .iter()
        .filter_map(|d| snaps.get(d.as_str()))
        .flat_map(|s| {
            s.placements.iter().flat_map(|pl| {
                recipe
                    .ingredients
                    .get_index(pl.ingredient_id as usize)
                    .map(|(_, ing)| ing.shape.world_spheres(pl.position, pl.rotation))
                    .into_iter()
                    .flatten()
            })
        })
        .collect();

    let mut rng = Xoshiro256PlusPlus::seed_from_u64(stage.effective_seed(pipeline_seed));
    let genome = chr_spec
        .genome
        .as_ref()
        .and_then(|p| crate::genome::Genome::from_csv(p).ok());

    for (si, strand) in strands.iter().enumerate() {
        if strand.len() < 2 {
            continue;
        }
        let share = lengths[si] / total_len;
        // This strand's share of each protein's total count.
        let strand_proteins: Vec<(String, u32)> = chr_spec
            .proteins
            .iter()
            .map(|(name, c)| (name.clone(), (*c as f32 * share).round() as u32))
            .collect();
        let fiber_world: Vec<nalgebra::Point3<f32>> =
            strand.iter().map(|p| chrom.center + p.coords).collect();
        // With a genome annotation, seat proteins at real transcription /
        // replication sites; otherwise spread them randomly along the strand.
        let binds = match &genome {
            Some(genome) => {
                let abundances: Vec<(String, u32)> = recipe
                    .directives
                    .iter()
                    .map(|d| (d.ingredient.clone(), d.count))
                    .collect();
                let sites = genome.binding_sites(&strand_proteins, &abundances, &mut rng);
                let mut at: Vec<(u32, &Ingredient, f32)> = Vec::new();
                for ((name, _), fracs) in strand_proteins.iter().zip(&sites) {
                    if let Some((idx, _, ing)) = recipe.ingredients.get_full(name) {
                        for &f in fracs {
                            at.push((idx as u32, ing, f));
                        }
                    }
                }
                crate::fiber_pack::pack_on_fiber_at(&fiber_world, &at, &obstacles, chrom.radius, chrom_shape, &mut rng)
            }
            None => {
                let proteins: Vec<(u32, &Ingredient, u32)> = strand_proteins
                    .iter()
                    .filter_map(|(name, c)| {
                        recipe
                            .ingredients
                            .get_full(name)
                            .map(|(idx, _, ing)| (idx as u32, ing, *c))
                    })
                    .collect();
                crate::fiber_pack::pack_on_fiber(&fiber_world, &proteins, &obstacles, chrom.radius, chrom_shape, &mut rng)
            }
        };
        for b in binds {
            snap.placements.push(Placement {
                instance_uid: snap.placements.len() as u64,
                ingredient_id: b.ingredient_id,
                variant_id: 0,
                compartment_id: 0,
                position: b.position,
                rotation: b.rotation,
            });
        }
    }
    snap
}

// ───── graph ─────────────────────────────────────────────────────────

/// Kahn topological sort over `depends_on`. Errors on duplicate ids,
/// unknown dependencies, or a cycle. Deterministic for a given graph.
fn topo_order(stages: &[Stage]) -> Result<Vec<usize>, PipelineError> {
    let mut id_to_idx: HashMap<&str, usize> = HashMap::new();
    for (i, s) in stages.iter().enumerate() {
        if id_to_idx.insert(&s.id, i).is_some() {
            return Err(PipelineError::DuplicateStage(s.id.clone()));
        }
    }
    let mut indeg = vec![0usize; stages.len()];
    let mut dependents: Vec<Vec<usize>> = vec![Vec::new(); stages.len()];
    for (i, s) in stages.iter().enumerate() {
        for d in &s.depends_on {
            let &j = id_to_idx.get(d.as_str()).ok_or_else(|| PipelineError::UnknownDep {
                stage: s.id.clone(),
                dep: d.clone(),
            })?;
            indeg[i] += 1;
            dependents[j].push(i);
        }
    }
    let mut queue: Vec<usize> = (0..stages.len()).filter(|&i| indeg[i] == 0).collect();
    queue.sort_unstable();
    let mut order = Vec::with_capacity(stages.len());
    let mut qi = 0;
    while qi < queue.len() {
        let n = queue[qi];
        qi += 1;
        order.push(n);
        for &m in &dependents[n] {
            indeg[m] -= 1;
            if indeg[m] == 0 {
                queue.push(m);
            }
        }
    }
    if order.len() != stages.len() {
        let unresolved: Vec<&str> = (0..stages.len())
            .filter(|i| !order.contains(i))
            .map(|i| stages[i].id.as_str())
            .collect();
        return Err(PipelineError::Cycle(unresolved.join(", ")));
    }
    Ok(order)
}

// ───── cache addressing ──────────────────────────────────────────────

fn cache_path(cache_dir: &Path, stage_id: &str, key: &str) -> PathBuf {
    cache_dir.join(format!("{}-{}.snapshot.json", sanitize(stage_id), key))
}

fn sanitize(s: &str) -> String {
    s.chars()
        .map(|c| if c.is_ascii_alphanumeric() || c == '-' || c == '_' { c } else { '_' })
        .collect()
}

// ───── content hashing (FNV-1a, 64-bit) ──────────────────────────────
// A small, stable, dependency-free hash. Floats are hashed by their
// exact bit pattern (`to_bits`) so the digest is fully deterministic;
// strings are length-prefixed so concatenations can't collide.

struct Fnv(u64);

impl Fnv {
    fn new() -> Self {
        Fnv(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &byte in b {
            self.0 ^= byte as u64;
            self.0 = self.0.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    fn u8(&mut self, v: u8) {
        self.bytes(&[v]);
    }
    fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn f32(&mut self, v: f32) {
        self.u32(v.to_bits());
    }
    fn string(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.bytes(s.as_bytes());
    }
    fn finish(&self) -> u64 {
        self.0
    }
}

fn region_tag(r: RegionKind) -> u8 {
    match r {
        RegionKind::Interior => 0,
        RegionKind::Surface => 1,
    }
}

fn hash_aabb(h: &mut Fnv, a: &parsimony_spatial::Aabb) {
    for v in [a.min.x, a.min.y, a.min.z, a.max.x, a.max.y, a.max.z] {
        h.f32(v);
    }
}

fn hash_compartment(h: &mut Fnv, c: &Compartment) {
    let tag = match &c.kind {
        CompartmentKind::Box(_) => 0u8,
        CompartmentKind::Sphere { .. } => 1,
        CompartmentKind::Capsule { .. } => 2,
        CompartmentKind::Mesh(_) => 3,
    };
    h.u8(tag);
    hash_aabb(h, &c.kind.aabb());
    match &c.kind {
        CompartmentKind::Sphere { center, radius } => {
            h.f32(center.x);
            h.f32(center.y);
            h.f32(center.z);
            h.f32(*radius);
        }
        CompartmentKind::Capsule { a, b, radius } => {
            for v in [a.x, a.y, a.z, b.x, b.y, b.z, *radius] {
                h.f32(v);
            }
        }
        _ => {}
    }
    h.u64(c.children.len() as u64);
}

fn hash_ingredient(h: &mut Fnv, ing: &Ingredient) {
    match &ing.shape {
        IngredientShape::SingleSphere { radius } => {
            h.u8(0);
            h.f32(*radius);
        }
        IngredientShape::MultiSphere { spheres } => {
            h.u8(1);
            h.u64(spheres.len() as u64);
            for s in spheres {
                h.f32(s.offset.x);
                h.f32(s.offset.y);
                h.f32(s.offset.z);
                h.f32(s.radius);
            }
        }
        IngredientShape::Mesh { trimesh, proxies } => {
            // Triangle/vertex/proxy counts + the LOD URLs below identify
            // the mesh cheaply (no per-vertex hashing).
            h.u8(2);
            h.u64(trimesh.vertices().len() as u64);
            h.u64(trimesh.indices().len() as u64);
            h.u64(proxies.len() as u64);
        }
        IngredientShape::Fiber { points, radius } => {
            h.u8(3);
            h.u64(points.len() as u64);
            h.f32(*radius);
        }
    }
    for lod in &ing.mesh_lods {
        h.string(&lod.url);
        h.f32(lod.voxel_size);
    }
    h.u8(matches!(ing.packing_mode, PackingMode::Tiled) as u8);
    h.f32(ing.principal_vector.x);
    h.f32(ing.principal_vector.y);
    h.f32(ing.principal_vector.z);
    h.u32(ing.jitter_attempts);
}

// ───── tests ─────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicU64, Ordering};
    use std::time::{SystemTime, UNIX_EPOCH};

    // A sphere cell with 10 proteins + a 100-bead chromosome with 2 nascent-RNA
    // strands.  Used by `pipeline_merge_preserves_rna_strands` to exercise the
    // merge-loop fix from commit 3aa6e0b: rna_strands were dropped during the
    // multi-stage merge.
    const RECIPE_WITH_RNAS: &str = r#"{
        "bounding_box": [[-100,-100,-100],[100,100,100]],
        "objects": { "prot": { "type": "single_sphere", "radius": 5 } },
        "composition": {
            "space": { "regions": { "interior": ["cell"] } },
            "cell": {
                "compartment": { "kind": "sphere", "center": [0,0,0], "radius": 80 },
                "regions": { "interior": [ { "object": "prot", "count": 10 } ] }
            }
        },
        "chromosome": {
            "beads": 100, "spacing": 8, "bead_radius": 4,
            "rnas": [
                {"root_coordinate": 10000, "root_domain": 0, "length_nt": 200, "is_mRNA": true},
                {"root_coordinate": -5000, "root_domain": 0, "length_nt": 100, "is_mRNA": false}
            ]
        }
    }"#;

    // A sphere cell with 50 proteins + a 100-bead chromosome.
    const RECIPE: &str = r#"{
        "bounding_box": [[-100,-100,-100],[100,100,100]],
        "objects": { "prot": { "type": "single_sphere", "radius": 5 } },
        "composition": {
            "space": { "regions": { "interior": ["cell"] } },
            "cell": {
                "compartment": { "kind": "sphere", "center": [0,0,0], "radius": 80 },
                "regions": { "interior": [ { "object": "prot", "count": 50 } ] }
            }
        },
        "chromosome": { "beads": 100, "spacing": 8, "bead_radius": 4 }
    }"#;

    fn scratch_dir() -> PathBuf {
        static N: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos();
        let dir = std::env::temp_dir().join(format!(
            "parsimony_pipeline_{}_{}",
            nanos,
            N.fetch_add(1, Ordering::Relaxed)
        ));
        std::fs::create_dir_all(&dir).unwrap();
        dir
    }

    fn two_stage_pipeline() -> Pipeline {
        Pipeline {
            name: "test".into(),
            recipe: "recipe.json".into(),
            seed: 0,
            strict_bounds: true,
            backend: PlacementBackend::Legacy,
            stages: vec![
                Stage {
                    id: "chromosome".into(),
                    kind: StageKind::Chromosome,
                    depends_on: vec![],
                    seed: None,
                },
                Stage {
                    id: "interior".into(),
                    kind: StageKind::Pack {
                        include: vec![],
                        exclude: vec![],
                        densify: false,
                        clearance_cell_size: None,
                    },
                    depends_on: vec!["chromosome".into()],
                    seed: None,
                },
            ],
        }
    }

    #[test]
    fn runs_caches_and_selectively_regenerates() {
        let base = scratch_dir();
        let cache = base.join("cache");
        std::fs::write(base.join("recipe.json"), RECIPE).unwrap();
        let mut pipeline = two_stage_pipeline();

        // First run: nothing cached, both stages execute.
        let run1 = pipeline.run(&base, &cache, false, None).unwrap();
        assert_eq!(run1.reports.len(), 2);
        assert!(run1.reports.iter().all(|r| !r.from_cache));
        let interior = run1.reports.iter().find(|r| r.id == "interior").unwrap();
        assert!(interior.placed > 0, "interior should place some proteins");
        assert!(run1.merged.chromosome.is_some(), "merged carries the chromosome");
        // The merged pack = interior placements + the chromosome stage's
        // (zero) placements; chromosome itself rides on `.chromosome`.
        assert_eq!(run1.merged.placements.len(), interior.placed);

        // Second run, unchanged: both stages come from cache.
        let run2 = pipeline.run(&base, &cache, false, None).unwrap();
        assert!(run2.reports.iter().all(|r| r.from_cache), "all cached on rerun");

        // Re-roll only the interior: chromosome stays cached, interior repacks.
        pipeline.stages[1].seed = Some(0xBEEF);
        let run3 = pipeline.run(&base, &cache, false, None).unwrap();
        let chrom = run3.reports.iter().find(|r| r.id == "chromosome").unwrap();
        let inter = run3.reports.iter().find(|r| r.id == "interior").unwrap();
        assert!(chrom.from_cache, "chromosome must be reused");
        assert!(!inter.from_cache, "interior must regenerate after a seed change");

        // The interior packs around the chromosome: no protein centre lands
        // inside a chromosome bead (radii 5 + 4 = 9 minimum separation).
        let chr = run3.merged.chromosome.as_ref().unwrap();
        for pl in &run3.merged.placements {
            let nearest = chr
                .points
                .iter()
                .map(|p| {
                    let d = pl.position - (chr.center + p.coords);
                    d.norm()
                })
                .fold(f32::INFINITY, f32::min);
            assert!(
                nearest >= 5.0 + chr.radius - 1e-2,
                "protein at {:?} overlaps the chromosome (nearest bead {nearest:.2})",
                pl.position
            );
        }

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn force_repacks_everything() {
        let base = scratch_dir();
        let cache = base.join("cache");
        std::fs::write(base.join("recipe.json"), RECIPE).unwrap();
        let pipeline = two_stage_pipeline();

        pipeline.run(&base, &cache, false, None).unwrap();
        let forced = pipeline.run(&base, &cache, true, None).unwrap();
        assert!(forced.reports.iter().all(|r| !r.from_cache), "--force ignores cache");

        let _ = std::fs::remove_dir_all(&base);
    }

    #[test]
    fn detects_cycles_and_unknown_deps() {
        let mut p = two_stage_pipeline();
        p.stages[0].depends_on = vec!["interior".into()]; // chromosome <-> interior
        assert!(matches!(topo_order(&p.stages), Err(PipelineError::Cycle(_))));

        let mut q = two_stage_pipeline();
        q.stages[1].depends_on = vec!["ghost".into()];
        assert!(matches!(
            topo_order(&q.stages),
            Err(PipelineError::UnknownDep { .. })
        ));
    }

    #[test]
    fn plan_matches_run_keys_and_reports_cached() {
        let base = scratch_dir();
        let cache = base.join("cache");
        std::fs::write(base.join("recipe.json"), RECIPE).unwrap();
        let pipeline = two_stage_pipeline();

        let plan_before = pipeline.plan(&base, &cache, None).unwrap();
        assert!(plan_before.iter().all(|s| !s.cached), "nothing cached yet");

        let run = pipeline.run(&base, &cache, false, None).unwrap();
        let plan_after = pipeline.plan(&base, &cache, None).unwrap();
        assert!(plan_after.iter().all(|s| s.cached), "all cached after a run");
        // Keys are stable between plan and run.
        for (p, r) in plan_after.iter().zip(run.reports.iter()) {
            assert_eq!(p.cache_key, r.cache_key);
        }

        let _ = std::fs::remove_dir_all(&base);
    }
    /// Regression test for commit 3aa6e0b: the multi-stage Pipeline merge loop
    /// was dropping `snapshot.rna_strands` from each stage's output, so a full
    /// chromosome+pack build rendered no RNA strands in the merged result.
    /// The fix added `merged.rna_strands.extend(snap.rna_strands)` to the loop.
    ///
    /// This test runs a two-stage pipeline (Chromosome + Pack) against a recipe
    /// that declares 2 nascent RNA strands and asserts that exactly 2 survive
    /// into `run.merged.rna_strands`.
    #[test]
    fn pipeline_merge_preserves_rna_strands() {
        let base = scratch_dir();
        let cache = base.join("cache");
        std::fs::write(base.join("recipe.json"), RECIPE_WITH_RNAS).unwrap();
        let pipeline = two_stage_pipeline();

        let run = pipeline.run(&base, &cache, true, None).unwrap();

        // The chromosome stage populates rna_strands (one per RnaSpec).
        // The merge loop must carry them through — the pre-3aa6e0b bug
        // silently dropped them, leaving merged.rna_strands empty.
        const N_RNAS: usize = 2;
        assert_eq!(
            run.merged.rna_strands.len(),
            N_RNAS,
            "pipeline merge must preserve rna_strands: expected {N_RNAS}, got {}",
            run.merged.rna_strands.len()
        );
        for (i, strand) in run.merged.rna_strands.iter().enumerate() {
            assert!(
                !strand.points.is_empty(),
                "rna_strand {i} has no beads after pipeline merge"
            );
        }

        let _ = std::fs::remove_dir_all(&base);
    }
}

