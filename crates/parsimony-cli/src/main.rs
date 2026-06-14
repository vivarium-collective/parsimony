//! `parsimony` CLI. Phase 2 surface:
//!
//! ```text
//! parsimony pack <recipe.json> -o <out.{simularium,json}> [--seed N] [--format ...]
//! ```

use std::fs;
use std::path::{Path, PathBuf};
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

mod mesh_gen;

use parsimony_core::{
    metrics, write_pack_json, write_simularium_json, write_transforms_json, GreedyRandomPlacer,
    MetricsConfig, PlacementBackend, Pipeline, PlacerConfig, PlacerOutcome, Recipe,
};

#[derive(Debug, Parser)]
#[command(
    name = "parsimony",
    version,
    about = "Pack molecular contents into cellular volumes.",
    long_about = None
)]
struct Cli {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Pack a recipe and write the result.
    Pack(PackArgs),
    /// Regenerate (or list) the viewer demo packs in viewer/data/.
    Demos(DemosArgs),
    /// Pack-and-serve the local three.js viewer in the browser.
    Viewer(ViewerArgs),
    /// Pack a recipe and compare it against cellPACK (validation).
    Compare(CompareArgs),
    /// Generate mesh LODs from PDB/CIF structures (PDB → SDF → OBJ LODs).
    Mesh(MeshArgs),
    /// Translate the whole-cell Mycoplasma recipe + meshes from the cellPACK
    /// data repo (clones it on first run) into a parsimony recipe.
    TranslateMycoplasma(TranslateMycoplasmaArgs),
    /// Render a Simularium file to a static PNG (the report images).
    Render(RenderArgs),
    /// Open this repo's REPORT.md as a live preview, or export it to HTML.
    Report(ReportArgs),
    /// Run a staged packing pipeline (DAG + content-addressed cache).
    Pipeline(PipelineArgs),
    /// Pack a recipe and report quantitative metrics — overlaps, fill,
    /// pair-correlation g(r), nearest-neighbour distances, free space.
    Metrics(MetricsArgs),
}

#[derive(Debug, Parser)]
struct PackArgs {
    /// Recipe JSON path.
    recipe: PathBuf,

    /// Output path. Extension is inferred by `--format` if given,
    /// otherwise by the extension of this path (`.simularium` →
    /// Simularium, anything else → transform-list JSON).
    #[arg(short, long)]
    out: PathBuf,

    /// RNG seed. Same seed + same recipe = bit-for-bit same output.
    #[arg(short, long, default_value_t = 0)]
    seed: u64,

    /// Output format. Overrides path extension when set.
    #[arg(long, value_enum)]
    format: Option<Format>,

    /// Quiet — suppress per-ingredient stats.
    #[arg(short, long)]
    quiet: bool,

    /// Use cellPACK-style loose root containment — centre inside the
    /// bounding box, sphere may protrude at the edge. Off by default
    /// (parsimony requires the whole sphere inside the box). Named
    /// compartments are always strict regardless of this flag.
    #[arg(long)]
    loose_bounds: bool,

    /// Interior-placement engine. `legacy` (default) is cellPACK-style:
    /// a dense clearance grid + a valid-cell list rebuilt per directive.
    /// `octree` shares one content-scaled sparse occupancy tree across all
    /// directives — ~5× faster on whole-cell recipes, and the engine that
    /// scales to far larger / sparser domains.
    #[arg(long, value_enum, default_value_t = Backend::Legacy)]
    backend: Backend,

    /// Override the legacy clearance-grid cell size, in Å (default:
    /// largest-ingredient-radius / 8). Smaller = finer grid (better
    /// resolution for small ingredients, more cells/memory). Used to
    /// explore adaptive cell sizing vs the octree. No effect on `octree`.
    #[arg(long)]
    cell_size: Option<f32>,

    /// Override the recipe's chromosome bead count (genome resolution).
    /// More beads = more DNA contour/volume + finer genome, at a heavier
    /// obstacle set for the interior pack. No effect without a chromosome.
    #[arg(long)]
    chromosome_beads: Option<usize>,

    /// Build mesh collision proxies from this LOD index (0 = coarsest;
    /// clamped per ingredient). Default = finest. Proxies are voxelised at
    /// `proxy_voxel_size` regardless, so a coarse LOD (near that resolution)
    /// packs far faster/lighter at whole-cell scale with ~the same result.
    #[arg(long)]
    proxy_lod: Option<usize>,
}

/// CLI mirror of [`PlacementBackend`] (keeps `clap` out of parsimony-core).
#[derive(Debug, Clone, Copy, Default, ValueEnum)]
enum Backend {
    #[default]
    Legacy,
    Octree,
}

impl From<Backend> for PlacementBackend {
    fn from(b: Backend) -> Self {
        match b {
            Backend::Legacy => PlacementBackend::Legacy,
            Backend::Octree => PlacementBackend::Octree,
        }
    }
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum Format {
    /// parsimony's native pack format (parsimony.pack.v1). This is
    /// what the local three.js viewer consumes; new rendering
    /// features land here first. Default for .pack / .json output.
    Pack,
    /// Optional Simularium export for the cellpack.allencell.org
    /// viewer. Lossy by construction (sphere-only).
    Simularium,
    /// Legacy flat transform list. Stable, debug-friendly.
    Transforms,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_target(false)
        .init();

    let cli = Cli::parse();
    match cli.command {
        Command::Pack(args) => run_pack(args),
        Command::Demos(args) => run_demos(args),
        Command::Viewer(args) => run_viewer(args),
        Command::Compare(args) => run_compare(args),
        Command::Mesh(args) => run_mesh(args),
        Command::TranslateMycoplasma(args) => run_translate_mycoplasma(args),
        Command::Render(args) => run_render(args),
        Command::Report(args) => run_report(args),
        Command::Pipeline(args) => run_pipeline(args),
        Command::Metrics(args) => run_metrics(args),
    }
}

/// Error unless `uv` is on PATH (drives the bundled scientific-Python
/// helpers — PDB meshing, the Mycoplasma translator, report rendering).
fn ensure_uv() -> Result<()> {
    let ok = std::process::Command::new("uv")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if ok {
        Ok(())
    } else {
        anyhow::bail!(
            "`uv` not on PATH — needed for the bundled Python helpers \
             (https://docs.astral.sh/uv/)"
        )
    }
}

/// Error unless `git` is on PATH (used to clone upstream data repos).
fn ensure_git() -> Result<()> {
    let ok = std::process::Command::new("git")
        .arg("--version")
        .output()
        .map(|o| o.status.success())
        .unwrap_or(false);
    if ok {
        Ok(())
    } else {
        anyhow::bail!("`git` not on PATH — needed to clone the cellPACK data repo")
    }
}

fn run_pack(args: PackArgs) -> Result<()> {
    let recipe = Recipe::from_file_with_proxy_lod(&args.recipe, args.proxy_lod)
        .with_context(|| format!("loading recipe `{}`", args.recipe.display()))?;
    let total_requested: u32 = recipe.directives.iter().map(|d| d.count).sum();
    if !args.quiet {
        eprintln!("recipe: {} ({} ingredient types, {} directives, {} instances requested)",
            recipe.name,
            recipe.ingredients.len(),
            recipe.directives.len(),
            total_requested,
        );
    }

    let placer_config = PlacerConfig {
        strict_bounds: !args.loose_bounds,
        backend: args.backend.into(),
        clearance_cell_size: args.cell_size,
        chromosome_beads: args.chromosome_beads,
        ..PlacerConfig::default()
    };
    let t = Instant::now();
    let placer = GreedyRandomPlacer::new(&recipe, placer_config);
    let out = placer.pack(args.seed);
    let elapsed = t.elapsed();

    if !args.quiet {
        eprintln!(
            "backend: {}",
            match args.backend {
                Backend::Legacy => "legacy (clearance grid + valid-cells)",
                Backend::Octree => "octree (sparse occupancy, content-scaled)",
            }
        );
        eprintln!(
            "packed: {}/{} instances in {:.2?}  ({:.0}%)",
            out.stats.placed,
            out.stats.requested,
            elapsed,
            100.0 * (out.stats.placed as f32) / (out.stats.requested.max(1) as f32),
        );
        eprintln!(
            "  total attempts: {}, success rate: {:.1}%",
            out.stats.total_attempts,
            100.0 * (out.stats.successful_attempts as f32)
                / (out.stats.total_attempts.max(1) as f32),
        );
        for (name, placed, requested, attempts) in &out.stats.per_ingredient {
            eprintln!(
                "  {:<24} {:>5}/{:<5} ({:>5} attempts)",
                name, placed, requested, attempts
            );
        }
    }

    let format = resolve_format(&args);
    let json = match format {
        Format::Pack => write_pack_json(&out.snapshot, &recipe),
        Format::Simularium => write_simularium_json(&out.snapshot, &recipe),
        Format::Transforms => write_transforms_json(&out.snapshot, &recipe),
    };
    let pretty = serde_json::to_string_pretty(&json)?;
    std::fs::write(&args.out, &pretty)
        .with_context(|| format!("writing `{}`", args.out.display()))?;

    if !args.quiet {
        eprintln!(
            "wrote {} ({} bytes, {})",
            args.out.display(),
            pretty.len(),
            match format {
                Format::Pack => "parsimony.pack.v1",
                Format::Simularium => "Simularium",
                Format::Transforms => "transform-list",
            }
        );
    }
    Ok(())
}

// ───── metrics ──────────────────────────────────────────────────────

#[derive(Debug, Parser)]
struct MetricsArgs {
    /// Recipe JSON path.
    recipe: PathBuf,

    /// RNG seed — drives both the pack and the Monte-Carlo free-space
    /// sampling, so the whole report is reproducible.
    #[arg(short, long, default_value_t = 0)]
    seed: u64,

    /// Interior-placement engine (see `pack --backend`).
    #[arg(long, value_enum, default_value_t = Backend::Legacy)]
    backend: Backend,

    /// cellPACK-style loose root containment (centre-in-box).
    #[arg(long)]
    loose_bounds: bool,

    /// Build mesh collision proxies from this LOD index (0 = coarsest).
    #[arg(long)]
    proxy_lod: Option<usize>,

    /// Override the legacy clearance-grid cell size, in Å.
    #[arg(long)]
    cell_size: Option<f32>,

    /// Override the recipe's chromosome bead count.
    #[arg(long)]
    chromosome_beads: Option<usize>,

    /// Emit the full metrics as JSON on stdout instead of a summary.
    #[arg(long)]
    json: bool,

    /// Number of pair-correlation g(r) bins (0 disables the RDF).
    #[arg(long, default_value_t = 64)]
    rdf_bins: usize,

    /// Monte-Carlo samples for the free-space estimate (0 disables it).
    #[arg(long, default_value_t = 8192)]
    samples: usize,
}

fn run_metrics(args: MetricsArgs) -> Result<()> {
    let recipe = Recipe::from_file_with_proxy_lod(&args.recipe, args.proxy_lod)
        .with_context(|| format!("loading recipe `{}`", args.recipe.display()))?;

    let placer_config = PlacerConfig {
        strict_bounds: !args.loose_bounds,
        backend: args.backend.into(),
        clearance_cell_size: args.cell_size,
        chromosome_beads: args.chromosome_beads,
        ..PlacerConfig::default()
    };
    let t = Instant::now();
    let placer = GreedyRandomPlacer::new(&recipe, placer_config);
    let out = placer.pack(args.seed);
    let pack_elapsed = t.elapsed();

    let cfg = MetricsConfig {
        rdf_bins: args.rdf_bins,
        free_space_samples: args.samples,
        seed: args.seed,
        ..MetricsConfig::default()
    };
    let t2 = Instant::now();
    let m = metrics::compute(&out.snapshot, &recipe, &cfg);
    let metrics_elapsed = t2.elapsed();

    if args.json {
        println!("{}", serde_json::to_string_pretty(&m)?);
        return Ok(());
    }

    let backend = match args.backend {
        Backend::Legacy => "legacy",
        Backend::Octree => "octree",
    };
    println!("recipe: {}  (seed {}, backend {})", m.recipe_name, m.seed, backend);
    println!(
        "packed {} / {} requested ({:.0}%) in {:.2?}; metrics in {:.2?}",
        m.n_placed,
        m.n_requested,
        100.0 * m.fraction_placed,
        pack_elapsed,
        metrics_elapsed,
    );
    println!("domain volume: {:.3e} Å³", m.domain_volume);

    let o = &m.geometry.overlaps;
    if o.pair_count == 0 {
        println!("overlaps: none");
    } else {
        println!(
            "overlaps: {} pairs across {} instances (max depth {:.3} Å, mean {:.3} Å)",
            o.pair_count, o.instance_count, o.max_depth, o.mean_depth,
        );
    }

    let nn = &m.geometry.nearest_neighbor;
    if nn.center.n > 0 {
        println!(
            "nearest neighbour (centre):  min {:.2}  median {:.2}  max {:.2}  (mean {:.2} ± {:.2}) Å",
            nn.center.min, nn.center.median, nn.center.max, nn.center.mean, nn.center.stddev,
        );
        println!(
            "nearest neighbour (surface): min {:.2}  median {:.2} Å  (negative ⇒ contact/overlap)",
            nn.surface_gap.min, nn.surface_gap.median,
        );
    }

    if let Some(rdf) = &m.geometry.rdf {
        let (pk_i, pk_g) = rdf
            .g
            .iter()
            .enumerate()
            .fold((0usize, 0.0f32), |(bi, bg), (i, &g)| if g > bg { (i, g) } else { (bi, bg) });
        println!(
            "g(r): peak g={:.2} at r={:.1} Å  (r_max {:.1} Å, {} bins, ρ={:.3e}/Å³)",
            pk_g,
            rdf.r.get(pk_i).copied().unwrap_or(0.0),
            rdf.r_max,
            rdf.g.len(),
            rdf.number_density,
        );
    }

    if let Some(fs) = &m.geometry.free_space {
        println!(
            "free space: {:.1}% occupied; void radius median {:.2}  max {:.2} Å  ({} samples)",
            100.0 * fs.occupied_fraction,
            fs.clearance.median,
            fs.clearance.max,
            fs.samples,
        );
    }

    println!("per-ingredient (placed / requested):");
    for f in &m.per_ingredient {
        println!("  {:<28} {:>6} / {:<6}", f.name, f.placed, f.requested);
    }
    Ok(())
}

fn resolve_format(args: &PackArgs) -> Format {
    if let Some(f) = args.format {
        return f;
    }
    match args.out.extension().and_then(|s| s.to_str()) {
        Some("simularium") => Format::Simularium,
        // Everything else (incl. .pack, .json, no extension) →
        // parsimony's native format. The viewer expects this.
        _ => Format::Pack,
    }
}

// ───── demos ────────────────────────────────────────────────────────
// The viewer dropdown is driven by viewer/data/index.json plus the
// viewer/data/*.pack.json files. Rather than packing each by hand,
// `parsimony demos regenerate` loops this manifest — the single source
// of truth for "which recipe → which demo pack" — and rewrites both the
// packs and index.json. Adding a demo is one row here. All recipes are
// in-repo, so this needs no cellPACK checkout.
struct Demo {
    id: &'static str,
    label: &'static str,
    /// Recipe path relative to the project root (`--root`).
    recipe: &'static str,
    /// Output filename under `<root>/viewer/data/`.
    out: &'static str,
    /// cellPACK-style loose containment (centre-inside vs whole-sphere-inside).
    loose_bounds: bool,
}

const DEMOS: &[Demo] = &[
    Demo { id: "shape_zoo", label: "shape_zoo — every ingredient shape",
           recipe: "examples/recipes/shape_zoo.json", out: "shape_zoo.pack.json", loose_bounds: false },
    Demo { id: "spheres_strict", label: "spheres_in_a_box (strict bounds)",
           recipe: "examples/recipes/spheres_in_a_box.json", out: "spheres_strict.pack.json", loose_bounds: false },
    Demo { id: "spheres_loose", label: "spheres_in_a_box (loose bounds / matches cellPACK)",
           recipe: "examples/recipes/spheres_in_a_box.json", out: "spheres_loose.pack.json", loose_bounds: true },
    Demo { id: "pdb_proteins", label: "pdb_proteins — GroEL + lysozyme PDB-derived meshes",
           recipe: "examples/recipes/pdb_proteins.json", out: "pdb_proteins.pack.json", loose_bounds: false },
    Demo { id: "blood_plasma", label: "blood_plasma — 6 real PDB meshes",
           recipe: "examples/recipes/blood_plasma.json", out: "blood_plasma.pack.json", loose_bounds: false },
    Demo { id: "mycoplasma_top30", label: "mycoplasma — top-30 species",
           recipe: "examples/recipes/mycoplasma.json", out: "mycoplasma_top30.pack.json", loose_bounds: false },
    Demo { id: "mycoplasma_full", label: "mycoplasma (full) — 642 species + lipid bilayer",
           recipe: "examples/recipes/mycoplasma_full.json", out: "mycoplasma_full.pack.json", loose_bounds: false },
    Demo { id: "ecoli_nucleoid", label: "E. coli — capsule + 4.6 Mbp rod nucleoid",
           recipe: "examples/recipes/ecoli_nucleoid.json", out: "ecoli_nucleoid.pack.json", loose_bounds: false },
];

#[derive(Debug, Parser)]
struct DemosArgs {
    /// `regenerate` (default) re-packs all demos; `list` just prints the manifest.
    #[arg(value_enum, default_value_t = DemosAction::Regenerate)]
    action: DemosAction,

    /// Project root: recipes resolve under `<root>/examples`, packs are
    /// written to `<root>/viewer/data`.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// RNG seed (same seed + recipe = bit-for-bit identical packing).
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// Only act on these demo ids (repeatable). Default: every demo.
    /// A filtered run still rewrites a complete index.json by reusing
    /// the placement counts of packs it didn't repack.
    #[arg(long)]
    only: Vec<String>,

    /// Skip these demo ids (repeatable).
    #[arg(long)]
    exclude: Vec<String>,
}

#[derive(Debug, Clone, Copy, ValueEnum)]
enum DemosAction {
    /// Re-pack every demo recipe into viewer/data/ and rewrite index.json.
    Regenerate,
    /// Print the demo manifest without packing anything.
    List,
}

/// Load + pack one recipe. Shared by `pack`, `demos`, `viewer`, `compare`.
fn pack_recipe(
    recipe_path: &Path,
    seed: u64,
    loose_bounds: bool,
    backend: PlacementBackend,
) -> Result<(Recipe, PlacerOutcome)> {
    let recipe = Recipe::from_file(recipe_path)
        .with_context(|| format!("loading recipe `{}`", recipe_path.display()))?;
    let config = PlacerConfig {
        strict_bounds: !loose_bounds,
        backend,
        ..PlacerConfig::default()
    };
    let placer = GreedyRandomPlacer::new(&recipe, config);
    let out = placer.pack(seed);
    Ok((recipe, out))
}

fn run_demos(args: DemosArgs) -> Result<()> {
    if matches!(args.action, DemosAction::List) {
        println!("{:<18} {:<7} {}", "id", "bounds", "recipe");
        for d in DEMOS {
            let bounds = if d.loose_bounds { "loose" } else { "strict" };
            println!("{:<18} {:<7} {}", d.id, bounds, d.recipe);
        }
        return Ok(());
    }

    let data_dir = args.root.join("viewer/data");
    fs::create_dir_all(&data_dir)
        .with_context(|| format!("creating {}", data_dir.display()))?;

    let selected = |id: &str| -> bool {
        let included = args.only.is_empty() || args.only.iter().any(|s| s == id);
        !args.exclude.iter().any(|s| s == id) && included
    };

    let mut index_demos = Vec::new();
    let (mut packed, mut kept, mut skipped) = (0usize, 0usize, 0usize);
    for d in DEMOS {
        let out_path = data_dir.join(d.out);
        // Repack if selected; otherwise reuse an existing pack's count so
        // a filtered run still emits a complete index.json.
        let count: Option<usize> = if selected(d.id) {
            let recipe_path = args.root.join(d.recipe);
            if !recipe_path.exists() {
                eprintln!("  skip   {:<16} recipe missing: {}", d.id, recipe_path.display());
                skipped += 1;
                None
            } else {
                eprint!("  pack   {:<16} … ", d.id);
                match pack_recipe(&recipe_path, args.seed, d.loose_bounds, PlacementBackend::Legacy) {
                    Ok((recipe, out)) => {
                        let value = write_pack_json(&out.snapshot, &recipe);
                        fs::write(&out_path, serde_json::to_string_pretty(&value)?)
                            .with_context(|| format!("writing {}", out_path.display()))?;
                        let placed = out.snapshot.placements.len();
                        eprintln!("{placed} placements → {}", d.out);
                        packed += 1;
                        Some(placed)
                    }
                    Err(e) => {
                        eprintln!("FAILED: {e:#}");
                        eprintln!("         (missing meshes? regenerate them, then re-run)");
                        skipped += 1;
                        None
                    }
                }
            }
        } else if out_path.exists() {
            kept += 1;
            placement_count(&out_path).ok()
        } else {
            None
        };

        if let Some(placed) = count {
            index_demos.push(serde_json::json!({
                "id": d.id,
                "label": format!("{} ({placed} placed)", d.label),
                "file": d.out,
            }));
        }
    }

    let index = serde_json::json!({
        "comment": "Generated by `parsimony demos regenerate`. Source of truth is the \
                    DEMOS manifest in crates/parsimony-cli/src/main.rs — edit there, not here.",
        "demos": index_demos,
    });
    let index_path = data_dir.join("index.json");
    fs::write(&index_path, serde_json::to_string_pretty(&index)?)
        .with_context(|| format!("writing {}", index_path.display()))?;

    eprintln!("demos: {packed} packed, {kept} kept, {skipped} skipped → {}", index_path.display());
    Ok(())
}

/// Count placements in an existing pack file — used to keep index
/// entries for demos a filtered run didn't repack.
fn placement_count(path: &Path) -> Result<usize> {
    let doc: serde_json::Value = serde_json::from_str(&fs::read_to_string(path)?)?;
    Ok(doc["placements"].as_array().map(|a| a.len()).unwrap_or(0))
}

// ───── viewer ───────────────────────────────────────────────────────
// Pack-and-serve front end for the local three.js viewer. Serves the
// project root over a static server (no extra deps) so the viewer can
// fetch root-relative mesh URLs like /examples/pdb_meshes/…, then opens
// a browser. Mesh (re)generation is `parsimony mesh` /
// `translate-mycoplasma`, kept out of this path.
#[derive(Debug, Parser)]
struct ViewerArgs {
    /// Pack this recipe to viewer/data/latest.pack.json and open it.
    #[arg(long)]
    recipe: Option<PathBuf>,

    /// Open an existing pack by filename (looked up under viewer/data/).
    #[arg(long, conflicts_with = "recipe")]
    pack: Option<String>,

    /// Project root, served as the web root.
    #[arg(long, default_value = ".")]
    root: PathBuf,

    /// HTTP port.
    #[arg(long, default_value_t = 8123)]
    port: u16,

    /// Don't try to open a browser.
    #[arg(long)]
    no_open: bool,

    /// RNG seed when packing a recipe.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// cellPACK-style loose bounds when packing a recipe.
    #[arg(long)]
    loose_bounds: bool,
}

fn run_viewer(args: ViewerArgs) -> Result<()> {
    let root = &args.root;

    // What the viewer should auto-load via its `?file=` param.
    let file_q: Option<String> = if let Some(recipe) = &args.recipe {
        let recipe_path = if recipe.is_absolute() {
            recipe.clone()
        } else {
            root.join(recipe)
        };
        eprintln!("packing {} …", recipe_path.display());
        let (recipe_doc, out) =
            pack_recipe(&recipe_path, args.seed, args.loose_bounds, PlacementBackend::Legacy)?;
        let data_dir = root.join("viewer/data");
        fs::create_dir_all(&data_dir)?;
        let out_path = data_dir.join("latest.pack.json");
        let value = write_pack_json(&out.snapshot, &recipe_doc);
        fs::write(&out_path, serde_json::to_string_pretty(&value)?)
            .with_context(|| format!("writing {}", out_path.display()))?;
        eprintln!(
            "packed {} placements → {}",
            out.snapshot.placements.len(),
            out_path.display()
        );
        Some("data/latest.pack.json".to_string())
    } else {
        args.pack.as_ref().map(|name| format!("data/{name}"))
    };

    let url = match &file_q {
        Some(f) => format!("http://localhost:{}/viewer/index.html?file={f}", args.port),
        None => format!("http://localhost:{}/viewer/index.html", args.port),
    };

    // Native no-cache static server rooted at the project, so root-relative
    // mesh URLs resolve and edited viewer.js / regenerated packs show up on a
    // plain reload. No Python.
    eprintln!("serving {} at {url}", root.display());
    eprintln!("(Ctrl-C to stop)");

    if !args.no_open {
        let open_url = url.clone();
        std::thread::spawn(move || {
            // Let the server bind, then best-effort open a browser.
            std::thread::sleep(std::time::Duration::from_millis(600));
            for opener in ["xdg-open", "open"] {
                if std::process::Command::new(opener).arg(&open_url).spawn().is_ok() {
                    break;
                }
            }
        });
    }

    serve_static(root, args.port)
}

/// Minimal native static file server (no Python): serves files under `root`
/// with no-cache headers, so edited viewer.js / regenerated packs show up on a
/// plain reload. HTTP/1.0, GET only, a thread per connection — enough for the
/// browser's parallel mesh fetches. Blocks until interrupted.
fn serve_static(root: &Path, port: u16) -> Result<()> {
    use std::io::{BufRead, BufReader, Write};
    let listener = std::net::TcpListener::bind(("127.0.0.1", port))
        .with_context(|| format!("binding 127.0.0.1:{port}"))?;
    let root = root.to_path_buf();
    for stream in listener.incoming() {
        let Ok(mut stream) = stream else { continue };
        let root = root.clone();
        std::thread::spawn(move || {
            let peek = match stream.try_clone() {
                Ok(s) => s,
                Err(_) => return,
            };
            let mut reader = BufReader::new(peek);
            let mut line = String::new();
            if reader.read_line(&mut line).is_err() {
                return;
            }
            // Drain the remaining request headers (avoids a reset on close).
            loop {
                let mut h = String::new();
                match reader.read_line(&mut h) {
                    Ok(0) | Err(_) => break,
                    Ok(_) if h == "\r\n" || h == "\n" => break,
                    Ok(_) => {}
                }
            }
            let raw = line.split_whitespace().nth(1).unwrap_or("/");
            let path = raw.split(['?', '#']).next().unwrap_or("/");
            let rel = path.trim_start_matches('/');
            let rel = if rel.is_empty() { "viewer/index.html" } else { rel };
            let body = if rel.contains("..") {
                None // reject path traversal
            } else {
                std::fs::read(root.join(rel)).ok()
            };
            match body {
                Some(bytes) => {
                    let hdr = format!(
                        "HTTP/1.0 200 OK\r\nContent-Type: {}\r\nContent-Length: {}\r\n\
                         Cache-Control: no-store, no-cache, must-revalidate\r\nConnection: close\r\n\r\n",
                        content_type(rel),
                        bytes.len(),
                    );
                    let _ = stream.write_all(hdr.as_bytes());
                    let _ = stream.write_all(&bytes);
                }
                None => {
                    let _ = stream.write_all(
                        b"HTTP/1.0 404 Not Found\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",
                    );
                }
            }
        });
    }
    Ok(())
}

/// Content-Type for a file path, by extension.
fn content_type(path: &str) -> &'static str {
    match path.rsplit('.').next().unwrap_or("") {
        "html" => "text/html; charset=utf-8",
        "js" | "mjs" => "application/javascript; charset=utf-8",
        "json" => "application/json; charset=utf-8",
        "css" => "text/css; charset=utf-8",
        "obj" => "text/plain; charset=utf-8",
        "wasm" => "application/wasm",
        "png" => "image/png",
        "svg" => "image/svg+xml",
        _ => "application/octet-stream",
    }
}

// ───── compare ──────────────────────────────────────────────────────
// Pack a recipe with parsimony (in-process) and, if a cellPACK checkout
// is available, pack the same recipe with cellPACK and report a
// side-by-side comparison (placement counts by radius + position spread).
// This is the validation harness — cellPACK / Maritan results are the
// reference we measure our own packing against. cellPACK is optional: no
// checkout → a parsimony-only report.
#[derive(Debug, Parser)]
struct CompareArgs {
    /// Recipe JSON path.
    recipe: PathBuf,

    /// cellPACK checkout to compare against (uses its `.venv` python).
    #[arg(long, default_value = "../cellpack")]
    cellpack: PathBuf,

    /// Skip cellPACK entirely; report parsimony only.
    #[arg(long)]
    no_cellpack: bool,

    /// Directory for the generated `.simularium` files.
    #[arg(long, default_value = "/tmp/parsimony_compare")]
    out_dir: PathBuf,

    /// RNG seed for parsimony.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// cellPACK `spacing` (it scales positions/radii by 1/spacing in its
    /// Simularium output; we rescale before comparing).
    #[arg(long, default_value_t = 1)]
    spacing: u32,

    /// Use strict bounds for parsimony (whole sphere inside). Default is
    /// loose (cellPACK semantics) for an apples-to-apples comparison.
    #[arg(long)]
    strict_bounds: bool,
}

fn run_compare(args: CompareArgs) -> Result<()> {
    use parsimony_bench::compare::{compare_counts, distribution_stats};
    use parsimony_bench::parse::SimulariumDoc;
    use parsimony_bench::runner::{run_cellpack, CellpackConfig};

    fs::create_dir_all(&args.out_dir)
        .with_context(|| format!("creating {}", args.out_dir.display()))?;

    // Parsimony — packed in-process with *both* backends, so the table puts
    // the cellPACK-method engine (legacy) and the content-scaled one
    // (octree) side by side against cellPACK itself.
    let pack_backend = |backend: PlacementBackend,
                        label: &'static str,
                        file: &str|
     -> Result<(SimulariumDoc, std::time::Duration)> {
        eprintln!("packing parsimony ({label})…");
        let t = Instant::now();
        let (recipe, out) = pack_recipe(&args.recipe, args.seed, !args.strict_bounds, backend)?;
        let elapsed = t.elapsed();
        let path = args.out_dir.join(file);
        let sim = write_simularium_json(&out.snapshot, &recipe);
        fs::write(&path, serde_json::to_string(&sim)?)
            .with_context(|| format!("writing {}", path.display()))?;
        let doc = SimulariumDoc::from_json(&fs::read_to_string(&path)?, label)?;
        eprintln!("  {} placements in {:.2?}", doc.agents.len(), elapsed);
        Ok((doc, elapsed))
    };

    let (legacy_doc, legacy_el) =
        pack_backend(PlacementBackend::Legacy, "legacy", "parsimony_legacy.simularium")?;
    let (octree_doc, octree_el) =
        pack_backend(PlacementBackend::Octree, "octree", "parsimony_octree.simularium")?;

    // cellPACK — only if its venv python is present.
    let py = args.cellpack.join(".venv/bin/python");
    let cp = if args.no_cellpack {
        None
    } else if !py.exists() {
        eprintln!(
            "cellpack python not found at {} — parsimony-only report.\n  \
             (point --cellpack at a checkout, or pass --no-cellpack)",
            py.display(),
        );
        None
    } else {
        eprintln!("packing cellpack via {}…", py.display());
        let cfg = CellpackConfig { python: py, spacing: args.spacing, ..Default::default() };
        match run_cellpack(&args.recipe, &args.out_dir, &cfg) {
            Ok(run) => {
                let mut doc = SimulariumDoc::from_json(
                    &fs::read_to_string(&run.simularium_path)?,
                    "cellpack",
                )?;
                doc.rescale(args.spacing as f32);
                eprintln!("  {} placements in {:.2?}", doc.agents.len(), run.elapsed);
                Some((doc, run.elapsed))
            }
            Err(e) => {
                eprintln!("  cellpack run failed: {e:#}");
                None
            }
        }
    };

    // Report.
    println!("\n=== compare: {} ===\n", args.recipe.display());
    println!("{:<18} {:>12} {:>12}", "engine", "placements", "wall");
    println!("{:-<18} {:->12} {:->12}", "", "", "");
    println!("{:<18} {:>12} {:>12.2?}", "parsimony-legacy", legacy_doc.agents.len(), legacy_el);
    println!("{:<18} {:>12} {:>12.2?}", "parsimony-octree", octree_doc.agents.len(), octree_el);
    if let Some((cp_doc, cp_el)) = &cp {
        println!("{:<18} {:>12} {:>12.2?}", "cellpack", cp_doc.agents.len(), cp_el);
        let secs = |d: std::time::Duration| d.as_secs_f64().max(1e-9);
        println!(
            "\nspeedup vs cellpack:  legacy {:.0}×   octree {:.0}×",
            secs(*cp_el) / secs(legacy_el),
            secs(*cp_el) / secs(octree_el),
        );
        println!("\nper-radius counts (cellpack vs parsimony-legacy):");
        println!("{:>10} {:>12} {:>12} {:>9}", "radius", "cellpack", "legacy", "%diff");
        println!("{:->10} {:->12} {:->12} {:->9}", "", "", "", "");
        for r in compare_counts(cp_doc, &legacy_doc) {
            println!(
                "{:>10.2} {:>12} {:>12} {:>+8.1}%",
                r.radius, r.a_count, r.b_count, r.pct_diff(),
            );
        }
    }
    println!("\nposition stddev (x, y, z):");
    let pl = distribution_stats(&legacy_doc.agents);
    let po = distribution_stats(&octree_doc.agents);
    println!("  parsimony-legacy ({:.1}, {:.1}, {:.1})", pl.stddev[0], pl.stddev[1], pl.stddev[2]);
    println!("  parsimony-octree ({:.1}, {:.1}, {:.1})", po.stddev[0], po.stddev[1], po.stddev[2]);
    if let Some((cp_doc, _)) = &cp {
        let cs = distribution_stats(&cp_doc.agents);
        println!("  cellpack         ({:.1}, {:.1}, {:.1})", cs.stddev[0], cs.stddev[1], cs.stddev[2]);
    }
    Ok(())
}

// ───── mesh ─────────────────────────────────────────────────────────
// Generate mesh LODs from PDB/mmCIF structures, natively (see mesh_gen):
// VdW SDF → gaussian smooth → surface nets → LOD downsample → OBJ. Takes a
// structure file, a 4-char RCSB ID (fetched + cached under examples/pdb_cache),
// or a whole directory, writing <slug>.lod<N>.obj per LOD voxel size.
#[derive(Debug, Parser)]
struct MeshArgs {
    /// A PDB/CIF file, a 4-char RCSB ID, or a directory of them.
    path: String,

    /// LOD voxel sizes in Å (coarse→fine), comma-separated.
    #[arg(long, default_value = "16,8,4,2.5")]
    lods: String,

    /// Output directory for the generated <slug>.lod<N>.obj files.
    #[arg(long, default_value = "examples/pdb_meshes")]
    out_dir: PathBuf,
}

fn run_mesh(args: MeshArgs) -> Result<()> {
    let lods = parse_lods(&args.lods)?;
    fs::create_dir_all(&args.out_dir)?;

    // Inputs: a directory of .pdb/.cif, a single file, or a 4-char PDB ID.
    let p = Path::new(&args.path);
    let inputs: Vec<PathBuf> = if p.is_dir() {
        let mut found: Vec<PathBuf> = fs::read_dir(p)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|pp| {
                pp.extension()
                    .and_then(|s| s.to_str())
                    .map(|x| x.eq_ignore_ascii_case("pdb") || x.eq_ignore_ascii_case("cif"))
                    .unwrap_or(false)
            })
            .collect();
        found.sort();
        found
    } else {
        vec![resolve_structure(&args.path)?]
    };
    anyhow::ensure!(!inputs.is_empty(), "no .pdb/.cif inputs under {}", args.path);

    let (mut ok, mut failed) = (0usize, 0usize);
    for input in &inputs {
        let slug = input
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("mesh")
            .to_string();
        eprint!("  {slug} ({} LODs) … ", lods.len());
        match mesh_one(input, &lods, &slug, &args.out_dir) {
            Ok(n) => {
                eprintln!("→ {n} verts (finest)");
                ok += 1;
            }
            Err(e) => {
                eprintln!("FAILED ({e})");
                failed += 1;
            }
        }
    }
    eprintln!("mesh: {ok} ok, {failed} failed → {}", args.out_dir.display());
    if failed > 0 {
        anyhow::bail!("{failed} mesh generation(s) failed");
    }
    Ok(())
}

/// Parse "16,8,4,1.5" → `[16, 8, 4, 1.5]`.
fn parse_lods(s: &str) -> Result<Vec<f32>> {
    let lods: Vec<f32> = s.split(',').filter_map(|x| x.trim().parse().ok()).collect();
    anyhow::ensure!(!lods.is_empty(), "no valid --lods (expected e.g. 16,8,4,1.5)");
    Ok(lods)
}

/// Mesh one structure file into `<out_dir>/<slug>.lod<N>.obj` per LOD.
/// Returns the finest LOD's vertex count.
fn mesh_one(path: &Path, lods: &[f32], slug: &str, out_dir: &Path) -> Result<usize> {
    let atoms = mesh_gen::load_atoms(path)?;
    let meshes = mesh_gen::mesh_lods(&atoms, lods)?;
    let (mut finest_verts, mut finest_res) = (0usize, f32::INFINITY);
    for (i, (res, mesh)) in lods.iter().zip(meshes.iter()).enumerate() {
        let out = out_dir.join(format!("{slug}.lod{i}.obj"));
        let header = format!(
            "parsimony mesh: {slug} (source {})\natoms: {}  voxel: {res} Å  verts: {}  tris: {}",
            path.display(),
            atoms.len(),
            mesh.0.len(),
            mesh.1.len()
        );
        mesh_gen::write_obj(mesh, &out, &header)?;
        if *res < finest_res {
            finest_res = *res;
            finest_verts = mesh.0.len();
        }
    }
    Ok(finest_verts)
}

/// Resolve a structure argument to a local file: an existing path is used as
/// is; a 4-char RCSB ID is taken from (or fetched into) examples/pdb_cache,
/// trying PDB then mmCIF.
fn resolve_structure(arg: &str) -> Result<PathBuf> {
    let p = Path::new(arg);
    if p.exists() {
        return Ok(p.to_path_buf());
    }
    let id = arg.trim();
    anyhow::ensure!(
        id.len() == 4 && id.chars().all(|c| c.is_ascii_alphanumeric()),
        "{arg} is neither an existing file nor a 4-char PDB ID"
    );
    let cache = Path::new("examples/pdb_cache");
    fs::create_dir_all(cache).ok();
    let lower = id.to_lowercase();
    for ext in ["pdb", "cif"] {
        let c = cache.join(format!("{lower}.{ext}"));
        if c.exists() {
            return Ok(c);
        }
    }
    for ext in ["pdb", "cif"] {
        let url = format!("https://files.rcsb.org/download/{}.{ext}", id.to_uppercase());
        let out = cache.join(format!("{lower}.{ext}"));
        eprint!("  fetching {url} … ");
        match fetch_to(&url, &out) {
            Ok(path) => {
                eprintln!("ok");
                return Ok(path);
            }
            Err(e) => eprintln!("({e})"),
        }
    }
    anyhow::bail!("could not fetch structure {id} from RCSB")
}

/// Fetch an RCSB chemical-component (ligand) CCD definition — an mmCIF with
/// atomic coordinates in `_chem_comp_atom` — into examples/pdb_cache. This is
/// how we get real small molecules that aren't deposited as their own
/// structure (e.g. a phospholipid).
fn fetch_ligand(ccd: &str) -> Result<PathBuf> {
    let cache = Path::new("examples/pdb_cache");
    fs::create_dir_all(cache).ok();
    let out = cache.join(format!("{}.cif", ccd.to_lowercase()));
    if out.exists() {
        return Ok(out);
    }
    let url = format!("https://files.rcsb.org/ligands/download/{}.cif", ccd.to_uppercase());
    eprint!("  fetching ligand {url} … ");
    let r = fetch_to(&url, &out);
    eprintln!("{}", if r.is_ok() { "ok" } else { "failed" });
    r
}

/// Download `url` to `out`, returning its path. Native HTTP (no curl/python).
/// Publishes atomically (unique temp + rename) so that when several species
/// share one RCSB ID and fetch it concurrently, a reader never sees a
/// half-written file.
fn fetch_to(url: &str, out: &Path) -> Result<PathBuf> {
    use std::sync::atomic::{AtomicU64, Ordering};
    static SEQ: AtomicU64 = AtomicU64::new(0);
    let resp = ureq::get(url).call().map_err(|e| anyhow::anyhow!("{e}"))?;
    let mut buf = Vec::new();
    std::io::Read::read_to_end(&mut resp.into_reader(), &mut buf)?;
    anyhow::ensure!(!buf.is_empty(), "empty response");
    let tmp = out.with_extension(format!("tmp{}", SEQ.fetch_add(1, Ordering::Relaxed)));
    fs::write(&tmp, &buf)?;
    fs::rename(&tmp, out)?;
    Ok(out.to_path_buf())
}

// ───── translate-mycoplasma ──────────────────────────────────────────
// The whole-cell Mycoplasma recipe comes from the cellPACK data repo
// (Maritan et al.). This subcommand does the whole thing natively (no
// Python): clone the data repo if needed, walk its curated recipe, mesh
// each species' structure with the native VdW mesher (mesh_gen), and
// compose a parsimony recipe — proteins + a coarse lipid bilayer, plus the
// chromosome (genome-driven supercoil with instanced dsDNA segments), its
// bound RNA/DNA polymerases, and free + nascent mRNA. Reproducible from one
// command. Run from the repo root.
#[derive(Debug, Parser)]
struct TranslateMycoplasmaArgs {
    /// Git URL of the cellPACK Mycoplasma data repo.
    #[arg(long, default_value = "https://github.com/ccsb-scripps/MycoplasmaGenitalium")]
    data_repo: String,

    /// Where to clone / find the data repo (its `cellPACK_Data/` is used).
    #[arg(long, default_value = ".cache/MycoplasmaGenitalium")]
    cache_dir: PathBuf,

    /// Output parsimony recipe path.
    #[arg(long, default_value = "examples/recipes/mycoplasma_full.json")]
    out_recipe: PathBuf,

    /// Output directory for the generated per-protein LOD meshes.
    #[arg(long, default_value = "examples/pdb_meshes/mycoplasma")]
    out_meshes: PathBuf,

    /// LOD voxel sizes in Å (coarse→fine), comma-separated.
    #[arg(long, default_value = "16,8,4,1.5")]
    lods: String,

    /// Most-abundant interior species to include (0 = all; 30 = the demo).
    #[arg(long, default_value_t = 0)]
    top_n: u32,

    /// Cell sphere radius in Å.
    #[arg(long, default_value_t = 2000.0)]
    cell_radius: f32,

    /// Membrane lipid *patches* tiled over the cell surface (0 disables). Each
    /// patch is one mesh baking a dense hex-packed disc of ~260 real lipids, so
    /// a few thousand patches cover the cell at full density. More = denser
    /// overlap (fewer gaps), heavier viewer.
    #[arg(long, default_value_t = 16000)]
    lipid_count: u32,

    /// Gene-annotation CSV for the genome (referenced by the chromosome block,
    /// path resolved relative to the output recipe).
    #[arg(long, default_value = "examples/genome/mycoplasma_g37_genes.csv")]
    genome: PathBuf,

    /// Re-mesh every species even if its OBJ files already exist (needed when
    /// changing --lods, since meshes are otherwise cached by filename).
    #[arg(long)]
    force: bool,
}

/// cellPACK curated-recipe JSON (only the fields we read).
#[derive(serde::Deserialize)]
struct CpRoot {
    #[serde(rename = "Compartments", default)]
    compartments: Vec<CpOuter>,
}
#[derive(serde::Deserialize)]
struct CpOuter {
    #[serde(rename = "Compartments", default)]
    compartments: Vec<CpRegion>,
}
#[derive(serde::Deserialize)]
struct CpRegion {
    #[serde(default)]
    name: String,
    #[serde(rename = "IngredientGroups", default)]
    groups: Vec<CpGroup>,
}
#[derive(serde::Deserialize)]
struct CpGroup {
    #[serde(rename = "Ingredients", default)]
    ingredients: Vec<CpIngredient>,
}
#[derive(serde::Deserialize, Clone)]
struct CpIngredient {
    name: String,
    #[serde(rename = "nbMol", default)]
    nb_mol: u64,
    #[serde(default)]
    source: Option<CpSource>,
}
#[derive(serde::Deserialize, Clone)]
struct CpSource {
    #[serde(default)]
    pdb: Option<String>,
}

/// Colour palette cycled across species (matches the former Python).
const TRANSLATE_PALETTE: [[f32; 3]; 10] = [
    [0.85, 0.35, 0.35], [0.95, 0.62, 0.35], [0.95, 0.85, 0.45], [0.55, 0.85, 0.45],
    [0.45, 0.85, 0.65], [0.45, 0.75, 0.95], [0.55, 0.55, 0.95], [0.80, 0.55, 0.85],
    [0.95, 0.55, 0.75], [0.75, 0.55, 0.45],
];

/// Sort by descending abundance, keep the top `n` (0 = all).
fn cp_select(mut ings: Vec<CpIngredient>, n: u32) -> Vec<CpIngredient> {
    ings.sort_by(|a, b| b.nb_mol.cmp(&a.nb_mol));
    if n > 0 {
        ings.truncate(n as usize);
    }
    ings
}

/// Find a species' structure file in cellPACK_Data/proteins/: a 4-char ID
/// lives as <ID>_BU1_.cif / <ID>.cif, or `source.pdb` is a literal filename.
fn find_structure_file(name: &str, src_pdb: &str, proteins_dir: &Path) -> Option<PathBuf> {
    let mut candidates: Vec<PathBuf> = Vec::new();
    if !src_pdb.is_empty() {
        if src_pdb.len() == 4 && src_pdb.chars().all(|c| c.is_ascii_alphanumeric()) {
            candidates.push(proteins_dir.join(format!("{}_BU1_.cif", src_pdb.to_uppercase())));
            candidates.push(proteins_dir.join(format!("{}.cif", src_pdb.to_uppercase())));
            candidates.push(proteins_dir.join(format!("{}.cif", src_pdb.to_lowercase())));
        }
        candidates.push(proteins_dir.join(src_pdb));
        if let Some(stem) = Path::new(src_pdb).file_stem().and_then(|s| s.to_str()) {
            candidates.push(proteins_dir.join(format!("{stem}.cif")));
            candidates.push(proteins_dir.join(format!("{stem}.pdb")));
            candidates.push(proteins_dir.join(format!("{stem}_BU1_.cif")));
        }
    }
    candidates.push(proteins_dir.join(format!("{name}.cif")));
    candidates.push(proteins_dir.join(format!("{name}.pdb")));
    candidates.into_iter().find(|c| c.exists())
}

/// Extract an RCSB PDB ID from a `source.pdb` reference for fetching: a bare
/// 4-char ID (`1abc`) or the leading ID of a filename (`6rut_bu1.pdb` → `6rut`).
/// RCSB IDs start with a digit, which avoids matching cellPACK's computed
/// names (`MG_…`, `computed.pdb`) that should only ever be local.
fn pdb_id_from(src: &str) -> Option<String> {
    let stem = Path::new(src.trim()).file_stem().and_then(|s| s.to_str()).unwrap_or("");
    let head = stem.get(..4)?;
    (head.chars().all(|c| c.is_ascii_alphanumeric()) && head.as_bytes()[0].is_ascii_digit())
        .then(|| head.to_string())
}

/// `target` expressed relative to `base_dir` (with `..`), forward-slashed.
fn relativize(target: &Path, base_dir: &Path) -> String {
    let t: Vec<_> = target.components().collect();
    let b: Vec<_> = base_dir.components().collect();
    let mut i = 0;
    while i < t.len() && i < b.len() && t[i] == b[i] {
        i += 1;
    }
    let mut parts: Vec<String> = std::iter::repeat("..".to_string()).take(b.len() - i).collect();
    for c in &t[i..] {
        parts.push(c.as_os_str().to_string_lossy().into_owned());
    }
    parts.join("/")
}

/// Mesh a structure into `<out_dir>/<slug>.lod<N>.obj` per LOD (cached: skipped
/// if all already exist) and return its parsimony mesh-ingredient spec.
fn mesh_to_spec(
    struct_path: &Path,
    slug: &str,
    lods: &[f32],
    out_dir: &Path,
    recipe_dir: &Path,
    color: [f32; 3],
    proxy_voxel: f32,
    force: bool,
    reorient_z: bool,
) -> Result<serde_json::Value> {
    let obj_paths: Vec<PathBuf> = (0..lods.len())
        .map(|i| out_dir.join(format!("{slug}.lod{i}.obj")))
        .collect();
    if force || !obj_paths.iter().all(|p| p.exists()) {
        let atoms = mesh_gen::load_atoms(struct_path)?;
        let mut meshes = mesh_gen::mesh_lods(&atoms, lods)?;
        if reorient_z {
            mesh_gen::reorient_to_z(&mut meshes);
        }
        for (i, (res, mesh)) in lods.iter().zip(meshes.iter()).enumerate() {
            let header = format!(
                "parsimony translate-mycoplasma: {slug} (source {})\natoms: {}  voxel: {res} Å  verts: {}  tris: {}",
                struct_path.display(),
                atoms.len(),
                mesh.0.len(),
                mesh.1.len()
            );
            mesh_gen::write_obj(mesh, &obj_paths[i], &header)?;
        }
    }
    let lod_specs: Vec<serde_json::Value> = obj_paths
        .iter()
        .zip(lods)
        .map(|(p, vs)| serde_json::json!({ "path": relativize(p, recipe_dir), "voxel_size": vs }))
        .collect();
    Ok(serde_json::json!({
        "type": "mesh",
        "color": color,
        "proxy_voxel_size": proxy_voxel,
        "mesh_lods": lod_specs,
    }))
}

/// Mesh every species in a region in parallel, returning `(slug, spec, count)`
/// for those whose structure was found.
fn mesh_cp_region(
    sel: &[CpIngredient],
    pal_off: usize,
    proteins_dir: &Path,
    lods: &[f32],
    out_meshes: &Path,
    recipe_dir: &Path,
    force: bool,
) -> Vec<(String, serde_json::Value, u64)> {
    use rayon::prelude::*;
    let proxy = (lods.iter().cloned().fold(f32::INFINITY, f32::min) * 2.5).max(4.0);
    sel.par_iter()
        .enumerate()
        .filter_map(|(idx, ing)| {
            let slug = ing.name.replace([' ', '/'], "_");
            let src = ing.source.as_ref().and_then(|s| s.pdb.clone()).unwrap_or_default();
            // Prefer the local cellPACK structure; otherwise cellPACK often
            // references an RCSB ID it didn't ship (bare "1abc", or a filename
            // like "6rut_bu1.pdb") — pull the ID and fetch it.
            let struct_path = match find_structure_file(&ing.name, &src, proteins_dir) {
                Some(p) => p,
                None => match pdb_id_from(&src) {
                    Some(id) => match resolve_structure(&id) {
                        Ok(p) => p,
                        Err(e) => {
                            eprintln!("  ! {slug}: fetch {id} failed ({e})");
                            return None;
                        }
                    },
                    None => {
                        eprintln!("  ! {slug}: no structure file (pdb={src:?})");
                        return None;
                    }
                },
            };
            let color = TRANSLATE_PALETTE[(idx + pal_off) % TRANSLATE_PALETTE.len()];
            match mesh_to_spec(&struct_path, &slug, lods, out_meshes, recipe_dir, color, proxy, force, false) {
                Ok(spec) => Some((slug, spec, ing.nb_mol)),
                Err(e) => {
                    eprintln!("  ! {slug}: {e}");
                    None
                }
            }
        })
        .collect()
}

fn run_translate_mycoplasma(args: TranslateMycoplasmaArgs) -> Result<()> {
    // Clone the data repo if its data dir is missing (idempotent otherwise).
    let data_dir = args.cache_dir.join("cellPACK_Data");
    if data_dir.exists() {
        eprintln!("using cellPACK data at {}", data_dir.display());
    } else {
        ensure_git()?;
        if let Some(parent) = args.cache_dir.parent().filter(|p| !p.as_os_str().is_empty()) {
            fs::create_dir_all(parent)?;
        }
        eprintln!("cloning {} → {}", args.data_repo, args.cache_dir.display());
        let st = std::process::Command::new("git")
            .args(["clone", "--depth", "1", &args.data_repo])
            .arg(&args.cache_dir)
            .status()
            .context("running git clone")?;
        anyhow::ensure!(st.success(), "git clone failed ({st})");
    }

    fs::create_dir_all(&args.out_meshes)?;
    let recipe_dir = args.out_recipe.parent().unwrap_or(Path::new(".")).to_path_buf();
    fs::create_dir_all(&recipe_dir).ok();
    let lods = parse_lods(&args.lods)?;

    // Locate + parse the curated cellPACK recipe.
    let recipes_dir = data_dir.join("recipes");
    let recipe_path = fs::read_dir(&recipes_dir)
        .with_context(|| format!("reading {}", recipes_dir.display()))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .find(|p| {
            p.file_name()
                .and_then(|s| s.to_str())
                .map(|n| n.starts_with("mg_curated_clean") && n.ends_with("serialized.json"))
                .unwrap_or(false)
        })
        .with_context(|| format!("no mg_curated_clean*serialized.json under {}", recipes_dir.display()))?;
    eprintln!("reading {}", recipe_path.display());
    let cp: CpRoot = serde_json::from_reader(std::io::BufReader::new(fs::File::open(&recipe_path)?))
        .context("parsing cellPACK recipe")?;

    // Walk to the interior + surface ingredient groups.
    let (mut interior, mut surface) = (Vec::new(), Vec::new());
    for outer in &cp.compartments {
        for region in &outer.compartments {
            let ings: Vec<CpIngredient> =
                region.groups.iter().flat_map(|g| g.ingredients.iter().cloned()).collect();
            match region.name.as_str() {
                "interior" => interior = ings,
                "surface" => surface = ings,
                _ => {}
            }
        }
    }
    let interior = cp_select(interior, args.top_n);
    let surface = cp_select(surface, args.top_n);
    eprintln!(
        "selected {} interior + {} surface species; meshing at LODs [{}]…",
        interior.len(),
        surface.len(),
        args.lods
    );

    let proteins_dir = data_dir.join("proteins");
    let interior_meshed =
        mesh_cp_region(&interior, 0, &proteins_dir, &lods, &args.out_meshes, &recipe_dir, args.force);
    let surface_meshed =
        mesh_cp_region(&surface, 5, &proteins_dir, &lods, &args.out_meshes, &recipe_dir, args.force);
    let skipped = (interior.len() - interior_meshed.len()) + (surface.len() - surface_meshed.len());
    let (n_interior, n_surface) = (interior_meshed.len(), surface_meshed.len());

    let mut objects = serde_json::Map::new();
    let mut interior_entries: Vec<serde_json::Value> = Vec::new();
    let mut surface_entries: Vec<serde_json::Value> = Vec::new();
    for (slug, spec, n) in interior_meshed {
        objects.insert(slug.clone(), spec);
        interior_entries.push(serde_json::json!({ "object": slug, "count": n }));
    }
    for (slug, spec, n) in surface_meshed {
        objects.insert(slug.clone(), spec);
        surface_entries.push(serde_json::json!({ "object": slug, "count": n }));
    }

    // Special ingredients for the chromosome + transcription machinery, meshed
    // from cached reference structures (examples/pdb_meshes, beside the recipe).
    let special_dir = args
        .out_meshes
        .parent()
        .unwrap_or(Path::new("examples/pdb_meshes"))
        .to_path_buf();
    // dna_segment: real B-DNA dodecamer (1BNA); local Z is the helix axis.
    let dna = resolve_structure("1BNA")?;
    let mut seg = mesh_to_spec(&dna, "1BNA", &[12.0, 6.0, 3.0, 1.5], &special_dir, &recipe_dir, [0.30, 0.55, 0.85], 6.0, args.force, true)?;
    seg["principal_vector"] = serde_json::json!([0, 0, 1]);
    objects.insert("dna_segment".into(), seg);
    // rna_segment: a real A-form RNA (1RNA), reoriented so its helix axis is Z,
    // tiled along each mRNA's bead chain (RNA-orange).
    let rna = resolve_structure("1RNA")?;
    let mut rseg = mesh_to_spec(&rna, "1RNA", &[12.0, 6.0, 3.0, 1.5], &special_dir, &recipe_dir, [0.95, 0.55, 0.15], 6.0, args.force, true)?;
    rseg["principal_vector"] = serde_json::json!([0, 0, 1]);
    objects.insert("rna_segment".into(), rseg);
    // RNA polymerase (1hqm) + DNA polymerase (2hpi) ride the chromosome.
    let rnap = resolve_structure("1hqm")?;
    objects.insert(
        "rnap".into(),
        mesh_to_spec(&rnap, "1hqm", &[16.0, 8.0, 4.0], &special_dir, &recipe_dir, [0.96, 0.85, 0.2], 12.0, args.force, false)?,
    );
    let dnap = resolve_structure("2hpi")?;
    objects.insert(
        "dnap".into(),
        mesh_to_spec(&dnap, "2hpi", &[16.0, 8.0, 4.0], &special_dir, &recipe_dir, [0.2, 0.9, 0.95], 10.0, args.force, false)?,
    );
    // mRNA: a coarse linear bead chain that the pack writer renders as the
    // rna_segment mesh tiled along it. Free copies in the cytoplasm, nascent
    // copies on the genome.
    objects.insert(
        "mrna".into(),
        serde_json::json!({
            "type": "multi_sphere",
            "color": [0.95, 0.55, 0.15],
            "principal_vector": [1.0, 0.0, 0.0],
            "positions": [[-72,0,0],[-56,7,2],[-40,-4,-3],[-24,6,4],[-8,-5,-2],[8,6,3],[24,-4,-4],[40,7,2],[56,-3,-3],[72,2,0]],
            "radii": [9,9,9,9,9,9,9,9,9,9],
            "segment": "rna_segment",
        }),
    );
    // Real phospholipid (a chemical component) meshed like everything else:
    // orient it head-tail along Z, then mirror into a tail-to-tail bilayer pair
    // so each surface placement is a full bilayer-spanning unit of real lipid
    // atoms (Z = the bilayer normal, aligned to the radial direction at
    // placement). Tiled densely over the cell surface.
    if args.lipid_count > 0 {
        let lipid_cif = fetch_ligand("LHG")?; // 1,2-dipalmitoyl-sn-glycero-3-phosphoglycerol
        let mut latoms = mesh_gen::load_atoms(&lipid_cif)?;
        // Drop hydrogens (VdW ~1.2 Å). The CCD ships explicit H, and on the
        // thin acyl tails each C–H spikes past the carbon tube at 1.5 Å — so we
        // mesh the heavy-atom surface, matching the (H-free) protein/DNA/RNA
        // structures from RCSB.
        latoms.retain(|a| a.radius > 1.3);
        mesh_gen::reorient_atoms_to_z(&mut latoms);
        // The phosphate/glycerol head is oxygen-rich (VdW ~1.52 Å) while the
        // tails are carbon — so put the O-heavy end at +Z. That makes the
        // mirrored pair deterministically heads-out / tails-in (a real
        // bilayer), instead of a 50/50 coin flip that can bury the heads.
        let is_o = |a: &mesh_gen::Atom| (a.radius - 1.52).abs() < 0.03;
        let o_top = latoms.iter().filter(|a| is_o(a) && a.pos.z > 0.0).count();
        let o_bot = latoms.iter().filter(|a| is_o(a) && a.pos.z < 0.0).count();
        if o_bot > o_top {
            for a in latoms.iter_mut() {
                a.pos.z = -a.pos.z;
            }
        }
        // Fatten the heavy atoms so the thin acyl tails mesh as smooth tubes
        // instead of a row of per-atom bumps (the residual spikiness). Done
        // after the O-based head test, which keys off the real oxygen radius.
        for a in latoms.iter_mut() {
            a.radius += 0.8;
        }
        let zmax = latoms.iter().map(|a| a.pos.z.abs()).fold(0.0_f32, f32::max).max(1.0);
        // One bilayer-spanning pair: the lipid + its Z-mirror, tails meeting at
        // the midplane (z = 0), heads at the two surfaces (z = bilayer normal).
        let pair: Vec<mesh_gen::Atom> = latoms
            .iter()
            .flat_map(|a| {
                [
                    mesh_gen::Atom { pos: nalgebra::Vector3::new(a.pos.x, a.pos.y, a.pos.z + zmax), radius: a.radius },
                    mesh_gen::Atom { pos: nalgebra::Vector3::new(a.pos.x, a.pos.y, -a.pos.z - zmax), radius: a.radius },
                ]
            })
            .collect();
        // A membrane PATCH: a hex-packed disc of those pairs (with positional
        // jitter so it isn't crystalline), baked into one mesh. A few thousand
        // tiled+rolled patches then cover the cell at full lipid density for a
        // tiny instance count — instead of millions of individual lipids.
        let (patch_r, spacing) = (110.0_f32, 13.0_f32);
        let dy = spacing * (3.0_f32).sqrt() / 2.0;
        let rows = (patch_r / dy).ceil() as i32;
        let cols = (patch_r / spacing).ceil() as i32 + 1;
        let jit = |a: i32, b: i32| {
            let v = (a as u32).wrapping_mul(73856093) ^ (b as u32).wrapping_mul(19349663);
            ((v & 0xffff) as f32 / 65535.0 - 0.5) * spacing * 0.4
        };
        let mut patch: Vec<mesh_gen::Atom> = Vec::new();
        let mut n_lipids = 0usize;
        for j in -rows..=rows {
            let y = j as f32 * dy;
            let xoff = if j & 1 == 0 { 0.0 } else { spacing * 0.5 };
            for i in -cols..=cols {
                let x = i as f32 * spacing + xoff;
                if x * x + y * y > patch_r * patch_r {
                    continue;
                }
                let (jx, jy) = (jit(i, j), jit(j, i));
                n_lipids += 1;
                for a in &pair {
                    patch.push(mesh_gen::Atom {
                        pos: nalgebra::Vector3::new(a.pos.x + x + jx, a.pos.y + y + jy, a.pos.z),
                        radius: a.radius,
                    });
                }
            }
        }
        // Coarser than the proteins on purpose: the acyl tails are thin
        // single-atom chains, so a fine voxel resolves every atom as a bump
        // ("spiky"). At ~2.5 Å the atoms merge into smooth tubes — plenty for a
        // membrane that's small on screen.
        let lipid_lods = [10.0_f32, 5.0, 2.5];
        let lipid_objs: Vec<PathBuf> = (0..lipid_lods.len())
            .map(|i| special_dir.join(format!("lipid.lod{i}.obj")))
            .collect();
        if args.force || !lipid_objs.iter().all(|p| p.exists()) {
            let meshes = mesh_gen::mesh_lods(&patch, &lipid_lods)?;
            for (i, (res, m)) in lipid_lods.iter().zip(meshes.iter()).enumerate() {
                mesh_gen::write_obj(
                    m,
                    &lipid_objs[i],
                    &format!("parsimony lipid patch ({n_lipids} LHG lipids)  voxel: {res} Å  verts: {}", m.0.len()),
                )?;
            }
            eprintln!("lipid patch: {n_lipids} lipids meshed");
        }
        let lipid_lod_specs: Vec<serde_json::Value> = lipid_objs
            .iter()
            .zip(lipid_lods)
            .map(|(p, vs)| serde_json::json!({ "path": relativize(p, &recipe_dir), "voxel_size": vs }))
            .collect();
        objects.insert(
            "lipid".into(),
            serde_json::json!({
                "type": "mesh",
                "color": [0.96, 0.86, 0.55],
                "principal_vector": [0, 0, 1],
                "proxy_voxel_size": 12.0,
                "packing_mode": "tiled",
                "mesh_lods": lipid_lod_specs,
            }),
        );
        surface_entries.push(serde_json::json!({ "object": "lipid", "count": args.lipid_count }));
    }
    // Free cytoplasmic mRNA (nascent mRNA + polymerases ride the chromosome).
    interior_entries.insert(0, serde_json::json!({ "object": "mrna", "count": 250 }));

    let n_objects = objects.len();
    let r = args.cell_radius;
    let recipe = serde_json::json!({
        "name": "mycoplasma_genitalium",
        "version": "0.1.0",
        "format_version": "2.1-parsimony",
        "description": format!(
            "Mycoplasma genitalium, translated natively from ccsb-scripps/MycoplasmaGenitalium \
             (Maritan et al., JMB 2022). {n_interior} interior + {n_surface} surface species, \
             VdW-surface meshed at LODs [{}]. Sphere cell of radius {r:.0} Å with a genome-driven \
             supercoiled chromosome (instanced dsDNA segments), bound RNA/DNA polymerases, and \
             free + nascent mRNA.",
            args.lods,
        ),
        "bounding_box": [[-r - 50.0, -r - 50.0, -r - 50.0], [r + 50.0, r + 50.0, r + 50.0]],
        "objects": serde_json::Value::Object(objects),
        "composition": {
            "space": { "regions": { "interior": ["cell"] } },
            "cell": {
                "compartment": { "kind": "sphere", "center": [0, 0, 0], "radius": r },
                "regions": { "interior": interior_entries, "surface": surface_entries },
            },
        },
        "chromosome": {
            "genome": relativize(&args.genome, &recipe_dir),
            "segment": "dna_segment",
            "beads": 90000,
            "spacing": 22,
            "bead_radius": 10,
            "color": [0.86, 0.30, 0.42],
            "supercoil": { "radius": 80, "pitch": 120, "domains": 50 },
            "proteins": [
                { "object": "mrna", "count": 150 },
                { "object": "rnap", "count": 250 },
                { "object": "dnap", "count": 24 }
            ],
        },
    });

    fs::write(&args.out_recipe, serde_json::to_string_pretty(&recipe)?)
        .with_context(|| format!("writing {}", args.out_recipe.display()))?;
    eprintln!(
        "wrote {} — {n_objects} objects ({n_interior} interior + {n_surface} surface species, \
         {skipped} skipped), meshes under {}",
        args.out_recipe.display(),
        args.out_meshes.display()
    );
    Ok(())
}

#[cfg(test)]
mod translate_tests {
    use super::*;

    #[test]
    fn parses_and_selects_cellpack_recipe() {
        // Minimal shape of a cellPACK curated recipe.
        let json = r#"{ "Compartments": [{ "Compartments": [
            {"name":"interior","IngredientGroups":[{"Ingredients":[
                {"name":"A","nbMol":10,"source":{"pdb":"1abc"}},
                {"name":"B","nbMol":50},
                {"name":"C","nbMol":30,"source":{"pdb":"computed.pdb"}}
            ]}]},
            {"name":"surface","IngredientGroups":[{"Ingredients":[
                {"name":"S1","nbMol":5}
            ]}]}
        ]}]}"#;
        let cp: CpRoot = serde_json::from_str(json).unwrap();
        let (mut interior, mut surface) = (Vec::new(), Vec::new());
        for o in &cp.compartments {
            for r in &o.compartments {
                let ings: Vec<CpIngredient> =
                    r.groups.iter().flat_map(|g| g.ingredients.iter().cloned()).collect();
                match r.name.as_str() {
                    "interior" => interior = ings,
                    "surface" => surface = ings,
                    _ => {}
                }
            }
        }
        assert_eq!(interior.len(), 3);
        assert_eq!(surface.len(), 1);
        // top-2 by abundance → B(50), C(30).
        let top = cp_select(interior, 2);
        assert_eq!(top.len(), 2);
        assert_eq!(top[0].name, "B");
        assert_eq!(top[0].nb_mol, 50);
        assert_eq!(top[1].name, "C");
    }

    #[test]
    fn relativize_sibling_dirs() {
        assert_eq!(
            relativize(
                Path::new("examples/pdb_meshes/mycoplasma/X.lod0.obj"),
                Path::new("examples/recipes"),
            ),
            "../pdb_meshes/mycoplasma/X.lod0.obj"
        );
        assert_eq!(
            relativize(Path::new("examples/genome/g.csv"), Path::new("examples/recipes")),
            "../genome/g.csv"
        );
    }

    #[test]
    fn extracts_rcsb_id_for_fetch() {
        // Bare ID, and a filename with an embedded biological-unit ID.
        assert_eq!(pdb_id_from("1abc").as_deref(), Some("1abc"));
        assert_eq!(pdb_id_from("5MG3").as_deref(), Some("5MG3"));
        assert_eq!(pdb_id_from("6rut_bu1.pdb").as_deref(), Some("6rut"));
        // cellPACK's computed names (no leading digit) stay local-only.
        assert_eq!(pdb_id_from("MG_191_MONOMER"), None);
        assert_eq!(pdb_id_from("computed.pdb"), None);
    }

    #[test]
    fn finds_structure_by_pdb_id_then_filename() {
        let dir = tempfile::tempdir().unwrap();
        let pd = dir.path();
        std::fs::write(pd.join("1ABC_BU1_.cif"), "x").unwrap();
        std::fs::write(pd.join("computed.pdb"), "x").unwrap();
        // 4-char ID prefers the biological-unit CIF.
        assert_eq!(
            find_structure_file("A", "1abc", pd).unwrap().file_name().unwrap(),
            "1ABC_BU1_.cif"
        );
        // Literal filename reference.
        assert_eq!(
            find_structure_file("C", "computed.pdb", pd).unwrap().file_name().unwrap(),
            "computed.pdb"
        );
        // Nothing matches.
        assert!(find_structure_file("Z", "9zzz", pd).is_none());
    }
}

// ───── render ────────────────────────────────────────────────────────
// Static PNG renderer for the report images. Wraps
// scripts/render_simularium.py (matplotlib via uv) so the report figures
// regenerate through the CLI rather than a loose `uv run` invocation.
#[derive(Debug, Parser)]
struct RenderArgs {
    /// Input `.simularium` file (pack with `--format simularium` first).
    input: PathBuf,

    /// Output `.png` path.
    output: PathBuf,

    /// Figure title.
    #[arg(long, default_value = "parsimony pack")]
    title: String,

    /// Cross-section axis (`x`/`y`/`z`); omit for the full volume.
    #[arg(long)]
    slice: Option<String>,

    /// Slice thickness in Å (used with `--slice`).
    #[arg(long, default_value_t = 50.0)]
    slice_thickness: f32,
}

fn run_render(args: RenderArgs) -> Result<()> {
    let script = Path::new("scripts/render_simularium.py");
    if !script.exists() {
        anyhow::bail!(
            "{} not found — run `parsimony render` from the repo root",
            script.display()
        );
    }
    ensure_uv()?;
    if let Some(parent) = args.output.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }
    let mut cmd = std::process::Command::new("uv");
    cmd.arg("run")
        .arg(script)
        .arg(&args.input)
        .arg(&args.output)
        .arg("--title")
        .arg(&args.title)
        .arg("--slice-thickness")
        .arg(args.slice_thickness.to_string());
    if let Some(s) = &args.slice {
        cmd.arg("--slice").arg(s);
    }
    let st = cmd.status().context("running render_simularium.py via uv")?;
    if !st.success() {
        anyhow::bail!("render failed ({st})");
    }
    eprintln!("wrote {}", args.output.display());
    Ok(())
}

// ───── report ────────────────────────────────────────────────────────
// Folds scripts/view_report.sh: a live grip preview by default, or a
// standalone HTML export with `--html` (pandoc, falling back to grip's
// exporter). The images live in docs/img/ and are referenced with
// relative paths, so any renderer that resolves them shows the figures.
#[derive(Debug, Parser)]
struct ReportArgs {
    /// Markdown file to render.
    #[arg(default_value = "docs/REPORT.md")]
    file: PathBuf,

    /// Export a standalone HTML file instead of opening a live preview.
    #[arg(long)]
    html: bool,

    /// Output path for `--html` (default: the input with a `.html` extension).
    #[arg(long)]
    out: Option<PathBuf>,

    /// With `--html`, open the generated file in a browser afterward.
    #[arg(long)]
    open: bool,
}

fn run_report(args: ReportArgs) -> Result<()> {
    if !args.file.exists() {
        anyhow::bail!("{} not found", args.file.display());
    }

    if !args.html {
        // Live preview via grip (GitHub-style, ephemeral uv venv).
        ensure_uv()?;
        eprintln!(
            "previewing {} via grip — open the printed URL; Ctrl-C to stop",
            args.file.display()
        );
        let st = std::process::Command::new("uv")
            .args(["tool", "run", "--from", "grip", "grip"])
            .arg(&args.file)
            .status()
            .context("running grip via uv")?;
        if !st.success() {
            anyhow::bail!("grip exited with status {st}");
        }
        return Ok(());
    }

    // HTML export. Default output is the markdown path with a .html suffix.
    let out = args
        .out
        .unwrap_or_else(|| args.file.with_extension("html"));
    let has = |bin: &str| {
        std::process::Command::new(bin)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false)
    };

    if has("pandoc") {
        eprintln!("rendering {} → {} (pandoc)", args.file.display(), out.display());
        let st = std::process::Command::new("pandoc")
            .arg(&args.file)
            .arg("-o")
            .arg(&out)
            .arg("--standalone")
            .arg("--metadata")
            .arg("title=parsimony — feature status")
            .arg("--css=https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown-light.min.css")
            .status()
            .context("running pandoc")?;
        if !st.success() {
            anyhow::bail!("pandoc exited with status {st}");
        }
    } else if has("uv") {
        eprintln!("rendering {} → {} (grip --export)", args.file.display(), out.display());
        let st = std::process::Command::new("uv")
            .args(["tool", "run", "--from", "grip", "grip"])
            .arg(&args.file)
            .arg("--export")
            .arg(&out)
            .status()
            .context("running grip --export via uv")?;
        if !st.success() {
            anyhow::bail!("grip --export exited with status {st}");
        }
    } else {
        anyhow::bail!(
            "need `pandoc` or `uv` for --html (install pandoc, or uv for grip)"
        );
    }
    eprintln!("wrote {}", out.display());

    if args.open {
        for opener in ["xdg-open", "open"] {
            if std::process::Command::new(opener).arg(&out).spawn().is_ok() {
                break;
            }
        }
    }
    Ok(())
}

// ───── pipeline ──────────────────────────────────────────────────────
// Staged packing as a small build system: a pipeline file lists stages
// (chromosome / pack-subset) with dependencies; each stage's partial
// snapshot is content-addressed and cached under <root>/.parsimony/cache.
// `run` repacks only the stages whose inputs changed (and their
// descendants) and merges everything into one pack.json the viewer reads;
// `status` shows what's fresh vs stale without packing. Recipe paths in
// the pipeline file resolve relative to the pipeline file itself.
#[derive(Debug, Parser)]
struct PipelineArgs {
    #[command(subcommand)]
    action: PipelineAction,
}

#[derive(Debug, Subcommand)]
enum PipelineAction {
    /// Pack stale stages (reuse cached ones) and write the merged pack.
    Run {
        /// Pipeline JSON file.
        file: PathBuf,
        /// Output pack path. Default: <root>/viewer/data/<name>.pack.json.
        #[arg(short, long)]
        out: Option<PathBuf>,
        /// Ignore the cache; repack every stage.
        #[arg(long)]
        force: bool,
        /// Stage cache directory. Default: <root>/.parsimony/cache.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Project root (for default cache + output locations).
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Run N iterations of a relaxation pass over the merged assembly to
        /// settle residual clashes at stage boundaries (0 = off).
        #[arg(long, default_value_t = 0)]
        relax: usize,
        /// Build mesh collision proxies from this LOD index (0 = coarsest;
        /// clamped per ingredient). Default = finest. A coarse LOD near the
        /// proxy resolution packs far faster/lighter at whole-cell scale.
        #[arg(long)]
        proxy_lod: Option<usize>,
    },
    /// Show each stage's cache key and whether it's fresh (cached) or stale.
    Status {
        /// Pipeline JSON file.
        file: PathBuf,
        /// Stage cache directory. Default: <root>/.parsimony/cache.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Project root (for the default cache location).
        #[arg(long, default_value = ".")]
        root: PathBuf,
        /// Proxy LOD to compute keys against (must match the `run` value for
        /// the displayed fresh/stale state to be accurate). Default = finest.
        #[arg(long)]
        proxy_lod: Option<usize>,
    },
    /// Empty the staged cache so the next `run` repacks every stage from
    /// scratch. (`run --force` recomputes but keeps writing into the cache;
    /// this deletes it entirely — a true "start over".)
    Clean {
        /// Pipeline JSON file. Optional and unused — accepted only so
        /// `clean` takes the same argument as `run`/`status`.
        file: Option<PathBuf>,
        /// Stage cache directory. Default: <root>/.parsimony/cache.
        #[arg(long)]
        cache_dir: Option<PathBuf>,
        /// Project root (for the default cache location).
        #[arg(long, default_value = ".")]
        root: PathBuf,
    },
}

fn run_pipeline(args: PipelineArgs) -> Result<()> {
    match args.action {
        PipelineAction::Run {
            file,
            out,
            force,
            cache_dir,
            root,
            relax,
            proxy_lod,
        } => {
            let pipeline = Pipeline::load(&file)
                .with_context(|| format!("loading pipeline `{}`", file.display()))?;
            let base_dir = file.parent().unwrap_or_else(|| Path::new("."));
            let cache_dir = cache_dir.unwrap_or_else(|| root.join(".parsimony/cache"));

            let t = Instant::now();
            let mut run = pipeline
                .run(base_dir, &cache_dir, force, proxy_lod)
                .with_context(|| format!("running pipeline `{}`", pipeline.name))?;
            let elapsed = t.elapsed();

            let (packed, cached) = run
                .reports
                .iter()
                .fold((0, 0), |(p, c), r| if r.from_cache { (p, c + 1) } else { (p + 1, c) });
            eprintln!(
                "pipeline `{}`: {} packed, {} cached in {:.2?}",
                pipeline.name, packed, cached, elapsed
            );
            for r in &run.reports {
                let tag = if r.from_cache { "cached" } else { "packed" };
                let chr = if r.chromosome { " +chromosome" } else { "" };
                eprintln!(
                    "  {tag:<6} {:<14} {:>6} placed  {:<22}{chr}  [{}]",
                    r.id, r.placed, r.kind, r.cache_key
                );
            }

            if relax > 0 {
                let rs = parsimony_core::relax(&mut run.merged, &run.recipe, relax);
                eprintln!(
                    "relax: {} iters, {} movable, {} proxy spheres — clashes {}→{}, max penetration {:.1}→{:.1} Å",
                    rs.iterations,
                    rs.movable,
                    rs.proxy_spheres,
                    rs.clashes_before,
                    rs.clashes_after,
                    rs.max_penetration_before,
                    rs.max_penetration_after,
                );
            }

            let out_path = out.unwrap_or_else(|| {
                root.join("viewer/data")
                    .join(format!("{}.pack.json", pipeline.name))
            });
            if let Some(parent) = out_path.parent() {
                fs::create_dir_all(parent)?;
            }
            let value = write_pack_json(&run.merged, &run.recipe);
            fs::write(&out_path, serde_json::to_string_pretty(&value)?)
                .with_context(|| format!("writing {}", out_path.display()))?;
            eprintln!(
                "merged {} placements{} → {}",
                run.merged.placements.len(),
                if run.merged.chromosome.is_some() { " + chromosome" } else { "" },
                out_path.display()
            );
            Ok(())
        }
        PipelineAction::Status {
            file,
            cache_dir,
            root,
            proxy_lod,
        } => {
            let pipeline = Pipeline::load(&file)
                .with_context(|| format!("loading pipeline `{}`", file.display()))?;
            let base_dir = file.parent().unwrap_or_else(|| Path::new("."));
            let cache_dir = cache_dir.unwrap_or_else(|| root.join(".parsimony/cache"));
            let plans = pipeline.plan(base_dir, &cache_dir, proxy_lod)?;

            println!(
                "pipeline `{}` — {} stages (cache: {})",
                pipeline.name,
                plans.len(),
                cache_dir.display()
            );
            println!("{:<14} {:<6} {:<18} {:<22} {}", "stage", "state", "key", "kind", "depends_on");
            for s in &plans {
                let state = if s.cached { "fresh" } else { "stale" };
                println!(
                    "{:<14} {state:<6} {:<18} {:<22} {}",
                    s.id,
                    s.cache_key,
                    s.kind,
                    s.depends_on.join(",")
                );
            }
            Ok(())
        }
        PipelineAction::Clean {
            file: _,
            cache_dir,
            root,
        } => {
            let cache_dir = cache_dir.unwrap_or_else(|| root.join(".parsimony/cache"));
            if cache_dir.exists() {
                fs::remove_dir_all(&cache_dir)
                    .with_context(|| format!("removing {}", cache_dir.display()))?;
                eprintln!("pipeline: cleared cache {}", cache_dir.display());
            } else {
                eprintln!("pipeline: cache already empty ({})", cache_dir.display());
            }
            Ok(())
        }
    }
}
