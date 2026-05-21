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

use parsimony_core::{
    write_pack_json, write_simularium_json, write_transforms_json, GreedyRandomPlacer,
    PlacementBackend, Pipeline, PlacerConfig, PlacerOutcome, Recipe,
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
    let recipe = Recipe::from_file(&args.recipe)
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

    // Static server rooted at the project so root-relative mesh URLs
    // resolve. Prefer the repo's no-cache server (scripts/serve.py) so
    // viewer.js edits and freshly regenerated packs show up on a normal
    // reload; fall back to the stdlib server if it's absent.
    let python = which_python()?;
    eprintln!("serving {} at {url}", root.display());
    eprintln!("(Ctrl-C to stop)");
    let serve_py = root.join("scripts/serve.py");
    let mut cmd = std::process::Command::new(&python);
    if serve_py.exists() {
        cmd.arg("scripts/serve.py").arg(args.port.to_string());
    } else {
        cmd.args(["-m", "http.server", &args.port.to_string()]);
    }
    let mut child = cmd
        .current_dir(root)
        .spawn()
        .with_context(|| format!("starting static server via {python}"))?;

    if !args.no_open {
        // Let the server bind, then best-effort open a browser.
        std::thread::sleep(std::time::Duration::from_millis(600));
        for opener in ["xdg-open", "open"] {
            if std::process::Command::new(opener).arg(&url).spawn().is_ok() {
                break;
            }
        }
    }

    let status = child.wait().context("waiting on http server")?;
    if !status.success() {
        anyhow::bail!("http server exited with status {status}");
    }
    Ok(())
}

/// First working `python3`/`python` on PATH (for the static server).
fn which_python() -> Result<String> {
    for cand in ["python3", "python"] {
        let ok = std::process::Command::new(cand)
            .arg("--version")
            .output()
            .map(|o| o.status.success())
            .unwrap_or(false);
        if ok {
            return Ok(cand.to_string());
        }
    }
    anyhow::bail!("no python3/python on PATH (needed for the viewer's static file server)")
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
// Generate mesh LODs from PDB/CIF structures. The heavy lifting (vdW
// signed-distance field + marching cubes) lives in scripts/pdb_to_mesh.py
// (scientific python deps via uv); this subcommand is the CLI front that
// drives it over a structure — or a whole directory — at each LOD voxel
// size, writing <slug>.lod<N>.obj. Run from the repo root.
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
    let script = Path::new("scripts/pdb_to_mesh.py");
    if !script.exists() {
        anyhow::bail!("{} not found — run `parsimony mesh` from the repo root", script.display());
    }
    ensure_uv()?;
    let lods: Vec<f32> = args
        .lods
        .split(',')
        .filter_map(|s| s.trim().parse().ok())
        .collect();
    if lods.is_empty() {
        anyhow::bail!("no valid --lods (expected e.g. 16,8,4,2.5)");
    }
    fs::create_dir_all(&args.out_dir)?;

    // Inputs: a directory of .pdb/.cif, a single file, or a PDB ID.
    let p = Path::new(&args.path);
    let inputs: Vec<String> = if p.is_dir() {
        let mut found: Vec<String> = fs::read_dir(p)?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|pp| {
                pp.extension()
                    .and_then(|s| s.to_str())
                    .map(|x| x.eq_ignore_ascii_case("pdb") || x.eq_ignore_ascii_case("cif"))
                    .unwrap_or(false)
            })
            .map(|pp| pp.to_string_lossy().into_owned())
            .collect();
        found.sort();
        found
    } else {
        vec![args.path.clone()] // file path or PDB ID
    };
    if inputs.is_empty() {
        anyhow::bail!("no .pdb/.cif inputs under {}", args.path);
    }

    let (mut ok, mut failed) = (0usize, 0usize);
    for input in &inputs {
        let slug = Path::new(input)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or(input.as_str())
            .to_string();
        for (i, voxel) in lods.iter().enumerate() {
            let out = args.out_dir.join(format!("{slug}.lod{i}.obj"));
            eprint!("  {slug} lod{i} (res {voxel} Å) … ");
            let st = std::process::Command::new("uv")
                .arg("run")
                .arg(script)
                .arg(input)
                .arg("--out")
                .arg(&out)
                .arg("--resolution")
                .arg(voxel.to_string())
                .status();
            match st {
                Ok(s) if s.success() => {
                    eprintln!("→ {}", out.display());
                    ok += 1;
                }
                Ok(s) => {
                    eprintln!("FAILED ({s})");
                    failed += 1;
                }
                Err(e) => {
                    eprintln!("FAILED ({e})");
                    failed += 1;
                }
            }
        }
    }
    eprintln!("mesh: {ok} OBJs generated, {failed} failed → {}", args.out_dir.display());
    if failed > 0 {
        anyhow::bail!("{failed} mesh generation(s) failed");
    }
    Ok(())
}

// ───── translate-mycoplasma ──────────────────────────────────────────
// The whole-cell Mycoplasma recipe comes from the cellPACK data repo
// (Maritan et al.). This subcommand folds the two manual steps — clone
// the data repo, then run the translator — into one command, the same
// way `mesh` wraps pdb_to_mesh.py. The translator itself (recipe walk +
// batch marching-cubes meshing) stays in scripts/translate_mycoplasma.py
// because it shares pdb_to_mesh.py's scientific-Python deps via uv. Run
// from the repo root.
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
    #[arg(long, default_value = "16,8,4,2.5")]
    lods: String,

    /// Most-abundant interior species to include (0 = all; 30 = the demo).
    #[arg(long, default_value_t = 0)]
    top_n: u32,
}

fn run_translate_mycoplasma(args: TranslateMycoplasmaArgs) -> Result<()> {
    let script = Path::new("scripts/translate_mycoplasma.py");
    if !script.exists() {
        anyhow::bail!(
            "{} not found — run `parsimony translate-mycoplasma` from the repo root",
            script.display()
        );
    }
    ensure_uv()?;

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
        if !st.success() {
            anyhow::bail!("git clone failed ({st})");
        }
    }

    fs::create_dir_all(&args.out_meshes)?;
    if let Some(parent) = args.out_recipe.parent().filter(|p| !p.as_os_str().is_empty()) {
        fs::create_dir_all(parent)?;
    }

    eprintln!(
        "translating mycoplasma (top-n {}, lods {}) → {}",
        args.top_n,
        args.lods,
        args.out_recipe.display()
    );
    let st = std::process::Command::new("uv")
        .arg("run")
        .arg(script)
        .arg("--cellpack-data")
        .arg(&data_dir)
        .arg("--out-recipe")
        .arg(&args.out_recipe)
        .arg("--out-meshes")
        .arg(&args.out_meshes)
        .arg("--lods")
        .arg(&args.lods)
        .arg("--top-n")
        .arg(args.top_n.to_string())
        .status()
        .context("running translate_mycoplasma.py via uv")?;
    if !st.success() {
        anyhow::bail!("mycoplasma translation failed ({st})");
    }
    eprintln!(
        "wrote recipe {} and meshes under {}",
        args.out_recipe.display(),
        args.out_meshes.display()
    );
    Ok(())
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
        } => {
            let pipeline = Pipeline::load(&file)
                .with_context(|| format!("loading pipeline `{}`", file.display()))?;
            let base_dir = file.parent().unwrap_or_else(|| Path::new("."));
            let cache_dir = cache_dir.unwrap_or_else(|| root.join(".parsimony/cache"));

            let t = Instant::now();
            let mut run = pipeline
                .run(base_dir, &cache_dir, force)
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
        } => {
            let pipeline = Pipeline::load(&file)
                .with_context(|| format!("loading pipeline `{}`", file.display()))?;
            let base_dir = file.parent().unwrap_or_else(|| Path::new("."));
            let cache_dir = cache_dir.unwrap_or_else(|| root.join(".parsimony/cache"));
            let plans = pipeline.plan(base_dir, &cache_dir)?;

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
