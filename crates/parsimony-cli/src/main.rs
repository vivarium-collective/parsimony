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
    write_pack_json, write_simularium_json, write_transforms_json, GreedyRandomPlacer, Pipeline,
    PlacerConfig, PlacerOutcome, Recipe,
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
    /// Pack-and-serve the local three.js viewer (supersedes view_pack.sh).
    Viewer(ViewerArgs),
    /// Pack a recipe and compare it against cellPACK (validation).
    Compare(CompareArgs),
    /// Generate mesh LODs from PDB/CIF structures (PDB → SDF → OBJ LODs).
    Mesh(MeshArgs),
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
        Command::Pipeline(args) => run_pipeline(args),
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
        ..PlacerConfig::default()
    };
    let t = Instant::now();
    let placer = GreedyRandomPlacer::new(&recipe, placer_config);
    let out = placer.pack(args.seed);
    let elapsed = t.elapsed();

    if !args.quiet {
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

/// Load + pack one recipe. Shared by `pack` and `demos`.
fn pack_recipe(
    recipe_path: &Path,
    seed: u64,
    loose_bounds: bool,
) -> Result<(Recipe, PlacerOutcome)> {
    let recipe = Recipe::from_file(recipe_path)
        .with_context(|| format!("loading recipe `{}`", recipe_path.display()))?;
    let config = PlacerConfig {
        strict_bounds: !loose_bounds,
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
                match pack_recipe(&recipe_path, args.seed, d.loose_bounds) {
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
// project root over Python's stdlib http.server (no extra deps) so the
// viewer can fetch root-relative mesh URLs like /examples/pdb_meshes/…,
// then opens a browser. Supersedes scripts/view_pack.sh; mesh
// (re)generation is `parsimony mesh …`, kept out of this path.
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
        let (recipe_doc, out) = pack_recipe(&recipe_path, args.seed, args.loose_bounds)?;
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

    // Parsimony — packed in-process (no subprocess, no hardcoded binary).
    eprintln!("packing parsimony…");
    let t = Instant::now();
    let (recipe, out) = pack_recipe(&args.recipe, args.seed, !args.strict_bounds)?;
    let psy_elapsed = t.elapsed();
    let psy_path = args.out_dir.join("parsimony.simularium");
    let sim = write_simularium_json(&out.snapshot, &recipe);
    fs::write(&psy_path, serde_json::to_string(&sim)?)
        .with_context(|| format!("writing {}", psy_path.display()))?;
    let psy_doc = SimulariumDoc::from_json(&fs::read_to_string(&psy_path)?, "parsimony")?;
    eprintln!("  {} placements in {:.2?}", psy_doc.agents.len(), psy_elapsed);

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
    println!("{:<12} {:>12} {:>12}", "engine", "placements", "wall");
    println!("{:-<12} {:->12} {:->12}", "", "", "");
    println!("{:<12} {:>12} {:>12.2?}", "parsimony", psy_doc.agents.len(), psy_elapsed);
    if let Some((cp_doc, cp_el)) = &cp {
        println!("{:<12} {:>12} {:>12.2?}", "cellpack", cp_doc.agents.len(), cp_el);
        println!("\nper-radius counts:");
        println!("{:>10} {:>12} {:>12} {:>9}", "radius", "cellpack", "parsimony", "%diff");
        println!("{:->10} {:->12} {:->12} {:->9}", "", "", "", "");
        for r in compare_counts(cp_doc, &psy_doc) {
            println!(
                "{:>10.2} {:>12} {:>12} {:>+8.1}%",
                r.radius, r.a_count, r.b_count, r.pct_diff(),
            );
        }
    }
    let ps = distribution_stats(&psy_doc.agents);
    println!("\nposition stddev (x, y, z):");
    println!("  parsimony ({:.1}, {:.1}, {:.1})", ps.stddev[0], ps.stddev[1], ps.stddev[2]);
    if let Some((cp_doc, _)) = &cp {
        let cs = distribution_stats(&cp_doc.agents);
        println!("  cellpack  ({:.1}, {:.1}, {:.1})", cs.stddev[0], cs.stddev[1], cs.stddev[2]);
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
    if std::process::Command::new("uv").arg("--version").output().is_err() {
        anyhow::bail!(
            "`uv` not on PATH — needed for the PDB→mesh script (https://docs.astral.sh/uv/)"
        );
    }
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
    }
}
