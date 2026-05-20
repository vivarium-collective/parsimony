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
