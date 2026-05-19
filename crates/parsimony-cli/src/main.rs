//! `parsimony` CLI. Phase 2 surface:
//!
//! ```text
//! parsimony pack <recipe.json> -o <out.{simularium,json}> [--seed N] [--format ...]
//! ```

use std::path::PathBuf;
use std::time::Instant;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand, ValueEnum};

use parsimony_core::{
    write_pack_json, write_simularium_json, write_transforms_json, GreedyRandomPlacer,
    PlacerConfig, Recipe,
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
