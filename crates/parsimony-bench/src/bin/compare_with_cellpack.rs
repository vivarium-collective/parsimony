//! `compare-with-cellpack` — run cellPACK Python and parsimony on the
//! same recipe, parse both Simularium outputs, report a side-by-side
//! comparison.

use std::path::PathBuf;

use anyhow::Result;
use clap::Parser;

use parsimony_bench::compare::{compare_counts, distribution_stats};
use parsimony_bench::parse::SimulariumDoc;
use parsimony_bench::runner::{run_cellpack, run_parsimony, CellpackConfig, ParsimonyConfig};

#[derive(Debug, Parser)]
#[command(name = "compare-with-cellpack")]
struct Args {
    /// Recipe JSON path.
    recipe: PathBuf,

    /// Output directory; created if absent. cellpack writes its files
    /// in a nested subdirectory based on the recipe name; parsimony
    /// writes one file here.
    #[arg(short, long, default_value = "/tmp/parsimony_compare")]
    out_dir: PathBuf,

    /// RNG seed for parsimony.
    #[arg(long, default_value_t = 0)]
    seed: u64,

    /// cellpack `spacing` config. cellpack scales positions/radii by
    /// `1/spacing` in its Simularium output; we re-multiply by spacing
    /// before comparing so both engines are in world units.
    #[arg(long, default_value_t = 1)]
    spacing: u32,

    /// Skip cellpack — only run parsimony and report its numbers.
    #[arg(long)]
    skip_cellpack: bool,
}

fn main() -> Result<()> {
    let args = Args::parse();

    // Run parsimony first (it's fast).
    let psy_cfg = ParsimonyConfig {
        seed: args.seed,
        ..Default::default()
    };
    eprintln!("running parsimony…");
    let psy = run_parsimony(&args.recipe, &args.out_dir, &psy_cfg)?;
    eprintln!("  done in {:.2?}", psy.elapsed);
    eprintln!("{}", indent(&psy.stderr_tail, "  | "));
    let psy_doc_raw = std::fs::read_to_string(&psy.simularium_path)?;
    let psy_doc = SimulariumDoc::from_json(&psy_doc_raw, "parsimony")?;

    let cp_doc = if args.skip_cellpack {
        None
    } else {
        let cp_cfg = CellpackConfig {
            spacing: args.spacing,
            ..Default::default()
        };
        eprintln!("running cellpack…");
        let cp = run_cellpack(&args.recipe, &args.out_dir, &cp_cfg)?;
        eprintln!("  done in {:.2?}", cp.elapsed);
        eprintln!("{}", indent(&cp.stderr_tail, "  | "));
        let cp_doc_raw = std::fs::read_to_string(&cp.simularium_path)?;
        let mut doc = SimulariumDoc::from_json(&cp_doc_raw, "cellpack")?;
        // cellpack writes scaled by 1/spacing; rescale to world units.
        doc.rescale(args.spacing as f32);
        Some((doc, cp.elapsed))
    };

    println!();
    println!("=== Recipe: {} ===", args.recipe.display());
    println!();
    println!(
        "{:<24}    {:>12}    {:>12}",
        "engine", "placements", "wall time"
    );
    println!("{:-<24}    {:->12}    {:->12}", "", "", "");
    println!(
        "{:<24}    {:>12}    {:>12.2?}",
        "parsimony",
        psy_doc.agents.len(),
        psy.elapsed
    );
    if let Some((cp_doc, cp_elapsed)) = &cp_doc {
        println!(
            "{:<24}    {:>12}    {:>12.2?}",
            "cellpack",
            cp_doc.agents.len(),
            cp_elapsed
        );
    }
    println!();

    if let Some((cp_doc, _)) = &cp_doc {
        println!("Per-ingredient counts (matched by radius):");
        println!(
            "{:>10}    {:>14}    {:>14}    {:>10}    {:>8}",
            "radius", "cellpack", "parsimony", "diff", "%diff"
        );
        println!(
            "{:->10}    {:->14}    {:->14}    {:->10}    {:->8}",
            "", "", "", "", ""
        );
        let rows = compare_counts(cp_doc, &psy_doc);
        for r in &rows {
            let cp_label = r.a_name.as_deref().unwrap_or("—");
            let psy_label = r.b_name.as_deref().unwrap_or("—");
            println!(
                "{:>10.3}    {:>6} {:<8}    {:>6} {:<8}    {:>+10}    {:>+7.1}%",
                r.radius, r.a_count, cp_label, r.b_count, psy_label, r.diff(), r.pct_diff()
            );
        }
        let cp_total: usize = rows.iter().map(|r| r.a_count).sum();
        let psy_total: usize = rows.iter().map(|r| r.b_count).sum();
        let diff = psy_total as isize - cp_total as isize;
        let pct = if cp_total > 0 {
            100.0 * diff as f32 / cp_total as f32
        } else {
            0.0
        };
        println!(
            "{:->10}    {:->14}    {:->14}    {:->10}    {:->8}",
            "", "", "", "", ""
        );
        println!(
            "{:>10}    {:>6} {:<8}    {:>6} {:<8}    {:>+10}    {:>+7.1}%",
            "total", cp_total, "", psy_total, "", diff, pct
        );
        println!();
    }

    println!("Position distribution:");
    let psy_stats = distribution_stats(&psy_doc.agents);
    if let Some((cp_doc, _)) = &cp_doc {
        let cp_stats = distribution_stats(&cp_doc.agents);
        println!(
            "{:<10}    {:>12}    {:>10}    {:>10}    {:>10}",
            "engine", "mean (x,y,z)", "stddev x", "stddev y", "stddev z"
        );
        let fmt_mean = |s: &parsimony_bench::compare::DistributionStats| {
            format!("({:.1},{:.1},{:.1})", s.mean[0], s.mean[1], s.mean[2])
        };
        println!(
            "{:<10}    {:>12}    {:>10.2}    {:>10.2}    {:>10.2}",
            "cellpack",
            fmt_mean(&cp_stats),
            cp_stats.stddev[0],
            cp_stats.stddev[1],
            cp_stats.stddev[2]
        );
        println!(
            "{:<10}    {:>12}    {:>10.2}    {:>10.2}    {:>10.2}",
            "parsimony",
            fmt_mean(&psy_stats),
            psy_stats.stddev[0],
            psy_stats.stddev[1],
            psy_stats.stddev[2]
        );
    } else {
        println!(
            "parsimony mean=({:.1},{:.1},{:.1}) stddev=({:.2},{:.2},{:.2})",
            psy_stats.mean[0],
            psy_stats.mean[1],
            psy_stats.mean[2],
            psy_stats.stddev[0],
            psy_stats.stddev[1],
            psy_stats.stddev[2]
        );
    }
    println!();

    Ok(())
}

fn indent(s: &str, prefix: &str) -> String {
    s.lines()
        .map(|l| format!("{prefix}{l}"))
        .collect::<Vec<_>>()
        .join("\n")
}
