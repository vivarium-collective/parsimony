//! Subprocess wrappers that run cellPACK Python pack and parsimony
//! pack on the same recipe.

use std::path::{Path, PathBuf};
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{anyhow, Context, Result};

pub struct CellpackConfig {
    /// Path to cellPACK's Python interpreter (with cellpack installed).
    pub python: PathBuf,
    /// `spacing` config value cellpack divides positions/radii by in
    /// its Simularium output. Default 1.
    pub spacing: u32,
    /// `place_method` config value: "jitter" or "spheresSST".
    pub place_method: String,
}

impl Default for CellpackConfig {
    fn default() -> Self {
        Self {
            python: PathBuf::from("/home/pattern/code/cellpack/.venv/bin/python"),
            spacing: 1,
            place_method: "jitter".into(),
        }
    }
}

pub struct CellpackRun {
    pub simularium_path: PathBuf,
    pub elapsed: Duration,
    pub stderr_tail: String,
}

pub fn run_cellpack(recipe: &Path, out_dir: &Path, config: &CellpackConfig) -> Result<CellpackRun> {
    std::fs::create_dir_all(out_dir)?;
    // Write a cellpack packing-config JSON in the same dir.
    let config_path = out_dir.join("cellpack_config.json");
    let config_body = serde_json::json!({
        "name": "parsimony_compare",
        "out": out_dir.to_string_lossy(),
        "overwrite_place_method": true,
        "place_method": config.place_method,
        "format": "simularium",
        "save_analyze_result": false,
        "spacing": config.spacing,
        "number_of_packings": 1,
        "use_periodicity": false,
    });
    std::fs::write(&config_path, serde_json::to_string_pretty(&config_body)?)?;

    let t = Instant::now();
    let output = Command::new(&config.python)
        .arg("-m")
        .arg("cellpack.bin.pack")
        .arg("--recipe")
        .arg(recipe)
        .arg("--config_path")
        .arg(&config_path)
        .output()
        .with_context(|| format!("running cellpack via {}", config.python.display()))?;
    let elapsed = t.elapsed();

    if !output.status.success() {
        return Err(anyhow!(
            "cellpack pack failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }

    // cellpack writes to `<out>/<recipe_name>/<place_method>/results_*.simularium`.
    let recipe_name = recipe
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown");
    let result_dir = out_dir.join(recipe_name).join(&config.place_method);
    let mut result_files = std::fs::read_dir(&result_dir)
        .with_context(|| format!("reading cellpack result dir {}", result_dir.display()))?
        .filter_map(|e| e.ok())
        .filter(|e| {
            e.file_name()
                .to_string_lossy()
                .ends_with(".simularium")
        })
        .map(|e| e.path())
        .collect::<Vec<_>>();
    result_files.sort();
    let path = result_files
        .pop()
        .ok_or_else(|| anyhow!("no .simularium file in {}", result_dir.display()))?;

    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail: String = stderr.lines().rev().take(5).collect::<Vec<_>>().join("\n");

    Ok(CellpackRun {
        simularium_path: path,
        elapsed,
        stderr_tail: tail,
    })
}

pub struct ParsimonyConfig {
    pub binary: PathBuf,
    pub seed: u64,
    /// Pass `--loose-bounds` to parsimony. cellPACK's default root
    /// containment is loose (centre-in-box only), so an apples-to-
    /// apples density comparison needs this on.
    pub loose_bounds: bool,
}

impl Default for ParsimonyConfig {
    fn default() -> Self {
        Self {
            binary: PathBuf::from(
                "/home/pattern/code/parsimony/target/release/parsimony",
            ),
            seed: 0,
            loose_bounds: false,
        }
    }
}

pub struct ParsimonyRun {
    pub simularium_path: PathBuf,
    pub elapsed: Duration,
    pub stderr_tail: String,
}

pub fn run_parsimony(
    recipe: &Path,
    out_dir: &Path,
    config: &ParsimonyConfig,
) -> Result<ParsimonyRun> {
    std::fs::create_dir_all(out_dir)?;
    let out_path = out_dir.join("parsimony_out.simularium");
    let t = Instant::now();
    let mut cmd = Command::new(&config.binary);
    cmd.arg("pack")
        .arg(recipe)
        .arg("--out")
        .arg(&out_path)
        .arg("--seed")
        .arg(config.seed.to_string());
    if config.loose_bounds {
        cmd.arg("--loose-bounds");
    }
    let output = cmd
        .output()
        .with_context(|| format!("running {}", config.binary.display()))?;
    let elapsed = t.elapsed();

    if !output.status.success() {
        return Err(anyhow!(
            "parsimony pack failed (status {:?}):\n{}",
            output.status.code(),
            String::from_utf8_lossy(&output.stderr)
        ));
    }
    let stderr = String::from_utf8_lossy(&output.stderr);
    let tail: String = stderr.lines().rev().take(8).collect::<Vec<_>>().join("\n");
    Ok(ParsimonyRun {
        simularium_path: out_path,
        elapsed,
        stderr_tail: tail,
    })
}
