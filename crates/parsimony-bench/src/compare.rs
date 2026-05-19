//! Compare two `SimulariumDoc`s: per-ingredient counts (matched by
//! radius), position-distribution statistics, and overall placement
//! totals.

use std::collections::BTreeMap;

use crate::parse::{Agent, SimulariumDoc};

/// One row of the comparison table — one ingredient bucket matched
/// by radius across the two engines.
#[derive(Debug, Clone)]
pub struct CountRow {
    pub radius: f32,
    pub a_count: usize,
    pub a_name: Option<String>,
    pub b_count: usize,
    pub b_name: Option<String>,
}

impl CountRow {
    pub fn diff(&self) -> isize {
        self.b_count as isize - self.a_count as isize
    }
    pub fn pct_diff(&self) -> f32 {
        if self.a_count == 0 {
            0.0
        } else {
            100.0 * (self.diff() as f32) / (self.a_count as f32)
        }
    }
}

/// Bucket agents by rounded radius. Two engines may name ingredients
/// differently (cellPACK uses composition entry names like `ext_A`;
/// we use the original ingredient names) — but the radii match for
/// any single-sphere ingredient. So we match by radius.
pub fn count_by_radius(agents: &[Agent]) -> BTreeMap<u32, (usize, String)> {
    let mut out: BTreeMap<u32, (usize, String)> = BTreeMap::new();
    for a in agents {
        let key = round_radius_key(a.radius);
        let entry = out.entry(key).or_insert((0, a.type_name.clone()));
        entry.0 += 1;
    }
    out
}

fn round_radius_key(r: f32) -> u32 {
    // Quantize to 0.01 — enough precision to distinguish typical recipe radii.
    (r * 100.0).round() as u32
}

pub fn compare_counts(a: &SimulariumDoc, b: &SimulariumDoc) -> Vec<CountRow> {
    let ac = count_by_radius(&a.agents);
    let bc = count_by_radius(&b.agents);
    let all_keys: std::collections::BTreeSet<u32> = ac.keys().chain(bc.keys()).copied().collect();
    all_keys
        .into_iter()
        .map(|k| {
            let (a_count, a_name) = ac.get(&k).cloned().map(|(c, n)| (c, Some(n))).unwrap_or((0, None));
            let (b_count, b_name) = bc.get(&k).cloned().map(|(c, n)| (c, Some(n))).unwrap_or((0, None));
            CountRow {
                radius: (k as f32) / 100.0,
                a_count,
                a_name,
                b_count,
                b_name,
            }
        })
        .collect()
}

#[derive(Debug, Clone)]
pub struct DistributionStats {
    pub n: usize,
    pub mean: [f32; 3],
    pub stddev: [f32; 3],
    pub bbox_min: [f32; 3],
    pub bbox_max: [f32; 3],
}

pub fn distribution_stats(agents: &[Agent]) -> DistributionStats {
    if agents.is_empty() {
        return DistributionStats {
            n: 0,
            mean: [0.0; 3],
            stddev: [0.0; 3],
            bbox_min: [0.0; 3],
            bbox_max: [0.0; 3],
        };
    }
    let n = agents.len();
    let mut sum = [0.0_f64; 3];
    let mut sumsq = [0.0_f64; 3];
    let mut bmin = [f32::INFINITY; 3];
    let mut bmax = [f32::NEG_INFINITY; 3];
    for a in agents {
        let xs = [a.x, a.y, a.z];
        for i in 0..3 {
            sum[i] += xs[i] as f64;
            sumsq[i] += (xs[i] as f64) * (xs[i] as f64);
            if xs[i] < bmin[i] {
                bmin[i] = xs[i];
            }
            if xs[i] > bmax[i] {
                bmax[i] = xs[i];
            }
        }
    }
    let mut mean = [0.0_f32; 3];
    let mut stddev = [0.0_f32; 3];
    for i in 0..3 {
        let m = sum[i] / n as f64;
        mean[i] = m as f32;
        let var = sumsq[i] / n as f64 - m * m;
        stddev[i] = (var.max(0.0)).sqrt() as f32;
    }
    DistributionStats {
        n,
        mean,
        stddev,
        bbox_min: bmin,
        bbox_max: bmax,
    }
}
