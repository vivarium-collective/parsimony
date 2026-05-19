//! Parse Simularium JSON output and extract per-instance data we need
//! for comparison: per-instance `(type_id, name, radius, position)`.
//! cellPACK writes positions scaled by `1/spacing` (a config knob) and
//! a synthetic `size` field; we re-scale into world units.

use anyhow::{anyhow, Context, Result};
use serde_json::Value;

#[derive(Debug, Clone)]
pub struct Agent {
    pub type_id: u32,
    pub type_name: String,
    pub radius: f32,
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

#[derive(Debug, Clone)]
pub struct SimulariumDoc {
    pub source: String,
    pub box_size: [f32; 3],
    pub spatial_unit: String,
    pub agents: Vec<Agent>,
}

impl SimulariumDoc {
    pub fn from_json(json_str: &str, source: &str) -> Result<Self> {
        let v: Value = serde_json::from_str(json_str)
            .with_context(|| format!("parsing simularium from {source}"))?;
        let info = v
            .get("trajectoryInfo")
            .ok_or_else(|| anyhow!("no trajectoryInfo in {source}"))?;
        let size = info
            .get("size")
            .ok_or_else(|| anyhow!("no trajectoryInfo.size in {source}"))?;
        let box_size = [
            size["x"].as_f64().unwrap_or(0.0) as f32,
            size["y"].as_f64().unwrap_or(0.0) as f32,
            size["z"].as_f64().unwrap_or(0.0) as f32,
        ];
        let spatial_unit = info
            .get("spatialUnits")
            .and_then(|u| u.get("name"))
            .and_then(|n| n.as_str())
            .unwrap_or("nm")
            .to_string();
        let type_mapping = info
            .get("typeMapping")
            .and_then(|t| t.as_object())
            .ok_or_else(|| anyhow!("no typeMapping in {source}"))?;
        let mut type_names: std::collections::HashMap<u32, String> =
            std::collections::HashMap::new();
        for (k, val) in type_mapping {
            let id: u32 = k.parse().context("non-numeric type id")?;
            let name = val
                .get("name")
                .and_then(|n| n.as_str())
                .unwrap_or("?")
                .to_string();
            type_names.insert(id, name);
        }
        let bundle = v
            .get("spatialData")
            .and_then(|s| s.get("bundleData"))
            .and_then(|b| b.as_array())
            .ok_or_else(|| anyhow!("no bundleData in {source}"))?;
        if bundle.is_empty() {
            return Err(anyhow!("empty bundleData in {source}"));
        }
        let frame = &bundle[0];
        let data = frame
            .get("data")
            .and_then(|d| d.as_array())
            .ok_or_else(|| anyhow!("no frame data in {source}"))?;

        let mut agents = Vec::new();
        let mut i = 0;
        while i < data.len() {
            let _vis = data[i].as_f64().unwrap_or(0.0);
            let _uid = data[i + 1].as_f64().unwrap_or(0.0);
            let type_id = data[i + 2].as_f64().unwrap_or(-1.0) as u32;
            let x = data[i + 3].as_f64().unwrap_or(0.0) as f32;
            let y = data[i + 4].as_f64().unwrap_or(0.0) as f32;
            let z = data[i + 5].as_f64().unwrap_or(0.0) as f32;
            let radius = data[i + 9].as_f64().unwrap_or(0.0) as f32;
            let n_sub = data[i + 10].as_f64().unwrap_or(0.0) as usize;
            let name = type_names.get(&type_id).cloned().unwrap_or_else(|| "?".into());
            agents.push(Agent {
                type_id,
                type_name: name,
                radius,
                x,
                y,
                z,
            });
            i += 11 + n_sub * 3;
        }

        Ok(SimulariumDoc {
            source: source.to_string(),
            box_size,
            spatial_unit,
            agents,
        })
    }

    /// Rescale agent positions and radii by `factor`. cellPACK's
    /// Simularium output writes everything divided by its `spacing`
    /// config knob (so the viewer shows a unit-scaled box); pass
    /// `factor = spacing` to recover world units.
    pub fn rescale(&mut self, factor: f32) {
        for a in self.agents.iter_mut() {
            a.x *= factor;
            a.y *= factor;
            a.z *= factor;
            a.radius *= factor;
        }
        for s in self.box_size.iter_mut() {
            *s *= factor;
        }
    }
}
