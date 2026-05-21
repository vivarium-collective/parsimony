//! Native Van-der-Waals surface mesher.
//!
//! Replaces the biopython + scipy + scikit-image pipeline that used to live
//! in `scripts/pdb_to_mesh.py` / `scripts/translate_mycoplasma.py`. The
//! algorithm is the same one those scripts settled on, ported to Rust:
//!
//!   1. parse atoms (PDB / mmCIF, tolerant of cellPACK's minimal/multi-model
//!      files) with their Van-der-Waals radii,
//!   2. rasterise an inside/outside field at the finest voxel size — −1 inside
//!      the union of VdW spheres, +1 outside,
//!   3. Gaussian-smooth that field in world Å (rounds the per-atom bumps into a
//!      molecular surface — exactly `scipy.ndimage.gaussian_filter` on the
//!      inside/outside field),
//!   4. for each LOD, trilinearly downsample the *same* smoothed field to that
//!      voxel size and extract the zero-isosurface with naive surface nets, so
//!      every LOD shares one shape and differs only in triangle count,
//!   5. centre every LOD on the finest LOD's centroid so placements line up
//!      across zoom levels.
//!
//! Surface nets (via `fast-surface-nets`) replaces marching cubes: it needs no
//! 256-entry tables and yields smoother surfaces from a smoothed field.

use anyhow::{Context, Result};
use fast_surface_nets::ndshape::RuntimeShape;
use fast_surface_nets::{surface_nets, SurfaceNetsBuffer};
use nalgebra::{Matrix3, UnitQuaternion, Vector3};
use std::path::Path;

/// Fallback VdW radius (Å) for elements pdbtbx can't identify — carbon.
const DEFAULT_VDW: f32 = 1.70;
/// Optional extra Gaussian blur (σ, world Å). Off by default: surface nets on
/// a true SDF is already smooth, and blurring rounds (erodes) the very convex
/// surfaces of thin molecules like DNA. (The old Python smoothed a *binary*
/// field at σ=4 Å, strong enough to hide marching-cubes staircasing but
/// eroding thin features.) Overridable via the MESH_SIGMA env var.
const SMOOTH_SIGMA_A: f32 = 0.0;

/// Surface is extracted at this many voxels *outside* the true VdW isosurface.
/// A small positive offset inflates the surface slightly: it matches the
/// Python output's mild rounding at the finest LOD, and — because it scales
/// with the LOD voxel size — keeps coarse LODs chunky (a readable cigar) where
/// a sharp 0-isosurface of a thin molecule would collapse to a sliver. Tuned
/// (vs the Python output on 1BNA) so the finest LOD matches and coarse LODs
/// stay near the true size; overridable via the MESH_ISO env var.
const ISO_OFFSET_VOXELS: f32 = 0.2;

/// Per-mesh vertex budget. A LOD that exceeds it is re-extracted at a coarser
/// voxel until it fits. 1.5 Å on a megadalton complex (a ribosome, the
/// 192-mer) is millions of verts / >100 MB of OBJ — the viewer can't stream
/// that, so it falls back to placeholders and the LOD reads as random.
/// Small/medium molecules keep their full requested resolution. Overridable
/// via the MESH_MAX_VERTS env var.
const MAX_VERTS: usize = 60_000;

/// One atom: centre (Å) and Van-der-Waals radius (Å).
pub struct Atom {
    pub pos: Vector3<f32>,
    pub radius: f32,
}

/// A triangle mesh: world-space vertices and triangle index triples.
pub type Mesh = (Vec<[f32; 3]>, Vec<[u32; 3]>);

/// Van-der-Waals radius (Å) for an element symbol; carbon as the fallback.
fn vdw_radius(elem: &str) -> f32 {
    match elem.trim().to_ascii_uppercase().as_str() {
        "H" => 1.20, "C" => 1.70, "N" => 1.55, "O" => 1.52, "S" => 1.80, "P" => 1.80,
        "F" => 1.47, "CL" => 1.75, "BR" => 1.83, "I" => 1.98, "SE" => 1.90, "B" => 1.92,
        "NA" => 2.27, "MG" => 1.73, "K" => 2.75, "CA" => 2.31, "FE" => 2.00,
        "ZN" => 1.39, "CU" => 1.40, "MN" => 2.00,
        _ => DEFAULT_VDW,
    }
}

/// Guess an element from a PDB atom-name field when the element column is
/// absent: the leading non-digit character. Sufficient for the VdW surface,
/// which is dominated by C/N/O/S/H/P (all single-letter).
fn guess_element(atom_name: &str) -> String {
    atom_name
        .trim()
        .trim_start_matches(|c: char| c.is_ascii_digit())
        .chars()
        .next()
        .map(|c| c.to_ascii_uppercase().to_string())
        .unwrap_or_default()
}

/// Tolerant fixed-column PDB ATOM/HETATM parser. cellPACK ships minimal PDBs
/// (often just x,y,z — no occupancy/B-factor/element columns), and real PDBs
/// carry header records (SEQADV, …) that strict parsers reject. We read only
/// the coordinate + element columns and ignore everything else.
fn parse_pdb_atoms(text: &str) -> Vec<Atom> {
    let mut atoms = Vec::new();
    for line in text.lines() {
        if !(line.starts_with("ATOM") || line.starts_with("HETATM")) {
            continue;
        }
        let col = |a: usize, b: usize| line.get(a..b.min(line.len())).map(str::trim);
        let num = |s: Option<&str>| s.filter(|x| !x.is_empty()).and_then(|x| x.parse::<f32>().ok());
        let (x, y, z) = match (num(col(30, 38)), num(col(38, 46)), num(col(46, 54))) {
            (Some(x), Some(y), Some(z)) => (x, y, z),
            _ => continue,
        };
        let elem = col(76, 78)
            .filter(|s| !s.is_empty() && s.chars().all(|c| c.is_ascii_alphabetic()))
            .map(str::to_string)
            .unwrap_or_else(|| guess_element(col(12, 16).unwrap_or("")));
        atoms.push(Atom { pos: Vector3::new(x, y, z), radius: vdw_radius(&elem) });
    }
    atoms
}

/// Tolerant mmCIF `_atom_site` parser: reads element + Cartesian coordinates
/// from the atom-site loop, keeping only the first model (cellPACK ships some
/// multi-model CIFs whose models don't correspond, which strict parsers
/// reject). Atom-site coordinates and element symbols are unquoted, so
/// whitespace tokenisation per row is sufficient.
fn parse_cif_atoms(text: &str) -> Vec<Atom> {
    let mut atoms = Vec::new();
    let mut lines = text.lines().peekable();
    while let Some(line) = lines.next() {
        if line.trim() != "loop_" {
            continue;
        }
        // Column tags following `loop_`.
        let mut tags: Vec<String> = Vec::new();
        while let Some(p) = lines.peek() {
            let p = p.trim();
            if p.starts_with('_') {
                tags.push(p.to_ascii_lowercase());
                lines.next();
            } else {
                break;
            }
        }
        // Find a column by trying suffixes in priority order — handles both
        // structures (`_atom_site.cartn_x`) and ligand chemical components
        // (`_chem_comp_atom.model_cartn_x`), so a fetched lipid CCD works too.
        let col = |suffixes: &[&str]| -> Option<usize> {
            suffixes.iter().find_map(|s| tags.iter().position(|t| t.ends_with(s)))
        };
        let (ix, iy, iz) = match (
            col(&[".cartn_x", ".model_cartn_x", ".pdbx_model_cartn_x_ideal"]),
            col(&[".cartn_y", ".model_cartn_y", ".pdbx_model_cartn_y_ideal"]),
            col(&[".cartn_z", ".model_cartn_z", ".pdbx_model_cartn_z_ideal"]),
        ) {
            (Some(a), Some(b), Some(c)) => (a, b, c),
            _ => continue, // not an atom loop — keep scanning
        };
        let ielem = col(&[".type_symbol"]);
        let imodel = col(&[".pdbx_pdb_model_num"]);
        let ncol = tags.len();
        let mut first_model: Option<String> = None;
        while let Some(p) = lines.peek() {
            let pt = p.trim();
            if pt.is_empty()
                || pt.starts_with('_')
                || pt.starts_with('#')
                || pt == "loop_"
                || pt.starts_with("data_")
                || pt.starts_with("save_")
            {
                break;
            }
            let row = lines.next().unwrap();
            let toks: Vec<&str> = row.split_whitespace().collect();
            if toks.len() < ncol {
                continue;
            }
            if let Some(im) = imodel {
                match &first_model {
                    None => first_model = Some(toks[im].to_string()),
                    Some(fm) if fm != toks[im] => continue, // models beyond the first
                    _ => {}
                }
            }
            let num = |i: usize| toks.get(i).and_then(|s| s.parse::<f32>().ok());
            if let (Some(x), Some(y), Some(z)) = (num(ix), num(iy), num(iz)) {
                let elem = ielem
                    .and_then(|i| toks.get(i))
                    .map(|s| s.trim_matches('"').to_string())
                    .unwrap_or_default();
                atoms.push(Atom { pos: Vector3::new(x, y, z), radius: vdw_radius(&elem) });
            }
        }
        if !atoms.is_empty() {
            break;
        }
    }
    atoms
}

/// Parse atoms from a PDB or mmCIF file (all atoms, hydrogens included — the
/// full-atom VdW envelope) via tolerant parsers that ignore the header and
/// multi-model quirks in cellPACK's structures.
pub fn load_atoms(path: &Path) -> Result<Vec<Atom>> {
    let ext = path.extension().and_then(|s| s.to_str()).unwrap_or("").to_ascii_lowercase();
    let text = std::fs::read_to_string(path).with_context(|| format!("read {}", path.display()))?;
    let atoms: Vec<Atom> = if ext == "cif" || ext == "mmcif" {
        parse_cif_atoms(&text)
    } else {
        parse_pdb_atoms(&text)
    };
    anyhow::ensure!(!atoms.is_empty(), "no atoms parsed from {}", path.display());
    if std::env::var("MESH_DEBUG").is_ok() {
        let mut lo = Vector3::repeat(f32::INFINITY);
        let mut hi = Vector3::repeat(f32::NEG_INFINITY);
        for a in &atoms {
            for k in 0..3 {
                lo[k] = lo[k].min(a.pos[k]);
                hi[k] = hi[k].max(a.pos[k]);
            }
        }
        eprintln!(
            "[mesh_debug] {}: {} atoms, bbox {:.1}×{:.1}×{:.1} Å, r∈[{:.2},{:.2}]",
            path.display(),
            atoms.len(),
            hi.x - lo.x,
            hi.y - lo.y,
            hi.z - lo.z,
            atoms.iter().map(|a| a.radius).fold(f32::INFINITY, f32::min),
            atoms.iter().map(|a| a.radius).fold(0.0, f32::max),
        );
    }
    Ok(atoms)
}

/// A scalar field on a regular grid, flat with x fastest.
struct Field {
    data: Vec<f32>,
    dims: [usize; 3],
    origin: Vector3<f32>,
    voxel: f32,
}

/// Rasterise the VdW signed-distance field at `voxel` spacing: for each grid
/// cell, min over atoms of (distance − radius) — negative inside the union of
/// VdW spheres, ~0 at the molecular surface, positive outside. Only cells
/// within `band` Å of an atom's surface get an exact value; the rest stay at
/// the positive constant `band` (so there's no cliff to make false surfaces).
/// A true SDF (not a binary inside/outside field) is what surface nets wants:
/// it places vertices at the interpolated zero-crossing, so the surface is
/// smooth and — unlike smoothing a binary field — thin features aren't eroded.
fn build_sdf(atoms: &[Atom], voxel: f32) -> Field {
    let mut lo = Vector3::repeat(f32::INFINITY);
    let mut hi = Vector3::repeat(f32::NEG_INFINITY);
    let mut rmax = 0.0f32;
    for a in atoms {
        rmax = rmax.max(a.radius);
        for k in 0..3 {
            lo[k] = lo[k].min(a.pos[k] - a.radius);
            hi[k] = hi[k].max(a.pos[k] + a.radius);
        }
    }
    let band = voxel * 6.0;
    let pad = rmax + band + voxel * 2.0;
    lo -= Vector3::repeat(pad);
    hi += Vector3::repeat(pad);
    let dims = [
        (((hi.x - lo.x) / voxel).ceil() as usize + 2).max(3),
        (((hi.y - lo.y) / voxel).ceil() as usize + 2).max(3),
        (((hi.z - lo.z) / voxel).ceil() as usize + 2).max(3),
    ];
    let mut data = vec![band; dims[0] * dims[1] * dims[2]];
    for a in atoms {
        let reach = a.radius + band;
        let gi = |c: f32, k: usize| {
            (((c - lo[k]) / voxel).floor() as isize).clamp(0, dims[k] as isize - 1) as usize
        };
        let (x0, x1) = (gi(a.pos.x - reach, 0), gi(a.pos.x + reach, 0));
        let (y0, y1) = (gi(a.pos.y - reach, 1), gi(a.pos.y + reach, 1));
        let (z0, z1) = (gi(a.pos.z - reach, 2), gi(a.pos.z + reach, 2));
        for z in z0..=z1 {
            let wz = lo.z + z as f32 * voxel;
            for y in y0..=y1 {
                let wy = lo.y + y as f32 * voxel;
                let base = (z * dims[1] + y) * dims[0];
                for x in x0..=x1 {
                    let wx = lo.x + x as f32 * voxel;
                    let d = ((wx - a.pos.x).powi(2) + (wy - a.pos.y).powi(2) + (wz - a.pos.z).powi(2))
                        .sqrt()
                        - a.radius;
                    let i = base + x;
                    if d < data[i] {
                        data[i] = d;
                    }
                }
            }
        }
    }
    Field { data, dims, origin: lo, voxel }
}

/// Separable Gaussian blur in place, σ given in voxels. Edges clamp to the
/// nearest in-grid value (the border is all +1 outside anyway).
fn gaussian_smooth(field: &mut Field, sigma_voxels: f32) {
    let sigma = sigma_voxels.max(0.2);
    let half = (3.0 * sigma).ceil() as isize;
    let mut kernel = vec![0.0f32; (2 * half + 1) as usize];
    let mut sum = 0.0;
    for (j, k) in kernel.iter_mut().enumerate() {
        let d = j as isize - half;
        *k = (-(d * d) as f32 / (2.0 * sigma * sigma)).exp();
        sum += *k;
    }
    for k in &mut kernel {
        *k /= sum;
    }
    let [nx, ny, nz] = field.dims;
    let mut tmp = vec![0.0f32; field.data.len()];
    // x
    for z in 0..nz {
        for y in 0..ny {
            let base = (z * ny + y) * nx;
            for x in 0..nx {
                let mut acc = 0.0;
                for (j, &kw) in kernel.iter().enumerate() {
                    let xx = (x as isize + j as isize - half).clamp(0, nx as isize - 1) as usize;
                    acc += kw * field.data[base + xx];
                }
                tmp[base + x] = acc;
            }
        }
    }
    // y
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut acc = 0.0;
                for (j, &kw) in kernel.iter().enumerate() {
                    let yy = (y as isize + j as isize - half).clamp(0, ny as isize - 1) as usize;
                    acc += kw * tmp[(z * ny + yy) * nx + x];
                }
                field.data[(z * ny + y) * nx + x] = acc;
            }
        }
    }
    // z
    for z in 0..nz {
        for y in 0..ny {
            for x in 0..nx {
                let mut acc = 0.0;
                for (j, &kw) in kernel.iter().enumerate() {
                    let zz = (z as isize + j as isize - half).clamp(0, nz as isize - 1) as usize;
                    acc += kw * field.data[(zz * ny + y) * nx + x];
                }
                tmp[(z * ny + y) * nx + x] = acc;
            }
        }
    }
    field.data = tmp;
}

/// Trilinearly resample `field` to voxel spacing `res` over the same world
/// extent (used to build coarser LODs from the one smoothed fine field).
fn downsample(field: &Field, res: f32) -> Field {
    let scale = field.voxel / res; // < 1 for coarser
    let [nx, ny, nz] = field.dims;
    let new = [
        (((nx - 1) as f32 * scale).floor() as usize + 1).max(3),
        (((ny - 1) as f32 * scale).floor() as usize + 1).max(3),
        (((nz - 1) as f32 * scale).floor() as usize + 1).max(3),
    ];
    let sample = |fx: f32, fy: f32, fz: f32| -> f32 {
        let x0 = (fx.floor() as isize).clamp(0, nx as isize - 1) as usize;
        let y0 = (fy.floor() as isize).clamp(0, ny as isize - 1) as usize;
        let z0 = (fz.floor() as isize).clamp(0, nz as isize - 1) as usize;
        let x1 = (x0 + 1).min(nx - 1);
        let y1 = (y0 + 1).min(ny - 1);
        let z1 = (z0 + 1).min(nz - 1);
        let tx = (fx - x0 as f32).clamp(0.0, 1.0);
        let ty = (fy - y0 as f32).clamp(0.0, 1.0);
        let tz = (fz - z0 as f32).clamp(0.0, 1.0);
        let at = |x: usize, y: usize, z: usize| field.data[(z * ny + y) * nx + x];
        let c00 = at(x0, y0, z0) * (1.0 - tx) + at(x1, y0, z0) * tx;
        let c10 = at(x0, y1, z0) * (1.0 - tx) + at(x1, y1, z0) * tx;
        let c01 = at(x0, y0, z1) * (1.0 - tx) + at(x1, y0, z1) * tx;
        let c11 = at(x0, y1, z1) * (1.0 - tx) + at(x1, y1, z1) * tx;
        let c0 = c00 * (1.0 - ty) + c10 * ty;
        let c1 = c01 * (1.0 - ty) + c11 * ty;
        c0 * (1.0 - tz) + c1 * tz
    };
    let mut data = vec![0.0f32; new[0] * new[1] * new[2]];
    for z in 0..new[2] {
        let fz = z as f32 / scale;
        for y in 0..new[1] {
            let fy = y as f32 / scale;
            let base = (z * new[1] + y) * new[0];
            for x in 0..new[0] {
                data[base + x] = sample(x as f32 / scale, fy, fz);
            }
        }
    }
    Field { data, dims: new, origin: field.origin, voxel: res }
}

/// Extract the `iso`-isosurface as a world-space triangle mesh (`iso` in the
/// field's units — Å; positive inflates the surface outward).
fn extract_surface(field: &Field, iso: f32) -> Mesh {
    let [nx, ny, nz] = field.dims;
    let shape = RuntimeShape::<u32, 3>::new([nx as u32, ny as u32, nz as u32]);
    let data: Vec<f32> = field.data.iter().map(|v| v - iso).collect();
    let mut buf = SurfaceNetsBuffer::default();
    surface_nets(
        &data,
        &shape,
        [0, 0, 0],
        [nx as u32 - 1, ny as u32 - 1, nz as u32 - 1],
        &mut buf,
    );
    let verts: Vec<[f32; 3]> = buf
        .positions
        .iter()
        .map(|p| {
            [
                field.origin.x + p[0] * field.voxel,
                field.origin.y + p[1] * field.voxel,
                field.origin.z + p[2] * field.voxel,
            ]
        })
        .collect();
    let faces: Vec<[u32; 3]> = buf.indices.chunks_exact(3).map(|c| [c[0], c[1], c[2]]).collect();
    (verts, faces)
}

/// Extract one LOD at voxel size `res` (downsampling the fine field unless
/// `res` already is the fine resolution), inflating by `iso_coef` voxels.
fn extract_lod(field: &Field, res: f32, fine: f32, iso_coef: f32) -> Mesh {
    let iso = iso_coef * res;
    if (res - fine).abs() < 1e-6 {
        extract_surface(field, iso)
    } else {
        extract_surface(&downsample(field, res), iso)
    }
}

/// Build one mesh per LOD voxel size (any order; finest is the smallest).
/// Every LOD derives from the same smoothed fine field and is centred on the
/// finest LOD's centroid.
pub fn mesh_lods(atoms: &[Atom], lods: &[f32]) -> Result<Vec<Mesh>> {
    anyhow::ensure!(!lods.is_empty(), "no LOD voxel sizes given");
    let fine = lods.iter().cloned().fold(f32::INFINITY, f32::min);
    anyhow::ensure!(fine > 1e-3, "LOD voxel size too small");
    let mut field = build_sdf(atoms, fine);
    let sigma_a = std::env::var("MESH_SIGMA")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(SMOOTH_SIGMA_A);
    if sigma_a > 1e-3 {
        gaussian_smooth(&mut field, sigma_a / fine);
    }

    let iso_coef = std::env::var("MESH_ISO")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(ISO_OFFSET_VOXELS);
    let max_verts = std::env::var("MESH_MAX_VERTS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(MAX_VERTS);
    let mut meshes: Vec<Mesh> = Vec::with_capacity(lods.len());
    for &res in lods {
        // Coarsen past the requested voxel if the mesh blows past the vertex
        // budget (a megadalton complex at 1.5 Å is millions of verts the
        // viewer can't stream); small/medium molecules keep `res`.
        let mut eff = res.max(fine);
        let mut m = extract_lod(&field, eff, fine, iso_coef);
        let mut guard = 0;
        while m.0.len() > max_verts && guard < 8 {
            eff *= 1.4;
            m = extract_lod(&field, eff, fine, iso_coef);
            guard += 1;
        }
        meshes.push(m);
    }

    // Centre on the finest LOD's centroid.
    let finest = lods
        .iter()
        .enumerate()
        .min_by(|a, b| a.1.partial_cmp(b.1).unwrap())
        .map(|(i, _)| i)
        .unwrap();
    let verts = &meshes[finest].0;
    anyhow::ensure!(!verts.is_empty(), "empty surface (no atoms enclosed?)");
    let mut c = [0.0f64; 3];
    for v in verts {
        for k in 0..3 {
            c[k] += v[k] as f64;
        }
    }
    let inv = 1.0 / verts.len() as f64;
    let centroid = [
        (c[0] * inv) as f32,
        (c[1] * inv) as f32,
        (c[2] * inv) as f32,
    ];
    for (vs, _) in &mut meshes {
        for v in vs {
            v[0] -= centroid[0];
            v[1] -= centroid[1];
            v[2] -= centroid[2];
        }
    }
    Ok(meshes)
}

/// Rotate every LOD so the molecule's principal (longest) axis lies on +Z.
/// Used for tiled *segment* meshes (dna_segment, rna_segment), whose local +Z
/// the tiler aligns to the path tangent — so the helix axis must be Z
/// regardless of how the source structure happened to be oriented. (Not used
/// for placed ingredients, whose recipe rotations assume the native frame.)
pub fn reorient_to_z(meshes: &mut [Mesh]) {
    let fi = match meshes.iter().enumerate().max_by_key(|(_, m)| m.0.len()) {
        Some((i, m)) if !m.0.is_empty() => i,
        _ => return,
    };
    // Covariance of the (already centroid-centered) finest LOD.
    let mut cov = Matrix3::zeros();
    for v in &meshes[fi].0 {
        let d = Vector3::new(v[0], v[1], v[2]);
        cov += d * d.transpose();
    }
    let eig = cov.symmetric_eigen();
    let mut imax = 0;
    for i in 1..3 {
        if eig.eigenvalues[i] > eig.eigenvalues[imax] {
            imax = i;
        }
    }
    let axis = eig.eigenvectors.column(imax).into_owned();
    let rot = UnitQuaternion::rotation_between(&axis, &Vector3::z())
        .unwrap_or_else(UnitQuaternion::identity);
    for (verts, _) in meshes.iter_mut() {
        for v in verts.iter_mut() {
            let p = rot * Vector3::new(v[0], v[1], v[2]);
            *v = [p.x, p.y, p.z];
        }
    }
}

/// Rotate + centre atoms so their principal (longest) axis lies on +Z. Used to
/// orient a fetched lipid head-tail along Z before mirroring it into a bilayer.
pub fn reorient_atoms_to_z(atoms: &mut [Atom]) {
    if atoms.is_empty() {
        return;
    }
    let mut mean = Vector3::zeros();
    for a in atoms.iter() {
        mean += a.pos;
    }
    mean /= atoms.len() as f32;
    let mut cov = Matrix3::zeros();
    for a in atoms.iter() {
        let d = a.pos - mean;
        cov += d * d.transpose();
    }
    let eig = cov.symmetric_eigen();
    let mut imax = 0;
    for i in 1..3 {
        if eig.eigenvalues[i] > eig.eigenvalues[imax] {
            imax = i;
        }
    }
    let axis = eig.eigenvectors.column(imax).into_owned();
    let rot = UnitQuaternion::rotation_between(&axis, &Vector3::z())
        .unwrap_or_else(UnitQuaternion::identity);
    for a in atoms.iter_mut() {
        a.pos = rot * (a.pos - mean);
    }
}

/// Write a triangle mesh as a Wavefront OBJ with a `#`-commented header.
pub fn write_obj(mesh: &Mesh, path: &Path, header: &str) -> Result<()> {
    use std::io::Write;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let (verts, faces) = mesh;
    let mut s = String::with_capacity(verts.len() * 24 + faces.len() * 18);
    for line in header.lines() {
        s.push_str("# ");
        s.push_str(line);
        s.push('\n');
    }
    for v in verts {
        s.push_str(&format!("v {:.4} {:.4} {:.4}\n", v[0], v[1], v[2]));
    }
    for f in faces {
        s.push_str(&format!("f {} {} {}\n", f[0] + 1, f[1] + 1, f[2] + 1));
    }
    let mut file = std::fs::File::create(path).with_context(|| format!("write {}", path.display()))?;
    file.write_all(s.as_bytes())?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meshes_a_sphere_at_known_radius() {
        // A single atom of radius 10 Å → a closed sphere of ~that radius,
        // centred on the origin (the finest-LOD centroid).
        let atoms = vec![Atom { pos: Vector3::zeros(), radius: 10.0 }];
        let meshes = mesh_lods(&atoms, &[1.0]).expect("mesh");
        let (verts, faces) = &meshes[0];
        assert!(verts.len() > 200, "expected a sphere, got {} verts", verts.len());
        assert!(!faces.is_empty());
        let mut lo = [f32::INFINITY; 3];
        let mut hi = [f32::NEG_INFINITY; 3];
        for v in verts {
            for k in 0..3 {
                lo[k] = lo[k].min(v[k]);
                hi[k] = hi[k].max(v[k]);
            }
        }
        for k in 0..3 {
            let ext = hi[k] - lo[k];
            assert!((ext - 20.0).abs() < 3.0, "axis {k} extent {ext}, expected ~20 Å");
            assert!((lo[k] + hi[k]).abs() < 2.0, "axis {k} not centred: {lo:?}..{hi:?}");
        }
    }

    #[test]
    fn lods_share_shape_and_centre() {
        // Two atoms → an elongated blob; every LOD should be non-empty and
        // share roughly the same extent/centre (all built from one field,
        // centred on the finest centroid).
        let atoms = vec![
            Atom { pos: Vector3::new(-8.0, 0.0, 0.0), radius: 6.0 },
            Atom { pos: Vector3::new(8.0, 0.0, 0.0), radius: 6.0 },
        ];
        let lods = [6.0, 3.0, 1.5];
        let meshes = mesh_lods(&atoms, &lods).expect("mesh");
        assert_eq!(meshes.len(), 3);
        for (i, (verts, _)) in meshes.iter().enumerate() {
            assert!(!verts.is_empty(), "lod {i} empty");
            let xext = verts.iter().map(|v| v[0]).fold(f32::NEG_INFINITY, f32::max)
                - verts.iter().map(|v| v[0]).fold(f32::INFINITY, f32::min);
            assert!(xext > 20.0, "lod {i} x-extent {xext} too small for a 16 Å-spanned pair");
        }
    }
}
