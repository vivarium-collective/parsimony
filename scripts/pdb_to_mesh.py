#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "biopython>=1.83",
#   "numpy>=1.25",
#   "scipy>=1.11",
#   "scikit-image>=0.22",
# ]
# ///
"""Convert a PDB structure to a Van der Waals surface mesh (OBJ).

Reads a PDB file (or fetches by 4-char ID from RCSB), computes a
signed distance field on a uniform voxel grid where the level set
SDF = 0 is the Van der Waals surface (union of atom spheres with
element-specific vdW radii), extracts that level set via marching
cubes, and writes a Wavefront OBJ.

Output is consumable by parsimony as a mesh ingredient:

    {
      "type": "mesh",
      "mesh_path": "examples/pdb_meshes/1aki.obj",
      "proxy_voxel_size": 4.0
    }

Usage (uv installed; deps auto-installed on first run):

    uv run scripts/pdb_to_mesh.py 1AKI --out examples/pdb_meshes/1aki.obj
    uv run scripts/pdb_to_mesh.py /path/to/local.pdb --out foo.obj --resolution 1.5
    uv run scripts/pdb_to_mesh.py 1AON --out groel.obj --resolution 2.0

Or run directly (the shebang hands off to `uv run`):

    ./scripts/pdb_to_mesh.py 1AKI --out examples/pdb_meshes/1aki.obj
"""

from __future__ import annotations
import argparse
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from scipy.spatial import cKDTree
from skimage.measure import marching_cubes


# Bondi (1964) Van der Waals radii in Ångström. Elements not listed
# fall back to 1.7 Å (a conservative average for organic atoms).
VDW_RADII = {
    "H":  1.20, "HE": 1.40,
    "LI": 1.82, "B":  1.92, "C":  1.70, "N":  1.55, "O":  1.52, "F":  1.47, "NE": 1.54,
    "NA": 2.27, "MG": 1.73, "AL": 1.84, "SI": 2.10, "P":  1.80, "S":  1.80, "CL": 1.75, "AR": 1.88,
    "K":  2.75, "CA": 2.31, "FE": 2.00, "ZN": 1.39, "CU": 1.40, "MN": 2.00,
    "BR": 1.83, "I":  1.98, "SE": 1.90,
}
DEFAULT_VDW = 1.70


def load_atoms(pdb_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return `(positions [N,3], radii [N])` for every atom in the file.

    Auto-selects between BioPython's PDB and mmCIF parsers by file
    extension. Hetatm + waters are included by default — the surface
    is just what's in the file.
    """
    ext = pdb_path.suffix.lower()
    if ext in (".cif", ".mmcif"):
        parser = MMCIFParser(QUIET=True)
    else:
        parser = PDBParser(QUIET=True)
    structure = parser.get_structure("s", str(pdb_path))
    positions = []
    radii = []
    for atom in structure.get_atoms():
        # `atom.element` is sometimes empty for non-standard residues;
        # fall back to the first character of the atom name.
        elem = (atom.element or atom.get_name()[0]).strip().upper()
        positions.append(atom.coord)
        radii.append(VDW_RADII.get(elem, DEFAULT_VDW))
    if not positions:
        raise SystemExit(f"no atoms parsed from {pdb_path}")
    return np.asarray(positions, dtype=np.float32), np.asarray(radii, dtype=np.float32)


def build_sdf(
    positions: np.ndarray,
    radii: np.ndarray,
    resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a 3D signed-distance grid for the union of vdW spheres.

    Returns `(sdf, origin)` where `sdf[i,j,k]` is the signed distance
    from the voxel centre to the nearest atom surface (negative
    inside, positive outside), and `origin` is the world-space
    coordinate of `sdf[0,0,0]`.
    """
    pad = float(radii.max()) + resolution * 2.0
    lo = positions.min(axis=0) - pad
    hi = positions.max(axis=0) + pad
    dims = np.ceil((hi - lo) / resolution).astype(int) + 1
    # Voxel grid centres.
    xs = lo[0] + np.arange(dims[0]) * resolution
    ys = lo[1] + np.arange(dims[1]) * resolution
    zs = lo[2] + np.arange(dims[2]) * resolution
    grid_pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    # Nearest atom per voxel; SDF = |voxel − atom| − atom_radius.
    tree = cKDTree(positions)
    dists, idxs = tree.query(grid_pts, workers=-1)
    sdf = (dists - radii[idxs]).astype(np.float32)
    return sdf.reshape(dims), lo.astype(np.float32)


def extract_mesh(
    sdf: np.ndarray, origin: np.ndarray, resolution: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Run marching cubes at the zero level set, return world-space
    vertices and triangle indices."""
    verts, faces, _, _ = marching_cubes(
        sdf,
        level=0.0,
        spacing=(resolution, resolution, resolution),
    )
    verts = verts.astype(np.float32) + origin
    return verts, faces


def center_on_origin(verts: np.ndarray) -> np.ndarray:
    """Translate the mesh so its centroid sits at the origin (makes
    the OBJ rotation-agnostic when placed by parsimony)."""
    centre = verts.mean(axis=0)
    return verts - centre


def write_obj(verts: np.ndarray, faces: np.ndarray, path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in header.splitlines():
            f.write(f"# {line}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            # OBJ vertex indices are 1-based.
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def fetch_pdb(pdb_id: str, cache_dir: Path) -> Path:
    """Download a PDB structure from RCSB if not already cached."""
    pdb_id = pdb_id.upper()
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = cache_dir / f"{pdb_id.lower()}.pdb"
    if out.exists():
        return out
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"  fetching {url}", file=sys.stderr)
    urllib.request.urlretrieve(url, out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument(
        "pdb",
        help="Path to a .pdb file, or a 4-character PDB ID (will be fetched from RCSB)",
    )
    ap.add_argument("--out", required=True, type=Path, help="Output .obj path")
    ap.add_argument(
        "--resolution", type=float, default=1.5,
        help="Voxel size in Å for the SDF grid (default: 1.5). Smaller = "
             "finer surface but more triangles. 1.0–2.0 is a good range; "
             "use 2.0+ for >10k-atom complexes.",
    )
    ap.add_argument(
        "--cache", type=Path, default=Path("examples/pdb_cache"),
        help="Where to cache downloaded PDB files",
    )
    ap.add_argument(
        "--no-center", action="store_true",
        help="Keep the mesh's original PDB coordinates instead of "
             "centring it on the origin (the default makes the OBJ "
             "rotation-agnostic when placed by parsimony).",
    )
    args = ap.parse_args()

    t0 = time.time()
    p = Path(args.pdb)
    if not p.exists() and len(args.pdb) == 4 and args.pdb.isalnum():
        p = fetch_pdb(args.pdb, args.cache)

    print(f"loading {p}", file=sys.stderr)
    positions, radii = load_atoms(p)
    print(f"  {len(positions):,} atoms, vdW radii {radii.min():.2f}–{radii.max():.2f} Å",
          file=sys.stderr)

    print(f"building SDF grid at resolution {args.resolution} Å…", file=sys.stderr)
    sdf, origin = build_sdf(positions, radii, args.resolution)
    print(f"  grid {sdf.shape[0]}×{sdf.shape[1]}×{sdf.shape[2]} "
          f"({sdf.size:,} voxels, range {sdf.min():.1f} … {sdf.max():.1f} Å)",
          file=sys.stderr)

    print("extracting surface via marching cubes…", file=sys.stderr)
    verts, faces = extract_mesh(sdf, origin, args.resolution)
    if not args.no_center:
        verts = center_on_origin(verts)
    extent = verts.max(axis=0) - verts.min(axis=0)
    print(f"  mesh: {len(verts):,} vertices, {len(faces):,} triangles "
          f"(extent {extent[0]:.1f}×{extent[1]:.1f}×{extent[2]:.1f} Å)",
          file=sys.stderr)

    header = (
        f"parsimony pdb_to_mesh: {p.name}\n"
        f"atoms: {len(positions)}  resolution: {args.resolution} Å  "
        f"vertices: {len(verts)}  triangles: {len(faces)}"
    )
    write_obj(verts, faces, args.out, header)
    elapsed = time.time() - t0
    print(f"wrote {args.out} ({len(verts):,} verts, {len(faces):,} tris) in {elapsed:.1f}s",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
