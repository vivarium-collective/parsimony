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
"""Translate the Maritan et al. Mycoplasma genitalium recipe (from
ccsb-scripps/MycoplasmaGenitalium) into a parsimony recipe.

For each protein species (filtered by abundance), this finds the
matching structure file in `cellPACK_Data/proteins/` (CIF or PDB),
converts it to a Van der Waals surface mesh via the same SDF +
marching-cubes routine used by pdb_to_mesh.py, and emits a
parsimony recipe JSON pointing at the produced OBJ files.

Defaults: top 30 interior species by abundance. Override with
`--top-n`. Use `--top-n 0` to convert every interior species (this
takes a while — ~500 species, ~10-20 min).

Usage:
    uv run scripts/translate_mycoplasma.py \\
        --cellpack-data /tmp/mycoplasma_repo/cellPACK_Data \\
        --out-recipe examples/recipes/mycoplasma.json \\
        --out-meshes examples/pdb_meshes/mycoplasma \\
        --top-n 30
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser, MMCIFParser
from scipy.spatial import cKDTree
from scipy.ndimage import gaussian_filter, zoom
from skimage.measure import marching_cubes


VDW_RADII = {
    "H": 1.20, "HE": 1.40,
    "LI": 1.82, "B": 1.92, "C": 1.70, "N": 1.55, "O": 1.52, "F": 1.47, "NE": 1.54,
    "NA": 2.27, "MG": 1.73, "AL": 1.84, "SI": 2.10, "P": 1.80, "S": 1.80, "CL": 1.75, "AR": 1.88,
    "K": 2.75, "CA": 2.31, "FE": 2.00, "ZN": 1.39, "CU": 1.40, "MN": 2.00,
    "BR": 1.83, "I": 1.98, "SE": 1.90,
}
DEFAULT_VDW = 1.70

# Colour palette — cycled across species.
PALETTE = [
    [0.85, 0.35, 0.35],
    [0.95, 0.62, 0.35],
    [0.95, 0.85, 0.45],
    [0.55, 0.85, 0.45],
    [0.45, 0.85, 0.65],
    [0.45, 0.75, 0.95],
    [0.55, 0.55, 0.95],
    [0.80, 0.55, 0.85],
    [0.95, 0.55, 0.75],
    [0.75, 0.55, 0.45],
]


def load_atoms(struct_path: Path) -> tuple[np.ndarray, np.ndarray] | None:
    ext = struct_path.suffix.lower()
    parser = MMCIFParser(QUIET=True) if ext in (".cif", ".mmcif") else PDBParser(QUIET=True)
    try:
        structure = parser.get_structure("s", str(struct_path))
    except Exception as exc:
        print(f"  ! parse failed for {struct_path.name}: {exc}", file=sys.stderr)
        return None
    positions, radii = [], []
    for atom in structure.get_atoms():
        elem = (atom.element or atom.get_name()[0]).strip().upper()
        positions.append(atom.coord)
        radii.append(VDW_RADII.get(elem, DEFAULT_VDW))
    if not positions:
        return None
    return np.asarray(positions, dtype=np.float32), np.asarray(radii, dtype=np.float32)


def build_meshes_all_lods(
    positions: np.ndarray, radii: np.ndarray, resolutions: list[float],
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Compute one Gaussian-smoothed SDF at the finest target voxel
    size, then run marching cubes against that single field once per
    LOD with an appropriate ``step_size`` to control vertex density.

    Every LOD therefore derives its surface from *exactly the same*
    smoothed field — they share a shape and differ only in triangle
    count. The previous per-LOD-SDF approach gave each level its own
    smoothing strength (in world units it varied with voxel size),
    so a thin tRNA arm might be preserved at one LOD and washed out
    at the next, producing the visible shape-jumping the user
    complained about as the camera zoomed across LODs.
    """
    fine_resolution = min(resolutions)
    pad = float(radii.max()) + fine_resolution * 2.0
    lo = positions.min(axis=0) - pad
    hi = positions.max(axis=0) + pad
    dims = np.ceil((hi - lo) / fine_resolution).astype(int) + 1
    xs = lo[0] + np.arange(dims[0]) * fine_resolution
    ys = lo[1] + np.arange(dims[1]) * fine_resolution
    zs = lo[2] + np.arange(dims[2]) * fine_resolution
    pts = np.stack(np.meshgrid(xs, ys, zs, indexing="ij"), axis=-1).reshape(-1, 3)
    tree = cKDTree(positions)
    dists, idxs = tree.query(pts, workers=-1)
    sdf = (dists - radii[idxs]).astype(np.float32).reshape(dims)
    # Single Gaussian smooth at fine resolution. σ pinned in world Å.
    inside_outside = np.where(sdf < 0, -1.0, 1.0).astype(np.float32)
    sigma_voxels = max(0.2, 4.0 / fine_resolution)
    field = gaussian_filter(inside_outside, sigma=sigma_voxels)

    meshes_raw: list[tuple[np.ndarray, np.ndarray]] = []
    finest_index = int(np.argmin(resolutions))
    for res in resolutions:
        # Downsample the *smoothed* field rather than the raw SDF —
        # zoom does proper linear-interpolation pooling, which (unlike
        # marching_cubes' step_size) preserves thin features by
        # averaging them into the coarser cells instead of skipping
        # over them. The resulting grid is at `res` spacing in world
        # units, and marching_cubes on it yields a mesh that's a
        # lower-fidelity-but-same-shape version of the finest LOD.
        if abs(res - fine_resolution) < 1e-6:
            target_field = field
            target_sdf = sdf
        else:
            scale = fine_resolution / res
            target_field = zoom(field, scale, order=1)
            target_sdf = zoom(sdf, scale, order=1)
        try:
            verts, faces, _, _ = marching_cubes(
                target_field, level=0.0, spacing=(res,) * 3,
            )
        except (ValueError, RuntimeError):
            verts, faces, _, _ = marching_cubes(
                target_sdf, level=0.0, spacing=(res,) * 3,
            )
        verts = verts.astype(np.float32) + lo.astype(np.float32)
        meshes_raw.append((verts, faces))

    # Centre every LOD on the finest LOD's centroid so identical
    # rotations / placements align across levels regardless of
    # per-LOD vertex sampling artefacts.
    ref_centroid = meshes_raw[finest_index][0].mean(axis=0).astype(np.float32)
    return [(v - ref_centroid, f) for v, f in meshes_raw]


def write_obj(verts: np.ndarray, faces: np.ndarray, path: Path, header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        for line in header.splitlines():
            f.write(f"# {line}\n")
        for v in verts:
            f.write(f"v {v[0]:.4f} {v[1]:.4f} {v[2]:.4f}\n")
        for face in faces:
            f.write(f"f {face[0]+1} {face[1]+1} {face[2]+1}\n")


def find_structure_file(name: str, src_pdb: str, proteins_dir: Path) -> Path | None:
    """The recipe's `source.pdb` can be either a 4-char ID (live
    in `proteins/<id>.cif`, sometimes with `_BU1_` for biological-
    unit assembly) or a literal filename like `MG_270_MONOMER_int.pdb`.
    Walk through the plausible candidates in priority order.
    """
    candidates = []
    if src_pdb:
        # 4-char PDB IDs — try biological-unit assembly first, then default.
        if len(src_pdb) == 4 and src_pdb.isalnum():
            candidates.extend([
                proteins_dir / f"{src_pdb.upper()}_BU1_.cif",
                proteins_dir / f"{src_pdb.upper()}.cif",
                proteins_dir / f"{src_pdb.lower()}.cif",
            ])
        # Literal filename references (computed structures).
        candidates.append(proteins_dir / src_pdb)
        # `something_BU1_` style without exact suffix.
        stem = Path(src_pdb).stem
        candidates.extend([
            proteins_dir / f"{stem}.cif",
            proteins_dir / f"{stem}.pdb",
            proteins_dir / f"{stem}_BU1_.cif",
        ])
    # Try the ingredient name itself.
    candidates.extend([
        proteins_dir / f"{name}.cif",
        proteins_dir / f"{name}.pdb",
    ])
    for c in candidates:
        if c.exists():
            return c
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cellpack-data", type=Path, required=True,
                    help="Path to ccsb-scripps/MycoplasmaGenitalium/cellPACK_Data")
    ap.add_argument("--recipe-json", type=Path, default=None,
                    help="Specific recipe JSON to read (defaults to mg_curated_clean_*)")
    ap.add_argument("--out-recipe", type=Path, required=True,
                    help="Output parsimony recipe path")
    ap.add_argument("--out-meshes", type=Path, required=True,
                    help="Output directory for generated OBJ meshes")
    ap.add_argument("--top-n", type=int, default=30,
                    help="Number of most-abundant interior species to include "
                         "(0 = all)")
    ap.add_argument("--resolution", type=float, default=2.0,
                    help="SDF voxel size in Å (default 2.0). Used when "
                         "--lods is empty.")
    ap.add_argument("--lods", default="",
                    help="Comma-separated voxel sizes for multi-LOD output "
                         "(coarse first, e.g. '8.0,2.5'). When set, each "
                         "species gets one OBJ per LOD level emitted as "
                         "<slug>.lod<N>.obj, and the recipe records all of "
                         "them under `mesh_lods` so the viewer can pick the "
                         "right resolution per zoom level.")
    ap.add_argument("--cell-radius", type=float, default=2000.0,
                    help="Cell sphere compartment radius in Å (M. genitalium "
                         "is ~150-200 nm diameter; default 2000 Å = 400 nm "
                         "diameter, deliberately generous so the proteins "
                         "have room)")
    ap.add_argument("--lipid-count", type=int, default=40000,
                    help="Coarse-grained bilayer lipids tiled evenly over the "
                         "cell surface (default 40000; 0 disables). Tune up for "
                         "a denser, more continuous-looking membrane.")
    args = ap.parse_args()

    if args.recipe_json is None:
        # Pick the curated recipe by default.
        cand = list((args.cellpack_data / "recipes").glob("mg_curated_clean*serialized.json"))
        if not cand:
            print("no curated recipe found", file=sys.stderr)
            return 1
        args.recipe_json = cand[0]

    print(f"reading {args.recipe_json.name}", file=sys.stderr)
    with args.recipe_json.open() as f:
        recipe = json.load(f)

    # Walk to interior + surface protein groups.
    interior_ingredients = []
    surface_ingredients = []
    for c in recipe["Compartments"][0]["Compartments"]:
        if c["name"] == "interior":
            interior_ingredients = c["IngredientGroups"][0]["Ingredients"]
        elif c["name"] == "surface":
            surface_ingredients = c["IngredientGroups"][0]["Ingredients"]

    def select(ings, label):
        ings = sorted(ings, key=lambda i: -i["nbMol"])
        if args.top_n > 0:
            ings = ings[: args.top_n]
        print(f"selected {len(ings)} {label} species "
              f"(total instances: {sum(i['nbMol'] for i in ings):,})",
              file=sys.stderr)
        return ings

    interior_selected = select(interior_ingredients, "interior")
    surface_selected = select(surface_ingredients, "surface")

    args.out_meshes.mkdir(parents=True, exist_ok=True)
    proteins_dir = args.cellpack_data / "proteins"

    objects = {}
    interior_entries = []
    surface_entries = []
    skipped = []

    lod_resolutions: list[float] = []
    if args.lods.strip():
        lod_resolutions = [float(x) for x in args.lods.split(",") if x.strip()]
    use_multi_lod = len(lod_resolutions) > 0

    def convert_one(ing, region_entries, color_idx):
        name = ing["name"]
        nbmol = ing["nbMol"]
        src_pdb = ing.get("source", {}).get("pdb", "")
        slug = name.replace(" ", "_").replace("/", "_")
        struct = find_structure_file(name, src_pdb, proteins_dir)
        if struct is None:
            skipped.append((name, src_pdb))
            return False

        # Load atoms once and reuse across LOD levels.
        if use_multi_lod:
            obj_paths = [
                args.out_meshes / f"{slug}.lod{i}.obj"
                for i in range(len(lod_resolutions))
            ]
            resolutions = lod_resolutions
        else:
            obj_paths = [args.out_meshes / f"{slug}.obj"]
            resolutions = [args.resolution]

        if all(p.exists() for p in obj_paths):
            pass  # already cached
        else:
            atoms = load_atoms(struct)
            if atoms is None:
                skipped.append((name, src_pdb))
                return False
            positions, radii = atoms
            t0 = time.time()
            try:
                meshes = build_meshes_all_lods(positions, radii, resolutions)
            except Exception as exc:
                print(f"  {slug}: mesh build failed: {exc}", file=sys.stderr)
                skipped.append((name, src_pdb))
                return False
            for obj_path, res, (verts, faces) in zip(obj_paths, resolutions, meshes):
                if obj_path.exists():
                    continue
                header = (
                    f"parsimony translate_mycoplasma: {name} (source {struct.name})\n"
                    f"atoms: {len(positions)}  resolution: {res} Å  "
                    f"verts: {len(verts)}  tris: {len(faces)}"
                )
                write_obj(verts, faces, obj_path, header)
                print(f"  {slug} @ {res} Å: {len(positions):,} atoms → "
                      f"{len(verts):,} verts",
                      file=sys.stderr)
            print(f"  {slug}: built {len(meshes)} LODs in {time.time()-t0:.1f}s",
                  file=sys.stderr)

        color = PALETTE[color_idx % len(PALETTE)]
        rel_paths = [
            str(p.resolve().relative_to(args.out_recipe.parent.resolve(), walk_up=True))
            for p in obj_paths
        ]
        spec = {
            "type": "mesh",
            "color": color,
            "proxy_voxel_size": max(min(resolutions) * 2.5, 4.0),
        }
        if use_multi_lod:
            spec["mesh_lods"] = [
                {"path": p, "voxel_size": vs}
                for p, vs in zip(rel_paths, resolutions)
            ]
        else:
            spec["mesh_path"] = rel_paths[0]
        objects[slug] = spec
        region_entries.append({"object": slug, "count": nbmol})
        return True

    print(f"\nconverting {len(interior_selected)} interior species…", file=sys.stderr)
    for idx, ing in enumerate(interior_selected):
        convert_one(ing, interior_entries, idx)

    print(f"\nconverting {len(surface_selected)} surface species…", file=sys.stderr)
    for idx, ing in enumerate(surface_selected):
        # Offset palette so surface proteins read differently from interior.
        convert_one(ing, surface_entries, idx + 5)

    # Coarse-grained bilayer-spanning lipid. Real membranes have ~10^6
    # lipids; Maritan uses atomic ones. We approximate with a multi-
    # sphere whose head/tail/tail/head beads straddle the membrane: when
    # the Surface region orients its principal_vector (+Z) along the
    # outward normal, the two heads land on the inner/outer faces and the
    # tails fill the core, so a field of these reads as a two-leaflet
    # bilayer (~60 A thick, matching the real lipid patches in the
    # Maritan data). `lipid_count` is the coarse-grained unit count.
    if args.lipid_count > 0:
        objects["lipid"] = {
            "type": "multi_sphere",
            "color": [0.96, 0.86, 0.55],
            "positions": [[0, 0, 25], [0, 0, 11], [0, 0, -11], [0, 0, -25]],
            "radii": [5.0, 3.2, 3.2, 5.0],
            "principal_vector": [0, 0, 1],
            # Even Fibonacci tiling over the surface (not collision-packed)
            # so the bilayer is dense + the pack stays O(count).
            "packing_mode": "tiled",
        }
        surface_entries.append(
            {"object": "lipid", "count": args.lipid_count}
        )
        print(f"\nadded coarse-grained lipid bilayer: {args.lipid_count:,} "
              f"bilayer-spanning surface placements", file=sys.stderr)

    # Compose the parsimony recipe — a Sphere compartment for the cell
    # carries both an interior region (cytoplasmic proteins) and a
    # surface region (membrane proteins + lipid placeholders).
    regions = {}
    if interior_entries:
        regions["interior"] = interior_entries
    if surface_entries:
        regions["surface"] = surface_entries

    parsimony_recipe = {
        "name": "mycoplasma_genitalium",
        "version": "0.1.0",
        "format_version": "2.1-parsimony",
        "description": (
            f"Mycoplasma genitalium recipe translated from "
            f"ccsb-scripps/MycoplasmaGenitalium "
            f"(Maritan et al., JMB 2022). "
            f"{len(interior_entries)} interior + {len(surface_entries)} "
            f"surface species. Each protein ingredient is a Van der "
            f"Waals surface mesh built from the corresponding PDB/CIF "
            f"by scripts/translate_mycoplasma.py. The cell is "
            f"approximated as a {args.cell_radius:.0f} Å sphere "
            f"compartment with surface lipid placements as a synthetic "
            f"bilayer."
        ),
        "bounding_box": [
            [-args.cell_radius - 50, -args.cell_radius - 50, -args.cell_radius - 50],
            [args.cell_radius + 50, args.cell_radius + 50, args.cell_radius + 50],
        ],
        "objects": objects,
        "composition": {
            "space": {"regions": {"interior": ["cell"]}},
            "cell": {
                "compartment": {
                    "kind": "sphere",
                    "center": [0, 0, 0],
                    "radius": args.cell_radius,
                },
                "regions": regions,
            },
        },
    }

    args.out_recipe.parent.mkdir(parents=True, exist_ok=True)
    with args.out_recipe.open("w") as f:
        json.dump(parsimony_recipe, f, indent=2)

    total_interior = sum(e["count"] for e in interior_entries)
    total_surface = sum(e["count"] for e in surface_entries)
    print(f"\nwrote {args.out_recipe}", file=sys.stderr)
    print(f"  {len(interior_entries)} interior species, "
          f"{total_interior:,} interior placements",
          file=sys.stderr)
    print(f"  {len(surface_entries)} surface species, "
          f"{total_surface:,} surface placements",
          file=sys.stderr)
    if skipped:
        print(f"  {len(skipped)} species skipped (no structure file found):",
              file=sys.stderr)
        for name, src in skipped[:10]:
            print(f"    - {name} (pdb={src})", file=sys.stderr)
        if len(skipped) > 10:
            print(f"    … and {len(skipped) - 10} more", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
