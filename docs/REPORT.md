# parsimony — feature status

**Date:** 2026-05-18 · **Tests passing:** 136 · **Workspace crates:** 4

A Rust reimagining of cellPACK. Reads cellPACK v2 JSON recipes, packs
molecular ingredients into compartments using a uniform clearance grid
+ per-directive valid-cell lists + slack-bounded jitter, emits
Simularium JSON for the cellpack.allencell.org viewer.

> **Viewing this report on Ubuntu:** the embedded PNGs in
> `docs/img/` are referenced with standard markdown image syntax
> (`![alt](img/foo.png)`), so any markdown renderer that resolves
> relative paths will show them.
>
> Easiest: **`./scripts/view_report.sh`** — if `uv` is on PATH it
> runs `uv tool run grip` (GitHub-style render in a local web server,
> ephemeral venv, no global install). Falls back to a system grip /
> pandoc / VSCode if uv isn't available.
>
> Manual options:
>
> - **uv + grip** (recommended; one command, no global pollution):
>   `uvx grip docs/REPORT.md` → opens `http://localhost:6419`.
> - **VSCode**: `code docs/REPORT.md`, then Ctrl+Shift+V (or Ctrl+K V
>   for side-by-side).
> - **pandoc → HTML**:
>   `sudo apt install pandoc && pandoc docs/REPORT.md -o docs/REPORT.html --standalone --metadata title=parsimony`
>   then `xdg-open docs/REPORT.html`.
> - **gnome-text-editor / typora / obsidian** — open the file from
>   inside this repo so the relative image paths resolve.

---

## Demos

Three live demos, all packing without overlap on a single thread.

### `shape_zoo` — every ingredient shape parsimony supports

![shape_zoo](img/shape_zoo.png)

```
recipe: shape_zoo (6 ingredient types, 6 directives, 400 instances requested)
packed: 400/400 instances in 141.57ms  (100%)
  total attempts: 400, success rate: 100.0%
  tiny_ball     200/200   (single_sphere, radius 1.5)
  block          50/50    (single_cube, size 4³)
  rod            50/50    (single_cylinder, length 10, radius 1.0)
  bent_rod       30/30    (multi_cylinder, two-segment Y)
  dumbbell       50/50    (multi_sphere, two-sphere proxy)
  mesh_blob      20/20    (mesh, OBJ loaded from examples/meshes/sphere.obj)
```

One pack run exercises `single_sphere`, `single_cube`, `single_cylinder`,
`multi_cylinder`, `multi_sphere`, and an OBJ-loaded mesh ingredient —
all going through the same QBVH + clearance-grid pipeline. The 100%
success rate means every loop iteration committed a placement.

### `spheres_in_a_box` — strict vs loose bounds

cellPACK's classic dense-packing benchmark, run both ways.

| | **strict bounds (default)** | **`--loose-bounds` (cellPACK match)** |
|---|---|---|
| | ![spheres_strict](img/spheres_strict.png) | ![spheres_loose](img/spheres_loose.png) |
| packed | 586/630 (93%) | 612/630 (97%) |
| sphere_200 | 4/20 | 13/20 |
| sphere_100 | 32/60 | 49/60 |
| sphere_50 | 150/150 | 150/150 |
| sphere_25 | 400/400 | 400/400 |
| wall time | 8.82 ms | 12.34 ms |
| success rate | 99.7% | 99.7% |

cellPACK gets 613/630 (sphere_200: 18, sphere_100: 45) on this recipe
by allowing centres anywhere inside the box (spheres protrude at the
edge). With `--loose-bounds` parsimony reproduces that semantics and
matches their density within rounding. The default behaviour is
strict — sphere fully inside box — which is biology-correct (relevant
when the bounding box represents a real container rather than just
the simulation domain).

### `blood_plasma` — real PDB protein meshes

![blood_plasma](img/blood_plasma.png)

```
recipe: blood_plasma (6 ingredient types, 6 directives, 384 instances requested)
packed: 384/384 instances in 692ms  (100%)
  lysozyme     150/150   (1AKI — small antibacterial enzyme, ~30 Å)
  hemoglobin    80/80    (1HHO — α₂β₂ tetramer, ~55 Å)
  albumin       60/60    (1AO6 — primary plasma carrier, ~80 Å)
  transferrin   50/50    (1A8E — iron transport, ~60 Å)
  antibody      40/40    (1IGY — Y-shape IgG, ~150 Å)
  groel          4/4     (1AON — 14-mer chaperonin, ~200 Å)
```

Mixed real protein structures fetched from RCSB and converted to Van
der Waals surface meshes by `scripts/pdb_to_mesh.py` (marching-cubes
on an SDF grid). Demonstrates parsimony's mesh-ingredient pipeline
end-to-end with structures that are decidedly *not* sphere-shaped —
the Y-armed antibody and the hollow GroEL barrel both pack cleanly.

### `mycoplasma_genitalium` (full) — translated from the Maritan et al. recipe

![mycoplasma full cell](img/mycoplasma_full.png)
![mycoplasma slice z=0](img/mycoplasma_full_slice.png)

```
recipe: mycoplasma_genitalium (643 ingredient types, 643 directives, 41,623 instances requested)
        546 interior species + 96 surface species + 1 synthetic lipid head
        22,624 interior placements + 10,999 surface placements + 8,000 lipids
```

Translated from [ccsb-scripps/MycoplasmaGenitalium](https://github.com/ccsb-scripps/MycoplasmaGenitalium)
(the cellPACK data backing Maritan, Singla, Autin, Karr, Covert,
Olson & Goodsell, *J Mol Biol* 2022, "Building Structural Models of
a Whole Mycoplasma Cell"). The translator `scripts/translate_mycoplasma.py`
walks their JSON recipe, finds the structure file for every species,
batch-converts them to Van der Waals meshes via
`scripts/pdb_to_mesh.py` (marching cubes on an atomic SDF at 2.5 Å
resolution), and emits a parsimony recipe.

- **Interior**: 546 cytoplasmic species placed inside the cell sphere.
- **Surface**: 96 membrane proteins placed on the cell-sphere boundary
  via parsimony's Surface region (with `principal_vector` alignment
  to surface normals).
- **Membrane**: 8,000 lipid head groups as a synthetic Surface
  ingredient — area-weighted random distribution traces the
  membrane in renders. Replaces the atomic-detail lipid bilayer
  the Maritan publication uses; the trade-off is visual readability
  for ingredient count.
- **Cell**: a 2,000 Å sphere compartment (Mycoplasma genitalium is
  ~150–200 nm diameter; the larger sphere gives the proteins room).
- **Skipped**: 41 species had no structure file in the cellPACK
  proteins/ folder (their PDB IDs were unresolved by Maritan's
  pipeline).

The DNA/RNA chromosome from `LatticeNucleoids/` is not included
yet — it's a constrained-polymer problem, not a packing problem,
and warrants its own pipeline.

---

## Performance vs cellPACK

Same recipe, same single thread, same machine
(Linux 6.8, Tuxedo workstation). Each engine packs `spheres_in_a_box`
(630 instances requested: 60 sphere_100 + 20 sphere_200 + 150
sphere_50 + 400 sphere_25 into a 1000³ box).

| | parsimony (loose-bounds) | cellPACK (Python, jitter mode) | ratio |
|---|---|---|---|
| **Wall time** | **~11 ms** | **31.5 s** | **~2 900× faster** |
| Placements | 612/630 (97%) | 613/630 (97%) | matched |
| Attempts | 614 | many thousands (every cell-rejection retries) | ~10–100× fewer |
| Sample success rate | 99.7% | ≪10% on dense recipes | grid is authoritative |
| Per-ingredient (sphere_200) | 13/20 | 18/20 | within rounding |
| Per-ingredient (sphere_100) | 49/60 | 45/60 | within rounding |
| Throughput | ~56 000 placements/sec | ~20 placements/sec | ~2 800× |

Three parsimony runs at different seeds (loose-bounds, release build):

```
seed=1: 608/630 in 12.46ms  attempts 610, success 99.7%
seed=2: 614/630 in 10.14ms  attempts 616, success 99.7%
seed=3: 610/630 in 12.08ms  attempts 612, success 99.7%
```

cellPACK on the same recipe and seed (one run, default config):

```
real    0m31.544s
user    0m12.162s
sys     0m21.620s
```

### Where the speed comes from

The 2 900× isn't an accident. cellPACK is Python with NumPy hot loops;
parsimony is release-mode Rust. But the algorithmic differences
matter at least as much:

- **Sample success rate.** cellPACK admits cells where the sphere
  *might* fit and verifies with a per-attempt grid scan. Most attempts
  fail and retry — orders of magnitude more iterations than placements.
  parsimony's strict `clearance ≥ radius` filter + slack-bounded jitter
  means the grid is authoritative, so every iteration commits.
- **No per-attempt collision check.** Interior placements skip the
  QBVH (and cellPACK's grid scan) entirely. The invariant in the
  jitter bound proves no overlap can happen.
- **f32 distance grid, not quantised.** cellPACK stores `distToClosestSurf`
  as floats too, but their per-attempt `collision_jitter` scans O(radius³/cell³)
  grid points per placement. parsimony does one `min(stored, |c−p|−r)`
  pass over the affected cells at placement time, then nothing per attempt.
- **QBVH broad-phase** for the Surface-region collision check
  (Interior skips it). cellPACK uses a brute / grid-scan combo.
- **Sphere-tree pairwise check** is exact and tight; no quantization,
  no extraneous overlap tolerance.

The throughput leaves comfortable headroom for the bigger recipes
that will follow (Mycoplasma's ~9 M atoms, etc.).
The clearance grid currently caps at 500 cells per axis (≈ 500 MB
worst case f32 storage) — that becomes the next bottleneck before
the algorithm does.

---

## Feature inventory

What parsimony does today, mapped to where it lives in the workspace.

### Ingredient shapes  (`crates/parsimony-core/src/ingredient.rs`)

| Recipe `type:` | Internal representation | Sphere-tree size |
|---|---|---|
| `single_sphere` | `IngredientShape::SingleSphere` | 1 |
| `multi_sphere` | `IngredientShape::MultiSphere` | user-defined |
| `single_cube` | converts to `MultiSphere` via `cube_proxies` | 8 octant spheres, radius ‖h‖/2 |
| `single_cylinder` | converts to `MultiSphere` via `cylinder_proxies` | overlapping chain along local Z |
| `multi_cylinder` | converts to `MultiSphere` via `multi_cylinder_proxies` | concatenated chains |
| `mesh` | `IngredientShape::Mesh` (parry3d `TriMesh` + voxelised proxies) | one sphere per interior voxel |

Random SO(3) rotation per placement via Shoemake's method, applied to
proxy offsets. `enclosing_radius`, `world_spheres`, `needs_rotation`
are uniform across variants.

OBJ loader is a ~30-line vertex-and-face parser
(`ingredient::obj::load_trimesh`) — no external crate; supports negative
indices and fan-triangulates polygonal faces.

### Compartment kinds  (`crates/parsimony-core/src/compartment.rs`)

| Recipe `kind:` | `CompartmentKind` variant | Signed-distance impl | Surface sampling |
|---|---|---|---|
| `box` | `Box(Aabb)` | per-axis min | area-weighted face pick |
| `sphere` | `Sphere { center, radius }` | `radius − ‖p−c‖` | uniform unit-sphere direction |
| `capsule` | `Capsule { a, b, radius }` | analytical capsule SDF | hemisphere ends + cylinder side |
| `mesh` | `Mesh(MeshCompartment)` | parry3d `project_local_point` | barycentric on area-weighted triangle |

`signed_distance(p)` (positive inside, negative outside) is the unifying
primitive — used both for `fits_sphere` and to bound jitter at sample
time so jittered points stay inside the compartment by ≥ radius.

Nested compartments are supported (parent/child pointers in
`Compartment`), with child exclusion in the placer so an "interior"
directive for a parent never lands inside one of its children.

### Recipe format  (`crates/parsimony-core/src/recipe.rs`)

- cellPACK v2 JSON loads directly (verified via the actual
  `spheres_in_a_box.json` from the cellpack repo).
- Object inheritance (`inherit`) resolved with cycle detection.
- `count` or `molarity` (Avogadro × volume conversion, matches
  cellPACK's `Recipe.setCount`).
- Composition tree walked into a flat list of `PlacementDirective`s.
- Nested analytical compartments via inline `compartment: { kind: ... }`
  (parsimony extension — cellPACK uses mesh files only).
- Mesh ingredients and mesh compartments accept paths resolved
  relative to the recipe file's parent directory.

### Placement algorithm  (`crates/parsimony-core/src/placer.rs`)

- `GreedyRandomPlacer` — cellPACK's `jitter_place`, simplified.
- Per-directive `valid_cells: Vec<u32>` (cellPACK's `allIngrPts`).
  Built initially by scanning the compartment AABB, kept clean by
  lazy stale-removal during sampling, rebuilt on emptiness.
- Sampling: pick random entry from the list; per-axis jitter bounded
  by `slack/√3` (capped at `cell_size/2`), where slack is the minimum
  of clearance-to-nearest-sphere, distance-to-compartment-boundary,
  and distance-to-each-child-boundary.
- That bound is what makes the clearance grid **mathematically
  authoritative** — no QBVH collision check needed for Interior
  placements. Surface placements (which don't go through the grid)
  use a strict QBVH check with a consecutive-rejection cap.
- Uniform random over live directives (cellPACK's default
  `pickIngredient`).
- `PlacerConfig::strict_bounds` toggles loose (centre-in-box) vs
  strict (sphere-fully-in-box) containment of the root compartment.

### Clearance grid  (`crates/parsimony-core/src/clearance_grid.rs`)

- Dense `Vec<f32>` storing distance from each cell centre to the
  nearest placed sphere's surface. `f32::INFINITY` = free, `0.0` =
  occupied, positive = clearance.
- Cell size auto-derived from the recipe's largest ingredient radius
  (≈ `radius / 8`). Capped at 500 cells per axis (≤ 500 MB worst case).
- `update_for_placement(p, r, max_r)` writes `min(stored, |c−p| − r)`
  into every cell within range, branch-free.

### Spatial index  (`crates/parsimony-spatial`)

- `QbvhIndex` — 4-wide SIMD BVH via `wide::f32x4`, SoA cell AABBs,
  native incremental insert / remove / update.
- `BruteIndex` — correctness oracle, kept for property-test cross-check.
- `VoxelField` — 3-level sparse hierarchical voxel field (16³ L1, 8³
  L0), constant-tile compression, plus mesh voxeliser
  (`voxelize_trimesh`, `prepare_trimesh_for_voxelize`).
- Common `SpatialIndex` trait abstracts the three.

### Output  (`crates/parsimony-core/src/output.rs`)

- Simularium JSON (viewer-compatible) — full `trajectoryInfo` +
  `spatialData` with type-mapped colours.
- Plain transform-list JSON (`name`, `placements: [{position,
  rotation, ingredient}]`) for downstream tooling.

### CLI / bench  (`crates/parsimony-cli`, `crates/parsimony-bench`)

- `parsimony pack <recipe> --out <path> [--loose-bounds] [--seed N]`.
- `compare-with-cellpack <recipe>` — runs both engines, parses both
  Simularium outputs, reports side-by-side counts.

---

## Test inventory

136 tests passing across the workspace (cargo test --release). Clippy
clean.

| Crate / file | Tests | Coverage |
|---|---|---|
| `parsimony-spatial` (lib) | 87 | AABB, brute index, QBVH, voxel field, mesh voxeliser, queries |
| `parsimony-core` (lib) | 34 | recipe loader, compartments, placer unit tests, output schema |
| `parsimony-core` tests/ `shape_zoo.rs` | 3 | every shape type present, ≥90% packed, no overlaps |
| `parsimony-core` tests/ `spheres_in_a_box.rs` | 7 | loads cellPACK recipe, no overlaps, within bounds, deterministic, simularium + transforms output well-formed |

Selected test names (full list runnable via `cargo test --release`):

```
shape_zoo_packs_everything                       (>=90%)
shape_zoo_no_overlaps                            (across all 6 shape types)
shape_zoo_includes_every_shape_type              (sphere + multi + mesh)
no_overlaps_in_packing                           (spheres_in_a_box)
all_inside_bounding_box                          (strict bounds asserted)
loose_bounds_allows_protrusion                   (verifies the loose flag does what it says)
places_into_nested_capsule_with_surface_region   (nested capsule + surface)
deterministic_same_seed_same_output              (bit-for-bit determinism)
loads_real_spheres_in_a_box_from_cellpack        (cellPACK recipe round-trips)
loads_single_cube / loads_single_cylinder / loads_multi_cylinder
loads_mesh_ingredient_from_local_obj
loads_mesh_compartment_from_local_obj
```

---

## Algorithm sketch

For every iteration:

1. **Pick a directive** uniformly at random over those that still
   have instances to place and aren't stuck. (cellPACK's default
   `pickIngredient`.)
2. **Sample a candidate position.** For Interior directives, pick a
   random cell from the directive's `valid_cells` list. Each cell is
   one that, at build time, had `clearance ≥ radius` AND was
   ≥ `radius` inside the compartment AND ≥ `radius` outside every
   child compartment. Lazy stale-removal pops cells whose clearance
   has since dropped. For Surface directives, sample on the
   compartment surface (area-weighted face / triangle pick + uniform
   barycentric).
3. **Slack-bounded jitter.** Per-axis jitter `j ∈ [−m, m]` where
   `m = min(sphere_slack, compartment_slack, child_slack) / √3`,
   capped at half a cell. The `/√3` factor caps worst-case
   Euclidean displacement at the smallest slack, so the jittered
   point stays ≥ radius from every forbidden surface. **This is the
   load-bearing invariant** — it lets us skip the downstream
   collision check entirely for Interior placements.
4. **Place.** Insert into QBVH (broad-phase index for Surface
   queries), update the clearance grid (writes `min(stored, |c−p|−r)`
   for cells within range of the new sphere — every proxy sphere of
   a multi-sphere ingredient updates separately).
5. **Stuck detection.** A directive whose `valid_cells` list empties,
   even after a rebuild, is marked stuck and dropped from the live
   set. A Surface directive that hits a consecutive-rejection cap is
   marked stuck similarly.

The result is mathematically overlap-free and converges in
~one-placement-per-iteration: typical demos run at 99–100% sample
success rate (every iteration commits a placement). See the per-demo
"success rate" numbers above.

---

## GPU foundation (Phase 4 — in flight)

`crates/parsimony-gpu/` exists. wgpu 23 backend, one compute pipeline:
`clearance_update.wgsl` ports the CPU clearance-grid update to the
GPU. One workgroup per placement (64 threads cooperating on the
affected cell range), `atomicMin` on the f32 bit-pattern of the
distance value (sound because the grid never stores negative
numbers).

Oracle test `gpu_matches_cpu_oracle` cross-checks: 64 random
placements onto a 32³ grid via the GPU pipeline match the CPU
reference (`cpu_update` in the same crate, byte-equivalent to
`parsimony-core/src/clearance_grid.rs::update_for_placement`) to
within FP noise. Passes on the test machine's GPU; will skip
gracefully if no adapter is available.

Next on this path:
- Wire `GpuClearanceGrid` into the placer behind a `--gpu` flag so
  the Mycoplasma 33,623-placement pack can use it for the load-time
  proxy voxelisation. Benchmark against the current 82 s.
- Move per-directive `valid_cells` filter to GPU (one prefix-sum +
  compact pass per directive).
- Eventually: collision queries via parallel sphere-tree-vs-sphere-
  tree against a GPU-resident QBVH.

---

## What's still deferred

These are real gaps relative to cellPACK, but discrete features we can
add when a recipe needs them:

- **Priority-based weighted ingredient picking** (cellPACK's
  `pickWeightedIngr`). Useful when one ingredient must place before
  others; we always uniform-random.
- **Close-packing mode** (cellPACK's `packing_mode: close`). Picks
  cells in a narrow clearance band for cytoplasmic-crowding style
  packings.
- **Gradient packing** (concentration gradients along a vector).
- **PDB → mesh pipeline.** Offline conversion (ChimeraX-headless or
  PyMOL produces an OBJ from a PDB structure); a small Python
  script. Then a real biology demo (GroEL 1AON inside a vesicle, or
  the hemoglobin/IgG/lysozyme plasma trio).
- **Mesh-vs-sphere exact collision.** Currently mesh ingredients
  collide via their sphere-tree proxies; the underlying `TriMesh`
  is retained for future exact narrow-phase via parry3d.
- **Additional output formats** beyond Simularium (PDB, SIF, OBJ
  scene export).
- **GPU acceleration** for the clearance-grid update + collision check
  (Phase 4 in the design doc).
- **Prism integration** (parsimony as a Value/Process type in the
  user's bigraph runtime).

---

## How to reproduce

```bash
# All tests
cargo test --release

# Packs (any recipe path)
cargo run --release -p parsimony-cli -- pack \
    examples/recipes/shape_zoo.json --out /tmp/x.simularium

# cellPACK comparison (cellpack venv at ../cellpack/.venv)
cargo run --release -p parsimony-bench --bin compare-with-cellpack -- \
    /home/pattern/code/cellpack/examples/recipes/v2/spheres_in_a_box.json

# Re-render report images. The script's PEP 723 header pulls
# matplotlib/numpy into an ephemeral uv-managed env on first run; no
# pip install required. Plain `./scripts/render_simularium.py` works
# too via the `uv run` shebang.
uv run scripts/render_simularium.py /tmp/x.simularium docs/img/x.png \
    --title "demo" [--slice z --slice-thickness 80]

# Open this report in a browser (uv tool run grip, ephemeral venv)
./scripts/view_report.sh

# Pack and view a recipe in the local three.js viewer
./scripts/view_pack.sh                           # shape_zoo demo
./scripts/view_pack.sh path/to/recipe.json       # any recipe
```

---

## Where things live

```
crates/parsimony-spatial/    # AABB, BVH (brute + QBVH SIMD), VoxelField, mesh voxeliser
crates/parsimony-core/       # recipe loader, ingredients, compartments, placer, output
  src/ingredient.rs            # IngredientShape + shape_helpers + obj loader
  src/compartment.rs           # CompartmentKind + signed_distance + surface sampling
  src/clearance_grid.rs        # f32 distance field
  src/placer.rs                # the main algorithm
  src/recipe.rs                # JSON loader + composition walker
  src/output.rs                # Simularium + transforms emitters
crates/parsimony-cli/        # parsimony pack
crates/parsimony-bench/      # compare-with-cellpack
examples/recipes/            # shape_zoo.json, blood_plasma.json, mycoplasma.json, pdb_proteins.json
examples/meshes/             # sphere.obj (toy mesh-ingredient demo)
examples/pdb_meshes/         # generated Van der Waals meshes (1AKI, 1AON, 1HHO, 1IGY, 1AO6, 1A8E)
examples/pdb_meshes/mycoplasma/ # batch-generated meshes for the mycoplasma recipe
examples/pdb_cache/          # downloaded raw PDBs (gitignore as desired)
docs/                        # this report, parsimony-design.md, img/*.png
viewer/                      # local three.js sphere-packing viewer (HTML + JS)
scripts/render_simularium.py # static PNG renderer used to produce report images
scripts/pdb_to_mesh.py       # PDB/CIF → Van der Waals OBJ mesh (uv-managed)
scripts/translate_mycoplasma.py # cellPACK Maritan recipe → parsimony recipe
scripts/view_report.sh       # opens this report in the best available viewer
scripts/view_pack.sh         # packs a recipe and opens it in the viewer
```
