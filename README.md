# parsimony

A Rust library for packing molecular contents into cellular volumes —
a next-generation rewrite of [cellPACK](https://github.com/mesoscope/cellpack).

It reads cellPACK v2 JSON recipes, packs molecular ingredients into
compartments without overlap, and emits its own pack format (with an
optional Simularium export) plus a local three.js viewer. First target:
*Escherichia coli*. Validation bar: Maritan et al. 2022 whole
*Mycoplasma genitalium* model — which it now packs end to end.

## Two placement engines

parsimony has two interchangeable interior-placement backends, selected
with `--backend`:

- **`legacy`** (default) — cellPACK's method, done faithfully and fast:
  a dense f32 **clearance grid** + a per-directive **valid-cell list** +
  **slack-bounded jitter**. The jitter bound is sized so the grid is
  *authoritative* — no per-attempt collision check is needed, so nearly
  every iteration commits a placement. Cost scales with box **volume**.
- **`octree`** — parsimony's own engine for whole-cell recipes: a sparse
  **occupancy octree** that subdivides only near placed geometry, caches
  free volume per node (so sampling is biased toward free space), and
  checks each candidate against the tree at proxy accuracy. One tree is
  shared across every directive. Cost scales with placed **content**, not
  empty space.

Rule of thumb: `legacy` for small/sparse recipes (the grid is more
efficient there); `octree` for whole-cell recipes (it's ~5× faster and
sidesteps the grid's volume-scaled, per-directive setup).

## Performance

Single-threaded, one workstation (Linux 6.8). Full numbers and method
discussion in [`docs/REPORT.md`](docs/REPORT.md).

**vs original cellPACK** (same recipe, matched density, ~2 400× faster):

| `spheres_in_a_box` (630 spheres) | parsimony `legacy` | cellPACK |
|---|---|---|
| wall time | **13 ms** | 31.5 s |
| placed | 612/630 | 613/630 |

(`one_sphere`, live via `parsimony compare`: cellPACK's multi-second
Python startup vs parsimony's sub-millisecond pack — thousands ×.)

**legacy vs octree at whole-cell scale** (*Mycoplasma genitalium*, seed 0):

| recipe | backend | placed | pack time | peak RAM |
|---|---|---|---|---|
| top-30 species | legacy | 50,248 | 74.8 s | 1.05 GB |
| top-30 species | **octree** | **50,313** | **15.5 s** | 1.50 GB |
| full (632 species) | legacy | 59,302 | 155.5 s | 2.39 GB |
| full (632 species) | **octree** | **60,177** | **28.2 s** | 2.51 GB |

## Quickstart

```bash
# Build
cargo build --release

# Pack a recipe (choose the engine)
cargo run --release -p parsimony-cli -- pack \
    examples/recipes/blood_plasma.json --out /tmp/plasma.pack.json
cargo run --release -p parsimony-cli -- pack \
    examples/recipes/mycoplasma_full.json --backend octree --out /tmp/cell.pack.json

# Compare both backends + cellPACK on one recipe (needs ../cellpack/.venv)
cargo run --release -p parsimony-cli -- compare \
    ../cellpack/examples/recipes/v2/spheres_in_a_box.json

# Pack-and-view in the local three.js viewer (serves :8123, opens a browser)
cargo run --release -p parsimony-cli -- viewer --recipe examples/recipes/shape_zoo.json
```

### Build a whole cell from scratch

The full *Mycoplasma* cell is a **staged pipeline** (chromosome →
membrane → fiber-bound proteins → densified interior), each stage
content-addressed and cached so edits only repack what changed.

```bash
# 1. Meshes. They're committed to the repo, so usually skipped. For a true
#    from-scratch rebuild, one command clones the cellPACK data and meshes
#    every species into the recipe (needs `uv` + `git`):
cargo run --release -p parsimony-cli -- translate-mycoplasma   # --top-n 30 for the demo
#    Standalone PDB structures (e.g. blood_plasma) use the CLI mesher:
#    cargo run --release -p parsimony-cli -- mesh 1AON

# 2. Start over (empty the stage cache), then run the pipeline. `run` packs
#    each stage in dependency order, caching every one, and writes the merged
#    result to viewer/data/<name>.pack.json.
cargo run --release -p parsimony-cli -- pipeline clean  examples/pipelines/mycoplasma_full.pipeline.json
cargo run --release -p parsimony-cli -- pipeline run    examples/pipelines/mycoplasma_full.pipeline.json

# Inspect stage freshness without packing; `run --force` recomputes every
# stage in place; `run --relax N` settles clashes at stage boundaries.
cargo run --release -p parsimony-cli -- pipeline status examples/pipelines/mycoplasma_full.pipeline.json
```

See [`docs/REPORT.md`](docs/REPORT.md) for demos, the cellPACK
comparison, the full feature inventory, and how to render report
images / open the viewer.

## Workspace

- **`crates/parsimony-spatial`** — `SpatialIndex` (brute + QBVH SIMD BVH)
  and `VoxelField` (sparse hierarchical voxel grid) + mesh voxeliser.
- **`crates/parsimony-core`** — recipe loader, ingredients, compartments,
  both placement backends (`clearance_grid` + `octree`), staged
  `pipeline`, `relax`, and output emitters.
- **`crates/parsimony-cli`** — the `parsimony` binary:
  `pack` / `compare` / `pipeline` / `mesh` / `demos` / `viewer`.
- **`crates/parsimony-gpu`** — wgpu clearance-grid update (Phase 4).
- **`crates/parsimony-bench`** — cellPACK comparison harness.

## Design

See [`docs/parsimony-design.md`](docs/parsimony-design.md) for the
architectural blueprint, roadmap, and open questions.
