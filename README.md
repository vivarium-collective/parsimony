# parsimony

A Rust library for packing molecular contents into cellular volumes —
a next-generation rewrite of [cellPACK](https://github.com/mesoscope/cellpack).

First target: *Escherichia coli*. Validation bar: Maritan et al. 2022
whole *Mycoplasma genitalium* model.

## Design

See [`docs/parsimony-design.md`](docs/parsimony-design.md) for the
architectural blueprint, roadmap, and open questions.

## Status

Phase 0 — design + workspace skeleton. No implementation yet.

## Workspace

- **`crates/parsimony-spatial`** — Phase 1. `SpatialIndex` + `VoxelField`
  (sparse hierarchical multiscale grid). Standalone.
- **`crates/parsimony-core`** — Phase 2. Recipe loader, ingredient,
  compartment, placement.
- **`crates/parsimony-cli`** — Phase 2. `parsimony pack <recipe>`.
- **`crates/parsimony-bench`** — Phase 2/3. Cross-language benchmark
  vs. cellPACK Python.
