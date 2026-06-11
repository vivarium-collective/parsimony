# parsimony — Roadmap / Task list

**Started:** 2026-06-11 · **Status of the core:** Phases 0–3 of the
design doc are delivered; the whole *M. genitalium* cell packs end to
end via the staged octree pipeline. This document is the living backlog
for what comes **next** — the post-whole-cell feature work, much of it
re-implementing original-cellPACK ("OC") capabilities that parsimony
hasn't ported yet.

## Where project tracking lives

There was no dedicated task file before this one. Tracking was split
across:

- [`docs/parsimony-design.md`](parsimony-design.md) §14 (Roadmap) and
  §15 (Open questions) — the *original* plan, now largely **delivered**.
- [`docs/REPORT.md`](REPORT.md) "Feature inventory" and "What's still
  deferred" — the most accurate snapshot of shipped vs. missing.

**This file supersedes the scattered "deferred" notes** and is the
single front door for forward work. When a task lands, move it from here
into REPORT.md's feature inventory.

> ⚠️ The design doc is aspirational in places — treat REPORT.md + the
> code as ground truth. Example: design §5.2 says cellPACK's `partners`
> field names are "kept verbatim," but there is currently **no** partner
> handling anywhere in the code (see below).

## Status legend

| | meaning |
|---|---|
| ✅ | exists and works |
| 🟡 | **partial** — parsed/recognized in the schema but not acted on |
| ⬜ | absent — not in the code at all |
| 🧪 | test / metrics infrastructure |

## Snapshot — meeting features vs. current code

| Feature (from meeting) | Status | Evidence in code |
|---|---|---|
| Partner packing | ⬜ absent | zero `partner` mentions in `crates/` |
| Gradients — global | 🟡 recognized-but-rejected | `recipe.rs:9` (schema-aware, not resolved into the `Recipe`) |
| Gradients — mesh-dependent | ⬜ absent | depends on global gradients + mesh SDF (have the SDF) |
| Pack meshes → update gradient map | ⬜ absent | needs gradients + mesh-as-source feedback loop |
| Visual integration tests + metrics | 🟡 partial | **metrics module landed (A1 ✅)** — `parsimony metrics`; still no visual-regression (A2) or gradient/partner metrics |
| Containers ≡ ingredients (one class) | ⬜ absent | separate `ingredients` / `compartments` maps (`recipe.rs:253-254`) |
| Realtime packing animation | ⬜ absent | output is single-frame Simularium (`output.rs`); static viewer |

Adjacent OC features REPORT.md already lists as deferred, folded in here
because they're cheap and half-parsed:

| Feature | Status | Evidence |
|---|---|---|
| Weighted / priority ingredient picking | 🟡 partial | `priority: f32` parsed (`recipe.rs:244`) but placer picks uniformly (`placer.rs:315`) |
| Close-packing mode (`packing_mode: close`) | 🟡 partial | `PackingMode { Random, Tiled }` only (`recipe.rs:225`) — no `Close` |

---

## Tasks

Grouped into tracks. Within a track, items are roughly dependency-ordered.
Each task notes **Today / Work / Depends on / Fit** (Fit = my honest read
on whether it makes sense and how hard it is).

### Track A — Metrics & visual test harness 🧪  *(do early; it defines "done" for B–D)*

**A1. Quantitative pack-metrics module. ✅ done (2026-06-11)**
- *Landed:* `parsimony-core/src/metrics.rs` — a reusable, QBVH-accelerated
  module computing from any `(Snapshot, Recipe)`: exact proxy-sphere
  **overlap** count + penetration depth, **fill** (placed/requested,
  overall + per ingredient), **nearest-neighbour** distance (centre +
  surface-gap), **pair-correlation g(r)**, and a Monte-Carlo **free-space /
  void-size** estimate. Exposed as `parsimony metrics <recipe>` (human
  summary or `--json`). The geometry metrics are pure functions of
  geometry + domain, unit-tested with a brute-force overlap oracle and a
  free-space-vs-analytic-volume check (6 unit tests + 1 integration test).
  Validated: spheres_in_a_box reports occupied-fraction == analytic sphere
  volume fraction; clean packs report zero overlaps.
- *Not yet:* histogram export for plotting; loading a saved `.pack.json`
  instead of repacking (currently packs fresh from the recipe).

**A2. Visual / acceptance integration tests.**
- *Today:* `parsimony render` makes PNG slices (matplotlib via the CLI);
  no automated visual check.
- *Work:* golden-recipe tests with hard assertions, one per capability:
  *spheres-in-a-box* (overlaps == 0, ≥ X% of requested placed),
  *partner packing* (adjacency metric, see C1), *global gradient*
  (measured density profile tracks the prescribed curve, see B1),
  *mesh-dependent gradient* (density vs. distance-to-mesh matches the
  falloff, see B2). Optionally a perceptual-hash visual-regression on
  the rendered PNGs.
- *Depends on:* A1; the feature under test. *Fit:* high value, but each
  sub-test lands **with** its feature, not before. Medium.

### Track B — Gradients (density control)

**B1. Global gradients.**
- *Today:* ⬜ — the loader recognizes a `gradients` block but rejects it
  (`recipe.rs:9`); there is no gradient field on the resolved `Recipe`.
- *Work:* port OC's model — each spatial sample point carries a **weight
  per ingredient type**, and multiple gradients on one ingredient
  **multiply**. Resolve a `gradients` section into the `Recipe`; evaluate
  a weight field for linear / radial / vector gradients; bias candidate
  sampling by weight. Natural home: the octree's `sample_free` descent
  (`octree.rs`) — weight the free-volume probability — and/or a weight
  channel on the `VoxelField`. (OC stored a weight per grid point; the
  octree lets us avoid a dense grid.)
- *Depends on:* A1 (to verify the resulting density profile). *Fit:*
  core cellPACK feature, clean fit, high value. Medium.

**B2. Mesh-dependent gradients.**
- *Today:* ⬜. We already have mesh signed-distance
  (`compartment.rs` mesh `project_local_point`, plus the voxeliser).
- *Work:* a gradient kind defined by **distance to a mesh** with a
  falloff function — the paper's "pack near the nuclear envelope with
  exponential falloff." Reuse the mesh SDF to produce the per-point
  weight feeding B1's machinery.
- *Depends on:* B1. *Fit:* exactly the published use case; the SDF is
  already there, so it's mostly plumbing on top of B1. Medium.

**B3. Pack meshes, then update the gradient map *(future / speculative)*.**
- *Today:* ⬜.
- *Work:* after placing some agents (as meshes), **recompute** the
  gradient field from their placed positions so subsequent ingredients
  pack relative to them — a feedback loop, not a static prescription.
- *Depends on:* B1+B2, and benefits from E1 (a packed mesh acting as a
  source). *Fit:* sensible but the most speculative item — park it
  behind a concrete recipe that needs it. Larger.

### Track C — Neighbour relations

**C1. Partner packing.**
- *Today:* ⬜ — no partner code at all.
- *Work:* probability *p* that an instance packs near a given partner
  type; **p = 1 means it only packs touching** a partner. Two regimes,
  and they differ a lot in difficulty (the meeting notes already flag
  this):
  - **p = 1 (touching):** tractable now. The QBVH gives nearest-instance
    queries; tag instances by type (need a `uid → ingredient` map) and
    restrict candidates to the contact shell of an existing partner.
  - **0 < p < 1 ("nearby"):** the hard case — biasing toward *near* a
    partner without a dense nearest-agent grid (OC leaned on the grid
    storing the closest agent's identity). Likely a weighted acceptance
    term layered on B1's weight field, recomputed as partners are placed.
- *Depends on:* A1 (adjacency metric to validate), shares machinery with
  B1. *Fit:* classic cellPACK, real value; scope **p = 1 first**, then
  probabilistic. Medium → hard.

**C2. Weighted / priority ingredient picking.**
- *Today:* 🟡 — `priority` parsed (`recipe.rs:244`); placer picks
  uniformly over live directives (`placer.rs:315`).
- *Work:* port OC's `pickWeightedIngr` — pick proportional to
  priority/count so high-priority ingredients place first.
- *Depends on:* nothing. *Fit:* cheap, half-done, unblocks realistic
  multi-ingredient density. Small.

**C3. Close-packing mode.**
- *Today:* 🟡 — `PackingMode` has only `Random` + `Tiled`
  (`recipe.rs:225`).
- *Work:* add `Close` — pick candidates in a narrow clearance band for
  crowding-style packings (OC's `packing_mode: close`).
- *Depends on:* clearance grid / octree clearance band (both expose the
  needed distance). *Fit:* small, self-contained, complements gradients.
  Small.

### Track D — Visualization

**D1. Realtime packing animation.**
- *Today:* ⬜ — output is a single Simularium frame (`output.rs`); the
  three.js viewer is static (`viewer/viewer.js`).
- *Work:* have the placer emit a **per-attempt event stream** (candidate
  point chosen → jitter → placed | rejected), serialize it as a
  multi-frame trajectory (design §9 already anticipates multi-frame
  output), and add viewer playback/scrubbing that shows the pick→jitter
  →accept/reject loop.
- *Depends on:* a placer event hook; nothing blocks it. *Fit:* great for
  debugging, demos, teaching; mostly self-contained (placer hook +
  trajectory format + viewer). Medium.

### Track E — Architecture

**E1. Unify containers and ingredients into one class.**
- *Today:* ⬜ — separate `ingredients` and `compartments` maps on the
  `Recipe` (`recipe.rs:253-254`), distinct `Ingredient`
  (`ingredient.rs`) and `Compartment` (`compartment.rs`) types.
  (Note: the octree work already unified the *interior-packing backend*;
  this is a separate, *data-model* unification.)
- *Work:* a single entity that can be packed **and** contain others, so
  a packed mesh can become a compartment — recursive recipes, and the
  enabling primitive for B3.
- *Depends on:* nothing hard-blocks it, but it touches the recipe model,
  placer, and output — widest blast radius here. *Fit:* genuinely
  enabling long-term; **highest risk**. Recommend parking behind a
  concrete use case (e.g. B3) rather than refactoring speculatively.
  Large.

---

## Pre-existing roadmap items (not from this meeting, still open)

Carried over from design §14 / REPORT so they aren't lost:

- **GPU acceleration (Phase 4, in flight).** `parsimony-gpu` ports the
  clearance-grid update to wgpu and matches the CPU oracle; next is
  wiring it behind a `--gpu` flag and moving the per-directive
  valid-cell filter to the GPU.
- **MD-relax kernel / FFT-spectral collision** (design §8.3) — denser
  packings, GPU.
- **prism integration** (design §12) — parsimony `Snapshot` as a
  `Value` / `Process` in the bigraph runtime.
- **Additional output formats** — PDB ensemble, mmCIF, OBJ scene.

## Suggested sequencing

1. ~~**A1** (metrics) — the ruler.~~ ✅ **done** — `parsimony metrics`.
2. **C2 + C3** (priority + close mode) — small, half-parsed, quick wins.
3. **B1 → B2** (global → mesh gradients) + their **A2** acceptance tests.
4. **C1** (partner packing, p = 1 first) + its adjacency test.
5. **D1** (realtime animation) — parallelizable; depends on nothing in B/C.
6. **B3 / E1** — defer until a concrete recipe demands them.
