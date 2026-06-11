# parsimony — Design

A Rust library for **packing molecular contents into cellular volumes** —
a next-generation rewrite of cellPACK (Goodsell, Autin, Olson; original
Python at `mesoscope/cellpack`). First target: *Escherichia coli*. Bar
to clear: Maritan et al. 2022's whole *M. genitalium* model (J Mol Biol
434:167351), built with CellPACKgpu + NVIDIA Flex.

This document is the architectural blueprint. The roadmap at the bottom
sequences implementation. Decisions are committed unless flagged
**(open)**.

---

## 1. Scope

**In scope (v0.1).** Static packed snapshot of one cellular volume from
a recipe: load → place ingredients → write a viewable result. Tested
against small cellPACK Python recipes for correctness.

**In scope (later versions).** Sparse hierarchical multiscale spatial
index. GPU broad-phase collision (wgpu). Flex-style rigid-body
relaxation. Warm-started incremental packing. E. coli proteome ingestion
pipeline (EcoCyc + PDB + AlphaFold + OPM). Adapter for prism so packed
configurations become bigraph values.

**Out of scope (forever, probably).** Atomic-resolution MD. Reaction
kinetics. Diffusion. Active transport. Cell-cycle dynamics. These belong
upstream (in prism) — parsimony is a *spatial configuration store with a
packer*, not a simulator. We expose the right APIs so prism processes
can drive temporal change; the dynamics live there.

## 2. The problem

A **recipe** specifies (a) a bounded volume — the cell — possibly with
nested compartments (membranes, organelles, nucleoid); (b) a set of
**ingredients** (proteins, lipid complexes, nucleic acids) with shape
representations, counts (or molarities), and placement constraints
(cytoplasm vs. membrane vs. interior of a specific compartment); (c)
optional gradients biasing density. A **packing** is a set of rigid-body
**placements** (per-instance translation + rotation) such that no two
ingredients overlap *and* each respects its compartment constraint.

The general 3D irregular-packing problem is NP-hard, but biology
practically isn't bin-packing — cytoplasm typical volume fraction is
30–40%, and counts come from concentrations with stochastic tolerance.
Greedy random placement plus local relaxation gets us within engineering
tolerance, which is exactly what every working tool (cellPACK,
CellPACKgpu, the Autin pipeline) actually does. We keep that pattern;
the room for improvement is *speed*, *scale*, and *honest
multi-resolution* — not the algorithmic core.

## 3. Design principles

1. **Compose primitives; don't fuse them.** The original cellPACK has a
   single regular grid that does three jobs poorly: candidate-point
   pool, free-space distance field, and compartment classifier. We
   split those: a **spatial index** over instances, a **voxel field**
   for occupancy/compartment, and a **placement strategy** that
   *consults* both but isn't tied to either.
2. **Snapshots are values.** A packed state is `(Vec<Placement>,
   compartment refs, recipe ref, RNG seed)` — pure data, serde-able. No
   runtime state hidden in the spatial index.
3. **Spatial structures behind traits.** `SpatialIndex` and `VoxelField`
   are interfaces. We can swap simple-BVH → sparse-hierarchical →
   GPU-BVH without touching the placement loop.
4. **CPU first, GPU when it pays.** Three places GPU genuinely earns
   its keep (broad-phase, relaxation, FFT-spectral collision for dense
   sub-regions). Everything else stays CPU. Use **wgpu** so we are
   portable across Apple Silicon, NVIDIA, AMD, Intel.
5. **Dynamics-ready by construction.** Every operation has an
   incremental form (`insert`, `remove`, `move`, `resolve_near`). Even
   in v0.1, we *never* assume "build once, query forever".
6. **No global state, no implicit threading.** Every operation threads
   through an explicit `Env`. Multi-threading is opt-in via Rayon at
   call sites we choose.
7. **Reuse cellPACK's recipe schema (v2.1) verbatim where it works.**
   Extend cautiously, with explicit `format_version` bumps. The Allen
   Institute already publishes example recipes against this schema; we
   inherit that ecosystem for free.

## 4. Workspace layout

```
parsimony/
├── Cargo.toml                      # workspace
├── README.md
├── docs/
│   └── parsimony-design.md         # this document
├── crates/
│   ├── parsimony-spatial/          # Phase 1: SpatialIndex + VoxelField (standalone)
│   ├── parsimony-core/             # Phase 2: recipe, ingredient, compartment, placement
│   ├── parsimony-cli/              # Phase 2: `parsimony pack ...`
│   └── parsimony-bench/            # Phase 2/3: cross-language benchmark vs cellPACK Python
└── examples/
    └── recipes/                    # mirrored / small test recipes
```

Rust edition 2024, matching prism. Workspace dependencies declared once
in the root `Cargo.toml`.

**Crate-level dependencies (target set).**

- Math + collision: `parry3d` (sphere-sphere, ray-mesh, point-in-mesh
  via nalgebra). Same numerics family as prism's `rapier2d`.
- Serde: `serde`, `serde_json`, `serde_with` for the recipe schema.
- RNG: `rand`, `rand_xoshiro` (deterministic, seekable streams).
- Errors: `thiserror` in libs, `anyhow` in the CLI.
- CLI: `clap` (derive).
- Logging: `tracing`.
- Parallelism (opt-in, later): `rayon`.
- GPU (deferred): `wgpu`.
- PDB parsing (deferred): `pdbtbx`.

## 5. Data model

### 5.1 Recipe

We adopt cellPACK's **recipe v2.1 schema** verbatim and add a forward
`parsimony_version` field. Three sections: `objects` (ingredient
prototypes, with inheritance), `composition` (a tree of regions
referring to objects with counts/molarities), `gradients` (density
biases). See `cellpack/docs/RECIPE_SCHEMA.md` and `examples/recipes/v2/`
for canonical examples; `spheres_in_a_box.json` is the MVP target.

Loader: a Rust crate of strongly-typed structs deserialized via serde,
with `#[serde(rename_all = "snake_case")]` and `#[serde(tag = "type")]`
for the ingredient variants. Conversion (`molarity` → `count` =
`molarity * 0.0006022 * volume_Å³`) is computed once at load.

We skip the v1 → v2 migration code that bloats the Python loader; if a
v1 recipe comes in, we error and ask the user to convert.

### 5.2 Ingredient

Three representations carried on every ingredient (mirroring cellPACK
v2):

- `packing`: a **sphere tree** (multi-level). Level 0 is the
  encapsulating sphere; deeper levels are clusters of smaller spheres.
  This is the *only* representation used for collision in v0.1.
- `mesh`: triangle mesh, for visualization and (later) mesh-precise
  collision when the sphere tree is too coarse.
- `atomic`: PDB / mmCIF, for final all-atom export.

Sphere trees come from `.sph` files (cellPACK's existing format) or are
derived from a single radius. Decomposition methods (K-means in
cellPACK; medial-axis or MIVS as future options) live in
`parsimony-core::sphere_tree`. **(open)** which decomposer is best for
typical proteins.

Each ingredient also carries: `count` (or `molarity` resolved at load),
`packing_mode`, `principal_vector` (for membrane orientation),
`available_regions`, `cutoff_boundary`, `cutoff_surface`, `partners`.
These are exactly the cellPACK v2 fields; we keep the names.

### 5.2.1 Variants — alternative shapes for the same ingredient

An ingredient may carry **multiple variants** — alternative shape
representations sharing one biological identity. Canonical examples:

- **Conformational cycles.** ATP synthase F1 β-subunit cycles through
  *E* (empty), *T* (tight, ADP+Pi → ATP), *O* (open, ATP released).
  Same protein, three sphere trees and three meshes; the cycle index is
  a variant identity.
- **Allosteric states.** Hemoglobin T ↔ R, GroEL apo / ATP / ADP, every
  kinase active/inactive flip. Same ingredient family, different
  conformation.
- **Ligand-bound forms.** A receptor with and without its bound
  signaling molecule; the bound form might pack as a coarser composite.

A recipe entry:

```json
"objects": {
  "atp_synthase_f1_beta": {
    "type": "multi_sphere",
    "variants": {
      "E": { "spheres": "atp_synth_beta_E.sph", "mesh": "..." },
      "T": { "spheres": "atp_synth_beta_T.sph", "mesh": "..." },
      "O": { "spheres": "atp_synth_beta_O.sph", "mesh": "..." }
    },
    "default_variant": "E"
  }
}
```

This is a recipe-schema **extension** over cellPACK v2.1; we'll bump
`parsimony_version` independently of `format_version`. Recipes without
`variants` collapse to a single canonical variant (id = 0) — the schema
is backward-compatible.

Variants share `ingredient_id` (family) but have distinct `variant_id`.
Placement records both. Most operations are variant-agnostic; collision
uses the variant's specific sphere tree.

### 5.3 Compartment

A compartment is a closed boundary classifying space into three states:
exterior, surface (a thin shell around the boundary), interior.

Three representations:

- **Analytical** (sphere, capsule, axis-aligned box). In/out and
  surface-distance are closed-form. E. coli is a capsule — ideal first
  target.
- **Mesh**. Triangle mesh; in/out via `parry3d::query::PointQuery` (BVH
  over triangles + signed distance) and/or pre-voxelization of the
  shell into the multiscale voxel field.
- **Implicit** (signed distance function). Future option; lets us
  represent organelles defined by isosurfaces of cryo-ET maps.

Nested compartments compose via the recipe's composition tree:
membrane-of-cell contains cytoplasm contains nucleoid. Classification
assigns each spatial query the *innermost* matching compartment.

### 5.4 Placement

```rust
pub struct Placement {
    pub ingredient_id: IngredientId,
    pub variant_id: VariantId,        // default 0 if ingredient has no variants
    pub compartment_id: CompartmentId,
    pub position: Vec3,               // Å
    pub rotation: UnitQuaternion,
    pub instance_uid: u64,            // stable across edits (incl. variant + identity changes)
}
```

A packed state is `Vec<Placement>` plus the spatial index seed. The
spatial index is *reconstructable* from the placement list — but in
practice we keep it live to avoid rebuilds.

The `instance_uid` is the **continuity anchor**. It survives variant
swaps (allosteric flips), identity swaps (catabolic / anabolic
transformation), and position updates. Downstream consumers — viewers,
trajectory exporters, prism processes — track an instance through
its lifetime by `uid`, not by `(ingredient_id, position)`.

### 5.5 Snapshot

```rust
pub struct Snapshot {
    pub recipe_ref: RecipeRef,           // (name, format_version, hash)
    pub rng_seed: u64,
    pub placements: Vec<Placement>,
    pub compartments: Vec<CompartmentInstance>,
    pub metadata: BTreeMap<String, Value>,
}
```

Serializable. The fundamental boundary between parsimony and the
outside world.

## 6. Spatial structures (the heart of v0.1)

### 6.1 The grid problem

cellPACK's regular grid is sized to the *smallest* ingredient's
`min_radius`. For an E. coli model with both 17 Å beads and ~0.5 µm
features, that's a `1000³`-cell grid eating ≥4 GB just to mark
occupancy — most of it empty. Worse, large ingredients are tested
against irrelevantly fine cells. The single-grid choice is the root
cause of cellPACK's scale ceiling.

### 6.2 Two-structure decomposition

We split the spatial work along the natural grain:

- **`SpatialIndex`** — over placed instances. Answers "which instances
  overlap this query region?" Drives collision and neighbor queries.
- **`VoxelField`** — over space itself. Answers "what compartment is
  this point in? is this region inside something? how far to the
  nearest occupied volume?" Drives candidate sampling and compartment
  classification.

They're complementary. The placement loop uses both: sample a candidate
point via the voxel field's free-space map, then test for collision via
the spatial index.

### 6.3 `SpatialIndex` trait

```rust
pub trait SpatialIndex {
    fn insert(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError>;
    fn remove(&mut self, uid: u64) -> Result<(), IndexError>;
    fn update(&mut self, uid: u64, aabb: Aabb) -> Result<(), IndexError>;
    fn query_aabb<F: FnMut(u64)>(&self, q: &Aabb, visit: F);
    fn query_sphere<F: FnMut(u64)>(&self, q: &Sphere, visit: F);
    fn len(&self) -> usize;
    fn stats(&self) -> IndexStats;
}
```

Production implementation: **`QbvhIndex`** — 4-wide SIMD BVH with SoA
storage (`f32x4` per axis), native O(log₄ n) incremental
insert/remove/update via leaf-splitting and cascading `detach_empty`,
top-down SAH build, manual rebuild on quality degradation. Patterned on
`parry3d::partitioning::Qbvh` but self-owning.

`BruteIndex` is retained as the correctness oracle (used in tests and
benchmarks; not a production option).

Originally the design called for a second backing (`HierGridIndex`) as
a hedge against tree-based indices being slow on uniform distributions.
The QBVH benchmarks closed that risk — query throughput is 70–2000× over
brute, and the mixed edit+query (dynamics) workload is 4–83× faster than
a binary-BVH baseline. A grid backend can be added later behind the same
trait if a specific workload demands it.

### 6.4 `VoxelField` — sparse hierarchical grid

A three-level sparse grid in the style of OpenVDB:

```
root: HashMap<RootCoord, Box<L1Tile>>
L1:   Vec<Option<Box<L0Tile>>>[4096]    (dense 16³ children, heap-resident)
L0:   [Cell; 512]                       (dense 8³ cells, 2 KB)
```

A single root coordinate covers `(16 × 8)³ = 128³` cells. The root
hashmap is sparse — empty regions cost zero memory. L1 tiles allocate
L0 children lazily; L0 tiles are freed automatically when their last
non-default cell is overwritten back to [`Cell::DEFAULT`].

**Cell** (`u16 + u8 + u8 = 4 bytes`): `compartment` (0 = exterior),
`flags` bitfield (`OCCUPIED`, `SURFACE`, `MEMBRANE_INNER`,
`MEMBRANE_OUTER`, room for more), `distance` (quantized 0..255).

**Per-tile aggregation** drives multi-scale rejection: each L0 tile
tracks `active_count` (non-default cells), each L1 tile tracks
`active_l0_count`. Region queries like
[`is_region_default`](crate::voxel::VoxelField::is_region_default)
prune at the highest level where the answer is unambiguous — a 200-nm
ribosome cluster never has to scan individual 5-nm cells in regions
known to be empty. This is the multiscale-grid story committed in §3
principle 1.

**Operations (delivered in Phase 1d):**

- `sample(p) / set(p, cell)` — world-point read/write.
- `get(c) / put(c, cell)` — direct cell-coord access.
- `mark_aabb(aabb, cell)` — fill a box.
- `mark_occupied / clear_occupied / mark_compartment(aabb, ...)` — bulk flag/id ops.
- `is_region_default(aabb) -> bool` — pruning region query.
- `any_cell_with_flags(aabb, mask) -> bool` — flag region query.
- `is_region_free(aabb) -> bool` — placement-loop convenience (no OCCUPIED in region).
- `is_l0_active(c) / is_l1_active(c)` — hierarchical activity probes.
- `iter_active_cells / iter_active_cells_in(aabb)` — visitor iteration.
- `find_free_point(near, radius, rng, attempts)` — rejection sampling.
- `bounds()` — AABB enclosing all written cells.

**Coordinate math:** cell coords are signed `i32`. Negative coords
work via arithmetic shift (`>>` floors) and two's-complement masking
(`-1 & 0xF == 15`). `root = cell >> 7`; L1 sub-index =
`(cell >> 3) & 0xF`; L0 sub-index = `cell & 0x7`.

**Why this and not OpenVDB itself**: OpenVDB is C++, GPL-incompatible
in places, and Rust ports are immature. The structure we need is
~750 LOC and tuned to our queries.

### 6.5 Multiscale, naturally

Different ingredients live at natural scales. The voxel field offers a
**level query**: `sample_at_level(p, L)` returns the coarsest tile that
fully contains `p` at active level L. Placement of a large ingredient
uses coarse levels for free-space rejection and fine levels only for
boundary refinement. We never quantize the world to one resolution.

The `SpatialIndex` adapts automatically — instances live at whatever
AABB they have. A 17 Å sphere and a 200 nm ribosome cluster share the
same tree; the QBVH's SAH-driven branching prevents cross-scale
pollution.

### 6.6 Phase 1 deliverable

`parsimony-spatial` standalone, exporting both traits and the
implementations, with:

- Unit tests for correctness against `BruteIndex` (the oracle) on
  randomized inputs and high-churn edit+query workloads.
- Microbenchmarks (Criterion): bulk build, incremental insert, AABB
  query, sphere query, mixed edit+query at 10² → 10⁶ instances.
- A correctness harness loading `cellpack/examples/recipes/v2/*.json`,
  building the voxel field over each compartment, and validating
  in/out classification matches the Python reference (within voxel
  resolution).
- A `compare_kdtree` example mirroring cellPACK's broad-phase workload
  for a single comparison ratio.

No placement, no recipe-level packing. Just the structures.

## 7. Compartments and boundaries

For v0.1 we implement analytical compartments (sphere, capsule, AABB)
and mesh compartments. Mesh in/out goes through `parry3d`'s
`PointQuery::project_point_with_max_dist` and `parry3d`'s `TriMesh`.

**Surface ingredients** (membrane proteins). The compartment exposes a
**surface sampler**: `sample_surface(rng) -> (Vec3, UnitVec3)` returning
a point and outward normal. For a capsule this is closed-form; for a
mesh we precompute per-triangle areas and sample proportionally,
returning the face normal.

Membrane orientation: the ingredient's `principal_vector` is rotated to
align with the local surface normal via `rotation_between_vectors`. We
keep cellPACK's convention here verbatim. **(open)** later: a richer
"attachment frame" with explicit lipid-tail axes for asymmetric
membrane proteins.

## 8. Placement strategy

### 8.1 Trait

```rust
pub trait Placer {
    fn step(&mut self, env: &mut Env, rng: &mut RngHandle) -> StepOutcome;
}

pub enum StepOutcome {
    Placed(Placement),
    NoFreeIngredient,        // all ingredients done
    NoCandidatePoint,        // ran out of dropable points
    Failed { ingredient_id, reason },
}
```

`Env` exposes the `SpatialIndex`, the `VoxelField`, the compartments,
the recipe, and the placement history. The `Placer` is small — one
step is one attempt.

### 8.2 Greedy random + sphere-tree rejection (v0.1)

The cellPACK algorithm, ported and simplified:

1. **Pick an ingredient** by priority/weight.
2. **Sample a candidate point** in the ingredient's allowed compartment
   via the voxel field's free-space sampler.
3. **Sample a rotation** (uniform random, or constrained for membrane
   ingredients).
4. **Test for collision** by querying the spatial index for instances
   within the ingredient's encapsulating radius, then running
   sphere-tree-vs-sphere-tree fine collision.
5. On success, insert into the spatial index and mark occupied tiles
   in the voxel field. On failure, increment a per-ingredient attempt
   counter; bail when over a threshold.

This is straightforward — most of the complexity is in the structures
underneath, which we'll have already.

### 8.3 Future strategies

- **FFT-spectral overlap** (SIGGRAPH 2023 paper, doi:10.1145/3592126):
  voxelize ingredient + voxelize free-space → cross-correlate via FFT →
  place at the global maximum. For *dense* sub-regions (paracrystalline
  arrays, condensates). Naturally GPU-parallel.
- **MD-relax** (Flex / LAMMPS equivalent): once placed, run rigid-body
  soft-contact relaxation. Maritan's pipeline does this on GPU with
  Flex; Autin's at 2M-bead scale switches to LAMMPS on CPU. We'll write
  our own minimal wgpu compute kernel for the rigid-body case (it's not
  much — soft-sphere force, Verlet integration, friction).

These plug in via the `Placer` trait without touching anything else.

## 9. Output

**Primary format: Simularium JSON** — the format consumed by the
existing `cellpack.allencell.org` viewer. Spec at simularium.allencell.
org. Each frame is a flat list of `{ uniqueId, agentType, x, y, z,
xrot, yrot, zrot }`. We emit a single-frame trajectory in v0.1; later
versions emit multi-frame trajectories for dynamics-ready output.

**Secondary format: transform-list JSON** — debugging / programmatic
consumption. Mirrors cellPACK's `save_Mixed_asJson` shape.

**Future formats: PDB ensemble, mmCIF.** Out of scope for v0.1.

## 10. Dynamics-readiness

Even though v0.1 does only one-shot packing, every interface is
designed for the dynamic case. Concretely:

- The `SpatialIndex` trait has `remove`/`update`, not just `insert`.
- The voxel field has `mark_occupied`/`mark_free`, not just `build`.
- The `Snapshot` is the load-bearing serialized form; "warm start from
  this snapshot" is the canonical entry point. v0.1 wraps it as
  "start from empty snapshot".
- The `Placer` trait is step-at-a-time; whatever loop drives it can be
  one-shot ("until done") or continuous ("react to external events").

So when prism eventually drives parsimony as a process, the integration
is just an `Engine`-side step loop calling `Placer::step` and emitting
`Snapshot` deltas — no new internal machinery.

### 10.1 The state API: incremental operations

The full set of incremental ops on a live packed state:

```rust
pub enum Op {
    Insert  { placement: Placement },
    Remove  { uid: u64 },
    Move    { uid: u64, position: Vec3, rotation: UnitQuaternion },
    Replace { uid: u64, change: ReplaceChange },
    // Batched application below; single ops are sugar for a singleton batch.
}

pub enum ReplaceChange {
    /// Variant-swap within the same ingredient family. Common for
    /// allosteric / conformational cycles (ATP synthase E↔T↔O, GroEL,
    /// hemoglobin T↔R). Cheap if the encapsulating sphere is unchanged.
    Variant { variant_id: VariantId },
    /// Identity-swap to a different ingredient. The biological case is
    /// in-place chemical transformation: ADP→ATP at the catalytic site,
    /// glucose→glucose-6-P after kinase action. Position/rotation may
    /// be preserved or perturbed.
    Identity { ingredient_id: IngredientId, variant_id: VariantId },
    /// Combined: change identity *and* pose simultaneously.
    Both { ingredient_id: IngredientId, variant_id: VariantId,
           position: Vec3, rotation: UnitQuaternion },
}

pub fn apply_batch(env: &mut Env, ops: &[Op]) -> Result<(), BatchError>;
```

`apply_batch` is atomic with respect to the snapshot: either every op
succeeds and the snapshot updates, or nothing changes and we get a
`BatchError` naming the offending op. This matters for prism: one
bigraph-reaction firing is one batch — partial application would
desynchronize parsimony from the bigraph state.

**Why batch is the primitive.** ATP synthase's catalytic step is a
single biological event:

```rust
apply_batch(env, &[
    Op::Remove  { uid: pi_uid },                                // Pi consumed
    Op::Replace { uid: adp_uid,
                  change: Identity { ingredient_id: ATP, variant_id: 0 } },
    Op::Replace { uid: f1_beta_uid,
                  change: Variant { variant_id: O_state } },    // E/T → O
])?;
```

Three coupled changes, one rewrite. Catabolism (proteasome chews up a
substrate), anabolism (ribosome adds an amino acid), signaling
(receptor binds ligand and changes conformation) all decompose into
batches of these ops.

**Why `Replace` is not just `Remove + Insert`.** The instance_uid is
preserved (continuity through transformation). The free-space search
is skipped — we *know* there's room because there already was one
there. The compartment ID is preserved. Most importantly, downstream
consumers see the change as a single event, not a deletion-creation
pair that briefly contradicts conservation laws.

### 10.2 Easing / interpolation (deferred, rendering-side)

Smooth visualization of conformational cycles is a rendering concern,
not a state-API concern. The snapshot model stays discrete: each batch
produces a new snapshot. Smoothness comes from:

- A multi-frame trajectory exporter (post-v0.1) emits sequential
  snapshots; the renderer interpolates positions/rotations between
  matching `instance_uid`s via slerp.
- Variant changes (shape changes) don't interpolate naturally. Future
  option: a per-variant **morph correspondence** (paired sphere /
  vertex sets) lets a renderer morph between conformations linearly.
  Out of scope for v0.1.

The state API stays discrete and exact. Easing lives in the viewer.
See open question §15.7.

## 11. GPU strategy (deferred but architected)

We use `wgpu` when we GPU-ize, for cross-platform compute (Apple Silicon
included). Three targets, in order of expected payoff:

1. **Broad-phase collision** — sphere–sphere queries in parallel; a
   compute kernel over the spatial index. Direct port from the CPU
   path; expected ~10–100× on real workloads.
2. **Rigid-body relaxation** — a soft-sphere force kernel + Verlet
   integration + position-based dynamics. Equivalent to Flex's
   `rbd_solver`. The Maritan paper validates this approach at scale.
3. **FFT-spectral collision** — voxelize ingredient + voxelize free
   space → 3D FFT → multiply → IFFT → find max. `wgpu`'s `wgpu-fft` or
   a hand-rolled radix-2. For dense sub-regions only.

What we do **not** GPU-ize: recipe loading, compartment construction,
snapshot serialization, anything that runs once per packing.

GPU integration is gated by a single feature flag and a `Backend` enum;
all logic that touches GPU lives in a `parsimony-gpu` crate (Phase 4+).

## 12. Prism integration (forward-compat, no current dep)

We do *not* depend on prism in v0.1. The integration shape is sketched
here so we don't paint ourselves into a corner.

**As a Value type.** A `Snapshot` is a `Value::Tree`-shaped structure
under prism's schema (`prism-schema`). The recipe-level structure
(objects → composition → placements) maps to nested `Value::Tree`s
naturally. We add a `parsimony-prism` adapter crate (Phase 5+) that
implements the `Schema::Custom` registration and `Value` ↔ `Snapshot`
conversion.

**As a Process.** A `PackProcess` is a `prism_bigraph::Process` that
takes an input port `recipe: Recipe`, runs the placer to completion,
and emits the resulting `Snapshot` on an output port. This is a
read-only mode — for incremental dynamics, a separate `PackingMonitor`
process maintains a live `Snapshot` and accepts `Insert` / `Remove` /
`Move` reaction-rule firings.

**Why deferred.** prism's API surface is still evolving (`Process`,
`Step`, `BRS`, `discover_processes`). Tying parsimony to it pre-1.0
would create coupling churn. The standalone-first path also keeps
parsimony useful to anyone outside this ecosystem.

## 13. Validation against cellPACK

We treat the Python cellPACK as our reference oracle.

**`parsimony-bench`** runs both packers against shared recipes and
reports:

- **Correctness**: do both produce the same per-ingredient placement
  counts (within a tolerance set by RNG variance)? Same compartment-ID
  distribution? Same in/out classification of test points?
- **Quality**: pair-correlation function, nearest-neighbor distance
  distribution, void-size distribution — match cellPACK?
- **Speed**: wall-clock, peak memory.

Bit-identity is not the goal (different RNGs, floating-point ops).
*Distributional match* is — and a *wall-clock ratio* on the same
hardware.

**Test recipes (initial set).** All from `cellpack/examples/recipes/v2/`:

1. `spheres_in_a_box.json` — 630 single spheres, four sizes. Smallest
   useful workload.
2. `nested_spheres.json` — nested compartments, surface region.
3. `peroxisome.json` — real mesh compartment, gradient.
4. Then a hand-authored E. coli starter (capsule, ~5 ingredient types,
   ~10⁴ instances total).

Plus a synthetic scaling series (random sphere recipes at sizes 10², 10³,
10⁴, 10⁵, 10⁶) to characterize asymptotic behavior.

The cellPACK Python install lives at `/home/pattern/code/cellpack/`;
the bench harness shells out via `python -m cellpack.bin.pack --recipe
... --seed N` and parses the resulting Simularium / transform JSON for
comparison.

## 14. Roadmap

> Phases 0–3 below are delivered (the whole *M. genitalium* cell packs
> end to end). Forward feature work — gradients, partner packing,
> realtime animation, container/ingredient unification — now lives in
> the living backlog at [`ROADMAP.md`](ROADMAP.md).

### Phase 0 — design (done 2026-05-18)

This document. Workspace skeleton.

### Phase 1 — spatial structures prototype

`parsimony-spatial` crate. Decomposed into sub-passes:

- **1a** *(done)* — AABB, `Sphere`, `Ray`, `SpatialIndex` trait,
  `BruteIndex` reference impl.
- **1b′** *(done; supersedes 1b/1c)* — `QbvhIndex`, 4-wide SIMD BVH with
  native O(log₄ n) incremental ops via `wide::f32x4`. Correctness vs.
  `BruteIndex`, 3000-op churn invariants. Bench numbers: 1.4–1.8× over
  binary BVH on queries, 4–83× over binary BVH on the dynamics-shaped
  mixed edit+query workload, 70–2000× over brute. Originally planned
  binary `BvhIndex` (1b) and `HierGridIndex` (1c) were collapsed into
  this single QBVH pass; the brute oracle is the only retained baseline.
- **1d** *(next)* — `VoxelField` (3-level sparse hierarchical OpenVDB-
  inspired voxel field) for compartment classification and free-space
  sampling.
- **1e** — `compare_kdtree` example mirroring cellPACK's broad-phase,
  plus a mesh in/out correctness harness against cellPACK's voxelize-
  and-flood-fill on `peroxisome.json`.

**Done when:** all queries pass correctness, benchmarks show ≥2× over
the cKDTree-rebuild-each-query baseline at 10⁴ instances (already
exceeded), voxel field classifies the cellPACK `peroxisome.json` mesh
correctly.

### Phase 2 — core packer MVP

`parsimony-core` (recipe loader, ingredient, compartment, placer) +
`parsimony-cli`. Greedy random + sphere-tree rejection over the
Phase-1 structures. Simularium output.

**Done when:** `parsimony pack spheres_in_a_box.json -o out.simularium`
runs in <1 s, the output loads in the Simularium viewer, and the
distribution matches cellPACK Python within 5% on per-ingredient counts.

### Phase 3 — E. coli starter

Hand-author a small E. coli recipe (capsule cell, inner membrane, ~3–5
ingredient types: ribosome, RNA polymerase, GroEL, a representative
soluble enzyme, lipids). Package the structures (PDB + sphere-tree
decomposition). Run end-to-end. Compare to cellPACK Python on the same
recipe.

**Done when:** the result visualizes plausibly, the distribution
matches cellPACK, runtime ≤ cellPACK Python.

### Phase 4+ (deferred)

GPU broad-phase (wgpu). MD-relax kernel. FFT-spectral collision. Full
EcoCyc + PDB + AlphaFold ingestion. prism `Value`/`Process` adapter.
Multi-frame trajectory output. Dynamics driven by external processes.

## 15. Open questions

1. **Sphere-tree decomposition.** cellPACK uses K-means on the protein's
   atomic centers. Better alternatives exist (medial-axis-based MIVS,
   inscribed-sphere packing). **(open)** — pick during Phase 3 once we
   have real ingredients.
2. **Nucleoid representation.** cellPACK treats the bacterial nucleoid
   as one big ingredient (LatticeNucleoid output). For E. coli that's
   ~4.6 Mbp at 10 bp/bead ≈ 460k beads. Do we keep it as one composite
   ingredient, or expose chromatin as a first-class compartment?
   **(open)**.
3. **Voxel field tile sizes.** 16/8/8 is a starting guess. Tune
   empirically in Phase 1.
4. **Snapshot versioning.** Embed `format_version`, hash inputs,
   schema-migrate on load. Details deferred to first format change.
5. **Determinism across hardware.** Floating-point reductions on GPU
   are non-deterministic. Do we tolerate run-to-run drift or do we pay
   for determinism via Kahan / pairwise summation? Deferred to GPU
   phase.
6. **Recipe v2.1 extensions.** Do we add `lod_radii`, explicit
   `attachment_frame`, or scale annotations to the schema, and version-
   bump? Or keep additions out-of-band until v3? **(open)**. (The
   `variants` extension in §5.2.1 is already committed.)
7. **Variant morph correspondences.** For renderers to *ease* between
   variants (E → T → O for ATP synthase, T → R for hemoglobin), we'd
   want paired-sphere or paired-vertex correspondences. Schema +
   tooling for that authoring is a real project on its own. **(open)** —
   defer until we have a viewer that wants it.

## 16. References

External context (papers, repos, databases) is catalogued in the
project memory note `reference-external`. Internal cross-links:

- `cellpack/autopack/Environment.py:1865` — the Python `pack_grid` loop
- `cellpack/autopack/ingredient/Ingredient.py:1241` — sphere-tree
  collision (the primitive worth preserving)
- `cellpack/autopack/Compartment.py:1160` — mesh voxelization for
  in/out classification
- `cellpack/docs/RECIPE_SCHEMA.md` — v2.1 recipe schema
- `cellpack/examples/recipes/v2/spheres_in_a_box.json` — MVP target
- `prism/docs/prism-architecture.md` — `Process` / `BRS` / `Engine` for
  Phase-5 integration shape
