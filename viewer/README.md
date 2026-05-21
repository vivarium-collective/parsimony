# parsimony viewer

A minimal three.js viewer for parsimony packings. Reads parsimony's
native pack format (`parsimony pack ... --out foo.pack.json`), renders
each ingredient type with instanced meshes, with orbit controls,
per-type visibility, and a cross-section slider.

## Run it

```bash
cargo run --release -p parsimony-cli -- viewer                                 # landing page
cargo run --release -p parsimony-cli -- viewer --recipe examples/recipes/shape_zoo.json   # pack + open
cargo run --release -p parsimony-cli -- viewer --pack mycoplasma_full_staged.pack.json    # existing pack
cargo run --release -p parsimony-cli -- viewer --port 9000 --no-open           # options
```

`parsimony viewer`:
1. Optionally packs `--recipe` to `viewer/data/latest.pack.json` (or
   opens an existing `--pack`).
2. Serves the project root over its own native no-cache static server
   (built into the CLI — no Python).
3. Opens your browser to the viewer, deep-linking the pack via `?file=`.

Press Ctrl-C to stop the server.

## Use the UI

- **Drag-and-drop** any `.pack.json` file onto the window to view it.
- **Left-click drag**: orbit the camera. **Right-click drag**: pan.
  **Scroll**: zoom.
- **Legend** (right side): click a row to toggle that ingredient
  type's visibility.
- **Cross-section**: pick an axis and slide the plane to slice
  through the volume. "flip" inverts the kept half.
- **Auto-spin**: gently rotates the camera around the centre. Useful
  for screenshots / recordings.

## Why not Simularium-viewer?

cellPACK's Simularium viewer is a hosted web tool — it works, but
requires uploading the data and gives you no control over rendering.
This local viewer is:

- Native to parsimony's output format (no upload step).
- Direct three.js — easy to extend (next planned: Goodsell-style
  cel + outline shaders, then a wgpu native viewer for live
  packing-in-progress views).
- One small HTML + one JS file. No build step, no node_modules.

## Files

- `index.html` — UI + importmap referencing three.js from a CDN.
- `viewer.js` — scene, instanced rendering, UI wiring.
- `data/` — staging area for `parsimony viewer` / `demos`; `*.pack.json` are gitignored (regenerated).
