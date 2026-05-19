# parsimony viewer

A minimal three.js viewer for parsimony packings. Reads the
Simularium JSON we emit (`parsimony pack ... --out foo.simularium`),
renders each ingredient type as a single instanced sphere mesh, with
orbit controls, per-type visibility, and a cross-section slider.

## Run it

```bash
./scripts/view_pack.sh                              # packs shape_zoo, opens browser
./scripts/view_pack.sh path/to/recipe.json          # other recipe
./scripts/view_pack.sh recipe.json --loose-bounds   # pass flags through to parsimony
PORT=9000 ./scripts/view_pack.sh                    # change port (default 8123)
```

The script:
1. Runs `cargo run --release -p parsimony-cli -- pack <recipe>` and
   writes `viewer/data/latest.simularium`.
2. Starts a local HTTP server (Python stdlib, no deps; uses uv if
   present, otherwise system `python3`).
3. Opens your browser to
   `http://localhost:8123/index.html?file=data/latest.simularium`.

Press Ctrl-C to stop the server.

## Use the UI

- **Drag-and-drop** any `.simularium` file onto the window to view it.
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
- `data/` — staging area for `view_pack.sh` (gitignore as needed).
