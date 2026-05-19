#!/usr/bin/env bash
# Pack a recipe, then open it in the local viewer.
#
# Usage:
#   ./scripts/view_pack.sh                                  # shape_zoo demo
#   ./scripts/view_pack.sh path/to/recipe.json
#   ./scripts/view_pack.sh path/to/recipe.json --loose-bounds
#   PORT=8000 ./scripts/view_pack.sh ...                     # override port
#
# Browsers won't fetch local files via file:// for security, so we
# serve the viewer + data over a tiny local HTTP server (Python
# stdlib, no deps).
set -e

cd "$(dirname "$0")/.."

RECIPE="${1:-examples/recipes/shape_zoo.json}"
shift || true

PORT="${PORT:-8123}"
mkdir -p viewer/data
OUT="viewer/data/latest.pack.json"

echo "[view_pack] packing ${RECIPE}…"
PARSI=(cargo run --release --quiet -p parsimony-cli -- pack "$RECIPE" --out "$OUT" "$@")
"${PARSI[@]}"

# Serve from the project root (not from `viewer/`) so the viewer
# can fetch root-relative mesh URLs like `/examples/pdb_meshes/x.obj`
# that the pack file references.
URL="http://localhost:${PORT}/viewer/index.html?file=data/latest.pack.json"
echo "[view_pack] serving viewer at ${URL}"

# Open in default browser shortly after the server starts.
(sleep 0.5 && (xdg-open "$URL" >/dev/null 2>&1 || open "$URL" >/dev/null 2>&1 || true)) &

# Prefer uv-managed python (lets us declare a python version), fall
# back to system python3. Either runs stdlib http.server only.
if command -v uv >/dev/null 2>&1; then
    exec uv run --python 3.12 python -m http.server "$PORT"
else
    exec python3 -m http.server "$PORT"
fi
