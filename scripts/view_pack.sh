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

# ---- preflight ----------------------------------------------------------
if [ ! -f "$RECIPE" ]; then
    echo "[view_pack] recipe not found: ${RECIPE}" >&2
    exit 1
fi
if ! command -v cargo >/dev/null 2>&1; then
    echo "[view_pack] cargo not on PATH — install Rust via https://rustup.rs" >&2
    exit 1
fi
if ! command -v uv >/dev/null 2>&1 && ! command -v python3 >/dev/null 2>&1; then
    echo "[view_pack] need either 'uv' or 'python3' for the local http server" >&2
    exit 1
fi

# Recipes that reference standalone PDB-derived meshes
# (examples/pdb_meshes/<id>[_label].obj). If any are missing, regenerate
# them from the PDB ID (first 4 chars of basename) via pdb_to_mesh.py.
ensure_pdb_meshes() {
    local recipe="$1"
    command -v python3 >/dev/null 2>&1 || return 0  # parsing needs python
    local missing
    missing="$(python3 - "$recipe" <<'PY'
import json, sys, re
from pathlib import Path
recipe = Path(sys.argv[1])
root = recipe.parent
needed = set()
def walk(node):
    if isinstance(node, dict):
        for k, v in node.items():
            if k == "mesh_path" and isinstance(v, str):
                needed.add(v)
            else:
                walk(v)
    elif isinstance(node, list):
        for v in node: walk(v)
walk(json.loads(recipe.read_text()))
for ref in sorted(needed):
    p = (root / ref).resolve()
    # Only auto-handle top-level pdb_meshes/<id>.obj — subdir meshes
    # (e.g. mycoplasma/) have their own bootstrap.
    try:
        rel = p.relative_to(Path("examples/pdb_meshes").resolve())
    except ValueError:
        continue
    if rel.parent != Path("."):
        continue
    if p.exists():
        continue
    m = re.match(r"^([0-9a-zA-Z]{4})", p.name)
    if not m: continue
    print(f"{m.group(1)}\t{p}")
PY
)"
    [ -z "$missing" ] && return 0

    if ! command -v uv >/dev/null 2>&1; then
        echo "[view_pack] missing PDB meshes and 'uv' is not installed;" >&2
        echo "            install uv (https://docs.astral.sh/uv/) to" >&2
        echo "            auto-generate them." >&2
        return 1
    fi
    while IFS=$'\t' read -r pdb_id obj_path; do
        echo "[view_pack] fetching + meshing ${pdb_id} → ${obj_path}"
        uv run scripts/pdb_to_mesh.py "$pdb_id" --out "$obj_path"
    done <<<"$missing"
}

# Mycoplasma recipes reference per-protein LOD meshes that have to be
# generated from the upstream cellPACK data. The meshes are committed
# to the repo, but if they're missing (fresh clone with LFS off, or
# someone wiped the dir) we clone the data repo and regenerate.
ensure_mycoplasma_meshes() {
    local recipe_name
    recipe_name="$(basename "$1")"
    case "$recipe_name" in
        mycoplasma.json|mycoplasma_full.json) ;;
        *) return 0 ;;
    esac

    local mesh_dir="examples/pdb_meshes/mycoplasma"
    if [ -d "$mesh_dir" ] && \
       [ -n "$(find "$mesh_dir" -maxdepth 1 -name '*.obj' -print -quit 2>/dev/null)" ]; then
        return 0
    fi

    if ! command -v uv >/dev/null 2>&1; then
        echo "[view_pack] ${mesh_dir} is empty and 'uv' is not installed;" >&2
        echo "            install uv (https://docs.astral.sh/uv/) so the" >&2
        echo "            mycoplasma mesh generator can fetch its deps." >&2
        return 1
    fi

    local cellpack_repo=".cache/MycoplasmaGenitalium"
    local cellpack_data="${cellpack_repo}/cellPACK_Data"
    if [ ! -d "$cellpack_data" ]; then
        echo "[view_pack] cloning ccsb-scripps/MycoplasmaGenitalium → ${cellpack_repo}"
        mkdir -p "$(dirname "$cellpack_repo")"
        git clone --depth 1 \
            https://github.com/ccsb-scripps/MycoplasmaGenitalium "$cellpack_repo"
    fi

    local top_n=30
    [ "$recipe_name" = "mycoplasma_full.json" ] && top_n=0

    echo "[view_pack] generating mycoplasma LOD meshes (--top-n ${top_n})…"
    uv run scripts/translate_mycoplasma.py \
        --cellpack-data "$cellpack_data" \
        --out-recipe "$1" \
        --out-meshes "$mesh_dir" \
        --lods 16.0,8.0,4.0,2.5 \
        --top-n "$top_n"
}
ensure_mycoplasma_meshes "$RECIPE"
ensure_pdb_meshes "$RECIPE"

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
# Use the repo's no-cache server so viewer.js edits + freshly packed
# data always show up on reload (stdlib http.server caches aggressively).
if command -v uv >/dev/null 2>&1; then
    exec uv run --python 3.12 python scripts/serve.py "$PORT"
else
    exec python3 scripts/serve.py "$PORT"
fi
