#!/usr/bin/env bash
#
# Remove generated artifacts from git — both from tracking AND from history —
# so the repo is pushable. Everything removed here is regenerated natively by
#   parsimony translate-mycoplasma   (cellpack data + RCSB fetch → meshes/recipe)
#   parsimony pipeline run           (→ viewer packs)
# so none of it belongs in version control.
#
# What it touches:
#   examples/pdb_meshes/        ~1.5 GB of VdW-surface OBJs  (the push blocker:
#                               pre-vertex-budget meshes were up to ~160 MB,
#                               over GitHub's 100 MB/file limit)
#   examples/pdb_cache/         PDB/mmCIF structures fetched from RCSB
#   viewer/data/*.pack.json     generated packs
#
# Stage 1 (untrack + commit) is safe. Stage 2 (history rewrite) is DESTRUCTIVE:
# it rewrites every commit hash, so anyone with a clone must re-clone. Each
# destructive step asks for confirmation. Run from anywhere inside the repo.
set -euo pipefail

cd "$(git rev-parse --show-toplevel)"
echo "repo: $(pwd)"

PATHS=(examples/pdb_meshes examples/pdb_cache)
PACK_GLOB="viewer/data/*.pack.json"

# ── Stage 1: stop tracking (files stay on disk) + commit ────────────────────
echo
echo "── Stage 1: untrack generated artifacts ──"
git rm -r --cached --ignore-unmatch "${PATHS[@]}" $PACK_GLOB >/dev/null || true
git add .gitignore 2>/dev/null || true
if git diff --cached --quiet; then
  echo "nothing newly untracked (already out of the index)."
else
  git commit -m "Drop generated meshes/packs from tracking (regenerated natively)"
fi

# ── Show what's still bloating history ──────────────────────────────────────
echo
echo "── largest blobs in history (MB) ──"
git rev-list --objects --all \
  | git cat-file --batch-check='%(objecttype) %(objectname) %(objectsize) %(rest)' \
  | sed -n 's/^blob //p' | sort -rn \
  | awk '{ printf "  %7.1f  %s\n", $2/1048576, $3 }' | head -15
echo "(if you see examples/pdb_meshes/*.obj above, Stage 2 is needed to push)"

# ── Stage 2: purge from ALL history (destructive) ───────────────────────────
echo
read -r -p "Stage 2: rewrite history to purge those paths everywhere? [y/N] " ans
if [[ "${ans:-}" != [yY] ]]; then
  echo "skipped. Working-tree untracking is committed; history is unchanged."
  exit 0
fi

if ! git diff --quiet || ! git diff --cached --quiet; then
  echo "!! uncommitted changes present — git-filter-repo needs a clean tree." >&2
  echo "   commit or stash everything, then re-run." >&2
  exit 1
fi

if command -v git-filter-repo >/dev/null 2>&1; then
  git filter-repo --force \
    --path examples/pdb_meshes \
    --path examples/pdb_cache \
    --path-glob 'viewer/data/*.pack.json' \
    --invert-paths
  echo
  echo "history rewritten. filter-repo removed 'origin' — re-add and force-push:"
  echo "  git remote add origin <url>"
  echo "  git push --force origin <branch>"
else
  echo "!! git-filter-repo not installed." >&2
  echo "   install:  pip install git-filter-repo   (then re-run this script)" >&2
  echo "   or BFG:   bfg --delete-folders '{pdb_meshes,pdb_cache}' && \\" >&2
  echo "             git reflog expire --expire=now --all && git gc --prune=now --aggressive" >&2
  exit 1
fi
