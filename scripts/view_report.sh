#!/usr/bin/env bash
# Open docs/REPORT.md in whichever markdown viewer is available.
# Prefers `uv tool run grip` (ephemeral venv, no global install).
# Falls back to pandoc → VSCode → install hint.
set -e

cd "$(dirname "$0")/.."

if command -v uv >/dev/null 2>&1; then
    echo "[view_report] using grip via uv (no global install needed)"
    echo "[view_report] open the printed URL in a browser; Ctrl-C to stop"
    exec uv tool run --from grip grip docs/REPORT.md
fi

if command -v grip >/dev/null 2>&1; then
    echo "[view_report] using locally-installed grip"
    exec grip docs/REPORT.md
fi

if command -v pandoc >/dev/null 2>&1; then
    echo "[view_report] using pandoc → HTML"
    pandoc docs/REPORT.md \
        -o docs/REPORT.html \
        --standalone \
        --metadata title="parsimony — feature status" \
        --css=https://cdn.jsdelivr.net/npm/github-markdown-css/github-markdown-light.min.css
    echo "[view_report] wrote docs/REPORT.html"
    if command -v xdg-open >/dev/null 2>&1; then
        exec xdg-open docs/REPORT.html
    fi
    exit 0
fi

if command -v code >/dev/null 2>&1; then
    echo "[view_report] opening in VSCode (press Ctrl+Shift+V for preview)"
    exec code docs/REPORT.md
fi

cat <<EOF
[view_report] no markdown renderer found. Install one of:
  uv (recommended; we'd then use it to run grip): https://docs.astral.sh/uv/getting-started/installation/
  sudo apt install pandoc                       # convert to standalone HTML
  sudo snap install code                        # VSCode preview (Ctrl+Shift+V)

The images live in docs/img/ and the markdown references them with
relative paths, so any renderer that resolves relative paths will
show them.
EOF
exit 1
