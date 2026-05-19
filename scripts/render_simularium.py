#!/usr/bin/env -S uv run --quiet
# /// script
# requires-python = ">=3.10"
# dependencies = [
#   "matplotlib>=3.8",
#   "numpy>=1.25",
# ]
# ///
"""Render a parsimony Simularium-format pack as a PNG.

Reads `<input>.simularium`, parses every placement (position +
enclosing-radius + type-id), groups by ingredient type, plots each
group as a 3D scatter sized by sphere radius and coloured per the
recipe's typeMapping colour. One PNG per call.

Usage (with uv installed — auto-provisions matplotlib/numpy):
    uv run scripts/render_simularium.py <input.simularium> <output.png>
        [--title "..."] [--azim 30] [--elev 20] [--alpha 0.5]
        [--slice {x,y,z} --slice-thickness 80]

Or run directly: `./scripts/render_simularium.py ...` — the shebang
hands off to `uv run` automatically.
"""

from __future__ import annotations
import argparse
import json
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers the projection)


def render(input_path: str, output_path: str, title: str,
           azim: float, elev: float, alpha: float,
           slice_axis: str | None = None, slice_thickness: float = 50.0) -> None:
    with open(input_path) as f:
        doc = json.load(f)

    type_meta: dict[int, dict] = {}
    for tid_str, info in doc["trajectoryInfo"]["typeMapping"].items():
        tid = int(tid_str)
        type_meta[tid] = {
            "name": info["name"],
            "color": info.get("geometry", {}).get("color", "#888888"),
        }

    # Simularium frame data layout: 11 floats per placement —
    # [vis_type, instance_id, type_id, x, y, z, rx, ry, rz, radius, subpoints_count].
    raw = doc["spatialData"]["bundleData"][0]["data"]
    n = len(raw) // 11

    grouped: dict[int, list[tuple[float, float, float, float]]] = {}
    for i in range(n):
        tid = int(raw[i * 11 + 2])
        x = raw[i * 11 + 3]
        y = raw[i * 11 + 4]
        z = raw[i * 11 + 5]
        r = raw[i * 11 + 9]
        grouped.setdefault(tid, []).append((x, y, z, r))

    # Compute world extent from the placements so we can convert sphere
    # radii into screen-area marker sizes that aren't dominated by
    # either the world scale or the figure dpi.
    all_xyz = np.array([(p[0], p[1], p[2]) for pts in grouped.values() for p in pts])
    world_extent = float((all_xyz.max(axis=0) - all_xyz.min(axis=0)).max())
    figure_inches = 10.0
    dpi = 130.0
    # ~half the rendered axes are content, so points-per-unit ≈ that.
    points_per_unit = (figure_inches * 72.0 * 0.5) / max(world_extent, 1.0)

    if slice_axis is not None:
        axis_idx = {"x": 0, "y": 1, "z": 2}[slice_axis]
        plot_axes = [i for i in (0, 1, 2) if i != axis_idx]
        labels = ["x", "y", "z"]
        fig, ax = plt.subplots(figsize=(figure_inches, figure_inches))
        points_per_unit_2d = (figure_inches * 72.0 * 0.85) / max(world_extent, 1.0)
        total_in_slice = 0
        for tid, pts in sorted(grouped.items()):
            arr = np.array(pts)
            in_slice = np.abs(arr[:, axis_idx]) <= slice_thickness * 0.5
            arr = arr[in_slice]
            if len(arr) == 0:
                continue
            total_in_slice += len(arr)
            name = type_meta.get(tid, {}).get("name", f"type_{tid}")
            color = type_meta.get(tid, {}).get("color", "#888888")
            marker_radius_pts = arr[:, 3] * points_per_unit_2d
            sizes = np.clip(marker_radius_pts ** 2, 4.0, 8000.0)
            ax.scatter(
                arr[:, plot_axes[0]], arr[:, plot_axes[1]],
                s=sizes, c=color, alpha=alpha,
                edgecolors="none",
                label=f"{name} ({len(arr)})",
            )
        ax.set_xlabel(labels[plot_axes[0]])
        ax.set_ylabel(labels[plot_axes[1]])
        ax.set_title(
            f"{title}\nslice {slice_axis}∈[−{slice_thickness/2:.0f},+{slice_thickness/2:.0f}]"
            f"  ({total_in_slice} placements visible)"
        )
        ax.set_aspect("equal")
        ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
        plt.tight_layout()
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close()
        print(f"wrote {output_path} ({total_in_slice} placements, slice along {slice_axis})")
        return

    fig = plt.figure(figsize=(figure_inches, figure_inches - 1))
    ax = fig.add_subplot(111, projection="3d")
    for tid, pts in sorted(grouped.items()):
        arr = np.array(pts)
        name = type_meta.get(tid, {}).get("name", f"type_{tid}")
        color = type_meta.get(tid, {}).get("color", "#888888")
        # matplotlib's scatter `s` is in points². Convert each sphere's
        # world-radius into screen radius (in points) before squaring.
        marker_radius_pts = arr[:, 3] * points_per_unit
        sizes = np.clip(marker_radius_pts ** 2, 4.0, 5000.0)
        ax.scatter(
            arr[:, 0], arr[:, 1], arr[:, 2],
            s=sizes, c=color, alpha=alpha,
            edgecolors="none",
            label=f"{name} ({len(pts)})",
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title(title)
    ax.view_init(elev=elev, azim=azim)
    # Keep aspect ratio honest (3D plots default to a stretched cube).
    ax.set_box_aspect((1, 1, 1))
    ax.legend(loc="upper right", fontsize=8, framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close()
    print(f"wrote {output_path} ({n} placements)")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input", help=".simularium file")
    ap.add_argument("output", help=".png output")
    ap.add_argument("--title", default="parsimony pack", help="figure title")
    ap.add_argument("--azim", type=float, default=30.0, help="azimuth (degrees)")
    ap.add_argument("--elev", type=float, default=20.0, help="elevation (degrees)")
    ap.add_argument("--alpha", type=float, default=0.55, help="marker opacity")
    ap.add_argument("--slice", choices=("x", "y", "z"), default=None,
                    help="render a 2D slice perpendicular to this axis instead of full 3D")
    ap.add_argument("--slice-thickness", type=float, default=50.0,
                    help="slab thickness around 0 (world units) when --slice is set")
    args = ap.parse_args()
    render(args.input, args.output, args.title, args.azim, args.elev, args.alpha,
           args.slice, args.slice_thickness)
    return 0


if __name__ == "__main__":
    sys.exit(main())
