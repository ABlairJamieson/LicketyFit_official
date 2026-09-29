#!/usr/bin/env python3
"""Plot signed reconstructed-minus-Michel-truth vertex residuals from vertices.csv."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def matched_residuals_cm(rows: list[dict]) -> dict[str, np.ndarray]:
    """Include only successfully reconstructed, time-matched truth pairs."""
    residuals = {axis: [] for axis in "xyz"}
    for row in rows:
        if row.get("status") != "ok" or row.get("truth_status") != "matched":
            continue
        try:
            point = {axis: (float(row[f"{axis}_mm"]) - float(row[f"truth_{axis}_mm"])) / 10.0
                     for axis in "xyz"}
        except (KeyError, TypeError, ValueError):
            continue
        if not all(np.isfinite(value) for value in point.values()):
            continue
        for axis, value in point.items():
            residuals[axis].append(value)
    return {axis: np.asarray(values, dtype=float) for axis, values in residuals.items()}


def plot_residuals(rows: list[dict], output: Path, *, bins: int = 30,
                   label: str = "") -> int:
    if bins < 1:
        raise ValueError("bins must be positive")
    residuals = matched_residuals_cm(rows)
    n_matched = len(residuals["x"])
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for axis, coordinate in zip(axes, "xyz"):
        values = residuals[coordinate]
        if n_matched:
            limit = max(1.0, float(np.max(np.abs(values))) * 1.05)
            axis.hist(values, bins=np.linspace(-limit, limit, bins + 1),
                      histtype="stepfilled", alpha=0.45, color="#1f77b4")
            median = float(np.median(values))
            rms = float(np.sqrt(np.mean(values**2)))
            axis.set_title(f"median {median:+.1f} cm; RMS {rms:.1f} cm")
        else:
            axis.text(0.5, 0.5, "No matched truth", ha="center", va="center",
                      transform=axis.transAxes)
        axis.axvline(0.0, color="black", linestyle="--", linewidth=1)
        axis.set(xlabel=f"Reco - truth {coordinate} (cm)", ylabel="Events / bin")
        axis.grid(alpha=0.2)
    sample = f"{label}: " if label else ""
    fig.suptitle(f"{sample}delayed-cluster vertex residuals (n={n_matched} matched events)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return n_matched


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("vertices_csv", type=Path, help="vertices.csv from reconstruct_delayed_vertices.py")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--primary-pid", type=int, default=None,
                        help="Plot only events with this primary PDG code (211, -211, or 22)")
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")
    with args.vertices_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.primary_pid is not None:
        rows = [row for row in rows if row.get("primary_pid") == str(args.primary_pid)]
    suffix = f"_pid{args.primary_pid}" if args.primary_pid is not None else ""
    output = args.output or args.vertices_csv.with_name(f"delayed_vertex_residuals_xyz{suffix}.png")
    n_matched = plot_residuals(rows, output, bins=args.bins,
                               label=f"PDG {args.primary_pid}" if args.primary_pid is not None else "")
    print(f"Matched events: {n_matched}/{len(rows)}")
    print(f"Plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
