#!/usr/bin/env python3
"""Plot prompt-to-best-delayed-cluster times from study_delayed_clusters.py."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARTICLE_LABELS = {"211": r"$\pi^+$", "-211": r"$\pi^-$", "22": r"$\gamma$"}
PARTICLE_COLORS = {"211": "#1f77b4", "-211": "#d95f02", "22": "#3a923a"}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_csv", type=Path, help="events.csv produced by study_delayed_clusters.py")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG or PDF path")
    parser.add_argument("--min-ns", type=float, default=100.0)
    parser.add_argument("--max-ns", type=float, default=6100.0)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()
    if args.bins < 1 or args.max_ns <= args.min_ns:
        parser.error("Require --bins >= 1 and --max-ns > --min-ns")
    output = args.output or args.events_csv.with_name("delayed_time_histogram.png")

    totals: Counter[str] = Counter()
    delays: dict[str, list[float]] = defaultdict(list)
    with args.events_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            particle = row.get("primary_pid", "") or row.get("selection_category", "unknown")
            totals[particle] += 1
            if row.get("delta_t_ns"):
                delay = float(row["delta_t_ns"])
                if np.isfinite(delay):
                    delays[particle].append(delay)

    if not totals:
        raise ValueError("The event table is empty")
    edges = np.linspace(args.min_ns, args.max_ns, args.bins + 1)
    groups = sorted(totals, key=lambda key: (key not in PARTICLE_LABELS, key))
    fig, axes = plt.subplots(len(groups), 1, figsize=(9, max(3.3, 2.8 * len(groups))), sharex=True, squeeze=False)
    for axis, particle in zip(axes[:, 0], groups):
        values = np.asarray(delays[particle])
        counts, _ = np.histogram(values, bins=edges)
        axis.stairs(counts, edges, linewidth=1.8, color=PARTICLE_COLORS.get(particle, "#555555"))
        axis.set_ylabel(f"Events / {(args.max_ns - args.min_ns) / args.bins:g} ns")
        axis.grid(alpha=0.2)
        label = PARTICLE_LABELS.get(particle, particle)
        axis.set_title(f"{label}: {len(values)}/{totals[particle]} with a delayed candidate; {counts.sum()} in plotted range", loc="left", fontsize=11)
        print(f"{particle}: {totals[particle]} events, {len(values)} candidates, {counts.sum()} in range")
    axes[-1, 0].set_xlabel("Best delayed-cluster time minus prompt-cluster time (ns)")
    axes[-1, 0].set_xlim(args.min_ns, args.max_ns)
    fig.suptitle("Delayed hit-cluster candidates", fontsize=13)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
