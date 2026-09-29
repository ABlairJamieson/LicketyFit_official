#!/usr/bin/env python3
"""Plot digit, distinct-PMT, and charge distributions of delayed clusters."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_csv", type=Path)
    parser.add_argument("--clusters-csv", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--bins", type=int, default=30)
    args = parser.parse_args()
    if args.bins < 1:
        parser.error("--bins must be positive")
    clusters_path = args.clusters_csv or args.events_csv.with_name("clusters.csv")
    output = args.output or args.events_csv.with_name("delayed_cluster_hits.png")
    labels = {}
    with args.events_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            labels[(row["input_file"], row["event_index"])] = row["primary_pid"]
    values: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    with clusters_path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["rank"] != "1":
                continue
            group = labels.get((row["input_file"], row["event_index"]), "unknown")
            for key in ("n_hits", "n_pmts", "charge"):
                values[group][key].append(float(row[key]))
    if not values:
        raise ValueError("No rank-1 delayed clusters found")
    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    display = (("n_hits", "Digit hits in delayed cluster"),
               ("n_pmts", "Distinct PMTs in delayed cluster"),
               ("charge", "Delayed-cluster charge (input units)"))
    colors = {"211": "#1f77b4", "-211": "#d95f02", "22": "#3a923a"}
    names = {"211": r"$\pi^+$", "-211": r"$\pi^-$", "22": r"$\gamma$"}
    for axis, (key, xlabel) in zip(axes, display):
        maximum = max(max(group[key]) for group in values.values() if group[key])
        edges = np.linspace(0, max(1.0, maximum * 1.01), args.bins + 1)
        for group in sorted(values):
            data = values[group][key]
            axis.hist(data, bins=edges, histtype="step", linewidth=1.8,
                      color=colors.get(group, "#555555"),
                      label=f"{names.get(group, group)} (n={len(data)})")
        axis.set_xlabel(xlabel)
        axis.set_ylabel("Candidates / bin")
        axis.grid(alpha=0.2)
        axis.legend()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)
    print(f"Plot: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
