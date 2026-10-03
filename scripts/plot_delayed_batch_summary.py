#!/usr/bin/env python3
"""Sum delayed-candidate histograms over completed tagged-gamma batch folders."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def _finite(value: str, *, column: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid {column} in {path}: {value!r}") from exc
    if not math.isfinite(number):
        raise ValueError(f"Nonfinite {column} in {path}: {value!r}")
    return number


def collect_batch(batch_dir: Path) -> dict:
    """Read only finished jobs and retain one record per best delayed cluster."""
    if not batch_dir.is_dir():
        raise ValueError(f"Batch output directory does not exist: {batch_dir}")
    combined = {"folders": [], "event_count": 0, "candidate_count": 0,
                "delays_ns": [], "n_hits": [], "n_pmts": [], "charge": []}
    for folder in sorted(path for path in batch_dir.iterdir() if path.is_dir()):
        events = folder / "clusters" / "events.csv"
        clusters = folder / "clusters" / "clusters.csv"
        if not (folder / "analysis.done").is_file() or not events.is_file() or not clusters.is_file():
            continue
        n_events = n_candidates = n_clusters = 0
        with events.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                n_events += 1
                if row.get("delta_t_ns"):
                    combined["delays_ns"].append(_finite(row["delta_t_ns"], column="delta_t_ns", path=events))
                    n_candidates += 1
        with clusters.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                if row.get("rank") != "1":
                    continue
                for key in ("n_hits", "n_pmts", "charge"):
                    combined[key].append(_finite(row.get(key, ""), column=key, path=clusters))
                n_clusters += 1
        if n_candidates != n_clusters:
            raise ValueError(f"Best-cluster count differs from candidate count in {folder}: "
                             f"{n_clusters} versus {n_candidates}")
        combined["folders"].append({"folder": folder.name, "events": n_events,
                                    "candidates": n_candidates})
        combined["event_count"] += n_events
        combined["candidate_count"] += n_candidates
    if not combined["folders"]:
        raise ValueError(f"No completed batch folders with CSVs found in {batch_dir}")
    return combined


def plot_summary(batch_dir: Path, output_dir: Path, *, min_ns: float = 200.0,
                 max_ns: float = 10_000.0, bin_ns: float = 200.0,
                 size_bins: int = 30) -> dict:
    if not (math.isfinite(min_ns) and math.isfinite(max_ns) and math.isfinite(bin_ns)):
        raise ValueError("Timing range and bin size must be finite")
    if max_ns <= min_ns or bin_ns <= 0 or size_bins < 1:
        raise ValueError("Require max_ns > min_ns, bin_ns > 0, and size_bins >= 1")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    data = collect_batch(batch_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    edges = np.arange(min_ns, max_ns, bin_ns, dtype=float)
    edges = np.append(edges, max_ns)
    delays = np.asarray(data["delays_ns"], dtype=float)
    time_counts, _ = np.histogram(delays, bins=edges)
    fig, ax = plt.subplots(figsize=(10, 4.5), constrained_layout=True)
    ax.stairs(time_counts, edges, linewidth=1.8, color="#2878a5")
    ax.set(xlabel="Best delayed-cluster time minus prompt-cluster time (ns)",
           ylabel="Candidates / bin",
           title=(f"Tagged-gamma delayed candidates: {data['candidate_count']:,} / "
                  f"{data['event_count']:,} events across {len(data['folders'])} files"),
           xlim=(min_ns, max_ns))
    ax.grid(alpha=0.2)
    time_path = output_dir / "combined_delayed_time_histogram.png"
    fig.savefig(time_path, dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(3, 1, figsize=(8, 10), constrained_layout=True)
    labels = (("n_hits", "Digit hits in delayed cluster"),
              ("n_pmts", "Distinct PMTs in delayed cluster"),
              ("charge", "Delayed-cluster charge (input units)"))
    for ax, (key, xlabel) in zip(axes, labels):
        values = np.asarray(data[key], dtype=float)
        upper = max(1.0, float(values.max()) * 1.01) if values.size else 1.0
        hist_edges = np.linspace(0, upper, size_bins + 1)
        counts, _ = np.histogram(values, bins=hist_edges)
        ax.stairs(counts, hist_edges, linewidth=1.8, color="#2878a5")
        if not values.size:
            ax.text(0.5, 0.5, "No delayed-cluster candidates", transform=ax.transAxes,
                    ha="center", va="center")
        ax.set(xlabel=xlabel, ylabel="Candidates / bin")
        ax.grid(alpha=0.2)
    fig.suptitle(f"Tagged-gamma delayed-cluster size ({data['candidate_count']:,} candidates)")
    size_path = output_dir / "combined_delayed_cluster_hits.png"
    fig.savefig(size_path, dpi=180)
    plt.close(fig)

    summary = {"batch_dir": str(batch_dir), "completed_folders": len(data["folders"]),
               "events": data["event_count"], "candidates": data["candidate_count"],
               "candidates_in_time_plot": int(time_counts.sum()),
               "time_range_ns": [min_ns, max_ns], "time_bin_ns": bin_ns,
               "size_bins": size_bins, "files": data["folders"],
               "plots": [str(time_path), str(size_path)],
               "note": "Raw summed candidate counts; no lifetime fit or per-file normalization."}
    (output_dir / "combined_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_dir", type=Path, help="Parent of completed per-NPZ output folders")
    parser.add_argument("--output-dir", type=Path, default=None,
                        help="Defaults to the batch directory")
    parser.add_argument("--min-ns", type=float, default=200.0)
    parser.add_argument("--max-ns", type=float, default=10_000.0)
    parser.add_argument("--bin-ns", type=float, default=200.0)
    parser.add_argument("--size-bins", type=int, default=30)
    args = parser.parse_args()
    summary = plot_summary(args.batch_dir, args.output_dir or args.batch_dir,
                           min_ns=args.min_ns, max_ns=args.max_ns,
                           bin_ns=args.bin_ns, size_bins=args.size_bins)
    print(f"Summed {summary['events']:,} events and {summary['candidates']:,} candidates "
          f"from {summary['completed_folders']} completed folders")
    for path in summary["plots"]:
        print(f"Plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
