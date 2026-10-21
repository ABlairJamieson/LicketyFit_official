#!/usr/bin/env python3
"""Compare simple prompt-to-delayed time choices using study CSV outputs.

The three choices are the earliest delayed cluster in time (the next cluster
after the prompt), the cluster with the most digit hits, and the existing
finder's rank-1 cluster (most distinct PMTs, charge as the tie-break).
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PARTICLE_LABELS = {"211": r"$\pi^+$", "-211": r"$\pi^-$", "22": r"$\gamma$"}
CHOICE_COLORS = {"second": "#4c78a8", "largest_hits": "#f58518", "rank1": "#54a24b"}
CHOICES = (
    ("second", "Earliest delayed cluster (second in time)"),
    ("largest_hits", "Largest delayed cluster (digit hits)"),
    ("rank1", "Current finder rank 1 (PMTs, then charge)"),
)


def label_mc(fig) -> None:
    """Mark every plot from the WCSim simulation workflow as MC."""
    fig.text(0.99, 0.995, "MC", ha="right", va="top", fontsize=10,
             color="#1d4ed8", weight="bold")


def load_delay_choices(events_csv: Path, clusters_csv: Path
                       ) -> dict[str, dict[str, list[float]]]:
    """Load one prompt-relative delay per event and selection rule."""
    event_particles: dict[tuple[str, str], str] = {}
    with events_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"input_file", "event_index", "primary_pid"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"events.csv needs columns: {', '.join(sorted(required))}")
        for row in reader:
            key = (row["input_file"], row["event_index"])
            event_particles[key] = row.get("primary_pid", "") or row.get("selection_category", "unknown")

    candidates: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    with clusters_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"input_file", "event_index", "rank", "delta_t_ns", "n_hits", "n_pmts", "charge"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"clusters.csv needs columns: {', '.join(sorted(required))}")
        for row in reader:
            try:
                delay = float(row["delta_t_ns"])
                hits = int(row["n_hits"])
                pmts = int(row["n_pmts"])
                charge = float(row["charge"])
                rank = int(row["rank"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(delay) and np.isfinite(charge):
                row["_delay"] = delay
                row["_hits"] = hits
                row["_pmts"] = pmts
                row["_charge"] = charge
                row["_rank"] = rank
                candidates[(row["input_file"], row["event_index"])].append(row)

    output: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {key: [] for key, _ in CHOICES})
    for key, rows in candidates.items():
        particle = event_particles.get(key, "unknown")
        output[particle]["second"].append(min(rows, key=lambda row: row["_delay"])["_delay"])
        output[particle]["largest_hits"].append(
            max(rows, key=lambda row: (row["_hits"], row["_pmts"], row["_charge"]))["_delay"])
        rank1 = [row for row in rows if row["_rank"] == 1]
        if rank1:
            output[particle]["rank1"].append(rank1[0]["_delay"])
    return dict(output)


def plot_choices(data: dict[str, dict[str, list[float]]], output: Path, *,
                 min_ns: float, max_ns: float, bins: int) -> None:
    edges = np.linspace(min_ns, max_ns, bins + 1)
    particles = sorted(data, key=lambda key: (key not in PARTICLE_LABELS, key))
    fig, axes = plt.subplots(len(particles), 1,
                             figsize=(10, max(4.0, 3.2 * len(particles))),
                             sharex=True, squeeze=False)
    for axis, particle in zip(axes[:, 0], particles):
        for choice, label in CHOICES:
            values = np.asarray(data[particle][choice], dtype=float)
            counts = np.histogram(values, bins=edges)[0]
            axis.stairs(counts, edges, linewidth=1.8,
                        color=CHOICE_COLORS[choice],
                        label=f"{label} (n={int(counts.sum()):,})")
        axis.set_ylabel(f"Events / {(max_ns - min_ns) / bins:g} ns")
        axis.set_title(PARTICLE_LABELS.get(particle, particle), loc="left")
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
    axes[-1, 0].set_xlabel("Selected delayed-cluster time − prompt-cluster time (ns)")
    axes[-1, 0].set_xlim(min_ns, max_ns)
    fig.suptitle("MC — Delayed-time choices: chronological, largest by hits, and current rank 1")
    label_mc(fig)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def load_basic_diagnostics(events_csv: Path, clusters_csv: Path):
    """Load per-event and per-cluster quantities for simple QA plots."""
    events: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"clusters": [], "prompt_hits": [], "prompt_time": []})
    event_particles: dict[tuple[str, str], str] = {}
    with events_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"input_file", "event_index", "primary_pid", "n_delayed_clusters", "prompt_time_ns"}
        if not reader.fieldnames or not required.issubset(reader.fieldnames):
            raise ValueError(f"events.csv needs columns: {', '.join(sorted(required))}")
        for row in reader:
            particle = row.get("primary_pid", "") or row.get("selection_category", "unknown")
            key = (row["input_file"], row["event_index"])
            event_particles[key] = particle
            try:
                ndelayed = int(row["n_delayed_clusters"] or 0)
                prompt_time = float(row["prompt_time_ns"])
            except (TypeError, ValueError):
                ndelayed, prompt_time = 0, np.nan
            events[particle]["clusters"].append(ndelayed + (int(np.isfinite(prompt_time))))
            if np.isfinite(prompt_time):
                events[particle]["prompt_time"].append(prompt_time)
            try:
                prompt_hits = float(row.get("prompt_n_hits", ""))
            except (TypeError, ValueError):
                prompt_hits = np.nan
            if np.isfinite(prompt_hits):
                events[particle]["prompt_hits"].append(prompt_hits)

    clusters: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"hits": [], "delay": []})
    with clusters_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            particle = event_particles.get((row["input_file"], row["event_index"]), "unknown")
            try:
                hits, delay = float(row["n_hits"]), float(row["delta_t_ns"])
            except (TypeError, ValueError):
                continue
            if np.isfinite(hits):
                clusters[particle]["hits"].append(hits)
            if np.isfinite(delay):
                clusters[particle]["delay"].append(delay)
    return dict(events), dict(clusters)


def plot_basic_diagnostics(events, clusters, output_dir: Path, *, bins: int,
                           min_ns: float, max_ns: float) -> list[Path]:
    """Write simple multiplicity, cluster-size, and prompt/delay-time plots."""
    output_dir.mkdir(parents=True, exist_ok=True)
    particles = sorted(set(events) | set(clusters),
                       key=lambda key: (key not in PARTICLE_LABELS, key))
    paths = []

    fig, ax = plt.subplots(figsize=(9, 5))
    for particle in particles:
        values = events.get(particle, {}).get("clusters", [])
        if values:
            edges = np.arange(-0.5, max(values) + 1.5, 1)
            ax.hist(values, bins=edges, histtype="step", linewidth=1.8,
                    label=f"{PARTICLE_LABELS.get(particle, particle)} (n={len(values):,})")
    ax.set(xlabel="Found clusters per event (prompt + delayed candidates)", ylabel="Events",
           title="MC — WCSim cluster multiplicity")
    ax.grid(alpha=.2); ax.legend()
    paths.append(output_dir / "clusters_per_event.png")
    label_mc(fig); fig.tight_layout(); fig.savefig(paths[-1], dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for particle in particles:
        ev = events.get(particle, {})
        cl = clusters.get(particle, {})
        label = PARTICLE_LABELS.get(particle, particle)
        if ev.get("prompt_hits"):
            all_hits = ev["prompt_hits"] + cl.get("hits", [])
            axes[0].hist(all_hits, bins=bins, range=(0, 500), histtype="step",
                         linewidth=1.8, label=f"{label} all (n={len(all_hits):,})")
            axes[1].hist(ev["prompt_hits"], bins=bins, range=(0, 500), histtype="step",
                         linewidth=1.8, label=f"{label} prompt (n={len(ev['prompt_hits']):,})")
        if cl.get("hits"):
            axes[2].hist(cl["hits"], bins=bins, range=(0, 500), histtype="step",
                         linewidth=1.8, label=f"{label} (n={len(cl['hits']):,})")
    axes[0].set(xlabel="Digit hits", ylabel="Clusters / bin", title="MC — All found clusters")
    axes[1].set(xlabel="Digit hits", ylabel="Clusters / bin", title="MC — Prompt cluster")
    axes[2].set(xlabel="Digit hits", ylabel="Clusters / bin", title="MC — Delayed candidates")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    paths.append(output_dir / "cluster_hits_all_prompt_delayed.png")
    label_mc(fig); fig.tight_layout(); fig.savefig(paths[-1], dpi=180); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    edges = np.linspace(min_ns, max_ns, bins + 1)
    for particle in particles:
        ev, cl = events.get(particle, {}), clusters.get(particle, {})
        label = PARTICLE_LABELS.get(particle, particle)
        if ev.get("prompt_time"):
            axes[0].hist(ev["prompt_time"], bins=edges, histtype="step", linewidth=1.8,
                         label=f"{label} (n={len(ev['prompt_time']):,})")
        if cl.get("delay"):
            axes[1].hist(cl["delay"], bins=edges, histtype="step", linewidth=1.8,
                         label=f"{label} (n={len(cl['delay']):,})")
    axes[0].set(xlabel="Prompt-cluster center on selected event time convention (ns)",
                ylabel="Events / bin", title="MC — Prompt cluster times")
    axes[1].set(xlabel="Delayed-cluster center − prompt center (ns)",
                ylabel="Candidates / bin", title="MC — Delayed candidate times")
    for ax in axes:
        ax.grid(alpha=.2); ax.legend(fontsize=8)
    paths.append(output_dir / "prompt_and_delayed_cluster_times.png")
    label_mc(fig); fig.tight_layout(); fig.savefig(paths[-1], dpi=180); plt.close(fig)
    return paths


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_csv", type=Path, help="events.csv from study_delayed_clusters.py")
    parser.add_argument("--clusters-csv", type=Path, default=None,
                        help="clusters.csv (defaults to sibling of events.csv)")
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument("--min-ns", type=float, default=100.0)
    parser.add_argument("--max-ns", type=float, default=6100.0)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--diagnostic-bins", type=int, default=50,
                        help="Bin count for hit-size and time diagnostic plots")
    args = parser.parse_args()
    if args.bins < 1 or args.max_ns <= args.min_ns:
        parser.error("Require --bins >= 1 and --max-ns > --min-ns")
    clusters_csv = args.clusters_csv or args.events_csv.with_name("clusters.csv")
    output = args.output or args.events_csv.with_name("delayed_time_selection_comparison.png")
    data = load_delay_choices(args.events_csv, clusters_csv)
    if not data:
        raise ValueError("No delayed candidates found in clusters.csv")
    plot_choices(data, output, min_ns=args.min_ns, max_ns=args.max_ns, bins=args.bins)
    events, clusters = load_basic_diagnostics(args.events_csv, clusters_csv)
    diagnostic_paths = plot_basic_diagnostics(
        events, clusters, output.parent, bins=args.diagnostic_bins,
        min_ns=args.min_ns, max_ns=args.max_ns)
    for particle, choices in sorted(data.items()):
        print(f"{particle}: " + "; ".join(
            f"{label}={len(choices[key])}" for key, label in CHOICES))
    print(f"Plot: {output}")
    for path in diagnostic_paths:
        print(f"Plot: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
