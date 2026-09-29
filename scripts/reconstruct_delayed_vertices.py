#!/usr/bin/env python3
"""Point-multilaterate rank-1 delayed clusters and plot their WCTE positions."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LicketyFit.MichelVertex import fit_delayed_point_vertex, match_michel_truth  # noqa: E402
from LicketyFit.TimeClustering import event_hit_times  # noqa: E402
from plot_delayed_vertex_residuals import plot_residuals  # noqa: E402


def load_pmt_positions(mapping_file: Path) -> dict[int, np.ndarray]:
    """Map zero-based DataTools digit PMT IDs to WCSim positions in mm."""
    table = np.atleast_2d(np.loadtxt(mapping_file))
    if table.shape[1] < 6:
        raise ValueError("Expected raw PMT ID and x,y,z coordinates in mapping table")
    # The mapping's first column is one-based, matching the existing WCSim
    # driver offset of +1. Its position columns are WCSim centimetres.
    return {int(row[0]) - 1: np.asarray(row[3:6] * 10.0, dtype=float) for row in table}


def plot_vertices(rows: list[dict], output: Path) -> None:
    accepted = [row for row in rows if row["status"] == "ok"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    colors = {"211": "#1f77b4", "-211": "#d95f02", "22": "#3a923a"}
    labels = {"211": r"$\pi^+$", "-211": r"$\pi^-$", "22": r"$\gamma$"}
    for pid in sorted({str(row["primary_pid"]) for row in accepted}):
        group = [row for row in accepted if str(row["primary_pid"]) == pid]
        color = colors.get(pid, "#555555")
        axes[0].scatter([row["y_mm"] / 10 for row in group],
                        [row["r_tank_mm"] / 10 for row in group],
                        s=22, alpha=0.6, color=color, label=f"{labels.get(pid, pid)} (n={len(group)})")
        axes[1].scatter([row["z_mm"] / 10 for row in group],
                        [row["x_mm"] / 10 for row in group],
                        s=22, alpha=0.6, color=color, label=f"{labels.get(pid, pid)} (n={len(group)})")
    axes[0].set(xlabel="Tank vertical coordinate y (cm)", ylabel=r"Tank radius $R=\sqrt{x^2+z^2}$ (cm)")
    axes[1].set(xlabel="Beam coordinate z (cm)", ylabel="Horizontal coordinate x (cm)")
    for axis in axes:
        axis.grid(alpha=0.2)
        if accepted:
            axis.legend(loc="best")
    fig.suptitle("Effective origins of delayed light (point multilateration)")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_truth_overlay(rows: list[dict], output: Path) -> None:
    matched = [row for row in rows if row["status"] == "ok" and row.get("truth_status") == "matched"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), constrained_layout=True)
    for row in matched:
        axes[0].plot([row["truth_y_mm"] / 10, row["y_mm"] / 10],
                     [row["truth_r_tank_mm"] / 10, row["r_tank_mm"] / 10],
                     color="#888888", alpha=0.5, linewidth=0.8, zorder=1)
        axes[1].plot([row["truth_z_mm"] / 10, row["z_mm"] / 10],
                     [row["truth_x_mm"] / 10, row["x_mm"] / 10],
                     color="#888888", alpha=0.5, linewidth=0.8, zorder=1)
    if matched:
        axes[0].scatter([row["truth_y_mm"] / 10 for row in matched],
                        [row["truth_r_tank_mm"] / 10 for row in matched],
                        marker="x", s=38, color="#d62728", label="Muon-decay e± truth", zorder=3)
        axes[0].scatter([row["y_mm"] / 10 for row in matched],
                        [row["r_tank_mm"] / 10 for row in matched],
                        marker="o", s=24, facecolors="none", edgecolors="#1f77b4",
                        label="Reconstructed light origin", zorder=3)
        axes[1].scatter([row["truth_z_mm"] / 10 for row in matched],
                        [row["truth_x_mm"] / 10 for row in matched],
                        marker="x", s=38, color="#d62728", zorder=3)
        axes[1].scatter([row["z_mm"] / 10 for row in matched],
                        [row["x_mm"] / 10 for row in matched],
                        marker="o", s=24, facecolors="none", edgecolors="#1f77b4", zorder=3)
        axes[0].legend(loc="best")
    else:
        axes[0].text(0.5, 0.5, "No time-matched muon-decay e± truth tracks",
                     transform=axes[0].transAxes, ha="center", va="center")
    axes[0].set(xlabel="Tank vertical coordinate y (cm)", ylabel=r"Tank radius $R=\sqrt{x^2+z^2}$ (cm)")
    axes[1].set(xlabel="Beam coordinate z (cm)", ylabel="Horizontal coordinate x (cm)")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(f"Delayed light: reconstructed versus Michel truth (n={len(matched)})")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("clusters_csv", type=Path, help="clusters.csv from study_delayed_clusters.py")
    parser.add_argument("--mapping-file", type=Path, default=ROOT / "tables/wcsim_wcte_mapping.txt")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--max-candidates", type=int, default=None)
    parser.add_argument("--min-pmts", type=int, default=12)
    parser.add_argument("--min-time-ns", type=float, default=-1_000_000.0)
    parser.add_argument("--time-mode", choices=("auto", "raw", "trigger-plus"), default="auto")
    parser.add_argument("--truth-match-tolerance-ns", type=float, default=100.0)
    args = parser.parse_args()
    if args.max_candidates is not None and args.max_candidates < 1:
        parser.error("--max-candidates must be positive")
    if args.truth_match_tolerance_ns <= 0:
        parser.error("--truth-match-tolerance-ns must be positive")
    output_dir = args.output_dir or args.clusters_csv.parent / "vertex_reconstruction"
    output_dir.mkdir(parents=True, exist_ok=True)
    positions = load_pmt_positions(args.mapping_file)
    by_file: dict[Path, list[dict]] = defaultdict(list)
    with args.clusters_csv.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["rank"] == "1":
                by_file[Path(row["input_file"])].append(row)
    candidates = [(source, row) for source, group in sorted(by_file.items()) for row in group]
    if args.max_candidates is not None:
        candidates = candidates[:args.max_candidates]
    if not candidates:
        print("No rank-1 delayed clusters found; writing empty diagnostic outputs.")
    by_file.clear()
    for source, row in candidates:
        by_file[source].append(row)

    output_rows = []
    status_counts: Counter[str] = Counter()
    for source, group in sorted(by_file.items()):
        with np.load(source, allow_pickle=True) as archive:
            fields = {name: archive[name] for name in (
                "digi_hit_time", "digi_hit_pmt", "digi_hit_charge", "digi_hit_trigger", "trigger_time", "pid",
                "track_pid", "track_parent", "track_start_time", "track_start_position"
            ) if name in archive.files}
            for candidate in group:
                index = int(candidate["event_index"])
                event = {name: array[index] for name, array in fields.items()}
                row = {"input_file": str(source), "event_index": index,
                       "primary_pid": int(event["pid"]) if "pid" in event else "",
                       "delta_t_ns": float(candidate["delta_t_ns"]),
                       "cluster_n_hits": int(candidate["n_hits"]),
                       "cluster_n_pmts": int(candidate["n_pmts"]),
                       "cluster_charge": float(candidate["charge"])}
                try:
                    times, mode = event_hit_times(event, args.time_mode)
                    pmts = np.asarray(event["digi_hit_pmt"], dtype=int).reshape(-1)
                    charges = np.asarray(event["digi_hit_charge"], dtype=float).reshape(-1)
                    mask = (np.isfinite(times) & (times >= args.min_time_ns)
                            & (times >= float(candidate["start_ns"]))
                            & (times <= float(candidate["end_ns"]))
                            & np.isfinite(charges) & (charges > 0))
                    matched = np.array([int(pmt) in positions for pmt in pmts[mask]], dtype=bool)
                    selected_ids, selected_times, selected_charges = pmts[mask][matched], times[mask][matched], charges[mask][matched]
                    selected_positions = np.asarray([positions[int(pmt)] for pmt in selected_ids], dtype=float).reshape(-1, 3)
                    fit = fit_delayed_point_vertex(selected_positions, selected_ids,
                                                   selected_charges, selected_times,
                                                   min_pmts=args.min_pmts)
                    x, y, z = map(float, fit["vertex_mm"])
                    row.update(status="ok", error="", time_mode=mode,
                               n_mapped_pmts=int(fit["n_window_pmts"]),
                               n_fit_pmts=int(fit["n_selected_pmts"]),
                               x_mm=x, y_mm=y, z_mm=z,
                               r_tank_mm=float(np.hypot(x, z)),
                               t0_ns=float(fit["t0_ns"]),
                               residual_rms_ns=float(fit["residual_rms_ns"]),
                               converged=bool(fit["converged"]),
                               linear_rank=int(fit["linear_rank"]))
                    truth = match_michel_truth(event, row["delta_t_ns"],
                                               tolerance_ns=args.truth_match_tolerance_ns)
                    row["truth_status"] = truth["status"]
                    row["n_truth_candidates"] = truth.get("n_truth_candidates", "")
                    row["truth_track_index"] = truth.get("truth_track_index", "")
                    row["truth_pdg"] = truth.get("truth_pdg", "")
                    row["truth_delay_ns"] = truth.get("truth_delay_ns", "")
                    row["truth_delay_difference_ns"] = truth.get("truth_delay_difference_ns", "")
                    if truth["status"] == "matched":
                        tx, ty, tz = map(float, truth["truth_vertex_mm"])
                        row.update(truth_x_mm=tx, truth_y_mm=ty, truth_z_mm=tz,
                                   truth_r_tank_mm=float(np.hypot(tx, tz)),
                                   reco_truth_distance_3d_mm=float(np.linalg.norm([x - tx, y - ty, z - tz])))
                except (ValueError, RuntimeError, KeyError, IndexError) as exc:
                    row.update(status="failed", error=str(exc))
                output_rows.append(row)
                status_counts[row["status"]] += 1
        print(f"{source.name}: {len(group)} candidates processed", flush=True)

    csv_path = output_dir / "vertices.csv"
    columns = ("input_file", "event_index", "primary_pid", "delta_t_ns", "cluster_n_hits",
               "cluster_n_pmts", "cluster_charge", "status", "error", "time_mode",
               "n_mapped_pmts", "n_fit_pmts", "x_mm", "y_mm", "z_mm",
               "r_tank_mm", "t0_ns", "residual_rms_ns",
               "converged", "linear_rank", "truth_status", "n_truth_candidates",
               "truth_track_index", "truth_pdg", "truth_delay_ns",
               "truth_delay_difference_ns", "truth_x_mm", "truth_y_mm", "truth_z_mm",
               "truth_r_tank_mm", "reco_truth_distance_3d_mm")
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(output_rows)
    plot_path = output_dir / "delayed_vertices_tank_ry.png"
    plot_vertices(output_rows, plot_path)
    overlay_path = output_dir / "delayed_vertices_truth_overlay.png"
    plot_truth_overlay(output_rows, overlay_path)
    residual_path = output_dir / "delayed_vertex_residuals_xyz.png"
    n_residuals = plot_residuals(output_rows, residual_path)
    print(f"Point fits: {dict(status_counts)}")
    print(f"Truth matches: {dict(Counter(row.get('truth_status', 'unavailable') for row in output_rows))}")
    print(f"Results: {csv_path}")
    print(f"Plot: {plot_path}")
    print(f"Truth overlay: {overlay_path}")
    print(f"Reco - truth x/y/z residuals: {residual_path} ({n_residuals} matched events)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
