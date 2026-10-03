#!/usr/bin/env python3
"""Audit rank-1 delayed clusters against WCSim muon-decay truth.

The five-bin 'purity' is a *timing-match fraction*, not hit-level Michel
purity: track_parent is a parent PDG code and digit ancestry is unavailable.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_TRUTH = ("track_pid", "track_parent", "track_start_time")
EVENT_COLUMNS = (
    "input_file", "event_index", "primary_pid", "truth_status", "muon_birth_ns",
    "positron_birth_ns", "truth_lifetime_ns", "truth_delay_from_primary_ns",
    "reco_delay_ns", "timing_matched", "time_residual_ns", "delay_difference_ns",
)
BIN_COLUMNS = (
    "bin_start_ns", "bin_end_ns", "truth_decays", "truth_matched",
    "efficiency", "efficiency_low_68", "efficiency_high_68",
    "reco_candidates", "reco_timing_matched", "timing_match_fraction",
    "timing_match_low_68", "timing_match_high_68",
)


def wilson_68(successes: int, trials: int) -> tuple[float | None, float | None, float | None]:
    """Binomial Wilson interval with z=1; undefined for an empty denominator."""
    if trials == 0:
        return None, None, None
    p = successes / trials
    z2 = 1.0
    denominator = 1.0 + z2 / trials
    center = (p + z2 / (2 * trials)) / denominator
    half = np.sqrt(p * (1 - p) / trials + z2 / (4 * trials * trials)) / denominator
    # Roundoff at p=0 or p=1 can otherwise put an endpoint infinitesimally
    # beyond the observed fraction, which Matplotlib rejects as negative yerr.
    return p, min(p, max(0.0, center - half)), max(p, min(1.0, center + half))


def truth_times(event: dict) -> dict:
    """Find one pion-parented mu+ and its one muon-parented e+.

    WCSim/DataTools parent IDs here are PDG codes, so events with multiple
    possible muons or positrons are excluded rather than assigned a lineage.
    """
    if any(name not in event for name in REQUIRED_TRUTH):
        return {"status": "missing_truth"}
    pids = np.asarray(event["track_pid"], dtype=int).reshape(-1)
    parents = np.asarray(event["track_parent"], dtype=int).reshape(-1)
    times = np.asarray(event["track_start_time"], dtype=float).reshape(-1)
    if not (len(pids) == len(parents) == len(times)) or not len(times):
        return {"status": "invalid_truth_shapes"}
    primary = (parents == 0) & np.isfinite(times)
    if not np.any(primary):
        return {"status": "no_primary_time"}
    reference = float(np.min(times[primary]))
    muons = np.flatnonzero((pids == -13) & (parents == 211) & np.isfinite(times))
    positrons = np.flatnonzero((pids == -11) & (parents == -13) & np.isfinite(times))
    if len(muons) > 1 or len(positrons) > 1:
        return {"status": "ambiguous_lineage"}
    if len(muons) == 0 or len(positrons) == 0:
        return {"status": "no_unique_michel"}
    muon_birth = float(times[muons[0]])
    positron_birth = float(times[positrons[0]])
    lifetime = positron_birth - muon_birth
    if lifetime <= 0:
        return {"status": "invalid_truth_lifetime"}
    return {
        "status": "unique_michel", "muon_birth_ns": muon_birth,
        "positron_birth_ns": positron_birth, "truth_lifetime_ns": lifetime,
        "truth_delay_from_primary_ns": positron_birth - reference,
    }


def bin_index(value: float, edges: np.ndarray) -> int | None:
    """Half-open bins, matching the lifetime fit's excluded upper edge."""
    if not np.isfinite(value) or value < edges[0] or value >= edges[-1]:
        return None
    return int(np.searchsorted(edges, value, side="right") - 1)


def analyze(events_csv: Path, edges: np.ndarray, tolerance_ns: float, pid: int) -> tuple[list[dict], list[dict], dict, list[float]]:
    by_file: dict[Path, list[dict]] = defaultdict(list)
    with events_csv.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"input_file", "event_index", "primary_pid", "delta_t_ns"}
        if not required.issubset(reader.fieldnames or []):
            raise ValueError(f"events.csv needs columns: {', '.join(sorted(required))}")
        for row in reader:
            if int(row["primary_pid"]) == pid:
                source = Path(row["input_file"])
                if not source.is_absolute() and not source.exists():
                    source = events_csv.parent / source
                by_file[source].append(row)
    if not by_file:
        raise ValueError(f"No primary PID {pid} rows in {events_csv}")

    truth_n = np.zeros(len(edges) - 1, dtype=int)
    truth_matched = np.zeros_like(truth_n)
    reco_n = np.zeros_like(truth_n)
    reco_matched = np.zeros_like(truth_n)
    rows: list[dict] = []
    statuses: Counter[str] = Counter()
    lifetimes: list[float] = []
    for source, group in sorted(by_file.items()):
        with np.load(source, allow_pickle=True) as archive:
            n_available = len(archive["track_pid"]) if "track_pid" in archive.files else 0
            fields = {name: archive[name] for name in REQUIRED_TRUTH if name in archive.files}
            for source_row in group:
                index = int(source_row["event_index"])
                if fields and not 0 <= index < n_available:
                    raise IndexError(f"{source}: event_index {index} outside NPZ")
                truth = truth_times({name: array[index] for name, array in fields.items()})
                status = truth["status"]
                statuses[status] += 1
                reco = float(source_row["delta_t_ns"]) if source_row["delta_t_ns"] else None
                record = {name: "" for name in EVENT_COLUMNS}
                record.update(input_file=str(source), event_index=index, primary_pid=pid,
                              truth_status=status, reco_delay_ns=reco if reco is not None else "",
                              timing_matched="", time_residual_ns="", delay_difference_ns="")
                for key in ("muon_birth_ns", "positron_birth_ns", "truth_lifetime_ns",
                            "truth_delay_from_primary_ns"):
                    if key in truth:
                        record[key] = truth[key]
                # Missing/ambiguous truth is unknown, never counted as a false tag.
                classifiable = status in {"unique_michel", "no_unique_michel"}
                matched = False
                if status == "unique_michel":
                    truth_delay = truth["truth_delay_from_primary_ns"]
                    residual = reco - truth_delay if reco is not None else None
                    difference = abs(residual) if residual is not None else None
                    matched = difference is not None and difference <= tolerance_ns
                    record["time_residual_ns"] = residual if residual is not None else ""
                    record["delay_difference_ns"] = difference if difference is not None else ""
                    lifetimes.append(truth["truth_lifetime_ns"])
                    truth_bin = bin_index(truth_delay, edges)
                    if truth_bin is not None:
                        truth_n[truth_bin] += 1
                        truth_matched[truth_bin] += int(matched)
                if classifiable:
                    record["timing_matched"] = int(matched)
                    if reco is not None:
                        reco_bin = bin_index(reco, edges)
                        if reco_bin is not None:
                            reco_n[reco_bin] += 1
                            reco_matched[reco_bin] += int(matched)
                rows.append(record)
        print(f"{source.name}: {len(group)} events audited", flush=True)

    bins: list[dict] = []
    for i in range(len(truth_n)):
        eff, eff_lo, eff_hi = wilson_68(int(truth_matched[i]), int(truth_n[i]))
        purity, purity_lo, purity_hi = wilson_68(int(reco_matched[i]), int(reco_n[i]))
        bins.append(dict(zip(BIN_COLUMNS, (
            float(edges[i]), float(edges[i + 1]), int(truth_n[i]), int(truth_matched[i]),
            eff, eff_lo, eff_hi, int(reco_n[i]), int(reco_matched[i]),
            purity, purity_lo, purity_hi,
        ))))
    summary = {
        "input_events_csv": str(events_csv), "primary_pid": pid,
        "bin_edges_ns": edges.tolist(), "match_tolerance_ns": tolerance_ns,
        "counts": {"events": len(rows), "truth_status": dict(statuses),
                   "unique_michel_lifetimes": len(lifetimes),
                   "truth_decays_in_range": int(truth_n.sum()),
                   "reco_candidates_in_range_with_classifiable_truth": int(reco_n.sum())},
        "truth_lifetime_mean_ns_descriptive": float(np.mean(lifetimes)) if lifetimes else None,
        "caveat": "Timing-match fraction is not hit-level purity. Efficiency is for the rank-1 cluster, "
                  "relative to uniquely identified pion-parented mu+ / muon-parented e+ tracks; "
                  "truth_delay is relative to earliest primary track, while reconstructed delay is "
                  "relative to prompt PMT light. Check timing origins and readout coverage.",
    }
    all_residuals = np.asarray([row["time_residual_ns"] for row in rows
                                if row["time_residual_ns"] != ""], dtype=float)
    matched_residuals = all_residuals[np.abs(all_residuals) <= tolerance_ns]
    summary["time_residuals"] = {
        "definition": "(rank-1 cluster time - prompt PMT time) - "
                      "(truth positron time - earliest primary track time)",
        "n_unique_truth_with_rank1": int(all_residuals.size),
        "n_within_match_tolerance": int(matched_residuals.size),
        "median_ns_all_pairs": float(np.median(all_residuals)) if all_residuals.size else None,
        "median_ns_matched": float(np.median(matched_residuals)) if matched_residuals.size else None,
        "std_ns_matched": float(np.std(matched_residuals, ddof=1)) if matched_residuals.size > 1 else None,
        "note": "The matched subset is cut at the stated tolerance: its width is not an "
                "unbiased detector resolution. Residual includes different time origins and "
                "uncorrected light travel times; far tails can be wrong rank-1 clusters.",
    }
    return rows, bins, summary, lifetimes


def plot_bins(bins: list[dict], output: Path) -> None:
    centers = np.array([(row["bin_start_ns"] + row["bin_end_ns"]) / 2 for row in bins])
    widths = np.array([row["bin_end_ns"] - row["bin_start_ns"] for row in bins])
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), sharex=True, constrained_layout=True)
    axes[0].bar(centers - widths * 0.17, [row["truth_decays"] for row in bins],
                width=widths * 0.33, label="Truth Michel decays", color="#4c78a8")
    axes[0].bar(centers + widths * 0.17, [row["reco_candidates"] for row in bins],
                width=widths * 0.33, label="Rank-1 delayed clusters", color="#f58518")
    axes[0].set_ylabel("Events / bin")
    axes[0].legend()
    for key, low, high, label, color in (
        ("efficiency", "efficiency_low_68", "efficiency_high_68", "Rank-1 efficiency", "#4c78a8"),
        ("timing_match_fraction", "timing_match_low_68", "timing_match_high_68",
         "Timing-match fraction (purity proxy)", "#f58518"),
    ):
        valid = [i for i, row in enumerate(bins) if row[key] is not None]
        if valid:
            y = np.array([bins[i][key] for i in valid])
            lower = np.array([bins[i][low] for i in valid])
            upper = np.array([bins[i][high] for i in valid])
            axes[1].errorbar(centers[valid], y,
                             yerr=[np.maximum(0.0, y - lower), np.maximum(0.0, upper - y)],
                             fmt="o-", capsize=3, color=color, label=label)
    axes[1].set(xlabel="Delay after primary / prompt (ns); see timing-origin caveat",
                ylabel="Fraction", ylim=(-0.05, 1.05))
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Simulated π⁺ delayed-cluster truth audit (five bins)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_truth_lifetimes(lifetimes: list[float], output: Path) -> None:
    fig, axis = plt.subplots(figsize=(9, 4.5), constrained_layout=True)
    edges = np.linspace(0, 10_000, 41)
    observed, _ = np.histogram(lifetimes, bins=edges)
    axis.stairs(observed, edges, color="#4c78a8", linewidth=1.8,
                label=f"Exported unique π⁺→μ⁺→e⁺ truth (n={len(lifetimes)})")
    if lifetimes:
        tau_ns = 2196.9811
        expected = np.exp(-edges[:-1] / tau_ns) - np.exp(-edges[1:] / tau_ns)
        expected /= 1 - np.exp(-edges[-1] / tau_ns)
        expected *= int(observed.sum())
        axis.stairs(expected, edges, color="#e45756", linestyle="--", linewidth=1.6,
                    label="2.197 µs exponential reference, normalized in range")
    axis.set(xlabel="μ⁺ creation to e⁺ creation in WCSim truth (ns)",
             ylabel="Truth decays / 250 ns", title="Generated-track muon decay times")
    axis.grid(alpha=0.2)
    axis.legend()
    fig.savefig(output, dpi=180)
    plt.close(fig)


def plot_time_residuals(rows: list[dict], tolerance_ns: float, output: Path) -> None:
    """Show all candidate/truth pairs, not only pairs passing the match cut."""
    residuals = np.asarray([row["time_residual_ns"] for row in rows
                            if row["time_residual_ns"] != ""], dtype=float)
    fig, axes = plt.subplots(2, 1, figsize=(9, 7), constrained_layout=True)
    if residuals.size:
        extent = max(1000.0, float(np.ceil(np.max(np.abs(residuals)) / 1000.0) * 1000.0))
        axes[0].hist(residuals, bins=np.linspace(-extent, extent, 101),
                     color="#4c78a8", histtype="step", linewidth=1.8)
        zoom = max(200.0, 2 * tolerance_ns)
        axes[1].hist(residuals, bins=np.linspace(-zoom, zoom, 81),
                     color="#4c78a8", histtype="step", linewidth=1.8)
        for axis in axes:
            axis.axvline(0, color="black", linewidth=0.9)
            axis.axvline(-tolerance_ns, color="#e45756", linestyle="--", linewidth=1,
                         label=f"±{tolerance_ns:g} ns match cut")
            axis.axvline(tolerance_ns, color="#e45756", linestyle="--", linewidth=1)
        axes[1].legend()
    else:
        for axis in axes:
            axis.text(0.5, 0.5, "No rank-1 candidates with unique Michel truth",
                      transform=axis.transAxes, ha="center", va="center")
    axes[0].set(title=f"All unique-truth / rank-1 pairs (n={len(residuals)}); full range",
                ylabel="Candidates / bin")
    axes[1].set(title="Core close-up; includes pairs outside the match cut",
                xlabel="Reconstructed prompt-to-cluster delay − truth primary-to-e⁺ delay (ns)",
                ylabel="Candidates / bin")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle("Delayed-cluster timing residual (different origins; no photon-TOF correction)")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_csv", type=Path, help="events.csv from study_delayed_clusters.py")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--min-ns", type=float, default=1000.0)
    parser.add_argument("--max-ns", type=float, default=6100.0)
    parser.add_argument("--bins", type=int, default=5)
    parser.add_argument("--match-tolerance-ns", type=float, default=100.0)
    parser.add_argument("--primary-pid", type=int, default=211)
    args = parser.parse_args()
    if args.bins < 1 or args.max_ns <= args.min_ns or args.match_tolerance_ns <= 0:
        parser.error("Require positive bins/tolerance and --max-ns > --min-ns")
    output = args.output_dir or args.events_csv.parent / "michel_truth_audit"
    output.mkdir(parents=True, exist_ok=True)
    edges = np.linspace(args.min_ns, args.max_ns, args.bins + 1)
    events, bins, summary, lifetimes = analyze(args.events_csv, edges, args.match_tolerance_ns, args.primary_pid)
    for filename, columns, records in (
        ("events_truth_audit.csv", EVENT_COLUMNS, events),
        ("delay_bin_metrics.csv", BIN_COLUMNS, bins),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    plot_bins(bins, output / "delay_bin_metrics.png")
    plot_truth_lifetimes(lifetimes, output / "truth_muon_lifetimes.png")
    plot_time_residuals(events, args.match_tolerance_ns, output / "time_residuals.png")
    for row in bins:
        print(f"{row['bin_start_ns']:.0f}–{row['bin_end_ns']:.0f} ns: "
              f"efficiency {row['truth_matched']}/{row['truth_decays']}; "
              f"timing-match {row['reco_timing_matched']}/{row['reco_candidates']}")
    if lifetimes:
        print(f"Available unique muon truth lifetimes: n={len(lifetimes)}, "
              f"mean={np.mean(lifetimes):.0f} ns (descriptive, not acceptance-corrected)")
    residual_summary = summary["time_residuals"]
    print(f"Timing residuals: {residual_summary['n_unique_truth_with_rank1']} rank-1/truth pairs; "
          f"{residual_summary['n_within_match_tolerance']} within ±{args.match_tolerance_ns:g} ns")
    print(f"Results: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
