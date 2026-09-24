#!/usr/bin/env python3
"""Validate basic charge/time distributions in a selected WCSim NPZ sample."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--input",
        type=Path,
        default=None,
        help="Compact NPZ; defaults to fit_input_file in the manifest.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/pion_shower_observable_checks"),
    )
    return parser.parse_args()


def fitter_window(times: np.ndarray) -> tuple[np.ndarray, float, float]:
    """Reproduce fit_single_event._apply_wcsim_peak_window exactly."""
    if times.size == 0:
        return np.zeros(0, dtype=bool), float("nan"), float("nan")
    counts, edges = np.histogram(times, bins=np.arange(0, 2000))
    peak_index = int(np.argmax(counts))
    cut_index = min(peak_index + 5, len(edges) - 1)
    cut_time = float(edges[cut_index])
    peak_center = float(0.5 * (edges[peak_index] + edges[peak_index + 1]))
    return (times > 0.0) & (times < cut_time), peak_center, cut_time


def weighted_mean_rms(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    total = float(np.sum(weights))
    if not values.size or total <= 0.0:
        return float("nan"), float("nan")
    mean = float(np.sum(values * weights) / total)
    rms = float(np.sqrt(np.sum(weights * (values - mean) ** 2) / total))
    return mean, rms


def finite_summary(values: list[float]) -> dict[str, float | int | None]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    if not array.size:
        return {"n": 0, "mean": None, "median": None, "q10": None, "q90": None}
    return {
        "n": int(array.size),
        "mean": float(np.mean(array)),
        "median": float(np.median(array)),
        "q10": float(np.quantile(array, 0.1)),
        "q90": float(np.quantile(array, 0.9)),
    }


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = sorted(
            csv.DictReader(handle), key=lambda row: int(row["fit_event_index"])
        )
    if not rows:
        raise RuntimeError("Manifest contains no events")
    input_path = args.input or Path(rows[0]["fit_input_file"])
    args.output_dir.mkdir(parents=True, exist_ok=True)

    metrics: list[dict[str, object]] = []
    hit_charge: dict[str, list[np.ndarray]] = defaultdict(list)
    hit_relative_time: dict[str, list[np.ndarray]] = defaultdict(list)
    with np.load(input_path, allow_pickle=True) as data:
        for row in rows:
            index = int(row["fit_event_index"])
            category = row["selection_category"]
            pmt = np.asarray(data["digi_hit_pmt"][index], dtype=int)
            charge = np.asarray(data["digi_hit_charge"][index], dtype=float)
            time = np.asarray(data["digi_hit_time"][index], dtype=float)
            if not (len(pmt) == len(charge) == len(time)):
                raise ValueError(f"Event {index} has inconsistent digit arrays")
            finite = np.isfinite(charge) & np.isfinite(time) & (charge >= 0.0)
            pmt, charge, time = pmt[finite], charge[finite], time[finite]
            keep, peak, cut_time = fitter_window(time)
            kept_charge, kept_time, kept_pmt = charge[keep], time[keep], pmt[keep]
            raw_total = float(np.sum(charge))
            selected_total = float(np.sum(kept_charge))
            time_mean, time_rms = weighted_mean_rms(kept_time, kept_charge)
            positive_charge = kept_charge[kept_charge > 0.0]
            metrics.append(
                {
                    "fit_event_index": index,
                    "source_event_index": int(row["event_index"]),
                    "selection_category": category,
                    "primary_energy_mev": float(row["primary_energy_mev"]),
                    "n_raw_digits": int(len(charge)),
                    "n_selected_digits": int(np.count_nonzero(keep)),
                    "n_raw_unique_pmts": int(np.unique(pmt).size),
                    "n_selected_unique_pmts": int(np.unique(kept_pmt).size),
                    "raw_total_charge": raw_total,
                    "selected_total_charge": selected_total,
                    "selected_charge_fraction": selected_total / raw_total if raw_total > 0 else None,
                    "selected_digit_fraction": float(np.mean(keep)) if len(keep) else None,
                    "mean_charge_per_selected_digit": (
                        float(np.mean(positive_charge)) if positive_charge.size else None
                    ),
                    "median_charge_per_selected_digit": (
                        float(np.median(positive_charge)) if positive_charge.size else None
                    ),
                    "q90_charge_per_selected_digit": (
                        float(np.quantile(positive_charge, 0.9)) if positive_charge.size else None
                    ),
                    "raw_time_min_ns": float(np.min(time)) if time.size else None,
                    "raw_time_max_ns": float(np.max(time)) if time.size else None,
                    "modal_time_ns": peak,
                    "fitter_cut_time_ns": cut_time,
                    "selected_time_mean_charge_weighted_ns": time_mean,
                    "selected_time_rms_charge_weighted_ns": time_rms,
                    "selected_time_iqr_ns": (
                        float(np.quantile(kept_time, 0.75) - np.quantile(kept_time, 0.25))
                        if kept_time.size else None
                    ),
                }
            )
            hit_charge[category].append(positive_charge)
            hit_relative_time[category].append(kept_time - peak)

    csv_path = args.output_dir / "event_observables.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    summary_fields = (
        "n_selected_unique_pmts",
        "selected_total_charge",
        "selected_charge_fraction",
        "mean_charge_per_selected_digit",
        "selected_time_rms_charge_weighted_ns",
        "selected_time_iqr_ns",
    )
    categories = list(dict.fromkeys(row["selection_category"] for row in rows))
    summary = {
        "input_file": str(input_path),
        "n_events": len(metrics),
        "fitter_preprocessing": (
            "Keep WCSim digits with 0 < time < five 1-ns bins beyond the modal bin; "
            "sum charge and form one charge-weighted mean time per PMT."
        ),
        "charge_model_caution": (
            "Track charge uses per-event total normalization and shower total PE floats; "
            "these fits currently compare charge shape, not absolute light yield."
        ),
        "categories": {
            category: {
                field: finite_summary(
                    [float(row[field]) for row in metrics if row["selection_category"] == category and row[field] is not None]
                )
                for field in summary_fields
            }
            for category in categories
        },
    }
    with (args.output_dir / "observable_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")

    figure, axes = plt.subplots(2, 3, figsize=(14, 8))
    plot_fields = (
        ("n_selected_unique_pmts", "Selected hit PMTs"),
        ("selected_total_charge", "Selected total charge [input units]"),
        ("mean_charge_per_selected_digit", "Mean charge / selected digit"),
        ("selected_charge_fraction", "Fraction of raw charge retained"),
        ("selected_time_rms_charge_weighted_ns", "Charge-weighted time RMS [ns]"),
        ("selected_time_iqr_ns", "Selected time IQR [ns]"),
    )
    for axis, (field, label) in zip(axes.flat, plot_fields):
        for category in categories:
            values = [
                float(row[field])
                for row in metrics
                if row["selection_category"] == category and row[field] is not None
            ]
            if values:
                axis.hist(values, bins="auto", histtype="step", linewidth=1.5, label=category)
        axis.set_xlabel(label)
        axis.set_ylabel("Events")
    axes[0, 0].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "event_observable_distributions.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    for category in categories:
        charges = np.concatenate(hit_charge[category]) if hit_charge[category] else np.array([])
        relative_times = (
            np.concatenate(hit_relative_time[category]) if hit_relative_time[category] else np.array([])
        )
        if charges.size:
            upper = max(float(np.quantile(charges, 0.995)), 1.0)
            axes[0].hist(
                charges, bins=np.linspace(0.0, upper, 80), density=True,
                histtype="step", linewidth=1.4, label=category,
            )
        if relative_times.size:
            axes[1].hist(
                relative_times, bins=np.linspace(-30.0, 6.0, 73), density=True,
                histtype="step", linewidth=1.4, label=category,
            )
    axes[0].set_xlabel("Selected digit charge [input units]")
    axes[0].set_ylabel("Normalized density")
    axes[1].set_xlabel("Selected digit time - modal time [ns]")
    axes[1].set_ylabel("Normalized density")
    axes[1].legend(fontsize=8)
    figure.tight_layout()
    figure.savefig(args.output_dir / "digit_charge_time_distributions.png", dpi=160)
    plt.close(figure)

    print("Event metrics:", csv_path)
    print("Summary:", args.output_dir / "observable_summary.json")
    print("Plots:", args.output_dir / "event_observable_distributions.png")
    print("       ", args.output_dir / "digit_charge_time_distributions.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
