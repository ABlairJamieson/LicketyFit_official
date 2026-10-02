#!/usr/bin/env python3
"""Check delayed-candidate timing, PMT multiplicity, and charge from events.csv.

This is a diagnostic plot, not a Michel-electron identification or lifetime fit.
The CSV is read one row at a time; no NPZ files or truth information are needed.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_FIELDS = ("delta_t_ns", "best_delayed_pmts", "best_delayed_charge")


def load_candidates(path: Path) -> tuple[int, Counter[str], dict[str, np.ndarray]]:
    total = 0
    modes: Counter[str] = Counter()
    values: dict[str, list[float]] = {field: [] for field in REQUIRED_FIELDS}
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        missing = set(REQUIRED_FIELDS) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"Missing events.csv columns: {', '.join(sorted(missing))}")
        for row in reader:
            total += 1
            modes[row.get("time_mode", "unknown")] += 1
            if not row["delta_t_ns"]:
                continue
            try:
                candidate = [float(row[field]) for field in REQUIRED_FIELDS]
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Invalid candidate values on CSV row {total + 1}") from exc
            if not all(np.isfinite(value) for value in candidate):
                raise ValueError(f"Non-finite candidate values on CSV row {total + 1}")
            for field, value in zip(REQUIRED_FIELDS, candidate):
                values[field].append(value)
    if total == 0:
        raise ValueError("The event table is empty")
    return total, modes, {field: np.asarray(items) for field, items in values.items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_csv", type=Path, help="events.csv from study_delayed_clusters.py")
    parser.add_argument("--output", type=Path, help="Output PNG (default: beside events.csv)")
    parser.add_argument("--max-delay-ns", type=float, default=10_000.0)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()
    if args.bins < 1 or not np.isfinite(args.max_delay_ns) or args.max_delay_ns <= 0:
        parser.error("Require --bins >= 1 and --max-delay-ns > 0")

    total, modes, values = load_candidates(args.events_csv)
    delay = values["delta_t_ns"]
    pmts = values["best_delayed_pmts"]
    charge = values["best_delayed_charge"]
    output = args.output or args.events_csv.with_name("delayed_candidate_checks.png")

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    axes[0].hist(delay, bins=np.linspace(0, args.max_delay_ns, args.bins + 1))
    axes[0].set(xlabel="Best delayed cluster after prompt (ns)", ylabel="Candidate events")
    outside = np.count_nonzero((delay < 0) | (delay > args.max_delay_ns))
    if outside:
        axes[0].set_title(f"{outside:,} outside plotted range", fontsize=10)

    # Limit long tails for readable diagnostics, and state how many points were hidden.
    for axis, data, xlabel in (
        (axes[1], pmts, "Distinct PMTs in best delayed cluster"),
        (axes[2], charge, "Best delayed-cluster charge (input units)"),
    ):
        if data.size:
            upper = max(float(np.percentile(data, 99.5)), float(data.min()) + 1.0)
            axis.hist(data, bins=np.linspace(min(0.0, float(data.min())), upper, args.bins + 1))
            clipped = np.count_nonzero(data > upper)
            if clipped:
                axis.set_title(f"Upper 0.5% clipped ({clipped:,} events)", fontsize=10)
        axis.set(xlabel=xlabel, ylabel="Candidate events")
    for axis in axes:
        axis.grid(alpha=0.2)
    fig.suptitle(f"Delayed candidates: {delay.size:,}/{total:,} events ({delay.size / total:.1%})")
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    plt.close(fig)

    print(f"Events: {total:,}; candidates: {delay.size:,} ({delay.size / total:.1%})")
    print(f"Time conventions: {dict(modes)}")
    print(f"Plot: {output}")
    print("Diagnostic only: verify readout timing and backgrounds before interpreting candidates as decays.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
