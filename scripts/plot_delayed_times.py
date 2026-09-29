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
from scipy.optimize import brentq, minimize, minimize_scalar


PARTICLE_LABELS = {"211": r"$\pi^+$", "-211": r"$\pi^-$", "22": r"$\gamma$"}
PARTICLE_COLORS = {"211": "#1f77b4", "-211": "#d95f02", "22": "#3a923a"}


def fit_delay_model(values: np.ndarray, start_ns: float, stop_ns: float) -> dict[str, float | None]:
    """Fit a truncated exponential plus a flat background, unbinned.

    The fitted tau is an apparent cluster-delay scale. In particular, this
    model does not correct for delay-dependent trigger/readout efficiency.
    """
    selected = np.asarray(values, dtype=float)
    selected = selected[np.isfinite(selected) & (selected >= start_ns) & (selected < stop_ns)]
    if selected.size < 30:
        raise ValueError("At least 30 candidates are needed for the lifetime fit")
    length = stop_ns - start_ns
    lower_tau, upper_tau = 100.0, 20_000.0

    def nll(tau_ns: float, signal_fraction: float) -> float:
        norm = -np.expm1(-length / tau_ns)
        signal_pdf = np.exp(-(selected - start_ns) / tau_ns) / (tau_ns * norm)
        pdf = signal_fraction * signal_pdf + (1.0 - signal_fraction) / length
        return float(-np.log(pdf).sum())

    result = minimize(
        lambda x: nll(float(np.exp(x[0])), float(x[1])),
        x0=(np.log(2200.0), 0.8),
        bounds=((np.log(lower_tau), np.log(upper_tau)), (0.0, 1.0)),
        method="L-BFGS-B",
    )
    if not result.success:
        raise RuntimeError(f"Lifetime fit failed: {result.message}")
    tau = float(np.exp(result.x[0]))
    fraction = float(result.x[1])

    def profile_difference(test_tau: float) -> float:
        profile = minimize_scalar(lambda f: nll(test_tau, f), bounds=(0.0, 1.0), method="bounded")
        return float(profile.fun - result.fun - 0.5)

    low = high = None
    if tau > lower_tau and profile_difference(lower_tau) > 0:
        low = float(brentq(profile_difference, lower_tau, tau))
    if tau < upper_tau and profile_difference(upper_tau) > 0:
        high = float(brentq(profile_difference, tau, upper_tau))
    return {"tau_ns": tau, "signal_fraction": fraction, "low_ns": low, "high_ns": high, "n_fit": int(selected.size)}


def expected_bin_counts(edges: np.ndarray, fit: dict[str, float | None], start_ns: float, stop_ns: float) -> np.ndarray:
    """Expected counts in bins wholly inside the fit interval."""
    tau = float(fit["tau_ns"])
    fraction = float(fit["signal_fraction"])
    length = stop_ns - start_ns
    lo, hi = edges[:-1], edges[1:]
    signal = (np.exp(-(lo - start_ns) / tau) - np.exp(-(hi - start_ns) / tau)) / (-np.expm1(-length / tau))
    background = (hi - lo) / length
    return int(fit["n_fit"]) * (fraction * signal + (1.0 - fraction) * background)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("events_csv", type=Path, help="events.csv produced by study_delayed_clusters.py")
    parser.add_argument("--output", type=Path, default=None, help="Output PNG or PDF path")
    parser.add_argument("--min-ns", type=float, default=100.0)
    parser.add_argument("--max-ns", type=float, default=6100.0)
    parser.add_argument("--bins", type=int, default=30)
    parser.add_argument("--fit-min-ns", type=float, default=500.0)
    parser.add_argument("--fit-pid", default="211", help="Primary PID to fit (default: pi+)")
    parser.add_argument("--no-fit", action="store_true", help="Plot without fitting a delay model")
    args = parser.parse_args()
    if args.bins < 1 or args.max_ns <= args.min_ns:
        parser.error("Require --bins >= 1 and --max-ns > --min-ns")
    if not args.no_fit and not (args.min_ns <= args.fit_min_ns < args.max_ns):
        parser.error("--fit-min-ns must lie inside the plotted range")
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
        if not args.no_fit and particle == args.fit_pid:
            fit = fit_delay_model(values, args.fit_min_ns, args.max_ns)
            fit_edges = edges[(edges >= args.fit_min_ns) & (edges <= args.max_ns)]
            # The fit start may fall inside a displayed bin. Include it as an
            # edge so the model line only covers the fitted interval.
            if fit_edges.size == 0 or fit_edges[0] > args.fit_min_ns:
                fit_edges = np.insert(fit_edges, 0, args.fit_min_ns)
            if fit_edges[-1] < args.max_ns:
                fit_edges = np.append(fit_edges, args.max_ns)
            expected = expected_bin_counts(fit_edges, fit, args.fit_min_ns, args.max_ns)
            interval = ""
            if fit["low_ns"] is not None and fit["high_ns"] is not None:
                interval = f" (68% stat: {fit['low_ns']:.0f}–{fit['high_ns']:.0f} ns)"
            axis.stairs(expected, fit_edges, linewidth=2, linestyle="--", color="black",
                        label=rf"exp + flat fit: $\tau_{{\rm app}}$={fit['tau_ns']:.0f} ns")
            axis.legend(loc="upper right")
            print(f"Apparent delay constant for PID {particle}: {fit['tau_ns']:.0f} ns{interval}; "
                  f"signal fraction={fit['signal_fraction']:.3f}; n={fit['n_fit']}")
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
