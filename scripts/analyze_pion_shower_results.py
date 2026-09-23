#!/usr/bin/env python3
"""Summarize pion/shower fit separation for the balanced validation sample."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np


PION_CATEGORIES = {"pi_plus_decay", "pi_plus_interacting", "pi_minus_interacting"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def finite_float(value: str | None) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def auc_pairwise(scores: np.ndarray, labels: np.ndarray) -> float:
    positive = scores[labels]
    negative = scores[~labels]
    if not len(positive) or not len(negative):
        return float("nan")
    wins = sum(float(np.sum(value > negative)) for value in positive)
    ties = sum(float(np.sum(value == negative)) for value in positive)
    return (wins + 0.5 * ties) / (len(positive) * len(negative))


def best_threshold(scores: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    unique = np.unique(scores)
    if len(unique) == 1:
        candidates = unique
    else:
        candidates = np.concatenate(
            ([unique[0] - 1.0], 0.5 * (unique[:-1] + unique[1:]), [unique[-1] + 1.0])
        )
    best: dict[str, float] | None = None
    for threshold in candidates:
        predicted = scores > threshold
        pion_eff = float(np.mean(predicted[labels]))
        background_rejection = float(np.mean(~predicted[~labels]))
        balanced = 0.5 * (pion_eff + background_rejection)
        row = {
            "threshold_delta_nll": float(threshold),
            "pion_efficiency": pion_eff,
            "background_rejection": background_rejection,
            "balanced_accuracy": balanced,
        }
        if best is None or row["balanced_accuracy"] > best["balanced_accuracy"]:
            best = row
    assert best is not None
    return best


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir or args.results_csv.parent / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    with args.results_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    usable: list[tuple[str, float]] = []
    by_category: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        if row.get("fit_comparison_valid", "").lower() not in {"true", "1"}:
            continue
        score = finite_float(row.get("delta_nll_shower_minus_pion"))
        if score is not None:
            category = row["selection_category"]
            usable.append((category, score))
            by_category[category].append(score)

    scores = np.asarray([score for _, score in usable], dtype=float)
    labels = np.asarray([category in PION_CATEGORIES for category, _ in usable], dtype=bool)
    category_summary = {
        category: {
            "n": len(values),
            "median_delta_nll": float(np.median(values)),
            "mean_delta_nll": float(np.mean(values)),
            "fraction_pion_favored": float(np.mean(np.asarray(values) > 0.0)),
        }
        for category, values in sorted(by_category.items())
    }
    summary = {
        "score_definition": "shower_nll - min(absorption_pion_nll, full_length_pion_nll)",
        "interpretation": "positive values favor the pion hypotheses",
        "n_rows": len(rows),
        "n_usable": len(usable),
        "pion_categories": sorted(PION_CATEGORIES),
        "auc": (
            auc_pairwise(scores, labels)
            if len(usable) and labels.any() and (~labels).any()
            else None
        ),
        "best_balanced_threshold_on_this_sample": (
            best_threshold(scores, labels) if len(usable) and labels.any() and (~labels).any() else None
        ),
        "categories": category_summary,
        "caution": (
            "This balanced truth-selected sample measures conditional separation, not purity in the "
            "naturally imbalanced tagged-gamma beam. Raw NLL differences also do not compensate "
            "for differing hypothesis flexibility; reserve an independent sample for calibration."
        ),
    }
    with (output_dir / "separation_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, allow_nan=False)
        handle.write("\n")

    print(f"Usable fits: {len(usable)}/{len(rows)}")
    if summary["auc"] is not None:
        print(f"AUC: {summary['auc']:.4f}")
    for category, values in category_summary.items():
        print(
            f"{category:24s} n={values['n']:3d}  "
            f"median delta-NLL={values['median_delta_nll']:10.3f}  "
            f"pion-favored={values['fraction_pion_favored']:7.1%}"
        )
    print("Summary:", output_dir / "separation_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
