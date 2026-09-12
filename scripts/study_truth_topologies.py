#!/usr/bin/env python3
"""Study inferred pion truth topologies in DataTools WCSim NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LicketyFit.TruthTopology import analyze_truth_file  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer conservative pion truth-topology signatures and true-light "
            "fractions. Labels are not exact Geant4 process identifications."
        )
    )
    parser.add_argument("inputs", nargs="+", type=Path, help="DataTools .npz file(s)")
    parser.add_argument("--first-event", type=int, default=0)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/truth_topologies"))
    return parser.parse_args()


def json_value(value):
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main() -> int:
    args = parse_args()
    rows = []
    for input_path in args.inputs:
        file_rows = analyze_truth_file(input_path, args.first_event, args.max_events)
        for row in file_rows:
            row["input_file"] = str(input_path)
        rows.extend(file_rows)

    if not rows:
        raise RuntimeError("No events were analyzed")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "event_topologies.csv"
    json_path = args.output_dir / "summary.json"

    fieldnames = sorted({key for row in rows for key in row})
    with csv_path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    topology_counts = Counter(row["topology"] for row in rows)
    n_events = len(rows)
    light_fractions = {}
    for field in (
        "true_hit_fraction_charged_pion",
        "true_hit_fraction_em",
        "true_hit_fraction_proton",
        "true_hit_fraction_muon",
        "true_hit_fraction_unmapped",
        "true_hit_fraction_ambiguous_track_id",
    ):
        values = np.asarray([row[field] for row in rows], dtype=float)
        finite = values[np.isfinite(values)]
        light_fractions[field] = {
            "mean": float(np.mean(finite)) if finite.size else None,
            "median": float(np.median(finite)) if finite.size else None,
        }

    summary = {
        "n_events": n_events,
        "warning": (
            "Topologies are inferred signatures, not exact Geant4 process labels; "
            "track_parent may encode parent PDG and sentinel values."
        ),
        "topologies": {
            label: {"count": count, "fraction": count / n_events}
            for label, count in sorted(topology_counts.items())
        },
        "light_fraction_statistics": light_fractions,
    }
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, default=json_value)
        output.write("\n")

    print(f"Analyzed {n_events} events")
    print("\nINFERRED TOPOLOGIES")
    for label, count in topology_counts.most_common():
        print(f"  {label:32s} {count:7d}  {count / n_events:8.3%}")
    print("\nMEAN TRUE-HIT FRACTIONS")
    for field, statistics in light_fractions.items():
        value = statistics["mean"]
        display = "unavailable" if value is None else f"{value:.3%}"
        print(f"  {field.removeprefix('true_hit_fraction_'):28s} {display}")
    print(f"\nPer-event table: {csv_path}")
    print(f"Summary:         {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
