#!/usr/bin/env python3
"""Study pion-production truth signatures in converted tagged-gamma NPZ files."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from LicketyFit.TruthTopology import infer_tagged_gamma_topology  # noqa: E402


LIGHT_FIELDS = (
    "true_hit_fraction_charged_pion",
    "true_hit_fraction_em",
    "true_hit_fraction_proton",
    "true_hit_fraction_muon",
    "true_hit_fraction_unmapped",
    "true_hit_fraction_ambiguous_track_id",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Infer charged-pion production and subsequent topology signatures "
            "in DataTools tagged-gamma NPZ files."
        )
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        type=Path,
        help="NPZ file(s) and/or directories containing converted files",
    )
    parser.add_argument(
        "--pattern",
        default="mdt*gamma*.npz",
        help="Glob used for directory inputs (default: %(default)s)",
    )
    parser.add_argument(
        "--include-skims",
        action="store_true",
        help="Include *_events*.npz files, which may duplicate full-file events",
    )
    parser.add_argument("--first-event", type=int, default=0)
    parser.add_argument("--max-events-per-file", type=int, default=None)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/tagged_gamma_truth"),
    )
    return parser.parse_args()


def discover_inputs(paths: list[Path], pattern: str, include_skims: bool) -> list[Path]:
    files: dict[str, Path] = {}
    for path in paths:
        candidates = sorted(path.glob(pattern)) if path.is_dir() else [path]
        for candidate in candidates:
            if candidate.suffix != ".npz":
                continue
            if not include_skims and "_events" in candidate.stem:
                continue
            files[str(candidate.resolve())] = candidate
    return [files[key] for key in sorted(files)]


def finite_or_none(value: float) -> float | None:
    return float(value) if np.isfinite(value) else None


def main() -> int:
    args = parse_args()
    input_files = discover_inputs(args.inputs, args.pattern, args.include_skims)
    if not input_files:
        raise FileNotFoundError("No matching non-skim NPZ input files were found")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "tagged_gamma_event_topologies.csv"
    json_path = args.output_dir / "summary.json"

    production_counts: Counter[str] = Counter()
    topology_counts: Counter[str] = Counter()
    per_file_counts: dict[str, Counter[str]] = defaultdict(Counter)
    light_sums: Counter[str] = Counter()
    light_n: Counter[str] = Counter()
    n_events = 0
    writer = None

    with csv_path.open("w", newline="", encoding="utf-8") as output:
        for file_index, input_path in enumerate(input_files, start=1):
            with np.load(input_path, allow_pickle=True) as data:
                file_events = len(data["pid"])
                start = args.first_event
                if start < 0 or start >= file_events:
                    raise IndexError(
                        f"{input_path}: first-event {start} outside {file_events} events"
                    )
                stop = file_events
                if args.max_events_per_file is not None:
                    stop = min(stop, start + args.max_events_per_file)

                print(
                    f"[{file_index}/{len(input_files)}] {input_path.name}: "
                    f"events {start}..{stop - 1}",
                    flush=True,
                )
                for event in range(start, stop):
                    row = infer_tagged_gamma_topology(data, event)
                    row["input_file"] = str(input_path)
                    if writer is None:
                        writer = csv.DictWriter(output, fieldnames=list(row.keys()))
                        writer.writeheader()
                    writer.writerow(row)

                    production = row["production_class"]
                    topology = row["tagged_gamma_topology"]
                    production_counts[production] += 1
                    topology_counts[topology] += 1
                    per_file_counts[input_path.name][production] += 1
                    n_events += 1
                    for field in LIGHT_FIELDS:
                        value = float(row[field])
                        if np.isfinite(value):
                            light_sums[field] += value
                            light_n[field] += 1

    light_means = {
        field: finite_or_none(light_sums[field] / light_n[field])
        if light_n[field]
        else None
        for field in LIGHT_FIELDS
    }
    summary = {
        "n_files": len(input_files),
        "n_events": n_events,
        "excluded_skims_by_default": not args.include_skims,
        "warning": (
            "These are inferred signatures, not exact Geant4 process labels. "
            "The available track_parent field behaves like parent-PDG metadata."
        ),
        "production_classes": {
            label: {"count": count, "fraction": count / n_events}
            for label, count in sorted(production_counts.items())
        },
        "topologies": {
            label: {"count": count, "fraction": count / n_events}
            for label, count in sorted(topology_counts.items())
        },
        "mean_true_hit_fractions": light_means,
        "per_file_production_counts": {
            filename: dict(sorted(counts.items()))
            for filename, counts in sorted(per_file_counts.items())
        },
    }
    with json_path.open("w", encoding="utf-8") as output:
        json.dump(summary, output, indent=2, allow_nan=False)
        output.write("\n")

    print(f"\nAnalyzed {n_events} events in {len(input_files)} files")
    print("\nPION PRODUCTION CLASSES")
    for label, count in production_counts.most_common():
        print(f"  {label:30s} {count:9d}  {count / n_events:8.3%}")
    print("\nINFERRED CHARGED-PION TOPOLOGIES")
    for label, count in topology_counts.most_common():
        print(f"  {label:38s} {count:9d}  {count / n_events:8.3%}")
    print(f"\nPer-event table: {csv_path}")
    print(f"Summary:         {json_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
