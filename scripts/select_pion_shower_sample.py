#!/usr/bin/env python3
"""Select a reproducible balanced truth sample and make a compact fit NPZ."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


CATEGORY_TO_TOPOLOGIES = {
    "pi_plus_decay": {"pi_plus_decay_candidate"},
    "pi_plus_interacting": {
        "pi_plus_interacting_no_pi0",
        "pi_plus_interacting_with_pi0",
    },
    "pi_minus_interacting": {
        "pi_minus_interacting_no_pi0",
        "pi_minus_interacting_with_pi0",
    },
    "no_pion": {"no_pion"},
    "pi0_only": {"pi0_without_charged_pion"},
}

FIT_FIELDS = ("digi_hit_pmt", "digi_hit_charge", "digi_hit_time")
SMALL_TRUTH_FIELDS = ("pid", "energy", "event_id", "position", "direction")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select equal-sized pion/shower truth classes for fit studies."
    )
    parser.add_argument("truth_csv", type=Path)
    parser.add_argument("--per-category", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/pion_shower_sample")
    )
    parser.add_argument(
        "--manifest-only",
        action="store_true",
        help="Do not create the compact NPZ used by the fit runner.",
    )
    parser.add_argument(
        "--allow-short",
        action="store_true",
        help="Keep all available rows instead of failing when a class is short.",
    )
    return parser.parse_args()


def classify(row: dict[str, str]) -> str | None:
    topology = row.get("tagged_gamma_topology", "")
    for category, labels in CATEGORY_TO_TOPOLOGIES.items():
        if topology in labels:
            return category
    return None


def stratified_select(
    rows: list[dict[str, str]], count: int, rng: np.random.Generator
) -> list[dict[str, str]]:
    """Spread selection across input files instead of taking one HD file."""
    by_file: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_file[row["input_file"]].append(row)
    for bucket in by_file.values():
        rng.shuffle(bucket)
    filenames = sorted(by_file)
    rng.shuffle(filenames)
    selected: list[dict[str, str]] = []
    while len(selected) < count:
        progressed = False
        for filename in filenames:
            if by_file[filename] and len(selected) < count:
                selected.append(by_file[filename].pop())
                progressed = True
        if not progressed:
            break
    return selected


def write_compact_npz(rows: list[dict[str, str]], output_path: Path) -> None:
    by_file: dict[Path, list[tuple[int, int]]] = defaultdict(list)
    for fit_index, row in enumerate(rows):
        by_file[Path(row["input_file"])].append((fit_index, int(row["event_index"])))

    collected: dict[str, list[object | None]] = {
        name: [None] * len(rows) for name in (*FIT_FIELDS, *SMALL_TRUTH_FIELDS)
    }
    for file_number, (source, selections) in enumerate(sorted(by_file.items()), 1):
        print(f"[{file_number}/{len(by_file)}] extracting {len(selections)} from {source.name}")
        with np.load(source, allow_pickle=True) as archive:
            for field in collected:
                if field not in archive.files:
                    continue
                source_array = archive[field]
                for fit_index, event_index in selections:
                    value = source_array[event_index]
                    collected[field][fit_index] = (
                        value.copy() if isinstance(value, np.ndarray) else value
                    )
                del source_array

    missing_fit = [name for name in FIT_FIELDS if any(v is None for v in collected[name])]
    if missing_fit:
        raise KeyError(f"Selected source files are missing fit fields: {missing_fit}")

    arrays: dict[str, np.ndarray] = {}
    for name, values in collected.items():
        if all(value is None for value in values):
            continue
        if name in FIT_FIELDS:
            array = np.empty(len(values), dtype=object)
            array[:] = values
        else:
            try:
                array = np.asarray(values)
            except ValueError:
                array = np.asarray(values, dtype=object)
        arrays[name] = array
    arrays["source_file"] = np.asarray([row["input_file"] for row in rows], dtype=object)
    arrays["source_event_index"] = np.asarray(
        [int(row["event_index"]) for row in rows], dtype=np.int64
    )
    arrays["selection_category"] = np.asarray(
        [row["selection_category"] for row in rows], dtype=object
    )
    np.savez_compressed(output_path, **arrays)


def main() -> int:
    args = parse_args()
    if args.per_category < 1:
        raise ValueError("--per-category must be positive")
    with args.truth_csv.open(newline="", encoding="utf-8") as handle:
        candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
        for row in csv.DictReader(handle):
            category = classify(row)
            if category is not None:
                candidates[category].append(row)

    rng = np.random.default_rng(args.seed)
    selected_by_category: dict[str, list[dict[str, str]]] = {}
    available = {category: len(candidates[category]) for category in CATEGORY_TO_TOPOLOGIES}
    for category in CATEGORY_TO_TOPOLOGIES:
        rows = candidates[category]
        if len(rows) < args.per_category and not args.allow_short:
            raise RuntimeError(
                f"{category} has {len(rows)} candidates; requested {args.per_category}. "
                "Use --allow-short to accept an unbalanced sample."
            )
        selected_by_category[category] = []
        for row in stratified_select(rows, min(args.per_category, len(rows)), rng):
            selected_row = dict(row)
            selected_row["selection_category"] = category
            selected_by_category[category].append(selected_row)

    # Interleave categories so --max-events 5 makes a one-per-class pilot.
    chosen: list[dict[str, str]] = []
    max_selected = max((len(rows) for rows in selected_by_category.values()), default=0)
    for index in range(max_selected):
        for category in CATEGORY_TO_TOPOLOGIES:
            rows = selected_by_category[category]
            if index < len(rows):
                chosen.append(rows[index])

    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output_dir / "fit_manifest.csv"
    skim_path = args.output_dir / "selected_events.npz"
    for fit_index, row in enumerate(chosen):
        row["fit_event_index"] = str(fit_index)
        row["fit_input_file"] = str(skim_path.resolve())
    fieldnames = list(chosen[0]) if chosen else []
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(chosen)

    if not args.manifest_only:
        write_compact_npz(chosen, skim_path)

    selected_counts = Counter(row["selection_category"] for row in chosen)
    summary = {
        "seed": args.seed,
        "requested_per_category": args.per_category,
        "available_counts": available,
        "selected_counts": dict(selected_counts),
        "manifest": str(manifest_path.resolve()),
        "compact_npz": None if args.manifest_only else str(skim_path.resolve()),
        "category_topologies": {
            key: sorted(value) for key, value in CATEGORY_TO_TOPOLOGIES.items()
        },
    }
    with (args.output_dir / "selection_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
        handle.write("\n")
    print("Selected:", dict(selected_counts))
    print("Manifest:", manifest_path)
    if not args.manifest_only:
        print("Compact fit input:", skim_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
