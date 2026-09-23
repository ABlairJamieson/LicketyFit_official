#!/usr/bin/env python3
"""Run the three-hypothesis pion/shower study from a selected manifest.

Each event is saved immediately and completed events are skipped on reruns.
The runner calls the established single-event smoke-test executable twice:
once for absorption-pion + shower, and once for a full-length pion.
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/pion_shower_fits"))
    parser.add_argument("--geometry-dir", default="Geometry")
    parser.add_argument("--geometry-file", default="Geometry/examples/wcte_bldg157.geo")
    parser.add_argument("--table-dir", default="tables")
    parser.add_argument("--ncall", type=int, default=15000)
    parser.add_argument("--shower-ncall", type=int, default=5000)
    parser.add_argument("--direction-seed-count", type=int, default=49)
    parser.add_argument("--max-attempts", type=int, default=2)
    parser.add_argument(
        "--pilot-seeds",
        action="store_true",
        help=(
            "Use a small vertex/length/energy seed bank for setup checks. "
            "Do not use this option for the final separation study."
        ),
    )
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--category", action="append", default=[])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--retry-failed", action="store_true")
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def nested(mapping: dict[str, Any], *keys: str) -> Any:
    value: Any = mapping
    for key in keys:
        if not isinstance(value, dict):
            return None
        value = value.get(key)
    return value


def build_common(args: argparse.Namespace, row: dict[str, str], output: Path) -> list[str]:
    energy = float(row.get("primary_energy_mev") or 1000.0)
    command = [
        sys.executable,
        "scripts/run_pion_shower_smoke_test.py",
        "--input", row["fit_input_file"],
        "--event-index", row["fit_event_index"],
        "--geometry-dir", args.geometry_dir,
        "--geometry-file", args.geometry_file,
        "--table-dir", args.table_dir,
        "--particle", "pion",
        "--energy-mev", str(energy),
        "--likelihood-mode", "charge_time",
        "--free-z",
        "--both-directions",
        "--direction-seeding", "isotropic",
        "--direction-seed-count", str(args.direction_seed_count),
        "--track-direction-parameterization", "theta_phi",
        "--ncall", str(args.ncall),
        "--max-attempts", str(args.max_attempts),
        "--output-dir", str(output),
    ]
    if args.pilot_seeds:
        command.extend(
            [
                "--x-seeds-mm", "0",
                "--y-seeds-mm", "0",
                "--z-seeds-mm=-1500,-1200,-900",
                "--visible-length-seeds-mm", "200,700,1200",
                "--ke-seeds-mev", "150,300,600",
            ]
        )
    return command


def run_command(command: list[str], log_path: Path, dry_run: bool) -> int:
    print(" ".join(command), flush=True)
    if dry_run:
        return 0
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
    return completed.returncode


def aggregate(rows: list[dict[str, str]], output_dir: Path) -> None:
    output_rows: list[dict[str, Any]] = []
    for row in rows:
        key = f"{int(row['fit_event_index']):04d}_{row['selection_category']}"
        event_dir = output_dir / key
        status_path = event_dir / "status.json"
        if not status_path.exists():
            continue
        status = load_json(status_path)
        record: dict[str, Any] = dict(row)
        record.update(status)
        reports: dict[str, dict[str, Any]] = {}
        for hypothesis, dirname in (("absorption", "absorption_shower"), ("full_length", "full_length")):
            path = event_dir / dirname / "report.json"
            if path.exists():
                reports[hypothesis] = load_json(path)
        absorption = reports.get("absorption", {})
        full = reports.get("full_length", {})
        shower_nll = nested(absorption, "shower_summary", "nll")
        absorption_nll = nested(absorption, "track_summary", "fcn")
        full_nll = nested(full, "track_summary", "fcn")
        finite_tracks = [float(v) for v in (absorption_nll, full_nll) if isinstance(v, (int, float))]
        best_track = min(finite_tracks) if finite_tracks else None
        record.update({
            "shower_nll": shower_nll,
            "absorption_pion_nll": absorption_nll,
            "full_length_pion_nll": full_nll,
            "best_pion_nll": best_track,
            "delta_nll_shower_minus_pion": (
                float(shower_nll) - best_track
                if isinstance(shower_nll, (int, float)) and best_track is not None
                else None
            ),
            "absorption_valid": nested(absorption, "track_summary", "valid"),
            "full_length_valid": nested(full, "track_summary", "valid"),
            "shower_valid": nested(absorption, "shower_summary", "valid"),
        })
        features = absorption.get("topology_features", {})
        for name, value in features.items():
            record[f"feature_{name}"] = value
        output_rows.append(record)
    if not output_rows:
        return
    fields: list[str] = []
    for row in output_rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with (output_dir / "fit_results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(output_rows)


def main() -> int:
    args = parse_args()
    with args.manifest.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if args.category:
        wanted = set(args.category)
        rows = [row for row in rows if row["selection_category"] in wanted]
    if args.max_events is not None:
        rows = rows[: args.max_events]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for number, row in enumerate(rows, 1):
        key = f"{int(row['fit_event_index']):04d}_{row['selection_category']}"
        event_dir = args.output_dir / key
        event_dir.mkdir(parents=True, exist_ok=True)
        status_path = event_dir / "status.json"
        if status_path.exists() and not args.retry_failed:
            status = load_json(status_path)
            if status.get("complete"):
                print(f"[{number}/{len(rows)}] skip complete {key}")
                continue
        print(f"[{number}/{len(rows)}] {key}", flush=True)
        absorption_dir = event_dir / "absorption_shower"
        full_dir = event_dir / "full_length"
        absorption = build_common(args, row, absorption_dir) + [
            "--fit-mode", "absorption",
            "--free-energy",
            "--shower",
            "--shower-model", "pdg_longitudinal",
            "--shower-max-beam-angle-deg", "30",
            "--shower-ncall", str(args.shower_ncall),
        ]
        full = build_common(args, row, full_dir) + ["--fit-mode", "full_length"]
        rc_abs = run_command(absorption, event_dir / "absorption_shower.log", args.dry_run)
        rc_full = run_command(full, event_dir / "full_length.log", args.dry_run)
        if not args.dry_run:
            status = {
                "complete": rc_abs == 0 and rc_full == 0,
                "absorption_shower_returncode": rc_abs,
                "full_length_returncode": rc_full,
            }
            with status_path.open("w", encoding="utf-8") as handle:
                json.dump(status, handle, indent=2)
                handle.write("\n")
            aggregate(rows, args.output_dir)
    aggregate(rows, args.output_dir)
    print("Results:", args.output_dir / "fit_results.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
