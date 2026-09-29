#!/usr/bin/env python3
"""Search converted WCSim NPZ events for delayed Michel-electron candidates."""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from LicketyFit.TimeClustering import event_hit_times, find_delayed_clusters  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path, help="Converted WCSim NPZ files")
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/delayed_clusters"))
    parser.add_argument("--max-events-per-file", type=int, default=100)
    parser.add_argument("--width-ns", type=float, default=50.0)
    parser.add_argument("--min-pmts", type=int, default=10)
    parser.add_argument("--prompt-min-pmts", type=int, default=10)
    parser.add_argument("--start-ns", type=float, default=200.0)
    parser.add_argument("--end-ns", type=float, default=10_000.0)
    parser.add_argument("--prompt-search-ns", type=float, default=200.0)
    parser.add_argument(
        "--min-time-ns", type=float, default=-1_000_000.0,
        help="Reject digits earlier than this common-clock time (default: %(default)g ns)",
    )
    parser.add_argument("--time-mode", choices=("auto", "raw", "trigger-plus"), default="auto")
    args = parser.parse_args()
    if args.max_events_per_file is not None and args.max_events_per_file < 1:
        parser.error("--max-events-per-file must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    event_fields = [
        "input_file", "event_index", "event_id", "primary_pid", "selection_category",
        "time_mode", "n_digits", "n_rejected_early_digits", "n_triggers", "time_span_ns", "prompt_time_ns",
        "prompt_pmts", "prompt_charge", "n_delayed_clusters", "best_delayed_time_ns",
        "delta_t_ns", "best_delayed_pmts", "best_delayed_charge",
    ]
    cluster_fields = ["input_file", "event_index", "rank", "delta_t_ns", "start_ns", "end_ns", "center_ns", "n_hits", "n_pmts", "charge"]
    events_path = args.output_dir / "events.csv"
    clusters_path = args.output_dir / "clusters.csv"
    total = tagged = 0
    conventions: Counter[str] = Counter()
    short_observed_spans = 0
    rejected_early_digits = 0
    with events_path.open("w", newline="", encoding="utf-8") as event_file, clusters_path.open("w", newline="", encoding="utf-8") as cluster_file:
        events = csv.DictWriter(event_file, fieldnames=event_fields)
        clusters = csv.DictWriter(cluster_file, fieldnames=cluster_fields)
        events.writeheader()
        clusters.writeheader()
        for path in args.inputs:
            with np.load(path, allow_pickle=True) as data:
                n_available = len(data["digi_hit_time"])
                n_events = n_available if args.max_events_per_file is None else min(n_available, args.max_events_per_file)
                # NPZ stores an entire object-array field in one member. Keep
                # the default pilot small for the large tagged-gamma files.
                field_names = [name for name in ("digi_hit_time", "digi_hit_pmt", "digi_hit_charge", "digi_hit_trigger", "trigger_time", "event_id", "pid", "selection_category") if name in data.files]
                fields = {name: data[name] for name in field_names}
                for index in range(n_events):
                    event = {name: fields[name][index] for name in field_names}
                    times, convention = event_hit_times(event, args.time_mode)
                    conventions[convention] += 1
                    pmts = np.asarray(event["digi_hit_pmt"], dtype=int).reshape(-1)
                    charge = np.asarray(event["digi_hit_charge"], dtype=float).reshape(-1)
                    n_raw_digits = len(times)
                    keep = np.isfinite(times) & (times >= args.min_time_ns)
                    n_rejected = int(np.count_nonzero(~keep))
                    rejected_early_digits += n_rejected
                    times, pmts, charge = times[keep], pmts[keep], charge[keep]
                    short_observed_spans += bool(len(times) and np.ptp(times) < args.end_ns)
                    prompt, delayed = find_delayed_clusters(times, pmts, charge, width_ns=args.width_ns, min_pmts=args.min_pmts, prompt_min_pmts=args.prompt_min_pmts, search_start_ns=args.start_ns, search_end_ns=args.end_ns, prompt_search_ns=args.prompt_search_ns)
                    best = delayed[0] if delayed else None
                    row = {
                        "input_file": str(path), "event_index": index,
                        "event_id": event.get("event_id", index),
                        "primary_pid": event.get("pid", ""),
                        "selection_category": event.get("selection_category", ""),
                        "time_mode": convention, "n_digits": n_raw_digits,
                        "n_rejected_early_digits": n_rejected,
                        "n_triggers": len(np.asarray(event.get("trigger_time", [])).reshape(-1)),
                        "time_span_ns": float(np.ptp(times)) if len(times) else "",
                        "prompt_time_ns": prompt.center_ns if prompt else "",
                        "prompt_pmts": prompt.n_pmts if prompt else "",
                        "prompt_charge": prompt.charge if prompt else "",
                        "n_delayed_clusters": len(delayed),
                        "best_delayed_time_ns": best.center_ns if best else "",
                        "delta_t_ns": best.center_ns - prompt.center_ns if best else "",
                        "best_delayed_pmts": best.n_pmts if best else "",
                        "best_delayed_charge": best.charge if best else "",
                    }
                    events.writerow(row)
                    for rank, candidate in enumerate(delayed, 1):
                        clusters.writerow({"input_file": str(path), "event_index": index, "rank": rank,
                            "delta_t_ns": candidate.center_ns - prompt.center_ns, **candidate.to_dict()})
                    total += 1
                    tagged += bool(delayed)
            print(f"{path.name}: {n_events} events", flush=True)
    print(f"Delayed clusters found in {tagged}/{total} events")
    print(f"Time conventions: {dict(conventions)}")
    print(f"Rejected early/nonfinite digits: {rejected_early_digits}")
    print(f"Observed digit span < {args.end_ns:g} ns: {short_observed_spans}/{total} events")
    print("Digit span is not a readout-coverage measure; check trigger/readout settings before interpreting missing clusters.")
    print(f"Event table: {events_path}")
    print(f"Cluster table: {clusters_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
