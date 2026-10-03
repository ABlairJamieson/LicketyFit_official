import csv
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "audit_delayed_michel_truth.py"


def test_wilson_interval_contains_boundary_fractions():
    spec = importlib.util.spec_from_file_location("audit_delayed_michel_truth", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    for successes, trials in ((0, 6), (6, 6), (10, 10), (3, 10)):
        value, low, high = module.wilson_68(successes, trials)
        assert low <= value <= high


def _object_events(events):
    result = np.empty(len(events), dtype=object)
    for i, event in enumerate(events):
        result[i] = np.asarray(event)
    return result


def test_five_bin_efficiency_and_timing_match_fraction(tmp_path):
    # Unique Michel at 1500 ns, missed Michel at 2500, mistagged Michel at
    # 3500, non-Michel false candidate, and an ambiguous event excluded.
    pids = [
        [211, -13, -11], [211, -13, -11], [211, -13, -11],
        [211, -13], [211, -13, -11, -11],
    ]
    parents = [
        [0, 211, -13], [0, 211, -13], [0, 211, -13],
        [0, 211], [0, 211, -13, -13],
    ]
    times = [
        [0, 15, 1500], [0, 15, 2500], [0, 15, 3500],
        [0, 15], [0, 15, 1500, 1700],
    ]
    source = tmp_path / "pions.npz"
    np.savez_compressed(source, track_pid=_object_events(pids),
                        track_parent=_object_events(parents),
                        track_start_time=_object_events(times))
    events_csv = tmp_path / "events.csv"
    with events_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("input_file", "event_index", "primary_pid", "delta_t_ns"))
        writer.writeheader()
        for index, delay in enumerate((1505, "", 5000, 1500, 1500)):
            writer.writerow({"input_file": str(source), "event_index": index,
                             "primary_pid": 211, "delta_t_ns": delay})
    output = tmp_path / "audit"
    subprocess.run([sys.executable, str(SCRIPT), str(events_csv), "--output-dir", str(output)],
                   check=True, capture_output=True, text=True)
    with (output / "delay_bin_metrics.csv").open(newline="", encoding="utf-8") as handle:
        bins = list(csv.DictReader(handle))
    assert len(bins) == 5
    assert (bins[0]["truth_decays"], bins[0]["truth_matched"]) == ("1", "1")
    assert (bins[0]["reco_candidates"], bins[0]["reco_timing_matched"]) == ("2", "1")
    assert float(bins[0]["efficiency"]) == 1.0
    assert float(bins[0]["timing_match_fraction"]) == 0.5
    assert (bins[1]["truth_decays"], bins[1]["truth_matched"]) == ("1", "0")
    assert (bins[2]["truth_decays"], bins[2]["truth_matched"]) == ("1", "0")
    assert (bins[3]["reco_candidates"], bins[3]["reco_timing_matched"]) == ("1", "0")
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["truth_status"]["ambiguous_lineage"] == 1
    assert summary["counts"]["truth_decays_in_range"] == 3
    assert (output / "delay_bin_metrics.png").is_file()
    assert (output / "truth_muon_lifetimes.png").is_file()


def test_missing_truth_not_counted_as_false_candidate(tmp_path):
    source = tmp_path / "no_truth.npz"
    np.savez_compressed(source, pid=np.array([211]))
    events_csv = tmp_path / "events.csv"
    events_csv.write_text("input_file,event_index,primary_pid,delta_t_ns\n"
                          f"{source},0,211,1500\n", encoding="utf-8")
    output = tmp_path / "audit"
    subprocess.run([sys.executable, str(SCRIPT), str(events_csv), "--output-dir", str(output)],
                   check=True, capture_output=True, text=True)
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    assert summary["counts"]["truth_status"]["missing_truth"] == 1
    assert summary["counts"]["reco_candidates_in_range_with_classifiable_truth"] == 0
