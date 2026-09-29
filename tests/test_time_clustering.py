import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from LicketyFit.TimeClustering import event_hit_times, find_delayed_clusters


def test_sliding_window_finds_boundary_crossing_and_delayed_burst():
    prompt = np.arange(12, dtype=float) + 15.0
    delayed = np.arange(11, dtype=float) + 2210.0
    times = np.concatenate((prompt, delayed))
    ids = np.arange(len(times))
    first, later = find_delayed_clusters(times, ids, np.ones(len(times)), width_ns=20, min_pmts=10)
    assert first.n_pmts == 12
    assert len(later) == 1
    assert later[0].n_pmts == 11
    assert 2180 < later[0].center_ns - first.center_ns < 2210


def test_prompt_is_first_burst_even_when_delayed_is_brighter():
    times = np.concatenate((np.arange(10), 2000 + np.arange(20))).astype(float)
    first, later = find_delayed_clusters(times, np.arange(30), np.ones(30), min_pmts=10)
    assert first.n_pmts == 10
    assert later[0].n_pmts == 20


def test_isolated_early_hit_does_not_turn_prompt_into_delayed_candidate():
    times = np.concatenate(([-1400.0], 10 + np.arange(300) / 10, 2200 + np.arange(12))).astype(float)
    ids = np.arange(len(times))
    first, later = find_delayed_clusters(times, ids, np.ones(len(times)), min_pmts=10)
    assert first.n_pmts >= 10
    assert 10 <= first.center_ns < 200
    assert len(later) == 1
    assert 2000 < later[0].center_ns - first.center_ns < 2200


def test_trigger_relative_digits_are_combined_on_common_clock():
    event = {
        "digi_hit_time": np.array([100.0, 100.0]),
        "digi_hit_trigger": np.array([0, 1]),
        "trigger_time": np.array([1000.0, 3000.0]),
    }
    times, mode = event_hit_times(event)
    assert mode == "trigger_plus"
    np.testing.assert_allclose(times, [1100.0, 3100.0])


def test_one_pmt_with_many_digits_does_not_pass_threshold():
    times = np.concatenate((np.arange(10), 2000 + np.arange(20))).astype(float)
    ids = np.concatenate((np.arange(10), np.full(20, 99)))
    _, later = find_delayed_clusters(times, ids, np.ones(30), min_pmts=10)
    assert later == []


def test_cli_reads_datatools_style_npz(tmp_path):
    arrays = {}
    for key, values in {
        "digi_hit_time": np.concatenate(([-214_749_368.0], 100 + np.arange(12), 100 + np.arange(11))),
        "digi_hit_pmt": np.arange(24),
        "digi_hit_charge": np.ones(24),
        "digi_hit_trigger": np.concatenate((np.zeros(13, dtype=int), np.ones(11, dtype=int))),
        "trigger_time": np.array([1000.0, 3200.0]),
    }.items():
        arrays[key] = np.empty(1, dtype=object)
        arrays[key][0] = values
    source = tmp_path / "pion.npz"
    np.savez_compressed(source, **arrays)
    output = tmp_path / "results"
    script = Path(__file__).resolve().parents[1] / "scripts" / "study_delayed_clusters.py"
    subprocess.run([sys.executable, str(script), str(source), "--output-dir", str(output)], check=True, capture_output=True, text=True)
    with (output / "events.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["time_mode"] == "trigger_plus"
    assert row["n_rejected_early_digits"] == "1"
    assert row["n_delayed_clusters"] == "1"
    assert 2190 < float(row["delta_t_ns"]) < 2210


def test_all_events_overrides_100_event_pilot_default(tmp_path):
    arrays = {}
    for key in ("digi_hit_time", "digi_hit_pmt", "digi_hit_charge"):
        arrays[key] = np.empty(101, dtype=object)
        for index in range(101):
            arrays[key][index] = np.array([], dtype=float)
    source = tmp_path / "gamma.npz"
    np.savez_compressed(source, **arrays)
    output = tmp_path / "all"
    script = Path(__file__).resolve().parents[1] / "scripts" / "study_delayed_clusters.py"
    subprocess.run([sys.executable, str(script), str(source), "--all-events",
                    "--output-dir", str(output)], check=True, capture_output=True, text=True)
    with (output / "events.csv").open(newline="", encoding="utf-8") as handle:
        assert len(list(csv.DictReader(handle))) == 101
