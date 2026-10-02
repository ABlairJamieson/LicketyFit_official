import csv
import sys

import pytest

from scripts.plot_delayed_candidate_checks import load_candidates, main


def test_candidate_checks_counts_and_writes_plot(tmp_path, monkeypatch):
    events = tmp_path / "events.csv"
    with events.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "time_mode", "delta_t_ns", "best_delayed_pmts", "best_delayed_charge",
        ])
        writer.writeheader()
        writer.writerow({"time_mode": "raw_missing_trigger_metadata", "delta_t_ns": "", "best_delayed_pmts": "", "best_delayed_charge": ""})
        writer.writerow({"time_mode": "raw_missing_trigger_metadata", "delta_t_ns": "250", "best_delayed_pmts": "10", "best_delayed_charge": "42"})
        writer.writerow({"time_mode": "raw_missing_trigger_metadata", "delta_t_ns": "3200", "best_delayed_pmts": "15", "best_delayed_charge": "90"})

    total, modes, values = load_candidates(events)
    assert total == 3
    assert modes == {"raw_missing_trigger_metadata": 3}
    assert values["delta_t_ns"].tolist() == [250, 3200]

    output = tmp_path / "checks.png"
    monkeypatch.setattr(sys, "argv", ["plot_delayed_candidate_checks.py", str(events), "--output", str(output)])
    assert main() == 0
    assert output.is_file() and output.stat().st_size > 0


def test_candidate_checks_rejects_missing_columns(tmp_path):
    events = tmp_path / "events.csv"
    events.write_text("delta_t_ns\n250\n", encoding="utf-8")
    with pytest.raises(ValueError, match="Missing events.csv columns"):
        load_candidates(events)
