import csv

from scripts.plot_delayed_time_choices import load_delay_choices, main


def test_selection_rules_and_plot(tmp_path, monkeypatch):
    events = tmp_path / "events.csv"
    with events.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "input_file", "event_index", "primary_pid", "delta_t_ns",
            "n_delayed_clusters", "prompt_time_ns", "prompt_n_hits",
        ])
        writer.writeheader()
        writer.writerow({"input_file": "sim.npz", "event_index": 0,
                         "primary_pid": "211", "delta_t_ns": "1000",
                         "n_delayed_clusters": "2", "prompt_time_ns": "150",
                         "prompt_n_hits": "12"})
    clusters = tmp_path / "clusters.csv"
    with clusters.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=[
            "input_file", "event_index", "rank", "delta_t_ns", "n_hits", "n_pmts", "charge",
        ])
        writer.writeheader()
        writer.writerow({"input_file": "sim.npz", "event_index": 0, "rank": 1,
                         "delta_t_ns": 1000, "n_hits": 5, "n_pmts": 10, "charge": 20})
        writer.writerow({"input_file": "sim.npz", "event_index": 0, "rank": 2,
                         "delta_t_ns": 500, "n_hits": 9, "n_pmts": 8, "charge": 10})

    choices = load_delay_choices(events, clusters)["211"]
    assert choices["second"] == [500.0]
    assert choices["largest_hits"] == [500.0]
    assert choices["rank1"] == [1000.0]

    output = tmp_path / "comparison.png"
    monkeypatch.setattr("sys.argv", ["plot_delayed_time_choices.py", str(events),
                                      "--output", str(output)])
    assert main() == 0
    assert output.is_file() and output.stat().st_size > 0
    for name in ("clusters_per_event.png", "cluster_hits_all_prompt_delayed.png",
                 "prompt_and_delayed_cluster_times.png"):
        plot = tmp_path / name
        assert plot.is_file() and plot.stat().st_size > 0
