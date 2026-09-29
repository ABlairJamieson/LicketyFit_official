import csv
import subprocess
import sys
from pathlib import Path

import numpy as np

from LicketyFit.MichelVertex import fit_delayed_point_vertex, match_michel_truth


def _synthetic_hits():
    rng = np.random.default_rng(290926)
    directions = rng.normal(size=(60, 3))
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    positions = 1500.0 * directions
    vertex = np.array([120.0, -80.0, 200.0])
    times = 2000.0 + np.linalg.norm(positions - vertex, axis=1) / (299.792458 / 1.373)
    times += rng.normal(0.0, 0.05, size=len(times))
    return positions, vertex, times


def test_point_multilateration_recovers_synthetic_vertex():
    positions, truth, times = _synthetic_hits()
    result = fit_delayed_point_vertex(positions, np.arange(len(times)), np.ones(len(times)), times)
    assert np.linalg.norm(result["vertex_mm"] - truth) < 100.0
    assert result["n_window_pmts"] == len(times)


def test_vertex_runner_reads_cluster_window_and_mapping(tmp_path):
    positions, truth, times = _synthetic_hits()
    mapping = np.zeros((len(times), 10))
    mapping[:, 0] = np.arange(1, len(times) + 1)
    mapping[:, 3:6] = positions / 10.0
    mapping_file = tmp_path / "mapping.txt"
    np.savetxt(mapping_file, mapping)
    arrays = {}
    for name, values in {
        "digi_hit_pmt": np.arange(len(times)),
        "digi_hit_time": times,
        "digi_hit_charge": np.ones(len(times)),
        "digi_hit_trigger": np.zeros(len(times), dtype=int),
        "trigger_time": np.array([0.0]),
        "track_pid": np.array([211, -13, -11]),
        "track_parent": np.array([0, 211, -13]),
        "track_start_time": np.array([0.0, 15.0, 2000.0]),
        "track_start_position": np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], truth / 10.0]),
    }.items():
        arrays[name] = np.empty(1, dtype=object)
        arrays[name][0] = values
    arrays["pid"] = np.array([211])
    source = tmp_path / "sample.npz"
    np.savez_compressed(source, **arrays)
    clusters_csv = tmp_path / "clusters.csv"
    with clusters_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("input_file", "event_index", "rank", "delta_t_ns", "start_ns", "end_ns", "n_hits", "n_pmts", "charge"))
        writer.writeheader()
        writer.writerow({"input_file": str(source), "event_index": 0, "rank": 1,
                         "delta_t_ns": 2000, "start_ns": 1990, "end_ns": 2040,
                         "n_hits": len(times), "n_pmts": len(times), "charge": len(times)})
    script = Path(__file__).resolve().parents[1] / "scripts" / "reconstruct_delayed_vertices.py"
    output_dir = tmp_path / "out"
    subprocess.run([sys.executable, str(script), str(clusters_csv), "--mapping-file", str(mapping_file), "--output-dir", str(output_dir)], check=True, capture_output=True, text=True)
    with (output_dir / "vertices.csv").open(newline="", encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    estimate = np.array([float(row[key]) for key in ("x_mm", "y_mm", "z_mm")])
    assert row["status"] == "ok"
    assert np.linalg.norm(estimate - truth) < 100.0
    assert (output_dir / "delayed_vertices_tank_ry.png").is_file()
    assert row["truth_status"] == "matched"
    assert float(row["reco_truth_distance_3d_mm"]) < 100.0
    assert (output_dir / "delayed_vertices_truth_overlay.png").is_file()
    assert (output_dir / "delayed_vertex_residuals_xyz.png").is_file()
    residual_script = Path(__file__).resolve().parents[1] / "scripts" / "plot_delayed_vertex_residuals.py"
    rerendered = tmp_path / "residuals_again.png"
    subprocess.run([sys.executable, str(residual_script), str(output_dir / "vertices.csv"),
                    "--output", str(rerendered), "--bins", "12"],
                   check=True, capture_output=True, text=True)
    assert rerendered.is_file()
    filtered = subprocess.run([sys.executable, str(residual_script),
                               str(output_dir / "vertices.csv"), "--primary-pid", "211"],
                              check=True, capture_output=True, text=True)
    assert "Matched events: 1/1" in filtered.stdout
    assert (output_dir / "delayed_vertex_residuals_xyz_pid211.png").is_file()


def test_michel_truth_rejects_wrong_delayed_window():
    event = {"track_pid": [-11, -13, 211], "track_parent": [-13, 211, 0],
             "track_start_time": [2000.0, 15.0, 0.0],
             "track_start_position": [[12.0, -8.0, 20.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]}
    assert match_michel_truth(event, 2000.0)["status"] == "matched"
    assert match_michel_truth(event, 900.0)["status"] == "time_mismatch"


def test_delayed_hit_plot_reads_rank_one_clusters(tmp_path):
    events_csv = tmp_path / "events.csv"
    with events_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("input_file", "event_index", "primary_pid"))
        writer.writeheader()
        writer.writerow({"input_file": "sample.npz", "event_index": 0, "primary_pid": 211})
    clusters_csv = tmp_path / "clusters.csv"
    with clusters_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("input_file", "event_index", "rank", "n_hits", "n_pmts", "charge"))
        writer.writeheader()
        writer.writerow({"input_file": "sample.npz", "event_index": 0, "rank": 1,
                         "n_hits": 15, "n_pmts": 12, "charge": 17.5})
    script = Path(__file__).resolve().parents[1] / "scripts" / "plot_delayed_cluster_hits.py"
    output = tmp_path / "delayed_hits.png"
    subprocess.run([sys.executable, str(script), str(events_csv), "--output", str(output)],
                   check=True, capture_output=True, text=True)
    assert output.is_file()


def test_vertex_runner_handles_gamma_control_with_no_delayed_clusters(tmp_path):
    mapping = np.zeros((1, 10))
    mapping[0, 0] = 1
    mapping_file = tmp_path / "mapping.txt"
    np.savetxt(mapping_file, mapping)
    clusters_csv = tmp_path / "clusters.csv"
    with clusters_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("input_file", "event_index", "rank"))
        writer.writeheader()
    output_dir = tmp_path / "out"
    script = Path(__file__).resolve().parents[1] / "scripts" / "reconstruct_delayed_vertices.py"
    subprocess.run([sys.executable, str(script), str(clusters_csv),
                    "--mapping-file", str(mapping_file), "--output-dir", str(output_dir)],
                   check=True, capture_output=True, text=True)
    with (output_dir / "vertices.csv").open(newline="", encoding="utf-8") as handle:
        assert list(csv.DictReader(handle)) == []
    assert (output_dir / "delayed_vertex_residuals_xyz.png").is_file()
