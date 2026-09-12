#!/usr/bin/env python3
"""Run one LicketyFit event and save track/topology/shower diagnostics.

Place this file in the repository's ``scripts/`` directory and run it from the
repository root, for example:

    python scripts/run_pion_shower_smoke_test.py

The defaults target the included 800 MeV proton WCSim event.  This is a
mechanical smoke test, not a pion-versus-shower performance measurement.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import numpy as np


DEFAULT_SAMPLE = (
    "work_dir/sample_data/wcsim/"
    "p+_800MeV_absorptionLike_x0y0zn1348.npz"
)


def parse_float_list(text: str) -> list[float]:
    values = [float(item) for item in text.replace(";", ",").split(",") if item.strip()]
    if not values:
        raise argparse.ArgumentTypeError("At least one numeric value is required.")
    return values


def isotropic_direction_seeds(count: int = 49) -> list[tuple[float, float]]:
    """Return nearly equal-solid-angle direction seeds on the +z hemisphere.

    ``cz`` is sampled uniformly, as required for uniform solid angle, while a
    golden-angle azimuth avoids aligned rings.  The driver applies the selected
    ``direction_z_sign`` to ``cz``; therefore ``--both-directions`` mirrors
    these seeds onto the -z hemisphere and covers the full sphere.
    """

    if count < 1:
        raise ValueError("isotropic direction seed count must be at least 1")
    seeds = []
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    for index in range(count):
        # Cell centres avoid the coordinate singularities at theta=0 and pi.
        cz = (index + 0.5) / count
        transverse = np.sqrt(max(0.0, 1.0 - cz * cz))
        phi = index * golden_angle
        seeds.append(
            (
                float(transverse * np.cos(phi)),
                float(transverse * np.sin(phi)),
            )
        )
    return seeds


def legacy_ring_direction_seeds() -> list[tuple[float, float]]:
    """Return the original three-ring hemisphere seeds for reproducibility."""

    seeds = [(0.0, 0.0)]
    for transverse in (0.45, 0.75, 0.92):
        for phi in np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False):
            seeds.append(
                (
                    float(transverse * np.cos(phi)),
                    float(transverse * np.sin(phi)),
                )
            )
    return seeds


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Fit one WCSim event, calculate topology features, and optionally "
            "compare with the empirical shower hypothesis."
        )
    )
    parser.add_argument("--input", default=DEFAULT_SAMPLE, help="WCSim input file.")
    parser.add_argument("--event-index", type=int, default=0)
    parser.add_argument("--geometry-dir", default="../Geometry")
    parser.add_argument("--geometry-file", default=None)
    parser.add_argument("--table-dir", default="tables")

    parser.add_argument("--particle", default="proton")
    parser.add_argument("--energy-mev", type=float, default=800.0)
    parser.add_argument(
        "--fit-mode",
        choices=("full_length", "absorption"),
        default="absorption",
    )
    parser.add_argument(
        "--likelihood-mode",
        choices=("charge_time", "charge_only", "timing_only"),
        default="charge_time",
    )
    parser.add_argument("--fixed-z-mm", type=float, default=-1348.0)
    parser.add_argument(
        "--z-seeds-mm",
        type=parse_float_list,
        default=None,
        help="Initial z0 seeds used when --free-z is enabled.",
    )
    parser.add_argument(
        "--free-z",
        action="store_true",
        help="Float z0 instead of fixing it to --fixed-z-mm.",
    )
    parser.add_argument(
        "--free-energy",
        action="store_true",
        help="Do not fix ke0_mev to --energy-mev in absorption mode.",
    )
    parser.add_argument(
        "--direction-z-sign",
        type=int,
        choices=(-1, 1),
        default=1,
        help="Discrete sign used in cz=sign*sqrt(1-cx^2-cy^2).",
    )
    parser.add_argument(
        "--both-directions",
        action="store_true",
        help="Fit both cz hemispheres and retain the lower-NLL result.",
    )
    parser.add_argument(
        "--direction-seeding",
        choices=("beam", "legacy_rings", "isotropic"),
        default="beam",
        help=(
            "Use near-beam seeds, the previous three-ring seeds, or new "
            "equal-solid-angle seeds spanning each fitted z hemisphere."
        ),
    )
    parser.add_argument(
        "--direction-seed-count",
        type=int,
        default=49,
        help=(
            "Number of equal-solid-angle cell-centre seeds per hemisphere. "
            "With --both-directions, twice this many directions are tested."
        ),
    )
    parser.add_argument(
        "--track-direction-parameterization",
        choices=("theta_phi", "cx_cy"),
        default="theta_phi",
        help=(
            "Smooth spherical angles for Minuit (recommended), or the legacy "
            "constrained direction-cosine parameterization."
        ),
    )

    parser.add_argument("--ncall", type=int, default=15000)
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument(
        "--visible-length-seeds-mm",
        type=parse_float_list,
        default=[300.0, 700.0, 1100.0, 1300.0],
    )
    parser.add_argument(
        "--ke-seeds-mev",
        type=parse_float_list,
        default=[100.0, 150.0, 200.0, 300.0, 450.0, 600.0],
        help=(
            "Initial kinetic-energy seeds used when absorption-mode energy is "
            "free. These are pion kinetic energies, not tagged photon energies."
        ),
    )
    parser.add_argument(
        "--x-seeds-mm",
        type=parse_float_list,
        default=[-150.0, 0.0, 150.0],
    )
    parser.add_argument(
        "--y-seeds-mm",
        type=parse_float_list,
        default=[-150.0, 0.0, 150.0],
    )

    parser.add_argument(
        "--shower",
        action="store_true",
        help="Also fit the empirical one-point shower hypothesis.",
    )
    parser.add_argument("--shower-ncall", type=int, default=5000)
    parser.add_argument(
        "--shower-width-max-deg",
        type=float,
        default=60.0,
        help="Upper bound on Gaussian ring-width sigma (not the Cherenkov angle).",
    )
    parser.add_argument(
        "--shower-max-beam-angle-deg",
        type=float,
        default=30.0,
        help=(
            "Constrain the shower axis within this angle of the +z tagged-gamma "
            "beam. The pion direction remains unrestricted."
        ),
    )
    parser.add_argument(
        "--shower-vertex-half-width-mm",
        type=float,
        default=1500.0,
        help="Allowed displacement in each coordinate from the shower vertex seed.",
    )
    parser.add_argument(
        "--shower-seed",
        choices=("prompt", "track"),
        default="prompt",
        help="Seed shower vertex from prompt-hit multilateration or the track fit.",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory. Default: outputs/smoke_event_<index>_<timestamp>.",
    )
    return parser


def resolve_from_root(value: str | Path, root: Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = root / path
    return path.resolve()


def load_payload(path: Path) -> Mapping[str, Any]:
    loaded = np.load(path, allow_pickle=True)
    if hasattr(loaded, "files"):
        return {name: loaded[name] for name in loaded.files}
    if isinstance(loaded, np.ndarray) and loaded.shape == ():
        item = loaded.item()
        if isinstance(item, Mapping):
            return item
    if isinstance(loaded, Mapping):
        return loaded
    return {}


def event_value(payload: Mapping[str, Any], name: str, event_index: int) -> Any:
    if name not in payload:
        return None
    values = payload[name]
    try:
        return values[event_index]
    except (IndexError, TypeError):
        return None


def compact_value(value: Any, max_items: int = 8) -> Any:
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    flattened = array.ravel()
    if flattened.size <= max_items and array.dtype != object:
        return array.tolist()
    return {
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        "first_values": [
            item.item() if isinstance(item, np.generic) else str(item)
            for item in flattened[:max_items]
        ],
    }


def extract_truth(payload: Mapping[str, Any], event_index: int) -> dict[str, Any]:
    truth = {}
    for name in (
        "position",
        "direction",
        "energy",
        "true_vis_length",
        "track_start_position",
        "track_stop_position",
    ):
        value = event_value(payload, name, event_index)
        if value is not None:
            truth[name] = compact_value(value)
    return truth


def json_safe(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def print_mapping(title: str, values: Mapping[str, Any]) -> None:
    print(f"\n{title}")
    for name, value in values.items():
        print(f"{name:48s} {value}")


def selected_topology_features(features: Mapping[str, float]) -> dict[str, float]:
    names = (
        "n_hit_pmts",
        "total_charge_pe",
        "mean_charge_per_hit_pe",
        "charge_top_10pct_fraction",
        "effective_n_hit_pmts",
        "beam_angle_mean_deg",
        "beam_angle_rms_deg",
        "charge_fraction_within_beam_30deg",
        "tof_residual_rms_ns",
        "prompt_charge_fraction",
        "fit_charge_poisson_deviance_per_dof",
        "fit_charge_shape_l1_distance",
        "fit_timing_residual_rms_ns",
        "fit_timing_pull_rms",
    )
    return {name: features.get(name, float("nan")) for name in names}


def main() -> int:
    args = build_parser().parse_args()

    # This script is intended to live in <repo>/scripts/.
    script_path = Path(__file__).resolve()
    root = script_path.parent.parent
    if not (root / "LicketyFit").is_dir() or not (root / "scripts").is_dir():
        raise RuntimeError(
            "Place this script in the repository's scripts/ directory before running it."
        )

    input_path = resolve_from_root(args.input, root)
    geometry_dir = resolve_from_root(args.geometry_dir, root)
    geometry_file = (
        resolve_from_root(args.geometry_file, root)
        if args.geometry_file
        else geometry_dir / "examples/wcte_bldg157.geo"
    )
    table_dir = resolve_from_root(args.table_dir, root)

    for path, label in (
        (input_path, "input file"),
        (geometry_file, "geometry file"),
        (table_dir, "table directory"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} does not exist: {path}")

    # Geometry is a sibling repository.  Importing Geometry.Device requires the
    # directory containing the Geometry package on sys.path.
    for path in (root, geometry_dir.parent):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    os.environ["PYTHONPATH"] = os.pathsep.join(
        [str(geometry_dir.parent), os.environ.get("PYTHONPATH", "")]
    ).rstrip(os.pathsep)

    from scripts.fit_single_event import (
        SingleEventConfig,
        fit_single_event,
        fit_track_both_directions,
        summarize_result,
        topology_features,
    )

    payload = load_payload(input_path)
    truth = extract_truth(payload, args.event_index)
    if truth:
        print_mapping("AVAILABLE SIMULATION TRUTH", truth)

    fixed_params: dict[str, float | None] = {
        "z0": None if args.free_z else float(args.fixed_z_mm)
    }
    if args.fit_mode == "absorption":
        fixed_params["ke0_mev"] = (
            None if args.free_energy else float(args.energy_mev)
        )

    config = SingleEventConfig(
        data_kind="wcsim",
        wcsim_input_file=input_path,
        event_index=args.event_index,
        geometry_path=geometry_dir,
        geometry_file=geometry_file,
        table_dir=table_dir,
        fit_particle=args.particle,
        fit_mode=args.fit_mode,
        likelihood_mode=args.likelihood_mode,
        energy_true=args.energy_mev,
        direction_z_sign=args.direction_z_sign,
        direction_parameterization=args.track_direction_parameterization,
        ring_mask_mode="none",
        fixed_params=fixed_params,
        fast_seed_x0=args.x_seeds_mm,
        fast_seed_y0=args.y_seeds_mm,
        fast_seed_z0=(
            [args.fixed_z_mm]
            if not args.free_z
            else (args.z_seeds_mm or [-1500.0, -1300.0, -1100.0])
        ),
        fast_seed_visible_lengths=args.visible_length_seeds_mm,
        fast_seed_ke0_mev=args.ke_seeds_mev,
        fast_seed_directions=(
            isotropic_direction_seeds(args.direction_seed_count)
            if args.direction_seeding == "isotropic"
            else (
                legacy_ring_direction_seeds()
                if args.direction_seeding == "legacy_rings"
                else [
                    (0.0, 0.0),
                    (0.04, 0.0),
                    (-0.04, 0.0),
                    (0.0, 0.04),
                    (0.0, -0.04),
                ]
            )
        ),
        fast_seed_full_cartesian=False,
        max_fit_attempts=args.max_attempts,
        ncall_migrad=args.ncall,
        ncall_simplex=args.ncall,
        enable_stage2_migrad_first=False,
        enable_stage3_adaptive_rescue=False,
        enable_stage4_length_profile=False,
        save_top_n_seeds=5,
        verbose=True,
    )

    print("\nTRACK FIT")
    print("Input:", input_path)
    print("Event:", args.event_index)
    print("Geometry:", geometry_file)
    print("Numba may compile kernels during the first run.")

    direction_scan = None
    if args.both_directions:
        direction_scan = fit_track_both_directions(config)
        track = direction_scan["best"]
        print("Best direction-z sign:", direction_scan["best_direction_z_sign"])
    else:
        track = fit_single_event(config)

    summary = summarize_result(track)
    diagnostics = {
        name: track["result"].get(name)
        for name in (
            "edm",
            "seed_stuck",
            "below_t_min",
            "visible_length_too_large",
            "attempts",
            "chosen_seed_index",
            "chosen_seed_fcn",
        )
    }
    features = topology_features(track, beam_direction=(0.0, 0.0, 1.0))
    if args.likelihood_mode == "charge_only":
        for name in (
            "fit_n_timing_residual_pmts",
            "fit_timing_residual_mean_ns",
            "fit_timing_residual_rms_ns",
            "fit_timing_pull_rms",
        ):
            features[name] = float("nan")

    print_mapping("TRACK FIT SUMMARY", summary)
    print_mapping("MINUIT DIAGNOSTICS", diagnostics)
    print_mapping("SELECTED TOPOLOGY FEATURES", selected_topology_features(features))
    print(
        "\nNote: vertex-based track-ring features are intentionally omitted from "
        "this report because they are not reliable for a long extended track."
    )

    shower_summary = None
    comparison = None
    shower = None
    if args.shower:
        from LicketyFit.Analysis import compare_fit_hypotheses
        from LicketyFit.ShowerFitter import ShowerFitConfig, ShowerFitter

        print("\nSHOWER FIT")
        shower_model = ShowerFitter(
            track["p_locations"],
            pmt_direction_zs=track["direction_zs"],
            config=ShowerFitConfig(
                ncall=args.shower_ncall,
                vertex_half_width_mm=(
                    args.shower_vertex_half_width_mm,
                    args.shower_vertex_half_width_mm,
                    args.shower_vertex_half_width_mm,
                ),
                angular_width_bounds_deg=(2.0, args.shower_width_max_deg),
                direction_theta_bounds_deg=(0.0, args.shower_max_beam_angle_deg),
                include_timing=args.likelihood_mode == "charge_time",
            ),
        )
        shower_seed_diagnostics = None
        if args.shower_seed == "prompt":
            nominal_seed = (0.0, 0.0, float(args.fixed_z_mm))
            shower_seed_diagnostics = shower_model.prompt_multilateration_seed(
                track["obs_pes"],
                track["obs_ts"],
                initial_vertex_mm=nominal_seed,
            )
            shower_vertex_seed = shower_seed_diagnostics["vertex_mm"]
            print_mapping(
                "PROMPT MULTILATERATION SEED",
                {
                    name: value
                    for name, value in shower_seed_diagnostics.items()
                    if name != "selected_pmt_indices"
                },
            )
        else:
            shower_vertex_seed = (
                track["values"]["x0"],
                track["values"]["y0"],
                track["values"]["z0"],
            )
        shower = shower_model.fit(
            track["obs_pes"],
            track["obs_ts"],
            vertex_seed_mm=shower_vertex_seed,
            direction_seed=(0.0, 0.0, 1.0),
            width_seed_deg=10.0,
        )
        shower_summary = {
            "valid": shower["valid"],
            "nll": shower["fval"],
            "effective_vertex_mm": shower["effective_vertex_mm"],
            "direction": shower["shower_direction"],
            "angular_width_deg": shower["angular_width_deg"],
            "ring_angular_sigma_deg": shower["ring_angular_sigma_deg"],
            "cherenkov_angle_deg": shower["metadata"]["cherenkov_angle_deg"],
            "max_beam_angle_deg": args.shower_max_beam_angle_deg,
            "angular_width_at_limit": shower["angular_width_at_limit"],
            "total_detected_pe": shower["total_detected_pe"],
        }
        comparison = compare_fit_hypotheses(track, shower)
        print_mapping("SHOWER FIT SUMMARY", shower_summary)
        print_mapping("TRACK VERSUS SHOWER", comparison)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = (
        resolve_from_root(args.output_dir, root)
        if args.output_dir
        else root / "outputs" / f"smoke_event_{args.event_index}_{timestamp}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    report = {
        "input_file": str(input_path),
        "event_index": args.event_index,
        "truth": truth,
        "track_summary": summary,
        "track_diagnostics": diagnostics,
        "topology_features": features,
        "direction_scan": (
            None
            if direction_scan is None
            else {
                "best_direction_z_sign": direction_scan["best_direction_z_sign"],
                "two_delta_nll_positive_minus_negative": direction_scan[
                    "two_delta_nll_positive_minus_negative"
                ],
            }
        ),
        "shower_summary": shower_summary,
        "shower_seed": (
            None
            if not args.shower or shower_seed_diagnostics is None
            else {
                name: value
                for name, value in shower_seed_diagnostics.items()
                if name != "selected_pmt_indices"
            }
        ),
        "track_shower_comparison": comparison,
    }
    with (output_dir / "report.json").open("w", encoding="utf-8") as handle:
        json.dump(json_safe(report), handle, indent=2, sort_keys=True)

    arrays = {
        "obs_pes": np.asarray(track["obs_pes"]),
        "obs_ts": np.asarray(track["obs_ts"]),
        "track_exp_pes": np.asarray(track["exp_pes"]),
        "track_exp_ts": np.asarray(track["exp_ts"]),
        "p_locations": np.asarray(track["p_locations"]),
        "pmt_direction_zs": np.asarray(track["direction_zs"]),
        "track_direction": np.asarray(track["track_direction"]),
    }
    if shower is not None:
        arrays["shower_exp_pes"] = np.asarray(shower["exp_pes"])
        arrays["shower_exp_ts"] = np.asarray(shower["exp_ts"])
        arrays["shower_direction"] = np.asarray(shower["shower_direction"])
    np.savez_compressed(output_dir / "diagnostic_arrays.npz", **arrays)

    print("\nOUTPUT")
    print("Report:", output_dir / "report.json")
    print("Arrays:", output_dir / "diagnostic_arrays.npz")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
