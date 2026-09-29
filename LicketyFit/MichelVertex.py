"""Exploratory point multilateration of a delayed PMT-hit cluster."""

from __future__ import annotations

import numpy as np

from .ShowerFitter import ShowerFitConfig, ShowerFitter


def fit_delayed_point_vertex(
    pmt_positions_mm: np.ndarray,
    pmt_ids: np.ndarray,
    charges: np.ndarray,
    times_ns: np.ndarray,
    *,
    min_pmts: int = 12,
) -> dict:
    """Fit an effective point of light using the existing timing multilaterator.

    Multiple digits on a PMT contribute their summed charge and earliest time.
    An electron emits along a short track, so this is an effective light origin,
    not an unbiased Michel-positron production vertex.
    """
    positions = np.asarray(pmt_positions_mm, dtype=float)
    ids = np.asarray(pmt_ids, dtype=int).reshape(-1)
    charge = np.asarray(charges, dtype=float).reshape(-1)
    times = np.asarray(times_ns, dtype=float).reshape(-1)
    if positions.shape != (len(ids), 3) or len(charge) != len(ids) or len(times) != len(ids):
        raise ValueError("PMT positions, IDs, charges, and times must align")
    keep = np.all(np.isfinite(positions), axis=1) & np.isfinite(times) & np.isfinite(charge) & (charge > 0)
    positions, ids, charge, times = positions[keep], ids[keep], charge[keep], times[keep]
    unique_ids, inverse = np.unique(ids, return_inverse=True)
    if len(unique_ids) < min_pmts:
        raise ValueError(f"Only {len(unique_ids)} mapped PMTs; need {min_pmts}")
    first_time = np.full(len(unique_ids), np.inf)
    total_charge = np.zeros(len(unique_ids))
    np.minimum.at(first_time, inverse, times)
    np.add.at(total_charge, inverse, charge)
    first_positions = positions[np.unique(inverse, return_index=True)[1]]
    config = ShowerFitConfig(
        refractive_index=1.373,
        prompt_seed_min_pmts=min_pmts,
        prompt_seed_fraction=0.5,
        prompt_seed_max_pmts=80,
        prompt_seed_iterations=30,
    )
    result = ShowerFitter(first_positions, config=config).prompt_multilateration_seed(
        total_charge, first_time, initial_vertex_mm=(0.0, 0.0, 0.0)
    )
    result["n_window_pmts"] = int(len(unique_ids))
    return result


def match_michel_truth(event: dict, delta_t_ns: float, *, tolerance_ns: float = 100.0) -> dict:
    """Match a delayed cluster to a muon-decay e+/e- track by relative time.

    DataTools stores WCSim track positions in cm and ``track_parent`` as the
    parent's PDG code, not its track ID.  This can identify a muon daughter,
    but cannot prove which individual muon produced it.  Relative times use
    the earliest primary track as the event reference and need validation on
    each simulation production before interpreting match efficiency.
    """
    required = ("track_pid", "track_parent", "track_start_time", "track_start_position")
    if any(key not in event for key in required):
        return {"status": "missing_truth"}
    pids = np.asarray(event["track_pid"], dtype=int).reshape(-1)
    parents = np.asarray(event["track_parent"], dtype=int).reshape(-1)
    times = np.asarray(event["track_start_time"], dtype=float).reshape(-1)
    positions_cm = np.asarray(event["track_start_position"], dtype=float)
    if not (len(pids) == len(parents) == len(times)) or positions_cm.shape != (len(pids), 3):
        return {"status": "invalid_truth_shapes"}
    if not len(times) or not np.isfinite(delta_t_ns):
        return {"status": "invalid_truth_time"}
    finite = np.isfinite(times) & np.all(np.isfinite(positions_cm), axis=1)
    primary = finite & (parents == 0)
    if not np.any(primary):
        return {"status": "no_primary_time"}
    t_reference = float(np.min(times[primary]))
    daughter = finite & (((pids == -11) & (parents == -13)) |
                         ((pids == 11) & (parents == 13)))
    indices = np.flatnonzero(daughter)
    if not len(indices):
        return {"status": "no_muon_decay_electron", "n_truth_candidates": 0}
    delays = times[indices] - t_reference
    offsets = np.abs(delays - delta_t_ns)
    best = int(np.argmin(offsets))
    index = int(indices[best])
    result = {"status": "matched" if offsets[best] <= tolerance_ns else "time_mismatch",
              "n_truth_candidates": int(len(indices)),
              "truth_track_index": index,
              "truth_pdg": int(pids[index]),
              "truth_delay_ns": float(delays[best]),
              "truth_delay_difference_ns": float(offsets[best])}
    if result["status"] == "matched":
        result["truth_vertex_mm"] = positions_cm[index] * 10.0
    return result
