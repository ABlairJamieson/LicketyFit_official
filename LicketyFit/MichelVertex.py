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
