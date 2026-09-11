"""Topology features and fit diagnostics for WCTE pion/shower studies.

The functions in this module are intentionally independent of the detector
geometry package.  They operate on the flat arrays already returned by
``scripts.fit_single_event``:

``obs_pes``, ``obs_ts``, ``p_locations``, ``exp_pes``, and ``exp_ts``.

Feature definitions use detected photoelectrons and PMT locations.  Quantities
such as the charge-weighted PMT radius are detector-response observables, not a
direct measurement of the physical shower radius.  They should be calibrated
and validated with simulation and control samples before receiving a physics
interpretation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


_EPS = 1.0e-12


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration shared by topology-feature calculations."""

    beam_direction: tuple[float, float, float] = (0.0, 0.0, 1.0)
    reference_vertex_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
    refractive_index: float = 1.344
    vacuum_light_speed_mm_per_ns: float = 299.792458
    cherenkov_angle_deg: float = 41.8
    ring_half_width_deg: float = 5.0
    prompt_window_ns: float = 2.0
    forward_cos_thresholds: tuple[float, ...] = (0.8, 0.9)
    forward_angle_thresholds_deg: tuple[float, ...] = (15.0, 30.0, 45.0)

    @property
    def light_speed_mm_per_ns(self) -> float:
        return self.vacuum_light_speed_mm_per_ns / self.refractive_index


def _vector3(value: Sequence[float], name: str, *, unit: bool = False) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.shape != (3,) or not np.all(np.isfinite(out)):
        raise ValueError(f"{name} must contain three finite numbers.")
    if unit:
        norm = float(np.linalg.norm(out))
        if norm <= 0.0:
            raise ValueError(f"{name} must have non-zero length.")
        out = out / norm
    return out


def _one_dimensional(value: Any, name: str, *, length: int | None = None) -> np.ndarray:
    out = np.asarray(value, dtype=np.float64)
    if out.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional array.")
    if length is not None and out.size != length:
        raise ValueError(f"{name} has length {out.size}; expected {length}.")
    return out


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    total = float(np.sum(weights))
    if total <= 0.0:
        return float("nan")
    return float(np.sum(weights * values) / total)


def _weighted_std(values: np.ndarray, weights: np.ndarray) -> float:
    mean = _weighted_mean(values, weights)
    if not np.isfinite(mean):
        return float("nan")
    variance = _weighted_mean((values - mean) ** 2, weights)
    return float(np.sqrt(max(0.0, variance)))


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if values.size == 0 or float(np.sum(weights)) <= 0.0:
        return float("nan")
    order = np.argsort(values)
    sorted_values = values[order]
    sorted_weights = weights[order]
    cumulative = np.cumsum(sorted_weights) - 0.5 * sorted_weights
    cumulative /= float(np.sum(sorted_weights))
    return float(np.interp(float(quantile), cumulative, sorted_values))


def _safe_angle_degrees(cosine: np.ndarray | float) -> np.ndarray:
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _suffix(value: float) -> str:
    return f"{float(value):g}".replace("-", "m").replace(".", "p")


def _poisson_deviance(observed: np.ndarray, expected: np.ndarray) -> float:
    mu = np.clip(expected, _EPS, None)
    positive = observed > 0.0
    terms = mu - observed
    terms[positive] += observed[positive] * np.log(observed[positive] / mu[positive])
    return float(2.0 * np.sum(terms))


def _add_charge_features(features: dict[str, float], charges: np.ndarray) -> None:
    hit = charges > 0.0
    hit_charge = charges[hit]
    n_hit = int(np.count_nonzero(hit))
    total = float(np.sum(hit_charge))

    features["n_active_pmts"] = float(charges.size)
    features["n_hit_pmts"] = float(n_hit)
    features["hit_pmt_fraction"] = float(n_hit / charges.size) if charges.size else float("nan")
    features["total_charge_pe"] = total

    if n_hit == 0:
        for name in (
            "mean_charge_per_hit_pe",
            "charge_rms_per_hit_pe",
            "charge_cv_per_hit",
            "charge_q10_pe",
            "charge_q50_pe",
            "charge_q90_pe",
            "charge_top_10pct_fraction",
            "effective_n_hit_pmts",
            "effective_hit_fraction",
        ):
            features[name] = float("nan")
        return

    mean = float(np.mean(hit_charge))
    rms = float(np.std(hit_charge))
    features["mean_charge_per_hit_pe"] = mean
    features["charge_rms_per_hit_pe"] = rms
    features["charge_cv_per_hit"] = rms / mean if mean > 0.0 else float("nan")
    features["charge_q10_pe"] = float(np.quantile(hit_charge, 0.10))
    features["charge_q50_pe"] = float(np.quantile(hit_charge, 0.50))
    features["charge_q90_pe"] = float(np.quantile(hit_charge, 0.90))

    n_top = max(1, int(np.ceil(0.10 * n_hit)))
    features["charge_top_10pct_fraction"] = (
        float(np.sum(np.partition(hit_charge, -n_top)[-n_top:]) / total)
        if total > 0.0
        else float("nan")
    )
    effective = total * total / float(np.sum(hit_charge * hit_charge))
    features["effective_n_hit_pmts"] = effective
    features["effective_hit_fraction"] = effective / n_hit


def _add_time_features(
    features: dict[str, float],
    charges: np.ndarray,
    times: np.ndarray,
    *,
    distances_mm: np.ndarray | None,
    config: FeatureConfig,
) -> None:
    mask = (charges > 0.0) & np.isfinite(times)
    features["n_timed_pmts"] = float(np.count_nonzero(mask))
    if not np.any(mask):
        for name in (
            "hit_time_mean_ns",
            "hit_time_rms_ns",
            "hit_time_iqr_ns",
            "tof_residual_rms_ns",
            "tof_residual_iqr_ns",
            "tof_residual_mad_ns",
            "prompt_charge_fraction",
        ):
            features[name] = float("nan")
        return

    q = charges[mask]
    t = times[mask]
    features["hit_time_mean_ns"] = _weighted_mean(t, q)
    features["hit_time_rms_ns"] = _weighted_std(t, q)
    features["hit_time_iqr_ns"] = _weighted_quantile(t, q, 0.75) - _weighted_quantile(t, q, 0.25)

    if distances_mm is None:
        residual = t
    else:
        residual = t - distances_mm[mask] / config.light_speed_mm_per_ns

    centre = _weighted_quantile(residual, q, 0.50)
    centred = residual - centre
    features["tof_residual_rms_ns"] = _weighted_std(residual, q)
    features["tof_residual_iqr_ns"] = (
        _weighted_quantile(residual, q, 0.75) - _weighted_quantile(residual, q, 0.25)
    )
    features["tof_residual_mad_ns"] = _weighted_quantile(np.abs(centred), q, 0.50)
    features["prompt_charge_fraction"] = float(
        np.sum(q[np.abs(centred) <= config.prompt_window_ns]) / np.sum(q)
    )


def _add_geometry_features(
    features: dict[str, float],
    charges: np.ndarray,
    positions: np.ndarray,
    *,
    vertex: np.ndarray,
    beam_direction: np.ndarray,
    fitted_direction: np.ndarray | None,
    config: FeatureConfig,
) -> np.ndarray:
    relative = positions - vertex[None, :]
    distances = np.linalg.norm(relative, axis=1)
    valid = (charges > 0.0) & np.isfinite(distances) & (distances > 0.0)
    if not np.any(valid):
        return distances

    q = charges[valid]
    rel = relative[valid]
    r = distances[valid]
    rays = rel / r[:, None]

    longitudinal = rel @ beam_direction
    transverse = np.sqrt(np.maximum(0.0, r * r - longitudinal * longitudinal))
    features["beam_transverse_radius_mean_mm"] = _weighted_mean(transverse, q)
    features["beam_transverse_radius_rms_mm"] = float(np.sqrt(_weighted_mean(transverse**2, q)))
    features["beam_longitudinal_mean_mm"] = _weighted_mean(longitudinal, q)
    features["beam_longitudinal_rms_mm"] = _weighted_std(longitudinal, q)

    centroid = np.sum(q[:, None] * positions[valid], axis=0) / np.sum(q)
    features["charge_centroid_x_mm"] = float(centroid[0])
    features["charge_centroid_y_mm"] = float(centroid[1])
    features["charge_centroid_z_mm"] = float(centroid[2])
    centred_positions = positions[valid] - centroid[None, :]
    spatial_r2 = np.sum(centred_positions * centred_positions, axis=1)
    features["charge_weighted_spatial_rms_mm"] = float(np.sqrt(_weighted_mean(spatial_r2, q)))

    beam_cos = rays @ beam_direction
    beam_angle = _safe_angle_degrees(beam_cos)
    features["beam_cos_mean"] = _weighted_mean(beam_cos, q)
    features["beam_cos_rms"] = _weighted_std(beam_cos, q)
    features["beam_angle_mean_deg"] = _weighted_mean(beam_angle, q)
    features["beam_angle_rms_deg"] = _weighted_std(beam_angle, q)
    for threshold in config.forward_cos_thresholds:
        features[f"charge_fraction_beam_cos_gt_{_suffix(threshold)}"] = float(
            np.sum(q[beam_cos > threshold]) / np.sum(q)
        )
    for threshold in config.forward_angle_thresholds_deg:
        features[f"charge_fraction_within_beam_{_suffix(threshold)}deg"] = float(
            np.sum(q[beam_angle <= threshold]) / np.sum(q)
        )

    resultant = np.sum(q[:, None] * rays, axis=0) / np.sum(q)
    resultant_length = float(np.linalg.norm(resultant))
    features["angular_resultant_length"] = resultant_length
    if resultant_length > 0.0:
        charge_axis = resultant / resultant_length
        features["charge_axis_beam_angle_deg"] = float(
            _safe_angle_degrees(np.dot(charge_axis, beam_direction))
        )
        spread = _safe_angle_degrees(rays @ charge_axis)
        features["angular_spread_about_charge_axis_deg"] = float(
            np.sqrt(_weighted_mean(spread**2, q))
        )
    else:
        features["charge_axis_beam_angle_deg"] = float("nan")
        features["angular_spread_about_charge_axis_deg"] = float("nan")

    ray_centroid = np.sum(q[:, None] * rays, axis=0) / np.sum(q)
    ray_delta = rays - ray_centroid[None, :]
    covariance = (ray_delta * q[:, None]).T @ ray_delta / np.sum(q)
    eigenvalues = np.sort(np.linalg.eigvalsh(covariance))[::-1]
    eigen_sum = float(np.sum(eigenvalues))
    if eigen_sum > 0.0:
        features["angular_pca_largest_fraction"] = float(eigenvalues[0] / eigen_sum)
        features["angular_pca_second_fraction"] = float(eigenvalues[1] / eigen_sum)
        features["angular_pca_smallest_fraction"] = float(eigenvalues[2] / eigen_sum)
        features["angular_pca_linearity"] = float((eigenvalues[0] - eigenvalues[1]) / eigen_sum)
        features["angular_pca_planarity"] = float((eigenvalues[1] - eigenvalues[2]) / eigen_sum)

    if fitted_direction is not None:
        track_cos = rays @ fitted_direction
        track_angle = _safe_angle_degrees(track_cos)
        ring_residual = track_angle - config.cherenkov_angle_deg
        features["fit_direction_beam_angle_deg"] = float(
            _safe_angle_degrees(np.dot(fitted_direction, beam_direction))
        )
        features["track_viewing_angle_mean_deg"] = _weighted_mean(track_angle, q)
        features["track_viewing_angle_rms_deg"] = _weighted_std(track_angle, q)
        features["track_ring_residual_mean_deg"] = _weighted_mean(ring_residual, q)
        features["track_ring_residual_rms_deg"] = float(
            np.sqrt(_weighted_mean(ring_residual**2, q))
        )
        features["track_ring_abs_residual_mean_deg"] = _weighted_mean(np.abs(ring_residual), q)
        features["track_ring_charge_fraction"] = float(
            np.sum(q[np.abs(ring_residual) <= config.ring_half_width_deg]) / np.sum(q)
        )

    return distances


def _add_fit_features(
    features: dict[str, float],
    observed: np.ndarray,
    expected: np.ndarray,
    *,
    observed_times: np.ndarray | None,
    expected_times: np.ndarray | None,
    n_fit_parameters: int,
    fit_nll: float | None,
    fit_valid: bool | None,
    time_resolution_ns: float,
) -> None:
    mu = np.asarray(expected, dtype=np.float64)
    if mu.shape != observed.shape:
        raise ValueError("expected_charges_pe must have the same shape as charges_pe.")
    if np.any(~np.isfinite(mu)) or np.any(mu < 0.0):
        raise ValueError("expected_charges_pe must be finite and non-negative.")

    n_dof = max(1, observed.size - int(n_fit_parameters))
    deviance = _poisson_deviance(observed, mu)
    features["fit_charge_poisson_deviance"] = deviance
    features["fit_charge_poisson_deviance_per_dof"] = deviance / n_dof
    features["fit_charge_pearson_chi2_per_dof"] = float(
        np.sum((observed - mu) ** 2 / np.clip(mu, 1.0, None)) / n_dof
    )
    features["fit_expected_total_charge_pe"] = float(np.sum(mu))
    features["fit_charge_total_ratio_expected_over_observed"] = (
        float(np.sum(mu) / np.sum(observed)) if np.sum(observed) > 0.0 else float("nan")
    )

    obs_sum = float(np.sum(observed))
    mu_sum = float(np.sum(mu))
    if obs_sum > 0.0 and mu_sum > 0.0:
        obs_shape = observed / obs_sum
        mu_shape = mu / mu_sum
        features["fit_charge_shape_l1_distance"] = float(0.5 * np.sum(np.abs(obs_shape - mu_shape)))
        features["fit_charge_cosine_similarity"] = float(
            np.dot(observed, mu)
            / (np.linalg.norm(observed) * np.linalg.norm(mu) + _EPS)
        )
    else:
        features["fit_charge_shape_l1_distance"] = float("nan")
        features["fit_charge_cosine_similarity"] = float("nan")

    if fit_nll is not None and np.isfinite(fit_nll):
        features["fit_nll"] = float(fit_nll)
        features["fit_nll_per_active_pmt"] = float(fit_nll) / max(1, observed.size)
        features["fit_nll_per_dof"] = float(fit_nll) / n_dof
        features["fit_aic"] = 2.0 * n_fit_parameters + 2.0 * float(fit_nll)
        features["fit_bic"] = (
            np.log(max(1, observed.size)) * n_fit_parameters + 2.0 * float(fit_nll)
        )
    if fit_valid is not None:
        features["fit_valid"] = float(bool(fit_valid))

    if observed_times is None or expected_times is None:
        return
    exp_t = _one_dimensional(expected_times, "expected_times_ns", length=observed.size)
    mask = (
        (observed > 0.0)
        & (mu > 0.0)
        & np.isfinite(observed_times)
        & np.isfinite(exp_t)
    )
    features["fit_n_timing_residual_pmts"] = float(np.count_nonzero(mask))
    if not np.any(mask):
        features["fit_timing_residual_mean_ns"] = float("nan")
        features["fit_timing_residual_rms_ns"] = float("nan")
        features["fit_timing_pull_rms"] = float("nan")
        return

    residual = observed_times[mask] - exp_t[mask]
    q = observed[mask]
    features["fit_timing_residual_mean_ns"] = _weighted_mean(residual, q)
    features["fit_timing_residual_rms_ns"] = float(np.sqrt(_weighted_mean(residual**2, q)))
    sigma = float(time_resolution_ns) / np.sqrt(np.clip(q, _EPS, None))
    features["fit_timing_pull_rms"] = float(np.sqrt(np.mean((residual / sigma) ** 2)))


def extract_event_features(
    charges_pe: Sequence[float],
    *,
    times_ns: Sequence[float] | None = None,
    pmt_positions_mm: Sequence[Sequence[float]] | None = None,
    reference_vertex_mm: Sequence[float] | None = None,
    beam_direction: Sequence[float] | None = None,
    fitted_direction: Sequence[float] | None = None,
    expected_charges_pe: Sequence[float] | None = None,
    expected_times_ns: Sequence[float] | None = None,
    n_fit_parameters: int = 0,
    fit_nll: float | None = None,
    fit_valid: bool | None = None,
    time_resolution_ns: float = 1.0,
    n_pulses: int | None = None,
    config: FeatureConfig | None = None,
) -> dict[str, float]:
    """Extract low-level, geometry, timing, and optional fit-quality features.

    ``charges_pe`` should contain one entry for every active PMT, including
    zero-charge PMTs.  Including unhit PMTs is essential for meaningful
    likelihood/deviance features.
    """

    cfg = config or FeatureConfig()
    charges = _one_dimensional(charges_pe, "charges_pe")
    if np.any(~np.isfinite(charges)) or np.any(charges < 0.0):
        raise ValueError("charges_pe must be finite and non-negative.")

    times = None
    if times_ns is not None:
        times = _one_dimensional(times_ns, "times_ns", length=charges.size)

    features: dict[str, float] = {}
    _add_charge_features(features, charges)
    if n_pulses is not None:
        features["n_pulses"] = float(n_pulses)
        features["pulses_per_hit_pmt"] = (
            float(n_pulses) / features["n_hit_pmts"]
            if features["n_hit_pmts"] > 0.0
            else float("nan")
        )

    distances = None
    if pmt_positions_mm is not None:
        positions = np.asarray(pmt_positions_mm, dtype=np.float64)
        if positions.shape != (charges.size, 3) or np.any(~np.isfinite(positions)):
            raise ValueError("pmt_positions_mm must be a finite array with shape (n_pmts, 3).")
        vertex = _vector3(
            cfg.reference_vertex_mm if reference_vertex_mm is None else reference_vertex_mm,
            "reference_vertex_mm",
        )
        beam = _vector3(
            cfg.beam_direction if beam_direction is None else beam_direction,
            "beam_direction",
            unit=True,
        )
        direction = (
            None
            if fitted_direction is None
            else _vector3(fitted_direction, "fitted_direction", unit=True)
        )
        distances = _add_geometry_features(
            features,
            charges,
            positions,
            vertex=vertex,
            beam_direction=beam,
            fitted_direction=direction,
            config=cfg,
        )

    if times is not None:
        _add_time_features(features, charges, times, distances_mm=distances, config=cfg)

    if expected_charges_pe is not None:
        _add_fit_features(
            features,
            charges,
            np.asarray(expected_charges_pe, dtype=np.float64),
            observed_times=times,
            expected_times=(
                None if expected_times_ns is None else np.asarray(expected_times_ns, dtype=np.float64)
            ),
            n_fit_parameters=int(n_fit_parameters),
            fit_nll=fit_nll,
            fit_valid=fit_valid,
            time_resolution_ns=float(time_resolution_ns),
        )

    return features


def features_from_fit_output(
    fit_output: Mapping[str, Any],
    *,
    beam_direction: Sequence[float] = (0.0, 0.0, 1.0),
    reference_vertex_mm: Sequence[float] | None = None,
    config: FeatureConfig | None = None,
) -> dict[str, float]:
    """Extract topology features from ``scripts.fit_single_event`` output."""

    values = dict(fit_output.get("values", {}))
    if reference_vertex_mm is None and all(name in values for name in ("x0", "y0", "z0")):
        reference_vertex_mm = (values["x0"], values["y0"], values["z0"])

    metadata = dict(fit_output.get("metadata", {}))
    free_params = metadata.get("free_params")
    if free_params is None:
        n_fit_parameters = len(values)
    else:
        n_fit_parameters = len(free_params)

    track_direction = fit_output.get("track_direction")
    if track_direction is None and "cx" in values and "cy" in values:
        sign = int(metadata.get("direction_z_sign", 1))
        cz2 = 1.0 - float(values["cx"]) ** 2 - float(values["cy"]) ** 2
        if cz2 >= 0.0:
            track_direction = (
                float(values["cx"]),
                float(values["cy"]),
                sign * np.sqrt(cz2),
            )

    return extract_event_features(
        fit_output["obs_pes"],
        times_ns=fit_output.get("obs_ts"),
        pmt_positions_mm=fit_output.get("p_locations"),
        reference_vertex_mm=reference_vertex_mm,
        beam_direction=beam_direction,
        fitted_direction=track_direction,
        expected_charges_pe=fit_output.get("exp_pes"),
        expected_times_ns=fit_output.get("exp_ts"),
        n_fit_parameters=n_fit_parameters,
        fit_nll=fit_output.get("fval"),
        fit_valid=fit_output.get("valid"),
        config=config,
    )


def _fit_parameter_count(fit: Mapping[str, Any]) -> int:
    if "n_parameters" in fit:
        return int(fit["n_parameters"])
    metadata = dict(fit.get("metadata", {}))
    if metadata.get("free_params") is not None:
        return len(metadata["free_params"])
    if fit.get("values") is not None:
        return len(fit["values"])
    return 0


def compare_fit_hypotheses(
    track_fit: Mapping[str, Any],
    shower_fit: Mapping[str, Any],
) -> dict[str, float]:
    """Return track-vs-shower likelihood, AIC, and BIC comparison features.

    Positive ``track_over_shower_*`` values favour the track hypothesis.  The
    comparison is valid only when both fits use the same PMT mask, charge/time
    definitions, and likelihood normalization.
    """

    track_obs = np.asarray(track_fit.get("obs_pes"), dtype=np.float64)
    shower_obs = np.asarray(shower_fit.get("obs_pes"), dtype=np.float64)
    if track_obs.shape != shower_obs.shape or not np.allclose(
        track_obs, shower_obs, equal_nan=True, rtol=1.0e-10, atol=1.0e-12
    ):
        raise ValueError("Track and shower fits do not contain the same observed PMT charges.")

    track_times = track_fit.get("obs_ts")
    shower_times = shower_fit.get("obs_ts")
    if track_times is not None and shower_times is not None:
        track_times = np.asarray(track_times, dtype=np.float64)
        shower_times = np.asarray(shower_times, dtype=np.float64)
        if track_times.shape != shower_times.shape or not np.allclose(
            track_times, shower_times, equal_nan=True, rtol=1.0e-10, atol=1.0e-12
        ):
            raise ValueError("Track and shower fits do not contain the same observed PMT times.")

    track_mode = dict(track_fit.get("metadata", {})).get("likelihood_mode")
    shower_mode = dict(shower_fit.get("metadata", {})).get("likelihood_mode")
    if track_mode is not None and shower_mode is not None and track_mode != shower_mode:
        raise ValueError(
            f"Likelihood modes differ: track={track_mode!r}, shower={shower_mode!r}."
        )

    track_nll = float(track_fit.get("fval", track_fit.get("nll", np.nan)))
    shower_nll = float(shower_fit.get("fval", shower_fit.get("nll", np.nan)))
    k_track = _fit_parameter_count(track_fit)
    k_shower = _fit_parameter_count(shower_fit)
    n = max(1, track_obs.size)

    track_aic = 2.0 * k_track + 2.0 * track_nll
    shower_aic = 2.0 * k_shower + 2.0 * shower_nll
    track_bic = np.log(n) * k_track + 2.0 * track_nll
    shower_bic = np.log(n) * k_shower + 2.0 * shower_nll

    return {
        "track_over_shower_2delta_log_likelihood": 2.0 * (shower_nll - track_nll),
        "track_over_shower_delta_aic": shower_aic - track_aic,
        "track_over_shower_delta_bic": shower_bic - track_bic,
        "track_nll": track_nll,
        "shower_nll": shower_nll,
        "track_n_parameters": float(k_track),
        "shower_n_parameters": float(k_shower),
    }
