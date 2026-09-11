"""A lightweight empirical electromagnetic-shower cone hypothesis.

This is a baseline comparison model, not a complete electromagnetic cascade
simulation.  It represents a shower as a fuzzy Cherenkov ring emitted from one
effective point.  The fitted point and width are therefore *effective*
reconstruction quantities.  The detected-PE normalization becomes an energy
estimate only after calibration against detector simulation or data control
samples.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np


_EPS = 1.0e-12


def _unit(vector: Sequence[float], name: str) -> np.ndarray:
    value = np.asarray(vector, dtype=np.float64)
    if value.shape != (3,) or np.any(~np.isfinite(value)):
        raise ValueError(f"{name} must contain three finite values.")
    norm = float(np.linalg.norm(value))
    if norm <= 0.0:
        raise ValueError(f"{name} must be non-zero.")
    return value / norm


def _direction_from_angles(theta: float, phi: float) -> np.ndarray:
    sin_theta = np.sin(theta)
    return np.asarray(
        [sin_theta * np.cos(phi), sin_theta * np.sin(phi), np.cos(theta)],
        dtype=np.float64,
    )


def _angles_from_direction(direction: Sequence[float]) -> tuple[float, float]:
    axis = _unit(direction, "direction")
    return float(np.arccos(np.clip(axis[2], -1.0, 1.0))), float(np.arctan2(axis[1], axis[0]))


@dataclass(frozen=True)
class ShowerFitConfig:
    """Configuration for :class:`ShowerFitter`."""

    refractive_index: float = 1.344
    vacuum_light_speed_mm_per_ns: float = 299.792458
    cherenkov_angle_deg: float = 41.8
    angular_width_bounds_deg: tuple[float, float] = (2.0, 35.0)
    isotropic_fraction: float = 0.01
    distance_power: float = 2.0
    # These defaults match the current LicketyFit PMT timing likelihood so NLL
    # comparisons use the same convention.  Increase/floor them only when both
    # hypotheses are changed consistently.
    time_resolution_ns: float = 1.0
    minimum_time_sigma_ns: float = 0.0
    include_timing: bool = True
    vertex_half_width_mm: tuple[float, float, float] = (500.0, 500.0, 500.0)
    t0_half_width_ns: float = 20.0
    ncall: int = 20000
    strategy: int = 1
    prompt_seed_fraction: float = 0.25
    prompt_seed_min_pmts: int = 12
    prompt_seed_max_pmts: int = 80
    prompt_seed_iterations: int = 20
    prompt_seed_huber_ns: float = 1.5

    @property
    def light_speed_mm_per_ns(self) -> float:
        return self.vacuum_light_speed_mm_per_ns / self.refractive_index


class ShowerFitter:
    """Fit a one-point fuzzy-cone shower hypothesis to PMT charge and time."""

    parameter_names = (
        "x0",
        "y0",
        "z0",
        "theta",
        "phi",
        "width_deg",
        "total_detected_pe",
        "t0",
    )

    def __init__(
        self,
        pmt_positions_mm: Sequence[Sequence[float]],
        *,
        pmt_direction_zs: Sequence[Sequence[float]] | None = None,
        relative_efficiency: Sequence[float] | None = None,
        config: ShowerFitConfig | None = None,
    ):
        self.config = config or ShowerFitConfig()
        self.pmt_positions_mm = np.asarray(pmt_positions_mm, dtype=np.float64)
        if (
            self.pmt_positions_mm.ndim != 2
            or self.pmt_positions_mm.shape[1] != 3
            or np.any(~np.isfinite(self.pmt_positions_mm))
        ):
            raise ValueError("pmt_positions_mm must be a finite (n_pmts, 3) array.")

        self.pmt_direction_zs = None
        if pmt_direction_zs is not None:
            normals = np.asarray(pmt_direction_zs, dtype=np.float64)
            if normals.shape != self.pmt_positions_mm.shape or np.any(~np.isfinite(normals)):
                raise ValueError("pmt_direction_zs must match pmt_positions_mm.")
            normal_norm = np.linalg.norm(normals, axis=1)
            if np.any(normal_norm <= 0.0):
                raise ValueError("Every PMT direction vector must be non-zero.")
            self.pmt_direction_zs = normals / normal_norm[:, None]

        if relative_efficiency is None:
            self.relative_efficiency = np.ones(self.pmt_positions_mm.shape[0], dtype=np.float64)
        else:
            efficiency = np.asarray(relative_efficiency, dtype=np.float64)
            if (
                efficiency.shape != (self.pmt_positions_mm.shape[0],)
                or np.any(~np.isfinite(efficiency))
                or np.any(efficiency < 0.0)
            ):
                raise ValueError("relative_efficiency must be finite, non-negative, and length n_pmts.")
            self.relative_efficiency = efficiency

    def predict(
        self,
        vertex_mm: Sequence[float],
        direction: Sequence[float],
        *,
        width_deg: float,
        total_detected_pe: float,
        t0_ns: float = 0.0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict detected PE and mean hit time at each active PMT."""

        vertex = np.asarray(vertex_mm, dtype=np.float64)
        if vertex.shape != (3,) or np.any(~np.isfinite(vertex)):
            raise ValueError("vertex_mm must contain three finite values.")
        axis = _unit(direction, "direction")
        if not np.isfinite(width_deg) or width_deg <= 0.0:
            raise ValueError("width_deg must be positive.")
        if not np.isfinite(total_detected_pe) or total_detected_pe < 0.0:
            raise ValueError("total_detected_pe must be finite and non-negative.")

        relative = self.pmt_positions_mm - vertex[None, :]
        distance = np.linalg.norm(relative, axis=1)
        if np.any(distance <= 0.0):
            raise ValueError("The shower vertex cannot coincide with a PMT position.")
        ray = relative / distance[:, None]

        viewing_angle = np.arccos(np.clip(ray @ axis, -1.0, 1.0))
        ring_angle = np.radians(self.config.cherenkov_angle_deg)
        width = np.radians(float(width_deg))
        angular = np.exp(-0.5 * ((viewing_angle - ring_angle) / width) ** 2)
        angular += float(self.config.isotropic_fraction)

        distance_scale = np.median(distance)
        geometry = (distance_scale / distance) ** float(self.config.distance_power)
        if self.pmt_direction_zs is not None:
            incidence = np.maximum(
                -np.sum(ray * self.pmt_direction_zs, axis=1),
                0.0,
            )
            geometry *= incidence

        shape = angular * geometry * self.relative_efficiency
        shape_sum = float(np.sum(shape))
        if not np.isfinite(shape_sum) or shape_sum <= 0.0:
            raise ValueError("The shower model has zero acceptance for this parameter point.")

        expected_pe = float(total_detected_pe) * shape / shape_sum
        expected_time = float(t0_ns) + distance / self.config.light_speed_mm_per_ns
        return expected_pe, expected_time

    def prompt_multilateration_seed(
        self,
        observed_pe: Sequence[float],
        observed_time_ns: Sequence[float],
        *,
        initial_vertex_mm: Sequence[float] = (0.0, 0.0, 0.0),
        early_fraction: float | None = None,
    ) -> dict:
        """Estimate an effective prompt-light vertex and time.

        PMTs are ranked by an approximate emission time
        ``t_hit - distance(initial_vertex, PMT)/c_water``.  A robust
        multilateration fit then solves

        ``t_hit = t0 + distance(vertex, PMT)/c_water``

        on the prompt subset.  This is intended only as a shower-fit seed.  In
        an extended cascade the result is an effective prompt-light origin,
        not necessarily the photon conversion point.
        """

        observed = np.asarray(observed_pe, dtype=np.float64)
        times = np.asarray(observed_time_ns, dtype=np.float64)
        if observed.shape != (self.pmt_positions_mm.shape[0],):
            raise ValueError("observed_pe must have length n_pmts.")
        if times.shape != observed.shape:
            raise ValueError("observed_time_ns must have the same length as observed_pe.")

        initial = np.asarray(initial_vertex_mm, dtype=np.float64)
        if initial.shape != (3,) or np.any(~np.isfinite(initial)):
            raise ValueError("initial_vertex_mm must contain three finite values.")

        valid = (observed > 0.0) & np.isfinite(times)
        valid_indices = np.flatnonzero(valid)
        minimum = max(4, int(self.config.prompt_seed_min_pmts))
        if valid_indices.size < minimum:
            raise ValueError(
                f"Prompt multilateration needs at least {minimum} timed hit PMTs; "
                f"found {valid_indices.size}."
            )

        fraction = (
            float(self.config.prompt_seed_fraction)
            if early_fraction is None
            else float(early_fraction)
        )
        if not 0.0 < fraction <= 1.0:
            raise ValueError("early_fraction must lie in (0, 1].")

        initial_distance = np.linalg.norm(
            self.pmt_positions_mm[valid_indices] - initial[None, :], axis=1
        )
        approximate_emission_time = (
            times[valid_indices] - initial_distance / self.config.light_speed_mm_per_ns
        )
        order = np.argsort(approximate_emission_time)
        n_use = max(minimum, int(np.ceil(fraction * valid_indices.size)))
        n_use = min(n_use, int(self.config.prompt_seed_max_pmts), valid_indices.size)
        selected = valid_indices[order[:n_use]]

        positions = self.pmt_positions_mm[selected]
        selected_times = times[selected]
        selected_charge = observed[selected]
        vertex = initial.copy()
        distance = np.linalg.norm(positions - vertex[None, :], axis=1)
        t0 = float(np.median(selected_times - distance / self.config.light_speed_mm_per_ns))

        converged = False
        rank = 0
        for iteration in range(int(self.config.prompt_seed_iterations)):
            delta_position = positions - vertex[None, :]
            distance = np.linalg.norm(delta_position, axis=1)
            usable = distance > 1.0e-6
            if np.count_nonzero(usable) < 4:
                break
            rays = delta_position[usable] / distance[usable, None]
            residual = (
                selected_times[usable]
                - t0
                - distance[usable] / self.config.light_speed_mm_per_ns
            )
            jacobian = np.column_stack(
                [rays / self.config.light_speed_mm_per_ns, -np.ones(np.count_nonzero(usable))]
            )

            huber = max(float(self.config.prompt_seed_huber_ns), 1.0e-6)
            robust_weight = np.minimum(1.0, huber / np.maximum(np.abs(residual), 1.0e-12))
            charge_weight = np.sqrt(np.clip(selected_charge[usable], 0.0, None))
            charge_weight /= max(float(np.median(charge_weight)), 1.0e-12)
            weight = np.sqrt(robust_weight * np.clip(charge_weight, 0.25, 4.0))

            weighted_jacobian = jacobian * weight[:, None]
            weighted_residual = residual * weight
            step, _, rank, _ = np.linalg.lstsq(
                weighted_jacobian, -weighted_residual, rcond=None
            )
            # Avoid an unstable first iteration jumping across the detector.
            spatial_norm = float(np.linalg.norm(step[:3]))
            if spatial_norm > 300.0:
                step[:3] *= 300.0 / spatial_norm
            vertex += step[:3]
            t0 += float(step[3])
            if np.linalg.norm(step[:3]) < 0.1 and abs(step[3]) < 1.0e-3:
                converged = True
                break

        final_distance = np.linalg.norm(positions - vertex[None, :], axis=1)
        final_residual = (
            selected_times - t0 - final_distance / self.config.light_speed_mm_per_ns
        )
        return {
            "vertex_mm": vertex,
            "t0_ns": float(t0),
            "selected_pmt_indices": selected,
            "n_selected_pmts": int(selected.size),
            "residual_rms_ns": float(np.sqrt(np.mean(final_residual**2))),
            "residual_median_ns": float(np.median(final_residual)),
            "converged": bool(converged),
            "linear_rank": int(rank),
            "iterations": int(iteration + 1),
        }

    def objective(
        self,
        observed_pe: Sequence[float],
        observed_time_ns: Sequence[float] | None,
        *,
        vertex_mm: Sequence[float],
        direction: Sequence[float],
        width_deg: float,
        total_detected_pe: float,
        t0_ns: float,
    ) -> tuple[float, float, float]:
        """Return total, charge, and timing negative log likelihood."""

        observed = np.asarray(observed_pe, dtype=np.float64)
        if (
            observed.shape != (self.pmt_positions_mm.shape[0],)
            or np.any(~np.isfinite(observed))
            or np.any(observed < 0.0)
        ):
            raise ValueError("observed_pe must be finite, non-negative, and length n_pmts.")
        expected, expected_time = self.predict(
            vertex_mm,
            direction,
            width_deg=width_deg,
            total_detected_pe=total_detected_pe,
            t0_ns=t0_ns,
        )

        charge_nll = float(np.sum(expected - observed * np.log(np.clip(expected, _EPS, None))))
        time_nll = 0.0
        if self.config.include_timing and observed_time_ns is not None:
            observed_time = np.asarray(observed_time_ns, dtype=np.float64)
            if observed_time.shape != observed.shape:
                raise ValueError("observed_time_ns must have the same length as observed_pe.")
            mask = (observed > 0.0) & (expected > 0.0) & np.isfinite(observed_time)
            if np.any(mask):
                sigma = np.maximum(
                    self.config.time_resolution_ns / np.sqrt(np.clip(observed[mask], _EPS, None)),
                    self.config.minimum_time_sigma_ns,
                )
                residual = (observed_time[mask] - expected_time[mask]) / sigma
                time_nll = float(0.5 * np.sum(residual * residual))
        return charge_nll + time_nll, charge_nll, time_nll

    def fit(
        self,
        observed_pe: Sequence[float],
        observed_time_ns: Sequence[float] | None = None,
        *,
        vertex_seed_mm: Sequence[float],
        direction_seed: Sequence[float] = (0.0, 0.0, 1.0),
        width_seed_deg: float = 10.0,
        pe_per_mev: float | None = None,
    ) -> dict:
        """Fit the empirical shower model with iminuit.

        ``pe_per_mev`` is optional and must come from an external calibration.
        When supplied, ``energy_calibrated_mev`` is reported as
        ``total_detected_pe / pe_per_mev``.
        """

        try:
            from iminuit import Minuit
        except ImportError as exc:
            raise ImportError(
                "ShowerFitter.fit requires iminuit, which is already a LicketyFit "
                "runtime dependency. Install it with `python -m pip install iminuit`."
            ) from exc

        observed = np.asarray(observed_pe, dtype=np.float64)
        if (
            observed.shape != (self.pmt_positions_mm.shape[0],)
            or np.any(~np.isfinite(observed))
            or np.any(observed < 0.0)
        ):
            raise ValueError("observed_pe must be finite, non-negative, and length n_pmts.")
        observed_time = (
            None if observed_time_ns is None else np.asarray(observed_time_ns, dtype=np.float64)
        )
        if observed_time is not None and observed_time.shape != observed.shape:
            raise ValueError("observed_time_ns must have the same length as observed_pe.")

        vertex_seed = np.asarray(vertex_seed_mm, dtype=np.float64)
        if vertex_seed.shape != (3,) or np.any(~np.isfinite(vertex_seed)):
            raise ValueError("vertex_seed_mm must contain three finite values.")
        theta_seed, phi_seed = _angles_from_direction(direction_seed)
        total_seed = max(float(np.sum(observed)), 1.0)

        if observed_time is not None:
            relative = self.pmt_positions_mm - vertex_seed[None, :]
            distance = np.linalg.norm(relative, axis=1)
            mask = (observed > 0.0) & np.isfinite(observed_time)
            if np.any(mask):
                t0_values = observed_time[mask] - distance[mask] / self.config.light_speed_mm_per_ns
                t0_seed = float(np.median(t0_values))
            else:
                t0_seed = 0.0
        else:
            t0_seed = 0.0

        def nll(x0, y0, z0, theta, phi, width_deg, total_detected_pe, t0):
            direction = _direction_from_angles(theta, phi)
            try:
                value, _, _ = self.objective(
                    observed,
                    observed_time,
                    vertex_mm=(x0, y0, z0),
                    direction=direction,
                    width_deg=width_deg,
                    total_detected_pe=total_detected_pe,
                    t0_ns=t0,
                )
            except ValueError:
                return 1.0e30
            return value if np.isfinite(value) else 1.0e30

        minimizer = Minuit(
            nll,
            x0=float(vertex_seed[0]),
            y0=float(vertex_seed[1]),
            z0=float(vertex_seed[2]),
            theta=theta_seed,
            phi=phi_seed,
            width_deg=float(width_seed_deg),
            total_detected_pe=total_seed,
            t0=t0_seed,
        )
        minimizer.errordef = Minuit.LIKELIHOOD
        minimizer.strategy = int(self.config.strategy)
        half_width = np.asarray(self.config.vertex_half_width_mm, dtype=np.float64)
        for index, name in enumerate(("x0", "y0", "z0")):
            minimizer.limits[name] = (
                float(vertex_seed[index] - half_width[index]),
                float(vertex_seed[index] + half_width[index]),
            )
            minimizer.errors[name] = max(5.0, 0.05 * half_width[index])
        minimizer.limits["theta"] = (0.0, np.pi)
        minimizer.limits["phi"] = (-np.pi, np.pi)
        minimizer.limits["width_deg"] = tuple(self.config.angular_width_bounds_deg)
        minimizer.limits["total_detected_pe"] = (max(1.0e-6, 0.05 * total_seed), 20.0 * total_seed)
        minimizer.limits["t0"] = (
            t0_seed - self.config.t0_half_width_ns,
            t0_seed + self.config.t0_half_width_ns,
        )
        minimizer.errors["theta"] = 0.05
        minimizer.errors["phi"] = 0.05
        minimizer.errors["width_deg"] = 1.0
        minimizer.errors["total_detected_pe"] = max(1.0, np.sqrt(total_seed))
        minimizer.errors["t0"] = 0.5
        if observed_time is None or not self.config.include_timing:
            minimizer.fixed["t0"] = True

        minimizer.migrad(ncall=int(self.config.ncall))
        values = minimizer.values.to_dict()
        direction = _direction_from_angles(values["theta"], values["phi"])
        expected_pe, expected_time = self.predict(
            (values["x0"], values["y0"], values["z0"]),
            direction,
            width_deg=values["width_deg"],
            total_detected_pe=values["total_detected_pe"],
            t0_ns=values["t0"],
        )
        total_nll, charge_nll, timing_nll = self.objective(
            observed,
            observed_time,
            vertex_mm=(values["x0"], values["y0"], values["z0"]),
            direction=direction,
            width_deg=values["width_deg"],
            total_detected_pe=values["total_detected_pe"],
            t0_ns=values["t0"],
        )

        n_parameters = sum(not bool(minimizer.fixed[name]) for name in self.parameter_names)
        output = {
            "values": values,
            "errors": minimizer.errors.to_dict(),
            "valid": bool(minimizer.valid),
            "fval": float(total_nll),
            "nll": float(total_nll),
            "charge_nll": float(charge_nll),
            "timing_nll": float(timing_nll),
            "n_parameters": int(n_parameters),
            "obs_pes": observed,
            "obs_ts": observed_time,
            "exp_pes": expected_pe,
            "exp_ts": expected_time,
            "p_locations": self.pmt_positions_mm,
            "effective_vertex_mm": np.asarray(
                [values["x0"], values["y0"], values["z0"]], dtype=np.float64
            ),
            "shower_direction": direction,
            "angular_width_deg": float(values["width_deg"]),
            "total_detected_pe": float(values["total_detected_pe"]),
            "metadata": {
                "hypothesis": "empirical_one_point_shower_cone",
                "likelihood_mode": (
                    "charge_time"
                    if self.config.include_timing and observed_time is not None
                    else "charge_only"
                ),
                "free_params": [
                    name for name in self.parameter_names if not bool(minimizer.fixed[name])
                ],
                "cherenkov_angle_deg": float(self.config.cherenkov_angle_deg),
                "warning": (
                    "Effective one-point shower model; vertex and width are empirical. "
                    "Energy requires external PE/MeV calibration."
                ),
            },
        }
        if pe_per_mev is not None:
            if not np.isfinite(pe_per_mev) or pe_per_mev <= 0.0:
                raise ValueError("pe_per_mev must be a positive finite calibration.")
            output["energy_calibrated_mev"] = float(values["total_detected_pe"] / pe_per_mev)
            output["pe_per_mev_calibration"] = float(pe_per_mev)
        return output
