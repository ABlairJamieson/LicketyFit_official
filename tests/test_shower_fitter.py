import unittest

import numpy as np

from LicketyFit.ShowerFitter import ShowerFitConfig, ShowerFitter


def fibonacci_sphere(n_points=300, radius_mm=1500.0):
    index = np.arange(n_points, dtype=np.float64)
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    z = 1.0 - 2.0 * (index + 0.5) / n_points
    radial = np.sqrt(np.maximum(0.0, 1.0 - z * z))
    phi = golden_angle * index
    return radius_mm * np.column_stack([radial * np.cos(phi), radial * np.sin(phi), z])


class ShowerModelTests(unittest.TestCase):
    def setUp(self):
        self.positions = fibonacci_sphere()
        self.model = ShowerFitter(
            self.positions,
            config=ShowerFitConfig(include_timing=True),
        )

    def test_prediction_normalizes_to_detected_pe(self):
        expected, times = self.model.predict(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            width_deg=8.0,
            total_detected_pe=250.0,
            t0_ns=3.0,
        )
        self.assertAlmostEqual(float(np.sum(expected)), 250.0, places=10)
        expected_time = 3.0 + 1500.0 / (299.792458 / 1.344)
        self.assertTrue(np.allclose(times, expected_time))

    def test_default_width_bound_is_not_the_cherenkov_angle(self):
        self.assertAlmostEqual(self.model.config.cherenkov_angle_deg, 41.8)
        self.assertGreater(self.model.config.angular_width_bounds_deg[1], 41.8)

    def test_true_direction_beats_opposite_direction(self):
        expected, times = self.model.predict(
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 1.0),
            width_deg=7.0,
            total_detected_pe=400.0,
            t0_ns=0.0,
        )
        correct, _, _ = self.model.objective(
            expected,
            times,
            vertex_mm=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, 1.0),
            width_deg=7.0,
            total_detected_pe=400.0,
            t0_ns=0.0,
        )
        wrong, _, _ = self.model.objective(
            expected,
            times,
            vertex_mm=(0.0, 0.0, 0.0),
            direction=(0.0, 0.0, -1.0),
            width_deg=7.0,
            total_detected_pe=400.0,
            t0_ns=0.0,
        )
        self.assertLess(correct, wrong)

    def test_pmt_incidence_is_supported(self):
        normals = -self.positions / np.linalg.norm(self.positions, axis=1)[:, None]
        model = ShowerFitter(self.positions, pmt_direction_zs=normals)
        expected, _ = model.predict(
            (0.0, 0.0, 0.0),
            (1.0, 0.0, 0.0),
            width_deg=10.0,
            total_detected_pe=100.0,
        )
        self.assertAlmostEqual(float(np.sum(expected)), 100.0, places=10)

    def test_prompt_multilateration_recovers_point_source(self):
        truth = np.asarray([85.0, -60.0, 120.0])
        t0 = 4.25
        distance = np.linalg.norm(self.positions - truth[None, :], axis=1)
        times = t0 + distance / (299.792458 / 1.344)
        charges = np.ones(self.positions.shape[0])
        seed = self.model.prompt_multilateration_seed(
            charges,
            times,
            initial_vertex_mm=(0.0, 0.0, 0.0),
            early_fraction=1.0,
        )
        self.assertTrue(np.allclose(seed["vertex_mm"], truth, atol=0.1))
        self.assertAlmostEqual(seed["t0_ns"], t0, places=3)
        self.assertLess(seed["residual_rms_ns"], 1.0e-5)


if __name__ == "__main__":
    unittest.main()
