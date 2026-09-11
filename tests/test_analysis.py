import unittest

import numpy as np

from LicketyFit.Analysis import compare_fit_hypotheses, extract_event_features
from LicketyFit.Classification import (
    classification_metrics_at_threshold,
    threshold_for_target_pion_efficiency,
)


class EventFeatureTests(unittest.TestCase):
    def test_basic_charge_features(self):
        features = extract_event_features([0.0, 1.0, 2.0, 3.0])
        self.assertEqual(features["n_active_pmts"], 4.0)
        self.assertEqual(features["n_hit_pmts"], 3.0)
        self.assertAlmostEqual(features["total_charge_pe"], 6.0)
        self.assertAlmostEqual(features["mean_charge_per_hit_pe"], 2.0)
        self.assertAlmostEqual(features["effective_n_hit_pmts"], 36.0 / 14.0)

    def test_geometry_tof_and_ring_features(self):
        theta = np.radians(41.8)
        phi = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
        rays = np.column_stack(
            [
                np.sin(theta) * np.cos(phi),
                np.sin(theta) * np.sin(phi),
                np.full(phi.shape, np.cos(theta)),
            ]
        )
        positions = 1000.0 * rays
        charges = np.ones(phi.size)
        times = np.full(phi.size, 7.5 + 1000.0 / (299.792458 / 1.344))
        features = extract_event_features(
            charges,
            times_ns=times,
            pmt_positions_mm=positions,
            reference_vertex_mm=(0.0, 0.0, 0.0),
            beam_direction=(0.0, 0.0, 1.0),
            fitted_direction=(0.0, 0.0, 1.0),
        )
        self.assertAlmostEqual(features["beam_angle_mean_deg"], 41.8, places=10)
        self.assertAlmostEqual(features["track_ring_residual_rms_deg"], 0.0, places=10)
        self.assertAlmostEqual(features["track_ring_charge_fraction"], 1.0)
        self.assertAlmostEqual(features["tof_residual_rms_ns"], 0.0, places=10)
        self.assertAlmostEqual(features["prompt_charge_fraction"], 1.0)

    def test_perfect_fit_has_zero_poisson_deviance(self):
        observed = np.asarray([0.0, 1.0, 4.0, 0.0])
        features = extract_event_features(
            observed,
            expected_charges_pe=observed,
            n_fit_parameters=1,
            fit_nll=3.0,
            fit_valid=True,
        )
        self.assertAlmostEqual(features["fit_charge_poisson_deviance"], 0.0)
        self.assertAlmostEqual(features["fit_charge_shape_l1_distance"], 0.0)
        self.assertAlmostEqual(features["fit_charge_cosine_similarity"], 1.0)
        self.assertEqual(features["fit_valid"], 1.0)

    def test_fit_comparison_sign_convention(self):
        observed = np.asarray([0.0, 1.0, 2.0])
        track = {
            "obs_pes": observed,
            "fval": 10.0,
            "n_parameters": 7,
            "valid": True,
        }
        shower = {
            "obs_pes": observed.copy(),
            "fval": 15.0,
            "n_parameters": 8,
            "valid": True,
        }
        comparison = compare_fit_hypotheses(track, shower)
        self.assertEqual(comparison["track_over_shower_2delta_log_likelihood"], 10.0)
        self.assertGreater(comparison["track_over_shower_delta_aic"], 0.0)

    def test_invalid_fit_suppresses_comparison(self):
        observed = np.asarray([0.0, 1.0, 2.0])
        comparison = compare_fit_hypotheses(
            {"obs_pes": observed, "fval": 10.0, "valid": False},
            {"obs_pes": observed.copy(), "fval": 15.0, "valid": True},
        )
        self.assertFalse(comparison["comparison_valid"])
        self.assertTrue(np.isnan(comparison["track_over_shower_2delta_log_likelihood"]))
        self.assertIn("track", comparison["comparison_reason"])

    def test_negative_charge_rejected(self):
        with self.assertRaises(ValueError):
            extract_event_features([1.0, -0.1])


class ClassificationUtilityTests(unittest.TestCase):
    def test_threshold_and_metrics(self):
        labels = np.asarray([1, 1, 1, 1, 0, 0])
        scores = np.asarray([0.9, 0.8, 0.7, 0.6, 0.5, 0.1])
        threshold = threshold_for_target_pion_efficiency(labels, scores, 0.75)
        self.assertEqual(threshold, 0.7)
        metrics = classification_metrics_at_threshold(labels, scores, threshold)
        self.assertAlmostEqual(metrics["pion_efficiency"], 0.75)
        self.assertAlmostEqual(metrics["shower_rejection"], 1.0)


if __name__ == "__main__":
    unittest.main()
