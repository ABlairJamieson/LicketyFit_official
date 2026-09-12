import importlib.util
import unittest
from pathlib import Path

import numpy as np


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "run_pion_shower_smoke_test.py"
SPEC = importlib.util.spec_from_file_location("run_pion_shower_smoke_test", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class DirectionSeedTests(unittest.TestCase):
    def test_equal_solid_angle_hemisphere_seeds(self):
        seeds = np.asarray(MODULE.isotropic_direction_seeds(49))
        self.assertEqual(seeds.shape, (49, 2))
        radius2 = np.sum(seeds * seeds, axis=1)
        self.assertTrue(np.all(radius2 < 1.0))

        cz = np.sqrt(1.0 - radius2[1:])
        expected = (np.arange(48, dtype=float) + 0.5) / 48.0
        self.assertTrue(np.allclose(np.sort(cz), expected))

    def test_seed_count_must_allow_pole_and_surface_sample(self):
        with self.assertRaises(ValueError):
            MODULE.isotropic_direction_seeds(1)

    def test_legacy_ring_seeds_remain_available(self):
        seeds = np.asarray(MODULE.legacy_ring_direction_seeds())
        self.assertEqual(seeds.shape, (49, 2))
        radii = np.sqrt(np.sum(seeds * seeds, axis=1))
        self.assertTrue(
            np.allclose(np.unique(np.round(radii, 8)), [0.0, 0.45, 0.75, 0.92])
        )


if __name__ == "__main__":
    unittest.main()
