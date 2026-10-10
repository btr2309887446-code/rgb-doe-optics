import sys
import unittest
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forward_models import make_grid, validate_sampling
from propagation_convergence_experiment import make_system
from rgb_imageplane_evaluation import db_array, masked_mse, pearson, rgb_array


class EvaluationToolTests(unittest.TestCase):
    def test_convergence_sweep_keeps_physical_field_of_view(self):
        expected_fov = 320e-6
        for n_grid in (16, 24, 32, 48, 64):
            system = make_system(n_grid, expected_fov, 0.01, 120e-6, 2)
            self.assertAlmostEqual(system.n_grid * system.pixel_pitch_m, expected_fov, places=15)
            grid = make_grid(system, torch.device("cpu"))
            report = validate_sampling(system, grid)
            self.assertTrue(report["aperture_fully_inside_base_grid"])
            self.assertGreaterEqual(report["field_to_aperture_diameter_ratio"], 1.2)

    def test_common_scale_similarity_metrics(self):
        first = torch.tensor([[0.0, 1.0], [2.0, 3.0]])
        second = first.clone()
        third = -first
        self.assertAlmostEqual(pearson(first, second), 1.0, places=6)
        self.assertLess(pearson(first, third), 0.0)
        self.assertAlmostEqual(masked_mse(first, second), 0.0, places=12)
        mask = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        self.assertGreater(masked_mse(first, third, mask), 0.0)

    def test_common_scale_and_db_rgb_mapping(self):
        channels = [
            torch.tensor([[1.0, 0.5], [0.0, 0.25]]).numpy(),
            torch.tensor([[0.8, 0.4], [0.0, 0.2]]).numpy(),
            torch.tensor([[0.6, 0.3], [0.0, 0.1]]).numpy(),
        ]
        rgb = rgb_array(channels, 1.0)
        self.assertEqual(rgb.shape, (2, 2, 3))
        self.assertAlmostEqual(float(rgb[0, 0, 2]), 1.0, places=6)
        self.assertAlmostEqual(float(rgb[0, 0, 1]), 0.8, places=6)
        self.assertAlmostEqual(float(rgb[0, 0, 0]), 0.6, places=6)
        db_rgb = db_array(channels)
        self.assertEqual(db_rgb.shape, (2, 2, 3))
        self.assertTrue(np.isfinite(db_rgb).all())
        self.assertGreaterEqual(float(db_rgb.min()), 0.0)
        self.assertLessEqual(float(db_rgb.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
