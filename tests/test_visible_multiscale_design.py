import sys
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

project = Path(__file__).resolve().parents[1]
fourier_dir = project / "多波长 Fourier 光学"
sys.path.insert(0, str(fourier_dir))

from fourier_optics_step11_free_screen_optimization import (  # noqa: E402
    Model,
    Problem,
    select_adaptive_wavelengths,
    upper_tail_mean_weights,
    validation_summary,
    verify_math,
)
from make_multiscale_resolution_target import (  # noqa: E402
    DEFAULT_PITCH_M,
    make_multiscale_target,
    make_region_masks,
)
from run_visible_multiscale_design import (  # noqa: E402
    checkpoint_callback,
    completed_checkpoint_candidates,
    training_wavelengths,
)


class VisibleMultiscaleDesignTests(unittest.TestCase):
    def test_multiscale_target_sampling_and_region_geometry(self):
        target, metadata = make_multiscale_target()
        self.assertEqual(target.shape, (256, 256))
        self.assertTrue(np.isfinite(target).all())
        self.assertGreaterEqual(float(target.min()), 0.0)
        self.assertAlmostEqual(float(target.sum()), 1.0, places=12)
        self.assertAlmostEqual(metadata["field_width_mm"], 4.0, places=12)
        self.assertGreater(metadata["finest_bar_width_pixels"], 8.0)
        self.assertGreater(metadata["finest_stripe_period_pixels"], 16.0)
        axis = (np.arange(256)-128)*DEFAULT_PITCH_M
        masks = make_region_masks(metadata, axis)
        self.assertEqual(len(masks), 9)
        values = list(masks.values())
        self.assertTrue(all(mask.shape == target.shape and mask.any() for mask in values))
        for index, first in enumerate(values):
            for second in values[index+1:]:
                self.assertFalse(np.any(first & second))

    def test_visible_training_grid_contains_all_anchors_and_unique_34_points(self):
        waves = training_wavelengths()
        np.testing.assert_array_equal(waves[0:4], [400.0, 410.0, 420.0, 430.0])
        self.assertEqual(waves.size, 34)
        self.assertTrue(all(wavelength in waves for wavelength in (488.0, 532.0, 633.0)))
        np.testing.assert_array_equal(waves[[0, -1]], [400.0, 700.0])

    def test_adaptive_selection_avoids_training_and_near_duplicate_points(self):
        rows = [
            {"wavelength_nm": 400.0, "relative_L2": 0.2, "roi_power": 0.7},
            {"wavelength_nm": 402.0, "relative_L2": 0.5, "roi_power": 0.8},
            {"wavelength_nm": 407.0, "relative_L2": 0.4, "roi_power": 0.8},
            {"wavelength_nm": 430.0, "relative_L2": 0.3, "roi_power": 0.8},
            {"wavelength_nm": 450.0, "relative_L2": 0.05, "roi_power": 0.95},
        ]
        selected = select_adaptive_wavelengths(
            rows, [400.0], [400.0, 700.0], max_additions=2, separation_nm=5.0)
        np.testing.assert_array_equal(selected, [402.0, 407.0])

    def test_two_gate_validation_summary(self):
        rows = [
            {"wavelength_nm": 400.0, "relative_L2": 0.08, "roi_power": 0.82},
            {"wavelength_nm": 401.0, "relative_L2": 0.11, "roi_power": 0.9},
        ]
        summary = validation_summary(rows, [400.0, 700.0])
        self.assertFalse(summary["passed"])
        self.assertEqual(summary["worst_wavelength_nm"], 401.0)
        rows[1]["relative_L2"] = 0.09
        rows[0]["roi_power"] = 0.79
        self.assertFalse(validation_summary(rows, [400.0, 700.0])["passed"])

    def test_worst_ten_percent_uses_fractional_boundary_sample(self):
        weights = upper_tail_mean_weights(np.arange(34.0), 0.10)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=14)
        self.assertEqual(int(np.count_nonzero(weights)), 4)
        ordered = np.sort(weights[weights > 0])[::-1]
        np.testing.assert_allclose(ordered[:3], np.full(3, 1/3.4))
        self.assertAlmostEqual(float(ordered[3]), 0.4/3.4, places=14)

    def test_optimizer_checkpoint_is_atomic_and_reloadable(self):
        pupil = np.zeros((16, 16), dtype=bool)
        pupil[5:11, 5:11] = True
        target = np.zeros((16, 16), dtype=float)
        target[7:9, 7:9] = 0.25
        problem = Problem(pupil, target, [532.0], [400.0, 700.0])
        model = Model(problem, "spectral", 2)
        coeff = np.zeros((3, problem.count), dtype=complex)
        coeff[0] = 1.0
        theta = model.pack(coeff)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            save = checkpoint_callback(root, "degree_2", model,
                                       np.asarray([532.0]), np.asarray([400.0, 700.0]))
            expected_loss = model.value_gradient(theta)[0]
            save(0, 25, theta, expected_loss)
            path = root/"checkpoints"/"degree_2"/"restart_00.npz"
            self.assertTrue(path.is_file())
            with np.load(path, allow_pickle=False) as checkpoint:
                np.testing.assert_array_equal(checkpoint["theta"], theta)
                self.assertAlmostEqual(float(checkpoint["best_loss"]), expected_loss)
            progress = json.loads((path.parent/"progress.json").read_text())
            self.assertEqual(progress["iteration"], 25)
            loaded = completed_checkpoint_candidates(root, "degree_2", model, 1, 25)
            self.assertIsNotNone(loaded)
            np.testing.assert_array_equal(loaded[0], theta)
            self.assertAlmostEqual(loaded[1], expected_loss)

    def test_step11_math_including_robust_loss_gradient(self):
        result = verify_math()
        self.assertEqual(result["robust_objective_gradient"]["objective_reconstruction"], "passed")
        self.assertLess(result["robust_objective_gradient"]["absolute_error"], 2e-7)


if __name__ == "__main__":
    unittest.main()
