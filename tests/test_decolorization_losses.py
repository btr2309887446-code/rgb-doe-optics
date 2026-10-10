import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from losses import LossSchedule, LossWeights, joint_loss, rgb_consistency_losses
from rgb_decolorization_experiment import ARCHITECTURES, VARIANTS
from targets import soft_circle_two_spots


class DecolorizationLossTests(unittest.TestCase):
    def setUp(self):
        coordinate = torch.linspace(-4e-4, 4e-4, 32)
        self.x, self.y = torch.meshgrid(coordinate, coordinate, indexing="xy")
        self.target = soft_circle_two_spots(self.x, self.y)
        base = torch.exp(-((self.x / 1.8e-4) ** 2 + (self.y / 1.4e-4) ** 2))
        self.field = (0.4 + base).to(torch.complex64)

    def test_identical_fields_have_zero_rgb_consistency(self):
        losses = rgb_consistency_losses([self.field, self.field, self.field], self.target)
        for value in losses.values():
            self.assertLess(float(value), 1e-10)

    def test_spatial_shift_increases_pixel_and_edge_terms(self):
        shifted = torch.roll(self.field, shifts=2, dims=0)
        losses = rgb_consistency_losses([self.field, shifted, self.field], self.target)
        self.assertGreater(float(losses["rgb_pixel"]), 1e-8)
        self.assertGreater(float(losses["rgb_edge"]), 1e-10)

    def test_single_channel_amplitude_change_increases_absolute_energy_terms(self):
        losses = rgb_consistency_losses([self.field, 2.0 * self.field, self.field], self.target)
        self.assertGreater(float(losses["rgb_roi_absolute"]), 1e-6)
        self.assertGreater(float(losses["rgb_energy"]), 1e-6)

    def test_common_amplitude_scale_does_not_create_energy_balance_penalty(self):
        losses = rgb_consistency_losses([2.0 * self.field, 2.0 * self.field, 2.0 * self.field], self.target)
        self.assertLess(float(losses["rgb_energy"]), 1e-10)
        self.assertLess(float(losses["rgb_roi_absolute"]), 1e-10)

    def test_rgb_losses_are_differentiable_and_finite(self):
        height = torch.nn.Parameter(torch.zeros((32, 32)))
        amplitude = 0.5 + 0.2 * height + torch.exp(-((self.x / 1.8e-4) ** 2 + (self.y / 1.4e-4) ** 2))
        field = amplitude.to(torch.complex64) * torch.exp(1j * height.to(torch.complex64))
        losses = rgb_consistency_losses([field, field * (1.0 + 0.01 * height), field], self.target)
        total = sum(losses.values())
        total.backward()
        self.assertIsNotNone(height.grad)
        self.assertTrue(torch.isfinite(total))
        self.assertTrue(torch.isfinite(height.grad).all())

    def test_input_power_length_is_validated(self):
        with self.assertRaises(ValueError):
            rgb_consistency_losses([self.field, self.field, self.field], self.target, input_powers=(1.0, 1.0))

    def test_zero_rgb_weights_keep_joint_loss_finite(self):
        height = torch.zeros((32, 32), requires_grad=True)
        fields = [self.field, self.field, self.field]
        weights = LossWeights(
            rgb_pixel_weight=0.0,
            rgb_edge_weight=0.0,
            rgb_background_weight=0.0,
            rgb_energy_weight=0.0,
            rgb_component_weight=0.0,
        )
        total, components = joint_loss(
            fields,
            self.target,
            height,
            1e-6,
            weights,
            LossSchedule(0, 1),
            1,
            wavelength_weights=(1.0, 1.0, 1.0),
            input_powers=(1.0, 1.0, 1.0),
        )
        self.assertTrue(torch.isfinite(total))
        self.assertIn("rgb_pixel", components)

    def test_experiment_excludes_fourier_lens(self):
        self.assertEqual(ARCHITECTURES, ("single_doe", "two_doe_planes"))
        self.assertEqual(len(VARIANTS), 7)


if __name__ == "__main__":
    unittest.main()
