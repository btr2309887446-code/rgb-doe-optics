import sys
import unittest
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forward_models import (
    OpticalSystem,
    angular_spectrum_transfer_batch,
    energy_relative_error,
    forward_field,
    forward_rgb_fourier_lens,
    forward_rgb_single_plane,
    forward_rgb_two_planes,
    make_grid,
    propagate,
    sampling_report,
    transfer_identity_error,
    validate_sampling,
)
from materials import CauchyMaterial
from losses import LossSchedule, LossWeights, joint_loss
from metrics import evaluate_output
from optimizers import AdamOptions, optimize_adam, optimize_dbs, optimize_mraf
from targets import (
    asymmetric_two_spots,
    cross_two_spots,
    gaussian_two_spots,
    ring_two_spots,
    soft_square_two_spots,
    soft_circle_two_spots,
    solid_triangle,
    target_factories,
    triangle_three_spots,
    supersampled_edge_target,
)


class ModularModelTests(unittest.TestCase):
    def test_cauchy_anchors_and_extrapolation(self):
        material = CauchyMaterial()
        self.assertAlmostEqual(material.index(532e-9), 1.6482, places=6)
        self.assertAlmostEqual(material.index(633e-9), 1.6347, places=6)
        metadata = material.metadata((488e-9, 532e-9, 633e-9, 660e-9))
        self.assertEqual(metadata["extrapolated"], [True, False, False, True])

    def test_centered_propagation_identity_and_energy(self):
        system = OpticalSystem(n_grid=32)
        device = torch.device("cpu")
        self.assertLessEqual(transfer_identity_error(system, device), 1e-7)
        self.assertLessEqual(energy_relative_error(system, device), 1e-6)

    def test_sampling_validation_checks_aperture_coverage(self):
        valid_system = OpticalSystem(n_grid=256, pixel_pitch_m=10e-6)
        valid_grid = make_grid(valid_system, torch.device("cpu"))
        report = validate_sampling(valid_system, valid_grid)
        self.assertTrue(report["aperture_fully_inside_base_grid"])
        self.assertGreater(report["base_aperture_pixel_count"], 0)

        invalid_system = OpticalSystem(n_grid=256, pixel_pitch_m=5e-6)
        invalid_grid = make_grid(invalid_system, torch.device("cpu"))
        with self.assertRaises(ValueError):
            validate_sampling(invalid_system, invalid_grid)
        self.assertFalse(sampling_report(invalid_system, invalid_grid)["aperture_fully_inside_base_grid"])

    def test_soft_circle_regions_partition(self):
        system = OpticalSystem(n_grid=64)
        grid = make_grid(system, torch.device("cpu"))
        target = soft_circle_two_spots(grid.base_x, grid.base_y, radius_m=30e-6, edge_width_m=15e-6)
        combined = target.signal_mask + target.transition_mask + target.background_mask
        self.assertTrue(torch.allclose(combined, torch.ones_like(combined)))
        self.assertGreater(float(target.intensity.max()), 0.0)

    def test_thin_lens_equation(self):
        system = OpticalSystem(
            n_grid=32,
            model="thin_lens",
            doe_to_lens_distance_m=0.1,
            lens_to_image_distance_m=0.1,
            lens_focal_length_m=0.05,
        )
        self.assertAlmostEqual(
            1.0 / system.doe_to_lens_distance_m + 1.0 / system.lens_to_image_distance_m,
            1.0 / system.lens_focal_length_m,
        )

    def test_generalization_targets_have_partitioned_components(self):
        system = OpticalSystem(n_grid=128)
        grid = make_grid(system, torch.device("cpu"))
        targets = [
            factory(grid.base_x, grid.base_y)
            for factory in target_factories().values()
        ]
        for target in targets:
            self.assertAlmostEqual(float(target.intensity.sum()), 1.0, places=5)
            self.assertEqual(len(target.component_masks), target.metadata["component_count"])
            partition = target.signal_mask + target.transition_mask + target.background_mask
            self.assertTrue(torch.allclose(partition, torch.ones_like(partition)))
            for index, first in enumerate(target.component_masks):
                self.assertGreater(float(first.sum()), 0.0)
                for second in target.component_masks[index + 1:]:
                    self.assertEqual(float((first * second).sum()), 0.0)

    def test_multi_component_metrics_and_optimizers(self):
        system = OpticalSystem(n_grid=32)
        grid = make_grid(system, torch.device("cpu"))
        target = triangle_three_spots(grid.base_x, grid.base_y)
        field = torch.ones((32, 32), dtype=torch.complex64)
        metrics = evaluate_output(field, target, grid.base_x, grid.base_y, grid.pixel_pitch_m)
        self.assertEqual(len(metrics["component_energies"]), 3)
        self.assertEqual(len(metrics["component_centroids_m"]), 3)

        aperture = grid.doe_aperture[grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end]
        def forward(height):
            return [height.to(torch.complex64)]

        weights = LossWeights()
        schedule = LossSchedule(0, 1)
        adam = optimize_adam(
            forward, target, 1e-6, 32, aperture,
            AdamOptions(iterations=1, learning_rate_um=0.01),
            weights, schedule,
        )
        mraf = optimize_mraf(
            forward, target, 1e-6, 32, aperture,
            AdamOptions(iterations=1, learning_rate_um=0.01),
            weights, schedule, background_bias=0.05,
        )
        dbs = optimize_dbs(
            forward, target, adam.height_m, 1e-6, aperture, weights,
            levels=4, sweeps=1, max_pixels_per_sweep=2,
        )
        self.assertEqual(adam.height_m.shape, (32, 32))
        self.assertEqual(mraf.height_m.shape, (32, 32))
        self.assertEqual(dbs.height_m.shape, (32, 32))

    def test_ring_masks_metrics_and_hole_loss(self):
        system = OpticalSystem(n_grid=128)
        grid = make_grid(system, torch.device("cpu"))
        target = ring_two_spots(
            grid.base_x,
            grid.base_y,
            radius_m=50e-6,
            edge_width_m=8e-6,
        )
        self.assertEqual(target.target_type, "ring")
        self.assertIsNotNone(target.hole_mask)
        self.assertIsNotNone(target.ring_mask)
        self.assertGreater(float(target.hole_mask.sum()), 0.0)
        self.assertGreater(float(target.ring_mask.sum()), 0.0)
        metrics = evaluate_output(
            torch.sqrt(target.intensity).to(torch.complex64),
            target,
            grid.base_x,
            grid.base_y,
            grid.pixel_pitch_m,
        )
        self.assertIn("mean_center_to_ring_ratio", metrics)
        self.assertIn("mean_ring_radius_error_m", metrics)
        field = torch.ones((128, 128), dtype=torch.complex64)
        height = torch.zeros((128, 128))
        loss, components = joint_loss(
            [field],
            target,
            height,
            1e-6,
            LossWeights(hole_weight=0.5),
            LossSchedule(0, 1, 0, 1),
            1,
        )
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("hole_0", components)

    def test_non_circular_targets_and_rgb_consistency(self):
        system = OpticalSystem(n_grid=128)
        grid = make_grid(system, torch.device("cpu"))
        targets = [
            soft_square_two_spots(grid.base_x, grid.base_y),
            cross_two_spots(grid.base_x, grid.base_y),
            triangle_three_spots(grid.base_x, grid.base_y),
            solid_triangle(grid.base_x, grid.base_y),
            asymmetric_two_spots(grid.base_x, grid.base_y),
        ]
        for target in targets:
            self.assertAlmostEqual(float(target.intensity.sum()), 1.0, places=5)
            partition = target.signal_mask + target.transition_mask + target.background_mask
            self.assertTrue(torch.allclose(partition, torch.ones_like(partition)))
            self.assertGreaterEqual(len(target.component_masks), 1)
        cross = targets[1]
        left_column = int(torch.argmin(torch.abs(grid.base_x[0] + 300e-6)))
        center_row = int(torch.argmin(torch.abs(grid.base_y[:, 0])))
        background_column = int(torch.argmin(torch.abs(grid.base_x[0])))
        background_row = int(torch.argmin(torch.abs(grid.base_y[:, 0] - 500e-6)))
        self.assertGreater(
            float(cross.intensity[center_row, left_column]),
            50.0 * float(cross.intensity[background_row, background_column]),
        )
        asymmetric = targets[-1]
        expected = torch.stack([
            torch.sum(asymmetric.intensity * mask)
            for mask in asymmetric.component_masks
        ])
        self.assertLess(float(expected[1] / expected[0]), 0.8)

        target = soft_square_two_spots(grid.base_x, grid.base_y)
        field = torch.ones((128, 128), dtype=torch.complex64, requires_grad=False)
        height = torch.zeros((128, 128), requires_grad=True)
        loss, components = joint_loss(
            [field, field, field],
            target,
            height,
            1e-6,
            LossWeights(position_weight=0.1),
            LossSchedule(0, 1),
            1,
            wavelength_weights=(0.5, 1.0, 1.0),
            pixel_pitch_m=grid.pixel_pitch_m,
        )
        loss.backward()
        self.assertTrue(torch.isfinite(loss))
        self.assertIn("centroid_consistency", components)

    def test_centroid_consistency_compares_wavelengths_not_components(self):
        system = OpticalSystem(n_grid=128, pixel_pitch_m=10e-6)
        grid = make_grid(system, torch.device("cpu"))
        target = soft_circle_two_spots(grid.base_x, grid.base_y)
        height = torch.nn.Parameter(torch.zeros((128, 128)))

        def two_spot_field(shift_m):
            first = torch.exp(
                -((grid.base_x + 300e-6 - shift_m) ** 2 + grid.base_y**2)
                / (2.0 * (20e-6) ** 2)
            )
            second = torch.exp(
                -((grid.base_x - 300e-6 - shift_m) ** 2 + grid.base_y**2)
                / (2.0 * (20e-6) ** 2)
            )
            amplitude = torch.sqrt(first + second)
            return (amplitude * (1.0 + height * 1e5)).to(torch.complex64)

        aligned_fields = [two_spot_field(0.0), two_spot_field(0.0)]
        _, aligned_components = joint_loss(
            aligned_fields,
            target,
            height,
            1e-6,
            LossWeights(position_weight=1.0),
            LossSchedule(0, 1),
            1,
            pixel_pitch_m=grid.pixel_pitch_m,
        )
        self.assertAlmostEqual(aligned_components["centroid_consistency"], 0.0, places=6)

        shifted_fields = [two_spot_field(0.0), two_spot_field(30e-6)]
        shifted_loss, shifted_components = joint_loss(
            shifted_fields,
            target,
            height,
            1e-6,
            LossWeights(position_weight=1.0),
            LossSchedule(0, 1),
            1,
            pixel_pitch_m=grid.pixel_pitch_m,
        )
        shifted_loss.backward()
        self.assertGreater(shifted_components["centroid_consistency"], 1e-5)
        self.assertTrue(torch.isfinite(height.grad).all())

    def test_rgb_batched_fft_matches_scalar_and_multi_plane_gradients(self):
        system = OpticalSystem(n_grid=32, pixel_pitch_m=80e-6, pad_factor=2)
        grid = make_grid(system, torch.device("cpu"))
        material = CauchyMaterial()
        wavelengths = (488e-9, 532e-9, 633e-9)
        indices = material.indices(wavelengths)
        wavelength_tensor = torch.tensor(wavelengths, dtype=torch.float64)
        index_tensor = torch.tensor(indices, dtype=torch.float64)
        height = torch.rand((32, 32)) * 1.0e-6
        transfer = angular_spectrum_transfer_batch(
            wavelength_tensor, grid.fx, grid.fy, system.propagation_distance_m
        )
        batched = forward_rgb_single_plane(
            height, grid, wavelength_tensor, index_tensor, transfer
        )
        scalar = torch.stack([
            forward_field(height, grid, system, wavelength, index)
            for wavelength, index in zip(wavelengths, indices)
        ])
        self.assertTrue(torch.allclose(batched, scalar, atol=2e-4, rtol=2e-4))
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            cuda_grid = make_grid(system, cuda_device)
            cuda_transfer = angular_spectrum_transfer_batch(
                wavelength_tensor.to(cuda_device), cuda_grid.fx, cuda_grid.fy,
                system.propagation_distance_m,
            )
            cuda_output = forward_rgb_single_plane(
                height.to(cuda_device), cuda_grid,
                wavelength_tensor.to(cuda_device), index_tensor.to(cuda_device), cuda_transfer,
            )
            self.assertTrue(torch.allclose(batched, cuda_output.cpu(), atol=3e-4, rtol=3e-4))

        first_transfer = angular_spectrum_transfer_batch(
            wavelength_tensor, grid.fx, grid.fy, 0.02
        )
        second_transfer = angular_spectrum_transfer_batch(
            wavelength_tensor, grid.fx, grid.fy, 0.03
        )
        two_plane_height = torch.nn.Parameter(torch.rand((2, 32, 32)) * 1.0e-6)
        two_plane = forward_rgb_two_planes(
            two_plane_height, grid, wavelength_tensor, index_tensor,
            first_transfer, second_transfer,
        )
        spatial_weights = torch.rand((3, 32, 32))
        torch.sum(torch.abs(two_plane) ** 2 * spatial_weights).backward()
        self.assertEqual(tuple(two_plane.shape), (3, 32, 32))
        self.assertTrue(torch.isfinite(two_plane_height.grad).all())
        self.assertGreater(float(two_plane_height.grad.abs().sum()), 0.0)

        lens_transfer = angular_spectrum_transfer_batch(
            wavelength_tensor, grid.fx, grid.fy, system.lens_focal_length_m
        )
        lens_output = forward_rgb_fourier_lens(
            height, grid, wavelength_tensor, index_tensor, lens_transfer,
            system.lens_focal_length_m,
        )
        self.assertEqual(tuple(lens_output.shape), (3, 32, 32))

    def test_supersampled_sharpened_target_is_partitioned_and_centered(self):
        system = OpticalSystem(n_grid=128, pixel_pitch_m=10e-6)
        grid = make_grid(system, torch.device("cpu"))
        target = supersampled_edge_target(
            lambda x, y: soft_circle_two_spots(x, y),
            grid.base_n,
            grid.pixel_pitch_m,
            torch.device("cpu"),
            sharpness_gain=1.6,
            supersample_factor=4,
        )
        self.assertAlmostEqual(float(target.intensity.sum()), 1.0, places=6)
        partition = target.signal_mask + target.transition_mask + target.background_mask
        self.assertTrue(torch.allclose(partition, torch.ones_like(partition), atol=1e-6))
        target_x = torch.sum(target.intensity * grid.base_x)
        target_y = torch.sum(target.intensity * grid.base_y)
        self.assertLess(float(torch.hypot(target_x, target_y)), 1e-6)
        self.assertEqual(target.metadata["edge_preprocess"], "supersampled_logit_sharpening")


if __name__ == "__main__":
    unittest.main()
