import sys
import unittest
import tempfile
import json
from dataclasses import asdict
from unittest.mock import patch
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forward_models import (
    OpticalSystem, OpticalStopConfig, centered_fft, forward_rgb_4f_stop,
    forward_rgb_lens_aperture, make_doe_field_rgb, make_fourier_stop,
    make_grid, make_image_stop, angular_spectrum_transfer_batch, propagate,
    validate_sampling,
)
from losses import (
    background_absolute_loss, background_peak_loss, efficiency_floor_loss,
    LossSchedule, LossWeights, joint_loss,
)
from rgb_background_filter_experiment import (
    OpticalConfig, acceptance_status, aggregate_rows, make_forward,
    optical_scan, sampling_checks, select_candidates, spectrum_reference, run, parse_args, run_one,
    weight_scan_jobs,
)
from rgb_decolorization_experiment import make_target
from targets import soft_circle_two_spots


class BackgroundFilterTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(2)
        self.system = OpticalSystem(n_grid=128, pixel_pitch_m=20e-6, pad_factor=2)
        self.grid = make_grid(self.system, torch.device("cpu"))
        self.waves = (488e-9, 532e-9, 633e-9)
        self.indices = (1.65, 1.6482, 1.6347)
        self.height = torch.rand(128, 128) * 1.4e-6
        self.target = soft_circle_two_spots(self.grid.base_x, self.grid.base_y)

    def test_relay_recovers_input_and_free_propagation(self):
        input_field = make_doe_field_rgb(self.height, self.grid, self.waves, self.indices)
        relay = forward_rgb_4f_stop(self.height, self.grid, self.waves, self.indices)
        crop = input_field[:, self.grid.crop_start:self.grid.crop_end, self.grid.crop_start:self.grid.crop_end]
        torch.testing.assert_close(relay, crop, atol=1e-6, rtol=1e-6)
        propagated = propagate(input_field, angular_spectrum_transfer_batch(self.waves, self.grid.fx, self.grid.fy, 0.05))
        relay = forward_rgb_4f_stop(self.height, self.grid, self.waves, self.indices, object_distance_m=0.05)
        torch.testing.assert_close(relay, propagated[:, self.grid.crop_start:self.grid.crop_end,
                                                    self.grid.crop_start:self.grid.crop_end], atol=1e-6, rtol=1e-6)

    def test_ones_stop_disabled_and_rgb_scalar_batch_agree(self):
        output = forward_rgb_4f_stop(self.height, self.grid, self.waves, self.indices)
        ones = forward_rgb_4f_stop(self.height, self.grid, self.waves, self.indices,
            stop=torch.ones(self.grid.work_n, self.grid.work_n), image_stop=torch.ones(self.grid.work_n, self.grid.work_n))
        torch.testing.assert_close(output, ones)
        for hard in (False, True):
            config = OpticalStopConfig(stop_type="fourier_annular", position="fourier",
                radius_m=0.4e-3, inner_radius_m=0.03e-3, soft_edge=300.0)
            batch = forward_rgb_4f_stop(self.height, self.grid, self.waves, self.indices, stop=config, hard_stop=hard)
            scalar = torch.cat([forward_rgb_4f_stop(self.height, self.grid, (wave,), (index,),
                                                  stop=config, hard_stop=hard) for wave, index in zip(self.waves, self.indices)])
            torch.testing.assert_close(batch, scalar)

    def test_stop_monotone_energy_and_physical_radius_shared(self):
        spectrum = centered_fft(make_doe_field_rgb(self.height, self.grid, self.waves, self.indices))
        small = make_fourier_stop(self.grid, self.waves, 0.05, radius_m=0.3e-3, hard=True)
        large = make_fourier_stop(self.grid, self.waves, 0.05, radius_m=0.5e-3, hard=True)
        self.assertTrue(torch.all(large >= small))
        self.assertTrue(torch.all((spectrum * large).abs().square().sum((-2, -1)) >=
                                  (spectrum * small).abs().square().sum((-2, -1))))
        self.assertGreater(float(small[0].sum()), float(small[2].sum()))
        for hard in (False, True):
            mask = make_fourier_stop(self.grid, self.waves, 0.05, 10000.0,
                inner_cutoff_frequency_cpm=500.0, stop_type="annular", soft_edge_cpm=100.0, hard=hard)
            self.assertEqual(mask.shape, (3, self.grid.work_n, self.grid.work_n))
            self.assertTrue(torch.isfinite(mask).all())
            self.assertLess(float(mask[:, self.grid.work_n // 2, self.grid.work_n // 2].max()), 0.01)

    def test_stop_and_height_gradients_are_finite(self):
        cutoff = torch.tensor(10000.0, requires_grad=True)
        inner = torch.tensor(500.0, requires_grad=True)
        mask = make_fourier_stop(self.grid, self.waves, 0.05, cutoff, inner,
                                 soft_edge_cpm=300.0, stop_type="annular")
        height = self.height.clone().requires_grad_()
        field = forward_rgb_4f_stop(height, self.grid, self.waves, self.indices, stop=mask)
        field.abs().square().sum().backward()
        for tensor in (cutoff, inner, height):
            self.assertTrue(torch.isfinite(tensor.grad).all())
            self.assertGreater(float(tensor.grad.abs().sum()), 0.0)
        radius = torch.tensor(0.4e-3, requires_grad=True)
        image = make_image_stop(self.grid, radius, 20e-6)
        image.sum().backward()
        self.assertTrue(torch.isfinite(radius.grad))

    def test_image_stop_centred_and_doe_coverage_independent(self):
        original = self.grid.doe_aperture.clone()
        soft = make_image_stop(self.grid, 0.5e-3, 20e-6)
        hard = make_image_stop(self.grid, 0.5e-3, hard=True)
        self.assertEqual(soft.shape, hard.shape)
        self.assertEqual(int(torch.argmax(soft)), self.grid.work_n // 2 * (self.grid.work_n + 1))
        self.assertTrue(torch.equal(original, self.grid.doe_aperture))
        self.assertTrue(validate_sampling(self.system, self.grid)["aperture_fully_inside_base_grid"])
        bad = OpticalSystem(n_grid=128, pixel_pitch_m=5e-6)
        with self.assertRaises(ValueError):
            validate_sampling(bad, make_grid(bad, torch.device("cpu")))

    def test_disabled_image_and_two_doe_relay_match_baseline(self):
        for architecture in ("single_doe", "two_doe_planes"):
            height = self.height if architecture == "single_doe" else torch.stack((self.height, self.height))
            baseline = torch.stack(make_forward(OpticalConfig(), architecture, self.grid, self.system)(height))
            image = OpticalConfig(branch="image", stop=OpticalStopConfig(stop_type="image", radius_m=1.0))
            cropped = torch.stack(make_forward(image, architecture, self.grid, self.system, hard=True)(height))
            torch.testing.assert_close(cropped, baseline)
            remaining = 0.05 if architecture == "single_doe" else 0.025
            relay = forward_rgb_4f_stop(height, self.grid, self.waves, self.indices,
                plane_separation_m=0.025, object_distance_m=remaining)
            # Different refractive indices are used by make_forward; compare
            # the manual no-stop relay to its own manual free-space path.
            first = make_doe_field_rgb(height[0] if height.ndim == 3 else height, self.grid, self.waves, self.indices)
            if height.ndim == 3:
                first = propagate(first, angular_spectrum_transfer_batch(self.waves, self.grid.fx, self.grid.fy, 0.025)) * make_doe_field_rgb(height[1], self.grid, self.waves, self.indices)
            expected = propagate(first, angular_spectrum_transfer_batch(self.waves, self.grid.fx, self.grid.fy, remaining))
            torch.testing.assert_close(relay, expected[:, self.grid.crop_start:self.grid.crop_end,
                                                      self.grid.crop_start:self.grid.crop_end], atol=1e-6, rtol=1e-6)

    def test_lens_focal_transform_energy_and_gradient(self):
        height = self.height.clone().requires_grad_()
        output, diagnostic = forward_rgb_lens_aperture(height, self.grid, self.waves, self.indices,
                                                     0.12, 0.5e-3, return_diagnostics=True)
        torch.testing.assert_close(diagnostic["native_focal_energy"], diagnostic["post_stop_energy"], atol=1e-3, rtol=1e-6)
        self.assertTrue((diagnostic["stop_throughput"] <= 1.0).all())
        output.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(height.grad).all())

    def test_finite_lens_relay_is_explicit_and_energy_bounded(self):
        height = self.height.clone().requires_grad_()
        output, diagnostic = forward_rgb_4f_stop(height, self.grid, self.waves, self.indices,
            focal_length_m=0.05, lens_aperture_radius_m=0.5e-3,
            object_distance_m=0.05, return_diagnostics=True)
        for plane in ("lens1", "lens2"):
            self.assertTrue((diagnostic[f"{plane}_post_energy"] <= diagnostic[f"{plane}_pre_energy"] + 1e-3).all())
        output.abs().square().mean().backward()
        self.assertTrue(torch.isfinite(height.grad).all())
        with self.assertRaisesRegex(ValueError, "Nyquist"):
            forward_rgb_4f_stop(height, self.grid, self.waves, self.indices,
                                lens_aperture_radius_m=1e-3)

    def test_losses_scale_response_and_bright_background(self):
        field = torch.sqrt(self.target.intensity + 1e-5).to(torch.complex64)
        first = background_absolute_loss(field, self.target)
        second = background_absolute_loss(2 * field, self.target)
        torch.testing.assert_close(first, second)
        peak = background_peak_loss(field, self.target)
        torch.testing.assert_close(background_peak_loss(2 * field, self.target), peak)
        roi_scaled = torch.where(self.target.signal_mask > 0, 2 * field, field)
        self.assertLess(background_peak_loss(roi_scaled, self.target), peak)
        bright = field.clone()
        bright[0, 0] = 5.0
        self.assertGreater(background_absolute_loss(bright, self.target), first)
        self.assertGreater(background_peak_loss(bright, self.target), peak)
        eta = float((field.abs().square() * self.target.signal_mask).sum())
        self.assertEqual(float(efficiency_floor_loss(field, self.target, 0.9 * eta)), 0.0)
        self.assertGreater(float(efficiency_floor_loss(field * 0.1, self.target, 0.9 * eta)), 0.0)

    def test_zero_peak_background_has_finite_gradients(self):
        for scale in (0.0, 1e-20, 1.0, 1e10):
            field = torch.full((128, 128), complex(scale), dtype=torch.complex64, requires_grad=True)
            loss = background_peak_loss(field, self.target)
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(torch.isfinite(field.grad).all())

    def test_new_joint_terms_and_required_floor(self):
        height = self.height.clone().requires_grad_()
        fields = [(height * 1e6 + 0.1).to(torch.complex64)] * 3
        weights = LossWeights(rgb_background_absolute_weight=0.1, rgb_background_peak_weight=0.05,
                              efficiency_floor_weight=0.1)
        with self.assertRaises(ValueError):
            joint_loss(fields, self.target, height, 1.4e-6, weights, LossSchedule(0, 0), 5)
        loss, parts = joint_loss(fields, self.target, height, 1.4e-6, weights, LossSchedule(0, 0), 5,
            efficiency_floor=(0.1,) * 3, input_powers=(1000.0,) * 3)
        loss.backward()
        self.assertTrue(torch.isfinite(height.grad).all())
        self.assertIn("rgb_background_absolute", parts)

    def test_sampling_rejects_stop_beyond_nyquist(self):
        config = OpticalConfig(branch="fourier_circular", stop=OpticalStopConfig(
            stop_type="fourier_circular", radius_m=2e-3, soft_edge=300.0))
        self.assertFalse(sampling_checks(self.system, self.grid, config)["valid"])
        for name in ("circle", "cross", "ring50"):
            target = make_target(name, self.grid, torch.device("cpu"))
            reference = spectrum_reference(target, self.grid)
            configs = optical_scan(reference, "baseline_image")
            self.assertEqual(len(configs), 33)
            for branch in ("none", "lens", "fourier_circular", "fourier_annular", "image"):
                self.assertTrue(any(sampling_checks(self.system, self.grid, config)["valid"]
                                    for config in configs if config.branch == branch))
            if name == "cross":
                self.assertTrue(any(not sampling_checks(self.system, self.grid, config)["valid"] for config in configs))

    def test_hard_constraints_per_channel_and_ring(self):
        baseline = {"mean_background_roi_ratio": 0.2, "mean_background_p99": 0.01,
                    "worst_correlation": 0.9, "roi_efficiencies": [0.5] * 3,
                    "max_component_centroid_drift_um": 2.0, "max_hole_leakage": 0.1}
        candidate = dict(baseline, mean_background_roi_ratio=0.1)
        self.assertTrue(acceptance_status(candidate, baseline, True)["accepted"])
        for key, value, failed in (("roi_efficiencies", [0.44, 0.9, 0.9], "efficiency"),
                                   ("worst_correlation", 0.84, "target_quality"),
                                   ("max_component_centroid_drift_um", 3.1, "centroid"),
                                   ("max_hole_leakage", 0.126, "hole")):
            status = acceptance_status(dict(candidate, **{key: value}), baseline, True)
            self.assertIn(failed, status["failed"])

    def test_candidates_aggregate_all_seeds_and_pareto_fallback(self):
        def row(optical_id, seed, accepted, reduction):
            return {"optical_id": optical_id, "seed": seed, "variant": "full_bg_absolute_peak",
                "weights_id": "default", "optical_config": {"branch": "fourier_circular"},
                "loss_parameters": (0.1, 0.05, 0.1), "mean_background_roi_ratio": 0.2 - reduction,
                "mean_background_p99": 0.001, "mean_background_absolute_fraction": 0.1,
                "mean_roi_absolute_efficiency": 0.5, "mean_common_scale_roi_pearson": 0.9,
                "rgb_roi_energy_cv": 0.1, "worst_correlation": 0.9,
                "max_component_centroid_drift_um": 0.5, "max_hole_leakage": 0.01,
                "acceptance_status": {"accepted": accepted, "background_ratio_reduction": reduction,
                                      "background_p99_reduction": 0.001}}
        rows = [row("a", 7, True, 0.1), row("a", 17, False, 0.1),
                row("b", 7, True, 0.05), row("b", 17, True, 0.05)]
        self.assertFalse(aggregate_rows(rows)[0]["acceptance_status"]["accepted"])
        self.assertEqual(select_candidates(rows)[0]["optical_id"], "b")
        fallback = select_candidates(rows[:2])
        self.assertIn("Pareto fallback", fallback[0]["selection_reason"])

    def test_weight_scan_is_limited_to_three_selected_optics(self):
        candidates = []
        for radius in (0.5e-3, 0.75e-3, 1.0e-3, 1.25e-3):
            config = OpticalConfig(branch="lens", lens_radius_m=radius)
            candidates.append({"optical_config": {
                **asdict(config), "stop": asdict(config.stop)}})
        jobs = weight_scan_jobs(candidates)
        self.assertEqual(len(jobs), 3 * 55)
        self.assertEqual({job[0].lens_radius_m for job in jobs}, {0.5e-3, 0.75e-3, 1.0e-3})

    def test_driver_artifacts_resume_and_config_mismatch(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parse_args(["--output-dir", directory, "--device", "cpu",
                "--screen-coarse-iterations", "2", "--screen-fine-iterations", "3"])
            args.output_dir = Path(directory)
            config = OpticalConfig()
            # Numerical optimization/evaluation and JSON/array persistence are
            # real; image rendering is exercised by the CLI smoke execution.
            def save_stub(path, *unused):
                path.mkdir(parents=True, exist_ok=True)
            with patch("rgb_background_filter_experiment.save_run", side_effect=save_stub), \
                 patch("rgb_background_filter_experiment.save_scalar_image"):
                baseline = run_one(args, "screen", "ring50", "two_doe_planes", 7,
                                   config, "full", (0.0, 0.0, 0.0))
                target = make_target("ring50", self.grid, torch.device("cpu"))
                stop = next(config for config in optical_scan(spectrum_reference(target, self.grid), "baseline_image")
                            if config.branch == "fourier_annular" and config.dc_fraction == 0.2)
                candidate = run_one(args, "screen", "ring50", "two_doe_planes", 7,
                    stop, "full_bg_constrained", (0.1, 0.05, 0.1), baseline)
                root = args.output_dir / candidate["run_dir"]
                result = json.loads((root / "result.json").read_text(encoding="utf-8"))
                for key in ("optical_config", "stop_throughput", "background_metrics", "acceptance_status"):
                    self.assertIn(key, result)
                self.assertTrue((root / "unfiltered_same_height_metrics.json").exists())
                self.assertTrue((root / "hard" / "fourier_stop_mask.npy").exists())
                self.assertEqual(len(result["background_metrics"]), 3)
                self.assertIn("max_hole_leakage", candidate)
                args.resume = True
                self.assertEqual(run_one(args, "screen", "ring50", "two_doe_planes", 7,
                    stop, "full_bg_constrained", (0.1, 0.05, 0.1), baseline)["optical_id"], candidate["optical_id"])
                args.screen_fine_iterations = 4
                with self.assertRaisesRegex(ValueError, "resume config mismatch"):
                    run_one(args, "screen", "ring50", "two_doe_planes", 7, stop,
                            "full_bg_constrained", (0.1, 0.05, 0.1), baseline)


if __name__ == "__main__":
    unittest.main()
