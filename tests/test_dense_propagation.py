import sys
import unittest
from pathlib import Path

import torch
import torch.nn.functional as functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from forward_models import (
    OpticalSystem,
    angular_spectrum_transfer,
    apply_dense_propagation,
    build_dense_angular_spectrum_matrix,
    build_dense_fresnel_matrix,
    make_grid,
    propagate,
)


class DensePropagationTests(unittest.TestCase):
    def make_system(self, device, pad_factor=1):
        system = OpticalSystem(
            n_grid=8,
            pixel_pitch_m=8e-6,
            aperture_radius_m=20e-6,
            propagation_distance_m=0.015,
            pad_factor=pad_factor,
        )
        return system, make_grid(system, device)

    def make_field(self, device):
        values = torch.arange(64, device=device, dtype=torch.float32).reshape(8, 8)
        return torch.complex(torch.sin(values / 7.0), torch.cos(values / 11.0))

    def fft_base_propagate(self, field, grid, wavelength_m, distance_m):
        right_pad = grid.work_n - grid.base_n - grid.crop_start
        work_field = functional.pad(
            field,
            (grid.crop_start, right_pad, grid.crop_start, right_pad),
        )
        transfer = angular_spectrum_transfer(
            wavelength_m, grid.fx, grid.fy, distance_m
        )
        output = propagate(work_field, transfer)
        return output[
            grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end
        ]

    def test_dense_angular_spectrum_matches_fft_with_and_without_padding(self):
        for pad_factor in (1, 2):
            system, grid = self.make_system(torch.device("cpu"), pad_factor)
            matrix = build_dense_angular_spectrum_matrix(
                532e-9, grid, system.propagation_distance_m, chunk_size=11
            )
            field = self.make_field(torch.device("cpu"))
            dense_output = apply_dense_propagation(matrix, field)
            fft_output = self.fft_base_propagate(
                field, grid, 532e-9, system.propagation_distance_m
            )
            error = torch.linalg.vector_norm(dense_output - fft_output) / torch.linalg.vector_norm(fft_output)
            self.assertLess(float(error), 1e-5)

    def test_zero_distance_dense_angular_spectrum_is_identity(self):
        _, grid = self.make_system(torch.device("cpu"), pad_factor=2)
        matrix = build_dense_angular_spectrum_matrix(532e-9, grid, 0.0, chunk_size=13)
        field = self.make_field(torch.device("cpu"))
        output = apply_dense_propagation(matrix, field)
        self.assertTrue(torch.allclose(output, field, atol=1e-6, rtol=1e-6))

    def test_fresnel_matrix_shape_linearity_and_gradient(self):
        system, grid = self.make_system(torch.device("cpu"))
        matrix = build_dense_fresnel_matrix(
            532e-9, grid, system.propagation_distance_m, chunk_size=17
        )
        self.assertEqual(matrix.shape, (64, 64))
        self.assertTrue(torch.isfinite(matrix.real).all())
        self.assertTrue(torch.isfinite(matrix.imag).all())

        first = self.make_field(torch.device("cpu")).requires_grad_()
        second = torch.flip(first.detach(), dims=(0,)).to(torch.complex64)
        combined = apply_dense_propagation(matrix, 0.3 * first + 0.7j * second)
        separate = 0.3 * apply_dense_propagation(matrix, first) + 0.7j * apply_dense_propagation(matrix, second)
        self.assertTrue(torch.allclose(combined, separate, atol=2e-5, rtol=2e-5))
        combined.abs().square().sum().backward()
        self.assertIsNotNone(first.grad)
        self.assertTrue(torch.isfinite(first.grad).all())

    def test_memory_guard_rejects_before_matrix_allocation(self):
        _, grid = self.make_system(torch.device("cpu"))
        with self.assertRaises(MemoryError):
            build_dense_angular_spectrum_matrix(
                532e-9, grid, 0.01, max_matrix_bytes=1
            )
        with self.assertRaises(MemoryError):
            build_dense_fresnel_matrix(532e-9, grid, 0.01, max_matrix_bytes=1)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is unavailable")
    def test_cuda_dense_and_cpu_dense_outputs_agree(self):
        _, cpu_grid = self.make_system(torch.device("cpu"), pad_factor=2)
        _, gpu_grid = self.make_system(torch.device("cuda"), pad_factor=2)
        cpu_matrix = build_dense_angular_spectrum_matrix(532e-9, cpu_grid, 0.015)
        gpu_matrix = build_dense_angular_spectrum_matrix(532e-9, gpu_grid, 0.015)
        cpu_output = apply_dense_propagation(cpu_matrix, self.make_field(torch.device("cpu")))
        gpu_output = apply_dense_propagation(
            gpu_matrix, self.make_field(torch.device("cuda"))
        ).cpu()
        torch.testing.assert_close(gpu_output, cpu_output, rtol=1e-4, atol=1e-5)


if __name__ == "__main__":
    unittest.main()
