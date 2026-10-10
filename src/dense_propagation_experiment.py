"""Compare dense Fresnel and angular-spectrum propagation on a small RGB grid."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from forward_models import (
    OpticalSystem,
    SimulationGrid,
    angular_spectrum_transfer,
    apply_dense_propagation,
    build_dense_angular_spectrum_matrix,
    build_dense_fresnel_matrix,
    make_grid,
    propagate,
    validate_sampling,
)
from materials import CauchyMaterial


DEFAULT_WAVELENGTHS_NM = (488.0, 532.0, 633.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=32)
    parser.add_argument("--pixel-pitch-um", type=float, default=10.0)
    parser.add_argument("--distance-mm", type=float, default=10.0)
    parser.add_argument("--aperture-radius-um", type=float, default=120.0)
    parser.add_argument("--grating-period-um", type=float, default=80.0)
    parser.add_argument("--grating-height-um", type=float, default=0.5)
    parser.add_argument(
        "--wavelengths-nm", type=float, nargs="+", default=DEFAULT_WAVELENGTHS_NM
    )
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--matrix-chunk-size", type=int, default=32)
    parser.add_argument("--fresnel-chunk-size", type=int, default=128)
    parser.add_argument("--max-matrix-mib", type=float, default=256.0)
    parser.add_argument("--repeats", type=int, default=10)
    parser.add_argument("--output-dir", type=Path, default=Path("results/dense_propagation"))
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device("cuda")
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device("cpu")


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_average(
    operation: Callable[[], torch.Tensor], device: torch.device, repeats: int
) -> tuple[torch.Tensor, float]:
    for _ in range(3):
        result = operation()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        result = operation()
    synchronize(device)
    return result, (time.perf_counter() - start) / repeats


def make_demo_height(grid: SimulationGrid, period_m: float, height_m: float) -> torch.Tensor:
    if period_m <= 0 or height_m < 0:
        raise ValueError("grating period must be positive and height nonnegative")
    grating = 0.5 * height_m * (
        1.0 + torch.cos(2.0 * math.pi * grid.base_x / period_m)
    )
    return grating.to(torch.float64)


def make_doe_input(
    height_m: torch.Tensor,
    grid: SimulationGrid,
    wavelength_m: float,
    refractive_index: float,
) -> torch.Tensor:
    phase = 2.0 * math.pi * (refractive_index - 1.0) * height_m / wavelength_m
    aperture = grid.doe_aperture[
        grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end
    ]
    return aperture.to(torch.complex64) * torch.exp(1j * phase).to(torch.complex64)


def fft_propagate_base(
    field: torch.Tensor,
    grid: SimulationGrid,
    transfer: torch.Tensor,
) -> torch.Tensor:
    right_pad = grid.work_n - grid.base_n - grid.crop_start
    work_field = functional.pad(
        field,
        (grid.crop_start, right_pad, grid.crop_start, right_pad),
    )
    output = propagate(work_field, transfer)
    return output[
        grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end
    ]


def relative_l2_error(actual: torch.Tensor, reference: torch.Tensor) -> float:
    difference_norm = torch.linalg.vector_norm(
        (actual - reference).to(torch.complex128)
    )
    reference_norm = torch.linalg.vector_norm(reference.to(torch.complex128))
    return float((difference_norm / (reference_norm + 1e-30)).detach().cpu())


def first_order_positions(
    intensity: torch.Tensor,
    grid: SimulationGrid,
    wavelength_m: float,
    distance_m: float,
    period_m: float,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    window_radius = max(2.0 * grid.pixel_pitch_m, wavelength_m * distance_m / period_m * 0.3)
    for order in (-1, 1):
        expected_x = order * wavelength_m * distance_m / period_m
        mask = (
            (grid.base_x - expected_x).square() + grid.base_y.square()
            <= window_radius**2
        )
        weights = intensity * mask
        total = weights.sum() + 1e-30
        measured_x = (weights * grid.base_x).sum() / total
        measured_y = (weights * grid.base_y).sum() / total
        results[str(order)] = {
            "expected_x_um": float(expected_x * 1e6),
            "measured_x_um": float(measured_x.detach().cpu() * 1e6),
            "measured_y_um": float(measured_y.detach().cpu() * 1e6),
            "window_energy": float(weights.sum().detach().cpu()),
        }
    return results


def normalized_intensity(field: torch.Tensor) -> torch.Tensor:
    intensity = field.abs().square()
    return intensity / (intensity.sum() + 1e-30)


def save_wavelength_figure(
    path: Path,
    wavelength_nm: float,
    input_field: torch.Tensor,
    fft_output: torch.Tensor,
    angular_dense_output: torch.Tensor,
    fresnel_output: torch.Tensor,
    grid: SimulationGrid,
) -> None:
    fields = (input_field, fft_output, angular_dense_output, fresnel_output)
    normalized = [normalized_intensity(field).detach().cpu().numpy() for field in fields]
    reference = fft_output.detach()
    angular_error = (angular_dense_output - reference).abs() / (reference.abs().max() + 1e-30)
    fresnel_error = (fresnel_output - reference).abs() / (reference.abs().max() + 1e-30)
    angular_error_array = angular_error.detach().cpu().numpy()
    fresnel_error_array = fresnel_error.detach().cpu().numpy()
    extent = [
        float(grid.base_x.min().cpu() * 1e3),
        float(grid.base_x.max().cpu() * 1e3),
        float(grid.base_y.min().cpu() * 1e3),
        float(grid.base_y.max().cpu() * 1e3),
    ]
    figure, axes = plt.subplots(2, 4, figsize=(14, 7), constrained_layout=True)
    titles = ("DOE input", "Angular spectrum FFT", "Angular spectrum dense", "Fresnel dense")
    for axis, data, title in zip(axes[0], normalized, titles):
        image = axis.imshow(data, origin="lower", extent=extent, cmap="inferno")
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        figure.colorbar(image, ax=axis, fraction=0.046)
    for axis, data, title in zip(
        axes[1],
        normalized[1:],
        ("FFT output, dB", "Dense angular spectrum, dB", "Dense Fresnel, dB"),
    ):
        db = 10.0 * np.log10(np.maximum(data / max(float(data.max()), 1e-30), 1e-4))
        image = axis.imshow(db, origin="lower", extent=extent, cmap="inferno", vmin=-40, vmax=0)
        axis.set_title(title)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        figure.colorbar(image, ax=axis, fraction=0.046, label="dB relative to peak")
    error_axis = axes[1, 3]
    error_image = error_axis.imshow(
        np.maximum(angular_error_array, fresnel_error_array),
        origin="lower",
        extent=extent,
        cmap="magma",
    )
    error_axis.set_title("max normalized complex-field error")
    error_axis.set_xlabel("x (mm)")
    error_axis.set_ylabel("y (mm)")
    figure.colorbar(error_image, ax=error_axis, fraction=0.046)
    figure.suptitle(f"Dense propagation comparison at {wavelength_nm:g} nm")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_profile_figure(
    path: Path,
    wavelength_nm: float,
    outputs: dict[str, torch.Tensor],
    grid: SimulationGrid,
) -> None:
    center = grid.base_n // 2
    x_um = grid.base_x[center].detach().cpu().numpy() * 1e6
    figure, axis = plt.subplots(figsize=(8, 4), constrained_layout=True)
    for label, field in outputs.items():
        intensity = normalized_intensity(field)[center].detach().cpu().numpy()
        axis.plot(x_um, intensity, label=label)
    axis.set(
        title=f"Center-line normalized intensity at {wavelength_nm:g} nm",
        xlabel="x (µm)",
        ylabel="normalized intensity per pixel",
    )
    axis.legend()
    axis.grid(True, alpha=0.25)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def run() -> dict[str, object]:
    args = parse_args()
    if args.repeats <= 0 or args.max_matrix_mib <= 0:
        raise ValueError("repeats and max-matrix-mib must be positive")
    if not args.wavelengths_nm or any(value <= 0 for value in args.wavelengths_nm):
        raise ValueError("wavelengths must be positive")
    device = resolve_device(args.device)
    max_matrix_bytes = int(args.max_matrix_mib * 1024**2)
    system = OpticalSystem(
        n_grid=args.n_grid,
        pixel_pitch_m=args.pixel_pitch_um * 1e-6,
        aperture_radius_m=args.aperture_radius_um * 1e-6,
        propagation_distance_m=args.distance_mm * 1e-3,
        pad_factor=2,
    )
    grid = make_grid(system, device)
    sampling = validate_sampling(system, grid)
    wavelengths_m = tuple(value * 1e-9 for value in args.wavelengths_nm)
    material = CauchyMaterial()
    indices = material.indices(wavelengths_m)
    height = make_demo_height(
        grid, args.grating_period_um * 1e-6, args.grating_height_um * 1e-6
    )
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    matrix_bytes = args.n_grid**4 * torch.empty((), dtype=torch.complex64).element_size()
    if matrix_bytes > max_matrix_bytes:
        raise MemoryError(
            f"one dense matrix needs {matrix_bytes / 1024**2:.1f} MiB, "
            f"above --max-matrix-mib={args.max_matrix_mib:g}"
        )

    results: list[dict[str, object]] = []
    for wavelength_nm, wavelength_m, refractive_index in zip(
        args.wavelengths_nm, wavelengths_m, indices
    ):
        input_field = make_doe_input(height, grid, wavelength_m, refractive_index)
        transfer = angular_spectrum_transfer(
            wavelength_m, grid.fx, grid.fy, system.propagation_distance_m
        )
        fft_output, fft_seconds = timed_average(
            lambda: fft_propagate_base(input_field, grid, transfer), device, args.repeats
        )

        synchronize(device)
        start = time.perf_counter()
        angular_matrix = build_dense_angular_spectrum_matrix(
            wavelength_m,
            grid,
            system.propagation_distance_m,
            chunk_size=args.matrix_chunk_size,
            max_matrix_bytes=max_matrix_bytes,
        )
        synchronize(device)
        angular_build_seconds = time.perf_counter() - start
        angular_dense_output, angular_dense_seconds = timed_average(
            lambda: apply_dense_propagation(angular_matrix, input_field),
            device,
            args.repeats,
        )
        angular_error = relative_l2_error(angular_dense_output, fft_output)
        del angular_matrix
        if device.type == "cuda":
            torch.cuda.empty_cache()

        synchronize(device)
        start = time.perf_counter()
        fresnel_matrix = build_dense_fresnel_matrix(
            wavelength_m,
            grid,
            system.propagation_distance_m,
            chunk_size=args.fresnel_chunk_size,
            max_matrix_bytes=max_matrix_bytes,
        )
        synchronize(device)
        fresnel_build_seconds = time.perf_counter() - start
        fresnel_output, fresnel_seconds = timed_average(
            lambda: apply_dense_propagation(fresnel_matrix, input_field),
            device,
            args.repeats,
        )
        fresnel_error = relative_l2_error(fresnel_output, fft_output)
        del fresnel_matrix
        if device.type == "cuda":
            torch.cuda.empty_cache()

        outputs = {
            "angular_spectrum_fft": fft_output,
            "angular_spectrum_dense": angular_dense_output,
            "fresnel_dense": fresnel_output,
        }
        wavelength_tag = f"{wavelength_nm:g}nm"
        for method, field in outputs.items():
            np.save(output_dir / f"{wavelength_tag}_{method}.npy", field.detach().cpu().numpy())
            np.save(
                output_dir / f"{wavelength_tag}_{method}_linear_intensity.npy",
                field.abs().square().detach().cpu().numpy(),
            )
        np.save(
            output_dir / f"{wavelength_tag}_dense_fft_abs_difference.npy",
            (angular_dense_output - fft_output).abs().detach().cpu().numpy(),
        )
        save_wavelength_figure(
            output_dir / f"{wavelength_tag}_comparison.png",
            wavelength_nm,
            input_field,
            fft_output,
            angular_dense_output,
            fresnel_output,
            grid,
        )
        save_profile_figure(
            output_dir / f"{wavelength_tag}_center_profile.png",
            wavelength_nm,
            outputs,
            grid,
        )
        fft_intensity = fft_output.abs().square()
        fresnel_intensity = fresnel_output.abs().square()
        expected_first_order = wavelength_m * system.propagation_distance_m / (
            args.grating_period_um * 1e-6
        )
        result = {
            "wavelength_nm": wavelength_nm,
            "refractive_index": refractive_index,
            "angular_dense_vs_fft_relative_l2": angular_error,
            "angular_dense_vs_fft_max_abs": float(
                (angular_dense_output - fft_output).abs().max().detach().cpu()
            ),
            "fresnel_vs_fft_relative_l2": fresnel_error,
            "expected_first_order_x_um": expected_first_order * 1e6,
            "fft_first_order_positions": first_order_positions(
                fft_intensity,
                grid,
                wavelength_m,
                system.propagation_distance_m,
                args.grating_period_um * 1e-6,
            ),
            "fresnel_first_order_positions": first_order_positions(
                fresnel_intensity,
                grid,
                wavelength_m,
                system.propagation_distance_m,
                args.grating_period_um * 1e-6,
            ),
            "matrix": {
                "shape": [args.n_grid**2, args.n_grid**2],
                "dtype": "complex64",
                "estimated_bytes": matrix_bytes,
            },
            "timing_seconds": {
                "fft_propagation_mean": fft_seconds,
                "angular_dense_build": angular_build_seconds,
                "angular_dense_matvec_mean": angular_dense_seconds,
                "fresnel_dense_build": fresnel_build_seconds,
                "fresnel_dense_matvec_mean": fresnel_seconds,
            },
        }
        results.append(result)
        print(
            f"{wavelength_nm:g} nm: dense-AS/FFT rel-L2={angular_error:.3e}; "
            f"Fresnel/FFT rel-L2={fresnel_error:.3e}; "
            f"FFT={fft_seconds * 1e3:.3f} ms, dense matvec="
            f"{angular_dense_seconds * 1e3:.3f} ms"
        )

    report = {
        "experiment": "small-grid dense propagation matrix validation",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "torch_version": torch.__version__,
        "operator_notes": {
            "angular_spectrum_dense": "exact dense expansion of padded/cropped FFT angular-spectrum propagation",
            "fresnel_dense": "sampled Fresnel diffraction integral with pixel-area quadrature",
            "relative_error_note": "Fresnel-vs-angular-spectrum error includes paraxial and finite-window differences",
        },
        "config": {
            "n_grid": args.n_grid,
            "pixel_pitch_m": system.pixel_pitch_m,
            "field_of_view_m": system.n_grid * system.pixel_pitch_m,
            "distance_m": system.propagation_distance_m,
            "aperture_radius_m": system.aperture_radius_m,
            "grating_period_m": args.grating_period_um * 1e-6,
            "grating_height_m": args.grating_height_um * 1e-6,
            "pad_factor": system.pad_factor,
            "wavelengths_m": list(wavelengths_m),
            "matrix_chunk_size": args.matrix_chunk_size,
            "fresnel_chunk_size": args.fresnel_chunk_size,
            "max_matrix_bytes": max_matrix_bytes,
            "repeats": args.repeats,
            "sampling": sampling,
            "material": material.metadata(wavelengths_m),
        },
        "per_wavelength": results,
    }
    with (output_dir / "summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    return report


if __name__ == "__main__":
    run()
