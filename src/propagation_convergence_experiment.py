"""Check propagation convergence, coordinate calibration, and dense-matrix fidelity."""
from __future__ import annotations

import argparse
import csv
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


DEFAULT_GRID_SIZES = (16, 24, 32, 48, 64)
DEFAULT_WAVELENGTHS_NM = (488.0, 532.0, 633.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-sizes", type=int, nargs="+", default=DEFAULT_GRID_SIZES)
    parser.add_argument("--wavelengths-nm", type=float, nargs="+", default=DEFAULT_WAVELENGTHS_NM)
    parser.add_argument("--field-of-view-um", type=float, default=320.0)
    parser.add_argument("--distance-mm", type=float, default=10.0)
    parser.add_argument("--aperture-radius-um", type=float, default=120.0)
    parser.add_argument("--grating-period-um", type=float, default=80.0)
    parser.add_argument("--grating-height-um", type=float, default=0.5)
    parser.add_argument("--pad-factor", type=int, choices=(1, 2, 4), default=2)
    parser.add_argument("--matrix-chunk-size", type=int, default=32)
    parser.add_argument("--fresnel-chunk-size", type=int, default=128)
    parser.add_argument("--max-matrix-mib", type=float, default=256.0)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/propagation_convergence"))
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


def timed_average(operation: Callable[[], torch.Tensor], device: torch.device, repeats: int) -> tuple[torch.Tensor, float]:
    for _ in range(2):
        result = operation()
    synchronize(device)
    start = time.perf_counter()
    for _ in range(repeats):
        result = operation()
    synchronize(device)
    return result, (time.perf_counter() - start) / repeats


def make_system(n_grid: int, field_of_view_m: float, distance_m: float, aperture_radius_m: float, pad_factor: int) -> OpticalSystem:
    return OpticalSystem(
        n_grid=n_grid,
        pixel_pitch_m=field_of_view_m / n_grid,
        aperture_radius_m=aperture_radius_m,
        propagation_distance_m=distance_m,
        pad_factor=pad_factor,
    )


def make_demo_height(grid: SimulationGrid, period_m: float, height_m: float) -> torch.Tensor:
    return (0.5 * height_m * (1.0 + torch.cos(2.0 * math.pi * grid.base_x / period_m))).to(torch.float64)


def make_doe_input(height_m: torch.Tensor, grid: SimulationGrid, wavelength_m: float, refractive_index: float) -> torch.Tensor:
    phase = 2.0 * math.pi * (refractive_index - 1.0) * height_m / wavelength_m
    aperture = grid.doe_aperture[grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end]
    return aperture.to(torch.complex64) * torch.exp(1j * phase).to(torch.complex64)


def fft_propagate_base(field: torch.Tensor, grid: SimulationGrid, transfer: torch.Tensor) -> torch.Tensor:
    right_pad = grid.work_n - grid.base_n - grid.crop_start
    work_field = functional.pad(field, (grid.crop_start, right_pad, grid.crop_start, right_pad))
    output = propagate(work_field, transfer)
    return output[grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end]


def fft_propagate_work(field: torch.Tensor, grid: SimulationGrid, transfer: torch.Tensor) -> torch.Tensor:
    right_pad = grid.work_n - grid.base_n - grid.crop_start
    work_field = functional.pad(field, (grid.crop_start, right_pad, grid.crop_start, right_pad))
    return propagate(work_field, transfer)


def relative_l2(actual: torch.Tensor, reference: torch.Tensor) -> float:
    numerator = torch.linalg.vector_norm((actual - reference).to(torch.complex128))
    denominator = torch.linalg.vector_norm(reference.to(torch.complex128))
    return float((numerator / (denominator + 1e-30)).detach().cpu())


def relative_energy(before: torch.Tensor, after: torch.Tensor) -> float:
    before_energy = torch.sum(torch.abs(before) ** 2)
    after_energy = torch.sum(torch.abs(after) ** 2)
    return float(torch.abs(after_energy / (before_energy + 1e-30) - 1.0).detach().cpu())


def order_positions(intensity: torch.Tensor, grid: SimulationGrid, wavelength_m: float, distance_m: float, period_m: float) -> dict[str, object]:
    result: dict[str, object] = {}
    for order in (-1, 1):
        expected_x = order * wavelength_m * distance_m / period_m
        window_radius = max(2.0 * grid.pixel_pitch_m, 0.30 * abs(expected_x))
        mask = (grid.base_x - expected_x).square() + grid.base_y.square() <= window_radius**2
        weights = intensity * mask
        total = weights.sum() + 1e-30
        measured_x = (weights * grid.base_x).sum() / total
        measured_y = (weights * grid.base_y).sum() / total
        x_error = torch.abs(measured_x - expected_x)
        tolerance = max(2.0 * grid.pixel_pitch_m, 0.05 * abs(expected_x))
        result[str(order)] = {
            "expected_x_um": float(expected_x * 1e6),
            "measured_x_um": float(measured_x.detach().cpu() * 1e6),
            "measured_y_um": float(measured_y.detach().cpu() * 1e6),
            "absolute_x_error_um": float(x_error.detach().cpu() * 1e6),
            "allowed_error_um": float(tolerance * 1e6),
            "within_tolerance": bool(x_error <= tolerance),
            "window_energy": float(weights.sum().detach().cpu()),
        }
    return result


def finite_tensor(value: torch.Tensor) -> bool:
    return bool(torch.isfinite(value.real).all() and torch.isfinite(value.imag).all())


def save_intensity(path: Path, field: torch.Tensor) -> None:
    np.save(path, field.abs().square().detach().cpu().numpy())


def plot_metric(path: Path, rows: list[dict[str, object]], wavelengths_nm: tuple[float, ...], key: str, title: str, ylabel: str, log_y: bool = False, percent: bool = False) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for wavelength_nm in wavelengths_nm:
        selected = sorted((row for row in rows if row["wavelength_nm"] == wavelength_nm), key=lambda row: row["n_grid"])
        values = [float(row[key]) * (100.0 if percent else 1.0) for row in selected]
        axis.plot([row["n_grid"] for row in selected], values, "o-", label=f"{wavelength_nm:g} nm")
    axis.set(title=title, xlabel="grid size", ylabel=ylabel)
    if log_y:
        axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_order_positions(path: Path, rows: list[dict[str, object]], wavelengths_nm: tuple[float, ...]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for wavelength_nm in wavelengths_nm:
        selected = sorted((row for row in rows if row["wavelength_nm"] == wavelength_nm), key=lambda row: row["n_grid"])
        sizes = [row["n_grid"] for row in selected]
        theory = [row["plus_one_expected_x_um"] for row in selected]
        measured = [row["plus_one_measured_x_um"] for row in selected]
        axis.plot(sizes, theory, "--", label=f"{wavelength_nm:g} nm theory")
        axis.plot(sizes, measured, "o-", label=f"{wavelength_nm:g} nm measured")
    axis.set(title="+1 diffraction-order coordinate", xlabel="grid size", ylabel="x (µm)")
    axis.grid(True, alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def plot_runtime(path: Path, rows: list[dict[str, object]], wavelengths_nm: tuple[float, ...]) -> None:
    figure, axis = plt.subplots(figsize=(8, 4.5), constrained_layout=True)
    for wavelength_nm in wavelengths_nm:
        selected = sorted((row for row in rows if row["wavelength_nm"] == wavelength_nm), key=lambda row: row["n_grid"])
        axis.plot([row["n_grid"] for row in selected], [1000.0 * row["fft_seconds"] for row in selected], "o-", label=f"FFT {wavelength_nm:g} nm")
        axis.plot([row["n_grid"] for row in selected], [1000.0 * row["angular_dense_matvec_seconds"] for row in selected], "s--", label=f"dense matvec {wavelength_nm:g} nm")
    axis.set(title="Propagation runtime", xlabel="grid size", ylabel="mean time (ms)")
    axis.set_yscale("log")
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(fontsize=8, ncol=2)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def real_model_sampling_report(device: torch.device) -> dict[str, object]:
    reports: dict[str, object] = {}
    for label, n_grid, pitch_um in (("coarse", 256, 10.0), ("fine", 512, 5.0)):
        system = OpticalSystem(n_grid=n_grid, pixel_pitch_m=pitch_um * 1e-6, aperture_radius_m=1e-3, propagation_distance_m=0.05, pad_factor=2)
        grid = make_grid(system, device)
        report = validate_sampling(system, grid)
        transfer = angular_spectrum_transfer(532e-9, grid.fx, grid.fy, system.propagation_distance_m)
        field = grid.doe_aperture.to(torch.complex64)
        report["work_energy_error"] = relative_energy(field, propagate(field, transfer))
        reports[label] = report
    return reports


def run() -> dict[str, object]:
    args = parse_args()
    if args.repeats <= 0 or args.field_of_view_um <= 0 or args.max_matrix_mib <= 0:
        raise ValueError("repeats, field of view, and matrix budget must be positive")
    if any(size < 8 or size % 2 for size in args.grid_sizes):
        raise ValueError("all grid sizes must be even integers >= 8")
    device = resolve_device(args.device)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "outputs").mkdir(exist_ok=True)
    wavelengths_nm = tuple(float(value) for value in args.wavelengths_nm)
    wavelengths_m = tuple(value * 1e-9 for value in wavelengths_nm)
    material = CauchyMaterial()
    refractive_indices = material.indices(wavelengths_m)
    field_of_view_m = args.field_of_view_um * 1e-6
    distance_m = args.distance_mm * 1e-3
    aperture_radius_m = args.aperture_radius_um * 1e-6
    period_m = args.grating_period_um * 1e-6
    height_m = args.grating_height_um * 1e-6
    max_matrix_bytes = int(args.max_matrix_mib * 1024**2)
    rows: list[dict[str, object]] = []

    for n_grid in args.grid_sizes:
        system = make_system(n_grid, field_of_view_m, distance_m, aperture_radius_m, args.pad_factor)
        grid = make_grid(system, device)
        sampling = validate_sampling(system, grid)
        unpadded_system = make_system(n_grid, field_of_view_m, distance_m, aperture_radius_m, 1)
        unpadded_grid = make_grid(unpadded_system, device)
        for wavelength_nm, wavelength_m, refractive_index in zip(wavelengths_nm, wavelengths_m, refractive_indices):
            height = make_demo_height(grid, period_m, height_m)
            input_field = make_doe_input(height, grid, wavelength_m, refractive_index)
            transfer = angular_spectrum_transfer(wavelength_m, grid.fx, grid.fy, distance_m)
            fft_output, fft_seconds = timed_average(lambda: fft_propagate_base(input_field, grid, transfer), device, args.repeats)
            work_input = functional.pad(input_field, (grid.crop_start, grid.work_n - grid.base_n - grid.crop_start, grid.crop_start, grid.work_n - grid.base_n - grid.crop_start))
            work_energy_error = relative_energy(work_input, fft_propagate_work(input_field, grid, transfer))

            synchronize(device)
            start = time.perf_counter()
            angular_matrix = build_dense_angular_spectrum_matrix(wavelength_m, grid, distance_m, chunk_size=args.matrix_chunk_size, max_matrix_bytes=max_matrix_bytes)
            synchronize(device)
            angular_build_seconds = time.perf_counter() - start
            angular_dense_output, angular_dense_matvec_seconds = timed_average(lambda: apply_dense_propagation(angular_matrix, input_field), device, args.repeats)
            angular_error = relative_l2(angular_dense_output, fft_output)
            angular_matrix_finite = finite_tensor(angular_matrix)
            del angular_matrix
            if device.type == "cuda":
                torch.cuda.empty_cache()

            synchronize(device)
            start = time.perf_counter()
            fresnel_matrix = build_dense_fresnel_matrix(wavelength_m, grid, distance_m, chunk_size=args.fresnel_chunk_size, max_matrix_bytes=max_matrix_bytes)
            synchronize(device)
            fresnel_build_seconds = time.perf_counter() - start
            fresnel_output, fresnel_matvec_seconds = timed_average(lambda: apply_dense_propagation(fresnel_matrix, input_field), device, args.repeats)
            fresnel_error = relative_l2(fresnel_output, fft_output)
            fresnel_matrix_finite = finite_tensor(fresnel_matrix)
            del fresnel_matrix
            if device.type == "cuda":
                torch.cuda.empty_cache()

            padded_positions = order_positions(fft_output.abs().square(), grid, wavelength_m, distance_m, period_m)
            unpadded_height = make_demo_height(unpadded_grid, period_m, height_m)
            unpadded_input = make_doe_input(unpadded_height, unpadded_grid, wavelength_m, refractive_index)
            unpadded_transfer = angular_spectrum_transfer(wavelength_m, unpadded_grid.fx, unpadded_grid.fy, distance_m)
            unpadded_output = fft_propagate_base(unpadded_input, unpadded_grid, unpadded_transfer)
            unpadded_positions = order_positions(unpadded_output.abs().square(), unpadded_grid, wavelength_m, distance_m, period_m)
            output_path = output_dir / "outputs" / f"n{n_grid}_{wavelength_nm:g}nm"
            output_path.mkdir(exist_ok=True)
            save_intensity(output_path / "fft_linear.npy", fft_output)
            save_intensity(output_path / "angular_dense_linear.npy", angular_dense_output)
            save_intensity(output_path / "fresnel_dense_linear.npy", fresnel_output)
            plus_order = padded_positions["1"]
            minus_order = padded_positions["-1"]
            row = {
                "n_grid": n_grid, "pixel_pitch_um": system.pixel_pitch_m * 1e6, "field_of_view_um": field_of_view_m * 1e6,
                "wavelength_nm": wavelength_nm, "refractive_index": refractive_index, "matrix_bytes": n_grid**4 * 8,
                "matrix_mib": n_grid**4 * 8 / 1024**2, "angular_matrix_finite": angular_matrix_finite,
                "fresnel_matrix_finite": fresnel_matrix_finite, "angular_dense_vs_fft_relative_l2": angular_error,
                "fresnel_vs_angular_relative_l2": fresnel_error, "work_energy_error": work_energy_error,
                "padding_relative_l2": relative_l2(fft_output, unpadded_output), "plus_one_expected_x_um": plus_order["expected_x_um"],
                "plus_one_measured_x_um": plus_order["measured_x_um"], "plus_one_error_um": plus_order["absolute_x_error_um"],
                "plus_one_allowed_error_um": plus_order["allowed_error_um"], "plus_one_within_tolerance": plus_order["within_tolerance"],
                "minus_one_error_um": minus_order["absolute_x_error_um"], "minus_one_within_tolerance": minus_order["within_tolerance"],
                "unpadded_plus_one_measured_x_um": unpadded_positions["1"]["measured_x_um"],
                "padded_unpadded_plus_one_shift_um": abs(float(plus_order["measured_x_um"]) - float(unpadded_positions["1"]["measured_x_um"])),
                "fft_seconds": fft_seconds, "angular_dense_build_seconds": angular_build_seconds,
                "angular_dense_matvec_seconds": angular_dense_matvec_seconds, "fresnel_dense_build_seconds": fresnel_build_seconds,
                "fresnel_dense_matvec_seconds": fresnel_matvec_seconds, "sampling_report": sampling,
            }
            rows.append(row)
            print(f"N={n_grid:>2}, {wavelength_nm:g} nm: AS/FFT={angular_error:.2e}, Fresnel/AS={fresnel_error:.2e}, +1 error={plus_order['absolute_x_error_um']:.2f} um")

    actual_sampling = real_model_sampling_report(device)
    csv_fields = [key for key in rows[0] if key != "sampling_report"]
    with (output_dir / "convergence_summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=csv_fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in csv_fields} for row in rows)
    with (output_dir / "sampling_report.json").open("w", encoding="utf-8") as stream:
        json.dump({"real_model": actual_sampling, "dense_model": rows[::len(wavelengths_nm)]}, stream, indent=2, ensure_ascii=False)
    report = {
        "experiment": "propagation convergence and coordinate validation", "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": {
            "grid_sizes": list(args.grid_sizes), "wavelengths_nm": list(wavelengths_nm), "field_of_view_m": field_of_view_m,
            "distance_m": distance_m, "aperture_radius_m": aperture_radius_m, "grating_period_m": period_m,
            "grating_height_m": height_m, "pad_factor": args.pad_factor, "max_matrix_bytes": max_matrix_bytes,
            "material": material.metadata(wavelengths_m),
        },
        "rows": rows, "real_model_sampling": actual_sampling,
    }
    with (output_dir / "convergence_summary.json").open("w", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False, allow_nan=False)
    plot_metric(output_dir / "fresnel_vs_angular_error.png", rows, wavelengths_nm, "fresnel_vs_angular_relative_l2", "Fresnel vs angular spectrum", "relative L2 error", percent=True)
    plot_metric(output_dir / "angular_dense_vs_fft_error.png", rows, wavelengths_nm, "angular_dense_vs_fft_relative_l2", "Dense angular spectrum vs FFT", "relative L2 error", log_y=True)
    plot_metric(output_dir / "energy_conservation.png", rows, wavelengths_nm, "work_energy_error", "Work-plane energy relative error", "|Eout/Ein - 1|", log_y=True)
    plot_order_positions(output_dir / "diffraction_order_position.png", rows, wavelengths_nm)
    plot_runtime(output_dir / "runtime_vs_grid_size.png", rows, wavelengths_nm)
    return report


if __name__ == "__main__":
    run()
