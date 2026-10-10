"""Coarse-to-fine optimization for circular and annular DOE spots."""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from forward_models import (
    OpticalSystem,
    angular_spectrum_transfer,
    energy_relative_error,
    forward_field,
    make_doe_field,
    make_grid,
    propagate,
    validate_sampling,
)
from losses import LossSchedule, LossWeights
from materials import CauchyMaterial
from metrics import evaluate_output, normalized_intensity
from optimizers import AdamOptions, OptimizationResult, optimize_adam
from targets import TargetSpec, ring_two_spots, soft_circle_two_spots


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-grid", type=int, default=256)
    parser.add_argument("--coarse-pitch-um", type=float, default=10.0)
    parser.add_argument("--fine-grid", type=int, default=512)
    parser.add_argument("--fine-pitch-um", type=float, default=5.0)
    parser.add_argument("--pad-factor", type=int, choices=[1, 2, 4], default=2)
    parser.add_argument("--aperture-radius-um", type=float, default=1000.0)
    parser.add_argument("--propagation-distance-mm", type=float, default=50.0)
    parser.add_argument("--height-max-um", type=float, default=1.4)
    parser.add_argument("--wavelengths", type=float, nargs="+", default=[532, 633])
    parser.add_argument("--coarse-single-iterations", type=int, default=300)
    parser.add_argument("--coarse-joint-iterations", type=int, default=300)
    parser.add_argument("--fine-iterations", type=int, default=800)
    parser.add_argument("--learning-rates", type=float, nargs="+", default=[0.01, 0.04, 0.10])
    parser.add_argument("--background-weights", type=float, nargs="+", default=[0.05, 0.10, 0.25])
    parser.add_argument("--smoothness-weights", type=float, nargs="+", default=[0.0, 1e-6])
    parser.add_argument("--hole-weights", type=float, nargs="+", default=[0.0, 0.1, 0.5, 1.0])
    parser.add_argument("--mraf-biases", type=float, nargs="+", default=[0.0, 0.02, 0.05])
    parser.add_argument("--mraf-mixings", type=float, nargs="+", default=[0.02, 0.05, 0.10])
    parser.add_argument("--targets", nargs="+", choices=["circle", "ring50", "ring60"], default=["circle", "ring50", "ring60"])
    parser.add_argument("--methods", nargs="+", choices=["adam", "mraf"], default=["adam", "mraf"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--output-dir", type=Path, default=Path("results/ring_optimization"))
    parser.add_argument("--radial-max-um", type=float, default=120.0)
    parser.add_argument("--radial-bin-um", type=float, default=2.5)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def make_system(args: argparse.Namespace, fine: bool) -> OpticalSystem:
    return OpticalSystem(
        n_grid=args.fine_grid if fine else args.coarse_grid,
        pixel_pitch_m=(args.fine_pitch_um if fine else args.coarse_pitch_um) * 1e-6,
        aperture_radius_m=args.aperture_radius_um * 1e-6,
        propagation_distance_m=args.propagation_distance_mm * 1e-3,
        pad_factor=args.pad_factor,
    )


def target_parameters(name: str) -> tuple[str, float | None, float | None]:
    if name == "circle":
        return "soft_circle_two", 40e-6, 15e-6
    if name == "ring50":
        return "ring_two", 50e-6, 8e-6
    if name == "ring60":
        return "ring_two", 60e-6, 10e-6
    raise ValueError(f"unknown target {name}")


def make_target(name: str, grid) -> TargetSpec:
    target_type, radius_m, width_m = target_parameters(name)
    if target_type == "soft_circle_two":
        return soft_circle_two_spots(grid.base_x, grid.base_y, radius_m=radius_m, edge_width_m=width_m)
    return ring_two_spots(grid.base_x, grid.base_y, radius_m=radius_m, edge_width_m=width_m)


def target_label(name: str, target: TargetSpec) -> str:
    if target.target_type == "ring":
        return f"{name}_r{float(target.metadata['ring_radius_m']) * 1e6:.0f}_w{float(target.metadata['ring_width_m']) * 1e6:.0f}"
    return name


def make_forward(grid, system: OpticalSystem, wavelengths_m: tuple[float, ...]):
    material = CauchyMaterial()
    indices = material.indices(wavelengths_m)

    def forward(height_m: torch.Tensor) -> list[torch.Tensor]:
        return [
            forward_field(height_m, grid, system, wavelength_m, refractive_index)
            for wavelength_m, refractive_index in zip(wavelengths_m, indices)
        ]

    return forward, material, indices


def make_schedule(iterations: int) -> LossSchedule:
    if iterations < 100:
        shape_end = max(0, iterations // 3)
    else:
        shape_end = 100
    return LossSchedule(
        shape_only_until=shape_end,
        background_ramp_end=min(300, max(1, iterations)),
        hole_ramp_start=min(300, max(0, iterations // 2)),
        hole_ramp_end=min(400, max(1, iterations)),
    )


def optimize_stage(
    method: str,
    config: dict[str, float],
    target: TargetSpec,
    forward,
    height_max_m: float,
    n_grid: int,
    aperture: torch.Tensor,
    iterations: int,
    seed: int,
    initial_height: torch.Tensor | None,
) -> OptimizationResult:
    weights = LossWeights(
        beta_background=config["background_weight"],
        gamma_efficiency=0.10,
        delta_balance=0.05,
        eta_smooth=config["smoothness_weight"],
        hole_weight=config.get("hole_weight", 0.0),
    )
    options = AdamOptions(
        iterations=iterations,
        learning_rate_um=config["learning_rate_um"],
        seed=seed,
        mraf_background_bias=config.get("mraf_bias") if method == "mraf" else None,
        mraf_mixing=config.get("mraf_mixing", 0.05),
    )
    return optimize_adam(
        forward,
        target,
        height_max_m,
        n_grid,
        aperture,
        options,
        weights,
        make_schedule(iterations),
        initial_height,
    )


def resize_height(height_m: torch.Tensor, target_n: int, aperture: torch.Tensor, height_max_m: float) -> torch.Tensor:
    resized = functional.interpolate(
        height_m[None, None],
        size=(target_n, target_n),
        mode="bilinear",
        align_corners=True,
    )[0, 0]
    return resized.clamp(0.0, height_max_m) * aperture


def score_metrics(metrics: list[dict[str, Any]], target: TargetSpec) -> dict[str, float]:
    correlations = np.asarray([item["correlation"] for item in metrics], dtype=float)
    backgrounds = np.asarray([item["background_fraction"] for item in metrics], dtype=float)
    efficiencies = np.asarray([item["roi_efficiency"] for item in metrics], dtype=float)
    balances = np.asarray([item["component_energy_balance_error"] for item in metrics], dtype=float)
    score = correlations.min() - 0.25 * backgrounds.mean() + 0.20 * efficiencies.mean() - 0.10 * balances.mean()
    result = {
        "score": float(score),
        "worst_correlation": float(correlations.min()),
        "mean_correlation": float(correlations.mean()),
        "mean_background": float(backgrounds.mean()),
        "mean_roi_efficiency": float(efficiencies.mean()),
        "mean_balance_error": float(balances.mean()),
    }
    if target.target_type == "ring":
        ratios = [item["mean_center_to_ring_ratio"] for item in metrics]
        radius_errors = [item["mean_ring_radius_error_m"] for item in metrics]
        radius_m = float(target.metadata["ring_radius_m"])
        result["mean_center_to_ring_ratio"] = float(np.mean(ratios))
        result["worst_center_to_ring_ratio"] = float(np.max(ratios))
        result["mean_ring_radius_error_m"] = float(np.mean(radius_errors))
        bounded_ratio = min(result["worst_center_to_ring_ratio"], 2.0)
        result["ring_quality_score"] = float(
            result["score"] - 0.5 * bounded_ratio
            - 0.1 * result["mean_ring_radius_error_m"] / max(radius_m, 1e-12)
        )
        result["score"] = result["ring_quality_score"]
    return result


def evaluate_height(height: torch.Tensor, forward, target: TargetSpec, grid, wavelengths_m: tuple[float, ...]):
    outputs = forward(height)
    metrics = [
        evaluate_output(field, target, grid.base_x, grid.base_y, grid.pixel_pitch_m)
        for field in outputs
    ]
    return metrics, outputs


def save_target_plot(path: Path, target: TargetSpec, grid, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    extent = [
        float(grid.base_x.min().cpu() * 1e3),
        float(grid.base_x.max().cpu() * 1e3),
        float(grid.base_y.min().cpu() * 1e3),
        float(grid.base_y.max().cpu() * 1e3),
    ]
    axis.imshow(target.intensity.detach().cpu().numpy(), origin="lower", extent=extent, cmap="inferno")
    axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_output_plots(
    linear_path: Path,
    db_path: Path,
    outputs: list[torch.Tensor],
    wavelengths_nm: tuple[float, ...],
    grid,
    title: str,
) -> None:
    extent = [
        float(grid.base_x.min().cpu() * 1e3),
        float(grid.base_x.max().cpu() * 1e3),
        float(grid.base_y.min().cpu() * 1e3),
        float(grid.base_y.max().cpu() * 1e3),
    ]
    linear_figure, linear_axes = plt.subplots(
        1, len(outputs), figsize=(4 * len(outputs), 4), squeeze=False, constrained_layout=True
    )
    db_figure, db_axes = plt.subplots(
        1, len(outputs), figsize=(4 * len(outputs), 4), squeeze=False, constrained_layout=True
    )
    for index, field in enumerate(outputs):
        intensity = normalized_intensity(field).detach().cpu().numpy()
        linear_axes[0, index].imshow(intensity, origin="lower", extent=extent, cmap="inferno")
        db_axes[0, index].imshow(
            10.0 * np.log10(np.maximum(intensity, 1e-12) / max(float(intensity.max()), 1e-12)),
            origin="lower",
            extent=extent,
            cmap="inferno",
            vmin=-40,
            vmax=0,
        )
        linear_axes[0, index].set(title=f"{wavelengths_nm[index]:g} nm", xlabel="x (mm)", ylabel="y (mm)")
        db_axes[0, index].set(title=f"{wavelengths_nm[index]:g} nm", xlabel="x (mm)", ylabel="y (mm)")
    linear_figure.suptitle(f"{title} linear intensity")
    db_figure.suptitle(f"{title} fixed -40 dB")
    linear_figure.savefig(linear_path, dpi=170)
    db_figure.savefig(db_path, dpi=170)
    plt.close(linear_figure)
    plt.close(db_figure)


def radial_profile(
    intensity: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    center: tuple[float, float],
    bin_width_m: float,
    max_radius_m: float,
) -> tuple[np.ndarray, np.ndarray]:
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    indices = torch.floor(distance / bin_width_m).to(torch.long)
    bin_count = int(math.ceil(max_radius_m / bin_width_m))
    valid = indices < bin_count
    sums = torch.zeros(bin_count, device=intensity.device, dtype=intensity.dtype)
    counts = torch.zeros(bin_count, device=intensity.device, dtype=intensity.dtype)
    sums.scatter_add_(0, indices[valid], intensity[valid])
    counts.scatter_add_(0, indices[valid], torch.ones_like(intensity[valid]))
    profile = sums / (counts + 1e-12)
    radii = (torch.arange(bin_count, device=intensity.device) + 0.5) * bin_width_m
    return radii.detach().cpu().numpy(), profile.detach().cpu().numpy()


def save_radial_plot(
    path: Path,
    csv_path: Path,
    outputs: list[torch.Tensor],
    wavelengths_nm: tuple[float, ...],
    target: TargetSpec,
    grid,
    bin_width_m: float,
    max_radius_m: float,
) -> None:
    centers = tuple(tuple(float(value) for value in center) for center in target.metadata["centers_m"])
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    rows: list[dict[str, float]] = []
    for wavelength_index, field in enumerate(outputs):
        intensity = normalized_intensity(field)
        profiles = [
            radial_profile(intensity, grid.base_x, grid.base_y, center, bin_width_m, max_radius_m)
            for center in centers
        ]
        radii = profiles[0][0]
        average = np.mean([profile for _, profile in profiles], axis=0)
        axis.plot(radii * 1e6, average, label=f"{wavelengths_nm[wavelength_index]:g} nm")
        for radius, value in zip(radii, average):
            rows.append({
                "wavelength_nm": wavelengths_nm[wavelength_index],
                "radius_um": float(radius * 1e6),
                "mean_normalized_intensity": float(value),
            })
    axis.set(
        xlabel="radius from spot center (µm)",
        ylabel="mean normalized intensity",
        title=f"{target.metadata['name']} radial profile",
    )
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(path, dpi=170)
    plt.close(figure)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save_height_plot(path: Path, height: torch.Tensor, grid, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    extent = [
        float(grid.base_x.min().cpu() * 1e3),
        float(grid.base_x.max().cpu() * 1e3),
        float(grid.base_y.min().cpu() * 1e3),
        float(grid.base_y.max().cpu() * 1e3),
    ]
    image = height.detach().cpu().numpy() * 1e6
    plotted = axis.imshow(image, origin="lower", extent=extent, cmap="viridis")
    axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(plotted, ax=axis, label="height (µm)")
    figure.savefig(path, dpi=170)
    plt.close(figure)


def save_loss_plot(path: Path, result: OptimizationResult, title: str) -> None:
    figure, axis = plt.subplots(figsize=(6, 4), constrained_layout=True)
    axis.plot(result.losses)
    axis.set(xlabel="iteration", ylabel="loss", title=title)
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def padding_comparison(
    height: torch.Tensor,
    system: OpticalSystem,
    wavelengths_m: tuple[float, ...],
    material: CauchyMaterial,
    device: torch.device,
) -> dict[str, Any]:
    padded_grid = make_grid(system, device)
    unpadded_system = OpticalSystem(
        n_grid=system.n_grid,
        pixel_pitch_m=system.pixel_pitch_m,
        aperture_radius_m=system.aperture_radius_m,
        propagation_distance_m=system.propagation_distance_m,
        pad_factor=1,
    )
    unpadded_grid = make_grid(unpadded_system, device)
    rows: list[dict[str, float]] = []
    for wavelength_m in wavelengths_m:
        index = material.index(wavelength_m)
        padded = normalized_intensity(forward_field(height, padded_grid, system, wavelength_m, index))
        unpadded = normalized_intensity(
            forward_field(height, unpadded_grid, unpadded_system, wavelength_m, index)
        )
        centered_padded = padded - padded.mean()
        centered_unpadded = unpadded - unpadded.mean()
        correlation = torch.sum(centered_padded * centered_unpadded) / (
            torch.sqrt(torch.sum(centered_padded**2) * torch.sum(centered_unpadded**2)) + 1e-12
        )
        padded_peak = torch.argmax(padded)
        unpadded_peak = torch.argmax(unpadded)
        padded_row, padded_column = divmod(int(padded_peak.item()), system.n_grid)
        unpadded_row, unpadded_column = divmod(int(unpadded_peak.item()), system.n_grid)
        padded_x = padded_grid.base_x[padded_row, padded_column]
        padded_y = padded_grid.base_y[padded_row, padded_column]
        unpadded_x = unpadded_grid.base_x[unpadded_row, unpadded_column]
        unpadded_y = unpadded_grid.base_y[unpadded_row, unpadded_column]
        rows.append({
            "wavelength_nm": wavelength_m * 1e9,
            "padded_vs_unpadded_correlation": float(correlation.detach().cpu()),
            "padded_vs_unpadded_mse": float(torch.mean((padded - unpadded) ** 2).detach().cpu()),
            "peak_position_shift_m": float(
                torch.sqrt((padded_x - unpadded_x) ** 2 + (padded_y - unpadded_y) ** 2).detach().cpu()
            ),
        })
    return {
        "padded_energy_relative_error": energy_relative_error(system, device),
        "unpadded_energy_relative_error": energy_relative_error(unpadded_system, device),
        "rows": rows,
    }


def single_wave_candidates(args: argparse.Namespace) -> list[dict[str, float]]:
    if args.quick:
        return [{"learning_rate_um": 0.04, "background_weight": 0.25, "smoothness_weight": 1e-6}]
    return [
        {
            "learning_rate_um": learning_rate,
            "background_weight": background,
            "smoothness_weight": smoothness,
        }
        for learning_rate in args.learning_rates
        for background in args.background_weights
        for smoothness in args.smoothness_weights
    ]


def choose_base_configuration(
    args: argparse.Namespace,
    target: TargetSpec,
    forward,
    grid,
    aperture: torch.Tensor,
    height_max_m: float,
    seed: int,
) -> tuple[dict[str, float], OptimizationResult, dict[str, float]]:
    best: tuple[float, dict[str, float], OptimizationResult, dict[str, float]] | None = None
    single_wavelength_forward = lambda height: [forward(height)[0]]
    for candidate in single_wave_candidates(args):
        result = optimize_stage(
            "adam",
            {**candidate, "hole_weight": 0.0},
            target,
            single_wavelength_forward,
            height_max_m,
            grid.base_n,
            aperture,
            args.coarse_single_iterations,
            seed,
            None,
        )
        metrics, _ = evaluate_height(result.height_m, single_wavelength_forward, target, grid, (532e-9,))
        score = score_metrics(metrics, target)
        if best is None or score["score"] > best[0]:
            best = (score["score"], candidate, result, score)
    assert best is not None
    return best[1], best[2], best[3]


def choose_hole_weight(
    args: argparse.Namespace,
    target: TargetSpec,
    forward,
    grid,
    aperture: torch.Tensor,
    height_max_m: float,
    base_config: dict[str, float],
    initial_height: torch.Tensor,
    seed: int,
) -> tuple[dict[str, float], OptimizationResult, dict[str, float]]:
    if target.target_type != "ring":
        config = {**base_config, "hole_weight": 0.0}
        result = optimize_stage(
            "adam",
            config,
            target,
            forward,
            height_max_m,
            grid.base_n,
            aperture,
            args.coarse_joint_iterations,
            seed,
            initial_height,
        )
        metrics, _ = evaluate_height(result.height_m, forward, target, grid, (532e-9, 633e-9))
        return config, result, score_metrics(metrics, target)
    candidates = [0.5] if args.quick else args.hole_weights
    best: tuple[float, dict[str, float], OptimizationResult, dict[str, float]] | None = None
    for hole_weight in candidates:
        config = {**base_config, "hole_weight": hole_weight}
        result = optimize_stage(
            "adam",
            config,
            target,
            forward,
            height_max_m,
            grid.base_n,
            aperture,
            args.coarse_joint_iterations,
            seed,
            initial_height,
        )
        metrics, _ = evaluate_height(result.height_m, forward, target, grid, (532e-9, 633e-9))
        score = score_metrics(metrics, target)
        if best is None or score["score"] > best[0]:
            best = (score["score"], config, result, score)
    assert best is not None
    return best[1], best[2], best[3]


def choose_mraf_configuration(
    args: argparse.Namespace,
    target: TargetSpec,
    forward,
    grid,
    aperture: torch.Tensor,
    height_max_m: float,
    base_config: dict[str, float],
    initial_height: torch.Tensor,
    seed: int,
) -> tuple[dict[str, float], OptimizationResult, dict[str, float]]:
    candidates = [(.05, .05)] if args.quick else [
        (bias, mixing) for bias in args.mraf_biases for mixing in args.mraf_mixings
    ]
    best: tuple[float, dict[str, float], OptimizationResult, dict[str, float]] | None = None
    for bias, mixing in candidates:
        config = {**base_config, "mraf_bias": bias, "mraf_mixing": mixing}
        result = optimize_stage(
            "mraf",
            config,
            target,
            forward,
            height_max_m,
            grid.base_n,
            aperture,
            args.coarse_joint_iterations,
            seed,
            initial_height,
        )
        metrics, _ = evaluate_height(result.height_m, forward, target, grid, (532e-9, 633e-9))
        score = score_metrics(metrics, target)
        if best is None or score["score"] > best[0]:
            best = (score["score"], config, result, score)
    assert best is not None
    return best[1], best[2], best[3]


def main() -> None:
    args = parse_args()
    if len(args.wavelengths) != 2:
        raise ValueError("this focused experiment expects exactly two wavelengths")
    if args.quick:
        args.seeds = args.seeds[:1]
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    wavelengths_nm = tuple(float(value) for value in args.wavelengths)
    wavelengths_m = tuple(value * 1e-9 for value in wavelengths_nm)
    height_max_m = args.height_max_um * 1e-6
    material = CauchyMaterial()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []

    for target_name in args.targets:
        coarse_system = make_system(args, fine=False)
        fine_system = make_system(args, fine=True)
        coarse_grid = make_grid(coarse_system, device)
        fine_grid = make_grid(fine_system, device)
        coarse_sampling = validate_sampling(coarse_system, coarse_grid)
        fine_sampling = validate_sampling(fine_system, fine_grid)
        coarse_target = make_target(target_name, coarse_grid)
        fine_target = make_target(target_name, fine_grid)
        label = target_label(target_name, fine_target)
        target_dir = output_dir / label
        target_dir.mkdir(parents=True, exist_ok=True)
        (target_dir / "coarse_sampling.json").write_text(
            json.dumps(coarse_sampling, indent=2), encoding="utf-8"
        )
        (target_dir / "fine_sampling.json").write_text(
            json.dumps(fine_sampling, indent=2), encoding="utf-8"
        )
        save_target_plot(target_dir / "target.png", fine_target, fine_grid, label)
        save_target_plot(target_dir / "coarse_target.png", coarse_target, coarse_grid, f"{label} coarse")
        coarse_aperture = coarse_grid.doe_aperture[
            coarse_grid.crop_start:coarse_grid.crop_end,
            coarse_grid.crop_start:coarse_grid.crop_end,
        ]
        fine_aperture = fine_grid.doe_aperture[
            fine_grid.crop_start:fine_grid.crop_end,
            fine_grid.crop_start:fine_grid.crop_end,
        ]
        coarse_forward, _, _ = make_forward(coarse_grid, coarse_system, wavelengths_m)
        fine_forward, _, fine_indices = make_forward(fine_grid, fine_system, wavelengths_m)
        coarse_stage_dir = target_dir / "stages"
        coarse_stage_dir.mkdir(parents=True, exist_ok=True)
        base_config, single_result, single_score = choose_base_configuration(
            args,
            coarse_target,
            coarse_forward,
            coarse_grid,
            coarse_aperture,
            height_max_m,
            args.seeds[0],
        )
        np.save(coarse_stage_dir / "single_532_height.npy", single_result.height_m.detach().cpu().numpy())
        (coarse_stage_dir / "single_532_score.json").write_text(
            json.dumps(single_score, indent=2), encoding="utf-8"
        )
        for method in args.methods:
            method_dir = target_dir / method
            method_dir.mkdir(parents=True, exist_ok=True)
            for seed in args.seeds:
                started = time.perf_counter()
                seed_single = optimize_stage(
                    "adam",
                    {**base_config, "hole_weight": 0.0},
                    coarse_target,
                    lambda height: [coarse_forward(height)[0]],
                    height_max_m,
                    coarse_grid.base_n,
                    coarse_aperture,
                    args.coarse_single_iterations,
                    seed,
                    None,
                )
                joint_config, joint_result, joint_score = choose_hole_weight(
                    args,
                    coarse_target,
                    coarse_forward,
                    coarse_grid,
                    coarse_aperture,
                    height_max_m,
                    base_config,
                    seed_single.height_m,
                    seed,
                )
                fine_initial = resize_height(
                    joint_result.height_m,
                    fine_grid.base_n,
                    fine_aperture,
                    height_max_m,
                )
                np.save(method_dir / f"seed_{seed}_fine_initial_height.npy", fine_initial.detach().cpu().numpy())
                final_config = dict(joint_config)
                final_initial = fine_initial
                coarse_method_score = joint_score
                if method == "mraf":
                    final_config, mraf_result, coarse_method_score = choose_mraf_configuration(
                        args,
                        coarse_target,
                        coarse_forward,
                        coarse_grid,
                        coarse_aperture,
                        height_max_m,
                        joint_config,
                        joint_result.height_m,
                        seed,
                    )
                    final_initial = resize_height(
                        mraf_result.height_m,
                        fine_grid.base_n,
                        fine_aperture,
                        height_max_m,
                    )
                final_result = optimize_stage(
                    method,
                    final_config,
                    fine_target,
                    fine_forward,
                    height_max_m,
                    fine_grid.base_n,
                    fine_aperture,
                    args.fine_iterations,
                    seed,
                    final_initial,
                )
                metrics, outputs = evaluate_height(
                    final_result.height_m,
                    fine_forward,
                    fine_target,
                    fine_grid,
                    wavelengths_m,
                )
                final_score = score_metrics(metrics, fine_target)
                run_dir = method_dir / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "height.npy", final_result.height_m.detach().cpu().numpy())
                (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                (run_dir / "loss_history.json").write_text(json.dumps(final_result.losses, indent=2), encoding="utf-8")
                save_output_plots(
                    run_dir / "output_linear.png",
                    run_dir / "output_db.png",
                    outputs,
                    wavelengths_nm,
                    fine_grid,
                    f"{label} / {method}",
                )
                save_radial_plot(
                    run_dir / "radial_profile.png",
                    run_dir / "radial_profile.csv",
                    outputs,
                    wavelengths_nm,
                    fine_target,
                    fine_grid,
                    args.radial_bin_um * 1e-6,
                    args.radial_max_um * 1e-6,
                )
                save_height_plot(run_dir / "height.png", final_result.height_m, fine_grid, f"{label} / {method} height")
                save_loss_plot(run_dir / "loss.png", final_result, f"{label} / {method} loss")
                padding = padding_comparison(
                    final_result.height_m,
                    fine_system,
                    wavelengths_m,
                    material,
                    device,
                )
                payload = {
                    "target": label,
                    "method": method,
                    "seed": seed,
                    "config": final_config,
                    "coarse_base_config": base_config,
                    "coarse_joint_score": coarse_method_score,
                    "score": final_score,
                    "metrics": metrics,
                    "wavelengths_nm": list(wavelengths_nm),
                    "refractive_indices": list(fine_indices),
                    "material": material.metadata(wavelengths_m),
                    "coarse_sampling": coarse_sampling,
                    "fine_sampling": fine_sampling,
                    "padding_comparison": padding,
                    "elapsed_seconds": time.perf_counter() - started,
                    "height_file": str(run_dir / "height.npy"),
                }
                (run_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                summary_rows.append(payload)
                print(
                    f"[{label}] method={method} seed={seed} "
                    f"corr={final_score['worst_correlation']:.4f} "
                    f"bg={final_score['mean_background']:.4f} "
                    f"score={final_score['score']:.4f}"
                )

    (output_dir / "summary.json").write_text(json.dumps(summary_rows, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["target", "method", "seed", "score", "metrics", "config"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow({field: json.dumps(row[field]) if isinstance(row[field], (dict, list)) else row[field] for field in fields})
    print(f"saved {len(summary_rows)} focused runs to {output_dir}")


if __name__ == "__main__":
    main()
