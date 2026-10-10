"""GPU-batched RGB DOE study for chromatic image-plane shifts."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from forward_models import (
    OpticalSystem,
    angular_spectrum_transfer_batch,
    forward_rgb_fourier_lens,
    forward_rgb_single_plane,
    forward_rgb_two_planes,
    make_grid,
    validate_sampling,
)
from losses import LossSchedule, LossWeights
from materials import CauchyMaterial
from metrics import evaluate_output, evaluate_wavelength_consistency, normalized_intensity
from optimizers import AdamOptions, OptimizationResult, optimize_adam
from targets import (
    TargetSpec,
    cross_two_spots,
    ring_two_spots,
    soft_circle_two_spots,
    supersampled_edge_target,
)


WAVELENGTHS_M = (488e-9, 532e-9, 633e-9)
WAVELENGTH_LABELS = ("488nm", "532nm", "633nm")
ARCHITECTURES = ("single_doe", "doe_fourier_lens", "two_doe_planes")
TARGETS = ("circle", "cross", "ring50")
RGB_CHANNELS = (2, 1, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-grid", type=int, default=256)
    parser.add_argument("--coarse-pitch-um", type=float, default=10.0)
    parser.add_argument("--fine-grid", type=int, default=512)
    parser.add_argument("--fine-pitch-um", type=float, default=5.0)
    parser.add_argument("--pad-factor", type=int, default=2, choices=(1, 2, 4))
    parser.add_argument("--aperture-radius-um", type=float, default=1000.0)
    parser.add_argument("--distance-mm", type=float, default=50.0)
    parser.add_argument("--plane-separation-mm", type=float, default=25.0)
    parser.add_argument("--height-max-um", type=float, default=1.4)
    parser.add_argument("--coarse-iterations", type=int, default=120)
    parser.add_argument("--fine-iterations", type=int, default=400)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--search-seed", type=int, default=7)
    parser.add_argument("--sharpness-gains", type=float, nargs="+", default=[1.0, 1.6])
    parser.add_argument("--position-weights", type=float, nargs="+", default=[0.10, 0.50])
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--output-dir", type=Path, default=Path("results/chromatic_shift_multiplane"))
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def target_factory(name: str) -> Callable[[torch.Tensor, torch.Tensor], TargetSpec]:
    if name == "circle":
        return lambda x, y: soft_circle_two_spots(x, y, radius_m=40e-6, edge_width_m=15e-6)
    if name == "cross":
        return lambda x, y: cross_two_spots(x, y, arm_length_m=40e-6, arm_width_m=10e-6, edge_width_m=8e-6)
    if name == "ring50":
        return lambda x, y: ring_two_spots(x, y, radius_m=50e-6, edge_width_m=8e-6)
    raise ValueError(f"unknown target: {name}")


def make_system(n_grid: int, pitch_um: float, args: argparse.Namespace) -> OpticalSystem:
    return OpticalSystem(
        n_grid=n_grid,
        pixel_pitch_m=pitch_um * 1e-6,
        aperture_radius_m=args.aperture_radius_um * 1e-6,
        propagation_distance_m=args.distance_mm * 1e-3,
        lens_focal_length_m=args.distance_mm * 1e-3,
        lens_aperture_radius_m=args.aperture_radius_um * 1e-6,
        pad_factor=args.pad_factor,
    )


def aperture_on_base(grid) -> torch.Tensor:
    return grid.doe_aperture[grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end]


def make_forward(
    architecture: str,
    grid,
    system: OpticalSystem,
    wavelengths: torch.Tensor,
    refractive_indices: torch.Tensor,
    plane_separation_m: float,
):
    if architecture == "single_doe":
        transfer = angular_spectrum_transfer_batch(
            wavelengths, grid.fx, grid.fy, system.propagation_distance_m
        )
        return lambda height: list(
            forward_rgb_single_plane(height, grid, wavelengths, refractive_indices, transfer).unbind(0)
        )
    if architecture == "doe_fourier_lens":
        transfer = angular_spectrum_transfer_batch(
            wavelengths, grid.fx, grid.fy, system.lens_focal_length_m
        )
        return lambda height: list(
            forward_rgb_fourier_lens(
                height, grid, wavelengths, refractive_indices, transfer, system.lens_focal_length_m
            ).unbind(0)
        )
    if architecture == "two_doe_planes":
        remaining_distance = system.propagation_distance_m - plane_separation_m
        if plane_separation_m <= 0 or remaining_distance <= 0:
            raise ValueError("plane separation must be between zero and total propagation distance")
        first_transfer = angular_spectrum_transfer_batch(
            wavelengths, grid.fx, grid.fy, plane_separation_m
        )
        second_transfer = angular_spectrum_transfer_batch(
            wavelengths, grid.fx, grid.fy, remaining_distance
        )
        return lambda height: list(
            forward_rgb_two_planes(
                height, grid, wavelengths, refractive_indices, first_transfer, second_transfer
            ).unbind(0)
        )
    raise ValueError(f"unknown architecture: {architecture}")


def make_initial_height(
    architecture: str,
    grid,
    height_max_m: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor | None:
    if architecture != "two_doe_planes":
        return None
    generator = torch.Generator(device=device).manual_seed(seed)
    aperture = aperture_on_base(grid)
    return height_max_m * torch.rand(
        (2, grid.base_n, grid.base_n), generator=generator, device=device
    ) * aperture[None, :, :]


def make_loss_schedule(iterations: int) -> LossSchedule:
    return LossSchedule(
        shape_only_until=max(1, iterations // 8),
        background_ramp_end=max(2, iterations // 2),
        hole_ramp_start=max(1, iterations // 3),
        hole_ramp_end=max(2, iterations * 2 // 3),
    )


def optimize_stage(
    architecture: str,
    target: TargetSpec,
    forward,
    grid,
    height_max_m: float,
    iterations: int,
    seed: int,
    position_weight: float,
    initial_height: torch.Tensor | None,
    centroid_tolerance_um: float = 5.0,
) -> OptimizationResult:
    ring_weight = 0.75 if target.target_type == "ring" else 0.0
    weights = LossWeights(
        beta_background=0.10,
        gamma_efficiency=0.05,
        delta_balance=0.05,
        eta_smooth=1e-6,
        hole_weight=ring_weight,
        position_weight=position_weight,
        centroid_consistency_weight=1.0,
        centroid_tolerance_m=centroid_tolerance_um * 1e-6,
        edge_weight=0.35,
    )
    options = AdamOptions(
        iterations=iterations,
        learning_rate_um=0.04,
        gradient_clip=1.0,
        seed=seed,
        pixel_pitch_m=grid.pixel_pitch_m,
        log_every=max(1, iterations // 20),
    )
    return optimize_adam(
        forward,
        target,
        height_max_m,
        grid.base_n,
        aperture_on_base(grid),
        options,
        weights,
        make_loss_schedule(iterations),
        initial_height,
    )


def resize_height(height: torch.Tensor, grid, aperture: torch.Tensor, height_max_m: float) -> torch.Tensor:
    if height.ndim == 2:
        resized = functional.interpolate(
            height[None, None], size=(grid.base_n, grid.base_n), mode="bilinear", align_corners=True
        )[0, 0]
    else:
        resized = functional.interpolate(
            height[:, None], size=(grid.base_n, grid.base_n), mode="bilinear", align_corners=True
        )[:, 0]
    return resized.clamp(0.0, height_max_m) * aperture


def evaluate(height: torch.Tensor, forward, target: TargetSpec, grid) -> tuple[list[dict[str, Any]], list[torch.Tensor], dict[str, float], float]:
    outputs = forward(height)
    metrics = [
        evaluate_output(field, target, grid.base_x, grid.base_y, grid.pixel_pitch_m)
        for field in outputs
    ]
    consistency = evaluate_wavelength_consistency(metrics, target)
    correlation_worst = min(item["correlation"] for item in metrics)
    background_mean = float(np.mean([item["background_fraction"] for item in metrics]))
    efficiency_mean = float(np.mean([item["roi_efficiency"] for item in metrics]))
    drift_um = consistency["max_centroid_drift_m"] * 1e6
    score = (
        correlation_worst - 0.20 * background_mean + 0.15 * efficiency_mean
        - 0.10 * max(0.0, drift_um - 5.0) / 5.0
    )
    if target.target_type == "ring":
        score -= 0.10 * consistency["max_radius_drift_m"] / float(target.metadata["ring_radius_m"])
    return metrics, outputs, consistency, float(score)


def save_intensity(path: Path, intensity: torch.Tensor, grid, title: str, db: bool = False) -> None:
    array = intensity.detach().cpu().numpy()
    extent = [
        float(grid.base_x.min().cpu() * 1e3), float(grid.base_x.max().cpu() * 1e3),
        float(grid.base_y.min().cpu() * 1e3), float(grid.base_y.max().cpu() * 1e3),
    ]
    figure, axis = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    if db:
        array = 10.0 * np.log10(np.maximum(array / max(float(array.max()), 1e-20), 1e-4))
        image = axis.imshow(array, origin="lower", extent=extent, cmap="inferno", vmin=-40, vmax=0)
        figure.colorbar(image, ax=axis, label="normalized intensity (dB)")
    else:
        image = axis.imshow(array, origin="lower", extent=extent, cmap="inferno")
        figure.colorbar(image, ax=axis, label="normalized intensity")
    axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def rgb_pseudocolor(outputs: list[torch.Tensor]) -> np.ndarray:
    channels = []
    for index in RGB_CHANNELS:
        intensity = normalized_intensity(outputs[index]).detach().cpu().numpy()
        channels.append(intensity / max(float(intensity.max()), 1e-20))
    rgb = np.stack(channels, axis=-1)
    return np.clip(rgb, 0.0, 1.0) ** 0.7


def save_rgb_image(path: Path, outputs: list[torch.Tensor], title: str, grid) -> None:
    figure, axis = plt.subplots(figsize=(4.5, 4), constrained_layout=True)
    extent = [
        float(grid.base_x.min().cpu() * 1e3), float(grid.base_x.max().cpu() * 1e3),
        float(grid.base_y.min().cpu() * 1e3), float(grid.base_y.max().cpu() * 1e3),
    ]
    axis.imshow(rgb_pseudocolor(outputs), origin="lower", extent=extent)
    axis.set_xlim(-0.48, 0.48)
    axis.set_ylim(-0.20, 0.20)
    axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_run(
    directory: Path,
    target: TargetSpec,
    outputs: list[torch.Tensor],
    height: torch.Tensor,
    result: OptimizationResult,
    grid,
    metrics: list[dict[str, Any]],
    consistency: dict[str, float],
    config: dict[str, Any],
) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_intensity(directory / "target.png", target.intensity, grid, "processed target")
    for wavelength_label, field in zip(WAVELENGTH_LABELS, outputs):
        intensity = normalized_intensity(field)
        save_intensity(directory / f"output_{wavelength_label}_linear.png", intensity, grid, wavelength_label)
        save_intensity(directory / f"output_{wavelength_label}_db.png", intensity, grid, f"{wavelength_label}, fixed -40 dB", True)
    save_rgb_image(directory / "output_rgb_pseudocolor.png", outputs, "RGB overlay (per-channel normalized)", grid)
    np.save(directory / "height.npy", height.detach().cpu().numpy())
    (directory / "metrics.json").write_text(
        json.dumps({"metrics_by_wavelength": metrics, "wavelength_consistency": consistency}, indent=2),
        encoding="utf-8",
    )
    (directory / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
    (directory / "loss_history.json").write_text(
        json.dumps({"losses": result.losses, "components": result.components}, indent=2),
        encoding="utf-8",
    )
    if result.losses:
        figure, axis = plt.subplots(figsize=(6, 3.5), constrained_layout=True)
        axis.plot(np.linspace(1, config["fine_iterations"], len(result.losses)), result.losses)
        axis.set(xlabel="logged iteration", ylabel="loss", title="optimization history")
        axis.grid(True, alpha=0.25)
        figure.savefig(directory / "loss.png", dpi=160)
        plt.close(figure)


def make_overview(
    path: Path,
    target_images: dict[str, np.ndarray],
    best_images: dict[tuple[str, str], np.ndarray],
    targets: list[str],
    architectures: list[str],
    grid,
) -> None:
    figure, axes = plt.subplots(len(targets), len(architectures) + 1, figsize=(4 * (len(architectures) + 1), 4 * len(targets)), squeeze=False)
    extent = [
        float(grid.base_x.min().cpu() * 1e3), float(grid.base_x.max().cpu() * 1e3),
        float(grid.base_y.min().cpu() * 1e3), float(grid.base_y.max().cpu() * 1e3),
    ]
    for row, target_name in enumerate(targets):
        axes[row, 0].imshow(target_images[target_name], origin="lower", extent=extent, cmap="inferno")
        axes[row, 0].set_title(f"{target_name}: target")
        for column, architecture in enumerate(architectures, start=1):
            axes[row, column].imshow(best_images[(target_name, architecture)], origin="lower", extent=extent)
            axes[row, column].set_title(f"{architecture}: RGB overlay")
        for axis in axes[row]:
            axis.set_xlim(-0.48, 0.48)
            axis.set_ylim(-0.20, 0.20)
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("y (mm)")
    figure.suptitle("Target and RGB image-plane reconstructions", fontsize=14)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_shift_plot(path: Path, records: list[dict[str, Any]], targets: list[str], architectures: list[str]) -> None:
    figure, axis = plt.subplots(figsize=(10, 5), constrained_layout=True)
    x = np.arange(len(targets))
    width = 0.8 / max(1, len(architectures))
    for index, architecture in enumerate(architectures):
        values = []
        for target_name in targets:
            choices = [
                row["max_centroid_drift_um"] for row in records
                if row["target"] == target_name and row["architecture"] == architecture
            ]
            values.append(float(np.mean(choices)) if choices else 0.0)
        axis.bar(x - 0.4 + width / 2 + index * width, values, width, label=architecture)
    axis.axhline(5.0, color="red", linestyle="--", label="5 µm target")
    axis.set_xticks(x, targets)
    axis.set_ylabel("maximum RGB component-centroid drift (µm)")
    axis.set_title("Chromatic image-plane shift by optical architecture")
    axis.legend()
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_target_processing(path: Path, targets: list[str], gains: list[float], grid, device: torch.device) -> None:
    figure, axes = plt.subplots(len(targets), len(gains), figsize=(3.4 * len(gains), 3.2 * len(targets)), squeeze=False)
    extent = [
        float(grid.base_x.min().cpu() * 1e3), float(grid.base_x.max().cpu() * 1e3),
        float(grid.base_y.min().cpu() * 1e3), float(grid.base_y.max().cpu() * 1e3),
    ]
    for row, target_name in enumerate(targets):
        for column, gain in enumerate(gains):
            target = supersampled_edge_target(
                target_factory(target_name), grid.base_n, grid.pixel_pitch_m, device,
                sharpness_gain=gain, supersample_factor=4,
            )
            axes[row, column].imshow(target.intensity.detach().cpu().numpy(), origin="lower", extent=extent, cmap="inferno")
            axes[row, column].set_xlim(-0.48, 0.48)
            axes[row, column].set_ylim(-0.20, 0.20)
            axes[row, column].set_title(f"{target_name}; gain={gain:g}")
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    figure.suptitle("Supersampled and edge-sharpened target variants")
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.coarse_iterations < 1 or args.fine_iterations < 1:
        raise ValueError("iteration counts must be positive")
    if any(gain < 1.0 for gain in args.sharpness_gains):
        raise ValueError("sharpness gains must be >= 1")
    if any(weight < 0.0 for weight in args.position_weights):
        raise ValueError("position weights must be nonnegative")
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.backends.cudnn.benchmark = device.type == "cuda"
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    material = CauchyMaterial()
    refractive_indices = torch.tensor(material.indices(WAVELENGTHS_M), device=device, dtype=torch.float64)
    wavelengths = torch.tensor(WAVELENGTHS_M, device=device, dtype=torch.float64)
    height_max_m = args.height_max_um * 1e-6
    plane_separation_m = args.plane_separation_mm * 1e-3

    coarse_system = make_system(args.coarse_grid, args.coarse_pitch_um, args)
    fine_system = make_system(args.fine_grid, args.fine_pitch_um, args)
    coarse_grid = make_grid(coarse_system, device)
    fine_grid = make_grid(fine_system, device)
    coarse_sampling = validate_sampling(coarse_system, coarse_grid)
    fine_sampling = validate_sampling(fine_system, fine_grid)
    (output_dir / "sampling.json").write_text(
        json.dumps({"coarse": coarse_sampling, "fine": fine_sampling}, indent=2), encoding="utf-8"
    )
    save_target_processing(output_dir / "target_processing_comparison.png", args.targets, args.sharpness_gains, fine_grid, device)

    rows: list[dict[str, Any]] = []
    candidate_rows: list[dict[str, Any]] = []
    best_rgb_images: dict[tuple[str, str], np.ndarray] = {}
    best_visual_scores: dict[tuple[str, str], float] = {}
    target_images: dict[str, np.ndarray] = {}
    best_configurations: dict[tuple[str, str], dict[str, Any]] = {}
    best_coarse_heights: dict[tuple[str, str], torch.Tensor] = {}
    best_scores: dict[tuple[str, str], float] = {}

    start_all = time.perf_counter()
    for target_name in args.targets:
        factory = target_factory(target_name)
        for architecture in args.architectures:
            key = (target_name, architecture)
            best_score = -float("inf")
            for sharpness_gain in args.sharpness_gains:
                target = supersampled_edge_target(
                    factory, coarse_grid.base_n, coarse_grid.pixel_pitch_m, device,
                    sharpness_gain=sharpness_gain, supersample_factor=4,
                )
                for position_weight in args.position_weights:
                    forward = make_forward(
                        architecture, coarse_grid, coarse_system, wavelengths,
                        refractive_indices, plane_separation_m,
                    )
                    coarse_initial = make_initial_height(
                        architecture, coarse_grid, height_max_m, args.search_seed, device
                    )
                    result = optimize_stage(
                        architecture, target, forward, coarse_grid, height_max_m,
                        args.coarse_iterations, args.search_seed, position_weight, coarse_initial,
                    )
                    metrics, outputs, consistency, score = evaluate(result.height_m, forward, target, coarse_grid)
                    candidate = {
                        "target": target_name,
                        "architecture": architecture,
                        "sharpness_gain": float(sharpness_gain),
                        "position_weight": float(position_weight),
                        "search_seed": args.search_seed,
                        "score": score,
                        "max_centroid_drift_um": consistency["max_centroid_drift_m"] * 1e6,
                        "worst_correlation": min(item["correlation"] for item in metrics),
                        "mean_background_fraction": float(np.mean([item["background_fraction"] for item in metrics])),
                        "mean_roi_efficiency": float(np.mean([item["roi_efficiency"] for item in metrics])),
                    }
                    candidate_rows.append(candidate)
                    if score > best_score:
                        best_score = score
                        best_configurations[key] = {
                            "sharpness_gain": float(sharpness_gain),
                            "position_weight": float(position_weight),
                        }
                        best_coarse_heights[key] = result.height_m.detach().clone()
                        best_scores[key] = score
                    print(
                        f"[coarse] {target_name}/{architecture} gain={sharpness_gain:g} "
                        f"position={position_weight:g} score={score:.4f} "
                        f"drift={candidate['max_centroid_drift_um']:.2f}um",
                        flush=True,
                    )
                    del outputs, metrics, result, forward
                    if device.type == "cuda":
                        torch.cuda.empty_cache()

            config = best_configurations[key]
            fine_target = supersampled_edge_target(
                factory, fine_grid.base_n, fine_grid.pixel_pitch_m, device,
                sharpness_gain=config["sharpness_gain"], supersample_factor=4,
            )
            target_images[target_name] = fine_target.intensity.detach().cpu().numpy()
            (output_dir / target_name).mkdir(parents=True, exist_ok=True)
            (output_dir / target_name / "target_metadata.json").write_text(
                json.dumps(fine_target.metadata, indent=2), encoding="utf-8"
            )
            for seed in args.seeds:
                if seed == args.search_seed:
                    coarse_height = best_coarse_heights[key]
                else:
                    coarse_target = supersampled_edge_target(
                        factory, coarse_grid.base_n, coarse_grid.pixel_pitch_m, device,
                        sharpness_gain=config["sharpness_gain"], supersample_factor=4,
                    )
                    coarse_forward = make_forward(
                        architecture, coarse_grid, coarse_system, wavelengths,
                        refractive_indices, plane_separation_m,
                    )
                    coarse_initial = make_initial_height(
                        architecture, coarse_grid, height_max_m, seed, device
                    )
                    coarse_result = optimize_stage(
                        architecture, coarse_target, coarse_forward, coarse_grid, height_max_m,
                        args.coarse_iterations, seed, config["position_weight"], coarse_initial,
                    )
                    coarse_height = coarse_result.height_m
                    del coarse_result, coarse_forward, coarse_target
                fine_initial = resize_height(coarse_height, fine_grid, aperture_on_base(fine_grid), height_max_m)
                fine_forward = make_forward(
                    architecture, fine_grid, fine_system, wavelengths,
                    refractive_indices, plane_separation_m,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                run_start = time.perf_counter()
                fine_result = optimize_stage(
                    architecture, fine_target, fine_forward, fine_grid, height_max_m,
                    args.fine_iterations, seed, config["position_weight"], fine_initial,
                )
                if device.type == "cuda":
                    torch.cuda.synchronize(device)
                runtime_seconds = time.perf_counter() - run_start
                metrics, outputs, consistency, score = evaluate(
                    fine_result.height_m, fine_forward, fine_target, fine_grid
                )
                run_dir = output_dir / target_name / architecture / f"seed_{seed}"
                run_config = {
                    "target": target_name,
                    "architecture": architecture,
                    "sharpness_gain": config["sharpness_gain"],
                    "position_weight": config["position_weight"],
                    "edge_weight": 0.35,
                    "centroid_tolerance_um": 5.0,
                    "wavelengths_m": list(WAVELENGTHS_M),
                    "refractive_indices": list(material.indices(WAVELENGTHS_M)),
                    "material": material.metadata(WAVELENGTHS_M),
                    "coarse_iterations": args.coarse_iterations,
                    "fine_iterations": args.fine_iterations,
                    "seed": seed,
                    "device": str(device),
                    "operator": "batched angular-spectrum FFT; matrix-free linear propagation",
                    "plane_separation_m": plane_separation_m if architecture == "two_doe_planes" else None,
                    "runtime_seconds": runtime_seconds,
                    "score": score,
                }
                save_run(run_dir, fine_target, outputs, fine_result.height_m, fine_result, fine_grid, metrics, consistency, run_config)
                rows.append({
                    "target": target_name,
                    "architecture": architecture,
                    "seed": seed,
                    "score": score,
                    "sharpness_gain": config["sharpness_gain"],
                    "position_weight": config["position_weight"],
                    "worst_correlation": min(item["correlation"] for item in metrics),
                    "mean_background_fraction": float(np.mean([item["background_fraction"] for item in metrics])),
                    "mean_roi_efficiency": float(np.mean([item["roi_efficiency"] for item in metrics])),
                    "max_centroid_drift_um": consistency["max_centroid_drift_m"] * 1e6,
                    "max_ring_radius_drift_um": consistency["max_radius_drift_m"] * 1e6,
                    "runtime_seconds": runtime_seconds,
                    "run_dir": str(run_dir.relative_to(output_dir)),
                })
                if key not in best_visual_scores or score > best_visual_scores[key]:
                    best_rgb_images[(target_name, architecture)] = rgb_pseudocolor(outputs)
                    best_visual_scores[key] = score
                print(
                    f"[fine] {target_name}/{architecture}/seed={seed} score={score:.4f} "
                    f"RGB-shift={consistency['max_centroid_drift_m'] * 1e6:.2f}um "
                    f"time={runtime_seconds:.1f}s",
                    flush=True,
                )
                del outputs, metrics, fine_result, fine_initial, fine_forward
                if device.type == "cuda":
                    torch.cuda.empty_cache()

    if not rows:
        raise RuntimeError("no runs completed")
    make_overview(
        output_dir / "image_plane_overview.png", target_images, best_rgb_images,
        args.targets, args.architectures, fine_grid,
    )
    save_shift_plot(output_dir / "chromatic_shift_comparison.png", rows, args.targets, args.architectures)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with (output_dir / "candidate_sweep.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(candidate_rows[0]))
        writer.writeheader()
        writer.writerows(candidate_rows)
    summary = {
        "experiment": "RGB chromatic image-plane shift and multi-element comparison",
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "wavelengths_m": list(WAVELENGTHS_M),
        "material": material.metadata(WAVELENGTHS_M),
        "operator": "matrix-free batched angular-spectrum propagation using CUDA FFT",
        "optical_architectures": list(args.architectures),
        "targets": list(args.targets),
        "sampling": {"coarse": coarse_sampling, "fine": fine_sampling},
        "coarse_iterations": args.coarse_iterations,
        "fine_iterations": args.fine_iterations,
        "seeds": list(args.seeds),
        "search_parameters": {
            "edge_sharpness_gains": list(args.sharpness_gains),
            "cross_wavelength_position_weights": list(args.position_weights),
            "centroid_drift_tolerance_um": 5.0,
            "edge_gradient_weight": 0.35,
        },
        "elapsed_seconds": time.perf_counter() - start_all,
        "runs": rows,
        "coarse_candidates": candidate_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    (output_dir / "experiment_config.json").write_text(
        json.dumps(vars(args) | {"output_dir": str(args.output_dir)}, indent=2, default=str), encoding="utf-8"
    )
    print(f"Results saved to {output_dir}", flush=True)
    print(f"Image-plane overview: {output_dir / 'image_plane_overview.png'}", flush=True)


if __name__ == "__main__":
    main()
