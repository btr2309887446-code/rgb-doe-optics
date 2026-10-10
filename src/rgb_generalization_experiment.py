"""RGB multi-pattern DOE generalization experiment."""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

from forward_models import OpticalSystem, forward_field, make_grid, validate_sampling
from losses import LossSchedule, LossWeights
from materials import CauchyMaterial
from metrics import evaluate_output, evaluate_wavelength_consistency, normalized_intensity
from optimizers import AdamOptions, OptimizationResult, optimize_adam
from targets import (
    TargetSpec,
    asymmetric_two_spots,
    cross_two_spots,
    ring_two_spots,
    soft_circle_two_spots,
    soft_square_two_spots,
    solid_triangle,
    triangle_three_spots,
)


TARGETS = (
    "circle",
    "ring50",
    "ring60",
    "square",
    "cross",
    "triangle_three",
    "solid_triangle",
    "asymmetric",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--coarse-grid", type=int, default=256)
    parser.add_argument("--coarse-pitch-um", type=float, default=10.0)
    parser.add_argument("--fine-grid", type=int, default=512)
    parser.add_argument("--fine-pitch-um", type=float, default=5.0)
    parser.add_argument("--pad-factor", type=int, default=2, choices=(1, 2, 4))
    parser.add_argument("--aperture-radius-um", type=float, default=1000.0)
    parser.add_argument("--propagation-distance-mm", type=float, default=50.0)
    parser.add_argument("--height-max-um", type=float, default=1.4)
    parser.add_argument("--wavelengths", type=float, nargs="+", default=[488, 532, 633])
    parser.add_argument("--coarse-iterations", type=int, default=300)
    parser.add_argument("--fine-iterations", type=int, default=1200)
    parser.add_argument("--position-weights", type=float, nargs="+", default=[0.0, 0.10, 0.25])
    parser.add_argument("--geometry-weights", type=float, nargs="+", default=[0.10])
    parser.add_argument("--radius-weights", type=float, nargs="+", default=[0.10])
    parser.add_argument("--hole-weights", type=float, nargs="+", default=[0.5, 1.0])
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--methods", nargs="+", choices=("adam", "mraf"), default=["adam", "mraf"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--output-dir", type=Path, default=Path("results/rgb_generalization"))
    parser.add_argument("--max-configs", type=int, default=2)
    parser.add_argument("--curriculum", action="store_true")
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def target_spec(name: str, grid) -> TargetSpec:
    x, y = grid.base_x, grid.base_y
    if name == "circle":
        return soft_circle_two_spots(x, y)
    if name == "ring50":
        return ring_two_spots(x, y, radius_m=50e-6, edge_width_m=8e-6)
    if name == "ring60":
        return ring_two_spots(x, y, radius_m=60e-6, edge_width_m=10e-6)
    if name == "square":
        return soft_square_two_spots(x, y)
    if name == "cross":
        return cross_two_spots(x, y)
    if name == "triangle_three":
        return triangle_three_spots(x, y)
    if name == "solid_triangle":
        return solid_triangle(x, y)
    if name == "asymmetric":
        return asymmetric_two_spots(x, y)
    raise ValueError(name)


def make_system(args: argparse.Namespace, fine: bool) -> OpticalSystem:
    return OpticalSystem(
        n_grid=args.fine_grid if fine else args.coarse_grid,
        pixel_pitch_m=(args.fine_pitch_um if fine else args.coarse_pitch_um) * 1e-6,
        aperture_radius_m=args.aperture_radius_um * 1e-6,
        propagation_distance_m=args.propagation_distance_mm * 1e-3,
        pad_factor=args.pad_factor,
    )


def schedule(iterations: int) -> LossSchedule:
    return LossSchedule(
        shape_only_until=min(100, max(1, iterations // 4)),
        background_ramp_end=min(300, max(1, iterations)),
        hole_ramp_start=min(300, max(1, iterations // 2)),
        hole_ramp_end=min(400, max(1, iterations)),
    )


def run_stage(
    method: str,
    target: TargetSpec,
    forward,
    height_max_m: float,
    grid,
    aperture: torch.Tensor,
    wavelengths: tuple[float, ...],
    config: dict[str, float],
    iterations: int,
    seed: int,
    initial: torch.Tensor | None,
    curriculum: bool,
) -> OptimizationResult:
    weights = LossWeights(
        beta_background=config["background_weight"],
        gamma_efficiency=0.10,
        delta_balance=0.05,
        eta_smooth=config["smoothness_weight"],
        hole_weight=config["hole_weight"],
        position_weight=config["position_weight"],
        centroid_consistency_weight=1.0,
        radius_weight=config["radius_weight"],
        geometry_weight=config["geometry_weight"],
    )

    def optimize(
        start: torch.Tensor | None,
        count: int,
        wavelength_weights: tuple[float, ...],
    ) -> OptimizationResult:
        options = AdamOptions(
            iterations=count,
            learning_rate_um=config["learning_rate_um"],
            seed=seed,
            wavelength_weights=wavelength_weights,
            mraf_background_bias=config.get("mraf_bias") if method == "mraf" else None,
            mraf_mixing=config.get("mraf_mixing", 0.05),
            pixel_pitch_m=grid.pixel_pitch_m,
        )
        return optimize_adam(
            forward,
            target,
            height_max_m,
            grid.base_n,
            aperture,
            options,
            weights,
            schedule(count),
            start,
        )

    if not curriculum or len(wavelengths) != 3 or iterations < 3:
        return optimize(initial, iterations, tuple(1.0 for _ in wavelengths))
    if iterations >= 600:
        first_count = 200
        second_count = 200
    else:
        first_count = max(1, iterations // 3)
        second_count = max(1, iterations // 3)
    third_count = iterations - first_count - second_count
    first = optimize(initial, first_count, (0.0, 1.0, 1.0))
    second = optimize(first.height_m, second_count, (0.5, 1.0, 1.0))
    third = optimize(second.height_m, third_count, (1.0, 1.0, 1.0))
    return OptimizationResult(
        third.height_m,
        first.losses + second.losses + third.losses,
        first.components + second.components + third.components,
        third.method,
    )


def resize_height(height: torch.Tensor, grid, aperture: torch.Tensor, height_max_m: float) -> torch.Tensor:
    resized = functional.interpolate(
        height[None, None], size=(grid.base_n, grid.base_n), mode="bilinear", align_corners=True
    )[0, 0]
    return resized.clamp(0.0, height_max_m) * aperture


def evaluate(height, forward, target, grid):
    outputs = forward(height)
    metrics = [
        evaluate_output(field, target, grid.base_x, grid.base_y, grid.pixel_pitch_m)
        for field in outputs
    ]
    consistency = evaluate_wavelength_consistency(metrics, target)
    correlations = [item["correlation"] for item in metrics]
    backgrounds = [item["background_fraction"] for item in metrics]
    efficiencies = [item["roi_efficiency"] for item in metrics]
    score = (
        min(correlations)
        - 0.20 * float(np.mean(backgrounds))
        + 0.15 * float(np.mean(efficiencies))
        - 0.20 * consistency["max_centroid_drift_m"] / 300e-6
    )
    if target.target_type == "ring":
        score -= 0.10 * consistency["max_radius_drift_m"] / float(target.metadata["ring_radius_m"])
    return metrics, outputs, consistency, float(score)


def save_image(path: Path, array: torch.Tensor, grid, title: str, cmap: str = "inferno") -> None:
    extent = [
        float(grid.base_x.min() * 1e3), float(grid.base_x.max() * 1e3),
        float(grid.base_y.min() * 1e3), float(grid.base_y.max() * 1e3),
    ]
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    axis.imshow(array.detach().cpu().numpy(), origin="lower", extent=extent, cmap=cmap)
    axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_outputs(
    linear_path: Path,
    db_path: Path,
    outputs: list[torch.Tensor],
    wavelengths_nm: tuple[float, ...],
    grid,
    title: str,
) -> None:
    extent = [
        float(grid.base_x.min() * 1e3), float(grid.base_x.max() * 1e3),
        float(grid.base_y.min() * 1e3), float(grid.base_y.max() * 1e3),
    ]
    linear_figure, linear_axes = plt.subplots(
        1, len(outputs), figsize=(4 * len(outputs), 4), squeeze=False
    )
    db_figure, db_axes = plt.subplots(
        1, len(outputs), figsize=(4 * len(outputs), 4), squeeze=False
    )
    for index, field in enumerate(outputs):
        intensity = normalized_intensity(field).detach().cpu().numpy()
        linear_axes[0, index].imshow(
            intensity, origin="lower", extent=extent, cmap="inferno"
        )
        linear_axes[0, index].set(
            title=f"{wavelengths_nm[index]:g} nm", xlabel="x (mm)", ylabel="y (mm)"
        )
        db_axes[0, index].imshow(
            10.0 * np.log10(np.maximum(intensity, 1e-12) / max(float(intensity.max()), 1e-12)),
            origin="lower", extent=extent, cmap="inferno", vmin=-40, vmax=0,
        )
        db_axes[0, index].set(
            title=f"{wavelengths_nm[index]:g} nm", xlabel="x (mm)", ylabel="y (mm)"
        )
    linear_figure.suptitle(f"{title} linear intensity")
    db_figure.suptitle(f"{title} fixed -40 dB")
    linear_figure.savefig(linear_path, dpi=160)
    db_figure.savefig(db_path, dpi=160)
    plt.close(linear_figure)
    plt.close(db_figure)


def save_masks(target: TargetSpec, grid, directory: Path) -> None:
    save_image(directory / "target.png", target.intensity, grid, "target")
    save_image(directory / "signal_mask.png", target.signal_mask, grid, "signal mask", "gray")
    save_image(directory / "transition_mask.png", target.transition_mask, grid, "transition mask", "gray")
    save_image(directory / "background_mask.png", target.background_mask, grid, "background mask", "gray")


def configs(args: argparse.Namespace, target_name: str) -> list[dict[str, float]]:
    hole_values = args.hole_weights if target_name.startswith("ring") else [0.0]
    radius_values = args.radius_weights if target_name.startswith("ring") else [0.0]
    result = []
    for position in args.position_weights:
        for geometry in args.geometry_weights:
            for hole in hole_values:
                for radius in radius_values:
                    result.append({
                        "learning_rate_um": 0.04,
                        "background_weight": 0.05,
                        "smoothness_weight": 1e-6,
                        "position_weight": position,
                        "geometry_weight": geometry,
                        "hole_weight": hole,
                        "radius_weight": radius,
                        "mraf_bias": 0.05,
                        "mraf_mixing": 0.05,
                    })
    return result


def main() -> None:
    args = parse_args()
    if len(args.wavelengths) != 3:
        raise ValueError("RGB experiment requires exactly three wavelengths")
    if args.quick:
        args.targets = args.targets[:2]
        args.methods = args.methods[:1]
        args.seeds = args.seeds[:1]
        args.coarse_iterations = min(args.coarse_iterations, 3)
        args.fine_iterations = min(args.fine_iterations, 3)
        args.position_weights = args.position_weights[:1]
        args.max_configs = 1
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    wavelengths_nm = tuple(float(value) for value in args.wavelengths)
    wavelengths_m = tuple(value * 1e-9 for value in wavelengths_nm)
    material = CauchyMaterial()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for target_name in args.targets:
        coarse_system = make_system(args, False)
        fine_system = make_system(args, True)
        coarse_grid = make_grid(coarse_system, device)
        fine_grid = make_grid(fine_system, device)
        coarse_sampling = validate_sampling(coarse_system, coarse_grid)
        fine_sampling = validate_sampling(fine_system, fine_grid)
        coarse_target = target_spec(target_name, coarse_grid)
        fine_target = target_spec(target_name, fine_grid)
        target_dir = output_dir / target_name
        target_dir.mkdir(parents=True, exist_ok=True)
        save_masks(fine_target, fine_grid, target_dir)
        (target_dir / "target_metadata.json").write_text(
            json.dumps(fine_target.metadata, indent=2), encoding="utf-8"
        )
        coarse_aperture = coarse_grid.doe_aperture[
            coarse_grid.crop_start:coarse_grid.crop_end,
            coarse_grid.crop_start:coarse_grid.crop_end,
        ]
        fine_aperture = fine_grid.doe_aperture[
            fine_grid.crop_start:fine_grid.crop_end,
            fine_grid.crop_start:fine_grid.crop_end,
        ]
        material_indices = material.indices(wavelengths_m)

        def coarse_forward(height):
            return [
                forward_field(height, coarse_grid, coarse_system, wave, index)
                for wave, index in zip(wavelengths_m, material_indices)
            ]

        def fine_forward(height):
            return [
                forward_field(height, fine_grid, fine_system, wave, index)
                for wave, index in zip(wavelengths_m, material_indices)
            ]

        candidates = configs(args, target_name)
        coarse_scores = []
        for index, config in enumerate(candidates):
            result = run_stage(
                "adam", coarse_target, coarse_forward, args.height_max_um * 1e-6,
                coarse_grid, coarse_aperture, wavelengths_m, config,
                args.coarse_iterations, args.seeds[0], None, args.curriculum,
            )
            metrics, _, consistency, score = evaluate(
                result.height_m, coarse_forward, coarse_target, coarse_grid
            )
            coarse_scores.append({
                "score": score,
                "index": index,
                "height": result.height_m.detach(),
                "drift": consistency["max_centroid_drift_m"],
                "worst_correlation": min(item["correlation"] for item in metrics),
                "background": float(np.mean([item["background_fraction"] for item in metrics])),
                "efficiency": float(np.mean([item["roi_efficiency"] for item in metrics])),
            })
        best_score = max(coarse_scores, key=lambda item: item["score"])
        selected = [best_score]
        if args.max_configs > 1 and len(coarse_scores) > 1:
            best_drift = min(coarse_scores, key=lambda item: item["drift"])
            if best_drift["index"] != best_score["index"]:
                selected.append(best_drift)
            if len(selected) < args.max_configs:
                next_score = max(
                    (
                        item
                        for item in coarse_scores
                        if item["index"] not in {chosen["index"] for chosen in selected}
                    ),
                    key=lambda item: item["score"],
                )
                selected.append(next_score)
        selected = selected[: max(1, args.max_configs)]

        for method in args.methods:
            for selected_config in selected:
                config_index = selected_config["index"]
                config = candidates[config_index]
                for seed in args.seeds:
                    started = time.perf_counter()
                    coarse_result = run_stage(
                        "adam", coarse_target, coarse_forward, args.height_max_um * 1e-6,
                        coarse_grid, coarse_aperture, wavelengths_m, config,
                        args.coarse_iterations, seed, None, args.curriculum,
                    )
                    initial = resize_height(
                        coarse_result.height_m, fine_grid, fine_aperture, args.height_max_um * 1e-6
                    )
                    final = run_stage(
                        method, fine_target, fine_forward, args.height_max_um * 1e-6,
                        fine_grid, fine_aperture, wavelengths_m, config,
                        args.fine_iterations, seed, initial, args.curriculum,
                    )
                    metrics, outputs, consistency, score = evaluate(
                        final.height_m, fine_forward, fine_target, fine_grid
                    )
                    run_dir = target_dir / method / f"config_{config_index}" / f"seed_{seed}"
                    run_dir.mkdir(parents=True, exist_ok=True)
                    np.save(run_dir / "height.npy", final.height_m.cpu().numpy())
                    (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                    (run_dir / "loss_history.json").write_text(json.dumps(final.losses), encoding="utf-8")
                    (run_dir / "config.json").write_text(json.dumps(config, indent=2), encoding="utf-8")
                    save_outputs(
                        run_dir / "output_linear.png",
                        run_dir / "output_db.png",
                        outputs,
                        wavelengths_nm,
                        fine_grid,
                        f"{target_name}/{method}",
                    )
                    save_image(run_dir / "height.png", final.height_m * 1e6, fine_grid, "DOE height", "viridis")
                    figure, axis = plt.subplots(figsize=(6, 4))
                    axis.plot(final.losses)
                    axis.set(xlabel="iteration", ylabel="loss", title=f"{target_name}/{method}")
                    figure.savefig(run_dir / "loss.png", dpi=160)
                    plt.close(figure)
                    payload = {
                        "target": target_name,
                        "method": method,
                        "seed": seed,
                        "config_index": config_index,
                        "config": config,
                        "score": score,
                        "metrics": metrics,
                        "wavelength_consistency": consistency,
                        "wavelengths_nm": wavelengths_nm,
                        "refractive_indices": material_indices,
                        "material": material.metadata(wavelengths_m),
                        "coarse_sampling": coarse_sampling,
                        "fine_sampling": fine_sampling,
                        "iterations": args.fine_iterations,
                        "elapsed_seconds": time.perf_counter() - started,
                    }
                    (run_dir / "result.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                    rows.append(payload)
                    print(
                        f"[{target_name}] {method} config={config_index} seed={seed} "
                        f"score={score:.4f} corr={min(item['correlation'] for item in metrics):.4f} "
                        f"drift={consistency['max_centroid_drift_m'] * 1e6:.2f}um"
                    )

    (output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["target", "method", "seed", "config_index", "score", "wavelength_consistency"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({
                field: json.dumps(row[field]) if isinstance(row[field], (dict, list)) else row[field]
                for field in fields
            })
    (output_dir / "experiment_config.json").write_text(
        json.dumps(vars(args), indent=2, default=str), encoding="utf-8"
    )
    print(f"saved {len(rows)} RGB runs to {output_dir}")


if __name__ == "__main__":
    main()
