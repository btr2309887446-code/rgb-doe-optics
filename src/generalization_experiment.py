"""Compare DOE optimizer generalization across several target patterns.

The script uses a small, explicit hyperparameter grid for coarse selection,
then repeats the selected configuration over the requested random seeds.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from forward_models import OpticalSystem, forward_field, make_grid, validate_sampling
from losses import LossSchedule, LossWeights
from materials import CauchyMaterial
from metrics import evaluate_output
from optimizers import (
    AdamOptions,
    optimize_adam,
    optimize_mraf,
)
from targets import TargetSpec, target_factories


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-grid", type=int, default=256)
    parser.add_argument("--pixel-pitch-um", type=float, default=10.0)
    parser.add_argument("--aperture-radius-um", type=float, default=1000.0)
    parser.add_argument("--pad-factor", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--propagation-distance-mm", type=float, default=50.0)
    parser.add_argument("--height-max-um", type=float, default=1.4)
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--validation-iterations", type=int)
    parser.add_argument("--wavelengths", type=float, nargs="+", default=[532, 633])
    parser.add_argument("--methods", nargs="+", choices=["adam", "mraf"], default=["adam", "mraf"])
    parser.add_argument("--targets", nargs="+", default=["all"])
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27, 37, 47])
    parser.add_argument("--coarse-seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/generalization"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--quick", action="store_true")
    return parser.parse_args()


def make_target(name: str, grid) -> TargetSpec:
    factory = target_factories()[name]
    return factory(grid.base_x, grid.base_y)


def make_system(args: argparse.Namespace) -> OpticalSystem:
    return OpticalSystem(
        n_grid=args.n_grid,
        pixel_pitch_m=args.pixel_pitch_um * 1e-6,
        aperture_radius_m=args.aperture_radius_um * 1e-6,
        pad_factor=args.pad_factor,
        propagation_distance_m=args.propagation_distance_mm * 1e-3,
    )


def target_names(requested: list[str]) -> list[str]:
    names = list(target_factories())
    if requested == ["all"] or "all" in requested:
        return names
    unknown = sorted(set(requested) - set(names))
    if unknown:
        raise ValueError(f"unknown target(s): {', '.join(unknown)}")
    return requested


def parameter_candidates(method: str, quick: bool) -> list[dict[str, float | int]]:
    if quick:
        if method == "adam":
            return [{"learning_rate_um": 0.04, "background_weight": 0.25, "smoothness_weight": 1e-5}]
        if method == "mraf":
            return [{"learning_rate_um": 0.04, "background_weight": 0.25, "smoothness_weight": 1e-5, "mraf_bias": 0.05}]
        return [{"levels": 32, "background_weight": 0.25, "smoothness_weight": 1e-5, "sweeps": 2}]
    if method == "adam":
        return [
            {"learning_rate_um": rate, "background_weight": background, "smoothness_weight": smooth}
            for rate in (0.01, 0.04, 0.10)
            for background in (0.05, 0.10, 0.25, 0.50)
            for smooth in (1e-6, 1e-5, 1e-4)
        ]
    if method == "mraf":
        return [
            {
                "learning_rate_um": rate,
                "background_weight": background,
                "smoothness_weight": smooth,
                "mraf_bias": bias,
            }
            for rate in (0.01, 0.04, 0.10)
            for background in (0.05, 0.10, 0.25, 0.50)
            for smooth in (1e-6, 1e-5, 1e-4)
            for bias in (0.0, 0.02, 0.05, 0.10)
        ]
    raise ValueError(f"unsupported optimizer method: {method}")


def score_metrics(metrics: list[dict[str, Any]]) -> dict[str, float]:
    correlations = np.asarray([item["correlation"] for item in metrics], dtype=float)
    backgrounds = np.asarray([item["background_fraction"] for item in metrics], dtype=float)
    efficiencies = np.asarray([item["roi_efficiency"] for item in metrics], dtype=float)
    balances = np.asarray([item["component_energy_balance_error"] for item in metrics], dtype=float)
    score = float(correlations.min() - 0.25 * backgrounds.mean() + 0.20 * efficiencies.mean() - 0.10 * balances.mean())
    return {
        "score": score,
        "worst_correlation": float(correlations.min()),
        "mean_correlation": float(correlations.mean()),
        "mean_background": float(backgrounds.mean()),
        "mean_roi_efficiency": float(efficiencies.mean()),
        "mean_balance_error": float(balances.mean()),
    }


def evaluate_height(
    height: torch.Tensor,
    forward,
    target: TargetSpec,
    grid,
    wavelengths_nm: list[float],
    material: CauchyMaterial,
) -> tuple[list[dict[str, Any]], list[torch.Tensor]]:
    outputs = forward(height)
    metrics = [
        evaluate_output(field, target, grid.base_x, grid.base_y, grid.pixel_pitch_m)
        for field in outputs
    ]
    return metrics, outputs


def optimize_one(
    method: str,
    config: dict[str, float | int],
    args: argparse.Namespace,
    target: TargetSpec,
    grid,
    base_aperture: torch.Tensor,
    forward,
    seed: int,
    iterations: int,
    initial_height: torch.Tensor | None = None,
):
    weights = LossWeights(
        beta_background=float(config.get("background_weight", 0.25)),
        gamma_efficiency=0.10,
        delta_balance=0.05,
        eta_smooth=float(config.get("smoothness_weight", 1e-5)),
    )
    schedule = LossSchedule(
        shape_only_until=min(100, max(0, iterations // 3)),
        background_ramp_end=min(300, max(1, iterations)),
    )
    height_max_m = args.height_max_um * 1e-6
    options = AdamOptions(
        iterations=iterations,
        learning_rate_um=float(config.get("learning_rate_um", 0.04)),
        seed=seed,
        mraf_background_bias=float(config["mraf_bias"]) if method == "mraf" else None,
    )
    return optimize_adam(
        forward, target, height_max_m, args.n_grid, base_aperture,
        options, weights, schedule, initial_height,
    )


def save_target_plot(path: Path, target: TargetSpec, grid, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    image = target.intensity.detach().cpu().numpy()
    extent = [float(grid.base_x.min() * 1e3), float(grid.base_x.max() * 1e3),
              float(grid.base_y.min() * 1e3), float(grid.base_y.max() * 1e3)]
    axis.imshow(image, origin="lower", extent=extent, cmap="inferno")
    axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_output_plot(path: Path, outputs: list[torch.Tensor], wavelengths_nm: list[float], grid, title: str) -> None:
    figure, axes = plt.subplots(1, len(outputs), figsize=(4 * len(outputs), 4), squeeze=False, constrained_layout=True)
    extent = [float(grid.base_x.min() * 1e3), float(grid.base_x.max() * 1e3),
              float(grid.base_y.min() * 1e3), float(grid.base_y.max() * 1e3)]
    for index, field in enumerate(outputs):
        image = (torch.abs(field) ** 2).detach().cpu().numpy()
        image /= image.sum() + 1e-12
        axes[0, index].imshow(10 * np.log10(np.maximum(image, 1e-12) / image.max()),
                              origin="lower", extent=extent, cmap="inferno", vmin=-40, vmax=0)
        axes[0, index].set(title=f"{wavelengths_nm[index]:g} nm", xlabel="x (mm)", ylabel="y (mm)")
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_height_plot(path: Path, height: torch.Tensor, grid, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    extent = [float(grid.base_x.min() * 1e3), float(grid.base_x.max() * 1e3),
              float(grid.base_y.min() * 1e3), float(grid.base_y.max() * 1e3)]
    image = height.detach().cpu().numpy() * 1e6
    axis.imshow(image, origin="lower", extent=extent, cmap="viridis")
    axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(axis.images[0], ax=axis, label="height (µm)")
    figure.savefig(path, dpi=160)
    plt.close(figure)


def aggregate_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["target"], row["method"]), []).append(row)
    result = []
    for (target, method), items in grouped.items():
        scores = np.asarray([item["score"]["score"] for item in items], dtype=float)
        worst = np.asarray([item["score"]["worst_correlation"] for item in items], dtype=float)
        background = np.asarray([item["score"]["mean_background"] for item in items], dtype=float)
        result.append({
            "target": target,
            "method": method,
            "n_runs": len(items),
            "score_mean": float(scores.mean()),
            "score_std": float(scores.std()),
            "score_best": float(scores.max()),
            "score_worst": float(scores.min()),
            "worst_correlation_mean": float(worst.mean()),
            "worst_correlation_std": float(worst.std()),
            "mean_background_mean": float(background.mean()),
            "mean_background_std": float(background.std()),
            "failure_rate": float(np.mean(worst < 0.50)),
            "config": items[0]["config"],
        })
    return result


def save_summary_plots(output_dir: Path, summary: list[dict[str, Any]]) -> None:
    targets = sorted({row["target"] for row in summary})
    methods = sorted({row["method"] for row in summary})
    values = np.asarray([[next(row for row in summary if row["target"] == target and row["method"] == method)["worst_correlation_mean"]
                          for method in methods] for target in targets])
    backgrounds = np.asarray([[next(row for row in summary if row["target"] == target and row["method"] == method)["mean_background_mean"]
                               for method in methods] for target in targets])
    for name, data, title in [
        ("generalization_heatmap.png", values, "Worst-wavelength correlation"),
        ("background_heatmap.png", backgrounds, "Mean background fraction"),
    ]:
        figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
        image = axis.imshow(data, cmap="viridis")
        axis.set_xticks(range(len(methods)), methods)
        axis.set_yticks(range(len(targets)), targets)
        axis.set_title(title)
        figure.colorbar(image, ax=axis)
        for row_index in range(data.shape[0]):
            for column_index in range(data.shape[1]):
                axis.text(column_index, row_index, f"{data[row_index, column_index]:.3f}",
                          ha="center", va="center", color="white")
        figure.savefig(output_dir / name, dpi=170)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    for method in methods:
        values_for_method = [row for row in summary if row["method"] == method]
        axis.scatter([row["mean_background_mean"] for row in values_for_method],
                     [row["worst_correlation_mean"] for row in values_for_method], label=method)
        for row in values_for_method:
            axis.annotate(row["target"], (row["mean_background_mean"], row["worst_correlation_mean"]), fontsize=7)
    axis.set(xlabel="mean background fraction", ylabel="worst-wavelength correlation", title="Generalization comparison")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.savefig(output_dir / "correlation_vs_background.png", dpi=170)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    names = target_names(args.targets)
    if args.quick:
        args.seeds = args.seeds[:1]
    if args.validation_iterations is None:
        args.validation_iterations = 300 if args.iterations >= 100 else args.iterations
    if args.quick:
        args.validation_iterations = min(args.validation_iterations, args.iterations)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    wavelengths_nm = [float(value) for value in args.wavelengths]
    wavelengths_m = tuple(value * 1e-9 for value in wavelengths_nm)
    system = make_system(args)
    grid = make_grid(system, device)
    sampling = validate_sampling(system, grid)
    base_aperture = grid.doe_aperture[grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end]
    material = CauchyMaterial()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, Any]] = []
    for target_name in names:
        target = make_target(target_name, grid)
        target_dir = output_dir / target_name
        target_dir.mkdir(parents=True, exist_ok=True)
        save_target_plot(target_dir / "target.png", target, grid, target_name)

        def forward(height: torch.Tensor) -> list[torch.Tensor]:
            return [
                forward_field(height, grid, system, wavelength, material.index(wavelength))
                for wavelength in wavelengths_m
            ]

        for method in args.methods:
            candidates = parameter_candidates(method, args.quick)
            best: tuple[float, dict[str, Any], torch.Tensor] | None = None
            print(f"[coarse] target={target_name} method={method} candidates={len(candidates)}")
            for candidate in candidates:
                started = time.perf_counter()
                result = optimize_one(method, candidate, args, target, grid, base_aperture, forward,
                                      args.coarse_seed, args.iterations)
                coarse_metrics, _ = evaluate_height(result.height_m, forward, target, grid, wavelengths_nm, material)
                score = score_metrics(coarse_metrics)
                if best is None or score["score"] > best[0]:
                    best = (score["score"], candidate, result.height_m)
                print(f"  {candidate} -> {score['score']:.4f} ({time.perf_counter() - started:.1f}s)")
            assert best is not None
            selected_config = best[1]
            for seed in args.seeds:
                started = time.perf_counter()
                result = optimize_one(
                    method, selected_config, args, target, grid, base_aperture, forward,
                    seed, args.validation_iterations,
                )
                metrics, outputs = evaluate_height(result.height_m, forward, target, grid, wavelengths_nm, material)
                score = score_metrics(metrics)
                run_dir = target_dir / method / f"seed_{seed}"
                run_dir.mkdir(parents=True, exist_ok=True)
                np.save(run_dir / "height.npy", result.height_m.detach().cpu().numpy())
                (run_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
                (run_dir / "loss_history.json").write_text(json.dumps(result.losses, indent=2), encoding="utf-8")
                save_output_plot(run_dir / "output.png", outputs, wavelengths_nm, grid, f"{target_name} / {method}")
                save_height_plot(run_dir / "height.png", result.height_m, grid, f"{target_name} / {method} height")
                row = {
                    "target": target_name,
                    "method": method,
                    "seed": seed,
                    "config": selected_config,
                    "score": score,
                    "metrics": metrics,
                    "elapsed_seconds": time.perf_counter() - started,
                    "target_metadata": target.metadata,
                    "material": material.metadata(wavelengths_m),
                    "sampling": sampling,
                }
                (run_dir / "result.json").write_text(json.dumps(row, indent=2), encoding="utf-8")
                all_rows.append(row)
                print(f"[validation] target={target_name} method={method} seed={seed} score={score['score']:.4f}")

    summary = aggregate_rows(all_rows)
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    with (output_dir / "method_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = list(summary[0].keys()) if summary else []
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(summary)
    save_summary_plots(output_dir, summary)
    print(f"saved {len(all_rows)} runs and {len(summary)} aggregate rows to {output_dir}")


if __name__ == "__main__":
    main()
