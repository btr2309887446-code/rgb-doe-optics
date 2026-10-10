"""RGB cross-wavelength decolorization loss ablation experiment."""
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

from forward_models import (
    OpticalSystem,
    angular_spectrum_transfer_batch,
    forward_rgb_single_plane,
    forward_rgb_two_planes,
    make_grid,
    validate_sampling,
)
from losses import LossSchedule, LossWeights
from materials import CauchyMaterial
from metrics import evaluate_output
from optimizers import AdamOptions, OptimizationResult, optimize_adam
from targets import TargetSpec, cross_two_spots, ring_two_spots, soft_circle_two_spots, supersampled_edge_target


WAVELENGTHS_M = (488e-9, 532e-9, 633e-9)
WAVELENGTH_NM = (488, 532, 633)
ARCHITECTURES = ("single_doe", "two_doe_planes")
TARGETS = ("circle", "cross", "ring50")
VARIANTS = ("baseline", "pixel", "edge", "background", "energy", "component", "full")
INPUT_POWERS = (1.0, 1.0, 1.0)
RGB_CHANNEL_INDICES = (2, 1, 0)
TARGET_CONFIG = {
    "circle": {"sharpness_gain": 1.0, "position_weight": 0.10},
    "cross": {"sharpness_gain": 1.6, "position_weight": 0.50},
    "ring50": {"sharpness_gain": 1.0, "position_weight": 0.10},
}
VARIANT_RGB_WEIGHTS = {
    "baseline": {},
    "pixel": {"rgb_pixel_weight": 0.50},
    "edge": {"rgb_edge_weight": 0.15},
    "background": {"rgb_background_weight": 0.10},
    "energy": {"rgb_energy_weight": 0.25},
    "component": {"rgb_component_weight": 0.10},
    "full": {
        "rgb_pixel_weight": 0.50,
        "rgb_edge_weight": 0.15,
        "rgb_background_weight": 0.10,
        "rgb_energy_weight": 0.25,
        "rgb_component_weight": 0.10,
    },
}


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
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--device", choices=("auto", "cuda", "cpu"), default="auto")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rgb_decolorization"))
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "cpu":
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device("cuda" if requested == "cuda" or torch.cuda.is_available() else "cpu")


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


def make_target(name: str, grid, device: torch.device) -> TargetSpec:
    if name == "circle":
        factory = lambda x, y: soft_circle_two_spots(x, y, radius_m=40e-6, edge_width_m=15e-6)
    elif name == "cross":
        factory = lambda x, y: cross_two_spots(x, y, arm_length_m=40e-6, arm_width_m=10e-6, edge_width_m=8e-6)
    elif name == "ring50":
        factory = lambda x, y: ring_two_spots(x, y, radius_m=50e-6, edge_width_m=8e-6)
    else:
        raise ValueError(f"unknown target: {name}")
    return supersampled_edge_target(
        factory,
        grid.base_n,
        grid.pixel_pitch_m,
        device,
        sharpness_gain=float(TARGET_CONFIG[name]["sharpness_gain"]),
        supersample_factor=4,
    )


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
    if architecture == "two_doe_planes":
        remaining_distance = system.propagation_distance_m - plane_separation_m
        if plane_separation_m <= 0 or remaining_distance <= 0:
            raise ValueError("plane separation must be between zero and total distance")
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
    raise ValueError(f"unsupported architecture: {architecture}")


def make_initial_height(
    architecture: str,
    grid,
    height_max_m: float,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    aperture = aperture_on_base(grid)
    shape = (2, grid.base_n, grid.base_n) if architecture == "two_doe_planes" else (grid.base_n, grid.base_n)
    height = height_max_m * torch.rand(shape, generator=generator, device=device)
    return height * (aperture[None, :, :] if height.ndim == 3 else aperture)


def resize_height(height: torch.Tensor, grid, aperture: torch.Tensor, height_max_m: float) -> torch.Tensor:
    if height.ndim == 2:
        resized = functional.interpolate(
            height[None, None], size=(grid.base_n, grid.base_n), mode="bilinear", align_corners=True
        )[0, 0]
    else:
        resized = functional.interpolate(
            height[:, None], size=(grid.base_n, grid.base_n), mode="bilinear", align_corners=True
        )[:, 0]
    return resized.clamp(0.0, height_max_m) * (aperture[None, :, :] if resized.ndim == 3 else aperture)


def make_schedule(iterations: int) -> LossSchedule:
    return LossSchedule(
        shape_only_until=max(1, iterations // 8),
        background_ramp_end=max(2, iterations // 2),
        hole_ramp_start=max(1, iterations // 3),
        hole_ramp_end=max(2, iterations * 2 // 3),
        rgb_ramp_start=max(1, iterations // 4),
        rgb_ramp_end=max(2, iterations * 3 // 4),
    )


def make_weights(target_name: str, variant: str) -> LossWeights:
    if variant not in VARIANT_RGB_WEIGHTS:
        raise ValueError(f"unknown loss variant: {variant}")
    ring_weight = 0.75 if target_name == "ring50" else 0.0
    return LossWeights(
        alpha_shape=1.0,
        beta_background=0.10,
        gamma_efficiency=0.05,
        delta_balance=0.05,
        eta_smooth=1e-6,
        hole_weight=ring_weight,
        position_weight=float(TARGET_CONFIG[target_name]["position_weight"]),
        centroid_consistency_weight=1.0,
        centroid_tolerance_m=5e-6,
        edge_weight=0.35,
        **VARIANT_RGB_WEIGHTS[variant],
    )


def make_options(iterations: int, seed: int, grid) -> AdamOptions:
    return AdamOptions(
        iterations=iterations,
        learning_rate_um=0.04,
        gradient_clip=1.0,
        cosine_decay=True,
        seed=seed,
        wavelength_weights=(1.0, 1.0, 1.0),
        pixel_pitch_m=grid.pixel_pitch_m,
        log_every=max(1, iterations // 20),
        input_powers=INPUT_POWERS,
    )


def optimize_two_stage(
    architecture: str,
    target_name: str,
    coarse_target: TargetSpec,
    fine_target: TargetSpec,
    coarse_forward,
    fine_forward,
    coarse_grid,
    fine_grid,
    height_max_m: float,
    coarse_iterations: int,
    fine_iterations: int,
    seed: int,
    variant: str,
    coarse_initial: torch.Tensor,
) -> tuple[OptimizationResult, OptimizationResult]:
    weights = make_weights(target_name, variant)
    coarse_result = optimize_adam(
        coarse_forward,
        coarse_target,
        height_max_m,
        coarse_grid.base_n,
        aperture_on_base(coarse_grid),
        make_options(coarse_iterations, seed, coarse_grid),
        weights,
        make_schedule(coarse_iterations),
        coarse_initial,
    )
    fine_initial = resize_height(
        coarse_result.height_m,
        fine_grid,
        aperture_on_base(fine_grid),
        height_max_m,
    )
    fine_result = optimize_adam(
        fine_forward,
        fine_target,
        height_max_m,
        fine_grid.base_n,
        aperture_on_base(fine_grid),
        make_options(fine_iterations, seed, fine_grid),
        weights,
        make_schedule(fine_iterations),
        fine_initial,
    )
    return coarse_result, fine_result


def json_default(value: Any) -> Any:
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    raise TypeError(f"cannot serialize {type(value)!r}")


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, default=json_default), encoding="utf-8")


def pearson(first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        first = first[mask]
        second = second[mask]
    first = np.asarray(first, dtype=np.float64).reshape(-1)
    second = np.asarray(second, dtype=np.float64).reshape(-1)
    if first.size < 2:
        return 0.0
    first = first - first.mean()
    second = second - second.mean()
    denominator = np.linalg.norm(first) * np.linalg.norm(second)
    if denominator <= 1e-30:
        return 1.0 if np.allclose(first, second) else 0.0
    return float(np.dot(first, second) / denominator)


def masked_mse(first: np.ndarray, second: np.ndarray, mask: np.ndarray | None = None) -> float:
    if mask is not None:
        first = first[mask]
        second = second[mask]
    if first.size == 0:
        return 0.0
    return float(np.mean((np.asarray(first, dtype=np.float64) - np.asarray(second, dtype=np.float64)) ** 2))


def make_extent(grid) -> list[float]:
    return [
        float(grid.base_x.min().detach().cpu() * 1e3),
        float(grid.base_x.max().detach().cpu() * 1e3),
        float(grid.base_y.min().detach().cpu() * 1e3),
        float(grid.base_y.max().detach().cpu() * 1e3),
    ]


def save_scalar_image(
    path: Path,
    array: np.ndarray,
    extent: list[float],
    title: str,
    cmap: str = "inferno",
    vmin: float | None = None,
    vmax: float | None = None,
    label: str = "intensity",
) -> None:
    figure, axis = plt.subplots(figsize=(4.5, 4.0), constrained_layout=True)
    image = axis.imshow(array, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(image, ax=axis, label=label)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_rgb_image(path: Path, rgb: np.ndarray, title: str) -> None:
    figure, axis = plt.subplots(figsize=(5.0, 4.0), constrained_layout=True)
    axis.imshow(np.clip(rgb, 0.0, 1.0), origin="lower")
    axis.set_title(title)
    axis.set_xticks([])
    axis.set_yticks([])
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_loss_plot(path: Path, losses: list[float], components: list[dict[str, float]]) -> None:
    figure, axis = plt.subplots(figsize=(7.0, 4.0), constrained_layout=True)
    if losses:
        axis.plot(np.arange(len(losses)), losses, label="total")
    for key in ("shape_0", "background_0", "rgb_pixel", "rgb_edge", "rgb_energy", "rgb_component"):
        values = [item[key] for item in components if key in item]
        if values:
            axis.plot(np.linspace(0, max(1, len(losses) - 1), len(values)), values, label=key)
    axis.set(xlabel="logged optimization step", ylabel="loss", title="Optimization loss history")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def evaluate_outputs(
    outputs: list[torch.Tensor],
    target: TargetSpec,
    grid,
    aperture: torch.Tensor,
    input_powers: Iterable[float],
) -> dict[str, Any]:
    powers = np.asarray(tuple(float(value) for value in input_powers), dtype=np.float64)
    if len(outputs) != len(powers):
        raise ValueError("input_powers must match output fields")
    raw_tensors = [field.detach().abs().square() for field in outputs]
    raw_arrays = [value.cpu().numpy().astype(np.float64) for value in raw_tensors]
    signal_mask = target.signal_mask.detach().cpu().numpy() > 0.5
    transition_mask = target.transition_mask.detach().cpu().numpy() > 0.05
    background_mask = target.background_mask.detach().cpu().numpy() > 0.5
    input_energy = float(aperture.detach().sum().cpu())
    common_power_arrays = [array / max(power, 1e-30) for array, power in zip(raw_arrays, powers)]
    common_vmax = max(float(array.max()) for array in common_power_arrays)
    common_arrays = [array / max(common_vmax, 1e-30) for array in common_power_arrays]
    shape_arrays = [array / max(float(array.max()), 1e-30) for array in raw_arrays]
    metrics_by_wavelength: list[dict[str, Any]] = []
    component_shares: list[list[float]] = []
    for index, (field, raw, common) in enumerate(zip(outputs, raw_arrays, common_power_arrays)):
        normalized_metrics = evaluate_output(
            field, target, grid.base_x, grid.base_y, grid.pixel_pitch_m
        )
        roi_energy = float((raw * signal_mask).sum())
        background_energy = float((raw * background_mask).sum())
        output_energy = float(raw.sum())
        components = [
            float((raw * mask.detach().cpu().numpy()).sum())
            for mask in target.component_masks
        ]
        share_total = sum(components) + 1e-30
        shares = [value / share_total for value in components]
        component_shares.append(shares)
        metrics_by_wavelength.append({
            "wavelength_nm": WAVELENGTH_NM[index],
            **normalized_metrics,
            "input_energy": input_energy,
            "output_energy": output_energy,
            "output_to_input_energy_ratio": output_energy / max(input_energy, 1e-30),
            "roi_energy": roi_energy,
            "background_energy": background_energy,
            "roi_absolute_efficiency": roi_energy / max(input_energy * powers[index], 1e-30),
            "background_absolute_fraction": background_energy / max(input_energy * powers[index], 1e-30),
            "power_normalized_output_energy": float(common.sum()),
            "power_normalized_peak": float(common.max()),
            "common_scale_channel_max": float(common.max() / max(common_vmax, 1e-30)),
            "component_absolute_energies": components,
            "component_energy_shares": shares,
        })
    pair_rows: list[dict[str, Any]] = []
    for first_index in range(len(common_arrays)):
        for second_index in range(first_index + 1, len(common_arrays)):
            pair_rows.append({
                "first_wavelength_nm": WAVELENGTH_NM[first_index],
                "second_wavelength_nm": WAVELENGTH_NM[second_index],
                "common_scale_roi_mse": masked_mse(common_arrays[first_index], common_arrays[second_index], signal_mask),
                "common_scale_roi_pearson": pearson(common_arrays[first_index], common_arrays[second_index], signal_mask),
                "common_scale_background_mse": masked_mse(common_arrays[first_index], common_arrays[second_index], background_mask),
                "shape_roi_mse": masked_mse(shape_arrays[first_index], shape_arrays[second_index], signal_mask),
                "shape_roi_pearson": pearson(shape_arrays[first_index], shape_arrays[second_index], signal_mask),
                "edge_mse": masked_mse(common_arrays[first_index], common_arrays[second_index], transition_mask),
            })
    roi_efficiencies = np.asarray([row["roi_absolute_efficiency"] for row in metrics_by_wavelength], dtype=np.float64)
    component_share_array = np.asarray(component_shares, dtype=np.float64)
    centroid_drifts: list[float] = []
    for component_index in range(len(target.component_masks)):
        centroids = [np.asarray(row["component_centroids_m"][component_index], dtype=np.float64) for row in metrics_by_wavelength]
        for first_index in range(len(centroids)):
            for second_index in range(first_index + 1, len(centroids)):
                centroid_drifts.append(float(np.linalg.norm(centroids[first_index] - centroids[second_index])))
    radius_values: list[float] = []
    for row in metrics_by_wavelength:
        radius_values.extend(float(item["ring_peak_radius_m"]) for item in row.get("ring_components", []))
    summary = {
        "common_scale_vmax": common_vmax,
        "worst_correlation": min(row["correlation"] for row in metrics_by_wavelength),
        "mean_background_fraction": float(np.mean([row["background_fraction"] for row in metrics_by_wavelength])),
        "mean_roi_efficiency": float(np.mean([row["roi_efficiency"] for row in metrics_by_wavelength])),
        "mean_roi_absolute_efficiency": float(np.mean(roi_efficiencies)),
        "rgb_roi_energy_mean": float(np.mean(roi_efficiencies)),
        "rgb_roi_energy_std": float(np.std(roi_efficiencies)),
        "rgb_roi_energy_cv": float(np.std(roi_efficiencies) / max(np.mean(roi_efficiencies), 1e-30)),
        "rgb_max_min_energy_ratio": float(np.max(roi_efficiencies) / max(np.min(roi_efficiencies), 1e-30)),
        "mean_common_scale_roi_pearson": float(np.mean([row["common_scale_roi_pearson"] for row in pair_rows])),
        "mean_common_scale_roi_mse": float(np.mean([row["common_scale_roi_mse"] for row in pair_rows])),
        "mean_shape_roi_pearson": float(np.mean([row["shape_roi_pearson"] for row in pair_rows])),
        "mean_shape_roi_mse": float(np.mean([row["shape_roi_mse"] for row in pair_rows])),
        "mean_background_consistency_mse": float(np.mean([row["common_scale_background_mse"] for row in pair_rows])),
        "mean_edge_consistency_mse": float(np.mean([row["edge_mse"] for row in pair_rows])),
        "max_component_share_drift": float(np.max(np.ptp(component_share_array, axis=0))) if component_share_array.size else 0.0,
        "max_component_centroid_drift_um": max(centroid_drifts, default=0.0) * 1e6,
        "max_ring_radius_drift_um": (max(radius_values) - min(radius_values)) * 1e6 if radius_values else 0.0,
    }
    return {
        "summary": summary,
        "metrics_by_wavelength": metrics_by_wavelength,
        "pairwise": pair_rows,
        "raw_arrays": raw_arrays,
        "common_arrays": common_arrays,
        "shape_arrays": shape_arrays,
        "common_vmax": common_vmax,
    }


def save_run(
    run_dir: Path,
    target: TargetSpec,
    grid,
    aperture: torch.Tensor,
    outputs: list[torch.Tensor],
    result: OptimizationResult,
    evaluation: dict[str, Any],
    config: dict[str, Any],
) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)
    extent = make_extent(grid)
    target_array = target.intensity.detach().cpu().numpy()
    save_scalar_image(run_dir / "target.png", target_array, extent, "processed target")
    for name, mask in (
        ("signal_mask", target.signal_mask),
        ("transition_mask", target.transition_mask),
        ("background_mask", target.background_mask),
    ):
        save_scalar_image(
            run_dir / f"{name}.png",
            mask.detach().cpu().numpy(),
            extent,
            name,
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
            label="mask",
        )
    for wavelength_nm, raw, common in zip(
        WAVELENGTH_NM, evaluation["raw_arrays"], evaluation["common_arrays"]
    ):
        np.save(run_dir / f"output_{wavelength_nm}nm_linear.npy", raw)
        np.save(run_dir / f"output_{wavelength_nm}nm_common_scale.npy", common)
        save_scalar_image(
            run_dir / f"output_{wavelength_nm}nm_linear.png",
            raw,
            extent,
            f"{wavelength_nm} nm linear intensity",
            label="raw intensity",
        )
        save_scalar_image(
            run_dir / f"output_{wavelength_nm}nm_common_scale.png",
            common,
            extent,
            f"{wavelength_nm} nm common scale",
            vmin=0.0,
            vmax=1.0,
            label="power-normalized / shared maximum",
        )
        common_db = np.clip(10.0 * np.log10(np.maximum(common, 1e-4)), -40.0, 0.0)
        save_scalar_image(
            run_dir / f"output_{wavelength_nm}nm_common_scale_db.png",
            common_db,
            extent,
            f"{wavelength_nm} nm common scale dB",
            vmin=-40.0,
            vmax=0.0,
            label="dB relative to shared maximum",
        )
    common_arrays = evaluation["common_arrays"]
    shape_arrays = evaluation["shape_arrays"]
    common_db_arrays = [np.clip(10.0 * np.log10(np.maximum(array, 1e-4)), -40.0, 0.0) for array in common_arrays]
    rgb_common = np.stack([common_arrays[index] for index in RGB_CHANNEL_INDICES], axis=-1)
    rgb_shape = np.stack([shape_arrays[index] for index in RGB_CHANNEL_INDICES], axis=-1)
    rgb_common_db = np.stack([(common_db_arrays[index] + 40.0) / 40.0 for index in RGB_CHANNEL_INDICES], axis=-1)
    np.save(run_dir / "output_rgb_common_scale.npy", rgb_common)
    np.save(run_dir / "output_rgb_shape_normalized.npy", rgb_shape)
    np.save(run_dir / "output_rgb_common_scale_db.npy", rgb_common_db)
    save_rgb_image(run_dir / "output_rgb_common_scale.png", rgb_common, "RGB common-scale output\nequal-input-power display; not CIE colorimetric rendering")
    save_rgb_image(run_dir / "output_rgb_shape_normalized.png", rgb_shape, "RGB shape-normalized output")
    save_rgb_image(run_dir / "output_rgb_common_scale_db.png", rgb_common_db, "RGB common-scale dB output\n-40 dB to 0 dB")
    height = result.height_m.detach().cpu().numpy()
    np.save(run_dir / "height.npy", height)
    height_display = height[0] if height.ndim == 3 else height
    save_scalar_image(
        run_dir / "height.png",
        height_display * 1e6,
        extent,
        "DOE height (first plane when two DOE)",
        cmap="viridis",
        label="height (µm)",
    )
    save_loss_plot(run_dir / "loss.png", result.losses, result.components)
    write_json(run_dir / "loss_history.json", {"losses": result.losses, "components": result.components})
    component_keys = sorted({key for item in result.components for key in item})
    with (run_dir / "loss_components.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=["index", *component_keys])
        writer.writeheader()
        for index, item in enumerate(result.components):
            writer.writerow({"index": index, **item})
    write_json(run_dir / "metrics.json", {
        "summary": evaluation["summary"],
        "metrics_by_wavelength": evaluation["metrics_by_wavelength"],
        "pairwise": evaluation["pairwise"],
    })
    write_json(run_dir / "config.json", config)
    write_json(run_dir / "result.json", {"method": result.method, "metrics": evaluation["summary"], "config": config})


def group_rows(rows: list[dict[str, Any]], key: str) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        groups.setdefault(str(row[key]), []).append(row)
    return groups


def mean_std_rows(rows: list[dict[str, Any]], keys: tuple[str, ...]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["variant"], row["target"], row["architecture"]), []).append(row)
    aggregate: list[dict[str, Any]] = []
    for (variant, target, architecture), values in sorted(grouped.items()):
        item: dict[str, Any] = {"variant": variant, "target": target, "architecture": architecture, "count": len(values)}
        for key in keys:
            numeric = np.asarray([float(value[key]) for value in values], dtype=np.float64)
            item[f"{key}_mean"] = float(numeric.mean())
            item[f"{key}_std"] = float(numeric.std())
            item[f"{key}_best"] = float(numeric.max())
            item[f"{key}_worst"] = float(numeric.min())
        aggregate.append(item)
    return aggregate


def save_metric_heatmap(
    path: Path,
    rows: list[dict[str, Any]],
    metric: str,
    title: str,
    label: str,
    lower: float | None = None,
    upper: float | None = None,
) -> None:
    variants = list(VARIANTS)
    units = [(target, architecture) for target in TARGETS for architecture in ARCHITECTURES]
    matrix = np.full((len(variants), len(units)), np.nan, dtype=np.float64)
    for row_index, variant in enumerate(variants):
        for column_index, (target, architecture) in enumerate(units):
            values = [
                float(row[metric]) for row in rows
                if row["variant"] == variant and row["target"] == target and row["architecture"] == architecture
            ]
            if values:
                matrix[row_index, column_index] = float(np.mean(values))
    figure, axis = plt.subplots(figsize=(max(9, len(units) * 1.2), 4.8), constrained_layout=True)
    image = axis.imshow(matrix, aspect="auto", cmap="viridis", vmin=lower, vmax=upper)
    axis.set_title(title)
    axis.set_xlabel("target / architecture")
    axis.set_ylabel("loss variant")
    axis.set_xticks(np.arange(len(units)), [f"{target}\n{architecture}" for target, architecture in units], rotation=45, ha="right")
    axis.set_yticks(np.arange(len(variants)), variants)
    figure.colorbar(image, ax=axis, label=label)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_aggregate_plots(output_dir: Path, rows: list[dict[str, Any]]) -> None:
    save_metric_heatmap(
        output_dir / "pixel_similarity_heatmap.png",
        rows,
        "mean_common_scale_roi_pearson",
        "RGB common-scale ROI Pearson similarity",
        "Pearson r",
        lower=-1.0,
        upper=1.0,
    )
    save_metric_heatmap(
        output_dir / "absolute_energy_cv_heatmap.png",
        rows,
        "rgb_roi_energy_cv",
        "RGB ROI absolute-energy coefficient of variation",
        "CV",
    )
    save_metric_heatmap(
        output_dir / "centroid_drift_heatmap.png",
        rows,
        "max_component_centroid_drift_um",
        "Maximum RGB component centroid drift",
        "drift (µm)",
    )
    save_metric_heatmap(
        output_dir / "background_consistency_heatmap.png",
        rows,
        "mean_background_consistency_mse",
        "Cross-wavelength background consistency",
        "MSE",
    )
    for image_name, variant in (("common_scale_overview.png", "full"), ("shape_normalized_overview.png", "full")):
        figure, axes = plt.subplots(len(TARGETS), len(ARCHITECTURES), figsize=(5.0 * len(ARCHITECTURES), 3.8 * len(TARGETS)), squeeze=False, constrained_layout=True)
        image_key = "common_image" if image_name.startswith("common") else "shape_image"
        for target_index, target in enumerate(TARGETS):
            for architecture_index, architecture in enumerate(ARCHITECTURES):
                axis = axes[target_index, architecture_index]
                selected = [row for row in rows if row["variant"] == variant and row["target"] == target and row["architecture"] == architecture]
                if not selected:
                    axis.set_axis_off()
                    continue
                axis.imshow(selected[0][image_key], origin="lower")
                axis.set_title(f"{target} / {architecture}")
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle(f"{variant} RGB image-plane outputs")
        figure.savefig(output_dir / image_name, dpi=180)
        plt.close(figure)
    figure, axes = plt.subplots(len(TARGETS), len(ARCHITECTURES), figsize=(5.0 * len(ARCHITECTURES), 3.8 * len(TARGETS)), squeeze=False, constrained_layout=True)
    for target_index, target in enumerate(TARGETS):
        for architecture_index, architecture in enumerate(ARCHITECTURES):
            axis = axes[target_index, architecture_index]
            selected = [row for row in rows if row["variant"] == "full" and row["target"] == target and row["architecture"] == architecture]
            if not selected or "common_image" not in selected[0]:
                axis.set_axis_off()
                continue
            common_image = selected[0]["common_image"]
            shape_image = selected[0]["shape_image"]
            axis.imshow(np.concatenate([common_image, shape_image], axis=1), origin="lower")
            axis.axvline(common_image.shape[1] - 0.5, color="white", linewidth=1.0)
            axis.set_title(f"{target} / {architecture}\ncommon scale | shape normalized")
            axis.set_xticks([])
            axis.set_yticks([])
    figure.suptitle("RGB decolorization overview")
    figure.savefig(output_dir / "rgb_decolorization_overview.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8.5, 5.0), constrained_layout=True)
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if selected:
            axis.scatter(
                [row["mean_common_scale_roi_mse"] for row in selected],
                [row["mean_common_scale_roi_pearson"] for row in selected],
                label=variant,
                alpha=0.75,
            )
    axis.set(xlabel="common-scale ROI MSE", ylabel="common-scale ROI Pearson", title="RGB decolorization Pareto view")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output_dir / "correlation_vs_pixel_mse.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(10.0, 4.5), constrained_layout=True)
    values = []
    labels = []
    for variant in VARIANTS:
        selected = [row["rgb_roi_energy_cv"] for row in rows if row["variant"] == variant]
        if selected:
            values.append(float(np.mean(selected)))
            labels.append(variant)
    axis.bar(np.arange(len(labels)), values)
    axis.set(title="RGB ROI energy variation by ablation", ylabel="coefficient of variation", xticks=np.arange(len(labels)), xticklabels=labels)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "energy_ratio_comparison.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(8.5, 4.5), constrained_layout=True)
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if selected:
            axis.plot(
                [row["mean_common_scale_roi_pearson"] for row in selected],
                [row["worst_correlation"] for row in selected],
                "o",
                label=variant,
            )
    axis.set(xlabel="common-scale ROI Pearson", ylabel="worst-wavelength correlation", title="Image quality versus RGB consistency")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    figure.savefig(output_dir / "architecture_comparison.png", dpi=180)
    plt.close(figure)
    figure, axis = plt.subplots(figsize=(9.0, 5.0), constrained_layout=True)
    plotted = False
    for variant in VARIANTS:
        selected = [row for row in rows if row["variant"] == variant]
        if not selected:
            continue
        history_path = output_dir / selected[0]["run_dir"] / "loss_history.json"
        if not history_path.exists():
            continue
        history = json.loads(history_path.read_text(encoding="utf-8"))
        losses = history.get("losses", [])
        if losses:
            axis.plot(np.arange(len(losses)), losses, label=variant)
            plotted = True
    if plotted:
        axis.set(xlabel="logged optimization step", ylabel="total loss", title="Loss convergence by RGB ablation")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
    else:
        axis.text(0.5, 0.5, "No loss histories available", ha="center", va="center")
        axis.set_axis_off()
    figure.savefig(output_dir / "loss_component_convergence.png", dpi=180)
    plt.close(figure)


def run() -> dict[str, Any]:
    args = parse_args()
    if args.coarse_iterations < 1 or args.fine_iterations < 1:
        raise ValueError("iteration counts must be positive")
    if any(seed < 0 for seed in args.seeds):
        raise ValueError("seeds must be nonnegative")
    device = resolve_device(args.device)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    start_time = time.perf_counter()
    material = CauchyMaterial()
    wavelengths = torch.tensor(WAVELENGTHS_M, device=device, dtype=torch.float64)
    refractive_indices = torch.tensor(material.indices(WAVELENGTHS_M), device=device, dtype=torch.float64)
    height_max_m = args.height_max_um * 1e-6
    plane_separation_m = args.plane_separation_mm * 1e-3
    coarse_system = make_system(args.coarse_grid, args.coarse_pitch_um, args)
    fine_system = make_system(args.fine_grid, args.fine_pitch_um, args)
    coarse_grid = make_grid(coarse_system, device)
    fine_grid = make_grid(fine_system, device)
    coarse_sampling = validate_sampling(coarse_system, coarse_grid)
    fine_sampling = validate_sampling(fine_system, fine_grid)
    coarse_targets = {name: make_target(name, coarse_grid, device) for name in args.targets}
    fine_targets = {name: make_target(name, fine_grid, device) for name in args.targets}
    write_json(output_dir / "experiment_config.json", {
        "experiment": "RGB cross-wavelength decolorization loss ablation",
        "created_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "architectures": list(args.architectures),
        "targets": list(args.targets),
        "variants": list(args.variants),
        "seeds": list(args.seeds),
        "wavelengths_m": list(WAVELENGTHS_M),
        "input_powers": list(INPUT_POWERS),
        "material": material.metadata(WAVELENGTHS_M),
        "coarse_sampling": coarse_sampling,
        "fine_sampling": fine_sampling,
        "coarse_grid": args.coarse_grid,
        "coarse_pitch_um": args.coarse_pitch_um,
        "fine_grid": args.fine_grid,
        "fine_pitch_um": args.fine_pitch_um,
        "pad_factor": args.pad_factor,
        "aperture_radius_um": args.aperture_radius_um,
        "distance_mm": args.distance_mm,
        "plane_separation_mm": args.plane_separation_mm,
        "height_max_um": args.height_max_um,
        "coarse_iterations": args.coarse_iterations,
        "fine_iterations": args.fine_iterations,
        "target_config": TARGET_CONFIG,
        "variant_rgb_weights": VARIANT_RGB_WEIGHTS,
        "historical_architecture_excluded": "doe_fourier_lens",
        "initial_height": "deterministic uniform random height per architecture and seed; no historical drift_refined initialization",
    })
    rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    initial_cache: dict[tuple[str, int], torch.Tensor] = {}
    for target_name in args.targets:
        for architecture in args.architectures:
            coarse_forward = make_forward(
                architecture,
                coarse_grid,
                coarse_system,
                wavelengths,
                refractive_indices,
                plane_separation_m,
            )
            fine_forward = make_forward(
                architecture,
                fine_grid,
                fine_system,
                wavelengths,
                refractive_indices,
                plane_separation_m,
            )
            for seed in args.seeds:
                cache_key = (architecture, int(seed))
                if cache_key not in initial_cache:
                    initial_cache[cache_key] = make_initial_height(
                        architecture, coarse_grid, height_max_m, int(seed), device
                    )
                for variant in args.variants:
                    run_start = time.perf_counter()
                    print(
                        f"[start] {variant}/{target_name}/{architecture}/seed_{seed} on {device}",
                        flush=True,
                    )
                    coarse_result, fine_result = optimize_two_stage(
                        architecture,
                        target_name,
                        coarse_targets[target_name],
                        fine_targets[target_name],
                        coarse_forward,
                        fine_forward,
                        coarse_grid,
                        fine_grid,
                        height_max_m,
                        args.coarse_iterations,
                        args.fine_iterations,
                        int(seed),
                        variant,
                        initial_cache[cache_key],
                    )
                    with torch.no_grad():
                        outputs = fine_forward(fine_result.height_m)
                    evaluation = evaluate_outputs(
                        outputs,
                        fine_targets[target_name],
                        fine_grid,
                        aperture_on_base(fine_grid),
                        INPUT_POWERS,
                    )
                    runtime_seconds = time.perf_counter() - run_start
                    run_dir = output_dir / variant / target_name / architecture / f"seed_{seed}"
                    run_config = {
                        "variant": variant,
                        "target": target_name,
                        "architecture": architecture,
                        "seed": int(seed),
                        "wavelengths_nm": list(WAVELENGTH_NM),
                        "input_powers": list(INPUT_POWERS),
                        "material": material.metadata(WAVELENGTHS_M),
                        "coarse_iterations": args.coarse_iterations,
                        "fine_iterations": args.fine_iterations,
                        "rgb_schedule_coarse": vars(make_schedule(args.coarse_iterations)),
                        "rgb_schedule_fine": vars(make_schedule(args.fine_iterations)),
                        "loss_weights": vars(make_weights(target_name, variant)),
                        "height_max_um": args.height_max_um,
                        "runtime_seconds": runtime_seconds,
                        "device": str(device),
                    }
                    save_run(
                        run_dir,
                        fine_targets[target_name],
                        fine_grid,
                        aperture_on_base(fine_grid),
                        outputs,
                        fine_result,
                        evaluation,
                        run_config,
                    )
                    row = {
                        "variant": variant,
                        "target": target_name,
                        "architecture": architecture,
                        "seed": int(seed),
                        "runtime_seconds": runtime_seconds,
                        "run_dir": str(run_dir.relative_to(output_dir)),
                        **evaluation["summary"],
                    }
                    if variant == "full" and int(seed) == int(args.seeds[0]):
                        row["common_image"] = np.stack([evaluation["common_arrays"][index] for index in RGB_CHANNEL_INDICES], axis=-1)
                        row["shape_image"] = np.stack([evaluation["shape_arrays"][index] for index in RGB_CHANNEL_INDICES], axis=-1)
                    rows.append(row)
                    for item in evaluation["metrics_by_wavelength"]:
                        energy_rows.append({
                            "variant": variant,
                            "target": target_name,
                            "architecture": architecture,
                            "seed": int(seed),
                            **item,
                        })
                    for item in evaluation["pairwise"]:
                        pair_rows.append({
                            "variant": variant,
                            "target": target_name,
                            "architecture": architecture,
                            "seed": int(seed),
                            **item,
                        })
                    print(
                        f"[done] {variant}/{target_name}/{architecture}/seed_{seed} "
                        f"corr={evaluation['summary']['worst_correlation']:.4f} "
                        f"RGB-r={evaluation['summary']['mean_common_scale_roi_pearson']:.4f} "
                        f"energy-CV={evaluation['summary']['rgb_roi_energy_cv']:.4f} "
                        f"time={runtime_seconds:.1f}s",
                        flush=True,
                    )
                    del coarse_result, fine_result, outputs, evaluation
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    if not rows:
        raise RuntimeError("no experiment runs completed")
    public_rows = [{key: value for key, value in row.items() if key not in {"common_image", "shape_image"}} for row in rows]
    summary_keys = (
        "worst_correlation",
        "mean_common_scale_roi_pearson",
        "mean_common_scale_roi_mse",
        "mean_shape_roi_pearson",
        "rgb_roi_energy_cv",
        "mean_background_consistency_mse",
        "mean_edge_consistency_mse",
        "max_component_share_drift",
        "max_component_centroid_drift_um",
    )
    ablation_rows = mean_std_rows(public_rows, summary_keys)
    for filename, data in (
        ("summary.csv", public_rows),
        ("energy_by_wavelength.csv", energy_rows),
        ("cross_wavelength_similarity.csv", pair_rows),
        ("ablation_summary.csv", ablation_rows),
    ):
        if data:
            fields = sorted({key for item in data for key in item})
            with (output_dir / filename).open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=fields)
                writer.writeheader()
                writer.writerows(data)
    save_aggregate_plots(output_dir, rows)
    report = {
        "experiment": "RGB cross-wavelength decolorization loss ablation",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "architectures": list(args.architectures),
        "targets": list(args.targets),
        "variants": list(args.variants),
        "wavelengths_nm": list(WAVELENGTH_NM),
        "input_powers": list(INPUT_POWERS),
        "material": material.metadata(WAVELENGTHS_M),
        "elapsed_seconds": time.perf_counter() - start_time,
        "runs": public_rows,
        "ablation_summary": ablation_rows,
        "historical_architecture_excluded": "doe_fourier_lens",
    }
    write_json(output_dir / "summary.json", report)
    print(f"RGB decolorization ablation saved to {output_dir}", flush=True)
    return report


if __name__ == "__main__":
    run()
