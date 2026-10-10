"""Re-evaluate saved RGB image-plane DOE runs with common intensity scales."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

import chromatic_shift_experiment as experiment
from forward_models import OpticalSystem, make_doe_field, make_grid
from materials import CauchyMaterial
from metrics import evaluate_output
from targets import supersampled_edge_target


WAVELENGTH_LABELS = ("488nm", "532nm", "633nm")
RGB_CHANNEL_INDICES = (2, 1, 0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/chromatic_shift_multiplane"))
    parser.add_argument("--targets", nargs="+", choices=experiment.TARGETS, default=list(experiment.TARGETS))
    parser.add_argument("--architectures", nargs="+", choices=experiment.ARCHITECTURES, default=list(experiment.ARCHITECTURES))
    parser.add_argument("--seeds", type=int, nargs="+", default=[7, 17, 27])
    parser.add_argument("--reference-sharpness", type=float, default=1.6)
    parser.add_argument("--power-weights", type=float, nargs=3, default=[1.0, 1.0, 1.0])
    parser.add_argument("--display-gamma", type=float, default=1.0)
    parser.add_argument("--prefer-refined", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rgb_imageplane_evaluation"))
    return parser.parse_args()


def resolve_device(force_cpu: bool) -> torch.device:
    return torch.device("cpu" if force_cpu or not torch.cuda.is_available() else "cuda")


def make_system(config: dict[str, Any]) -> OpticalSystem:
    return OpticalSystem(
        n_grid=int(config["fine_grid"]),
        pixel_pitch_m=float(config["fine_pitch_um"]) * 1e-6,
        aperture_radius_m=float(config["aperture_radius_um"]) * 1e-6,
        propagation_distance_m=float(config["distance_mm"]) * 1e-3,
        lens_focal_length_m=float(config["distance_mm"]) * 1e-3,
        lens_aperture_radius_m=float(config["aperture_radius_um"]) * 1e-6,
        pad_factor=int(config["pad_factor"]),
    )


def finite_array(value: np.ndarray) -> bool:
    return bool(np.isfinite(value).all())


def pearson(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        first = first[mask > 0]
        second = second[mask > 0]
    first = first.reshape(-1).to(torch.float64)
    second = second.reshape(-1).to(torch.float64)
    first = first - first.mean()
    second = second - second.mean()
    denominator = torch.linalg.vector_norm(first) * torch.linalg.vector_norm(second)
    return float((torch.sum(first * second) / (denominator + 1e-30)).detach().cpu())


def masked_mse(first: torch.Tensor, second: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        first = first[mask > 0]
        second = second[mask > 0]
    return float(torch.mean((first.to(torch.float64) - second.to(torch.float64)).square()).detach().cpu())


def component_absolute_energies(intensity: torch.Tensor, target, input_energy: float) -> list[float]:
    return [float((intensity * mask).sum().detach().cpu()) / max(input_energy, 1e-30) for mask in target.component_masks]


def component_centroid_drift(metrics: list[dict[str, Any]]) -> tuple[float, list[float]]:
    if len(metrics) < 2:
        return 0.0, []
    component_count = len(metrics[0]["component_centroids_m"])
    drifts: list[float] = []
    for component_index in range(component_count):
        values = [torch.tensor(item["component_centroids_m"][component_index], dtype=torch.float64) for item in metrics]
        component_max = 0.0
        for first_index, first in enumerate(values):
            for second in values[first_index + 1:]:
                component_max = max(component_max, float(torch.linalg.vector_norm(first - second)))
        drifts.append(component_max)
    return max(drifts, default=0.0), drifts


def safe_name(value: str) -> str:
    return value.replace("/", "_").replace("\\", "_")


def save_image(path: Path, array: np.ndarray, extent: list[float], title: str, vmin: float | None = None, vmax: float | None = None, cmap: str = "inferno", colorbar_label: str = "") -> None:
    figure, axis = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    image = axis.imshow(array, origin="lower", extent=extent, cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set(title=title, xlabel="x (mm)", ylabel="y (mm)")
    figure.colorbar(image, ax=axis, label=colorbar_label)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_rgb(path: Path, intensities: list[np.ndarray], vmax: float, title: str, gamma: float) -> None:
    rgb = rgb_array(intensities, vmax, gamma)
    figure, axis = plt.subplots(figsize=(5.2, 4.1), constrained_layout=True)
    axis.imshow(rgb, origin="lower")
    axis.set(title=title, xlabel="pixel x", ylabel="pixel y")
    figure.suptitle("equal-input-power display; not CIE colorimetric rendering", fontsize=9)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def rgb_array(intensities: list[np.ndarray], vmax: float, gamma: float = 1.0) -> np.ndarray:
    channels = [np.clip(intensities[index] / max(vmax, 1e-30), 0.0, 1.0) for index in RGB_CHANNEL_INDICES]
    rgb = np.stack(channels, axis=-1)
    if gamma != 1.0:
        rgb = np.clip(rgb, 0.0, 1.0) ** (1.0 / gamma)
    return rgb


def db_array(common_arrays: list[np.ndarray], floor_db: float = -40.0) -> np.ndarray:
    db_channels = [np.clip(10.0 * np.log10(np.maximum(array, 10.0 ** (floor_db / 10.0))), floor_db, 0.0) for array in common_arrays]
    scaled_channels = [(array - floor_db) / (-floor_db) for array in db_channels]
    return rgb_array(scaled_channels, 1.0, 1.0)


def save_db_channels(path: Path, common_arrays: list[np.ndarray], title: str, floor_db: float = -40.0) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(11.5, 3.6), constrained_layout=True)
    for axis, array, label in zip(axes, common_arrays, ("488 nm (B)", "532 nm (G)", "633 nm (R)")):
        db = np.clip(10.0 * np.log10(np.maximum(array, 10.0 ** (floor_db / 10.0))), floor_db, 0.0)
        image = axis.imshow(db, origin="lower", cmap="inferno", vmin=floor_db, vmax=0.0)
        axis.set_title(label)
        axis.set_xticks([])
        axis.set_yticks([])
        figure.colorbar(image, ax=axis, fraction=0.046, label="dB")
    figure.suptitle(f"{title}\nshared reference peak; equal-input-power display; not CIE colorimetric rendering")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def create_overview(path: Path, rows: list[dict[str, Any]], image_key: str, targets: list[str], architectures: list[str], title: str, cmap: str = "inferno") -> None:
    figure, axes = plt.subplots(len(targets), len(architectures), figsize=(4.2 * len(architectures), 3.5 * len(targets)), squeeze=False, constrained_layout=True)
    for row_index, target in enumerate(targets):
        for column_index, architecture in enumerate(architectures):
            selected = [row for row in rows if row["target"] == target and row["architecture"] == architecture]
            axis = axes[row_index, column_index]
            if not selected:
                axis.set_axis_off()
                continue
            row = sorted(selected, key=lambda item: item["seed"])[0]
            image = axis.imshow(np.asarray(row[image_key]), origin="lower", cmap=cmap)
            axis.set_title(f"{target} / {architecture}\nseed {row['seed']}")
            axis.set_xticks([])
            axis.set_yticks([])
            if cmap is not None:
                figure.colorbar(image, ax=axis, fraction=0.046)
    figure.suptitle(title)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def run_one(
    run_dir: Path,
    config: dict[str, Any],
    target_name: str,
    architecture: str,
    seed: int,
    device: torch.device,
    output_root: Path,
    power_weights: tuple[float, float, float],
    reference_sharpness: float,
    display_gamma: float,
    prefer_refined: bool,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    selected_dir = run_dir / "drift_refined" if prefer_refined and (run_dir / "drift_refined" / "height.npy").exists() else run_dir
    height = torch.from_numpy(np.load(selected_dir / "height.npy")).to(device=device, dtype=torch.float32)
    system = make_system(config)
    grid = make_grid(system, device)
    target = supersampled_edge_target(
        experiment.target_factory(target_name),
        grid.base_n,
        grid.pixel_pitch_m,
        device,
        sharpness_gain=reference_sharpness,
        supersample_factor=4,
    )
    wavelengths = torch.tensor(experiment.WAVELENGTHS_M, dtype=torch.float64, device=device)
    material = CauchyMaterial()
    indices = torch.tensor(material.indices(experiment.WAVELENGTHS_M), dtype=torch.float64, device=device)
    forward = experiment.make_forward(
        architecture,
        grid,
        system,
        wavelengths,
        indices,
        float(config["plane_separation_mm"]) * 1e-3,
    )
    with torch.no_grad():
        outputs = forward(height)

    intensities = [field.abs().square() for field in outputs]
    input_energies: list[float] = []
    for wavelength_m, refractive_index in zip(experiment.WAVELENGTHS_M, material.indices(experiment.WAVELENGTHS_M)):
        first_height = height[0] if architecture == "two_doe_planes" else height
        input_field = make_doe_field(first_height, grid, wavelength_m, refractive_index)
        input_energies.append(float(input_field.abs().square().sum().detach().cpu()))
    weighted_intensities = [intensity * power for intensity, power in zip(intensities, power_weights)]
    common_vmax = max(float(intensity.max().detach().cpu()) for intensity in weighted_intensities)
    extent = [float(grid.base_x.min().cpu() * 1e3), float(grid.base_x.max().cpu() * 1e3), float(grid.base_y.min().cpu() * 1e3), float(grid.base_y.max().cpu() * 1e3)]
    metrics_by_wavelength: list[dict[str, Any]] = []
    run_dir_output = output_root / target_name / architecture / f"seed_{seed}"
    run_dir_output.mkdir(parents=True, exist_ok=True)
    run_info = {
        "target": target_name,
        "architecture": architecture,
        "seed": seed,
        "source_run_dir": str(selected_dir),
        "power_weights": list(power_weights),
        "common_scale_vmax": common_vmax,
        "reference_sharpness": reference_sharpness,
    }
    (run_dir_output / "input_config.json").write_text(json.dumps(run_info, indent=2), encoding="utf-8")

    energy_rows: list[dict[str, Any]] = []
    for index, (label, wavelength_nm, intensity, input_energy) in enumerate(zip(WAVELENGTH_LABELS, (488.0, 532.0, 633.0), intensities, input_energies)):
        normalized_field = outputs[index]
        base_metrics = evaluate_output(normalized_field, target, grid.base_x, grid.base_y, grid.pixel_pitch_m)
        output_energy = float(intensity.sum().detach().cpu())
        roi_energy = float((intensity * target.signal_mask).sum().detach().cpu())
        background_energy = float((intensity * target.background_mask).sum().detach().cpu())
        weighted_intensity = weighted_intensities[index]
        shape_array = (intensity / (intensity.max() + 1e-30)).detach().cpu().numpy()
        common_array = (weighted_intensity / max(common_vmax, 1e-30)).detach().cpu().numpy()
        raw_array = intensity.detach().cpu().numpy()
        np.save(run_dir_output / f"output_{label}_linear.npy", raw_array)
        np.save(run_dir_output / f"output_{label}_common_scale.npy", common_array)
        save_image(run_dir_output / f"shape_normalized_{label}.png", shape_array, extent, f"{target_name} {architecture} {label}: shape normalized", vmin=0.0, vmax=1.0)
        save_image(run_dir_output / f"common_scale_linear_{label}.png", common_array, extent, f"{target_name} {architecture} {label}: common scale", vmin=0.0, vmax=1.0, colorbar_label="power-weighted intensity / shared vmax")
        common_db = 10.0 * np.log10(np.maximum(common_array, 1e-4))
        save_image(run_dir_output / f"common_scale_db_{label}.png", common_db, extent, f"{target_name} {architecture} {label}: common dB", vmin=-40.0, vmax=0.0, colorbar_label="dB relative to shared vmax")
        component_energies = component_absolute_energies(intensity, target, input_energy)
        wavelength_metrics = {
            **base_metrics,
            "wavelength_nm": wavelength_nm,
            "input_energy": input_energy,
            "output_energy": output_energy,
            "output_to_input_energy_ratio": output_energy / max(input_energy, 1e-30),
            "roi_energy_absolute_efficiency": roi_energy / max(input_energy, 1e-30),
            "background_energy_absolute_fraction": background_energy / max(input_energy, 1e-30),
            "peak_intensity": float(intensity.max().detach().cpu()),
            "common_scale_channel_max": float(common_array.max()),
            "component_absolute_efficiencies": component_energies,
        }
        metrics_by_wavelength.append(wavelength_metrics)
        energy_rows.append({
            "target": target_name, "architecture": architecture, "seed": seed, "wavelength_nm": wavelength_nm,
            "power_weight": power_weights[index], "input_energy": input_energy, "output_energy": output_energy,
            "output_to_input_energy_ratio": wavelength_metrics["output_to_input_energy_ratio"],
            "roi_energy_absolute_efficiency": wavelength_metrics["roi_energy_absolute_efficiency"],
            "background_energy_absolute_fraction": wavelength_metrics["background_energy_absolute_fraction"],
            "peak_intensity": wavelength_metrics["peak_intensity"], "common_scale_channel_max": wavelength_metrics["common_scale_channel_max"],
        })

    common_arrays = [(weighted_intensities[index] / max(common_vmax, 1e-30)).detach().cpu().numpy() for index in range(3)]
    shape_arrays = [(intensities[index] / (intensities[index].max() + 1e-30)).detach().cpu().numpy() for index in range(3)]
    common_db_arrays = [np.clip(10.0 * np.log10(np.maximum(array, 1e-4)), -40.0, 0.0) for array in common_arrays]
    rgb_common = rgb_array(common_arrays, 1.0, display_gamma)
    rgb_shape = rgb_array(shape_arrays, 1.0, display_gamma)
    rgb_common_db = db_array(common_arrays)
    np.save(run_dir_output / "output_rgb_common_scale.npy", rgb_common)
    np.save(run_dir_output / "output_rgb_shape_normalized.npy", rgb_shape)
    np.save(run_dir_output / "output_rgb_common_scale_db.npy", rgb_common_db)
    save_rgb(run_dir_output / "output_rgb_common_scale.png", common_arrays, 1.0, f"{target_name} {architecture} seed {seed}", display_gamma)
    save_rgb(run_dir_output / "output_rgb_shape_normalized.png", shape_arrays, 1.0, f"{target_name} {architecture} seed {seed} shape", display_gamma)
    save_rgb(run_dir_output / "output_rgb_common_scale_db.png", [(array + 40.0) / 40.0 for array in common_db_arrays], 1.0, f"{target_name} {architecture} seed {seed} common dB", 1.0)
    save_db_channels(run_dir_output / "output_common_scale_db_channels.png", common_arrays, f"{target_name} {architecture} seed {seed}")

    pair_rows: list[dict[str, Any]] = []
    for first_index in range(3):
        for second_index in range(first_index + 1, 3):
            first_common = weighted_intensities[first_index] / max(common_vmax, 1e-30)
            second_common = weighted_intensities[second_index] / max(common_vmax, 1e-30)
            first_shape = intensities[first_index] / (intensities[first_index].max() + 1e-30)
            second_shape = intensities[second_index] / (intensities[second_index].max() + 1e-30)
            pair_rows.append({
                "target": target_name, "architecture": architecture, "seed": seed,
                "first_wavelength_nm": (488.0, 532.0, 633.0)[first_index], "second_wavelength_nm": (488.0, 532.0, 633.0)[second_index],
                "common_scale_full_mse": masked_mse(first_common, second_common),
                "common_scale_roi_mse": masked_mse(first_common, second_common, target.signal_mask),
                "common_scale_background_mse": masked_mse(first_common, second_common, target.background_mask),
                "common_scale_full_pearson": pearson(first_common, second_common),
                "common_scale_roi_pearson": pearson(first_common, second_common, target.signal_mask),
                "shape_full_mse": masked_mse(first_shape, second_shape),
                "shape_roi_mse": masked_mse(first_shape, second_shape, target.signal_mask),
                "shape_full_pearson": pearson(first_shape, second_shape),
                "shape_roi_pearson": pearson(first_shape, second_shape, target.signal_mask),
            })
    max_drift, component_drifts = component_centroid_drift(metrics_by_wavelength)
    summary_row = {
        "target": target_name, "architecture": architecture, "seed": seed, "source_run_dir": str(selected_dir),
        "common_scale_vmax": common_vmax, "mean_output_to_input_energy_ratio": float(np.mean([row["output_to_input_energy_ratio"] for row in metrics_by_wavelength])),
        "mean_roi_absolute_efficiency": float(np.mean([row["roi_energy_absolute_efficiency"] for row in metrics_by_wavelength])),
        "mean_background_absolute_fraction": float(np.mean([row["background_energy_absolute_fraction"] for row in metrics_by_wavelength])),
        "min_correlation": min(row["correlation"] for row in metrics_by_wavelength),
        "mean_roi_efficiency": float(np.mean([row["roi_efficiency"] for row in metrics_by_wavelength])),
        "max_component_centroid_drift_um": max_drift * 1e6,
        "component_centroid_drifts_um": [value * 1e6 for value in component_drifts],
        "output_energy_by_wavelength": [row["output_energy"] for row in metrics_by_wavelength],
        "roi_absolute_efficiency_by_wavelength": [row["roi_energy_absolute_efficiency"] for row in metrics_by_wavelength],
    }
    (run_dir_output / "metrics.json").write_text(json.dumps({"summary": summary_row, "metrics_by_wavelength": metrics_by_wavelength, "pairwise_similarity": pair_rows}, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_row, energy_rows, {
        "pairs": pair_rows,
        "shape_images": shape_arrays,
        "common_images": common_arrays,
        "rgb_common": rgb_common,
        "rgb_shape": rgb_shape,
        "rgb_common_db": rgb_common_db,
    }


def save_aggregate_plots(output_dir: Path, summary_rows: list[dict[str, Any]], pair_rows: list[dict[str, Any]], targets: list[str], architectures: list[str]) -> None:
    plot_rows = []
    for row in summary_rows:
        copied = dict(row)
        run_path = Path(row["_run_output_dir"])
        copied["common_image"] = np.load(run_path / "output_rgb_common_scale.npy")
        copied["shape_image"] = np.load(run_path / "output_rgb_shape_normalized.npy")
        copied["common_db_image"] = np.load(run_path / "output_rgb_common_scale_db.npy")
        plot_rows.append(copied)
    create_overview(output_dir / "common_scale_overview.png", plot_rows, "common_image", targets, architectures, "RGB common-scale image-plane outputs", cmap=None)
    create_overview(output_dir / "shape_normalized_overview.png", plot_rows, "shape_image", targets, architectures, "RGB shape-normalized image-plane outputs", cmap=None)
    create_overview(output_dir / "common_scale_db_rgb.png", plot_rows, "common_db_image", targets, architectures, "RGB common-scale dB outputs\n-40 dB to 0 dB; not CIE colorimetric rendering", cmap=None)

    labels = [f"{row['target']}\n{row['architecture']}" for row in summary_rows]
    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 0.55), 5), constrained_layout=True)
    values = np.asarray([row["output_energy_by_wavelength"] for row in summary_rows])
    x = np.arange(len(labels))
    width = 0.24
    for index, label in enumerate(("488 nm", "532 nm", "633 nm")):
        axis.bar(x + (index - 1) * width, values[:, index], width, label=label)
    axis.set(title="Output energy by wavelength", ylabel="cropped output energy", xticks=x, xticklabels=labels)
    axis.tick_params(axis="x", labelrotation=70)
    axis.legend()
    figure.savefig(output_dir / "energy_ratio_comparison.png", dpi=180)
    plt.close(figure)

    pair_values = []
    pair_labels = []
    for row in summary_rows:
        selected = [pair for pair in pair_rows if pair["target"] == row["target"] and pair["architecture"] == row["architecture"] and pair["seed"] == row["seed"]]
        if not selected:
            continue
        pair_values.append([pair["common_scale_roi_pearson"] for pair in selected])
        pair_labels.append(f"{row['target']}\n{row['architecture']}")
    if pair_values:
        figure, axis = plt.subplots(figsize=(max(9, len(pair_labels) * 0.55), 4.5), constrained_layout=True)
        matrix = np.asarray(pair_values, dtype=float)
        image = axis.imshow(matrix.T, aspect="auto", vmin=-1.0, vmax=1.0, cmap="coolwarm")
        axis.set(title="Pairwise common-scale ROI Pearson similarity", xlabel="target / architecture / seed", ylabel="wavelength pair")
        axis.set_xticks(np.arange(len(pair_labels)), pair_labels, rotation=70)
        axis.set_yticks(np.arange(3), ["488-532", "488-633", "532-633"])
        figure.colorbar(image, ax=axis, label="Pearson r")
        figure.savefig(output_dir / "pixelwise_similarity_heatmap.png", dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(max(9, len(labels) * 0.55), 4.5), constrained_layout=True)
    axis.bar(np.arange(len(labels)), [row["max_component_centroid_drift_um"] for row in summary_rows])
    axis.set(title="Maximum component centroid drift", ylabel="drift (µm)", xticks=np.arange(len(labels)), xticklabels=labels)
    axis.tick_params(axis="x", labelrotation=70)
    axis.grid(axis="y", alpha=0.25)
    figure.savefig(output_dir / "centroid_drift_comparison.png", dpi=180)
    plt.close(figure)

    rgb_rows = [row for row in summary_rows if row.get("_rgb_image") is not None]
    if rgb_rows:
        figure, axes = plt.subplots(len(targets), len(architectures), figsize=(4.8 * len(architectures), 3.7 * len(targets)), squeeze=False, constrained_layout=True)
        for target_index, target in enumerate(targets):
            for architecture_index, architecture in enumerate(architectures):
                candidates = [row for row in rgb_rows if row["target"] == target and row["architecture"] == architecture]
                axis = axes[target_index, architecture_index]
                if not candidates:
                    axis.set_axis_off()
                    continue
                array = candidates[0]["_rgb_image"]
                axis.imshow(array)
                axis.set_title(f"{target} / {architecture}")
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle("RGB common-scale composites\nEqual-input-power display; not CIE colorimetric rendering")
        figure.savefig(output_dir / "rgb_common_scale_overview.png", dpi=180)
        plt.close(figure)

    if summary_rows:
        first_rows = []
        for row in summary_rows:
            if any(item["target"] == row["target"] and item["architecture"] == row["architecture"] for item in first_rows):
                continue
            first_rows.append(row)
        figure, axes = plt.subplots(len(targets), len(architectures), figsize=(4.8 * len(architectures), 4.0 * len(targets)), squeeze=False, constrained_layout=True)
        for target_index, target in enumerate(targets):
            for architecture_index, architecture in enumerate(architectures):
                candidates = [row for row in first_rows if row["target"] == target and row["architecture"] == architecture]
                axis = axes[target_index, architecture_index]
                if not candidates:
                    axis.set_axis_off()
                    continue
                axis.imshow(candidates[0]["_rgb_db_image"], origin="lower")
                axis.set_title(f"{target} / {architecture}")
                axis.set_xticks([])
                axis.set_yticks([])
        figure.suptitle("RGB common-scale dB composites\n-40 dB to 0 dB; equal-input-power display; not CIE colorimetric rendering")
        figure.savefig(output_dir / "rgb_common_scale_db_overview.png", dpi=180)
        plt.close(figure)


def run() -> dict[str, Any]:
    args = parse_args()
    if any(value <= 0 for value in args.power_weights):
        raise ValueError("power weights must be positive")
    if args.display_gamma <= 0:
        raise ValueError("display gamma must be positive")
    device = resolve_device(args.cpu)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    config = json.loads((args.results_dir / "experiment_config.json").read_text(encoding="utf-8"))
    summary = json.loads((args.results_dir / "summary.json").read_text(encoding="utf-8"))
    power_weights = tuple(float(value) for value in args.power_weights)
    summary_rows: list[dict[str, Any]] = []
    energy_rows: list[dict[str, Any]] = []
    pair_rows: list[dict[str, Any]] = []
    for run in summary["runs"]:
        target_name = run["target"]
        architecture = run["architecture"]
        seed = int(run["seed"])
        if target_name not in args.targets or architecture not in args.architectures or seed not in args.seeds:
            continue
        run_dir = args.results_dir / run["run_dir"]
        summary_row, run_energy_rows, extra = run_one(run_dir, config, target_name, architecture, seed, device, output_dir, power_weights, args.reference_sharpness, args.display_gamma, args.prefer_refined)
        summary_row["_run_output_dir"] = str(output_dir / target_name / architecture / f"seed_{seed}")
        rgb_path = output_dir / target_name / architecture / f"seed_{seed}" / "output_rgb_common_scale.png"
        summary_row["_rgb_image"] = extra["rgb_common"]
        summary_row["_rgb_db_image"] = extra["rgb_common_db"]
        summary_row["_shape_image"] = extra["rgb_shape"]
        summary_rows.append(summary_row)
        energy_rows.extend(run_energy_rows)
        pair_rows.extend(extra["pairs"])
        if device.type == "cuda":
            torch.cuda.empty_cache()

    public_summary_rows = [{key: value for key, value in row.items() if not key.startswith("_")} for row in summary_rows]
    fields = sorted({key for row in public_summary_rows for key in row})
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(public_summary_rows)
    energy_fields = sorted({key for row in energy_rows for key in row})
    with (output_dir / "energy_by_wavelength.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=energy_fields)
        writer.writeheader()
        writer.writerows(energy_rows)
    pair_fields = sorted({key for row in pair_rows for key in row})
    with (output_dir / "cross_wavelength_similarity.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=pair_fields)
        writer.writeheader()
        writer.writerows(pair_rows)
    public_report = {
        "experiment": "RGB image-plane common-scale and absolute-energy evaluation",
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "config": {
            "results_dir": str(args.results_dir), "targets": args.targets, "architectures": args.architectures, "seeds": args.seeds,
            "power_weights": list(power_weights), "display_gamma": args.display_gamma, "prefer_refined": args.prefer_refined,
            "reference_sharpness": args.reference_sharpness, "color_mapping": {"488nm": "B", "532nm": "G", "633nm": "R"},
            "colorimetric_warning": "equal-input-power display composite; not CIE colorimetric rendering",
        },
        "runs": public_summary_rows,
        "energy_by_wavelength": energy_rows,
        "cross_wavelength_similarity": pair_rows,
    }
    (output_dir / "summary.json").write_text(json.dumps(public_report, indent=2, ensure_ascii=False), encoding="utf-8")
    if summary_rows:
        save_aggregate_plots(output_dir, summary_rows, pair_rows, args.targets, args.architectures)
    print(f"RGB common-scale evaluation saved to {output_dir}")
    return public_report


if __name__ == "__main__":
    run()
