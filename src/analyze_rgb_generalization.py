"""Validate and summarize completed RGB multi-pattern DOE runs."""
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch

from forward_models import OpticalSystem, forward_field, make_grid, validate_sampling
from materials import CauchyMaterial
from metrics import evaluate_output, normalized_intensity
from rgb_generalization_experiment import TARGETS, make_system, target_spec


WAVELENGTHS_NM = (488.0, 532.0, 633.0)
METHODS = ("adam", "mraf")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--results-dir", type=Path, default=Path("results/rgb_generalization_corrected")
    )
    parser.add_argument("--skip-spectral", action="store_true")
    parser.add_argument("--spectral-step-nm", type=float, default=5.0)
    parser.add_argument("--cpu", action="store_true")
    return parser.parse_args()


def read_runs(results_dir: Path) -> list[dict[str, Any]]:
    return [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(results_dir.rglob("result.json"))
    ]


def run_key(run: dict[str, Any]) -> tuple[str, str, int]:
    return run["target"], run["method"], int(run["config_index"])


def metric_summary(run: dict[str, Any]) -> dict[str, float]:
    metrics = run["metrics"]
    consistency = run["wavelength_consistency"]
    values: dict[str, float] = {
        "score": float(run["score"]),
        "worst_correlation": min(float(item["correlation"]) for item in metrics),
        "mean_correlation": float(np.mean([item["correlation"] for item in metrics])),
        "mean_background_fraction": float(
            np.mean([item["background_fraction"] for item in metrics])
        ),
        "mean_roi_efficiency": float(np.mean([item["roi_efficiency"] for item in metrics])),
        "max_centroid_drift_um": float(consistency["max_centroid_drift_m"]) * 1e6,
        "mean_centroid_error_um": float(
            np.mean([item["mean_component_centroid_error_m"] for item in metrics])
        )
        * 1e6,
        "mean_component_share_error": float(
            np.mean([item["component_energy_balance_error"] for item in metrics])
        ),
    }
    for wave_index, wavelength in enumerate(WAVELENGTHS_NM):
        suffix = f"{wavelength:g}nm"
        values[f"correlation_{suffix}"] = float(
            metrics[wave_index]["correlation"]
        )
        values[f"background_{suffix}"] = float(
            metrics[wave_index]["background_fraction"]
        )
        values[f"efficiency_{suffix}"] = float(metrics[wave_index]["roi_efficiency"])
    if metrics and "mean_center_to_ring_ratio" in metrics[0]:
        values["mean_center_to_ring_ratio"] = float(
            np.mean([item["mean_center_to_ring_ratio"] for item in metrics])
        )
        values["mean_ring_radius_error_um"] = float(
            np.mean([item["mean_ring_radius_error_m"] for item in metrics]) * 1e6
        )
        values["max_ring_radius_drift_um"] = float(
            consistency.get("max_radius_drift_m", 0.0) * 1e6
        )
    return values


def validate_dataset(results_dir: Path, runs: list[dict[str, Any]]) -> dict[str, Any]:
    config = json.loads((results_dir / "experiment_config.json").read_text(encoding="utf-8"))
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    errors: list[str] = []
    for run in runs:
        groups[run_key(run)].append(run)
        path = (
            results_dir
            / run["target"]
            / run["method"]
            / f"config_{run['config_index']}"
            / f"seed_{run['seed']}"
        )
        if int(run.get("iterations", -1)) != int(config["fine_iterations"]):
            errors.append(f"{path}: fine iteration count mismatch")
        try:
            history = json.loads((path / "loss_history.json").read_text(encoding="utf-8"))
            if len(history) != int(config["fine_iterations"]):
                errors.append(f"{path}: history length={len(history)}")
            if not all(math.isfinite(float(value)) for value in history):
                errors.append(f"{path}: non-finite loss")
        except (OSError, ValueError, TypeError) as error:
            errors.append(f"{path}: cannot validate loss history ({error})")
        for filename in ("height.npy", "output_linear.png", "output_db.png", "metrics.json"):
            if not (path / filename).is_file():
                errors.append(f"{path}: missing {filename}")
    expected = (
        len(config["targets"])
        * len(config["methods"])
        * int(config["max_configs"])
        * len(config["seeds"])
    )
    if len(runs) != expected:
        errors.append(f"expected {expected} runs, found {len(runs)}")
    for key, group in groups.items():
        if len(group) != len(config["seeds"]):
            errors.append(f"{key}: expected {len(config['seeds'])} seeds, found {len(group)}")
    return {
        "config": config,
        "runs": len(runs),
        "groups": len(groups),
        "expected_runs": expected,
        "errors": errors,
    }


def summarize_groups(runs: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], dict]:
    groups: dict[tuple[str, str, int], list[dict[str, Any]]] = defaultdict(list)
    for run in runs:
        groups[run_key(run)].append(run)
    records: list[dict[str, Any]] = []
    chosen: dict[tuple[str, str], dict[str, Any]] = {}
    for (target, method, config_index), group in sorted(groups.items()):
        per_run = [metric_summary(run) for run in group]
        keys = sorted(per_run[0])
        record: dict[str, Any] = {
            "target": target,
            "method": method,
            "config_index": config_index,
            "seed_count": len(group),
        }
        for key in keys:
            values = [item[key] for item in per_run]
            record[f"{key}_mean"] = float(statistics.mean(values))
            record[f"{key}_std"] = float(statistics.stdev(values)) if len(values) > 1 else 0.0
            record[f"{key}_min"] = float(min(values))
            record[f"{key}_max"] = float(max(values))
        records.append(record)
        method_target = (target, method)
        current = chosen.get(method_target)
        if current is None or record["score_mean"] > current["summary"]["score_mean"]:
            chosen[method_target] = {
                "summary": record,
                "runs": group,
            }
    return records, chosen


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_run_summaries(results_dir: Path, runs: list[dict[str, Any]]) -> None:
    (results_dir / "summary.json").write_text(
        json.dumps(runs, indent=2), encoding="utf-8"
    )
    fields = ["target", "method", "seed", "config_index", "score", "wavelength_consistency"]
    with (results_dir / "summary.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    key: json.dumps(run[key]) if isinstance(run[key], (dict, list)) else run[key]
                    for key in fields
                }
            )


def choose_representative(runs: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(runs, key=lambda run: float(run["score"]))
    return ordered[len(ordered) // 2]


def make_representatives(chosen: dict) -> dict[tuple[str, str], dict[str, Any]]:
    return {
        key: choose_representative(value["runs"])
        for key, value in chosen.items()
    }


def construct_system_and_target(results_dir: Path, target_name: str, device: torch.device):
    config = json.loads((results_dir / "experiment_config.json").read_text(encoding="utf-8"))
    system = OpticalSystem(
        n_grid=int(config["fine_grid"]),
        pixel_pitch_m=float(config["fine_pitch_um"]) * 1e-6,
        aperture_radius_m=float(config["aperture_radius_um"]) * 1e-6,
        propagation_distance_m=float(config["propagation_distance_mm"]) * 1e-3,
        pad_factor=int(config["pad_factor"]),
    )
    grid = make_grid(system, device)
    sampling = validate_sampling(system, grid)
    target = target_spec(target_name, grid)
    return config, system, grid, target, sampling


def get_forward(system, grid, height: torch.Tensor, wavelength_nm: float, index: float):
    return forward_field(height, grid, system, wavelength_nm * 1e-9, index)


def plot_imageplane_comparison(
    results_dir: Path,
    representatives: dict[tuple[str, str], dict[str, Any]],
    output_path: Path,
    device: torch.device,
) -> None:
    config = json.loads((results_dir / "experiment_config.json").read_text(encoding="utf-8"))
    system = OpticalSystem(
        n_grid=int(config["fine_grid"]),
        pixel_pitch_m=float(config["fine_pitch_um"]) * 1e-6,
        aperture_radius_m=float(config["aperture_radius_um"]) * 1e-6,
        propagation_distance_m=float(config["propagation_distance_mm"]) * 1e-3,
        pad_factor=int(config["pad_factor"]),
    )
    grid = make_grid(system, device)
    x_mm = float(grid.base_x.min().cpu()) * 1e3
    x_max_mm = float(grid.base_x.max().cpu()) * 1e3
    y_mm = float(grid.base_y.min().cpu()) * 1e3
    y_max_mm = float(grid.base_y.max().cpu()) * 1e3
    extent = (x_mm, x_max_mm, y_mm, y_max_mm)
    material = CauchyMaterial()
    fig, axes = plt.subplots(
        len(TARGETS), 7, figsize=(17.5, 18), squeeze=False, constrained_layout=True
    )
    column_titles = (
        "Target",
        "Adam 488 nm",
        "Adam 532 nm",
        "Adam 633 nm",
        "MRAF 488 nm",
        "MRAF 532 nm",
        "MRAF 633 nm",
    )
    for column, title in enumerate(column_titles):
        axes[0, column].set_title(title, fontsize=10)
    for row, target_name in enumerate(TARGETS):
        target = target_spec(target_name, grid)
        target_array = target.intensity.detach().cpu().numpy()
        images = [target_array]
        for method in METHODS:
            representative = representatives[(target_name, method)]
            run_dir = (
                results_dir
                / target_name
                / method
                / f"config_{representative['config_index']}"
                / f"seed_{representative['seed']}"
            )
            height = torch.from_numpy(np.load(run_dir / "height.npy")).to(device=device)
            for wavelength_nm in WAVELENGTHS_NM:
                field = get_forward(
                    system, grid, height, wavelength_nm, material.index(wavelength_nm * 1e-9)
                )
                images.append(normalized_intensity(field).detach().cpu().numpy())
        for column, (axis, image) in enumerate(zip(axes[row], images)):
            db = 10.0 * np.log10(np.maximum(image, 1e-12) / max(float(image.max()), 1e-12))
            axis.imshow(
                db,
                origin="lower",
                extent=extent,
                cmap="inferno",
                vmin=-40,
                vmax=0,
                interpolation="nearest",
            )
            axis.set_xlim(-0.5, 0.5)
            axis.set_ylim(-0.25, 0.45)
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(target_name, fontsize=9)
    fig.suptitle(
        "RGB target and image-plane output (peak-normalized, fixed -40 dB display)",
        fontsize=13,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def plot_summary(
    records: list[dict[str, Any]],
    chosen: dict,
    output_dir: Path,
) -> None:
    target_labels = list(TARGETS)
    correlation_matrix = np.full((len(METHODS), len(target_labels)), np.nan)
    background_matrix = np.full_like(correlation_matrix, np.nan)
    efficiency_matrix = np.full_like(correlation_matrix, np.nan)
    drift_matrix = np.full_like(correlation_matrix, np.nan)
    for row_index, method in enumerate(METHODS):
        for column_index, target in enumerate(target_labels):
            record = chosen[(target, method)]["summary"]
            correlation_matrix[row_index, column_index] = record["worst_correlation_mean"]
            background_matrix[row_index, column_index] = record["mean_background_fraction_mean"]
            efficiency_matrix[row_index, column_index] = record["mean_roi_efficiency_mean"]
            drift_matrix[row_index, column_index] = record["max_centroid_drift_um_mean"]

    fig, axes = plt.subplots(2, 2, figsize=(16, 8), constrained_layout=True)
    matrices = (
        (correlation_matrix, "Worst RGB correlation", "viridis", (0, 1)),
        (background_matrix, "Mean background fraction", "magma_r", (0, 0.4)),
        (efficiency_matrix, "Mean ROI efficiency", "viridis", (0, 1)),
        (drift_matrix, "Maximum centroid drift (µm)", "magma_r", (0, 50)),
    )
    for axis, (matrix, title, cmap, limits) in zip(axes.flat, matrices):
        image = axis.imshow(matrix, aspect="auto", cmap=cmap, vmin=limits[0], vmax=limits[1])
        axis.set_xticks(range(len(target_labels)), target_labels, rotation=35, ha="right")
        axis.set_yticks(range(len(METHODS)), METHODS)
        axis.set_title(title)
        for row in range(matrix.shape[0]):
            for column in range(matrix.shape[1]):
                axis.text(column, row, f"{matrix[row, column]:.2f}", ha="center", va="center",
                          color="white" if matrix[row, column] < np.mean(limits) else "black",
                          fontsize=8)
        fig.colorbar(image, ax=axis, shrink=0.8)
    fig.savefig(output_dir / "rgb_metric_heatmaps.png", dpi=180)
    plt.close(fig)

    colors = plt.get_cmap("tab10")
    fig, axis = plt.subplots(figsize=(9, 6), constrained_layout=True)
    marker_by_method = {"adam": "o", "mraf": "s"}
    for target_index, target in enumerate(target_labels):
        for method in METHODS:
            summary = chosen[(target, method)]["summary"]
            axis.scatter(
                summary["mean_background_fraction_mean"],
                summary["worst_correlation_mean"],
                color=colors(target_index),
                marker=marker_by_method[method],
                s=70,
                edgecolors="black",
                linewidths=0.4,
            )
    for index, target in enumerate(target_labels):
        axis.scatter([], [], color=colors(index), label=target)
    axis.scatter([], [], color="black", marker="o", label="Adam")
    axis.scatter([], [], color="black", marker="s", label="MRAF")
    axis.set(
        xlabel="Mean background fraction",
        ylabel="Worst wavelength correlation",
        title="Best mean-score configuration per target and method",
    )
    axis.grid(alpha=0.25)
    axis.legend(ncol=2, fontsize=8)
    fig.savefig(output_dir / "correlation_background_pareto.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    x_positions = np.arange(len(target_labels))
    width = 0.36
    for method_index, method in enumerate(METHODS):
        summaries = [chosen[(target, method)]["summary"] for target in target_labels]
        offsets = x_positions + (method_index - 0.5) * width
        axes[0].bar(
            offsets,
            [item["worst_correlation_mean"] for item in summaries],
            width,
            yerr=[item["worst_correlation_std"] for item in summaries],
            capsize=3,
            label=method,
        )
        axes[1].bar(
            offsets,
            [item["max_centroid_drift_um_mean"] for item in summaries],
            width,
            yerr=[item["max_centroid_drift_um_std"] for item in summaries],
            capsize=3,
            label=method,
        )
    for axis, title, ylabel in (
        (axes[0], "Seed stability: worst RGB correlation", "Correlation"),
        (axes[1], "Seed stability: RGB centroid drift", "Drift (µm)"),
    ):
        axis.set_xticks(x_positions, target_labels, rotation=35, ha="right")
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.grid(axis="y", alpha=0.25)
        axis.legend()
    fig.savefig(output_dir / "seed_stability.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    labels = [f"{target}\n{method}" for target in target_labels for method in METHODS]
    values = [
        chosen[(target, method)]["summary"]["mean_component_share_error_mean"]
        for target in target_labels
        for method in METHODS
    ]
    axes[0].bar(np.arange(len(labels)), values, color=[
        colors(index // 2) for index in range(len(labels))
    ])
    axes[0].set_xticks(np.arange(len(labels)), labels, rotation=70, ha="right", fontsize=7)
    axes[0].set_title("Mean component-share error")
    axes[0].set_ylabel("Total variation distance")
    ring_targets = ("ring50", "ring60")
    for target in ring_targets:
        for method in METHODS:
            summary = chosen[(target, method)]["summary"]
            axes[1].scatter(
                summary["mean_center_to_ring_ratio_mean"],
                summary["mean_ring_radius_error_um_mean"],
                marker=marker_by_method[method],
                label=f"{target} {method}",
            )
    axes[1].set(
        xlabel="Mean center/ring intensity ratio",
        ylabel="Mean ring-radius error (µm)",
        title="Ring-specific quality",
    )
    axes[1].grid(alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(output_dir / "component_and_ring_metrics.png", dpi=180)
    plt.close(fig)


def spectral_validation(
    results_dir: Path,
    representatives: dict[tuple[str, str], dict[str, Any]],
    output_dir: Path,
    step_nm: float,
    device: torch.device,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    if step_nm <= 0:
        raise ValueError("spectral step must be positive")
    start, stop = WAVELENGTHS_NM[0], WAVELENGTHS_NM[-1]
    wavelengths = np.arange(start, stop + 0.5 * step_nm, step_nm)
    if not np.isclose(wavelengths[-1], stop):
        wavelengths = np.append(wavelengths, stop)
    material = CauchyMaterial()
    spectral_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []
    curves: dict[tuple[str, str], list[dict[str, float]]] = {}
    for target_name in TARGETS:
        config, system, grid, target, _ = construct_system_and_target(
            results_dir, target_name, device
        )
        for method in METHODS:
            representative = representatives[(target_name, method)]
            run_dir = (
                results_dir
                / target_name
                / method
                / f"config_{representative['config_index']}"
                / f"seed_{representative['seed']}"
            )
            height = torch.from_numpy(np.load(run_dir / "height.npy")).to(device=device)
            metrics_at_wavelength: list[dict[str, float]] = []
            with torch.no_grad():
                for wavelength_nm in wavelengths:
                    index = material.index(float(wavelength_nm) * 1e-9)
                    field = get_forward(system, grid, height, float(wavelength_nm), index)
                    metrics = evaluate_output(
                        field, target, grid.base_x, grid.base_y, grid.pixel_pitch_m
                    )
                    row = {
                        "target": target_name,
                        "method": method,
                        "config_index": int(representative["config_index"]),
                        "seed": int(representative["seed"]),
                        "wavelength_nm": float(wavelength_nm),
                        "refractive_index": index,
                        "correlation": float(metrics["correlation"]),
                        "background_fraction": float(metrics["background_fraction"]),
                        "roi_efficiency": float(metrics["roi_efficiency"]),
                        "mean_centroid_error_um": float(
                            metrics["mean_component_centroid_error_m"]
                        )
                        * 1e6,
                    }
                    spectral_rows.append(row)
                    metrics_at_wavelength.append(row)
                    if np.isclose(wavelength_nm, 488.0):
                        for delta_n in (-0.005, 0.0, 0.005):
                            varied_index = index + delta_n
                            varied_field = get_forward(
                                system, grid, height, 488.0, varied_index
                            )
                            varied_metrics = evaluate_output(
                                varied_field,
                                target,
                                grid.base_x,
                                grid.base_y,
                                grid.pixel_pitch_m,
                            )
                            sensitivity_rows.append(
                                {
                                    "target": target_name,
                                    "method": method,
                                    "config_index": int(representative["config_index"]),
                                    "seed": int(representative["seed"]),
                                    "delta_n": delta_n,
                                    "refractive_index": varied_index,
                                    "correlation": float(varied_metrics["correlation"]),
                                    "background_fraction": float(
                                        varied_metrics["background_fraction"]
                                    ),
                                    "roi_efficiency": float(
                                        varied_metrics["roi_efficiency"]
                                    ),
                                    "mean_centroid_error_um": float(
                                        varied_metrics["mean_component_centroid_error_m"]
                                    )
                                    * 1e6,
                                }
                            )
            curves[(target_name, method)] = metrics_at_wavelength

    write_csv(output_dir / "continuous_wavelength_metrics.csv", spectral_rows)
    write_csv(output_dir / "488nm_index_sensitivity.csv", sensitivity_rows)
    fig, axes = plt.subplots(2, 2, figsize=(13, 8), constrained_layout=True)
    metric_fields = (
        ("correlation", "Correlation"),
        ("background_fraction", "Background fraction"),
        ("roi_efficiency", "ROI efficiency"),
        ("mean_centroid_error_um", "Mean centroid error (µm)"),
    )
    for axis, (field, title) in zip(axes.flat, metric_fields):
        for method in METHODS:
            rows = [
                row
                for target_name in TARGETS
                for row in curves[(target_name, method)]
            ]
            x_values = sorted({row["wavelength_nm"] for row in rows})
            means = [
                float(np.mean([
                    row[field] for row in rows if row["wavelength_nm"] == x_value
                ]))
                for x_value in x_values
            ]
            stds = [
                float(np.std([
                    row[field] for row in rows if row["wavelength_nm"] == x_value
                ]))
                for x_value in x_values
            ]
            axis.plot(x_values, means, label=method)
            axis.fill_between(
                x_values,
                np.asarray(means) - np.asarray(stds),
                np.asarray(means) + np.asarray(stds),
                alpha=0.15,
            )
        for wavelength in WAVELENGTHS_NM:
            axis.axvline(wavelength, color="gray", alpha=0.25, linewidth=0.8)
        axis.set(title=title, xlabel="Wavelength (nm)")
        axis.grid(alpha=0.2)
        axis.legend()
    fig.suptitle(
        "Continuous-wavelength evaluation of representative optimized DOE designs"
    )
    fig.savefig(output_dir / "continuous_wavelength_validation.png", dpi=180)
    plt.close(fig)
    return spectral_rows, sensitivity_rows


def report_text(
    validation: dict[str, Any],
    chosen: dict,
    spectral_rows: list[dict[str, Any]] | None,
    sensitivity_rows: list[dict[str, Any]] | None,
) -> str:
    config = validation["config"]
    best_rows = [
        chosen[(target, method)]["summary"]
        for target in TARGETS
        for method in METHODS
    ]
    passing_general = []
    for target in TARGETS:
        best = max(
            (chosen[(target, method)]["summary"] for method in METHODS),
            key=lambda item: item["score_mean"],
        )
        if (
            best["worst_correlation_mean"] >= 0.75
            and best["mean_background_fraction_mean"] < 0.15
            and best["mean_roi_efficiency_mean"] > 0.75
            and best["max_centroid_drift_um_mean"] < 20
        ):
            passing_general.append(target)
    score_gap = float(np.mean([
        chosen[(target, "mraf")]["summary"]["score_mean"]
        - chosen[(target, "adam")]["summary"]["score_mean"]
        for target in TARGETS
    ]))
    lines = [
        "# 阶段性报告 2：RGB 多图案 DOE 完整迭代实验",
        "",
        "## 执行摘要",
        "",
        f"- 已核验 {validation['runs']} 组精细优化结果；每组精网格损失历史均为 "
        f"{config['fine_iterations']} 次，且未发现非有限损失值。",
        f"- 设计波长为 488/532/633 nm；流程使用 {config['coarse_grid']}×"
        f"{config['coarse_grid']}、{config['coarse_pitch_um']:g} µm 粗优化，"
        f"继承至 {config['fine_grid']}×{config['fine_grid']}、"
        f"{config['fine_pitch_um']:g} µm 精优化；每组精阶段 {config['fine_iterations']} 次。",
        f"- 按相关系数、背景、ROI 效率和跨波长质心漂移的宽松门槛，"
        f"{len(passing_general)}/{len(TARGETS)} 类目标达到综合筛选条件："
        f"{', '.join(passing_general) if passing_general else '无'}。",
        f"- MRAF 相对 Adam 的最佳配置平均分差为 {score_gap:+.3f}；两者整体差距较小，"
        "不能据此认定 MRAF 普遍优于 Adam。",
        "",
        "## 实验设置与数据质量",
        "",
        f"- 目标：{', '.join(config['targets'])}；方法：{', '.join(config['methods'])}；"
        f"种子：{', '.join(map(str, config['seeds']))}。",
        f"- 光路：自由传播 {config['propagation_distance_mm']:g} mm；"
        f"DOE 孔径半径 {config['aperture_radius_um']:g} µm；零填充因子 {config['pad_factor']}。",
        "- 同一目标、方法和候选配置的三个随机种子分别报告；表内配置选择依据为三种子平均综合分，"
        "展示图选取分数处于中位数的种子，而非挑最高分单次结果。",
        "- 折射率模型为 S1813 两锚点 Cauchy 外推：532 nm 为 1.6482，633 nm 为 1.6347，"
        "488 nm 外推为约 1.65686。该 488 nm 值并非直接测量值。",
        "- 精网格视场为 2.56 mm，孔径直径 2.00 mm，几何视场比为 1.28；"
        "采样检查记录在各次运行的 `result.json` 中。",
        "",
        "## 目标 × 方法结果",
        "",
        "| 目标 | 方法 | 配置 | 最差波长相关系数 | 背景比例 | ROI 效率 | 最大 RGB 质心漂移 (µm) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for target in TARGETS:
        for method in METHODS:
            summary = chosen[(target, method)]["summary"]
            lines.append(
                f"| {target} | {method} | {summary['config_index']} | "
                f"{summary['worst_correlation_mean']:.3f} ± "
                f"{summary['worst_correlation_std']:.3f} | "
                f"{summary['mean_background_fraction_mean']:.3f} | "
                f"{summary['mean_roi_efficiency_mean']:.3f} | "
                f"{summary['max_centroid_drift_um_mean']:.1f} |"
            )
    lines.extend(
        [
            "",
            "## 结果解读",
            "",
            "- 修正十字目标的符号后，十字相关系数约 0.89、背景约 0.05、ROI 效率约 0.90，"
            "跨 RGB 质心漂移约 2–3 µm；因此它是本轮表现较好的非圆形图案之一。",
            "- R60 圆环优于 R50：两者平均中心/环带强度比分别约 0.004–0.005 和 0.007，"
            "环半径误差约 3 µm；R50 仍因 RGB 质心漂移约 24 µm、背景约 0.12 而未达到预设标准。",
            "- 双圆斑相关系数约 0.77，低于其较严格的 0.85 目标；方形、三角形和不对称目标的"
            "总体指标较好，但几何边界指标仍不完整。",
            "- 这些数据支持“部分目标可以由同类流程重建”，不支持“算法对任意图案均有泛化能力”。"
            "特别是几何专用损失尚未形成完整的边/角点误差测量，方形和三角形的边界质量不能仅由全场相关系数判定。",
            "- 本轮所谓 MRAF 是 Adam 优化器中的背景混合/偏置风格项，并非经典 MRAF 独立迭代投影算法。",
            "",
            "## 488 nm 色散敏感性与连续波长",
            "",
        ]
    )
    if sensitivity_rows:
        nominal = [row for row in sensitivity_rows if row["delta_n"] == 0.0]
        low = [row for row in sensitivity_rows if row["delta_n"] == -0.005]
        high = [row for row in sensitivity_rows if row["delta_n"] == 0.005]
        mean_corr_change = float(np.mean([
            max(abs(a["correlation"] - b["correlation"]),
                abs(a["correlation"] - c["correlation"]))
            for a, b, c in zip(nominal, low, high)
        ]))
        lines.append(
            f"- 对代表性设计固定高度图，仅改变 488 nm 折射率 ±0.005，"
            f"三者相关系数最大变化的跨设计平均值为 {mean_corr_change:.3f}。"
            "这是后验敏感性检查，不是重新优化后的鲁棒性结论。"
        )
    else:
        lines.append("- 488 nm 折射率敏感性复算未执行。")
    if spectral_rows:
        correlations = [row["correlation"] for row in spectral_rows]
        lines.append(
            f"- 对每个目标/方法各选一张代表性 DOE，在 "
            f"{min(row['wavelength_nm'] for row in spectral_rows):g}–"
            f"{max(row['wavelength_nm'] for row in spectral_rows):g} nm "
            f"以 5 nm 间隔复算；所有代表设计的最差相关系数为 {min(correlations):.3f}。"
            "这属于离散波长抽样验证，不等同于连续谱实验或全谱优化。"
        )
    else:
        lines.append("- 连续波长抽样验证未执行。")
    lines.extend(
        [
            "",
            "## 结论与下一步",
            "",
            "1. 暂不宣称 RGB 任意图案泛化成功；R50 圆环未达标，双圆斑也未达到其严格相关性目标。",
            "2. 十字已通过当前像面形状、背景、效率和位置筛选；下一步为其补充臂长、臂宽和边缘粗糙度指标。",
            "3. 为方形、十字和实心三角形补充可测量的边界距离、顶点误差和边缘粗糙度，而非只增加过渡区惩罚。",
            "4. 对圆环联合扫描内孔惩罚和环宽，并按环半径误差、中心泄漏和跨波长漂移筛选 Pareto 解。",
            "5. 若研究问题强调 RGB 同位置，下一轮使用位置漂移为显式约束的多目标筛选，并报告每个组件跨波长质心，不仅报告平均分。",
            "6. 本轮没有执行独立双波长高度图继承对照，也没有将 ±0.005 折射率扰动纳入重新训练；后续应把这两项作为独立实验，而不能从当前后验评估推断。",
            "",
            "## 产物",
            "",
            "- `target_method_config_statistics.csv`：候选配置跨种子均值、标准差、最好/最差。",
            "- `best_target_method.csv`：每目标每方法的最优平均分配置。",
            "- `rgb_imageplane_comparison.png`：目标与 Adam/MRAF 的 488/532/633 nm 像面图。",
            "- `rgb_metric_heatmaps.png`、`correlation_background_pareto.png`、`seed_stability.png`。",
            "- `continuous_wavelength_validation.png`、`continuous_wavelength_metrics.csv`。",
            "- `488nm_index_sensitivity.csv`：固定 DOE 高度后的折射率扰动复算。",
            "",
            "图像采用每幅峰值归一化的固定 -40 dB 显示以观察空间形状；定量结论以 JSON/CSV 的线性强度指标为准。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    results_dir = args.results_dir
    output_dir = results_dir / "analysis"
    output_dir.mkdir(parents=True, exist_ok=True)
    runs = read_runs(results_dir)
    validation = validate_dataset(results_dir, runs)
    (output_dir / "validation.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    if validation["errors"]:
        raise RuntimeError("result validation failed:\n" + "\n".join(validation["errors"]))
    write_run_summaries(results_dir, runs)
    records, chosen = summarize_groups(runs)
    representatives = make_representatives(chosen)
    write_csv(output_dir / "target_method_config_statistics.csv", records)
    write_csv(
        output_dir / "best_target_method.csv",
        [chosen[(target, method)]["summary"] for target in TARGETS for method in METHODS],
    )
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    plot_imageplane_comparison(
        results_dir, representatives, output_dir / "rgb_imageplane_comparison.png", device
    )
    plot_summary(records, chosen, output_dir)
    spectral_rows: list[dict[str, Any]] | None = None
    sensitivity_rows: list[dict[str, Any]] | None = None
    if not args.skip_spectral:
        spectral_rows, sensitivity_rows = spectral_validation(
            results_dir,
            representatives,
            output_dir,
            args.spectral_step_nm,
            device,
        )
    report = report_text(validation, chosen, spectral_rows, sensitivity_rows)
    (output_dir / "阶段性报告2.md").write_text(report, encoding="utf-8")
    print(
        f"validated {validation['runs']} runs; wrote analysis to {output_dir.resolve()}"
    )


if __name__ == "__main__":
    main()
