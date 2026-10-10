"""Re-score saved RGB DOE runs against shared, architecture-independent targets."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

import chromatic_shift_experiment as experiment
from forward_models import make_grid
from materials import CauchyMaterial
from metrics import evaluate_output, evaluate_wavelength_consistency
from targets import supersampled_edge_target


def score(metrics: list[dict[str, Any]], consistency: dict[str, float], target) -> float:
    worst_correlation = min(item["correlation"] for item in metrics)
    background_mean = float(np.mean([item["background_fraction"] for item in metrics]))
    efficiency_mean = float(np.mean([item["roi_efficiency"] for item in metrics]))
    drift_um = consistency["max_centroid_drift_m"] * 1e6
    value = worst_correlation - 0.20 * background_mean + 0.15 * efficiency_mean
    value -= 0.10 * max(0.0, drift_um - 5.0) / 5.0
    if target.target_type == "ring":
        value -= 0.10 * consistency["max_radius_drift_m"] / float(target.metadata["ring_radius_m"])
    return float(value)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/chromatic_shift_multiplane"))
    parser.add_argument("--reference-sharpness", type=float, default=1.6)
    parser.add_argument("--prefer-refined", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    results_dir = args.results_dir.resolve()
    summary = json.loads((results_dir / "summary.json").read_text(encoding="utf-8"))
    experiment_config = json.loads((results_dir / "experiment_config.json").read_text(encoding="utf-8"))
    run_config = SimpleNamespace(**experiment_config)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    material = CauchyMaterial()
    wavelengths = torch.tensor(experiment.WAVELENGTHS_M, dtype=torch.float64, device=device)
    indices = torch.tensor(material.indices(experiment.WAVELENGTHS_M), dtype=torch.float64, device=device)
    fine_system = experiment.make_system(run_config.fine_grid, run_config.fine_pitch_um, run_config)
    grid = make_grid(fine_system, device)
    targets = list(summary["targets"])
    architectures = list(summary["optical_architectures"])
    target_images: dict[str, np.ndarray] = {}
    best_images: dict[tuple[str, str], np.ndarray] = {}
    best_scores: dict[tuple[str, str], float] = {}
    rows: list[dict[str, Any]] = []
    reference_targets = {}
    for target_name in targets:
        target = supersampled_edge_target(
            experiment.target_factory(target_name), grid.base_n, grid.pixel_pitch_m,
            device, sharpness_gain=args.reference_sharpness, supersample_factor=4,
        )
        reference_targets[target_name] = target
        target_images[target_name] = target.intensity.detach().cpu().numpy()

    for run in summary["runs"]:
        target_name = run["target"]
        architecture = run["architecture"]
        target = reference_targets[target_name]
        run_dir = results_dir / run["run_dir"]
        output_dir = run_dir / "drift_refined" if args.prefer_refined else run_dir
        if not (output_dir / "height.npy").exists():
            output_dir = run_dir
        height = torch.from_numpy(np.load(output_dir / "height.npy")).to(device=device, dtype=torch.float32)
        forward = experiment.make_forward(
            architecture,
            grid,
            fine_system,
            wavelengths,
            indices,
            run_config.plane_separation_mm * 1e-3,
        )
        with torch.no_grad():
            outputs = forward(height)
        metrics = [
            evaluate_output(field, target, grid.base_x, grid.base_y, grid.pixel_pitch_m)
            for field in outputs
        ]
        consistency = evaluate_wavelength_consistency(metrics, target)
        common_score = score(metrics, consistency, target)
        row = {
            "target": target_name,
            "architecture": architecture,
            "seed": run["seed"],
            "used_refined_solution": output_dir.name == "drift_refined",
            "common_reference_sharpness": args.reference_sharpness,
            "optimized_target_sharpness": json.loads((run_dir / "config.json").read_text(encoding="utf-8"))["sharpness_gain"],
            "common_reference_score": common_score,
            "worst_correlation": min(item["correlation"] for item in metrics),
            "mean_background_fraction": float(np.mean([item["background_fraction"] for item in metrics])),
            "mean_roi_efficiency": float(np.mean([item["roi_efficiency"] for item in metrics])),
            "max_centroid_drift_um": consistency["max_centroid_drift_m"] * 1e6,
            "max_ring_radius_drift_um": consistency["max_radius_drift_m"] * 1e6,
            "correlation_by_wavelength": [item["correlation"] for item in metrics],
            "background_by_wavelength": [item["background_fraction"] for item in metrics],
            "roi_efficiency_by_wavelength": [item["roi_efficiency"] for item in metrics],
            "component_centroids_by_wavelength_m": [item["component_centroids_m"] for item in metrics],
        }
        rows.append(row)
        key = (target_name, architecture)
        if key not in best_scores or common_score > best_scores[key]:
            best_scores[key] = common_score
            best_images[key] = experiment.rgb_pseudocolor(outputs)
        del outputs, height

    overview_path = results_dir / "common_reference_overview.png"
    experiment.make_overview(overview_path, target_images, best_images, targets, architectures, grid)
    experiment.save_shift_plot(
        results_dir / "common_reference_shift.png", rows, targets, architectures
    )
    experiment.save_target_processing(
        results_dir / "common_reference_target_processing.png",
        targets,
        [1.0, args.reference_sharpness],
        grid,
        device,
    )
    with (results_dir / "common_reference_metrics.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        fields = [
            "target", "architecture", "seed", "common_reference_sharpness",
            "optimized_target_sharpness", "used_refined_solution", "common_reference_score", "worst_correlation",
            "mean_background_fraction", "mean_roi_efficiency", "max_centroid_drift_um",
            "max_ring_radius_drift_um", "correlation_by_wavelength", "background_by_wavelength",
            "roi_efficiency_by_wavelength", "component_centroids_by_wavelength_m",
        ]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    report = {
        "reference_sharpness_gain": args.reference_sharpness,
        "target_shared_across_architectures": True,
        "device": str(device),
        "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "runs": rows,
    }
    (results_dir / "common_reference_metrics.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    print(f"Shared-target re-evaluation saved: {overview_path}")


if __name__ == "__main__":
    main()
