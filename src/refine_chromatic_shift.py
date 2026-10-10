"""Tighten the RGB centroid-drift constraint on saved DOE solutions."""
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path, default=Path("results/chromatic_shift_multiplane"))
    parser.add_argument("--iterations", type=int, default=120)
    parser.add_argument("--position-weight", type=float, default=2.0)
    parser.add_argument("--centroid-tolerance-um", type=float, default=3.0)
    parser.add_argument("--max-runs", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    root = args.results_dir.resolve()
    summary = json.loads((root / "summary.json").read_text(encoding="utf-8"))
    experiment_config = json.loads((root / "experiment_config.json").read_text(encoding="utf-8"))
    config_args = SimpleNamespace(**experiment_config)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    material = CauchyMaterial()
    wavelengths = torch.tensor(experiment.WAVELENGTHS_M, dtype=torch.float64, device=device)
    indices = torch.tensor(material.indices(experiment.WAVELENGTHS_M), dtype=torch.float64, device=device)
    system = experiment.make_system(config_args.fine_grid, config_args.fine_pitch_um, config_args)
    grid = make_grid(system, device)
    rows: list[dict[str, Any]] = []

    for index, run in enumerate(summary["runs"]):
        if args.max_runs and index >= args.max_runs:
            break
        run_dir = root / run["run_dir"]
        saved_config = json.loads((run_dir / "config.json").read_text(encoding="utf-8"))
        target = experiment.supersampled_edge_target(
            experiment.target_factory(run["target"]),
            grid.base_n,
            grid.pixel_pitch_m,
            device,
            sharpness_gain=saved_config["sharpness_gain"],
            supersample_factor=4,
        )
        height = torch.from_numpy(np.load(run_dir / "height.npy")).to(device=device, dtype=torch.float32)
        forward = experiment.make_forward(
            run["architecture"],
            grid,
            system,
            wavelengths,
            indices,
            config_args.plane_separation_mm * 1e-3,
        )
        baseline_metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
        baseline_drift_um = baseline_metrics["wavelength_consistency"]["max_centroid_drift_m"] * 1e6
        baseline_worst_correlation = min(
            item["correlation"] for item in baseline_metrics["metrics_by_wavelength"]
        )
        result = experiment.optimize_stage(
            run["architecture"],
            target,
            forward,
            grid,
            config_args.height_max_um * 1e-6,
            args.iterations,
            run["seed"],
            args.position_weight,
            height,
            args.centroid_tolerance_um,
        )
        metrics, outputs, consistency, score = experiment.evaluate(result.height_m, forward, target, grid)
        refined_dir = run_dir / "drift_refined"
        refined_config = dict(saved_config)
        refined_config.update({
            "fine_iterations": args.iterations,
            "refinement_iterations": args.iterations,
            "refinement_position_weight": args.position_weight,
            "refinement_centroid_tolerance_um": args.centroid_tolerance_um,
            "refinement_parent_run": str(run_dir.relative_to(root)),
            "score": score,
        })
        experiment.save_run(
            refined_dir, target, outputs, result.height_m, result, grid,
            metrics, consistency, refined_config,
        )
        row = {
            "target": run["target"],
            "architecture": run["architecture"],
            "seed": run["seed"],
            "baseline_max_drift_um": baseline_drift_um,
            "refined_max_drift_um": consistency["max_centroid_drift_m"] * 1e6,
            "baseline_worst_correlation": baseline_worst_correlation,
            "refined_worst_correlation": min(item["correlation"] for item in metrics),
            "refined_mean_background_fraction": float(np.mean([item["background_fraction"] for item in metrics])),
            "refined_mean_roi_efficiency": float(np.mean([item["roi_efficiency"] for item in metrics])),
            "iterations": args.iterations,
            "position_weight": args.position_weight,
            "centroid_tolerance_um": args.centroid_tolerance_um,
            "refined_run_dir": str(refined_dir.relative_to(root)),
        }
        rows.append(row)
        print(
            f"[{index + 1}/{len(summary['runs'])}] {run['target']}/{run['architecture']}/seed={run['seed']} "
            f"drift {baseline_drift_um:.2f}->{row['refined_max_drift_um']:.2f}um; "
            f"worst corr {baseline_worst_correlation:.3f}->{row['refined_worst_correlation']:.3f}",
            flush=True,
        )
        del outputs, height, result, target, forward
        if device.type == "cuda":
            torch.cuda.empty_cache()

    with (root / "drift_refinement_comparison.csv").open(
        "w", newline="", encoding="utf-8-sig"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (root / "drift_refinement_comparison.json").write_text(
        json.dumps({
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "iterations": args.iterations,
            "position_weight": args.position_weight,
            "centroid_tolerance_um": args.centroid_tolerance_um,
            "runs": rows,
        }, indent=2),
        encoding="utf-8",
    )
    print(f"Refinement comparison saved under {root}", flush=True)


if __name__ == "__main__":
    main()
