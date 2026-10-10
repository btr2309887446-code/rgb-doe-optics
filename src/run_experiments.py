"""Run modular multicolor DOE experiments.

Examples:
    py -3 src/run_experiments.py --iterations 20 --n-grid 64 --cpu --self-test
    py -3 src/run_experiments.py --wavelengths 532 633 --algorithm compare --iterations 100 --cpu
    py -3 src/run_experiments.py --model thin_lens --focal-length-mm 50 --magnification 1 --cpu
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from forward_models import (
    OpticalSystem,
    energy_relative_error,
    forward_field,
    make_grid,
    validate_sampling,
    transfer_identity_error,
)
from losses import LossSchedule, LossWeights
from materials import CauchyMaterial
from metrics import evaluate_output
from optimizers import AdamOptions, optimize_adam, optimize_dbs
from targets import TargetSpec, soft_circle_two_spots


def parse_wavelengths(values: list[float]) -> tuple[float, ...]:
    if not values:
        raise ValueError("at least one wavelength is required")
    return tuple(value * 1e-9 for value in values)


def thin_lens_distances(focal_length_m: float, magnification: float) -> tuple[float, float]:
    if focal_length_m <= 0 or magnification <= 0:
        raise ValueError("focal length and magnification must be positive")
    return focal_length_m * (1.0 + 1.0 / magnification), focal_length_m * (1.0 + magnification)


def make_system(args: argparse.Namespace) -> OpticalSystem:
    common = {
        "n_grid": args.n_grid,
        "pixel_pitch_m": args.pixel_pitch_um * 1e-6,
        "pad_factor": args.pad_factor,
        "aperture_radius_m": args.aperture_radius_um * 1e-6,
    }
    if args.model == "thin_lens":
        focal_length_m = args.focal_length_mm * 1e-3
        object_distance_m, image_distance_m = thin_lens_distances(focal_length_m, args.magnification)
        return OpticalSystem(
            **common,
            model="thin_lens",
            doe_to_lens_distance_m=object_distance_m,
            lens_to_image_distance_m=image_distance_m,
            lens_focal_length_m=focal_length_m,
            lens_aperture_radius_m=args.lens_aperture_um * 1e-6 if args.lens_aperture_um else None,
        )
    return OpticalSystem(**common, model="free_space", propagation_distance_m=args.propagation_distance_mm * 1e-3)


def make_target(args: argparse.Namespace, grid) -> TargetSpec:
    return soft_circle_two_spots(
        grid.base_x,
        grid.base_y,
        spot_x_m=args.spot_x_um * 1e-6,
        radius_m=args.radius_um * 1e-6,
        edge_width_m=args.edge_width_um * 1e-6,
    )


def plot_outputs(path: Path, outputs: list[torch.Tensor], wavelengths_m: tuple[float, ...], grid, title: str) -> None:
    columns = len(outputs)
    figure, axes = plt.subplots(1, columns, figsize=(4.0 * columns, 4.0), squeeze=False, constrained_layout=True)
    extent = (
        float(grid.base_x.min().cpu() * 1e3),
        float(grid.base_x.max().cpu() * 1e3),
        float(grid.base_y.min().cpu() * 1e3),
        float(grid.base_y.max().cpu() * 1e3),
    )
    for index, field in enumerate(outputs):
        intensity = torch.abs(field).square().detach().cpu().numpy()
        intensity /= intensity.sum() + 1e-12
        display = 10.0 * np.log10(np.maximum(intensity, 1e-12) / intensity.max())
        axes[0, index].imshow(display, cmap="inferno", origin="lower", extent=extent, vmin=-40, vmax=0)
        axes[0, index].set_title(f"{wavelengths_m[index] * 1e9:.0f} nm")
        axes[0, index].set_xlabel("x (mm)")
        axes[0, index].set_ylabel("y (mm)")
    figure.suptitle(title)
    figure.savefig(path, dpi=160)
    plt.close(figure)



def validate_index_sensitivity(
    height_m: torch.Tensor,
    grid,
    system: OpticalSystem,
    material: CauchyMaterial,
    target: TargetSpec,
    wavelengths_m: tuple[float, ...],
    delta_n: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for wavelength_m in wavelengths_m:
        nominal = material.index(wavelength_m)
        cases = []
        for label, index in (("minus", nominal - delta_n), ("nominal", nominal), ("plus", nominal + delta_n)):
            field = forward_field(height_m, grid, system, wavelength_m, index)
            cases.append({
                "label": label,
                "refractive_index": index,
                "metrics": evaluate_output(field, target, grid.base_x, grid.base_y, system.pixel_pitch_m),
            })
        rows.append({"wavelength_nm": wavelength_m * 1e9, "cases": cases})
    return rows

def validate_band(
    height_m: torch.Tensor,
    grid,
    system: OpticalSystem,
    material: CauchyMaterial,
    target: TargetSpec,
    start_nm: float,
    stop_nm: float,
    step_nm: float,
) -> list[dict[str, object]]:
    values = np.arange(start_nm, stop_nm + 0.5 * step_nm, step_nm)
    validation: list[dict[str, object]] = []
    for wavelength_nm in values:
        wavelength_m = float(wavelength_nm) * 1e-9
        field = forward_field(height_m, grid, system, wavelength_m, material.index(wavelength_m))
        validation.append({
            "wavelength_nm": float(wavelength_nm),
            "refractive_index": material.index(wavelength_m),
            "metrics": evaluate_output(field, target, grid.base_x, grid.base_y, system.pixel_pitch_m),
        })
    return validation


def run_one(
    args: argparse.Namespace,
    wavelengths_m: tuple[float, ...],
    system: OpticalSystem,
    output_dir: Path,
    algorithm: str,
    seed: int,
    initial_height: torch.Tensor | None = None,
) -> tuple[dict[str, object], torch.Tensor]:
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    grid = make_grid(system, device)
    sampling = validate_sampling(system, grid)
    material = CauchyMaterial()
    indices = material.indices(wavelengths_m)
    target = make_target(args, grid)
    base_aperture = grid.doe_aperture[grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end]

    def forward(height_m: torch.Tensor) -> list[torch.Tensor]:
        return [
            forward_field(height_m, grid, system, wavelength, refractive_index)
            for wavelength, refractive_index in zip(wavelengths_m, indices)
        ]

    weights = LossWeights(
        alpha_shape=1.0,
        beta_background=args.background_weight,
        gamma_efficiency=args.efficiency_weight,
        delta_balance=args.balance_weight,
        eta_smooth=args.smoothness_weight,
    )
    schedule = LossSchedule(args.shape_only_until, args.background_ramp_end)
    start = time.perf_counter()
    if algorithm in ("adam", "mraf"):
        options = AdamOptions(
            iterations=args.iterations,
            learning_rate_um=args.learning_rate_um,
            gradient_clip=args.gradient_clip,
            cosine_decay=not args.no_cosine_decay,
            seed=seed,
            wavelength_weights=tuple(args.wavelength_weights) if args.wavelength_weights else None,
            mraf_background_bias=args.mraf_bias if algorithm == "mraf" else None,
            mraf_mixing=args.mraf_mixing,
        )
        result = optimize_adam(
            forward, target, args.height_max_um * 1e-6,
            system.n_grid, base_aperture, options, weights, schedule, initial_height,
        )
    elif algorithm == "dbs":
        if initial_height is None:
            generator = torch.Generator(device=device).manual_seed(seed)
            initial_height = args.height_max_um * 1e-6 * torch.rand(
                (system.n_grid, system.n_grid), generator=generator, device=device
            )
        result = optimize_dbs(
            forward, target, initial_height, args.height_max_um * 1e-6,
            base_aperture, weights, levels=args.dbs_levels, sweeps=args.dbs_sweeps,
            seed=seed, max_pixels_per_sweep=args.dbs_max_pixels,
        )
    else:
        raise ValueError(f"unknown algorithm: {algorithm}")
    elapsed = time.perf_counter() - start
    outputs = forward(result.height_m)
    metrics = [
        evaluate_output(field, target, grid.base_x, grid.base_y, system.pixel_pitch_m)
        for field in outputs
    ]
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_outputs(
        output_dir / f"{algorithm}_seed{seed}.png",
        outputs,
        wavelengths_m,
        grid,
        f"{algorithm}, {system.model}",
    )
    height_path = output_dir / f"{algorithm}_seed{seed}_height.npy"
    np.save(height_path, result.height_m.detach().cpu().numpy())
    payload = {
        "algorithm": result.method,
        "seed": seed,
        "elapsed_seconds": elapsed,
        "system": asdict(system),
        "wavelengths_nm": [value * 1e9 for value in wavelengths_m],
        "refractive_indices": list(indices),
        "material": material.metadata(wavelengths_m),
        "sampling": sampling,
        "target": {
            "radius_um": args.radius_um,
            "edge_width_um": args.edge_width_um,
            "spot_x_um": args.spot_x_um,
        },
        "loss_weights": asdict(weights),
        "loss_schedule": asdict(schedule),
        "metrics": metrics,
        "final_loss": result.losses[-1] if result.losses else None,
        "loss_history": result.losses,
        "component_history": result.components,
        "height_file": str(height_path),
    }
    if args.validate_band:
        payload["band_validation"] = validate_band(
            result.height_m, grid, system, material, target,
            args.band_start_nm, args.band_stop_nm, args.band_step_nm,
        )
    if args.index_sensitivity:
        payload["index_sensitivity"] = validate_index_sensitivity(
            result.height_m, grid, system, material, target, wavelengths_m, args.index_delta_n,
        )
    json_path = output_dir / f"{algorithm}_seed{seed}.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload, result.height_m


def run_curriculum(
    args: argparse.Namespace,
    wavelengths_m: tuple[float, ...],
    system: OpticalSystem,
    output_dir: Path,
    seed: int,
) -> dict[str, object]:
    """Run the planned 532/633 pretraining followed by four-wavelength tuning."""
    if tuple(round(value * 1e9) for value in wavelengths_m) != (488, 532, 633, 660):
        raise ValueError("--curriculum requires wavelengths 488 532 633 660")
    center_indices = (1, 2)
    center_wavelengths = tuple(wavelengths_m[index] for index in center_indices)
    pre_args = argparse.Namespace(**vars(args))
    pre_args.iterations = args.curriculum_pre_iterations
    pre_args.wavelength_weights = None
    pre_args.validate_band = False
    pre_dir = output_dir / "curriculum_pretrain"
    pre_payload, pre_height = run_one(pre_args, center_wavelengths, system, pre_dir, "adam", seed)
    final_payload, final_height = run_one(
        args, wavelengths_m, system, output_dir / "curriculum_finetune", "adam", seed,
        initial_height=pre_height,
    )
    combined = {
        "algorithm": "curriculum_adam",
        "seed": seed,
        "pretrain": pre_payload,
        "finetune": final_payload,
        "height_file": final_payload["height_file"],
    }
    return combined


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wavelengths", type=float, nargs="+", default=[488, 532, 633, 660])
    parser.add_argument("--algorithm", choices=["adam", "mraf", "dbs", "compare"], default="adam")
    parser.add_argument("--curriculum", action="store_true", help="pretrain on 532/633 nm, then continue on all wavelengths")
    parser.add_argument("--model", choices=["free_space", "thin_lens"], default="free_space")
    parser.add_argument("--n-grid", type=int, default=256)
    parser.add_argument("--pixel-pitch-um", type=float, default=10.0)
    parser.add_argument("--aperture-radius-um", type=float, default=1000.0)
    parser.add_argument("--pad-factor", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--propagation-distance-mm", type=float, default=50.0)
    parser.add_argument("--focal-length-mm", type=float, default=50.0)
    parser.add_argument("--magnification", type=float, default=1.0)
    parser.add_argument("--lens-aperture-um", type=float)
    parser.add_argument("--iterations", type=int, default=600)
    parser.add_argument("--curriculum-pre-iterations", type=int, default=300)
    parser.add_argument("--learning-rate-um", type=float, default=0.04)
    parser.add_argument("--height-max-um", type=float, default=1.4)
    parser.add_argument("--radius-um", type=float, default=40.0)
    parser.add_argument("--edge-width-um", type=float, default=15.0)
    parser.add_argument("--spot-x-um", type=float, default=300.0)
    parser.add_argument("--background-weight", type=float, default=0.25)
    parser.add_argument("--efficiency-weight", type=float, default=0.10)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=1e-5)
    parser.add_argument("--shape-only-until", type=int, default=100)
    parser.add_argument("--background-ramp-end", type=int, default=300)
    parser.add_argument("--mraf-bias", type=float, default=0.05)
    parser.add_argument("--mraf-mixing", type=float, default=0.05)
    parser.add_argument("--wavelength-weights", type=float, nargs="+")
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--no-cosine-decay", action="store_true")
    parser.add_argument("--dbs-levels", type=int, default=64)
    parser.add_argument("--dbs-sweeps", type=int, default=10)
    parser.add_argument("--dbs-max-pixels", type=int)
    parser.add_argument("--seeds", type=int, nargs="+", default=[7])
    parser.add_argument("--validate-band", action="store_true")
    parser.add_argument("--index-sensitivity", action="store_true")
    parser.add_argument("--index-delta-n", type=float, default=0.005)
    parser.add_argument("--band-start-nm", type=float, default=488.0)
    parser.add_argument("--band-stop-nm", type=float, default=660.0)
    parser.add_argument("--band-step-nm", type=float, default=10.0)
    parser.add_argument("--output-dir", type=Path, default=Path("results/modular_experiments"))
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    wavelengths_m = parse_wavelengths(args.wavelengths)
    if args.wavelength_weights and len(args.wavelength_weights) != len(wavelengths_m):
        parser.error("--wavelength-weights must match --wavelengths")
    try:
        system = make_system(args)
    except ValueError as error:
        parser.error(str(error))
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    if args.self_test:
        print(json.dumps({
            "identity_error": transfer_identity_error(system, device),
            "energy_error": energy_relative_error(system, device),
        }, indent=2))
        return
    all_results: list[dict[str, object]] = []
    if args.curriculum:
        for seed in args.seeds:
            print(
                f"running curriculum Adam, model={system.model}, seed={seed}, "
                f"wavelengths={[round(value * 1e9) for value in wavelengths_m]}"
            )
            all_results.append(run_curriculum(args, wavelengths_m, system, args.output_dir, seed))
    else:
        for seed in args.seeds:
            print(
                f"running algorithm={args.algorithm}, model={system.model}, seed={seed}, "
                f"wavelengths={[round(value * 1e9) for value in wavelengths_m]}"
            )
            if args.algorithm == "compare":
                adam_result, adam_height = run_one(args, wavelengths_m, system, args.output_dir / "adam", "adam", seed)
                mraf_result, _ = run_one(args, wavelengths_m, system, args.output_dir / "mraf", "mraf", seed, initial_height=adam_height)
                dbs_result, _ = run_one(args, wavelengths_m, system, args.output_dir / "dbs", "dbs", seed, initial_height=adam_height)
                all_results.extend([adam_result, mraf_result, dbs_result])
            else:
                result, _ = run_one(args, wavelengths_m, system, args.output_dir, args.algorithm, seed)
                all_results.append(result)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    archive = {}
    for result in all_results:
        key = f"{result.get('algorithm', 'experiment')}_seed{result.get('seed', 0)}"
        height_file = result.get("height_file")
        if height_file:
            archive[key] = np.load(height_file)
    if archive:
        np.savez_compressed(args.output_dir / "heights.npz", **archive)
    print(f"saved {len(all_results)} experiment(s) to {args.output_dir}")


if __name__ == "__main__":
    main()
