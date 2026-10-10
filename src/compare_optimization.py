"""Sweep loss functions and learning rates for the two-wavelength two-spot DOE.

This script keeps the physical model fixed and compares objective functions and
Adam learning rates. It is intended to answer whether the rough spot edges are
caused mainly by the loss function or by the optimizer step size.

Run from the repository root:
    py -3 src/compare_optimization.py --iterations 200 --cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_spot_multicolor import (  # noqa: E402
    Config,
    angular_spectrum_transfer,
    image_plane_field,
    make_grid,
    normalized_intensity,
    propagate,
    target_two_spots,
)


LOSS_NAMES = ("amplitude_mse", "intensity_mse", "roi_background")


def make_spot_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x: float,
    spot_radius: float,
) -> torch.Tensor:
    left = (x + spot_x) ** 2 + y**2 <= spot_radius**2
    right = (x - spot_x) ** 2 + y**2 <= spot_radius**2
    return (left | right).to(torch.float32)


def smoothness_loss(height: torch.Tensor, height_max: float) -> torch.Tensor:
    return (
        torch.mean((height[:, 1:] - height[:, :-1]) ** 2)
        + torch.mean((height[1:, :] - height[:-1, :]) ** 2)
    ) / (height_max**2)


def correlation(
    intensity: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    intensity_centered = intensity - intensity.mean()
    target_centered = target - target.mean()
    denominator = torch.sqrt(
        torch.sum(intensity_centered**2)
        * torch.sum(target_centered**2)
        + 1e-12
    )
    return torch.sum(intensity_centered * target_centered) / denominator


def loss_for_variant(
    loss_name: str,
    output_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
    spot_mask: torch.Tensor,
    target_spot_energy: torch.Tensor,
    height: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    output_amplitude = torch.sqrt(output_intensity + 1e-12)
    smoothness = smoothness_loss(height, config.height_max)

    if loss_name == "amplitude_mse":
        pattern_loss = torch.mean((output_amplitude - target_amplitude) ** 2)
        return pattern_loss + 1e-4 * smoothness

    if loss_name == "intensity_mse":
        pattern_loss = torch.mean((output_intensity - target_intensity) ** 2)
        return pattern_loss + 1e-4 * smoothness

    if loss_name == "roi_background":
        region_weight = 1.0 + 4.0 * spot_mask
        shape_loss = torch.mean(
            region_weight * (output_amplitude - target_amplitude) ** 2
        )
        background = 1.0 - spot_mask
        background_mean = torch.sum(output_intensity * background) / (
            torch.sum(background) + 1e-12
        )
        spot_energy = torch.sum(output_intensity * spot_mask)
        energy_loss = (spot_energy - target_spot_energy) ** 2
        return (
            shape_loss
            + 3.0 * background_mean
            + 0.02 * energy_loss
            + 2e-4 * smoothness
        )

    raise ValueError(f"Unknown loss function: {loss_name}")


def run_variant(
    config: Config,
    target_intensity: torch.Tensor,
    target_amplitude: torch.Tensor,
    spot_mask: torch.Tensor,
    aperture: torch.Tensor,
    transfers: list[torch.Tensor],
    loss_name: str,
    learning_rate_um: float,
    iterations: int,
    device: torch.device,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(config.seed + 101)
    initial_height = config.height_max * torch.rand(
        (config.n_grid, config.n_grid),
        generator=generator,
        device=device,
    )
    height = torch.nn.Parameter(initial_height)
    optimizer = torch.optim.Adam(
        [height],
        lr=learning_rate_um * 1e-6,
    )
    target_spot_energy = torch.sum(target_intensity * spot_mask)
    beta = [
        2.0 * torch.pi * (n_refractive - 1.0) / wavelength
        for wavelength, n_refractive in zip(
            config.wavelengths,
            config.refractive_indices,
        )
    ]
    losses: list[float] = []

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        for wavelength_index in range(len(config.wavelengths)):
            output_field = image_plane_field(
                height,
                aperture,
                config.wavelengths[wavelength_index],
                config.refractive_indices[wavelength_index],
                transfers[wavelength_index],
            )
            output_intensity = normalized_intensity(output_field)
            total_loss = total_loss + loss_for_variant(
                loss_name,
                output_intensity,
                target_intensity,
                target_amplitude,
                spot_mask,
                target_spot_energy,
                height,
                config,
            )

        total_loss = total_loss / len(config.wavelengths)
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            height.clamp_(0.0, config.height_max)
            height.mul_(aperture)
        losses.append(float(total_loss.detach().cpu()))

    intensities = []
    correlations = []
    background_fractions = []
    spot_fractions = []
    with torch.no_grad():
        for wavelength_index in range(len(config.wavelengths)):
            output_field = image_plane_field(
                height,
                aperture,
                config.wavelengths[wavelength_index],
                config.refractive_indices[wavelength_index],
                transfers[wavelength_index],
            )
            output_intensity = normalized_intensity(output_field)
            intensities.append(output_intensity.detach().cpu().numpy())
            correlations.append(
                float(correlation(output_intensity, target_intensity).cpu())
            )
            background = 1.0 - spot_mask
            background_fractions.append(
                float(torch.sum(output_intensity * background).cpu())
            )
            spot_fractions.append(
                float(torch.sum(output_intensity * spot_mask).cpu())
            )

    return {
        "loss_name": loss_name,
        "learning_rate_um": learning_rate_um,
        "height": height.detach().cpu().numpy(),
        "intensities": intensities,
        "losses": losses,
        "correlations": correlations,
        "background_fractions": background_fractions,
        "spot_fractions": spot_fractions,
    }


def save_sweep_outputs(
    output_dir: Path,
    config: Config,
    x: torch.Tensor,
    target_intensity: torch.Tensor,
    results: list[dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_mm = x.detach().cpu().numpy() * 1e3
    extent = (float(x_mm.min()), float(x_mm.max()), float(x_mm.min()), float(x_mm.max()))

    target = target_intensity.detach().cpu().numpy()
    figure, axis = plt.subplots(figsize=(5, 4), constrained_layout=True)
    axis.imshow(target, cmap="inferno", extent=extent, origin="lower")
    axis.set_title("Target two spots")
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    figure.savefig(output_dir / "target.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(results),
        2,
        figsize=(10, 3.3 * len(results)),
        squeeze=False,
        constrained_layout=True,
    )
    for row, result in enumerate(results):
        label = f"{result['loss_name']}, lr={result['learning_rate_um']} um"
        for wavelength_index, axis in enumerate(axes[row]):
            image = result["intensities"][wavelength_index]
            display = 10.0 * np.log10(np.maximum(image, 1e-12) / image.max())
            axis.imshow(
                display,
                cmap="inferno",
                extent=extent,
                origin="lower",
                vmin=-30,
                vmax=0,
            )
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("y (mm)")
            axis.set_title(
                f"{label}\n"
                f"{config.wavelengths[wavelength_index] * 1e9:.0f} nm, "
                f"corr={result['correlations'][wavelength_index]:.3f}, "
                f"bg={result['background_fractions'][wavelength_index]:.3f}"
            )
    figure.savefig(output_dir / "all_results.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    for result in results:
        label = f"{result['loss_name']} / {result['learning_rate_um']}"
        axes[0].plot(result["losses"], label=label)
        axes[1].plot(
            [result["correlations"][0]] * len(config.wavelengths),
            label=label,
        )
    axes[0].set_yscale("log")
    axes[0].set_title("Training loss")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("loss")
    axes[1].set_title("Final 532 nm correlation")
    axes[1].set_ylabel("correlation")
    axes[1].set_xticks([])
    axes[0].grid(alpha=0.25)
    axes[1].grid(alpha=0.25)
    axes[0].legend(fontsize=8)
    figure.savefig(output_dir / "training_comparison.png", dpi=180)
    plt.close(figure)

    summary = []
    for result in results:
        summary.append(
            {
                "loss_name": result["loss_name"],
                "learning_rate_um": result["learning_rate_um"],
                "final_loss": result["losses"][-1],
                "correlations": result["correlations"],
                "background_fractions": result["background_fractions"],
                "spot_fractions": result["spot_fractions"],
            }
        )
    metadata = {
        "config": asdict(config),
        "results": summary,
    }
    (output_dir / "summary.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "sweep_arrays.npz",
        x=x.detach().cpu().numpy(),
        target=target,
        **{
            f"height_{index}": result["height"]
            for index, result in enumerate(results)
        },
        **{
            f"intensity_{index}_{wavelength_index}": result["intensities"][wavelength_index]
            for index, result in enumerate(results)
            for wavelength_index in range(len(config.wavelengths))
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=200)
    parser.add_argument(
        "--learning-rates",
        type=float,
        nargs="+",
        default=[0.01, 0.04, 0.10],
        help="Adam learning rates in micrometres per step",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/optimization_sweep"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    config = Config()
    torch.manual_seed(config.seed)
    print(f"device: {device}")
    print(f"losses: {LOSS_NAMES}")
    print(f"learning rates (um): {args.learning_rates}")

    x, y, fx, fy, aperture = make_grid(config, device)
    target_intensity = target_two_spots(x, y, config.spot_x, config.spot_sigma)
    target_amplitude = torch.sqrt(target_intensity)
    spot_mask = make_spot_mask(x, y, config.spot_x, spot_radius=0.11e-3)
    transfers = [
        angular_spectrum_transfer(
            config.wavelengths[index],
            fx,
            fy,
            config.propagation_distance,
        )
        for index in range(len(config.wavelengths))
    ]

    results: list[dict[str, object]] = []
    for loss_name in LOSS_NAMES:
        for learning_rate_um in args.learning_rates:
            print(f"\n[{loss_name}, lr={learning_rate_um} um]")
            result = run_variant(
                config,
                target_intensity,
                target_amplitude,
                spot_mask,
                aperture,
                transfers,
                loss_name,
                learning_rate_um,
                args.iterations,
                device,
            )
            results.append(result)
            print(
                "  correlation:",
                [round(value, 4) for value in result["correlations"]],
                "background:",
                [round(value, 4) for value in result["background_fractions"]],
            )

    save_sweep_outputs(
        args.output_dir,
        config,
        x,
        target_intensity,
        results,
    )
    print(f"\nsaved sweep to {args.output_dir}")


if __name__ == "__main__":
    main()