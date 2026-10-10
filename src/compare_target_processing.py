"""Compare target-spot preprocessing under one fixed optimization objective.

All variants use the same two-wavelength forward model, amplitude-MSE loss,
Adam learning rate, initialization, and iteration count. Only the target
intensity is changed:

1. two Gaussian spots;
2. two soft circular spots;
3. soft circular spots followed by mild Gaussian smoothing;
4. a numerically generated two-PSF target.

Run from the repository root:
    py -3 src/compare_target_processing.py --iterations 300 --cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as functional

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


TARGET_NAMES = (
    "gaussian",
    "soft_circle",
    "soft_circle_smoothed",
    "numerical_psf",
)


def gaussian_kernel(size: int, sigma: float, device: torch.device) -> torch.Tensor:
    coordinate = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    kernel_1d = torch.exp(-(coordinate**2) / (2.0 * sigma**2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    return kernel_1d[:, None] * kernel_1d[None, :]


def gaussian_blur(image: torch.Tensor, size: int, sigma: float) -> torch.Tensor:
    kernel = gaussian_kernel(size, sigma, image.device)
    return functional.conv2d(
        image[None, None],
        kernel[None, None],
        padding=size // 2,
    )[0, 0]


def normalize_target(target: torch.Tensor) -> torch.Tensor:
    target = torch.clamp(target, min=0.0)
    return target / (target.sum() + 1e-12)


def soft_circular_spot(
    x: torch.Tensor,
    y: torch.Tensor,
    center_x: float,
    radius: float,
    edge_width: float,
) -> torch.Tensor:
    radial_distance = torch.sqrt((x - center_x) ** 2 + y**2 + 1e-12)
    return 0.5 * (
        1.0 - torch.tanh((radial_distance - radius) / edge_width)
    )


def make_soft_circle_target(
    x: torch.Tensor,
    y: torch.Tensor,
    config: Config,
    radius: float = 60e-6,
    edge_width: float = 12e-6,
) -> torch.Tensor:
    left = soft_circular_spot(x, y, -config.spot_x, radius, edge_width)
    right = soft_circular_spot(x, y, config.spot_x, radius, edge_width)
    return normalize_target(left + right)


def make_numerical_psf_target(
    x: torch.Tensor,
    y: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    aperture: torch.Tensor,
    transfer: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    """Generate a two-spot target from a single simulated circular-aperture PSF."""
    del x, y, fx, fy
    point_field = propagate(aperture.to(torch.complex64), transfer)
    point_intensity = normalized_intensity(point_field)
    point_intensity = torch.fft.fftshift(point_intensity)

    # Roll the numerically generated PSF to the two target positions.
    pixel_shift = round(config.spot_x / config.pixel_pitch)
    left = torch.roll(point_intensity, shifts=-pixel_shift, dims=1)
    right = torch.roll(point_intensity, shifts=pixel_shift, dims=1)
    return normalize_target(left + right)


def make_targets(
    x: torch.Tensor,
    y: torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    aperture: torch.Tensor,
    transfer: torch.Tensor,
    config: Config,
) -> dict[str, torch.Tensor]:
    gaussian = target_two_spots(x, y, config.spot_x, config.spot_sigma)
    soft_circle = make_soft_circle_target(x, y, config)
    soft_circle_smoothed = normalize_target(
        gaussian_blur(soft_circle, size=7, sigma=1.5)
    )
    numerical_psf = make_numerical_psf_target(
        x, y, fx, fy, aperture, transfer, config
    )
    return {
        "gaussian": gaussian,
        "soft_circle": soft_circle,
        "soft_circle_smoothed": soft_circle_smoothed,
        "numerical_psf": numerical_psf,
    }


def amplitude_mse_loss(
    output_intensity: torch.Tensor,
    target_intensity: torch.Tensor,
    height: torch.Tensor,
    config: Config,
) -> torch.Tensor:
    output_amplitude = torch.sqrt(output_intensity + 1e-12)
    target_amplitude = torch.sqrt(target_intensity + 1e-12)
    shape_loss = torch.mean((output_amplitude - target_amplitude) ** 2)
    smoothness = (
        torch.mean((height[:, 1:] - height[:, :-1]) ** 2)
        + torch.mean((height[1:, :] - height[:-1, :]) ** 2)
    ) / (config.height_max**2)
    return shape_loss + 1e-4 * smoothness


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


def radiality_score(
    intensity: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x: float,
    radius: float,
) -> float:
    """Measure left/right radial symmetry inside the two spot neighborhoods."""
    scores = []
    for center_x in (-spot_x, spot_x):
        local_x = x - center_x
        local_radius = torch.sqrt(local_x**2 + y**2 + 1e-12)
        region = local_radius <= radius
        reflected = torch.flip(intensity, dims=(0, 1))
        reflected_local_x = torch.flip(local_x, dims=(0, 1))
        reflected_region = reflected_local_x**2 + torch.flip(y, dims=(0, 1))**2 <= radius**2
        overlap = region & reflected_region
        if torch.any(overlap):
            scores.append(
                float(
                    torch.mean(
                        torch.abs(intensity[overlap] - reflected[overlap])
                    ).cpu()
                )
            )
    return float(np.mean(scores)) if scores else float("nan")


def optimize_target(
    config: Config,
    target: torch.Tensor,
    aperture: torch.Tensor,
    transfers: list[torch.Tensor],
    iterations: int,
    learning_rate_um: float,
    device: torch.device,
    seed: int,
) -> dict[str, object]:
    generator = torch.Generator(device=device).manual_seed(seed)
    initial_height = config.height_max * torch.rand(
        (config.n_grid, config.n_grid),
        generator=generator,
        device=device,
    )
    height = torch.nn.Parameter(initial_height)
    optimizer = torch.optim.Adam([height], lr=learning_rate_um * 1e-6)
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
            total_loss = total_loss + amplitude_mse_loss(
                output_intensity,
                target,
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
        correlations.append(float(correlation(output_intensity, target).cpu()))

        radius = 0.11e-3
        spot_mask = (
            (((torch.arange(config.n_grid, device=device) - config.n_grid // 2) * config.pixel_pitch)[None, :] + config.spot_x) ** 2
            + (((torch.arange(config.n_grid, device=device) - config.n_grid // 2) * config.pixel_pitch)[:, None]) ** 2
            <= radius**2
        )
        spot_mask = spot_mask | torch.roll(spot_mask, shifts=round(2 * config.spot_x / config.pixel_pitch), dims=1)
        background_fractions.append(
            float(torch.sum(output_intensity * (~spot_mask)).cpu())
        )

    return {
        "height": height.detach().cpu().numpy(),
        "intensities": intensities,
        "losses": losses,
        "correlations": correlations,
        "background_fractions": background_fractions,
    }


def save_outputs(
    output_dir: Path,
    config: Config,
    x: torch.Tensor,
    targets: dict[str, torch.Tensor],
    results: dict[str, dict[str, object]],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_mm = x.detach().cpu().numpy() * 1e3
    extent = (float(x_mm.min()), float(x_mm.max()), float(x_mm.min()), float(x_mm.max()))

    figure, axes = plt.subplots(1, len(targets), figsize=(16, 4), constrained_layout=True)
    for axis, name in zip(axes, TARGET_NAMES):
        target = targets[name].detach().cpu().numpy()
        axis.imshow(target, cmap="inferno", extent=extent, origin="lower")
        axis.set_title(name)
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    figure.savefig(output_dir / "targets.png", dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(
        len(TARGET_NAMES), 2,
        figsize=(10, 4 * len(TARGET_NAMES)),
        constrained_layout=True,
    )
    for row, name in enumerate(TARGET_NAMES):
        result = results[name]
        for wavelength_index, axis in enumerate(axes[row]):
            image = result["intensities"][wavelength_index]
            display = 10.0 * np.log10(np.maximum(image, 1e-12) / image.max())
            axis.imshow(display, cmap="inferno", extent=extent, origin="lower", vmin=-30, vmax=0)
            axis.set_title(
                f"{name}, {config.wavelengths[wavelength_index] * 1e9:.0f} nm\n"
                f"corr={result['correlations'][wavelength_index]:.3f}, "
                f"bg={result['background_fractions'][wavelength_index]:.3f}"
            )
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("y (mm)")
    figure.savefig(output_dir / "outputs.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for name in TARGET_NAMES:
        axis.semilogy(results[name]["losses"], label=name)
    axis.set_xlabel("iteration")
    axis.set_ylabel("fixed amplitude-MSE loss")
    axis.set_title("Convergence under the same objective")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "losses.png", dpi=180)
    plt.close(figure)

    summary = {
        name: {
            "correlations": result["correlations"],
            "background_fractions": result["background_fractions"],
            "final_loss": result["losses"][-1],
        }
        for name, result in results.items()
    }
    (output_dir / "summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    np.savez_compressed(
        output_dir / "target_processing_arrays.npz",
        x=x.detach().cpu().numpy(),
        **{
            f"target_{name}": target.detach().cpu().numpy()
            for name, target in targets.items()
        },
        **{
            f"height_{name}": result["height"]
            for name, result in results.items()
        },
        **{
            f"intensity_{name}_{index}": result["intensities"][index]
            for name, result in results.items()
            for index in range(len(config.wavelengths))
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--output-dir", type=Path, default=Path("results/target_processing"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(config.seed)
    print(f"device: {device}")
    print(f"fixed loss: amplitude MSE, learning rate: {args.learning_rate} um")

    x, y, fx, fy, aperture = make_grid(config, device)
    transfers = [
        angular_spectrum_transfer(
            config.wavelengths[index], fx, fy, config.propagation_distance
        )
        for index in range(len(config.wavelengths))
    ]
    targets = make_targets(x, y, fx, fy, aperture, transfers[0], config)

    results: dict[str, dict[str, object]] = {}
    for index, name in enumerate(TARGET_NAMES):
        print(f"\n[{index + 1}/{len(TARGET_NAMES)}] target={name}")
        result = optimize_target(
            config,
            targets[name],
            aperture,
            transfers,
            args.iterations,
            args.learning_rate,
            device,
            seed=config.seed + 100,
        )
        results[name] = result
        print(
            "  correlation:",
            [round(value, 4) for value in result["correlations"]],
            "background:",
            [round(value, 4) for value in result["background_fractions"]],
        )

    save_outputs(args.output_dir, config, x, targets, results)
    print(f"\nsaved target-processing comparison to {args.output_dir}")


if __name__ == "__main__":
    main()
