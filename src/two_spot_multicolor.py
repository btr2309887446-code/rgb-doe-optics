"""Two-wavelength DOE design for a common two-spot target.

The DOE is represented by one physical height map h(x, y).  Each wavelength
therefore sees a different phase

    phi_lambda(x, y) = 2*pi*(n(lambda)-1)*h(x, y)/lambda.

The forward model is a complete scalar image-plane path:

    uniform incident field -> finite DOE aperture -> air propagation
    -> common physical image plane.

The image plane is the plane 5 cm after the DOE. This is a real-space
observation plane, not the Fourier plane of a lens. PyTorch is
used only for automatic differentiation; NumPy and Matplotlib are used for
saving results.

Run from the repository root:

    py -3 src/two_spot_multicolor.py

The first run may take a few minutes on CPU.  Use --iterations to shorten it.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    n_grid: int = 256
    pixel_pitch: float = 10e-6
    propagation_distance: float = 0.05
    aperture_radius: float = 1.0e-3
    spot_x: float = 0.30e-3
    spot_sigma: float = 35e-6
    height_max: float = 1.4e-6
    refractive_indices: tuple[float, float] = (1.6482, 1.6347)
    refractive_index_source: str = "Shipley S1813; Kim et al. (2012)"
    wavelengths: tuple[float, float] = (532e-9, 633e-9)
    seed: int = 7


def make_grid(config: Config, device: torch.device) -> tuple[torch.Tensor, ...]:
    n = config.n_grid
    coordinate = (torch.arange(n, device=device) - n // 2) * config.pixel_pitch
    x, y = torch.meshgrid(coordinate, coordinate, indexing="xy")
    fx = torch.fft.fftfreq(n, d=config.pixel_pitch, device=device)
    fy = torch.fft.fftfreq(n, d=config.pixel_pitch, device=device)
    fx, fy = torch.meshgrid(fx, fy, indexing="xy")
    aperture = ((x**2 + y**2) <= config.aperture_radius**2).to(torch.float32)
    return x, y, fx, fy, aperture


def angular_spectrum_transfer(
    wavelength: float,
    fx: torch.Tensor,
    fy: torch.Tensor,
    distance: float,
) -> torch.Tensor:
    spatial_frequency_radius_squared = fx**2 + fy**2
    propagating = spatial_frequency_radius_squared <= (1.0 / wavelength) ** 2
    kz_cycles = torch.sqrt(
        torch.clamp((1.0 / wavelength) ** 2 - spatial_frequency_radius_squared, min=0.0)
    )
    transfer = torch.exp(1j * 2.0 * torch.pi * distance * kz_cycles)
    return transfer * propagating


def make_doe_field(
    height: torch.Tensor,
    aperture: torch.Tensor,
    wavelength: float,
    refractive_index: float,
) -> torch.Tensor:
    """Return the complex field immediately after a phase-only DOE."""
    phase = 2.0 * torch.pi * (refractive_index - 1.0) * height / wavelength
    return aperture.to(torch.complex64) * torch.exp(1j * phase)


def target_two_spots(
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x: float,
    spot_sigma: float,
) -> torch.Tensor:
    left = torch.exp(-((x + spot_x) ** 2 + y**2) / (2.0 * spot_sigma**2))
    right = torch.exp(-((x - spot_x) ** 2 + y**2) / (2.0 * spot_sigma**2))
    target = left + right
    return target / target.sum()


def propagate(
    field: torch.Tensor,
    transfer: torch.Tensor,
) -> torch.Tensor:
    """Propagate a field whose spatial origin is at the array centre."""
    field_fft_order = torch.fft.ifftshift(field, dim=(-2, -1))
    propagated_fft_order = torch.fft.ifft2(
        torch.fft.fft2(field_fft_order) * transfer
    )
    return torch.fft.fftshift(propagated_fft_order, dim=(-2, -1))


def image_plane_field(
    height: torch.Tensor,
    aperture: torch.Tensor,
    wavelength: float,
    refractive_index: float,
    transfer: torch.Tensor,
) -> torch.Tensor:
    """Propagate the DOE field to the common physical image plane."""
    doe_field = make_doe_field(height, aperture, wavelength, refractive_index)
    return propagate(doe_field, transfer)


def normalized_amplitude(intensity_field: torch.Tensor) -> torch.Tensor:
    amplitude = torch.abs(intensity_field)
    return amplitude / torch.sqrt(torch.sum(amplitude**2) + 1e-12)


def normalized_intensity(field: torch.Tensor) -> torch.Tensor:
    intensity = torch.abs(field) ** 2
    return intensity / (intensity.sum() + 1e-12)


def design_doe(
    config: Config,
    target_amplitude: torch.Tensor,
    aperture: torch.Tensor,
    transfers: list[torch.Tensor],
    wavelengths_to_fit: list[int],
    iterations: int,
    learning_rate: float,
    device: torch.device,
) -> tuple[torch.Tensor, list[float]]:
    generator = torch.Generator(device=device).manual_seed(config.seed + len(wavelengths_to_fit))
    initial_height = config.height_max * torch.rand(
        (config.n_grid, config.n_grid), generator=generator, device=device
    )
    height = torch.nn.Parameter(initial_height)
    optimizer = torch.optim.Adam([height], lr=learning_rate)
    losses: list[float] = []

    for iteration in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        total_loss = torch.zeros((), device=device)
        phase_mask_losses = []
        for wavelength_index in wavelengths_to_fit:
            output_field = image_plane_field(
                height,
                aperture,
                config.wavelengths[wavelength_index],
                config.refractive_indices[wavelength_index],
                transfers[wavelength_index],
            )
            output_amplitude = normalized_amplitude(output_field)
            amplitude_loss = torch.mean((output_amplitude - target_amplitude) ** 2)
            phase_mask_losses.append(amplitude_loss)
            total_loss = total_loss + amplitude_loss

        # Mild smoothness regularization prevents isolated pixel-to-pixel height jumps.
        smoothness = (
            torch.mean((height[:, 1:] - height[:, :-1]) ** 2)
            + torch.mean((height[1:, :] - height[:-1, :]) ** 2)
        ) / (config.height_max**2)
        total_loss = total_loss / len(wavelengths_to_fit) + 1e-4 * smoothness
        total_loss.backward()
        optimizer.step()
        with torch.no_grad():
            height.clamp_(0.0, config.height_max)
            height.mul_(aperture)

        losses.append(float(total_loss.detach().cpu()))
        if iteration == 0 or (iteration + 1) % max(1, iterations // 10) == 0:
            component_losses = ", ".join(f"{float(item.detach().cpu()):.4e}" for item in phase_mask_losses)
            print(f"iter {iteration + 1:4d}/{iterations}: loss={losses[-1]:.4e} [{component_losses}]")

    return height.detach(), losses


def evaluate(
    height: torch.Tensor,
    config: Config,
    aperture: torch.Tensor,
    target_intensity: torch.Tensor,
    transfers: list[torch.Tensor],
) -> dict[str, list[torch.Tensor] | list[float]]:
    fields = []
    intensities = []
    correlations = []
    target_centered = target_intensity - target_intensity.mean()
    for wavelength_index in range(len(config.wavelengths)):
        output_field = image_plane_field(
            height,
            aperture,
            config.wavelengths[wavelength_index],
            config.refractive_indices[wavelength_index],
            transfers[wavelength_index],
        )
        intensity = normalized_intensity(output_field)
        intensity_centered = intensity - intensity.mean()
        correlation = torch.sum(intensity_centered * target_centered) / (
            torch.sqrt(torch.sum(intensity_centered**2) * torch.sum(target_centered**2)) + 1e-12
        )
        fields.append(output_field)
        intensities.append(intensity)
        correlations.append(float(correlation.detach().cpu()))
    return {"fields": fields, "intensities": intensities, "correlations": correlations}


def save_image(path: Path, image: np.ndarray, extent_mm: tuple[float, float, float, float], title: str) -> None:
    figure, axis = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
    display = np.maximum(image, 1e-12)
    axis.imshow(10.0 * np.log10(display / display.max()), cmap="inferno", extent=extent_mm, origin="lower", vmin=-40, vmax=0)
    axis.set_xlabel("x (mm)")
    axis.set_ylabel("y (mm)")
    axis.set_title(title)
    figure.colorbar(axis.images[0], ax=axis, label="relative intensity (dB)")
    figure.savefig(path, dpi=180)
    plt.close(figure)


def save_results(
    output_dir: Path,
    config: Config,
    x: torch.Tensor,
    target: torch.Tensor,
    baseline: dict[str, list[torch.Tensor] | list[float]],
    multicolor: dict[str, list[torch.Tensor] | list[float]],
    baseline_height: torch.Tensor,
    multicolor_height: torch.Tensor,
    baseline_loss: list[float],
    multicolor_loss: list[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_mm = x.detach().cpu().numpy() * 1e3
    extent = (float(x_mm.min()), float(x_mm.max()), float(x_mm.min()), float(x_mm.max()))
    target_np = target.detach().cpu().numpy()

    save_image(output_dir / "target_two_spots.png", target_np, extent, "Target: two spots")
    for label, result in (("baseline", baseline), ("multicolor", multicolor)):
        intensities = result["intensities"]
        figure, axes = plt.subplots(1, 2, figsize=(9.5, 4.2), constrained_layout=True)
        for index, axis in enumerate(axes):
            image = intensities[index].detach().cpu().numpy()
            display = np.maximum(image, 1e-12)
            plot = axis.imshow(10.0 * np.log10(display / display.max()), cmap="inferno", extent=extent, origin="lower", vmin=-40, vmax=0)
            axis.set_xlabel("x (mm)")
            axis.set_ylabel("y (mm)")
            axis.set_title(f"{label}, {config.wavelengths[index] * 1e9:.0f} nm\ncorr={result['correlations'][index]:.4f}")
        figure.colorbar(plot, ax=axes, label="relative intensity (dB)")
        figure.savefig(output_dir / f"{label}_results.png", dpi=180)
        plt.close(figure)

    for label, height in (("baseline", baseline_height), ("multicolor", multicolor_height)):
        figure, axis = plt.subplots(figsize=(5.2, 4.4), constrained_layout=True)
        image = height.detach().cpu().numpy() * 1e6
        plot = axis.imshow(image, cmap="viridis", extent=extent, origin="lower")
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
        axis.set_title(f"{label} DOE height")
        figure.colorbar(plot, ax=axis, label="height (µm)")
        figure.savefig(output_dir / f"{label}_height.png", dpi=180)
        plt.close(figure)

    figure, axis = plt.subplots(figsize=(6.2, 4.0), constrained_layout=True)
    axis.semilogy(baseline_loss, label="single wavelength")
    axis.semilogy(multicolor_loss, label="two wavelengths")
    axis.set_xlabel("iteration")
    axis.set_ylabel("loss")
    axis.set_title("Optimization history")
    axis.grid(True, alpha=0.25)
    axis.legend()
    figure.savefig(output_dir / "loss_history.png", dpi=180)
    plt.close(figure)

    np.savez_compressed(
        output_dir / "two_spot_design.npz",
        x=x.detach().cpu().numpy(),
        target=target_np,
        baseline_height=baseline_height.detach().cpu().numpy(),
        multicolor_height=multicolor_height.detach().cpu().numpy(),
        baseline_532=baseline["intensities"][0].detach().cpu().numpy(),
        baseline_633=baseline["intensities"][1].detach().cpu().numpy(),
        multicolor_532=multicolor["intensities"][0].detach().cpu().numpy(),
        multicolor_633=multicolor["intensities"][1].detach().cpu().numpy(),
    )

    metadata = {
        "grid_size": config.n_grid,
        "pixel_pitch_m": config.pixel_pitch,
        "propagation_distance_m": config.propagation_distance,
        "aperture_radius_m": config.aperture_radius,
        "spot_x_m": config.spot_x,
        "spot_sigma_m": config.spot_sigma,
        "height_max_m": config.height_max,
        "wavelengths_m": list(config.wavelengths),
        "refractive_indices": list(config.refractive_indices),
        "refractive_index_source": config.refractive_index_source,
        "optical_path": "uniform illumination -> finite S1813 phase DOE -> air angular-spectrum propagation -> physical image plane",
        "baseline_correlations": baseline["correlations"],
        "multicolor_correlations": multicolor["correlations"],
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=400, help="iterations for each optimization")
    parser.add_argument("--learning-rate", type=float, default=0.04, help="Adam learning rate in micrometres")
    parser.add_argument("--output-dir", type=Path, default=Path("results/two_spot"))
    parser.add_argument("--cpu", action="store_true", help="force CPU even when CUDA is available")
    args = parser.parse_args()

    torch.set_default_dtype(torch.float32)
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    config = Config()
    torch.manual_seed(config.seed)
    print(f"device: {device}")
    print(f"grid: {config.n_grid} x {config.n_grid}, dx={config.pixel_pitch * 1e6:.1f} um")
    print(f"wavelengths: {[round(item * 1e9) for item in config.wavelengths]} nm")

    x, y, fx, fy, aperture = make_grid(config, device)
    target_intensity = target_two_spots(x, y, config.spot_x, config.spot_sigma)
    target_amplitude = torch.sqrt(target_intensity)
    transfers = [
        angular_spectrum_transfer(
            config.wavelengths[index], fx, fy, config.propagation_distance
        )
        for index in range(len(config.wavelengths))
    ]

    print("\n[1/2] single-wavelength baseline")
    baseline_height, baseline_loss = design_doe(
        config=config,
        target_amplitude=target_amplitude,
        aperture=aperture,
        transfers=transfers,
        wavelengths_to_fit=[0],
        iterations=args.iterations,
        learning_rate=args.learning_rate * 1e-6,
        device=device,
    )
    baseline = evaluate(baseline_height, config, aperture, target_intensity, transfers)

    print("\n[2/2] shared-height two-wavelength DOE")
    multicolor_height, multicolor_loss = design_doe(
        config=config,
        target_amplitude=target_amplitude,
        aperture=aperture,
        transfers=transfers,
        wavelengths_to_fit=[0, 1],
        iterations=args.iterations,
        learning_rate=args.learning_rate * 1e-6,
        device=device,
    )
    multicolor = evaluate(multicolor_height, config, aperture, target_intensity, transfers)

    print("\nCorrelation with target intensity:")
    for index, wavelength in enumerate(config.wavelengths):
        print(
            f"  {wavelength * 1e9:.0f} nm: baseline={baseline['correlations'][index]:.4f}, "
            f"two-wavelength={multicolor['correlations'][index]:.4f}"
        )

    save_results(
        output_dir=args.output_dir,
        config=config,
        x=x,
        target=target_intensity,
        baseline=baseline,
        multicolor=multicolor,
        baseline_height=baseline_height,
        multicolor_height=multicolor_height,
        baseline_loss=baseline_loss,
        multicolor_loss=multicolor_loss,
    )
    print(f"saved results to {args.output_dir}")


if __name__ == "__main__":
    main()
