"""Scan soft-circle target parameters with a fixed objective and optimizer.

Only the target radius and edge transition are varied. The forward model,
amplitude-MSE objective, Adam learning rate, initialization, and iteration
count stay fixed for a fair comparison.

Run from the repository root:
    py -3 src/scan_target_parameters.py --iterations 250 --cpu
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_target_processing import (  # noqa: E402
    amplitude_mse_loss,
    correlation,
    make_soft_circle_target,
)
from two_spot_multicolor import (  # noqa: E402
    Config,
    angular_spectrum_transfer,
    image_plane_field,
    make_grid,
    normalized_intensity,
    propagate,
)


def make_target(
    x: torch.Tensor,
    y: torch.Tensor,
    config: Config,
    radius_um: float,
    edge_width_um: float,
) -> torch.Tensor:
    return make_soft_circle_target(
        x,
        y,
        config,
        radius=radius_um * 1e-6,
        edge_width=edge_width_um * 1e-6,
    )


def make_spot_mask(
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x: float,
    radius: float,
) -> torch.Tensor:
    left = (x + spot_x) ** 2 + y**2 <= radius**2
    right = (x - spot_x) ** 2 + y**2 <= radius**2
    return left | right


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
    target_amplitude = torch.sqrt(target)
    beta = [
        2.0 * torch.pi * (n_refractive - 1.0) / wavelength
        for wavelength, n_refractive in zip(
            config.wavelengths,
            config.refractive_indices,
        )
    ]
    losses: list[float] = []

    for _ in range(iterations):
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
    spot_fractions = []
    radius_for_metric = 1.5e-6 * 0.0
    del radius_for_metric
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
        correlations.append(float(correlation(output_intensity, target).detach().cpu()))
        # The caller supplies the target-specific mask after optimization.
        background_fractions.append(float("nan"))
        spot_fractions.append(float("nan"))

    return {
        "height": height.detach().cpu().numpy(),
        "intensities": intensities,
        "losses": losses,
        "correlations": correlations,
        "background_fractions": background_fractions,
        "spot_fractions": spot_fractions,
    }


def fill_region_metrics(
    result: dict[str, object],
    x: torch.Tensor,
    y: torch.Tensor,
    config: Config,
    radius_um: float,
) -> None:
    mask = make_spot_mask(x, y, config.spot_x, radius_um * 1e-6)
    background = ~mask
    background_fractions = []
    spot_fractions = []
    for image in result["intensities"]:
        intensity = torch.as_tensor(image, device=x.device)
        spot_fraction = torch.sum(intensity[mask])
        background_fraction = torch.sum(intensity[background])
        spot_fractions.append(float(spot_fraction.cpu()))
        background_fractions.append(float(background_fraction.cpu()))
    result["background_fractions"] = background_fractions
    result["spot_fractions"] = spot_fractions


def save_outputs(
    output_dir: Path,
    config: Config,
    x: torch.Tensor,
    y: torch.Tensor,
    targets: dict[str, torch.Tensor],
    results: dict[str, dict[str, object]],
    radii: list[float],
    edge_widths: list[float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    x_mm = x.detach().cpu().numpy() * 1e3
    extent = (float(x_mm.min()), float(x_mm.max()), float(x_mm.min()), float(x_mm.max()))

    mean_correlation = np.zeros((len(radii), len(edge_widths)))
    worst_correlation = np.zeros_like(mean_correlation)
    mean_background = np.zeros_like(mean_correlation)
    for row, radius in enumerate(radii):
        for column, edge_width in enumerate(edge_widths):
            result = results[f"r{radius:g}_e{edge_width:g}"]
            mean_correlation[row, column] = np.mean(result["correlations"])
            worst_correlation[row, column] = np.min(result["correlations"])
            mean_background[row, column] = np.mean(result["background_fractions"])

    score = mean_correlation - 0.25 * mean_background
    best_index = np.unravel_index(np.argmax(score), score.shape)
    best_key = f"r{radii[best_index[0]]:g}_e{edge_widths[best_index[1]]:g}"

    def heatmap(data: np.ndarray, title: str, filename: str, fmt: str = ".3f") -> None:
        figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True)
        image = axis.imshow(
            data,
            origin="lower",
            cmap="viridis",
            aspect="auto",
            vmin=float(np.min(data)),
            vmax=float(np.max(data)),
        )
        axis.set_xticks(range(len(edge_widths)), [f"{value:g}" for value in edge_widths])
        axis.set_yticks(range(len(radii)), [f"{value:g}" for value in radii])
        axis.set_xlabel("edge width (µm)")
        axis.set_ylabel("spot radius (µm)")
        axis.set_title(title)
        for row in range(data.shape[0]):
            for column in range(data.shape[1]):
                axis.text(column, row, format(data[row, column], fmt), ha="center", va="center", color="white")
        figure.colorbar(image, ax=axis)
        figure.savefig(output_dir / filename, dpi=180)
        plt.close(figure)

    heatmap(mean_correlation, "Mean correlation", "mean_correlation.png")
    heatmap(worst_correlation, "Worst-wavelength correlation", "worst_correlation.png")
    heatmap(mean_background, "Mean background fraction", "mean_background.png")
    heatmap(score, "Selection score = mean correlation - 0.25 background", "selection_score.png")

    best_target = targets[best_key]
    best_result = results[best_key]
    figure, axes = plt.subplots(1, 3, figsize=(13.5, 4.2), constrained_layout=True)
    axes[0].imshow(best_target.detach().cpu().numpy(), cmap="inferno", extent=extent, origin="lower")
    axes[0].set_title(f"Best target: {best_key}")
    for index, axis in enumerate(axes[1:]):
        image = best_result["intensities"][index]
        display = 10.0 * np.log10(np.maximum(image, 1e-12) / image.max())
        axis.imshow(display, cmap="inferno", extent=extent, origin="lower", vmin=-30, vmax=0)
        axis.set_title(
            f"{config.wavelengths[index] * 1e9:.0f} nm\n"
            f"corr={best_result['correlations'][index]:.3f}, "
            f"bg={best_result['background_fractions'][index]:.3f}"
        )
    for axis in axes:
        axis.set_xlabel("x (mm)")
        axis.set_ylabel("y (mm)")
    figure.savefig(output_dir / "best_case.png", dpi=180)
    plt.close(figure)

    summary = {
        "radii_um": radii,
        "edge_widths_um": edge_widths,
        "best_key": best_key,
        "best_score": float(score[best_index]),
        "mean_correlation": mean_correlation.tolist(),
        "worst_correlation": worst_correlation.tolist(),
        "mean_background": mean_background.tolist(),
        "selection_score": score.tolist(),
        "cases": {
            key: {
                "correlations": value["correlations"],
                "background_fractions": value["background_fractions"],
                "spot_fractions": value["spot_fractions"],
                "final_loss": value["losses"][-1],
            }
            for key, value in results.items()
        },
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    np.savez_compressed(
        output_dir / "scan_arrays.npz",
        radii_um=np.asarray(radii),
        edge_widths_um=np.asarray(edge_widths),
        mean_correlation=mean_correlation,
        worst_correlation=worst_correlation,
        mean_background=mean_background,
        selection_score=score,
        **{
            f"target_{key}": value.detach().cpu().numpy()
            for key, value in targets.items()
        },
        **{
            f"height_{key}": value["height"]
            for key, value in results.items()
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--iterations", type=int, default=250)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--radii", type=float, nargs="+", default=[40, 50, 60, 70, 80])
    parser.add_argument("--edge-widths", type=float, nargs="+", default=[6, 10, 14, 18, 24])
    parser.add_argument("--output-dir", type=Path, default=Path("results/target_parameter_scan"))
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    config = Config()
    device = torch.device("cpu" if args.cpu or not torch.cuda.is_available() else "cuda")
    torch.manual_seed(config.seed)
    print(f"device: {device}")
    print(f"fixed objective: amplitude MSE, learning rate: {args.learning_rate} um")
    print(f"radii (um): {args.radii}")
    print(f"edge widths (um): {args.edge_widths}")

    x, y, fx, fy, aperture = make_grid(config, device)
    transfers = [
        angular_spectrum_transfer(
            config.wavelengths[index], fx, fy, config.propagation_distance
        )
        for index in range(len(config.wavelengths))
    ]

    targets: dict[str, torch.Tensor] = {}
    results: dict[str, dict[str, object]] = {}
    total_cases = len(args.radii) * len(args.edge_widths)
    case_index = 0
    for radius in args.radii:
        for edge_width in args.edge_widths:
            case_index += 1
            key = f"r{radius:g}_e{edge_width:g}"
            print(f"\n[{case_index}/{total_cases}] {key}")
            target = make_target(x, y, config, radius, edge_width)
            result = optimize_target(
                config,
                target,
                aperture,
                transfers,
                args.iterations,
                args.learning_rate,
                device,
                seed=config.seed + 100,
            )
            fill_region_metrics(result, x, y, config, radius)
            targets[key] = target
            results[key] = result
            print(
                "  corr:", [round(value, 4) for value in result["correlations"]],
                "bg:", [round(value, 4) for value in result["background_fractions"]],
            )

    save_outputs(args.output_dir, config, x, y, targets, results, args.radii, args.edge_widths)
    summary = json.loads((args.output_dir / "summary.json").read_text(encoding="utf-8"))
    print(f"\nbest case: {summary['best_key']}, score={summary['best_score']:.4f}")
    print(f"saved scan to {args.output_dir}")


if __name__ == "__main__":
    main()
