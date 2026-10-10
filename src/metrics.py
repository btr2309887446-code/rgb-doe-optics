"""Quantitative output metrics for DOE comparisons."""
from __future__ import annotations

import math
from typing import Any

import torch

from targets import TargetSpec


def normalized_intensity(field: torch.Tensor) -> torch.Tensor:
    intensity = torch.abs(field) ** 2
    return intensity / (intensity.sum() + 1e-12)


def normalized_amplitude(field: torch.Tensor) -> torch.Tensor:
    amplitude = torch.abs(field)
    return amplitude / torch.sqrt(torch.sum(amplitude**2) + 1e-12)


def _weighted_centroid(
    intensity: torch.Tensor,
    mask: torch.Tensor,
    x: torch.Tensor,
    y: torch.Tensor,
) -> tuple[float, float]:
    weights = intensity * mask
    total = weights.sum() + 1e-12
    return (
        float(((weights * x).sum() / total).detach().cpu()),
        float(((weights * y).sum() / total).detach().cpu()),
    )


def _fwhm_1d(profile: torch.Tensor, pixel_pitch_m: float) -> float:
    peak = profile.max()
    if float(peak.detach().cpu()) <= 0:
        return 0.0
    return float((torch.sum(profile >= 0.5 * peak) * pixel_pitch_m).detach().cpu())


def _circularity(intensity: torch.Tensor, mask: torch.Tensor) -> float:
    region = intensity * mask
    threshold = 0.5 * region.max()
    binary = (region >= threshold).to(torch.float32) * mask
    area = binary.sum()
    edge_x = torch.abs(binary[:, 1:] - binary[:, :-1]).sum()
    edge_y = torch.abs(binary[1:, :] - binary[:-1, :]).sum()
    perimeter = edge_x + edge_y + 1e-12
    return float((4.0 * math.pi * area / perimeter**2).detach().cpu())


def _component_fwhm(
    intensity: torch.Tensor,
    mask: torch.Tensor,
    center: tuple[float, float],
    x: torch.Tensor,
    y: torch.Tensor,
    pixel_pitch_m: float,
) -> dict[str, float]:
    row = int(torch.argmin(torch.abs(y[:, 0] - center[1])).item())
    column = int(torch.argmin(torch.abs(x[0, :] - center[0])).item())
    horizontal = intensity[row, :] * mask[row, :]
    vertical = intensity[:, column] * mask[:, column]
    return {
        "x_m": _fwhm_1d(horizontal, pixel_pitch_m),
        "y_m": _fwhm_1d(vertical, pixel_pitch_m),
    }


def _ring_component_metrics(
    intensity: torch.Tensor,
    center: tuple[float, float],
    radius_m: float,
    width_m: float,
    x: torch.Tensor,
    y: torch.Tensor,
    pixel_pitch_m: float,
) -> dict[str, float]:
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    hole_radius = max(radius_m - 2.5 * width_m, 0.0)
    hole = distance <= hole_radius
    ring = torch.abs(distance - radius_m) <= 2.0 * width_m
    hole_energy = torch.sum(intensity * hole)
    ring_energy = torch.sum(intensity * ring)
    center_mean = torch.sum(intensity * hole) / (hole.sum() + 1e-12)
    ring_mean = torch.sum(intensity * ring) / (ring.sum() + 1e-12)
    lower_radius = max(0.5 * radius_m, pixel_pitch_m)
    upper_radius = 1.5 * radius_m
    bin_width = pixel_pitch_m
    bin_index = torch.floor(distance / bin_width).to(torch.long)
    bin_count = max(1, int(math.ceil(upper_radius / bin_width)) + 1)
    valid = (distance >= lower_radius) & (distance <= upper_radius)
    sums = torch.zeros(bin_count, device=intensity.device, dtype=intensity.dtype)
    counts = torch.zeros_like(sums)
    indices = bin_index[valid].clamp(0, bin_count - 1)
    sums.scatter_add_(0, indices, intensity[valid])
    counts.scatter_add_(0, indices, torch.ones_like(intensity[valid]))
    radial_mean = sums / (counts + 1e-12)
    eligible = counts > 0
    selected = torch.where(eligible, radial_mean, torch.full_like(radial_mean, -1.0))
    peak_index = int(torch.argmax(selected).item())
    peak_radius = (peak_index + 0.5) * bin_width
    return {
        "center_mean_intensity": float(center_mean.detach().cpu()),
        "ring_mean_intensity": float(ring_mean.detach().cpu()),
        "center_to_ring_mean_ratio": float((center_mean / (ring_mean + 1e-8)).detach().cpu()),
        "hole_leakage_fraction": float((hole_energy / (hole_energy + ring_energy + 1e-12)).detach().cpu()),
        "ring_energy": float(ring_energy.detach().cpu()),
        "ring_peak_radius_m": peak_radius,
        "ring_radius_error_m": abs(peak_radius - radius_m),
    }


def evaluate_output(
    field: torch.Tensor,
    target: TargetSpec,
    x: torch.Tensor,
    y: torch.Tensor,
    pixel_pitch_m: float,
) -> dict[str, Any]:
    intensity = normalized_intensity(field)
    amplitude = normalized_amplitude(field)
    target_amplitude = torch.sqrt(target.intensity)
    signal = target.signal_mask
    background = target.background_mask
    roi_efficiency = torch.sum(intensity * signal)
    background_fraction = torch.sum(intensity * background)
    centered_intensity = intensity - intensity.mean()
    centered_target = target.intensity - target.intensity.mean()
    correlation = torch.sum(centered_intensity * centered_target) / (
        torch.sqrt(torch.sum(centered_intensity**2) * torch.sum(centered_target**2)) + 1e-12
    )
    roi_amp_mse = torch.sum(signal * (amplitude - target_amplitude) ** 2) / (signal.sum() + 1e-12)
    roi_intensity_mse = torch.sum(signal * (intensity - target.intensity) ** 2) / (signal.sum() + 1e-12)
    background_peak = torch.max(intensity * background)
    mean_background = torch.sum(intensity * background) / (background.sum() + 1e-12)

    centers = tuple(tuple(float(value) for value in center) for center in target.metadata["centers_m"])
    component_energies: list[float] = []
    component_centroids: list[list[float]] = []
    component_centroid_errors: list[float] = []
    component_fwhm: list[dict[str, float]] = []
    component_circularity: list[float] = []
    expected_energies = torch.stack([
        torch.sum(target.intensity * mask) for mask in target.component_masks
    ])
    expected_shares = expected_energies / (expected_energies.sum() + 1e-12)
    predicted_energies = torch.stack([
        torch.sum(intensity * mask) for mask in target.component_masks
    ])
    predicted_shares = predicted_energies / (predicted_energies.sum() + 1e-12)
    share_errors = torch.abs(predicted_shares - expected_shares)
    for mask, center, energy in zip(target.component_masks, centers, predicted_energies):
        centroid_x, centroid_y = _weighted_centroid(intensity, mask, x, y)
        component_energies.append(float(energy.detach().cpu()))
        component_centroids.append([centroid_x, centroid_y])
        component_centroid_errors.append(math.hypot(centroid_x - center[0], centroid_y - center[1]))
        component_fwhm.append(_component_fwhm(intensity, mask, center, x, y, pixel_pitch_m))
        component_circularity.append(_circularity(intensity, mask))

    energy_std = float(torch.std(predicted_energies, unbiased=False).detach().cpu())
    energy_tv_error = float((0.5 * share_errors.sum()).detach().cpu())
    max_relative_share_error = float(
        (share_errors / (expected_shares + 1e-12)).max().detach().cpu()
    )
    expected_layout_center = torch.tensor(centers, dtype=x.dtype, device=x.device).mean(dim=0)
    observed_layout_center = torch.tensor(component_centroids, dtype=x.dtype, device=x.device).mean(dim=0)
    layout_center_error = float(torch.linalg.vector_norm(observed_layout_center - expected_layout_center).cpu())

    result: dict[str, Any] = {
        "correlation": float(correlation.detach().cpu()),
        "roi_amplitude_mse": float(roi_amp_mse.detach().cpu()),
        "roi_intensity_mse": float(roi_intensity_mse.detach().cpu()),
        "background_fraction": float(background_fraction.detach().cpu()),
        "roi_efficiency": float(roi_efficiency.detach().cpu()),
        "component_energies": component_energies,
        "component_energy_std": energy_std,
        "component_energy_shares": [float(value) for value in predicted_shares.detach().cpu()],
        "target_component_energy_shares": [float(value) for value in expected_shares.detach().cpu()],
        "component_energy_balance_error": energy_tv_error,
        "max_component_relative_share_error": max_relative_share_error,
        "component_centroids_m": component_centroids,
        "component_centroid_errors_m": component_centroid_errors,
        "mean_component_centroid_error_m": float(sum(component_centroid_errors) / len(component_centroid_errors)),
        "component_fwhm_m": component_fwhm,
        "mean_fwhm_x_m": float(sum(item["x_m"] for item in component_fwhm) / len(component_fwhm)),
        "mean_fwhm_y_m": float(sum(item["y_m"] for item in component_fwhm) / len(component_fwhm)),
        "component_circularity": component_circularity,
        "mean_component_circularity": float(sum(component_circularity) / len(component_circularity)),
        "background_peak_to_mean": float((background_peak / (mean_background + 1e-12)).detach().cpu()),
        "layout_centroid_error_m": layout_center_error,
    }
    if len(component_energies) >= 2:
        result["left_energy"] = component_energies[0]
        result["right_energy"] = component_energies[1]
        result["left_right_balance_error"] = energy_tv_error
        result["left_centroid_m"] = component_centroids[0]
        result["right_centroid_m"] = component_centroids[1]
        result["left_fwhm_m"] = component_fwhm[0]["x_m"]
        result["right_fwhm_m"] = component_fwhm[1]["x_m"]
        result["left_circularity"] = component_circularity[0]
        result["right_circularity"] = component_circularity[1]
    if target.target_type == "ring" and target.hole_mask is not None and target.ring_mask is not None:
        radius_m = float(target.metadata["ring_radius_m"])
        width_m = float(target.metadata["ring_width_m"])
        ring_metrics = [
            _ring_component_metrics(intensity, center, radius_m, width_m, x, y, pixel_pitch_m)
            for center in centers
        ]
        result["ring_components"] = ring_metrics
        result["mean_center_to_ring_ratio"] = float(
            sum(item["center_to_ring_mean_ratio"] for item in ring_metrics) / len(ring_metrics)
        )
        result["max_center_to_ring_ratio"] = float(
            max(item["center_to_ring_mean_ratio"] for item in ring_metrics)
        )
        result["mean_ring_radius_error_m"] = float(
            sum(item["ring_radius_error_m"] for item in ring_metrics) / len(ring_metrics)
        )
        result["mean_hole_leakage_fraction"] = float(
            sum(item["hole_leakage_fraction"] for item in ring_metrics) / len(ring_metrics)
        )
        result["mean_ring_energy"] = float(sum(item["ring_energy"] for item in ring_metrics) / len(ring_metrics))
    return result


def evaluate_wavelength_consistency(
    metrics: list[dict[str, Any]],
    target: TargetSpec,
) -> dict[str, float]:
    if len(metrics) < 2:
        return {
            "max_centroid_drift_m": 0.0,
            "mean_centroid_drift_m": 0.0,
            "max_radius_drift_m": 0.0,
        }
    centroid_drifts: list[float] = []
    component_count = len(target.component_masks)
    for component_index in range(component_count):
        centroids = [
            torch.tensor(item["component_centroids_m"][component_index], dtype=torch.float64)
            for item in metrics
        ]
        for first_index, first in enumerate(centroids):
            for second in centroids[first_index + 1:]:
                centroid_drifts.append(float(torch.linalg.vector_norm(first - second)))
    radius_drifts: list[float] = []
    if target.target_type == "ring":
        radii_by_wave = [
            [
                float(component["ring_peak_radius_m"])
                for component in item.get("ring_components", [])
            ]
            for item in metrics
        ]
        for component_index in range(component_count):
            values = [wave[component_index] for wave in radii_by_wave if len(wave) > component_index]
            for first_index, first in enumerate(values):
                radius_drifts.extend(abs(first - second) for second in values[first_index + 1:])
    return {
        "max_centroid_drift_m": max(centroid_drifts, default=0.0),
        "mean_centroid_drift_m": sum(centroid_drifts) / max(1, len(centroid_drifts)),
        "max_radius_drift_m": max(radius_drifts, default=0.0),
    }
