"""Region-aware multicolor DOE losses."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import torch

from metrics import normalized_amplitude, normalized_intensity
from targets import TargetSpec


@dataclass(frozen=True)
class LossWeights:
    alpha_shape: float = 1.0
    beta_background: float = 0.25
    gamma_efficiency: float = 0.10
    delta_balance: float = 0.05
    eta_smooth: float = 1e-5
    hole_weight: float = 0.0
    position_weight: float = 0.0
    centroid_consistency_weight: float = 1.0
    centroid_tolerance_m: float = 5e-6
    edge_weight: float = 0.0
    radius_weight: float = 0.0
    radius_consistency_weight: float = 1.0
    geometry_weight: float = 0.0
    rgb_pixel_weight: float = 0.0
    rgb_edge_weight: float = 0.0
    rgb_background_weight: float = 0.0
    # Absolute background terms are deliberately separate from
    # ``rgb_background_weight``, which only enforces cross-wavelength
    # background similarity.
    rgb_background_absolute_weight: float = 0.0
    rgb_background_peak_weight: float = 0.0
    efficiency_floor_weight: float = 0.0
    background_peak_order: float = 8.0
    rgb_energy_weight: float = 0.0
    rgb_component_weight: float = 0.0


@dataclass(frozen=True)
class LossSchedule:
    shape_only_until: int = 100
    background_ramp_end: int = 300
    hole_ramp_start: int = 300
    hole_ramp_end: int = 400
    rgb_ramp_start: int = 0
    rgb_ramp_end: int = 0

    def background_scale(self, step: int) -> float:
        if step < self.shape_only_until:
            return 0.0
        if step >= self.background_ramp_end:
            return 1.0
        return (step - self.shape_only_until) / max(1, self.background_ramp_end - self.shape_only_until)

    def hole_scale(self, step: int) -> float:
        if step < self.hole_ramp_start:
            return 0.0
        if step >= self.hole_ramp_end:
            return 1.0
        return (step - self.hole_ramp_start) / max(1, self.hole_ramp_end - self.hole_ramp_start)

    def rgb_scale(self, step: int) -> float:
        if self.rgb_ramp_end <= self.rgb_ramp_start:
            return 1.0
        if step < self.rgb_ramp_start:
            return 0.0
        if step >= self.rgb_ramp_end:
            return 1.0
        return (step - self.rgb_ramp_start) / max(1, self.rgb_ramp_end - self.rgb_ramp_start)


def smoothness_loss(height_m: torch.Tensor, height_max_m: float) -> torch.Tensor:
    dx = torch.mean((height_m[..., 1:] - height_m[..., :-1]) ** 2)
    dy = torch.mean((height_m[..., 1:, :] - height_m[..., :-1, :]) ** 2)
    return (dx + dy) / (height_max_m**2 + 1e-12)


def edge_gradient_loss(amplitude: torch.Tensor, target_amplitude: torch.Tensor) -> torch.Tensor:
    amplitude_dx = amplitude[:, 1:] - amplitude[:, :-1]
    amplitude_dy = amplitude[1:, :] - amplitude[:-1, :]
    target_dx = target_amplitude[:, 1:] - target_amplitude[:, :-1]
    target_dy = target_amplitude[1:, :] - target_amplitude[:-1, :]
    scale = torch.maximum(target_dx.abs().max(), target_dy.abs().max()) + 1e-12
    weight_x = target_dx.abs() / scale
    weight_y = target_dy.abs() / scale
    numerator = torch.sum(weight_x * (amplitude_dx - target_dx).square())
    numerator = numerator + torch.sum(weight_y * (amplitude_dy - target_dy).square())
    denominator = weight_x.sum() + weight_y.sum() + 1e-12
    return numerator / denominator


def _single_loss(
    field: torch.Tensor,
    target: TargetSpec,
    weights: LossWeights,
    background_scale: float,
    hole_scale: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    intensity = normalized_intensity(field)
    amplitude = normalized_amplitude(field)
    target_amplitude = torch.sqrt(target.intensity)
    signal = target.signal_mask
    transition = target.transition_mask
    background = target.background_mask
    shape = torch.sum(signal * (amplitude - target_amplitude) ** 2) / (signal.sum() + 1e-12)
    geometry = torch.sum(transition * (amplitude - target_amplitude) ** 2) / (
        transition.sum() + 1e-12
    )
    edge = edge_gradient_loss(amplitude, target_amplitude)
    background_loss = torch.sum(background * intensity)
    efficiency = torch.sum(signal * intensity)
    efficiency_loss = (1.0 - efficiency) ** 2
    component_energies = torch.stack([
        torch.sum(mask * intensity) for mask in target.component_masks
    ])
    predicted_shares = component_energies / (component_energies.sum() + 1e-12)
    expected_energies = torch.stack([
        torch.sum(mask * target.intensity) for mask in target.component_masks
    ])
    expected_shares = expected_energies / (expected_energies.sum() + 1e-12)
    balance_loss = torch.mean((predicted_shares - expected_shares) ** 2)
    if target.hole_mask is not None and target.ring_mask is not None:
        hole_energy = torch.sum(target.hole_mask * intensity)
        ring_energy = torch.sum(target.ring_mask * intensity)
        hole_loss = hole_energy / (ring_energy + 1e-6)
    else:
        hole_loss = torch.zeros((), device=intensity.device, dtype=intensity.dtype)
    total = (
        weights.alpha_shape * shape
        + weights.beta_background * background_scale * background_loss
        + background_scale * weights.gamma_efficiency * efficiency_loss
        + background_scale * weights.delta_balance * balance_loss
        + weights.hole_weight * hole_scale * hole_loss
        + weights.geometry_weight * geometry
        + weights.edge_weight * edge
    )
    return total, {
        "shape": shape,
        "background": background_loss,
        "efficiency": efficiency_loss,
        "balance": balance_loss,
        "hole": hole_loss,
        "geometry": geometry,
        "edge": edge,
    }


def _centroid_loss(
    intensity: torch.Tensor,
    target: TargetSpec,
    pixel_pitch_m: float,
    position_scale_m: float = 300e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    centers = target.metadata.get("centers_m", ((0.0, 0.0),))
    height, width = intensity.shape
    coordinate = torch.linspace(
        -(width // 2), width // 2 - 1, width, device=intensity.device, dtype=intensity.dtype
    )
    x = coordinate * pixel_pitch_m
    y = coordinate * pixel_pitch_m
    grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
    centroids = []
    target_error = torch.zeros((), device=intensity.device, dtype=intensity.dtype)
    for mask, center in zip(target.component_masks, centers):
        weights = intensity * mask
        total = weights.sum() + 1e-12
        centroid_x = (weights * grid_x).sum() / total
        centroid_y = (weights * grid_y).sum() / total
        target_x = torch.as_tensor(float(center[0]), device=intensity.device, dtype=intensity.dtype)
        target_y = torch.as_tensor(float(center[1]), device=intensity.device, dtype=intensity.dtype)
        target_error = target_error + ((centroid_x - target_x) ** 2 + (centroid_y - target_y) ** 2)
        centroids.append(torch.stack((centroid_x, centroid_y)))
    target_error = target_error / max(1, len(centroids)) / (position_scale_m**2)
    return target_error, torch.stack(centroids)


def _radius_loss(
    intensity: torch.Tensor,
    target: TargetSpec,
    pixel_pitch_m: float = 5e-6,
) -> tuple[torch.Tensor, torch.Tensor]:
    if target.target_type != "ring" or target.ring_mask is None:
        empty = torch.zeros((0,), device=intensity.device, dtype=intensity.dtype)
        return empty, empty
    centers = target.metadata.get("centers_m", ())
    radius_m = float(target.metadata.get("ring_radius_m", 50e-6))
    coordinate = torch.linspace(
        -(intensity.shape[-1] // 2),
        intensity.shape[-1] // 2 - 1,
        intensity.shape[-1],
        device=intensity.device,
        dtype=intensity.dtype,
    ) * pixel_pitch_m
    grid_x, grid_y = torch.meshgrid(coordinate, coordinate, indexing="xy")
    values = []
    for mask, center in zip(target.component_masks, centers):
        distance = torch.sqrt((grid_x - float(center[0])) ** 2 + (grid_y - float(center[1])) ** 2)
        weights = intensity * mask
        values.append((weights * distance).sum() / (weights.sum() + 1e-12))
    if not values:
        empty = torch.zeros((0,), device=intensity.device, dtype=intensity.dtype)
        return empty, empty
    return torch.stack(values), torch.as_tensor(
        radius_m, device=intensity.device, dtype=intensity.dtype
    )


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    normalized_weights = weights / (weights.sum() + 1e-12)
    return torch.sum(values * normalized_weights[:, None, None], dim=0)


def _masked_pairwise_mse(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.to(dtype=first.dtype)
    return torch.sum((first - second).square() * mask) / (mask.sum() + 1e-12)


def _field_list(fields: Iterable[torch.Tensor] | torch.Tensor) -> list[torch.Tensor]:
    if isinstance(fields, torch.Tensor):
        if fields.ndim == 2:
            return [fields]
        if fields.ndim == 3:
            return list(fields.unbind(0))
        raise ValueError("field tensor must have shape (height, width) or (channels, height, width)")
    values = list(fields)
    if not values:
        raise ValueError("at least one field is required")
    return values


def _power_tensor(
    count: int,
    reference: torch.Tensor,
    input_powers: Iterable[float] | None,
) -> torch.Tensor:
    if input_powers is None:
        return torch.ones(count, device=reference.device, dtype=reference.dtype)
    values = torch.as_tensor(list(input_powers), device=reference.device, dtype=reference.dtype)
    if values.numel() != count or torch.any(values <= 0):
        raise ValueError("input_powers must match fields and contain positive values")
    return values


def _wavelength_weight_tensor(
    count: int,
    reference: torch.Tensor,
    wavelength_weights: Iterable[float] | None,
) -> torch.Tensor:
    if wavelength_weights is None:
        weights = torch.ones(count, device=reference.device, dtype=reference.dtype)
    else:
        weights = torch.as_tensor(list(wavelength_weights), device=reference.device, dtype=reference.dtype)
        if weights.numel() != count or torch.any(weights < 0) or float(weights.sum().detach().cpu()) <= 0:
            raise ValueError("wavelength_weights must match fields and have a positive sum")
    return weights / (weights.sum() + 1e-12)


def background_absolute_loss(
    fields: Iterable[torch.Tensor] | torch.Tensor,
    target: TargetSpec,
    input_powers: Iterable[float] | None = None,
    wavelength_weights: Iterable[float] | None = None,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean absolute background-to-ROI energy ratio.

    Intensities are divided by their input powers before integration.  This
    keeps unequal RGB illumination comparable while preserving the requested
    absolute field scale in the numerator and denominator.
    """
    values = _field_list(fields)
    intensities = torch.stack([torch.abs(field).square() for field in values])
    signal = target.signal_mask.to(dtype=intensities.dtype)
    background = target.background_mask.to(dtype=intensities.dtype)
    powers = _power_tensor(len(values), intensities, input_powers)
    weights = _wavelength_weight_tensor(len(values), intensities, wavelength_weights)
    normalized = intensities / powers[:, None, None]
    background_energy = torch.sum(normalized * background[None, :, :], dim=(-2, -1))
    roi_energy = torch.sum(normalized * signal[None, :, :], dim=(-2, -1))
    ratios = background_energy / (roi_energy + eps)
    return torch.sum(weights * ratios)


def background_peak_loss(
    fields: Iterable[torch.Tensor] | torch.Tensor,
    target: TargetSpec,
    input_powers: Iterable[float] | None = None,
    wavelength_weights: Iterable[float] | None = None,
    p: float = 8.0,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Mean smooth-background-peak to ROI-energy ratio across wavelengths.

    An unnormalized p-norm provides a smooth approximation to the peak of the
    power-normalized background intensity. Dividing by power-normalized ROI
    energy makes the term insensitive to uniform field-amplitude scaling and
    measures background hotspots relative to useful signal. The p-norm is
    evaluated after per-channel rescaling to avoid overflow and underflow.
    """
    if p < 1.0:
        raise ValueError("background peak order p must be at least one")
    values = _field_list(fields)
    intensities = torch.stack([torch.abs(field).square() for field in values])
    background = target.background_mask.to(dtype=intensities.dtype)
    signal = target.signal_mask.to(dtype=intensities.dtype)
    powers = _power_tensor(len(values), intensities, input_powers)
    weights = _wavelength_weight_tensor(len(values), intensities, wavelength_weights)
    normalized = intensities / powers[:, None, None]
    samples = (normalized * background[None]).clamp_min(0.0).flatten(1)
    # Rescale before taking high powers: absolute intensities can be tiny or
    # very large, and the unscaled p-norm underflows/overflows in float32.
    scale = samples.amax(dim=1).detach().clamp_min(torch.finfo(samples.dtype).tiny)
    smooth_peak = scale * torch.linalg.vector_norm(samples / scale[:, None], ord=p, dim=1)
    roi_energy = torch.sum(normalized * signal[None, :, :], dim=(-2, -1))
    ratios = smooth_peak / (roi_energy + eps)
    return torch.sum(weights * ratios)


def efficiency_floor_loss(
    fields: Iterable[torch.Tensor] | torch.Tensor,
    target: TargetSpec,
    eta_min: float | torch.Tensor,
    input_powers: Iterable[float] | None = None,
    wavelength_weights: Iterable[float] | None = None,
) -> torch.Tensor:
    """Penalize ROI efficiency below a fixed per-wavelength floor."""
    if isinstance(fields, torch.Tensor) and fields.ndim < 2:
        raise ValueError("fields must contain spatial dimensions")
    values = _field_list(fields)
    intensities = torch.stack([torch.abs(field).square() for field in values])
    signal = target.signal_mask.to(dtype=intensities.dtype)
    powers = _power_tensor(len(values), intensities, input_powers)
    weights = _wavelength_weight_tensor(len(values), intensities, wavelength_weights)
    eta = torch.sum(intensities * signal[None, :, :], dim=(-2, -1)) / (powers + 1e-12)
    floor = torch.as_tensor(eta_min, device=intensities.device, dtype=intensities.dtype)
    if floor.ndim == 0:
        floor = floor.expand_as(eta)
    if floor.shape != eta.shape:
        raise ValueError("eta_min must be scalar or match the number of fields")
    if not torch.isfinite(floor).all() or torch.any(floor < 0):
        raise ValueError("eta_min must be finite and nonnegative")
    return torch.sum(weights * torch.relu(floor - eta).square())


def rgb_consistency_losses(
    fields: Iterable[torch.Tensor],
    target: TargetSpec,
    input_powers: Iterable[float] | None = None,
    wavelength_weights: Iterable[float] | None = None,
    *,
    include_absolute: bool = False,
    background_peak_order: float = 8.0,
) -> dict[str, torch.Tensor]:
    """Return differentiable cross-wavelength consistency losses.

    ``include_absolute`` is opt-in to preserve the historical return schema;
    the joint objective uses the standalone absolute background and efficiency
    functions below so existing RGB ablation reports remain comparable.
    """
    fields = list(fields)
    if len(fields) < 2:
        zero = torch.zeros((), device=fields[0].device if fields else "cpu")
        result = {
            "rgb_pixel": zero,
            "rgb_roi_absolute": zero,
            "rgb_edge": zero,
            "rgb_background": zero,
            "rgb_energy": zero,
            "rgb_component": zero,
        }
        if include_absolute:
            result.update({
                "rgb_background_absolute": background_absolute_loss(fields, target, input_powers, wavelength_weights),
                "rgb_background_peak": background_peak_loss(fields, target, input_powers, wavelength_weights, p=background_peak_order),
            })
        return result
    intensities = torch.stack([torch.abs(field).square() for field in fields])
    wavelength_count = intensities.shape[0]
    if input_powers is None:
        input_power_tensor = torch.ones(
            wavelength_count, device=intensities.device, dtype=intensities.dtype
        )
    else:
        input_power_tensor = torch.as_tensor(
            list(input_powers), device=intensities.device, dtype=intensities.dtype
        )
        if input_power_tensor.numel() != wavelength_count or torch.any(input_power_tensor <= 0):
            raise ValueError("input_powers must match fields and contain positive values")
    if wavelength_weights is None:
        wavelength_weight_tensor = torch.ones(
            wavelength_count, device=intensities.device, dtype=intensities.dtype
        )
    else:
        wavelength_weight_tensor = torch.as_tensor(
            list(wavelength_weights), device=intensities.device, dtype=intensities.dtype
        )
        if wavelength_weight_tensor.numel() != wavelength_count or torch.any(wavelength_weight_tensor < 0):
            raise ValueError("wavelength_weights must match fields and be nonnegative")
        if float(wavelength_weight_tensor.sum().detach().cpu()) <= 0:
            raise ValueError("wavelength_weights must have positive sum")
    wavelength_weight_tensor = wavelength_weight_tensor / (wavelength_weight_tensor.sum() + 1e-12)

    signal = target.signal_mask.to(dtype=intensities.dtype)
    transition = target.transition_mask.to(dtype=intensities.dtype)
    background = target.background_mask.to(dtype=intensities.dtype)
    roi_energy = torch.sum(intensities * signal[None, :, :], dim=(-2, -1))
    shape_intensity = intensities * signal[None, :, :]
    shape_intensity = shape_intensity / (roi_energy[:, None, None] + 1e-12)
    mean_shape = _weighted_mean(shape_intensity, wavelength_weight_tensor)
    pixel_loss = torch.sum(
        wavelength_weight_tensor[:, None, None]
        * (shape_intensity - mean_shape[None, :, :]).square()
        * signal[None, :, :]
    ) / (signal.sum() + 1e-12)

    absolute_intensity = intensities / (input_power_tensor[:, None, None] + 1e-12)
    mean_absolute = _weighted_mean(absolute_intensity, wavelength_weight_tensor)
    roi_absolute_loss = torch.sum(
        wavelength_weight_tensor[:, None, None]
        * (absolute_intensity - mean_absolute[None, :, :]).square()
        * signal[None, :, :]
    ) / (torch.sum(mean_absolute.square() * signal) + 1e-12)
    background_loss = torch.sum(
        wavelength_weight_tensor[:, None, None]
        * (absolute_intensity - mean_absolute[None, :, :]).square()
        * background[None, :, :]
    ) / (torch.sum(mean_absolute.square() * background) + 1e-12)

    amplitude = torch.sqrt(shape_intensity.clamp_min(0.0) + 1e-12)
    amplitude_dx = amplitude[..., :, 1:] - amplitude[..., :, :-1]
    amplitude_dy = amplitude[..., 1:, :] - amplitude[..., :-1, :]
    edge_mask_x = torch.maximum(transition[:, 1:], transition[:, :-1])
    edge_mask_y = torch.maximum(transition[1:, :], transition[:-1, :])
    mean_dx = _weighted_mean(amplitude_dx, wavelength_weight_tensor)
    mean_dy = _weighted_mean(amplitude_dy, wavelength_weight_tensor)
    edge_loss = (
        torch.sum(
            wavelength_weight_tensor[:, None, None]
            * (amplitude_dx - mean_dx[None, :, :]).square()
            * edge_mask_x[None, :, :]
        ) / (edge_mask_x.sum() + 1e-12)
        + torch.sum(
            wavelength_weight_tensor[:, None, None]
            * (amplitude_dy - mean_dy[None, :, :]).square()
            * edge_mask_y[None, :, :]
        ) / (edge_mask_y.sum() + 1e-12)
    ) / 2.0

    efficiency = roi_energy / (input_power_tensor + 1e-12)
    mean_efficiency = torch.sum(efficiency * wavelength_weight_tensor)
    energy_loss = torch.sum(
        wavelength_weight_tensor * (efficiency - mean_efficiency).square()
    ) / (mean_efficiency.square() + 1e-12)

    component_shares = []
    for component_mask in target.component_masks:
        component_energy = torch.sum(
            intensities * component_mask.to(dtype=intensities.dtype)[None, :, :], dim=(-2, -1)
        )
        component_shares.append(component_energy / (roi_energy + 1e-12))
    shares = torch.stack(component_shares, dim=-1)
    mean_shares = torch.sum(
        shares * wavelength_weight_tensor[:, None], dim=0
    )
    component_loss = torch.sum(
        wavelength_weight_tensor[:, None] * (shares - mean_shares[None, :]).square()
    ) / max(1, shares.shape[-1])
    result = {
        "rgb_pixel": pixel_loss,
        "rgb_roi_absolute": roi_absolute_loss,
        "rgb_edge": edge_loss,
        "rgb_background": background_loss,
        "rgb_energy": energy_loss,
        "rgb_component": component_loss,
    }
    if include_absolute:
        result.update({
            "rgb_background_absolute": background_absolute_loss(
                fields, target, input_powers, wavelength_weights
            ),
            "rgb_background_peak": background_peak_loss(
                fields, target, input_powers, wavelength_weights, p=background_peak_order
            ),
        })
    return result


def joint_loss(
    fields: Iterable[torch.Tensor],
    target: TargetSpec,
    height_m: torch.Tensor,
    height_max_m: float,
    weights: LossWeights,
    schedule: LossSchedule,
    step: int,
    wavelength_weights: Iterable[float] | None = None,
    pixel_pitch_m: float = 5e-6,
    record_components: bool = True,
    input_powers: Iterable[float] | None = None,
    efficiency_floor: float | torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    fields = list(fields)
    if not fields:
        raise ValueError("at least one field is required")
    if wavelength_weights is None:
        wavelength_weights = [1.0] * len(fields)
    wavelength_weights = list(wavelength_weights)
    if len(wavelength_weights) != len(fields):
        raise ValueError("wavelength_weights length mismatch")
    total_weight = sum(wavelength_weights)
    if total_weight <= 0:
        raise ValueError("wavelength weights must have positive sum")
    total = torch.zeros((), device=height_m.device)
    components: dict[str, float] = {}
    position_losses = []
    centroid_sets = []
    radius_sets = []
    scale = schedule.background_scale(step)
    hole_scale = schedule.hole_scale(step)
    for index, (field, wavelength_weight) in enumerate(zip(fields, wavelength_weights)):
        current, current_components = _single_loss(field, target, weights, scale, hole_scale)
        total = total + wavelength_weight * current / total_weight
        if record_components:
            for name, value in current_components.items():
                components[f"{name}_{index}"] = float(value.detach().cpu())
        position, centroids = _centroid_loss(
            normalized_intensity(field), target, pixel_pitch_m
        )
        position_losses.append(position)
        centroid_sets.append(centroids)
        radii, target_radius = _radius_loss(
            normalized_intensity(field), target, pixel_pitch_m
        )
        if radii.numel():
            radius_sets.append(radii)

    active_indices = [index for index, value in enumerate(wavelength_weights) if value > 0]
    active_weights = torch.as_tensor(
        [wavelength_weights[index] for index in active_indices],
        device=height_m.device,
        dtype=height_m.dtype,
    )
    active_weights = active_weights / active_weights.sum()
    active_positions = torch.stack([position_losses[index] for index in active_indices])
    position = torch.sum(active_positions * active_weights)
    active_centroids = [centroid_sets[index] for index in active_indices]
    centroid_drift = torch.zeros((), device=height_m.device, dtype=height_m.dtype)
    if len(active_centroids) > 1:
        centroid_stack = torch.stack(active_centroids)
        pairwise = centroid_stack[:, None, :, :] - centroid_stack[None, :, :, :]
        pairwise_distance = torch.linalg.vector_norm(pairwise, dim=-1)
        pair_indices = torch.triu_indices(
            centroid_stack.shape[0], centroid_stack.shape[0], offset=1,
            device=centroid_stack.device,
        )
        unique_distances = pairwise_distance[pair_indices[0], pair_indices[1], :]
        centroid_drift = unique_distances.max()
        excess = torch.relu(unique_distances - weights.centroid_tolerance_m)
        consistency = torch.mean(
            (excess / max(weights.centroid_tolerance_m, 1e-12)).square()
        )
    else:
        consistency = torch.zeros((), device=height_m.device, dtype=height_m.dtype)
    active_radii = [radius_sets[index] for index in active_indices if index < len(radius_sets)]
    if active_radii:
        radius_stack = torch.stack(active_radii)
        radius_m = float(target.metadata["ring_radius_m"])
        radius_target = torch.mean((radius_stack - radius_m) ** 2) / (radius_m**2)
        radius_consistency = (
            torch.sum(
                (radius_stack - torch.sum(radius_stack * active_weights[:, None], dim=0)) ** 2
                * active_weights[:, None],
                dim=0,
            ).mean() / (radius_m**2)
            if len(active_radii) > 1
            else torch.zeros((), device=height_m.device, dtype=height_m.dtype)
        )
    else:
        radius_target = torch.zeros((), device=height_m.device, dtype=height_m.dtype)
        radius_consistency = torch.zeros((), device=height_m.device, dtype=height_m.dtype)
    if position_losses:
        total = total + weights.position_weight * (
            position + weights.centroid_consistency_weight * consistency
        )
        total = total + weights.radius_weight * (
            radius_target + weights.radius_consistency_weight * radius_consistency
        )
        if record_components:
            components["position"] = float(position.detach().cpu())
            components["centroid_consistency"] = float(consistency.detach().cpu())
            components["centroid_drift_um"] = float(centroid_drift.detach().cpu()) * 1e6
            components["radius_target"] = float(radius_target.detach().cpu())
            components["radius_consistency"] = float(radius_consistency.detach().cpu())
    rgb_scale = schedule.rgb_scale(step)
    rgb_losses = rgb_consistency_losses(
        fields,
        target,
        input_powers=input_powers,
        wavelength_weights=wavelength_weights,
    )
    rgb_pixel = rgb_losses["rgb_pixel"] + rgb_losses["rgb_roi_absolute"]
    total = total + rgb_scale * (
        weights.rgb_pixel_weight * rgb_pixel
        + weights.rgb_edge_weight * rgb_losses["rgb_edge"]
        + weights.rgb_background_weight * rgb_losses["rgb_background"]
        + weights.rgb_energy_weight * rgb_losses["rgb_energy"]
        + weights.rgb_component_weight * rgb_losses["rgb_component"]
    )
    absolute_background = background_absolute_loss(
        fields,
        target,
        input_powers=input_powers,
        wavelength_weights=wavelength_weights,
    )
    background_peak = background_peak_loss(
        fields,
        target,
        input_powers=input_powers,
        wavelength_weights=wavelength_weights,
        p=weights.background_peak_order,
    )
    total = total + rgb_scale * (weights.rgb_background_absolute_weight * absolute_background
                                + weights.rgb_background_peak_weight * background_peak)
    if weights.efficiency_floor_weight and efficiency_floor is None:
        raise ValueError("efficiency_floor is required when its weight is nonzero")
    if efficiency_floor is not None and weights.efficiency_floor_weight != 0.0:
        floor_loss = efficiency_floor_loss(
            fields,
            target,
            efficiency_floor,
            input_powers=input_powers,
            wavelength_weights=wavelength_weights,
        )
        total = total + rgb_scale * weights.efficiency_floor_weight * floor_loss
    else:
        floor_loss = torch.zeros((), device=height_m.device, dtype=height_m.dtype)
    if record_components:
        components["rgb_pixel"] = float(rgb_losses["rgb_pixel"].detach().cpu())
        components["rgb_roi_absolute"] = float(rgb_losses["rgb_roi_absolute"].detach().cpu())
        components["rgb_edge"] = float(rgb_losses["rgb_edge"].detach().cpu())
        components["rgb_background"] = float(rgb_losses["rgb_background"].detach().cpu())
        components["rgb_background_absolute"] = float(absolute_background.detach().cpu())
        components["rgb_background_peak"] = float(background_peak.detach().cpu())
        components["efficiency_floor"] = float(floor_loss.detach().cpu())
        components["rgb_energy"] = float(rgb_losses["rgb_energy"].detach().cpu())
        components["rgb_component"] = float(rgb_losses["rgb_component"].detach().cpu())
        components["rgb_scale"] = rgb_scale
    smooth = smoothness_loss(height_m, height_max_m)
    total = total + weights.eta_smooth * smooth
    if record_components:
        components["smoothness"] = float(smooth.detach().cpu())
        components["background_scale"] = scale
        components["hole_scale"] = hole_scale
        components["total"] = float(total.detach().cpu())
    return total, components


def mraf_amplitude_loss(
    field: torch.Tensor,
    target: TargetSpec,
    background_bias: float,
    mixing: float = 0.05,
) -> torch.Tensor:
    """Differentiable MRAF/OMRAF-style signal/background amplitude objective."""
    if not 0.0 <= mixing <= 1.0:
        raise ValueError("mixing must be in [0, 1]")
    amplitude = normalized_amplitude(field)
    target_amplitude = torch.sqrt(target.intensity)
    signal = target.signal_mask
    background = target.background_mask
    bias = torch.as_tensor(background_bias, device=field.device, dtype=amplitude.dtype)
    background_target = mixing * amplitude.detach() + (1.0 - mixing) * bias
    desired = signal * target_amplitude + background * background_target
    active = signal + background
    return torch.sum(active * (amplitude - desired) ** 2) / (active.sum() + 1e-12)
