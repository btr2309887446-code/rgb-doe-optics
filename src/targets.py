"""Target patterns and signal/background regions."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Callable, Iterable

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class TargetSpec:
    intensity: torch.Tensor
    signal_mask: torch.Tensor
    transition_mask: torch.Tensor
    background_mask: torch.Tensor
    component_masks: tuple[torch.Tensor, ...]
    metadata: dict[str, object]
    hole_mask: torch.Tensor | None = None
    ring_mask: torch.Tensor | None = None
    target_type: str = "generic"

    @property
    def left_mask(self) -> torch.Tensor:
        return self.component_masks[0]

    @property
    def right_mask(self) -> torch.Tensor:
        return self.component_masks[1]

    @property
    def spot_x_m(self) -> float:
        centers = self.metadata.get("centers_m", ((0.0, 0.0),))
        return abs(float(centers[0][0]))

    @property
    def radius_m(self) -> float:
        return float(self.metadata.get("radius_m", 40e-6))

    @property
    def edge_width_m(self) -> float:
        return float(self.metadata.get("edge_width_m", 15e-6))


def _make_target(
    centers_m: Iterable[tuple[float, float]],
    profiles: Iterable[torch.Tensor],
    masks: Iterable[torch.Tensor],
    transition_masks: Iterable[torch.Tensor],
    metadata: dict[str, object],
    hole_masks: Iterable[torch.Tensor] | None = None,
    ring_masks: Iterable[torch.Tensor] | None = None,
    target_type: str = "generic",
) -> TargetSpec:
    profiles = tuple(profiles)
    masks = tuple(mask.to(torch.float32) for mask in masks)
    transition_masks = tuple(mask.to(torch.bool) for mask in transition_masks)
    intensity = sum(profiles)
    intensity = intensity / (intensity.sum() + 1e-12)
    signal = torch.zeros_like(intensity, dtype=torch.bool)
    transition = torch.zeros_like(signal)
    for mask in masks:
        signal |= mask > 0.5
    for mask in transition_masks:
        transition |= mask
    transition &= ~signal
    background = ~(signal | transition)
    hole = torch.zeros_like(signal)
    for mask in hole_masks or ():
        hole |= mask.to(torch.bool)
    ring = torch.zeros_like(signal)
    for mask in ring_masks or ():
        ring |= mask.to(torch.bool)
    metadata = dict(metadata)
    metadata["centers_m"] = tuple(tuple(float(value) for value in center) for center in centers_m)
    metadata["component_count"] = len(masks)
    return TargetSpec(
        intensity=intensity,
        signal_mask=signal.to(torch.float32),
        transition_mask=transition.to(torch.float32),
        background_mask=background.to(torch.float32),
        component_masks=masks,
        metadata=metadata,
        hole_mask=hole.to(torch.float32) if hole_masks is not None else None,
        ring_mask=ring.to(torch.float32) if ring_masks is not None else None,
        target_type=target_type,
    )


def _circle_profile(
    x: torch.Tensor,
    y: torch.Tensor,
    center: tuple[float, float],
    radius_m: float,
    edge_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    profile = 0.5 * (1.0 - torch.tanh((distance - radius_m) / edge_width_m))
    signal = distance <= radius_m + edge_width_m
    transition = distance <= radius_m + 2.0 * edge_width_m
    return profile, signal, transition


def _gaussian_profile(
    x: torch.Tensor,
    y: torch.Tensor,
    center: tuple[float, float],
    sigma_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    distance_squared = (x - center[0]) ** 2 + (y - center[1]) ** 2
    profile = torch.exp(-distance_squared / (2.0 * sigma_m**2))
    signal = distance_squared <= (2.0 * sigma_m) ** 2
    transition = distance_squared <= (3.0 * sigma_m) ** 2
    return profile, signal, transition


def _ring_profile(
    x: torch.Tensor,
    y: torch.Tensor,
    center: tuple[float, float],
    radius_m: float,
    edge_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    distance = torch.sqrt((x - center[0]) ** 2 + (y - center[1]) ** 2)
    profile = torch.exp(-((distance - radius_m) ** 2) / (2.0 * edge_width_m**2))
    signal = torch.abs(distance - radius_m) <= 2.0 * edge_width_m
    transition = torch.abs(distance - radius_m) <= 3.0 * edge_width_m
    hole = distance <= max(radius_m - 2.5 * edge_width_m, 0.0)
    return profile, signal, transition, hole


def _rectangle_profile(
    x: torch.Tensor,
    y: torch.Tensor,
    center: tuple[float, float],
    half_width_m: float,
    half_height_m: float,
    edge_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.abs(x - center[0])
    dy = torch.abs(y - center[1])
    profile = 0.5 * (1.0 - torch.tanh((dx - half_width_m) / edge_width_m))
    profile = profile * 0.5 * (1.0 - torch.tanh((dy - half_height_m) / edge_width_m))
    signal = (dx <= half_width_m + edge_width_m) & (dy <= half_height_m + edge_width_m)
    transition = (dx <= half_width_m + 2.0 * edge_width_m) & (
        dy <= half_height_m + 2.0 * edge_width_m
    )
    return profile, signal, transition


def _cross_profile(
    x: torch.Tensor,
    y: torch.Tensor,
    center: tuple[float, float],
    arm_length_m: float,
    arm_width_m: float,
    edge_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dx = torch.abs(x - center[0])
    dy = torch.abs(y - center[1])
    horizontal = 0.5 * (1.0 - torch.tanh((dx - arm_length_m) / edge_width_m))
    horizontal = horizontal * 0.5 * (1.0 - torch.tanh((dy - arm_width_m) / edge_width_m))
    vertical = 0.5 * (1.0 - torch.tanh((dx - arm_width_m) / edge_width_m))
    vertical = vertical * 0.5 * (1.0 - torch.tanh((dy - arm_length_m) / edge_width_m))
    profile = torch.clamp(horizontal + vertical, 0.0, 1.0)
    signal = ((dx <= arm_length_m + edge_width_m) & (dy <= arm_width_m + edge_width_m)) | (
        (dx <= arm_width_m + edge_width_m) & (dy <= arm_length_m + edge_width_m)
    )
    transition = ((dx <= arm_length_m + 2.0 * edge_width_m) & (dy <= arm_width_m + 2.0 * edge_width_m)) | (
        (dx <= arm_width_m + 2.0 * edge_width_m) & (dy <= arm_length_m + 2.0 * edge_width_m)
    )
    return profile, signal, transition


def _soft_triangle_profile(
    x: torch.Tensor,
    y: torch.Tensor,
    circumradius_m: float,
    edge_width_m: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, tuple[tuple[float, float], ...]]:
    vertices = (
        (0.0, circumradius_m),
        (-math.sqrt(3.0) * circumradius_m / 2.0, -circumradius_m / 2.0),
        (math.sqrt(3.0) * circumradius_m / 2.0, -circumradius_m / 2.0),
    )
    signed_distances = []
    for index in range(3):
        first = vertices[index]
        second = vertices[(index + 1) % 3]
        edge_x = second[0] - first[0]
        edge_y = second[1] - first[1]
        length = math.sqrt(edge_x**2 + edge_y**2)
        signed = (edge_x * (y - first[1]) - edge_y * (x - first[0])) / length
        signed_distances.append(signed)
    inside_soft = [0.5 * (1.0 - torch.tanh(-distance / edge_width_m)) for distance in signed_distances]
    profile = inside_soft[0] * inside_soft[1] * inside_soft[2]
    inside = (signed_distances[0] >= 0) & (signed_distances[1] >= 0) & (signed_distances[2] >= 0)
    transition = (
        (signed_distances[0] >= -2.0 * edge_width_m)
        & (signed_distances[1] >= -2.0 * edge_width_m)
        & (signed_distances[2] >= -2.0 * edge_width_m)
    )
    centers = tuple((float(vertex[0]), float(vertex[1])) for vertex in vertices)
    return profile, inside, transition, centers


def _validate_geometry(centers_m: Iterable[tuple[float, float]], radius_m: float) -> None:
    centers = tuple(centers_m)
    if len(centers) < 2 or radius_m <= 0:
        raise ValueError("at least two centers and positive radius are required")
    for index, first in enumerate(centers):
        for second in centers[index + 1:]:
            distance = ((first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2) ** 0.5
            if distance <= 2.0 * radius_m:
                raise ValueError("target component regions overlap")


def soft_circle_spots(
    x: torch.Tensor,
    y: torch.Tensor,
    centers_m: Iterable[tuple[float, float]],
    radius_m: float,
    edge_width_m: float,
    name: str = "soft_circle",
) -> TargetSpec:
    centers = tuple(centers_m)
    _validate_geometry(centers, radius_m)
    values = [_circle_profile(x, y, center, radius_m, edge_width_m) for center in centers]
    return _make_target(
        centers,
        [value[0] for value in values],
        [value[1] for value in values],
        [value[2] for value in values],
        {"name": name, "radius_m": radius_m, "edge_width_m": edge_width_m},
        target_type="soft_circle",
    )


def soft_circle_two_spots(
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x_m: float = 0.30e-3,
    radius_m: float = 40e-6,
    edge_width_m: float = 15e-6,
) -> TargetSpec:
    return soft_circle_spots(
        x,
        y,
        ((-spot_x_m, 0.0), (spot_x_m, 0.0)),
        radius_m,
        edge_width_m,
        "soft_circle_two",
    )


def gaussian_two_spots(
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x_m: float = 0.30e-3,
    sigma_m: float = 40e-6,
) -> TargetSpec:
    centers = ((-spot_x_m, 0.0), (spot_x_m, 0.0))
    values = [_gaussian_profile(x, y, center, sigma_m) for center in centers]
    return _make_target(
        centers,
        [value[0] for value in values],
        [value[1] for value in values],
        [value[2] for value in values],
        {"name": "gaussian_two", "sigma_m": sigma_m},
        target_type="gaussian",
    )


def ring_two_spots(
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x_m: float = 0.30e-3,
    radius_m: float = 50e-6,
    edge_width_m: float = 8e-6,
) -> TargetSpec:
    centers = ((-spot_x_m, 0.0), (spot_x_m, 0.0))
    values = [_ring_profile(x, y, center, radius_m, edge_width_m) for center in centers]
    return _make_target(
        centers,
        [value[0] for value in values],
        [value[1] for value in values],
        [value[2] for value in values],
        {
            "name": "ring_two",
            "radius_m": radius_m,
            "ring_radius_m": radius_m,
            "edge_width_m": edge_width_m,
            "ring_width_m": edge_width_m,
        },
        hole_masks=[value[3] for value in values],
        ring_masks=[value[1] for value in values],
        target_type="ring",
    )


def asymmetric_two_spots(
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x_m: float = 0.30e-3,
    radius_m: float = 40e-6,
    edge_width_m: float = 15e-6,
    right_scale: float = 0.65,
) -> TargetSpec:
    centers = ((-spot_x_m, 0.0), (spot_x_m, 0.0))
    values = [_circle_profile(x, y, center, radius_m, edge_width_m) for center in centers]
    return _make_target(
        centers,
        [values[0][0], right_scale * values[1][0]],
        [value[1] for value in values],
        [value[2] for value in values],
        {
            "name": "asymmetric_two",
            "radius_m": radius_m,
            "edge_width_m": edge_width_m,
            "right_scale": right_scale,
        },
        target_type="asymmetric_circle",
    )


def triangle_three_spots(
    x: torch.Tensor,
    y: torch.Tensor,
    radius_m: float = 40e-6,
    edge_width_m: float = 15e-6,
) -> TargetSpec:
    return soft_circle_spots(
        x,
        y,
        ((-300e-6, 0.0), (300e-6, 0.0), (0.0, 300e-6)),
        radius_m,
        edge_width_m,
        "triangle_three",
    )


def soft_square_two_spots(
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x_m: float = 0.30e-3,
    side_m: float = 80e-6,
    edge_width_m: float = 15e-6,
) -> TargetSpec:
    centers = ((-spot_x_m, 0.0), (spot_x_m, 0.0))
    half_side = side_m / 2.0
    values = [
        _rectangle_profile(x, y, center, half_side, half_side, edge_width_m)
        for center in centers
    ]
    return _make_target(
        centers,
        [value[0] for value in values],
        [value[1] for value in values],
        [value[2] for value in values],
        {
            "name": "soft_square_two",
            "side_m": side_m,
            "edge_width_m": edge_width_m,
        },
        target_type="soft_square",
    )


def cross_two_spots(
    x: torch.Tensor,
    y: torch.Tensor,
    spot_x_m: float = 0.30e-3,
    arm_length_m: float = 40e-6,
    arm_width_m: float = 10e-6,
    edge_width_m: float = 8e-6,
) -> TargetSpec:
    centers = ((-spot_x_m, 0.0), (spot_x_m, 0.0))
    values = [
        _cross_profile(x, y, center, arm_length_m, arm_width_m, edge_width_m)
        for center in centers
    ]
    return _make_target(
        centers,
        [value[0] for value in values],
        [value[1] for value in values],
        [value[2] for value in values],
        {
            "name": "cross_two",
            "arm_length_m": arm_length_m,
            "arm_width_m": arm_width_m,
            "edge_width_m": edge_width_m,
        },
        target_type="cross",
    )


def solid_triangle(
    x: torch.Tensor,
    y: torch.Tensor,
    circumradius_m: float = 60e-6,
    edge_width_m: float = 10e-6,
) -> TargetSpec:
    profile, signal, transition, vertices = _soft_triangle_profile(
        x, y, circumradius_m, edge_width_m
    )
    center = ((0.0, 0.0),)
    return _make_target(
        center,
        [profile],
        [signal],
        [transition],
        {
            "name": "solid_triangle",
            "circumradius_m": circumradius_m,
            "edge_width_m": edge_width_m,
            "vertices_m": vertices,
        },
        target_type="solid_triangle",
    )


def target_factories() -> dict[str, Callable[..., TargetSpec]]:
    return {
        "soft_circle_two": soft_circle_two_spots,
        "gaussian_two": gaussian_two_spots,
        "ring_two": ring_two_spots,
        "asymmetric_two": asymmetric_two_spots,
        "triangle_three": triangle_three_spots,
        "soft_square_two": soft_square_two_spots,
        "cross_two": cross_two_spots,
        "solid_triangle": solid_triangle,
    }


def supersampled_edge_target(
    target_factory: Callable[[torch.Tensor, torch.Tensor], TargetSpec],
    n_grid: int,
    pixel_pitch_m: float,
    device: torch.device,
    sharpness_gain: float = 1.0,
    supersample_factor: int = 4,
) -> TargetSpec:
    if n_grid < 2 or pixel_pitch_m <= 0:
        raise ValueError("grid size and pixel pitch must be positive")
    if sharpness_gain < 1.0:
        raise ValueError("sharpness_gain must be at least 1")
    if supersample_factor < 1:
        raise ValueError("supersample_factor must be at least 1")
    high_n = n_grid * supersample_factor
    subpixel_offset = (supersample_factor - 1) / 2.0
    coordinate = (
        torch.arange(high_n, device=device, dtype=torch.float32)
        - high_n // 2
        - subpixel_offset
    ) * (pixel_pitch_m / supersample_factor)
    high_x, high_y = torch.meshgrid(coordinate, coordinate, indexing="xy")
    high_target = target_factory(high_x, high_y)

    def downsample(value: torch.Tensor) -> torch.Tensor:
        if supersample_factor == 1:
            return value.to(torch.float32)
        return functional.avg_pool2d(
            value.to(torch.float32)[None, None, :, :],
            kernel_size=supersample_factor,
            stride=supersample_factor,
        )[0, 0]

    intensity = downsample(high_target.intensity)
    if sharpness_gain > 1.0:
        relative = (intensity / intensity.max().clamp_min(1e-12)).clamp(1e-7, 1.0 - 1e-7)
        log_odds = torch.log(relative) - torch.log1p(-relative)
        intensity = torch.sigmoid(sharpness_gain * log_odds)
    intensity = intensity / intensity.sum().clamp_min(1e-12)
    component_masks = tuple(downsample(mask) for mask in high_target.component_masks)
    signal_mask = downsample(high_target.signal_mask)
    transition_mask = downsample(high_target.transition_mask)
    background_mask = downsample(high_target.background_mask)
    hole_mask = downsample(high_target.hole_mask) if high_target.hole_mask is not None else None
    ring_mask = downsample(high_target.ring_mask) if high_target.ring_mask is not None else None
    metadata = dict(high_target.metadata)
    metadata["edge_preprocess"] = "supersampled_logit_sharpening"
    metadata["edge_sharpness_gain"] = float(sharpness_gain)
    metadata["supersample_factor"] = int(supersample_factor)
    return TargetSpec(
        intensity=intensity,
        signal_mask=signal_mask,
        transition_mask=transition_mask,
        background_mask=background_mask,
        component_masks=component_masks,
        metadata=metadata,
        hole_mask=hole_mask,
        ring_mask=ring_mask,
        target_type=high_target.target_type,
    )
