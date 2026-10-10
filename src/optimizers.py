"""Adam, MRAF/OMRAF-style, and direct binary/level search optimizers."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Callable, Iterable

import torch

from losses import LossSchedule, LossWeights, joint_loss, mraf_amplitude_loss
from targets import TargetSpec

FieldForward = Callable[[torch.Tensor], list[torch.Tensor]]


@dataclass(frozen=True)
class AdamOptions:
    iterations: int = 600
    learning_rate_um: float = 0.04
    gradient_clip: float = 1.0
    cosine_decay: bool = True
    seed: int = 7
    wavelength_weights: tuple[float, ...] | None = None
    mraf_background_bias: float | None = None
    mraf_mixing: float = 0.05
    pixel_pitch_m: float = 5e-6
    log_every: int = 1
    input_powers: tuple[float, ...] | None = None
    efficiency_floor: float | torch.Tensor | None = None


@dataclass
class OptimizationResult:
    height_m: torch.Tensor
    losses: list[float]
    components: list[dict[str, float]]
    method: str


def _initial_height(n_grid: int, height_max_m: float, seed: int, device: torch.device) -> torch.Tensor:
    generator = torch.Generator(device=device).manual_seed(seed)
    return height_max_m * torch.rand((n_grid, n_grid), generator=generator, device=device)


def optimize_adam(
    forward: FieldForward,
    target: TargetSpec,
    height_max_m: float,
    n_grid: int,
    aperture_mask: torch.Tensor,
    options: AdamOptions,
    weights: LossWeights,
    schedule: LossSchedule,
    initial_height_m: torch.Tensor | None = None,
) -> OptimizationResult:
    if initial_height_m is None:
        initial_height_m = _initial_height(n_grid, height_max_m, options.seed, aperture_mask.device)
    else:
        if initial_height_m.ndim not in (2, 3) or initial_height_m.shape[-2:] != (n_grid, n_grid):
            raise ValueError("initial_height_m must end with dimensions (n_grid, n_grid)")
        initial_height_m = initial_height_m.detach().clone().to(aperture_mask.device)
    if options.log_every < 1:
        raise ValueError("log_every must be positive")
    height = torch.nn.Parameter(initial_height_m)
    optimizer = torch.optim.Adam([height], lr=options.learning_rate_um * 1e-6)
    scheduler = (
        torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, options.iterations))
        if options.cosine_decay else None
    )
    losses: list[float] = []
    components: list[dict[str, float]] = []
    for step in range(options.iterations):
        log_now = step % options.log_every == 0 or step == options.iterations - 1
        optimizer.zero_grad(set_to_none=True)
        fields = forward(height)
        total, current_components = joint_loss(
            fields,
            target,
            height,
            height_max_m,
            weights,
            schedule,
            step,
            options.wavelength_weights,
            options.pixel_pitch_m,
            record_components=log_now,
            input_powers=options.input_powers,
            efficiency_floor=options.efficiency_floor,
        )
        if options.mraf_background_bias is not None:
            mraf_terms = torch.stack([
                mraf_amplitude_loss(field, target, options.mraf_background_bias, options.mraf_mixing)
                for field in fields
            ])
            wavelength_weights = options.wavelength_weights or tuple(1.0 for _ in fields)
            mraf_weights = torch.as_tensor(
                wavelength_weights, device=mraf_terms.device, dtype=mraf_terms.dtype
            )
            mraf = torch.sum(mraf_terms * mraf_weights) / (mraf_weights.sum() + 1e-12)
            total = total + mraf
            if log_now:
                current_components["mraf"] = float(mraf.detach().cpu())
                current_components["total"] = float(total.detach().cpu())
        if not torch.isfinite(total):
            raise FloatingPointError(f"nonfinite objective at step {step}")
        total.backward()
        if height.grad is None or not torch.isfinite(height.grad).all():
            raise FloatingPointError(f"missing or nonfinite height gradient at step {step}")
        if options.gradient_clip > 0:
            torch.nn.utils.clip_grad_norm_([height], options.gradient_clip * height_max_m)
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        with torch.no_grad():
            height.clamp_(0.0, height_max_m)
            height.mul_(aperture_mask)
        if log_now:
            losses.append(float(total.detach().cpu()))
            components.append(current_components)
    return OptimizationResult(
        height.detach(), losses, components,
        "mraf_adam" if options.mraf_background_bias is not None else "adam",
    )


def optimize_mraf(
    forward: FieldForward,
    target: TargetSpec,
    height_max_m: float,
    n_grid: int,
    aperture_mask: torch.Tensor,
    options: AdamOptions,
    weights: LossWeights,
    schedule: LossSchedule,
    background_bias: float,
    mixing: float = 0.05,
    initial_height_m: torch.Tensor | None = None,
) -> OptimizationResult:
    if not 0.0 <= background_bias:
        raise ValueError("background_bias must be nonnegative")
    mraf_options = replace(
        options,
        mraf_background_bias=background_bias,
        mraf_mixing=mixing,
    )
    result = optimize_adam(
        forward,
        target,
        height_max_m,
        n_grid,
        aperture_mask,
        mraf_options,
        weights,
        schedule,
        initial_height_m,
    )
    result.method = "mraf"
    return result


def quantize_height(height_m: torch.Tensor, height_max_m: float, levels: int) -> torch.Tensor:
    if levels < 2:
        raise ValueError("levels must be >= 2")
    step = height_max_m / (levels - 1)
    return torch.round(height_m / step).clamp(0, levels - 1) * step


def optimize_dbs(
    forward: FieldForward,
    target: TargetSpec,
    initial_height_m: torch.Tensor,
    height_max_m: float,
    aperture_mask: torch.Tensor,
    weights: LossWeights,
    levels: int = 64,
    sweeps: int = 10,
    seed: int = 7,
    max_pixels_per_sweep: int | None = None,
) -> OptimizationResult:
    """Greedy level search; max_pixels_per_sweep enables coarse smoke runs."""
    height = quantize_height(initial_height_m.detach(), height_max_m, levels) * aperture_mask
    rng = torch.Generator(device=height.device).manual_seed(seed)
    aperture_indices = torch.nonzero(aperture_mask.flatten() > 0.5, as_tuple=False).flatten()
    losses: list[float] = []
    components: list[dict[str, float]] = []

    def objective(candidate: torch.Tensor) -> tuple[torch.Tensor, dict[str, float]]:
        fields = forward(candidate)
        return joint_loss(fields, target, candidate, height_max_m, weights, LossSchedule(0, 0), 9999)

    current, current_components = objective(height)
    for sweep in range(sweeps):
        order = aperture_indices[torch.randperm(aperture_indices.numel(), generator=rng)]
        if max_pixels_per_sweep is not None:
            order = order[:max_pixels_per_sweep]
        for flat_index in order.tolist():
            row, column = divmod(flat_index, height.shape[-1])
            old = height[row, column].clone()
            step = height_max_m / (levels - 1)
            candidates = [old]
            if float(old.detach().cpu()) > 0:
                candidates.append(old - step)
            if float(old.detach().cpu()) < height_max_m:
                candidates.append(old + step)
            best_height = height
            best_value = current
            for candidate_value in candidates[1:]:
                trial = height.clone()
                trial[row, column] = candidate_value
                trial_value, _ = objective(trial)
                if trial_value < best_value:
                    best_height = trial
                    best_value = trial_value
            height = best_height
            current = best_value
        current, current_components = objective(height)
        current_components = dict(current_components)
        current_components["sweep"] = float(sweep + 1)
        losses.append(float(current.detach().cpu()))
        components.append(current_components)
    return OptimizationResult(height.detach(), losses, components, f"dbs_{levels}levels")
