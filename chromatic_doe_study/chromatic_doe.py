"""Physically constrained broadband inverse design for a static relief DOE.

The forward model evaluates the scalar Fraunhofer field on a fixed physical
detector grid behind an ideal achromatic thin lens. One material height map is
shared by all wavelengths. A straight-through multilevel quantizer keeps the
training design at a specified fabrication level count.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch


@dataclass(frozen=True)
class Config:
    n_grid: int = 48
    detector_grid: int = 128
    dx_m: float = 6e-6
    aperture_radius_m: float = 140e-6
    focal_length_m: float = 50e-3
    detector_half_width_m: float = 1.2e-3
    height_max_um: float = 1.4
    height_levels: int = 32
    target_sigma_m: float = 72e-6
    efficiency_floor: float = 0.12
    efficiency_weight: float = 0.6
    consistency_weight: float = 0.2
    smoothness_weight: float = 0.05
    worst_weight: float = 0.5
    softmax_temperature: float = 0.04


DESIGN_NM = (488.0, 532.0, 633.0)
TARGET_SPOTS = ((-320e-6, 0.0, 1.0), (190e-6, 240e-6, 0.72), (120e-6, -290e-6, 0.48))


def s1813_index(wavelength_m: torch.Tensor) -> torch.Tensor:
    """Cauchy fit through the S1813 index anchors reported by Kim et al. (2012)."""
    l1, l2 = 532e-9, 633e-9
    n1, n2 = 1.6482, 1.6347
    b = (n1 - n2) / (1.0 / l1**2 - 1.0 / l2**2)
    a = n1 - b / l1**2
    return a + b / wavelength_m**2


def make_coordinates(cfg: Config, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    n = cfg.n_grid
    x = (torch.arange(n, device=device, dtype=torch.float64) - n // 2) * cfg.dx_m
    m = cfg.detector_grid
    x_det = (torch.arange(m, device=device, dtype=torch.float64) - (m - 1) / 2) * (
        2.0 * cfg.detector_half_width_m / (m - 1)
    )
    xx, yy = torch.meshgrid(x, x, indexing="xy")
    pupil = ((xx.square() + yy.square()) <= cfg.aperture_radius_m**2).to(torch.float64)
    return x, x_det, pupil


class FraunhoferModel:
    """Separable direct Fourier-integral evaluator on common physical X,Y."""

    def __init__(self, cfg: Config, wavelengths_nm: Sequence[float], device: torch.device):
        self.cfg = cfg
        self.device = device
        self.wavelengths_nm = tuple(float(v) for v in wavelengths_nm)
        self.wavelengths = torch.tensor(
            [v * 1e-9 for v in self.wavelengths_nm], device=device, dtype=torch.float64
        )
        self.x, self.x_det, self.pupil = make_coordinates(cfg, device)
        self.dx_out = float(self.x_det[1] - self.x_det[0])
        self.input_power = float(torch.sum(self.pupil).item()) * cfg.dx_m**2
        self.target, self.roi = make_target(cfg, self.x_det, device)
        phase = -2j * math.pi * (
            self.x_det[None, :, None] * self.x[None, None, :]
        ) / (self.wavelengths[:, None, None] * cfg.focal_length_m)
        self.kernel = torch.complex(torch.cos(phase.imag), torch.sin(phase.imag))
        self.kernel_t = self.kernel.transpose(-1, -2)
        self.prefactor = (cfg.dx_m**2 / (self.wavelengths * cfg.focal_length_m))[:, None, None]

    def quantize(self, height_um: torch.Tensor, levels: int | None) -> torch.Tensor:
        bounded = height_um.clamp(0.0, self.cfg.height_max_um) * self.pupil
        if levels is None or levels < 2:
            return bounded
        step = self.cfg.height_max_um / (levels - 1)
        hard = torch.round(bounded / step).clamp(0, levels - 1) * step
        # The physical forward pass is quantized; gradients pass through the
        # continuous height parameter using a straight-through estimator.
        return bounded + (hard - bounded).detach()

    def forward(self, height_um: torch.Tensor, levels: int | None = None) -> torch.Tensor:
        h_um = self.quantize(height_um, levels)
        h_m = h_um * 1e-6
        n = s1813_index(self.wavelengths)
        phase = 2.0 * math.pi * (n - 1.0)[:, None, None] * h_m[None] / self.wavelengths[:, None, None]
        screen = self.pupil[None] * torch.complex(torch.cos(phase), torch.sin(phase))
        transformed = torch.matmul(torch.matmul(self.kernel, screen), self.kernel_t)
        field = transformed * self.prefactor
        return field.abs().square()

    def power_efficiency(self, intensity: torch.Tensor) -> torch.Tensor:
        return intensity.sum(dim=(-2, -1)) * self.dx_out**2 / self.input_power

    def roi_efficiency(self, intensity: torch.Tensor) -> torch.Tensor:
        roi = self.roi.to(dtype=intensity.dtype)
        return (intensity * roi).sum(dim=(-2, -1)) * self.dx_out**2 / self.input_power


def make_target(cfg: Config, x_det: torch.Tensor, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
    xx, yy = torch.meshgrid(x_det, x_det, indexing="xy")
    target = torch.zeros_like(xx)
    for x0, y0, weight in TARGET_SPOTS:
        radius2 = (xx - x0) ** 2 + (yy - y0) ** 2
        target = target + weight * torch.exp(-radius2 / (2.0 * cfg.target_sigma_m**2))
    target = target / target.sum()
    roi = target >= 0.08 * target.max()
    return target.to(device), roi.to(device)


def loss_components(
    model: FraunhoferModel,
    height_um: torch.Tensor,
    levels: int | None,
    cfg: Config,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    intensity = model.forward(height_um, levels)
    normalized = intensity / (intensity.sum(dim=(-2, -1), keepdim=True) + 1e-30)
    target = model.target[None]
    denominator = target.square().sum() + 1e-30
    per_wave = ((normalized - target).square().sum(dim=(-2, -1)) / denominator)
    smooth_max = cfg.softmax_temperature * torch.logsumexp(
        per_wave / cfg.softmax_temperature, dim=0
    )
    shape = (1.0 - cfg.worst_weight) * per_wave.mean() + cfg.worst_weight * smooth_max
    pairwise = []
    for i in range(normalized.shape[0]):
        for j in range(i + 1, normalized.shape[0]):
            pairwise.append((normalized[i] - normalized[j]).square().mean())
    consistency = torch.stack(pairwise).mean() if pairwise else torch.zeros_like(shape)
    efficiency = model.roi_efficiency(intensity)
    efficiency_penalty = torch.relu(cfg.efficiency_floor - efficiency).square().mean()
    bounded_h = model.quantize(height_um, levels)
    tv = (
        (bounded_h[1:, :] - bounded_h[:-1, :]).square().mean()
        + (bounded_h[:, 1:] - bounded_h[:, :-1]).square().mean()
    ) / (cfg.height_max_um**2 + 1e-30)
    total = (
        shape
        + cfg.consistency_weight * consistency
        + cfg.efficiency_weight * efficiency_penalty
        + cfg.smoothness_weight * tv
    )
    return total, {
        "shape": shape,
        "shape_by_wavelength": per_wave,
        "consistency": consistency,
        "roi_efficiency": efficiency,
        "efficiency_penalty": efficiency_penalty,
        "smoothness": tv,
        "intensity": intensity,
    }


def optimize(
    cfg: Config,
    wavelengths_nm: Sequence[float],
    initial_um: np.ndarray | torch.Tensor,
    iterations: int,
    levels: int,
    device: torch.device,
) -> tuple[np.ndarray, list[float]]:
    model = FraunhoferModel(cfg, wavelengths_nm, device)
    initial = torch.as_tensor(initial_um, device=device, dtype=torch.float64)
    h = torch.nn.Parameter(initial.clone())
    optimizer = torch.optim.Adam([h], lr=0.035)
    history: list[float] = []
    for step in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = loss_components(model, h, levels, cfg)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"non-finite loss at iteration {step}")
        loss.backward()
        optimizer.step()
        with torch.no_grad():
            h.clamp_(0.0, cfg.height_max_um)
            h.mul_(model.pupil)
        history.append(float(loss.detach().cpu()))
    return h.detach().cpu().numpy(), history


def evaluate_design(
    name: str,
    heights_um: np.ndarray,
    cfg: Config,
    wavelengths_nm: Sequence[float],
    levels: int | None,
    device: torch.device,
    noise_sigma_nm: float = 0.0,
    seed: int = 0,
) -> tuple[list[dict[str, float | str]], np.ndarray]:
    model = FraunhoferModel(cfg, wavelengths_nm, device)
    h = torch.as_tensor(heights_um, device=device, dtype=torch.float64)
    if noise_sigma_nm > 0:
        # Add height error to the fabricated, quantized relief, not to the
        # optimizer's latent continuous parameters.
        h = model.quantize(h, levels if levels is not None else cfg.height_levels).detach()
        generator = torch.Generator(device=device).manual_seed(seed)
        noise = torch.randn(h.shape, device=device, dtype=h.dtype, generator=generator)
        h = (h + noise * (noise_sigma_nm * 1e-3)).clamp(0.0, cfg.height_max_um) * model.pupil
        effective_levels = None
    else:
        effective_levels = levels
    with torch.no_grad():
        intensity = model.forward(h, effective_levels)
        normalized = intensity / (intensity.sum(dim=(-2, -1), keepdim=True) + 1e-30)
        target = model.target
        rows: list[dict[str, float | str]] = []
        roi = model.roi
        for i, wavelength in enumerate(wavelengths_nm):
            p = normalized[i]
            a = p[roi]
            b = target[roi]
            corr = torch.corrcoef(torch.stack([a, b]))[0, 1] if a.numel() > 1 else torch.tensor(0.0, device=device)
            rows.append({
                "design": name,
                "wavelength_nm": float(wavelength),
                "relative_l2_shape": float(torch.linalg.vector_norm(p - target) / (torch.linalg.vector_norm(target) + 1e-30)),
                "pearson_roi": float(corr),
                "roi_efficiency_input_fraction": float(model.roi_efficiency(intensity[i:i+1])[0]),
                "detector_capture_input_fraction": float(model.power_efficiency(intensity[i:i+1])[0]),
            })
        return rows, intensity.detach().cpu().numpy()


def run_self_test(device: torch.device) -> dict[str, float]:
    # A matched DFT lattice must agree with the direct physical Fourier integral.
    n, wavelength, f, dx = 16, 532e-9, 0.04, 4e-6
    x = (torch.arange(n, device=device, dtype=torch.float64) - n // 2) * dx
    freq = torch.fft.fftshift(torch.fft.fftfreq(n, d=dx, device=device, dtype=torch.float64))
    x_out = freq * wavelength * f
    angle = -2j * math.pi * x_out[:, None] * x[None, :] / (wavelength * f)
    kernel = torch.complex(torch.cos(angle.imag), torch.sin(angle.imag))
    g = torch.Generator(device=device).manual_seed(91)
    real = torch.randn((n, n), generator=g, device=device, dtype=torch.float64)
    imag = torch.randn((n, n), generator=g, device=device, dtype=torch.float64)
    field = torch.complex(real, imag)
    direct = (dx**2 / (wavelength * f)) * kernel @ field @ kernel.T
    fft_field = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(field))) * (dx**2 / (wavelength * f))
    dft_error = float(torch.linalg.vector_norm(direct - fft_field) / torch.linalg.vector_norm(fft_field))

    # Validate the differentiable material-phase objective with a directional FD.
    cfg = Config(n_grid=16, detector_grid=24, dx_m=5e-6, aperture_radius_m=32e-6,
                 detector_half_width_m=0.6e-3, height_levels=8)
    model = FraunhoferModel(cfg, [488.0, 532.0, 633.0], device)
    initial = torch.rand((cfg.n_grid, cfg.n_grid), generator=g, device=device, dtype=torch.float64) * cfg.height_max_um
    initial = (initial * model.pupil).requires_grad_(True)
    value, _ = loss_components(model, initial, None, cfg)
    gradient = torch.autograd.grad(value, initial)[0]
    direction = torch.randn(initial.shape, generator=g, device=device, dtype=initial.dtype) * model.pupil
    direction = direction / torch.linalg.vector_norm(direction)
    analytic = torch.sum(gradient * direction)
    eps = 1e-4
    with torch.no_grad():
        plus, _ = loss_components(model, initial + eps * direction, None, cfg)
        minus, _ = loss_components(model, initial - eps * direction, None, cfg)
        finite = (plus - minus) / (2 * eps)
    grad_error = float(torch.abs(analytic - finite) / (torch.abs(analytic) + torch.abs(finite) + 1e-12))

    # Discrete Parseval check on the same DFT grid.
    pupil = torch.ones((n, n), device=device, dtype=torch.float64)
    spectrum = torch.fft.fftshift(torch.fft.fft2(torch.fft.ifftshift(pupil)))
    dx_out = wavelength * f / (n * dx)
    field_out = spectrum * (dx**2 / (wavelength * f))
    p_in = pupil.square().sum() * dx**2
    p_out = field_out.abs().square().sum() * dx_out**2
    energy_error = float(torch.abs(p_out - p_in) / p_in)
    result = {"matched_dft_relative_error": dft_error, "directional_gradient_relative_error": grad_error,
              "discrete_parseval_relative_error": energy_error}
    if dft_error > 1e-11 or grad_error > 2e-4 or energy_error > 1e-11:
        raise AssertionError(f"self-test failed: {result}")
    return result


def save_csv(path: Path, rows: Sequence[dict[str, float | str]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_figures(
    output: Path,
    cfg: Config,
    target: np.ndarray,
    designs: dict[str, np.ndarray],
    pattern_data: dict[str, np.ndarray],
    metric_rows: list[dict[str, float | str]],
    history: dict[str, list[float]],
    scan_rows: list[dict[str, float | str]],
) -> None:
    wave_index = {v: i for i, v in enumerate(DESIGN_NM)}
    shape_patterns = {
        wavelength: pattern_data["joint_32level"][wave_index[wavelength]]
        / pattern_data["joint_32level"][wave_index[wavelength]].sum()
        for wavelength in DESIGN_NM
    }
    display_max = max(float(target.max()), *(float(v.max()) for v in shape_patterns.values()))
    fig, axes = plt.subplots(1, 4, figsize=(15, 4), constrained_layout=True)
    axes[0].imshow(target, cmap="magma", origin="lower", vmin=0.0, vmax=display_max)
    axes[0].set_title("Target intensity")
    for ax, wavelength in zip(axes[1:], DESIGN_NM):
        ax.imshow(shape_patterns[wavelength], cmap="magma", origin="lower", vmin=0.0, vmax=display_max)
        ax.set_title(f"Joint DOE, {wavelength:.0f} nm")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    fig.savefig(output / "pattern_comparison.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 4), constrained_layout=True)
    for ax, (name, h) in zip(axes, list(designs.items())[:2]):
        im = ax.imshow(h, cmap="viridis", origin="lower", vmin=0, vmax=cfg.height_max_um)
        ax.set_title(name.replace("_", " "))
        ax.set_xticks([]); ax.set_yticks([])
    fig.colorbar(im, ax=axes, shrink=0.8, label="Relief height (µm)")
    fig.savefig(output / "height_maps.png", dpi=180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, values in history.items():
        ax.plot(np.arange(1, len(values) + 1), values, label=name)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Training objective")
    ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.savefig(output / "optimization_history.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    names = ["flat", "532nm_only", "joint_32level"]
    colors = ["#888888", "#2676b8", "#d04a35"]
    for name, color in zip(names, colors):
        selected = [row for row in metric_rows if row["design"] == name and row["wavelength_nm"] in DESIGN_NM]
        axes[0].plot([row["wavelength_nm"] for row in selected], [row["relative_l2_shape"] for row in selected], "o-", label=name, color=color)
        axes[1].plot([row["wavelength_nm"] for row in selected], [row["roi_efficiency_input_fraction"] for row in selected], "o-", label=name, color=color)
    axes[0].set_ylabel("Relative L2 (normalized shape)")
    axes[1].set_ylabel("ROI fraction of incident power")
    for ax in axes:
        ax.set_xlabel("Wavelength (nm)"); ax.grid(alpha=0.25); ax.legend(frameon=False)
    fig.savefig(output / "baseline_vs_joint.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    wave = np.asarray([float(row["wavelength_nm"]) for row in scan_rows])
    for name in ["532nm_only", "joint_32level"]:
        selected = [row for row in scan_rows if row["design"] == name]
        axes[0].plot([row["wavelength_nm"] for row in selected], [row["relative_l2_shape"] for row in selected], label=name)
        axes[1].plot([row["wavelength_nm"] for row in selected], [row["roi_efficiency_input_fraction"] for row in selected], label=name)
    for ax in axes:
        for nm in DESIGN_NM:
            ax.axvline(nm, color="#777777", alpha=0.25, linewidth=0.8)
        ax.grid(alpha=0.25); ax.set_xlabel("Wavelength (nm)"); ax.legend(frameon=False)
    axes[0].set_ylabel("Relative L2")
    axes[1].set_ylabel("ROI fraction of incident power")
    fig.savefig(output / "spectral_holdout.png", dpi=180)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results"))
    parser.add_argument("--iterations", type=int, default=140)
    parser.add_argument("--starts", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--self-test-only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.iterations < 1 or args.starts < 1:
        raise SystemExit("--iterations and --starts must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA was requested but is not available")
    device = torch.device("cuda" if args.device == "cuda" or (args.device == "auto" and torch.cuda.is_available()) else "cpu")
    torch.set_num_threads(max(1, min(4, os.cpu_count() or 1)))
    checks = run_self_test(device)
    print(json.dumps({"device": str(device), "self_tests": checks}, indent=2))
    if args.self_test_only:
        return

    cfg = Config()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    model_all = FraunhoferModel(cfg, DESIGN_NM, device)
    target = model_all.target.detach().cpu().numpy()
    pupil = model_all.pupil.detach().cpu().numpy()
    rng = np.random.default_rng(args.seed)
    zero = np.zeros((cfg.n_grid, cfg.n_grid), dtype=np.float64)

    # The green-only design is a useful conventional single-colour baseline.
    green_model = FraunhoferModel(cfg, [532.0], device)
    green_candidates: list[tuple[float, np.ndarray, list[float]]] = []
    history: dict[str, list[float]] = {}
    for start in range(args.starts):
        init = rng.uniform(0.0, cfg.height_max_um, size=zero.shape) * pupil
        h, trace = optimize(cfg, [532.0], init, args.iterations, cfg.height_levels, device)
        check_model = FraunhoferModel(cfg, [532.0], device)
        check_loss, _ = loss_components(check_model, torch.as_tensor(h, device=device), cfg.height_levels, cfg)
        green_candidates.append((float(check_loss.detach().cpu()), h, trace))
    green_candidates.sort(key=lambda item: item[0])
    green_h = green_candidates[0][1]
    history["532 nm, best start"] = green_candidates[0][2]

    joint_candidates: list[tuple[float, np.ndarray, list[float], str]] = []
    joint_initials: list[tuple[str, np.ndarray]] = [("green warm start", green_h)]
    for start in range(args.starts):
        joint_initials.append((f"random start {start + 1}", rng.uniform(0.0, cfg.height_max_um, size=zero.shape) * pupil))
    for label, init in joint_initials:
        h, trace = optimize(cfg, DESIGN_NM, init, args.iterations, cfg.height_levels, device)
        check_loss, _ = loss_components(model_all, torch.as_tensor(h, device=device), cfg.height_levels, cfg)
        joint_candidates.append((float(check_loss.detach().cpu()), h, trace, label))
    joint_candidates.sort(key=lambda item: item[0])
    joint_h = joint_candidates[0][1]
    history["RGB/broadband, best start"] = joint_candidates[0][2]

    designs = {"flat": zero, "532nm_only": green_h, "joint_32level": joint_h}
    def quantized_array(values: np.ndarray, levels: int) -> np.ndarray:
        step = cfg.height_max_um / (levels - 1)
        return np.round(np.clip(values, 0.0, cfg.height_max_um) / step) * step * pupil

    physical_designs = {
        "532nm_only": quantized_array(green_h, cfg.height_levels),
        "joint_32level": quantized_array(joint_h, cfg.height_levels),
    }
    metric_rows: list[dict[str, float | str]] = []
    pattern_data: dict[str, np.ndarray] = {}
    for name, h in designs.items():
        rows, images = evaluate_design(name, h, cfg, DESIGN_NM, cfg.height_levels, device)
        metric_rows.extend(rows)
        pattern_data[name] = images
    # Manufacturing robustness: evaluate the selected physical levels at a
    # coarser 16-level process and with independent 10 nm height noise.
    quant_rows, quant_images = evaluate_design(
        "joint_16level", physical_designs["joint_32level"], cfg, DESIGN_NM, 16, device
    )
    metric_rows.extend(quant_rows); pattern_data["joint_16level"] = quant_images
    for seed in (11, 23, 37):
        noisy_rows, _ = evaluate_design(f"joint_height_noise_seed_{seed}", joint_h, cfg, DESIGN_NM,
                                        cfg.height_levels, device, noise_sigma_nm=10.0, seed=seed)
        metric_rows.extend(noisy_rows)

    # A spectral scan samples between design lines; it is a robustness check,
    # not a proof over a mathematical continuum.
    scan_wavelengths = tuple(sorted(set(float(v) for v in np.arange(488.0, 634.0, 5.0)) | set(DESIGN_NM)))
    scan_rows: list[dict[str, float | str]] = []
    for name, h in (("532nm_only", green_h), ("joint_32level", joint_h)):
        rows, _ = evaluate_design(name, h, cfg, scan_wavelengths, cfg.height_levels, device)
        scan_rows.extend(rows)

    save_csv(output / "metrics.csv", metric_rows)
    save_csv(output / "spectral_scan.csv", scan_rows)
    np.savez_compressed(output / "designs.npz", flat_um=zero,
                        single_532_optimizer_parameter_um=green_h,
                        joint_32level_optimizer_parameter_um=joint_h,
                        single_532_height_um=physical_designs["532nm_only"],
                        joint_32level_height_um=physical_designs["joint_32level"],
                        target=target, pupil=pupil, wavelengths_nm=np.asarray(DESIGN_NM),
                        joint_intensity=pattern_data["joint_32level"])
    summary = {
        "study": "shared-height multilevel S1813 DOE, fixed physical Fourier-plane target",
        "config": asdict(cfg),
        "wavelengths_design_nm": list(DESIGN_NM),
        "wavelengths_scanned_nm": list(scan_wavelengths),
        "material_indices_design": {str(w): float(s1813_index(torch.tensor(w * 1e-9, dtype=torch.float64))) for w in DESIGN_NM},
        "material_note": "532 and 633 nm indices are literature anchors; 488 nm is a Cauchy extrapolation.",
        "optimization": {"iterations": args.iterations, "random_starts": args.starts,
                          "joint_warm_start": "best 532 nm design", "selected_joint_start": joint_candidates[0][3],
                          "selected_joint_objective": joint_candidates[0][0],
                          "joint_candidate_objectives": [{"start": row[3], "objective": row[0]} for row in joint_candidates],
                          "selected_single_532_objective": green_candidates[0][0],
                          "device": str(device)},
        "self_tests": checks,
        "roi_definition": "target intensity >= 8% of target peak",
        "efficiency_definition": "absolute scalar power integrated over target ROI divided by incident DOE pupil power; no output-window renormalization",
        "height_neighbors": {
            "joint_32level_p95_absolute_step_um": float(np.percentile(np.abs(np.concatenate([
                np.diff(physical_designs["joint_32level"], axis=0)[pupil[1:, :].astype(bool)],
                np.diff(physical_designs["joint_32level"], axis=1)[pupil[:, 1:].astype(bool)]
            ])), 95)),
            "joint_32level_max_absolute_step_um": float(np.max(np.abs(np.concatenate([
                np.diff(physical_designs["joint_32level"], axis=0)[pupil[1:, :].astype(bool)],
                np.diff(physical_designs["joint_32level"], axis=1)[pupil[:, 1:].astype(bool)]
            ]))))
        },
        "outputs": ["metrics.csv", "spectral_scan.csv", "designs.npz", "pattern_comparison.png",
                    "height_maps.png", "optimization_history.png", "baseline_vs_joint.png", "spectral_holdout.png"],
        "limitations": ["scalar thin-phase relief", "ideal achromatic thin lens", "Fraunhofer focal plane",
                        "finite detector window", "no polarization/reflection/absorption/full-wave response", "discrete scan is not a continuum proof"]
    }
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    make_figures(output, cfg, target, physical_designs,
                 pattern_data, metric_rows, history, scan_rows)
    print(f"Results written to {output}")
    for row in metric_rows:
        if row["design"] in {"flat", "532nm_only", "joint_32level", "joint_16level"}:
            print(f"{row['design']:>14s} {row['wavelength_nm']:6.1f} nm  "
                  f"L2={row['relative_l2_shape']:.4f}  Pearson={row['pearson_roi']:.4f}  "
                  f"eta_ROI={row['roi_efficiency_input_fraction']:.4f}  capture={row['detector_capture_input_fraction']:.4f}")


if __name__ == "__main__":
    main()
