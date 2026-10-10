"""Fixed-optics RGB background screening, weight scans, and full validation."""
from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass, replace
import hashlib
import itertools
import json
from pathlib import Path
import time
from typing import Any

import numpy as np
import torch
import matplotlib.pyplot as plt

from forward_models import (
    OpticalStopConfig, OpticalSystem, angular_spectrum_transfer_batch,
    centered_fft, forward_rgb_4f_stop, forward_rgb_lens_aperture,
    make_doe_field_rgb, make_grid, make_image_stop, propagate, validate_sampling,
)
from losses import LossWeights
from materials import CauchyMaterial
from optimizers import AdamOptions, optimize_adam
from rgb_decolorization_experiment import (
    ARCHITECTURES, INPUT_POWERS, TARGETS, WAVELENGTHS_M, WAVELENGTH_NM,
    aperture_on_base, evaluate_outputs, make_initial_height, make_schedule,
    make_target, make_weights as historical_weights, resize_height, resolve_device,
    save_run, save_scalar_image, write_json,
)

VARIANTS = ("full", "full_bg_absolute", "full_bg_peak", "full_bg_absolute_peak", "full_bg_constrained")
BRANCHES = ("none", "lens", "fourier_circular", "fourier_annular", "image")
REFERENCE_WAVELENGTH_M = 532e-9
INTERPRETATION = {
    "none": "DOE numerical background",
    "lens": "two explicit finite thin lenses in a 1:1 relay; angular-spectrum propagation between planes",
    "fourier_circular": "ideal 1:1 relay with optical low-pass filtering",
    "fourier_annular": "ideal 1:1 relay with optical DC rejection",
    "image": "detector spatial selection; cropping is not DOE background elimination",
}


@dataclass(frozen=True)
class OpticalConfig:
    branch: str = "none"
    focal_length_m: float = 0.05
    lens_radius_m: float | None = None
    stop: OpticalStopConfig = OpticalStopConfig()
    cutoff_fraction: float | None = None
    dc_fraction: float = 0.0
    image_margin: float | None = None
    relay_object: str = "baseline_image"
    reference_wavelength_m: float = REFERENCE_WAVELENGTH_M


def config_id(config: OpticalConfig) -> str:
    digest = hashlib.sha256(json.dumps(asdict(config), sort_keys=True).encode()).hexdigest()[:12]
    return f"{config.branch}_{digest}"


def make_weights(target: str, variant: str, absolute: float = 0.1,
                 peak: float = 0.05, floor: float = 0.1) -> LossWeights:
    if variant not in VARIANTS:
        raise ValueError(f"unknown variant {variant}")
    return replace(historical_weights(target, "full"),
                   rgb_background_absolute_weight=absolute if variant in (
                       "full_bg_absolute", "full_bg_absolute_peak", "full_bg_constrained") else 0.0,
                   rgb_background_peak_weight=peak if variant in (
                       "full_bg_peak", "full_bg_absolute_peak", "full_bg_constrained") else 0.0,
                   efficiency_floor_weight=floor if variant == "full_bg_constrained" else 0.0)


def spectrum_reference(target, grid, energy_fraction: float = 0.95) -> dict[str, float]:
    """Freeze radii from the target AMPLITUDE spectrum before optimization.

    The target spectrum radius contains 95% of amplitude-spectrum energy.
    The DC reference radius contains 10%; both definitions are saved so the
    otherwise ambiguous 'target/DC radius' can be reproduced.
    """
    amplitude = torch.sqrt(target.intensity)
    pad = (grid.work_n - grid.base_n) // 2
    amplitude = torch.nn.functional.pad(amplitude, (pad, pad, pad, pad))
    energy = centered_fft(amplitude).abs().square().flatten()
    radius = torch.fft.fftshift(torch.hypot(grid.fx, grid.fy)).flatten()
    indices = torch.argsort(radius)
    cumulative = energy[indices].cumsum(0) / energy.sum()

    def quantile(value):
        index = torch.searchsorted(cumulative, torch.tensor(value, device=cumulative.device)).clamp_max(len(indices) - 1)
        return float(radius[indices[index]].cpu())

    envelope = torch.hypot(grid.base_x, grid.base_y)[
        (target.signal_mask + target.transition_mask) > 0.05
    ]
    return {"target_radius_cpm": quantile(energy_fraction), "dc_radius_cpm": max(
        quantile(0.1), 1.0 / (grid.work_n * grid.pixel_pitch_m)),
        "spectrum_energy_fraction": energy_fraction, "dc_energy_fraction": 0.1,
        "target_envelope_radius_m": float(envelope.max().cpu())}


def optical_scan(reference: dict[str, float], relay_object: str) -> list[OpticalConfig]:
    configs = [OpticalConfig(relay_object=relay_object)]
    configs.extend(OpticalConfig(branch="lens", focal_length_m=f * 1e-3,
                                 lens_radius_m=r * 1e-3, relay_object=relay_object)
                   for f, r in itertools.product((50, 100), (0.5, 0.75, 1.0, 1.25)))
    for branch in ("fourier_circular", "fourier_annular"):
        for fraction, dc in itertools.product((0.6, 0.8, 1.0, 1.2),
                                              (0.0,) if branch == "fourier_circular" else (0.0, 0.05, 0.1, 0.2)):
            fc = fraction * reference["target_radius_cpm"]
            inner = dc * reference["dc_radius_cpm"]
            # One piece of hardware shared by RGB: freeze physical radii at
            # 532 nm, rather than building three different wavelength stops.
            stop = OpticalStopConfig(stop_type=branch, position="fourier",
                cutoff_frequency_cpm=fc, inner_cutoff_frequency_cpm=inner,
                radius_m=REFERENCE_WAVELENGTH_M * 0.05 * fc,
                inner_radius_m=REFERENCE_WAVELENGTH_M * 0.05 * inner,
                soft_edge=max(0.03 * fc, 100.0))
            configs.append(OpticalConfig(branch=branch, stop=stop,
                cutoff_fraction=fraction, dc_fraction=dc, relay_object=relay_object))
    for margin in (1.0, 1.25, 1.5, 2.0):
        configs.append(OpticalConfig(branch="image", image_margin=margin, relay_object=relay_object,
            stop=OpticalStopConfig(stop_type="image", position="image",
                radius_m=margin * reference["target_envelope_radius_m"], soft_edge=10e-6)))
    return configs


def sampling_checks(system, grid, config: OpticalConfig) -> dict[str, Any]:
    report = {"doe": validate_sampling(system, grid),
              "output_coordinates": "registered physical image plane, metres",
              "frequency_nyquist_cpm": 0.5 / grid.pixel_pitch_m,
              "frequency_step_cpm": 1.0 / (grid.work_n * grid.pixel_pitch_m),
              "valid": True, "reasons": []}
    f = config.focal_length_m
    pitches = [wave * f / (grid.work_n * grid.pixel_pitch_m) for wave in WAVELENGTHS_M]
    extents = [(grid.work_n // 2 - 1) * pitch for pitch in pitches]
    report["fourier_pitch_m_by_wavelength"] = pitches
    report["fourier_half_extent_m_by_wavelength"] = extents
    report["fourier_paraxial_max_sin_angle"] = max(WAVELENGTHS_M) / (2 * grid.pixel_pitch_m)
    report["relay_magnification"] = 1.0
    report["relay_length_m"] = 4 * f if config.branch.startswith("fourier") or config.branch == "lens" else None
    if config.branch.startswith("fourier"):
        outer = float(config.stop.radius_m)
        inner = config.stop.inner_radius_m
        cutoffs = [outer / (wave * f) for wave in WAVELENGTHS_M]
        report["effective_cutoff_cpm_by_wavelength"] = cutoffs
        report["physical_stop_radius_m"] = outer
        # Soft edges map to metres with the REFERENCE wavelength, independent
        # of the RGB channel. Include 4 edge widths in the Nyquist guard.
        tau_m = config.stop.soft_edge * config.reference_wavelength_m * f
        report["physical_soft_edge_m"] = tau_m
        report["inner_radius_pixels_by_wavelength"] = [inner / pitch for pitch in pitches]
        report["inner_edge_underresolved"] = inner > 0 and any(inner < pitch for pitch in pitches)
        if outer + 4 * tau_m >= min(extents):
            report["reasons"].append("Fourier stop including soft edge exceeds sampled Nyquist disk")
        if inner >= outer:
            report["reasons"].append("annular inner radius must be smaller than outer radius")
    if config.branch == "image":
        half_base = (grid.base_n // 2 - 1) * grid.pixel_pitch_m
        if float(config.stop.radius_m) + 4 * config.stop.soft_edge >= half_base:
            report["reasons"].append("detector stop extends beyond physical output field of view")
    if config.branch == "lens":
        half_work = (grid.work_n // 2 - 1) * grid.pixel_pitch_m
        if float(config.lens_radius_m) > half_work:
            report["reasons"].append("lens pupil extends beyond computational field")
        report["finite_pupil_model"] = "explicit finite thin lenses and angular-spectrum propagation"
        report["pupil_cutoff_cpm_by_wavelength"] = [config.lens_radius_m / (wave * f) for wave in WAVELENGTHS_M]
        report["lens_phase_frequency_cpm"] = config.lens_radius_m / (min(WAVELENGTHS_M) * f)
        report["fourier_pitch_m_by_wavelength"] = [grid.pixel_pitch_m] * 3
        report["fourier_half_extent_m_by_wavelength"] = [half_work] * 3
        report["pupil_exceeds_frequency_window"] = report["lens_phase_frequency_cpm"] >= report["frequency_nyquist_cpm"]
        if report["pupil_exceeds_frequency_window"]:
            report["reasons"].append("finite lens phase exceeds sampled Nyquist limit")
    report["valid"] = not report["reasons"]
    return report


def make_forward(config, architecture, grid, system, *, hard=False, diagnostics=False):
    material = CauchyMaterial()
    waves = torch.tensor(WAVELENGTHS_M, device=grid.base_x.device, dtype=torch.float64)
    indices = material.indices(WAVELENGTHS_M)
    between = angular_spectrum_transfer_batch(waves, grid.fx, grid.fy, 0.025)
    remaining = system.propagation_distance_m - (0.025 if architecture == "two_doe_planes" else 0.0)
    transfer = angular_spectrum_transfer_batch(waves, grid.fx, grid.fy, remaining)

    def forward(height):
        if config.branch.startswith("fourier"):
            # Translate the fixed physical edge to equivalent frequency width
            # separately per channel. No optical parameter is trainable here.
            from forward_models import make_fourier_stop
            masks = torch.stack([make_fourier_stop(grid, wave, config.focal_length_m,
                radius_m=config.stop.radius_m, inner_radius_m=config.stop.inner_radius_m,
                stop_type=config.branch, soft_edge_cpm=config.stop.soft_edge * config.reference_wavelength_m / wave,
                hard=hard) for wave in WAVELENGTHS_M]).to(torch.float32)
            output = forward_rgb_4f_stop(height, grid, waves, indices, config.focal_length_m,
                masks, plane_separation_m=0.025,
                object_distance_m=remaining if config.relay_object == "baseline_image" else 0.0,
                return_diagnostics=diagnostics)
        elif config.branch == "lens":
            output = forward_rgb_4f_stop(height, grid, waves, indices,
                config.focal_length_m, lens_aperture_radius_m=config.lens_radius_m,
                plane_separation_m=0.025, object_distance_m=remaining,
                return_diagnostics=diagnostics)
        else:
            field = make_doe_field_rgb(height[0] if height.ndim == 3 else height, grid, waves, indices)
            if height.ndim == 3:
                field = propagate(field, between) * make_doe_field_rgb(height[1], grid, waves, indices)
            work = propagate(field, transfer)
            before = work
            if config.branch == "image":
                work = work * make_image_stop(grid, config.stop.radius_m, config.stop.soft_edge, hard=hard).to(torch.float32)
            cropped = work[..., grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end]
            if diagnostics:
                pre = before.abs().square().sum((-2, -1))
                post = work.abs().square().sum((-2, -1))
                output = cropped, {"fourier_field": centered_fft(before),
                    "before_image_stop_field": before, "pre_stop_energy": pre,
                    "post_stop_energy": post, "stop_throughput": post / pre.clamp_min(1e-12)}
            else:
                output = cropped
        if diagnostics:
            return output
        return list(output.unbind(0))
    return forward


def background_metrics(outputs, target, grid) -> list[dict[str, Any]]:
    power = float(aperture_on_base(grid).sum())
    rows = []
    for wave, field in zip(WAVELENGTH_NM, outputs):
        intensity = field.detach().abs().square()
        weighted_bg = intensity * target.background_mask
        bg = intensity[target.background_mask > 0.5] / power
        roi = float((intensity * target.signal_mask).sum())
        total_bg = float(weighted_bg.sum())
        rows.append({"wavelength_nm": wave, "background_energy": total_bg,
            "background_absolute_fraction": total_bg / power,
            "background_roi_ratio": total_bg / max(roi, 1e-12),
            "background_max": float(bg.max()) if bg.numel() else 0.0,
            "background_p99": float(torch.quantile(bg, 0.99)) if bg.numel() else 0.0,
            "background_mean": float(bg.mean()) if bg.numel() else 0.0,
            "roi_absolute_efficiency": roi / power,
            "output_absolute_efficiency": float(intensity.sum()) / power,
            "peak_units": "fraction of incident power per output pixel"})
    return rows


def evaluate(outputs, target, grid):
    if not all(torch.isfinite(field).all() for field in outputs):
        raise FloatingPointError("nonfinite output field")
    evaluation = evaluate_outputs(outputs, target, grid, aperture_on_base(grid), INPUT_POWERS)
    bg = background_metrics(outputs, target, grid)
    for row, background in zip(evaluation["metrics_by_wavelength"], bg):
        row.update(background)
    evaluation["background_metrics"] = bg
    summary = evaluation["summary"]
    for key in ("background_roi_ratio", "background_absolute_fraction", "background_p99", "background_max", "background_mean"):
        summary[f"mean_{key}"] = float(np.mean([row[key] for row in bg]))
    summary["roi_efficiencies"] = [row["roi_absolute_efficiency"] for row in bg]
    summary["mean_roi_absolute_efficiency"] = float(np.mean(summary["roi_efficiencies"]))
    summary["max_hole_leakage"] = max((item["hole_leakage_fraction"] for row in evaluation["metrics_by_wavelength"]
                                      for item in row.get("ring_components", [])), default=0.0)
    summary["mean_center_to_ring_ratio"] = float(np.mean([
        row.get("mean_center_to_ring_ratio", 0.0) for row in evaluation["metrics_by_wavelength"]]))
    return evaluation


def acceptance_status(candidate: dict, baseline: dict, ring: bool = False) -> dict:
    checks = {
        "background": candidate["mean_background_roi_ratio"] <= baseline["mean_background_roi_ratio"] + 0.005,
        "target_quality": candidate["worst_correlation"] >= baseline["worst_correlation"] - 0.05,
        "efficiency": all(c >= 0.9 * b - 1e-12 for c, b in zip(candidate["roi_efficiencies"], baseline["roi_efficiencies"])),
        "centroid": candidate["max_component_centroid_drift_um"] <= baseline["max_component_centroid_drift_um"] + 1.0,
        "hole": not ring or candidate["max_hole_leakage"] <= 1.25 * baseline["max_hole_leakage"] + 1e-12,
    }
    return {"accepted": all(checks.values()), "checks": checks,
        "failed": [name for name, passed in checks.items() if not passed],
        "background_ratio_reduction": baseline["mean_background_roi_ratio"] - candidate["mean_background_roi_ratio"],
        "background_p99_reduction": baseline["mean_background_p99"] - candidate["mean_background_p99"]}


def rank_key(row):
    status = row["acceptance_status"]
    return (-status["background_ratio_reduction"], -status["background_p99_reduction"],
            -row["mean_roi_absolute_efficiency"], -row["mean_common_scale_roi_pearson"], row["rgb_roi_energy_cv"])


def pareto_front(rows):
    def objectives(row):
        return (row["mean_background_roi_ratio"], row["mean_background_p99"],
                -row["mean_roi_absolute_efficiency"], -row["worst_correlation"],
                row["max_component_centroid_drift_um"], row["max_hole_leakage"])
    return [row for row in rows if not any(
        all(a <= b for a, b in zip(objectives(other), objectives(row))) and
        any(a < b for a, b in zip(objectives(other), objectives(row)))
        for other in rows if other is not row)]


def aggregate_rows(rows):
    groups = {}
    for row in rows:
        groups.setdefault((row["optical_id"], row["variant"], row["weights_id"]), []).append(row)
    result = []
    for values in groups.values():
        item = {key: values[0][key] for key in ("optical_id", "variant", "weights_id", "optical_config", "loss_parameters")}
        item["seeds"] = sorted(row["seed"] for row in values)
        for key in ("mean_background_roi_ratio", "mean_background_p99", "mean_background_absolute_fraction",
                    "mean_roi_absolute_efficiency", "mean_common_scale_roi_pearson", "rgb_roi_energy_cv",
                    "worst_correlation", "max_component_centroid_drift_um", "max_hole_leakage"):
            numeric = np.asarray([row[key] for row in values])
            item[key] = float(numeric.mean())
            item[f"{key}_std"] = float(numeric.std())
        item["acceptance_status"] = {"accepted": all(row["acceptance_status"]["accepted"] for row in values),
            "background_ratio_reduction": float(np.mean([row["acceptance_status"]["background_ratio_reduction"] for row in values])),
            "background_p99_reduction": float(np.mean([row["acceptance_status"]["background_p99_reduction"] for row in values]))}
        result.append(item)
    return result


def select_candidates(rows, limit=3):
    """Pick distinct strongest suppression, RGB tradeoff, and shape candidates."""
    aggregates = aggregate_rows([row for row in rows if row["optical_config"]["branch"] != "none"])
    accepted = [row for row in aggregates if row["acceptance_status"]["accepted"]]
    pool = accepted or pareto_front(aggregates)
    selected = []
    selectors = [rank_key,
        lambda row: (-row["mean_common_scale_roi_pearson"], *rank_key(row)),
        lambda row: (-row["worst_correlation"], *rank_key(row))]
    reasons = ("background suppression", "background/RGB tradeoff", "target shape")
    for selector, reason in zip(selectors, reasons):
        remaining = [row for row in pool if row["optical_id"] not in {value["optical_id"] for value in selected}]
        if remaining and len(selected) < limit:
            item = min(remaining, key=selector).copy()
            item["selection_reason"] = reason if accepted else f"Pareto fallback: {reason}"
            selected.append(item)
    return selected


def optical_from_dict(value):
    value = dict(value)
    value["stop"] = OpticalStopConfig(**value["stop"])
    return OpticalConfig(**value)


def weights_scan():
    for absolute in (0.05, 0.1, 0.25, 0.5):
        yield "full_bg_absolute", (absolute, 0.0, 0.0)
    for peak in (0.02, 0.05, 0.1):
        yield "full_bg_peak", (0.0, peak, 0.0)
    for absolute, peak in itertools.product((0.05, 0.1, 0.25, 0.5), (0.02, 0.05, 0.1)):
        yield "full_bg_absolute_peak", (absolute, peak, 0.0)
        for floor in (0.05, 0.1, 0.25):
            yield "full_bg_constrained", (absolute, peak, floor)


def weight_scan_jobs(candidates, scan=None, limit=3):
    """Expand the prescribed weight sweep only over selected optical candidates."""
    scan = list(weights_scan()) if scan is None else list(scan)
    jobs = [(optical_from_dict(item["optical_config"]), variant, parameters)
            for item, (variant, parameters) in itertools.product(candidates[:limit], scan)]
    return list({(config_id(config), variant, tuple(parameters)): (config, variant, parameters)
                 for config, variant, parameters in jobs}.values())


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def run_one(args, stage, target_name, architecture, seed, config, variant, parameters, baseline=None):
    device = resolve_device(args.device)
    if stage == "validate":
        sizes = ((args.validation_coarse_grid, args.validation_coarse_pitch_um, args.validation_coarse_iterations),
                 (args.validation_fine_grid, args.validation_fine_pitch_um, args.validation_fine_iterations))
    else:
        sizes = ((args.screen_grid, args.screen_pitch_um, args.screen_coarse_iterations),
                 (args.screen_grid, args.screen_pitch_um, args.screen_fine_iterations))
    grids, systems, targets = [], [], []
    for size, pitch, _ in sizes:
        system = OpticalSystem(n_grid=size, pixel_pitch_m=pitch * 1e-6, pad_factor=args.pad_factor)
        grid = make_grid(system, device)
        check = sampling_checks(system, grid, config)
        if not check["valid"]:
            raise ValueError("; ".join(check["reasons"]))
        systems.append(system)
        grids.append(grid)
        targets.append(make_target(target_name, grid, device))
    weights = make_weights(target_name, variant, *parameters)
    weights_id = "_".join(f"{number:g}" for number in parameters)
    run_dir = args.output_dir / stage / target_name / architecture / config_id(config) / variant / weights_id / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    floors = None if baseline is None else tuple(0.9 * eta for eta in baseline["roi_efficiencies"])
    coarse_floors = None if baseline is None else tuple(0.9 * eta for eta in baseline["coarse_roi_efficiencies"])
    run_config = {"schema_version": 2, "stage": stage, "target": target_name, "architecture": architecture, "seed": seed,
        "optical_config": asdict(config), "variant": variant, "loss_parameters": parameters,
        "background_peak_definition": "power-normalized background Lp norm divided by power-normalized signal ROI energy",
        "loss_weights": asdict(weights), "wavelengths_m": WAVELENGTHS_M, "input_powers": INPUT_POWERS,
        "incident_power_convention": "unit-amplitude DOE aperture; P_lambda = illuminated aperture pixel count",
        "height_max_m": 1.4e-6, "grids": [asdict(system) for system in systems],
        "sampling": [sampling_checks(system, grid, config) for system, grid in zip(systems, grids)],
        "iterations": [item[2] for item in sizes], "schedules": [asdict(make_schedule(item[2])) for item in sizes],
        "efficiency_floor": floors, "coarse_efficiency_floor": coarse_floors,
        "plane_separation_m": 0.025, "total_baseline_distance_m": 0.05,
        "interpretation": INTERPRETATION[config.branch], "soft_train_hard_validate": True,
        "initialization": "identical uniform random heights by architecture and seed", "device": str(device)}
    run_config["spectrum_reference"] = spectrum_reference(targets[0], grids[0])
    run_config["baseline_reference"] = None if baseline is None else baseline["run_dir"]
    run_config["acceptance_thresholds"] = {"background_ratio_tolerance": 0.005,
        "correlation_drop": 0.05, "per_channel_efficiency_ratio": 0.9,
        "centroid_increase_um": 1.0, "hole_leakage_ratio": 1.25}
    run_config["optical_positions_m"] = {
        "relay_object_after_last_doe": 0.0 if config.relay_object == "doe" and config.branch.startswith("fourier") else
            (0.025 if architecture == "two_doe_planes" else 0.05),
        "lens1_from_relay_object": config.focal_length_m,
        "fourier_stop_from_relay_object": 2 * config.focal_length_m,
        "lens2_from_relay_object": 3 * config.focal_length_m,
        "output_from_relay_object": 4 * config.focal_length_m,
    } if config.branch.startswith("fourier") or config.branch == "lens" else {"detector_after_first_doe": 0.05}
    existing = run_dir / "result.json"
    if args.resume and existing.exists():
        saved = json.loads(existing.read_text(encoding="utf-8"))
        if saved.get("config") != json.loads(json.dumps(run_config)):
            raise ValueError(f"resume config mismatch: {existing}; use a new output directory")
        return saved["row"]
    write_json(run_dir / "config.json", run_config)
    print(f"[start] {stage}/{target_name}/{architecture}/{config_id(config)}/{variant}/seed_{seed}", flush=True)
    start = time.perf_counter()
    initial = make_initial_height(architecture, grids[0], 1.4e-6, seed, device)
    results = []
    for index, (grid, system, target, (_, _, iterations)) in enumerate(zip(grids, systems, targets, sizes)):
        if index:
            initial = resize_height(results[-1].height_m, grid, aperture_on_base(grid), 1.4e-6)
        power = float(aperture_on_base(grid).sum())
        options = AdamOptions(iterations=iterations, seed=seed, input_powers=(power,) * 3,
            wavelength_weights=INPUT_POWERS, pixel_pitch_m=grid.pixel_pitch_m,
            efficiency_floor=coarse_floors if index == 0 else floors, log_every=max(1, iterations // 20))
        results.append(optimize_adam(make_forward(config, architecture, grid, system), target,
            1.4e-6, grid.base_n, aperture_on_base(grid), options, weights, make_schedule(iterations), initial))
    grid, system, target, result = grids[-1], systems[-1], targets[-1], results[-1]
    evaluations, diagnostics = {}, {}
    for label, hard in (("soft", False), ("hard", True)):
        with torch.no_grad():
            batch, diagnostic = make_forward(config, architecture, grid, system, hard=hard, diagnostics=True)(result.height_m)
        outputs = list(batch.unbind(0))
        evaluation = evaluate(outputs, target, grid)
        if baseline is None:
            status = {"accepted": True, "reference": "unfiltered full baseline", "checks": {}, "failed": [],
                      "background_ratio_reduction": 0.0, "background_p99_reduction": 0.0}
        else:
            status = acceptance_status(evaluation["summary"], baseline, target_name == "ring50")
        evaluation["acceptance_status"] = status
        evaluations[label] = {key: evaluation[key] for key in ("summary", "metrics_by_wavelength", "pairwise", "background_metrics", "acceptance_status")}
        energy_diagnostic = {key: value.detach().cpu().tolist() for key, value in diagnostic.items()
                             if "field" not in key and "mask" not in key}
        diagnostics[label] = energy_diagnostic
        save_run(run_dir / label, target, grid, aperture_on_base(grid), outputs, result, evaluation, run_config)
        write_json(run_dir / label / "metrics.json", {**evaluations[label], "stop_throughput": energy_diagnostic})
        write_json(run_dir / label / "result.json", {"config": run_config, "optical_config": asdict(config),
            "background_metrics": evaluation["background_metrics"], "stop_throughput": energy_diagnostic,
            "acceptance_status": status, "metrics": evaluation["summary"]})
        if "fourier_stop_mask" in diagnostic:
            np.save(run_dir / label / "fourier_stop_mask.npy", diagnostic["fourier_stop_mask"].cpu().numpy())
        if "before_image_stop_field" in diagnostic:
            before = diagnostic["before_image_stop_field"][:, grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end]
            write_json(run_dir / label / "before_detector_metrics.json", {
                "background_metrics": background_metrics(list(before.unbind(0)), target, grid)})
            np.save(run_dir / label / "before_detector_intensity.npy", before.abs().square().cpu().numpy())
        if "fourier_field" in diagnostic:
            focal = diagnostic["fourier_field"]
            pitches = run_config["sampling"][-1]["fourier_pitch_m_by_wavelength"]
            for wave, field, pitch in zip(WAVELENGTH_NM, focal, pitches):
                array = field.abs().square().cpu().numpy()
                np.save(run_dir / label / f"fourier_{wave}nm.npy", array)
                extent = [-grid.work_n // 2 * pitch * 1e3, (grid.work_n // 2 - 1) * pitch * 1e3] * 2
                save_scalar_image(run_dir / label / f"fourier_{wave}nm.png", array, extent,
                    f"{wave} nm focal spectrum" if config.branch in ("lens", "fourier_circular", "fourier_annular")
                    else f"{wave} nm diagnostic spectrum (virtual plane)")
    hard_summary = evaluations["hard"]["summary"]
    with torch.no_grad():
        unfiltered = make_forward(OpticalConfig(), architecture, grid, system)(result.height_m)
        coarse_outputs = make_forward(config, architecture, grids[0], systems[0], hard=True)(results[0].height_m)
    coarse_efficiencies = evaluate(coarse_outputs, targets[0], grids[0])["summary"]["roi_efficiencies"]
    write_json(run_dir / "unfiltered_same_height_metrics.json", {
        "summary": evaluate(unfiltered, target, grid)["summary"],
        "background_metrics": background_metrics(unfiltered, target, grid),
        "interpretation": "DOE numerical background at baseline image plane for the same optimized heights"})
    row = {"stage": stage, "target": target_name, "architecture": architecture, "seed": seed,
           "optical_id": config_id(config), "optical_config": asdict(config), "variant": variant,
           "weights_id": weights_id, "loss_parameters": parameters, "run_dir": str(run_dir.relative_to(args.output_dir)),
           **hard_summary, "coarse_roi_efficiencies": coarse_efficiencies,
           "acceptance_status": evaluations["hard"]["acceptance_status"]}
    write_json(existing, {"config": run_config, "optical_config": asdict(config), "row": row,
        "evaluations": evaluations, "stop_throughput": diagnostics,
        "background_metrics": evaluations["hard"]["background_metrics"], "acceptance_status": row["acceptance_status"],
        "runtime_seconds": time.perf_counter() - start})
    write_json(run_dir / "coarse_loss_history.json", {"losses": results[0].losses, "components": results[0].components})
    print(f"[done] accepted={row['acceptance_status']['accepted']} bg/ROI={row['mean_background_roi_ratio']:.5g}", flush=True)
    return row


def stage_report(output_dir, stage, rows, rejected):
    root = output_dir / stage
    root.mkdir(parents=True, exist_ok=True)
    units = {}
    for row in rows:
        units.setdefault((row["target"], row["architecture"]), []).append(row)
    candidates, aggregates, pareto = [], [], []
    for (target, architecture), values in units.items():
        public = [row for row in values if not (row["variant"] == "full" and row["optical_config"]["branch"] == "none")]
        unit_aggregates = aggregate_rows(public)
        for item in unit_aggregates:
            item.update(target=target, architecture=architecture)
        aggregates.extend(unit_aggregates)
        selected = select_candidates(public)
        candidates.extend({**item, "target": target, "architecture": architecture} for item in selected)
        pareto.extend(pareto_front(unit_aggregates))
    report = {"stage": stage, "runs": rows, "aggregates": aggregates, "candidates": candidates,
              "pareto_front": pareto, "rejected_sampling": rejected,
              "accepted_optical_filter_configs": [item for item in aggregates if item["optical_config"]["branch"]
                in ("lens", "fourier_circular", "fourier_annular") and item["acceptance_status"]["accepted"]],
              "accepted_detector_selection_configs": [item for item in aggregates if item["optical_config"]["branch"]
                == "image" and item["acceptance_status"]["accepted"]]}
    write_json(root / "summary.json", report)
    write_json(root / "candidates.json", {"stage": stage, "candidates": candidates})
    write_csv(root / "summary.csv", rows)
    write_csv(root / "aggregate.csv", aggregates)
    figure, axis = plt.subplots(figsize=(7.5, 5), constrained_layout=True)
    for branch in BRANCHES:
        values = [row for row in aggregates if row["optical_config"]["branch"] == branch]
        if values:
            axis.scatter([row["mean_background_roi_ratio"] for row in values],
                         [row["mean_roi_absolute_efficiency"] for row in values], label=branch)
    axis.set(xlabel="Mean background / ROI energy", ylabel="Absolute ROI efficiency", title=f"{stage}: background / efficiency tradeoff")
    axis.grid(alpha=0.25)
    if aggregates:
        axis.legend(fontsize=8)
    figure.savefig(root / "background_efficiency_pareto.png", dpi=160)
    plt.close(figure)
    text = [f"# RGB background experiment: {stage}", "", f"Completed runs: {len(rows)}. Sampling rejections: {len(rejected)}.", "",
            "Acceptance is checked per seed and per RGB channel against the same target/architecture/seed unfiltered full baseline.",
            "A configuration is accepted across seeds only when every seed passes.", "",
            "| Target | Architecture | Optics | Variant | Accepted | Background/ROI | ROI efficiency | Worst correlation |", "|---|---|---|---|---|---:|---:|---:|"]
    for item in aggregates:
        text.append(f"| {item['target']} | {item['architecture']} | {item['optical_config']['branch']} | {item['variant']} | "
                    f"{item['acceptance_status']['accepted']} | {item['mean_background_roi_ratio']:.6g} | "
                    f"{item['mean_roi_absolute_efficiency']:.6g} | {item['worst_correlation']:.4f} |")
    text.extend(["", "If no configuration passes all constraints, the candidates are explicitly marked Pareto fallbacks.",
                 "Background reduction on that frontier can involve losses in efficiency or target quality.", "",
                 "Image stops are detector spatial selection. Report before-detector metrics separately.",
                 "Finite lens pupils use explicit thin lens phases and apertures with angular-spectrum propagation.",
                 "Ideal relay modeling excludes lens scattering, reflections, fabrication error, absorption, polarization and detector dark current.",
                 "Fourier stop masks include coherent edge diffraction through the second focal transform."])
    (root / "report.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return report


def run_stage(args, stage, source=None):
    rows, rejected = [], []
    if source is None and stage != "screen":
        path = args.output_dir / ("screen" if stage == "weights" else "weights") / "summary.json"
        source = json.loads(path.read_text(encoding="utf-8"))
    for target_name, architecture in itertools.product(args.targets, args.architectures):
        system = OpticalSystem(n_grid=args.screen_grid, pixel_pitch_m=args.screen_pitch_um * 1e-6, pad_factor=args.pad_factor)
        grid = make_grid(system, resolve_device(args.device))
        reference = spectrum_reference(make_target(target_name, grid, grid.base_x.device), grid)
        default_parameters = (0.1, 0.05, 0.1)
        if stage == "screen":
            configs = [config for config in optical_scan(reference, args.relay_object) if config.branch in args.branches]
            if args.smoke:
                configs = [next(config for config in configs if config.branch == branch and
                                (branch != "fourier_annular" or config.dc_fraction > 0)) for branch in args.branches]
            jobs = [(config, "full_bg_absolute_peak", default_parameters) for config in configs]
        elif stage == "weights":
            selected = [item for item in source["candidates"] if item["target"] == target_name and item["architecture"] == architecture]
            scans = list(weights_scan()) if not args.smoke else [("full_bg_constrained", default_parameters)]
            jobs = weight_scan_jobs(selected, scans, limit=3)
        else:
            unit = [item for item in source["aggregates"] if item["target"] == target_name and item["architecture"] == architecture]
            # Keep the best representative of EACH requested optical branch,
            # even when the three cross-branch candidates omit a branch.
            representatives = [min([item for item in unit if item["optical_config"]["branch"] == branch],
                                   key=lambda item: (not item["acceptance_status"]["accepted"], *rank_key(item)))
                               for branch in args.branches if any(item["optical_config"]["branch"] == branch for item in unit)]
            selected = [item for item in source["candidates"] if item["target"] == target_name and item["architecture"] == architecture]
            combined = { (item["optical_id"], item["variant"], item["weights_id"]): item for item in representatives + selected }
            jobs = [(optical_from_dict(item["optical_config"]), item["variant"], tuple(item["loss_parameters"])) for item in combined.values()]
        seeds = args.validation_seeds if stage == "validate" else args.screen_seeds
        jobs = list({(config_id(config), variant, tuple(parameters)): (config, variant, parameters)
                     for config, variant, parameters in jobs}.values())
        for seed in seeds:
            baseline = run_one(args, stage, target_name, architecture, seed, OpticalConfig(relay_object=args.relay_object), "full", (0.0, 0.0, 0.0))
            rows.append(baseline)
            for config, variant, parameters in jobs:
                try:
                    rows.append(run_one(args, stage, target_name, architecture, seed, config, variant, parameters, baseline))
                except ValueError as error:
                    # Reject only physical sampling failures; implementation
                    # errors must fail loudly rather than becoming scan data.
                    if not any(word in str(error) for word in ("sampled Nyquist", "field of view", "computational field", "native Fourier field")):
                        raise
                    rejected.append({"target": target_name, "architecture": architecture, "seed": seed,
                                     "optical_config": asdict(config), "reason": str(error)})
                    print(f"[sampling rejection] {config_id(config)}: {error}", flush=True)
            # Incremental report permits review/resume during long scans.
            stage_report(args.output_dir, stage, rows, rejected)
    return stage_report(args.output_dir, stage, rows, rejected)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("screen", "weights", "validate", "all"), default="screen")
    parser.add_argument("--output-dir", type=Path, default=Path("results/rgb_background_filter"))
    parser.add_argument("--targets", nargs="+", choices=TARGETS, default=list(TARGETS))
    parser.add_argument("--architectures", nargs="+", choices=ARCHITECTURES, default=list(ARCHITECTURES))
    parser.add_argument("--branches", nargs="+", choices=BRANCHES, default=list(BRANCHES))
    parser.add_argument("--relay-object", choices=("baseline_image", "doe"), default="baseline_image")
    parser.add_argument("--screen-grid", type=int, default=128)
    parser.add_argument("--screen-pitch-um", type=float, default=20.0)
    parser.add_argument("--screen-coarse-iterations", type=int, default=40)
    parser.add_argument("--screen-fine-iterations", type=int, default=80)
    parser.add_argument("--screen-seeds", nargs="+", type=int, default=[7, 17])
    parser.add_argument("--validation-coarse-grid", type=int, default=256)
    parser.add_argument("--validation-coarse-pitch-um", type=float, default=10.0)
    parser.add_argument("--validation-fine-grid", type=int, default=512)
    parser.add_argument("--validation-fine-pitch-um", type=float, default=5.0)
    parser.add_argument("--validation-coarse-iterations", type=int, default=120)
    parser.add_argument("--validation-fine-iterations", type=int, default=400)
    parser.add_argument("--validation-seeds", nargs="+", type=int, default=[7, 17, 27])
    parser.add_argument("--pad-factor", choices=(1, 2, 4), type=int, default=2)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="one optical configuration per branch and a reduced weight scan")
    return parser.parse_args(argv)


def run(argv=None):
    args = parse_args(argv)
    if args.threads < 1 or any(getattr(args, name) < 1 for name in (
        "screen_coarse_iterations", "screen_fine_iterations", "validation_coarse_iterations", "validation_fine_iterations")):
        raise ValueError("threads and iteration counts must be positive")
    if any(seed < 0 for seed in args.screen_seeds + args.validation_seeds):
        raise ValueError("seeds must be nonnegative")
    torch.set_num_threads(args.threads)
    args.output_dir = args.output_dir.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if args.dry_run:
        manifest = []
        system = OpticalSystem(n_grid=args.screen_grid, pixel_pitch_m=args.screen_pitch_um * 1e-6, pad_factor=args.pad_factor)
        grid = make_grid(system, resolve_device(args.device))
        for target_name in args.targets:
            reference = spectrum_reference(make_target(target_name, grid, grid.base_x.device), grid)
            configs = [config for config in optical_scan(reference, args.relay_object) if config.branch in args.branches]
            manifest.append({"target": target_name, "spectrum_reference": reference,
                "configs": [{"optical_id": config_id(config), "optical_config": asdict(config),
                             "sampling": sampling_checks(system, grid, config)} for config in configs]})
        report = {"arguments": {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
                  "targets": manifest, "weight_combinations_per_optical_candidate": len(list(weights_scan()))}
        write_json(args.output_dir / "scan_manifest.json", report)
        print(f"Scan manifest: {args.output_dir / 'scan_manifest.json'}", flush=True)
        return report
    source = None
    for stage in (("screen", "weights", "validate") if args.stage == "all" else (args.stage,)):
        source = run_stage(args, stage, source)
    return source


if __name__ == "__main__":
    run()
