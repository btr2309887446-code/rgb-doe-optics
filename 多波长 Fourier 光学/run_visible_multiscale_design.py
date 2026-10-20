"""Run the two-vs-three-order visible-band free-screen experiment.

This deliberately remains a material-free scalar Fourier-lens experiment.
It uses the Step 11 fixed pupil and lens, a common physical detector grid,
robust spectral training, dense wavelength validation, and optional adaptive
training-point insertion for the multiscale chart.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from fourier_optics_step11_free_screen_optimization import (  # noqa: E402
    Model,
    Problem,
    load_case,
    objective_from_metrics,
    save_json,
    select_adaptive_wavelengths,
    solve,
    validation_summary,
    verify_math,
)
from make_multiscale_resolution_target import (  # noqa: E402
    DEFAULT_PITCH_M,
    DEFAULT_SUPERSAMPLE,
    make_multiscale_target,
    make_region_masks,
)


def training_wavelengths() -> np.ndarray:
    coarse = np.arange(400.0, 701.0, 10.0)
    return np.unique(np.r_[coarse, [488.0, 532.0, 633.0]])


def initial_degree_two_starts(count: int, starts: int, rng: np.random.Generator):
    values = []
    for _ in range(starts):
        coeff = np.zeros((3, count), dtype=np.complex128)
        coeff[0] = np.exp(1j*rng.uniform(-np.pi, np.pi, count))
        coeff[1:] = 0.15*(rng.normal(size=(2, count))
                          + 1j*rng.normal(size=(2, count))) / np.sqrt(2)
        values.append(Model.pack(coeff))
    return values


def padded_degree_three_starts(model_two: Model, candidates: list[np.ndarray]):
    values = []
    for theta in candidates:
        coeff_two = model_two.unpack(theta)
        coeff_three = np.zeros((4, coeff_two.shape[1]), dtype=np.complex128)
        coeff_three[:3] = coeff_two
        values.append(Model.pack(coeff_three))
    return values


def scan_model(model: Model, theta: np.ndarray, waves: np.ndarray,
               image_path: Path | None = None, batch_size: int = 12):
    rows = []
    image_map = None
    if image_path is not None:
        image_map = np.lib.format.open_memmap(
            image_path, mode="w+", dtype=np.float32,
            shape=(len(waves), model.p.m, model.p.m))
    for start in range(0, len(waves), batch_size):
        stop = min(start+batch_size, len(waves))
        batch_rows, images, _ = model.evaluate(theta, waves[start:stop],
                                               save_arrays=image_map is not None)
        for row in batch_rows:
            if not all(np.isfinite(float(value)) for value in row.values()):
                raise FloatingPointError("NaN/Inf in wavelength-scan output metrics")
            if row["captured_power"] > 1.005:
                raise AssertionError("Output-plane quadrature exceeds incident screen power")
        rows.extend(batch_rows)
        if image_map is not None:
            image_map[start:stop] = images.astype(np.float32)
    if image_map is not None:
        image_map.flush()
        del image_map
    return rows


def summarize_regions(rows, region_metadata):
    result = {}
    names = [region["name"] for region in region_metadata.get("regions", [])]
    for name in names:
        key = f"correlation_{name}"
        values = np.asarray([row[key] for row in rows if key in row], dtype=float)
        if values.size:
            result[name] = {"mean": float(values.mean()),
                            "minimum": float(values.min())}
    by_width = {}
    by_orientation = {}
    for name, values in result.items():
        region = next(item for item in region_metadata["regions"]
                      if item["name"] == name)
        width_key = f"{int(region['line_width_um'])}um"
        orient_key = f"{int(region['orientation_deg'])}deg"
        by_width.setdefault(width_key, []).append(values)
        by_orientation.setdefault(orient_key, []).append(values)
    return {
        "per_cell": result,
        "by_line_width": {
            key: {"mean": float(np.mean([x["mean"] for x in values])),
                  "minimum": float(min(x["minimum"] for x in values))}
            for key, values in by_width.items()},
        "by_orientation": {
            key: {"mean": float(np.mean([x["mean"] for x in values])),
                  "minimum": float(min(x["minimum"] for x in values))}
            for key, values in by_orientation.items()},
    }


def save_metrics_csv(path: Path, sections):
    all_rows = []
    for model_name, split, rows in sections:
        all_rows.extend({"model": model_name, "split": split, **row} for row in rows)
    fieldnames = ["model", "split"]
    for row in all_rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)


def checkpoint_callback(output_dir: Path, label: str, model: Model,
                        training_nm: np.ndarray, band_nm: np.ndarray):
    checkpoint_dir = output_dir/"checkpoints"/label
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    def save(restart: int, iteration: int, theta: np.ndarray, best_loss: float):
        path = checkpoint_dir/f"restart_{restart:02d}.npz"
        temporary = path.with_suffix(".tmp")
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream, theta=theta, coefficients=model.unpack(theta),
                training_nm=training_nm, band_nm=band_nm,
                restart=np.asarray(restart), iteration=np.asarray(iteration),
                best_loss=np.asarray(best_loss),
            )
        os.replace(temporary, path)
        save_json(checkpoint_dir/"progress.json", {
            "restart": restart, "iteration": iteration,
            "best_loss": best_loss, "checkpoint": path.name,
        })
    return save


def completed_checkpoint_candidates(output_dir: Path, label: str, model: Model,
                                    starts: int, iterations: int):
    """Load a complete restart set after an interrupted post-optimization step."""
    folder = output_dir/"checkpoints"/label
    paths = [folder/f"restart_{index:02d}.npz" for index in range(starts)]
    if not any(path.exists() for path in paths):
        return None
    if not all(path.is_file() for path in paths):
        raise ValueError(f"Incomplete checkpoint set for {label}; no checkpoints were overwritten")
    candidates, losses, runs = [], [], []
    for index, path in enumerate(paths):
        with np.load(path, allow_pickle=False) as checkpoint:
            if int(checkpoint["iteration"]) < iterations:
                raise ValueError(
                    f"Checkpoint {path} stopped at {int(checkpoint['iteration'])} of "
                    f"{iterations} iterations; automatic partial-optimizer resume is not enabled")
            if (not np.array_equal(checkpoint["training_nm"], model.p.waves)
                or not np.array_equal(checkpoint["band_nm"], model.p.band)):
                raise ValueError(f"Checkpoint configuration mismatch: {path}")
            theta = checkpoint["theta"].copy()
            value = model.value_gradient(theta)[0]
            if not np.isclose(value, float(checkpoint["best_loss"]),
                              rtol=1e-8, atol=1e-10):
                raise ValueError(f"Checkpoint objective does not reproduce: {path}")
            candidates.append(theta)
            losses.append(float(value))
            runs.append({
                "restart": index, "best_loss": float(value),
                "success": False, "message": "Loaded completed atomic checkpoint",
                "iterations": int(checkpoint["iteration"]), "evaluations": 0,
                "seconds": 0.0, "history": [], "resumed_checkpoint": str(path),
            })
    best = int(np.argmin(losses))
    return candidates[best], losses[best], runs, candidates


def make_model_figures(model_dir: Path, model_name: str, target: np.ndarray,
                       detector_axis_m: np.ndarray, scan_nm: np.ndarray,
                       rows: list[dict], image_file: Path, region_metadata: dict | None):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    pitch_m = float(np.median(np.diff(detector_axis_m)))
    extent = [(detector_axis_m[0]-pitch_m/2)*1e3,
              (detector_axis_m[-1]+pitch_m/2)*1e3]*2
    image_map = np.load(image_file, mmap_mode="r", allow_pickle=False)
    choose_nm = [400.0, 488.0, 532.0, 633.0, 700.0]
    in_band = [row for row in rows if 400 <= row["wavelength_nm"] <= 700]
    worst = max(in_band, key=lambda row: row["relative_L2"])
    choose_nm.append(float(worst["wavelength_nm"]))
    indices = []
    for wavelength in choose_nm:
        indices.append(int(np.argmin(np.abs(scan_nm-wavelength))))
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), squeeze=False)
    panels = [(target, "Target")] + [
        (image_map[index], f"{scan_nm[index]:g} nm, L2={rows[index]['relative_L2']:.3f}")
        for index in indices
    ]
    vmax = float(target.max())
    for ax, (image, title) in zip(axes.ravel(), panels):
        ax.imshow(image/vmax, origin="lower", extent=extent, cmap="magma",
                  vmin=0, vmax=1)
        ax.set(title=title, xlabel="X (mm)", ylabel="Y (mm)")
        ax.set_aspect("equal")
    fig.suptitle(f"{model_name}: common target scale")
    fig.tight_layout()
    fig.savefig(model_dir/"validation_patterns.png", dpi=160)
    plt.close(fig)
    del image_map

    wavelengths = np.asarray([row["wavelength_nm"] for row in rows])
    errors = np.asarray([row["relative_L2"] for row in rows])
    roi = np.asarray([row["roi_power"] for row in rows])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    axes[0].plot(wavelengths, errors, color="#1769aa")
    axes[0].axhline(0.10, color="#c62828", ls="--", label="L2 limit 0.10")
    axes[0].axvspan(400, 700, color="#90caf9", alpha=.18,
                    label="design band")
    axes[1].plot(wavelengths, roi, color="#2e7d32")
    axes[1].axhline(0.80, color="#c62828", ls="--", label="ROI floor 0.80")
    axes[1].axvspan(400, 700, color="#90caf9", alpha=.18,
                    label="design band")
    axes[0].set(xlabel="Wavelength (nm)", ylabel="Relative L2",
                title="Fixed-power intensity error")
    axes[1].set(xlabel="Wavelength (nm)", ylabel="ROI / incident screen power",
                title="ROI power fraction")
    for ax in axes:
        ax.grid(alpha=.25)
        ax.legend()
    fig.tight_layout()
    fig.savefig(model_dir/"spectral_validation.png", dpi=160)
    plt.close(fig)


def run_case(case_name: str, pupil: np.ndarray, target: np.ndarray,
             initial_waves: np.ndarray, scan_nm: np.ndarray, band: np.ndarray,
             grid_nm: float, region_metadata: dict | None,
             screen_axis_m: np.ndarray, detector_axis_m: np.ndarray,
             output_dir: Path,
             iterations: int, starts: int, seed: int,
             adaptive: bool, adaptive_rounds: int, adaptive_max_additions: int,
             roi_floor: float, roi_weight: float, cvar_weight: float,
              cvar_tail_fraction: float, resume: bool = False):
    output_dir.mkdir(parents=True, exist_ok=resume)
    region_masks = (make_region_masks(region_metadata, detector_axis_m)
                    if region_metadata else {})
    rng = np.random.default_rng(seed)
    train_nm = initial_waves.copy()
    common_config = {
        "loss_profile": "robust_cvar",
        "loss_definition": f"{1-cvar_weight:g}*mean(relative_L2^2) + {cvar_weight:g}*mean(worst_{100*cvar_tail_fraction:g}_percent(relative_L2^2)) + {roi_weight:g}*mean(max(0, {roi_floor:g}-ROI_power)^2)",
        "roi_definition": "target >= 1% of target peak intensity",
        "roi_floor": roi_floor,
        "roi_weight": roi_weight,
        "cvar_weight": cvar_weight,
        "cvar_tail_fraction": cvar_tail_fraction,
        "train_initial_nm": initial_waves.tolist(),
        "design_validation_nm": [400.0, 700.0, 1.0],
        "pressure_test_nm": [380.0, 780.0, 1.0],
        "starts_per_order": starts,
        "iterations_per_start": iterations,
        "adaptive_rounds_max": adaptive_rounds if adaptive else 0,
        "adaptive_additions_per_round_max": adaptive_max_additions,
        "optimizer": "SciPy L-BFGS-B, scalar point-sampled Fourier-lens model",
    }
    save_json(output_dir/"config.json", common_config)
    records = []
    sections = []

    p_two = Problem(pupil, target, train_nm, band, grid_nm,
                    loss_profile="robust_cvar", roi_floor=roi_floor,
                    roi_weight=roi_weight, cvar_weight=cvar_weight,
                    cvar_tail_fraction=cvar_tail_fraction,
                    region_masks=region_masks)
    model_two = Model(p_two, "spectral", 2)
    starts_two = initial_degree_two_starts(p_two.count, starts, rng)
    resumed_two = completed_checkpoint_candidates(
        output_dir, "spectral_degree_2", model_two, starts, iterations) if resume else None
    if resumed_two is not None:
        theta_two, loss_two, runs_two, candidates_two = resumed_two
        print(f"{case_name}/spectral_degree_2: resumed {len(candidates_two)} completed checkpoints",
              flush=True)
    else:
        theta_two, loss_two, runs_two, candidates_two = solve(
            model_two, starts_two, iterations, f"{case_name}/spectral_degree_2",
            return_candidates=True,
            on_progress=checkpoint_callback(output_dir, "spectral_degree_2",
                                            model_two, train_nm, band))
    training_two, _, _ = model_two.evaluate(theta_two, train_nm)
    reconstructed_two = objective_from_metrics(
        training_two, "robust_cvar", roi_floor, roi_weight,
        cvar_weight, cvar_tail_fraction)
    if not np.isclose(loss_two, reconstructed_two, rtol=1e-9, atol=1e-12):
        raise AssertionError("Degree-2 saved training metrics do not reproduce its objective")
    rec_two = {"name": "spectral_degree_2", "degree": 2,
               "parameters": int(theta_two.size), "loss": loss_two,
               "basis_rank": int(np.linalg.matrix_rank(model_two.basis)),
               "training_nm": train_nm.tolist(), "runs": runs_two,
               "training": training_two}
    model_two_dir = output_dir/"spectral_degree_2"
    model_two_dir.mkdir(exist_ok=resume)
    rec_two["coefficients"] = model_two.unpack(theta_two).copy()
    records.append(rec_two)

    p_three = Problem(pupil, target, train_nm, band, grid_nm,
                      loss_profile="robust_cvar", roi_floor=roi_floor,
                      roi_weight=roi_weight, cvar_weight=cvar_weight,
                      cvar_tail_fraction=cvar_tail_fraction,
                      region_masks=region_masks)
    model_three = Model(p_three, "spectral", 3)
    starts_three = padded_degree_three_starts(model_two, candidates_two)
    resumed_three = completed_checkpoint_candidates(
        output_dir, "spectral_degree_3", model_three, starts, iterations) if resume else None
    if resumed_three is not None:
        theta_three, loss_three, runs_three, candidates_three = resumed_three
        print(f"{case_name}/spectral_degree_3: resumed {len(candidates_three)} completed checkpoints",
              flush=True)
    else:
        theta_three, loss_three, runs_three, candidates_three = solve(
            model_three, starts_three, iterations,
            f"{case_name}/spectral_degree_3", return_candidates=True,
            on_progress=checkpoint_callback(output_dir, "spectral_degree_3",
                                            model_three, train_nm, band))
    training_three, _, _ = model_three.evaluate(theta_three, train_nm)
    reconstructed_three = objective_from_metrics(
        training_three, "robust_cvar", roi_floor, roi_weight,
        cvar_weight, cvar_tail_fraction)
    if not np.isclose(loss_three, reconstructed_three, rtol=1e-9, atol=1e-12):
        raise AssertionError("Degree-3 saved training metrics do not reproduce its objective")
    if loss_three > loss_two + 1e-10*max(1.0, loss_two):
        raise AssertionError("Nested cubic model did not preserve the quadratic solution")
    rec_three = {"name": "spectral_degree_3", "degree": 3,
                 "parameters": int(theta_three.size), "loss": loss_three,
                 "basis_rank": int(np.linalg.matrix_rank(model_three.basis)),
                 "training_nm": train_nm.tolist(), "runs": runs_three,
                 "training": training_three,
                 "starts_zero_extended_from_degree_2": True,
                 "initial_order3_coefficients": "zero for every start",
                 "adaptation_history": []}
    rec_three["coefficients"] = model_three.unpack(theta_three).copy()
    records.append(rec_three)
    sections += [("spectral_degree_2", "training_initial", training_two),
                 ("spectral_degree_3", "training_initial", training_three)]

    # Initial dense validation first establishes whether the cubic design needs
    # extra spectral samples. The same 1 nm samples are used for both bands.
    rows_three = scan_model(model_three, theta_three, scan_nm)
    design_summary_three = validation_summary(rows_three, [400.0, 700.0],
                                              0.10, roi_floor)
    adaptive_log = []
    for round_index in range(1, (adaptive_rounds if adaptive else 0)+1):
        if design_summary_three["passed"]:
            break
        additions = select_adaptive_wavelengths(
            rows_three, train_nm, [400.0, 700.0], l2_limit=0.10,
            roi_floor=roi_floor, max_additions=adaptive_max_additions,
            separation_nm=5.0)
        if additions.size == 0:
            adaptive_log.append({"round": round_index, "added_nm": [],
                                 "status": "no_untrained_failed_wavelengths_to_add"})
            break
        before_summary = design_summary_three
        train_nm = np.unique(np.r_[train_nm, additions])
        p_three = Problem(pupil, target, train_nm, band, grid_nm,
                          loss_profile="robust_cvar", roi_floor=roi_floor,
                          roi_weight=roi_weight, cvar_weight=cvar_weight,
                          cvar_tail_fraction=cvar_tail_fraction,
                          region_masks=region_masks)
        model_three = Model(p_three, "spectral", 3)
        # Coefficients use the fixed band-normalized Chebyshev basis, so they
        # can be carried exactly to the denser training set.
        adaptive_start = model_three.pack(rec_three["coefficients"])
        theta_three, loss_three, round_runs = solve(
            model_three, [adaptive_start], iterations,
            f"{case_name}/spectral_degree_3/adaptive_{round_index}",
            on_progress=checkpoint_callback(
                output_dir, f"spectral_degree_3_adaptive_{round_index}",
                model_three, train_nm, band))
        rec_three["coefficients"] = model_three.unpack(theta_three).copy()
        training_three, _, _ = model_three.evaluate(theta_three, train_nm)
        if not np.isclose(
                loss_three,
                objective_from_metrics(training_three, "robust_cvar", roi_floor,
                                       roi_weight, cvar_weight, cvar_tail_fraction),
                rtol=1e-9, atol=1e-12):
            raise AssertionError("Adaptive training metrics do not reproduce its objective")
        rows_three = scan_model(model_three, theta_three, scan_nm)
        design_summary_three = validation_summary(rows_three, [400.0, 700.0],
                                                  0.10, roi_floor)
        adaptive_row = {"round": round_index, "added_nm": additions.tolist(),
                        "training_nm": train_nm.tolist(), "loss": loss_three,
                        "runs": round_runs, "before_validation": before_summary,
                        "after_validation": design_summary_three}
        adaptive_log.append(adaptive_row)
        rec_three["adaptation_history"].append(adaptive_row)
        sections.append(("spectral_degree_3", f"training_adaptive_{round_index}",
                         training_three))

    rec_three["training_nm"] = train_nm.tolist()
    rec_three["loss"] = float(loss_three)
    rec_three["training"] = training_three
    rec_three["runs_adaptive"] = [x.get("runs", []) for x in adaptive_log]

    # Re-evaluate the quadratic model over the same dense grid, then evaluate
    # the final cubic model and persist every 1 nm pressure/design image.
    rows_two = scan_model(model_two, theta_two, scan_nm)
    if not adaptive_log:
        rows_three = scan_model(model_three, theta_three, scan_nm)
    summary_two = validation_summary(rows_two, [400.0, 700.0], 0.10, roi_floor)
    summary_three = validation_summary(rows_three, [400.0, 700.0], 0.10, roi_floor)
    pressure_two = validation_summary(rows_two, [380.0, 780.0], 0.10, roi_floor)
    pressure_three = validation_summary(rows_three, [380.0, 780.0], 0.10, roi_floor)
    rec_two["design_band_validation"] = summary_two
    rec_two["pressure_test"] = pressure_two
    rec_two["region_correlations_design_band"] = summarize_regions(
        [r for r in rows_two if 400 <= r["wavelength_nm"] <= 700],
        region_metadata or {"regions": []})
    rec_three["design_band_validation"] = summary_three
    rec_three["pressure_test"] = pressure_three
    rec_three["region_correlations_design_band"] = summarize_regions(
        [r for r in rows_three if 400 <= r["wavelength_nm"] <= 700],
        region_metadata or {"regions": []})
    rec_three["adaptive_rounds"] = adaptive_log
    rec_three["training"] = training_three
    rec_three["training_nm"] = train_nm.tolist()
    # Recompute final training rows after any adaptive insertion; this avoids
    # reporting the old 34-point rows against the final 46-point coefficient set.
    if len(train_nm) != len(initial_waves):
        rec_three["training"] = model_three.evaluate(theta_three, train_nm)[0]
        training_three = rec_three["training"]
    common_config["train_final_nm"] = train_nm.tolist()
    common_config["adaptive_rounds_actual"] = len(adaptive_log)
    save_json(output_dir/"config.json", common_config)

    image_file_two = model_two_dir/"spectral_scan_images.npy"
    rows_two_saved = scan_model(model_two, theta_two, scan_nm,
                                image_path=image_file_two)
    model_three_dir = output_dir/"spectral_degree_3"
    model_three_dir.mkdir(exist_ok=resume)
    image_file_three = model_three_dir/"spectral_scan_images.npy"
    rows_three_saved = scan_model(model_three, theta_three, scan_nm,
                                  image_path=image_file_three)
    sections += [("spectral_degree_2", "validation_1nm", rows_two_saved),
                 ("spectral_degree_3", "validation_1nm_final", rows_three_saved)]
    save_metrics_csv(output_dir/"metrics.csv", sections)

    for model_dir, model_name, model_obj, theta, training_rows, scan_rows, image_file in [
        (model_two_dir, "spectral_degree_2", model_two, theta_two,
         training_two, rows_two_saved, image_file_two),
        (model_three_dir, "spectral_degree_3", model_three, theta_three,
         training_three, rows_three_saved, image_file_three),
    ]:
        coefficients = model_obj.unpack(theta)
        np.savez_compressed(
            model_dir/"design.npz", coefficients=coefficients, theta=theta,
            pupil=pupil, target=target, training_nm=np.asarray(
                rec_two["training_nm"] if model_name.endswith("2") else train_nm),
            band_nm=band, scan_nm=scan_nm,
            screen_axis_m=screen_axis_m,
            detector_axis_m=detector_axis_m,
        )
        make_model_figures(model_dir, model_name, target, detector_axis_m,
                           scan_nm, scan_rows, image_file, region_metadata)

    payload = {
        "case": case_name,
        "configuration": common_config,
        "models": [{key: value for key, value in record.items()
                    if key != "coefficients"} for record in records],
        "acceptance_status": {
            "degree_2_design_band": summary_two["passed"],
            "degree_3_design_band": summary_three["passed"],
            "primary_target_passed": summary_two["passed"] or summary_three["passed"],
            "selected_model": ("spectral_degree_3" if summary_three["passed"]
                               else "spectral_degree_2" if summary_two["passed"]
                               else "none_meets_both_gates"),
        },
        "interpretation_limits": [
            "Free complex screen coefficients are not a material-realizable passive DOE.",
            "ROI power fraction is a numerical collection fraction, not device efficiency.",
            "Results are for a scalar point-sampled Fourier-lens model.",
            "The 1 nm sweep is dense validation, not a mathematical guarantee at every real wavelength.",
        ],
    }
    save_json(output_dir/"results.json", payload)
    save_json(output_dir/"summary.json", {
        "case": case_name,
        "degree_2": {"design_band": summary_two, "pressure": pressure_two},
        "degree_3": {"design_band": summary_three, "pressure": pressure_three},
        "degree_3_adaptive_rounds": adaptive_log,
    })
    report_lines = [
        f"# {case_name}: 二阶与三阶可见光谱联合设计", "",
        "损失采用 70% 全波长平均相对 L2²、30% 最差 10% 相对 L2² 均值，以及 ROI 下限惩罚。",
        f"ROI 定义为目标强度不低于峰值 1% 的像素；门槛为 {roi_floor:.2f}。",
        "训练波长初始为 400–700 nm 每 10 nm 加 488、532、633 nm；验收与压力扫描均为 1 nm 步长。", "",
        "| 模型 | 训练波长数 | 参数数 | 设计波段平均 L2 | P95 L2 | 最大 L2（波长） | 最小 ROI | 通过 | 压力段最大 L2（波长） | 压力段最小 ROI |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|---:|---:|",
    ]
    for record, design, pressure in [
        (rec_two, summary_two, pressure_two),
        (rec_three, summary_three, pressure_three),
    ]:
        report_lines.append(
            f"| {record['name']} | {len(record['training_nm'])} | {record['parameters']} | "
            f"{design['mean_relative_L2']:.5g} | {design['p95_relative_L2']:.5g} | "
            f"{design['worst_relative_L2']:.5g}（{design['worst_wavelength_nm']:g} nm） | "
            f"{design['minimum_roi_power']:.5g} | {'是' if design['passed'] else '否'} | "
            f"{pressure['worst_relative_L2']:.5g}（{pressure['worst_wavelength_nm']:g} nm） | "
            f"{pressure['minimum_roi_power']:.5g} |"
        )
    report_lines += ["", "## 多尺度区域相关性", ""]
    for width, values in rec_three["region_correlations_design_band"].get("by_line_width", {}).items():
        report_lines.append(f"- {width} 线宽：平均 Pearson {values['mean']:.5g}，最差 {values['minimum']:.5g}。")
    for orientation, values in rec_three["region_correlations_design_band"].get("by_orientation", {}).items():
        report_lines.append(f"- {orientation} 方向：平均 Pearson {values['mean']:.5g}，最差 {values['minimum']:.5g}。")
    report_lines += ["", "## 自适应补点", ""]
    if adaptive_log:
        for item in adaptive_log:
            report_lines.append(
                f"- 第 {item['round']} 轮加入 {item.get('added_nm', [])} nm；"
                f"轮后设计波段 {'通过' if item.get('after_validation', {}).get('passed') else '未通过'}。")
    elif not adaptive:
        report_lines.append("- 本次为控制组，未启用自适应补点。")
    elif adaptive_rounds == 0:
        report_lines.append("- 此次运行将自适应轮数设为 0，未执行补点。")
    elif summary_three["passed"]:
        report_lines.append("- 初始三阶解已通过双门槛，因此没有补点。")
    else:
        report_lines.append("- 自适应补点未运行；请检查记录中的停止状态。")
    report_lines += ["", "结果不能解释为实际材料器件效率；详细逐波长结果见 metrics.csv，"
                     "每个模型的全波段强度阵列见 spectral_scan_images.npy。", ""]
    (output_dir/"report.md").write_text("\n".join(report_lines), encoding="utf-8")
    return payload


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", type=Path,
                        default=HERE/"fourier_optics_step3_phase_screens.npz")
    parser.add_argument("--output", type=Path,
                        help="New run directory; it must not already exist")
    parser.add_argument("--grid", type=int, default=320,
                        help="Screen grid; the 380 nm Nyquist check is enforced at runtime")
    parser.add_argument("--detector-grid", type=int, default=256)
    parser.add_argument("--target-pitch-um", type=float, default=DEFAULT_PITCH_M*1e6)
    parser.add_argument("--supersample", type=int, default=DEFAULT_SUPERSAMPLE)
    parser.add_argument("--focal-m", type=float, default=0.2)
    parser.add_argument("--grid-wavelength-nm", type=float, default=400.0)
    parser.add_argument("--starts", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=400)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--adaptive-rounds", type=int, default=2)
    parser.add_argument("--adaptive-max-additions", type=int, default=6)
    parser.add_argument("--roi-floor", type=float, default=0.80)
    parser.add_argument("--roi-weight", type=float, default=0.10)
    parser.add_argument("--cvar-weight", type=float, default=0.30)
    parser.add_argument("--cvar-tail-fraction", type=float, default=0.10)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--resume", action="store_true",
                        help="Continue from completed model checkpoints in --output")
    args = parser.parse_args()
    if (args.grid < 16 or args.grid % 2 or args.detector_grid < 16
        or args.detector_grid % 2 or args.starts < 1 or args.iterations < 1
        or args.adaptive_rounds < 0 or args.adaptive_max_additions < 1
        or not np.isfinite(args.focal_m) or args.focal_m <= 0
        or not np.isfinite(args.grid_wavelength_nm) or args.grid_wavelength_nm <= 0
        or not np.isfinite(args.target_pitch_um) or args.target_pitch_um <= 0
        or not np.isfinite(args.roi_floor) or not np.isfinite(args.roi_weight)
        or not np.isfinite(args.cvar_weight) or not np.isfinite(args.cvar_tail_fraction)
        or not 0 <= args.roi_floor <= 1 or args.roi_weight < 0
        or not 0 <= args.cvar_weight <= 1
        or not 0 < args.cvar_tail_fraction <= 1):
        parser.error("Invalid sampling, optimization, or loss parameters")
    if args.detector_grid != 256:
        parser.error("The saved multiscale target is defined on a 256x256 detector grid")
    checks = verify_math()
    print("CZT, adjoint, mean/robust loss gradients, ROI penalty, and nesting checks PASSED.",
          flush=True)
    if args.check_only:
        print(json.dumps(checks, indent=2))
        return
    if args.output is None:
        parser.error("--output is required for a full experiment")
    if args.output.exists() and not args.resume:
        parser.error("Output directory already exists")
    if args.resume and not args.output.is_dir():
        parser.error("--resume requires an existing output directory")
    if args.resume:
        provenance_path = args.output/"provenance.json"
        if not provenance_path.is_file():
            parser.error("Existing run has no provenance.json; refusing an unverified resume")
        saved_configuration = json.loads(
            provenance_path.read_text(encoding="utf-8"))
        old_args = saved_configuration.get("configuration", {})
        current_args = vars(args)
        compare_keys = (
            "archive", "grid", "detector_grid", "target_pitch_um", "supersample",
            "focal_m", "grid_wavelength_nm", "starts", "iterations", "seed",
            "adaptive_rounds", "adaptive_max_additions", "roi_floor", "roi_weight",
            "cvar_weight", "cvar_tail_fraction",
        )
        mismatches = [key for key in compare_keys
                      if str(old_args.get(key)) != str(current_args.get(key))]
        if mismatches:
            parser.error("Resume configuration mismatch: " + ", ".join(mismatches))
    if args.grid_wavelength_nm*args.detector_grid/(380.0*args.grid) > 1:
        parser.error("Screen grid undersamples the detector at 380 nm; increase --grid")
    args.output.mkdir(parents=True, exist_ok=args.resume)
    target_dir = args.output/"target"
    target_dir.mkdir(exist_ok=args.resume)
    target, target_metadata = make_multiscale_target(
        args.detector_grid, args.target_pitch_um*1e-6, args.supersample)
    np.save(target_dir/"target.npy", target)
    save_json(target_dir/"target.json", target_metadata)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    target_axis_m = (np.arange(args.detector_grid)-args.detector_grid//2)*args.target_pitch_um*1e-6
    target_extent = [(target_axis_m[0]-args.target_pitch_um*0.5e-6)*1e3,
                     (target_axis_m[-1]+args.target_pitch_um*0.5e-6)*1e3]*2
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(target/target.max(), origin="lower", extent=target_extent,
              cmap="magma", vmin=0, vmax=1)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", title="Multiscale resolution target")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(target_dir/"target_preview.png", dpi=180)
    plt.close(fig)

    waves = training_wavelengths()
    scan_nm = np.arange(380.0, 781.0, 1.0)
    band = np.asarray([400.0, 700.0])
    with np.load(args.archive, allow_pickle=False) as archive:
        pupil, custom_target, _, screen_axis_m, detector_axis_m, metadata = load_case(
            archive, "multiscale", args.grid, args.detector_grid,
            args.grid_wavelength_nm, args.focal_m, target_dir/"target.npy")
        if not np.allclose(detector_axis_m, (np.arange(args.detector_grid)
              - args.detector_grid//2)*args.target_pitch_um*1e-6, atol=1e-12):
            raise AssertionError("Target sampling does not match the physical detector coordinates")
        if not np.isclose(target.sum(), 1.0, atol=1e-12) or np.any(target < 0):
            raise AssertionError("Generated target failed positivity/normalization checks")
        metadata.update({"target": "multiscale_resolution_chart",
                         "target_metadata": target_metadata,
                         "screen_sampling_check_380_nm": float(
                             args.grid_wavelength_nm*args.detector_grid/(380.0*args.grid))})
        save_json(target_dir/"sampling_report.json", metadata)
        top_config = {
            "configuration": vars(args),
            "training_nm_initial": waves.tolist(),
            "validation_nm": scan_nm.tolist(),
            "target": target_metadata,
            "screen_sampling": metadata,
            "math_checks": checks,
            "provenance": {
                "python": sys.version,
                "numpy": np.__version__,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "model_source_sha256": hashlib.sha256(
                    (HERE/"fourier_optics_step11_free_screen_optimization.py").read_bytes()
                ).hexdigest(),
            },
        }
        save_json(args.output/"math_checks.json", checks)
        save_json(args.output/"provenance.json", top_config)
        common_case_args = dict(
            initial_waves=waves, scan_nm=scan_nm, band=band,
            grid_nm=args.grid_wavelength_nm, detector_axis_m=detector_axis_m,
            screen_axis_m=screen_axis_m,
            iterations=args.iterations, starts=args.starts, seed=args.seed,
            adaptive_rounds=args.adaptive_rounds,
            adaptive_max_additions=args.adaptive_max_additions,
            roi_floor=args.roi_floor, roi_weight=args.roi_weight,
            cvar_weight=args.cvar_weight,
            cvar_tail_fraction=args.cvar_tail_fraction,
            resume=args.resume,
        )
        primary = run_case(
            "multiscale_resolution_chart", pupil, custom_target,
            region_metadata=target_metadata, output_dir=args.output/"multiscale",
            adaptive=True, **common_case_args)
        ring_pupil, ring_target, _, _, ring_axis, ring_metadata = load_case(
            archive, "annular_ring", args.grid, args.detector_grid,
            args.grid_wavelength_nm, args.focal_m)
        if not np.allclose(ring_axis, detector_axis_m):
            raise AssertionError("Control target is not on the same physical detector grid")
        control = run_case(
            "annular_ring_control", ring_pupil, ring_target,
            region_metadata=None, output_dir=args.output/"annular_ring_control",
            adaptive=False, seed=args.seed+1, **{
                key: value for key, value in common_case_args.items()
                if key != "seed"})
    root_summary = {
        "primary_target": primary["acceptance_status"],
        "annular_ring_control": control["acceptance_status"],
        "interpretation": "Pass requires every 1 nm design-band sample to satisfy both thresholds.",
    }
    save_json(args.output/"summary.json", root_summary)
    report_lines = [
        "# 可见光全波段多尺度联合设计实验", "",
        "固定 Step 11 孔径、焦距和自由复屏模型，对 400–700 nm 做 1 nm 验收；380–780 nm 为压力测试。",
        "训练目标使用 34 个初始波长、70/30 平均与最差 10% 相对 L2² 稳健项，以及 ROI=0.80 的效率下限惩罚。",
        "主目标最多补两轮、每轮最多六个波长；圆环是控制组。", "",
        "验收要求设计波段全部 301 个点均满足相对 L2≤0.10 且 ROI 功率比例≥0.80。",
    ]
    for folder, title in [("multiscale", "多尺度分辨率目标"),
                          ("annular_ring_control", "圆环控制组")]:
        result = json.loads((args.output/folder/"summary.json").read_text(encoding="utf-8"))
        report_lines += ["", f"## {title}", "",
            "| 模型 | 平均 L2 | P95 L2 | 最大 L2（波长） | 最小 ROI | 设计波段通过 | 压力段最大 L2（波长） | 压力段最小 ROI |",
            "|---|---:|---:|---:|---:|:---:|---:|---:|"]
        for order in ("degree_2", "degree_3"):
            design = result[order]["design_band"]
            pressure = result[order]["pressure"]
            report_lines.append(
                f"| {order.replace('_', ' ')} | {design['mean_relative_L2']:.5g} | "
                f"{design['p95_relative_L2']:.5g} | "
                f"{design['worst_relative_L2']:.5g}（{design['worst_wavelength_nm']:g} nm） | "
                f"{design['minimum_roi_power']:.5g} | {'是' if design['passed'] else '否'} | "
                f"{pressure['worst_relative_L2']:.5g}（{pressure['worst_wavelength_nm']:g} nm） | "
                f"{pressure['minimum_roi_power']:.5g} |"
            )
        report_lines.append(f"详细逐波长数据、图表和运行日志：`{folder}/`。")
        if folder == "multiscale":
            rounds = result.get("degree_3_adaptive_rounds", [])
            if rounds:
                report_lines.append("三阶自适应补点：" + "; ".join(
                    f"第 {item['round']} 轮 {item['added_nm']} nm"
                    for item in rounds) + "。")
    report_lines += ["", "屏面自由复系数不能解释为可加工的被动材料器件；ROI 比例也不是真实器件效率。",
                     "本结果来自标量、点采样 Fourier 透镜数值模型；1 nm 扫描是稠密离散验证，不是连续波段的数学证明。", ""]
    (args.output/"report.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"Completed experiment: {args.output.resolve()}", flush=True)


if __name__ == "__main__":
    main()
