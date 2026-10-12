"""Step 9: explore integer relief branches and joint multiwavelength height design.

The integer-branch heuristic preserves the exact 532 nm phase screen. A second
experiment directly optimizes one bounded continuous relief height profile for
three wavelengths under the prescribed lambda*f=constant Fourier operator.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = OUT_DIR / "fourier_optics_step3_phase_screens.npz"
LAMBDA0_M = 532.0e-9
F0_M = 0.200
WAVELENGTHS_NM = np.array([450.0, 532.0, 633.0])
INTEGER_CAPS = [1, 2, 4, 8]
MAX_HEIGHT_CYCLES = 3.0
ADAM_ITERATIONS = 160
ADAM_LEARNING_RATE_CYCLES = 0.012
ROI_POWER_FLOOR = 0.85
ROI_PENALTY_WEIGHT = 100.0

SELLMEIER_B = np.array([0.6961663, 0.4079426, 0.8974794])
SELLMEIER_C_UM = np.array([0.0684043, 0.1162414, 9.896161])


def fused_silica_n(wavelength_m: float | np.ndarray) -> np.ndarray:
    wavelength_um = np.asarray(wavelength_m, dtype=np.float64) * 1e6
    lam2 = wavelength_um[..., None] ** 2
    n2 = 1.0 + np.sum(SELLMEIER_B * lam2 / (lam2 - SELLMEIER_C_UM**2), axis=-1)
    return np.sqrt(n2)


def centered_fft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def centered_ifft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def metrics(target: np.ndarray, image: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    target_norm = target / max(float(np.sum(target)), 1e-300)
    image_norm = image / max(float(np.sum(image)), 1e-300)
    diff = image_norm - target_norm
    return {
        "relative_L2_to_target": float(np.linalg.norm(diff.ravel()) / max(float(np.linalg.norm(target_norm.ravel())), 1e-300)),
        "squared_Bhattacharyya_fidelity_to_target": float(np.sum(np.sqrt(target_norm * image_norm)) ** 2),
        "power_fraction_in_target_1pct_ROI": float(np.sum(image_norm[roi])),
    }


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = a / max(float(np.sum(a)), 1e-300)
    bb = b / max(float(np.sum(b)), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / max(float(np.linalg.norm(bb.ravel())), 1e-300))


def evaluate_height_cycles(
    q: np.ndarray,
    pupil: np.ndarray,
    gammas: np.ndarray,
    target: np.ndarray,
    roi: np.ndarray,
) -> tuple[list[np.ndarray], list[dict[str, float]]]:
    images: list[np.ndarray] = []
    rows: list[dict[str, float]] = []
    for wavelength_nm, gamma in zip(WAVELENGTHS_NM, gammas):
        screen = np.zeros(q.shape, dtype=np.complex128)
        screen[pupil] = np.exp(1j * 2.0 * np.pi * gamma * q[pupil])
        image = np.abs(centered_fft(screen)) ** 2
        image /= max(float(np.sum(image)), 1e-300)
        images.append(image)
        rows.append({"wavelength_nm": float(wavelength_nm), **metrics(target, image, roi)})
    return images, rows


def make_local_integer_branch(q0: np.ndarray, pupil: np.ndarray, gammas: np.ndarray, cap: int) -> tuple[np.ndarray, np.ndarray]:
    q_best = q0.copy()
    m_best = np.zeros(q0.shape, dtype=np.int16)
    reference = np.exp(1j * 2.0 * np.pi * q0[pupil])
    best_cost = np.full(reference.shape, np.inf, dtype=np.float64)
    # 532 nm contributes no preference because every integer branch is exact there.
    off_reference_gammas = [float(gammas[0]), float(gammas[2])]
    for m in range(cap + 1):
        candidate = q0[pupil] + m
        cost = np.zeros(candidate.shape, dtype=np.float64)
        for gamma in off_reference_gammas:
            phasor = np.exp(1j * 2.0 * np.pi * gamma * candidate)
            cost += np.abs(phasor - reference) ** 2
        if m == 0:
            best_cost[:] = cost
        else:
            update = cost < best_cost
            best_cost[update] = cost[update]
            active_m = m_best[pupil]
            active_m[update] = m
            m_best[pupil] = active_m
    q_best[pupil] = q0[pupil] + m_best[pupil]
    return q_best, m_best


def objective_and_gradient(
    q_active: np.ndarray,
    pupil: np.ndarray,
    gammas: np.ndarray,
    target_norm: np.ndarray,
    target_norm_sq: float,
    energy: float,
    roi: np.ndarray,
    roi_power_floor: float | None = None,
    roi_penalty_weight: float = 0.0,
) -> tuple[float, np.ndarray, list[float], list[float]]:
    n_grid = pupil.shape[0]
    q = np.zeros(pupil.shape, dtype=np.float64)
    q[pupil] = q_active
    grad = np.zeros(pupil.shape, dtype=np.float64)
    total_loss = 0.0
    per_wave_relative_squared = []
    per_wave_roi_power = []
    wavelength_count = float(len(gammas))

    for gamma in gammas:
        screen = np.zeros(pupil.shape, dtype=np.complex128)
        screen[pupil] = np.exp(1j * 2.0 * np.pi * gamma * q_active)
        spectrum = centered_fft(screen)
        intensity = np.abs(spectrum) ** 2
        probability = intensity / energy
        diff = probability - target_norm
        relative_squared = float(np.sum(diff**2) / target_norm_sq)
        roi_power = float(np.sum(probability[roi]))
        per_wave_relative_squared.append(relative_squared)
        per_wave_roi_power.append(roi_power)
        total_loss += 0.5 * relative_squared / wavelength_count

        roi_shortage = 0.0
        if roi_power_floor is not None and roi_penalty_weight > 0.0:
            roi_shortage = max(0.0, roi_power_floor - roi_power)
            total_loss += 0.5 * roi_penalty_weight * roi_shortage**2 / wavelength_count

        # Adjoint of the centered, unnormalized FFT is N^2 times centered IFFT.
        intensity_adjoint = diff / (energy * target_norm_sq * wavelength_count)
        if roi_shortage > 0.0:
            intensity_adjoint -= roi_penalty_weight * roi_shortage * roi / (energy * wavelength_count)
        field_adjoint = (n_grid**2) * centered_ifft(intensity_adjoint * spectrum)
        phase_gradient = -2.0 * np.imag(np.conj(field_adjoint[pupil]) * screen[pupil])
        grad[pupil] += (2.0 * np.pi * gamma) * phase_gradient
    return total_loss, grad[pupil], per_wave_relative_squared, per_wave_roi_power


def run_adam(
    q_initial: np.ndarray,
    pupil: np.ndarray,
    gammas: np.ndarray,
    target_norm: np.ndarray,
    target_norm_sq: float,
    energy: float,
    roi: np.ndarray,
    roi_power_floor: float | None = None,
    roi_penalty_weight: float = 0.0,
) -> tuple[np.ndarray, float, int, dict[str, list]]:
    q_active = q_initial.copy()
    adam_m = np.zeros_like(q_active)
    adam_v = np.zeros_like(q_active)
    best_q = q_active.copy()
    best_loss = float("inf")
    best_iteration = 0
    history: dict[str, list] = {"iteration": [], "objective": [], "relative_L2_by_wavelength": [], "roi_power_by_wavelength": []}
    for iteration in range(ADAM_ITERATIONS + 1):
        loss, grad, relative_squared, roi_powers = objective_and_gradient(
            q_active, pupil, gammas, target_norm, target_norm_sq, energy, roi,
            roi_power_floor, roi_penalty_weight,
        )
        history["iteration"].append(iteration)
        history["objective"].append(float(loss))
        history["relative_L2_by_wavelength"].append(np.sqrt(np.maximum(relative_squared, 0.0)).tolist())
        history["roi_power_by_wavelength"].append(roi_powers)
        if loss < best_loss:
            best_loss = float(loss)
            best_q = q_active.copy()
            best_iteration = iteration
        if iteration == ADAM_ITERATIONS:
            break
        grad_rms = float(np.sqrt(np.mean(grad**2)))
        grad_normalized = grad / max(grad_rms, 1e-30)
        adam_m = 0.9 * adam_m + 0.1 * grad_normalized
        adam_v = 0.999 * adam_v + 0.001 * grad_normalized**2
        m_hat = adam_m / (1.0 - 0.9 ** (iteration + 1))
        v_hat = adam_v / (1.0 - 0.999 ** (iteration + 1))
        q_active = np.clip(
            q_active - ADAM_LEARNING_RATE_CYCLES * m_hat / (np.sqrt(v_hat) + 1e-8),
            0.0,
            MAX_HEIGHT_CYCLES,
        )
    return best_q, best_loss, best_iteration, history


def profile_summary(q: np.ndarray, pupil: np.ndarray, height_2pi_m: float) -> dict[str, float]:
    height = q[pupil] * height_2pi_m
    dx_values = np.abs(np.diff(q, axis=1))
    dy_values = np.abs(np.diff(q, axis=0))
    valid_x = pupil[:, 1:] & pupil[:, :-1]
    valid_y = pupil[1:, :] & pupil[:-1, :]
    edge_steps = np.concatenate((dx_values[valid_x], dy_values[valid_y])) * height_2pi_m
    return {
        "height_min_um": float(np.min(height) * 1e6),
        "height_max_um": float(np.max(height) * 1e6),
        "height_mean_um": float(np.mean(height) * 1e6),
        "height_rms_neighbor_step_um": float(np.sqrt(np.mean(edge_steps**2)) * 1e6) if edge_steps.size else 0.0,
        "neighbor_step_95th_percentile_um": float(np.percentile(edge_steps, 95) * 1e6) if edge_steps.size else 0.0,
    }


def main() -> None:
    with np.load(ARCHIVE_PATH, allow_pickle=False) as archive:
        pupil = archive["pupil"].astype(bool)
        phase_ref = archive["three_spots_phase_rad"].astype(np.float64)
        target = archive["three_spots_target_intensity"].astype(np.float64)
        dx = float(archive["input_pixel_pitch_m"])
        lambda_ref_archive = float(archive["wavelength_m"])
        f_ref_archive = float(archive["focal_length_m"])

    if abs(lambda_ref_archive - LAMBDA0_M) > 1e-15 or abs(f_ref_archive - F0_M) > 1e-12:
        raise ValueError("Step 3 archive does not match Step 9 reference settings")

    n_grid = int(pupil.shape[0])
    phi0 = np.mod(phase_ref, 2.0 * np.pi)
    q0 = np.where(pupil, phi0 / (2.0 * np.pi), 0.0)
    n0 = float(fused_silica_n(LAMBDA0_M))
    height_2pi_m = LAMBDA0_M / (n0 - 1.0)
    n_values = fused_silica_n(WAVELENGTHS_NM * 1e-9)
    gammas = ((n_values - 1.0) / (WAVELENGTHS_NM * 1e-9)) / ((n0 - 1.0) / LAMBDA0_M)
    target_norm = target / max(float(np.sum(target)), 1e-300)
    target_norm_sq = float(np.sum(target_norm**2))
    roi = target >= 0.01 * float(np.max(target))
    energy = float(n_grid) ** 2 * float(np.sum(pupil))

    baseline_images, baseline_rows = evaluate_height_cycles(q0, pupil, gammas, target_norm, roi)
    reference_image = baseline_images[1]
    baseline_rows = [
        {**row, "relative_L2_to_532nm_principal_branch_image": relative_l2(image, reference_image)}
        for row, image in zip(baseline_rows, baseline_images)
    ]
    integer_results = []
    integer_images_by_cap: dict[str, list[np.ndarray]] = {}
    integer_profiles: dict[str, np.ndarray] = {}
    integer_m_maps: dict[str, np.ndarray] = {}
    for cap in INTEGER_CAPS:
        q_branch, m_map = make_local_integer_branch(q0, pupil, gammas, cap)
        images, rows = evaluate_height_cycles(q_branch, pupil, gammas, target_norm, roi)
        rows = [
            {**row, "relative_L2_to_532nm_principal_branch_image": relative_l2(image, reference_image)}
            for row, image in zip(rows, images)
        ]
        integer_images_by_cap[str(cap)] = images
        integer_profiles[str(cap)] = q_branch.astype(np.float32)
        integer_m_maps[str(cap)] = m_map.astype(np.int8)
        integer_results.append({
            "maximum_added_2pi_orders": cap,
            "mean_three_wavelength_relative_L2": float(np.mean([row["relative_L2_to_target"] for row in rows])),
            "maximum_three_wavelength_relative_L2": float(np.max([row["relative_L2_to_target"] for row in rows])),
            "branch_pixels_with_m_nonzero_fraction": float(np.mean(m_map[pupil] > 0)),
            "selected_m_mean": float(np.mean(m_map[pupil])),
            "selected_m_max": int(np.max(m_map[pupil])),
            "profile": profile_summary(q_branch, pupil, height_2pi_m),
            "wavelength_results": rows,
        })

    # Directly optimize one continuous height profile against all three wavelengths.
    # q is height expressed in units of the 532 nm 2pi relief depth.
    best_q, best_loss, best_iteration, history_joint = run_adam(
        q0[pupil], pupil, gammas, target_norm, target_norm_sq, energy, roi
    )
    q_joint = np.zeros(pupil.shape, dtype=np.float64)
    q_joint[pupil] = best_q
    joint_images, joint_rows = evaluate_height_cycles(q_joint, pupil, gammas, target_norm, roi)
    joint_rows = [
        {**row, "relative_L2_to_532nm_principal_branch_image": relative_l2(image, reference_image)}
        for row, image in zip(joint_rows, joint_images)
    ]
    joint_result = {
        "method": "analytic-gradient Adam optimization of continuous relief height, equal-weight three-wavelength normalized intensity L2 objective",
        "iterations": ADAM_ITERATIONS,
        "best_iteration": best_iteration,
        "best_objective_half_mean_squared_relative_L2": best_loss,
        "maximum_height_cycles": MAX_HEIGHT_CYCLES,
        "maximum_height_um": MAX_HEIGHT_CYCLES * height_2pi_m * 1e6,
        "learning_rate_cycles": ADAM_LEARNING_RATE_CYCLES,
        "profile": profile_summary(q_joint, pupil, height_2pi_m),
        "wavelength_results": joint_rows,
    }

    # Continue from the shape-only solution and penalize any wavelength below
    # the chosen target-ROI power floor. This illustrates the accuracy/throughput tradeoff.
    best_q_roi, best_loss_roi, best_iteration_roi, history_roi = run_adam(
        best_q, pupil, gammas, target_norm, target_norm_sq, energy, roi,
        ROI_POWER_FLOOR, ROI_PENALTY_WEIGHT,
    )
    q_roi = np.zeros(pupil.shape, dtype=np.float64)
    q_roi[pupil] = best_q_roi
    roi_images, roi_rows = evaluate_height_cycles(q_roi, pupil, gammas, target_norm, roi)
    roi_rows = [
        {**row, "relative_L2_to_532nm_principal_branch_image": relative_l2(image, reference_image)}
        for row, image in zip(roi_rows, roi_images)
    ]
    roi_result = {
        "method": "analytic-gradient Adam continuing from the shape-only solution; adds a quadratic penalty when target-ROI power is below the selected floor",
        "iterations": ADAM_ITERATIONS,
        "best_iteration": best_iteration_roi,
        "best_combined_objective": best_loss_roi,
        "roi_power_floor": ROI_POWER_FLOOR,
        "roi_penalty_weight": ROI_PENALTY_WEIGHT,
        "maximum_height_cycles": MAX_HEIGHT_CYCLES,
        "maximum_height_um": MAX_HEIGHT_CYCLES * height_2pi_m * 1e6,
        "learning_rate_cycles": ADAM_LEARNING_RATE_CYCLES,
        "profile": profile_summary(q_roi, pupil, height_2pi_m),
        "wavelength_results": roi_rows,
        "constraint_note": "soft penalty, not a hard guarantee that every wavelength reaches the floor",
    }

    fig, axes = plt.subplots(5, 3, figsize=(12.5, 16), constrained_layout=True)
    row_images = [baseline_images, integer_images_by_cap[str(INTEGER_CAPS[-1])], joint_images, roi_images]
    row_titles = ["Principal minimum-height branch", f"Local integer branch (m ≤ {INTEGER_CAPS[-1]})",
                  "Joint shape optimization", f"Joint with ROI power floor {ROI_POWER_FLOOR:.2f}"]
    images_grid = [[target_norm] * 3] + row_images
    titles_grid = [[f"Target — {w:.0f} nm" for w in WAVELENGTHS_NM]] + [
        [f"{row_titles[i]}\n{WAVELENGTHS_NM[j]:.0f} nm" for j in range(3)] for i in range(4)
    ]
    fx = np.fft.fftshift(np.fft.fftfreq(n_grid, d=dx))
    output_axis_mm = LAMBDA0_M * F0_M * fx * 1e3
    crop_idx = np.flatnonzero(np.abs(output_axis_mm) <= 2.0)
    lo, hi = int(crop_idx[0]), int(crop_idx[-1]) + 1
    extent = [output_axis_mm[lo], output_axis_mm[hi - 1], output_axis_mm[lo], output_axis_mm[hi - 1]]
    for r in range(5):
        for c in range(3):
            small = images_grid[r][c][lo:hi, lo:hi]
            axes[r, c].imshow(np.log10(np.maximum(small / max(float(np.max(small)), 1e-30), 1e-7)),
                              extent=extent, origin="lower", cmap="magma", vmin=-7, vmax=0)
            axes[r, c].set_title(titles_grid[r][c], fontsize=9)
            axes[r, c].set_xlabel("X (mm)")
            axes[r, c].set_ylabel("Y (mm)")
    fig.suptitle("Three-wavelength fused-silica relief designs on a common Fourier grid", fontsize=13)
    fig.savefig(OUT_DIR / "fourier_optics_step9_height_patterns.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), constrained_layout=True)
    axes[0].plot(history_joint["iteration"], [np.mean(x) for x in history_joint["relative_L2_by_wavelength"]],
                 color="#315a9b", label="shape-only design")
    axes[0].plot(history_roi["iteration"], [np.mean(x) for x in history_roi["relative_L2_by_wavelength"]],
                 color="#b54136", label="ROI-penalized design")
    axes[0].set(xlabel="Iteration", ylabel="Mean relative L2 to target", title="Joint height optimization")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False, fontsize=8)
    caps = [0] + INTEGER_CAPS
    cap_means = [float(np.mean([row["relative_L2_to_target"] for row in baseline_rows]))] + [x["mean_three_wavelength_relative_L2"] for x in integer_results]
    axes[1].plot(caps, cap_means, "o-", color="#b54136", label="local integer branch")
    axes[1].axhline(float(np.mean([row["relative_L2_to_target"] for row in baseline_rows])),
                    color="#777777", ls="--", label="principal branch")
    axes[1].axhline(float(np.mean([row["relative_L2_to_target"] for row in joint_rows])),
                    color="#228477", ls=":", label="joint continuous height")
    axes[1].set(xlabel="Maximum added integer 2π orders", ylabel="Mean three-wavelength relative L2",
                title="Height branch heuristic vs joint design")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT_DIR / "fourier_optics_step9_optimization.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7.5, 4.5), constrained_layout=True)
    for label, rows, color, marker in [
        ("Principal branch", baseline_rows, "#777777", "o"),
        ("Shape-only joint", joint_rows, "#315a9b", "s"),
        ("ROI-penalized joint", roi_rows, "#b54136", "^"),
    ]:
        ax.plot([r["power_fraction_in_target_1pct_ROI"] for r in rows],
                [r["relative_L2_to_target"] for r in rows], marker=marker, color=color, label=label)
        for row in rows:
            ax.annotate(f"{row['wavelength_nm']:.0f} nm",
                        (row["power_fraction_in_target_1pct_ROI"], row["relative_L2_to_target"]),
                        xytext=(4, 4), textcoords="offset points", fontsize=8)
    ax.axvline(ROI_POWER_FLOOR, color="#b54136", ls="--", lw=1, alpha=0.7, label="selected ROI floor")
    ax.set(xlabel="Power fraction in target 1% ROI", ylabel="Relative L2 to target",
           title="Shape error and target-region power")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(OUT_DIR / "fourier_optics_step9_roi_tradeoff.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 4, figsize=(14, 4), constrained_layout=True)
    profiles = [q0, integer_profiles[str(INTEGER_CAPS[-1])], q_joint, q_roi]
    profile_titles = ["Principal branch", f"Integer branch m≤{INTEGER_CAPS[-1]}", "Shape-only joint", "ROI-penalized joint"]
    for ax, q_profile, title in zip(axes, profiles, profile_titles):
        display = np.where(pupil, q_profile, np.nan)
        im = ax.imshow(display, origin="lower", cmap="viridis", vmin=0, vmax=max(float(np.nanmax(display)), 1.0))
        ax.set_title(title)
        ax.set_xticks([])
        ax.set_yticks([])
        fig.colorbar(im, ax=ax, shrink=0.8, label="height / 2π relief")
    fig.savefig(OUT_DIR / "fourier_optics_step9_height_maps.png", dpi=170)
    plt.close(fig)

    results = {
        "step": 9,
        "title": "Integer relief branch heuristic and direct three-wavelength height optimization",
        "status": "preliminary nonconvex scalar optimization; heuristic branch map and one bounded Adam run, not a global optimum",
        "reference": {
            "wavelength_nm": float(LAMBDA0_M * 1e9),
            "focal_length_m": F0_M,
            "n_fused_silica": n0,
            "one_2pi_height_um": height_2pi_m * 1e6,
            "wavelengths_nm": WAVELENGTHS_NM.tolist(),
            "phase_scales_gamma": [float(x) for x in gammas],
            "pupil_pixels": int(np.sum(pupil)),
            "grid_N": n_grid,
            "input_pitch_um": dx * 1e6,
            "height_design_variable": "q=h/[lambda0/(n(lambda0)-1)] in reference 2pi cycles",
            "ideal_operator": "f(lambda)=f0*lambda0/lambda; same physical Fourier coordinate grid",
        },
        "principal_minimum_height_branch": {
            "profile": profile_summary(q0, pupil, height_2pi_m),
            "mean_three_wavelength_relative_L2": float(np.mean([row["relative_L2_to_target"] for row in baseline_rows])),
            "wavelength_results": baseline_rows,
        },
        "integer_branch_heuristics": {
            "selection_rule": "per pupil pixel, choose m in [0, cap] minimizing the sum of squared phasor differences from the 532 nm reference screen at 450 and 633 nm; 532 nm remains exact",
            "results_by_maximum_m": integer_results,
            "note": "local phase matching is only a heuristic; it does not directly minimize image error and does not constrain neighboring height steps",
        },
        "joint_continuous_height_design": joint_result,
        "joint_height_with_roi_power_penalty": roi_result,
        "optimizer_history": {"shape_only": history_joint, "roi_penalized": history_roi},
        "figures": [
            "fourier_optics_step9_height_patterns.png",
            "fourier_optics_step9_optimization.png",
            "fourier_optics_step9_height_maps.png",
            "fourier_optics_step9_roi_tradeoff.png",
        ],
    }
    with (OUT_DIR / "fourier_optics_step9_results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)
    np.savez_compressed(
        OUT_DIR / "fourier_optics_step9_height_designs.npz",
        q_principal=q0.astype(np.float32),
        q_integer_cap8=integer_profiles[str(INTEGER_CAPS[-1])],
        m_integer_cap8=integer_m_maps[str(INTEGER_CAPS[-1])],
        q_joint_continuous=q_joint.astype(np.float32),
        q_joint_roi_penalized=q_roi.astype(np.float32),
        pupil=pupil.astype(np.uint8),
        target_normalized=target_norm.astype(np.float32),
        wavelength_nm=WAVELENGTHS_NM.astype(np.float32),
        baseline_images=np.stack(baseline_images).astype(np.float32),
        integer_cap8_images=np.stack(integer_images_by_cap[str(INTEGER_CAPS[-1])]).astype(np.float32),
        joint_images=np.stack(joint_images).astype(np.float32),
        roi_penalized_images=np.stack(roi_images).astype(np.float32),
        input_pitch_m=np.asarray(dx),
        one_2pi_height_m=np.asarray(height_2pi_m),
    )


if __name__ == "__main__":
    main()
