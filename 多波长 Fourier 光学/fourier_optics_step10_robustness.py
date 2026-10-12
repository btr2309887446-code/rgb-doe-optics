"""Step 10: ROI constraint sensitivity, warm-start variation, and band sweep."""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

font_path = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False


OUT = Path(__file__).resolve().parent
INPUT = OUT / "fourier_optics_step9_height_designs.npz"
STEP9_PATH = OUT / "fourier_optics_step9_height_optimization.py"
FLOOR = 0.85
MAX_Q = 3.0
OUTER_ROUNDS = 4
INNER_STEPS = 25
RHO = 40.0
LR = 0.012
SEED = 20260930


def load_step9():
    spec = importlib.util.spec_from_file_location("step9", STEP9_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def evaluate(q, pupil, gammas, target, roi, module):
    images = []
    rows = []
    for i, gamma in enumerate(gammas):
        screen = np.zeros(q.shape, dtype=np.complex128)
        screen[pupil] = np.exp(1j * 2.0 * np.pi * gamma * q[pupil])
        image = np.abs(module.centered_fft(screen)) ** 2
        image /= max(float(np.sum(image)), 1e-300)
        images.append(image.astype(np.float32))
        rows.append({"wavelength_nm": float([450.0, 532.0, 633.0][i]), **module.metrics(target, image, roi)})
    return np.stack(images), rows


def objective_gradient(q_active, pupil, gammas, target_norm, target_norm_sq, energy, roi, multipliers, rho, module):
    n_grid = pupil.shape[0]
    q = np.zeros(pupil.shape, dtype=np.float64)
    q[pupil] = q_active
    grad = np.zeros(pupil.shape, dtype=np.float64)
    wavelength_count = float(len(gammas))
    total = 0.0
    deficits = []
    errors_sq = []
    roi_powers = []
    for wi, gamma in enumerate(gammas):
        screen = np.zeros(pupil.shape, dtype=np.complex128)
        screen[pupil] = np.exp(1j * 2.0 * np.pi * gamma * q_active)
        spectrum = module.centered_fft(screen)
        probability = np.abs(spectrum) ** 2 / energy
        diff = probability - target_norm
        relative_squared = float(np.sum(diff**2) / target_norm_sq)
        roi_power = float(np.sum(probability[roi]))
        deficit = max(0.0, FLOOR - roi_power)
        errors_sq.append(relative_squared)
        roi_powers.append(roi_power)
        deficits.append(deficit)
        total += 0.5 * relative_squared / wavelength_count
        total += (multipliers[wi] * deficit + 0.5 * rho * deficit**2) / wavelength_count
        intensity_adjoint = diff / (energy * target_norm_sq * wavelength_count)
        if deficit > 0.0:
            intensity_adjoint -= (multipliers[wi] + rho * deficit) * roi / (energy * wavelength_count)
        field_adjoint = (n_grid**2) * module.centered_ifft(intensity_adjoint * spectrum)
        phase_gradient = -2.0 * np.imag(np.conj(field_adjoint[pupil]) * screen[pupil])
        grad[pupil] += (2.0 * np.pi * gamma) * phase_gradient
    return total, grad[pupil], np.asarray(deficits), np.sqrt(np.maximum(errors_sq, 0.0)), np.asarray(roi_powers)


def augmented_lagrange(q_start, pupil, gammas, target_norm, target_norm_sq, energy, roi, module):
    q_active = q_start[pupil].copy()
    moments = np.zeros_like(q_active)
    variances = np.zeros_like(q_active)
    multipliers = np.zeros(len(gammas), dtype=np.float64)
    beta1, beta2 = 0.9, 0.999
    history = []
    best_q = q_start.copy()
    best_key = None
    for outer in range(OUTER_ROUNDS):
        for inner in range(INNER_STEPS + 1):
            iteration = outer * INNER_STEPS + inner
            loss, grad, deficits, l2s, etas = objective_gradient(
                q_active, pupil, gammas, target_norm, target_norm_sq, energy, roi, multipliers, RHO, module
            )
            max_deficit = float(np.max(deficits))
            key = (max_deficit, float(np.mean(l2s)))
            if best_key is None or key < best_key:
                best_key = key
                best_q[:] = 0.0
                best_q[pupil] = q_active
            history.append({
                "iteration": iteration,
                "outer_round": outer,
                "objective": float(loss),
                "relative_L2_by_wavelength": l2s.tolist(),
                "roi_power_by_wavelength": etas.tolist(),
                "floor_deficit_by_wavelength": deficits.tolist(),
                "lagrange_multipliers": multipliers.tolist(),
            })
            if inner == INNER_STEPS:
                break
            grad_rms = float(np.sqrt(np.mean(grad**2)))
            normalized = grad / max(grad_rms, 1e-30)
            moments = beta1 * moments + (1.0 - beta1) * normalized
            variances = beta2 * variances + (1.0 - beta2) * normalized**2
            t = iteration + 1
            mhat = moments / (1.0 - beta1**t)
            vhat = variances / (1.0 - beta2**t)
            q_active = np.clip(q_active - LR * mhat / (np.sqrt(vhat) + 1e-8), 0.0, MAX_Q)
        _, _, deficits, _, _ = objective_gradient(
            q_active, pupil, gammas, target_norm, target_norm_sq, energy, roi, multipliers, RHO, module
        )
        multipliers = np.maximum(0.0, multipliers + RHO * deficits)
    return best_q, history, multipliers.tolist()


def roi_maximin_gradient(q_active, pupil, gammas, energy, roi, module, temperature=0.025):
    """Gradient of a smooth minimum of the wavelength ROI powers."""
    n_grid = pupil.shape[0]
    q = np.zeros(pupil.shape, dtype=np.float64)
    q[pupil] = q_active
    spectra = []
    etas = []
    for gamma in gammas:
        screen = np.zeros(pupil.shape, dtype=np.complex128)
        screen[pupil] = np.exp(1j * 2.0 * np.pi * gamma * q_active)
        spectrum = module.centered_fft(screen)
        probability = np.abs(spectrum) ** 2 / energy
        spectra.append((screen, spectrum))
        etas.append(float(np.sum(probability[roi])))
    etas = np.asarray(etas)
    log_weights = -etas / temperature
    shift = float(np.max(log_weights))
    unnormalized = np.exp(log_weights - shift)
    weights = unnormalized / np.sum(unnormalized)
    soft_min = -temperature * (shift + np.log(np.mean(unnormalized)))
    grad = np.zeros(pupil.shape, dtype=np.float64)
    for wi, (gamma, (screen, spectrum)) in enumerate(zip(gammas, spectra)):
        intensity_adjoint = -weights[wi] * roi / energy
        field_adjoint = (n_grid**2) * module.centered_ifft(intensity_adjoint * spectrum)
        phase_gradient = -2.0 * np.imag(np.conj(field_adjoint[pupil]) * screen[pupil])
        grad[pupil] += (2.0 * np.pi * gamma) * phase_gradient
    return -float(soft_min), grad[pupil], etas


def maximize_minimum_roi(q_start, pupil, gammas, energy, roi, module, iterations=120, learning_rate=0.012):
    q_active = q_start[pupil].copy()
    moments = np.zeros_like(q_active)
    variances = np.zeros_like(q_active)
    best_q = q_start.copy()
    best_soft_min = -np.inf
    history = []
    for iteration in range(iterations + 1):
        loss, grad, etas = roi_maximin_gradient(q_active, pupil, gammas, energy, roi, module)
        soft_min = -loss
        if soft_min > best_soft_min:
            best_soft_min = soft_min
            best_q[:] = 0.0
            best_q[pupil] = q_active
        history.append({"iteration": iteration, "soft_min_ROI": float(soft_min), "ROI_by_wavelength": etas.tolist()})
        if iteration == iterations:
            break
        grad_rms = float(np.sqrt(np.mean(grad**2)))
        normalized = grad / max(grad_rms, 1e-30)
        moments = 0.9 * moments + 0.1 * normalized
        variances = 0.999 * variances + 0.001 * normalized**2
        t = iteration + 1
        mhat = moments / (1.0 - 0.9**t)
        vhat = variances / (1.0 - 0.999**t)
        q_active = np.clip(q_active - learning_rate * mhat / (np.sqrt(vhat) + 1e-8), 0.0, MAX_Q)
    return best_q, history


def make_band_sweep(q_by_name, pupil, dx, target, roi, n_ref, module):
    wavelengths_nm = np.arange(400.0, 701.0, 5.0)
    lam_ref = 532e-9
    n_ref_value = float(n_ref)
    # The 532 nm relief phase is the height coordinate; gamma(lambda) is the
    # physical fused-silica optical-path coefficient relative to that reference.
    records = {name: [] for name in q_by_name}
    curves = {name: {"L2": [], "ROI": []} for name in q_by_name}
    height_2pi = 1.1547486e-6
    for wavelength_nm in wavelengths_nm:
        lam = wavelength_nm * 1e-9
        n_lam = float(module.fused_silica_n(lam))
        gamma = ((n_lam - 1.0) / lam) / ((n_ref_value - 1.0) / lam_ref)
        for name, q in q_by_name.items():
            screen = np.zeros(q.shape, dtype=np.complex128)
            screen[pupil] = np.exp(1j * 2.0 * np.pi * gamma * q[pupil])
            image = np.abs(module.centered_fft(screen)) ** 2
            image /= max(float(np.sum(image)), 1e-300)
            m = module.metrics(target, image, roi)
            row = {"wavelength_nm": float(wavelength_nm), **m}
            records[name].append(row)
            curves[name]["L2"].append(m["relative_L2_to_target"])
            curves[name]["ROI"].append(m["power_fraction_in_target_1pct_ROI"])
    return wavelengths_nm, records, curves


def main():
    module = load_step9()
    with np.load(INPUT, allow_pickle=False) as z:
        pupil = z["pupil"].astype(bool)
        target = z["target_normalized"].astype(np.float64)
        roi = target >= 0.01 * np.max(target)
        q_principal = z["q_principal"].astype(np.float64)
        q_shape = z["q_joint_continuous"].astype(np.float64)
        q_soft = z["q_joint_roi_penalized"].astype(np.float64)
        gammas = np.array([
            ((float(module.fused_silica_n(w * 1e-9)) - 1.0) / (w * 1e-9))
            / ((float(module.fused_silica_n(532e-9)) - 1.0) / 532e-9)
            for w in [450.0, 532.0, 633.0]
        ])
        dx = float(z["input_pitch_m"])
    energy = float(pupil.shape[0] ** 2 * np.count_nonzero(pupil))
    target_norm_sq = float(np.sum(target**2))

    rng = np.random.default_rng(SEED)
    starts = {
        "soft-penalty solution": q_soft,
        "shape solution + 0.10-cycle noise": np.where(pupil, np.clip(q_shape + rng.normal(0.0, 0.10, q_shape.shape), 0.0, MAX_Q), 0.0),
        "principal branch + 0.10-cycle noise": np.where(pupil, np.clip(q_principal + rng.normal(0.0, 0.10, q_principal.shape), 0.0, MAX_Q), 0.0),
    }
    restart_results = {}
    histories = {}
    final_multipliers = {}
    candidates = {"principal minimum-height": q_principal, "joint shape only": q_shape, "ROI soft penalty": q_soft}
    for name, q_start in starts.items():
        q, hist, multipliers = augmented_lagrange(q_start, pupil, gammas, target, target_norm_sq, energy, roi, module)
        key = f"augmented Lagrange: {name}"
        candidates[key] = q
        histories[key] = hist
        final_multipliers[key] = multipliers
        _, rows = evaluate(q, pupil, gammas, target, roi, module)
        restart_results[key] = {
            "initialization": name,
            "rows": rows,
            "max_height_um": float(np.max(q[pupil]) * 1.1547486),
            "mean_neighbor_step_um": module.profile_summary(q, pupil, 1.1547486e-6)["height_rms_neighbor_step_um"],
            "minimum_ROI_power": float(min(x["power_fraction_in_target_1pct_ROI"] for x in rows)),
            "minimum_L2": float(min(x["relative_L2_to_target"] for x in rows)),
            "mean_L2": float(np.mean([x["relative_L2_to_target"] for x in rows])),
        }
    roi_only_results = {}
    roi_only_histories = {}
    for name, q_start in {
        "principal branch": q_principal,
        "shape solution": q_shape,
        "ROI soft penalty solution": q_soft,
    }.items():
        q, hist = maximize_minimum_roi(q_start, pupil, gammas, energy, roi, module)
        key = f"ROI-only maximin from {name}"
        candidates[key] = q
        roi_only_histories[key] = hist
        _, rows = evaluate(q, pupil, gammas, target, roi, module)
        roi_only_results[key] = {
            "initialization": name,
            "rows": rows,
            "minimum_ROI_power": float(min(x["power_fraction_in_target_1pct_ROI"] for x in rows)),
            "mean_ROI_power": float(np.mean([x["power_fraction_in_target_1pct_ROI"] for x in rows])),
            "mean_L2": float(np.mean([x["relative_L2_to_target"] for x in rows])),
            "max_height_um": float(np.max(q[pupil]) * 1.1547486),
        }
    band_wavelengths, band_rows, curves = make_band_sweep(
        {k: candidates[k] for k in ["principal minimum-height", "joint shape only", "ROI soft penalty"] + list(restart_results) + list(roi_only_results)},
        pupil, dx, target, roi, float(module.fused_silica_n(532e-9)), module
    )
    sweep_summary = {}
    for name, rows in band_rows.items():
        worst_l2 = max(rows, key=lambda r: r["relative_L2_to_target"])
        worst_roi = min(rows, key=lambda r: r["power_fraction_in_target_1pct_ROI"])
        sweep_summary[name] = {
            "worst_sampled_L2": worst_l2,
            "worst_sampled_ROI": worst_roi,
            "band_mean_L2": float(np.mean([r["relative_L2_to_target"] for r in rows])),
            "band_mean_ROI": float(np.mean([r["power_fraction_in_target_1pct_ROI"] for r in rows])),
        }

    # Plot dense-band robustness and the three design tradeoff anchors.
    fig, axes = plt.subplots(1, 2, figsize=(13.5, 4.8), constrained_layout=True)
    colors = plt.cm.tab10(np.linspace(0.0, 0.9, len(curves)))
    for color, (name, values) in zip(colors, curves.items()):
        axes[0].plot(band_wavelengths, values["L2"], label=name, color=color, lw=1.5)
        axes[1].plot(band_wavelengths, values["ROI"], label=name, color=color, lw=1.5)
    for ax in axes:
        ax.axvline(450, color="0.6", ls=":", lw=0.8)
        ax.axvline(532, color="0.6", ls=":", lw=0.8)
        ax.axvline(633, color="0.6", ls=":", lw=0.8)
        ax.set_xlabel("波长 (nm)")
        ax.grid(alpha=0.25)
    axes[0].set_ylabel("归一化强度相对 L2")
    axes[0].set_title("连续波段离散采样误差")
    axes[1].set_ylabel("目标 ROI 功率比例")
    axes[1].axhline(FLOOR, color="black", ls="--", lw=1.0, label="85% 目标线")
    axes[1].set_title("连续波段离散采样 ROI 功率")
    axes[1].legend(fontsize=7, loc="best")
    fig.savefig(OUT / "fourier_optics_step10_band_robustness.png", dpi=170)
    plt.close(fig)

    np.savez_compressed(
        OUT / "fourier_optics_step10_results.npz",
        band_wavelengths_nm=band_wavelengths,
        band_L2_by_design=np.array([curves[k]["L2"] for k in curves]),
        band_ROI_by_design=np.array([curves[k]["ROI"] for k in curves]),
        design_names=np.array(list(curves), dtype="U64"),
        **{f"q_{i}": candidates[k] for i, k in enumerate(candidates)},
    )
    result = {
        "step": 10,
        "title": "Warm-start variation, augmented-Lagrangian ROI experiment, and 400-700 nm band sweep",
        "status": "three perturbed starts and a 5 nm scalar-model sweep; not a device validation or global optimum",
        "settings": {
            "ROI_floor": FLOOR,
            "augmented_Lagrangian_outer_rounds": OUTER_ROUNDS,
            "Adam_steps_per_outer_round": INNER_STEPS,
            "rho": RHO,
            "height_cycles_bound": [0.0, MAX_Q],
            "band_nm": [400, 700],
            "band_step_nm": 5,
            "wavelengths_nm_for_optimization": [450, 532, 633],
            "random_seed": SEED,
        },
        "warm_start_results": restart_results,
        "ROI_only_maximin_results": roi_only_results,
        "dense_band_summary": sweep_summary,
        "multiplier_history_final": final_multipliers,
        "ROI_only_optimization_history": roi_only_histories,
        "figures": ["fourier_optics_step10_band_robustness.png"],
    }
    (OUT / "fourier_optics_step10_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"warm_start_results": restart_results, "dense_band_summary": sweep_summary}, indent=2))


if __name__ == "__main__":
    main()
