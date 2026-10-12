"""Step 3: phase-only Gerchberg-Saxton intensity holograms.

Run with the configured Miniconda interpreter:
    C:/Users/lenovo/miniconda3/envs/SI100B_Project/python.exe fourier_optics_step3.py

The screen is phase-only inside a finite circular pupil. Results include
multiple deterministic starts, convergence curves, and reconstructed fields.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
N = 1024
DX = 5.0e-6
WAVELENGTH = 532.0e-9
FOCAL_LENGTH = 0.200
ITERATIONS = 150
SEEDS = [260930, 260931, 260932]
EXPORT_BEST_PHASES_ONLY = os.environ.get("FOURIER_OPTICS_EXPORT_PHASES_ONLY") == "1"
BEST_SEED_BY_TARGET = {
    "Three Gaussian spots": 260930,
    "Annular ring": 260931,
    "Binary letter A": 260930,
}
L = N * DX
DF = 1.0 / L
OUTPUT_DX = WAVELENGTH * FOCAL_LENGTH / L


def centered_fft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def centered_ifft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def segment_distance(
    X_mm: np.ndarray,
    Y_mm: np.ndarray,
    a: tuple[float, float],
    b: tuple[float, float],
) -> np.ndarray:
    ax, ay = a
    bx, by = b
    vx, vy = bx - ax, by - ay
    denom = vx * vx + vy * vy
    u = np.clip(((X_mm - ax) * vx + (Y_mm - ay) * vy) / denom, 0.0, 1.0)
    return np.hypot(X_mm - (ax + u * vx), Y_mm - (ay + u * vy))


def normalized_metrics(
    intensity_target: np.ndarray,
    intensity_reconstructed: np.ndarray,
    roi: np.ndarray,
) -> dict[str, float | None]:
    p_target = intensity_target / max(float(np.sum(intensity_target)), 1e-300)
    p_reconstructed = intensity_reconstructed / max(
        float(np.sum(intensity_reconstructed)), 1e-300
    )
    difference = p_reconstructed - p_target
    target_l2 = max(float(np.linalg.norm(p_target.ravel())), 1e-300)
    fidelity = float(np.sum(np.sqrt(p_target * p_reconstructed)) ** 2)
    def correlation(a: np.ndarray, b: np.ndarray) -> float | None:
        a_flat = a.ravel()
        b_flat = b.ravel()
        if float(np.std(a_flat)) <= 1e-15 or float(np.std(b_flat)) <= 1e-15:
            return None
        return float(np.corrcoef(a_flat, b_flat)[0, 1])

    corr_global = correlation(p_target, p_reconstructed)
    p_t_roi = p_target[roi]
    p_r_roi = p_reconstructed[roi]
    corr_roi = correlation(p_t_roi, p_r_roi)
    return {
        "normalized_intensity_relative_L2": float(
            np.linalg.norm(difference.ravel()) / target_l2
        ),
        "normalized_intensity_RMSE_per_pixel": float(
            np.sqrt(np.mean(difference**2))
        ),
        "intensity_correlation_global": corr_global,
        "intensity_correlation_signal_ROI": corr_roi,
        "squared_Bhattacharyya_fidelity": fidelity,
        "reconstructed_energy_fraction_in_signal_ROI": float(np.sum(p_reconstructed[roi])),
        "target_energy_fraction_in_signal_ROI": float(np.sum(p_target[roi])),
    }


def convergence_metrics(
    intensity_target: np.ndarray,
    intensity_reconstructed: np.ndarray,
    roi: np.ndarray,
) -> dict[str, float]:
    p_target = intensity_target / max(float(np.sum(intensity_target)), 1e-300)
    p_reconstructed = intensity_reconstructed / max(
        float(np.sum(intensity_reconstructed)), 1e-300
    )
    difference = p_reconstructed - p_target
    target_l2 = max(float(np.linalg.norm(p_target.ravel())), 1e-300)
    return {
        "normalized_intensity_relative_L2": float(
            np.linalg.norm(difference.ravel()) / target_l2
        ),
        "squared_Bhattacharyya_fidelity": float(
            np.sum(np.sqrt(p_target * p_reconstructed)) ** 2
        ),
        "reconstructed_energy_fraction_in_signal_ROI": float(
            np.sum(p_reconstructed[roi])
        ),
    }


def main() -> None:
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=DX))
    X_axis = WAVELENGTH * FOCAL_LENGTH * fx
    x = (np.arange(N) - N // 2) * DX
    X, Y = np.meshgrid(X_axis, X_axis)
    X_mm, Y_mm = X * 1e3, Y * 1e3

    # Circular phase-only pupil, diameter 1.28 mm.
    pupil_radius = 128 * DX
    pupil = (np.hypot(*np.meshgrid(x, x)) <= pupil_radius).astype(float)
    input_power_samples = float(np.sum(pupil**2))

    sigma_spot = 0.12  # mm, intensity standard deviation
    spot_target = np.zeros((N, N), dtype=float)
    for x0, y0, weight in [(-0.72, -0.48, 1.0), (0.72, -0.48, 0.85), (0.0, 0.74, 0.90)]:
        spot_target += weight * np.exp(
            -((X_mm - x0) ** 2 + (Y_mm - y0) ** 2) / (2.0 * sigma_spot**2)
        )
    spot_target /= float(np.max(spot_target))

    ring_radius_mm = 0.78
    ring_sigma_mm = 0.14
    radius_mm = np.hypot(X_mm, Y_mm)
    ring_target = np.exp(
        -((radius_mm - ring_radius_mm) ** 2) / (2.0 * ring_sigma_mm**2)
    )

    line_thickness_mm = 0.16
    letter_mask = (
        (segment_distance(X_mm, Y_mm, (-0.72, -0.78), (0.0, 0.88)) < line_thickness_mm / 2)
        | (segment_distance(X_mm, Y_mm, (0.0, 0.88), (0.72, -0.78)) < line_thickness_mm / 2)
        | (segment_distance(X_mm, Y_mm, (-0.39, -0.12), (0.39, -0.12)) < line_thickness_mm / 2)
    )
    letter_target = letter_mask.astype(float)

    target_specs = [
        ("Three Gaussian spots", spot_target),
        ("Annular ring", ring_target),
        ("Binary letter A", letter_target),
    ]
    summaries: list[dict[str, object]] = []
    final_plot_data: list[dict[str, object]] = []
    phase_archive: dict[str, np.ndarray] = {
        "screen_x_m": x.astype(np.float64),
        "fourier_x_m_at_reference": X_axis.astype(np.float64),
        "pupil": pupil.astype(np.uint8),
        "wavelength_m": np.asarray(WAVELENGTH),
        "focal_length_m": np.asarray(FOCAL_LENGTH),
        "N": np.asarray(N),
        "input_pixel_pitch_m": np.asarray(DX),
        "best_seed_by_target": np.asarray(
            [BEST_SEED_BY_TARGET[name] for name, _ in target_specs], dtype=np.int64
        ),
    }
    target_keys = ["three_spots", "annular_ring", "letter_A"]

    for target_index, (target_name, target_intensity) in enumerate(target_specs):
        target_amplitude = np.sqrt(target_intensity)
        target_amplitude *= N * math.sqrt(
            input_power_samples / max(float(np.sum(target_amplitude**2)), 1e-300)
        )
        roi = target_intensity >= 0.01 * float(np.max(target_intensity))
        seed_runs: list[dict[str, object]] = []
        curves: list[dict[str, np.ndarray]] = []

        seeds_for_target = (
            [BEST_SEED_BY_TARGET[target_name]] if EXPORT_BEST_PHASES_ONLY else SEEDS
        )
        for seed in seeds_for_target:
            rng = np.random.default_rng(seed)
            phase = rng.uniform(-np.pi, np.pi, size=(N, N))
            screen = pupil * np.exp(1j * phase)
            history: dict[str, list[float]] = {
                "iteration": [],
                "normalized_intensity_relative_L2": [],
                "squared_Bhattacharyya_fidelity": [],
                "reconstructed_energy_fraction_in_signal_ROI": [],
            }
            final_F = None

            for iteration in range(ITERATIONS + 1):
                F = centered_fft(screen)
                intensity = np.abs(F) ** 2

                if iteration % 3 == 0 or iteration == ITERATIONS:
                    metric = convergence_metrics(target_intensity, intensity, roi)
                    history["iteration"].append(float(iteration))
                    history["normalized_intensity_relative_L2"].append(
                        metric["normalized_intensity_relative_L2"]
                    )
                    history["squared_Bhattacharyya_fidelity"].append(
                        metric["squared_Bhattacharyya_fidelity"]
                    )
                    history["reconstructed_energy_fraction_in_signal_ROI"].append(
                        metric["reconstructed_energy_fraction_in_signal_ROI"]
                    )

                final_F = F
                if iteration == ITERATIONS:
                    break

                # Target-plane amplitude constraint; retain current phase.
                constrained_F = target_amplitude * np.exp(1j * np.angle(F))
                back = centered_ifft(constrained_F)
                # Screen-plane phase-only constraint inside the circular pupil.
                screen = pupil * np.exp(1j * np.angle(back))

            assert final_F is not None
            final_intensity = np.abs(final_F) ** 2
            final_metrics = normalized_metrics(target_intensity, final_intensity, roi)
            seed_runs.append(
                {
                    "seed": seed,
                    "metrics": final_metrics,
                    "screen_phase": np.angle(screen).astype(np.float32),
                    "field": final_F,
                }
            )
            curves.append({key: np.asarray(values, dtype=float) for key, values in history.items()})

        best_index = int(
            np.argmax([
                run["metrics"]["squared_Bhattacharyya_fidelity"]
                for run in seed_runs
            ])
        )
        best = seed_runs[best_index]
        best_intensity = np.abs(best["field"]) ** 2
        best_power_map = best_intensity / max(float(np.sum(best_intensity)), 1e-300)
        best_phase = best["screen_phase"]
        best_scores = [run["metrics"] for run in seed_runs]
        all_curves = [
            {
                key: value.tolist()
                for key, value in curve.items()
            }
            for curve in curves
        ]

        key = target_keys[target_index]
        phase_archive[f"{key}_target_intensity"] = target_intensity.astype(np.float32)
        phase_archive[f"{key}_reconstructed_normalized_intensity"] = best_power_map.astype(
            np.float32
        )
        phase_archive[f"{key}_phase_rad"] = best_phase.astype(np.float32)
        phase_archive[f"{key}_screen_complex"] = (
            pupil * np.exp(1j * best_phase)
        ).astype(np.complex64)
        phase_archive[f"{key}_best_seed"] = np.asarray(int(best["seed"]))

        summaries.append(
            {
                "target": target_name,
                "phase_only_pupil_diameter_m": 2.0 * pupil_radius,
                "pupil_samples_across": 2 * 128,
                "iterations_per_start": ITERATIONS,
                "random_starts": SEEDS,
                "best_seed": int(best["seed"]),
                "best_final_metrics": best["metrics"],
                "all_seed_final_metrics": [
                    {"seed": seed_runs[i]["seed"], **seed_runs[i]["metrics"]}
                    for i in range(len(seed_runs))
                ],
                "convergence_curves_all_starts": all_curves,
                "phase_levels_before_quantization": "continuous [−π, π)",
                "reconstructed_energy_fraction_in_ROI_across_starts": [
                    float(score["reconstructed_energy_fraction_in_signal_ROI"])
                    for score in best_scores
                ],
            }
        )
        final_plot_data.append(
            {
                "name": target_name,
                "target": target_intensity.astype(np.float32),
                "reconstructed": best_power_map.astype(np.float32),
                "phase": best_phase,
                "curves": curves,
                "best_seed": int(best["seed"]),
            }
        )

    if EXPORT_BEST_PHASES_ONLY:
        archive_path = OUT_DIR / "fourier_optics_step3_phase_screens.npz"
        np.savez_compressed(archive_path, **phase_archive)
        print(json.dumps({"phase_archive": str(archive_path), "targets": target_keys}, ensure_ascii=True))
        return

    # Pattern, reconstruction, and phase-only screen.
    fig, axes = plt.subplots(3, 3, figsize=(14, 13), constrained_layout=True)
    fig.suptitle(
        "Gerchberg–Saxton: intensity target → phase-only screen → Fourier-plane field\n"
        f"λ={WAVELENGTH * 1e9:.0f} nm, f={FOCAL_LENGTH * 1e3:.0f} mm, "
        f"N={N}, dx={DX * 1e6:.1f} µm, circular pupil D={2*pupil_radius*1e3:.2f} mm",
        fontsize=13,
    )
    focus_extent = [
        X_axis[0] * 1e3,
        X_axis[-1] * 1e3,
        X_axis[0] * 1e3,
        X_axis[-1] * 1e3,
    ]
    screen_extent = [x[0] * 1e3, x[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3]
    for row, item in enumerate(final_plot_data):
        axes[row, 0].imshow(
            item["target"], extent=focus_extent, origin="lower", cmap="magma", vmin=0
        )
        axes[row, 0].set_xlim(-1.8, 1.8)
        axes[row, 0].set_ylim(-1.8, 1.8)
        axes[row, 0].set_title(f"{item['name']}: target")
        axes[row, 0].set_xlabel("X (mm)")
        axes[row, 0].set_ylabel("Y (mm)")

        axes[row, 1].imshow(
            item["reconstructed"],
            extent=focus_extent,
            origin="lower",
            cmap="magma",
            vmin=0,
            vmax=max(float(np.max(item["reconstructed"])), 1e-12),
        )
        axes[row, 1].set_xlim(-1.8, 1.8)
        axes[row, 1].set_ylim(-1.8, 1.8)
        axes[row, 1].set_title(f"Reconstruction (best seed {item['best_seed']})")
        axes[row, 1].set_xlabel("X (mm)")
        axes[row, 1].set_ylabel("Y (mm)")

        phase_cmap = plt.get_cmap("twilight").copy()
        phase_cmap.set_bad(color="black")
        phase_display = np.ma.masked_where(pupil == 0, item["phase"])
        axes[row, 2].imshow(
            phase_display,
            extent=screen_extent,
            origin="lower",
            cmap=phase_cmap,
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[row, 2].set_title("Phase-only screen (outside pupil black)")
        axes[row, 2].set_xlabel("x (mm)")
        axes[row, 2].set_ylabel("y (mm)")

    fig.savefig(OUT_DIR / "fourier_optics_step3_phase_only_results.png", dpi=180)
    plt.close(fig)

    # Convergence curves: three metrics for each target, with all starts shown.
    curve_fig, curve_axes = plt.subplots(3, 3, figsize=(15, 11), constrained_layout=True)
    curve_fig.suptitle("GS convergence over 3 deterministic random starts", fontsize=13)
    metric_specs = [
        ("normalized_intensity_relative_L2", "Normalized intensity relative L2", "log"),
        ("squared_Bhattacharyya_fidelity", "Squared Bhattacharyya fidelity", "linear"),
        (
            "reconstructed_energy_fraction_in_signal_ROI",
            "Reconstructed energy in target ROI",
            "linear",
        ),
    ]
    for row, item in enumerate(final_plot_data):
        for col, (key, title, scale) in enumerate(metric_specs):
            ax = curve_axes[row, col]
            for curve in item["curves"]:
                yvals = curve[key]
                if scale == "log":
                    ax.semilogy(curve["iteration"], np.maximum(yvals, 1e-12), alpha=0.75)
                else:
                    ax.plot(curve["iteration"], yvals, alpha=0.75)
            ax.set_title(f"{item['name']}\n{title}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel(title)
            ax.grid(True, alpha=0.25)
    curve_fig.savefig(OUT_DIR / "fourier_optics_step3_convergence.png", dpi=170)
    plt.close(curve_fig)

    results = {
        "runtime": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "matplotlib": matplotlib.__version__,
        },
        "model": {
            "wavelength_m": WAVELENGTH,
            "focal_length_m": FOCAL_LENGTH,
            "N": N,
            "input_pixel_pitch_m": DX,
            "screen_width_m": L,
            "output_pixel_pitch_m": OUTPUT_DX,
            "output_frequency_spacing_per_m": DF,
            "circular_pupil_diameter_m": 2.0 * pupil_radius,
            "phase_only_constraint": "|t|=1 inside circular pupil; t=0 outside",
            "target_amplitude_constraint": "sqrt(I_target), globally scaled to match screen power",
            "iterations_per_start": ITERATIONS,
            "seeds": SEEDS,
            "target_ROI_definition": "I_target >= 0.01 * max(I_target)",
        },
        "targets": summaries,
        "figures": [
            "fourier_optics_step3_phase_only_results.png",
            "fourier_optics_step3_convergence.png",
        ],
    }
    with (OUT_DIR / "fourier_optics_step3_results.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    # Keep the Windows console output safe under legacy code pages such as GBK.
    print(json.dumps(results, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
