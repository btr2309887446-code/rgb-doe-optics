"""Step 7: a shared phase-only screen for three ordinary-lens wavelengths.

Uses a Gerchberg-Saxton-style amplitude-consensus projection. At each DFT
frequency bin, the consensus target amplitude is the least-squares mean of the
three wavelength-specific target amplitudes sampled on their natural focal
plane grids. This is a practical heuristic for a generally inconsistent set
of constraints, not a proof of a global optimum.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import map_coordinates


OUT_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = OUT_DIR / "fourier_optics_step3_phase_screens.npz"
WAVELENGTHS = [450.0e-9, 532.0e-9, 633.0e-9]
WAVELENGTH0 = 532.0e-9
FOCAL_LENGTH = 0.200
ITERATIONS = 180
SEED = 260931


def centered_fft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def centered_ifft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def target_intensity(X_mm: np.ndarray, Y_mm: np.ndarray) -> np.ndarray:
    sigma = 0.12
    result = np.zeros_like(X_mm, dtype=np.float64)
    for x0, y0, weight in [(-0.72, -0.48, 1.0), (0.72, -0.48, 0.85), (0.0, 0.74, 0.90)]:
        result += weight * np.exp(
            -((X_mm - x0) ** 2 + (Y_mm - y0) ** 2) / (2.0 * sigma**2)
        )
    return result / float(np.max(result))


def metrics(target: np.ndarray, reconstruction: np.ndarray) -> dict[str, float]:
    pt = target / max(float(np.sum(target)), 1e-300)
    pr = reconstruction / max(float(np.sum(reconstruction)), 1e-300)
    d = pr - pt
    return {
        "normalized_intensity_relative_L2": float(
            np.linalg.norm(d.ravel()) / max(float(np.linalg.norm(pt.ravel())), 1e-300)
        ),
        "normalized_intensity_RMSE_per_pixel": float(np.sqrt(np.mean(d**2))),
        "intensity_correlation_global": float(np.corrcoef(pt.ravel(), pr.ravel())[0, 1]),
        "squared_Bhattacharyya_fidelity": float(np.sum(np.sqrt(pt * pr)) ** 2),
    }


def resample_complex(
    field: np.ndarray,
    frequency_axis: np.ndarray,
    query_fx: np.ndarray,
    query_fy: np.ndarray,
) -> np.ndarray:
    delta = float(frequency_axis[1] - frequency_axis[0])
    ix = (query_fx - float(frequency_axis[0])) / delta
    iy = (query_fy - float(frequency_axis[0])) / delta
    coords = np.stack((iy.ravel(), ix.ravel()))
    real = map_coordinates(field.real, coords, order=3, mode="constant", cval=0.0).reshape(query_fx.shape)
    imag = map_coordinates(field.imag, coords, order=3, mode="constant", cval=0.0).reshape(query_fx.shape)
    return real + 1j * imag


def common_grid_outputs(
    screen: np.ndarray,
    wavelength: float,
    fx: np.ndarray,
    reference_frequency_grid: np.ndarray,
) -> np.ndarray:
    spectrum = centered_fft(screen)
    scale = WAVELENGTH0 / wavelength
    qx = reference_frequency_grid * scale
    qy = reference_frequency_grid * scale
    QX, QY = np.meshgrid(qx, qy)
    sampled = resample_complex(spectrum, fx, QX, QY)
    return np.abs(sampled) ** 2


def main() -> None:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Missing Step 3 phase archive: {ARCHIVE_PATH}")
    with np.load(ARCHIVE_PATH, allow_pickle=False) as archive:
        pupil = archive["pupil"].astype(bool)
        baseline_screen = archive["three_spots_screen_complex"].astype(np.complex128)
        baseline_target = archive["three_spots_target_intensity"].astype(np.float64)
        dx = float(archive["input_pixel_pitch_m"])
        n = int(baseline_screen.shape[0])

    fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    L = n * dx
    pupil_power = float(np.sum(pupil))
    target_amplitudes = []
    targets_on_native_grids = []
    natural_axes = []
    for wavelength in WAVELENGTHS:
        axis_m = wavelength * FOCAL_LENGTH * fx
        X, Y = np.meshgrid(axis_m * 1e3, axis_m * 1e3)
        target = target_intensity(X, Y)
        target_norm = target / float(np.sum(target))
        amplitude = n * np.sqrt(pupil_power * target_norm)
        targets_on_native_grids.append(target_norm)
        target_amplitudes.append(amplitude)
        natural_axes.append(axis_m)

    consensus_amplitude = np.mean(np.stack(target_amplitudes, axis=0), axis=0)
    amplitude_spread = {}
    for i in range(len(WAVELENGTHS)):
        for j in range(i + 1, len(WAVELENGTHS)):
            a, b = target_amplitudes[i], target_amplitudes[j]
            mean_ab = (a + b) / 2.0
            amplitude_spread[f"{WAVELENGTHS[i]*1e9:.0f}nm_vs_{WAVELENGTHS[j]*1e9:.0f}nm"] = float(
                np.linalg.norm((a - b).ravel()) / max(float(np.linalg.norm(mean_ab.ravel())), 1e-300)
            )

    rng = np.random.default_rng(SEED)
    phase = rng.uniform(-np.pi, np.pi, size=(n, n))
    screen = pupil * np.exp(1j * phase)
    history = {"iteration": [], "mean_native_grid_relative_L2": []}
    for iteration in range(ITERATIONS + 1):
        spectrum = centered_fft(screen)
        current_intensity = np.abs(spectrum) ** 2
        native_errors = []
        normalized_current = current_intensity / max(float(np.sum(current_intensity)), 1e-300)
        for target_norm in targets_on_native_grids:
            d = normalized_current - target_norm
            native_errors.append(
                float(np.linalg.norm(d.ravel()) / max(float(np.linalg.norm(target_norm.ravel())), 1e-300))
            )
        if iteration % 3 == 0 or iteration == ITERATIONS:
            history["iteration"].append(iteration)
            history["mean_native_grid_relative_L2"].append(float(np.mean(native_errors)))

        if iteration == ITERATIONS:
            break
        constrained_spectrum = consensus_amplitude * np.exp(1j * np.angle(spectrum))
        screen_plane = centered_ifft(constrained_spectrum)
        screen = pupil * np.exp(1j * np.angle(screen_plane))

    joint_spectrum = centered_fft(screen)
    joint_intensity = np.abs(joint_spectrum) ** 2
    frequency_ref = fx
    reference_axis = WAVELENGTH0 * FOCAL_LENGTH * frequency_ref
    target_common = baseline_target / float(np.sum(baseline_target))
    joint_images_common = {}
    baseline_images_common = {}
    wavelength_results = []
    for i, wavelength in enumerate(WAVELENGTHS):
        native_metrics = metrics(targets_on_native_grids[i], joint_intensity)
        common_joint_intensity = common_grid_outputs(screen, wavelength, fx, frequency_ref)
        common_baseline_intensity = common_grid_outputs(baseline_screen, wavelength, fx, frequency_ref)
        common_joint_images = common_joint_intensity / max(float(np.sum(common_joint_intensity)), 1e-300)
        common_baseline_images = common_baseline_intensity / max(float(np.sum(common_baseline_intensity)), 1e-300)
        common_metrics_joint = metrics(target_common, common_joint_images)
        common_metrics_baseline = metrics(target_common, common_baseline_images)
        wavelength_key = f"{int(round(wavelength*1e9))}nm"
        joint_images_common[wavelength_key] = common_joint_images.astype(np.float32)
        baseline_images_common[wavelength_key] = common_baseline_images.astype(np.float32)
        wavelength_results.append(
            {
                "wavelength_nm": wavelength * 1e9,
                "natural_focal_plane_pixel_pitch_m": wavelength * FOCAL_LENGTH / L,
                "target_amplitude_constraint_spread_to_532nm": (
                    0.0
                    if int(round(wavelength * 1e9)) == 532
                    else amplitude_spread[
                        "450nm_vs_532nm" if wavelength < WAVELENGTH0 else "532nm_vs_633nm"
                    ]
                ),
                "joint_screen_native_grid_metrics": native_metrics,
                "joint_screen_common_532nm_physical_grid_metrics": common_metrics_joint,
                "single_532nm_screen_common_grid_metrics": common_metrics_baseline,
            }
        )

    common_l2_joint = [
        result["joint_screen_common_532nm_physical_grid_metrics"]["normalized_intensity_relative_L2"]
        for result in wavelength_results
    ]
    common_l2_baseline = [
        result["single_532nm_screen_common_grid_metrics"]["normalized_intensity_relative_L2"]
        for result in wavelength_results
    ]
    results = {
        "method": "GS-style shared-spectrum amplitude consensus: arithmetic mean of the three normalized target amplitudes on each wavelength's natural FFT plane, then phase-only pupil projection",
        "status": "preliminary heuristic; not guaranteed to minimize the reported common-grid error or reach a global optimum",
        "model": {
            "wavelengths_m": WAVELENGTHS,
            "focal_length_m": FOCAL_LENGTH,
            "screen_assumed_wavelength_independent": True,
            "screen_pitch_m": dx,
            "screen_width_m": L,
            "pupil_samples": int(pupil_power),
            "iterations": ITERATIONS,
            "random_seed": SEED,
            "output_comparison_grid": "common physical coordinate grid of the 532 nm Fourier plane; spectra are cubic-interpolated",
            "natural_grid_note": "at each wavelength the same DFT bins map to different physical focal-plane coordinates for an ordinary fixed-f lens",
        },
        "pairwise_target_amplitude_constraint_spread": amplitude_spread,
        "single_screen_vs_joint_screen_common_grid_relative_L2": {
            "single_532nm_screen": common_l2_baseline,
            "joint_three_wavelength_screen": common_l2_joint,
            "mean_single_screen_error": float(np.mean(common_l2_baseline)),
            "mean_joint_screen_error": float(np.mean(common_l2_joint)),
            "worst_single_screen_error": float(np.max(common_l2_baseline)),
            "worst_joint_screen_error": float(np.max(common_l2_joint)),
        },
        "wavelength_results": wavelength_results,
        "convergence": history,
        "figures": [
            "fourier_optics_step7_joint_multicolor.png",
            "fourier_optics_step7_convergence.png",
        ],
    }
    result_path = OUT_DIR / "fourier_optics_step7_results.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)
    np.savez_compressed(
        OUT_DIR / "fourier_optics_step7_joint_phase_screen.npz",
        phase_rad=np.angle(screen).astype(np.float32),
        screen_complex=screen.astype(np.complex64),
        pupil=pupil.astype(np.uint8),
        wavelengths_m=np.asarray(WAVELENGTHS),
        focal_length_m=np.asarray(FOCAL_LENGTH),
        pixel_pitch_m=np.asarray(dx),
        natural_output_axes_m=np.asarray(natural_axes),
        target_amplitudes=np.asarray(target_amplitudes, dtype=np.float32),
    )

    x_ref_mm = reference_axis * 1e3
    extent = [x_ref_mm[0], x_ref_mm[-1], x_ref_mm[0], x_ref_mm[-1]]
    fig, axes = plt.subplots(3, 3, figsize=(14, 12), constrained_layout=True)
    fig.suptitle("Same physical target at 450, 532, and 633 nm with an ordinary fixed-f lens", fontsize=14)
    for col, wavelength in enumerate(WAVELENGTHS):
        axes[0, col].imshow(
            np.log10(np.maximum(target_common / np.max(target_common), 1e-7)),
            extent=extent, origin="lower", cmap="magma", vmin=-7, vmax=0,
        )
        axes[0, col].set_title(f"Target — {wavelength*1e9:.0f} nm")
        key = f"{int(round(wavelength*1e9))}nm"
        axes[1, col].imshow(
            np.log10(np.maximum(baseline_images_common[key] / np.max(baseline_images_common[key]), 1e-7)),
            extent=extent, origin="lower", cmap="magma", vmin=-7, vmax=0,
        )
        axes[1, col].set_title(f"Single-color screen — {wavelength*1e9:.0f} nm")
        axes[2, col].imshow(
            np.log10(np.maximum(joint_images_common[key] / np.max(joint_images_common[key]), 1e-7)),
            extent=extent, origin="lower", cmap="magma", vmin=-7, vmax=0,
        )
        axes[2, col].set_title(f"Joint screen — {wavelength*1e9:.0f} nm")
    for ax in axes.ravel():
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    fig.savefig(OUT_DIR / "fourier_optics_step7_joint_multicolor.png", dpi=170)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    ax.plot(history["iteration"], history["mean_native_grid_relative_L2"], "o-", markersize=3)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Mean native-grid normalized intensity relative L2")
    ax.set_title("Shared-spectrum amplitude-consensus iteration")
    ax.grid(True, alpha=0.25)
    fig.savefig(OUT_DIR / "fourier_optics_step7_convergence.png", dpi=160)
    plt.close(fig)

    print(json.dumps({"results": str(result_path), "iterations": ITERATIONS}, ensure_ascii=True))


if __name__ == "__main__":
    main()
