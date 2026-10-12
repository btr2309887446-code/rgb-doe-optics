"""Step 8: fused-silica relief phase dispersion for the Step 3 phase screen.

The relief is chosen from the wrapped 532 nm phase using the minimum-height
principal branch. A prescribed ideal Fourier lens with lambda*f constant
holds the output physical coordinate grid fixed, isolating the screen's
material dispersion from focal-plane magnification.
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
WAVELENGTHS_NM = [450.0, 532.0, 633.0]
SWEEP_NM = np.arange(400.0, 700.0 + 0.1, 5.0)

# I. H. Malitson, JOSA 55, 1205-1209 (1965), wavelength in micrometres.
SELLMEIER_B = np.array([0.6961663, 0.4079426, 0.8974794])
SELLMEIER_C_UM = np.array([0.0684043, 0.1162414, 9.896161])


def fused_silica_n(wavelength_m: float | np.ndarray) -> np.ndarray:
    wavelength_um = np.asarray(wavelength_m, dtype=np.float64) * 1e6
    lam2 = wavelength_um[..., None] ** 2
    b = SELLMEIER_B
    c2 = SELLMEIER_C_UM**2
    n2 = 1.0 + np.sum(b * lam2 / (lam2 - c2), axis=-1)
    return np.sqrt(n2)


def centered_fft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def normalized_intensity(field: np.ndarray) -> np.ndarray:
    intensity = np.abs(field) ** 2
    total = float(np.sum(intensity))
    return intensity / max(total, 1e-300)


def distribution_metrics(target: np.ndarray, image: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    t = target / max(float(np.sum(target)), 1e-300)
    r = image / max(float(np.sum(image)), 1e-300)
    d = r - t
    return {
        "normalized_intensity_relative_L2_to_target": float(
            np.linalg.norm(d.ravel()) / max(float(np.linalg.norm(t.ravel())), 1e-300)
        ),
        "global_intensity_correlation_to_target": float(np.corrcoef(t.ravel(), r.ravel())[0, 1]),
        "squared_Bhattacharyya_fidelity_to_target": float(np.sum(np.sqrt(t * r)) ** 2),
        "power_fraction_in_target_1pct_ROI": float(np.sum(r[roi])),
    }


def relative_l2(a: np.ndarray, b: np.ndarray) -> float:
    aa = a / max(float(np.sum(a)), 1e-300)
    bb = b / max(float(np.sum(b)), 1e-300)
    return float(np.linalg.norm((aa - bb).ravel()) / max(float(np.linalg.norm(bb.ravel())), 1e-300))


def make_figure_phase_transfer(wavelengths_m: np.ndarray, n_values: np.ndarray, gamma: np.ndarray) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(wavelengths_m * 1e9, n_values, color="#315a9b", lw=2)
    axes[0].scatter(np.asarray(WAVELENGTHS_NM), fused_silica_n(np.asarray(WAVELENGTHS_NM) * 1e-9),
                    color="#c04b35", zorder=3)
    axes[0].set(xlabel="Wavelength (nm)", ylabel="Refractive index n", title="Fused-silica Sellmeier model")
    axes[0].grid(alpha=0.25)

    phi_cycles = np.linspace(0.0, 1.0, 401)
    axes[1].plot(phi_cycles, phi_cycles, "k--", label="wavelength-invariant phase")
    for wavelength_nm in WAVELENGTHS_NM:
        lam = wavelength_nm * 1e-9
        g = float(((fused_silica_n(lam) - 1.0) / lam) / ((fused_silica_n(LAMBDA0_M) - 1.0) / LAMBDA0_M))
        axes[1].plot(phi_cycles, g * phi_cycles, label=f"{wavelength_nm:.0f} nm (×{g:.4f})")
    axes[1].set(xlabel="wrapped 532 nm phase / 2π", ylabel="relief phase / 2π",
                title="Phase response of minimum-height relief")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False, fontsize=8)
    fig.savefig(OUT_DIR / "fourier_optics_step8_phase_transfer.png", dpi=180)
    plt.close(fig)


def main() -> None:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Missing Step 3 archive: {ARCHIVE_PATH}")
    with np.load(ARCHIVE_PATH, allow_pickle=False) as archive:
        pupil = archive["pupil"].astype(bool)
        phase_532 = archive["three_spots_phase_rad"].astype(np.float64)
        target = archive["three_spots_target_intensity"].astype(np.float64)
        dx = float(archive["input_pixel_pitch_m"])
        n_grid = int(archive["N"])
        archive_lambda0 = float(archive["wavelength_m"])
        archive_f0 = float(archive["focal_length_m"])

    if abs(archive_lambda0 - LAMBDA0_M) > 1e-15 or abs(archive_f0 - F0_M) > 1e-12:
        raise ValueError("Step 3 reference wavelength/focal length differs from Step 8 assumptions")
    if phase_532.shape != (n_grid, n_grid) or pupil.shape != phase_532.shape:
        raise ValueError("Unexpected Step 3 array shape")

    phi_wrapped = np.mod(phase_532, 2.0 * np.pi)
    n0 = float(fused_silica_n(LAMBDA0_M))
    height_2pi = LAMBDA0_M / (n0 - 1.0)
    height_m = np.where(pupil, phi_wrapped * height_2pi / (2.0 * np.pi), 0.0)

    target_norm = target / max(float(np.sum(target)), 1e-300)
    roi = target >= 0.01 * float(np.max(target))
    frequency = np.fft.fftshift(np.fft.fftfreq(n_grid, d=dx))
    output_axis_m = LAMBDA0_M * F0_M * frequency

    screen_ideal = pupil * np.exp(1j * phi_wrapped)
    ideal_image = normalized_intensity(centered_fft(screen_ideal))
    ideal_metrics = distribution_metrics(target_norm, ideal_image, roi)

    spectrum_at_samples: dict[str, np.ndarray] = {}
    wavelength_results = []
    n_sample_values = []
    for wavelength_nm in WAVELENGTHS_NM:
        wavelength_m = wavelength_nm * 1e-9
        n_lambda = float(fused_silica_n(wavelength_m))
        gamma = ((n_lambda - 1.0) / wavelength_m) / ((n0 - 1.0) / LAMBDA0_M)
        phase_lambda = 2.0 * np.pi * (n_lambda - 1.0) * height_m / wavelength_m
        screen_relief = pupil * np.exp(1j * phase_lambda)
        image = normalized_intensity(centered_fft(screen_relief))
        phase_error = np.exp(1j * phase_lambda[pupil]) - np.exp(1j * phi_wrapped[pupil])
        metrics = distribution_metrics(target_norm, image, roi)
        metrics["relative_L2_to_532nm_ideal_screen_image"] = relative_l2(image, ideal_image)
        metrics["pupil_phasor_RMS_difference_from_532nm_screen"] = float(np.sqrt(np.mean(np.abs(phase_error) ** 2)))
        metrics["phase_scale_gamma"] = float(gamma)
        metrics["n_fused_silica"] = n_lambda
        metrics["ideal_compensating_focal_length_m"] = F0_M * LAMBDA0_M / wavelength_m
        wavelength_results.append({"wavelength_nm": wavelength_nm, **metrics})
        spectrum_at_samples[f"{int(wavelength_nm)}nm"] = image.astype(np.float32)
        n_sample_values.append(n_lambda)

    sweep_n = fused_silica_n(SWEEP_NM * 1e-9)
    sweep_gamma = ((sweep_n - 1.0) / (SWEEP_NM * 1e-9)) / ((n0 - 1.0) / LAMBDA0_M)
    sweep_target_error = []
    sweep_drift = []
    sweep_phase_rms = []
    for wavelength_nm, n_lambda in zip(SWEEP_NM, sweep_n):
        wavelength_m = float(wavelength_nm * 1e-9)
        phase_lambda = 2.0 * np.pi * (float(n_lambda) - 1.0) * height_m / wavelength_m
        screen = pupil * np.exp(1j * phase_lambda)
        image = normalized_intensity(centered_fft(screen))
        sweep_target_error.append(distribution_metrics(target_norm, image, roi)["normalized_intensity_relative_L2_to_target"])
        sweep_drift.append(relative_l2(image, ideal_image))
        delta = np.exp(1j * phase_lambda[pupil]) - np.exp(1j * phi_wrapped[pupil])
        sweep_phase_rms.append(float(np.sqrt(np.mean(np.abs(delta) ** 2))))

    make_figure_phase_transfer(SWEEP_NM * 1e-9, sweep_n, sweep_gamma)

    # Show the intended target, wavelength-invariant ideal screen, and material relief.
    keys = ["450nm", "532nm", "633nm"]
    fig, axes = plt.subplots(2, 3, figsize=(11, 7.4), constrained_layout=True)
    images = [target_norm, ideal_image] + [spectrum_at_samples[key] for key in keys]
    titles = ["Target", "Ideal invariant screen"] + [f"Fused-silica relief\n{key[:-2]} nm" for key in keys]
    # Plot normalized-to-peak logarithmic intensities on a common physical grid.
    extent_mm = [output_axis_m[0] * 1e3, output_axis_m[-1] * 1e3,
                 output_axis_m[0] * 1e3, output_axis_m[-1] * 1e3]
    crop = np.abs(output_axis_m * 1e3) <= 2.0
    sl = np.flatnonzero(crop)
    lo, hi = int(sl[0]), int(sl[-1]) + 1
    crop_extent = [output_axis_m[lo] * 1e3, output_axis_m[hi - 1] * 1e3,
                   output_axis_m[lo] * 1e3, output_axis_m[hi - 1] * 1e3]
    for i, (ax, image, title) in enumerate(zip(axes.ravel(), images, titles)):
        small = image[lo:hi, lo:hi]
        ax.imshow(np.log10(np.maximum(small / max(float(np.max(small)), 1e-30), 1e-7)),
                  extent=crop_extent, origin="lower", cmap="magma", vmin=-7, vmax=0)
        ax.set_title(title)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    axes.ravel()[-1].axis("off")
    fig.suptitle("Same output grid using ideal λf compensation", fontsize=13)
    fig.savefig(OUT_DIR / "fourier_optics_step8_relief_patterns.png", dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), constrained_layout=True)
    axes[0].plot(SWEEP_NM, sweep_target_error, color="#b54136", lw=2, label="relief vs target")
    axes[0].axhline(ideal_metrics["normalized_intensity_relative_L2_to_target"], color="#315a9b",
                    ls="--", label="ideal screen baseline")
    axes[0].set(xlabel="Wavelength (nm)", ylabel="Normalized intensity relative L2",
                title="Target error over 400–700 nm")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].plot(SWEEP_NM, sweep_drift, color="#7a4d9b", lw=2, label="image drift from ideal")
    axes[1].plot(SWEEP_NM, sweep_phase_rms, color="#228477", lw=1.8, label="pupil phasor RMS change")
    axes[1].set(xlabel="Wavelength (nm)", ylabel="Relative L2 / RMS phasor difference",
                title="Dispersion-induced change")
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)
    fig.savefig(OUT_DIR / "fourier_optics_step8_band_sweep.png", dpi=180)
    plt.close(fig)

    n_values = [float(fused_silica_n(w * 1e-9)) for w in WAVELENGTHS_NM]
    result = {
        "step": 8,
        "title": "Fused-silica relief phase dispersion with prescribed ideal Fourier-lens compensation",
        "status": "scalar thin-relief preliminary model; dense wavelength sweep is sampled at 5 nm, not a proof for every wavelength",
        "source_phase_screen": "Step 3 three-spot phase-only screen, seed 260930",
        "material_model": {
            "material": "fused silica in air",
            "dispersion": "Malitson 3-term Sellmeier equation",
            "sellmeier_B": SELLMEIER_B.tolist(),
            "sellmeier_C_um": SELLMEIER_C_UM.tolist(),
            "validity_range_um": [0.21, 3.71],
            "reference_temperature_C": 20,
            "reference_wavelength_nm": LAMBDA0_M * 1e9,
            "reference_index": n0,
            "reference_2pi_relief_height_um": height_2pi * 1e6,
        },
        "device_and_propagation_model": {
            "relief_height_rule": "h = mod(phi_532, 2pi) * lambda0/[2pi*(n(lambda0)-1)] within the pupil; zero outside",
            "relief_transmission": "unit-amplitude exp(i*2pi*[n(lambda)-1]*h/lambda); Fresnel reflection, absorption, substrate propagation, and relief scattering omitted",
            "lens": "prescribed ideal thin-lens Fourier operator with f(lambda)=f0*lambda0/lambda; used only to keep lambda*f and physical output coordinates fixed",
            "input_grid_N": n_grid,
            "input_pixel_pitch_um": dx * 1e6,
            "input_width_mm": n_grid * dx * 1e3,
            "pupil_samples": int(np.sum(pupil)),
            "comparison_grid": "fixed 532 nm focal-plane physical grid; same DFT bins because lambda*f is constant",
            "target_roi": "target intensity >= 1 percent of its maximum",
            "phase_wrapping_branch": "principal minimum-height branch, 0 <= h < h_2pi",
        },
        "ideal_wavelength_invariant_screen_baseline": ideal_metrics,
        "sampled_wavelength_results": wavelength_results,
        "band_sweep": {
            "wavelength_nm": SWEEP_NM.tolist(),
            "sellmeier_index": [float(x) for x in sweep_n],
            "phase_scale_gamma": [float(x) for x in sweep_gamma],
            "relative_L2_to_target": [float(x) for x in sweep_target_error],
            "relative_L2_drift_from_ideal_image": [float(x) for x in sweep_drift],
            "pupil_phasor_RMS_difference_from_reference": [float(x) for x in sweep_phase_rms],
            "worst_sampled_target_error": float(np.max(sweep_target_error)),
            "worst_sampled_drift_from_ideal": float(np.max(sweep_drift)),
            "worst_sampled_wavelength_nm_for_target_error": float(SWEEP_NM[int(np.argmax(sweep_target_error))]),
        },
        "figures": [
            "fourier_optics_step8_phase_transfer.png",
            "fourier_optics_step8_relief_patterns.png",
            "fourier_optics_step8_band_sweep.png",
        ],
    }
    with (OUT_DIR / "fourier_optics_step8_results.json").open("w", encoding="utf-8") as stream:
        json.dump(result, stream, indent=2, ensure_ascii=False)
    np.savez_compressed(
        OUT_DIR / "fourier_optics_step8_relief_simulation.npz",
        wavelength_nm=SWEEP_NM.astype(np.float32),
        input_phase_reference_rad=phi_wrapped.astype(np.float32),
        relief_height_m=height_m.astype(np.float32),
        output_axis_m=output_axis_m.astype(np.float64),
        ideal_invariant_normalized_intensity=ideal_image.astype(np.float32),
        target_normalized_intensity=target_norm.astype(np.float32),
        sampled_relief_normalized_intensity=np.stack([spectrum_at_samples[k] for k in keys]),
        sweep_target_relative_L2=np.asarray(sweep_target_error, dtype=np.float32),
        sweep_ideal_drift_relative_L2=np.asarray(sweep_drift, dtype=np.float32),
    )


if __name__ == "__main__":
    main()
