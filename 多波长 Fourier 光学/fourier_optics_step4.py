"""Step 4: fixed-screen multiwavelength Fourier-plane comparison.

The phase-only screen is the saved best three-spot design from Step 3.
The propagation model treats the screen transmission as wavelength-independent
and compares an ordinary fixed-focal-length lens with an idealized
f(lambda) = f0 * lambda0 / lambda Fourier operator.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
from scipy.ndimage import map_coordinates


OUT_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = OUT_DIR / "fourier_optics_step3_phase_screens.npz"
WAVELENGTHS_M = [450e-9, 532e-9, 633e-9]
REFERENCE_WAVELENGTH_M = 532e-9
REFERENCE_FOCAL_LENGTH_M = 0.200
PAD_FACTOR = 2


def centered_fft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def resample_complex_spectrum(
    spectrum: np.ndarray,
    frequency_axis: np.ndarray,
    query_fx: np.ndarray,
    query_fy: np.ndarray,
) -> np.ndarray:
    step = float(frequency_axis[1] - frequency_axis[0])
    ix = (query_fx - float(frequency_axis[0])) / step
    iy = (query_fy - float(frequency_axis[0])) / step
    coords = np.stack((iy.ravel(), ix.ravel()))
    real = map_coordinates(
        spectrum.real, coords, order=3, mode="constant", cval=0.0, prefilter=True
    ).reshape(query_fx.shape)
    imag = map_coordinates(
        spectrum.imag, coords, order=3, mode="constant", cval=0.0, prefilter=True
    ).reshape(query_fx.shape)
    return real + 1j * imag


def resample_intensity(
    intensity: np.ndarray,
    frequency_axis: np.ndarray,
    query_fx: np.ndarray,
    query_fy: np.ndarray,
) -> np.ndarray:
    step = float(frequency_axis[1] - frequency_axis[0])
    ix = (query_fx - float(frequency_axis[0])) / step
    iy = (query_fy - float(frequency_axis[0])) / step
    coords = np.stack((iy.ravel(), ix.ravel()))
    return map_coordinates(
        intensity, coords, order=3, mode="constant", cval=0.0, prefilter=True
    ).reshape(query_fx.shape)


def normalized_metrics(
    target: np.ndarray, reconstructed: np.ndarray, roi: np.ndarray
) -> dict[str, float]:
    p_target = target / max(float(np.sum(target)), 1e-300)
    p_reconstructed = reconstructed / max(float(np.sum(reconstructed)), 1e-300)
    difference = p_reconstructed - p_target
    relative_l2 = float(
        np.linalg.norm(difference.ravel())
        / max(float(np.linalg.norm(p_target.ravel())), 1e-300)
    )
    fidelity = float(np.sum(np.sqrt(p_target * p_reconstructed)) ** 2)
    correlation = float(np.corrcoef(p_target.ravel(), p_reconstructed.ravel())[0, 1])
    return {
        "normalized_intensity_relative_L2": relative_l2,
        "normalized_intensity_RMSE_per_pixel": float(np.sqrt(np.mean(difference**2))),
        "intensity_correlation_global": correlation,
        "squared_Bhattacharyya_fidelity": fidelity,
        "reconstructed_energy_fraction_in_signal_ROI": float(
            np.sum(p_reconstructed[roi])
        ),
        "target_energy_fraction_in_signal_ROI": float(np.sum(p_target[roi])),
    }


def peak_position(
    intensity: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    expected_xy_mm: tuple[float, float],
    half_width_mm: float = 0.24,
) -> tuple[float, float]:
    ex, ey = expected_xy_mm
    x_indices = np.flatnonzero(np.abs(x_mm - ex) <= half_width_mm)
    y_indices = np.flatnonzero(np.abs(y_mm - ey) <= half_width_mm)
    patch = intensity[np.ix_(y_indices, x_indices)]
    iy, ix = np.unravel_index(int(np.argmax(patch)), patch.shape)
    return float(x_mm[x_indices[ix]]), float(y_mm[y_indices[iy]])


def main() -> None:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(
            f"Missing {ARCHIVE_PATH}; run fourier_optics_step3.py with "
            "FOURIER_OPTICS_EXPORT_PHASES_ONLY=1 first."
        )

    with np.load(ARCHIVE_PATH, allow_pickle=False) as archive:
        screen = archive["three_spots_screen_complex"].astype(np.complex128)
        target = archive["three_spots_target_intensity"].astype(np.float64)
        pupil = archive["pupil"].astype(bool)
        x_m = archive["screen_x_m"].astype(np.float64)
        reference_X_m = archive["fourier_x_m_at_reference"].astype(np.float64)
        wavelength0 = float(archive["wavelength_m"])
        focal0 = float(archive["focal_length_m"])
        dx = float(archive["input_pixel_pitch_m"])

    n = int(screen.shape[0])
    width_m = n * dx
    m = n * PAD_FACTOR
    padded = np.zeros((m, m), dtype=np.complex128)
    start = (m - n) // 2
    padded[start : start + n, start : start + n] = screen
    frequency_dense = np.fft.fftshift(np.fft.fftfreq(m, d=dx))
    spectrum_dense = centered_fft(padded) * dx**2
    intensity_dense = np.abs(spectrum_dense) ** 2

    frequency_ref = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    query_x_ref, query_y_ref = np.meshgrid(frequency_ref, frequency_ref)
    X_ref, Y_ref = np.meshgrid(reference_X_m, reference_X_m)
    target_normalized = target / max(float(np.sum(target)), 1e-300)
    roi = target >= 0.01 * float(np.max(target))
    x_ref_mm = reference_X_m * 1e3
    output_pitch_ref_m = float(reference_X_m[1] - reference_X_m[0])
    y_idx, x_idx = np.where(pupil)
    aperture_diameter_m = float((max(x_idx.max() - x_idx.min(), y_idx.max() - y_idx.min()) + 1) * dx)

    conditions: list[dict[str, object]] = []
    maps: dict[str, np.ndarray] = {}
    scaling_errors: dict[str, float] = {}
    target_spot_centers = [(-0.72, -0.48), (0.72, -0.48), (0.0, 0.74)]

    for lens_model in ["ordinary_fixed_f", "ideal_inverse_lambda_f"]:
        for wavelength in WAVELENGTHS_M:
            if lens_model == "ordinary_fixed_f":
                focal_length = focal0
            else:
                focal_length = focal0 * wavelength0 / wavelength
            scale = wavelength0 * focal0 / (wavelength * focal_length)
            query_fx = query_x_ref * scale
            query_fy = query_y_ref * scale
            spectrum_at_query = resample_complex_spectrum(
                spectrum_dense, frequency_dense, query_fx, query_fy
            )

            # Scalar lens Fourier transform: U_f = F[t](X/(lambda*f),Y/(lambda*f))/(i*lambda*f),
            # up to a spatially constant phase and the conventional propagation phase.
            field_at_plane = spectrum_at_query / (1j * wavelength * focal_length)
            intensity_raw = np.abs(field_at_plane) ** 2
            intensity_normalized = intensity_raw / max(float(np.sum(intensity_raw)), 1e-300)

            # Independently check the scale law by interpolating the dense reference
            # intensity and applying the predicted intensity Jacobian scale^2.
            predicted_raw = scale**2 * resample_intensity(
                intensity_dense, frequency_dense, query_fx, query_fy
            )
            predicted_normalized = predicted_raw / max(float(np.sum(predicted_raw)), 1e-300)
            scale_error = float(
                np.linalg.norm((intensity_normalized - predicted_normalized).ravel())
                / max(float(np.linalg.norm(predicted_normalized.ravel())), 1e-300)
            )
            scaling_errors[f"{lens_model}:{wavelength*1e9:.0f}nm"] = scale_error

            key = f"{lens_model}_{int(round(wavelength * 1e9))}nm"
            maps[key] = intensity_normalized.astype(np.float32)
            metrics = normalized_metrics(target_normalized, intensity_normalized, roi)
            centers = []
            expected_scale = wavelength / wavelength0 if lens_model == "ordinary_fixed_f" else 1.0
            for cx, cy in target_spot_centers:
                observed = peak_position(
                    intensity_normalized,
                    x_ref_mm,
                    x_ref_mm,
                    (cx * expected_scale, cy * expected_scale),
                )
                centers.append({"expected_xy_mm": [cx * expected_scale, cy * expected_scale],
                                "observed_peak_xy_mm": list(observed)})

            conditions.append(
                {
                    "lens_model": lens_model,
                    "wavelength_nm": wavelength * 1e9,
                    "effective_focal_length_m": focal_length,
                    "lambda_times_f_m2": wavelength * focal_length,
                    "Fourier_transform_argument_scale_relative_to_532nm": scale,
                    "physical_pattern_coordinate_scale_relative_to_532nm": 1.0 / scale,
                    "natural_output_pixel_pitch_m": wavelength * focal_length / width_m,
                    "metrics_vs_532nm_target": metrics,
                    "spot_peak_positions": centers,
                    "scale_law_interpolation_relative_L2": scale_error,
                    "fraction_of_target_ROI_within_sampled_frequency_band": float(
                        np.mean(
                            (np.abs(query_fx[roi]) <= frequency_dense[-1])
                            & (np.abs(query_fy[roi]) <= frequency_dense[-1])
                        )
                    ),
                }
            )

    results = {
        "model": {
            "screen": "fixed, phase-only three-spot screen from Step 3 seed 260930",
            "screen_is_assumed_wavelength_independent": True,
            "screen_pitch_m": dx,
            "screen_samples_per_axis": n,
            "screen_width_m": width_m,
            "pupil_diameter_m": aperture_diameter_m,
            "reference_wavelength_m": wavelength0,
            "reference_focal_length_m": focal0,
            "reference_fourier_plane_pixel_pitch_m": output_pitch_ref_m,
            "input_nyquist_spatial_frequency_per_m": 1.0 / (2.0 * dx),
            "dense_fft_zero_padding_factor": PAD_FACTOR,
            "dense_frequency_spacing_per_m": float(frequency_dense[1] - frequency_dense[0]),
            "comparison_coordinates": "all intensity maps resampled onto the 532 nm reference physical X,Y grid",
            "ideal_compensator": "f(lambda)=f0*lambda0/lambda; an ideal effective operator, not a fabricated lens model",
        },
        "conditions": conditions,
        "figures": [
            "fourier_optics_step4_wavelength_patterns.png",
            "fourier_optics_step4_wavelength_differences.png",
            "fourier_optics_step4_scaling_metrics.png",
        ],
    }
    result_path = OUT_DIR / "fourier_optics_step4_results.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)

    extent = [x_ref_mm[0], x_ref_mm[-1], x_ref_mm[0], x_ref_mm[-1]]
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
    fig.suptitle(
        "One fixed phase-only screen across wavelength\n"
        "Top: ordinary fixed focal length | Bottom: ideal f(λ)=f₀λ₀/λ",
        fontsize=14,
    )
    axes[0, 0].imshow(
        np.log10(np.maximum(target_normalized / np.max(target_normalized), 1e-7)),
        extent=extent, origin="lower", cmap="magma", vmin=-7, vmax=0,
    )
    axes[0, 0].set_title("532 nm target")
    axes[1, 0].imshow(
        np.log10(np.maximum(target_normalized / np.max(target_normalized), 1e-7)),
        extent=extent, origin="lower", cmap="magma", vmin=-7, vmax=0,
    )
    axes[1, 0].set_title("532 nm target")
    for row, model in enumerate(["ordinary_fixed_f", "ideal_inverse_lambda_f"]):
        for col, wavelength in enumerate(WAVELENGTHS_M, start=1):
            key = f"{model}_{int(round(wavelength * 1e9))}nm"
            image_data = maps[key]
            axes[row, col].imshow(
                np.log10(np.maximum(image_data / np.max(image_data), 1e-7)),
                extent=extent, origin="lower", cmap="magma", vmin=-7, vmax=0,
            )
            axes[row, col].set_title(f"{wavelength*1e9:.0f} nm")
    for ax in axes.ravel():
        ax.set_xlim(-1.8, 1.8)
        ax.set_ylim(-1.8, 1.8)
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    fig.savefig(OUT_DIR / "fourier_optics_step4_wavelength_patterns.png", dpi=170)
    plt.close(fig)

    difference_images = []
    for model in ["ordinary_fixed_f", "ideal_inverse_lambda_f"]:
        for wavelength in WAVELENGTHS_M:
            key = f"{model}_{int(round(wavelength * 1e9))}nm"
            difference_images.append((model, wavelength, maps[key] - target_normalized))
    differences_peak_scaled = [
        diff / max(float(np.max(target_normalized)), 1e-300)
        for _, _, diff in difference_images
    ]
    # Use a common robust scale for readable signed-error maps.
    vmax = max(
        float(np.percentile(np.abs(diff), 99.8)) for diff in differences_peak_scaled
    )
    vmax = max(vmax, 1e-6)
    fig, axes = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    fig.suptitle("Signed intensity error relative to the 532 nm target", fontsize=14)
    for i, (model, wavelength, diff) in enumerate(difference_images):
        row, col = divmod(i, 3)
        diff = diff / max(float(np.max(target_normalized)), 1e-300)
        im = axes[row, col].imshow(
            diff,
            extent=extent,
            origin="lower",
            cmap="coolwarm",
            norm=SymLogNorm(linthresh=max(vmax * 0.015, 1e-7), vmin=-vmax, vmax=vmax),
        )
        axes[row, col].set_title(
            f"{'Fixed f' if model == 'ordinary_fixed_f' else 'Ideal compensated'} — {wavelength*1e9:.0f} nm"
        )
        axes[row, col].set_xlim(-1.8, 1.8)
        axes[row, col].set_ylim(-1.8, 1.8)
        axes[row, col].set_xlabel("X (mm)")
        axes[row, col].set_ylabel("Y (mm)")
        fig.colorbar(im, ax=axes[row, col], shrink=0.82, label="(I − target) / target peak")
    fig.savefig(OUT_DIR / "fourier_optics_step4_wavelength_differences.png", dpi=170)
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    names = ["450 nm", "532 nm", "633 nm"]
    l2_fixed = [
        c["metrics_vs_532nm_target"]["normalized_intensity_relative_L2"]
        for c in conditions if c["lens_model"] == "ordinary_fixed_f"
    ]
    l2_comp = [
        c["metrics_vs_532nm_target"]["normalized_intensity_relative_L2"]
        for c in conditions if c["lens_model"] == "ideal_inverse_lambda_f"
    ]
    xloc = np.arange(3)
    axes[0].plot(xloc, l2_fixed, "o-", label="Fixed f")
    axes[0].plot(xloc, l2_comp, "s-", label="Ideal compensation")
    axes[0].set_xticks(xloc, names)
    axes[0].set_ylabel("Normalized intensity relative L2")
    axes[0].set_title("Error against the same physical target plane")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend()

    # Show the geometric position scaling for one off-axis spot. The prediction
    # is anchored at the measured 532 nm maximum, whose slight offset from the
    # target center is already present in the phase-only reconstruction.
    target_x_mm = 0.72
    baseline_peak_x, _ = peak_position(
        maps["ordinary_fixed_f_532nm"], x_ref_mm, x_ref_mm, (target_x_mm, -0.48)
    )
    for model, marker, label in [
        ("ordinary_fixed_f", "o", "Fixed f"),
        ("ideal_inverse_lambda_f", "s", "Ideal compensation"),
    ]:
        observed_x = []
        for wavelength in WAVELENGTHS_M:
            key = f"{model}_{int(round(wavelength * 1e9))}nm"
            scaling = wavelength / wavelength0 if model == "ordinary_fixed_f" else 1.0
            observed = peak_position(
                maps[key], x_ref_mm, x_ref_mm, (target_x_mm * scaling, -0.48 * scaling)
            )
            observed_x.append(observed[0])
        axes[1].plot(xloc, observed_x, f"{marker}-", label=f"{label}: measured peak")
    axes[1].plot(xloc, [baseline_peak_x * w / wavelength0 for w in WAVELENGTHS_M],
                 "k--", alpha=0.7, label="Fixed-f prediction from 532 nm peak")
    axes[1].plot(xloc, [baseline_peak_x] * len(WAVELENGTHS_M),
                 "k:", alpha=0.7, label="Compensated prediction")
    axes[1].axhline(target_x_mm, color="gray", linestyle=":", label="Target x")
    axes[1].set_xticks(xloc, names)
    axes[1].set_ylabel("Right spot X position (mm)")
    axes[1].set_title("Measured spot displacement")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)
    fig.savefig(OUT_DIR / "fourier_optics_step4_scaling_metrics.png", dpi=170)
    plt.close(fig)

    print(json.dumps({"results": str(result_path), "conditions": len(conditions)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
