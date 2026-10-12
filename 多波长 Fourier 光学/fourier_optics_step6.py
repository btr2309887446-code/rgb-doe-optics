"""Step 6: illustrative phase-quantization, pixel-pitch, and NA limits."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FormatStrFormatter


OUT_DIR = Path(__file__).resolve().parent
ARCHIVE_PATH = OUT_DIR / "fourier_optics_step3_phase_screens.npz"
WAVELENGTH = 532.0e-9
FOCAL_LENGTH = 0.200
PHASE_LEVEL_COUNTS = [2, 4, 8, 16, 32, 64]
PIXEL_BLOCKS = [1, 2, 4, 8, 16]
NA_VALUES = [0.0025, 0.0030, 0.0040, 0.0050, 0.0070, 0.0100]


def centered_fft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def metrics(target: np.ndarray, reconstructed: np.ndarray, roi: np.ndarray) -> dict[str, float]:
    p_target = target / max(float(np.sum(target)), 1e-300)
    p_reconstructed = reconstructed / max(float(np.sum(reconstructed)), 1e-300)
    difference = p_reconstructed - p_target
    correlation = float(np.corrcoef(p_target.ravel(), p_reconstructed.ravel())[0, 1])
    return {
        "normalized_intensity_relative_L2": float(
            np.linalg.norm(difference.ravel())
            / max(float(np.linalg.norm(p_target.ravel())), 1e-300)
        ),
        "normalized_intensity_RMSE_per_pixel": float(np.sqrt(np.mean(difference**2))),
        "intensity_correlation_global": correlation,
        "squared_Bhattacharyya_fidelity": float(
            np.sum(np.sqrt(p_target * p_reconstructed)) ** 2
        ),
        "reconstructed_energy_fraction_in_target_ROI": float(
            np.sum(p_reconstructed[roi])
        ),
    }


def quantize_phase(phase: np.ndarray, levels: int, pupil: np.ndarray) -> np.ndarray:
    step = 2.0 * np.pi / levels
    quantized = (np.round((phase + np.pi) / step) * step - np.pi + np.pi) % (2.0 * np.pi) - np.pi
    return np.where(pupil, quantized, 0.0)


def evaluate_screen(
    phase: np.ndarray,
    pupil: np.ndarray,
    target: np.ndarray,
    roi: np.ndarray,
    label: str,
) -> tuple[dict[str, object], np.ndarray, np.ndarray]:
    screen = pupil * np.exp(1j * phase)
    spectrum = centered_fft(screen)
    intensity = np.abs(spectrum) ** 2
    result: dict[str, object] = {"case": label, **metrics(target, intensity, roi)}
    result["screen_power_relative_to_continuous"] = float(
        np.sum(np.abs(screen) ** 2) / max(float(np.sum(pupil)), 1.0)
    )
    return result, spectrum, intensity


def block_sample_phase(phase: np.ndarray, block: int) -> np.ndarray:
    if block == 1:
        return phase.copy()
    n = phase.shape[0]
    coarse = phase[block // 2 :: block, block // 2 :: block]
    if coarse.shape != (n // block, n // block):
        coarse = phase[block // 2 : n : block, block // 2 : n : block]
    return np.repeat(np.repeat(coarse, block, axis=0), block, axis=1)


def main() -> None:
    if not ARCHIVE_PATH.exists():
        raise FileNotFoundError(f"Missing Step 3 archive: {ARCHIVE_PATH}")
    with np.load(ARCHIVE_PATH, allow_pickle=False) as archive:
        phase = archive["three_spots_phase_rad"].astype(np.float64)
        target = archive["three_spots_target_intensity"].astype(np.float64)
        pupil = archive["pupil"].astype(bool)
        dx = float(archive["input_pixel_pitch_m"])

    n = phase.shape[0]
    L = n * dx
    fx = np.fft.fftshift(np.fft.fftfreq(n, d=dx))
    FX, FY = np.meshgrid(fx, fx)
    rho_frequency = np.hypot(FX, FY)
    roi = target >= 0.01 * float(np.max(target))
    target_roi_frequency_max = float(np.max(rho_frequency[roi]))
    target_roi_minimum_na = WAVELENGTH * target_roi_frequency_max
    continuous_result, continuous_spectrum, continuous_intensity = evaluate_screen(
        phase, pupil, target, roi, "continuous reference"
    )

    quantization_results = []
    for levels in PHASE_LEVEL_COUNTS:
        qphase = quantize_phase(phase, levels, pupil)
        result, _, _ = evaluate_screen(
            qphase, pupil, target, roi, f"{levels} phase levels"
        )
        result["phase_levels"] = levels
        result["maximum_phase_rounding_error_rad"] = float(np.pi / levels)
        quantization_results.append(result)

    pixel_results = []
    pixel_screens: dict[int, np.ndarray] = {}
    for block in PIXEL_BLOCKS:
        sampled_phase = block_sample_phase(phase, block)
        result, _, _ = evaluate_screen(
            sampled_phase, pupil, target, roi, f"{block*dx*1e6:.0f} um phase pixel"
        )
        result["effective_phase_pixel_pitch_m"] = block * dx
        result["phase_pixel_pitch_nyquist_per_m"] = 1.0 / (2.0 * block * dx)
        pixel_results.append(result)
        pixel_screens[block] = sampled_phase

    # A hard circular spatial-frequency cutoff models a specified paraxial NA.
    na_results = []
    for na in NA_VALUES:
        cutoff = na / WAVELENGTH
        passband = rho_frequency <= cutoff
        filtered_spectrum = continuous_spectrum * passband
        filtered_intensity = np.abs(filtered_spectrum) ** 2
        result: dict[str, object] = {
            "NA": na,
            "cutoff_spatial_frequency_per_m": cutoff,
            "fraction_of_target_ROI_inside_cutoff": float(
                np.mean(rho_frequency[roi] <= cutoff)
            ),
            "transmitted_power_fraction_relative_to_no_NA_cutoff": float(
                np.sum(filtered_intensity) / max(float(np.sum(continuous_intensity)), 1e-300)
            ),
            **metrics(target, filtered_intensity, roi),
        }
        na_results.append(result)

    combined_results = []
    combined_scenarios = [(8, 1, 0.01), (8, 4, 0.007), (8, 4, 0.005), (4, 4, 0.005)]
    for levels, block, na in combined_scenarios:
        qphase = quantize_phase(block_sample_phase(phase, block), levels, pupil)
        screen = pupil * np.exp(1j * qphase)
        spec = centered_fft(screen)
        passband = rho_frequency <= na / WAVELENGTH
        filtered_spec = spec * passband
        filtered_intensity = np.abs(filtered_spec) ** 2
        result: dict[str, object] = {
            "phase_levels": levels,
            "phase_pixel_pitch_m": block * dx,
            "NA": na,
            "transmitted_power_fraction": float(
                np.sum(filtered_intensity) / max(float(np.sum(np.abs(spec) ** 2)), 1e-300)
            ),
            **metrics(target, filtered_intensity, roi),
        }
        combined_results.append(result)

    results = {
        "scope": "post-process a single-wavelength continuous GS phase screen; no re-optimization at each hardware setting",
        "model": {
            "wavelength_m": WAVELENGTH,
            "focal_length_m": FOCAL_LENGTH,
            "screen_samples_per_axis": n,
            "screen_pixel_pitch_m": dx,
            "screen_width_m": L,
            "pupil_diameter_m_nominal": 1.28e-3,
            "target_ROI": "target intensity >= 1% of maximum",
            "target_ROI_max_spatial_frequency_per_m": target_roi_frequency_max,
            "minimum_NA_to_include_every_ROI_sample": target_roi_minimum_na,
            "NA_model": "hard circular cutoff sqrt(fx^2+fy^2) <= NA/lambda in the Fourier plane",
            "phase_quantization": "nearest of Q equally spaced levels in [-pi, pi)",
            "pixel_pitch_model": "sample the saved phase at each coarse pixel center, then hold it over that pixel; preserve the original circular pupil mask",
        },
        "continuous_reference": continuous_result,
        "phase_quantization_sweep": quantization_results,
        "phase_pixel_pitch_sweep": pixel_results,
        "NA_sweep": na_results,
        "combined_examples": combined_results,
        "figure": "fourier_optics_step6_hardware_limits.png",
    }
    result_path = OUT_DIR / "fourier_optics_step6_results.json"
    with result_path.open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.8), constrained_layout=True)
    axes[0].plot(
        PHASE_LEVEL_COUNTS,
        [r["normalized_intensity_relative_L2"] for r in quantization_results],
        "o-",
    )
    axes[0].axhline(continuous_result["normalized_intensity_relative_L2"], color="gray", linestyle="--", label="Continuous reference")
    axes[0].set_xscale("log", base=2)
    axes[0].set_xticks(PHASE_LEVEL_COUNTS, [str(q) for q in PHASE_LEVEL_COUNTS])
    axes[0].set_xlabel("Number of phase levels Q")
    axes[0].set_ylabel("Normalized intensity relative L2")
    axes[0].set_title("Phase quantization")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=8)

    axes[1].plot(
        [r["effective_phase_pixel_pitch_m"] * 1e6 for r in pixel_results],
        [r["normalized_intensity_relative_L2"] for r in pixel_results],
        "o-",
    )
    axes[1].axhline(continuous_result["normalized_intensity_relative_L2"], color="gray", linestyle="--", label="Continuous reference")
    axes[1].set_xlabel("Phase pixel pitch (µm)")
    axes[1].set_ylabel("Normalized intensity relative L2")
    axes[1].set_title("Spatial pixel sampling")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=8)

    axes[2].plot(
        [r["NA"] for r in na_results],
        [r["normalized_intensity_relative_L2"] for r in na_results],
        "o-",
        label="Normalized shape error",
    )
    axes[2].set_xlabel("Numerical aperture NA")
    axes[2].set_ylabel("Normalized intensity relative L2")
    axes[2].set_title("Hard Fourier-frequency cutoff")
    axes[2].xaxis.set_major_formatter(FormatStrFormatter("%.4f"))
    axes[2].grid(True, alpha=0.25)
    ax_power = axes[2].twinx()
    ax_power.plot(
        [r["NA"] for r in na_results],
        [r["transmitted_power_fraction_relative_to_no_NA_cutoff"] for r in na_results],
        "s--",
        color="tab:orange",
        label="Power passed",
    )
    ax_power.set_ylabel("Fraction of power passed", color="tab:orange")
    axes[2].axvline(target_roi_minimum_na, color="gray", linestyle=":", label="ROI cutoff")
    handles1, labels1 = axes[2].get_legend_handles_labels()
    handles2, labels2 = ax_power.get_legend_handles_labels()
    axes[2].legend(handles1 + handles2, labels1 + labels2, fontsize=8)

    fig.suptitle("Hardware constraints on one fixed GS phase screen", fontsize=14)
    fig.savefig(OUT_DIR / "fourier_optics_step6_hardware_limits.png", dpi=170)
    plt.close(fig)
    print(json.dumps({"results": str(result_path), "phase_cases": len(quantization_results), "NA_cases": len(na_results)}, ensure_ascii=True))


if __name__ == "__main__":
    main()
