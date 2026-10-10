from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

# Narrow-band RGB values used throughout the project.
WAVELENGTHS_NM = {
    "B": 450.0,
    "G": 532.0,
    "R": 638.0,
}
COLORS = {"B": "#1f77b4", "G": "#2ca02c", "R": "#d62728"}

# A realistic first-pass geometry for a table-top experiment.
Z_M = 0.30
SLIT_WIDTH_M = 100e-6
SLIT_SPACING_M = 400e-6
N_SLITS = 8

# Physical screen coordinate. This is deliberately wide enough to show
# the RGB scaling of the single-slit envelope.
X_M = np.linspace(-8e-3, 8e-3, 12001)


def sinc_un_normalized(x: np.ndarray) -> np.ndarray:
    """Return sin(x) / x with the removable singularity filled in."""
    result = np.ones_like(x, dtype=float)
    nonzero = np.abs(x) > 1e-14
    result[nonzero] = np.sin(x[nonzero]) / x[nonzero]
    return result


def single_slit_intensity(x_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    argument = np.pi * SLIT_WIDTH_M * x_m / (wavelength_m * Z_M)
    return sinc_un_normalized(argument) ** 2


def double_slit_intensity(x_m: np.ndarray, wavelength_m: float) -> np.ndarray:
    envelope = single_slit_intensity(x_m, wavelength_m)
    beta = np.pi * SLIT_SPACING_M * x_m / (wavelength_m * Z_M)
    return envelope * np.cos(beta) ** 2


def multi_slit_intensity(
    x_m: np.ndarray, wavelength_m: float, n_slits: int
) -> np.ndarray:
    envelope = single_slit_intensity(x_m, wavelength_m)
    beta = np.pi * SLIT_SPACING_M * x_m / (wavelength_m * Z_M)
    numerator = np.sin(n_slits * beta)
    denominator = np.sin(beta)
    array_factor = np.empty_like(beta)
    near_zero = np.abs(denominator) < 1e-12
    array_factor[near_zero] = float(n_slits) ** 2
    array_factor[~near_zero] = (numerator[~near_zero] / denominator[~near_zero]) ** 2
    return envelope * array_factor / float(n_slits) ** 2


def normalize(intensity: np.ndarray) -> np.ndarray:
    peak = float(np.max(intensity))
    return intensity / peak if peak > 0 else intensity


def first_zero_single_slit(wavelength_m: float) -> float:
    return wavelength_m * Z_M / SLIT_WIDTH_M


def fwhm(x_m: np.ndarray, intensity: np.ndarray) -> float:
    curve = normalize(intensity)
    half = 0.5
    center = int(np.argmin(np.abs(x_m)))

    left_candidates = np.where(curve[:center] <= half)[0]
    right_candidates = np.where(curve[center:] <= half)[0]
    if len(left_candidates) == 0 or len(right_candidates) == 0:
        return float("nan")

    left = left_candidates[-1]
    right = center + right_candidates[0]
    return float(x_m[right] - x_m[left])


def plot_physical_coordinate(
    filename: str,
    title: str,
    model,
    x_limit_mm: float = 8.0,
    show_legend: bool = True,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    for label, wavelength_nm in WAVELENGTHS_NM.items():
        intensity = normalize(model(X_M, wavelength_nm * 1e-9))
        ax.plot(
            X_M * 1e3,
            intensity,
            color=COLORS[label],
            linewidth=1.6,
            label=f"{label} {wavelength_nm:.0f} nm",
        )
    ax.set_xlim(-x_limit_mm, x_limit_mm)
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel("Screen coordinate X (mm)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    if show_legend:
        ax.legend(frameon=False)
    fig.savefig(RESULTS / filename, dpi=180)
    plt.close(fig)


def plot_single_slit_normalized() -> None:
    # u = X / lambda collapses the Fraunhofer single-slit curves.
    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    u = np.linspace(-20e3, 20e3, 12001)
    for label, wavelength_nm in WAVELENGTHS_NM.items():
        wavelength_m = wavelength_nm * 1e-9
        x_m = u * wavelength_m
        intensity = normalize(single_slit_intensity(x_m, wavelength_m))
        ax.plot(
            u / 1e3,
            intensity,
            color=COLORS[label],
            linewidth=1.6,
            label=f"{label} {wavelength_nm:.0f} nm",
        )
    ax.set_ylim(-0.02, 1.05)
    ax.set_xlabel(r"Normalized coordinate $u=X/\lambda$ (10$^3$)")
    ax.set_ylabel("Normalized intensity")
    ax.set_title("Single-slit RGB pattern in normalized coordinate")
    ax.grid(True, alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(RESULTS / "single_slit_normalized_coordinate.png", dpi=180)
    plt.close(fig)


def write_metrics() -> None:
    rows = []
    for label, wavelength_nm in WAVELENGTHS_NM.items():
        wavelength_m = wavelength_nm * 1e-9
        single = single_slit_intensity(X_M, wavelength_m)
        double = double_slit_intensity(X_M, wavelength_m)
        multi = multi_slit_intensity(X_M, wavelength_m, N_SLITS)
        rows.append(
            {
                "channel": label,
                "wavelength_nm": wavelength_nm,
                "single_slit_first_zero_mm": first_zero_single_slit(wavelength_m) * 1e3,
                "single_slit_fwhm_mm": fwhm(X_M, single) * 1e3,
                "double_slit_fwhm_mm": fwhm(X_M, double) * 1e3,
                "multi_slit_fwhm_mm": fwhm(X_M, multi) * 1e3,
            }
        )

    with (RESULTS / "phase1_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_summary() -> None:
    summary = {
        "geometry": {
            "z_m": Z_M,
            "slit_width_um": SLIT_WIDTH_M * 1e6,
            "slit_spacing_um": SLIT_SPACING_M * 1e6,
            "n_slits": N_SLITS,
            "screen_range_mm": [float(X_M.min() * 1e3), float(X_M.max() * 1e3)],
        },
        "wavelengths_nm": WAVELENGTHS_NM,
        "expected_single_slit_scaling": {
            "R_over_B": WAVELENGTHS_NM["R"] / WAVELENGTHS_NM["B"],
            "G_over_B": WAVELENGTHS_NM["G"] / WAVELENGTHS_NM["B"],
        },
        "outputs": [
            "single_slit_physical_coordinate.png",
            "single_slit_normalized_coordinate.png",
            "double_slit_physical_coordinate.png",
            "eight_slit_physical_coordinate.png",
            "phase1_metrics.csv",
        ],
    }
    (RESULTS / "phase1_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    plot_physical_coordinate(
        "single_slit_physical_coordinate.png",
        "Single-slit RGB diffraction on the same physical screen",
        single_slit_intensity,
    )
    plot_single_slit_normalized()
    plot_physical_coordinate(
        "double_slit_physical_coordinate.png",
        "Double-slit RGB diffraction on the same physical screen",
        double_slit_intensity,
    )
    plot_physical_coordinate(
        "eight_slit_physical_coordinate.png",
        f"{N_SLITS}-slit RGB diffraction on the same physical screen",
        lambda x, wavelength: multi_slit_intensity(x, wavelength, N_SLITS),
    )
    write_metrics()
    write_summary()
    print(f"Phase 1 results written to: {RESULTS}")


if __name__ == "__main__":
    main()
