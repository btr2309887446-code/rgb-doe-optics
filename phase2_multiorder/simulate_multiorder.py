from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)

WAVELENGTHS_NM = {"B": 450.0, "G": 532.0, "R": 638.0}
COLORS = {"B": "#1769aa", "G": "#1b8a3d", "R": "#c53929"}

# A finite grating that can be represented by a printed mask or a fabricated DOE.
PERIOD_M = 100e-6
SLIT_WIDTH_M = 5e-6
N_PERIODS = 40
FOCAL_LENGTH_M = 0.100

# Fine angular sampling is used because the compensated peaks differ by only a
# few hundredths of a degree.
THETA_DEG = np.linspace(-4.0, 4.0, 320001)
THETA_RAD = np.deg2rad(THETA_DEG)

BASELINE_ORDERS = {"B": 1, "G": 1, "R": 1}
COMPENSATED_ORDERS = {"B": 7, "G": 6, "R": 5}


def finite_grating_intensity(
    theta_rad: np.ndarray,
    wavelength_m: float,
    period_m: float = PERIOD_M,
    slit_width_m: float = SLIT_WIDTH_M,
    n_periods: int = N_PERIODS,
) -> np.ndarray:
    """Normalized Fraunhofer intensity of a finite multi-slit grating."""
    sine = np.sin(theta_rad)
    single_slit = np.sinc(slit_width_m * sine / wavelength_m) ** 2
    beta = np.pi * period_m * sine / wavelength_m
    denominator = np.sin(beta)
    ratio = np.empty_like(beta)
    near_zero = np.abs(denominator) < 1e-12
    ratio[near_zero] = float(n_periods)
    ratio[~near_zero] = np.sin(n_periods * beta[~near_zero]) / denominator[~near_zero]
    array_factor = (ratio / float(n_periods)) ** 2
    return single_slit * array_factor


def predicted_angle_deg(wavelength_nm: float, order: int) -> float:
    argument = order * wavelength_nm * 1e-9 / PERIOD_M
    if abs(argument) >= 1.0:
        return float("nan")
    return float(np.rad2deg(np.arcsin(argument)))


def nearest_peak(theta_deg: np.ndarray, intensity: np.ndarray, expected_deg: float) -> tuple[float, float]:
    window = np.abs(theta_deg - expected_deg) <= 0.12
    if not np.any(window):
        raise ValueError("Peak search window is outside the angular grid")
    indices = np.where(window)[0]
    local = indices[int(np.argmax(intensity[window]))]
    return float(theta_deg[local]), float(intensity[local])


def screen_position_mm(theta_deg: float) -> float:
    return float(FOCAL_LENGTH_M * np.tan(np.deg2rad(theta_deg)) * 1e3)


def normalized(value: np.ndarray) -> np.ndarray:
    peak = float(np.max(value))
    return value / peak if peak > 0 else value


def plot_angular_comparison() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    configs = [
        ("Ordinary grating: same order m=1", BASELINE_ORDERS),
        ("Compensated grating: B7 / G6 / R5", COMPENSATED_ORDERS),
    ]

    for ax, (title, orders) in zip(axes, configs):
        for label, wavelength_nm in WAVELENGTHS_NM.items():
            intensity = finite_grating_intensity(
                THETA_RAD, wavelength_nm * 1e-9
            )
            expected = predicted_angle_deg(wavelength_nm, orders[label])
            ax.plot(
                THETA_DEG,
                intensity,
                color=COLORS[label],
                linewidth=1.15,
                label=f"{label}: order {orders[label]}",
            )
            ax.axvline(expected, color=COLORS[label], alpha=0.25, linewidth=0.8)
        ax.set_ylim(-0.01, 1.05)
        ax.set_xlim(-0.2, 2.8)
        ax.set_ylabel("Normalized intensity")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncols=3)
    axes[-1].set_xlabel("Diffraction angle theta (degree)")
    fig.savefig(RESULTS / "phase2_angular_intensity.png", dpi=190)
    plt.close(fig)


def plot_screen_coordinate() -> None:
    fig, axes = plt.subplots(2, 1, figsize=(10, 7.5), sharex=True, constrained_layout=True)
    configs = [
        ("Ordinary grating at f = 100 mm", BASELINE_ORDERS),
        ("Multi-order compensation at f = 100 mm", COMPENSATED_ORDERS),
    ]
    x_mm = FOCAL_LENGTH_M * np.tan(THETA_RAD) * 1e3
    for ax, (title, orders) in zip(axes, configs):
        for label, wavelength_nm in WAVELENGTHS_NM.items():
            intensity = finite_grating_intensity(
                THETA_RAD, wavelength_nm * 1e-9
            )
            expected = predicted_angle_deg(wavelength_nm, orders[label])
            ax.plot(
                x_mm,
                intensity,
                color=COLORS[label],
                linewidth=1.15,
                label=f"{label}: order {orders[label]}",
            )
            ax.axvline(
                screen_position_mm(expected),
                color=COLORS[label],
                alpha=0.25,
                linewidth=0.8,
            )
        ax.set_ylim(-0.01, 1.05)
        ax.set_xlim(-0.1, 5.0)
        ax.set_ylabel("Normalized intensity")
        ax.set_title(title)
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False, ncols=3)
    axes[-1].set_xlabel("Back focal-plane coordinate X (mm)")
    fig.savefig(RESULTS / "phase2_screen_coordinate.png", dpi=190)
    plt.close(fig)


def plot_peak_positions(metrics: list[dict[str, float | str]]) -> None:
    labels = list(WAVELENGTHS_NM)
    x = np.arange(len(labels))
    width = 0.36
    baseline = [float(next(row["baseline_x_mm"] for row in metrics if row["channel"] == label)) for label in labels]
    compensated = [float(next(row["compensated_x_mm"] for row in metrics if row["channel"] == label)) for label in labels]

    fig, ax = plt.subplots(figsize=(9, 5.2), constrained_layout=True)
    ax.bar(x - width / 2, baseline, width, label="same order m=1", color="#9aa3ad")
    ax.bar(x + width / 2, compensated, width, label="B7 / G6 / R5", color="#6f8fae")
    ax.set_xticks(x, [f"{label}\n{WAVELENGTHS_NM[label]:.0f} nm" for label in labels])
    ax.set_ylabel("Peak position X (mm)")
    ax.set_title("RGB peak positions at the same observation plane")
    ax.grid(axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.savefig(RESULTS / "phase2_peak_positions.png", dpi=190)
    plt.close(fig)


def search_order_triplets() -> list[dict[str, float | int]]:
    """Search useful integer order triples near the 3 um wavelength product."""
    products = np.array(list(WAVELENGTHS_NM.values()), dtype=float)
    candidates: list[dict[str, float | int]] = []
    for mb in range(1, 13):
        for mg in range(1, 13):
            for mr in range(1, 13):
                order_product = np.array([mb, mg, mr], dtype=float) * products
                mean_product = float(np.mean(order_product))
                if mean_product < 1800 or mean_product > 5000:
                    continue
                relative_rms = float(
                    np.sqrt(np.mean(((order_product - mean_product) / mean_product) ** 2))
                )
                candidates.append(
                    {
                        "m_B": mb,
                        "m_G": mg,
                        "m_R": mr,
                        "mean_m_lambda_nm": mean_product,
                        "relative_rms": relative_rms,
                    }
                )
    candidates.sort(key=lambda row: float(row["relative_rms"]))
    return candidates[:12]


def plot_order_search(candidates: list[dict[str, float | int]]) -> None:
    labels = [
        f"B{row['m_B']}/G{row['m_G']}/R{row['m_R']}" for row in candidates
    ]
    errors = [float(row["relative_rms"]) * 100 for row in candidates]
    fig, ax = plt.subplots(figsize=(10, 5.4), constrained_layout=True)
    positions = np.arange(len(labels))
    ax.barh(positions, errors, color="#587a9f")
    ax.set_yticks(positions, labels)
    ax.invert_yaxis()
    ax.set_xlabel("RMS mismatch of m*lambda (%)")
    ax.set_title("Best RGB integer-order compensation candidates")
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(RESULTS / "phase2_order_search.png", dpi=190)
    plt.close(fig)


def calculate_metrics() -> list[dict[str, float | str]]:
    rows: list[dict[str, float | str]] = []
    for label, wavelength_nm in WAVELENGTHS_NM.items():
        wavelength_m = wavelength_nm * 1e-9
        intensity = finite_grating_intensity(THETA_RAD, wavelength_m)
        baseline_expected = predicted_angle_deg(wavelength_nm, BASELINE_ORDERS[label])
        compensated_expected = predicted_angle_deg(
            wavelength_nm, COMPENSATED_ORDERS[label]
        )
        baseline_peak, baseline_value = nearest_peak(
            THETA_DEG, intensity, baseline_expected
        )
        compensated_peak, compensated_value = nearest_peak(
            THETA_DEG, intensity, compensated_expected
        )
        rows.append(
            {
                "channel": label,
                "wavelength_nm": wavelength_nm,
                "baseline_order": BASELINE_ORDERS[label],
                "compensated_order": COMPENSATED_ORDERS[label],
                "baseline_expected_theta_deg": baseline_expected,
                "baseline_peak_theta_deg": baseline_peak,
                "baseline_x_mm": screen_position_mm(baseline_peak),
                "compensated_expected_theta_deg": compensated_expected,
                "compensated_peak_theta_deg": compensated_peak,
                "compensated_x_mm": screen_position_mm(compensated_peak),
                "compensated_peak_relative_intensity": compensated_value,
            }
        )
    return rows


def write_outputs(metrics: list[dict[str, float | str]], candidates: list[dict[str, float | int]]) -> None:
    with (RESULTS / "phase2_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)

    best = candidates[0]
    compensated_positions = [float(row["compensated_x_mm"]) for row in metrics]
    compensated_angles = [float(row["compensated_peak_theta_deg"]) for row in metrics]
    summary = {
        "geometry": {
            "period_um": PERIOD_M * 1e6,
            "slit_width_um": SLIT_WIDTH_M * 1e6,
            "n_periods": N_PERIODS,
            "focal_length_mm": FOCAL_LENGTH_M * 1e3,
        },
        "wavelengths_nm": WAVELENGTHS_NM,
        "baseline_orders": BASELINE_ORDERS,
        "compensated_orders": COMPENSATED_ORDERS,
        "best_search_candidate": best,
        "compensated_angle_span_deg": max(compensated_angles) - min(compensated_angles),
        "compensated_screen_span_mm": max(compensated_positions) - min(compensated_positions),
        "outputs": [
            "phase2_angular_intensity.png",
            "phase2_screen_coordinate.png",
            "phase2_peak_positions.png",
            "phase2_order_search.png",
            "phase2_metrics.csv",
        ],
    }
    (RESULTS / "phase2_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )


def main() -> None:
    metrics = calculate_metrics()
    candidates = search_order_triplets()
    plot_angular_comparison()
    plot_screen_coordinate()
    plot_peak_positions(metrics)
    plot_order_search(candidates)
    write_outputs(metrics, candidates)

    print(f"Phase 2 results written to: {RESULTS}")
    print("Top order candidates:")
    for row in candidates[:5]:
        print(
            f"  B{row['m_B']}/G{row['m_G']}/R{row['m_R']}: "
            f"{float(row['relative_rms']) * 100:.4f}% RMS mismatch"
        )


if __name__ == "__main__":
    main()
