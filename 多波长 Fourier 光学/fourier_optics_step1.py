"""Step 1: sampled thin-lens Fourier diffraction with physical coordinates.

Run with a Miniconda Python that has NumPy, SciPy, and Matplotlib installed:
    C:/Users/lenovo/miniconda3/envs/SI100B_Project/python.exe fourier_optics_step1.py

The script writes a PNG figure and a JSON results file beside itself.
"""

from __future__ import annotations

import json
import math
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1


OUT_DIR = Path(__file__).resolve().parent
N = 1024
DX = 5.0e-6  # input-plane pitch [m]
WAVELENGTH = 532.0e-9  # [m]
FOCAL_LENGTH = 0.200  # [m]
L = N * DX
DF = 1.0 / L
DX_F = WAVELENGTH * FOCAL_LENGTH * DF
MAX_FREQUENCY = 1.0 / (2.0 * DX)
MAX_FOCAL_COORD = WAVELENGTH * FOCAL_LENGTH * MAX_FREQUENCY


def centered_fft2(field: np.ndarray, dx: float) -> np.ndarray:
    """Riemann-sum approximation to the continuous FT, centered in both planes."""
    return (dx * dx) * np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(field))
    )


def centered_axis(n: int, pitch: float) -> np.ndarray:
    return (np.arange(n) - n // 2) * pitch


def disk_transform(radius: float, fx: np.ndarray, fy: np.ndarray) -> np.ndarray:
    """Continuous FT of a unit-amplitude disk centered at the origin."""
    rho = np.hypot(fx, fy)
    u = 2.0 * np.pi * radius * rho
    area = np.pi * radius * radius
    out = np.empty_like(u, dtype=np.float64)
    zero = np.isclose(u, 0.0)
    out[zero] = area
    nz = ~zero
    out[nz] = area * 2.0 * j1(u[nz]) / u[nz]
    return out


def sinc2_rect_transform(
    width: float, height: float, fx: np.ndarray, fy: np.ndarray,
    shift_x: float = 0.0, shift_y: float = 0.0,
) -> np.ndarray:
    """Continuous FT of a centered rectangle times a plane-wave carrier."""
    return (
        width
        * height
        * np.sinc(width * (fx - shift_x))
        * np.sinc(height * (fy - shift_y))
    )


def sampled_rect_grating_transform(
    aperture_samples: int,
    dx: float,
    fx: np.ndarray,
    fy: np.ndarray,
    carrier_x: float,
) -> np.ndarray:
    """Exact finite-sum transform of the sampled, centered rectangular grating."""
    # The odd number of aperture samples is centered at x=y=0. This
    # Dirichlet-kernel expression exactly sums the samples independently of FFT.
    qx = fx - carrier_x
    qy = fy
    sx = (aperture_samples * dx) * np.sinc(aperture_samples * dx * qx) / np.sinc(dx * qx)
    sy = (aperture_samples * dx) * np.sinc(aperture_samples * dx * qy) / np.sinc(dx * qy)
    return sx * sy


def analytic_disk_carriers(
    radius: float,
    fx: np.ndarray,
    fy: np.ndarray,
    carriers: list[tuple[complex, float, float]],
) -> np.ndarray:
    result = np.zeros_like(fx, dtype=np.complex128)
    for coefficient, shift_x, shift_y in carriers:
        result += coefficient * disk_transform(
            radius, fx - shift_x, fy - shift_y
        )
    return result


def direct_discrete_ft(
    field: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
    fx: float,
    fy: float,
    dx: float,
) -> complex:
    """Direct Riemann sum at one frequency; independent of np.fft.fft2."""
    phase_x = np.exp(-2j * np.pi * fx * x)
    phase_y = np.exp(-2j * np.pi * fy * y)
    return dx * dx * (phase_y @ field @ phase_x)


def fresnel_back_focal_plane(
    ft: np.ndarray, X: np.ndarray, Y: np.ndarray,
    wavelength: float, focal_length: float,
) -> np.ndarray:
    """Apply the ideal thin-lens prefactor and residual output quadratic phase."""
    k = 2.0 * np.pi / wavelength
    phase = np.exp(1j * k * focal_length) / (1j * wavelength * focal_length)
    phase = phase * np.exp(
        1j * np.pi * (X * X + Y * Y) / (wavelength * focal_length)
    )
    return phase * ft


def relative_l2(actual: np.ndarray, reference: np.ndarray, mask=None) -> float:
    if mask is not None:
        actual = actual[mask]
        reference = reference[mask]
    denom = np.linalg.norm(reference.ravel())
    if denom == 0:
        return float(np.linalg.norm(actual.ravel()))
    return float(np.linalg.norm((actual - reference).ravel()) / denom)


def main() -> None:
    fx_axis = np.fft.fftshift(np.fft.fftfreq(N, d=DX))
    fy_axis = fx_axis.copy()
    FX, FY = np.meshgrid(fx_axis, fy_axis)
    X_axis = WAVELENGTH * FOCAL_LENGTH * fx_axis
    Y_axis = WAVELENGTH * FOCAL_LENGTH * fy_axis
    X, Y = np.meshgrid(X_axis, Y_axis)
    x_axis = centered_axis(N, DX)
    y_axis = x_axis.copy()
    X_IN, Y_IN = np.meshgrid(x_axis, y_axis)

    # Case 1: sampled circular pupil.
    radius_px = 128
    radius = radius_px * DX
    diameter = 2.0 * radius
    circular_pupil = ((X_IN * X_IN + Y_IN * Y_IN) <= radius * radius).astype(float)
    ft_circle = centered_fft2(circular_pupil, DX)
    analytic_circle = disk_transform(radius, FX, FY)
    circle_intensity = np.abs(ft_circle) ** 2
    circle_intensity /= float(np.max(circle_intensity))
    circle_analytic_intensity = np.abs(analytic_circle) ** 2
    circle_analytic_intensity /= float(np.max(circle_analytic_intensity))
    u_circle = 2.0 * np.pi * radius * np.hypot(FX, FY)
    circle_compare_mask = u_circle <= 12.0
    circle_complex_rel_l2 = relative_l2(
        ft_circle, analytic_circle, circle_compare_mask
    )
    circle_intensity_rmse = float(
        np.sqrt(np.mean(
            (circle_intensity[circle_compare_mask]
             - circle_analytic_intensity[circle_compare_mask]) ** 2
        ))
    )
    j1_first_zero = 3.8317059702075125
    airy_first_zero_x = j1_first_zero * WAVELENGTH * FOCAL_LENGTH / (
        2.0 * np.pi * radius
    )
    center = N // 2
    airy_cut = circle_intensity[center, :]
    airy_nearest_null_sample = int(np.argmin(np.abs(X_axis - airy_first_zero_x)))
    airy_nearest_null_sample_x = float(X_axis[airy_nearest_null_sample])
    airy_nearest_null_sample_intensity = float(airy_cut[airy_nearest_null_sample])
    airy_nearest_null_sample_error = (
        abs(airy_nearest_null_sample_x - airy_first_zero_x) / airy_first_zero_x
    )

    # Case 2: a linear phase ramp (blazed grating) on a finite square pupil.
    # 32.35 cycles across the aperture puts the lobe between FFT samples so
    # the sampled sinc envelope is visible instead of collapsing to one bin.
    grating_cycles_across_window = 32.35
    grating_period = L / grating_cycles_across_window
    grating_period_samples = grating_period / DX
    grating_frequency = 1.0 / grating_period
    grating_aperture_samples = 801
    grating_aperture_half_width = (grating_aperture_samples // 2) * DX
    rectangular_pupil = (
        (np.abs(X_IN) <= grating_aperture_half_width)
        & (np.abs(Y_IN) <= grating_aperture_half_width)
    ).astype(float)
    grating_aperture_width = grating_aperture_samples * DX
    phase_grating = rectangular_pupil * np.exp(
        2j * np.pi * X_IN / grating_period
    )
    ft_grating = centered_fft2(phase_grating, DX)
    analytic_grating = sinc2_rect_transform(
        grating_aperture_width,
        grating_aperture_width,
        FX,
        FY,
        shift_x=grating_frequency,
    )
    exact_sampled_grating = sampled_rect_grating_transform(
        grating_aperture_samples, DX, FX, FY, carrier_x=grating_frequency
    )
    grating_discrete_rel_l2 = relative_l2(ft_grating, exact_sampled_grating)
    grating_continuous_rel_l2_full = relative_l2(ft_grating, analytic_grating)
    grating_main_lobe_mask = (
        (np.abs(L * (FX - grating_frequency)) <= 3.0)
        & (np.abs(L * FY) <= 3.0)
    )
    grating_continuous_rel_l2_main_lobe = relative_l2(
        ft_grating, analytic_grating, grating_main_lobe_mask
    )
    grating_peak_y, grating_peak_x = np.unravel_index(
        int(np.argmax(np.abs(ft_grating) ** 2)), ft_grating.shape
    )
    grating_peak_x_m = float(X_axis[grating_peak_x])
    grating_peak_y_m = float(Y_axis[grating_peak_y])
    grating_expected_x_m = WAVELENGTH * FOCAL_LENGTH / grating_period
    grating_peak_position_error = abs(grating_peak_x_m - grating_expected_x_m)
    center_line_num = np.abs(ft_grating[center, :]) ** 2
    center_line_ana = np.abs(analytic_grating[center, :]) ** 2
    line_window = np.abs(X_axis - grating_expected_x_m) <= 0.20e-3
    grating_line_num_norm = center_line_num[line_window] / np.max(center_line_num[line_window])
    grating_line_ana_norm = center_line_ana[line_window] / np.max(center_line_ana[line_window])
    grating_line_rel_l2 = relative_l2(grating_line_num_norm, grating_line_ana_norm)

    # Case 3: a complex-valued field over a disk, built from shifted carriers.
    # The carrier amplitudes keep |t| <= 0.95; their shifted Airy transforms
    # provide an independent continuous reference for the sampled FFT.
    carriers = [
        (0.55 + 0.0j, 0.0, 0.0),
        (0.25 + 0.0j, 12.0 / L, -4.0 / L),
        (0.15 + 0.0j, -17.0 / L, 13.0 / L),
    ]
    complex_screen = np.zeros((N, N), dtype=np.complex128)
    for coefficient, shift_x, shift_y in carriers:
        complex_screen += coefficient * np.exp(
            2j * np.pi * (shift_x * X_IN + shift_y * Y_IN)
        )
    complex_screen *= circular_pupil
    ft_complex = centered_fft2(complex_screen, DX)
    analytic_complex = analytic_disk_carriers(
        radius, FX, FY, carriers
    )
    carrier_mask = np.zeros((N, N), dtype=bool)
    for _, shift_x, shift_y in carriers:
        carrier_mask |= (
            (2.0 * np.pi * radius
             * np.hypot(FX - shift_x, FY - shift_y)) <= 12.0
        )
    complex_field_rel_l2 = relative_l2(ft_complex, analytic_complex, carrier_mask)
    complex_peak = float(np.max(np.abs(analytic_complex)))
    complex_error_peak_norm = float(
        np.max(np.abs(ft_complex[carrier_mask] - analytic_complex[carrier_mask]))
        / complex_peak
    )
    complex_intensity = np.abs(ft_complex) ** 2

    # Independent direct-sum checks on deterministic random complex data.
    rng = np.random.default_rng(20260930)
    random_amplitude = 0.2 + 0.8 * rng.random((N, N))
    random_phase = rng.uniform(-np.pi, np.pi, size=(N, N))
    random_complex_screen = circular_pupil * random_amplitude * np.exp(1j * random_phase)
    ft_random = centered_fft2(random_complex_screen, DX)
    sample_bins = [
        (center, center),
        (center + 31, center - 57),
        (center - 121, center + 93),
        (center + 217, center + 164),
        (center - 286, center - 271),
        (center + 402, center - 385),
        (center - 467, center + 431),
        (center + 119, center + 378),
    ]
    direct_errors = []
    for iy, ix in sample_bins:
        direct = direct_discrete_ft(
            random_complex_screen,
            x_axis,
            y_axis,
            float(fx_axis[ix]),
            float(fy_axis[iy]),
            DX,
        )
        direct_errors.append(abs(ft_random[iy, ix] - direct))
    random_ft_relative_error = float(max(direct_errors) / np.max(np.abs(ft_random)))
    df = 1.0 / (N * DX)
    input_power = float(np.sum(np.abs(random_complex_screen) ** 2) * DX * DX)
    focal_power = float(np.sum(np.abs(ft_random) ** 2) * df * df)
    parseval_relative_error = abs(focal_power - input_power) / input_power

    # Propagation prefactor: verify that its intensity integral conserves power.
    random_focal_field = fresnel_back_focal_plane(
        ft_random, X, Y, WAVELENGTH, FOCAL_LENGTH
    )
    focal_plane_power = float(np.sum(np.abs(random_focal_field) ** 2) * DX_F * DX_F)
    fresnel_power_relative_error = abs(focal_plane_power - input_power) / input_power

    # Output plot: physical axes and cross-checks are included in the figure.
    fig, axes = plt.subplots(3, 3, figsize=(15.5, 13.0), constrained_layout=True)
    fig.suptitle(
        "Thin-lens Fourier diffraction: sampled fields and analytical checks\n"
        f"λ={WAVELENGTH * 1e9:.0f} nm, f={FOCAL_LENGTH * 1e3:.0f} mm, "
        f"N={N}, dx={DX * 1e6:.1f} µm, L={L * 1e3:.2f} mm",
        fontsize=14,
    )

    axes[0, 0].imshow(
        circular_pupil,
        extent=[x_axis[0] * 1e3, x_axis[-1] * 1e3, y_axis[0] * 1e3, y_axis[-1] * 1e3],
        origin="lower",
        cmap="gray",
    )
    axes[0, 0].set_title("Circular pupil amplitude")
    axes[0, 0].set_xlabel("x (mm)")
    axes[0, 0].set_ylabel("y (mm)")
    crop_air = np.abs(X_axis) <= 0.45e-3
    axes[0, 1].imshow(
        np.maximum(circle_intensity, 1e-8),
        extent=[X_axis[0] * 1e3, X_axis[-1] * 1e3, Y_axis[0] * 1e3, Y_axis[-1] * 1e3],
        origin="lower",
        cmap="magma",
        norm=matplotlib.colors.LogNorm(vmin=1e-8, vmax=1.0),
    )
    axes[0, 1].set_xlim(-0.45, 0.45)
    axes[0, 1].set_ylim(-0.45, 0.45)
    axes[0, 1].axvline(airy_first_zero_x * 1e3, color="cyan", lw=0.8, ls="--")
    axes[0, 1].axvline(-airy_first_zero_x * 1e3, color="cyan", lw=0.8, ls="--")
    axes[0, 1].set_title("Circular-pupil diffraction (log scale)")
    axes[0, 1].set_xlabel("X (mm)")
    axes[0, 1].set_ylabel("Y (mm)")
    axes[0, 2].plot(X_axis[crop_air] * 1e3, circle_intensity[center, crop_air], label="FFT")
    axes[0, 2].plot(
        X_axis[crop_air] * 1e3,
        circle_analytic_intensity[center, crop_air],
        "--",
        label="Airy analytic",
    )
    axes[0, 2].set_yscale("log")
    axes[0, 2].set_ylim(1e-8, 1.2)
    axes[0, 2].axvline(airy_first_zero_x * 1e3, color="gray", lw=0.8, ls=":")
    axes[0, 2].set_title("Airy line cut: numerical vs analytic")
    axes[0, 2].set_xlabel("X (mm), Y=0")
    axes[0, 2].set_ylabel("Normalized intensity")
    axes[0, 2].legend(fontsize=8)

    phase_cmap = plt.get_cmap("twilight").copy()
    phase_cmap.set_bad(color="black")
    grating_phase_display = np.ma.masked_where(
        rectangular_pupil == 0, np.angle(phase_grating)
    )
    axes[1, 0].imshow(
        grating_phase_display,
        extent=[x_axis[0] * 1e3, x_axis[-1] * 1e3, y_axis[0] * 1e3, y_axis[-1] * 1e3],
        origin="lower",
        cmap=phase_cmap,
        vmin=-np.pi,
        vmax=np.pi,
    )
    axes[1, 0].set_title("Linear phase grating phase")
    axes[1, 0].set_xlabel("x (mm)")
    axes[1, 0].set_ylabel("y (mm)")
    grating_log = np.maximum(np.abs(ft_grating) ** 2 / np.max(np.abs(ft_grating) ** 2), 1e-12)
    axes[1, 1].imshow(
        grating_log,
        extent=[X_axis[0] * 1e3, X_axis[-1] * 1e3, Y_axis[0] * 1e3, Y_axis[-1] * 1e3],
        origin="lower",
        cmap="viridis",
        norm=matplotlib.colors.LogNorm(vmin=1e-12, vmax=1.0),
    )
    axes[1, 1].set_xlim(-1.0, 1.5)
    axes[1, 1].set_ylim(-0.4, 0.4)
    axes[1, 1].plot(grating_peak_x_m * 1e3, grating_peak_y_m * 1e3, "r+", ms=9)
    axes[1, 1].set_title("Finite-aperture grating diffraction (log)")
    axes[1, 1].set_xlabel("X (mm)")
    axes[1, 1].set_ylabel("Y (mm)")
    axes[1, 2].plot(
        X_axis[line_window] * 1e3,
        center_line_num[line_window] / np.max(center_line_num),
        ".",
        ms=3,
        label="FFT samples",
    )
    axes[1, 2].plot(
        X_axis[line_window] * 1e3,
        center_line_ana[line_window] / np.max(center_line_ana),
        "-",
        lw=1,
        label="finite-square analytic",
    )
    axes[1, 2].axvline(grating_expected_x_m * 1e3, color="gray", lw=0.8, ls=":")
    axes[1, 2].set_yscale("log")
    axes[1, 2].set_ylim(1e-12, 1.2)
    axes[1, 2].set_title("Grating order envelope")
    axes[1, 2].set_xlabel("X (mm), Y=0")
    axes[1, 2].set_ylabel("Normalized intensity")
    axes[1, 2].legend(fontsize=8)

    complex_magnitude = np.abs(complex_screen)
    complex_hsv = np.stack(
        [
            (np.angle(complex_screen) + np.pi) / (2.0 * np.pi),
            circular_pupil,
            complex_magnitude / max(float(np.max(complex_magnitude)), 1e-30),
        ],
        axis=-1,
    )
    axes[2, 0].imshow(
        matplotlib.colors.hsv_to_rgb(complex_hsv),
        extent=[x_axis[0] * 1e3, x_axis[-1] * 1e3, y_axis[0] * 1e3, y_axis[-1] * 1e3],
        origin="lower",
        cmap="viridis",
    )
    axes[2, 0].set_title("Complex screen: phase=hue, amplitude=brightness")
    axes[2, 0].set_xlabel("x (mm)")
    axes[2, 0].set_ylabel("y (mm)")
    complex_log = np.maximum(complex_intensity / np.max(complex_intensity), 1e-9)
    axes[2, 1].imshow(
        complex_log,
        extent=[X_axis[0] * 1e3, X_axis[-1] * 1e3, Y_axis[0] * 1e3, Y_axis[-1] * 1e3],
        origin="lower",
        cmap="magma",
        norm=matplotlib.colors.LogNorm(vmin=1e-9, vmax=1.0),
    )
    axes[2, 1].set_xlim(-1.6, 1.6)
    axes[2, 1].set_ylim(-1.6, 1.6)
    axes[2, 1].set_title("Complex screen diffraction (log)")
    axes[2, 1].set_xlabel("X (mm)")
    axes[2, 1].set_ylabel("Y (mm)")
    complex_error_map = np.abs(ft_complex - analytic_complex) / complex_peak
    complex_error_display = np.where(
        carrier_mask, np.maximum(complex_error_map, 1e-12), np.nan
    )
    error_cmap = plt.get_cmap("inferno").copy()
    error_cmap.set_bad(color="white")
    axes[2, 2].imshow(
        complex_error_display,
        extent=[X_axis[0] * 1e3, X_axis[-1] * 1e3, Y_axis[0] * 1e3, Y_axis[-1] * 1e3],
        origin="lower",
        cmap=error_cmap,
        norm=matplotlib.colors.LogNorm(vmin=1e-7, vmax=1e-3),
    )
    axes[2, 2].set_xlim(-1.6, 1.6)
    axes[2, 2].set_ylim(-1.6, 1.6)
    axes[2, 2].set_title("Complex-field error vs shifted-Airy model")
    axes[2, 2].set_xlabel("X (mm)")
    axes[2, 2].set_ylabel("Y (mm)")

    fig.savefig(OUT_DIR / "fourier_optics_step1.png", dpi=180)
    plt.close(fig)

    try:
        import importlib.metadata as metadata

        pyoptica_version = metadata.version("pyoptica")
        pyoptica_status = f"installed ({pyoptica_version})"
    except metadata.PackageNotFoundError:
        pyoptica_status = "not installed in the inspected Miniconda environments"

    results = {
        "runtime": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": matplotlib.__version__,
            "pyoptica": pyoptica_status,
        },
        "parameters": {
            "N": N,
            "input_pixel_pitch_m": DX,
            "input_window_width_m": L,
            "wavelength_m": WAVELENGTH,
            "focal_length_m": FOCAL_LENGTH,
            "input_frequency_spacing_per_m": DF,
            "output_pixel_pitch_m": DX_F,
            "nyquist_frequency_per_m": MAX_FREQUENCY,
            "max_focal_plane_coordinate_m": MAX_FOCAL_COORD,
            "circular_aperture_diameter_m": diameter,
            "circular_aperture_pixels_across": 2 * radius_px,
            "paraxial_NA_approx": radius / math.sqrt(radius * radius + FOCAL_LENGTH * FOCAL_LENGTH),
        },
        "circular_aperture": {
            "continuous_first_zero_m": airy_first_zero_x,
            "nearest_sample_to_first_zero_m": airy_nearest_null_sample_x,
            "nearest_sample_relative_offset_from_first_zero": airy_nearest_null_sample_error,
            "normalized_intensity_at_nearest_sample": airy_nearest_null_sample_intensity,
            "continuous_Airy_intensity_RMSE_u_le_12": circle_intensity_rmse,
            "complex_field_relative_L2_u_le_12": circle_complex_rel_l2,
        },
        "phase_grating": {
            "period_m": grating_period,
            "samples_per_period": grating_period_samples,
            "cycles_across_window": grating_cycles_across_window,
            "rectangular_aperture_width_m": grating_aperture_width,
            "rectangular_aperture_pixels_across": grating_aperture_samples,
            "spatial_frequency_per_m": grating_frequency,
            "expected_first_order_X_m": grating_expected_x_m,
            "sampled_peak_X_m": grating_peak_x_m,
            "sampled_peak_Y_m": grating_peak_y_m,
            "peak_position_error_m": grating_peak_position_error,
            "exact_sampled_Dirichlet_field_relative_L2": grating_discrete_rel_l2,
            "continuous_sinc_field_relative_L2_full_Nyquist_band": grating_continuous_rel_l2_full,
            "continuous_sinc_field_relative_L2_main_lobe_region": grating_continuous_rel_l2_main_lobe,
            "center_line_normalized_intensity_relative_L2_window": grating_line_rel_l2,
        },
        "complex_screen": {
            "continuous_shifted_Airy_field_relative_L2_in_carrier_regions": complex_field_rel_l2,
            "max_complex_field_error_over_peak_in_carrier_regions": complex_error_peak_norm,
            "random_screen_direct_DFT_max_relative_error": random_ft_relative_error,
            "random_screen_parseval_relative_error": parseval_relative_error,
            "random_screen_fresnel_power_relative_error": fresnel_power_relative_error,
        },
        "figure": "fourier_optics_step1.png",
    }
    with (OUT_DIR / "fourier_optics_step1_results.json").open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
