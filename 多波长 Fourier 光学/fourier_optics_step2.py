"""Step 2: invert complex focal-plane targets into sampled complex screens.

Run with the configured Miniconda interpreter:
    C:/Users/lenovo/miniconda3/envs/SI100B_Project/python.exe fourier_optics_step2.py

The script writes a figure, a dense Gaussian cut, and a JSON summary beside itself.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUT_DIR = Path(__file__).resolve().parent
N = 1024
PAD_FACTOR = 2
DX = 5.0e-6
WAVELENGTH = 532.0e-9
FOCAL_LENGTH = 0.200
U0 = 1.0 + 0.0j
L = N * DX
K = 2.0 * np.pi / WAVELENGTH


def centered_fft2(field: np.ndarray, dx: float) -> np.ndarray:
    """Riemann-sum approximation to the continuous FT on centered grids."""
    return dx * dx * np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(field))
    )


def centered_ifft2_from_continuous_spectrum(
    spectrum: np.ndarray, dx: float
) -> np.ndarray:
    """Inverse of centered_fft2 for matched N×N Fourier samples."""
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(spectrum / (dx * dx)))
    )


def lens_fourier_factor(X_row: np.ndarray, Y_col: np.ndarray) -> np.ndarray:
    """Thin-lens back-focal-plane factor multiplying the screen FT."""
    return (
        U0
        * np.exp(1j * K * FOCAL_LENGTH)
        / (1j * WAVELENGTH * FOCAL_LENGTH)
        * np.exp(
            1j
            * np.pi
            * (X_row * X_row + Y_col * Y_col)
            / (WAVELENGTH * FOCAL_LENGTH)
        )
    )


def field_metrics(target: np.ndarray, reconstructed: np.ndarray) -> dict[str, float]:
    target_norm = max(float(np.max(np.abs(target))), 1e-300)
    target_i = np.abs(target) ** 2
    reconstructed_i = np.abs(reconstructed) ** 2
    target_i_norm = target_i / max(float(np.max(target_i)), 1e-300)
    reconstructed_i_norm = reconstructed_i / max(float(np.max(reconstructed_i)), 1e-300)
    target_flat = target.ravel()
    error_flat = (reconstructed - target).ravel()
    intensity_correlation = float(
        np.corrcoef(target_i_norm.ravel(), reconstructed_i_norm.ravel())[0, 1]
    )
    return {
        "complex_field_relative_L2": float(
            np.linalg.norm(error_flat) / max(np.linalg.norm(target_flat), 1e-300)
        ),
        "complex_field_max_error_over_target_peak": float(
            np.max(np.abs(reconstructed - target)) / target_norm
        ),
        "normalized_intensity_RMSE": float(
            np.sqrt(np.mean((reconstructed_i_norm - target_i_norm) ** 2))
        ),
        "normalized_intensity_correlation": intensity_correlation,
    }


def main() -> None:
    fx = np.fft.fftshift(np.fft.fftfreq(N, d=DX))
    fy = fx.copy()
    x = (np.arange(N) - N // 2) * DX
    X_axis = WAVELENGTH * FOCAL_LENGTH * fx
    Y_axis = WAVELENGTH * FOCAL_LENGTH * fy
    X_row = X_axis[None, :]
    Y_col = Y_axis[:, None]
    f_lens = lens_fourier_factor(X_row, Y_col)

    sigma_g = 0.45e-3

    def gaussian(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        return np.exp(-(X * X + Y * Y) / (4.0 * sigma_g * sigma_g))

    spot_sigma = 0.18e-3
    spot_spec = [
        (0.95 + 0.0j, -0.85e-3, -0.35e-3),
        (0.78 * np.exp(0.8j), 0.78e-3, -0.35e-3),
        (0.70 * np.exp(-1.1j), 0.05e-3, 0.92e-3),
    ]

    def multiple_spots(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        result = np.zeros(np.broadcast_shapes(X.shape, Y.shape), dtype=np.complex128)
        for coefficient, x0, y0 in spot_spec:
            result += coefficient * np.exp(
                -((X - x0) ** 2 + (Y - y0) ** 2)
                / (4.0 * spot_sigma * spot_sigma)
            )
        return result

    ring_radius = 1.35e-3
    ring_sigma = 0.22e-3

    def ring(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        radius = np.hypot(X, Y)
        phase = np.arctan2(Y, X)
        return np.exp(-((radius - ring_radius) ** 2) / (4.0 * ring_sigma**2)) * np.exp(1j * phase)

    random_rng = np.random.default_rng(20260930)
    random_modes: list[tuple[complex, float, float]] = []
    for _ in range(14):
        coefficient = (
            random_rng.normal() + 1j * random_rng.normal()
        ) / math.sqrt(28.0)
        kx = float(random_rng.uniform(-1800.0, 1800.0))
        ky = float(random_rng.uniform(-1800.0, 1800.0))
        random_modes.append((coefficient, kx, ky))
    random_sigma = 1.40e-3

    def random_complex(X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        result = np.zeros(np.broadcast_shapes(X.shape, Y.shape), dtype=np.complex128)
        for coefficient, kx, ky in random_modes:
            result += coefficient * np.exp(2j * np.pi * (kx * X + ky * Y))
        return result * np.exp(-(X * X + Y * Y) / (4.0 * random_sigma**2))

    targets = [
        ("Gaussian", gaussian),
        ("Multiple coherent spots", multiple_spots),
        ("Vortex ring", ring),
        ("Random smooth complex field", random_complex),
    ]

    # Shared plotting buffers retain only normalized float images.
    plot_data: list[dict[str, np.ndarray | str]] = []
    summaries: list[dict[str, object]] = []
    dense_cuts: dict[str, dict[str, np.ndarray]] = {}
    dense_N = PAD_FACTOR * N
    dense_frequency = np.fft.fftshift(np.fft.fftfreq(dense_N, d=DX))
    dense_X_axis = WAVELENGTH * FOCAL_LENGTH * dense_frequency
    dense_df = 1.0 / (dense_N * DX)
    dense_dX = WAVELENGTH * FOCAL_LENGTH * dense_df

    for name, target_function in targets:
        target_unscaled = target_function(X_row, Y_col)
        target_peak_unscaled = float(np.max(np.abs(target_unscaled)))
        target_unit_peak = target_unscaled / max(target_peak_unscaled, 1e-300)

        desired_screen_spectrum = target_unit_peak / f_lens
        screen_raw = centered_ifft2_from_continuous_spectrum(
            desired_screen_spectrum, DX
        )

        # Passive complex-transmission constraint: a global attenuation keeps
        # every screen sample at |t| <= 1 without changing normalized shape.
        screen_peak_raw = float(np.max(np.abs(screen_raw)))
        passive_scale = min(1.0, 1.0 / max(screen_peak_raw, 1e-300))
        screen = screen_raw * passive_scale
        target = target_unit_peak * passive_scale
        desired_screen_spectrum = desired_screen_spectrum * passive_scale

        ft_reconstructed = centered_fft2(screen, DX)
        reconstructed = f_lens * ft_reconstructed
        metrics = field_metrics(target, reconstructed)

        input_power = L * L * abs(U0) ** 2
        screen_power = float(np.sum(np.abs(screen) ** 2) * DX * DX)
        focal_plane_spacing = WAVELENGTH * FOCAL_LENGTH / L
        focal_power = float(
            np.sum(np.abs(reconstructed) ** 2)
            * focal_plane_spacing
            * focal_plane_spacing
        )
        transmitted_fraction = screen_power / input_power

        edge_px = 16
        edge_mask = np.zeros((N, N), dtype=bool)
        edge_mask[:edge_px, :] = True
        edge_mask[-edge_px:, :] = True
        edge_mask[:, :edge_px] = True
        edge_mask[:, -edge_px:] = True
        screen_energy = np.abs(screen) ** 2
        edge_power_fraction = float(
            np.sum(screen_energy[edge_mask]) / max(np.sum(screen_energy), 1e-300)
        )
        edge_amplitude_over_peak = float(
            np.max(np.abs(screen[edge_mask])) / max(np.max(np.abs(screen)), 1e-300)
        )
        screen_amplitude = np.abs(screen)
        phase_only_rms_mismatch = float(
            np.sqrt(np.mean((screen_amplitude - 1.0) ** 2))
        )

        # Zero-pad the finite screen to sample its continuous Fourier transform
        # between the original DFT frequencies. This interpolates; it does not
        # add aperture bandwidth or physical resolution.
        padded = np.zeros((dense_N, dense_N), dtype=np.complex128)
        start = (dense_N - N) // 2
        padded[start : start + N, start : start + N] = screen
        ft_dense = centered_fft2(padded, DX)
        Xd_row = dense_X_axis[None, :]
        Yd_col = dense_X_axis[:, None]
        dense_factor = lens_fourier_factor(Xd_row, Yd_col)
        reconstructed_dense = dense_factor * ft_dense
        target_dense = target_function(Xd_row, Yd_col)
        target_dense = target_dense / max(target_peak_unscaled, 1e-300) * passive_scale
        dense_mask = np.abs(target_dense) >= 0.05 * max(
            float(np.max(np.abs(target_dense))), 1e-300
        )
        dense_target_norm = float(np.linalg.norm(target_dense[dense_mask]))
        dense_metrics = {
            "zero_padded_output_pixel_pitch_m": dense_dX,
            "complex_field_relative_L2_where_target_amplitude_ge_5pct_peak": float(
                np.linalg.norm(
                    (reconstructed_dense - target_dense)[dense_mask]
                )
                / max(dense_target_norm, 1e-300)
            ),
            "max_complex_error_over_target_peak_in_same_region": float(
                np.max(np.abs(reconstructed_dense[dense_mask] - target_dense[dense_mask]))
                / max(float(np.max(np.abs(target_dense))), 1e-300)
            ),
            "samples_in_dense_output_grid": dense_N,
        }

        # Keep only display-scaled data, keeping the saved file compact.
        target_intensity = np.abs(target_unit_peak) ** 2
        reconstructed_intensity = np.abs(reconstructed) ** 2
        target_intensity /= max(float(np.max(target_intensity)), 1e-300)
        reconstructed_intensity /= max(float(np.max(reconstructed_intensity)), 1e-300)
        target_phase = np.angle(target_unit_peak)
        target_phase_mask = np.abs(target_unit_peak) >= 0.02
        screen_hsv = np.stack(
            [
                (np.angle(screen) + np.pi) / (2.0 * np.pi),
                (screen_amplitude > 1e-12).astype(float),
                screen_amplitude / max(float(np.max(screen_amplitude)), 1e-300),
            ],
            axis=-1,
        )
        screen_rgb = matplotlib.colors.hsv_to_rgb(screen_hsv).astype(np.float32)
        plot_data.append(
            {
                "name": name,
                "target_i": target_intensity.astype(np.float32),
                "target_phase": target_phase.astype(np.float32),
                "target_phase_mask": target_phase_mask,
                "reconstructed_i": reconstructed_intensity.astype(np.float32),
                "screen_rgb": screen_rgb,
            }
        )

        summaries.append(
            {
                "name": name,
                "target_peak_normalization_before_passive_scaling": target_peak_unscaled,
                "screen_peak_before_passive_scaling": screen_peak_raw,
                "global_passive_amplitude_scale": passive_scale,
                "screen_peak_after_scaling": float(np.max(screen_amplitude)),
                "screen_rms_amplitude": float(np.sqrt(np.mean(screen_amplitude**2))),
                "phase_only_RMS_amplitude_mismatch": phase_only_rms_mismatch,
                "transmitted_power_fraction_for_unit_incident_intensity": transmitted_fraction,
                "screen_edge_strip_power_fraction_16_pixels": edge_power_fraction,
                "screen_edge_max_amplitude_over_peak": edge_amplitude_over_peak,
                "input_screen_power": screen_power,
                "sampled_focal_plane_power": focal_power,
                "power_relative_error": abs(focal_power - screen_power)
                / max(screen_power, 1e-300),
                **metrics,
                "off_grid_zero_padded_comparison": dense_metrics,
            }
        )

        if name in ("Gaussian", "Random smooth complex field"):
            # Store selected dense central-axis cuts, normalized to unit peak.
            dense_center = dense_N // 2
            x_mm = dense_X_axis * 1e3
            dense_intensity = np.abs(reconstructed_dense[dense_center, :]) ** 2
            dense_intensity /= max(float(np.max(dense_intensity)), 1e-300)
            analytic_dense = np.abs(target_dense[dense_center, :]) ** 2
            analytic_dense /= max(float(np.max(analytic_dense)), 1e-300)
            dense_relative_error = np.abs(
                reconstructed_dense[dense_center, :]
                - target_dense[dense_center, :]
            ) / max(float(np.max(np.abs(target_dense))), 1e-300)
            dense_cuts[name] = {
                "x_mm": x_mm.astype(np.float32),
                "intensity": dense_intensity.astype(np.float32),
                "target_intensity": analytic_dense.astype(np.float32),
                "relative_field_error": dense_relative_error.astype(np.float32),
            }

    # Summary figure: target and reconstructed intensities, target phase, and
    # the inverse complex screen (phase hue, amplitude brightness).
    fig, axes = plt.subplots(4, 4, figsize=(17, 15), constrained_layout=True)
    fig.suptitle(
        "Complex target → inverse Fourier screen → focal-plane reconstruction\n"
        f"λ={WAVELENGTH * 1e9:.0f} nm, f={FOCAL_LENGTH * 1e3:.0f} mm, "
        f"N={N}, dx={DX * 1e6:.1f} µm, screen width={L * 1e3:.2f} mm",
        fontsize=14,
    )
    extent_focus = [
        X_axis[0] * 1e3,
        X_axis[-1] * 1e3,
        Y_axis[0] * 1e3,
        Y_axis[-1] * 1e3,
    ]
    extent_screen = [x[0] * 1e3, x[-1] * 1e3, x[0] * 1e3, x[-1] * 1e3]
    for row, data in enumerate(plot_data):
        axes[row, 0].imshow(
            np.maximum(data["target_i"], 1e-7),
            extent=extent_focus,
            origin="lower",
            cmap="magma",
            norm=matplotlib.colors.LogNorm(vmin=1e-7, vmax=1.0),
        )
        axes[row, 0].set_xlim(-3.0, 3.0)
        axes[row, 0].set_ylim(-3.0, 3.0)
        axes[row, 0].set_title(f"{data['name']}: target intensity")
        axes[row, 0].set_xlabel("X (mm)")
        axes[row, 0].set_ylabel("Y (mm)")

        masked_phase = np.ma.masked_where(
            ~data["target_phase_mask"], data["target_phase"]
        )
        phase_cmap = plt.get_cmap("twilight").copy()
        phase_cmap.set_bad(color="black")
        axes[row, 1].imshow(
            masked_phase,
            extent=extent_focus,
            origin="lower",
            cmap=phase_cmap,
            vmin=-np.pi,
            vmax=np.pi,
        )
        axes[row, 1].set_xlim(-3.0, 3.0)
        axes[row, 1].set_ylim(-3.0, 3.0)
        axes[row, 1].set_title("Target phase (low-amplitude pixels masked)")
        axes[row, 1].set_xlabel("X (mm)")
        axes[row, 1].set_ylabel("Y (mm)")

        axes[row, 2].imshow(
            np.maximum(data["reconstructed_i"], 1e-7),
            extent=extent_focus,
            origin="lower",
            cmap="magma",
            norm=matplotlib.colors.LogNorm(vmin=1e-7, vmax=1.0),
        )
        axes[row, 2].set_xlim(-3.0, 3.0)
        axes[row, 2].set_ylim(-3.0, 3.0)
        axes[row, 2].set_title("Reconstructed intensity")
        axes[row, 2].set_xlabel("X (mm)")
        axes[row, 2].set_ylabel("Y (mm)")

        axes[row, 3].imshow(
            data["screen_rgb"],
            extent=extent_screen,
            origin="lower",
        )
        axes[row, 3].set_title("Inverse screen: phase=hue, amplitude=brightness")
        axes[row, 3].set_xlabel("x (mm)")
        axes[row, 3].set_ylabel("y (mm)")

    fig.savefig(OUT_DIR / "fourier_optics_step2_complex_targets.png", dpi=170)
    plt.close(fig)

    if dense_cuts:
        dense_fig, dense_axes = plt.subplots(
            len(dense_cuts), 2, figsize=(12, 4.5 * len(dense_cuts)),
            constrained_layout=True, squeeze=False
        )
        dense_keep = None
        for row, (name, cut) in enumerate(dense_cuts.items()):
            dense_keep = np.abs(cut["x_mm"]) <= 3.0
            dense_axes[row, 0].plot(
                cut["x_mm"][dense_keep],
                cut["target_intensity"][dense_keep],
                "--",
                label="continuous target",
            )
            dense_axes[row, 0].plot(
                cut["x_mm"][dense_keep],
                cut["intensity"][dense_keep],
                label="zero-padded finite-screen transform",
            )
            dense_axes[row, 0].set_title(f"{name}: intensity cut, Y=0")
            dense_axes[row, 0].set_xlabel("X (mm)")
            dense_axes[row, 0].set_ylabel("Normalized intensity")
            dense_axes[row, 0].legend(fontsize=8)
            dense_axes[row, 1].semilogy(
                cut["x_mm"][dense_keep],
                np.maximum(cut["relative_field_error"][dense_keep], 1e-9),
            )
            dense_axes[row, 1].set_title("Complex-field error between DFT samples")
            dense_axes[row, 1].set_xlabel("X (mm)")
            dense_axes[row, 1].set_ylabel("Absolute error / target peak")
            dense_axes[row, 1].set_ylim(1e-8, 2.0)
        dense_fig.suptitle(
            "Zero-padding samples the finite screen between DFT bins; it adds no bandwidth",
            fontsize=11,
        )
        dense_fig.savefig(OUT_DIR / "fourier_optics_step2_dense_comparison.png", dpi=180)
        plt.close(dense_fig)

    try:
        import importlib.metadata as metadata

        pyoptica = metadata.version("pyoptica")
    except metadata.PackageNotFoundError:
        pyoptica = "not installed; NumPy/SciPy implementation used"

    results = {
        "runtime": {
            "python": __import__("platform").python_version(),
            "numpy": np.__version__,
            "scipy": __import__("scipy").__version__,
            "matplotlib": matplotlib.__version__,
            "pyoptica": pyoptica,
        },
        "model": {
            "target": "physical complex field in the back focal plane",
            "screen_inverse": "centered_ifft2(T_required / dx^2)",
            "screen_samples": N,
            "input_pixel_pitch_m": DX,
            "finite_screen_width_m": L,
            "wavelength_m": WAVELENGTH,
            "focal_length_m": FOCAL_LENGTH,
            "coarse_output_pixel_pitch_m": WAVELENGTH * FOCAL_LENGTH / L,
            "coarse_output_frequency_spacing_per_m": 1.0 / L,
            "zero_padding_factor": PAD_FACTOR,
            "dense_output_pixel_pitch_m": dense_dX,
            "input_screen_constraint": "passive complex transmission, |t| <= 1 by global scaling",
        },
        "targets": summaries,
        "figures": [
            "fourier_optics_step2_complex_targets.png",
            "fourier_optics_step2_dense_comparison.png",
        ],
    }
    with (OUT_DIR / "fourier_optics_step2_results.json").open(
        "w", encoding="utf-8"
    ) as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(json.dumps(results, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
