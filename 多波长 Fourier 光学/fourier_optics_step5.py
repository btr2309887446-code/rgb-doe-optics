"""Step 5: field bases and finite-aperture Fourier modes."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.special import j1


OUT_DIR = Path(__file__).resolve().parent
N = 256
DX = 10.0e-6
L = N * DX
WAVELENGTH = 532.0e-9
FOCAL_LENGTH = 0.200
APERTURE_DIAMETER = 1.28e-3
PAD_FACTOR = 4
MODE_INDEX_MAX = 4


def centered_fft(field: np.ndarray) -> np.ndarray:
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def gram_matrix_stats(mask: np.ndarray, pairs: list[tuple[int, int]]) -> tuple[np.ndarray, dict[str, float]]:
    y = (np.arange(N) - N // 2) * DX
    x = y.copy()
    X, Y = np.meshgrid(x, y)
    modes = np.empty((len(pairs), N * N), dtype=np.complex64)
    for i, (m, n) in enumerate(pairs):
        mode = mask * np.exp(2j * np.pi * (m * X / L + n * Y / L))
        modes[i] = mode.ravel().astype(np.complex64)
    norms = np.linalg.norm(modes, axis=1)
    normalized = modes / norms[:, None]
    gram = normalized.conj() @ normalized.T
    abs_gram = np.abs(gram)
    np.fill_diagonal(abs_gram, 0.0)
    upper = abs_gram[np.triu_indices(len(pairs), 1)]
    nearest = []
    pair_lookup = {pair: i for i, pair in enumerate(pairs)}
    for i, (m, n) in enumerate(pairs):
        for neighbor in [(m + 1, n), (m, n + 1)]:
            if neighbor in pair_lookup:
                nearest.append(float(abs(gram[i, pair_lookup[neighbor]])))
    stats = {
        "max_absolute_off_diagonal_inner_product": float(np.max(upper)),
        "median_absolute_off_diagonal_inner_product": float(np.median(upper)),
        "mean_adjacent_mode_coherence": float(np.mean(nearest)),
    }
    return gram, stats


def main() -> None:
    coordinates = (np.arange(N) - N // 2) * DX
    X, Y = np.meshgrid(coordinates, coordinates)
    radius = np.hypot(X, Y)
    pupil = radius <= APERTURE_DIAMETER / 2
    mode_pairs = [
        (m, n)
        for n in range(-MODE_INDEX_MAX, MODE_INDEX_MAX + 1)
        for m in range(-MODE_INDEX_MAX, MODE_INDEX_MAX + 1)
    ]

    # A unitary DFT is an orthonormal basis transform on C^(N^2).
    rng = np.random.default_rng(260930)
    arbitrary_field = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    unitary_coefficients = np.fft.fft2(arbitrary_field) / N
    field_reconstruction = np.fft.ifft2(unitary_coefficients) * N
    basis_reconstruction_error = float(
        np.linalg.norm((field_reconstruction - arbitrary_field).ravel())
        / np.linalg.norm(arbitrary_field.ravel())
    )
    unitary_parseval_error = float(
        abs(
            np.sum(np.abs(arbitrary_field) ** 2)
            - np.sum(np.abs(unitary_coefficients) ** 2)
        )
        / np.sum(np.abs(arbitrary_field) ** 2)
    )

    rectangular_gram, rectangular_stats = gram_matrix_stats(np.ones_like(pupil, dtype=float), mode_pairs)
    circular_gram, circular_stats = gram_matrix_stats(pupil.astype(float), mode_pairs)

    # A single sampled Fourier mode, truncated by the circular aperture.
    grating_m, grating_n = 3, -2
    screen = pupil * np.exp(2j * np.pi * (grating_m * X / L + grating_n * Y / L))
    M = N * PAD_FACTOR
    padded = np.zeros((M, M), dtype=np.complex128)
    offset = (M - N) // 2
    padded[offset : offset + N, offset : offset + N] = screen
    spectrum = centered_fft(padded)
    intensity = np.abs(spectrum) ** 2
    frequency = np.fft.fftshift(np.fft.fftfreq(M, d=DX))
    focal_axis_m = WAVELENGTH * FOCAL_LENGTH * frequency
    peak_y, peak_x = np.unravel_index(int(np.argmax(intensity)), intensity.shape)
    measured_peak_xy_m = [float(focal_axis_m[peak_x]), float(focal_axis_m[peak_y])]
    predicted_peak_xy_m = [
        WAVELENGTH * FOCAL_LENGTH * grating_m / L,
        WAVELENGTH * FOCAL_LENGTH * grating_n / L,
    ]
    peak_position_error_pixels = float(
        np.hypot(
            measured_peak_xy_m[0] - predicted_peak_xy_m[0],
            measured_peak_xy_m[1] - predicted_peak_xy_m[1],
        )
        / (focal_axis_m[1] - focal_axis_m[0])
    )

    # The circular pupil transform is an Airy amplitude shifted to the grating order.
    line = intensity[peak_y, :]
    minima = [
        i
        for i in range(peak_x + 1, M - 1)
        if line[i] <= line[i - 1] and line[i] < line[i + 1]
    ]
    first_minimum_index = minima[0]
    measured_first_minimum_offset_m = float(focal_axis_m[first_minimum_index] - focal_axis_m[peak_x])
    predicted_first_zero_offset_m = 1.22 * WAVELENGTH * FOCAL_LENGTH / APERTURE_DIAMETER
    input_power = float(np.sum(np.abs(screen) ** 2))
    fft_power = float(np.sum(intensity) / (M * M))
    parseval_error_screen = abs(fft_power - input_power) / input_power

    delta_X_m = focal_axis_m - focal_axis_m[peak_x]
    delta_frequency = delta_X_m / (WAVELENGTH * FOCAL_LENGTH)
    q = np.pi * APERTURE_DIAMETER * delta_frequency
    airy_amplitude = np.ones_like(q)
    nonzero = np.abs(q) > 1e-12
    airy_amplitude[nonzero] = 2.0 * j1(q[nonzero]) / q[nonzero]
    airy_intensity = np.abs(airy_amplitude) ** 2
    measured_line_normalized = line / max(float(line[peak_x]), 1e-300)

    results = {
        "definition": {
            "discrete_complex_field_space": f"C^({N*N})",
            "fourier_basis": "unitary 2-D DFT basis on a periodic rectangular sample grid",
            "finite_aperture_modes": "pupil(x,y)*exp(2*pi*i*(m*x/L+n*y/L))",
            "inner_product": "discrete sum over sampled pupil; modes individually normalized",
        },
        "sampling": {
            "N": N,
            "pixel_pitch_m": DX,
            "window_width_m": L,
            "nyquist_frequency_per_m": 1.0 / (2.0 * DX),
            "circular_pupil_diameter_m": APERTURE_DIAMETER,
            "circular_pupil_samples_across_nominal_diameter": APERTURE_DIAMETER / DX,
            "wavelength_m": WAVELENGTH,
            "focal_length_m": FOCAL_LENGTH,
            "padded_fft_factor": PAD_FACTOR,
            "padded_fourier_plane_pixel_pitch_m": float(
                focal_axis_m[1] - focal_axis_m[0]
            ),
        },
        "complete_discrete_dft_basis_check": {
            "random_complex_field_relative_reconstruction_error": basis_reconstruction_error,
            "unitary_parseval_relative_error": unitary_parseval_error,
        },
        "selected_modes": {
            "integer_orders_m_n": [list(pair) for pair in mode_pairs],
            "sampled_full_rectangle_gram_stats": rectangular_stats,
            "sampled_circular_pupil_gram_stats": circular_stats,
            "circular_pupil_area_fraction_of_square": float(np.mean(pupil)),
        },
        "finite_aperture_single_mode": {
            "grating_orders_m_n": [grating_m, grating_n],
            "predicted_peak_position_m": predicted_peak_xy_m,
            "measured_peak_position_m": measured_peak_xy_m,
            "peak_position_error_pixels": peak_position_error_pixels,
            "analytic_circular_aperture_first_zero_offset_m": predicted_first_zero_offset_m,
            "measured_first_axis_minimum_offset_m": measured_first_minimum_offset_m,
            "first_minimum_offset_error_pixels": abs(
                measured_first_minimum_offset_m - predicted_first_zero_offset_m
            ) / float(focal_axis_m[1] - focal_axis_m[0]),
            "parseval_relative_error_for_screen": float(parseval_error_screen),
        },
        "figure": "fourier_optics_step5_basis_and_finite_aperture.png",
    }
    with (OUT_DIR / "fourier_optics_step5_results.json").open("w", encoding="utf-8") as stream:
        json.dump(results, stream, indent=2, ensure_ascii=False)

    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), constrained_layout=True)
    axes[0].imshow(np.abs(circular_gram), origin="lower", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_title("|Gram matrix|: circular pupil")
    axes[0].set_xlabel("Mode index")
    axes[0].set_ylabel("Mode index")

    extent = [focal_axis_m[0] * 1e3, focal_axis_m[-1] * 1e3,
              focal_axis_m[0] * 1e3, focal_axis_m[-1] * 1e3]
    axes[1].imshow(
        np.log10(np.maximum(intensity / max(float(np.max(intensity)), 1e-300), 1e-8)),
        extent=extent,
        origin="lower",
        cmap="magma",
        vmin=-8,
        vmax=0,
    )
    axes[1].set_xlim(measured_peak_xy_m[0] * 1e3 - 0.35, measured_peak_xy_m[0] * 1e3 + 0.35)
    axes[1].set_ylim(measured_peak_xy_m[1] * 1e3 - 0.35, measured_peak_xy_m[1] * 1e3 + 0.35)
    axes[1].set_title("Finite circular pupil: shifted Airy spot")
    axes[1].set_xlabel("X (mm)")
    axes[1].set_ylabel("Y (mm)")

    radius_window = 0.32e-3
    keep = np.abs(delta_X_m) <= radius_window
    axes[2].semilogy(
        delta_X_m[keep] * 1e3,
        np.maximum(measured_line_normalized[keep], 1e-8),
        label="Numerical pupil transform",
    )
    axes[2].semilogy(
        delta_X_m[keep] * 1e3,
        np.maximum(airy_intensity[keep], 1e-8),
        "--",
        label="Continuous-disk Airy model",
    )
    axes[2].axvline(predicted_first_zero_offset_m * 1e3, color="gray", linestyle=":")
    axes[2].set_title("Line profile through grating order")
    axes[2].set_xlabel("Offset from order peak (mm)")
    axes[2].set_ylabel("Normalized intensity")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(fontsize=8)
    fig.savefig(OUT_DIR / "fourier_optics_step5_basis_and_finite_aperture.png", dpi=170)
    plt.close(fig)

    print(json.dumps({"results": str(OUT_DIR / "fourier_optics_step5_results.json")}, ensure_ascii=True))


if __name__ == "__main__":
    main()
