"""Run an end-to-end example of the reusable Fourier-optics functions."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
from matplotlib import font_manager
import matplotlib.pyplot as plt
import numpy as np

from fourier_optics_framework import (
    centered_fft,
    fused_silica_index,
    intensity_metrics,
    make_complex_screen,
    make_screen,
    relief_screen,
    simulate_multicolor,
)

font_path = r"C:\Windows\Fonts\msyh.ttc"
font_manager.fontManager.addfont(font_path)
plt.rcParams["font.family"] = font_manager.FontProperties(fname=font_path).get_name()
plt.rcParams["axes.unicode_minus"] = False


OUT = Path(__file__).resolve().parent
ARCHIVE = OUT / "fourier_optics_step3_phase_screens.npz"
WAVELENGTHS = np.array([450e-9, 532e-9, 633e-9])


def main():
    rng = np.random.default_rng(2718)
    known_transmission = 0.72 * np.exp(1j * rng.uniform(-np.pi, np.pi, (64, 64)))
    complex_inverse = make_complex_screen(
        centered_fft(known_transmission), np.ones_like(known_transmission)
    )
    complex_inverse_error = float(np.max(np.abs(complex_inverse["screen"] - known_transmission)))

    with np.load(ARCHIVE, allow_pickle=False) as z:
        pupil = z["pupil"].astype(bool)
        targets = {
            "three_spots": z["three_spots_target_intensity"].astype(np.float64),
            "annular_ring": z["annular_ring_target_intensity"].astype(np.float64),
            "letter_A": z["letter_A_target_intensity"].astype(np.float64),
        }
        saved_screens = {
            name: z[f"{name}_screen_complex"].astype(np.complex128)
            for name in targets
        }
        dx = float(z["input_pixel_pitch_m"])
        lambda_ref = float(z["wavelength_m"])
        focal_ref = float(z["focal_length_m"])

    # Exercise the arbitrary-target phase retrieval on an independent 256^2
    # three-spot target, avoiding dependence on the precomputed Step 3 screen.
    n_small = 256
    axis = (np.arange(n_small) - n_small // 2) / n_small
    xx, yy = np.meshgrid(axis, axis)
    toy_target = np.zeros((n_small, n_small), dtype=np.float64)
    for cx, cy in [(-0.22, -0.12), (0.22, -0.12), (0.0, 0.22)]:
        toy_target += np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * 0.025**2))
    toy_pupil = xx**2 + yy**2 <= 0.42**2
    gs = make_screen(toy_target, input_amplitude=toy_pupil.astype(float), iterations=120, seed=314159)
    gs_result = simulate_multicolor(
        gs["screen"], np.array([lambda_ref]), 10e-6, lambda_ref, 0.2, system="diffractive"
    )[0]
    toy_roi = toy_target >= 0.01 * np.max(toy_target)
    gs_metrics = intensity_metrics(toy_target, gs_result["normalized_intensity"], toy_roi)

    n_ref = float(fused_silica_index(lambda_ref))
    height_2pi = lambda_ref / (n_ref - 1.0)
    target_results = {}
    height_maps = {}
    image_rows = []
    image_titles = []
    for name, target in targets.items():
        roi = target >= 0.01 * np.max(target)
        phase = np.angle(saved_screens[name])
        q = np.mod(phase, 2.0 * np.pi) / (2.0 * np.pi)
        height = q * height_2pi
        height_maps[name] = height
        ideal = simulate_multicolor(
            saved_screens[name], WAVELENGTHS, dx, lambda_ref, focal_ref, system="diffractive"
        )
        ordinary = simulate_multicolor(
            saved_screens[name], WAVELENGTHS, dx, lambda_ref, focal_ref, system="ordinary"
        )
        relief = simulate_multicolor(
            lambda wavelength: relief_screen(height, wavelength, pupil),
            WAVELENGTHS, dx, lambda_ref, focal_ref, system="diffractive"
        )
        per_design = {}
        for design_name, results, image_key in [
            ("ideal_wavelength_invariant_screen", ideal, "common_normalized_intensity"),
            ("ordinary_fixed_focal_length", ordinary, "common_normalized_intensity"),
            ("fused_silica_principal_height_relief", relief, "common_normalized_intensity"),
        ]:
            rows = []
            for wavelength, item in zip(WAVELENGTHS, results):
                image = item[image_key]
                row = {"wavelength_nm": float(wavelength * 1e9), **intensity_metrics(target, image, roi)}
                rows.append(row)
                if name == "three_spots":
                    image_rows.append(image)
                    image_titles.append(f"{design_name}\n{wavelength*1e9:.0f} nm")
            per_design[design_name] = rows
        target_results[name] = per_design

    result = {
        "title": "Fourier-optics reusable pipeline demonstration",
        "matched_DFT_complex_inverse_demo": {
            "grid": [64, 64],
            "maximum_transmission": 0.72,
            "maximum_absolute_screen_reconstruction_error": complex_inverse_error,
            "note": "A finite matched DFT target is inverted exactly to numerical precision when the required screen transmission is passive.",
        },
        "GS_256_three_spot_demo": {
            "grid": [256, 256],
            "iterations": gs["iterations"],
            "seed": gs["seed"],
            "final_relative_L2": gs_metrics["relative_L2"],
            "final_ROI_power": gs_metrics["power_fraction_in_roi"],
        },
        "reused_1024_targets": target_results,
        "limitations": [
            "The three stored phase screens were designed at the reference wavelength; the relief conversion uses the principal 0-to-2pi height branch.",
            "The Fourier lens is idealized; diffractive mode imposes f proportional to inverse wavelength by definition.",
            "Each distribution is normalized, so reported ROI fractions are not absolute throughput or diffraction efficiency.",
            "Scalar paraxial propagation omits vector, polarization, reflection, absorption, fabrication, and feature-size effects.",
        ],
    }
    (OUT / "fourier_optics_framework_example_results.json").write_text(json.dumps(result, indent=2), encoding="utf-8")

    fig, axes = plt.subplots(4, 4, figsize=(12.8, 11.0), constrained_layout=True)
    axes[0, 0].imshow(targets["three_spots"], cmap="magma", origin="lower")
    axes[0, 0].set_title("三光斑目标")
    axes[0, 1].imshow(np.angle(saved_screens["three_spots"]), cmap="twilight", origin="lower")
    axes[0, 1].set_title("示例相位屏")
    axes[0, 2].imshow(height_maps["three_spots"] * 1e6, cmap="viridis", origin="lower")
    axes[0, 2].set_title("浮雕高度 (μm)")
    axes[0, 3].imshow(targets["three_spots"], cmap="magma", origin="lower")
    axes[0, 3].set_title("目标")
    axes[1, 0].axis("off")
    axes[2, 0].axis("off")
    axes[3, 0].axis("off")
    for ax, image, title in zip(axes[1:, 1:].ravel(), image_rows, image_titles):
        ax.imshow(image, cmap="magma", origin="lower", vmin=0, vmax=max(float(np.max(image)), 1e-12))
        ax.set_title(title, fontsize=8)
    axes[1, 0].text(0.0, 0.5, "理想屏\nλf 恒定", transform=axes[1, 0].transAxes, va="center")
    axes[2, 0].text(0.0, 0.5, "固定焦距\n普通透镜", transform=axes[2, 0].transAxes, va="center")
    axes[3, 0].text(0.0, 0.5, "石英主值\n浮雕", transform=axes[3, 0].transAxes, va="center")
    for ax in axes.ravel():
        ax.set_xticks([])
        ax.set_yticks([])
    fig.savefig(OUT / "fourier_optics_framework_example.png", dpi=170)
    plt.close(fig)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
