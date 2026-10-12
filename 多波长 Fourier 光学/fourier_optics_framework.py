"""Small scalar Fourier-optics toolkit for phase screens and relief surfaces.

Conventions
-----------
Arrays are sampled on centered, square, even-sized grids. ``input_pitch_m`` is
the screen-plane sample pitch. The focal-plane sample pitch is
``wavelength_m * focal_length_m / (N * input_pitch_m)``. Intensities returned
by :func:`simulate` are normalized to unit discrete sum; the field amplitude
is the sampled Fourier-lens integral, up to its irrelevant global phase.

This module models a thin scalar screen and paraxial Fourier lens. It does not
model vector fields, reflection, absorption, substrate interfaces, or feature
scale-dependent electromagnetic efficiency.
"""

from __future__ import annotations

from typing import Callable

import numpy as np
from scipy.ndimage import map_coordinates


Array = np.ndarray
ScreenSource = Array | Callable[[float], Array]


def centered_fft(field: Array) -> Array:
    """Centered, unnormalized 2-D DFT (matching NumPy's FFT convention)."""
    return np.fft.fftshift(np.fft.fft2(np.fft.ifftshift(field)))


def centered_ifft(field: Array) -> Array:
    """Inverse of :func:`centered_fft`, including NumPy's 1/N^2 factor."""
    return np.fft.fftshift(np.fft.ifft2(np.fft.ifftshift(field)))


def spatial_axis(n: int, pitch_m: float) -> Array:
    """Centered screen-plane coordinate samples in metres."""
    return (np.arange(n, dtype=np.float64) - n // 2) * pitch_m


def frequency_axis(n: int, pitch_m: float) -> Array:
    """Centered spatial-frequency samples in cycles per metre."""
    return np.fft.fftshift(np.fft.fftfreq(n, d=pitch_m))


def _validate_square(array: Array, name: str) -> Array:
    result = np.asarray(array)
    if result.ndim != 2 or result.shape[0] != result.shape[1]:
        raise ValueError(f"{name} must be a square 2-D array")
    if result.shape[0] % 2:
        raise ValueError(f"{name} must have an even side length")
    return result


def make_screen(
    target_intensity: Array,
    input_amplitude: Array | None = None,
    iterations: int = 200,
    seed: int = 0,
    initial_phase: Array | None = None,
) -> dict[str, Array | list[float] | float | int]:
    """Find a phase-only screen for a sampled target intensity using GS.

    The input and target use the same square DFT size. ``input_amplitude`` is
    the desired screen-plane field magnitude (for example, a pupil mask). The
    returned ``screen`` is a complex transmission of unit magnitude wherever
    the input amplitude is nonzero and zero elsewhere. The target specifies
    intensity only; its phase is free and is iteratively estimated.
    """
    target = np.asarray(_validate_square(target_intensity, "target_intensity"), dtype=np.float64)
    if not np.all(np.isfinite(target)) or np.any(target < 0.0) or float(np.sum(target)) <= 0.0:
        raise ValueError("target_intensity must be finite, nonnegative, and have positive sum")
    if iterations < 1:
        raise ValueError("iterations must be positive")
    if input_amplitude is None:
        amp_in = np.ones_like(target)
    else:
        amp_in = np.asarray(_validate_square(input_amplitude, "input_amplitude"), dtype=np.float64)
        if amp_in.shape != target.shape or np.any(~np.isfinite(amp_in)) or np.any(amp_in < 0.0):
            raise ValueError("input_amplitude must be finite, nonnegative, and match target_intensity")
    power = float(np.sum(amp_in**2))
    if power <= 0.0:
        raise ValueError("input_amplitude must illuminate at least one sample")
    target_norm = target / float(np.sum(target))
    # NumPy's forward FFT is unnormalized: Parseval gives total Fourier-plane
    # intensity N^2 times the input-plane discrete power.
    target_amp = np.sqrt(target_norm * (target.shape[0] ** 2 * power))
    rng = np.random.default_rng(seed)
    if initial_phase is None:
        phase = rng.uniform(-np.pi, np.pi, size=target.shape)
    else:
        phase = np.asarray(_validate_square(initial_phase, "initial_phase"), dtype=np.float64)
        if phase.shape != target.shape or np.any(~np.isfinite(phase)):
            raise ValueError("initial_phase must be finite and match target_intensity")
    field = amp_in * np.exp(1j * phase)
    errors: list[float] = []
    for _ in range(iterations):
        far = centered_fft(field)
        far = target_amp * np.exp(1j * np.angle(far))
        field = centered_ifft(far)
        field = amp_in * np.exp(1j * np.angle(field))
        estimated = np.abs(centered_fft(field)) ** 2
        estimated /= max(float(np.sum(estimated)), 1e-300)
        errors.append(float(np.linalg.norm(estimated - target_norm) / max(np.linalg.norm(target_norm), 1e-300)))
    phase = np.angle(field)
    screen = np.zeros(target.shape, dtype=np.complex128)
    illuminated = amp_in > 0.0
    screen[illuminated] = np.exp(1j * phase[illuminated])
    return {
        "screen": screen,
        "phase_rad": phase,
        "input_amplitude": amp_in,
        "target_normalized": target_norm,
        "relative_l2_history": errors,
        "iterations": iterations,
        "seed": seed,
    }


def make_complex_screen(
    target_field: Array,
    input_field: Array,
    max_transmission: float = 1.0,
    normalize_to_passive: bool = False,
) -> dict[str, Array | float | bool]:
    """Invert a matched finite DFT for a complex target field.

    This is exact on the discrete model: the inverse DFT of the target is the
    required screen-plane field. If ``normalize_to_passive`` is true, a single
    global scale is applied so that the largest transmission magnitude is at
    most ``max_transmission``; the returned ``target_scale`` records that loss
    of absolute field amplitude. Illumination zeros cannot be divided through.
    """
    desired = np.asarray(_validate_square(target_field, "target_field"), dtype=np.complex128)
    incident = np.asarray(_validate_square(input_field, "input_field"), dtype=np.complex128)
    if desired.shape != incident.shape:
        raise ValueError("target_field and input_field must have the same shape")
    if np.any(~np.isfinite(desired)) or np.any(~np.isfinite(incident)):
        raise ValueError("fields must be finite")
    if max_transmission <= 0.0 or not np.isfinite(max_transmission):
        raise ValueError("max_transmission must be finite and positive")
    required = centered_ifft(desired)
    if np.any((np.abs(incident) == 0.0) & (np.abs(required) > 1e-12 * max(float(np.max(np.abs(required))), 1.0))):
        raise ValueError("target requires nonzero field where input_field is zero")
    screen = np.zeros_like(required)
    active = np.abs(incident) > 0.0
    screen[active] = required[active] / incident[active]
    maximum = float(np.max(np.abs(screen)))
    scale = 1.0
    if maximum > max_transmission:
        if not normalize_to_passive:
            raise ValueError(f"required transmission magnitude {maximum:.6g} exceeds the passive limit")
        scale = max_transmission / maximum
        screen *= scale
    return {
        "screen": screen,
        "max_transmission_used": float(np.max(np.abs(screen))),
        "target_scale": float(scale),
        "passive_bound_satisfied": bool(np.max(np.abs(screen)) <= max_transmission * (1.0 + 1e-12)),
    }


def fused_silica_index(wavelength_m: float | Array) -> Array:
    """Fused-silica refractive index from Malitson's 20 C Sellmeier fit.

    The published fit covers wavelengths from 0.21 to 3.71 micrometres.
    """
    wavelength_um = np.asarray(wavelength_m, dtype=np.float64) * 1e6
    b = np.array([0.6961663, 0.4079426, 0.8974794])
    c_um = np.array([0.0684043, 0.1162414, 9.896161])
    lam2 = wavelength_um[..., None] ** 2
    n2 = 1.0 + np.sum(b * lam2 / (lam2 - c_um**2), axis=-1)
    return np.sqrt(n2)


def relief_screen(height_m: Array, wavelength_m: float, pupil: Array | None = None) -> Array:
    """Complex screen for a fused-silica relief in air at one wavelength."""
    height = _validate_square(np.asarray(height_m, dtype=np.float64), "height_m")
    if np.any(~np.isfinite(height)) or np.any(height < 0.0):
        raise ValueError("height_m must be finite and nonnegative")
    if wavelength_m <= 0.0 or not np.isfinite(wavelength_m):
        raise ValueError("wavelength_m must be finite and positive")
    n = float(fused_silica_index(wavelength_m))
    phase = 2.0 * np.pi * (n - 1.0) * height / wavelength_m
    screen = np.exp(1j * phase)
    if pupil is not None:
        mask = np.asarray(_validate_square(pupil, "pupil"), dtype=bool)
        if mask.shape != height.shape:
            raise ValueError("pupil must match height_m")
        screen = np.where(mask, screen, 0.0)
    return screen


def _simulate_one(
    screen: Array,
    wavelength_m: float,
    input_pitch_m: float,
    focal_length_m: float,
    input_field: Array | None = None,
    common_axis_m: Array | None = None,
) -> dict[str, Array | float]:
    transmittance = np.asarray(_validate_square(screen, "screen"), dtype=np.complex128)
    n = transmittance.shape[0]
    if input_field is None:
        incident = np.ones_like(transmittance)
    else:
        incident = np.asarray(_validate_square(input_field, "input_field"), dtype=np.complex128)
        if incident.shape != transmittance.shape:
            raise ValueError("input_field must match screen")
    if wavelength_m <= 0.0 or input_pitch_m <= 0.0 or focal_length_m <= 0.0:
        raise ValueError("wavelength, input pitch, and focal length must be positive")
    spectrum = centered_fft(incident * transmittance)
    field = (input_pitch_m**2 / (wavelength_m * focal_length_m)) * spectrum
    intensity = np.abs(field) ** 2
    power_sum = float(np.sum(intensity))
    normalized = intensity / max(power_sum, 1e-300)
    delta_x = wavelength_m * focal_length_m / (n * input_pitch_m)
    axis = spatial_axis(n, delta_x)
    result: dict[str, Array | float] = {
        "wavelength_m": float(wavelength_m),
        "focal_length_m": float(focal_length_m),
        "sample_pitch_m": float(delta_x),
        "x_m": axis,
        "y_m": axis,
        "field": field,
        "intensity": intensity,
        "normalized_intensity": normalized,
    }
    if common_axis_m is not None:
        # For separable axis-aligned grids, map_coordinates applies bilinear
        # interpolation at the common physical coordinates.
        indices = common_axis_m / delta_x + n // 2
        yy, xx = np.meshgrid(indices, indices, indexing="ij")
        remapped = map_coordinates(normalized, [yy, xx], order=1, mode="constant", cval=0.0)
        remapped /= max(float(np.sum(remapped)), 1e-300)
        result["common_x_m"] = common_axis_m
        result["common_y_m"] = common_axis_m
        result["common_normalized_intensity"] = remapped
    return result


def simulate(
    screen: Array,
    wavelength_m: float,
    input_pitch_m: float,
    focal_length_m: float,
    input_field: Array | None = None,
) -> dict[str, Array | float]:
    """Propagate one screen through a paraxial thin Fourier lens."""
    return _simulate_one(screen, wavelength_m, input_pitch_m, focal_length_m, input_field)


def simulate_multicolor(
    screen: ScreenSource,
    wavelengths_m: Array,
    input_pitch_m: float,
    reference_wavelength_m: float,
    reference_focal_length_m: float,
    system: str = "ordinary",
    input_field: Array | None = None,
) -> list[dict[str, Array | float]]:
    """Propagate one screen source at several wavelengths.

    ``system="ordinary"`` keeps the focal length fixed. ``system="diffractive"``
    uses the idealized ``f(lambda)=f_ref*lambda_ref/lambda`` law, so all colors
    share a physical Fourier-plane sample grid. A callable screen receives the
    wavelength in metres and can implement a dispersive material relief.
    Results include native coordinates and intensities; for a fixed-focus
    system they also include normalized intensity interpolated onto the
    reference wavelength's physical grid.
    """
    wavelengths = np.asarray(wavelengths_m, dtype=np.float64).ravel()
    if wavelengths.size == 0 or np.any(~np.isfinite(wavelengths)) or np.any(wavelengths <= 0.0):
        raise ValueError("wavelengths_m must contain finite positive values")
    if system not in {"ordinary", "diffractive"}:
        raise ValueError("system must be 'ordinary' or 'diffractive'")
    prototype = np.asarray(screen(float(wavelengths[0])) if callable(screen) else screen)
    n = _validate_square(prototype, "screen").shape[0]
    ref_pitch = reference_wavelength_m * reference_focal_length_m / (n * input_pitch_m)
    common_axis = spatial_axis(n, ref_pitch)
    results = []
    for wavelength in wavelengths:
        focal = reference_focal_length_m
        if system == "diffractive":
            focal = reference_focal_length_m * reference_wavelength_m / float(wavelength)
        transmittance = screen(float(wavelength)) if callable(screen) else prototype
        results.append(_simulate_one(
            transmittance,
            float(wavelength),
            input_pitch_m,
            focal,
            input_field,
            common_axis_m=common_axis,
        ))
    return results


def intensity_metrics(target_intensity: Array, image_intensity: Array, roi: Array | None = None) -> dict[str, float]:
    """Compare normalized intensity distributions, with optional ROI power."""
    target = np.asarray(target_intensity, dtype=np.float64)
    image = np.asarray(image_intensity, dtype=np.float64)
    if target.shape != image.shape or target.ndim != 2:
        raise ValueError("target_intensity and image_intensity must have the same 2-D shape")
    t = target / max(float(np.sum(target)), 1e-300)
    p = image / max(float(np.sum(image)), 1e-300)
    result = {
        "relative_L2": float(np.linalg.norm((p - t).ravel()) / max(float(np.linalg.norm(t.ravel())), 1e-300)),
        "squared_Bhattacharyya_fidelity": float(np.sum(np.sqrt(np.maximum(t, 0.0) * np.maximum(p, 0.0))) ** 2),
    }
    if roi is not None:
        mask = np.asarray(roi, dtype=bool)
        if mask.shape != target.shape:
            raise ValueError("roi must match the intensity shape")
        result["power_fraction_in_roi"] = float(np.sum(p[mask]))
    return result
