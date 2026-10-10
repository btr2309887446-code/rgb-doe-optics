"""Differentiable scalar forward models for DOE image-plane experiments."""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Iterable, Literal

import torch
import torch.nn.functional as functional


@dataclass(frozen=True)
class OpticalSystem:
    n_grid: int = 256
    pixel_pitch_m: float = 10e-6
    aperture_radius_m: float = 1e-3
    model: Literal["free_space", "thin_lens"] = "free_space"
    propagation_distance_m: float = 0.05
    doe_to_lens_distance_m: float = 0.10
    lens_to_image_distance_m: float = 0.10
    lens_focal_length_m: float = 0.05
    lens_aperture_radius_m: float | None = None
    pad_factor: int = 1

    def __post_init__(self) -> None:
        if self.n_grid < 8 or self.n_grid % 2:
            raise ValueError("n_grid must be an even integer >= 8")
        if self.pixel_pitch_m <= 0 or self.aperture_radius_m <= 0:
            raise ValueError("sampling and aperture must be positive")
        if self.model not in ("free_space", "thin_lens"):
            raise ValueError("model must be free_space or thin_lens")
        if self.pad_factor not in (1, 2, 4):
            raise ValueError("pad_factor must be 1, 2, or 4")
        if self.model == "free_space" and self.propagation_distance_m <= 0:
            raise ValueError("free-space propagation distance must be positive")
        if self.model == "thin_lens":
            if min(self.doe_to_lens_distance_m, self.lens_to_image_distance_m, self.lens_focal_length_m) <= 0:
                raise ValueError("thin-lens distances must be positive")
            left = 1.0 / self.doe_to_lens_distance_m + 1.0 / self.lens_to_image_distance_m
            right = 1.0 / self.lens_focal_length_m
            if abs(left - right) > 1e-5 * max(right, 1.0):
                raise ValueError("thin-lens distances must satisfy 1/so+1/si=1/f")

    @property
    def lens_radius_m(self) -> float:
        return self.aperture_radius_m if self.lens_aperture_radius_m is None else self.lens_aperture_radius_m


@dataclass(frozen=True)
class OpticalStopConfig:
    """Serializable description of a spatial stop.

    ``cutoff_frequency_cpm`` and ``inner_cutoff_frequency_cpm`` are specified
    in cycles/m in the Fourier plane.  ``radius_m`` and ``inner_radius_m`` are
    physical radii for an image-plane stop.  The soft edge is expressed in the
    same coordinate units as the corresponding stop and is set to zero for a
    hard mask.
    """

    stop_type: Literal[
        "none", "fourier_circular", "fourier_annular", "image", "image_circular"
    ] = "none"
    cutoff_frequency_cpm: float | None = None
    inner_cutoff_frequency_cpm: float = 0.0
    radius_m: float | None = None
    inner_radius_m: float = 0.0
    soft_edge: float = 0.0
    position: Literal["none", "fourier", "image"] = "none"

    def __post_init__(self) -> None:
        normalized = "image" if self.stop_type == "image_circular" else self.stop_type
        if normalized not in {"none", "fourier_circular", "fourier_annular", "image"}:
            raise ValueError(f"unsupported stop_type: {self.stop_type}")
        if self.cutoff_frequency_cpm is not None and self.cutoff_frequency_cpm < 0:
            raise ValueError("cutoff_frequency_cpm must be nonnegative")
        if self.inner_cutoff_frequency_cpm < 0:
            raise ValueError("inner_cutoff_frequency_cpm must be nonnegative")
        if self.radius_m is not None and self.radius_m < 0:
            raise ValueError("radius_m must be nonnegative")
        if self.inner_radius_m < 0:
            raise ValueError("inner_radius_m must be nonnegative")
        if self.soft_edge < 0:
            raise ValueError("soft_edge must be nonnegative")
        if self.position not in {"none", "fourier", "image"}:
            raise ValueError("position must be none, fourier, or image")
        if self.cutoff_frequency_cpm is not None and self.inner_cutoff_frequency_cpm > self.cutoff_frequency_cpm:
            raise ValueError("inner frequency must not exceed outer frequency")
        if self.radius_m is not None and self.inner_radius_m > self.radius_m:
            raise ValueError("inner radius must not exceed outer radius")

    @property
    def kind(self) -> str:
        """Alias useful in reports and backwards-compatible callers."""
        return "image" if self.stop_type == "image_circular" else self.stop_type

    def as_dict(self) -> dict[str, Any]:
        return {
            "stop_type": self.stop_type,
            "cutoff_frequency_cpm": self.cutoff_frequency_cpm,
            "inner_cutoff_frequency_cpm": self.inner_cutoff_frequency_cpm,
            "radius_m": self.radius_m,
            "inner_radius_m": self.inner_radius_m,
            "soft_edge": self.soft_edge,
            "position": self.position,
        }


@dataclass(frozen=True)
class SimulationGrid:
    base_x: torch.Tensor
    base_y: torch.Tensor
    work_x: torch.Tensor
    work_y: torch.Tensor
    fx: torch.Tensor
    fy: torch.Tensor
    doe_aperture: torch.Tensor
    lens_aperture: torch.Tensor
    base_n: int
    work_n: int
    crop_start: int
    pixel_pitch_m: float

    @property
    def crop_end(self) -> int:
        return self.crop_start + self.base_n


def make_grid(system: OpticalSystem, device: torch.device) -> SimulationGrid:
    base_coordinate = (torch.arange(system.n_grid, device=device) - system.n_grid // 2) * system.pixel_pitch_m
    base_x, base_y = torch.meshgrid(base_coordinate, base_coordinate, indexing="xy")
    work_n = system.n_grid * system.pad_factor
    work_coordinate = (torch.arange(work_n, device=device) - work_n // 2) * system.pixel_pitch_m
    work_x, work_y = torch.meshgrid(work_coordinate, work_coordinate, indexing="xy")
    frequency = torch.fft.fftfreq(work_n, d=system.pixel_pitch_m, device=device)
    fx, fy = torch.meshgrid(frequency, frequency, indexing="xy")
    doe_aperture = ((work_x**2 + work_y**2) <= system.aperture_radius_m**2).to(torch.float32)
    lens_aperture = ((work_x**2 + work_y**2) <= system.lens_radius_m**2).to(torch.float32)
    return SimulationGrid(
        base_x=base_x,
        base_y=base_y,
        work_x=work_x,
        work_y=work_y,
        fx=fx,
        fy=fy,
        doe_aperture=doe_aperture,
        lens_aperture=lens_aperture,
        base_n=system.n_grid,
        work_n=work_n,
        crop_start=(work_n - system.n_grid) // 2,
        pixel_pitch_m=system.pixel_pitch_m,
    )


def _soft_disk(
    radius: torch.Tensor,
    outer: torch.Tensor,
    inner: torch.Tensor | None = None,
    soft_edge: float = 0.0,
) -> torch.Tensor:
    """Return a differentiable disk/annulus mask for a radial coordinate."""
    if soft_edge <= 0:
        mask = (radius <= outer).to(dtype=radius.dtype)
        if inner is not None:
            mask = mask * (radius >= inner).to(dtype=radius.dtype)
        return mask
    edge = torch.as_tensor(soft_edge, device=radius.device, dtype=radius.dtype)
    mask = torch.sigmoid((outer - radius) / edge)
    if inner is not None:
        mask = mask * torch.where(inner > 0, torch.sigmoid((radius - inner) / edge), 1.0)
    return mask


def make_fourier_stop(
    grid: SimulationGrid,
    wavelength_m: float | torch.Tensor | Iterable[float],
    focal_length_m: float,
    cutoff_frequency_cpm: float | None = None,
    inner_cutoff_frequency_cpm: float = 0.0,
    soft_edge_cpm: float = 0.0,
    stop_type: Literal["circular", "annular", "fourier_circular", "fourier_annular"] = "circular",
    *,
    radius_m: float | None = None,
    inner_radius_m: float | None = None,
    hard: bool = False,
    soft_edge: float | None = None,
    tau: float | None = None,
) -> torch.Tensor:
    """Construct a wavelength-aware Fourier-plane circular/annular stop.

    A scalar wavelength returns ``(work_n, work_n)``.  A wavelength vector
    returns ``(n_wavelengths, work_n, work_n)`` so it can be broadcast over a
    batched RGB field.  Frequency cutoffs are converted to physical Fourier
    plane radii with ``r = lambda * f * f_c``.
    """
    if focal_length_m <= 0:
        raise ValueError("focal_length_m must be positive")
    normalized_type = "annular" if stop_type == "fourier_annular" else (
        "circular" if stop_type == "fourier_circular" else stop_type
    )
    if normalized_type not in {"circular", "annular"}:
        raise ValueError("stop_type must be circular or annular")
    if cutoff_frequency_cpm is None and radius_m is None:
        raise ValueError("cutoff_frequency_cpm or radius_m is required")
    if cutoff_frequency_cpm is not None and cutoff_frequency_cpm < 0:
        raise ValueError("cutoff_frequency_cpm must be nonnegative")
    if inner_cutoff_frequency_cpm < 0:
        raise ValueError("inner_cutoff_frequency_cpm must be nonnegative")
    if radius_m is not None and radius_m < 0:
        raise ValueError("radius_m must be nonnegative")
    if inner_radius_m is not None and inner_radius_m < 0:
        raise ValueError("inner_radius_m must be nonnegative")
    if soft_edge is not None:
        soft_edge_cpm = soft_edge
    if tau is not None:
        soft_edge_cpm = tau
    if hard:
        soft_edge_cpm = 0.0
    if soft_edge_cpm < 0:
        raise ValueError("soft_edge_cpm must be nonnegative")

    wavelengths = torch.as_tensor(wavelength_m, device=grid.work_x.device, dtype=torch.float64)
    scalar = wavelengths.ndim == 0
    wavelengths = wavelengths.reshape(-1)
    if wavelengths.numel() == 0 or torch.any(wavelengths <= 0):
        raise ValueError("wavelength_m must contain positive values")
    # A focal transform has pitch lambda*f/(N*dx), not the object-plane pitch.
    radial_frequency = torch.fft.fftshift(torch.hypot(grid.fx, grid.fy)).to(torch.float64)
    radius = radial_frequency[None] * wavelengths[:, None, None] * focal_length_m
    if radius_m is None:
        outer_radius = wavelengths * focal_length_m * torch.as_tensor(
            cutoff_frequency_cpm, device=wavelengths.device, dtype=wavelengths.dtype
        )
    else:
        outer_radius = torch.as_tensor(radius_m, device=wavelengths.device, dtype=wavelengths.dtype).expand_as(wavelengths)
    if inner_radius_m is not None:
        inner_radius = torch.as_tensor(inner_radius_m, device=wavelengths.device, dtype=wavelengths.dtype).expand_as(wavelengths)
    else:
        inner_radius = wavelengths * focal_length_m * torch.as_tensor(
            inner_cutoff_frequency_cpm, device=wavelengths.device, dtype=wavelengths.dtype
        )
    edge = wavelengths * focal_length_m * soft_edge_cpm
    if soft_edge_cpm > 0:
        result = torch.sigmoid((outer_radius[:, None, None] - radius) / edge[:, None, None])
        if normalized_type == "annular":
            result = result * torch.where(
                inner_radius[:, None, None] > 0,
                torch.sigmoid((radius - inner_radius[:, None, None]) / edge[:, None, None]), 1.0,
            )
    else:
        result = (radius <= outer_radius[:, None, None]).to(torch.float64)
        if normalized_type == "annular":
            result = result * (radius >= inner_radius[:, None, None])
    return result[0] if scalar else result


def make_image_stop(
    grid: SimulationGrid,
    radius_m: float,
    soft_edge_m: float = 0.0,
    *,
    center_m: tuple[float, float] = (0.0, 0.0),
    hard: bool = False,
    soft_edge: float | None = None,
    tau: float | None = None,
) -> torch.Tensor:
    """Construct a circular output/detector-plane aperture mask."""
    if radius_m < 0:
        raise ValueError("radius_m must be nonnegative")
    if soft_edge is not None:
        soft_edge_m = soft_edge
    if tau is not None:
        soft_edge_m = tau
    if soft_edge_m < 0:
        raise ValueError("soft_edge_m must be nonnegative")
    if hard:
        soft_edge_m = 0.0
    x = grid.work_x - float(center_m[0])
    y = grid.work_y - float(center_m[1])
    radius = torch.sqrt(x.to(torch.float64).square() + y.to(torch.float64).square())
    return _soft_disk(
        radius,
        torch.as_tensor(radius_m, device=radius.device, dtype=radius.dtype),
        None,
        soft_edge_m,
    )


def stop_sampling_report(
    grid: SimulationGrid,
    mask: torch.Tensor,
    *,
    coordinate: Literal["physical", "frequency"] = "physical",
) -> dict[str, float | int | bool]:
    """Report basic finite-view and nonempty checks for a stop mask."""
    mask = mask.detach()
    if mask.shape[-2:] != (grid.work_n, grid.work_n):
        raise ValueError("stop mask must match the padded work grid")
    finite = bool(torch.isfinite(mask).all().item())
    nonempty = bool(torch.any(mask > 1e-8).item())
    return {
        "coordinate": coordinate,
        "work_n": grid.work_n,
        "mask_min": float(mask.min().cpu()),
        "mask_max": float(mask.max().cpu()),
        "mask_mean": float(mask.mean().cpu()),
        "finite": finite,
        "nonempty": nonempty,
        "within_unit_interval": bool((mask.min() >= -1e-6 and mask.max() <= 1.0 + 1e-6).item()),
    }


def sampling_report(
    system: OpticalSystem,
    grid: SimulationGrid,
    minimum_fov_ratio: float = 1.2,
) -> dict[str, float | int | bool]:
    base_aperture = grid.doe_aperture[grid.crop_start:grid.crop_end, grid.crop_start:grid.crop_end]
    half_extent_x = min(system.n_grid // 2, system.n_grid // 2 - 1) * system.pixel_pitch_m
    aperture_covered = half_extent_x >= system.aperture_radius_m
    fov_m = system.n_grid * system.pixel_pitch_m
    return {
        "n_grid": system.n_grid,
        "pixel_pitch_m": system.pixel_pitch_m,
        "base_field_of_view_m": fov_m,
        "aperture_radius_m": system.aperture_radius_m,
        "aperture_diameter_m": 2.0 * system.aperture_radius_m,
        "field_to_aperture_diameter_ratio": fov_m / (2.0 * system.aperture_radius_m),
        "minimum_required_ratio": minimum_fov_ratio,
        "aperture_fully_inside_base_grid": aperture_covered,
        "base_aperture_pixel_count": int(torch.count_nonzero(base_aperture > 0.5).item()),
        "work_aperture_pixel_count": int(torch.count_nonzero(grid.doe_aperture > 0.5).item()),
        "expected_aperture_pixel_count": math.pi * system.aperture_radius_m**2 / system.pixel_pitch_m**2,
        "pad_factor": system.pad_factor,
    }


def validate_sampling(
    system: OpticalSystem,
    grid: SimulationGrid,
    minimum_fov_ratio: float = 1.2,
) -> dict[str, float | int | bool]:
    report = sampling_report(system, grid, minimum_fov_ratio)
    if report["field_to_aperture_diameter_ratio"] < minimum_fov_ratio:
        raise ValueError(
            "physical field of view is too small for the DOE aperture: "
            f"got {report['field_to_aperture_diameter_ratio']:.3f} aperture diameters, "
            f"need at least {minimum_fov_ratio:.3f}"
        )
    if not report["aperture_fully_inside_base_grid"]:
        raise ValueError("the DOE aperture is not fully represented by the base grid")
    return report


def angular_spectrum_transfer(
    wavelength_m: float,
    fx: torch.Tensor,
    fy: torch.Tensor,
    distance_m: float,
) -> torch.Tensor:
    return angular_spectrum_transfer_batch((wavelength_m,), fx, fy, distance_m)[0]


def angular_spectrum_transfer_batch(
    wavelengths_m: Iterable[float] | torch.Tensor,
    fx: torch.Tensor,
    fy: torch.Tensor,
    distance_m: float,
) -> torch.Tensor:
    wavelengths = torch.as_tensor(wavelengths_m, device=fx.device, dtype=torch.float64)
    if wavelengths.ndim != 1 or wavelengths.numel() == 0 or torch.any(wavelengths <= 0):
        raise ValueError("wavelengths_m must be a nonempty vector of positive values")
    fx_precise = fx.to(torch.float64)
    fy_precise = fy.to(torch.float64)
    radius_squared = fx_precise.square() + fy_precise.square()
    inverse_wavelength_squared = wavelengths.reciprocal().square()[:, None, None]
    kz_cycles = torch.sqrt(
        torch.clamp(inverse_wavelength_squared - radius_squared[None, :, :], min=0.0)
    )
    propagating = radius_squared[None, :, :] <= inverse_wavelength_squared
    transfer = torch.exp(1j * 2.0 * torch.pi * distance_m * kz_cycles) * propagating
    return transfer.to(torch.complex64)


def _pad_center(field: torch.Tensor, target_n: int) -> torch.Tensor:
    current_n = field.shape[-1]
    if current_n == target_n:
        return field
    if current_n > target_n or field.shape[-2] != current_n:
        raise ValueError("field must be square and no larger than target_n")
    delta = target_n - current_n
    left = delta // 2
    right = delta - left
    return functional.pad(field, (left, right, left, right))


def _crop_center(field: torch.Tensor, start: int, size: int) -> torch.Tensor:
    return field[..., start:start + size, start:start + size]


def propagate(field: torch.Tensor, transfer: torch.Tensor) -> torch.Tensor:
    """Propagate a centred physical field with the corrected FFT convention."""
    if field.shape[-2:] != transfer.shape[-2:]:
        raise ValueError("field and transfer function shapes must match")
    fft_order = torch.fft.ifftshift(field, dim=(-2, -1))
    propagated = torch.fft.ifft2(torch.fft.fft2(fft_order) * transfer)
    return torch.fft.fftshift(propagated, dim=(-2, -1))


def _validate_dense_matrix_budget(n_grid: int, max_matrix_bytes: int) -> int:
    if max_matrix_bytes <= 0:
        raise ValueError("max_matrix_bytes must be positive")
    matrix_bytes = n_grid**4 * torch.empty((), dtype=torch.complex64).element_size()
    if matrix_bytes > max_matrix_bytes:
        raise MemoryError(
            f"dense propagation matrix needs {matrix_bytes / 1024**2:.1f} MiB, "
            f"exceeding the {max_matrix_bytes / 1024**2:.1f} MiB limit"
        )
    return matrix_bytes


def build_dense_angular_spectrum_matrix(
    wavelength_m: float,
    grid: SimulationGrid,
    distance_m: float,
    chunk_size: int = 32,
    max_matrix_bytes: int = 256 * 1024**2,
) -> torch.Tensor:
    """Expand the padded, cropped FFT angular-spectrum operator into a matrix."""
    if wavelength_m <= 0 or distance_m < 0:
        raise ValueError("wavelength_m must be positive and distance_m nonnegative")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    _validate_dense_matrix_budget(grid.base_n, max_matrix_bytes)

    pixel_count = grid.base_n**2
    matrix = torch.empty(
        (pixel_count, pixel_count), device=grid.base_x.device, dtype=torch.complex64
    )
    transfer = angular_spectrum_transfer(wavelength_m, grid.fx, grid.fy, distance_m)
    with torch.no_grad():
        for start in range(0, pixel_count, chunk_size):
            stop = min(start + chunk_size, pixel_count)
            batch_size = stop - start
            basis = torch.zeros(
                (batch_size, pixel_count),
                device=grid.base_x.device,
                dtype=torch.complex64,
            )
            batch_indices = torch.arange(batch_size, device=basis.device)
            column_indices = torch.arange(start, stop, device=basis.device)
            basis[batch_indices, column_indices] = 1.0
            base_fields = basis.reshape(batch_size, grid.base_n, grid.base_n)
            work_fields = _pad_center(base_fields, grid.work_n)
            propagated = propagate(work_fields, transfer)
            output_fields = _crop_center(propagated, grid.crop_start, grid.base_n)
            matrix[:, start:stop] = output_fields.reshape(batch_size, pixel_count).T
    return matrix


def build_dense_fresnel_matrix(
    wavelength_m: float,
    grid: SimulationGrid,
    distance_m: float,
    chunk_size: int = 128,
    max_matrix_bytes: int = 256 * 1024**2,
) -> torch.Tensor:
    """Build a sampled point-to-point Fresnel diffraction matrix."""
    if wavelength_m <= 0 or distance_m <= 0:
        raise ValueError("wavelength_m and distance_m must be positive")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    _validate_dense_matrix_budget(grid.base_n, max_matrix_bytes)

    pixel_count = grid.base_n**2
    coordinates = torch.stack(
        (grid.base_x.reshape(-1), grid.base_y.reshape(-1)), dim=-1
    ).to(torch.float64)
    matrix = torch.empty(
        (pixel_count, pixel_count), device=grid.base_x.device, dtype=torch.complex64
    )
    prefactor = (grid.pixel_pitch_m**2 / (wavelength_m * distance_m)) * complex(
        math.cos(2.0 * math.pi * distance_m / wavelength_m - math.pi / 2.0),
        math.sin(2.0 * math.pi * distance_m / wavelength_m - math.pi / 2.0),
    )
    with torch.no_grad():
        for start in range(0, pixel_count, chunk_size):
            stop = min(start + chunk_size, pixel_count)
            offsets = coordinates[start:stop, None, :] - coordinates[None, :, :]
            radius_squared = torch.sum(offsets.square(), dim=-1)
            phase = math.pi * radius_squared / (wavelength_m * distance_m)
            kernel = torch.exp(1j * phase).to(torch.complex64)
            matrix[start:stop] = kernel * prefactor
    return matrix


def apply_dense_propagation(matrix: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """Apply a square dense propagator to one or more square complex fields."""
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be a square rank-2 tensor")
    if (
        field.ndim < 2
        or field.shape[-2] != field.shape[-1]
        or field.shape[-1] ** 2 != matrix.shape[1]
    ):
        raise ValueError("the field's last two dimensions must match the square matrix grid")
    if not field.is_complex():
        raise ValueError("field must have a complex dtype")
    if field.device != matrix.device:
        raise ValueError("matrix and field must be on the same device")
    compatible_matrix = matrix.to(dtype=field.dtype)
    flat_field = field.reshape(*field.shape[:-2], -1)
    flat_output = torch.matmul(flat_field, compatible_matrix.transpose(0, 1))
    return flat_output.reshape(*field.shape[:-2], *field.shape[-2:])


def make_doe_field(
    height_m: torch.Tensor,
    grid: SimulationGrid,
    wavelength_m: float,
    refractive_index: float,
) -> torch.Tensor:
    height_work = _pad_center(height_m, grid.work_n)
    phase_scale = torch.as_tensor(
        2.0 * math.pi * (refractive_index - 1.0) / wavelength_m,
        device=height_work.device,
        dtype=height_work.dtype,
    )
    phase = phase_scale * height_work
    return grid.doe_aperture.to(height_work.dtype) * torch.exp(1j * phase)


def make_doe_field_rgb(
    height_m: torch.Tensor,
    grid: SimulationGrid,
    wavelengths_m: Iterable[float] | torch.Tensor,
    refractive_indices: Iterable[float] | torch.Tensor,
) -> torch.Tensor:
    wavelengths = torch.as_tensor(wavelengths_m, device=height_m.device, dtype=torch.float64)
    indices = torch.as_tensor(refractive_indices, device=height_m.device, dtype=torch.float64)
    if wavelengths.ndim != 1 or indices.shape != wavelengths.shape:
        raise ValueError("wavelengths_m and refractive_indices must have matching vector shapes")
    height_work = _pad_center(height_m, grid.work_n)
    phase_scale = (2.0 * torch.pi * (indices - 1.0) / wavelengths).to(height_work.dtype)
    phase = phase_scale[:, None, None] * height_work[None, :, :]
    aperture = grid.doe_aperture.to(height_work.dtype)
    return aperture[None, :, :] * torch.exp(1j * phase)


def forward_field(
    height_m: torch.Tensor,
    grid: SimulationGrid,
    system: OpticalSystem,
    wavelength_m: float,
    refractive_index: float,
) -> torch.Tensor:
    """DOE-to-image-plane field for free-space or thin-lens configuration."""
    doe_field = make_doe_field(height_m, grid, wavelength_m, refractive_index)
    if system.model == "free_space":
        transfer = angular_spectrum_transfer(wavelength_m, grid.fx, grid.fy, system.propagation_distance_m)
        output = propagate(doe_field, transfer)
    else:
        to_lens = angular_spectrum_transfer(wavelength_m, grid.fx, grid.fy, system.doe_to_lens_distance_m)
        to_image = angular_spectrum_transfer(wavelength_m, grid.fx, grid.fy, system.lens_to_image_distance_m)
        field_at_lens = propagate(doe_field, to_lens)
        lens_phase = -torch.pi * (grid.work_x**2 + grid.work_y**2) / (wavelength_m * system.lens_focal_length_m)
        lens = grid.lens_aperture.to(field_at_lens.dtype) * torch.exp(1j * lens_phase)
        output = propagate(field_at_lens * lens, to_image)
    return _crop_center(output, grid.crop_start, grid.base_n)


def forward_rgb_single_plane(
    height_m: torch.Tensor,
    grid: SimulationGrid,
    wavelengths_m: Iterable[float] | torch.Tensor,
    refractive_indices: Iterable[float] | torch.Tensor,
    transfer: torch.Tensor,
) -> torch.Tensor:
    """Batched RGB propagation through one shared phase-only DOE."""
    field = make_doe_field_rgb(height_m, grid, wavelengths_m, refractive_indices)
    output = propagate(field, transfer)
    return _crop_center(output, grid.crop_start, grid.base_n)


def forward_rgb_two_planes(
    heights_m: torch.Tensor,
    grid: SimulationGrid,
    wavelengths_m: Iterable[float] | torch.Tensor,
    refractive_indices: Iterable[float] | torch.Tensor,
    first_transfer: torch.Tensor,
    second_transfer: torch.Tensor,
) -> torch.Tensor:
    """Batched RGB propagation through two independently optimized DOE planes."""
    if heights_m.ndim != 3 or heights_m.shape[0] != 2:
        raise ValueError("heights_m must have shape (2, n_grid, n_grid)")
    first_field = make_doe_field_rgb(heights_m[0], grid, wavelengths_m, refractive_indices)
    field_at_second = propagate(first_field, first_transfer)
    second_screen = make_doe_field_rgb(heights_m[1], grid, wavelengths_m, refractive_indices)
    output = propagate(field_at_second * second_screen, second_transfer)
    return _crop_center(output, grid.crop_start, grid.base_n)


def forward_rgb_fourier_lens(
    height_m: torch.Tensor,
    grid: SimulationGrid,
    wavelengths_m: Iterable[float] | torch.Tensor,
    refractive_indices: Iterable[float] | torch.Tensor,
    transfer: torch.Tensor,
    focal_length_m: float,
) -> torch.Tensor:
    """Batched RGB propagation through a DOE co-located with an ideal thin lens."""
    if focal_length_m <= 0:
        raise ValueError("focal_length_m must be positive")
    wavelengths = torch.as_tensor(wavelengths_m, device=height_m.device, dtype=torch.float64)
    field = make_doe_field_rgb(height_m, grid, wavelengths, refractive_indices)
    radius_squared = grid.work_x.to(torch.float64).square() + grid.work_y.to(torch.float64).square()
    lens_phase = -torch.pi * radius_squared[None, :, :] / (wavelengths[:, None, None] * focal_length_m)
    lens = grid.lens_aperture.to(field.dtype)[None, :, :] * torch.exp(1j * lens_phase).to(field.dtype)
    output = propagate(field * lens, transfer)
    return _crop_center(output, grid.crop_start, grid.base_n)


def _lens_batch(
    grid: SimulationGrid,
    wavelengths: torch.Tensor,
    focal_length_m: float,
    aperture_radius_m: float | None = None,
) -> torch.Tensor:
    if focal_length_m <= 0:
        raise ValueError("focal_length_m must be positive")
    radius_squared = grid.work_x.to(torch.float64).square() + grid.work_y.to(torch.float64).square()
    if aperture_radius_m is None:
        aperture = grid.lens_aperture
    else:
        if aperture_radius_m <= 0:
            raise ValueError("aperture_radius_m must be positive")
        aperture = (
            radius_squared <= float(aperture_radius_m) ** 2
        ).to(dtype=torch.float32)
    lens_phase = -torch.pi * radius_squared[None, :, :] / (
        wavelengths[:, None, None] * float(focal_length_m)
    )
    return aperture.to(torch.complex64)[None, :, :] * torch.exp(1j * lens_phase).to(torch.complex64)


def _coerce_stop_mask(
    stop: torch.Tensor | OpticalStopConfig | None,
    grid: SimulationGrid,
    wavelengths: torch.Tensor,
    focal_length_m: float,
    *,
    hard_stop: bool,
) -> torch.Tensor:
    if stop is None:
        return torch.ones(
            (wavelengths.numel(), grid.work_n, grid.work_n),
            device=grid.work_x.device,
            dtype=torch.float32,
        )
    if isinstance(stop, OpticalStopConfig):
        if stop.kind == "none":
            return torch.ones(
                (wavelengths.numel(), grid.work_n, grid.work_n),
                device=grid.work_x.device,
                dtype=torch.float32,
            )
        if stop.kind not in {"fourier_circular", "fourier_annular"}:
            raise ValueError("a Fourier stop is required for the relay stop argument")
        if stop.cutoff_frequency_cpm is None and stop.radius_m is None:
            raise ValueError("Fourier stop requires a cutoff frequency or physical radius")
        return make_fourier_stop(
            grid,
            wavelengths,
            focal_length_m,
            cutoff_frequency_cpm=stop.cutoff_frequency_cpm,
            inner_cutoff_frequency_cpm=stop.inner_cutoff_frequency_cpm,
            soft_edge_cpm=stop.soft_edge,
            stop_type=stop.kind,
            radius_m=stop.radius_m,
            inner_radius_m=stop.inner_radius_m if stop.radius_m is not None else None,
            hard=hard_stop,
        ).to(torch.float32)
    mask = torch.as_tensor(stop, device=grid.work_x.device)
    if mask.ndim == 2:
        mask = mask[None, :, :]
    if mask.ndim != 3 or mask.shape[-2:] != (grid.work_n, grid.work_n):
        raise ValueError("Fourier stop mask must have shape (work_n, work_n) or (wavelengths, work_n, work_n)")
    if mask.shape[0] not in (1, wavelengths.numel()):
        raise ValueError("Fourier stop wavelength dimension does not match wavelengths")
    if mask.shape[0] == 1:
        mask = mask.expand(wavelengths.numel(), -1, -1)
    return mask.to(dtype=torch.float32)


def _coerce_image_mask(
    image_stop: torch.Tensor | OpticalStopConfig | None,
    grid: SimulationGrid,
    *,
    hard_stop: bool,
) -> torch.Tensor:
    if image_stop is None:
        return torch.ones(
            (grid.work_n, grid.work_n), device=grid.work_x.device, dtype=torch.float32
        )
    if isinstance(image_stop, OpticalStopConfig):
        if image_stop.kind == "none":
            return torch.ones(
                (grid.work_n, grid.work_n), device=grid.work_x.device, dtype=torch.float32
            )
        if image_stop.kind != "image" or image_stop.radius_m is None:
            raise ValueError("image_stop config requires stop_type='image' and radius_m")
        return make_image_stop(
            grid,
            image_stop.radius_m,
            image_stop.soft_edge,
            hard=hard_stop,
        ).to(torch.float32)
    mask = torch.as_tensor(image_stop, device=grid.work_x.device)
    if mask.ndim != 2 or mask.shape != (grid.work_n, grid.work_n):
        raise ValueError("image stop mask must have shape (work_n, work_n)")
    return mask.to(torch.float32)


def forward_rgb_4f_stop(
    heights_m: torch.Tensor,
    grid: SimulationGrid,
    wavelengths_m: Iterable[float] | torch.Tensor,
    refractive_indices: Iterable[float] | torch.Tensor,
    focal_length_m: float = 0.05,
    stop: torch.Tensor | OpticalStopConfig | None = None,
    image_stop: torch.Tensor | OpticalStopConfig | None = None,
    *,
    lens1_focal_length_m: float | None = None,
    lens2_focal_length_m: float | None = None,
    lens_aperture_radius_m: float | None = None,
    plane_separation_m: float | None = None,
    object_distance_m: float = 0.0,
    register_inversion: bool = True,
    hard_stop: bool = False,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Ideal paraxial 1:1 4f relay evaluated as two scaled focal transforms.

    Object--L1--stop--L2--image separations are f, f, f, f.  The second
    transform restores physical image coordinates, with the expected parity
    inversion; ``register_inversion`` labels output in registered coordinates.
    ``object_distance_m`` adds free propagation after the last DOE before the
    relay object plane.  Zero images the DOE itself (a phase-only DOE cannot
    form a nonuniform intensity with an unfiltered ideal relay).

    The default uses infinite ideal lenses. A finite lens radius enables
    explicit angular-spectrum propagation between two finite thin lenses.
    Use ``forward_rgb_lens_aperture`` for a literal co-located finite pupil.
    """
    if focal_length_m <= 0:
        raise ValueError("focal_length_m must be positive")
    f1 = float(focal_length_m if lens1_focal_length_m is None else lens1_focal_length_m)
    f2 = float(focal_length_m if lens2_focal_length_m is None else lens2_focal_length_m)
    if f1 <= 0 or f2 <= 0:
        raise ValueError("relay focal lengths must be positive")
    if not math.isclose(f1, f2, rel_tol=1e-10):
        raise ValueError("only 1:1 relays (equal focal lengths) are supported")
    if object_distance_m < 0:
        raise ValueError("object_distance_m must be nonnegative")
    wavelengths = torch.as_tensor(wavelengths_m, device=heights_m.device, dtype=torch.float64)
    indices = torch.as_tensor(refractive_indices, device=heights_m.device, dtype=torch.float64)
    if wavelengths.ndim != 1 or indices.shape != wavelengths.shape or wavelengths.numel() == 0:
        raise ValueError("wavelengths_m and refractive_indices must be matching nonempty vectors")
    if heights_m.ndim not in (2, 3) or heights_m.shape[-2:] != (grid.base_n, grid.base_n):
        raise ValueError("heights_m must have shape (n_grid, n_grid) or (2, n_grid, n_grid)")
    if heights_m.ndim == 3 and heights_m.shape[0] != 2:
        raise ValueError("a multi-plane height tensor must have exactly two planes")

    if heights_m.ndim == 2:
        field = make_doe_field_rgb(heights_m, grid, wavelengths, indices)
    else:
        if plane_separation_m is None:
            plane_separation_m = f1 / 2.0
        if plane_separation_m <= 0:
            raise ValueError("plane_separation_m must be positive for two DOE planes")
        between = angular_spectrum_transfer_batch(
            wavelengths, grid.fx, grid.fy, float(plane_separation_m)
        )
        first_doe = make_doe_field_rgb(heights_m[0], grid, wavelengths, indices)
        second_doe = make_doe_field_rgb(heights_m[1], grid, wavelengths, indices)
        field = propagate(first_doe, between) * second_doe
    if object_distance_m > 0:
        field = propagate(field, angular_spectrum_transfer_batch(wavelengths, grid.fx, grid.fy, object_distance_m))
    before_pupil_energy = field.abs().square().sum((-2, -1))
    if lens_aperture_radius_m is not None:
        if lens_aperture_radius_m <= 0:
            raise ValueError("lens_aperture_radius_m must be positive")
        if lens_aperture_radius_m / (float(wavelengths.min()) * f1) >= 0.5 / grid.pixel_pitch_m:
            raise ValueError("finite lens phase exceeds sampled Nyquist limit")
        transfer = angular_spectrum_transfer_batch(wavelengths, grid.fx, grid.fy, f1)
        lens = _lens_batch(grid, wavelengths, f1, lens_aperture_radius_m)
        at_first_lens = propagate(field, transfer)
        after_first_lens = at_first_lens * lens
        at_fourier = propagate(after_first_lens, transfer)
    else:
        at_fourier = centered_fft(field)
    pre_stop_energy = torch.sum(torch.abs(at_fourier) ** 2, dim=(-2, -1))
    if lens_aperture_radius_m is not None and isinstance(stop, OpticalStopConfig) and stop.kind != "none":
        # Explicit ASM uses the same physical pitch at every plane, whereas
        # the ideal focal FFT uses lambda*f/(N*dx). Build the physical mask on
        # the ASM coordinates, not on the scaled FFT lattice.
        outer = stop.radius_m
        inner = stop.inner_radius_m
        if outer is None:
            outer = wavelengths[:, None, None] * f1 * stop.cutoff_frequency_cpm
            inner = wavelengths[:, None, None] * f1 * stop.inner_cutoff_frequency_cpm
        edge_m = stop.soft_edge * float(wavelengths[0]) * f1
        radial = torch.hypot(grid.work_x, grid.work_y)
        stop_mask = _soft_disk(radial, torch.as_tensor(outer, device=radial.device),
            torch.as_tensor(inner, device=radial.device) if stop.kind == "fourier_annular" else None,
            0.0 if hard_stop else edge_m)
        if stop_mask.ndim == 2:
            stop_mask = stop_mask[None].expand(len(wavelengths), -1, -1)
    else:
        stop_mask = _coerce_stop_mask(stop, grid, wavelengths, f1, hard_stop=hard_stop)
    after_stop = at_fourier * stop_mask.to(at_fourier.dtype)
    post_stop_energy = torch.sum(torch.abs(after_stop) ** 2, dim=(-2, -1))
    if lens_aperture_radius_m is not None:
        at_second_lens = propagate(after_stop, transfer)
        after_second_lens = at_second_lens * lens
        before_image_stop = propagate(after_second_lens, transfer)
    else:
        before_image_stop = centered_fft(after_stop)
    if register_inversion:
        before_image_stop = invert_centered(before_image_stop)
    image_mask = _coerce_image_mask(image_stop, grid, hard_stop=hard_stop)
    after_image = before_image_stop * image_mask.to(before_image_stop.dtype)
    output = _crop_center(after_image, grid.crop_start, grid.base_n)
    if not return_diagnostics:
        return output
    diagnostics = {
        "fourier_field": at_fourier,
        "fourier_stop_mask": stop_mask,
        "before_image_stop_field": before_image_stop,
        "before_pupil_energy": before_pupil_energy,
        "lens_pupil_throughput": pre_stop_energy / (before_pupil_energy + 1e-12),
        "pre_stop_energy": pre_stop_energy,
        "post_stop_energy": post_stop_energy,
        "before_image_stop_energy": torch.sum(torch.abs(before_image_stop) ** 2, dim=(-2, -1)),
        "after_image_stop_energy": torch.sum(torch.abs(after_image) ** 2, dim=(-2, -1)),
        "stop_throughput": post_stop_energy / (pre_stop_energy + 1e-12),
        "image_stop_throughput": torch.sum(torch.abs(after_image) ** 2, dim=(-2, -1))
        / (torch.sum(torch.abs(before_image_stop) ** 2, dim=(-2, -1)) + 1e-12),
    }
    if lens_aperture_radius_m is not None:
        diagnostics.update({"lens1_pre_energy": at_first_lens.abs().square().sum((-2, -1)),
            "lens1_post_energy": after_first_lens.abs().square().sum((-2, -1)),
            "lens2_pre_energy": at_second_lens.abs().square().sum((-2, -1)),
            "lens2_post_energy": after_second_lens.abs().square().sum((-2, -1))})
    return output, diagnostics


def centered_fft(field: torch.Tensor) -> torch.Tensor:
    """Unitary focal transform; energy is in object-pixel-equivalent units."""
    return torch.fft.fftshift(torch.fft.fft2(
        torch.fft.ifftshift(field, dim=(-2, -1)), norm="ortho"
    ), dim=(-2, -1))


def invert_centered(field: torch.Tensor) -> torch.Tensor:
    """Parity inversion on an even centred periodic lattice (keeps DC fixed)."""
    return torch.roll(torch.flip(field, (-2, -1)), (1, 1), (-2, -1))


def forward_rgb_lens_aperture(
    heights_m: torch.Tensor,
    grid: SimulationGrid,
    wavelengths_m: Iterable[float] | torch.Tensor,
    refractive_indices: Iterable[float] | torch.Tensor,
    focal_length_m: float,
    lens_aperture_radius_m: float,
    *,
    plane_separation_m: float = 0.025,
    return_diagnostics: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """DOE co-located with a finite thin lens; detector at its back focal plane.

    Analytically cancelling the lens and Fresnel input chirps avoids sampling
    the rapidly varying quadratic lens phase on the coarse DOE grid.  The
    native focal grid has dx_f=lambda*f/(N*dx); complex interpolation onto the
    common physical detector grid includes the pixel-area energy factor.
    """
    if focal_length_m <= 0 or lens_aperture_radius_m <= 0:
        raise ValueError("focal length and pupil radius must be positive")
    wavelengths = torch.as_tensor(wavelengths_m, device=heights_m.device, dtype=torch.float64)
    if heights_m.ndim == 2:
        field = make_doe_field_rgb(heights_m, grid, wavelengths, refractive_indices)
    elif heights_m.ndim == 3 and heights_m.shape[0] == 2:
        field = propagate(make_doe_field_rgb(heights_m[0], grid, wavelengths, refractive_indices),
                          angular_spectrum_transfer_batch(wavelengths, grid.fx, grid.fy, plane_separation_m))
        field = field * make_doe_field_rgb(heights_m[1], grid, wavelengths, refractive_indices)
    else:
        raise ValueError("height must be a single map or two DOE maps")
    pupil = (torch.hypot(grid.work_x, grid.work_y) <= lens_aperture_radius_m).to(field.real.dtype)
    after_pupil = field * pupil
    spectrum = centered_fft(after_pupil)
    pitches = wavelengths * focal_length_m / (grid.work_n * grid.pixel_pitch_m)
    if (grid.base_n // 2) * grid.pixel_pitch_m > float(pitches.min()) * (grid.work_n // 2 - 1):
        raise ValueError("requested focal detector exceeds native Fourier field; increase focal length or refine object sampling")
    sample_x = grid.base_x[None] / pitches[:, None, None] + grid.work_n // 2
    sample_y = grid.base_y[None] / pitches[:, None, None] + grid.work_n // 2
    sample_grid = torch.stack((2 * sample_x / (grid.work_n - 1) - 1,
                               2 * sample_y / (grid.work_n - 1) - 1), dim=-1).to(torch.float32)
    samples = functional.grid_sample(torch.stack((spectrum.real, spectrum.imag), dim=1),
                                     sample_grid, align_corners=True, padding_mode="zeros")
    output = torch.complex(samples[:, 0], samples[:, 1]) * (grid.pixel_pitch_m / pitches[:, None, None])
    phase = torch.pi * (grid.base_x.square() + grid.base_y.square())[None] / (
        wavelengths[:, None, None] * focal_length_m
    )
    output = output * torch.exp(1j * phase).to(output.dtype)
    if not return_diagnostics:
        return output
    before = field.abs().square().sum((-2, -1))
    after = after_pupil.abs().square().sum((-2, -1))
    return output, {"fourier_field": spectrum, "pre_stop_energy": before, "post_stop_energy": after,
                    "stop_throughput": after / before.clamp_min(1e-12), "fourier_pitch_m": pitches,
                    "native_focal_energy": spectrum.abs().square().sum((-2, -1)),
                    "detector_energy": output.abs().square().sum((-2, -1))}


def transfer_identity_error(system: OpticalSystem, device: torch.device) -> float:
    grid = make_grid(system, device)
    field = torch.zeros((grid.work_n, grid.work_n), dtype=torch.complex64, device=device)
    field[grid.work_n // 2, grid.work_n // 2] = 1
    transfer = angular_spectrum_transfer(532e-9, grid.fx, grid.fy, 0.0)
    return float(torch.max(torch.abs(propagate(field, transfer) - field)).detach().cpu())


def energy_relative_error(system: OpticalSystem, device: torch.device) -> float:
    grid = make_grid(system, device)
    field = grid.doe_aperture.to(torch.complex64)
    transfer = angular_spectrum_transfer(532e-9, grid.fx, grid.fy, 0.05)
    output = propagate(field, transfer)
    before = torch.sum(torch.abs(field) ** 2)
    after = torch.sum(torch.abs(output) ** 2)
    return float(torch.abs(after / before - 1.0).detach().cpu())
