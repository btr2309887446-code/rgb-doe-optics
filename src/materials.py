"""Material dispersion models for the DOE experiments."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np


@dataclass(frozen=True)
class CauchyMaterial:
    """Two-anchor Cauchy model for the S1813 simulation material."""

    name: str = "Shipley S1813"
    anchor_wavelengths_m: tuple[float, float] = (532e-9, 633e-9)
    anchor_indices: tuple[float, float] = (1.6482, 1.6347)
    source: str = "Kim et al. (2012), Shipley S1813 values"

    @property
    def coefficients(self) -> tuple[float, float]:
        wavelengths = np.asarray(self.anchor_wavelengths_m, dtype=float)
        indices = np.asarray(self.anchor_indices, dtype=float)
        x_values = 1.0 / wavelengths**2
        coefficient_b = float((indices[0] - indices[1]) / (x_values[0] - x_values[1]))
        coefficient_a = float(indices[0] - coefficient_b * x_values[0])
        return coefficient_a, coefficient_b

    def index(self, wavelength_m: float) -> float:
        if wavelength_m <= 0:
            raise ValueError("wavelength must be positive")
        coefficient_a, coefficient_b = self.coefficients
        return float(coefficient_a + coefficient_b / wavelength_m**2)

    def indices(self, wavelengths_m: Iterable[float]) -> tuple[float, ...]:
        return tuple(self.index(float(value)) for value in wavelengths_m)

    def metadata(self, wavelengths_m: Iterable[float]) -> dict[str, object]:
        wavelengths = tuple(float(value) for value in wavelengths_m)
        lower, upper = sorted(self.anchor_wavelengths_m)
        return {
            "material": self.name,
            "source": self.source,
            "model": "n(lambda)=A+B/lambda^2",
            "cauchy_coefficients": list(self.coefficients),
            "anchor_wavelengths_m": list(self.anchor_wavelengths_m),
            "anchor_indices": list(self.anchor_indices),
            "wavelengths_m": list(wavelengths),
            "indices": list(self.indices(wavelengths)),
            "extrapolated": [not lower <= value <= upper for value in wavelengths],
            "sensitivity_delta_n": 0.005,
        }
