"""Create a physically sampled, antialiased 3x3 multiscale target chart."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_GRID = 256
DEFAULT_PITCH_M = 15.625e-6
DEFAULT_SUPERSAMPLE = 8
LINE_WIDTHS_UM = (160.0, 240.0, 360.0)
ORIENTATIONS_DEG = (0.0, 45.0, 90.0)
CELL_HALF_WIDTH_UM = 500.0
CELL_HALF_HEIGHT_UM = 325.0


def _downsample_mean(image: np.ndarray, factor: int) -> np.ndarray:
    n = image.shape[0] // factor
    return image.reshape(n, factor, n, factor).mean(axis=(1, 3))


def make_multiscale_target(
    grid: int = DEFAULT_GRID,
    detector_pitch_m: float = DEFAULT_PITCH_M,
    supersample: int = DEFAULT_SUPERSAMPLE,
) -> tuple[np.ndarray, dict]:
    """Return a unit-sum target and JSON-serializable physical-region metadata."""
    if (grid < 32 or grid % 2 or not np.isfinite(detector_pitch_m)
        or detector_pitch_m <= 0 or supersample < 2):
        raise ValueError("grid must be even >= 32; pitch positive; supersample >= 2")
    pitch_um = detector_pitch_m * 1e6
    high_n = grid * supersample
    # Match Step 11's even-grid Fourier convention: sample centers are
    # (index-grid//2)*pitch, so subpixels tile around those exact centers.
    coord_um = (np.arange(high_n) - (grid//2)*supersample
                - supersample/2 + 0.5) * pitch_um / supersample
    yy, xx = np.meshgrid(coord_um, coord_um, indexing="ij")
    chart = np.zeros((grid, grid), dtype=np.float64)
    regions = []

    for row, width_um in enumerate(LINE_WIDTHS_UM):
        y_center_um = (720.0, 0.0, -720.0)[row]
        for col, orientation_deg in enumerate(ORIENTATIONS_DEG):
            x_center_um = (-1080.0, 0.0, 1080.0)[col]
            x_local = xx - x_center_um
            y_local = yy - y_center_um
            cell = ((np.abs(x_local) <= CELL_HALF_WIDTH_UM)
                    & (np.abs(y_local) <= CELL_HALF_HEIGHT_UM))
            theta = np.deg2rad(orientation_deg)
            normal_coordinate = -x_local * np.sin(theta) + y_local * np.cos(theta)
            stripe = np.mod(normal_coordinate + width_um, 2.0 * width_um) < width_um
            tile = _downsample_mean((cell & stripe).astype(np.float64), supersample)
            y0, y1 = y_center_um - CELL_HALF_HEIGHT_UM, y_center_um + CELL_HALF_HEIGHT_UM
            x0, x1 = x_center_um - CELL_HALF_WIDTH_UM, x_center_um + CELL_HALF_WIDTH_UM
            detector_axis_um = (np.arange(grid) - grid // 2) * pitch_um
            rows = np.flatnonzero((detector_axis_um >= y0) & (detector_axis_um <= y1))
            cols = np.flatnonzero((detector_axis_um >= x0) & (detector_axis_um <= x1))
            if tile.sum() <= 0 or rows.size == 0 or cols.size == 0:
                raise ValueError("A resolution-chart cell is not represented on this detector grid")
            # Assign equal integrated weight to each scale/direction cell.
            chart[np.ix_(rows, cols)] += tile[np.ix_(rows, cols)] * (0.90 / 9.0) / tile.sum()
            regions.append({
                "name": f"width_{int(width_um)}um_orientation_{int(orientation_deg)}deg",
                "x_min_um": x0, "x_max_um": x1,
                "y_min_um": y0, "y_max_um": y1,
                "line_width_um": width_um,
                "orientation_deg": orientation_deg,
            })

    # A 40 um border and a small cross at the optical-axis center make chart
    # registration visible while reserving 90% of the target energy for tiles.
    border_outer = ((np.abs(xx) <= 1660.0) & (np.abs(yy) <= 1420.0))
    border_inner = ((np.abs(xx) <= 1620.0) & (np.abs(yy) <= 1380.0))
    border = border_outer & ~border_inner
    marker = ((np.abs(xx) <= 130.0) & (np.abs(yy) <= 22.0)) | (
        (np.abs(yy) <= 130.0) & (np.abs(xx) <= 22.0)
    )
    detail = _downsample_mean((border | marker).astype(np.float64), supersample)
    if detail.sum() <= 0:
        raise ValueError("Registration border/marker is not represented on this detector grid")
    chart += detail * (0.10 / detail.sum())
    chart /= chart.sum()

    metadata = {
        "schema_version": 1,
        "target_name": "multiscale_resolution_chart",
        "grid": grid,
        "detector_pitch_um": pitch_um,
        "field_width_mm": grid * detector_pitch_m * 1e3,
        "coordinate_convention": "x,y=(index-grid//2)*detector_pitch; field boundaries are +/-field_width/2",
        "first_pixel_center_um": [-grid//2*pitch_um, -grid//2*pitch_um],
        "last_pixel_center_um": [(grid//2-1)*pitch_um, (grid//2-1)*pitch_um],
        "supersample": supersample,
        "line_widths_um": list(LINE_WIDTHS_UM),
        "finest_bar_width_pixels": min(LINE_WIDTHS_UM) / pitch_um,
        "finest_stripe_period_um": 2 * min(LINE_WIDTHS_UM),
        "finest_stripe_period_pixels": 2 * min(LINE_WIDTHS_UM) / pitch_um,
        "orientations_deg": list(ORIENTATIONS_DEG),
        "equal_energy_per_cell": True,
        "cell_energy_fraction": 0.90 / 9.0,
        "registration_energy_fraction": 0.10,
        "roi_definition": "target >= 1% of maximum target intensity",
        "regions": regions,
    }
    return chart, metadata


def make_region_masks(metadata: dict, detector_axis_m: np.ndarray) -> dict[str, np.ndarray]:
    axis_um = np.asarray(detector_axis_m) * 1e6
    x, y = np.meshgrid(axis_um, axis_um, indexing="xy")
    masks = {}
    for region in metadata.get("regions", []):
        masks[region["name"]] = (
            (x >= region["x_min_um"]) & (x <= region["x_max_um"])
            & (y >= region["y_min_um"]) & (y <= region["y_max_um"])
        )
    return masks


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", type=int, default=DEFAULT_GRID)
    parser.add_argument("--detector-pitch-um", type=float, default=DEFAULT_PITCH_M * 1e6)
    parser.add_argument("--supersample", type=int, default=DEFAULT_SUPERSAMPLE)
    parser.add_argument("--output-dir", type=Path, default=Path("results/step11_visible_multiscale/target"))
    args = parser.parse_args()
    target, metadata = make_multiscale_target(args.grid, args.detector_pitch_um * 1e-6, args.supersample)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    np.save(args.output_dir / "target.npy", target)
    (args.output_dir / "target.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    axis = (np.arange(args.grid) - args.grid//2) * args.detector_pitch_um
    extent = [(axis[0]-args.detector_pitch_um/2)/1000,
              (axis[-1]+args.detector_pitch_um/2)/1000,
              (axis[0]-args.detector_pitch_um/2)/1000,
              (axis[-1]+args.detector_pitch_um/2)/1000]
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.imshow(target / target.max(), origin="lower", extent=extent, cmap="magma", vmin=0, vmax=1)
    ax.set(xlabel="X (mm)", ylabel="Y (mm)", title="Multiscale resolution chart target")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(args.output_dir / "target.png", dpi=180)
    plt.close(fig)
    print(f"Saved target, preview, and metadata to {args.output_dir.resolve()}")
    print(f"sum={target.sum():.12g}; min={target.min():.6g}; max={target.max():.6g}")


if __name__ == "__main__":
    main()
