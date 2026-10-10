"""Scan target geometry or loss weights with the modular DOE pipeline.

Examples:
    py -3 src/scan_modular.py --kind target --radii 30 40 --edge-widths 10 15 --iterations 100 --cpu
    py -3 src/scan_modular.py --kind weights --background-weights 0.05 0.25 0.5 --iterations 100 --cpu
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import make_system, parse_wavelengths, run_one


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--wavelengths", type=float, nargs="+", default=[532, 633])
    parser.add_argument("--model", choices=["free_space", "thin_lens"], default="free_space")
    parser.add_argument("--n-grid", type=int, default=256)
    parser.add_argument("--pixel-pitch-um", type=float, default=10.0)
    parser.add_argument("--aperture-radius-um", type=float, default=1000.0)
    parser.add_argument("--pad-factor", type=int, choices=[1, 2, 4], default=1)
    parser.add_argument("--propagation-distance-mm", type=float, default=50.0)
    parser.add_argument("--focal-length-mm", type=float, default=50.0)
    parser.add_argument("--magnification", type=float, default=1.0)
    parser.add_argument("--lens-aperture-um", type=float)
    parser.add_argument("--iterations", type=int, default=300)
    parser.add_argument("--learning-rate-um", type=float, default=0.04)
    parser.add_argument("--height-max-um", type=float, default=1.4)
    parser.add_argument("--spot-x-um", type=float, default=300.0)
    parser.add_argument("--efficiency-weight", type=float, default=0.10)
    parser.add_argument("--balance-weight", type=float, default=0.05)
    parser.add_argument("--smoothness-weight", type=float, default=1e-5)
    parser.add_argument("--shape-only-until", type=int, default=100)
    parser.add_argument("--background-ramp-end", type=int, default=300)
    parser.add_argument("--mraf-bias", type=float, default=0.05)
    parser.add_argument("--mraf-mixing", type=float, default=0.05)
    parser.add_argument("--gradient-clip", type=float, default=1.0)
    parser.add_argument("--no-cosine-decay", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", type=Path, default=Path("results/modular_scan"))


def summarize(payload: dict[str, object]) -> dict[str, object]:
    metrics = payload["metrics"]
    correlations = [float(item["correlation"]) for item in metrics]
    backgrounds = [float(item["background_fraction"]) for item in metrics]
    efficiencies = [float(item["roi_efficiency"]) for item in metrics]
    return {
        "algorithm": payload["algorithm"],
        "seed": payload["seed"],
        "wavelengths_nm": payload["wavelengths_nm"],
        "correlations": correlations,
        "background_fractions": backgrounds,
        "roi_efficiencies": efficiencies,
        "mean_correlation": float(np.mean(correlations)),
        "worst_correlation": float(np.min(correlations)),
        "mean_background": float(np.mean(backgrounds)),
        "mean_roi_efficiency": float(np.mean(efficiencies)),
        "final_loss": payload["final_loss"],
    }


def save_pareto(path: Path, rows: list[dict[str, object]]) -> None:
    figure, axis = plt.subplots(figsize=(6.5, 4.5), constrained_layout=True)
    x_values = [row["mean_background"] for row in rows]
    y_values = [row["worst_correlation"] for row in rows]
    axis.scatter(x_values, y_values, c=[row["mean_roi_efficiency"] for row in rows], cmap="viridis", s=55)
    for row in rows:
        axis.annotate(str(row["case"]), (row["mean_background"], row["worst_correlation"]), fontsize=7)
    axis.set_xlabel("mean background fraction")
    axis.set_ylabel("worst-wavelength correlation")
    axis.set_title("Target/loss scan Pareto view")
    axis.grid(alpha=0.25)
    figure.savefig(path, dpi=170)
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--kind", choices=["target", "weights"], default="target")
    parser.add_argument("--radii", type=float, nargs="+", default=[30, 40, 50, 60])
    parser.add_argument("--edge-widths", type=float, nargs="+", default=[10, 15, 20, 25])
    parser.add_argument("--background-weights", type=float, nargs="+", default=[0.05, 0.10, 0.25, 0.50])
    parser.add_argument("--radius-um", type=float, default=40.0)
    parser.add_argument("--edge-width-um", type=float, default=15.0)
    add_common_arguments(parser)
    args = parser.parse_args()
    args.wavelengths = list(args.wavelengths)
    args.wavelength_weights = None
    args.validate_band = False
    args.no_cosine_decay = args.no_cosine_decay
    wavelengths_m = parse_wavelengths(args.wavelengths)
    system = make_system(args)
    rows: list[dict[str, object]] = []
    cases: list[tuple[str, float, float, float]] = []
    if args.kind == "target":
        for radius in args.radii:
            for edge_width in args.edge_widths:
                cases.append((f"r{radius:g}_e{edge_width:g}", radius, edge_width, 0.25))
    else:
        for background_weight in args.background_weights:
            cases.append((f"bg{background_weight:g}", args.radius_um, args.edge_width_um, background_weight))
    for index, (case, radius, edge_width, background_weight) in enumerate(cases, start=1):
        print(f"[{index}/{len(cases)}] {case}")
        case_args = argparse.Namespace(**vars(args))
        case_args.radius_um = radius
        case_args.edge_width_um = edge_width
        case_args.background_weight = background_weight
        case_dir = args.output_dir / case
        payload, _ = run_one(case_args, wavelengths_m, system, case_dir, "adam", args.seed)
        row = summarize(payload)
        row["case"] = case
        row["radius_um"] = radius
        row["edge_width_um"] = edge_width
        row["background_weight"] = background_weight
        rows.append(row)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda row: (-row["worst_correlation"], row["mean_background"]))
    (args.output_dir / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    save_pareto(args.output_dir / "pareto.png", rows)
    print(f"best case: {rows[0]['case']}")
    print(f"saved scan to {args.output_dir}")


if __name__ == "__main__":
    main()
