"""Continue one saved screen model on a finer grid or more wavelengths.

This is a sensitivity/continuation experiment, not a new fair multistart
comparison. It preserves the source target, aperture and physical detector.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from fourier_optics_step11_free_screen_optimization import (
    Model, Problem, load_case, numbers, save_json, solve, verify_math,
)


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--source",type=Path,required=True,help="Completed case directory")
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--model",default="spectral_degree_2")
    cli.add_argument("--grid",type=int)
    cli.add_argument("--train-nm",help="Comma-separated replacement design wavelengths")
    cli.add_argument("--iterations",type=int,default=300)
    args = cli.parse_args()
    source = json.loads((args.source/"results.json").read_text(encoding="utf-8"))
    cfg,meta = source["configuration"],source["metadata"]
    record = next(r for r in source["models"] if r["name"]==args.model)
    with np.load(args.source/"designs.npz",allow_pickle=False) as z:
        saved = {k:z[k] for k in z.files}
    n = args.grid or meta["screen_N"]
    if n<16 or n%2 or args.iterations<1:
        cli.error("Grid must be even >=16 and iterations positive")
    waves = numbers(args.train_nm) if args.train_nm else saved["train_nm"]
    band = saved["band_nm"]
    if len(waves)==0 or np.any(~np.isfinite(waves)) or np.any(waves<band[0]) or np.any(waves>band[1]):
        cli.error("Training wavelengths must be finite and within the source band")
    with np.load(cfg["archive"],allow_pickle=False) as archive:
        pupil,target,_,x,X,metadata = load_case(archive,meta["target"],n,meta["detector_M"],
            cfg["grid_wavelength_nm"],meta["focal_length_m"],cfg.get("target_npy"))
    if not np.allclose(target,saved["target"]) or not np.allclose(X,saved["detector_axis_m"]):
        raise ValueError("Physical target/detector changed")
    p = Problem(pupil,target,waves,band,cfg["grid_wavelength_nm"])
    kind = "phase" if args.model=="phase" else "complex" if args.model=="complex" else "spectral"
    model = Model(p,kind,record["degree"])
    old_x = saved["screen_axis_m"]
    ix = (x-old_x[0])/(old_x[1]-old_x[0])
    yy,xx = np.meshgrid(ix,ix,indexing="ij")
    coeff = []
    for c in saved[args.model+"_coefficients"]:
        old = np.zeros(saved["pupil"].shape,complex)
        old[saved["pupil"].astype(bool)] = c
        coeff.append(map_coordinates(old,[yy,xx],order=0,mode="nearest")[pupil])
    coeff = np.asarray(coeff)
    theta = np.angle(coeff[0]) if kind=="phase" else model.pack(coeff)
    checks = verify_math()
    args.output.mkdir(parents=True,exist_ok=False)
    scan = np.unique(np.r_[np.arange(band[0],band[1]+.1,5.),waves,saved["train_nm"]])
    before,_,_ = model.evaluate(theta,waves)
    before_scan,_,_ = model.evaluate(theta,scan)
    fitted,loss,runs = solve(model,[theta],args.iterations,meta["target"]+"/validation/"+args.model)
    after,images,screens = model.evaluate(fitted,waves,True)
    after_scan,_,_ = model.evaluate(fitted,scan)
    if max(x["captured_power"] for x in after_scan)>1.005:
        raise AssertionError("Detector power quadrature failed")
    payload = {"source":str(args.source.resolve()),"model":args.model,"metadata":metadata,
        "training_nm":waves.tolist(),"band_nm":band.tolist(),"best_loss":loss,"runs":runs,
        "before_training":before,"after_training":after,"before_scan":before_scan,"after_scan":after_scan}
    save_json(args.output/"results.json",payload)
    save_json(args.output/"math_checks.json",checks)
    np.savez_compressed(args.output/"design.npz",coefficients=model.unpack(fitted),theta=fitted,
        pupil=pupil,target=target,screen_axis_m=x,detector_axis_m=X,training_nm=waves,band_nm=band,
        images=images.astype(np.float32),screens=screens.astype(np.complex64))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig,ax = plt.subplots(figsize=(8,4))
    for label,rows in [("Before continuation",before_scan),("After continuation",after_scan)]:
        ax.plot([r["wavelength_nm"] for r in rows],[r["relative_L2"] for r in rows],label=label)
    for w in waves:
        ax.axvline(w,color="gray",lw=.5,ls=":")
    ax.set(xlabel="Wavelength (nm)",ylabel="Relative L2, fixed power")
    ax.legend()
    ax.grid(alpha=.2)
    fig.tight_layout()
    fig.savefig(args.output/"validation_scan.png",dpi=160)
    plt.close(fig)
    summary = {"before_training_RMS":float(np.sqrt(np.mean([r["relative_L2"]**2 for r in before]))),
        "after_training_RMS":float(np.sqrt(2*loss)),
        "before_scan_worst":max(r["relative_L2"] for r in before_scan),
        "after_scan_worst":max(r["relative_L2"] for r in after_scan)}
    save_json(args.output/"summary.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__=="__main__":
    main()
