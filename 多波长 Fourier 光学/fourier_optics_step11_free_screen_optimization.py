"""Material-free inverse design with a fixed Fourier lens and a fixed pupil.

Requires numpy, scipy and matplotlib. See fourier_optics_step11_README.md.
Screen power is normalized BEFORE propagation, never inside the detector.
This is the project's scalar, point-sampled Fourier-lens model, not a material
or passive-device simulation. All saved errors are best-found, not certified
global minima. Chebyshev coefficients are followed by per-wavelength screen
power normalization, which is included in the analytic derivative.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import minimize
from scipy.signal import CZT

HERE = Path(__file__).resolve().parent


class FourierLens:
    """Exact discrete Fourier quadrature on a common physical detector grid.

    alpha = dx*dX/(lambda*f), with dX=grid_lambda*f/(N*dx).
    K[k,j]=exp(-2*pi*i*alpha*(j-N/2)*(k-M/2)); A(u)=alpha*K*u*K.T.
    The rectangular CZT adjoint is constructed explicitly (not an inverse).
    The physical screen width N*dx and detector pitch stay fixed on refinement.
    """
    def __init__(self, n, m, wavelength_nm, grid_nm=400.0):
        self.n, self.m = n, m
        self.alpha = grid_nm / (wavelength_nm * n)
        if self.alpha * m > 1 + 1e-12:
            raise ValueError("Detector exceeds the native Nyquist window; increase --grid")
        self.fwd = self._plan(n, m, self.alpha)
        self.adj = self._plan(m, n, -self.alpha)

    @staticmethod
    def _plan(n, m, alpha):
        plan = CZT(n, m=m, w=np.exp(-2j*np.pi*alpha),
                   a=np.exp(-2j*np.pi*alpha*(m//2)))
        post = np.exp(2j*np.pi*alpha*(n//2)*(np.arange(m)-m//2))
        return plan, post

    def _apply(self, field, plans):
        plan, post = plans
        temp = plan(field, axis=1) * post[None, :]
        return self.alpha * plan(temp, axis=0) * post[:, None]

    def forward(self, field):
        return self._apply(field, self.fwd)

    def adjoint(self, field):
        return self._apply(field, self.adj)


class Problem:
    def __init__(self, pupil, target, waves, band, grid_nm=400.0,
                 loss_profile="mean", roi_floor=0.80, roi_weight=0.10,
                 cvar_weight=0.30, cvar_tail_fraction=0.10, region_masks=None):
        self.pupil, self.target = pupil, target
        self.waves, self.band, self.grid_nm = np.asarray(waves), np.asarray(band), grid_nm
        self.n, self.m = pupil.shape[0], target.shape[0]
        self.count = int(pupil.sum())
        self.target_sq = float(np.sum(target**2))
        self.roi = target >= 0.01*target.max()
        if (target.ndim != 2 or target.shape[0] != target.shape[1]
            or np.any(~np.isfinite(target)) or np.any(target < 0)
            or self.target_sq <= 0 or self.count < 1):
            raise ValueError("Target and pupil must be finite, nonnegative, and nonempty")
        if loss_profile not in {"mean", "robust_cvar"}:
            raise ValueError("loss_profile must be mean or robust_cvar")
        if (not np.isfinite(roi_floor) or not np.isfinite(roi_weight)
            or not 0 <= roi_floor <= 1 or roi_weight < 0):
            raise ValueError("ROI floor must be in [0, 1] and its weight nonnegative")
        if (not np.isfinite(cvar_weight) or not np.isfinite(cvar_tail_fraction)
            or not 0 <= cvar_weight <= 1 or not 0 < cvar_tail_fraction <= 1):
            raise ValueError("CVaR weights/fraction must be in [0, 1] and (0, 1]")
        self.loss_profile = loss_profile
        self.roi_floor = float(roi_floor)
        self.roi_weight = float(roi_weight)
        self.cvar_weight = float(cvar_weight)
        self.cvar_tail_fraction = float(cvar_tail_fraction)
        self.region_masks = region_masks or {}
        for name, mask in self.region_masks.items():
            if np.asarray(mask).shape != target.shape or not np.any(mask):
                raise ValueError(f"Region {name!r} must be a nonempty detector-grid mask")
        self.op_by_wavelength = {}
        self.ops = [self.operator(w) for w in waves]

    def operator(self, wavelength_nm):
        key = float(wavelength_nm)
        if key not in self.op_by_wavelength:
            self.op_by_wavelength[key] = FourierLens(
                self.n, self.m, key, self.grid_nm)
        return self.op_by_wavelength[key]

    def basis(self, waves, degree):
        x = (2*np.asarray(waves)-sum(self.band))/(self.band[1]-self.band[0])
        return np.polynomial.chebyshev.chebvander(x, degree)


class Model:
    def __init__(self, p, kind, degree=0):
        self.p, self.kind, self.degree = p, kind, degree
        self.basis = p.basis(p.waves, degree)

    def unpack(self, theta):
        if self.kind == "phase":
            return np.exp(1j*theta)[None, :]
        count = (self.degree+1)*self.p.count
        return (theta[:count]+1j*theta[count:]).reshape(self.degree+1, -1)

    @staticmethod
    def pack(coeff):
        return np.concatenate([coeff.real.ravel(), coeff.imag.ravel()])

    def value_gradient(self, theta):
        p = self.p
        coeff = self.unpack(theta)
        raw = self.basis @ coeff
        gc = np.zeros_like(coeff)
        residuals = []
        roi_powers = []
        normalized_fields = []
        intensities = []
        for q, op in zip(raw, p.ops):
            norm = np.linalg.norm(q)
            if not np.isfinite(norm) or norm < 1e-14:
                raise FloatingPointError("Screen power is zero/nonfinite")
            u = np.zeros((p.n, p.n), complex)
            u[p.pupil] = a = q/norm
            v = op.forward(u)
            intensity = np.abs(v)**2
            residuals.append(intensity-p.target)
            intensities.append(intensity)
            roi_powers.append(float(np.sum(intensity[p.roi])))
            normalized_fields.append((a, v, norm))

        residuals = np.asarray(residuals)
        roi_powers = np.asarray(roi_powers)
        relative_l2_sq = np.sum(residuals**2, axis=(1, 2))/p.target_sq
        if p.loss_profile == "mean":
            spectral_weights = np.full(len(p.waves), 0.5/len(p.waves))
            loss = float(np.mean(relative_l2_sq)*0.5)
        else:
            tail_weights = upper_tail_mean_weights(relative_l2_sq,
                                                   p.cvar_tail_fraction)
            spectral_weights = ((1-p.cvar_weight)/len(p.waves)
                                + p.cvar_weight*tail_weights)
            loss = float((1-p.cvar_weight)*np.mean(relative_l2_sq)
                         + p.cvar_weight*np.dot(tail_weights, relative_l2_sq))
            efficiency_deficit = np.maximum(p.roi_floor-roi_powers, 0.0)
            loss += p.roi_weight*float(np.mean(efficiency_deficit**2))

        for index, (b, residual, (a, v, norm), op) in enumerate(
                zip(self.basis, residuals, normalized_fields, p.ops)):
            image_gradient = (2*spectral_weights[index]/p.target_sq)*residual
            if p.loss_profile == "robust_cvar":
                deficit = max(p.roi_floor-roi_powers[index], 0.0)
                image_gradient -= (2*p.roi_weight*deficit/len(p.waves))*p.roi
            gu = 2*op.adjoint(image_gradient*v)[p.pupil]
            # dL=Re(vdot(gq,dq)); differentiate u=q/||q|| as a REAL map.
            gq = (gu-a*np.vdot(a, gu).real)/norm
            gc += b[:, None]*gq[None, :]
        grad = np.imag(np.conj(coeff[0])*gc[0]) if self.kind == "phase" else self.pack(gc)
        return float(loss), grad

    def evaluate(self, theta, waves, save_arrays=False):
        p = self.p
        raw = p.basis(waves, self.degree) @ self.unpack(theta)
        rows, images, screens = [], [], []
        for w, q in zip(waves, raw):
            norm = np.linalg.norm(q)
            if norm < 1e-14 or not np.isfinite(norm):
                raise FloatingPointError(f"Undefined zero/nonfinite screen at {w} nm")
            u = np.zeros((p.n, p.n), complex)
            u[p.pupil] = q/norm
            image = np.abs(p.operator(w).forward(u))**2
            captured = float(image.sum())
            screen = np.sqrt(p.count)*u
            metrics = {"wavelength_nm":float(w),
                "relative_L2":float(np.linalg.norm(image-p.target)/np.sqrt(p.target_sq)),
                "captured_power":captured, "roi_power":float(image[p.roi].sum()),
                "renormalized_detector_L2_diagnostic":float(np.linalg.norm(image/max(captured,1e-300)-p.target)/np.sqrt(p.target_sq)),
                "peak_transmission_amplitude":float(np.abs(screen).max())}
            for name, mask in p.region_masks.items():
                a, b = image[mask], p.target[mask]
                aa, bb = a-a.mean(), b-b.mean()
                denom = np.linalg.norm(aa)*np.linalg.norm(bb)
                metrics[f"correlation_{name}"] = float(np.sum(aa*bb)/denom) if denom > 0 else 0.0
            rows.append(metrics)
            if save_arrays:
                images.append(image)
                screens.append(screen)
        return rows, np.asarray(images), np.asarray(screens)


def objective_from_metrics(rows, loss_profile="mean", roi_floor=0.80,
                           roi_weight=0.10, cvar_weight=0.30,
                           cvar_tail_fraction=0.10):
    """Reconstruct the scalar training objective from saved per-wave metrics."""
    errors_sq = np.asarray([row["relative_L2"]**2 for row in rows], dtype=float)
    if errors_sq.size == 0 or np.any(~np.isfinite(errors_sq)):
        raise ValueError("Metrics must contain finite per-wavelength errors")
    if loss_profile == "mean":
        return float(0.5 * errors_sq.mean())
    if loss_profile != "robust_cvar":
        raise ValueError("loss_profile must be mean or robust_cvar")
    tail_weights = upper_tail_mean_weights(errors_sq, cvar_tail_fraction)
    roi = np.asarray([row["roi_power"] for row in rows], dtype=float)
    deficit = np.maximum(roi_floor-roi, 0.0)
    return float((1-cvar_weight)*errors_sq.mean()
                 + cvar_weight*np.dot(tail_weights, errors_sq)
                 + roi_weight*np.dot(deficit, deficit)/len(rows))


def upper_tail_mean_weights(values, tail_fraction):
    """Exact empirical upper-tail CVaR weights, including a fractional sample."""
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or values.size == 0 or np.any(~np.isfinite(values)):
        raise ValueError("values must be a nonempty finite vector")
    if not 0 < tail_fraction <= 1:
        raise ValueError("tail_fraction must be in (0, 1]")
    mass = values.size*tail_fraction
    full_count = int(np.floor(mass))
    fractional = mass-full_count
    order = np.argsort(values)[::-1]
    weights = np.zeros(values.size, dtype=float)
    weights[order[:full_count]] = 1.0
    if fractional > 1e-12:
        weights[order[full_count]] = fractional
    return weights/mass


def verify_math():
    rng = np.random.default_rng(117)
    checks = {"operators":[], "gradients":[]}
    for n, m in [(16,16), (24,16)]:
        j, k = np.arange(n)-n//2, np.arange(m)-m//2
        u = rng.normal(size=(n,n))+1j*rng.normal(size=(n,n))
        v = rng.normal(size=(m,m))+1j*rng.normal(size=(m,m))
        for w in [400.,450.,532.,700.]:
            op = FourierLens(n,m,w)
            kernel = np.exp(-2j*np.pi*op.alpha*np.outer(k,j))
            direct = op.alpha*kernel@u@kernel.T
            matrix_error = np.linalg.norm(op.forward(u)-direct)/np.linalg.norm(direct)
            lhs, rhs = np.vdot(op.forward(u),v), np.vdot(u,op.adjoint(v))
            adjoint_error = abs(lhs-rhs)/max(abs(lhs),abs(rhs),1e-12)
            assert max(matrix_error,adjoint_error)<1e-10, (n,m,w,matrix_error,adjoint_error)
            checks["operators"].append({"N":n,"M":m,"wavelength_nm":w,
                "matrix_relative_error":float(matrix_error),"adjoint_relative_error":float(adjoint_error)})
    n,m = 24,16
    yy,xx = np.mgrid[:n,:n]-n//2
    pupil = xx**2+yy**2<=7**2
    yy,xx = np.mgrid[:m,:m]-m//2
    target = np.exp(-((xx-2)**2+(yy+1)**2)/5)+.4*np.exp(-((xx+3)**2+(yy-2)**2)/3)
    target /= target.sum()
    p = Problem(pupil,target,[450.,532.,633.],[400.,700.])
    for kind, degree in [("phase",0),("complex",0),("spectral",1),("spectral",2)]:
        model = Model(p,kind,degree)
        coeff = .2*(rng.normal(size=(degree+1,p.count))+1j*rng.normal(size=(degree+1,p.count)))
        coeff[0] += 1
        theta = rng.normal(size=p.count) if kind=="phase" else model.pack(coeff)
        value, gradient = model.value_gradient(theta)
        for trial in range(3):
            d = rng.normal(size=theta.size)
            d /= np.linalg.norm(d)
            h = 1e-5
            numerical = (model.value_gradient(theta+h*d)[0]-model.value_gradient(theta-h*d)[0])/(2*h)
            analytic = float(gradient@d)
            error = abs(numerical-analytic)
            assert error < 2e-7*max(1.,abs(numerical),abs(analytic)), (kind,degree,numerical,analytic)
            checks["gradients"].append({"kind":kind,"degree":degree,"trial":trial,
                "analytic":analytic,"finite_difference":float(numerical),"absolute_error":float(error)})
        if kind!="phase":
            assert np.isclose(value,model.value_gradient(theta*2.3)[0],rtol=1e-12)
    phi = rng.normal(size=p.count)
    base_value = Model(p,"phase").value_gradient(phi)[0]
    for degree in [0,1,2]:
        model = Model(p,"complex",degree)
        c = np.zeros((degree+1,p.count),complex)
        c[0] = np.exp(1j*phi)
        assert np.isclose(model.value_gradient(model.pack(c))[0],base_value,rtol=1e-12)
    model_two = Model(p,"spectral",2)
    model_three = Model(p,"spectral",3)
    nested_two = .2*(rng.normal(size=(3,p.count))+1j*rng.normal(size=(3,p.count)))
    nested_three = np.zeros((4,p.count),complex)
    nested_three[:3] = nested_two
    theta_two = model_two.pack(nested_two)
    theta_three = model_three.pack(nested_three)
    assert np.isclose(model_two.value_gradient(theta_two)[0],
                      model_three.value_gradient(theta_three)[0],rtol=1e-12)
    checks["degree_2_to_degree_3_zero_extension"] = "passed"
    op = FourierLens(m,m,400.)
    desired = np.sqrt(target)*np.exp(1j*rng.uniform(-np.pi,np.pi,(m,m)))
    recovered = op.forward(op.adjoint(desired))
    inverse_error = np.linalg.norm(np.abs(recovered)**2-target)/np.linalg.norm(target)
    assert inverse_error < 1e-11
    checks["full_aperture_single_wave_inverse_DFT_relative_L2"] = float(inverse_error)
    checks["normalization_scale_invariance_and_model_nesting"] = "passed"

    # Check the production robust objective, including its active ROI-floor
    # penalty, away from a tie in the selected upper-tail wavelengths.
    robust_p = Problem(pupil, target, [420., 460., 510., 570., 620., 680.],
                       [400., 700.], loss_profile="robust_cvar", roi_floor=0.99,
                       roi_weight=0.10, cvar_weight=0.30,
                       cvar_tail_fraction=1/3)
    robust_model = Model(robust_p, "spectral", 2)
    robust_coeff = .2*(rng.normal(size=(3,p.count))+1j*rng.normal(size=(3,p.count)))
    robust_coeff[0] += 1
    robust_theta = robust_model.pack(robust_coeff)
    value, gradient = robust_model.value_gradient(robust_theta)
    d = rng.normal(size=robust_theta.size)
    d /= np.linalg.norm(d)
    h = 2e-6
    numerical = (robust_model.value_gradient(robust_theta+h*d)[0]
                 - robust_model.value_gradient(robust_theta-h*d)[0])/(2*h)
    analytic = float(gradient@d)
    error = abs(numerical-analytic)
    assert np.isfinite(value) and error < 2e-7*max(1.,abs(numerical),abs(analytic)), (
        value, numerical, analytic, error)
    robust_rows, _, _ = robust_model.evaluate(robust_theta, robust_p.waves)
    reconstructed = objective_from_metrics(robust_rows, "robust_cvar", 0.99,
                                           0.10, 0.30, 1/3)
    assert np.isclose(value, reconstructed, rtol=1e-10, atol=1e-12)
    checks["robust_objective_gradient"] = {
        "analytic": analytic, "finite_difference": float(numerical),
        "absolute_error": float(error), "objective_reconstruction": "passed"}
    return checks


def solve(model, starts, iterations, label, return_candidates=False,
          on_progress=None):
    best_theta, best_loss, runs = None, np.inf, []
    candidate_thetas = []
    for restart, initial in enumerate(starts):
        tic = time.perf_counter()
        state = {"loss":np.inf,"theta":initial.copy(),"calls":0,"last_theta":None}
        history = []
        def objective(theta):
            value, gradient = model.value_gradient(theta)
            state["calls"] += 1
            if value < state["loss"]:
                state.update(loss=value,theta=theta.copy())
            state.update(last_theta=theta.copy(),last_value=value,last_gradient=gradient.copy())
            return value, gradient
        def callback(theta):
            if state["last_theta"] is not None and np.array_equal(theta,state["last_theta"]):
                value,gradient = state["last_value"],state["last_gradient"]
            else:
                value,gradient = model.value_gradient(theta)
            i = len(history)
            history.append({"iteration":i,"loss":value,"gradient_inf":float(np.max(np.abs(gradient)))})
            if i%25==0:
                print(f"{label} start={restart} iter={i} objective={value:.6g}",flush=True)
                if on_progress is not None:
                    on_progress(restart, i, state["theta"].copy(), float(state["loss"]))
        objective(initial)
        callback(initial)
        result = minimize(objective,initial,jac=True,method="L-BFGS-B",callback=callback,
            options={"maxiter":iterations,"maxcor":12,"maxls":40,"maxfun":max(2000,40*iterations),
                     "ftol":1e-12,"gtol":1e-9})
        if state["loss"] < best_loss:
            best_loss,best_theta = state["loss"],state["theta"].copy()
        runs.append({"restart":restart,"best_loss":float(state["loss"]),"success":bool(result.success),
            "message":str(result.message),"iterations":int(result.nit),"evaluations":state["calls"],
            "seconds":time.perf_counter()-tic,"history":history})
        candidate_thetas.append(state["theta"].copy())
        if on_progress is not None:
            on_progress(restart, int(result.nit), state["theta"].copy(),
                        float(state["loss"]))
        print(f"{label} start={restart} best objective={state['loss']:.6g}; {result.message}",flush=True)
    if return_candidates:
        return best_theta,float(best_loss),runs,candidate_thetas
    return best_theta,float(best_loss),runs


def validation_summary(rows, band, l2_limit=0.10, roi_floor=0.80):
    in_band = [row for row in rows if band[0] <= row["wavelength_nm"] <= band[1]]
    if not in_band:
        raise ValueError("Validation scan has no points inside the design band")
    errors = np.asarray([row["relative_L2"] for row in in_band], dtype=float)
    roi = np.asarray([row["roi_power"] for row in in_band], dtype=float)
    if np.any(~np.isfinite(errors)) or np.any(~np.isfinite(roi)):
        raise FloatingPointError("Validation metrics contain NaN or Inf")
    worst_index = int(np.argmax(errors))
    return {
        "design_band_nm": [float(band[0]), float(band[1])],
        "sample_count": len(in_band),
        "mean_relative_L2": float(errors.mean()),
        "p95_relative_L2": float(np.quantile(errors, 0.95)),
        "worst_relative_L2": float(errors[worst_index]),
        "worst_wavelength_nm": float(in_band[worst_index]["wavelength_nm"]),
        "mean_roi_power": float(roi.mean()),
        "p05_roi_power": float(np.quantile(roi, 0.05)),
        "minimum_roi_power": float(roi.min()),
        "relative_L2_limit": float(l2_limit),
        "roi_power_floor": float(roi_floor),
        "passed": bool(np.all(errors <= l2_limit) and np.all(roi >= roi_floor)),
    }


def select_adaptive_wavelengths(rows, training_waves, band, l2_limit=0.10,
                                roi_floor=0.80, max_additions=6, separation_nm=5.0):
    """Pick separated in-band failures, prioritizing normalized constraint violation."""
    existing = np.asarray(training_waves, dtype=float)
    candidates = []
    for row in rows:
        wavelength = float(row["wavelength_nm"])
        if not band[0] <= wavelength <= band[1]:
            continue
        if np.any(np.isclose(existing, wavelength, atol=1e-8, rtol=0)):
            continue
        score = max(row["relative_L2"] / l2_limit,
                    roi_floor / max(row["roi_power"], 1e-12))
        if score > 1.0:
            candidates.append((score, wavelength))
    selected = []
    while candidates and len(selected) < max_additions:
        score, wavelength = max(candidates)
        selected.append(wavelength)
        candidates = [item for item in candidates
                      if abs(item[1] - wavelength) >= separation_nm]
    return np.asarray(sorted(selected), dtype=float)


def load_case(archive, prefix, n, m, grid_nm, focal, custom=None):
    old_pupil = archive["pupil"].astype(bool)
    old_n = old_pupil.shape[0]
    old_dx = float(archive["input_pixel_pitch_m"])
    dx = old_n*old_dx/n
    x = (np.arange(n)-n//2)*dx
    indices = x/old_dx+old_n//2
    yi,xi = np.meshgrid(indices,indices,indexing="ij")
    pupil = map_coordinates(old_pupil.astype(float),[yi,xi],order=0,mode="constant",cval=0)>0.5
    dX = grid_nm*1e-9*focal/(n*dx)
    X = (np.arange(m)-m//2)*dX
    if custom:
        target = np.asarray(np.load(custom,allow_pickle=False),float)
        if target.shape != (m,m):
            raise ValueError("Custom .npy must match the detector grid, not the screen grid")
        retained = None
    else:
        old_target = np.asarray(archive[prefix+"_target_intensity"],float)
        old_dX = float(archive["wavelength_m"])*float(archive["focal_length_m"])/(old_n*old_dx)
        ti = X/old_dX+old_n//2
        ty,tx = np.meshgrid(ti,ti,indexing="ij")
        target = map_coordinates(old_target,[ty,tx],order=1,mode="constant",cval=0)
        target *= (dX/old_dX)**2/old_target.sum()
        retained = float(target.sum())
        if not 0.995<=retained<=1.005:
            raise ValueError(f"{prefix}: target mass={retained:.6g}; increase both detector and screen grids")
    if np.any(~np.isfinite(target)) or np.any(target<0) or target.sum()<=0 or pupil.sum()<4:
        raise ValueError("Invalid target or unresolved aperture")
    target /= target.sum()
    key = prefix+"_phase_rad"
    phase = (map_coordinates(archive[key],[yi,xi],order=0,mode="constant",cval=0)[pupil]
             if not custom and key in archive.files else np.zeros(int(pupil.sum())))
    metadata = {"target":prefix,"screen_N":n,"detector_M":m,"input_pitch_m":dx,
        "detector_pitch_m":dX,"focal_length_m":focal,"aperture_pixels":int(pupil.sum()),
        "original_aperture_area_m2":float(old_pupil.sum()*old_dx**2),
        "sampled_aperture_area_m2":float(pupil.sum()*dx**2),"target_retained_mass":retained}
    return pupil,target,phase,x,X,metadata


def save_json(path, payload):
    path.write_text(json.dumps(payload,indent=2,ensure_ascii=False,default=str),encoding="utf-8")


def plots(out, target, X, waves, records, arrays):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    extent = [X[0]*1e3,X[-1]*1e3]*2
    fig,axes = plt.subplots(len(records)+1,len(waves),figsize=(4*len(waves),2.6*(len(records)+1)),squeeze=False)
    for col,w in enumerate(waves):
        axes[0,col].imshow(target/target.max(),origin="lower",extent=extent,cmap="magma",vmin=0,vmax=1)
        axes[0,col].set_title(f"Target, {w:g} nm")
    for row,r in enumerate(records,1):
        for col,metric in enumerate(r["training"]):
            axes[row,col].imshow(arrays[r["name"]+"_images"][col]/target.max(),origin="lower",
                                  extent=extent,cmap="magma",vmin=0,vmax=1)
            axes[row,col].set_title(f'{r["name"]}, L2={metric["relative_L2"]:.3g}')
    for ax in axes.ravel():
        ax.set_xlabel("X (mm)")
        ax.set_ylabel("Y (mm)")
    fig.tight_layout()
    fig.savefig(out/"patterns.png",dpi=140)
    plt.close(fig)
    fig,axes = plt.subplots(1,3,figsize=(15,4))
    for r in records:
        rows = r["scan"]
        axes[0].plot([x["wavelength_nm"] for x in rows],[x["relative_L2"] for x in rows],label=r["name"])
        axes[1].plot([x["wavelength_nm"] for x in rows],[x["roi_power"] for x in rows],label=r["name"])
        run = min(r["runs"],key=lambda run:run["best_loss"])
        axes[2].semilogy([h["iteration"] for h in run["history"]],
                        [max(h["loss"],1e-16) for h in run["history"]],label=r["name"])
    for ax in axes[:2]:
        for w in waves:
            ax.axvline(w,color="gray",lw=.5,ls=":")
        ax.set_xlabel("Wavelength (nm)")
    axes[0].set_ylabel("Relative L2 (fixed power)")
    axes[1].set_ylabel("ROI / total transmitted power")
    axes[2].set(xlabel="Iteration, best restart",ylabel="Training objective")
    for ax in axes:
        ax.grid(alpha=.2)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out/"scan_and_convergence.png",dpi=150)
    plt.close(fig)


def report(out, metadata, records, waves, problem=None):
    lines = ["# 自由屏函数数值比较", "", "本报告由实际运行生成；数值为当前离散模型和预算下找到的最好解，不是全局最优证明。", "",
        f"屏网格 {metadata['screen_N']}；观察面网格 {metadata['detector_M']}；固定焦距 {metadata['focal_length_m']} m。",
        f"训练波长：{list(map(float,waves))} nm。所有模型使用同一孔径、物理观察面、目标及全平面总透射功率。", "",
        "| 模型 | 实参数数 | 训练 RMS 相对 L2 | 训练平均 L2 | 训练最差 L2 | 未训练扫描最差 L2 | 训练最小 ROI |",
        "|---|---:|---:|---:|---:|---:|---:|"]
    for r in records:
        held = [x for x in r["scan"] if not np.any(np.isclose(x["wavelength_nm"],waves,rtol=0,atol=1e-9))]
        worst = f'{max(x["relative_L2"] for x in held):.6g}' if held else "无"
        training_rms = float(np.sqrt(np.mean([x["relative_L2"]**2 for x in r["training"]])))
        lines.append(f'| {r["name"]} | {r["parameters"]} | {training_rms:.6g} | '
            f'{np.mean([x["relative_L2"] for x in r["training"]]):.6g} | '
            f'{max(x["relative_L2"] for x in r["training"]):.6g} | {worst} | '
            f'{min(x["roi_power"] for x in r["training"]):.6g} |')
    if problem is not None and problem.loss_profile == "robust_cvar":
        loss_description = (f"损失为 {(1-problem.cvar_weight):g}×全波长平均相对 L2² + "
            f"{problem.cvar_weight:g}×最差 {100*problem.cvar_tail_fraction:g}% 相对 L2² 的经验 CVaR "
            f"+ {problem.roi_weight:g}×平均 max(0, {problem.roi_floor:g}−ROI 功率)²。")
    else:
        loss_description = "损失是各波长相对 L2 平方均值的一半。"
    lines += ["", loss_description + "没有按观察窗内功率重新归一化。",
        "自由复屏允许局部振幅超过 1；ROI 分母为总透射功率，不是现实器件效率。", "",
        "色散复屏采用 Chebyshev 复系数展开后进行逐波长屏面功率归一化。基矩阵秩达到训练波长数时，"
        "训练点可表达独立复屏；未训练波长误差可能增加。", "",
        "固定焦距的波长缩放包含在 CZT 算子中。数学检查见上级 math_checks.json。"
        "网格与优化收敛需要独立验证；点采样标量模型未包含像素 sinc、矢量效应或材料。"]
    for r in records:
        lines += ["", f"## {r['name']}", "", f"训练基矩阵秩：{r['basis_rank']}。", "",
                  "| 波长 nm | 相对 L2 | 观察窗功率 | ROI 功率 | 峰值透射振幅 |", "|---:|---:|---:|---:|---:|"]
        for x in r["training"]:
            lines.append(f'| {x["wavelength_nm"]:g} | {x["relative_L2"]:.6g} | {x["captured_power"]:.6g} | '
                         f'{x["roi_power"]:.6g} | {x["peak_transmission_amplitude"]:.6g} |')
        lines += ["", "各起点停止原因：", ""]
        lines += [f'- {run["restart"]}: {run["message"]}; best loss={run["best_loss"]:.8g}' for run in r["runs"]]
    (out/"report.md").write_text("\n".join(lines)+"\n",encoding="utf-8")


def numbers(value):
    return np.asarray([float(x) for x in value.split(",")])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive",type=Path,default=HERE/"fourier_optics_step3_phase_screens.npz")
    parser.add_argument("--targets",default="all")
    parser.add_argument("--target-npy",type=Path)
    parser.add_argument("--grid",type=int,default=256)
    parser.add_argument("--detector-grid",type=int,default=256)
    parser.add_argument("--grid-wavelength-nm",type=float,default=400.)
    parser.add_argument("--focal-m",type=float,default=.2)
    parser.add_argument("--train-nm",default="450,532,633")
    parser.add_argument("--band-nm",default="400,700")
    parser.add_argument("--scan-step-nm",type=float,default=10.)
    parser.add_argument("--degrees",default="1,2")
    parser.add_argument("--loss-profile",choices=("mean","robust_cvar"),default="mean")
    parser.add_argument("--roi-floor",type=float,default=0.80)
    parser.add_argument("--roi-weight",type=float,default=0.10)
    parser.add_argument("--cvar-weight",type=float,default=0.30)
    parser.add_argument("--cvar-tail-fraction",type=float,default=0.10)
    parser.add_argument("--region-metadata",type=Path,
                        help="JSON metadata from make_multiscale_resolution_target.py")
    parser.add_argument("--starts",type=int,default=3)
    parser.add_argument("--iterations",type=int,default=300)
    parser.add_argument("--seed",type=int,default=20261009)
    parser.add_argument("--warm-start",type=Path,help="designs.npz from a previous run; same target/system required")
    parser.add_argument("--check-only",action="store_true")
    parser.add_argument("--output",type=Path,help="New directory; never overwrite an existing run")
    args = parser.parse_args()
    waves,band = numbers(args.train_nm),numbers(args.band_nm)
    degrees = sorted(set(int(x) for x in args.degrees.split(",") if x.strip()))
    if (min(args.grid,args.detector_grid)<16 or args.grid%2 or args.detector_grid%2
        or args.starts<1 or args.iterations<1 or args.focal_m<=0 or args.grid_wavelength_nm<=0
        or len(band)!=2 or not 0<band[0]<band[1] or args.scan_step_nm<=0
        or np.any(~np.isfinite(waves)) or np.any(~np.isfinite(band)) or len(waves)<1
        or len(np.unique(waves))!=len(waves) or np.any(waves<band[0]) or np.any(waves>band[1])
        or any(d<1 for d in degrees) or not 0<=args.roi_floor<=1
        or args.roi_weight<0 or not 0<=args.cvar_weight<=1
        or not 0<args.cvar_tail_fraction<=1):
        parser.error("Invalid grid, budget, band or polynomial degrees")
    checks = verify_math()
    print("Operator, adjoint, gradient, normalization, nesting and inverse-DFT checks PASSED.",flush=True)
    if args.check_only:
        print(json.dumps(checks,indent=2))
        return
    out = args.output or HERE/("step11_results_"+time.strftime("%Y%m%d_%H%M%S"))
    out.mkdir(parents=True,exist_ok=False)
    save_json(out/"math_checks.json",checks)
    import scipy
    save_json(out/"provenance.json",{"python":sys.version,"numpy":np.__version__,"scipy":scipy.__version__,
        "source_sha256":hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),"configuration":vars(args)})
    scan = np.unique(np.concatenate([np.arange(band[0],band[1],args.scan_step_nm),band[1:],waves]))
    rng = np.random.default_rng(args.seed)
    summary = []
    with np.load(args.archive,allow_pickle=False) as archive:
        available = sorted(k[:-len("_target_intensity")] for k in archive.files if k.endswith("_target_intensity"))
        targets = ["custom"] if args.target_npy else available if args.targets=="all" else args.targets.split(",")
        if not targets or (not args.target_npy and any(t not in available for t in targets)):
            raise ValueError(f"Available targets: {available}")
        for prefix in targets:
            if not prefix or Path(prefix).name!=prefix or prefix in {".",".."} or "/" in prefix or "\\" in prefix:
                raise ValueError("Target must be a plain name")
            dest = out/prefix
            dest.mkdir()
            pupil,target,phi,x,X,metadata = load_case(archive,prefix,args.grid,args.detector_grid,
                args.grid_wavelength_nm,args.focal_m,args.target_npy)
            region_masks = {}
            region_metadata = None
            if args.region_metadata:
                from make_multiscale_resolution_target import make_region_masks
                region_metadata = json.loads(args.region_metadata.read_text(encoding="utf-8"))
                region_masks = make_region_masks(region_metadata, X)
            p = Problem(pupil,target,waves,band,args.grid_wavelength_nm,
                        loss_profile=args.loss_profile, roi_floor=args.roi_floor,
                        roi_weight=args.roi_weight, cvar_weight=args.cvar_weight,
                        cvar_tail_fraction=args.cvar_tail_fraction,
                        region_masks=region_masks)
            metadata["region_metadata"] = region_metadata
            arrays = {"pupil":pupil,"target":target,"screen_axis_m":x,"detector_axis_m":X,
                      "train_nm":waves,"band_nm":band,"scan_nm":scan,"focal_m":np.asarray(args.focal_m)}
            warm = None
            if args.warm_start:
                with np.load(args.warm_start,allow_pickle=False) as w:
                    warm = {k:w[k] for k in w.files}
                if (warm["target"].shape!=target.shape or not np.allclose(warm["target"],target)
                    or not np.allclose(warm["detector_axis_m"],X)
                    or not np.isclose(float(warm["focal_m"]),args.focal_m)
                    or not np.array_equal(warm["band_nm"],band)):
                    raise ValueError("Warm start must use the same target, detector, band and focal length")
            records,prev,prev_loss = [],None,np.inf
            for kind,degree in [("phase",0),("complex",0)]+[("spectral",d) for d in degrees]:
                name = kind if kind!="spectral" else f"spectral_degree_{degree}"
                model = Model(p,kind,degree)
                if kind=="phase":
                    starts = [phi.copy()]+[rng.uniform(-np.pi,np.pi,p.count) for _ in range(args.starts-1)]
                else:
                    base = np.zeros((degree+1,p.count),complex)
                    base[:len(prev)] = prev
                    starts = [model.pack(base)]
                    for i in range(1,args.starts):
                        noise = (rng.normal(size=base.shape)+1j*rng.normal(size=base.shape))/np.sqrt(2)
                        starts.append(model.pack(base+(.1 if i%2 else .5)*max(np.sqrt(np.mean(np.abs(base)**2)),1e-4)*noise))
                if warm is not None and name+"_coefficients" in warm:
                    old_x = warm["screen_axis_m"]
                    ix = (x-old_x[0])/(old_x[1]-old_x[0])
                    yy,xx = np.meshgrid(ix,ix,indexing="ij")
                    cnew = []
                    for c in warm[name+"_coefficients"]:
                        field = np.zeros(warm["pupil"].shape,complex)
                        field[warm["pupil"].astype(bool)] = c
                        cnew.append(map_coordinates(field,[yy,xx],order=0,mode="nearest")[pupil])
                    cnew = np.asarray(cnew)
                    candidate = np.angle(cnew[0]) if kind=="phase" else model.pack(cnew)
                    # Keep the nested-model candidate and add the old solution.
                    starts = [candidate] if kind=="phase" and args.starts==1 else [starts[0],candidate]+starts[2:]
                theta,loss,runs = solve(model,starts,args.iterations,prefix+"/"+name)
                if loss>prev_loss+1e-10*max(1.,prev_loss):
                    raise AssertionError("Nested model loss increased")
                prev,prev_loss = model.unpack(theta).copy(),loss
                training,images,screens = model.evaluate(theta,waves,True)
                scan_rows,_,_ = model.evaluate(theta,scan)
                reconstructed_loss = objective_from_metrics(
                    training, p.loss_profile, p.roi_floor, p.roi_weight,
                    p.cvar_weight, p.cvar_tail_fraction)
                if not np.isclose(loss,reconstructed_loss,rtol=1e-9,atol=1e-12):
                    raise AssertionError("Saved output does not reproduce objective")
                if max(r["captured_power"] for r in scan_rows)>1.005:
                    raise AssertionError("Power quadrature exceeds unity: refine detector sampling")
                records.append({"name":name,"degree":degree,"loss":loss,"parameters":int(theta.size),
                    "basis_rank":int(np.linalg.matrix_rank(model.basis)),"training":training,"scan":scan_rows,"runs":runs})
                arrays[name+"_coefficients"] = prev
                arrays[name+"_theta"] = theta
                arrays[name+"_images"] = images.astype(np.float32)
                arrays[name+"_screens"] = screens.astype(np.complex64)
                save_json(dest/"results.json",{"metadata":metadata,"configuration":vars(args),"models":records})
                np.savez_compressed(dest/"designs.npz",**arrays)
            with (dest/"metrics.csv").open("w",newline="",encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream,fieldnames=["model","split"]+list(records[0]["training"][0]))
                writer.writeheader()
                for r in records:
                    for split in ["training","scan"]:
                        writer.writerows({"model":r["name"],"split":split,**row} for row in r[split])
            report(dest,metadata,records,waves,p)
            plots(dest,target,X,waves,records,arrays)
            summary.append({"target":prefix,"models":[{"name":r["name"],
                "training_RMS_L2":float(np.sqrt(np.mean([x["relative_L2"]**2 for x in r["training"]])))}
                for r in records]})
            save_json(out/"summary.json",summary)
    print(f"Completed: {out.resolve()}",flush=True)


if __name__ == "__main__":
    main()
