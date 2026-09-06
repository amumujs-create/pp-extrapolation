"""Distance-shell calibration for label-free selective extrapolation at inference."""
from dataclasses import dataclass
from typing import Iterable
import numpy as np

@dataclass(frozen=True)
class ShellEvidence:
    lower: float
    upper: float
    count: int
    relative_mse_gain: float | None
    normalized_seed_disagreement: float | None
    accepted: bool
    reasons: tuple[str, ...]

@dataclass(frozen=True)
class ShellCalibration:
    edges: tuple[float, ...]
    shells: tuple[ShellEvidence, ...]
    minimum_count: int
    minimum_relative_gain: float
    maximum_seed_disagreement: float

def calibrate_distance_shells(
    distances: np.ndarray, targets: np.ndarray, candidate_predictions: np.ndarray,
    baseline_predictions: np.ndarray, *,
    edges: Iterable[float] = (0.0, 0.5, 1.0, 2.0, 4.0, np.inf),
    minimum_count: int = 20, minimum_relative_gain: float = 0.0,
    maximum_seed_disagreement: float = 0.25,
) -> ShellCalibration:
    """Calibrate PP-vs-baseline evidence using validation labels only."""
    d=np.asarray(distances,dtype=float); y=np.asarray(targets,dtype=float)
    pred=np.asarray(candidate_predictions,dtype=float); base=np.asarray(baseline_predictions,dtype=float)
    bounds=tuple(float(v) for v in edges)
    if pred.ndim!=2 or pred.shape[1]!=len(y) or d.shape!=y.shape or base.shape!=y.shape:
        raise ValueError("expected distances/targets/baseline=(n,), predictions=(seeds,n)")
    if len(bounds)<2 or bounds[0]!=0 or any(b<=a for a,b in zip(bounds,bounds[1:])):
        raise ValueError("edges must increase strictly from zero")
    if minimum_count<1 or minimum_relative_gain<0 or maximum_seed_disagreement<0:
        raise ValueError("invalid shell thresholds")
    if not np.isfinite(np.r_[d,y,pred.ravel(),base]).all() or np.any(d<0):
        raise ValueError("calibration arrays must be finite and distances nonnegative")
    ensemble=pred.mean(0); target_sd=max(float(np.std(y)),1e-12); shells=[]
    for lower,upper in zip(bounds,bounds[1:]):
        mask=(d>=lower)&(d<upper)
        count=int(mask.sum()); reasons=[]
        if count<minimum_count:
            gain=disagreement=None; reasons.append("insufficient_samples")
        else:
            baseline_mse=float(np.mean((base[mask]-y[mask])**2))
            candidate_mse=float(np.mean((ensemble[mask]-y[mask])**2))
            gain=(baseline_mse-candidate_mse)/max(baseline_mse,1e-12)
            disagreement=float(np.mean(np.std(pred[:,mask],axis=0))/target_sd)
            if gain<minimum_relative_gain: reasons.append("no_baseline_gain")
            if disagreement>maximum_seed_disagreement: reasons.append("seed_unstable")
        shells.append(ShellEvidence(lower,upper,count,gain,disagreement,not reasons,tuple(reasons)))
    return ShellCalibration(bounds,tuple(shells),minimum_count,float(minimum_relative_gain),float(maximum_seed_disagreement))

def apply_shell_calibration(calibration: ShellCalibration, source_distances: np.ndarray) -> np.ndarray:
    """Return a label-free per-observation predict mask for frozen shells."""
    d=np.asarray(source_distances,dtype=float)
    if d.ndim!=1 or not np.isfinite(d).all() or np.any(d<0):
        raise ValueError("source distances must be a finite nonnegative vector")
    accepted=np.zeros(len(d),dtype=bool)
    for shell in calibration.shells:
        accepted |= (d>=shell.lower)&(d<shell.upper)&shell.accepted
    return accepted
