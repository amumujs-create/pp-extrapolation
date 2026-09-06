"""Distance and epistemic-uncertainty conditioned PP residual shrinkage."""
from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np
from .model import PPFit
from .support_gate import predict_components

DEFAULT_GAMMAS=(0.0,1.0,2.0,4.0,8.0,16.0,32.0,64.0)

@dataclass(frozen=True)
class UncertaintyGateSelection:
    beta: float
    gamma: float
    validation_mse: float
    validation_ensemble_mse: float
    candidates: tuple[dict,...]

def component_ensemble(fits: Sequence[PPFit], x: np.ndarray):
    if len(fits)<2: raise ValueError("at least two independently seeded fits required")
    components=[predict_components(fit,x) for fit in fits]
    affine=np.asarray([a for a,_ in components]); residual=np.asarray([r for _,r in components])
    cap=float(min(fit.target_scale for fit in fits))
    uncertainty=np.std(residual,axis=0)/max(cap,1e-12)
    return affine,residual,uncertainty,cap

def combine_uncertainty_gated(affine,residual,distance,uncertainty,*,beta,gamma,output_cap):
    affine=np.asarray(affine,dtype=float); residual=np.asarray(residual,dtype=float)
    distance=np.asarray(distance,dtype=float); uncertainty=np.asarray(uncertainty,dtype=float)
    if affine.shape!=residual.shape or affine.ndim!=2 or affine.shape[1:]!=distance.shape or distance.shape!=uncertainty.shape:
        raise ValueError("components=(seeds,n), distance/uncertainty=(n,) required")
    if min(beta,gamma)<0 or not np.isfinite([beta,gamma,output_cap]).all(): raise ValueError("invalid gate parameters")
    trust=np.exp(-float(beta)*np.maximum(distance,0)-float(gamma)*np.maximum(uncertainty,0))
    return np.clip(affine+trust[None,:]*residual,0,float(output_cap))

def select_uncertainty_gate(fits,validation_x,validation_y,validation_distance,*,
    betas:Iterable[float]=(0.,.05,.1,.25,.5,1.,2.,4.,8.),gammas:Iterable[float]=DEFAULT_GAMMAS):
    affine,residual,u,cap=component_ensemble(fits,validation_x); truth=np.asarray(validation_y,dtype=float)
    rows=[]
    for beta in betas:
        for gamma in gammas:
            seeded=combine_uncertainty_gated(affine,residual,validation_distance,u,beta=float(beta),gamma=float(gamma),output_cap=cap)
            rows.append({'beta':float(beta),'gamma':float(gamma),
              'validation_mse':float(np.mean((seeded-truth[None,:])**2)),
              'validation_ensemble_mse':float(np.mean((seeded.mean(0)-truth)**2))})
    best=min(rows,key=lambda r:(r['validation_mse'],r['beta'],r['gamma']))
    return UncertaintyGateSelection(best['beta'],best['gamma'],best['validation_mse'],best['validation_ensemble_mse'],tuple(rows))

def predict_uncertainty_gated(fits,x,distance,selection):
    affine,residual,u,cap=component_ensemble(fits,x)
    return combine_uncertainty_gated(affine,residual,distance,u,beta=selection.beta,gamma=selection.gamma,output_cap=cap)
