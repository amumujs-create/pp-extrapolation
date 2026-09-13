"""Nested-source extrapolation error model; no distribution-free guarantee."""
from dataclasses import dataclass
import numpy as np
from pp_extrapolation.relation_local_transport import group_weights


def distance_features(x,boundary,span):
    d=(boundary-np.asarray(x)[:,0])/max(span,1e-6)
    return np.column_stack((np.ones(len(d)),d))


@dataclass
class TransferError:
    coef: np.ndarray
    residual: np.ndarray
    weight: np.ndarray
    audit: dict


def fit_error(features,log_error,groups):
    features=np.asarray(features,float);log_error=np.asarray(log_error,float)
    if not np.isfinite(features).all() or not np.isfinite(log_error).all():raise ValueError('finite errors required')
    w=group_weights(groups);penalty=np.diag([1e-6,.1])
    coef=np.linalg.solve(features.T@(w[:,None]*features)+penalty,features.T@(w*log_error))
    residual=np.empty(len(log_error))
    for u in np.unique(groups):
        held=groups==u;v=group_weights(groups[~held]);a=features[~held]
        c=np.linalg.solve(a.T@(v[:,None]*a)+penalty,a.T@(v*log_error[~held]))
        residual[held]=log_error[held]-features[held]@c
    return TransferError(coef,residual,w,dict(units=len(np.unique(groups)),rows=len(w),guarantee=False))


def factors(fit,features,mode='conditional',samples=256):
    if mode not in ('conditional','global','spread_only'):raise ValueError('unknown mode')
    # Deterministic equal-mass quadrature under the unit-balanced empirical law.
    values=fit.residual
    order=np.argsort(values);cdf=np.cumsum(fit.weight[order]);cdf[-1]=1
    ix=order[np.searchsorted(cdf,(np.arange(samples)+.5)/samples)]
    shifts=np.clip(values[ix],-np.log(4),np.log(4))
    location=np.clip(features@fit.coef,-np.log(4),np.log(4)) if mode=='conditional' else np.zeros(len(features))
    f=np.exp(location[:,None]+shifts[None])
    if mode=='spread_only':f=f/f.mean(1,keepdims=True)
    return f
