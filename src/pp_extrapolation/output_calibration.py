"""Group-robust validation-only calibration for extrapolation predictions."""
from __future__ import annotations
from dataclasses import dataclass
from math import comb
import numpy as np

@dataclass(frozen=True)
class OutputCalibrator:
    kind:str;slope:float=1.;intercept:float=0.
    def predict(self,prediction,lower=0.,upper=None):
        value=self.slope*np.asarray(prediction,dtype=np.float64)+self.intercept
        return np.clip(value,lower,upper)

def fit_output_calibrator(kind,target,prediction,groups=None):
    y=np.asarray(target,dtype=np.float64);p=np.asarray(prediction,dtype=np.float64)
    if kind=='identity':return OutputCalibrator(kind)
    if kind=='bias':return OutputCalibrator(kind,1.,float(np.mean(y-p)))
    if kind=='group_bias':
        if groups is None:raise ValueError('groups are required for group_bias')
        g=np.asarray(groups);offset=np.mean([np.mean(y[g==u]-p[g==u]) for u in np.unique(g)])
        return OutputCalibrator(kind,1.,float(offset))
    if kind=='affine':
        slope,intercept=np.polyfit(p,y,1)
        return OutputCalibrator(kind,float(np.clip(slope,.5,1.5)),float(intercept))
    raise ValueError(f'unknown calibrator: {kind}')

def select_group_loo_calibrator(target,prediction,groups,*,upper,candidates=('identity','bias','group_bias','affine')):
    """Select a calibrator using leave-one-validation-group-out error."""
    y=np.asarray(target,dtype=np.float64);p=np.asarray(prediction,dtype=np.float64);g=np.asarray(groups);scores={}
    for kind in candidates:
        errors=[]
        for held in np.unique(g):
            keep=g!=held;cal=fit_output_calibrator(kind,y[keep],p[keep],g[keep])
            errors.extend((cal.predict(p[~keep],upper=upper)-y[~keep])**2)
        scores[kind]=float(np.mean(errors))
    # ``min`` is stable, so the declared candidate order is the complexity
    # preference when validation scores tie (identity is the default).
    chosen=min(candidates,key=lambda kind:scores[kind])
    return fit_output_calibrator(chosen,y,p,g),scores

def consensus_tail_probability(votes,total,*,null_probability=.5):
    """Exact upper-tail probability for repeated seed-level selections."""
    if not (0 <= votes <= total) or total < 1:
        raise ValueError('require 0 <= votes <= total and total >= 1')
    if not (0 < null_probability < 1):
        raise ValueError('null_probability must lie strictly between zero and one')
    return float(sum(comb(total,k)*null_probability**k*(1-null_probability)**(total-k)
                     for k in range(votes,total+1)))

def group_loo_affine_evidence(target,prediction,groups,*,upper,alpha=.05,
                              bootstrap_draws=20000,bootstrap_seed=271828):
    """Test whether affine transport improves held-out-group mean squared error."""
    y=np.asarray(target,dtype=np.float64);p=np.asarray(prediction,dtype=np.float64);g=np.asarray(groups)
    rows=[]
    for held in np.unique(g):
        keep=g!=held
        cal=fit_output_calibrator('affine',y[keep],p[keep],g[keep])
        identity_mse=float(np.mean((p[~keep]-y[~keep])**2))
        affine_mse=float(np.mean((cal.predict(p[~keep],upper=upper)-y[~keep])**2))
        rows.append({'group':str(held),'identity_mse':identity_mse,'affine_mse':affine_mse,
                     'mse_gain':identity_mse-affine_mse,
                     'relative_gain':(identity_mse-affine_mse)/max(identity_mse,1e-12)})
    wins=sum(row['affine_mse']<row['identity_mse'] for row in rows)
    probability=consensus_tail_probability(wins,len(rows))
    gains=np.asarray([row['mse_gain'] for row in rows])
    rng=np.random.default_rng(int(bootstrap_seed))
    means=gains[rng.integers(0,len(gains),size=(int(bootstrap_draws),len(gains)))].mean(axis=1)
    interval=np.quantile(means,[alpha/2,1-alpha/2])
    return {'groups':len(rows),'wins':wins,'sign_test_p':probability,
            'mean_mse_gain':float(gains.mean()),
            'bootstrap_ci':[float(interval[0]),float(interval[1])],
            'median_relative_gain':float(np.median([row['relative_gain'] for row in rows])),
            'approved':bool(interval[0]>0),'per_group':rows}

def approve_dual_evidence(calibrators,group_evidence,required_kind='affine',*,alpha=.05):
    """Require both optimization-seed and held-out-group evidence."""
    return (approve_seed_consensus(calibrators,required_kind,alpha=alpha)
            and bool(group_evidence.get('approved',False)))

def transport_direction_cosine(train_coordinates,validation_coordinates,test_coordinates):
    """Cosine between validation and test shift rays from the train centroid."""
    train=np.asarray(train_coordinates,dtype=np.float64)
    validation=np.asarray(validation_coordinates,dtype=np.float64)
    test=np.asarray(test_coordinates,dtype=np.float64)
    if train.ndim!=2 or validation.ndim!=2 or test.ndim!=2 or not (train.shape[1]==validation.shape[1]==test.shape[1]):
        raise ValueError('coordinate arrays must have shape (n,d) with a shared d')
    center=train.mean(axis=0);vray=validation.mean(axis=0)-center;tray=test.mean(axis=0)-center
    denominator=np.linalg.norm(vray)*np.linalg.norm(tray)
    return float(vray@tray/denominator) if denominator>0 else float('nan')

def approve_transport(calibrators,group_evidence,*,direction_cosine,alpha=.05,
                      minimum_direction_cosine=0.):
    """Require statistical replication and a compatible extrapolation ray."""
    return (np.isfinite(direction_cosine) and direction_cosine>minimum_direction_cosine
            and approve_dual_evidence(calibrators,group_evidence,alpha=alpha))

def approve_seed_consensus(calibrators,required_kind='affine',*,alpha=.05,null_probability=.5):
    """Approve when seed votes reject chance selection by an exact sign test.

    This is an optimization-stability check. Scientific inference still requires
    independent physical units or datasets.
    """
    values=list(calibrators)
    if not values or not (0 < alpha < 1):
        return False
    votes=sum(value.kind==required_kind for value in values)
    return consensus_tail_probability(votes,len(values),null_probability=null_probability)<=alpha
