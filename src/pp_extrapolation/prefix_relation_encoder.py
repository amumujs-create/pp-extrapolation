"""Experimental causal prefix -> coherent function coefficient distribution.

Covariance is an empirical leave-unit-out residual approximation, not calibrated
predictive uncertainty. No query labels or future observations are consumed.
"""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .function_prior_sharing import DESCRIPTOR,integral
from .relation_local_transport import group_weights


def prefix_features(rows):
    x=np.asarray(rows['x'],float);g=np.asarray(rows['groups']);t=np.asarray(rows['cycles'])
    out=np.empty((len(x),15))
    for u in np.unique(g):
        ix=np.flatnonzero(g==u);ix=ix[np.argsort(t[ix],kind='stable')]
        if len(np.unique(t[ix]))!=len(ix):raise ValueError('unique cycle per unit required')
        z=x[ix][:,DESCRIPTOR]
        out[ix]=np.column_stack((z,np.cumsum(z,axis=0)/np.arange(1,len(ix)+1)[:,None],
                                np.log1p(np.arange(1,len(ix)+1))))
    return out


def ridge_fit(x,y,g):
    w=group_weights(g);center=w@x;scale=np.sqrt(w@(x-center)**2);scale[scale<1e-8]=1
    a=np.column_stack((np.ones(len(x)),(x-center)/scale))
    penalty=np.eye(a.shape[1])*.1;penalty[0,0]=1e-8
    coef=np.linalg.solve(a.T@(w[:,None]*a)+penalty,a.T@(w[:,None]*y))
    return center,scale,coef


class Encoder(nn.Module):
    def __init__(self,center,scale,coef,cov,seed):
        super().__init__();torch.manual_seed(seed)
        for name,value in dict(center=center,scale=scale,coef=coef,chol=np.linalg.cholesky(cov)).items():
            self.register_buffer(name,torch.tensor(value,dtype=torch.float64))
        self.residual=nn.Sequential(nn.Linear(len(center),16),nn.Tanh(),nn.Linear(16,3)).double()
        nn.init.zeros_(self.residual[-1].weight);nn.init.zeros_(self.residual[-1].bias)

    def mean(self,z):
        z=(z-self.center)/self.scale
        a=torch.cat((torch.ones((len(z),1),dtype=z.dtype),z),1)
        return a@self.coef+.5*torch.tanh(self.residual(z))

    def draws(self,z,m,noise):
        theta=self.mean(z)[:,None]+torch.einsum('ij,sj->si',self.chol,noise)[None]
        return integral(theta,m)


@dataclass
class PrefixFit:
    model: Encoder
    audit: dict


def fit_prefix(rows,bank,*,neural=True,consistency=True,seed=42,steps=160):
    z=prefix_features(rows);lookup={str(u):i for i,u in enumerate(bank.groups)}
    labels=np.array([bank.means[lookup[str(u)]] for u in rows['groups']])
    center,scale,coef=ridge_fit(z,labels,rows['groups'])
    residuals=[]
    for u in np.unique(rows['groups']):
        held=rows['groups']==u
        c,s,b=ridge_fit(z[~held],labels[~held],rows['groups'][~held])
        p=np.column_stack((np.ones(held.sum()),(z[held]-c)/s))@b
        residuals.append(np.mean(labels[held]-p,axis=0))
    e=np.array(residuals);cov=e.T@e/len(e)+np.diag([.01,.01,.01])
    # Conservative shrinkage retains covariance while limiting small-unit estimates.
    cov=.5*cov+.5*np.diag(np.diag(cov))
    model=Encoder(center,scale,coef,cov,seed)
    tz=torch.tensor(z);m=torch.tensor(rows['x'][:,0],dtype=torch.float64)
    y=torch.tensor(rows['y'],dtype=torch.float64);w=torch.tensor(group_weights(rows['groups']))
    left=[];right=[]
    for u in np.unique(rows['groups']):
        ix=np.flatnonzero(rows['groups']==u);ix=ix[np.argsort(rows['cycles'][ix])]
        left.extend(ix[:-1]);right.extend(ix[1:])
    rng=np.random.default_rng(seed);opt=torch.optim.Adam(model.residual.parameters(),lr=.01)
    history=[]
    for step in range(steps if neural else 0):
        # Random rows sampled with unit-balanced probabilities. Targets remain source-only.
        ix=rng.choice(len(z),size=min(128,len(z)),replace=True,p=w.numpy())
        noise=torch.tensor(rng.normal(size=(16,3)),dtype=torch.float64)
        a=model.draws(tz[ix],m[ix],noise[:8]);b=model.draws(tz[ix],m[ix],noise[8:])
        energy=(a-y[ix,None]).abs().mean()-.5*(a-b).abs().mean()
        theta=model.mean(tz)
        smooth=(theta[left]-theta[right]).square().mean() if left else theta.sum()*0
        loss=energy+(.1*smooth if consistency else 0)+.001*sum(p.square().mean() for p in model.residual.parameters())
        if not torch.isfinite(loss):raise RuntimeError('nonfinite prefix loss')
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if step%40==0:history.append(float(loss.detach()))
    return PrefixFit(model,dict(seed=seed,neural=neural,consistency=consistency,steps=steps if neural else 0,
        source_units=len(bank.groups),residual_covariance=cov.tolist(),history=history,
        covariance_calibrated=False,held_query_labels_used=False))


def predict_prefix(fit,rows,*,samples=128,seed=1729):
    z=prefix_features(rows);noise=torch.tensor(np.random.default_rng(seed).normal(size=(samples,3)),dtype=torch.float64)
    with torch.no_grad():
        chunks=[fit.model.draws(torch.tensor(z[i:i+64]),torch.tensor(rows['x'][i:i+64,0],dtype=torch.float64),noise).numpy()
                for i in range(0,len(z),64)]
    draws=np.concatenate(chunks)
    return dict(mean=draws.mean(1),q05=np.quantile(draws,.05,axis=1),q95=np.quantile(draws,.95,axis=1))
