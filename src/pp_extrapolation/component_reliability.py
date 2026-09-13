"""Experimental transfer-trained coefficient reliability; no identifiability guarantee."""
import numpy as np
import torch
from torch import nn
from .prefix_relation_encoder import prefix_features
from .function_prior_sharing import integral


def prepare(base,bank,source,query):
    z=prefix_features(query);m=base.model
    with torch.no_grad():
        mean=m.mean(torch.tensor(z)).detach()
    cov=(m.chol@m.chol.T).detach()
    prior_mean=torch.tensor(bank.means.mean(0))
    centered=bank.means-bank.means.mean(0)
    prior_cov=torch.tensor(bank.covariances.mean(0)+centered.T@centered/len(centered))
    span=max(float(np.ptp(source['x'][:,0])),1e-6)
    distance=(source['x'][:,0].min()-query['x'][:,0])/span
    age=z[:,-1]/max(float(np.log1p(len(source['y'])/len(np.unique(source['groups'])))),1.)
    uncertainty=torch.sqrt(torch.diag(cov)/torch.diag(prior_cov).clamp_min(1e-8)).numpy()
    features=np.empty((len(z),3,4));features[:,:,0]=1
    features[:,:,1]=age[:,None];features[:,:,2]=distance[:,None];features[:,:,3]=np.log1p(uncertainty)[None]
    return dict(mean=mean,cov=cov,prior_mean=prior_mean,prior_cov=prior_cov,
        features=torch.tensor(features),margin=torch.tensor(query['x'][:,0],dtype=torch.float64))


class Reliability(nn.Module):
    def __init__(self,mode='component'):
        super().__init__()
        if mode not in ('ridge','fixed','tied','component'):raise ValueError('unknown reliability mode')
        self.mode=mode
        self.weight=nn.Parameter(torch.zeros((1 if mode=='tied' else 3,4),dtype=torch.float64))
        with torch.no_grad():self.weight[:,0]=2.

    def distribution(self,p):
        if self.mode=='ridge':w=torch.ones_like(p['mean'])
        elif self.mode=='fixed':w=torch.full_like(p['mean'],.5)
        else:
            w=(p['features']*self.weight[None]).sum(-1).sigmoid()
            # Tied means exactly one shared gate, including shared uncertainty summary.
            if self.mode=='tied':w=w.mean(1,keepdim=True).expand(-1,3)
        mean=w*p['mean']+(1-w)*p['prior_mean']
        cov=w[:,:,None]*p['cov']*w[:,None,:]+(1-w)[:,:,None]*p['prior_cov']*(1-w)[:,None,:]
        disagreement=w*(1-w)*(p['mean']-p['prior_mean']).square()
        cov=cov+torch.diag_embed(disagreement)
        return mean,cov,w

    def draws(self,p,noise):
        mean,cov,_=self.distribution(p)
        chol=torch.linalg.cholesky(cov+1e-10*torch.eye(3,dtype=torch.float64))
        theta=mean[:,None]+torch.einsum('bij,sj->bsi',chol,noise)
        return integral(theta,p['margin'])


def fit_reliability(episodes,mode,seed=42,steps=120):
    model=Reliability(mode);rng=np.random.default_rng(seed)
    if mode in ('ridge','fixed'):return model
    opt=torch.optim.Adam(model.parameters(),lr=.02)
    for step in range(steps):
        e=episodes[int(rng.integers(len(episodes)))];groups=e['groups']
        unit=rng.choice(np.unique(groups));ix=np.flatnonzero(groups==unit)
        ix=rng.choice(ix,min(48,len(ix)),replace=False)
        p={k:(v[ix] if k in ('mean','features','margin') else v) for k,v in e['prepared'].items()}
        noise=torch.tensor(rng.normal(size=(24,3)),dtype=torch.float64)
        a=model.draws(p,noise[:12]);b=model.draws(p,noise[12:]);y=torch.tensor(e['y'][ix],dtype=torch.float64)
        loss=(a-y[:,None]).abs().mean()-.5*(a-b).abs().mean()+.001*model.weight[:,1:].square().mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite reliability loss')
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
    return model


def predict_reliability(model,p,samples=256):
    noise=torch.tensor(np.random.default_rng(1729).normal(size=(samples,3)),dtype=torch.float64)
    with torch.no_grad():
        draws=model.draws(p,noise).numpy();w=model.distribution(p)[2].numpy()
    return draws,w
