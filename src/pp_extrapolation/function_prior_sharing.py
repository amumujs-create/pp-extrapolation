"""Experimental component-specific sharing of coherent positive function priors.

log slowness(h) = a+b*h+c*h², RUL(m)=integral_0^m exp(log slowness(h))dh.
Unit coefficient Gaussians are Laplace approximations, not calibrated posteriors.
Routing and proxy-rate conditioning are empirical approximations, not a proof
of physical identifiability, novelty, or out-of-support coverage.
"""
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .relation_local_transport import group_weights

DESCRIPTOR=(2,3,4,6,7,8,9)
NODES,WEIGHTS=np.polynomial.legendre.leggauss(16)
NODES=(NODES+1)/2;WEIGHTS=WEIGHTS/2


def basis(h):
    return torch.stack((torch.ones_like(h),h,h.square()),dim=-1)


def integral(coefficients,margin,target=None):
    """One coefficient draw defines an entire path, not independent row noise."""
    if coefficients.ndim!=3 or coefficients.shape[0]!=len(margin) or coefficients.shape[2]!=3:
        raise ValueError('coefficients must be query x sample x 3')
    target=torch.zeros_like(margin) if target is None else target
    if target.shape!=margin.shape or torch.any(target<0) or torch.any(target>margin):
        raise ValueError('require 0 <= target <= current margin')
    nodes=torch.as_tensor(NODES,dtype=margin.dtype,device=margin.device)
    weights=torch.as_tensor(WEIGHTS,dtype=margin.dtype,device=margin.device)
    h=target[:,None]+(margin-target)[:,None]*nodes
    logs=torch.einsum('bsk,bqk->bsq',coefficients,basis(h))
    return (margin-target)[:,None]*(logs.clamp(-8,8).exp()*weights).sum(-1)


@dataclass
class FunctionBank:
    means: np.ndarray
    covariances: np.ndarray
    descriptors: np.ndarray
    groups: np.ndarray
    center: np.ndarray
    scale: np.ndarray
    calibration: np.ndarray
    evidence_variance: float
    evidence_reliability: float
    audit: dict


def fit_bank(rows,*,steps=80):
    x,y,g,rate=[np.asarray(rows[k]) for k in ('x','y','groups','rate')]
    if x.ndim!=2 or x.shape[1]!=11 or y.shape!=(len(x),) or g.shape!=y.shape or rate.shape!=y.shape:
        raise ValueError('aligned 11-feature battery rows required')
    if any(not np.isfinite(v).all() for v in (x,y,rate)) or np.any(x[:,0]<=0) or np.any(y<=0) or np.any(rate<=0):
        raise ValueError('positive finite training margin, rate and RUL required')
    units,ix=np.unique(g.astype(str),return_inverse=True)
    if len(units)<2:raise ValueError('at least two source units required')
    w=group_weights(g)
    init=np.array([np.median(np.log(y[ix==i]/x[ix==i,0])) for i in range(len(units))])
    theta=nn.Parameter(torch.tensor(np.column_stack((init,np.zeros((len(units),2)))),dtype=torch.float64))
    prior=theta.detach().clone();precision=torch.tensor([1.,1/.75**2,1/.75**2],dtype=torch.float64)
    m=torch.tensor(x[:,0],dtype=torch.float64);ty=torch.tensor(y,dtype=torch.float64)
    rowix=torch.tensor(ix);tw=torch.tensor(w)
    opt=torch.optim.LBFGS([theta],lr=.5,max_iter=steps,line_search_fn='strong_wolfe')
    def closure():
        opt.zero_grad()
        p=integral(theta[rowix,None],m).squeeze(1).clamp_min(1e-10)
        data=(tw*(p.log()-ty.log()).square()).sum()/.25**2
        penalty=((theta-prior).square()*precision).sum(1).mean()/8
        loss=data+penalty
        if not torch.isfinite(loss):raise RuntimeError('nonfinite unit-prior objective')
        loss.backward();return loss
    opt.step(closure)
    means=theta.detach().numpy();cov=[]
    with torch.no_grad():
        nodes=torch.tensor(NODES,dtype=torch.float64);weights=torch.tensor(WEIGHTS,dtype=torch.float64)
        ph=basis(m[:,None]*nodes)
        logs=torch.einsum('bk,bqk->bq',theta[rowix],ph)
        terms=weights*logs.clamp(-8,8).exp()
        jac=(terms[:,:,None]*ph).sum(1)/terms.sum(1)[:,None]
        jac=jac.numpy()
        for i in range(len(units)):
            j=jac[ix==i];neff=min(max(len(j)/16,1.),8.)
            cov.append(np.linalg.inv(np.diag(precision.numpy())+neff*(j.T@j)/len(j)/.25**2))
    desc=x[:,DESCRIPTOR].astype(float)
    center=w@desc;scale=np.sqrt(w@(desc-center)**2);scale[scale<1e-8]=1.
    descriptors=np.array([desc[ix==i].mean(0) for i in range(len(units))])
    latent=np.sum(means[ix]*np.column_stack((np.ones(len(x)),x[:,0],x[:,0]**2)),1)
    proxy=-np.log(rate.astype(float))
    a=np.column_stack((np.ones(len(x)),proxy))
    coef=np.linalg.solve(a.T@(w[:,None]*a)+np.diag([1e-6,.01]),a.T@(w*latent))
    # A negative proxy relation is not treated as a valid degradation-rate observation.
    if coef[1]<0:coef=np.array([w@latent,0.])
    error=latent-a@coef;var=max(float(w@error**2),.5**2)
    total=max(float(w@(latent-w@latent)**2),1e-8)
    reliability=max(0.,min(1.,1-float(w@error**2)/total))
    return FunctionBank(means,np.asarray(cov),descriptors,units,center,scale,coef,var,reliability,
        dict(units=len(units),rows=len(y),proxy_reliability=reliability,proxy_variance=var,
             source_only=True,approximation='regularized unit log-RUL MAP + effective-sample Gauss-Newton covariance'))


def condition_gaussian(mean,cov,h,observation,variance,reliability):
    """Rank-one information update; no observations of the other two directions."""
    phi=basis(h);v=torch.einsum('bij,bj->bi',cov,phi)
    if reliability<=1e-8:return mean,cov
    denominator=(phi*v).sum(1)+variance/reliability
    new_mean=mean+v*((observation-(phi*mean).sum(1))/denominator)[:,None]
    new_cov=cov-v[:,:,None]*v[:,None,:]/denominator[:,None,None]
    return new_mean,(new_cov+new_cov.transpose(1,2))*.5


class SharingRule(nn.Module):
    def __init__(self,mode='component',condition=True,distribution=True):
        super().__init__()
        if mode not in ('component','tied','fixed','none'):raise ValueError('unknown sharing mode')
        self.mode,self.condition,self.distribution=mode,condition,distribution
        # Allocated parameter count is the same for learned tied/component arms.
        self.log_metric=nn.Parameter(torch.zeros(3,len(DESCRIPTOR),dtype=torch.float64))

    def coefficient_distribution(self,bank,x,rate):
        x=torch.as_tensor(x,dtype=torch.float64);rate=torch.as_tensor(rate,dtype=torch.float64)
        if x.ndim!=2 or x.shape[1]!=11 or rate.shape!=(len(x),) or torch.any(rate<=0) or torch.any(x[:,0]<0):
            raise ValueError('nonnegative query margin and positive rate required')
        if not torch.isfinite(x).all() or not torch.isfinite(rate).all():raise ValueError('finite query inputs required')
        mu=torch.tensor(bank.means);cov=torch.tensor(bank.covariances)
        desc=torch.tensor(bank.descriptors);scale=torch.tensor(bank.scale)
        delta=(x[:,DESCRIPTOR,None].transpose(1,2)-desc[None])/scale
        if self.mode=='none':
            # No individual-unit coefficient borrowing: generic population level and slope priors.
            mean=torch.zeros((len(x),3),dtype=torch.float64);mean[:,0]=mu[:,0].mean()
            sigma=torch.diag(torch.tensor([1.,.75**2,.75**2],dtype=torch.float64)).expand(len(x),-1,-1)
            weights=torch.zeros((len(x),3,len(mu)),dtype=torch.float64)
        else:
            if self.mode=='fixed':weights=torch.full((len(x),3,len(mu)),1/len(mu),dtype=torch.float64)
            else:
                metric=self.log_metric.clamp(-3,3)
                if self.mode=='tied':metric=metric.mean(0,keepdim=True).expand(3,-1)
                score=-.5*torch.einsum('buf,kf->bku',delta.square(),metric.exp())/len(DESCRIPTOR)
                weights=score.softmax(-1)
            mean=torch.einsum('bku,uk->bk',weights,mu)
            sigma=torch.einsum('biu,uij,bju->bij',weights,cov,weights)
            disagreement=torch.einsum('bku,buk->bk',weights,(mu[None]-mean[:,None]).square())
            sigma=sigma+torch.diag_embed(disagreement)
        if self.condition:
            obs=bank.calibration[0]+bank.calibration[1]*(-rate.log())
            mean,sigma=condition_gaussian(mean,sigma,x[:,0],obs,bank.evidence_variance,bank.evidence_reliability)
        return mean,sigma,weights

    def draws(self,bank,x,rate,noise,target=None):
        mean,cov,_=self.coefficient_distribution(bank,x,rate)
        if self.distribution:
            chol=torch.linalg.cholesky(cov+1e-8*torch.eye(3,dtype=torch.float64))
            coeff=mean[:,None]+torch.einsum('bij,sj->bsi',chol,noise)
        else:coeff=mean[:,None].expand(-1,len(noise),-1)
        m=torch.as_tensor(np.asarray(x)[:,0],dtype=torch.float64)
        t=None if target is None else torch.as_tensor(target,dtype=torch.float64)
        return integral(coeff,m,t)


@dataclass
class SharingFit:
    model: SharingRule
    bank: FunctionBank
    selection: dict


def predict_sharing(fit,rows,*,samples=256,seed=1729):
    if samples<2:raise ValueError('at least two distribution draws required')
    noise=torch.tensor(np.random.default_rng(seed).normal(size=(samples,3)),dtype=torch.float64)
    predictions=[];fit.model.eval()
    with torch.no_grad():
        for start in range(0,len(rows['x']),64):
            predictions.append(fit.model.draws(fit.bank,rows['x'][start:start+64],rows['rate'][start:start+64],noise).numpy())
    draws=np.concatenate(predictions)
    return dict(mean=draws.mean(1),median=np.median(draws,axis=1),samples=draws,
                q05=np.quantile(draws,.05,axis=1),q95=np.quantile(draws,.95,axis=1))


def fit_sharing(bank,episodes,validation,*,mode='component',condition=True,distribution=True,
                seed=42,max_epochs=150,patience=35):
    model=SharingRule(mode,condition,distribution);fit=SharingFit(model,bank,{})
    w=group_weights(validation['groups'])
    def risk():
        p=predict_sharing(fit,validation,samples=64)['mean']
        return float(w@(p-validation['y'])**2)
    best,best_epoch=risk(),0;state=copy.deepcopy(model.state_dict());history=[]
    opt=torch.optim.Adam(model.parameters(),lr=.025)
    rng=np.random.default_rng(seed)
    for epoch in range(1,max_epochs+1) if mode in ('component','tied') else ():
        losses=[]
        for ei in rng.choice(len(episodes),size=min(3,len(episodes)),replace=False):
            e=episodes[ei];query=e['query']
            # Uniform query-unit sampling, then rows, avoids long-unit dominance.
            unit=rng.choice(np.unique(query['groups']));ix=np.flatnonzero(query['groups']==unit)
            ix=rng.choice(ix,size=min(32,len(ix)),replace=False)
            noise=torch.tensor(rng.normal(size=(32,3)),dtype=torch.float64)
            a=model.draws(e['bank'],query['x'][ix],query['rate'][ix],noise[:16])
            b=model.draws(e['bank'],query['x'][ix],query['rate'][ix],noise[16:])
            y=torch.tensor(query['y'][ix],dtype=torch.float64)
            loss=(a-y[:,None]).abs().mean()-.5*(a-b).abs().mean()+.001*model.log_metric.square().mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite functional energy objective')
            opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
            losses.append(float(loss.detach()))
        if epoch%5==0:
            current=risk();history.append(dict(epoch=epoch,source_energy=float(np.mean(losses)),validation_unit_mse=current))
            if current<best-1e-8:best,best_epoch,state=current,epoch,copy.deepcopy(model.state_dict())
            if epoch-best_epoch>=patience:break
    model.load_state_dict(state)
    fit.selection=dict(seed=seed,mode=mode,condition=condition,distribution=distribution,
        selected_epoch=best_epoch,validation_unit_mse=best,history=history,parameters=sum(p.numel() for p in model.parameters()),
        metric=model.log_metric.detach().clamp(-3,3).exp().tolist())
    return fit
