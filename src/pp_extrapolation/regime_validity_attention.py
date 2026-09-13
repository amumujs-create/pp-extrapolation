"""Minimal RAVEN-X: window validity, competing regime hazards and path decoder."""
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .function_prior_sharing import DESCRIPTOR
from .piecewise_health_path import one_switch_rul,no_switch_rul
from .relation_intervention_generator import RelationInterventionGenerator

WINDOW_FRACTIONS=(.125,.25,.5,1.)


def sparsemax(x,dim=-1):
    z=x-x.max(dim=dim,keepdim=True).values
    zs=torch.sort(z,dim=dim,descending=True).values
    k=torch.arange(1,z.shape[dim]+1,dtype=z.dtype,device=z.device)
    shape=[1]*z.ndim;shape[dim]=-1;k=k.reshape(shape)
    cumulative=zs.cumsum(dim)
    support=(1+k*zs>cumulative)
    count=support.sum(dim=dim,keepdim=True).clamp_min(1)
    tau=(cumulative.gather(dim,count-1)-1)/count
    return (z-tau).clamp_min(0)


def _window_rows(context,group,cycle):
    ix=np.flatnonzero((context['groups']==group)&(context['cycles']<=cycle))
    return ix[np.argsort(context['cycles'][ix],kind='stable')]


def prepare_windows(base,bank,source,context,query):
    """Causal observed X history is allowed; query RUL is never read."""
    zall=[];theta=[];valid=[]
    center=base.model.center.detach().numpy();scale=base.model.scale.detach().numpy()
    for group,cycle in zip(query['groups'],query['cycles']):
        ix=_window_rows(context,group,cycle)
        if len(ix)==0:raise ValueError('query row absent from context')
        row_z=[];row_valid=[]
        current=context['x'][ix[-1],DESCRIPTOR]
        for fraction in WINDOW_FRACTIONS:
            n=max(1,int(np.ceil(len(ix)*fraction)));take=ix[-n:]
            value=np.r_[current,context['x'][take][:,DESCRIPTOR].mean(0),np.log1p(n)]
            row_z.append((value-center)/scale);row_valid.append(n>=4 or fraction==1.)
        zall.append(row_z);valid.append(row_valid)
    z=torch.tensor(np.asarray(zall),dtype=torch.float64);valid=torch.tensor(valid,dtype=torch.bool)
    with torch.no_grad():
        flat=z.reshape(-1,15)*base.model.scale+base.model.center
        th=base.model.mean(flat).reshape(len(z),4,3)
    population=np.asarray(bank.means);centered=population-population.mean(0)
    pcov=bank.covariances.mean(0)+centered.T@centered/max(len(centered),1)+np.eye(3)*1e-6
    span=max(float(np.ptp(source['x'][:,0])),1e-6)
    distance=np.maximum(0,(source['x'][:,0].min()-query['x'][:,0])/span)
    base_cov=base.model.chol@base.model.chol.T
    return PreparedWindows(z,th,valid,torch.tensor(query['x'][:,0],dtype=torch.float64),
        torch.tensor(distance,dtype=torch.float64),base_cov.detach().clone(),torch.tensor(pcov),torch.tensor(population.mean(0)))


@dataclass
class PreparedWindows:
    z: torch.Tensor
    theta: torch.Tensor
    valid: torch.Tensor
    margin: torch.Tensor
    distance: torch.Tensor
    base_cov: torch.Tensor
    population_cov: torch.Tensor
    population_mean: torch.Tensor


class RegimeValidityAttention(nn.Module):
    def __init__(self,attention='sparse',validity=True,generator='sparse',point_transition=False,coherent=True,transitions=True):
        super().__init__()
        if attention not in ('sparse','softmax'):raise ValueError('unknown attention')
        self.attention,self.validity,self.point_transition,self.coherent,self.transitions=attention,validity,point_transition,coherent,transitions
        self.encoder=nn.Sequential(nn.Linear(15,16),nn.Tanh()).double()
        self.query=nn.Parameter(torch.randn((3,16),dtype=torch.float64)*.1)
        self.route=nn.Linear(16,3).double()
        self.bin_bias=nn.Parameter(torch.zeros((8,3),dtype=torch.float64))
        self.distance_effect=nn.Parameter(torch.zeros((8,2),dtype=torch.float64))
        self.generator=RelationInterventionGenerator(15,generator)

    def window_attention(self,p):
        h=self.encoder(p.z);score=torch.einsum('bwh,kh->bkw',h,self.query)/4
        score=score.masked_fill(~p.valid[:,None],-1e9)
        weights=sparsemax(score,-1) if self.attention=='sparse' else score.softmax(-1)
        return h,weights

    def path_probabilities(self,context,distance):
        """First-event competing-risk probabilities: no switch + 2 types x 8 bins."""
        if not self.transitions:
            answer=torch.zeros((len(context),17),dtype=context.dtype);answer[:,0]=1
            return answer
        base=self.route(context)[:,1:]
        logits=base[:,None,:]+self.bin_bias[None,:,1:]
        if self.validity:logits=logits+self.distance_effect[None]*distance[:,None,None]
        if self.point_transition:
            mask=torch.full_like(logits,-30);mask[:,3]=logits[:,3];logits=mask
        survival=torch.ones(len(context),dtype=context.dtype);events=[]
        for b in range(8):
            q=torch.cat((torch.zeros((len(context),1),dtype=context.dtype),logits[:,b]),1).softmax(1)
            events.append(survival[:,None]*q[:,1:]);survival=survival*q[:,0]
        event=torch.stack(events,1)
        return torch.cat((survival[:,None],event.reshape(len(context),-1)),1)

    def forward(self,p,noise):
        h,a=self.window_attention(p)
        theta_path=torch.einsum('bkw,bwj->bkj',a,p.theta)
        relation=torch.einsum('bkw,bwf->bkf',a[:,1:],p.z)
        context=torch.einsum('bkw,bwh->bkh',a,h).mean(1)
        probabilities=self.path_probabilities(context,p.distance)
        # Before-regime covariance retained for every path.
        chol=torch.linalg.cholesky(p.base_cov)
        theta0=theta_path[:,:,None,:]+torch.einsum('ij,sj->si',chol,noise)[None,None]
        persist=no_switch_rul(theta0[:,0],p.margin)[:,None,:]
        theta1,gate=self.generator.draw(theta_path[:,1:],relation,noise)
        event_paths=[]
        for b in range(8):
            switch=p.margin*(1-(b+.5)/8)
            for k in range(2):
                if self.coherent:
                    value=one_switch_rul(theta0[:,k+1],theta1[:,k],p.margin,switch)
                else:
                    # Control: ignores pre-switch travel and predicts each horizon independently.
                    value=no_switch_rul(theta1[:,k],p.margin)
                event_paths.append(value[:,None,:])
        draws=torch.cat([persist]+event_paths,1)
        return draws,probabilities,a,gate


def mixture_energy(draws,probabilities,y):
    """Deterministic weighted sample energy score."""
    b,c,s=draws.shape;flat=draws.reshape(b,c*s)
    weight=(probabilities[:,:,None]/s).expand(-1,-1,s).reshape(b,c*s)
    first=(weight*(flat-y[:,None]).abs()).sum(1)
    second=.5*torch.einsum('bi,bij,bj->b',weight,(flat[:,:,None]-flat[:,None,:]).abs(),weight)
    return first-second


def predictive_summary(model,p,samples=1024,seed=1729,y=None,return_support=False):
    u=torch.quasirandom.SobolEngine(3,scramble=True,seed=seed).draw(samples).double().clamp(1e-9,1-1e-9)
    noise=torch.erfinv(2*u-1)*np.sqrt(2);model.eval()
    means=[];lo=[];hi=[];crps=[];probabilities=[];attentions=[];supports=[];support_weights=[];gate=None
    with torch.no_grad():
        for start in range(0,len(p.margin),16):
            batch=subset_prepared(p,np.arange(start,min(start+16,len(p.margin))))
            draws,prob,a,gate=model(batch,noise)
            b,c,s=draws.shape;flat=draws.reshape(b,c*s).numpy()
            weight=(prob[:,:,None]/s).expand(-1,-1,s).reshape(b,c*s).numpy()
            means.extend((flat*weight).sum(1));probabilities.append(prob.numpy());attentions.append(a.numpy())
            if return_support:supports.append(flat);support_weights.append(weight)
            for row,(values,w) in enumerate(zip(flat,weight)):
                order=np.argsort(values);v=values[order];sw=w[order];cw=np.cumsum(sw);cw[-1]=1
                lo.append(v[np.searchsorted(cw,.05)]);hi.append(v[np.searchsorted(cw,.95)])
                if y is not None:
                    first=np.sum(sw*np.abs(v-np.asarray(y)[start+row]))
                    half_pair=np.sum(sw*v*(2*(cw-sw)+sw-1))
                    crps.append(first-half_pair)
    answer=dict(mean=np.asarray(means),q05=np.asarray(lo),q95=np.asarray(hi),
        path_probability=np.concatenate(probabilities),attention=np.concatenate(attentions),edge_gate=gate.numpy())
    if y is not None:answer['crps']=np.asarray(crps)
    if return_support:
        answer['support']=np.concatenate(supports);answer['support_weight']=np.concatenate(support_weights)
    return answer


def subset_prepared(p,ix):
    return PreparedWindows(p.z[ix],p.theta[ix],p.valid[ix],p.margin[ix],p.distance[ix],p.base_cov,p.population_cov,p.population_mean)


def fit_raven(episodes,config,seed=42,steps=80,counterfactual=True):
    model=RegimeValidityAttention(**config);rng=np.random.default_rng(seed)
    opt=torch.optim.Adam(model.parameters(),lr=.015)
    history=[]
    for step in range(steps):
        e=episodes[int(rng.integers(len(episodes)))];g=e['groups']
        unit=rng.choice(np.unique(g));available=np.flatnonzero(g==unit)
        ix=rng.choice(available,min(12,len(available)),replace=False)
        p=subset_prepared(e['prepared'],ix);y=torch.tensor(e['y'][ix],dtype=torch.float64)
        noise=torch.tensor(rng.normal(size=(8,3)),dtype=torch.float64)
        draws,prob,a,gate=model(p,noise);risk=mixture_energy(draws,prob,y)
        loss=risk.mean()+.002*gate.mean()
        if counterfactual and step%4==0:
            # Outer query is untouched: importance supervision uses inner suffix labels only.
            importance=[]
            for w in range(4):
                masked=PreparedWindows(p.z,p.theta,p.valid.clone(),p.margin,p.distance,p.base_cov,p.population_cov,p.population_mean)
                can=masked.valid[:,w] & (masked.valid.sum(1)>1);masked.valid[can,w]=False
                cd,cp,_,_=model(masked,noise)
                importance.append((mixture_energy(cd,cp,y)-risk).detach())
            target=torch.stack(importance,1).softmax(1)
            observed=a.mean(1).clamp_min(1e-10)
            loss=loss-.05*(target*observed.log()).sum(1).mean()
        if not torch.isfinite(loss):raise RuntimeError('nonfinite RAVEN objective')
        opt.zero_grad();loss.backward();nn.utils.clip_grad_norm_(model.parameters(),2);opt.step()
        if step%20==0:history.append(float(loss.detach()))
    return model,dict(seed=seed,steps=steps,counterfactual=counterfactual,history=history,
        parameters=sum(v.numel() for v in model.parameters()))
