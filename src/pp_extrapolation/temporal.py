"""Experimental domain-contract priors with causal history and neural correction.

The trend-disagreement signal is a diagnostic, not a calibrated regime probability.
History-only priors assume the current trend persists until a declared boundary.
"""
import copy
from dataclasses import dataclass
import numpy as np
import torch
from torch import nn
from .model import equal_group_weights


@dataclass(frozen=True)
class DegradationContract:
    boundary: float
    direction: str  # 'decreasing' capacity or 'increasing' crack/wear
    short_window: int = 8
    long_window: int = 24
    sequence_window: int = 16

    def coordinate(self, value):
        value = np.asarray(value, dtype=np.float64)
        if value.ndim != 1 or not len(value) or not np.isfinite(value).all():
            raise ValueError('finite nonempty 1D observations required')
        if self.direction not in ('decreasing', 'increasing'):
            raise ValueError('declare increasing or decreasing degradation')
        if not np.isfinite(self.boundary) or not 2 <= self.short_window < self.long_window or self.sequence_window < 1:
            raise ValueError('invalid boundary or window sizes')
        sign = 1. if self.direction == 'decreasing' else -1.
        margin = sign*(value[0]-self.boundary)
        if margin <= 0:
            raise ValueError('first observation must be on the healthy side of boundary')
        return sign*(value-self.boundary)/margin


def causal_history(time, observations, contract):
    """Build each row using its own prefix only; no labels accepted by this API."""
    q = contract.coordinate(observations)
    time = np.asarray(time, dtype=np.float64)
    if time.shape != q.shape or not np.isfinite(time).all() or np.any(np.diff(time) <= 0):
        raise ValueError('time must be finite and strictly increasing')
    n = len(q); rates = np.zeros((n,2)); errors = np.zeros((n,2))
    valid = np.zeros((n,2))
    for i in range(n):
        if i:
            # Errors are of predictions issued one observation earlier.
            innovation = np.abs(q[i]-(q[i-1]-rates[i-1]*(time[i]-time[i-1])))
            errors[i] = .8*errors[i-1]+.2*innovation
        for j,window in enumerate((contract.short_window,contract.long_window)):
            start=max(0,i-window+1)
            t=time[start:i+1]; v=q[start:i+1]
            if len(t)>1:
                centered=t-t.mean()
                rates[i,j]=-float(centered@(v-v.mean()))/float(centered@centered)
                valid[i,j]=float(rates[i,j]>1e-6 and len(t)>=4)
    disagreement=np.abs(rates[:,0]-rates[:,1])/(np.abs(rates).sum(1)+1e-6)
    # An uncalibrated online score favoring smaller observed forecast errors.
    logits=-errors/(errors.mean(1,keepdims=True)+1e-6)
    logits=logits-logits.max(1,keepdims=True)
    weights=np.exp(logits); weights/=weights.sum(1,keepdims=True)
    remaining=np.maximum(q,0)[:,None]/np.maximum(rates,1e-6)
    remaining=np.where(valid>0,remaining,1e9)
    features=np.column_stack((q,time-time[0],rates,errors,disagreement,valid))
    sequences=np.stack([features[np.maximum(np.arange(i-contract.sequence_window+1,i+1),0)] for i in range(n)])
    return dict(x=sequences.astype('float32'), prior=remaining.astype('float32'),
                reliability=weights.astype('float32'), q=q.astype('float32'),
                disagreement=disagreement.astype('float32'))


class TemporalPriorNet(nn.Module):
    def __init__(self, features, mode='corrected', width=16):
        super().__init__()
        if mode not in ('direct','corrected','fixed_gate'):
            raise ValueError('invalid temporal mode')
        self.mode=mode
        self.encoder=nn.GRU(features,width,batch_first=True)
        self.gate=nn.Linear(width,2)
        self.correction=nn.Linear(width,1)
        self.direct=nn.Linear(width,1)
        nn.init.zeros_(self.gate.weight); nn.init.zeros_(self.gate.bias)
        nn.init.zeros_(self.correction.weight); nn.init.zeros_(self.correction.bias)

    def forward(self,x,prior,reliability):
        _,h=self.encoder(x); h=h[-1]
        gate=reliability if self.mode=='fixed_gate' else torch.softmax(torch.log(reliability.clamp(min=1e-8))+self.gate(h),dim=1)
        if self.mode=='direct':
            return self.direct(h).squeeze(1),gate
        base=(gate*prior).sum(1)
        return base+.25*torch.tanh(self.correction(h).squeeze(1)),gate


def fit_temporal(train,validation,*,seed,mode='corrected',epochs=300,patience=70):
    torch.manual_seed(seed)
    # Current-row train statistics avoid repeated/padded history reweighting.
    center=train['x'][:,-1,:].mean(0); scale=train['x'][:,-1,:].std(0)
    scale=np.where(scale<1e-8,1.,scale)
    cap=max(float(np.max(train['y'])),1.)
    def tensors(data):
        return (torch.tensor((data['x']-center)/scale),
                torch.tensor(np.clip(data['prior']/cap,0,1)),
                torch.tensor(data['reliability']))
    tx,tp,tr=tensors(train);vx,vp,vr=tensors(validation)
    y=torch.tensor(train['y']/cap); weights=torch.tensor(equal_group_weights(train['groups']),dtype=torch.float32)
    model=TemporalPriorNet(tx.shape[-1],mode)
    optimizer=torch.optim.AdamW(model.parameters(),lr=5e-4,weight_decay=.05)
    rng=np.random.default_rng(seed)
    def val_loss():
        model.eval()
        with torch.no_grad():p=model(vx,vp,vr)[0].clamp(0,1).numpy()*cap
        return float(np.mean((p-validation['y'])**2))
    best=val_loss();epoch_best=0;state=copy.deepcopy(model.state_dict())
    for epoch in range(1,epochs+1):
        model.train()
        order=rng.permutation(len(y))
        for start in range(0,len(y),512):
            idx=order[start:start+512]
            p,g=model(tx[idx],tp[idx],tr[idx])
            loss=(weights[idx]*(p-y[idx]).square()).mean()
            if not torch.isfinite(loss):raise RuntimeError('nonfinite temporal loss')
            optimizer.zero_grad();loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(),2.,error_if_nonfinite=True)
            optimizer.step()
        current=val_loss()
        if current<best-1e-10:
            best=current;epoch_best=epoch;state=copy.deepcopy(model.state_dict())
        if epoch-epoch_best>patience:break
    model.load_state_dict(state);model.eval()
    return dict(model=model,center=center,scale=scale,cap=cap,
                selection=dict(seed=seed,mode=mode,epoch=epoch_best,validation_mse=best))


def predict_temporal(fit,data):
    with torch.no_grad():
        p,g=fit['model'](torch.tensor((data['x']-fit['center'])/fit['scale']),
                         torch.tensor(np.clip(data['prior']/fit['cap'],0,1)),
                         torch.tensor(data['reliability']))
    return np.clip(p.numpy()*fit['cap'],0,fit['cap']),g.numpy()
