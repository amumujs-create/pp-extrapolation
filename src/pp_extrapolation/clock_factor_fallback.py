"""Data-audited affine clock factorization control, not a novelty claim."""
import copy
import numpy as np
import torch
from torch import nn
from .relation_local_transport import group_weights


def audit_clock(train, index=0):
    x,y,g=np.asarray(train['x'],float),np.asarray(train['y'],float),np.asarray(train['groups'])
    slopes=[]; errors=[]
    for unit in np.unique(g):
        t=x[g==unit,index];v=y[g==unit]
        if len(t)<3 or np.ptp(t)<=0:
            raise ValueError('clock audit requires varying clock within every unit')
        slope=np.sum((t-t.mean())*(v-v.mean()))/np.sum((t-t.mean())**2)
        slopes.append(slope)
        errors.append(float(np.max(abs(v-v.mean()-slope*(t-t.mean())))))
    common=float(np.median(slopes))
    if common>=0 or max(abs(np.asarray(slopes)-common))>1e-5 or max(errors)>1e-4:
        raise ValueError('common decreasing affine clock relation not established on TRAIN')
    return dict(slope=common,unit_slopes=slopes,max_residual=max(errors))


def predict_clock(fit,x):
    x=np.asarray(x,float)
    if x.ndim!=2 or x.shape[1]!=len(fit['feature_mask']) or not np.isfinite(x).all():
        raise ValueError('finite aligned clock features required')
    z=(x[:,fit['feature_mask']]-fit['center'])/fit['scale']
    fit['model'].eval()
    with torch.no_grad():
        endpoint=fit['model'](torch.tensor(z,dtype=torch.float32)).squeeze(1).numpy()*fit['cap']
    return np.clip(endpoint+fit['clock_slope']*x[:,fit['clock_index']],0,fit['cap'])


def fit_clock(train,val,*,seed=42,consistency=.1,clock_index=0,max_epochs=300):
    if set(train['groups']) & set(val['groups']):
        raise ValueError('TRAIN and validation units must be disjoint')
    if consistency<0:
        raise ValueError('nonnegative consistency penalty required')
    audit=audit_clock(train,clock_index)
    x=np.asarray(train['x'],float);y=np.asarray(train['y'],float)
    mask=np.arange(x.shape[1])!=clock_index
    center=x[:,mask].mean(0);scale=x[:,mask].std(0);scale[scale<1e-6]=1.
    cap=float(max(y.max(),1.))
    target=(y-audit['slope']*x[:,clock_index])/cap
    torch.manual_seed(seed)
    model=nn.Sequential(nn.Linear(int(mask.sum()),32),nn.Tanh(),nn.Linear(32,32),nn.Tanh(),nn.Linear(32,1))
    fit=dict(model=model,feature_mask=mask,center=center,scale=scale,cap=cap,
             clock_slope=audit['slope'],clock_index=clock_index)
    tx=torch.tensor((x[:,mask]-center)/scale,dtype=torch.float32)
    ty=torch.tensor(target,dtype=torch.float32)
    w=torch.tensor(group_weights(train['groups']),dtype=torch.float32)
    groups=[np.flatnonzero(train['groups']==g) for g in np.unique(train['groups'])]
    vw=group_weights(val['groups']);best=float('inf');best_epoch=0;state=None
    opt=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.1)
    history=[]
    for epoch in range(1,max_epochs+1):
        endpoint=model(tx).squeeze(1)
        invariant=torch.stack([endpoint[ix].var(unbiased=False) for ix in groups]).mean()
        loss=(w*(endpoint-ty).square()).sum()+consistency*invariant
        opt.zero_grad();loss.backward();opt.step()
        pred=predict_clock(fit,val['x']);risk=float(vw @ (pred-val['y'])**2)
        if risk<best-1e-9:
            best,best_epoch,state=risk,epoch,copy.deepcopy(model.state_dict())
        if epoch%25==0:
            history.append(dict(epoch=epoch,loss=float(loss.detach()),validation_unit_mse=risk))
        if epoch-best_epoch>50:
            break
    model.load_state_dict(state)
    fit['selection']=dict(seed=seed,consistency=consistency,selected_epoch=best_epoch,
        validation_unit_mse=best,history=history,clock_audit=audit,
        parameters=sum(p.numel() for p in model.parameters()))
    return fit
