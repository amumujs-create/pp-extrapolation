"""Matched gate ablations; existing artifact-producing module stays unchanged."""
import copy
import numpy as np
import torch
from .component_transfer import ComponentGate, ComponentFit, predict_components
from .relation_local_transport import group_weights


class TiedComponentGate(ComponentGate):
    """Same allocated head parameters, one effective logit shared by components."""
    def forward(self, context):
        logits=self.net(context)
        return torch.sigmoid(logits.mean(1,keepdim=True)).expand(-1,3)


def fit_ablation(episodes,validation,*,seed,shared=False,cap=None,max_epochs=200,patience=40):
    data={k:np.concatenate([e[k] for e in episodes]) for k in ('c','z','base','y','groups')}
    w=group_weights(data['groups'])
    center=w@data['z'];scale=np.sqrt(w@(data['z']-center)**2);scale[scale<1e-6]=1.
    torch.manual_seed(seed)
    model=TiedComponentGate() if shared else ComponentGate()
    fit=ComponentFit(model,center,scale,False,cap,{})
    vw=group_weights(validation['groups'])
    best=float(vw@(validation['base']-validation['y'])**2)
    best_epoch,best_enabled,state=-1,False,copy.deepcopy(model.state_dict())
    z=torch.tensor(np.clip((data['z']-center)/scale,-8.,8.),dtype=torch.float32)
    c,base,y=[torch.tensor(data[k],dtype=torch.float32) for k in ('c','base','y')]
    ix=[np.flatnonzero(data['groups']==g) for g in np.unique(data['groups'])]
    floor=max(float(np.mean((data['base']-data['y'])**2))*.05,1e-8)
    risks0=[(base[a]-y[a]).square().mean().clamp_min(floor) for a in ix]
    opt=torch.optim.AdamW(model.parameters(),lr=.01,weight_decay=.01)
    history=[]
    for epoch in range(max_epochs):
        correction=(c*model(z)).sum(1)
        p=(base+correction).clamp_min(0.)
        if cap is not None:p=p.clamp_max(cap)
        risks=torch.stack([(p[a]-y[a]).square().mean()/b for a,b in zip(ix,risks0)])
        loss=risks.mean()+.25*(risks-1).relu().square().mean()+.01*correction.square().mean()/max(floor/.05,1e-8)
        if not torch.isfinite(loss):raise RuntimeError('nonfinite ablation objective')
        opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
        if (epoch+1)%5==0:
            fit.enabled=True
            p=predict_components(fit,validation['c'],validation['z'],validation['base'])
            val=float(vw@(p-validation['y'])**2)
            history.append(dict(epoch=epoch+1,source_ratio=float(risks.mean().detach()),validation_unit_mse=val))
            if val<best-1e-8:
                best,best_epoch,best_enabled,state=val,epoch+1,True,copy.deepcopy(model.state_dict())
            if epoch+1-max(best_epoch,0)>=patience:break
    model.load_state_dict(state);fit.enabled=best_enabled
    fit.selection=dict(seed=seed,shared=shared,selected_epoch=best_epoch,enabled=best_enabled,
        validation_unit_mse=best,history=history,parameters=sum(p.numel() for p in model.parameters()))
    return fit
