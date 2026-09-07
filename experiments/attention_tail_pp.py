#!/usr/bin/env python3
"""Attention-inside PP: support-conditioned transition to an affine tail."""
import json
from pathlib import Path

import numpy as np
import torch

from extended_nn_benchmark import Model
from pp_extrapolation import regression_metrics, select_affine_initialization, support_distance
from pp_extrapolation.model import _affine_prediction, transform_features

SEEDS=(42,43,44,45,46)
BETAS=(0.0,0.02,0.05,0.1,0.25,0.5,1.0,2.0,4.0)

def load_split(folder):
    d=np.load(folder/'split.npz')
    return [{k:d[f'{label}_{k}'] for k in ('x','y','groups')} for label in ('train','validation','test')]

def attention_predictions(folder,parts,aff):
    out=[]
    tr,va,te=parts;z=transform_features(tr['x'],aff['center'],aff['scale']);vz=transform_features(va['x'],aff['center'],aff['scale'])
    for seed in SEEDS:
        state=torch.load(folder/f'seed{seed}.pt',map_location='cpu',weights_only=False);model=Model('ft_transformer',z.shape[1],state['config'],aff,z,vz);model.load_state_dict(state['state']);model.eval()
        pred=[]
        for part in (va,te):
            x=torch.tensor(transform_features(part['x'],state['center'],state['scale']))
            with torch.no_grad():raw=torch.cat([model(batch) for batch in x.split(512)]).numpy()*state['target_scale']
            pred.append(np.clip(raw,0,state['target_scale']))
        out.append(pred)
    return np.asarray([x[0] for x in out]),np.asarray([x[1] for x in out])

def blend(attention,affine,distance,beta,cap):
    trust=np.exp(-float(beta)*np.maximum(distance,0.0))
    return np.clip(affine[None,:]+trust[None,:]*(attention-affine[None,:]),0,cap)

def evaluate(name):
    folder=Path('results/ft_original_budget_v1')/name;tr,va,te=parts=load_split(folder);aff=select_affine_initialization(tr,va);cap=aff['target_scale']
    vat,tat=attention_predictions(folder,parts,aff);vaa=_affine_prediction(aff['initialization'],va['x'],aff['center'],aff['scale'],cap);taa=_affine_prediction(aff['initialization'],te['x'],aff['center'],aff['scale'],cap)
    vd,_=support_distance(tr['x'][:,[0]],va['x'][:,[0]]);td,_=support_distance(tr['x'][:,[0]],te['x'][:,[0]])
    candidates=[]
    base_errors=(vat.mean(0)-va['y'])**2;groups=np.unique(va['groups'])
    for beta in BETAS:
        p=blend(vat,vaa,vd,beta,cap).mean(0);err=(p-va['y'])**2;mse=float(err.mean());base=float(base_errors.mean());wins=sum(float(err[va['groups']==g].mean())<float(base_errors[va['groups']==g].mean()) for g in groups)
        candidates.append({'beta':beta,'validation_mse':mse,'relative_gain_vs_attention':(base-mse)/max(base,1e-12),'group_win_fraction':wins/max(len(groups),1)})
    eligible=[r for r in candidates if r['beta']==0 or (r['relative_gain_vs_attention']>=.02 and r['group_win_fraction']>=.8)]
    selected=min(eligible,key=lambda r:(r['validation_mse'],r['beta']));pred=blend(tat,taa,td,selected['beta'],cap)
    def pack(matrix):
        runs=[regression_metrics(te['y'],p,te['groups']) for p in matrix];r=np.asarray([x['pooled']['r2'] for x in runs]);return {'mean_r2':float(r.mean()),'sample_sd_r2':float(r.std(ddof=1)),'ensemble':regression_metrics(te['y'],matrix.mean(0),te['groups'])}
    return {'selected':selected,'candidates':candidates,'attention':pack(tat),'attention_tail_pp':pack(pred)}

def main():
    torch.set_num_threads(2);datasets={}
    for name in ('hust','virkler','matr2019'):
        datasets[name]=evaluate(name);a=datasets[name]['attention']['ensemble']['pooled']['r2'];p=datasets[name]['attention_tail_pp']['ensemble']['pooled']['r2'];print(name,datasets[name]['selected']['beta'],a,p,flush=True)
    out=Path('results/attention_tail_pp_v1');out.mkdir(parents=True,exist_ok=True);(out/'results.json').write_text(json.dumps({'status':'post-hoc development on inspected datasets','formula':'affine + exp(-beta*support_distance)*(attention-affine)','selection':'validation mean gain >=2% and >=80% validation-group wins; beta=0 fallback','datasets':datasets},indent=2))
if __name__=='__main__':main()
