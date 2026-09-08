"""Cross-fitted development selection of a boundary-consistent lifetime path."""
import json, sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments')]
import znion_bq_confirmatory as data
from naion_prefix_gate_eval import prefix_descriptor
from pp_extrapolation import fit_boundary_quotient_pp,predict_boundary_quotient,regression_metrics
OUT=ROOT/'results/znion_boundary_consistent_development'
TAUS=(0.,.001,.003,.01,.03,.1)

def components(memory_cells,cells,fit,boundary,scale):
    rows=data.make_rows(cells,boundary,scale,'tail')
    base=predict_boundary_quotient(fit,rows)
    desc=np.array([prefix_descriptor(c['cycle'],c['capacity'],min(boundary,len(c['cycle'])-1)) for c in memory_cells.values()])
    center=np.median(desc,0);spread=np.maximum(np.quantile(desc,.75,axis=0)-np.quantile(desc,.25,axis=0),1e-5)
    memory=(desc-center)/spread;life=np.log([c['cycle'][-1] for c in memory_cells.values()])
    latent=[];gates=[]
    for uid,c in sorted(cells.items()):
        idx=np.flatnonzero(c['cycle']-c['cycle'][0]>boundary)
        if not len(idx):continue
        q=(prefix_descriptor(c['cycle'],c['capacity'],boundary)-center)/spread
        dist=((memory-q)**2).sum(1);w=np.exp(-(dist-dist.min()));lh=np.exp(w@life/w.sum())
        use=c['cycle']-c['cycle'][0]<=boundary
        slope=np.polyfit(c['cycle'][use],c['capacity'][use],1)[0]
        gate=1/(1+np.exp(np.clip(-slope/1e-4,-700,700)))
        latent.extend(np.maximum(lh-c['cycle'][idx],0));gates.extend([gate]*len(idx))
    return rows,base,np.array(latent),np.array(gates)

def predict(parts,tau):
    rows,base,latent,g=parts
    factor=np.ones(len(base)) if tau==0 else rows['margin']/(rows['margin']+tau)
    return (1-g)*base+g*latent*factor

def main():
    torch.set_num_threads(2)
    tr,_=data.load_split('train');va,_=data.load_split('validation');dev={**tr,**va}
    boundary=152.;scale=370.;folds=[]
    # Every held-out cell is excluded from scaling, affine fitting and RBF memory.
    for uid,c in sorted(dev.items()):
        if len(c['cycle'])-1-boundary<50:continue
        rest={u:v for u,v in dev.items() if u!=uid}
        train=data.make_rows(rest,boundary,scale,'prefix')
        fit=fit_boundary_quotient_pp(train,train,seed=42,alpha=1000,residual_bound=.5,max_epochs=0)
        parts=components(rest,{uid:c},fit,boundary,scale)
        scores={str(t):regression_metrics(parts[0]['y'],predict(parts,t),parts[0]['groups'])['pooled']['r2'] for t in TAUS}
        folds.append({'unit':uid,'r2':scores})
    selection=[{'tau':t,'mean_unit_r2':float(np.mean([f['r2'][str(t)] for f in folds]))} for t in TAUS]
    chosen=max(selection,key=lambda v:v['mean_unit_r2'])['tau']
    OUT.mkdir(parents=True,exist_ok=True)
    # Write selection before loading any evaluation cohort.
    (OUT/'selection.json').write_text(json.dumps({'folds':folds,'selection':selection,'chosen_tau':chosen},indent=2))
    train=data.make_rows(tr,boundary,scale,'prefix');val=data.make_rows(va,boundary,scale,'tail')
    fit=fit_boundary_quotient_pp(train,val,seed=42,alpha=1000,residual_bound=.5,max_epochs=300,patience=50)
    results={}
    for name,directory in [('seen','znion_bq_confirmatory/test'),('v2','znion_rbf_regime_confirmation_v2'),('v3','znion_rbf_regime_confirmation_v3')]:
        cells={p.stem:data.read_cell(p) for p in sorted((ROOT/'data'/directory).glob('*.xlsx'))}
        cells={u:c for u,c in cells.items() if c is not None and len(c['cycle'])-1-boundary>=50}
        parts=components(dev,cells,fit,boundary,scale);rows=parts[0]
        results[name]={label:regression_metrics(rows['y'],predict(parts,t),rows['groups']) for label,t in [('baseline',0.),('selected',chosen)]}
        np.savez_compressed(OUT/f'{name}.npz',y=rows['y'],groups=rows['groups'],baseline=predict(parts,0),selected=predict(parts,chosen))
    result={'status':'retrospective development; not untouched confirmation','chosen_tau':chosen,'selection':selection,'folds':folds,'limitation':'LOO selection uses affine-only BQ; replay uses validation-selected residual. Fixed boundary/alpha inherited from prior development.','replay':results}
    (OUT/'results.json').write_text(json.dumps(result,indent=2))
    print('SELECTION',selection,'chosen',chosen,flush=True)
    for name,v in results.items():print(name,{k:(m['pooled']['r2'],m['unit_macro_r2']) for k,m in v.items()},flush=True)
if __name__=='__main__':main()
