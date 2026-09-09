"""FEMTO development: grouped endpoint selection and actual bounded-residual PP.

All six Learning bearings supply leave-one-bearing-out selection. Official
truncated Test_set endpoints are evaluated only after the selection is saved.
This changes the historical 5+1 training protocol; both arms receive the change.
"""
import copy
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
from torch import nn

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'experiments'), str(ROOT / 'src')]
from femto_sensor_adapter_v2 import load, LEARN, TEST, OFFICIAL

OUT = ROOT / 'results/femto_grouped_structural_v4'


def features(d):
    """Prefix-only, signed-log sensor history; no trajectory-length feature."""
    out = []
    for name in np.unique(d['bearing']):
        ids = np.flatnonzero(d['bearing'] == name)
        ids = ids[np.argsort(d['recording_index'][ids])]
        z = d['sensor'][ids].astype(float)
        z = np.sign(z) * np.log1p(np.abs(z))
        t = d['elapsed_s'][ids]
        for j, ix in enumerate(ids):
            base = np.median(z[:min(10, j+1)], axis=0)
            blocks = [np.eye(3)[int(d['condition'][ix])-1], [np.log1p(t[j])/10], z[j], z[j]-base]
            for w in (4, 16, 64):
                a = max(0, j-w+1)
                v = z[a:j+1]
                duration = max(float(t[j]-t[a]), 10.)
                blocks += [v.mean(0)-base, v.std(0), (z[j]-z[a])*1000/duration,
                           [len(v)/w, np.log1p(duration)/10]]
            out.append((ix, np.concatenate(blocks)))
    x = np.zeros((len(d['bearing']), len(out[0][1])), np.float32)
    for ix, v in out:
        x[ix] = v
    return x


def indices(d, names, endpoints=False):
    result = []
    for name in names:
        ix = np.flatnonzero(d['bearing'] == name)
        ix = ix[np.argsort(d['recording_index'][ix])]
        if endpoints:
            ix = ix[np.unique(np.round((len(ix)-1)*np.array([.25,.5,.75,.9])).astype(int))]
        else:
            ix = ix[np.unique(np.linspace(0, len(ix)-1, min(128,len(ix))).astype(int))]
        result.extend(ix)
    return np.asarray(result)


class Net(nn.Module):
    def __init__(self, dim, weight, bias, mode, bound):
        super().__init__()
        self.affine = nn.Linear(dim, 1)
        with torch.no_grad():
            self.affine.weight.copy_(torch.tensor(weight[None], dtype=torch.float32))
            self.affine.bias.fill_(float(bias))
        self.affine.requires_grad_(False)
        self.residual = nn.Sequential(nn.Linear(dim,32),nn.Tanh(),nn.Linear(32,1))
        nn.init.zeros_(self.residual[-1].weight)
        nn.init.zeros_(self.residual[-1].bias)
        self.mode, self.bound = mode, bound

    def forward(self, x):
        r = self.residual(x).squeeze(-1)
        if self.mode == 'pp':
            r = self.affine(x).squeeze(-1) + self.bound*torch.tanh(r/self.bound)
        return torch.nn.functional.softplus(r)


def train_model(x,y,g, config,seed, vx=None,vy=None, epochs=180):
    torch.manual_seed(seed)
    center, scale = x.mean(0), np.maximum(x.std(0),.05)
    target_scale = max(float(np.median([np.max(y[g==u]) for u in np.unique(g)])), 1.)
    z = (x-center)/scale
    target = np.maximum(y/target_scale, 1e-4)
    latent = target+np.log(-np.expm1(-target))
    weights = np.asarray([1/np.sum(g==u) for u in g]); weights /= weights.mean()
    dim = config.get('prior_dim', x.shape[1])
    affine = Ridge(alpha=100.).fit(z[:,:dim],latent,sample_weight=weights)
    coef=np.zeros(x.shape[1]); coef[:dim]=affine.coef_
    model = Net(x.shape[1],coef,affine.intercept_,config['mode'],config['bound'])
    anchor={k:v.detach().clone() for k,v in model.affine.named_parameters()}
    soft=config.get('soft_anchor',False)
    model.affine.requires_grad_(soft)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=config['lr'],weight_decay=.1)
    tx,ty,tw = [torch.tensor(a,dtype=torch.float32) for a in (z,y/target_scale,weights)]
    tv = None if vx is None else torch.tensor((vx-center)/scale,dtype=torch.float32)
    history=[]; best_loss=float('inf'); best_epoch=0; best=None
    for epoch in range(epochs+1):
        if epoch:
            model.train(); optimizer.zero_grad()
            p=model(tx)
            loss=(tw*((p-ty).square()+.1*(torch.log1p(p)-torch.log1p(ty)).square())).mean()
            if soft:
                loss=loss+.1*sum((p-anchor[k]).square().sum() for k,p in model.affine.named_parameters())
            loss.backward(); nn.utils.clip_grad_norm_(model.parameters(),2.); optimizer.step()
        model.eval()
        with torch.no_grad():
            train_loss=float(((model(tx)-ty).square()*tw).mean())
            val_loss=train_loss if tv is None else float(np.mean((model(tv).numpy()*target_scale-vy)**2))
        history.append([epoch,train_loss,val_loss])
        if tv is None or val_loss<best_loss:
            best_loss=val_loss; best_epoch=epoch; best=copy.deepcopy(model.state_dict())
    model.load_state_dict(best)
    return (model,center,scale,target_scale), {'epoch':best_epoch,'loss':best_loss,'history':history}


def predict(fit,x):
    model,center,scale,target_scale=fit
    with torch.no_grad():
        return model(torch.tensor((x-center)/scale,dtype=torch.float32)).numpy()*target_scale


def main():
    global OUT
    ap=argparse.ArgumentParser(); ap.add_argument('--lowdim',action='store_true'); args=ap.parse_args()
    if args.lowdim:
        OUT=ROOT/'results/femto_grouped_structural_v5'
    torch.set_num_threads(2); OUT.mkdir(parents=True,exist_ok=True)
    d=load(); x=features(d)
    configs=[{'mode':m,'bound':b,'lr':lr} for m in ('direct','pp')
             for b in ((1.,6.) if m=='pp' else (1.,)) for lr in (.001,.003)]
    if args.lowdim:
        configs=[{'mode':m,'bound':6.,'lr':lr,'prior_dim':4,'soft_anchor':soft}
                 for m in ('direct','pp') for lr in (.001,.003)
                 for soft in ((False,True) if m=='pp' else (False,))]
    search=[]
    for ci,c in enumerate(configs):
        folds=[]
        for held in LEARN:
            tr=indices(d,[n for n in LEARN if n!=held]); va=indices(d,[held],True)
            fit,log=train_model(x[tr],d['y'][tr],d['bearing'][tr],c,42,x[va],d['y'][va])
            folds.append({'held':held,'mse':log['loss'],'epoch':log['epoch']})
            (OUT/f'curve_{ci}_{held}.json').write_text(json.dumps(log))
        search.append({'config':c,'folds':folds,'mse':float(np.mean([r['mse'] for r in folds]))})
        print('screen',ci,c,search[-1]['mse'],flush=True)
    selected={m:min([r for r in search if r['config']['mode']==m],key=lambda r:r['mse']) for m in ('direct','pp')}
    manifest={'status':'retrospective development','protocol':'6 Learning bearings grouped pseudoendpoints; final all-six refit; different from historical 5+1',
              'seeds':list(range(42,47)),'search':search,'selected':selected,
              'source_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    (OUT/'runner_snapshot.py').write_text(Path(__file__).read_text())
    (OUT/'selection_manifest.json').write_text(json.dumps(manifest,indent=2))
    tr=indices(d,LEARN)
    te=np.asarray([np.flatnonzero(d['bearing']==n)[np.argmax(d['recording_index'][d['bearing']==n])] for n in TEST])
    assert np.allclose(d['y'][te],[OFFICIAL[n] for n in TEST])
    results={}
    for mode,sel in selected.items():
        epoch=int(np.median([r['epoch'] for r in sel['folds']]))
        predictions=[]
        for seed in range(42,47):
            fit,log=train_model(x[tr],d['y'][tr],d['bearing'][tr],sel['config'],seed,epochs=epoch)
            predictions.append(predict(fit,x[te]))
            (OUT/f'final_curve_{mode}_{seed}.json').write_text(json.dumps(log))
        p=np.asarray(predictions); y=d['y'][te]
        results[mode]={'epochs':epoch,'config':sel['config'],'pooled_r2':float(r2_score(y,p.mean(0))),
                       'rmse':float(np.sqrt(np.mean((y-p.mean(0))**2))),
                       'per_seed_r2':[float(r2_score(y,q)) for q in p]}
        np.savez_compressed(OUT/f'{mode}.npz',predictions=p,y=y,groups=d['bearing'][te])
        print('FINAL',mode,results[mode],flush=True)
    (OUT/'results.json').write_text(json.dumps(results,indent=2))


if __name__=='__main__':
    main()
