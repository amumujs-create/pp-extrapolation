#!/usr/bin/env python3
"""Causal multiscale latent PP for NASA batteries; validation-only selection."""
from pathlib import Path
import argparse,json,sys
import numpy as np,pandas as pd,torch
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'src'),str(ROOT/'experiments'),str(ROOT.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,regression_metrics,select_affine_initialization
from pp_extrapolation.model import AffineInitialization,fit_feature_scale
from run_affine_tail_external_three import NASA_BOUNDARY,nasa_cells
SEEDS=range(42,47);OUT=ROOT/'results/nasa_causal_multiscale_pp_v1';PRESETS=('short','multiscale','moments');SEPS=(0.,.001,.01)

def coordinate_affine(train,validation):
 base=select_affine_initialization({'x':train['x'][:,:1],'y':train['y'],'groups':train['groups']},{'x':validation['x'][:,:1],'y':validation['y'],'groups':validation['groups']})
 center,scale=fit_feature_scale(train['x']);w=np.zeros(train['x'].shape[1],dtype=np.float32);w[0]=base['initialization'].weight[0]
 return {**base,'center':center,'scale':scale,'initialization':AffineInitialization(w,base['initialization'].bias,base['initialization'].alpha)}

def cell_rows(frame,name,preset):
 f=frame.copy();initial=float(f.iloc[0].capacity);h=((f.capacity.to_numpy(float)-NASA_BOUNDARY)/(initial-NASA_BOUNDARY));cycles=f.cycle.to_numpy(float);n=len(h);x=[];y=[];idx=[]
 for i in range(2,n):
  blocks=[h[i]]
  horizons=(3,) if preset=='short' else (3,5,10)
  for k in horizons:
   a=max(0,i-k+1);z=h[a:i+1];dt=max(cycles[i]-cycles[a],1.);s=(z[-1]-z[0])/dt
   blocks += [z.mean(),s]
   if preset in ('multiscale','moments'):blocks += [z.std()]
  if preset=='moments':
   d=np.diff(h[max(0,i-9):i+1]);blocks += [np.mean(d[-3:]) if len(d) else 0.,(d[-1]-d[0]) if len(d)>1 else 0.]
  x.append(blocks);y.append(n-1-i);idx.append(i)
 return np.asarray(x,np.float32),np.asarray(y,np.float32),np.asarray(idx),h[2:]
def subset(cells,names,preset,cut,mode):
 xs=[];ys=[];gs=[]
 for name in names:
  x,y,idx,h=cell_rows(cells[name],name,preset);m=h>=cut if mode=='train' else h<cut;xs.append(x[m]);ys.append(y[m]);gs.extend([name]*int(m.sum()))
 return {'x':np.concatenate(xs),'y':np.concatenate(ys),'groups':np.asarray(gs)}
def folds(preset):
 cells=nasa_cells();order=['B0005','B0006','B0007','B0018'];out=[]
 for i,test in enumerate(order):
  val=order[(i+1)%4];train=[c for c in order if c not in (test,val)];health=[]
  for c in train:
   f=cells[c];initial=float(f.iloc[0].capacity);health.extend(((f.capacity-NASA_BOUNDARY)/(initial-NASA_BOUNDARY)).to_numpy()[2:])
  cut=max(.5,float(np.min(np.asarray(health)[np.asarray(health)>=.5])))
  out.append({'test_cell':test,'validation_cell':val,'train_cells':train,'train':subset(cells,train,preset,cut,'train'),'validation':subset(cells,[val],preset,cut,'tail'),'test':subset(cells,[test],preset,cut,'tail')})
 return out
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--fixed-preset',choices=PRESETS);ap.add_argument('--fixed-separation',type=float);ap.add_argument('--coordinate-affine',action='store_true');ap.add_argument('--output',type=Path,default=OUT);args=ap.parse_args();out=args.output
 affine_select=coordinate_affine if args.coordinate_affine else select_affine_initialization
 torch.set_num_threads(2);out.mkdir(parents=True,exist_ok=True);prepared={p:folds(p) for p in PRESETS};audit=[];pred=[[] for _ in SEEDS];ys=[];gs=[]
 for fi in range(4):
  search=[]
  for preset in ((args.fixed_preset,) if args.fixed_preset else PRESETS):
   f=prepared[preset][fi];a=affine_select(f['train'],f['validation'])
   for sep in ((args.fixed_separation,) if args.fixed_separation is not None else SEPS):
    q=fit_latent_regime_pp(f['train'],f['validation'],seed=42,affine_selection=a,max_epochs=400,patience=80,separation_weight=sep,gate_weight=0.,width=32);search.append({'preset':preset,'separation':sep,'validation_mse':q.selection['validation_mse']})
  best=min(search,key=lambda z:z['validation_mse']);f=prepared[best['preset']][fi];a=affine_select(f['train'],f['validation']);runs=[]
  for j,s in enumerate(SEEDS):
   q=fit_latent_regime_pp(f['train'],f['validation'],seed=s,affine_selection=a,max_epochs=400,patience=80,separation_weight=best['separation'],gate_weight=0.,width=32);p=predict_latent_regime(q,f['test']['x']);pred[j].append(p);runs.append({'seed':s,**q.selection})
  ys.append(f['test']['y']);gs.append(f['test']['groups']);audit.append({'test_cell':f['test_cell'],'validation_cell':f['validation_cell'],'selected':best,'search':search,'runs':runs});print(f['test_cell'],best,flush=True)
 y=np.concatenate(ys);g=np.concatenate(gs);P=np.asarray([np.concatenate(x) for x in pred]);r={'status':'post-hoc NASA development; causal multiscale latent PP; preset and regularizer selected by validation','fixed_preset':args.fixed_preset,'fixed_separation':args.fixed_separation,'coordinate_affine':args.coordinate_affine,'folds':audit,'ensemble':regression_metrics(y,P.mean(0),g),'per_seed':[regression_metrics(y,p,g) for p in P]};(out/'results.json').write_text(json.dumps(r,indent=2)+'\n');np.savez_compressed(out/'predictions.npz',prediction=P,y=y,groups=g);print('FINAL',r['ensemble']['pooled']['r2'],r['ensemble']['unit_macro_r2'])
if __name__=='__main__':main()
