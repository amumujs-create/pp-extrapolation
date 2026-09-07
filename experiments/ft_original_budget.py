"""FT original-PP budget replay: 9 trials per seed, 300 epochs, patience 70."""
import os
os.environ.setdefault('OMP_NUM_THREADS','2');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
import argparse,copy,hashlib,json,sys,time,subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np,torch
from pp_extrapolation import regression_metrics,select_affine_initialization
from pp_extrapolation.model import transform_features,equal_group_weights
from extended_nn_benchmark import Model,GRID,data as matr_data
from regime_spline_deep_future import splits
OUT=Path('results/ft_original_budget_v1')

def prepare():
 OUT.mkdir(parents=True,exist_ok=True);parts=splits();parts['matr2019']=matr_data()
 for name,triplet in parts.items():
  d=OUT/name;d.mkdir(exist_ok=True);payload={}
  for label,p in zip(['train','validation','test'],triplet):
   for k in ['x','y','groups']:payload[f'{label}_{k}']=np.asarray(p[k],dtype=str if k=='groups' else np.float32)
  np.savez_compressed(d/'split.npz',**payload)

def worker(name,seed):
 torch.set_num_threads(2);folder=OUT/name;dest=folder/f'seed{seed}.json'
 if dest.exists():return
 data=np.load(folder/'split.npz');parts=[{k:data[f'{label}_{k}'] for k in ['x','y','groups']} for label in ['train','validation','test']];tr,va,te=parts;aff=select_affine_initialization(tr,va);cap=float(aff['target_scale']);z=transform_features(tr['x'],aff['center'],aff['scale']);vz=transform_features(va['x'],aff['center'],aff['scale']);x=torch.tensor(z);v=torch.tensor(vz);y=torch.tensor(tr['y']/cap);weights=torch.tensor(equal_group_weights(tr['groups']),dtype=torch.float32);rows=[];best_candidate=None
 for ci,cfg in enumerate(GRID):
  torch.manual_seed(seed);model=Model('ft_transformer',z.shape[1],cfg,aff,z,vz);opt=torch.optim.AdamW(model.parameters(),lr=cfg['lr'],weight_decay=cfg['wd']);rng=np.random.default_rng(seed)
  def val():
   model.eval()
   with torch.no_grad():p=torch.cat([model(b) for b in v.split(512)]).numpy().astype(np.float64)*cap
   return float(np.mean((np.clip(p,0,cap)-va['y'].astype(np.float64))**2))
  best=val();be=0;state=copy.deepcopy(model.state_dict());started=time.monotonic()
  for epoch in range(1,301):
   model.train();ix=rng.permutation(len(x))
   for start in range(0,len(x),512):
    b=ix[start:start+512];loss=(weights[b]*(model(x[b])-y[b])**2).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),2.);opt.step()
   score=val()
   if score<best-1e-10:best=score;be=epoch;state=copy.deepcopy(model.state_dict())
   if epoch-be>70:break
   if epoch%25==0:(folder/f'progress{seed}.json').write_text(json.dumps({'candidate':ci,'epoch':epoch,'best_epoch':be,'best_validation_mse':best}))
  row={'candidate':ci,'config':cfg,'validation_mse':best,'epoch':be,'epochs_executed':epoch,'seconds':time.monotonic()-started};rows.append(row);(folder/f'candidates{seed}.json').write_text(json.dumps(rows,indent=2));print('CANDIDATE',name,seed,ci,round(best,3),flush=True)
  if best_candidate is None or best<best_candidate[0]:best_candidate=(best,ci,state)
 selected=rows[best_candidate[1]];model=Model('ft_transformer',z.shape[1],selected['config'],aff,z,vz);model.load_state_dict(best_candidate[2]);model.eval();t=torch.tensor(transform_features(te['x'],aff['center'],aff['scale']))
 # Only the validation-selected model is scored on test.
 with torch.no_grad():raw=torch.cat([model(b) for b in t.split(512)]).numpy().astype(np.float64)*cap
 pred=np.clip(raw,0,cap);metrics=regression_metrics(te['y'],pred,te['groups']);np.savez_compressed(folder/f'seed{seed}.npz',raw=raw,prediction=pred,y=te['y'],groups=te['groups']);torch.save({'state':model.state_dict(),'config':selected['config'],'center':aff['center'],'scale':aff['scale'],'target_scale':cap},folder/f'seed{seed}.pt');dest.write_text(json.dumps({'seed':seed,'selected':selected,'candidates':rows,'metrics':metrics},indent=2));print('SEED_DONE',name,seed,metrics['pooled']['r2'],flush=True)

def summarize():
 datasets={}
 for name in ['hust','virkler','matr2019']:
  folder=OUT/name;rows=[json.load(open(folder/f'seed{s}.json')) for s in range(42,47)];ds=[np.load(folder/f'seed{s}.npz') for s in range(42,47)];r2=[r['metrics']['pooled']['r2'] for r in rows];datasets[name]={'runs':rows,'mean_r2':float(np.mean(r2)),'sd_r2':float(np.std(r2)),'ensemble':regression_metrics(ds[0]['y'],np.mean([d['prediction'] for d in ds],0),ds[0]['groups'])}
 (OUT/'results.json').write_text(json.dumps({'status':'post-hoc FT comparison against original PP training budget','datasets':datasets},indent=2));print('ALL_DONE',{n:d['ensemble']['pooled']['r2'] for n,d in datasets.items()},flush=True)

def main():
 a=argparse.ArgumentParser();a.add_argument('--dataset');a.add_argument('--seed',type=int);args=a.parse_args()
 if args.dataset:worker(args.dataset,args.seed);return
 prepare()
 # Two independent seed jobs, each with two torch CPU threads.
 jobs=[(n,s) for n in ['virkler','matr2019','hust'] for s in range(42,47)]
 def run(job):
  subprocess.run([sys.executable,__file__,'--dataset',job[0],'--seed',str(job[1])],check=True)
 # Three CPU workers keep the HUST completion practical while each torch job
 # remains capped at two intra-op threads.
 with ThreadPoolExecutor(max_workers=3) as pool:list(pool.map(run,jobs))
 summarize()
if __name__=='__main__':main()
