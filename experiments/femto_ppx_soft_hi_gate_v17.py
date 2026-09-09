"""Validation-selected soft PP-X blend of waveform NN and constrained HI prior."""
import hashlib,json,sys
from pathlib import Path
import numpy as np
from sklearn.metrics import r2_score
ROOT=Path(__file__).resolve().parents[1];sys.path[:0]=[str(ROOT/'experiments'),str(ROOT/'src')]
from femto_sensor_adapter_v2 import load,LEARN,TEST
from femto_corrected_benchmark_v2 import TRAIN,VAL
from femto_constrained_hi_pp_v16 import make_features,fit_hi,trajectory,rul
OUT=ROOT/'results/femto_ppx_soft_hi_gate_v17';WEIGHTS=(0.,.05,.1,.15,.2,.3,.4,.5)

def hi_predictions(d,x,model,mu,sd,names,c,fractions=None):
 out=[]
 for name in names:
  ix,h,t=trajectory(d,x,model,mu,sd,name);ends=[len(ix)-1] if fractions is None else [int(round((len(ix)-1)*q)) for q in fractions]
  out.extend(rul(h,t,e,c['window'],c['scale'],c['min_rate']) for e in ends)
 return np.asarray(out)

def main():
 OUT.mkdir(parents=True,exist_ok=True);d=load();cfg=json.load(open(ROOT/'results/femto_constrained_hi_pp_v16/selection_manifest.json'))['selected']
 x=make_features(d,cfg['kind']);model,mu,sd=fit_hi(d,x,[u for u in LEARN if int(u.replace('Bearing','').replace('_','')) in TRAIN],cfg['alpha'])
 source=np.load(ROOT/'results/femto_waveform_pp_v8/direct.npz');vp=source['validation_prediction'].mean(0);vy=source['validation_y']
 hp=hi_predictions(d,x,model,mu,sd,['Bearing3_2'],cfg,(.5,.6,.7,.8,.9));assert len(hp)==len(vp)==len(vy)
 screen=[]
 for weight in WEIGHTS:
  q=(1-weight)*vp+weight*hp;screen.append({'prior_weight':weight,'validation_mse':float(np.mean((q-vy)**2))})
 selected=min(screen,key=lambda q:q['validation_mse']);weight=selected['prior_weight']
 manifest={'status':'retrospective inductive development; no TTA','gate':'single global HI-prior weight selected on held-out Bearing3_2 five prefix endpoints',
  'weights':WEIGHTS,'screen':screen,'selected':selected,'test_inputs_used_for_selection':False,'test_batch_statistics_used':False,
  'source_hash':hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
 (OUT/'selection_manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 hp=hi_predictions(d,x,model,mu,sd,TEST,cfg);P=source['predictions'];Q=(1-weight)*P+weight*hp[None,:];y=source['y']
 result={'model':'PP-X waveform executor + constrained HI prior','prior_weight':weight,'pooled_r2':float(r2_score(y,Q.mean(0))),
  'rmse':float(np.sqrt(np.mean((y-Q.mean(0))**2))),'seed_r2':[float(r2_score(y,q)) for q in Q],
  'baseline_waveform_r2':float(r2_score(y,P.mean(0)))}
 (OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n');np.savez_compressed(OUT/'predictions.npz',prediction=Q,y=y,groups=source['groups'],hi_prior=hp);print(json.dumps(result,indent=2))
if __name__=='__main__':main()
