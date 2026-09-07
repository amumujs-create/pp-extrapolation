#!/usr/bin/env python3
from pathlib import Path
import json,sys,time,platform,resource
import numpy as np,torch
R=Path(__file__).resolve().parents[1];sys.path[:0]=[str(R/'src'),str(R/'experiments'),str(R.parent/'ca-css-ncmapss')]
from pp_extrapolation import fit_latent_regime_pp,predict_latent_regime,select_affine_initialization

def measure(name,tr,va,tx,cfg):
 torch.manual_seed(42);t=time.perf_counter();m=fit_latent_regime_pp(tr,va,seed=42,affine_selection=select_affine_initialization(tr,va),max_epochs=cfg.get('epochs',400),patience=cfg.get('patience',80),separation_weight=cfg['separation'],gate_weight=0.,width=cfg.get('width',32));train=time.perf_counter()-t
 for _ in range(3):predict_latent_regime(m,tx)
 t=time.perf_counter();reps=20
 for _ in range(reps):predict_latent_regime(m,tx)
 infer=(time.perf_counter()-t)/reps
 return {'parameter_count':sum(p.numel() for p in m.model.parameters()),'train_seconds':train,'inference_seconds_per_batch':infer,'inference_microseconds_per_row':infer/len(tx)*1e6,'n_train':len(tr['y']),'n_validation':len(va['y']),'n_inference':len(tx),'selected_epoch':m.selection['selected_epoch']}
def main():
 torch.set_num_threads(2);out={}
 from nasa_causal_multiscale_pp import folds
 a=json.load(open(R/'results/nasa_causal_multiscale_pp_v1/results.json'))['folds'][0]['selected'];f=folds(a['preset'])[0];out['nasa_representative_fold']=measure('nasa',f['train'],f['validation'],f['test']['x'],{'separation':a['separation'],'width':32,'epochs':400,'patience':80})
 from apps.ncmapss_data_utils import FEATURE_COLS
 from ncmapss_tra_quantile_split import make_tra_hard_split
 from ncmapss_pp_multiscale import as_rows,make_features
 s=make_tra_hard_split((R/'data/N-CMAPSS_DS02-006.h5').resolve(),max_windows_per_unit=1500,random_seed=42);a=json.load(open(R/'results/ncmapss_pp_multiscale_v1/results.json'))['runs'][0]['selected'];tr=as_rows(s.train,list(FEATURE_COLS),a['preset']);va=as_rows(s.val,list(FEATURE_COLS),a['preset']);tx=make_features(s.all_windows,list(FEATURE_COLS),a['preset']);out['ncmapss_seed42']=measure('ncmapss',tr,va,tx,{'separation':a['separation_weight'],'width':24,'epochs':300,'patience':70})
 result={'hardware':{'platform':platform.platform(),'processor':platform.processor(),'torch':torch.__version__,'threads':torch.get_num_threads()},'peak_process_rss_bytes_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,'peak_process_rss_mib_macos':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024**2,'profiles':out,'scope':'single-process CPU representative fit; excludes dataset loading and hyperparameter search'}
 d=R/'results/final_pp_compute_profile_v1';d.mkdir(exist_ok=True);(d/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
