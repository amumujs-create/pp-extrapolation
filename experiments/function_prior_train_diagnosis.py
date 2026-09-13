"""TRAIN-only diagnostic, no validation/test reads or successor selection."""
from pathlib import Path
import json
import sys
import hashlib
from dataclasses import replace
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from pp_extrapolation.function_prior_sharing import fit_bank,integral,SharingRule,SharingFit,predict_sharing,DESCRIPTOR
from component_transfer_screen import source_episode_indices
from relation_local_transport_screen import subset,write


def score(y,p,g):
    total=len(y);mask=np.isfinite(p)
    y,p,g=y[mask],p[mask],g[mask]
    errors=[float(np.mean((y[g==u]-p[g==u])**2)) for u in np.unique(g)]
    return dict(unit_mse=float(np.mean(errors)),rmse=float(np.sqrt(np.mean((y-p)**2))),
                unit_mse_values=errors,scored_rows=len(y),total_rows=total,scored_units=len(errors))


def own_prediction(bank,rows,allow_missing=False):
    lookup={str(u):i for i,u in enumerate(bank.groups)}
    present=np.array([str(u) in lookup for u in rows['groups']])
    if not present.all():
        if not allow_missing:raise ValueError('query unit absent from own-unit bank')
        p=np.full(len(present),np.nan)
        p[present]=own_prediction(bank,subset(rows,present))
        return p
    ix=[lookup[str(u)] for u in rows['groups']]
    with torch.no_grad():
        return integral(torch.tensor(bank.means[ix,None]),torch.tensor(rows['x'][:,0],dtype=torch.float64)).numpy()[:,0]


def donor_predictions(bank,rows):
    with torch.no_grad():
        return integral(torch.tensor(bank.means)[None].expand(len(rows['y']),-1,-1),
                        torch.tensor(rows['x'][:,0],dtype=torch.float64)).numpy()


def ridge_predict(x,y,q,alpha=1.):
    # Every normalization statistic and fitted label excludes the held-out unit.
    center=x.mean(0);scale=x.std(0);scale[scale<1e-8]=1.
    a=np.column_stack((np.ones(len(x)),(x-center)/scale))
    b=np.column_stack((np.ones(len(q)),(q-center)/scale))
    penalty=np.eye(a.shape[1])*alpha;penalty[0,0]=0
    return b@np.linalg.solve(a.T@a+penalty,a.T@y)


def main():
    torch.set_num_threads(2)
    out=ROOT/'results/function_prior_train_diagnosis_v3';out.mkdir(exist_ok=False)
    path=ROOT/'results/relation_local_transport_mich_v1/train_rows.npz'
    write(out/'protocol.json',dict(scope='TRAIN only; validation/test never loaded',
        fixed_design='six original unit/support splits; alpha=1 ridge; no tuning',
        oracle='own-full uses all own labels; own-early uses own prefix RUL labels; best-donor uses query labels; all diagnostic only',
        correction='v1 signed cutoff error invalid; v2 correctly split but stopped on absent prefix unit; v3 reports missing own-prefix coverage explicitly',
        refit='TRAIN-internal donor early->full-support surrogate; not actual TRAIN+VAL refit',
        hashes={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in
                (path,Path(__file__),ROOT/'src/pp_extrapolation/function_prior_sharing.py')}))
    train=dict(np.load(path));full=fit_bank(train)
    results=dict(full_unit_fit=score(train['y'],own_prediction(full,train),train['groups']),episodes=[])
    arrays={}
    for fold in range(3):
        for frac in (.6,.8):
            donor,query,cut,held=source_episode_indices(train,'mich',fold,frac)
            d,q=subset(train,donor),subset(train,query)
            assert not set(d['groups'])&set(q['groups'])
            assert d['x'][:,0].min()>q['x'][:,0].max()
            bank=fit_bank(d)
            early_rows=subset(train,train['x'][:,0]>=-cut)
            assert early_rows['x'][:,0].min()>q['x'][:,0].max()
            assert len(early_rows['y'])<len(train['y'])
            early=fit_bank(early_rows)
            donor_full=fit_bank(subset(train,np.isin(train['groups'],np.unique(d['groups']))))
            assert list(bank.groups)==list(donor_full.groups)
            # Unit's own later labels are used only in this retrospective family-fit diagnostic.
            predictions=dict(own_full_oracle=own_prediction(full,q),own_early_label_oracle=own_prediction(early,q,allow_missing=True))
            present=np.isfinite(predictions['own_early_label_oracle'])
            # Matched rows for the early/full family-fit contrast; unavailable prefixes stay missing.
            predictions['own_full_matched_prefix_oracle']=np.where(present,predictions['own_full_oracle'],np.nan)
            matrix=donor_predictions(bank,q)
            predictions['uniform_coefficient_mean']=own_mean=np.zeros(len(q['y']))
            with torch.no_grad():
                own_mean[:]=integral(torch.tensor(bank.means.mean(0))[None,None].expand(len(q['y']),1,3),
                    torch.tensor(q['x'][:,0],dtype=torch.float64)).numpy()[:,0]
            predictions['uniform_function_mean']=matrix.mean(1)
            predictions['nearest_descriptor']=np.zeros(len(q['y']))
            predictions['oracle_best_single_donor']=np.zeros(len(q['y']))
            ranks=[];matches=[];distances=[];losses=[]
            for u in np.unique(q['groups']):
                ix=q['groups']==u
                loss=((matrix[ix]-q['y'][ix,None])**2).mean(0)
                # Same causal current-row descriptors as the deployed router, not query suffix summaries.
                delta=(q['x'][ix][:,DESCRIPTOR,None].transpose(0,2,1)-bank.descriptors[None])/bank.scale
                distance=np.mean(delta**2,axis=(0,2))
                nearest=np.argmin((delta**2).mean(-1),axis=1)
                predictions['nearest_descriptor'][ix]=matrix[ix][np.arange(ix.sum()),nearest]
                best=int(loss.argmin());predictions['oracle_best_single_donor'][ix]=matrix[ix,best]
                rank=np.argsort(np.argsort(distance));ranks.append(float(rank[best]/max(len(rank)-1,1)))
                matches.append(bool(distance.argmin()==best));distances.append(distance);losses.append(loss)
            for name,b in dict(early=bank,full=donor_full,
                function_only=replace(bank,means=donor_full.means,covariances=donor_full.covariances),
                descriptor_only=replace(bank,descriptors=donor_full.descriptors,scale=donor_full.scale,center=donor_full.center),
                calibration_only=replace(bank,calibration=donor_full.calibration,evidence_variance=donor_full.evidence_variance,
                                         evidence_reliability=donor_full.evidence_reliability)).items():
                predictions['router_'+name]=predict_sharing(SharingFit(SharingRule(),b,{}),q)['mean']
            # LOOU coefficient prediction: early descriptors -> full-history coefficient labels.
            lookup={str(u):i for i,u in enumerate(full.groups)}
            labels=np.array([full.means[lookup[str(u)]] for u in early.groups])
            ridge=[];base=[]
            for i in range(len(early.groups)):
                mask=np.arange(len(early.groups))!=i
                ridge.append(ridge_predict(early.descriptors[mask],labels[mask],early.descriptors[i:i+1])[0])
                base.append(labels[mask].mean(0))
            ridge=np.array(ridge);base=np.array(base)
            ratio=np.mean((ridge-labels)**2,axis=0)/np.maximum(np.mean((base-labels)**2,axis=0),1e-12)
            raw_proxy=-np.log(d['rate']);latent=np.sum(bank.means[[list(bank.groups).index(str(u)) for u in d['groups']]]*
                np.column_stack((np.ones(len(d['y'])),d['x'][:,0],d['x'][:,0]**2)),axis=1)
            shifts={k:float(np.sqrt(np.mean((predictions['router_'+k]-predictions['router_early'])**2)))
                    for k in ('full','function_only','descriptor_only','calibration_only')}
            result=dict(fold=fold,fraction=frac,coordinate_cutoff=float(cut),health_cutoff=float(-cut),
                early_rows=len(early_rows['y']),query_units=len(np.unique(q['groups'])),
                scores={k:score(q['y'],p,q['groups']) for k,p in predictions.items()},
                oracle_best_donor_distance_percentile=float(np.mean(ranks)),nearest_unit_top1_match=float(np.mean(matches)),
                coefficient_loou_ridge_mse_ratio=ratio.tolist(),proxy_reliability=bank.evidence_reliability,
                raw_proxy_latent_correlation=float(np.corrcoef(raw_proxy,latent)[0,1]),
                refit_prediction_rms_shift=shifts)
            results['episodes'].append(result)
            prefix=f'f{fold}_q{frac}'
            arrays.update({prefix+'_'+k:v for k,v in predictions.items()})
            arrays[prefix+'_y']=q['y'];arrays[prefix+'_groups']=q['groups']
            arrays[prefix+'_donor_predictions']=matrix
            arrays[prefix+'_donor_distance']=np.array(distances);arrays[prefix+'_donor_loss']=np.array(losses)
            print('EPISODE',fold,frac,'own full/early/uniform/oracle',
                [round(result['scores'][k]['unit_mse'],5) for k in ('own_full_oracle','own_early_label_oracle','uniform_function_mean','oracle_best_single_donor')],flush=True)
    results['aggregate_unit_mse']={k:float(np.mean([e['scores'][k]['unit_mse'] for e in results['episodes']]))
                                  for k in results['episodes'][0]['scores']}
    results['complete']=True;results['successor_promoted']=False
    results['limits']=['six overlapping episodes, not independent replications','unit coefficients are estimated labels, not physical truth',
        'fixed linear probe failure does not prove all nonlinear routing impossible','own-full and best-donor are retrospective oracles',
        'refit uses donor later-support labels for diagnosis only; not an admissible early-support predictor']
    np.savez_compressed(out/'predictions.npz',**arrays)
    write(out/'results.json',results)
    print(json.dumps(results['aggregate_unit_mse'],indent=2),flush=True)


if __name__=='__main__':main()
