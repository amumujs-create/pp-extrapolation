#!/usr/bin/env python3
"""Unit-paired inference for the completed PP versus matched plain-MLP ablation."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'results/plain_mlp_ablation_v1/results.json'
OUT=ROOT/'results/plain_mlp_paired_inference_v1'

def exact_or_mc_signflip(delta,rng,n_mc=200000):
    n=len(delta); observed=abs(float(np.mean(delta)))
    if n<=20:
        masks=np.arange(1<<n,dtype=np.uint64)[:,None]
        bits=((masks>>np.arange(n,dtype=np.uint64))&1).astype(float)
        signs=2*bits-1
        vals=np.abs(signs@delta/n)
        return float(np.mean(vals>=observed-1e-15)),int(len(vals))
    signs=rng.choice((-1.,1.),size=(n_mc,n))
    return float((1+np.sum(np.abs(signs@delta/n)>=observed))/(n_mc+1)),n_mc

def main():
    src=json.load(open(SOURCE));rng=np.random.default_rng(20260908);out={}
    for name,row in src['datasets'].items():
        units=sorted(row['runs'][0]['plain_metrics']['per_unit'])
        plain=np.array([[run['plain_metrics']['per_unit'][u]['rmse'] for u in units] for run in row['runs']]).mean(0)
        pp=np.array([[run['pp_metrics']['per_unit'][u]['rmse'] for u in units] for run in row['runs']]).mean(0)
        # Positive values mean PP reduces RMSE.
        delta=plain-pp
        boot=np.mean(rng.choice(delta,size=(50000,len(delta)),replace=True),axis=1)
        p,nperm=exact_or_mc_signflip(delta,rng)
        out[name]={'n_units':len(units),'mean_rmse_reduction_pp_vs_plain':float(delta.mean()),
                   'median_rmse_reduction':float(np.median(delta)),
                   'ci95_unit_bootstrap':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
                   'two_sided_paired_signflip_p':p,'permutations':nperm,
                   'pp_unit_wins':int(np.sum(delta>0)),'unit_ids':units,
                   'unit_rmse_reduction':delta.tolist()}
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'results.json').write_text(json.dumps({'experiment':'plain_mlp_paired_inference_v1',
      'estimand':'mean across physical units of five-seed-average RMSE(plain)-RMSE(PP)',
      'bootstrap_resamples':50000,'datasets':out},indent=2)+'\n')
    lines=['# PP 대 matched plain MLP의 unit-paired 통계','',
      '각 physical unit에서 5개 seed RMSE를 먼저 평균한 뒤, unit을 표본 단위로 사용했다. 양의 차이는 PP의 RMSE 감소를 뜻한다.', '',
      '| 데이터셋 | unit 수 | PP 승 unit | 평균 RMSE 감소 | unit-bootstrap 95% CI | paired sign-flip p |','|---|---:|---:|---:|---:|---:|']
    for name,r in out.items():
      lines.append(f"| {name.upper()} | {r['n_units']} | {r['pp_unit_wins']}/{r['n_units']} | {r['mean_rmse_reduction_pp_vs_plain']:.3f} | [{r['ci95_unit_bootstrap'][0]:.3f}, {r['ci95_unit_bootstrap'][1]:.3f}] | {r['two_sided_paired_signflip_p']:.4g} |")
    lines += ['', '이 검정은 행 단위 pseudo-replication을 피하지만 데이터셋 수가 아니라 unit 수에 대한 조건부 추론이다. 세 데이터셋 모두 개발에 사용되었으므로 외부 일반화의 확증 검정은 아니다.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps(out,indent=2))
if __name__=='__main__':main()
