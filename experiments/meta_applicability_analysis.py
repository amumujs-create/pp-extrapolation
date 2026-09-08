#!/usr/bin/env python3
"""Leave-one-domain-out applicability analysis from pre-outcome descriptors."""
from __future__ import annotations
import json,itertools
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import balanced_accuracy_score

ROOT=Path(__file__).resolve().parents[1];SRC=ROOT/'results/cross_domain_mechanism_v1/results.json';OUT=ROOT/'results/meta_applicability_v1'
FEATURES=['normalized_horizon','trajectory_heterogeneity','degradation_snr','curvature_gain']

def bacc_score(y,p):
    vals=[]
    for c in (0,1):
        m=y==c
        if m.any(): vals.append(float(np.mean(p[m]==c)))
    return float(np.mean(vals))

def transform(x):
    z=x.copy();z[:,0]=np.log1p(z[:,0]);z[:,1]=np.log1p(z[:,1]);z[:,2]=np.log1p(z[:,2]);return z

def standardize(train,test):
    mu=train.mean(0);sd=train.std(0);sd[sd<1e-9]=1;return (train-mu)/sd,(test-mu)/sd

def ridge_lodo(x,y):
    pred=[];alphas=[];grid=[.01,.1,1.,10.,100.]
    for i in range(len(y)):
        keep=np.arange(len(y))!=i;xt,xi=standardize(x[keep],x[i:i+1]);yt=y[keep]
        # Inner leave-one-domain-out selection.
        losses=[]
        for alpha in grid:
            pp=[];tt=[]
            for j in range(len(yt)):
                kk=np.arange(len(yt))!=j;a,b=standardize(x[keep][kk],x[keep][j:j+1]);m=Ridge(alpha=alpha).fit(a,yt[kk]);pp.append(m.predict(b)[0]);tt.append(yt[j])
            losses.append(np.mean((np.array(pp)-tt)**2))
        alpha=grid[int(np.argmin(losses))];alphas.append(alpha);pred.append(Ridge(alpha=alpha).fit(xt,yt).predict(xi)[0])
    return np.array(pred),alphas

def stump_lodo(x,label):
    pred=[];rules=[]
    for i in range(len(label)):
        keep=np.arange(len(label))!=i;best=None
        for j,name in enumerate(FEATURES):
            vals=np.unique(x[keep,j]);cuts=(vals[:-1]+vals[1:])/2
            for cut in cuts:
                for direction in (1,-1):
                    p=((x[keep,j]-cut)*direction>0).astype(int);score=bacc_score(label[keep],p)
                    cand=(score,-j,-abs(cut),j,cut,direction)
                    if best is None or cand>best:best=cand
        _,_,_,j,cut,direction=best;pred.append(int((x[i,j]-cut)*direction>0));rules.append({'feature':FEATURES[j],'threshold':float(cut),'positive_when':'above' if direction==1 else 'below'})
    return np.array(pred),rules

def main():
    d=json.load(open(SRC))['datasets'];names=list(d);raw=np.array([[d[n]['descriptors'][f] for f in FEATURES] for n in names],float);x=transform(raw);gain=np.array([d[n]['pp_gain_r2'] for n in names]);label=(gain>0).astype(int)
    corr={}
    for j,f in enumerate(FEATURES):
        rho,p=spearmanr(x[:,j],gain);corr[f]={'spearman_rho':float(rho),'p':float(p)}
    pred,alphas=ridge_lodo(x,gain);rho=float(spearmanr(pred,gain).statistic);mae=float(np.mean(np.abs(pred-gain)))
    gate,rules=stump_lodo(x,label);bacc=bacc_score(label,gate);acc=float(np.mean(label==gate))
    rng=np.random.default_rng(20260908);rhos=[];baccs=[]
    # Exploratory domain count is small; 199 deterministic Monte-Carlo
    # permutations provide an honest coarse null check without excessive refits.
    for _ in range(199):
        yp=rng.permutation(gain);rp,_=ridge_lodo(x,yp);rhos.append(spearmanr(rp,yp).statistic)
        lp=rng.permutation(label);gp,_=stump_lodo(x,lp);baccs.append(bacc_score(lp,gp))
    pr=float((1+np.sum(np.abs(rhos)>=abs(rho)))/(len(rhos)+1));pg=float((1+np.sum(np.array(baccs)>=bacc))/(len(baccs)+1))
    rows=[]
    for i,n in enumerate(names):rows.append({'dataset':n,'gain_r2':float(gain[i]),'pp_helped':bool(label[i]),'predicted_gain_lodo':float(pred[i]),'gate_prediction':bool(gate[i]),'correct':bool(gate[i]==label[i]),'fold_rule':rules[i],**{f:float(raw[i,j]) for j,f in enumerate(FEATURES)}})
    result={'n_domains':len(names),'feature_definition':'train labels + test covariates, no test outcomes','univariate':corr,
      'nested_lodo_ridge':{'spearman':rho,'permutation_p':pr,'mae_gain_r2':mae,'selected_alphas':alphas},
      'nested_lodo_stump_gate':{'accuracy':acc,'balanced_accuracy':bacc,'permutation_p':pg,'predicted_positive':int(gate.sum())},'domains':rows}
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# PP applicability meta-analysis','',f"총 {len(names)}개 도메인에서 test outcome을 사용하지 않는 네 descriptor로 matched backbone PP의 plain-MLP 대비 R² gain을 leave-one-domain-out 예측했다.",'',
      '| 분석 | 결과 | permutation p |','|---|---:|---:|',f'| Nested LODO ridge: 예측 gain과 실제 gain Spearman | {rho:.3f} | {pr:.4f} |',f'| Nested LODO one-variable stump gate: balanced accuracy | {bacc:.3f} | {pg:.4f} |','',
      '| dataset | actual ΔR² | gate | correct | fold-selected variable |','|---|---:|---:|---:|---|']
    for r in rows:lines.append(f"| {r['dataset']} | {r['gain_r2']:+.3f} | {int(r['gate_prediction'])} | {int(r['correct'])} | {r['fold_rule']['feature']} {r['fold_rule']['positive_when']} {r['fold_rule']['threshold']:.3g} |")
    lines += ['', '이 결과는 도메인 수가 12개인 탐색적 메타분석이다. LODO는 각 held-out domain의 outcome을 rule fitting에 쓰지 않지만, descriptor 정의 자체가 이 프로젝트의 개발 과정에서 정해졌으므로 외부 확증은 아니다.']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
