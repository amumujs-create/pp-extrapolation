"""Low-cost prediction-only validation on three opened external cohorts."""
from pathlib import Path
import itertools
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'results/ppx_three_cohort_low_cost_validation_v1'


def r2(y, p):
    return float(1 - np.sum((y-p)**2) / np.sum((y-y.mean())**2))


def rmse(y, p):
    return float(np.sqrt(np.mean((y-p)**2)))


def exact_sign_flip(values):
    values = np.asarray(values, float)
    observed = abs(values.mean())
    null = [np.mean(values*np.asarray(signs)) for signs in itertools.product((-1, 1), repeat=len(values))]
    return float(np.mean(np.abs(null) >= observed-1e-15))


def exact_sign(values):
    values = np.asarray(values, float); values = values[values != 0]
    n = len(values); k = max(int((values > 0).sum()), int((values < 0).sum()))
    return min(1., 2*sum(math.comb(n, j) for j in range(k, n+1))/2**n)


def bh(pvalues):
    order = np.argsort(pvalues); out = np.empty(len(pvalues)); running = 0.
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues)-rank)*pvalues[index]); out[index] = min(running, 1.)
    return out.tolist()


def load():
    base = ROOT/'results/two_success_cohorts_extrapolation_competitors_v2_nonnegative'
    for name, file in [('Alloy A','alloy_a_ensemble_predictions.npz'),
                       ('MultiStage RPT','multistage_rpt_ensemble_predictions.npz')]:
        z=np.load(base/file)
        yield name,z['truth'].astype(float),z['groups'].astype(str),z['PP_latest_successful'].astype(float),z['Engression'].astype(float),'CCMR trajectory executor'
    p=np.load(ROOT/'results/ncmapss_ds03_ccmr_v22/predictions.npz')
    e=np.load(ROOT/'results/ncmapss_ds03_equal_budget_v1/engression/predictions.npz')
    assert np.allclose(p['y'],e['y']) and np.array_equal(p['groups'].astype(str),e['groups'].astype(str))
    yield 'N-CMAPSS DS03',p['y'].astype(float),p['groups'].astype(str),p['prediction'].astype(float),e['prediction'].mean(0).astype(float),'prospectively selected direct fallback'


def main():
    records=[]
    for name,y,groups,ppx,eng,route in load():
        unit=[]
        for group in np.unique(groups):
            m=groups==group; pr=rmse(y[m],ppx[m]); er=rmse(y[m],eng[m])
            unit.append(dict(unit=str(group),n=int(m.sum()),ppx_rmse=pr,engression_rmse=er,
                             log_rmse_effect=float(np.log(max(er,1e-12)/max(pr,1e-12)))))
        effects=np.array([u['log_rmse_effect'] for u in unit])
        records.append(dict(cohort=name,route=route,rows=len(y),units=len(unit),
            ppx=dict(r2=r2(y,ppx),rmse=rmse(y,ppx)),engression=dict(r2=r2(y,eng),rmse=rmse(y,eng)),
            r2_difference=r2(y,ppx)-r2(y,eng),units_ppx_better=int((effects>0).sum()),
            mean_unit_log_rmse_effect=float(effects.mean()),geometric_rmse_ratio=float(np.exp(-effects.mean())),
            exact_unit_sign_p_two_sided=exact_sign(effects),exact_unit_sign_flip_mean_p_two_sided=exact_sign_flip(effects),
            unit_results=unit))
    q=bh([r['exact_unit_sign_flip_mean_p_two_sided'] for r in records])
    for r,v in zip(records,q): r['unit_sign_flip_bh_q']=v
    gaps=np.array([r['r2_difference'] for r in records])
    aggregate=dict(cohorts_ppx_higher_r2=int((gaps>0).sum()),cohorts=3,
        mean_r2_difference=float(gaps.mean()),cohort_sign_p_two_sided=exact_sign(gaps),
        exact_cohort_sign_flip_mean_p_two_sided=exact_sign_flip(gaps),
        note='n=3 cannot support a strong cross-cohort superiority or stability inference.')
    protocol=dict(status='retrospective prediction-only low-cost validation',training='none',
        cohorts=['Alloy A','MultiStage RPT','N-CMAPSS DS03'],comparator='Engression',
        caveats=['All outcomes were already opened before this aggregation.',
                 'Alloy/MultiStage use CCMR trajectory executors; DS03 uses the selected direct fallback.',
                 'This is a PP-X framework stress test, not one identical neural-network checkpoint.',
                 'Only DS03 route selection was prospective; predictive superiority failed there.',
                 'Cohort-level n=3 is too small for a positive generalization claim.'])
    OUT.mkdir(parents=True,exist_ok=False)
    (OUT/'results.json').write_text(json.dumps(dict(protocol=protocol,records=records,aggregate=aggregate),indent=2))
    lines=['# PP-X 저비용 3고호트 검증','',
        '새 학습 없이 저장된 평가 예측을 재집계했다. 결과를 이미 본 세 고호트이므로 retrospective external stress test다. Alloy/MultiStage의 CCMR 경로와 DS03 fallback을 PP-X framework의 실행 경로로 평가했으며 하나의 동일 신경망 검증은 아니다.','',
        '| 고호트 | PP-X 경로 | 개체 | PP-X R² | Engression R² | 차이 | PP-X 우위 개체 | PP-X/Eng RMSE 기하비 | unit exact p | BH q |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        lines.append(f"| {r['cohort']} | {r['route']} | {r['units']} | {r['ppx']['r2']:.4f} | {r['engression']['r2']:.4f} | {r['r2_difference']:+.4f} | {r['units_ppx_better']}/{r['units']} | {r['geometric_rmse_ratio']:.3f} | {r['exact_unit_sign_flip_mean_p_two_sided']:.4f} | {r['unit_sign_flip_bh_q']:.4f} |")
    lines += ['','## 판정','',
        f"PP-X가 pooled R²에서 Engression보다 높은 고호트는 {aggregate['cohorts_ppx_higher_r2']}/3이다. 세 고호트 평균 R² 차이는 {aggregate['mean_r2_difference']:+.4f}, cohort exact sign-flip p={aggregate['exact_cohort_sign_flip_mean_p_two_sided']:.4f}다.",'',
        '이 세 고호트는 최종 9-setting 개발 결과를 외부에서 그대로 재현하지 않는다. Alloy A에서는 Engression이 명확히 우세하고, MultiStage는 사실상 동률이며, DS03에서도 Engression이 우세하다. 따라서 **외부 세 고호트에서 PP-X가 Engression보다 평균 정확도 또는 변동성이 우월하다는 주장은 지지되지 않는다.**','',
        'MultiStage의 PP-X 경로는 persistence 대비 최악 개체 regret를 증가시키지 않았다는 기존 결과가 있지만, 이것은 Engression 대비 정확도 우월과 다른 주장이다. DS03는 부적합한 prior를 거절해 PP-X 후보 중 최선 경로를 선택했지만 Engression보다 낮았다.','',
        '## 논문에서의 사용','',
        '- 주 9개 개발 설정: 높은 성능과 낮은 시드 변동의 retrospective evidence.',
        '- 이 3개 외부 고호트: 평균 정확도 우월을 재현하지 못한 stress test와 적용 범위의 한계.',
        '- 사용할 수 있는 메시지: PP-X는 모든 외부 고호트에서 최고 정확도를 목표로 하지 않으며, 계약에 따라 prior를 거절하거나 위험 제한 경로를 선택한다.',
        '- 사용할 수 없는 메시지: PP-X가 외부 고호트에서도 Engression보다 덜 흔들리거나 항상 우월하다.',
        '- 세 고호트만으로 variance 비교는 정의하기 어렵고 n=3 exact 검정의 최소 양측 p도 0.25다.','',
        '## 비용 및 재현성','',
        '- 새 학습 비용: 0.',
        '- test prediction은 변경하지 않았다.',
        '- 개체 ID가 있는 동일 행에서 PP-X와 Engression RMSE를 비교했다.',
        '- exact unit 검정은 각 고호트 안에서만 수행하고 세 고호트의 개체를 하나의 독립 표본으로 합치지 않았다.','']
    (OUT/'RESULTS_KO.md').write_text('\n'.join(lines))
    print(json.dumps(dict(records=[{k:v for k,v in r.items() if k!='unit_results'} for r in records],aggregate=aggregate),indent=2))


if __name__=='__main__': main()
