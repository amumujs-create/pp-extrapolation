"""Prediction-only stability analysis of the canonical final PP-X portfolio."""
from pathlib import Path
import json
import sys
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'experiments'), str(ROOT.parent / 'ca-css-ncmapss')]
from final_modular_pp_evidence import build_datasets
from summarize_full_equal_candidate_budget import baseline, NAMES, r2

OUT = ROOT / 'results/final_ppx_seed_unit_stability_v1'
MODELS = ('ppx', 'engression', 'plain_mlp')


def metrics(y, groups, predictions):
    ensemble = predictions.mean(0)
    scores = np.array([r2(y, p) for p in predictions])
    units = []
    for unit in np.unique(groups):
        mask = groups == unit
        error = ensemble[mask] - y[mask]
        units.append(dict(unit=str(unit), n=int(mask.sum()),
                          rmse=float(np.sqrt(np.mean(error ** 2))),
                          seed_rmse=[float(np.sqrt(np.mean((p[mask] - y[mask]) ** 2))) for p in predictions]))
    return dict(ensemble_r2=r2(y, ensemble), seed_r2=scores.tolist(),
                seed_sd_r2=float(scores.std(ddof=1)), worst_seed_r2=float(scores.min()),
                positive_seeds=int((scores > 0).sum()), units=units)


def main():
    OUT.mkdir(parents=True, exist_ok=False)
    protocol = dict(model='canonical final paper PP-X; not per-test selection of later candidates',
                    seeds=list(range(42, 47)), scope='nine retrospective main settings',
                    analysis='existing predictions only; no training or hyperparameter selection',
                    source='final_modular_pp_evidence.build_datasets and 30-candidate baselines',
                    uncertainty='descriptive analysis; no new significance claims',
                    limits=['Baseline candidate counts are matched; historical PP-X development budgets are not.',
                            'Seed variation is conditional on fixed data and selected hyperparameters.',
                            'Domain R2 dispersion is not physical-unit risk or universal robustness.'])
    (OUT / 'protocol.json').write_text(json.dumps(protocol, indent=2))
    print('Loading canonical five-seed PP-X predictions', flush=True)
    data = build_datasets()
    records = []
    for name, values in data.items():
        y, groups, pp = map(np.asarray, values[:3])
        groups = groups.astype(str)
        predictions = {'ppx': pp}
        for model in MODELS[1:]:
            by, bg, prediction = baseline(NAMES[name], model)
            if np.shape(by) != y.shape or not np.allclose(by, y):
                raise ValueError(f'Target mismatch: {name}/{model}')
            if not np.array_equal(bg, groups):
                raise ValueError(f'Group mismatch: {name}/{model}')
            predictions[model] = prediction
        for model, p in predictions.items():
            if p.shape != (5, len(y)) or not np.isfinite(p).all():
                raise ValueError(f'Invalid predictions: {name}/{model}')
        result = dict(setting=name, rows=len(y), models={m: metrics(y, groups, p) for m, p in predictions.items()}, paired={})
        for model in MODELS[1:]:
            pu = result['models']['ppx']['units']
            bu = result['models'][model]['units']
            ratios = np.array([a['rmse'] / max(b['rmse'], 1e-12) for a, b in zip(pu, bu)])
            result['paired'][model] = dict(unit_count=len(ratios), units_ppx_better=int((ratios < 1).sum()),
                median_unit_rmse_ratio=float(np.median(ratios)), q90_unit_rmse_ratio=float(np.quantile(ratios, .9)),
                worst_unit_rmse_ratio=float(ratios.max()), units_ratio_above_2=int((ratios > 2).sum()),
                mean_unit_log_base_over_ppx=float(np.mean(-np.log(np.maximum(ratios, 1e-12)))),
                unit_ratios=[dict(unit=a['unit'], n=a['n'], ratio=float(r)) for a, r in zip(pu, ratios)])
        records.append(result)
        print('SCORED', name, {m: dict(r2=v['ensemble_r2'], sd=v['seed_sd_r2'], worst=v['worst_seed_r2']) for m, v in result['models'].items()}, flush=True)
        (OUT / 'progress.json').write_text(json.dumps(records, indent=2))
    summary = {}
    for model in MODELS:
        vals = [r['models'][model] for r in records]
        es = np.array([v['ensemble_r2'] for v in vals])
        summary[model] = dict(mean_setting_r2=float(es.mean()), domain_sd_ddof1=float(es.std(ddof=1)),
            domain_mad=float(np.median(np.abs(es - np.median(es)))), minimum_setting_r2=float(es.min()),
            mean_within_setting_seed_sd=float(np.mean([v['seed_sd_r2'] for v in vals])),
            median_within_setting_seed_sd=float(np.median([v['seed_sd_r2'] for v in vals])),
            minimum_seed_r2=float(min(v['worst_seed_r2'] for v in vals)),
            positive_seed_runs=sum(v['positive_seeds'] for v in vals),
            settings_all_seeds_positive=sum(v['positive_seeds'] == 5 for v in vals))
    comparisons = {}
    for model in MODELS[1:]:
        pairs = [r['paired'][model] for r in records]
        comparisons[model] = dict(
            settings_lower_seed_sd=sum(r['models']['ppx']['seed_sd_r2'] < r['models'][model]['seed_sd_r2'] for r in records),
            settings_higher_worst_seed=sum(r['models']['ppx']['worst_seed_r2'] > r['models'][model]['worst_seed_r2'] for r in records),
            units_ppx_better=sum(p['units_ppx_better'] for p in pairs), units=sum(p['unit_count'] for p in pairs),
            units_ratio_above_2=sum(p['units_ratio_above_2'] for p in pairs),
            equal_setting_mean_unit_log_base_over_ppx=float(np.mean([p['mean_unit_log_base_over_ppx'] for p in pairs])))
    payload = dict(protocol=protocol, records=records, summary=summary, comparisons=comparisons)
    (OUT / 'results.json').write_text(json.dumps(payload, indent=2))
    lines = ['# 최종 PP-X 시드·개체 안정성 분석', '',
        '기존 최종 5-seed 예측 재집계. 새 학습 없음. 9개 후향적 주 평가 설정이며 모든 역사적 코호트의 통합 결과가 아니다.', '',
        '## 설정별 결과', '',
        '| 설정 | PP-X R² | Engression R² | MLP R² | PP-X seed SD | Engression seed SD | MLP seed SD | PP-X 최악 seed |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for r in records:
        m = r['models']
        nums = [m[k]['ensemble_r2'] for k in MODELS] + [m[k]['seed_sd_r2'] for k in MODELS] + [m['ppx']['worst_seed_r2']]
        lines.append('| ' + r['setting'] + ' | ' + ' | '.join(f'{v:.4f}' for v in nums) + ' |')
    lines += ['', '## 요약', '', '| 모델 | 평균 R² | 설정 간 MAD | 평균 seed SD | 최악 seed R² | 양수 seed /45 | 모든 seed 양수 설정 /9 |', '|---|---:|---:|---:|---:|---:|---:|']
    for m, v in summary.items():
        lines.append(f"| {m} | {v['mean_setting_r2']:.4f} | {v['domain_mad']:.4f} | {v['mean_within_setting_seed_sd']:.4f} | {v['minimum_seed_r2']:.4f} | {v['positive_seed_runs']} | {v['settings_all_seeds_positive']} |")
    lines += ['', '## 개체별 PP-X / 비교모델 RMSE 비율', '',
              '1 미만은 PP-X 우위, 2 초과는 비교모델의 두 배를 넘는 오차다. 2는 기술적 기준이며 사전 안전 임계값이 아니다. 한 행만 있는 개체도 포함하고 행 수를 JSON에 기록했다.', '',
              '| 설정 | 비교모델 | PP-X 우위 개체 | 비율 중앙값 | 비율 90% 분위수 | 최악 비율 |', '|---|---|---:|---:|---:|---:|']
    for r in records:
        for m, p in r['paired'].items():
            lines.append(f"| {r['setting']} | {m} | {p['units_ppx_better']}/{p['unit_count']} | {p['median_unit_rmse_ratio']:.3f} | {p['q90_unit_rmse_ratio']:.3f} | {p['worst_unit_rmse_ratio']:.3f} |")
    lines += ['', '## 직접적인 판단', '']
    for m, c in comparisons.items():
        lines.append(f"- {m} 대비 PP-X seed SD가 작은 설정은 {c['settings_lower_seed_sd']}/9, 최악 seed R²가 높은 설정은 {c['settings_higher_worst_seed']}/9이다.")
        lines.append(f"- {m} 대비 개체 RMSE 우위는 {c['units_ppx_better']}/{c['units']}이며, PP-X RMSE가 두 배를 넘는 개체는 {c['units_ratio_above_2']}개다. 개체들을 독립적인 다중 도메인 확증 표본으로 합산 검정하지 않는다.")
    lines += ['', '## 범위와 추가 실행 판단', '',
        '- 이번 결과는 canonical final paper PP-X의 9-setting prediction portfolio 기준이다. 새로운 후보의 설정별 최고 점수를 섞지 않았다.',
        '- 기존 확장 registry의 XJTU, FEMTO, milling은 서로 다른 프로토콜의 개발 tier다. 이번 9-setting 표와 경쟁모델 승수를 합치지 않는다.',
        '- FEMTO 최종 경로는 기존 기록상 ensemble R²만 양수이고 모든 개별 seed가 음수다. 전체 코호트 시드 안정성이라는 주장은 불가능하다.',
        '- DS03 최종 선택은 fallback이며 Engression보다 낮았다. 기존 결과가 있으므로 단지 성능 확인을 위해 재학습하지 않는다.',
        '- 초기 v1으로만 평가한 코호트는 현재 최종 선택 절차의 평가를 대체하지 않는다. 재실행한다면 이미 열린 데이터의 후향적 전이 평가다.',
        '- 모든 코호트를 합친 주장에는 최종 버전별 artifact manifest와 동등한 baseline/평가 행을 추가로 연결해야 한다. 이번에는 그 공백을 숨겨 새 점수를 만들지 않았다.',
        '- 시드 SD는 데이터와 선택된 hyperparameter를 고정한 optimizer variation이다. 독립 cohort 안정성, 안전성, prior validity의 증명이 아니다.',
        '- 학습 데이터 규모와 R²의 target 분산 차이를 고려해야 한다. 평균 SD와 개체 상대오차는 다른 질문에 답한다.',
        '- 본 분석은 기술통계다. 기존 Holm 보정 후 분산 우월성 미확증이라는 결론을 변경하지 않는다.',
        '- 새 학습은 시행하지 않았다. 우선 본 결과의 범위로 원고 주장을 정하고, 범위를 넓힐 때만 누락 cohort의 고정 버전 평가를 실행한다.', '']
    (OUT / 'RESULTS_KO.md').write_text('\n'.join(lines))
    print('SUMMARY', json.dumps(summary), flush=True)
    print('COMPARISONS', json.dumps(comparisons), flush=True)


if __name__ == '__main__':
    main()
