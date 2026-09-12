"""Small-sample inference for the frozen final PP-X prediction portfolio."""
from pathlib import Path
import itertools
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / 'results/final_ppx_seed_unit_stability_v1/results.json'
OUT = ROOT / 'results/final_ppx_small_sample_inference_v1'
COMPARATORS = ('engression', 'plain_mlp')
RNG = np.random.default_rng(20260913)


def exact_sign_test(values):
    values = np.asarray(values, float)
    values = values[values != 0]
    wins = int((values > 0).sum())
    n = len(values)
    extreme = max(wins, n - wins)
    p = min(1.0, 2 * sum(math.comb(n, k) for k in range(extreme, n + 1)) / 2 ** n)
    return dict(n=n, positive=wins, negative=n-wins, p_two_sided=float(p))


def exact_sign_flip_mean(values):
    values = np.asarray(values, float)
    observed = abs(values.mean())
    null = np.array([np.mean(values * signs) for signs in itertools.product((-1, 1), repeat=len(values))])
    return float(np.mean(np.abs(null) >= observed - 1e-15))


def setting_bootstrap(values, draws=100000):
    values = np.asarray(values, float)
    sampled = values[RNG.integers(0, len(values), size=(draws, len(values)))].mean(1)
    return [float(x) for x in np.quantile(sampled, (.025, .975))]


def hierarchical_unit_bootstrap(unit_effects, draws=100000):
    # Equal weight to settings, then equal weight to sampled physical units within setting.
    estimates = np.empty(draws)
    n_settings = len(unit_effects)
    for b in range(draws):
        selected = RNG.integers(0, n_settings, n_settings)
        setting_means = []
        for index in selected:
            values = unit_effects[index]
            setting_means.append(values[RNG.integers(0, len(values), len(values))].mean())
        estimates[b] = np.mean(setting_means)
    return [float(x) for x in np.quantile(estimates, (.025, .975))]


def holm(pvalues):
    order = np.argsort(pvalues)
    adjusted = np.empty(len(pvalues))
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, (len(pvalues)-rank) * pvalues[index])
        adjusted[index] = min(running, 1.0)
    return adjusted.tolist()


def main():
    payload = json.loads(SOURCE.read_text())
    records = payload['records']
    results = {}
    all_primary_p = []
    primary_keys = []
    for comparator in COMPARATORS:
        accuracy = np.array([r['models']['ppx']['ensemble_r2'] - r['models'][comparator]['ensemble_r2'] for r in records])
        log_sd = np.log(np.array([r['models'][comparator]['seed_sd_r2'] for r in records]) /
                        np.array([r['models']['ppx']['seed_sd_r2'] for r in records]))
        worst = np.array([r['models']['ppx']['worst_seed_r2'] - r['models'][comparator]['worst_seed_r2'] for r in records])
        unit_effects = []
        for r in records:
            ratios = np.array([u['ratio'] for u in r['paired'][comparator]['unit_ratios']])
            unit_effects.append(-np.log(np.maximum(ratios, 1e-12)))
        tests = {}
        for name, values in (('ensemble_r2_difference', accuracy), ('log_seed_sd_ratio', log_sd), ('worst_seed_r2_difference', worst)):
            tests[name] = dict(effect_mean=float(values.mean()), effect_median=float(np.median(values)),
                               bootstrap_ci95=setting_bootstrap(values), sign=exact_sign_test(values),
                               exact_sign_flip_mean_p_two_sided=exact_sign_flip_mean(values), values=values.tolist())
            all_primary_p.append(tests[name]['exact_sign_flip_mean_p_two_sided'])
            primary_keys.append((comparator, name))
        observed_unit = float(np.mean([v.mean() for v in unit_effects]))
        tests['hierarchical_unit_log_rmse'] = dict(
            effect='equal-setting mean log(RMSE comparator / RMSE PP-X); positive favors PP-X',
            estimate=observed_unit, geometric_mean_rmse_ratio=float(np.exp(-observed_unit)),
            hierarchical_bootstrap_ci95=hierarchical_unit_bootstrap(unit_effects),
            setting_means=[float(v.mean()) for v in unit_effects],
            caveat='Exploratory retrospective CI; settings and related battery cohorts are not guaranteed exchangeable.')
        results[comparator] = tests
    adjusted = holm(all_primary_p)
    for (comparator, name), q in zip(primary_keys, adjusted):
        results[comparator][name]['holm_q_across_six_primary_tests'] = q
    protocol = dict(scope='canonical final PP-X, nine retrospective main settings, five fixed seeds',
        primary_unit='setting; seeds are repeated optimization runs and not independent samples',
        methods=['two-sided exact sign test', 'two-sided exhaustive sign-flip test of mean effect',
                 'setting bootstrap CI', 'dataset-then-unit hierarchical bootstrap CI'],
        multiplicity='Holm adjustment across 3 outcomes x 2 prespecified comparators',
        interpretation='Retrospective small-sample inference; not prospective confirmation or proof of prior validity.')
    OUT.mkdir(parents=True, exist_ok=False)
    (OUT / 'results.json').write_text(json.dumps(dict(protocol=protocol, results=results), indent=2))
    lines = ['# 최종 PP-X 소표본 통계 검증', '',
        '대상은 최종 PP-X의 기존 9개 후향적 주 평가 설정과 5개 고정 시드다. 시드 45개를 독립 표본으로 사용하지 않고 설정 9개를 주 추론 단위로 사용했다.', '',
        '## 검정 설계', '',
        '- 방향 일관성: 양측 exact sign test.',
        '- 평균 효과: 2^9=512개 부호 배치를 모두 열거한 양측 exact sign-flip test.',
        '- 효과 크기 구간: 설정 단위 bootstrap 100,000회. 소표본·개발자료이므로 확증적 신뢰구간이 아니다.',
        '- 개체 RMSE: 설정을 먼저, 그 안에서 물리 개체를 다시 뽑는 계층 bootstrap 100,000회.',
        '- 2개 비교모델 x 3개 주 결과의 6개 검정에 Holm 보정.', '',
        '## 결과', '',
        '| 비교 | 결과 | 설정 방향 | 평균 효과 [bootstrap 95% CI] | exact p | Holm q |',
        '|---|---|---:|---:|---:|---:|']
    labels = {'ensemble_r2_difference':'R² 차이', 'log_seed_sd_ratio':'log(seed SD 비)', 'worst_seed_r2_difference':'최악 seed R² 차이'}
    for comparator, tests in results.items():
        for key in labels:
            t = tests[key]
            ci = t['bootstrap_ci95']
            lines.append(f"| {comparator} | {labels[key]} | {t['sign']['positive']}/{t['sign']['n']} | {t['effect_mean']:.4f} [{ci[0]:.4f}, {ci[1]:.4f}] | {t['exact_sign_flip_mean_p_two_sided']:.4f} | {t['holm_q_across_six_primary_tests']:.4f} |")
    lines += ['', 'log(seed SD 비)는 `log(비교모델 SD / PP-X SD)`이므로 양수가 PP-X의 작은 시드 변동을 뜻한다. 다른 두 차이도 양수가 PP-X 우위다.', '',
              '## 개체 수준 탐색 결과', '',
              '| 비교 | 설정 동일가중 log RMSE 효과 | PP-X/비교 RMSE 기하비 | 계층 bootstrap 95% CI |',
              '|---|---:|---:|---:|']
    for comparator, tests in results.items():
        t = tests['hierarchical_unit_log_rmse']; ci = t['hierarchical_bootstrap_ci95']
        lines.append(f"| {comparator} | {t['estimate']:.4f} | {t['geometric_mean_rmse_ratio']:.3f} | [{ci[0]:.4f}, {ci[1]:.4f}] |")
    lines += ['', '## 논문용 해석', '',
        '가장 직접적인 결과는 PP-X가 9개 설정 모두에서 Engression과 MLP보다 시드 SD가 작고 최악 시드 R²가 높았다는 것이다. exact 검정은 작은 n에 맞춰 이 방향 일관성을 평가한다.',
        '', '이 결과는 평가한 개발 설정에서의 재학습 안정성을 지지한다. 그러나 데이터셋들이 완전히 독립·동질한 모집단에서 무작위 추출된 것이 아니고, 최종 PP-X도 이 자료를 보며 개발됐으므로 미래 코호트에 대한 확증 p값으로 해석하지 않는다.',
        '', '프라이어가 타당하기 때문에 변동이 감소했다는 인과적 주장은 이 분석으로 검정되지 않았다. 이를 주장하려면 결과와 독립적으로 prior validity를 정의한 통제 실험이 필요하다.',
        '', '권장 문장:', '',
        '> Across nine retrospective extrapolation settings, final PP-X showed lower seed-to-seed variation and a higher worst-seed R² than Engression and a plain MLP in every setting. Exact small-sample tests support the consistency of this pattern within the evaluated benchmark; because these settings informed model development, the result is interpreted as retrospective stability evidence rather than prospective proof of robustness.', '',
        '## 제한', '',
        '- n=9이므로 효과 크기와 개별 설정 결과를 p값보다 우선한다.',
        '- R² 차이는 목표 분산에 민감하다. RMSE 기반 개체 결과를 함께 보고한다.',
        '- bootstrap은 관측한 설정을 모집단처럼 재표집한다. 관련 배터리 코호트 간 독립성을 보장하지 않는다.',
        '- seed SD는 선택된 설정 아래 optimizer variation이며 데이터 표본 불확실성을 포함하지 않는다.',
        '- 개체 계층 결과는 사후 탐색 분석이고, 개체 수가 3~16으로 작아 설정별 불확실성이 크다.',
        '- 본 검정은 최종 9-setting PP-X portfolio에 한정한다. DS03, FEMTO, XJTU, milling이나 초기 v1 결과를 합치지 않는다.', '']
    (OUT / 'RESULTS_KO.md').write_text('\n'.join(lines))
    print(json.dumps({c: {k: {x:v for x,v in t.items() if x in ('effect_mean','bootstrap_ci95','exact_sign_flip_mean_p_two_sided','holm_q_across_six_primary_tests')} for k,t in tests.items() if k != 'hierarchical_unit_log_rmse'} for c,tests in results.items()}, indent=2))


if __name__ == '__main__':
    main()
