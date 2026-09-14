# PP-X executor ablation 및 통계 감사

작성자: 박진서  
기준 산출물: `results/ppx_final_ablation_statistics_v1/`  
Prior: PP-X Final (`known_boundary` → BQ, 아니면 affine). FEMTO abstention은 `v1_declared` 보관.

## 범위

PP-X의 optional executor인 bounded/unbounded residual, support-adaptive
dual-scale, causal/multiscale history, support-distance decay, regime
transport를 점검했다. 총 7개 데이터셋(Sunwoda, RWTH, MICH, NASA battery,
N-CMAPSS, MATR batch 2, HUST)의 matched on/off 예측을 사용했다.

모든 executor를 모든 데이터셋에 강제로 적용하지 않았다. PP-X의 typed
contract상 해당 prior family에서 정의되는 executor만 비교했다. 데이터셋
내에서는 동일 test row, unit, seeds 42–46을 유지했다.

## 통계 규칙

- 주 효과: prediction-ensemble의 paired unit RMSE 감소
- 불확실성: 물리 unit bootstrap 50,000회, 95% CI
- 검정: 양측 exact sign-flip
- 다중 비교: 25개 component-dataset 비교에 Benjamini–Hochberg 보정
- seed 검정: seeds 5개의 exact sign-flip을 보조 지표로만 사용

## 핵심 결과

| Executor | 데이터셋 | ΔR² | unit 통계 | 판정 |
|---|---|---:|---|---|
| Bounded vs unbounded | Sunwoda | +0.221 | CI [+0.074,+0.225], q=.033 | 개선 |
| Bounded vs unbounded | RWTH | +0.090 | CI [+0.013,+0.332], q=.156 | 방향 양성, BH 미통과 |
| Bounded vs unbounded | MICH | −0.291 | CI [−0.074,−0.031], q=.033 | 악화 |
| Dual-scale vs fixed | Sunwoda | −0.005 | CI 0 포함 | 차이 없음 |
| Dual-scale vs fixed | RWTH | −0.037 | CI [−0.084,−0.042], q=.022 | 악화 |
| Dual-scale vs fixed | MICH | +0.283 | CI [+0.029,+0.087], q=.033 | 개선 |
| Full history vs current | Sunwoda | +0.350 | CI [+0.154,+0.263], q=.020 | 개선 |
| Full history vs current | RWTH | +1.256 | CI [+1.036,+1.243], q=.022 | 개선 |
| Full history vs current | MICH | −0.204 | CI [−0.051,−0.015], q=.042 | 악화 |
| Selected history | NASA battery | +0.012 | 2/4 units, CI 0 포함 | 방향성만 |
| Selected history | N-CMAPSS | +0.009 | 3/3 units, n=3 | 방향성만 |
| Distance decay, transport off | MATR batch 2 | +0.002 | CI 0 포함, q=.277 | 단독 기각 |
| Decay×transport | MATR batch 2 | +0.042 | CI 0 포함 | 상호작용 방향성 |
| Regime transport | HUST | +0.128 | 15/16 units, q=.003 | 강한 근거 |
| Regime transport | MATR batch 2 | +0.187 | 9/9 units, q=.020 | 강한 근거 |

## 결론

모듈을 전역 기본값으로 모두 켜는 주장은 기각한다. Bounded residual,
dual-scale, full history는 데이터셋에 따라 부호가 바뀐다. Distance decay의
단독 정확도 효과도 지지되지 않는다. 반면 regime transport는 적용 가능한 두
데이터셋에서 일관된 unit-level 근거가 있다.

따라서 현재 증거는 PP-X의 설계 원칙인 “contract가 허용한 executor만
validation evidence로 승인하고, 실패하면 fallback”을 지지한다. 이 결과는
retrospective development split에 대한 것이며 untouched cohort 확증을
대체하지 않는다.
