# PP-X v1.2 적응형 prior shrinkage 개발

기록: 박진서, 2026-09-10.

## 왜 바꿨나

v1.1은 prior 승인에 실패하면 exact matched MLP가 된다. 이는 PP로 인한
추가 손실을 막지만, MLP 자체가 강한 모델이라는 뜻은 아니다. ISU 250 mAh
retrospective ablation에서는 약한 prior가 test R² 0.555였지만 hard gate가
이를 버려 MLP R² 0.451로 후퇴했다.

## 검토한 대안

Ridge, ExtraTrees, histogram gradient boosting을 fallback portfolio 후보로
같은 train/validation에 비교했다.

- Stanford validation 최선 classical: ExtraTrees, RMSE 106.60
  (MLP 46.27)
- ISU validation 최선 classical: HistGradientBoosting, RMSE 2.28
  (MLP 계열 약 1.81)

따라서 약한 classical expert를 섞으면 구조만 복잡해지고 validation 성능이
낮아졌다. 이번 버전은 fallback 종류를 늘리지 않고 **prior 사용량을 연속적으로
줄이는 문제**를 직접 해결한다.

## v1.2 구조

validation evidence를 세 단계로 나눈다.

1. **Strong:** 상대 RMSE 2% 이상 개선, unit 60% 이상 승리,
   최악 unit 비율 1.10 이하, unit-bootstrap 95% 하한 양수  
   → validation-selected prior 후보를 100% 사용
2. **Weak:** pooled RMSE 방향이 양수이고 unit 과반에서 개선  
   → prior weight를 0–0.5로 연속 shrinkage
3. **None:** pooled 이득이 없거나 unit 과반 승리 실패  
   → weight 0, exact matched MLP

Weak weight는 상대 이득과 unit 승률로 계산한 뒤, **실제 혼합 예측의 모든
validation unit RMSE 비율이 1.10 이하**가 되도록 다시 줄인다. 따라서 full
prior 후보의 최악 unit이 나빠도 작은 prior 신호만 안전 envelope 안에서
남길 수 있다.

최종 예측은

`fallback + prior_weight × (prior_candidate − fallback)`

이다. weight 0은 수치적으로 exact fallback이다.

## Retrospective stress test

| Cohort | v1.1 hard gate R² | v1.2 R² | Always-on R² | 선택 |
|---|---:|---:|---:|---|
| Stanford | 0.034 | **0.034** | 0.034 | evidence none, weight 0 |
| ISU 250 mAh | 0.451 | **0.488** | 0.555 | weak, weight 0.213 |

ISU validation evidence:

- 상대 RMSE 이득 1.65%
- unit 승리 32/45
- full 후보 최악 unit 비율 6.21
- bootstrap 95% CI `[−0.145, 0.192]`
- 최초 제안 weight 0.455
- worst-unit 1.10 envelope 적용 후 weight **0.213**
- effective affine trust **0.0043**

즉 full prior의 위험을 그대로 받지 않고, v1.1보다 test R²를 0.037 높였다.
Stanford에서는 validation 방향 자체가 음수라 exact fallback을 유지했다.

## 증거 수준

이 정책과 0.5 cap은 이미 본 Stanford·ISU를 사용해 만든 **post-test
development**다. 두 고호트에서 나아졌다는 사실만으로 일반화를 주장하지
않는다. 새 미개봉 고호트에서 아래를 결과 전에 고정해야 한다.

- trust·architecture grid
- strong/weak/none 문턱
- weak cap 0.5
- worst-unit envelope 1.10
- split·endpoint·strict-tail cutoff

## 구현

- `src/pp_extrapolation/adaptive_shrinkage.py`
- `src/pp_extrapolation/presets.py::adaptive_shrinkage_policy_config`
- `experiments/ppx_v12_adaptive_shrinkage_development.py`
- `results/ppx_v12_adaptive_shrinkage_development/`
- `tests/test_adaptive_shrinkage.py`
