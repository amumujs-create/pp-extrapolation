# CCMR v1.7 Concrete 재료-shift 봉인 결과

## 판정

두 번째이자 마지막 counted sealed cohort는 완료됐지만 **성능 성공은
아니다**. CCMR v1.7이 validation에서 최소 개선폭을 충족하지 못해 test
전체를 보정 전에 거부했고, exact persistence로 안전 복귀했다.

- train: RH-1 HPC Lab 1--3, 40 specimens, 75,819 origins
- validation: RH-1 HPC Lab 4, 13 specimens, 14,260 origins
- test: RU-1 UHPC Lab 5--7, 38 specimens, 72,949 origins
- train 진행도 최대 30%, test 최소 75%, strict OOD 100%
- 공개 원본: DOI `10.25835/l9n63a7u`, ODbL 1.0, ZIP 총 25.1 MB

## Validation에서 거부된 이유

causal gate는 validation 행의 17.07%에서만 보정을 허용했다. 위험은
낮았지만 개선폭이 사전 기준 0.5%보다 작았다.

- pooled RMSE 개선: 0.188%
- unit-macro RMSE 개선: 0.260%
- raw unit regret mean/CVaR20/max:
  -0.546% / +0.070% / +0.210%

따라서 frozen v1.7의 global veto가 발동했다. 이는 test를 본 뒤 내린
결정이 아니다.

## 봉인 test 결과

- Persistence R²: 0.89146, RMSE: 0.00804094
- 배포 CCMR v1.7 R²: 0.89146, RMSE: 0.00804094
- pooled/macro 개선: 0% / 0%
- raw unit regret mean/CVaR20/max: 0% / 0% / 0%
- deployed correction coverage: 0%
- exact fallback replay error: 0

후보 보정은 test 행의 23.23%에서 causal gate를 통과했지만, validation
veto 때문에 배포 예측에는 하나도 사용되지 않았다. 따라서 이 자료는
“재료가 HPC에서 UHPC로 바뀌어도 성능이 개선된다”는 증거가 아니라,
“불충분한 개선을 자동으로 감지해 대규모 condition shift에서 baseline
손상을 막는다”는 강건성 증거다.

## v1.7에서 추가된 구조

CCMR v1.6의 cross-fit sign consensus, context support veto, raw
unit-regret minimax에 두 안전장치를 추가했다.

1. 현재 origin에서 이미 관측 가능한 최근 5개 비중첩 동일-horizon
   shadow forecast가 모두 persistence를 이긴 경우에만 행별 보정;
2. 이 gate까지 적용한 validation에서 coverage, pooled/macro 개선,
   raw-risk cap을 모두 통과하지 못하면 test 전체를 exact persistence로
   복원.

열린 luminosity 개발자료에서는 첫 장치가 기존 최대 손상을
29.34%에서 1.44%로 크게 낮췄지만 validation이 위험을 검출했고, 두 번째
장치가 최종 배포를 exact persistence로 만들었다.

## 봉인 정책

counted sealed cohort는 요청대로 두 개만 유지한다.

1. Alloy A: 독립 성능 성공, 영구 봉인;
2. Concrete material shift: 독립 safe fallback, 성능 성공 아님.

기존 Luminosity 실패 파일과 해시는 삭제하거나 고치지 않고
`audit_only_not_counted`로 보존한다. 새 고호트를 추가해 성공 사례 수를
늘리지 않는다.

관련 파일:

- `protocols/CCMR_V17_CAUSAL_BACKTEST_PROTOCOL.md`
- `protocols/CONCRETE_MATERIAL_SHIFT_CCMR_V17_PROTOCOL.md`
- `experiments/concrete_material_shift_ccmr_v17.py`
- `results/concrete_material_shift_ccmr_v17/sealed_predictions.npz`
- `results/concrete_material_shift_ccmr_v17/results.json`
- `protocols/SEALED_COHORT_REGISTRY.json`
