# CCMR v1.6 및 미개봉 Alloy A 결과

## 결론

CCMR v1.6은 방금 실패한 NASA test를 다시 맞추지 않고, 이미 열린 LED
개발자료에서 보정 gate를 설계한 뒤 완전히 미개봉이었던 실제 금속
피로균열 고호트에서 one-shot 성공했다.

- Persistence: pooled R² 0.75792, RMSE 0.10262, MAE 0.09000
- CCMR v1.6: pooled R² 0.76433, RMSE 0.10125, MAE 0.08637
- RMSE 개선: 1.33%
- test raw-unit regret: mean -7.87%, CVaR20 0%, maximum 0%
- exact fallback replay error: 0

CCMR은 test 행의 70%를 context OOD로 거부해 persistence를 정확히
복원했고, 합의된 30%만 보정했다. 네 test 시편 중 하나도 persistence보다
악화되지 않았다.

## v1.5에서 바뀐 구조

OAIR v1.5는 validation 평균에서 안전하면 모든 support 내부 행에 같은
약한 보정을 적용했다. NASA B0051/B0052처럼 slope 범위는 비슷하지만
후기 형상이 다른 경우 이를 구분하지 못했다.

CCMR(Cross-fit Consensus--Minimax Residual) v1.6은 다음 세 인증을 모두
통과한 행만 보정한다.

1. train physical-unit leave-one-out ridge의 90% 이상이 보정 방향에 합의;
2. fold MAD 대비 보정 크기가 충분하고, 전체 context의 validation
   99% support 안에 존재;
3. validation raw-unit mean/CVaR20/max regret가 0%/1%/2% 이내이며
   macro-MSE도 실제 개선.

하나라도 실패하면 행별 또는 모델 전체가 exact persistence로 돌아간다.
이는 단일 실패 유형 보정보다 **선택적 안전 보정**에 가깝다.

## LED post-test development와 ablation

이미 열린 LED Data_ID_2/5/6에서 구조를 고정했다.

- Data_ID_2: raw cell-risk 인증 부족으로 mass 0, exact persistence
- Data_ID_5: RMSE 0.008109 → 0.008090, mass 0.010
- Data_ID_6: validation 인증 부족으로 mass 0, exact persistence

진짜 raw cell regret를 적용하면 기존 regularized regret보다 훨씬
보수적으로 작동한다. 즉 Data_ID_2/6의 작은 평균 이득을 포기하고
레짐 이동 안정성을 택한 결과다. 이 LED 결과는 post-test development이며
독립 증거가 아니다.

## 미개봉 Alloy A 계약

- GPL-2 `SMRD.data`의 실제 금속 피로균열 실험
- 원본 `alloya.rda`: 773 bytes, SHA-256
  `9fb4295f12393fbfdb81183aed4dce7fc829ea922228c342e15990895a72380e`
- 21 specimens, 262 observations
- hash 고정 분할: train 13 / validation 4 / test 4 specimens
- train 25 / validation 12 / test 10 forecast origins
- train origin 진행도 최대 30%, test 최소 75%
- test origin의 100%가 train ordered-axis support 밖

결과를 보기 전에 split, feature, gate, risk cap, 성공 기준을
`protocols/ALLOYA_UNOPENED_CCMR_V16_PROTOCOL.md`에 고정했다. 예측은
점수 계산 전 `sealed_predictions.npz`로 저장했다.

사후 코드 감사에서 기존 `unit_regret`이 중앙 MSE stabilizer를 포함한다는
정의 불일치를 발견해 selector를 진짜 raw excess MSE로 교정했다. 교정된
selector도 mass 0.25를 선택했고 봉인 예측과 최대 절대 차이 0, active
mask 완전 동일이었다. Validation raw mean/CVaR/max regret는
-33.09%/-20.09%/-20.09%, test는 -7.87%/0%/0%였다. 따라서 결과는
보존되며 감사 내역은 `raw_regret_audit.json`에 별도로 남겼다.

## 해석과 한계

이번 결과는 “모든 곳에서 작은 prior를 적용”하는 것보다 “근거가 있는
부분만 보정하고 나머지는 정확히 baseline으로 복원”하는 구조가 더
강건할 수 있다는 독립 증거다.

하지만 test가 4 specimens, 10 origins뿐이라 통계적 검정력은 낮다.
또한 동일 재료·동일 하중 안의 unit-disjoint 시간 외삽이지,
새 재료나 새 하중 조건으로의 regime 외삽은 아니다. 따라서 v1.6을 최종
범용 모형으로 확정하지 않고, 더 큰 미개봉 condition-shift 고호트가
추가로 필요하다.

다음 미개봉 후보는 CC BY 4.0 Perovskite Solar Cells Ageing 자료다.
공개 metadata상 33 MB, 실제 cell 2,245개, 약 901 ageing 시점으로 현재
저검정력 한계를 해소할 수 있다. 다만 fabrication batch 단위 분할과
미래 누출 없는 정규화를 먼저 고정해야 한다. 또한 in-support concept
shift까지 막으려면 최근 비중첩 동일-horizon shadow forecast가 실제로
persistence를 이긴 경우만 보정하는 causal backtest gate를 다음
development 버전에서 검토해야 한다. 이 자료는 아직 다운로드하거나
outcome을 열지 않았다.

관련 파일:

- `src/pp_extrapolation/consensus_residual.py`
- `experiments/ccmr_v16_led_development.py`
- `results/ccmr_v16_led_development/results.json`
- `experiments/alloya_unopened_ccmr_v16.py`
- `results/alloya_unopened_ccmr_v16/results.json`
- `results/alloya_unopened_ccmr_v16/sealed_predictions.npz`
- `results/alloya_unopened_ccmr_v16/raw_regret_audit.json`
