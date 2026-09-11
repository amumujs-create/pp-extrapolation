# LG M50T Experiment 1 CCMR v1.7 봉인 결과

## 결론

추가 성능 성공 고호트를 얻지 못했다. 모델은 test 두 cell을 모두
context OOD로 판정해 exact persistence로 복귀했다. 손상은 없지만
개선도 없으므로 성공 registry에는 추가하지 않았다.

- 실제 상용 LG M50T 21700 cell 열화 자료
- DOI `10.5281/zenodo.10637534`, CC BY 4.0
- 9.1 GB 원격 ZIP 전체 대신 Performance Summary 9개만 range 추출
- 실제 저장량: 약 60 KB
- unit split: train 5 / validation 2 / test 2 cells
- origins: train 11 / validation 6 / test 6
- train 진행도 최대 28.6%, test 최소 78.6%, strict OOD 100%

## Validation

- causal correction coverage: 16.67%
- pooled RMSE 개선: 2.07%
- unit-macro RMSE 개선: 1.96%
- raw unit regret mean/CVaR20/max: -3.62% / 0% / 0%
- 판정: 통과

## 봉인 test

- test cells: B, F
- CCMR base support 통과율: 0%
- deployed correction coverage: 0%
- Persistence 및 CCMR RMSE: 0.082052
- pooled/macro 개선: 0% / 0%
- pooled R²: -0.91664
- raw unit regret mean/CVaR20/max: 0% / 0% / 0%
- exact fallback replay error: 0
- 판정: 안전 복귀, 성능 실패

Validation에서 보정이 유효했더라도 test context가 training support와
달라 v1.7은 보정을 적용하지 않았다. 이는 강건성 동작의 증거지만,
성능 일반화의 증거는 아니다.

## 투명성 기록

처음 검토한 `resistor2`는 unit당 5시점, `metalwear`는 강한 tail test
origin이 총 3개뿐이라 각각 사전 eligibility 단계에서 중단했다. 모델,
예측, test 점수는 생성하지 않았다.

LG M50T는 요약 파일을 받은 뒤 수치 결과를 보지 않은 schema-only
검사에서 11--15 RPT임을 확인했다. 5개 causal shadow forecast에 필요한
수학적 최소치가 10임에 따라 eligibility를 12에서 10으로, test-origin
최소치를 8에서 4로 pre-score 수정했다. 이 수정 때문에 완전 무수정
one-shot보다 증거 등급이 낮다.

성공할 때까지 다른 고호트를 반복 선택하면 selection bias가 생기므로
추가 교체 실험은 하지 않았다.

관련 파일:

- `protocols/LGM50T_EXPT1_UNOPENED_CCMR_V17_PROTOCOL.md`
- `experiments/lgm50t_expt1_ccmr_v17.py`
- `results/lgm50t_expt1_ccmr_v17/sealed_predictions.npz`
- `results/lgm50t_expt1_ccmr_v17/results.json`
- `protocols/SEALED_COHORT_REGISTRY.json`
