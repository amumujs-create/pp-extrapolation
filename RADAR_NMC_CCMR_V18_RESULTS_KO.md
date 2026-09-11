# RADAR NMC CCMR v1.8 최종 순차 확증 결과

## 결론

정식 판정은 **배포 확증 실패(safe fallback)**다. 다만 봉인된 raw CCMR
candidate는 사전 test 성능·위험 기준을 모두 통과해, 모형 성능 가능성은
확인됐다. Router가 validation 단계에서 이를 보수적으로 차단했다.

## 데이터와 외삽

- 실제 commercial NMC/C+SiO cyclic-aging cells: 116개
- unit-disjoint train/validation/test: 69/23/24 cells
- train/validation/test origins: 48,756/8,437/7,009
- train 진행도 최대 30%, test 최소 75%
- strict ordered-axis OOD: 100%
- 원본 result-data TAR: 333.4 MB
- 사용 EOC ZIP: 93.2 MB

## 사전 고정 v1.8 판정

Base validation은 macro RMSE 7.74% 개선, raw mean/CVaR/max regret
-20.69%/+0.62%/+1.17%, active 98.99%였다. Stable route의 사전 기준은
macro 개선 10% 이상과 세 regret 모두 -5% 이하였다. 따라서
`cautious_causal` route가 선택됐고, 그 validation certificate가 실패해
test 배포를 exact persistence로 복원했다.

배포 test 결과:

- pooled R²: 0.99548
- RMSE 개선: 0%
- macro RMSE 개선: 0%
- correction coverage: 0%
- raw mean/CVaR/max regret: 0%/0%/0%
- fallback replay error: 0

## 봉인 후 candidate 감사

배포 판정과 별도로, 이미 봉인된 candidate를 사후 채점하면:

- pooled R²: 0.99561
- pooled RMSE 개선: 1.43%
- unit-macro RMSE 개선: 1.86%
- correction coverage: 59.37%
- raw unit mean/CVaR20/max regret: -5.34%/+0.64%/+1.66%

따라서 raw candidate는 원래의 test 성능·위험 한도를 모두 만족한다.
그러나 validation-only router의 결정을 test를 본 뒤 무효화할 수 없으므로
이는 **탐색적 candidate 성공**이지 정식 배포 성공 고호트가 아니다.

관련 파일:

- `protocols/CCMR_V18_REGIME_ROUTER_PROTOCOL.md`
- `protocols/RADAR_NMC_CYCLIC_CCMR_V18_PROTOCOL.md`
- `experiments/radar_nmc_cyclic_ccmr_v18.py`
- `results/radar_nmc_cyclic_ccmr_v18/results.json`
- `results/radar_nmc_cyclic_ccmr_v18/sealed_predictions.npz`
