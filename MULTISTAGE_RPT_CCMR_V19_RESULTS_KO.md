# Multi-Stage RPT CCMR v1.9 성공 결과

## 판정

CCMR v1.9가 새 실제 배터리 고호트에서 사전 고정한 validation과 test
성공 기준을 모두 통과했다. 순차 탐색 후 선택된 자료이므로
selection-conditioned replication이지만, test 예측과 기준은 점수 확인
전에 봉인됐다.

## 데이터와 외삽

- Samsung INR21700-50E Stage-1 cyclic-aging 적격 cell: 72개
- unit-disjoint train/validation/test: 43/14/15 cells
- train/validation/test origins: 110/50/33
- test 진행도는 모든 origin에서 train 최대 진행도를 초과: 100%
- DOI: `10.6084/m9.figshare.25975315.v1`
- 전체 대상 archive 3.87 GB 중 RPT member만 range 추출

세 개 `TP_z04` cell은 사전에 고정한 최소 10개 유효 RPT 기준을
충족하지 않아 모델 적합 전에 제외됐다. 최초 parser 오류와 이 QC
변경 과정은 protocol에 모두 기록되고 이전 inconclusive artifact도
보존됐다.

## Validation

- route: `stable_base`
- pooled RMSE 개선: 13.80%
- unit-macro RMSE 개선: 17.56%
- raw mean/CVaR20/max regret: -38.56%/-21.98%/-16.43%
- validation 승인: 통과

## 봉인 test

- pooled R²: 0.97940
- pooled RMSE 개선: 13.01%
- unit-macro RMSE 개선: 16.39%
- raw unit mean/CVaR20/max regret: -35.30%/-5.05%/0%
- correction coverage: 81.82%
- fallback replay error: 0
- confirmatory success: **true**

관련 파일:

- `protocols/CCMR_V19_REPLICATED_ROUTER_PROTOCOL.md`
- `protocols/MULTISTAGE_RPT_CCMR_V19_PROTOCOL.md`
- `experiments/extract_multistage_rpt.py`
- `experiments/multistage_rpt_ccmr_v19.py`
- `results/multistage_rpt_ccmr_v19_amended2/results.json`
- `results/multistage_rpt_ccmr_v19_amended2/sealed_predictions.npz`
