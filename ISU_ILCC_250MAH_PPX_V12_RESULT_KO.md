# 250 mAh 외부 고호트 PP-X v1.2 결과

## 결론

ISU–ILCC 250 mAh NMC pouch 고호트에서 강건한 v1.2 시스템은
**안전 후퇴 성공**, PP prior 확장 자체는 **실패**했다.

- 적격 셀: **229**
- split: train/validation/test = **137/46/46셀**
- strict-tail 행: **1,375/167/148**
- validation/test tail coverage: **100%/100%**
- 선택: **trust=0 matched MLP**
- test pooled R²: **0.592**
- test RMSE: **1.479일**
- test MAE: **1.152일**
- trust-zero 독립 MLP 재현 오차: **0.0**

## 왜 PP prior를 끘나

가장 좋은 positive-trust 후보는 validation pooled RMSE를 **2.29%**
줄여 첫 문턱은 넘었다. 그러나:

- 개선된 validation 셀 비율: **33.3%** (필요 60%)
- 최악 셀 RMSE 비율: **3.05** (허용 1.10)

즉, 평균만 보면 좋아 보이지만 일부 셀의 손상이 너무 컸다. v1.2는 test를
보기 전에 이 후보를 거절하고 trust=0으로 고정했다. 따라서 test의
R² 0.592는 PP prior 성과가 아니라 **matched MLP fallback 성과**다.

## 해석상 주의

pooled R²는 양수지만 test cell-macro R²는 **−1.449**다. strict tail에서
개별 셀당 평가점이 1–11개로 적어 셀별 R²가 불안정하며, 모든 셀에
균일하게 강하다고 말할 수 없다. 현재 증거가 지지하는 주장은 다음이다.

> PP-X v1.2의 prior 승인문은 평균 이득에 속지 않고 위험한 prior를
> 차단했으며, OOD 조건 그룹에서 matched MLP로 안전하게 후퇴해 pooled
> 예측력을 유지했다.

“PP prior가 250 mAh 고호트로 일반화됐다”는 주장은 지지되지 않는다.

## 재현·증거

- 사전 고정: `protocols/ISU_ILCC_250MAH_PPX_V12_PROTOCOL.md`
- 실행: `experiments/isu_ilcc_250mah_ppx_v12.py`
- 전체 결과: `results/isu_ilcc_250mah_ppx_v12/results.json`
- 예측: `results/isu_ilcc_250mah_ppx_v12/predictions.npz`
- 데이터 SHA-256:
  `2dada0f57db9dbc002abe8ed839a61403003739086cbcc207be479689b5dc7af`
- 출처: DOI `10.25380/iastate.22582234.v2`

추가로 검토한 독립 UConn NMC 고호트는 48셀 모두 65% EOL을 교차했지만,
동결한 최소 15 RPT 조건을 만족한 셀이 23개뿐이라 학습 전에
`inconclusive`로 중단했다.
