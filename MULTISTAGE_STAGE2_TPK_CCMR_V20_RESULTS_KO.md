# Multi-Stage Stage-2 TP_k CCMR v2.0 1차 실행 결과

작성자: 박진서  
상태: **모형 적합 전 inconclusive** (프로토콜 유지)

## 판정

동결 프로토콜의 최소 적격 기준을 통과하지 못해 CCMR을 적합하지
않았다. 성공/실패로 재라벨하지 않는다.

## 추출

- 실행: `experiments/extract_multistage_stage2_tpk.py`
- 인구: Stage-2 `TP_k*` 66 cell 전부
- 행: 351, 유한 capacity 351
- CSV SHA-256: `47520dd90fe2f64fec65916ec42e2a862665bbe190134357acb6ecc2b0bce0c3`

## 적격 탈락 원인

프로토콜은 cell당 최소 10개 RPT를 요구한다. 실제 길이 분포:

| RPT 수 | cell 수 |
|---:|---:|
| 2 | 33 |
| 7 | 12 |
| 9 | 9 |
| 10 | 12 |

적격 cell은 12개뿐이라 최소 50 cell 기준도 미달했다.
아카이브 샘플 확인 결과 필터 누락이 아니라, 짧은 cell은 원래
`ET_T23`+`AT_T23` 두 RPT만 갖고 있었다.

## 봉인 전 중단 요약

- eligible cells: 12
- split: train/val/test 7/2/3
- validation origins: 4
- test origins: 6
- 결과 파일: `results/multistage_stage2_tpk_ccmr_v20/results.json`

## 해석

Stage-2 `TP_k`는 Stage-1 `TP_z`와 달리 RPT 궤적이 짧아, 현재 CCMR
progress-window 계약(history 3, train≤0.30, val 0.40–0.60, test
0.75–0.90)에 맞지 않는다. MultiStage형 “긴 capacity 궤적” 승기 레짐이
아니다.

다음 후보를 고를 때는 **추출 전에** public metadata만으로 RPT member
개수 하한을 스크리닝하거나, Stage-1에서 이미 이긴 MultiStage
retrospective 승기 셋을 비교 문장의 주 근거로 유지하는 편이 맞다.
