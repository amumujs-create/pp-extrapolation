# Deterministic RBF-regime PP 확증 v2 결과

사전 동결 commit `0934b42`이 난 뒤 공식 Zn-ion test 7–12번 셀 6개를 다운로드하고 한 번 평가했다.

## 적합성 감사

- 이전 원본과 정확히 중복된 파일: 0개
- 오른쪽 중단: 0개
- 152-cycle 경계 뒤 tail이 50행 미만: 5개
- unique eligible cell: `205-3` 1개

프로토콜은 적합 셀이 2개 미만이면 inconclusive로 판정하도록 사전 고정했다. 따라서 **정식 확증 성공으로 선언하지 않는다.**

## 적합 셀의 성능

| 모델 | pooled R² | RMSE | MAE |
|---|---:|---:|---:|
| single-seed plain MLP | 0.057 | 47.936 | 38.693 |
| deterministic RBF-regime PP | **0.959** | **9.972** | **7.123** |

`205-3`의 prefix slope는 `-1.54e-4`였고 latent-scale gate는 0.176이었다. 따라서 예측의 대부분은 boundary-quotient path가 맡고 RBF lifetime prior가 작게 보정했다. 이는 ensemble 없이 단일 결정론적 모델로 얻은 결과다.

## 판단

신규 구조는 기존 latent MLP의 seed 분산을 제거했고, 첫 번째 unique untouched 적합 셀에서 plain MLP를 큰 차이로 넘었다. 다만 n=1이므로 일반화 증거는 아니다. 다음 확증에서는 모델과 50-row 적합 규칙을 그대로 유지하고, 공식 test 목록의 남은 셀로 eligible unit 수를 늘려야 한다.
