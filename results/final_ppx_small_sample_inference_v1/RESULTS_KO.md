# 최종 PP-X 소표본 통계 검증

대상은 최종 PP-X의 기존 9개 후향적 주 평가 설정과 5개 고정 시드다. 시드 45개를 독립 표본으로 사용하지 않고 설정 9개를 주 추론 단위로 사용했다.

## 검정 설계

- 방향 일관성: 양측 exact sign test.
- 평균 효과: 2^9=512개 부호 배치를 모두 열거한 양측 exact sign-flip test.
- 효과 크기 구간: 설정 단위 bootstrap 100,000회. 소표본·개발자료이므로 확증적 신뢰구간이 아니다.
- 개체 RMSE: 설정을 먼저, 그 안에서 물리 개체를 다시 뽑는 계층 bootstrap 100,000회.
- 2개 비교모델 x 3개 주 결과의 6개 검정에 Holm 보정.

## 결과

| 비교 | 결과 | 설정 방향 | 평균 효과 [bootstrap 95% CI] | exact p | Holm q |
|---|---|---:|---:|---:|---:|
| engression | R² 차이 | 9/9 | 0.5504 [0.1009, 1.1243] | 0.0039 | 0.0234 |
| engression | log(seed SD 비) | 9/9 | 1.9388 [1.4197, 2.4484] | 0.0039 | 0.0234 |
| engression | 최악 seed R² 차이 | 9/9 | 1.3430 [0.2713, 2.7210] | 0.0039 | 0.0234 |
| plain_mlp | R² 차이 | 9/9 | 0.7137 [0.1816, 1.4109] | 0.0039 | 0.0234 |
| plain_mlp | log(seed SD 비) | 9/9 | 2.9459 [2.0564, 3.8326] | 0.0039 | 0.0234 |
| plain_mlp | 최악 seed R² 차이 | 9/9 | 3.3664 [1.2014, 5.9367] | 0.0039 | 0.0234 |

log(seed SD 비)는 `log(비교모델 SD / PP-X SD)`이므로 양수가 PP-X의 작은 시드 변동을 뜻한다. 다른 두 차이도 양수가 PP-X 우위다.

## 개체 수준 탐색 결과

| 비교 | 설정 동일가중 log RMSE 효과 | PP-X/비교 RMSE 기하비 | 계층 bootstrap 95% CI |
|---|---:|---:|---:|
| engression | 0.4888 | 0.613 | [0.1342, 0.8095] |
| plain_mlp | 0.7728 | 0.462 | [0.3785, 1.2300] |

## 논문용 해석

가장 직접적인 결과는 PP-X가 9개 설정 모두에서 Engression과 MLP보다 시드 SD가 작고 최악 시드 R²가 높았다는 것이다. exact 검정은 작은 n에 맞춰 이 방향 일관성을 평가한다.

이 결과는 평가한 개발 설정에서의 재학습 안정성을 지지한다. 그러나 데이터셋들이 완전히 독립·동질한 모집단에서 무작위 추출된 것이 아니고, 최종 PP-X도 이 자료를 보며 개발됐으므로 미래 코호트에 대한 확증 p값으로 해석하지 않는다.

프라이어가 타당하기 때문에 변동이 감소했다는 인과적 주장은 이 분석으로 검정되지 않았다. 이를 주장하려면 결과와 독립적으로 prior validity를 정의한 통제 실험이 필요하다.

권장 문장:

> Across nine retrospective extrapolation settings, final PP-X showed lower seed-to-seed variation and a higher worst-seed R² than Engression and a plain MLP in every setting. Exact small-sample tests support the consistency of this pattern within the evaluated benchmark; because these settings informed model development, the result is interpreted as retrospective stability evidence rather than prospective proof of robustness.

## 제한

- n=9이므로 효과 크기와 개별 설정 결과를 p값보다 우선한다.
- R² 차이는 목표 분산에 민감하다. RMSE 기반 개체 결과를 함께 보고한다.
- bootstrap은 관측한 설정을 모집단처럼 재표집한다. 관련 배터리 코호트 간 독립성을 보장하지 않는다.
- seed SD는 선택된 설정 아래 optimizer variation이며 데이터 표본 불확실성을 포함하지 않는다.
- 개체 계층 결과는 사후 탐색 분석이고, 개체 수가 3~16으로 작아 설정별 불확실성이 크다.
- 본 검정은 최종 9-setting PP-X portfolio에 한정한다. DS03, FEMTO, XJTU, milling이나 초기 v1 결과를 합치지 않는다.
