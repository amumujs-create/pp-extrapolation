# FT 원래 PP 예산 비교: 중간 결과

아직 전체 실험 완료 아님. MATR2019 진행 중, HUST 대기. 최종 결과는 results/ft_original_budget_v1/results.json 생성 후 확인한다.

FT에는 일반적인 validation 기반 학습률/구조/checkpoint 선택만 적용했다. PP의 prior, gate, residual 제어, 특수 loss를 이식하지 않았다. 양쪽 모두 seed별 9개 후보, 최대 300 epoch, patience 70이지만 탐색 축과 계산량까지 동일한 것은 아니다. 이미 결과를 본 데이터에서의 사후 비교다.

| Virkler | 원래 PP | FT 재실험 |
|---|---:|---:|
| 5 seed 평균 예측 pooled R² | -5.373762 | -3.658347 |
| 단일 seed R² 평균 | -5.495880 | -15.658022 |
| seed 간 표준편차 (ddof=0) | 1.248462 | 30.472505 |

FT seed별 pooled R²:

- 42: 0.828366
- 43: 0.933369
- 44: 0.649576
- 45: -76.478243
- 46: -4.223180

FT 일부 seed는 양호하지만 큰 실패 seed가 있다. 평균 예측 성능과 재학습 안정성은 서로 다른 결론을 준다. 이 결과만으로 FT 또는 PP의 보편적 우월성을 주장할 수 없다. 두 ensemble R² 모두 음수다.
