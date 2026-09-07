# MATR PP 용량·최적화 추가 탐색 비교

사후 개발 실험. PP와 FT 각각 seed별 9후보, 최대 300 epoch / patience 70, validation MSE로 선택한 뒤 선택 모델만 test 평가. PP는 기존 seed별 정규화 계수를 재사용하므로 총 누적 탐색 예산이 동일한 확증 비교는 아니다.

PP: width 24/48/96 × (lr, weight decay) = (0.0002,0.1)/(0.0005,2)/(0.001,0.01). affine 동결, 원래 gate와 expert 구조 및 loss 유지. 원래 후보의 validation 점수·선택 epoch가 모든 seed에서 정확히 재현됨. FT는 기존 ft_original_budget_v1 결과를 사용.

| 모델 | pooled ensemble R² | seed R² 평균 | SD |
|---|---:|---:|---:|
| 원래 PP | 0.257392 | 0.171052 | 0.108738 |
| PP 추가 튜닝 | 0.182935 | 0.087227 | 0.147567 |
| FT 추가 튜닝 | 0.331059 | 0.260064 | 0.073914 |

추가 탐색의 validation 최적 선택이 test 개선을 보장하지 않았다. 본 실험에서 튜닝 PP는 기존 PP와 FT보다 낮다. PP의 최적 가능 성능을 확정하거나 모든 데이터에서 FT의 우월성을 입증하는 결과는 아니다. 더 긴 학습 한도, 깊이 탐색 및 전체 loss-optimizer 공동 탐색은 수행하지 않았다. test 점수로 기존/추가 PP를 골라 확증 성능으로 보고하지 않는다.
