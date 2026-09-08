# Zn-ion PP matched-information controls

이 분석은 이미 관측한 v3에서 실행한 사후 공정성 감사다. untouched 확증으로 재분류하지 않는다.

## 결과

| 대조군 | pooled R² | unit-macro R² |
|---|---:|---:|
| 기존 MLP 5-seed mean | -0.082 | -3.768 |
| 상한을 제거한 기존 MLP | -0.082 | -3.768 |
| validation EOL까지 같이 쓴 refit MLP | 0.537 | -2.167 |
| 같은 dev EOL memory만 쓴 RBF lifetime | 0.780 | -2.611 |
| BQ + RBF regime PP | **0.822** | **0.375** |

## 해석

기존 MLP의 학습 RUL 상한은 367이지만 5개 seed 모두 test 예측이 그 상한에 닿지 않았다. 따라서 clipping은 잠재적 외삽 제약이지만 현재 v3 점수차의 실제 원인은 아니다.

PP만 validation EOL을 memory label로 쓴 정보 비대칭은 실제로 크다. MLP에도 모든 development prefix와 EOL label을 주면 pooled R²가 -0.082에서 0.537로 올랐다. 그럼에도 개선 PP 0.822보다 낮고 macro R²는 음수다.

RBF lifetime-only는 같은 descriptor와 EOL memory를 쓰면서 pooled 0.780을 냈지만 macro가 -2.611이다. BQ 경로를 결합한 PP가 pooled를 0.042 높이고 macro를 양수로 바꾸었다. 이 비교에서는 memory 정보 하나만으로 현재 PP의 전체 성능을 설명할 수 없다.

## 남은 공정성 문제

refit MLP는 전체 dev prefix로 학습하고 원래 train unit의 tail로 epoch을 골랐다. 학습 prefix와 validation tail이 같은 physical unit을 공유하므로 엄밀한 nested unit holdout은 아니다. 다음 비교에서는 모든 모델에 동일한 outer unit split과 inner unit validation을 적용해야 한다.

기존 MLP는 하나의 architecture만 쓴다. 이 결과로 충분히 튜닝한 일반 NN을 넘었다고 주장하지 않는다. width, depth, activation, weight decay를 inner validation으로 선택한 강한 양의-output MLP와 temporal model이 필요하다.
