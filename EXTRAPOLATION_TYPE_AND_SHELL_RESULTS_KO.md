# 외삽 유형 분리와 거리 shell 결과

## Seen-unit future tail

기존 결과는 train과 test unit이 분리된 unseen-unit late-tail이었다. 이번에는 같은 unit의 early 구간으로 같은 unit의 더 깊은 future tail을 예측했다.

| 데이터 | Ridge R² | PP single R² | PP ensemble R² | plain NN ensemble R² |
|---|---:|---:|---:|---:|
| HUST | -4.191 | -1.069±0.682 | -0.565 | **0.340** |
| Virkler | -2.330 | -4.576±2.271 | -4.180 | **-0.688** |

PP는 기존 unseen-unit late-tail에서는 HUST `0.773`, Virkler `0.888`이었지만 seen-unit의 훨씬 깊은 tail에서는 실패했다. PP의 강점은 모든 future forecasting이 아니라 여러 unit 사이에 공유되는 late-tail affine trend의 전이다.

HUST에서 일반 NN ensemble만 양의 R²를 얻은 것은 깊은 future 구간에 affine tail로 표현되지 않는 곡률 또는 regime 변화가 있음을 시사한다. 이 외삽 유형까지 목표로 한다면 단순히 PP 폭을 키우는 것보다 change-point/regime gate와 piecewise curvature head가 필요하다. Virkler는 두 개의 매우 먼 crack-length 지점만 남아 target support가 제한되므로 이 실험만으로 고급 모델을 선택하면 안 된다.

## Strict-hull distance shells

기존 unseen-unit strict-tail test를 support distance 순서로 같은 크기의 near/mid/far 세 shell로 나눴다. 제한된 shell에서는 R² 분모가 불안정하므로 ensemble RMSE를 주로 해석한다.

| 데이터 | Shell | affine RMSE | PP RMSE | support-PP RMSE |
|---|---|---:|---:|---:|
| HUST | near | **75.62** | 100.34 | 99.92 |
| HUST | mid | 84.29 | 51.81 | **49.19** |
| HUST | far | 119.82 | **52.32** | 57.65 |
| Virkler | near | 14.34 | **3.59** | 3.59 |
| Virkler | mid | 10.16 | **2.46** | 2.46 |
| Virkler | far | 0.00 | 0.00 | 0.00 |
| NASA | near | 21.39 | **20.57** | 20.61 |
| NASA | mid | 15.58 | **13.69** | 13.72 |
| NASA | far | 13.03 | 12.31 | **12.25** |

HUST에서 PP residual은 near shell을 손상시키지만 mid/far에서는 affine보다 크게 좋다. 거리 증가에 따라 residual을 단조 감쇠하는 support gate가 far shell에서 PP보다 나빠진 이유도 이것이다. NASA는 모든 shell에서 PP가 affine보다 낫지만 개선 폭은 작다.

따라서 거리 하나로 gate를 정하는 모델은 충분하지 않다. 다음 고급화가 필요하다면 `distance-only`가 아니라 validation에서 관측된 residual regime와 변화점을 조건으로 affine/curved residual의 신뢰도를 조절해야 한다. 이 확장은 seen-unit deep-future를 논문 claim에 포함할 때만 필요하다. 현재 unseen-unit late-tail PP 논문만 목표로 하면 기존 PP와 applicability certificate가 더 단순하고 방어 가능하다.
