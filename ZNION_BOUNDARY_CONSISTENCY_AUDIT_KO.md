# Boundary-consistent lifetime path 개발 감사

이번 실험은 기존 결과를 본 이후의 개발 실험이다. 새로운 확증이 아니다.

기존 BQ 경로는 margin=0에서 0이지만, 혼합한 lifetime 경로는 그렇지 않다. 예를 들어 기존 v3 `446-1`의 EOL 예측은 약 79 cycle이었다. 따라서 통합 모델 전체에 대해 hard-zero boundary를 주장할 수 없다.

수정 후보는 lifetime 예측에 margin/(margin+tau)를 곱한다. tau>0일 때 두 경로 모두 경계에서 0이다. tau=0은 기존 모델 대조군이다.

## 선택 방법과 결과

Development 8개 셀 중 152-cycle 경계 이후 50개 이상 시점이 있는 5개 셀을 하나씩 제외했다. 제외 셀은 affine fitting, descriptor scaling, RBF memory에서 모두 제거했다. 셀별 R²의 평균으로 tau를 선택했다.

| tau | LOO 평균 셀 R² |
|---:|---:|
| 0 | -0.773188 |
| 0.001 | -0.774154 |
| 0.003 | -0.776028 |
| 0.01 | -0.782081 |
| 0.03 | -0.796354 |
| 0.1 | -0.828856 |

선택은 tau=0이므로 수정안을 채택하지 않는다. 기존 v3 pooled R² 0.821521, macro R² 0.374656을 그대로 보존한다. 개선 실패 후보도 selection.json과 results.json에 저장했다.

## 중요한 한계

LOO에서는 residual 학습을 하지 않은 affine-only BQ를 사용했으며, 최종 재생에서는 기존 validation으로 선택한 residual을 사용했다. 따라서 위 LOO 점수를 최종 배포 모델의 엄밀한 nested CV 점수로 해석하면 안 된다. boundary=152, alpha=1000도 앞선 개발에서 고정된 값으로, 완전한 nested CV가 아니다.

장수명 `442-2`를 memory에서 제외했을 때 tau=0의 R²가 -2.340이었다. 장수명 regime의 독립적인 학습 사례가 하나뿐이라는 취약성을 보여준다. 감쇠 강도를 더 높이는 것만으로는 해결되지 않았다.

RBF head 자체는 결정론적이지만 BQ residual은 seed를 사용하는 NN이다. seed=42를 고정한 실행을 '재학습 불안정성 전체 제거'로 표현하면 안 된다. 현재 gate는 학습한 hazard가 아니라 prefix slope의 고정 sigmoid 규칙이며 두 경로를 end-to-end 공동 학습한 네트워크도 아니다.

통계 주의: 4/4 승리 이후 한 번 더 이기는 셀을 확보하면 p<0.05가 된다는 식으로 표본을 추가해서는 안 된다. 표본 수와 종료 규칙을 사전에 정한 별도 평가가 필요하다. 기존 작은 표본 bootstrap CI는 탐색적 기술통계다.

후속 개발은 경계 일관성을 갖는 lifetime quotient를 직접 학습하고, unit 전체를 제외하는 검증과 여러 장수명 regime의 독립 unit을 확보하는 방향이 필요하다. 이번 후보는 기존 모델을 대체하지 않는다.
