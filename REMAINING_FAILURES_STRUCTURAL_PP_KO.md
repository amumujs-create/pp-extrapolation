# XJTU·FEMTO structural PP 개선

새 cohort를 열지 않고 이미 관측된 XJTU·FEMTO split에서 수행한 post-test development다. 모든 구조·optimizer 선택은 train·validation으로 했고 test는 선택 후 평가했다.

## 통합 모델링 원리

PP executor는 prior contract에 따라 다음 세 경로를 갖는다.

1. 알려진 failure boundary가 있으면 inspection-corrected boundary quotient
2. 수명 범위가 train target을 넘으면 scale-free progress quotient
3. prior evidence가 약하면 exact direct-NN safety path

XJTU에서는 RUL을 바로 학습하면 train 최대 RUL 범위를 넘지 못한다. 대신 진행률 `p=t/(t+RUL)`의 logit을 affine progress head로 학습하고

\[
\widehat{RUL}=t\exp[-\widehat{\operatorname{logit}(p)}]
\]

로 복원했다. Ridge alpha 8개는 validation RMSE로 선택했다. 이 표현은 target scale을 직접 제한하지 않는다.

FEMTO에서는 각 bearing prefix의 elapsed time, baseline 대비 센서 변화, 8/32-step slope, recent variation/max를 causal feature로 만들었다. 독립 train bearing이 5개뿐이므로 NN width를 `next_power_of_two(group_count)=8`로 제한하고 learning rate·weight decay만 validation으로 튜닝했다. 이는 단일 validation bearing에서 큰 network가 잘못 선택되는 현상을 줄이는 group-evidence capacity control이다.

## 결과

| 설정 | 이전 PP R2 | 개선 PP R2 | 이전 최고 경쟁모델 | 판정 |
|---|---:|---:|---:|---|
| XJTU condition transfer | -1.229 | **-0.843** | linear-tail RBF -1.418 | 상대 우세, 절대 실패 |
| FEMTO endpoint transfer | -1.165 | **-0.571** | monotone NN -0.973 | 상대 우세, 절대 실패 |
| NASA milling material transfer | -4.826 | **0.341** | tuned NN -0.476 | 양수 복구·PP 우세 |

세 설정 모두에서 관측된 비-PP 경쟁모델보다 높아졌지만, XJTU·FEMTO는 pooled R2가 음수이므로 성공 설정으로 세지 않는다. 두 데이터의 남은 병목은 최적화보다 조건별·개체별 lifetime scale의 식별 근거 부족이다.

재현 코드는 `experiments/remaining_failures_structural_pp.py`, 공통 route는 `src/pp_extrapolation/adaptive_routes.py`, 원시 결과는 `results/remaining_failures_structural_pp_v1/results.json`에 있다.
