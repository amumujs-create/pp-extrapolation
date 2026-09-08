# Na-ion boundary-quotient PP 개선

## 변경한 모델

공식 80%-EOL에는 `capacity=0.8 Ah`라는 알려진 failure boundary가 있다. 일반 PP가
RUL 자체를 직접 회귀하도록 두지 않고 다음처럼 경계 여유와 양의 quotient로
factorization했다.

\[
\widehat{RUL}(t)=\max(C_t-0.8,0)\,\operatorname{softplus}
\{f_{affine}(x_t)+r_{NN}(x_t)\}.
\]

이는 PP 안의 frozen affine quotient path와 bounded neural residual을 사용한다. 일반
NN이나 다른 예측기의 output ensemble이 아니다. 경계에서 RUL=0이 구조적으로
보장되고, NN은 단위 capacity margin당 남은 시간과 history-dependent acceleration만
학습한다.

## 개발 결과

앞선 공식 prospective split을 그대로 재사용해 five-seed로 실행했다. 이 구조는 generic
PP 결과를 확인한 뒤 개발했으므로 독립 confirmatory evidence가 아니다.

| 모델 | pooled R² | RMSE | MAE |
|---|---:|---:|---:|
| Plain MLP | 0.470 | 30.715 | 22.963 |
| Generic PP | 0.457 | 31.081 | 25.379 |
| **Boundary-quotient PP** | **0.781** | **19.747** | **15.575** |

BQ-PP validation pooled R²는 `0.929`였다. 다섯 seed 모두 validation이 epoch 0을
선택해 zero-initialized NN residual을 켜지 않고 affine quotient를 보존했다. 따라서
이번 상승은 NN 표현력이나 seed ensemble의 우연한 상쇄가 아니라 올바른 target
factorization에서 발생했다. 이 데이터 하나로 learned residual의 추가 기여를
주장하지 않는다.

## 해석

정적 gate가 모든 데이터에서 generic PP 사용 여부를 맞히도록 강제하는 것이 해법은
아니다. failure boundary가 알려진 데이터는 **typed prior contract**로 BQ-PP를 선택하고,
경계가 없거나 신뢰할 수 없을 때 generic PP와 applicability gate를 사용해야 한다.

따라서 논문의 개선된 구조는 다음 계층이다.

1. EOL boundary가 알려짐: boundary-quotient PP.
2. 명시적 식은 없지만 monotone tail 구조가 있음: generic PP.
3. validation transport와 regime compatibility가 깨짐: abstain 또는 generic NN.

이 결과는 Na-ion 실패 원인을 모델 구조로 해결했지만 이미 본 test를 사용한 개발
결과다. `BQ-PP > generic PP/MLP`의 확증 주장은 동일 80%-EOL 정의의 새 untouched
cohort에서 프로토콜을 고정한 뒤 검증해야 한다.
