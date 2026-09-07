# MATR open-loop conditioned PP 개발 결과

## 실험 질문

기존 MATR tail 평가는 매 미래 시점의 실제 health를 다시 입력받는 pointwise RUL 회귀다.
이번 실험은 test cell마다 cycle 100, 150, 200, 250, 300, 350, 400까지만 관측하고,
각 고정 prefix 이후 RUL을 예측하는 open-loop 문제로 바꿨다. 충전 정책별로 cell을
train/validation/test에 분리해 모든 정책을 각 split에 유지했다. 입력은 마지막 64-cycle의
상대 SOH, capacity loss, 평균 온도와 관측된 충전 정책이다. Test cell은 train과 겹치지 않는다.

이 결과는 이미 확인한 MATR2019 cohort의 사후 모델 개발 결과이며 독립 확증이 아니다.

## 모델

- **Direct temporal attention**: prefix encoder가 RUL을 직접 회귀한다.
- **Conditioned hazard-PP**: 동일한 temporal attention encoder가 미래 구간별 hazard를
  출력한다. Survival은 hazard의 누적곱으로 구성되어 시간에 따라 단조 감소하고,
  RUL은 survival curve의 적분으로 계산된다. 특정 열화식을 forward 함수에 삽입하지 않는다.

두 모델 모두 동일 policy context, train/validation/test rows, seeds 42--46과
validation-only 구조 선택을 사용했다.

## 결과

| 모델 | ensemble pooled R² | 단일 seed 평균±SD | unit-macro R² |
|---|---:|---:|---:|
| Direct temporal attention | **0.637** | 0.555 ± 0.157 | -0.537 |
| Conditioned hazard-PP | 0.626 | **0.621 ± 0.009** | -0.585 |
| 추가 hazard tuning | 0.618 | 0.613 ± 0.013 | -0.619 |

Direct attention 개별 seed R²는 0.294--0.679였고, hazard-PP는 0.613--0.632였다.
따라서 ensemble 최고점에서는 direct attention이 0.011 높지만, 임의의 한 모델을 다시
학습해 배포하는 조건에서는 hazard-PP의 평균이 0.065 높고 seed SD는 약 17배 작다.

| 관측 종료 cycle | Direct attention R² | Hazard-PP R² |
|---:|---:|---:|
| 100 | **0.509** | 0.356 |
| 150 | 0.372 | **0.396** |
| 200 | **0.432** | 0.423 |
| 250 | **0.513** | 0.441 |
| 300 | 0.587 | **0.597** |
| 350 | 0.554 | **0.663** |
| 400 | **0.708** | 0.698 |

각 horizon은 test cell 9개뿐이므로 horizon별 R²는 탐색적 결과다. 특히 unit-macro R²는
cell마다 일곱 landmark만 있어 분모 분산이 작으므로 주 성능 판단에는 적합하지 않다.
Cell 단위 uncertainty와 RMSE를 함께 보고해야 한다.

## 판정

Hazard-PP가 FT 계열을 pooled ensemble에서 이겼다는 주장은 아직 할 수 없다. 다만
open-loop 정의에서는 기존 pointwise MATR보다 격차가 크게 줄었고, 단일 재학습 성능과
안정성에서는 hazard-PP가 명확히 우세했다. Validation으로 hazard bin width와 loss weight를
추가 선택한 결과는 test ensemble 0.618로 낮아졌으므로 채택하지 않는다.

## 전체 causal history 보강

예측 시점 이전 정보를 직전 64 cycle로 제한하지 않고, 최근 64 cycle 원자료와
8·16·32·64·128-cycle 및 전체 prefix의 SOH·loss·temperature 통계를 함께 입력했다.
Train에는 cycle 50부터 425까지 25-cycle 간격의 variable-length prefix를 사용했고,
validation/test landmark는 바꾸지 않았다. 두 모델은 동일한 history를 받았다.

| 모델 | ensemble pooled R² | 단일 seed 평균±SD | unit-macro R² |
|---|---:|---:|---:|
| Multiscale direct attention | **0.696** | **0.672 ± 0.039** | **-0.290** |
| Multiscale hazard-PP | 0.658 | 0.642 ± 0.031 | -0.450 |
| Survival + bounded residual hybrid PP | 0.647 | 0.624 ± 0.036 | -0.494 |

전체 history는 direct attention을 `0.637→0.696`, hazard-PP를 `0.626→0.658`로
각각 개선했다. 따라서 causal history 보강은 채택한다. 그러나 동일 정보를 주면 direct
attention도 더 크게 개선되므로 PP가 Transformer를 이겼다는 증거는 아니다. Survival
RUL에 bounded attention residual을 공동 학습한 hybrid head도 0.647로 낮아 폐기한다.

다음 확장에서는 cycle 100의 장기 horizon 약점을 해결해야 한다. 관측 prefix로부터
미래 health trajectory를 여러 horizon에서 직접 복원하는 auxiliary decoder와,
condition별 latent transition을 hazard head 앞에 추가하는 것이 타당하다. 구조와 loss를
다른 개발 cohort에서 고정한 뒤 새 cell cohort에서 ensemble pooled R²와 single-model
mean/SD를 함께 검증해야 한다.

기계 판독 결과: `results/matr_open_loop_conditioned_pp_v1/results.json`

실행 코드: `experiments/matr_open_loop_conditioned_pp.py`
