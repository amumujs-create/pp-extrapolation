# MATR 외삽 정의 및 split 민감도 감사

## 판정

기존 MATR2019 평가는 누수가 확인된 split은 아니다. Train/validation/test cell은 서로
겹치지 않고, test feature는 현재와 과거 8-cycle 관측만 사용한다. 그러나 이 평가는
`unseen-cell + capacity tail + charging-policy shift + lifetime-distribution shift`가 동시에
섞인 복합 stress test다. 또한 각 test tail 시점의 실제 health를 입력으로 받으므로
고정 prefix에서 미래 궤적을 예측하는 open-loop temporal extrapolation은 아니다.

파일순서 split의 평균 cycle life는 train 769, validation 683, test 885 cycle이었다.
Train에 없는 충전 정책도 validation/test에 포함되었다. 절대 capacity boundary를 사용해
cell별 초기 capacity 차이도 tail 정의에 섞였다.

## 정책 균형 상대-SOH 대조 split

각 충전 정책에서 마지막 두 eligible cell을 validation/test로, 나머지를 train으로 배정했다.
모든 split에 각 정책이 존재하도록 했고, capacity를 각 cell의 첫 8-cycle median으로
나눈 상대 SOH로 tail boundary를 정의했다. Train/validation/test는 26/9/9 unseen cells이며,
validation/test tail은 train 상대-SOH hull 밖 100%다. 모델당 9개 설정을 seed 42
validation으로 선택하고 같은 설정을 seeds 42--46에 적용했다. 이 실험은 이미 본 cohort를
이용한 사후 split 민감도 분석이며 새 확증 결과가 아니다.

| 입력/모델 | ensemble pooled R² | seed 평균 pooled R² | unit-macro R² |
|---|---:|---:|---:|
| health-only PP | 0.269 | 0.094 | -0.715 |
| health-only FT-Transformer | **0.535** | **0.487** | **0.089** |
| health + current age PP | 0.089 | 0.023 | -0.553 |
| health + current age FT-Transformer | **0.476** | **0.428** | **0.128** |
| latent-EOL attention | -0.807 | -0.979 | -2.599 |

정책 균형과 상대 SOH로 교정해도 FT 우위가 유지되고 오히려 안정적이다. 따라서 기존
파일순서 split 하나 때문에 FT가 우세했다는 설명은 기각한다. Current age의 단순 추가도
두 모델을 개선하지 않았다. Attention으로 cell별 공통 EOL 시점을 추정한 뒤
`RUL = EOL - current cycle`로 계산하는 구조도 실패했다. 같은 SOH에서 정책별 total life가
달라 health-only EOL이 식별되지 않는 것이 원인으로 해석된다.

## 외삽 질문을 분리하는 권장 프로토콜

1. **Matched-condition unseen-unit tail**: 정책마다 train/validation/test cell을 분리하고
   상대 SOH tail을 평가한다. 새로운 엔진·cell 일반화의 주 실험으로 사용한다.
2. **Unseen-condition tail**: charging policy 또는 operating condition을 통째로
   leave-one-condition-out한다. 조건 이동에 대한 별도 stress test로 사용한다.
3. **Open-loop future extrapolation**: test unit마다 하나의 고정 prefix까지만 입력하고,
   이후 trajectory 또는 여러 horizon의 RUL을 예측한다. 현재의 모든 tail window를 입력하는
   pointwise RUL 회귀와 별도 표로 보고한다.
4. Pooled R²와 unit-macro R²를 함께 내고, 통계 검정과 bootstrap의 단위는 겹치는 window가
   아니라 cell로 둔다.

## 모델 개발 판단

MATR에서 FT를 이기기 위한 다음 구조는 affine prior를 더 강하게 거는 방식이 아니다.
필요한 모델은 policy/operating context를 조건으로 받는 temporal attention과,
관측 prefix에서 추정한 regime-change hazard를 이용해 미래 latent state를 rollout하는
모델이다. 이 모델은 matched-condition tail에서 개발하고 구조를 고정한 뒤,
unseen-condition 및 새 cohort에 평가해야 한다. 이미 본 MATR test 점수를 기준으로 계속
구조를 선택하면 수치는 개선될 수 있어도 논문상 확증 근거가 되지 않는다.

기계 판독 결과는 `results/matr_split_sensitivity_v1/`에 저장했다. 실행 코드는
`experiments/matr_split_sensitivity.py`다.
