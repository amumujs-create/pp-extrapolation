# 과거 구간 조건부 열화속도 적분 모델: 구현 기록

## 완료 결과: 구현·검증 완료, 후보 미채택

주 후보 `variable_prefix`는 고정된 Engression 비교보다 세 데이터에서 좋지만,
기존 PP-X보다 세 데이터 모두 RMSE가 높았다. 정확도·커버리지 동시 개선 조건을
통과하지 못했다. 기본 모델 교체, 전체 benchmark 확대, 후속 모델 성공 주장은
하지 않는다. 다른 ablation을 test 점수로 골라 주 후보로 승격하지도 않는다.

### 주 평가: train+validation 재학습 후 3-seed 앙상블 R²

| 모델 | SUNWODA | RWTH | MICH | 양의 R² 장비 |
|---|---:|---:|---:|---:|
| 일정 속도 + RUL 학습 | .80554 | .31241 | .59824 | 24/25 |
| 속도 변화 + RUL 학습 | .80636 | .53326 | .52138 | 24/25 |
| 속도 변화 + 구간 학습 (주 후보) | .79592 | .49717 | .33448 | 24/25 |
| 기존 PP-X | .90955 | .83941 | .71003 | 25/25 |
| 공식 Engression, 고정 설정 | -.05235 | .20247 | -3.49244 | 14/25 |

| 데이터 | 주 후보 RMSE | PP-X RMSE | Engression RMSE | 주 후보 최악 장비 RMSE | PP-X 최악 장비 RMSE |
|---|---:|---:|---:|---:|---:|
| SUNWODA | 43.2970 | 28.8245 | 98.3176 | 58.4337 | 44.5129 |
| RWTH | 101.0395 | 57.1008 | 127.2495 | 129.4010 | 102.1392 |
| MICH | 13.0744 | 8.6301 | 33.9691 | 15.7545 | 10.5841 |

주 후보와 PP-X 모두 양의 R² 데이터셋×seed는 9/9이다. 이는 3데이터×3seeds이며
원래 9개 benchmark 전체를 뜻하지 않는다. 주 후보는 RWTH에서 장비 coverage가
8/8→7/8로 떨어지고 세 데이터 모두 최악 장비 오차가 증가했다.

### 재학습 없는 결과도 전부 보존

| 모델 | SUNWODA R² | RWTH R² | MICH R² |
|---|---:|---:|---:|
| 일정 속도 + RUL 학습 | .80356 | .62881 | .82080 |
| 속도 변화 + RUL 학습 | .70579 | .47917 | .26135 |
| 속도 변화 + 구간 학습 | .62936 | .66577 | -.24244 |
| 기존 PP-X | .91560 | .86740 | .32256 |
| Engression | -.40709 | .51411 | -4.14518 |

일정 속도/no-refit의 MICH .82080만 골라 개선으로 제시하지 않는다. 동일 모델이
나머지 데이터에서 PP-X에 미달하고, 주 후보도 사전에 다른 것으로 고정했다.

구간 학습은 재학습 전 RWTH에는 도움이 됐지만, 주 평가에서는 RUL-only 속도
모델보다 세 데이터 모두 낮아졌다. **이번 고정 설정에서 추가 학습 방식의 일관된
이득은 확인되지 않았다.** 왜 악화했는지는 아직 실험으로 분리하지 않았으므로
단조성 가정, rate proxy 오류, 손실 충돌 중 어느 하나로 단정하지 않는다.

세 버전 모두 5,123 parameters이다. 선택 epoch은 일정 속도 413/314/405,
속도 변화 257/280/351, 구간 학습 195/191/270이다. 9개 선택 학습 + 9개 refit의
측정 학습·예측·체크포인트 처리시간 합은 약84.4초(CPU 2 threads)이며 데이터
로딩/전체 프로그램 시작시간은 제외한다. epoch/optimizer update별 기록은
`results.json`에 저장했다.

## 무엇을 구현했나

PP-X 기본 경로와 분리된 실험 모듈 `prefix_speed_integral.py`를 추가했다.
현재 관측까지의 특징 z, 고장 경계까지의 margin m, 관측된 양의 열화속도 r을 받아
앞으로의 열화속도 변화를 학습하고 경계 도달시간을 적분해 RUL을 반환한다.

```
(a,b,c) = 2 tanh(MLP(standardize_train(z)))
q = (현재 margin - 미래 margin) / 현재 margin
미래 역속도 = exp(a + b q + c q²) / r
T(m → h) = (m-h)/r × ∫₀¹ exp(a + b f u + c (f u)²) du
f = (m-h)/m
RUL = T(m → 0)
```

2층 SiLU, 폭64, 출력3개이며 Gauss–Legendre 24점 적분을 쓴다. 모든 계수는
[-2,2]로 제한한다. 초기 출력층은 0이라 처음에는 m/r이다. a는 관측 rate의
보정이므로 현재 속도와 정확히 같도록 강제하는 구조는 아니다. b,c는 경계로
진행하면서 속도가 바뀌는 정도를 학습한다. 같은 prefix에서 더 먼 경계로의
이동시간은 증가하고 margin=0이면 RUL=0이다. 서로 다른 prefix의 예측 사이
일관성이나 실제 열화의 단조성까지 보장하지 않는다.

### 기존 실패 모델과 다른 부분

기존 `event_flow.py`도 역속도를 적분했다. 그 자체는 새 아이디어가 아니다.
이 구현은 causal observed rate를 기준으로 한 제한된 속도변화와 **실제 관측
구간 사이의 경과시간을 partial integral에 지도하는 학습**을 시험한다.
기존 ETO의 latent ODE/decoder나 frozen PP-X 뒤에 붙이는 잔차 모델은 아니다.

각 source train 장비에서 시간순 anchor와 이후 endpoint를 연결한다. offsets는
사용 가능한 window index 기준 1/4/16이고 실제 cycle 차이가 8 이상인 쌍만 쓴다.
health가 감소한 쌍을 사용하며, 실제 경과시간을 target으로 삼는다. suffix의
health는 적분 끝점의 **학습 target**일 뿐 입력 encoder에 주지 않는다.
test RUL은 미래 health 없이 알려진 경계 h=0만으로 계산한다.

이 쌍 생성은 RUL 정답을 읽지 않는다. source의 관측 뒤 구간만 이용한다.
다만 본 모델의 주 손실에는 다른 RUL 모델처럼 source train RUL 정답이 들어가므로
전체 학습을 self-supervised 또는 라벨 불필요 모델이라고 부르지 않는다.

## 고정 제거 실험

1. `constant_rul`: b,c를 0으로 마스킹, RUL 손실만 사용.
2. `variable_rul`: a,b,c 모두 학습, RUL 손실만 사용.
3. `variable_prefix`: a,b,c 학습 + 실제 구간 경과시간 손실. 사전에 지정한 주 후보.

세 버전 모두 같은 network parameter 수와 초기화 규칙을 사용하지만 constant의
b,c 출력은 비활성이다. source pair 학습의 추가 계산량은 동일 FLOPs가 아니다.

주 손실은 dataset/unit 균등 가중 normalized RUL MSE다. 추가 손실은
`0.1 × weighted_mean(((T_segment - Δt)/(Δt + 0.1))²)`이며 시간은 해당
데이터셋의 원본 train RUL 표준편차로 정규화한다. 상대오차 안정화 상수와
가중치는 test 결과를 보기 전에 고정했다.

## 평가 범위와 한계

SUNWODA/RWTH/MICH의 원래 early-health train/validation과 더 낮은 health의
source test, 동일 3529 test rows를 유지했다. 세 데이터 모두 장비 분리 및
엄격한 lower-health support 분리를 검사했다. train 쌍 11,180개, refit의
train+validation 쌍 14,674개다. refit 쌍은 같은 장비 내에서만 만든다.

각 버전은 seeds42/43/44, cap500/patience70, validation dataset-macro RUL MSE로
epoch을 선택한다. 재학습 없음과 train+validation 고정 epoch 재학습을 모두
보고하며, **주 판정은 variable_prefix 재학습 후 결과**다. test-best 버전이나
데이터셋별 switching으로 주 후보를 바꾸지 않는다.

비교는 직전 factorial 진단의 같은 입력·정답·분할·정규화·seed수인 frozen
joint dual-scale PP-X와 공식 Engression 예측을 hash/행 일치 검사 후 재사용한다.
이는 해당 세 배터리 설정의 기존 완성된 PP-X 경로이지 약한 PP 코어가 아니다.
그러나 원래 전체 9+5 설정을 재현한 것은 아니다. 후보의 추가 source-transition
지도와 비교 모델의 native loss/optimizer 차이도 존재한다. 성능 선별을 통과하더라도
엄밀한 동등 예산·추가 정보 대조와 잠근 외부 코호트 검증 없이 최종 우월성을
주장하지 않는다.

알려진 고장 경계, 과거 관측 순서, 양의 causal rate가 필요하다. 따라서 원래
범용 입력 설정의 적용 범위를 모두 보존하지 못한다. MICH의 경계는 published
life label과 물리적으로 완전히 일치하는 threshold가 아니므로 모델의 구조 가정이
특히 취약할 수 있다. 확률적 예측구간 coverage는 제공하지 않는다.

학습 동역학과 적분 자체는 이미 [Neural ODE](https://proceedings.neurips.cc/paper_files/paper/2018/hash/69386f6bb1dfed68692a24c8686939b9-Abstract.html)
등의 선행연구가 있다. 이번 구현의 조합이 논문 수준의 새 기여라는 결론은 내리지
않는다. 완전한 최근접 문헌 대조와 제거 실험의 기여가 별도로 필요하다.

## 재현

```sh
PYTHONPATH=src python -m pytest -q tests/test_prefix_speed_integral.py
python experiments/prefix_speed_integral_screen.py
python experiments/verify_prefix_speed_integral_screen.py
```

screen은 기존 출력 디렉터리를 덮어쓰지 않는다. 결과 위치는
`results/prefix_speed_integral_screen_v1/`이다. 설정과 코드·비교 예측 hash,
원본 정규화 rows, source pair anchor/future index, seed별 예측, 선택 이력,
재학습 가중치 체크포인트를 저장한다. package 기본 export와 기존 PP-X 모델은
변경하지 않았다.

단위 테스트는 상수속도 해, 지수속도 해, 경계0, 양의 이동시간, gradient,
잘못된 입력 거부, 쌍의 시간순서/장비 분리/RUL-label 독립성, batch 독립성,
동일 seed 재현, 24/64점 수치적분 일치를 포함한다. 신규 13개 및 관련 기존
회귀 테스트를 합쳐 37개가 통과했다.

별도 검증도 완료했다. 저장된 30개 데이터셋별 결과의 R²/RMSE/장비 평균·최악
오차/seed·장비 coverage를 독립적으로 재계산했고, 18개 checkpoint를 다시
로드해 저장 예측과 일치를 확인했다. source pair 순서·소속 장비·target 일치,
평가 행 일치, 코드·비교 예측 hash 불변 검사를 통과했다.
검증 결과는 `verification.json`에 보관한다.
