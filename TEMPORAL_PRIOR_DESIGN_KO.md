# PP 시계열 prior 조립 및 보정: 연구 프로토타입

현재 scalar PP와 별개의 실험 모델이다. 기존 방식의 성능 보장이나 새로운 논문 기여가 확립된 상태가 아니다. 사용자 의도에 맞춰 도메인 경계, 과거 추세, 변화 징후, NN 보정을 연결한다.

## 도메인 설정

`DegradationContract`는 `boundary`, `direction`, short/long window, sequence window를 받는다. 배터리는 감소하는 용량과 고장 용량, 균열은 증가하는 길이와 임계 길이, 마모는 증가하는 마모량과 한계를 선언한다.

관측값은 최초 관측에서 1, 고장 경계에서 0인 무차원 좌표 q로 변환된다. 물리 단위 변경은 동일한 좌표를 준다. 이것은 모델이 도메인 이름만 보고 과학적 법칙을 자동 발견한다는 의미가 아니다. 의미 있는 열화 지표와 경계가 필요하다. 비단조 다변수 센서만 있는 기계 도메인은 추가 상태추정기가 필요하다. Paris/Arrhenius 등 별도 법칙 플러그인은 이번에 구현하지 않았다.

## prior 생성과 온라인 신뢰도

각 관측 시점 t에서 과거 8개/24개 관측을 이용해 q의 OLS 감소 속도를 추정한다. 각 속도가 유지된다는 가정에서 `q / rate`로 고장 경계까지의 시간을 계산한다. 속도가 양수가 아니거나 관측이 4개 미만이면 해당 prior는 미확정으로 취급해 학습 RUL 상한으로 대체한다. 현재 구현은 결국 trend-to-threshold 가정을 포함하므로 formula-free라고 부르지 않는다.

이전 시점에 발행한 한 단계 q 예측이 지금 관측과 얼마나 다른지 EWMA로 누적한다. 짧은/긴 추세의 신뢰도는 이 관측 가능한 예측오차를 상대 비교해 산출한다. 미래 RUL 정답을 사용하지 않는다. 단기 관측 예측오차가 작은 prior가 장기 RUL에도 정확하다는 보장은 없다.

## 시계열 네트워크

입력은 길이 16의 과거 feature sequence다. 각 시점의 feature는 q, 경과 시간, short/long rate, short/long 예측오차, 추세 불일치, rate 유효성 두 개로 총 9개다. 초기 부족분은 최초 관측을 왼쪽으로 반복해 채운다. 모든 행은 자신의 prefix만 사용한다. 정규화는 train 행의 현재 feature 통계로 계산한다.

GRU(9 inputs, 16 hidden)에서 얻은 상태 h를 이용해 prior의 비중을 조절하고 RUL을 보정한다.

    gate = softmax(log(online_reliability) + gate_head(h))
    prediction = sum(gate * prior_RUL) + 0.25 * train_RUL_cap * tanh(residual_head(h))

gate head와 residual head는 0으로 초기화해서 시작 예측은 online-reliability prior와 같다. 최종 출력은 다른 모델과 동일하게 [0, max train RUL]로 clip한다. 보정 폭 제한 자체도 새로운 가정이다. 두 prior가 모두 틀리면 충분한 보정이 안 될 수 있고, 자동 abstention이나 예측구간은 아직 없다.

학습은 unit-balanced RUL MSE, AdamW lr=5e-4/weight_decay=.05, 300 epochs/patience70, gradient clip2를 쓴다. validation MSE로 checkpoint를 선택한다. 이번 버전은 추가 physics loss를 쓰지 않는다. 기존 실패한 변화율 정규화를 다시 넣지 않고 prior가 예측 구조에 제공하는 정보를 먼저 검증한다.

## regime 관측

추세 불일치는 `abs(short_rate-long_rate)/(abs(short_rate)+abs(long_rate)+1e-6)`다. 이는 관측상의 변화 징후이며 regime posterior, 실제 전환 확률, 인과효과가 아니다. 센서 잡음·회복·불규칙한 관측도 신호를 만들 수 있다.

합성 감사는 short/long window와 threshold=.35, 연속 3점 조건을 고정해 stable/abrupt/gradual 세 경우를 측정한다. 진짜 전환 시점은 75이고 10 seeds를 사용한다. NASA에는 별도 검증된 regime 라벨이 없으므로 NASA의 전환 검출 정확도를 주장하지 않는다. 미래에 아직 관측되지 않은 전환은 이 모델로 알아낼 수 없다.

## 분리 비교

- PP_history_stats: 동일한 9개 현재 시점 요약 feature를 사용하는 원래 PP 구조
- GRU_direct: 같은 16x9 이력을 받아 RUL을 직접 예측
- prior_only: 과거 예측오차로 prior 두 개를 결합, NN 없음
- GRU_fixed_gate: prior gate를 고정하고 GRU 잔차만 학습
- GRU_corrected: GRU gate와 잔차를 같이 학습

같은 NASA health-v2 4 folds, 255 hull-out test 행을 사용한다. 각 fold에서 unit은 완전히 분리된다. 4개 학습 arm x 5 seeds x 4 folds = 80 fits이며 prior-only는 결정론적 계산이다. 이전 scalar PP는 과거정보를 덜 받으므로 참고점수다. 그 점수와의 상승만으로 구조의 우수성을 주장하지 않는다. 입력 정보와 구조 비교를 분리한다.

## 실행

```bash
python experiments/prepare_nasa_temporal.py --research-root /path/to/ca-css-ncmapss --output /tmp/pp_nasa_temporal_folds.npz
python experiments/nasa_temporal_prior.py --folds /tmp/pp_nasa_temporal_folds.npz --output results/nasa_temporal_prior_v1
python experiments/regime_signal_audit.py
```

adapter의 원래 NASA 로더는 로컬 원본 데이터를 요구한다. 연구 결과를 재사용한 post-hoc 실험이며 새 데이터의 confirmatory 검증이 아니다. 시계열의 causality는 prefix를 잘라도 이전 행의 feature/prior가 동일한지 검사한다. 기존 원본의 1.401 Ah 라벨 판정과 1.4 Ah 좌표를 그대로 유지했다.

## 선행연구와 남은 과제

물리 prior와 NN 결합, switching dynamics, mixture-of-experts 자체는 기존 연구가 있다.

- [HybridNet, CoRL 2018](https://proceedings.mlr.press/v87/long18a.html)
- [Hybrid system identification using switching density networks, CoRL 2019 proceedings](https://proceedings.mlr.press/v100/burke20a.html)
- [Physics-informed neural network for lithium-ion battery degradation, Nature Communications 2024](https://www.nature.com/articles/s41467-024-48779-z)

연구 기여 후보는 도메인별 prior의 적용 조건과 실패를 명시하고, 현재까지의 관측 예측오차로 신뢰도를 갱신하며, 외삽 거리 및 관측된 전환 전후에서 NN 보정의 효과를 검증하는 것이다. 이를 새로운 방법이라고 확정하려면 더 폭넓은 선행연구 비교와 미사용 시계열 검증이 필요하다.
