# Ferrara bearing PP 개선 결과

## 결론

동결된 최초 E5/E6 평가에서 BQ-PP는 pooled R² `-2359.980`으로 실패했다. 실패 후 개발한
**regime-conditioned log-boundary-quotient PP**는 같은 E5/E6에서 pooled R² `0.539`,
RMSE `520.9 min`을 기록했다. 이 결과는 최초 실패를 대체하는 확증 결과가 아니라, 이미
E5/E6 라벨을 확인한 뒤 수행한 **후향적 모델 개발 결과**이다.

| 모델 | pooled R² | RMSE (min) | unit-macro R² |
|---|---:|---:|---:|
| 동결 BQ-PP | -2359.980 | 37276.5 | -4640.460 |
| 동결 plain MLP | -5.871 | 2010.9 | -19.379 |
| 동결 FT-Transformer | -151.444 | 9472.0 | -227.281 |
| 동결 boundary-rate | -0.227 | 849.9 | -0.325 |
| log-quotient affine | 0.512 | 535.7 | -1.070 |
| log-quotient PP | 0.506 | 539.1 | -0.874 |
| **regime log-quotient PP** | **0.539** | **520.9** | **0.006** |

## 무엇을 바꿨는가

기존 BQ-PP는 다음 raw quotient를 학습했다.

\[
q(x)=\frac{RUL}{m},\qquad \widehat{RUL}=m\,\operatorname{softplus}(a(x)+r_{NN}(x))
\]

여기서 `m=20g-running peak`이다. Ferrara에서는 같은 margin이 서로 다른 bearing에서 매우
다른 시간척도를 뜻하기 때문에 `q`가 수십 배 이상 달라졌다. 여기에 개발 자료에서 뒤집힌
load-life 상관을 affine path가 외삽하면서 E5/E6 예측 중앙값이 약 45,242분까지 발산했다.

개선 구조는 raw quotient 대신 log quotient를 학습한다.

\[
z(x)=\log\left(1+\frac{RUL}{m}\right),\qquad
\widehat{RUL}_{LQ}=m\{\exp(a(x)+r_{NN}(x))-1\}
\]

이 변환은 unit 간 시간척도 차이를 압축하면서 `m=0`에서 RUL이 정확히 0이 되는 경계 조건을
유지한다. 잘못된 transport를 막기 위해 load와 상수 speed는 입력에서 제외했다. elapsed time과
모든 vibration/history 변수는 예측 시점까지의 값만 사용한다.

E5는 고장 직전까지 RMS와 대역에너지가 거의 변하지 않는 충격형 궤적이고, E6는 수명
72--82%부터 점진적 진동 증가가 관측됐다. 하나의 quotient만으로 두 양상을 처리하기 어려워,
초기 100개 관측의 robust baseline 대비 RMS/고주파 에너지가 5 MAD를 넘으면 local boundary-rate
경로를 켜는 causal regime gate를 추가했다.

\[
\widehat{RUL}=(1-g)\widehat{RUL}_{LQ}
 +g\frac{m}{\max(v_{local},v_{floor})}
\]

이는 서로 독립적으로 고른 기성 모델의 사후 ensemble이 아니라, 알려진 failure boundary를
공유하는 두 PP 실행 경로를 관측된 regime 증거로 선택하는 구조이다.

## 수정된 평가 의미

동결 프로토콜은 개발 bearing도 앞 70%만 학습에 사용했다. 하지만 test의 convex-hull 밖 비율은
24.6%에 불과해 엄밀한 state-hull 외삽 기준을 통과하지 못했고, 동시에 어떤 bearing에서도
late failure regime을 학습하지 않는 문제가 있었다. 개선 실험은 다음과 같이 estimand를 명확히
바꿨다.

- 하이퍼파라미터: E1--E3의 앞 70%로 학습, 동일 unit의 뒤 30% shell로 선택
- 최종 refit: E1--E4의 완전한 run-to-failure 궤적
- 평가: 처음 보는 E5--E6의 뒤 30%, 과거 history만 제공
- 해석: 다차원 hull 외삽이 아니라 **unseen-bearing/unit 외삽**

alpha는 개발 shell에서 `10`으로 선택됐고, NN stopping epoch만 seed별로 선택했다. seed별 PP
pooled R²는 `0.571, 0.617, 0.246, 0.593, 0.359`였으며 5-seed 평균 예측을 사용했다.

## 남은 한계

| unit | regime PP R² | 해석 |
|---|---:|---|
| E5 | 0.404 | 충격형 궤적에서 log-scale 안정화가 작동 |
| E6 | -0.391 | local-rate 분기로 크게 개선됐지만 개별 양의 R²에는 미달 |

pooled와 macro가 모두 양수가 됐지만 unit이 2개뿐이고, 구조 개발 전에 E5/E6를 이미 봤다.
따라서 이 수치는 모델 개선 가능성의 증거이지 독립 외부 확증은 아니다. 저널의 주효과 표에는
`posthoc development`로 표시하고, 같은 구조를 새 bearing cohort에 고정 적용해야 한다.

## 재현

```bash
python experiments/ferrara_log_quotient_pp_development.py
```

결과는 `results/ferrara_log_quotient_pp_posthoc_v1/results.json`과 `predictions.npz`에 저장된다.
