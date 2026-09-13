# 약한-prior projection authority synthetic 실험

작성자: 박진서  
프로토콜:
`protocols/WEAK_PRIOR_PROJECTION_AUTHORITY_SYNTHETIC_PROTOCOL.md`

## 결론

약한 비음수·단조·최소기울기 prior를 differentiable projection으로 구현하고,
source pseudo-tail evidence에서 continuous authority를 예측하는 신경망 head를
결합했다.

구조는 의도대로 작동했지만 **현재 PP-X로 승격하지 않는다.**

- 약한 prior가 실제로 맞는 30개 task에서는 hard projection이 direct MLP보다
  평균 RMSE를 25.3% 줄였다.
- prior가 unseen tail에서 깨지는 30개 task에서는 hard projection이 RMSE를
  42.4% 악화했다.
- 학습형 authority는 hard projection의 위해를 줄였지만 전체 task에서
  direct 대비 평균 RMSE가 2.45% 악화됐다.
- authority와 test-oracle authority의 상관은 0.034로 사실상 0이었다.

따라서 forward-level 모델 노벨티는 구현됐지만 prior falsification 능력과
전체 성능은 입증되지 않았다.

## 모델

base MLP의 ordered extrapolation-ray 예측을 \(z_\theta\)라 할 때,
weak-prior projection은 비음수성과 source에서 추정한 최소 증가량을 강제한다.

\[
\Pi_{\mathcal C}(z_\theta)
=
\operatorname{cummax}
\left(
z_\theta-k\delta
\right)
+k\delta.
\]

authority head는 source 및 source-pseudo-tail feature만 사용한다.

\[
\hat y
=
z_\theta
+a_\phi(s)
\left[
\Pi_{\mathcal C}(z_\theta)-z_\theta
\right],
\qquad a_\phi(s)\in[0,1].
\]

\(a=0\)이면 exact direct fallback이고 \(a=1\)이면 hard weak-prior
projection이다.

## 실험

- authority development task: 180개
- 미사용 synthetic confirmation task: 60개
- weak prior valid: 30개
- weak prior violated in unseen tail: 30개
- base model: 모든 arm에서 동일한 two-layer tanh MLP
- authority target: development task의 tail-oracle alpha
- confirmation tail outcome은 authority feature 및 head 학습에 미사용

## 전체 결과

| arm | mean RMSE | direct 대비 평균 변화 | task win | worst-20% harm | mean authority |
|---|---:|---:|---:|---:|---:|
| direct MLP | 0.3681 | 기준 | 0% | 0 | 0 |
| hard projection | 0.3568 | **8.57% 악화** | 51.7% | 80.1% | 1.000 |
| pseudo-tail authority | 0.3568 | **8.57% 악화** | 51.7% | 80.1% | 1.000 |
| ablated head | 0.3602 | 2.70% 악화 | 53.3% | 37.4% | 0.512 |
| full authority head | 0.3607 | 2.45% 악화 | 53.3% | 34.9% | 0.484 |
| tail oracle | 0.3126 | **13.8% 개선** | 53.3% | 0% | 0.520 |

mean RMSE 자체는 hard projection이 direct보다 작지만 task별 상대 RMSE를 동일
가중하면 8.57% 악화다. valid task의 큰 direct error가 pooled mean을 지배하기
때문이다. 논문 판단에는 equal-task relative effect와 worst-tail harm을
우선한다.

## Prior-valid task

| arm | mean RMSE | direct 대비 변화 | win |
|---|---:|---:|---:|
| direct | 0.4334 | 기준 | 0% |
| hard projection | 0.3250 | **25.3% 개선** | 100% |
| full authority | 0.3806 | **12.3% 개선** | 100% |
| oracle | 0.3250 | **25.3% 개선** | 100% |

weak prior가 맞으면 projection은 30/30 task에서 모두 유효했다. 모델이 weak
constraint를 외삽 구조로 이용할 수 있다는 메커니즘은 확인됐다.

## Prior-violated task

| arm | mean RMSE | direct 대비 변화 | win |
|---|---:|---:|---:|
| direct | 0.3028 | 기준 | 0% |
| hard projection | 0.3887 | **42.4% 악화** | 3.3% |
| full authority | 0.3409 | **17.2% 악화** | 6.7% |
| oracle | 0.3003 | **2.27% 개선** | 6.7% |

continuous authority는 hard projection의 악화를 절반 이하로 줄였지만 완전히
거절하지 못했다.

## 실패 원인: source에서 식별할 수 없는 prior violation

실험의 invalid task는 \(x>0.60\)에서만 무작위 tail reversal이 시작된다.
reversal 크기는 source 함수와 독립적으로 생성했다. 따라서 \(x\le0.60\)의
source 및 pseudo-tail evidence만 보면 valid task와 invalid task를 구별할
정보가 없다.

이 때문에:

- pseudo-tail selector는 모든 task에서 hard projection을 선택했다.
- full head의 validation epoch는 1에서 끝났다.
- authority는 valid 0.485, invalid 0.482로 거의 같았다.
- oracle authority correlation은 0.034였다.

이는 단순한 모델 용량 부족이 아니라 **식별 불가능성**이다. unseen tail에서만
독립적으로 발생하는 prior 위반은 어떤 source-only gate도 안정적으로 예측할
수 없다.

## 모델링적 교훈

약한 prior 모델은 두 상황을 구분해야 한다.

1. **Falsifiable weak prior:** source trajectory에 위반 징후가 존재한다.
   이 경우 pseudo-tail falsification과 continuous authority를 학습할 수 있다.
2. **Non-falsifiable weak prior:** 위반이 unseen tail에서 독립적으로 나타난다.
   이 경우 point prediction authority를 학습한다고 주장하면 안 된다.
   set-valued prediction, interval 확대 또는 abstention이 필요하다.

따라서 다음 구조는 단순 authority head가 아니라
**identifiability-aware weak-prior model**이어야 한다.

\[
\text{falsifiability evidence present}
\Rightarrow
\text{learn authority},
\]

\[
\text{falsifiability evidence absent}
\Rightarrow
\text{credal interval or abstention}.
\]

이 분기는 외삽에서 약한 prior가 어디까지 검증 가능한지를 모델 출력에 직접
반영하므로 더 강한 모델적 기여가 될 수 있다.

## 판정

- differentiable weak-prior projection: 구현 및 작동 확인
- exact \(a=0\) fallback: 확인
- prior-valid task 개선: 통과
- violated-prior harm 감소: hard projection 대비 부분 통과
- direct 대비 전체 개선: 실패
- authority-oracle calibration: 실패
- pseudo-tail falsification: 실패
- 실제 데이터 검증: 미실시

현재 구조는 **기각된 mechanism challenger**로 보존한다. 다음 실험은 source
precursor가 있는 falsifiable violation과 precursor가 없는 non-falsifiable
violation을 분리하고, 후자에는 point authority가 아니라 set prediction을
출력해야 한다.
