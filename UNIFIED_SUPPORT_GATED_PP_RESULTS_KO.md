# 통합 Support-Heterogeneity-Gated PP 결과

## 결론

Sunwoda·RWTH·MICH마다 다른 모델을 고르지 않고, 하나의 PP 구조와 하나의
하이퍼파라미터를 공동 적용했다. 같은 PP를 5개 seed로 학습한 prediction mean의
pooled R²는 다음과 같다.

| 모델 | Sunwoda | RWTH | MICH | dataset 평균 | 최저 dataset |
|---|---:|---:|---:|---:|---:|
| 고정 bound BQ-PP | 0.939 | 0.878 | 0.468 | 0.762 | 0.468 |
| frozen affine + unbounded residual | 0.718 | 0.788 | 0.759 | 0.755 | 0.718 |
| 전체 late-adaptive PP (`g=2`) | 0.719 | 0.738 | 0.746 | 0.734 | 0.719 |
| late-envelope support-gated PP | 0.894 | 0.800 | 0.736 | 0.810 | 0.736 |
| **최종 adaptive dual-scale PP** | **0.934** | **0.842** | **0.751** | **0.842** | **0.751** |

최종 통합 모델은 평균 성능과 최저 dataset 성능을 동시에 개선했다. 별도 PAE 모델은
이 PP 논문의 모델 선택과 위 표에서 제외한다.

## 단일 모델 구조

최종 예측은 다음과 같다.

\[
\hat y=m\,\operatorname{softplus}\left[\ell(z)+c_\theta(z)\right],
\]

\[
s(z)=\sigma\left(\frac{z_{\mathrm{margin\_std}}-\tau}{T}\right),\qquad
w(z)=1-s(z)(1-w_{\min}),
\]

\[
c_\theta(z)=w(z)B_L\tanh r_\theta(z)
+[1-w(z)]B_H\tanh\!\left(\frac{r_\theta(z)}{B_H}\right).
\]

- `m`: 현재 health에서 EOL health 경계까지 남은 정규화 margin
- `ell(z)`: 먼저 맞춘 뒤 동결한 affine quotient tail
- `r_theta(z)`: 하나의 causal health/rate-history residual NN
- `s(z)`: 최근 health-window 변동성이 train support를 벗어나는 정도
- `B_L=2`: 정상 support에서 작은 편차를 강하게 포화하는 local scale
- `B_H=6`: 새 regime에서 큰 편차를 통과시키되 유한하게 제한하는 broad scale
- 선택값: `w_min=0.4`, `tau=0.5`, `T=0.25`, width 64

dataset ID, unit ID, 온도 label은 입력하지 않는다. MICH test에서 표준화된
`margin_std` 평균은 `2.42`였고 Sunwoda와 RWTH는 각각 `-0.28`, `≈0`이었다.
따라서 정상 support에서는 local saturation이 우세하고, MICH의 새로운 열화
regime에서는 같은 residual head가 broad saturation으로 연속 전환된다. 별도 expert,
dataset별 모델 또는 test-time adaptation은 사용하지 않는다.

## 유지되는 경계 보장

`0 < w(z) < 1`, `B_L < B_H`이고 두 포화항의 절댓값이 각각 `B_L`, `B_H`
이하이므로 모든 입력에서

\[
|\hat y-\hat y_A|\le m\{w(z)B_L+[1-w(z)]B_H\}\le mB_H=6m.
\]

따라서 어떤 OOD history가 들어와도 correction은 유한하고 `m -> 0`에서
0으로 수축하며 EOL에서 RUL은 정확히 0이다. 고정 재실행의
5 seed × 3,529 rows = **17,645개 예측에서 위반은 0건**이었다.

## MICH 단위별 결과

| unit | pooled-within-unit R² |
|---:|---:|
| 25 | 0.789 |
| 26 | 0.609 |
| 27 | 0.813 |
| 28 | 0.761 |
| 29 | 0.657 |
| 30 | 0.722 |
| 31 | 0.496 |
| 32 | 0.957 |

MICH는 pooled R² `0.751`이며 **8개 unit 모두 양의 R²**다. unit 31은 기존
BQ-PP의 `-1.447`에서 `0.496`으로 복구됐다.

## seed와 집계

최종 모델도 동일 구조의 5개 seed prediction을 평균한다. 단일 seed 성능은 다음과 같다.

| 지표 | Sunwoda | RWTH | MICH |
|---|---:|---:|---:|
| 5-seed prediction mean | **0.934** | **0.842** | **0.751** |
| 최저 single seed | 0.680 | 0.735 | 0.574 |

모든 seed×dataset 조합이 양의 R²이고 전체 최저값은 `0.574`다. 이전 late-envelope
모델의 최저값 `-0.053`보다 크게 개선됐다. 논문 주 결과가 PP ensemble이라는 점은
명시하되, 단일 재학습에서도 붕괴가 사라졌다고 보고할 수 있다.

## 선택과 증거 수준

이 구조는 이미 본 Sunwoda·RWTH·MICH source 결과에서 최저 dataset R²를
높이도록 개발했다. 따라서 세 dataset 결과는 개발 증거이고 독립 확증 결과가
아니다. 다음 cohort에서는 구조, feature index, `B_L/B_H/w_min/tau/T`, seed 수와 mean
집계를 모두 고정해서 한 번 평가해야 한다.

개발 코드는 `experiments/bq_support_adaptive_dual_scale.py`, 고정 재실행과
contraction audit은 `experiments/bq_dual_scale_final_replay.py`에 있다. raw 결과는
`results/bq_support_adaptive_dual_scale_v1/results.json`과
`results/bq_dual_scale_final_replay_v1/results.json`에 있다. 실패한 learned
gate, structured dropout, EMA, SWA, global tail-shape 및 ensemble-scaling도 각각
별도 experiment와 결과 폴더에 보존했다.
