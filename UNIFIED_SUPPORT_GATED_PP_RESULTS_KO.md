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
| **통합 support-gated PP** | **0.894** | **0.800** | **0.736** | **0.810** | **0.736** |

통합 모델은 평균 성능과 최저 dataset 성능을 동시에 개선했다. 별도 PAE 모델은
이 PP 논문의 모델 선택과 위 표에서 제외한다.

## 단일 모델 구조

예측은 다음과 같다.

\[
\hat y=m\,\operatorname{softplus}\!\left[
\ell(z)+B\{1+g(1-\operatorname{clip}(m,0,1))s(z)\}\tanh f_\theta(z)
\right],
\]

\[
s(z)=\sigma\!\left(\frac{z_{\mathrm{margin\_std}}-\tau}{T}\right).
\]

- `m`: 현재 health에서 EOL health 경계까지 남은 정규화 margin
- `ell(z)`: 먼저 맞춘 뒤 동결한 affine quotient tail
- `f_theta(z)`: causal health/rate-history residual NN
- `s(z)`: 최근 health-window 변동성이 train support를 벗어나는 정도
- 선택값: `B=2`, `g=3`, `tau=0.5`, `T=0.5`, width 64

dataset ID, unit ID, 온도 label은 입력하지 않는다. MICH test에서 표준화된
`margin_std` 평균은 `2.42`였고 Sunwoda와 RWTH는 각각 `-0.28`, `≈0`이었다.
따라서 MICH의 새로운 열화 regime에서만 envelope가 주로 열리고, 다른 두
dataset에서는 원래 보수적 PP tail을 유지한다.

## 유지되는 경계 보장

`0 < s(z) < 1`이고 `|tanh(f)| <= 1`이므로 모든 입력에서

\[
|\hat y-\hat y_A|
\le mB\{1+g(1-\operatorname{clip}(m,0,1))s(z)\}
\le mB(1+g).
\]

따라서 어떤 OOD history가 들어와도 correction은 유한하고 `m -> 0`에서
0으로 수축하며 EOL에서 RUL은 정확히 0이다. 기존 고정 `mB` envelope보다
넓지만 support heterogeneity가 없으면 expansion이 억제된다.

## MICH 단위별 결과

| unit | pooled-within-unit R² |
|---:|---:|
| 25 | 0.666 |
| 26 | 0.611 |
| 27 | 0.858 |
| 28 | 0.868 |
| 29 | 0.801 |
| 30 | 0.820 |
| 31 | -0.367 |
| 32 | 0.560 |

MICH는 pooled R² `0.736`, macro-unit R² `0.602`이며 8개 unit 중 7개가 양수다.
unit 31은 기존 BQ-PP의 `-1.447`보다 개선됐지만 완전히 복구되지는 않았다.
따라서 “모든 dataset에서 높다”는 dataset-level pooled 결과를 뜻하며 모든
개별 unit 성공을 뜻하지 않는다.

## seed와 집계

동일 구조의 5개 seed prediction을 평균했다. 대체 집계 결과는 다음과 같다.

| 집계 | Sunwoda | RWTH | MICH |
|---|---:|---:|---:|
| **mean** | **0.894** | **0.800** | **0.736** |
| median | 0.882 | 0.783 | 0.724 |
| trimmed mean | 0.878 | 0.792 | 0.692 |

단일 seed는 여전히 불안정하다. 선택 구조의 single-seed pooled R² 평균±SD는
Sunwoda `0.642±0.307`, RWTH `0.759±0.069`, MICH `0.247±0.274`다. seed 45는
`0.872/0.760/0.711`로 세 dataset 모두 높았지만 test를 보고 이 seed만 고르는
방식은 허용하지 않는다. 논문 주 결과는 **PP ensemble**로 명시해야 하며,
단일-deployment 안정성은 해결된 것으로 주장하지 않는다.

## 선택과 증거 수준

이 구조는 이미 본 Sunwoda·RWTH·MICH source 결과에서 최저 dataset R²를
높이도록 개발했다. 따라서 세 dataset 결과는 개발 증거이고 독립 확증 결과가
아니다. 다음 cohort에서는 구조, feature index, `B/g/tau/T`, seed 수와 mean
집계를 모두 고정해서 한 번 평가해야 한다.

재현 코드는 `experiments/bq_support_heterogeneity_gate.py`, 고정 재실행은
`experiments/bq_support_gate_final_replay.py`, raw 결과는
`results/bq_support_heterogeneity_gate_v1/results.json` 및
`results/bq_support_gate_final_replay_v1/results.json`에 있다. 실패한 learned
gate, structured dropout, global tail-shape 및 stability tuning도 각각 별도
experiment와 결과 폴더에 보존했다.
