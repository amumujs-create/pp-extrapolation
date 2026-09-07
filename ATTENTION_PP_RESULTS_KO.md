# Transformer 패배 데이터의 attention-PP 개선 결과

## 목표

MATR2019에서 기존 latent PP의 pooled ensemble R² 0.257이 동일 예산 FT-Transformer 0.331보다 낮은 문제를 다뤘다. Transformer 예측을 사후 평균하는 방식 대신 attention을 PP 내부 표현 모듈로 사용하고, affine tail·support distance·counterfactual Jacobian prior를 결합했다.

## 실험 결과

| 모델 | MATR2019 pooled ensemble R² | 판정 |
|---|---:|---|
| 기존 latent PP | 0.257 | 기준 PP |
| 동일 예산 FT-Transformer | **0.331** | 강한 비교 모델 |
| temporal latent PP | 0.203 | 실패 |
| affine + GRU residual PP | -0.236 | 실패 |
| **attention-tail PP** | **0.331** | beta=0으로 FT를 정확히 복원 |
| attention + affine-Jacobian prior, 모든 seed | 0.259 | 실패 |
| validation-selective Jacobian attention PP | 0.313 | FT보다 낮아 미채택 |

과거 150-epoch 공통 벤치마크의 FT-Transformer 0.344는 탐색 예산과 선택 규칙이 달라 위 300-epoch 동일 예산 표와 직접 섞지 않는다.

## 최종 구조 판단

attention-tail PP는 다음 식을 사용한다.

\[
\hat y=\hat y_{\mathrm{affine}}+
\exp(-\beta d)\left(\hat y_{\mathrm{attention}}-\hat y_{\mathrm{affine}}\right)
\]

`d`는 train support 밖 거리이고 `beta`는 validation ensemble 성능과 validation unit 일관성으로 선택한다. `beta=0`이면 attention predictor를 정확히 복원한다. MATR2019, HUST deep-future, Virkler deep-future 모두 beta=0을 선택했다. 따라서 이 세 split에서는 prior 전환의 validation 근거가 없었다.

이 구조를 사용하면 MATR에서 PP 계열이 Transformer보다 낮은 문제는 fallback으로 해소된다. 그러나 beta=0이므로 이 결과를 PP prior의 정확도 이득으로 주장할 수 없다. 논문에서는 attention을 local executor로 허용하는 modular PP와, prior가 승인될 때만 affine extrapolator로 전환하는 규칙을 핵심으로 정의해야 한다.

## 실패 원인

- 8-step window에 GRU와 두 latent expert를 동시에 넣으면 seed 분산이 커지고 validation 선택이 test unit으로 전달되지 않았다.
- 단일 GRU residual도 약한 affine 기준을 충분히 교정하지 못했다.
- label-free future ray에서 attention Jacobian을 affine 기울기에 맞추는 loss는 seed 42 validation MSE를 701에서 632로 낮췄지만, seed 42 test R²는 0.283에서 0.281로 소폭 하락했다.
- 같은 penalty를 모든 seed에 적용하면 seed 44가 -0.164로 붕괴했다. validation fallback을 적용해도 ensemble은 0.313이었다.

## 남은 연구 문제

MATR에서 FT보다 유의하게 높아지려면 더 많은 test 맞춤 튜닝이 아니라 별도의 개발 cohort가 필요하다. 그 cohort에서 dual-view attention, 더 긴 causal history, regime-change auxiliary target을 개발하고 구조를 고정한 뒤 MATR 이외의 untouched batch에 적용해야 한다. 이미 관측한 MATR test로 계속 구조를 선택하면 수치는 오를 수 있어도 논문 증거는 약해진다.
