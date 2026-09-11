# Selective Regression equal-coverage audit protocol

작성자: 박진서  
상태: retrospective benchmark protocol, test 재계산 전 고정  
대상: PP-X 논문의 9개 메인 strict-extrapolation setting

## 목적

Noskov, Fishkov, Panov의 *Selective Nonparametric Regression via Testing*
Algorithm 1을 현재 train/validation/test split에 구현하고, PP-X와 같은
row coverage에서 accepted-set risk를 비교한다. 이 실험은 선택적 회귀가
strict out-of-support 조건에서 얼마나 예측을 유지하는지와, 유지한 예측의
오차가 어떤지를 동시에 본다.

## 데이터와 누출 방지

- `experiments/full_equal_candidate_budget.py::load_settings`의 9개 adapter를
  그대로 사용한다.
- train으로 평균·표준편차와 PCA를 적합한다. PCA 차원은
  `min(3, n_features, n_train-1)`로 고정한다.
- bandwidth 선택에는 validation label만 사용한다.
- test label은 최종 accepted-risk 계산에만 사용한다.
- PP-X 예측은 동결된 5-seed prediction artifact를 사용한다.

## Selective Nonparametric Regression

표준화·PCA 공간에서 Gaussian kernel을 사용하여 논문 식 (1)의
Nadaraya–Watson 평균, 조건부 분산 및 kernel density를 계산한다.

\[
\hat f(x)=\sum_i w_i(x)y_i,\qquad
\hat\sigma^2(x)=\sum_iw_i(x)y_i^2-\hat f(x)^2.
\]

Algorithm 1의 acceptance 조건을 그대로 계산한다.

1. \(\hat p(x)\ge 4a/(nh^d)\)
2. \(\hat\sigma^2(x)\le
   \lambda[1-z_{1-\beta}\lVert K\rVert_2
   \sqrt{2/(nh^d\hat p(x))}]\)

Gaussian kernel에 대해 \(b=1\),
\(a=(2\pi)^{-d/2}\exp(-1/2)\),
\(\lVert K\rVert_2^2=2^{-d}\pi^{-d/2}\)를 사용한다.
\(\beta=0.10\)으로 고정한다.

## Bandwidth

train PCA 좌표의 표본 중앙 최근접 거리 \(m\)을 기준으로
`h ∈ {0.5m, 1m, 2m, 4m}`을 평가하고 validation MSE가 가장 작은 값을 고른다.
동률이면 작은 bandwidth를 선택한다. 계산은 전 train row를 사용하되 kernel
행렬은 chunk로 계산한다.

## 동일 coverage 비교

Algorithm 1 부등식을 만족시키는 최소 abstention cost를 각 점의
acceptance score로 사용한다. density 조건이 실패하거나 우변 계수가 0 이하이면
score는 무한대로 둔다.

PP-X의 point confidence score는 5-seed prediction의 행별 표준편차를
validation target 표준편차로 나눈 값이다. 이는 label-free test score다.

각 setting에서 test label을 보지 않고 score가 작은 순서대로 정확히 같은
개수의 행을 선택한다.

- 목표 coverage: 25%, 50%, 75%, 90%
- 비교 A: Selective NW predictor + Algorithm-1 score
- 비교 B: frozen PP-X predictor + PP-X seed-disagreement score
- 비교 C: frozen PP-X predictor + Algorithm-1 score

C는 predictor 효과와 selector 효과를 분리하는 control이다. 동점은 원래 row
index로 고정한다.

## 지표

- setting별 accepted normalized RMSE:
  `accepted RMSE / full-test target SD`
- accepted \(R^2\)는 subset target 분산에 따라 불안정하므로 보조로만 저장
- physical-unit macro normalized RMSE
- accepted row coverage 및 적어도 한 행이 남은 unit coverage
- 9개 setting 동일가중 macro risk
- coverage별 setting win count와 paired exact sign test
- risk–coverage AUC(네 target coverage의 사다리꼴 적분)

## 판정

- PP-X의 macro accepted normalized RMSE가 Selective NW보다 작고
  9개 setting 중 7개 이상에서 우세하면 실증 장표에 포함한다.
- 그렇지 않아도 결과를 삭제하지 않고 Selective Regression의 적용 범위
  비교로 기록한다.
- 이 결과는 이미 열린 9개 setting의 retrospective audit이며 PP-X 구조 선택에
  다시 사용하지 않는다.
