# PP-X 승패 메커니즘·게이트·불확실성·통계 검증 감사

> **Canonical paper pointer:** 아래 내용은 legacy 공통 PP backbone과 초기 gate의
> 역사적 감사다. 현재 논문 메인은 PP-X이며 동결 Algorithm 1과 최신 판정은
> `PPX_FINAL_PAPER_MODEL_KO.md`,
> `PPX_TOP_JOURNAL_VALIDATION_PACKAGE_KO.md`를 우선한다. 이후
> support-scaled 90% block-conformal coverage가 실행됐지만 독립 prospective
> predictive superiority는 아직 확보되지 않았다.

## 판정

현재 저장소는 다중 데이터셋 성능과 일부 구조 ablation은 강하지만, “어떤
관측 가능한 조건에서 legacy PP backbone이 plain MLP보다 유리한가”를 하나의
보편적 사전 법칙으로 확정하지 못했다. 이 실패가 PP-X에서 typed contract와
validation-approved executor/fallback을 분리한 근거다.

| 요구사항 | 현재 판정 | 실제 근거 |
|---|---|---|
| 외삽 거리·이질성·SNR·곡률로 승패 설명 | **미검증** | hull 거리는 전 데이터셋에 있으나 나머지 세 변수의 공통 정의와 matched plain-MLP 결과가 전 데이터셋에 없음 |
| test label 이전 적용 가능한 applicability gate | **부분검증/확증 실패** | validation-only gate와 pretest 기록은 있으나 XJTU 절대 성능 실패, Oxford 사전 성공 기준 실패 |
| 예측구간과 hull-distance별 coverage | **미검증** | OOF uncertainty head와 risk–coverage는 있으나 conformal interval empirical coverage가 없음 |
| PP 대 plain MLP 통계 유의성 | **부분검증** | 이번에 HUST·Virkler·NASA의 unit-paired bootstrap 및 sign-flip 검정 추가; 전체 데이터셋 matched 비교는 없음 |

## 현재 전제의 수정

“NASA·Virkler 승, HUST·XJTU 패”는 하나의 모델 버전을 가리킬 때만 맞다.

- 동일 입력·예산의 **기본 affine-tail PP 대 plain MLP**에서는 HUST가 패배하고 Virkler·NASA가 승리한다.
- PP-X의 paper-selected HUST route는 validation-only regime transport를 추가해 pooled ensemble \(R^2=0.958\)까지 개선됐다.
- XJTU는 PP가 일부 비교모델보다 상대적으로 높아도 \(R^2=-1.229\)이므로 절대 실패다.

따라서 설명할 outcome은 둘로 나눠야 한다.

1. 공통 backbone의 이득: \(\Delta\mathrm{RMSE}=\mathrm{RMSE}_{MLP}-\mathrm{RMSE}_{base\ PP}\)
2. 최종 시스템의 성공: PP-X paper-selected route의 pooled \(R^2>0\) 및
   strongest comparator 대비 regret

두 outcome을 섞으면 HUST처럼 backbone은 지지만 적절한 executor가 살리는 경우를 설명할 수 없다.

## 새로 계산한 PP 대 matched plain MLP 통계

행을 독립 표본으로 취급하지 않았다. 각 physical unit에서 5개 seed RMSE를 먼저 평균하고 unit을 bootstrap/sign-flip 단위로 사용했다.

| 데이터셋 | unit | PP 승 unit | 평균 RMSE 감소 | unit-bootstrap 95% CI | paired sign-flip p | 판정 |
|---|---:|---:|---:|---:|---:|---|
| HUST | 16 | 8/16 | −5.914 | [−17.935, 5.111] | 0.350 | 기본 PP 우월 근거 없음 |
| Virkler | 10 | 10/10 | +6.892 | [5.989, 7.821] | 0.00195 | 유의한 PP 우세 |
| NASA battery | 4 | 4/4 | +2.305 | [0.855, 4.997] | 0.125 | 효과는 일관되나 exact 검정 표본력 부족 |

NASA의 bootstrap CI가 0을 제외하더라도 unit이 4개뿐이라 이를 강한 유의성 증거로 쓰면 안 된다. 네 unit의 모든 부호가 같아도 양측 exact sign-flip 검정의 최소 p-value가 0.125이기 때문이다.

## 승패 설명변수 분석이 아직 성립하지 않는 이유

요청한 설명변수는 네 개지만, 현재 matched PP 대 plain MLP 결과는 HUST·Virkler·NASA 세 데이터셋뿐이다. 3개 관측치로 4개 변수를 회귀하는 것은 식별 불가능하다. NASA·Virkler·HUST·XJTU·MATR 다섯 개만 사용해도 절편을 포함하면 잔여 자유도가 0이다. 이 상태에서 상관계수나 회귀계수를 보고하면 설명이 아니라 데이터 암기가 된다.

논문용으로 유효한 분석은 다음 조건을 만족해야 한다.

1. 최소 10–12개 데이터셋 모두에서 같은 정보·seed·튜닝 예산의 plain MLP와 backbone PP를 실행한다.
2. 설명변수는 test label 없이 train과 관측 history에서 계산한다.
3. 네 변수를 사전 고정한다.
   - `normalized horizon`: 마지막 관측 health에서 사전 정의된 failure boundary까지의 거리 / train health SD
   - `trajectory heterogeneity`: unit별 정규화 열화율의 median absolute deviation
   - `degradation SNR`: train trajectory의 smooth trend variance / first-difference residual variance
   - `curvature`: unit-held-out train에서 quadratic smoother가 affine fit보다 줄인 상대 오차
4. 데이터셋 수가 적으므로 p-value 중심 다중회귀보다 rank correlation, bootstrap CI, leave-one-domain-out prediction을 주 분석으로 둔다.
5. 후보 네 변수 중 하나를 선택한 뒤에는 별도 데이터에서 threshold를 고정한다. 같은 데이터로 변수 선택과 성능 평가를 동시에 하면 사후 설명이다.

## 사전 게이트 검증 판정

현재 validation-only gate는 test label을 입력으로 쓰지는 않지만, threshold와 feature가 이미 본 데이터 결과를 바탕으로 개발됐다. 따라서 실행 시점에는 사전이어도 연구 전체 관점에서는 retrospective gate다.

- XJTU: pretest decision 기록은 있으나 모든 모델의 절대 성능이 음수라 성공 사례가 아니다.
- Oxford: 외부 outcome-held-out 실험에서 PP가 비교모델보다 높았지만 \(R^2=-0.119\)로 사전 정의한 양수 기준을 통과하지 못했다.
- MATR batch 2: sealed confirmatory 기록과 양의 성능은 있으나 PP 연구 전체에서 완전히 새로운 외부 데이터셋은 아니다.

그러므로 “사전 게이트가 untouched 외부 데이터에서 성공했다”는 주장은 아직 할 수 없다. 로컬에 남아 있는 데이터 중 하나를 다시 떼어 untouched라고 부르는 것도 정당하지 않다. 다음 외부 cohort 전에는 위 네 meta-feature, 결측 처리, gate threshold, 성공 기준을 코드와 protocol hash로 고정해야 한다.

## 불확실성 검증 판정

`oof_uncertainty_pp.py`는 nested OOF disagreement target으로 단일 inference-time uncertainty head를 학습하고 error correlation 및 risk–coverage를 계산한다. 이는 “불확실성 모델이 전혀 없다”는 비판에는 반박 근거가 된다. 그러나 다음이 없으므로 prediction interval 주장은 불가능하다.

- validation/calibration residual로 정한 interval radius
- 80%, 90%, 95% nominal coverage별 empirical coverage
- physical-unit block conformal 또는 leave-one-unit-out calibration
- hull-distance shell별 coverage와 interval width

권장 방법은 support-adaptive normalized conformal이다.

\[
I(x)=\left[\hat y(x)-q_{1-\alpha}\,s(x),\;\hat y(x)+q_{1-\alpha}\,s(x)\right],
\qquad
s(x)=\epsilon+\hat u(x)+\lambda d_{hull}(x)
\]

여기서 \(q\)는 validation unit을 block 단위로 사용한 nonconformity score에서만 구한다. test에서는 interval coverage를 한 번 평가하며 \(q,\lambda\)를 다시 조정하지 않는다.

## 논문에서 지금 가능한 주장

현재 증거로는 다음 정도가 정확하다.

> The affine-tail PP backbone shows a significant unit-level advantage over a matched plain MLP on Virkler, a consistent but underpowered advantage on four NASA battery units, and no advantage on HUST before regime transport. This contrast supports a modular rather than universally activated prior architecture, but a predictive cross-domain applicability law and calibrated extrapolation intervals remain to be confirmed.

강한 저널 주장으로 올리려면 우선순위는 다음과 같다.

1. 전 데이터셋 matched plain-MLP/backbone PP 결과를 확보한다.
2. 네 meta-feature를 train-only로 계산하고 leave-one-domain-out gate를 개발한다.
3. calibration unit이 충분한 데이터에 support-adaptive conformal interval을 적용한다.
4. 그 모든 규칙을 고정한 뒤 새 외부 cohort를 한 번 평가한다.

원시 통계 결과: `results/plain_mlp_paired_inference_v1/results.json`  
재현 코드: `experiments/plain_mlp_paired_inference.py`
