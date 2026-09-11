# PP-X v1.2 group-OOF continuous portfolio

기록: 박진서, 2026-09-10.

## 최종 개발 구조

초기 adaptive shrinkage 안을 다시 검토한 결과, 같은 validation으로
architecture 선택과 gate 검정을 반복하고 단일 MLP로 후퇴하는 문제가 남았다.
최종 v1.2 개발안은 다음처럼 바꿨다.

1. trust=0의 4개 width×learning-rate, 각 5-seed 예측을 고정 평균해
   **baseline portfolio**를 만든다.
2. positive trust `.02,.05,.10,.20,.40`도 trust별로 4 architecture×5 seed를
   고정 평균한다.
3. validation physical unit 하나씩을 제외한 group-OOF fold에서 baseline과
   다섯 prior-route portfolio의 convex weight를 학습한다.
4. 목적함수는 unit 평균 MSE, 최악 20% unit의 baseline 대비 excess-loss
   CVaR, prior mass penalty, prior weight concentration penalty를 결합한다.
5. fold weight의 좌표별 중앙값을 정규화해 test 전에 최종 고정한다.

고정 regularization은 `rho=.50`, `tau=.02`, `gamma=.01`,
CVaR fraction `.20`이다.

## 왜 이 구조인가

- 단일 validation architecture winner를 없애 selection variance를 줄인다.
- 행 수가 많은 unit이 결정을 지배하지 않도록 unit-balanced loss를 쓴다.
- 이진 승인 대신 여러 약한 prior route를 연속 가중한다.
- 한 unit의 baseline RMSE가 거의 0일 때 폭발하는 단일 worst-ratio 대신
  worst-tail excess-loss CVaR를 사용한다.
- test label은 weight 선택 함수에 전달되지 않는다.

각 trust route는 따로 재학습되므로 nominal trust는 실제 affine 혼합률과
동일하지 않다. 보고하는 effective trust는 weight×nominal trust의 설명값이지
인과적인 affine 기여율이 아니다.

## Retrospective 결과

| Cohort | 단일 winner MLP R² | Baseline portfolio R² | v1.2 portfolio R² | Prior mass | Effective trust |
|---|---:|---:|---:|---:|---:|
| Stanford | 0.034 | **0.071** | **0.071** | 0.000 | 0.000 |
| ISU 250 mAh | 0.451 | 0.480 | **0.492** | 1.000 | 0.0299 |

Stanford의 8개 OOF fold는 모두 prior mass 0을 선택했다. ISU의 45개 fold는
모두 낮은-trust prior portfolio를 선택했고, 최종 weight는 trust .02에
0.671, trust .05에 0.329였다. 높은 trust `.10–.40`은 0이다.

ISU에서 prior mass가 1이어도 effective trust는 약 0.03이다. 이는
“강한 prior로 전환”이 아니라 여러 low-trust residual route를 평균한 것이다.

## 기존 안과 비교

- v1.1 hard gate: unsupported prior를 버리지만 단일 MLP가 남음
- v1.2 초기 shrinkage: 연속 weight이나 동일 validation 재사용과
  single-winner 문제가 남음
- v1.2 continuous portfolio: architecture/seed portfolio +
  group-OOF + continuous robust stacking

두 고호트 모두 기존 단일 MLP보다 높았지만 이미 본 test를 사용하는
post-test development다. 최종 채택에는 새 미개봉 고호트 한 번이 필요하다.

## 구현

- `src/pp_extrapolation/continuous_portfolio.py`
- `experiments/ppx_v12_continuous_portfolio_development.py`
- `results/ppx_v12_continuous_portfolio_development/`
- `tests/test_continuous_portfolio.py`
- 차기 고정 계약:
  `protocols/NEXT_UNTOUCHED_PPX_V12_CONTINUOUS_PROTOCOL.md`
