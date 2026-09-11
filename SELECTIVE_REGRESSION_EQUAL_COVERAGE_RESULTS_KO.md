# Selective Regression 동일 coverage 비교

작성자: 박진서  
대상 논문: Noskov, Fishkov, Panov, *Selective Nonparametric Regression via
Testing*, Algorithm 1  
프로토콜: `protocols/SELECTIVE_REGRESSION_EQUAL_COVERAGE_PROTOCOL.md`

## 비교 질문

검정 기반 selective regression을 PP-X의 9개 strict-extrapolation setting에
그대로 적용했을 때 예측을 얼마나 유지하며, 동일한 수의 test row만 남겼을 때
어느 쪽의 accepted risk가 작은지 비교했다.

Algorithm 1의 Gaussian Nadaraya–Watson 평균·조건부분산·density test를
train-only 표준화 및 3차원 train-PCA 공간에 구현했다. Bandwidth는 validation
MSE로 선택했고 test label은 accepted risk 계산에만 사용했다.

## Coverage

| setting | Algorithm 1 admissible coverage |
|---|---:|
| HUST | 0.15% |
| Virkler | 0% |
| NASA battery | 25.49% |
| Sunwoda | 0% |
| RWTH | 0% |
| MATR2019 | 1.92% |
| MATR batch 2 | 1.23% |
| N-CMAPSS | 100% |
| MICH | 0% |

Strict out-of-support 평가에서는 9개 중 4개 setting을 전부 거절했다. 목표
coverage를 25%, 50%, 75%, 90%로 높여도 Algorithm 1의 density 조건을 유지한
실제 9-setting 평균 coverage는 각각 5.84%, 8.72%, 11.52%, 13.19%였다.
이는 해당 방법이 가정하는 충분한 local support와 본 연구의 strict
extrapolation 조건이 구조적으로 다르기 때문이다.

## 동일 achieved-coverage risk

각 setting에서 Algorithm 1이 실제로 허용한 행 수만큼 PP-X도 seed
disagreement가 작은 행을 선택했다.

| 목표 coverage | 비교 가능한 setting | PP-X macro normalized RMSE | Selective NW | PP-X 승리 |
|---|---:|---:|---:|---:|
| 25% | 5 | **0.272** | 1.279 | **5/5** |
| 50% | 5 | **0.296** | 1.275 | **5/5** |
| 75% | 5 | **0.303** | 1.274 | **5/5** |
| 90% | 5 | **0.305** | 1.266 | **5/5** |

5/5 승리의 양측 exact sign-flip \(p\)는 0.0625다. 표본이 5개이면 가능한
최소 양측 p-value가 0.0625이므로 방향은 완전히 일관되지만 0.05 기준에는
도달하지 않는다.

## 결론

Selective Regression은 관측 support 안에서 조건부분산이 낮은 점을 골라
예측하는 방법이다. PP-X는 support 밖에서도 승인된 prior-residual executor로
예측 경로를 구성한다. 현재 9개 setting에서는 PP-X가 전체 setting에 예측을
제공하면서, Selective Regression이 허용한 소수의 행만 맞춰 비교해도 5/5에서
더 작은 normalized RMSE를 보였다.

따라서 발표에서는 다음처럼 정리한다.

> 검정 기반 selective regression은 strict extrapolation에서 대부분의 예측을
> 거절했다. PP-X는 validation이 지지한 구조로 그 coverage를 유지했고, 동일
> achieved coverage에서도 더 작은 accepted risk를 보였다.

재현:

```bash
python experiments/selective_regression_equal_coverage.py
```

원시 결과: `results/selective_regression_equal_coverage_v1/results.json`
