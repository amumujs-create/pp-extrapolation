# 봉인 MATR batch2 confirmatory 결과

## 사전 동결과 데이터

Protocol `7b3eeea`를 공개한 뒤 처음으로 MATR batch2 원본을 다운로드·역직렬화했다.
고정 split은 저장 순서 기준 train 30 cells, validation 9 cells, test 9 cells다.
원본 SHA-256은 `63ab200d09ecb237fee5ef3a5c5db76e3212e3206a0bd92f769e1427fed338b8`이다.

Strict capacity-tail 표본은 train 11,553, validation 673, test 733개이며 validation과
test는 모두 train capacity convex hull 밖이다.

## Test label 공개 전 gate

고정 shell gate는 733개 중 53개를 승인했다.

- prediction coverage: **7.23%**
- `[4,inf)` shell: validation n=53, Ridge 대비 gain=0, seed disagreement=0
- 나머지 populated shell은 seed instability 또는 negative baseline gain으로 거절

## 전체 test 성능

| 모델 | 5-seed R² 평균±SD | ensemble R² | ensemble RMSE | unit-macro R² |
|---|---:|---:|---:|---:|
| Ridge/affine | deterministic | -1.276 | 49.35 | -5.084 |
| 일반 NN | 0.012±0.306 | **0.744** | **16.54** | **0.275** |
| PP | **0.435±0.037** | 0.471 | 23.80 | 0.001 |
| support-PP | **0.496±0.068** | 0.523 | 22.58 | -0.036 |

일반 NN ensemble이 가장 높다. 그러나 개별 seed는 `0.482, -0.363, -0.019,
-0.248, 0.209`로 불안정하다. PP의 다섯 seed는 `0.451, 0.365, 0.431,
0.467, 0.461`, support-PP는 `0.601, 0.388, 0.489, 0.506, 0.497`로 모두 양수다.

PP와 support-PP는 9/9 test cells에서 Ridge보다 RMSE가 낮았다. Unit-paired bootstrap
평균 RMSE 차이는 PP `-29.71` (95% CI `[-35.72,-21.96]`), support-PP `-30.22`
(`[-35.66,-23.16]`)였다. 일반 NN ensemble도 9/9에서 Ridge보다 낮았다.

## 사전 confirmatory 판정

| 사전 조건 | 결과 |
|---|---|
| nonzero coverage | PASS |
| selective PP R² > 0 | **FAIL** (`-1.470`) |
| selective PP RMSE < selective Ridge | PASS (`3.075 < 3.242`) |
| full PP R² > Ridge | PASS (`0.471 > -1.276`) |

네 조건 중 하나가 실패했으므로 전체 confirmatory verdict는 **FAIL**이다. 승인된 53개는
수명 종료 바로 근처라 target 분산이 작고, 작은 절대 오차에도 R²가 음수가 됐다. 결과를
본 뒤 metric이나 shell threshold를 바꾸지 않는다.

## 논문에서 가능한 주장

이 봉인 cohort는 PP가 Ridge를 안정적으로 이긴 독립 증거다. 특히 단일 학습 run의 seed
robustness는 일반 NN보다 강하다. 반면 best ensemble accuracy는 일반 NN이므로 PP가 모든
비교 모델보다 정확하다고 주장할 수 없다. 또한 현재 shell gate는 nonzero coverage를 냈지만
selective positive R² 기준을 통과하지 못해 완성된 selective predictor의 성공 증거도 아니다.

따라서 결과는 **predictive architecture 부분 성공 + selective gate confirmatory 실패**로
분리해 보고한다. 이 구분이 post-hoc 성공 선언보다 논문 신뢰도를 높인다.
