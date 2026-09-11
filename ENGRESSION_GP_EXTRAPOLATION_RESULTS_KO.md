# PP-X 관련 Engression 및 linear-mean GP 역사적 외삽 비교

> **Canonical paper pointer:** 이 문서는 legacy PP 개발 당시의 8-setting
> Engression/GP 비교와 그 artifact를 보존한다. 현재 PP-X paper-main
> equal-budget 비교는 `FULL_EQUAL_CANDIDATE_BUDGET_RESULTS_KO.md`가 기준이며,
> 그 결과는 8/9, p=0.0391이다. strongest same-split mixed-comparator 9/9와
> 이 역사적 8/8 분석을 현재의 동일 evidence처럼 섞지 않는다. DS03에서는
> PP-X fallback 0.8818보다 Engression 0.9013이 높아 prospective predictive
> superiority가 실패했다.

## 결론

동일한 고정 split과 pooled R²에서 당시 개선 PP route는 양의 성능을 낸 8개
평가 설정 모두 Engression과 linear-mean GP보다 높았다. 이는 retrospective
historical audit이며 현재 PP-X의 universal 또는 prospective 우월성 주장이
아니다. 가장 좁은 차이는 N-CMAPSS의 0.937 대 Engression 0.932다.

| 데이터셋 | test hull-out | median hull 거리 | 개선 PP | 기존 최강 경쟁모델 | Engression | Linear-mean GP | PP−Engression |
|---|---:|---:|---:|---:|---:|---:|---:|
| HUST | 100% | 1.608 | **0.958** | GroupDRO 0.934 | 0.878 | -0.320 | +0.080 |
| Virkler | 100% | 1.623 | **0.888** | Linear-tail RBF 0.805 | 0.552 | 0.539 | +0.336 |
| NASA battery | 100% | 2.005 | **0.584** | Linear-tail RBF 0.550 | 0.549 | 0.438 | +0.035 |
| Sunwoda | 100% | 4.421 | **0.865** | Linear-tail RBF 0.838 | 0.619 | -1.598 | +0.246 |
| RWTH | 100% | 2.870 | **0.743** | V-REx 0.645 | 0.526 | -0.474 | +0.217 |
| MATR2019 | 100% | 5.554 | **0.466** | FT-Transformer 0.344 | -0.726 | -2.461 | +1.192 |
| MATR batch2 | 100% | 1.898 | **0.862** | V-REx 0.850 | 0.739 | 0.213 | +0.123 |
| N-CMAPSS | 100% | 0.228 | **0.937** | TabPFN 0.934 | 0.932 | 0.804 | +0.005 |

기존 최강 경쟁모델 열은 지금까지 실행한 Ridge, boosting, spline, MLP, ResNet, GRU, TCN, temporal Transformer, FT-Transformer, TabPFN, V-REx, GroupDRO, monotone NN, linear-tail RBF 중 데이터셋별 최고값이다. N-CMAPSS는 짧은 입력 hull 외삽이며 target 범위 밖 외삽은 아니다.

## 공정한 선택 절차

Engression은 공식 `engression==0.1.9` 구현을 사용했다. hidden width 32/64, learning rate 0.001/0.005, beta 0.5/1.0, 2-layer·250 epoch/3-layer·500 epoch의 16개 후보를 validation MSE로만 선택했다. 선택 설정을 seed 42--46으로 다시 학습한 뒤 예측을 평균했다. 계산량을 통제하기 위해 train은 결정론적 최대 5,000행을 사용했다. NASA는 leave-one-battery-out fold 예측을 pooled 집계했다.

Engression seed별 pooled R²의 평균±표준편차는 HUST 0.765±0.141, Virkler 0.496±0.242, Sunwoda 0.606±0.111, RWTH 0.520±0.053, MATR2019 -0.865±0.275, MATR batch2 0.616±0.183, N-CMAPSS 0.920±0.019였다. 따라서 ensemble 결과만 비교할 때 가려지는 재학습 분산도 PP 논문의 robustness 분석에 포함해야 한다.

Linear-mean GP는 `DotProduct + Constant×RBF + WhiteKernel` 구조에서 RBF 길이척도 0.3/1/3과 noise 0.01/0.1/1의 9개 후보를 validation으로 선택했다. 정확 GP의 세제곱 계산량 때문에 결정론적 최대 750행을 사용하고 seed 5개 예측을 평균했다. 따라서 특히 HUST·Sunwoda·RWTH·MATR·N-CMAPSS에서는 sparse/variational GP의 완전한 대체 결과가 아니라 고전적 uncertainty baseline이다.

## 해석과 남은 근거

Engression은 N-CMAPSS에서 PP와 사실상 동률이고 HUST에서도 강하다. 이는 분포 회귀가 짧은 hull 경계 외삽과 비교적 매끄러운 degradation coordinate에서 유효하다는 근거다. 반면 가장 먼 health-tail인 MATR2019에서 음의 R²가 되어, 분포 학습 자체가 안정적인 먼 꼬리 기울기를 보장하지는 않았다. PP는 affine/regime tail과 제한된 neural residual을 결합하기 때문에 이 설정에서 차이가 가장 컸다.

당시 PP의 8/8 최고점은 데이터를 보며 구조를 개발한 결과다. 8개
데이터셋·68개 유효 물리 unit의 계층 bootstrap은
`PAIRED_PP_ENGRESSION_ANALYSIS_KO.md`에 역사적 secondary analysis로
보존한다. 현재 PP-X 결론에서는 DS03 predictive-superiority 실패와 독립
prospective superiority 미확보를 함께 공개한다.

Progression은 논문 자체는 직접적인 회귀 외삽 경쟁군이지만, 이번 저장소에서 검증 가능한 공식 실행 구현을 확인하지 못했다. 논문 수식만 보고 임의 재구현한 수치를 공식 baseline처럼 넣으면 비교 재현성이 약해지므로 결과표에 숫자를 만들지 않았다. 저자 코드가 확보되면 동일 split·validation-only protocol로 추가한다.

## 재현 위치

- Engression 코드: `experiments/engression_all_positive.py`
- Engression raw 결과: `results/engression_all_positive_v2/results.json`
- GP 코드: `experiments/linear_mean_gp_all_positive.py`
- GP raw 결과: `results/linear_mean_gp_all_positive_v1/results.json`
- hull 코드: `experiments/all_dataset_hull_audit.py`
- hull raw 결과: `results/all_dataset_hull_audit_v1/results.json`

## 사후 통계 점검

개체 단위 paired bootstrap 결과, NASA의 PP−Engression 평균 상대 RMSE 차이는 -2.1%이나 95% CI [-28.2%, 29.8%]였고 N-CMAPSS는 +1.098, 95% CI [-0.222, 3.690]였다. 두 신뢰구간 모두 0을 포함한다. N-CMAPSS의 동일 가중 결과는 test 관측이 하나뿐인 엔진 14에 민감하며, 관측이 충분한 엔진 11과 15에서는 PP가 모두 소폭 우세했다. 상세 분석은 `PAIRED_PP_ENGRESSION_ANALYSIS_KO.md`에 있다.
