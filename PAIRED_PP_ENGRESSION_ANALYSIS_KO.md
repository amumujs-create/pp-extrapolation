# PP 대 Engression 통계 분석

## 판정

PP는 pooled R²에서 8개 데이터셋 모두 Engression보다 높지만, N-CMAPSS와 NASA 각각의 개체 수준 분석에서는 유의한 우월성을 확정할 수 없다. 따라서 논문에는 전체 방향 일관성과 데이터셋 내부 불확실성을 따로 보고해야 한다.

## 8개 도메인 계층 bootstrap

서로 다른 target scale을 비교하기 위해 개체별 `(PP RMSE - Engression RMSE) / Engression RMSE`를 사용했다. 한 개의 오차로 개체 성능을 정의하지 않도록 최소 2개 관측이 있는 물리적 개체만 주 분석에 포함했다. bootstrap은 먼저 데이터셋을 동일 가중으로 재표본화하고, 선택된 데이터셋 안에서 개체를 다시 재표본화했다.

| 항목 | 결과 |
|---|---:|
| 데이터셋 | 8 |
| 물리적 개체 | 68 |
| PP가 더 낮은 RMSE를 보인 개체 | 60/68 (88.2%) |
| 데이터셋 동일 가중 상대 RMSE 차이 | -24.0% |
| 계층 bootstrap 95% CI | **[-40.9%, -5.5%]** |
| bootstrap에서 PP 우위 확률 | 99.35% |

이 분석에서는 평균 차이의 신뢰구간이 0을 포함하지 않는다. 8개 데이터셋을 하나씩 제외한 leave-one-dataset-out 계층 bootstrap에서도 모든 95% CI가 0 아래였다. 따라서 관측된 전체 벤치마크 범위에서 PP의 Engression 대비 평균 오차 우위는 특정 데이터셋 하나에만 의존하지 않는다.

다만 68개 unit를 모두 독립으로 가정한 sign test의 매우 작은 p값은 주요 근거로 사용하지 않는다. 위의 데이터셋–개체 계층 bootstrap을 주 통계로 사용한다. 모델 구조가 이미 이 벤치마크를 보며 개발됐으므로, 이는 **강화된 retrospective evidence**이며 새 cohort의 prospective confirmation을 대체하지는 않는다.

| 분석 단위 | 관측 결과 | 불확실성 | 판정 |
|---|---|---|---|
| 8개 데이터셋 방향 | PP 8/8 승 | 사전 독립검정으로 가정할 때 단측 sign-test p=0.0039 | 강한 방향 일관성이나 post-hoc이므로 확증 p값 아님 |
| NASA 배터리 4개 | PP 2/4 승; 평균 RMSE 차이 -0.021 | unit bootstrap 95% CI [-0.282, 0.298] | 유의하지 않음 |
| N-CMAPSS 엔진 3개 | PP 2/3 승; 평균 RMSE 차이 +1.098 | unit bootstrap 95% CI [-0.222, 3.690] | 유의하지 않음; 한 엔진은 n=1 |
| N-CMAPSS seed 5개 | PP 4/5 승 | 단측 exact binomial p=0.1875 | 재학습 안정성의 기술통계이며 유의하지 않음 |

RMSE 차이는 `PP − Engression`이므로 음수가 PP 우위다. 행 단위 bootstrap은 같은 기계의 인접 시계열을 독립 표본으로 잘못 간주하므로 사용하지 않았다.

## NASA: pooled 우위가 어디서 생겼는가

| 배터리 | n | PP RMSE | Engression RMSE | PP−Engression |
|---|---:|---:|---:|---:|
| B0005 | 56 | 7.385 | **5.049** | +0.984 |
| B0006 | 59 | **7.729** | 9.623 | +2.408 |
| B0007 | 91 | 21.784 | **21.456** | -0.305 |
| B0018 | 49 | **8.633** | 13.629 | -4.897 |

PP의 pooled R² 0.584 대 Engression 0.549 우위는 네 배터리 모두에서 조금씩 이긴 결과가 아니다. B0006과 B0018의 개선이 B0005·B0007 열세를 상쇄했다. 평균 unit 상대 RMSE는 PP에 2.1% 유리하지만 신뢰구간이 넓다. 이 결과는 PP가 특정 degradation regime에서 큰 실패를 억제할 가능성을 보여주지만, 배터리 4개로 일반화하기에는 부족하다.

## N-CMAPSS: pooled R²와 unit 평균의 불일치

| 엔진 | n | PP RMSE | Engression RMSE | PP−Engression |
|---|---:|---:|---:|---:|
| 11 | 104 | **5.100** | 5.323 | -0.222 |
| 14 | 1 | 4.950 | **1.261** | +3.690 |
| 15 | 54 | **4.463** | 4.635 | -0.172 |

PP는 평가점이 충분한 엔진 11과 15에서 모두 이겼다. 엔진 14는 test window가 하나뿐이어서 엔진 수준 성능이나 R²를 정의할 수 없고, 이 한 점이 동일 가중 unit 평균을 뒤집는다. pooled R²는 159개 행을 가중하므로 PP 0.937 대 Engression 0.932가 된다. 따라서 본문 주지표는 사전 선언한 pooled R²를 유지하되, 엔진 14의 `n=1`과 두 유효 엔진에서의 작은 RMSE 차이를 반드시 같이 공개해야 한다.

엔진 14를 사후 제외한 두 엔진 모두에서 PP가 이긴다는 사실은 민감도 분석일 뿐이다. 독립 엔진이 두 개밖에 없어 통계적 일반화 근거가 되지는 않는다.

## 외삽 거리와 PP 이득

8개 데이터셋에서 median hull distance와 `PP R² − Engression R²`의 Spearman 상관은 ρ=0.667, 양측 p=0.071이다. 먼 외삽에서 PP의 affine/regime tail이 더 유리하다는 가설과 방향은 맞지만, 표본 8개이고 MATR2019의 큰 차이에 영향을 받는 탐색적 결과다. 논문의 메커니즘 근거로 사용하려면 각 데이터셋 안에서 distance shell별 paired error를 계산하고 새 cohort에서 같은 기울기를 확인해야 한다.

## 논문에서 사용할 수 있는 주장

- 사용할 수 있음: PP가 고정된 pooled R²에서 8개 평가 설정 모두 Engression보다 높은 점수를 냈다.
- 사용할 수 있음: Engression 대비 이득은 외삽 거리가 멀수록 커지는 경향을 보였다.
- 제한해서 사용: N-CMAPSS와 NASA에서는 점추정상 PP가 높았지만 개체 수준 신뢰구간은 0을 포함했다.
- 아직 사용할 수 없음: PP가 Engression보다 통계적으로 유의하게 보편 우월하다.

확증 실험은 새 데이터의 split·모델·튜닝 예산·pooled R²·unit bootstrap을 미리 고정해야 한다. 최소한 NASA형 독립 배터리 수와 N-CMAPSS형 test engine 수를 늘려야 한다. 현재 자료에서는 seed 수를 늘리는 것보다 물리적 개체 수를 늘리는 편이 더 중요하다.

## 재현

- 분석 코드: `experiments/paired_significance_analysis.py`
- 원시 통계: `results/paired_pp_engression_v1/results.json`
- 입력 결과: `results/engression_all_positive_v2/results.json`, `results/nasa_regime_spline_tuning_v1/results.json`, `results/ncmapss_pp_multiscale_v1/seed*.npz`
