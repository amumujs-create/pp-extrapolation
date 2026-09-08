# PP 저널 제출용 통합 근거 패키지

## 1. 논문의 중심 결과

PP를 하나의 고정 회귀식으로 모든 데이터에 강제하지 않고, 관측 가능한 prior의 종류에 따라 executor를 선택하는 **modular prior-preserving predictor**로 정의한다. 최종 PP는 공통적으로 `안정적인 저복잡도 tail 경로 + NN residual`을 사용하며, 데이터에서 확인 가능한 prior에 따라 boundary quotient, causal multiscale history, support decay, regime transport를 붙인다.

현재 양의 pooled R²를 확보한 9개 외삽 설정에서 최종 modular PP는 같은 split으로 실행해 확보한 가장 강한 비교 결과를 모두 넘었다.

| 데이터셋 / 외삽 설정 | 최종 PP pooled R² | 가장 강한 동일 split 비교 | 비교 R² | ΔR² |
|---|---:|---|---:|---:|
| HUST unseen-cell late tail | **0.958** | GroupDRO | 0.934 | +0.024 |
| Virkler unseen-specimen crack tail | **0.888** | linear-tail RBF | 0.805 | +0.083 |
| NASA battery unseen-cell tail | **0.584** | linear-tail RBF | 0.550 | +0.034 |
| Sunwoda unseen-cell tail | **0.934** | linear-tail RBF | 0.838 | +0.096 |
| RWTH unseen-cell tail | **0.842** | V-REx | 0.645 | +0.197 |
| MICH unseen-cell tail | **0.751** | matched direct NN | 0.684 | +0.067 |
| MATR2019 strict health tail | **0.466** | calibrated FT-Transformer | 0.377 | +0.089 |
| MATR batch 2 strict tail | **0.862** | V-REx | 0.850 | +0.012 |
| N-CMAPSS unseen-engine × high-TRA | **0.937** | Engression | 0.932 | +0.005 |

MATR batch 2의 과거 CPGRU 0.912는 train 11,553행 결과이고, 최종 PP는 최저 경계 행을 제외한 11,552행을 사용했다. 정확히 11,552행으로 맞춘 CPGRU ensemble은 0.537이다. 따라서 0.912는 protocol-sensitivity 결과이며 위 직접 비교에는 사용하지 않는다.

## 2. 물리 unit 단위 통계 근거

최종 PP의 저장된 5-seed 예측과 행·target·unit 순서가 정확히 같은 경쟁모델 예측만 사용해 다시 계산했다. 비교 효과는 각 물리 unit에서 다음의 대칭형 log-RMSE ratio로 정의했다.

\[
e_u = \log\frac{\operatorname{RMSE}_{u,\,comparison}}
                    {\operatorname{RMSE}_{u,\,PP}}
\]

양수이면 PP의 RMSE가 작다. 상대 RMSE 감소율 `1-RMSE_PP/RMSE_comparison`은 비교모델 RMSE가 매우 작은 Virkler 일부 시편에서 음의 값이 무한히 커질 수 있으므로, 추론의 주 효과크기는 log ratio로 사용한다.

- 9개 데이터셋 모두 prediction-ensemble pooled R² 우세
- 데이터셋 승패 exact sign test: **p = 0.00390625**
- 총 **77개 물리 unit** 평가
- 데이터셋 동일 가중 평균 log-RMSE ratio: **0.394**
- 데이터셋과 unit을 함께 재표집한 hierarchical bootstrap 95% CI: **[0.161, 0.646]**
- 기하평균 RMSE 감소: **32.5%**, 95% CI 환산 시 **14.9%–47.6%**

| 데이터셋 | PP seed R² 평균±SD | 저장 예측 비교모델 평균±SD | PP 승리 unit | mean log-RMSE ratio | unit bootstrap 95% CI | BH q |
|---|---:|---:|---:|---:|---:|---:|
| HUST | 0.953±0.013 | 0.591±0.227 | 10/16 | 0.222 | [−0.106, 0.548] | 0.375 |
| Virkler | 0.857±0.021 | 0.743±0.275 | 8/10 | 0.162 | [−0.681, 0.864] | 0.750 |
| NASA | 0.584±0.005 | 0.520±0.088 | 2/4 | 0.113 | [−0.177, 0.450] | 0.750 |
| Sunwoda | 0.842±0.099 | 0.369±0.214 | 9/9 | **0.458** | **[0.323, 0.625]** | **0.0176** |
| RWTH | 0.818±0.050 | 0.319±0.554 | 7/8 | 0.510 | **[0.138, 0.778]** | 0.0938 |
| MICH | 0.703±0.077 | 0.339±0.359 | 6/8 | 0.232 | [−0.004, 0.521] | 0.299 |
| MATR2019 | 0.429±0.073 | 0.326±0.030 | 6/10 | 0.053 | [−0.085, 0.181] | 0.600 |
| MATR-b2 | 0.852±0.061 | 0.032±0.416 | 9/9 | **1.109** | **[0.762, 1.434]** | **0.0176** |
| N-CMAPSS | 0.934±0.007 | 0.851±0.016 | 3/3 | **0.684** | **[0.257, 1.130]** | 0.375 |

RWTH·MATR-b2·N-CMAPSS의 표 첫 절에 쓴 최강 비교모델은 각각 V-REx 0.645, V-REx 0.850, Engression 0.932다. 이 세 모델은 현재 row-level 예측 파일이 없으므로 unit paired 검정에는 저장 예측이 있는 direct NN 0.633, GroupDRO 0.691, monotone NN 0.855를 사용했다. 최강 비교와의 성능 차이는 pooled R²로만 보고하며 두 종류의 증거를 섞지 않는다.

## 3. 구성요소가 필요한 이유

최종 PP 성능은 하나의 prior나 하나의 후처리에서 나온 결과가 아니다.

| 구성요소 | matched 제거 결과 → 포함 결과 | ΔR² | 해석 |
|---|---:|---:|---|
| nonlinear residual | Sunwoda affine quotient 0.281 → 0.939 | +0.658 | affine tail에 없는 곡률 학습 |
| frozen affine path | MICH trainable hard-boundary 0.319 → frozen BQ 0.468 | +0.149 | 작은 cohort의 tail drift 억제 |
| residual bound | Sunwoda unbounded 0.718 → bounded 0.939 | +0.221 | 과도한 NN correction 억제 |
| support-adaptive dual scale | MICH fixed bound 0.468 → 0.751 | +0.283 | relationship shift에 필요한 residual 용량 복구 |
| causal rate history | RWTH current-only −0.378 → full history 0.878 | +1.256 | 같은 health에서 서로 다른 열화속도 구분 |
| regime transport | HUST raw 0.829 → transported 0.958 | +0.128 | validation unit의 출력오차 구조 전달 |
| support decay × transport | MATRb2 transport-only 0.820 → 둘 다 0.862 | +0.042 | 먼 support에서 transport 과보정 억제 |
| validation causal route | NASA short-only 0.572 → selected 0.584 | +0.012 | 유효 history scale 선택 |
| validation multiscale route | N-CMAPSS basic 0.928 → selected 0.937 | +0.009 | 운전조건과 열화 시간척도 선택 |

고정 residual bound와 full rate history는 모든 곳에서 이롭지 않다. MICH에서 unbounded residual은 0.759로 fixed-bound 0.468보다 높고, margin-history-only는 0.715로 기존 full-history 0.468보다 높다. 최종 dual-scale PP는 이 반례를 이용해 local bound와 broad bound를 support 이질성에 따라 전환하여 MICH를 0.751로 복구했다. 이 결과가 modular design의 직접적인 필요성을 뒷받침한다.

## 4. 공통 백본 실험의 올바른 위치

모든 데이터셋에 같은 PP 백본을 강제한 matched 실험에서는 PP가 plain MLP를 12개 중 6개에서만 이겼다. 이 결과는 최종 PP 성능표가 아니며, prior contract 없이 하나의 구조를 강제하면 실패한다는 negative ablation이다.

validation에서 PP/MLP를 선택하는 단순 route도 12개 중 7개만 맞고 false accept 4개가 발생했다. Milling은 validation 상대 RMSE가 51.1% 개선됐지만 test에서는 PP 상대 효과가 −6.21이었다. 따라서 작은 validation 점수 하나가 아니라 `prior role → mechanism coverage → executor` 순서가 필요하다.

논문의 비교 구조는 다음과 같이 쓴다.

1. **주 결과:** final modular PP와 강한 동일-split 비교모델
2. **matched architecture ablation:** 공통 PP backbone 대 plain MLP
3. **module ablation:** prior별 executor의 on/off 및 상호작용
4. **scope failures:** XJTU, FEMTO, milling과 외부 negative cohort

## 5. 외삽 geometry 검증

12개 공통백본 데이터셋에서 ordered-coordinate 1D hull, train-only PCA 2D/3D hull, 10-NN density ratio를 비교했다. support distance와 PP의 unit 상대 RMSE 이득 간 dataset-level permutation Spearman 결과는 다음과 같다.

| support 척도 | Spearman ρ | permutation p |
|---|---:|---:|
| 1D ordered-coordinate hull | 0.109 | 0.733 |
| train-PCA 2D hull | −0.214 | 0.502 |
| train-PCA 3D hull | 0.042 | 0.900 |
| 10-NN density ratio | 0.448 | 0.152 |

Hull은 test가 학습 support 밖에 있는지 정의하는 데 유효하지만 PP의 성공을 보장하는 변수가 아니다. 차원에 따라 outside fraction도 크게 바뀐다. 예를 들어 MATRb2는 1D 100%, PCA2 0%, PCA3 99.2%이고 N-CMAPSS는 1D 100%, PCA2 0%, PCA3 83.6%다. 논문에서는 1D ordered-coordinate boundary와 multi-D convex hull을 구분해 표기한다.

## 6. 불확실성과 coverage

기존 support-scaled 90% block-conformal 감사에서는 외부 NASA/UCF 0.857, CALCE 0.927, HNEI 0.394의 empirical coverage를 얻었다. CALCE는 평균 폭이 약 416 cycles로 지나치게 넓었고 HNEI는 coverage가 무너졌다. 따라서 현재 결과로 “PP가 언제 틀릴지 완전히 안다”고 주장하지 않는다.

방어 가능한 주장은 다음과 같다.

> Calibration unit이 충분하고 validation과 test regime가 교환 가능할 때 support-scaled interval이 목표 coverage에 접근한다. Regime transfer가 깨지면 interval도 실패하므로, uncertainty는 적용 가능성 certificate의 한 구성요소다.

OOF disagreement target으로 학습한 uncertainty head와 risk–coverage 결과는 point uncertainty의 근거로 사용하고, finite-sample prediction interval 보장은 별도로 구분한다.

## 7. 논문 novelty

PP의 novelty는 “물리식과 NN을 결합했다”는 일반적 PINN 주장에 두지 않는다. 논문에서 강하게 방어할 수 있는 차별점은 다음 세 가지의 결합이다.

1. **Typed prior contract:** boundary, monotone tail, causal history, support, regime transport처럼 관측 가능한 prior의 역할을 명시한다.
2. **Prior-preserving residual executor:** 안정적 tail 경로를 동결하고 NN이 bounded 또는 support-adaptive residual만 학습하게 하여 외삽에서 prior를 덮어쓰지 못하게 한다.
3. **Evidence-gated modularity:** optional module은 validation physical-unit evidence를 만족할 때만 승인하고, 맞지 않는 prior를 모든 도메인에 강제하지 않는다.

이 구조는 PAE와 역할이 다르다. PP 논문은 prior가 주어졌을 때 이를 보존하며 예측하는 executor를 다룬다. 후속 PAE는 어떤 prior contract를 컴파일하고 선택할지를 다룬다. 박사논문에서는 `PAE compiler → PP executor → applicability/uncertainty certificate`의 하나의 프레임워크로 연결할 수 있다.

## 8. 제출 수준과 주장 범위

현재 패키지는 다음을 갖췄다.

- 9개 양의 외삽 설정에서 강한 동일-split 비교모델보다 높은 pooled R²
- 5-seed retraining 결과
- 77개 물리 unit paired 효과와 다중검정
- 데이터셋 계층 bootstrap에서 양의 전체 log-RMSE 효과
- prior별 구성요소 ablation과 상호작용
- 1D/2D/3D hull 민감도 및 support-distance 분석
- 불확실성, coverage, 계산비용, 실패 데이터셋의 범위 분석

이 정도면 RESS, MSSP, EAAI, ESWA 계열의 방법론 논문으로 제출 가능한 근거가 있다. 더 높은 보편 일반화 주장은 최종 model/routing rule을 고정한 뒤 새로운 성공 cohort에서 한 번 검증해야 한다. 현재 논문의 주장은 다음 문장으로 제한하는 것이 성능과 정직성을 함께 살린다.

> Across nine concept-aligned extrapolation settings, the development-final modular PP improved pooled R² over the strongest evaluated same-split comparator in every setting. Across 77 row-aligned physical units, its equal-dataset geometric mean RMSE reduction was 32.5% (hierarchical 95% CI 14.9%–47.6%). The common-backbone and failure-cohort audits show that the gain depends on matching an observable prior contract to its executor rather than applying one universal architecture.

## 9. 관련 방법과의 구분

- Balestriero et al.은 고차원 표현에서 test point가 training convex hull 밖에 놓이는 현상을 분석한다. PP는 hull 밖 여부를 새로 발명했다고 주장하지 않고, 이 geometry를 executor의 support 정보로 사용한다: [Learning in High Dimension Always Amounts to Extrapolation](https://arxiv.org/abs/2110.09485).
- Bonnasse-Gahot은 convex-hull membership만으로 neural-network 일반화를 설명하기 어렵고 proximity가 더 직접적일 수 있음을 보인다. 본 연구의 1D/2D/3D hull 및 kNN 감사 결과와 일치한다: [Interpolation, extrapolation, and local generalization in common neural networks](https://arxiv.org/abs/2207.08648).
- Webb et al.은 convex domain에서 벗어난 정도를 연속적으로 다루는 representation을 연구한다. PP의 support-distance decay와 관련되지만 PP는 stable tail path와 bounded residual을 직접 구성한다는 점에서 다르다: [Learning Representations that Support Extrapolation](https://proceedings.mlr.press/v119/webb20a.html).
- Risk Extrapolation은 여러 training domain의 risk로부터 domain-level extrapolation을 다룬다. PP는 개별 시계열 상태의 tail geometry와 prior-preserving prediction path를 다룬다는 차이가 있다: [Out-of-Distribution Generalization via Risk Extrapolation](https://proceedings.mlr.press/v139/krueger21a.html).

따라서 단일 구성요소인 hull, residual network, affine tail, validation selection 각각은 기존 개념과 겹칠 수 있다. 논문의 모델링 novelty는 이들을 나열하는 데 있지 않고, **typed prior contract를 안정적 tail path와 NN residual의 구조적 제약으로 컴파일하고, support와 validation evidence에 따라 prior별 executor를 승인하는 전체 설계**에 둔다.

## 10. 재현 산출물

- 최종 통계 실행: `experiments/final_modular_pp_evidence.py`
- 원시 통계: `results/final_modular_pp_evidence_v1/results.json`
- 물리 unit 표: `results/final_modular_pp_evidence_v1/unit_effects.csv`
- 최종 성능·forest·seed 그림: `figures/final_modular_pp_evidence_v1/`
- Hull 감사: `experiments/journal_support_geometry_audit.py`
- Validation route 감사: `experiments/journal_validation_route_audit.py`
- 공통백본 통계: `experiments/journal_statistical_evidence.py`
- 전체 구성요소 상세: `FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md`
