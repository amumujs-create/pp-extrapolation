# PP 논문 그림 구성 가이드

모든 그림은 `figures/paper/`에 벡터 PDF와 600 dpi PNG로 저장한다. 수치는 현재 저장된 최종 PP 실험을 사용한다. 현재 결과는 여러 차례의 모델 개발에 사용된 데이터셋을 포함하므로, 논문에서는 **retrospective development benchmark**로 명시하고 독립 봉인 코호트 결과와 구분해야 한다.

## 이번 통합 감사에서 추가된 제출 우선 그림

아래 세 그림은 PNG 600 dpi와 vector PDF로 함께 생성됐으며 기존 개발 단계 그림보다 우선 사용한다.

1. `figures/final_modular_pp_evidence_v1/fig_F1_final_pp_vs_strongest`: 9개 양의 외삽 설정에서 최종 PP와 가장 강한 동일-split 비교 결과.
2. `figures/final_modular_pp_evidence_v1/fig_F2_final_pp_unit_forest`: 저장 예측이 행 단위로 일치하는 77개 물리 unit의 log-RMSE ratio와 bootstrap CI.
3. `figures/final_modular_pp_evidence_v1/fig_F3_final_pp_seed_stability`: 동일 5 seeds에서 최종 PP와 저장 비교모델의 재학습 안정성.

`figures/journal_evidence_v1/`의 공통백본 그림은 최종 PP 성능이 아니라 modular prior routing의 필요성을 보이는 mechanism/negative ablation으로 배치한다.

## 본문 권장 그림

### Figure 1. 전체 strict-extrapolation 성능

- 파일: `fig1_final_benchmark.pdf`
- 목적: 12개 데이터셋에서 최종 PP와 각 데이터셋의 가장 강한 matched comparator를 함께 제시한다.
- 캡션 초안: **Final modular PP performance under the fixed strict-extrapolation protocols.** Points report pooled prediction-ensemble \(R^2\). The comparator is the strongest model evaluated under the same split and metric for each dataset. Negative results are retained to expose failure regimes rather than filtered from the benchmark.
- 주의: NASA milling의 큰 음수 때문에 양의 성능 구간이 압축된다. 정확한 값은 본문 표와 함께 제시한다.

### Figure 2. PP 구조와 정보 흐름

- 파일: `fig2_model_overview.pdf`
- 목적: frozen affine tail, neural residual, support-adaptive dual-scale gate, boundary quotient, validation-evidence executor의 역할을 한 장에 설명한다.
- 캡션 초안: **Architecture of modular PP.** A stable affine tail provides the extrapolative backbone, while a bounded neural residual represents deviations supported by observed history. Support distance adjusts residual capacity, the boundary quotient enforces the end-of-life boundary when available, and validation-only evidence activates optional history, decay, and regime-transport executors.

### Figure 3. 구조적 matched ablation

- 파일: `fig3_bq_matched_ablation.pdf`
- 목적: 동일 데이터와 평가 조건에서 여섯 구조를 비교해 PP 구성의 필요성을 보인다.
- 캡션 초안: **Matched structural ablation of the boundary-quotient PP executor.** Bars are prediction-ensemble pooled \(R^2\) on Sunwoda, RWTH, and MICH. The comparison separates the effects of a neural residual, hard boundary construction, frozen affine extrapolator, and bounded correction.

### Figure 5. support decay × regime transport

- 파일: `fig5_matr_support_transport_interaction.pdf`
- 목적: MATR batch 2에서 두 외삽 모듈의 독립 효과와 결합 효과를 5-seed 평균±표준편차로 보여준다.
- 캡션 초안: **Interaction between support decay and validation-only regime transport on MATR batch 2.** Points show mean single-seed pooled \(R^2\), and error bars show one standard deviation over five retrainings. Regime transport provides the main gain and support decay improves the transported solution further.

### Figure 6. 재학습 안정성

- 파일: `fig6_matr_seed_stability.pdf`
- 목적: 단일 최고 seed 대신 5회 재학습 분포를 공개한다.
- 캡션 초안: **Retraining stability under the exact MATR batch-2 split.** Each point is one random seed and the horizontal segment is the seed mean. PP is compared with V-REx, TabPFN, CPGRU, and CPTransformer under the same evaluation split.

### Figure 7. convex-hull 외삽 강도

- 파일: `fig7_hull_extrapolation_severity.pdf`
- 목적: test가 train hull 밖으로 얼마나 멀리 나갔는지와 validation-test 난이도 이동을 정량화한다.
- 캡션 초안: **Geometric severity of the extrapolation protocols.** The horizontal coordinate is the median distance beyond the training hull in training-standard-deviation units. Colour indicates the ratio between test and validation distances, clipped at eight for display. Crosses denote datasets whose median validation distance is zero, making the ratio undefined. NASA battery is represented by the median over leave-one-cell-out folds.

## 본문 또는 보충자료 선택

### Figure 4. 구성요소별 \(\Delta R^2\)

- 파일: `fig4_component_delta_heatmap.pdf`
- 목적: 각 모듈이 어느 failure mode에서 기여하는지 보여준다.
- 해석 제한: 셀마다 기준 ablation arm이 다르므로 행 사이의 절댓값을 인과효과처럼 직접 비교하지 않는다. 각 셀은 표에 정의된 matched removal 대비 변화량이다.

### Figure 9. ablation 복합 그림

- 파일: `fig9_ablation_composite.pdf`
- 목적: Figure 3의 구조 ablation과 승인된 executor gain을 한 장으로 압축한다. 본문 지면이 제한될 때 Figure 3과 Figure 8 대신 사용한다.
- 권장: Figure 9를 쓰면 Figure 3과 Figure 8은 보충자료로 이동한다.

## 보충자료 권장 그림

### Figure 8. 데이터셋별 executor 기여

- 파일: `fig8_executor_contributions.pdf`
- 목적: history, multiscale, transport, dual-scale이 각각 어떤 데이터셋에서 선택되고 얼마나 개선했는지 보여준다.
- 주의: 이는 하나의 모듈을 모든 데이터셋에 강제로 적용한 결과가 아니라 validation-evidence rule로 승인된 모듈의 변화량이다.

### Figure 10. 전체 데이터셋 ablation coverage

- 파일: `fig10_ablation_coverage.pdf`
- 목적: 최종 benchmark의 모든 데이터셋을 빠짐없이 나열하고, 어느 구성요소가 실제 matched/on-off ablation 되었는지와 어디가 아직 미분리 상태인지 공개한다.
- 기호: `✓`는 정량 matched/on-off 근거, `△`는 실패 또는 stress-test 근거, `—`는 해당 효과를 별도로 분리하지 않았음을 뜻한다. `—`는 그 모듈이 반드시 적용 불가능하다는 의미가 아니다.
- 해석: 현재 ablation은 모든 데이터셋에 모든 모듈을 적용한 full factorial이 아니다. prior가 있는 executor는 해당 prior가 정의되는 데이터에서 검증했고, 나머지 빈칸은 보충 실험 범위로 남긴다.

### Figure 11. PP 대 plain MLP 단위 대응 추론

- 파일: `fig11_plain_mlp_paired_inference.pdf`
- 목적: HUST, Virkler, NASA battery에서 unit별 RMSE 차이의 bootstrap 95% CI와 exact sign-flip p를 제시한다.

### Figure 12. 적용 가능성 descriptor 지도

- 파일: `fig12_applicability_map.pdf`
- 목적: 기존 12개 도메인과 누출 감사가 끝난 NASA/UCF, CALCE, HNEI 외부 코호트를 horizon–heterogeneity 평면에 표시한다.
- 해석: HNEI 성공과 두 외부 실패가 단순 threshold로 분리되지 않으므로 이 그림은 성공 gate가 아니라 **정적 descriptor gate의 반증**이다.

### Figure 13. Conformal coverage 감사

- 파일: `fig13_conformal_coverage.pdf`
- 목적: 15개 평가 도메인의 nominal 90% 대비 empirical coverage와 calibration-unit 수를 동시에 보인다.
- 주의: 외부 3개 코호트의 calibration unit이 1–2개뿐이므로 finite-sample 90% 보장으로 해석하지 않는다.

### Figure 14–16. 외부 코호트 결과

- 파일: `fig14_nasa_alt_external_predictions.pdf`, `fig15_external_cohort_comparison.pdf`, `fig16_hnei_external_predictions.pdf`
- 목적: NASA/UCF 실패 궤적, 세 외부 데이터의 pooled R², HNEI의 unit별 예측을 각각 공개한다.
- 주의: 모든 수치는 held-out cell의 최종 수명을 입력 정규화에 쓰지 않은 leakage-audited 재실행 결과다.

### Figure 17. 외부 실패 코호트 구조 개선

- 파일: `fig17_external_failure_recovery.pdf`
- 목적: HNEI locked 성공 결과는 그대로 두고, NASA/UCF와 CALCE에서 validation-selected safe-continuation PP가 동일 조건 NN을 얼마나 개선했는지 보인다.
- 주의: NASA/UCF와 CALCE test는 구조 개발 전에 이미 확인됐으므로 development evidence로 표기한다. NASA/UCF의 개선 PP R²도 여전히 음수다.

## 그림과 근거 파일 연결

| 그림 | 주요 근거 |
|---|---|
| Fig. 1 | `FINAL_PP_BENCHMARK_TABLE_KO.md`, `ALL_DATASET_COMPARISON_KO.md` |
| Fig. 3, 9a | `results/bq_pp_matched_controls_v1/results.json` |
| Fig. 4, 8, 9b | `FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md` |
| Fig. 5 | `results/matr_batch2_support_transport_ablation_v1/results.json` |
| Fig. 6 | MATR batch-2 PP/competitor 5-seed JSON 결과 |
| Fig. 7 | `results/all_dataset_hull_audit_v1/results.json` |
| Fig. 10 | `FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md`, 각 executor 결과 JSON |
| Fig. 11 | `results/plain_mlp_paired_inference_v1/results.json` |
| Fig. 12–13 | `results/cross_domain_mechanism_v1/results.json`, 외부 3개 `results.json` |
| Fig. 14–16 | `results/nasa_alt_external_locked_v1/`, `results/calce_external_locked_v1/`, `results/hnei_external_locked_v1/` |
| Fig. 17 | `results/external_failure_pp_recovery_v1/safe_final_results.json` |

## 원고 배치안

1. Introduction 마지막: Figure 2로 전체 아이디어를 제시한다.
2. Experimental setup: Figure 7로 단순 미래예측과 hull 외삽의 차이를 정량화한다.
3. Main results: Figure 1과 정확한 수치 표를 함께 둔다.
4. Ablation: Figure 9 또는 Figure 3+5를 둔다.
5. Robustness: Figure 6을 둔다.
6. Supplement: Figure 4와 8, 전체 seed 표, unit-level 통계를 둔다.

그래프를 다시 만들 때는 저장소 루트에서 다음을 실행한다.

```bash
MPLCONFIGDIR=/tmp/pp-mpl python experiments/make_paper_figures.py
```
