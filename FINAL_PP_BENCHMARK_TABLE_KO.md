# 최종 PP 기준 통합 경쟁모델 표

이 표가 PP 발표와 논문 초안에서 사용할 **현재 최종 개발 모델** 기준 표다. 모든 값은 각 데이터셋의 고정 strict extrapolation test에서 계산한 pooled R²다. `최종 PP`는 validation으로 선택한 구조·학습 설정·승인된 PP 내부 모듈을 반영한다. 따라서 이전의 기본 PP, support-PP, 봉인 confirmatory PP 수치와 섞어 비교하지 않는다.

| 데이터셋 / 외삽 설정 | 최종 PP | 현재 최고 비-PP 경쟁모델 | TabPFN v3 | PP 판정 |
|---|---:|---:|---:|---|
| HUST protocol-tail | **0.958** | GroupDRO 0.934 | 0.218 | 우세 |
| Virkler crack-tail | **0.888** | linear-tail RBF 0.805 | 0.621 | 우세 |
| NASA battery LOO-tail | **0.584** | linear-tail RBF 0.550 | −0.691 | 우세 |
| Sunwoda unseen-cell tail | **0.939** | linear-tail RBF 0.838 | −0.886 | validation-approved bounded BQ-PP 우세 |
| RWTH unseen-cell tail | **0.878** | V-REx 0.645 | −2.174 | validation-approved bounded BQ-PP 우세 |
| MATR2019 strict health-tail | **0.466** | 같은 validation calibrator를 적용한 FT-Transformer 0.377 | 미실행 | 우세 |
| MATR batch 2 strict tail | **0.862** | V-REx 0.850 | 0.618† | 우세 |
| N-CMAPSS hard TRA extrapolation | **0.937** | Engression 0.932 | 0.934 | 동률권에 가까운 우세 |
| MICH unseen-cell tail | **0.751** | direct NN 0.684 | 미실행 | 개선 dual-scale PP 우세 |
| XJTU condition transfer | **0.257** | linear-tail RBF −1.418 | 미실행 | progress-temporal PP + validation opposite-ray scale transport; post-test 개발 성공§ |
| FEMTO endpoint transfer | **0.075*** | 교정 causal GRU −0.248 | 미실행 | prior abstention → waveform neural safety; ensemble만 양수 |
| NASA milling material transfer | **0.341** | tuned direct NN −0.476 (GroupDRO −0.691) | 미실행 | inspection-calibrated boundary-quotient PP 우세‡ |

## 이 표를 읽는 방법

- **양의 R² 12개 설정**은 final selected PP-X route에서 확보됐다. 단, FEMTO의 .075는 다섯 예측의 ensemble만 양수이고 개별 seed는 모두 음수이므로 강건한 성공이나 경쟁모델 우세에 포함하지 않는다. XJTU의 0.257은 이미 관측된 test에서 구조를 개발한 결과라 독립 확증 성공에는 포함하지 않는다. N-CMAPSS의 PP 0.937 대 Engression 0.932 차이는 작으므로 엄밀한 우월성보다 동률권에 가까운 우세로 쓴다.
- MICH는 support-adaptive dual-scale boundary executor로 0.751까지 복구됐다. Sunwoda·RWTH는 dual-scale을 전역으로 쓰지 않고 fixed bounded executor를 사용한다. Milling은 공식 고장경계 `VB=0.50`을 사용한 quotient prior로 양의 R2를 회복했다.

> **2026-09-09 FEMTO 정정:** 과거 −0.571 경로는 6열 CSV에서 실제 진동 열 4/5가 아니라 시간 metadata 열 0/1을 사용했다. 해당 값과 같은 입력에서 나온 비교값은 최종 성능 근거에서 철회한다. 교정 이후 PP-X는 source evidence로 physical prior를 거절하고 waveform neural safety로 간다. `*.075`는 ensemble R²이며 individual seed 안정성은 해결되지 않았다.

§ **XJTU 0.257:** RUL을 직접 제한해 예측하지 않고 `log1p(RUL/position)`을 affine+GRU residual PP가 학습한다. Train condition을 좌표 0, validation/test 조건을 반대 방향 −1/+1로 놓고, validation의 zero-intercept scale을 identity 1 주위로 반사해 test scale을 사전 계산한다. Seed 42~46 개별 R²는 0.241~0.256, 평균 0.252±0.006, prediction ensemble 0.257이다. Test label을 scale 계산에 사용하지 않았지만 test를 이미 본 뒤 개발한 구조이므로 `results/xjtu_reflected_scale_pp_v3/`의 retrospective 결과이며 새 cohort 확증이 필요하다.

‡ Milling 0.341은 이미 관측된 test에서 개발한 결과다. 공식 고장경계 0.50은 유지하고, 희소한 마모 측정 간격을 보정하는 margin `+0.03`을 validation MAE로 선택했다. Residual hyperparameter도 train/validation으로 튜닝했다. Test material-2가 train material-1에 없다는 label-free 인증으로 NN residual을 끄고 quotient route를 적용했다. 외부 확증 결과로는 쓰지 않는다.
- TabPFN은 로컬 CPU v3에서 seed 42–46 예측을 평균했다. HUST·Virkler·NASA·C-MAPSS는 최대 3,000 train행, Sunwoda·RWTH·MATRb2는 최대 1,000 equal-unit train행을 사용한 보조 비교다. 다른 모델과 train cap이 다를 수 있어, 최고 비-PP 경쟁모델 열의 순위 결정에는 쓰지 않는다.

† MATRb2 BatteryLife CPGRU는 최종 PP와 동일한 11,552 train행에서 ensemble 0.537이다. 봉인 confirmatory train 11,553행에서 나온 0.912는 protocol-sensitivity 결과로 분리하며 최종 직접 비교에 사용하지 않는다.

MATRb2의 seeds 42–46 완전 비교에서 PP는 `0.852 ± 0.061`, prediction ensemble `0.862`이며, 다섯 seed 모두 CPGRU·CPTransformer·V-REx·TabPFN보다 높았다. 전체 seed 표와 통계는 `MATR_BATCH2_FIVE_SEED_COMPARISON_KO.md`에 있다.

| MATRb2 동일 5-seed 보조표 | 개별 pooled R² 평균±SD | prediction ensemble R² |
|---|---:|---:|
| **최종 PP** | **0.852±0.061** | **0.862** |
| V-REx | 0.046±0.468 | 0.850 |
| TabPFN v3† | 0.616±0.041 | 0.618 |
| BatteryLife CPGRU | 0.386±0.334 | 0.537 |
| BatteryLife CPTransformer | −0.081±1.044 | 0.380 |

MATRb2 temporal PP 후보는 단일 seed에서 0.914를 기록했지만 5-seed 평균 예측은 0.860으로 최종 PP 0.862를 넘지 못했다. 따라서 현재 최종 표에는 채택하지 않았다. 상세 결과는 `MATR_BATCH2_TEMPORAL_PP_RESULTS_KO.md`에 보존한다.

## 이전 MATR batch 2 수치와의 관계

MATR batch 2의 `0.471`(기본 PP) 및 `0.523`(support-PP)은 test 공개 전에 동결한 봉인 confirmatory 실험 결과다. 이후 validation-only selection으로 PP 구조·optimizer·state/rate transport를 개발한 최종 PP는 `0.862`이다. 따라서 **최종 모델 성능 표에는 0.862**를 쓴다. 독립 확증 문단에서는 봉인 결과 0.523을 별도로 보고하며, 0.862를 새 cohort 확인 결과처럼 쓰지 않는다.

## 근거 문서

- 제출용 성능·통계·ablation·geometry 통합본: `JOURNAL_EVIDENCE_COMPLETE_KO.md`
- 최종 77-unit paired 통계: `results/final_modular_pp_evidence_v1/results.json`
- 최종 PP와 V-REx·GroupDRO·monotone NN·linear-tail RBF: `ALL_DATASET_EXTRAPOLATION_COMPETITORS_KO.md`
- Engression·GP 및 후보 중 최고값: `ENGRESSION_GP_EXTRAPOLATION_RESULTS_KO.md`
- TabPFN HUST/Virkler/NASA/C-MAPSS: `FAIR_PFN_COMPARISON_KO.md`
- TabPFN Sunwoda/RWTH/MATRb2: `TABPFN_EXTERNAL_BATTERIES_RESULTS_KO.md`
- 문헌 SOTA·대표 모델과 protocol 대응: `LITERATURE_SOTA_MAPPING_KO.md`
- BatteryLife 공식 CPGRU·CPTransformer backbone strict-tail 재실행: `BATTERYLIFE_BACKBONE_STRICT_TAIL_RESULTS_KO.md`
- PP 구성요소 matched ablation: `FINAL_PP_COMPONENT_ABLATION_RESULTS_KO.md`
- 배터리 boundary executor 개선: `UNIFIED_DUAL_SCALE_PP_IMPROVEMENT_KO.md`
- 실패 도메인 NN safety route 감사: `FAILED_DOMAIN_SAFETY_CONTINUATION_KO.md`
- NASA milling known-boundary route: `results/milling_boundary_quotient_route_v1/results.json`
- XJTU/FEMTO structural routes: `REMAINING_FAILURES_STRUCTURAL_PP_KO.md`
- 남은 음수 R2의 식별가능성 한계: `NEGATIVE_R2_IDENTIFIABILITY_AUDIT_KO.md`
