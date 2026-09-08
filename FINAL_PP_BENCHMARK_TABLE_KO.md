# 최종 PP 기준 통합 경쟁모델 표

이 표가 PP 발표와 논문 초안에서 사용할 **현재 최종 개발 모델** 기준 표다. 모든 값은 각 데이터셋의 고정 strict extrapolation test에서 계산한 pooled R²다. `최종 PP`는 validation으로 선택한 구조·학습 설정·승인된 PP 내부 모듈을 반영한다. 따라서 이전의 기본 PP, support-PP, 봉인 confirmatory PP 수치와 섞어 비교하지 않는다.

| 데이터셋 / 외삽 설정 | 최종 PP | 현재 최고 비-PP 경쟁모델 | TabPFN v3 | PP 판정 |
|---|---:|---:|---:|---|
| HUST protocol-tail | **0.958** | GroupDRO 0.934 | 0.218 | 우세 |
| Virkler crack-tail | **0.888** | linear-tail RBF 0.805 | 0.621 | 우세 |
| NASA battery LOO-tail | **0.584** | linear-tail RBF 0.550 | −0.691 | 우세 |
| Sunwoda unseen-cell tail | **0.934** | linear-tail RBF 0.838 | −0.886 | 개선 dual-scale PP 우세 |
| RWTH unseen-cell tail | **0.842** | V-REx 0.645 | −2.174 | 개선 dual-scale PP 우세 |
| MATR2019 strict health-tail | **0.466** | 같은 validation calibrator를 적용한 FT-Transformer 0.377 | 미실행 | 우세 |
| MATR batch 2 strict tail | **0.862** | V-REx 0.850 | 0.618† | 우세 |
| N-CMAPSS hard TRA extrapolation | **0.937** | Engression 0.932 | 0.934 | 동률권에 가까운 우세 |
| MICH unseen-cell tail | **0.751** | direct NN 0.684 | 미실행 | 개선 dual-scale PP 우세 |
| XJTU condition transfer | −1.229 | linear-tail RBF −1.418 | 미실행 | 모두 실패, PP 상대 우세 |
| FEMTO endpoint transfer | −1.378 | monotone NN −0.973 | 미실행 | 패배; 적용 범위 밖 |
| NASA milling material transfer | −4.826 | GroupDRO −0.691 | 미실행 | 패배; 적용 범위 밖 |

## 이 표를 읽는 방법

- **양의 R² 9개 설정**에서는 최종 PP가 현재 동일 행으로 실행된 비-PP 비교모델 중 최고값보다 높다. N-CMAPSS의 PP 0.937 대 Engression 0.932 차이는 작으므로 엄밀한 우월성보다 동률권에 가까운 우세로 쓴다.
- XJTU는 전 모델이 음수이므로 PP의 상대 수치가 더 높아도 성공 데이터셋으로 세지 않는다.
- MICH는 support-adaptive dual-scale boundary executor로 0.751까지 복구됐다. FEMTO·milling은 최종 PP도 해결하지 못했으므로 적용 범위 반례로 유지한다.
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
