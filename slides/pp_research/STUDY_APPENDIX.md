# PP 공부용 부록 색인

발표 고정본: `ppt/pp/PP_Research_Detailed_v2.pptx` (본문 + Appendix)

## 명칭

- **SAAR** = Support-Aware Affine–Residual Network = **최종 PP** (adaptive dual-scale)
- 상세: `ppt/pp/SAAR_NAMING.md` · `pp-extrapolation/SAAR_NAMING_KO.md`

## 권장 공부 순서

1. `ppt/외삽자료/extrapolation-papers/외삽_문헌조사_통합.pptx`
2. `ppt/외삽자료/extrapolation-papers/외삽_50분_장표별_설명_v5.md`
3. PPT 본문 A–D
4. PPT Appendix: 데이터셋 표 → 스플릿 규칙 → 성공/실패 지도 → 결과 표
5. 아래 md 원문

## 부록에서 보는 것

| 슬라이드 주제 | 내용 |
|---|---|
| PP 연구 바운더리 | unit-disjoint out-of-support RUL / equation-free |
| 외삽 범위 규칙 | 계약→unit분리→hull-out→val-only |
| 조건 적합도 | 6항 검사 · Pass/Weak/Fail · 예측+적용증명 |
| 게이트 on/off | always-on vs val 증거 게이트 |
| 데이터셋 한눈에 | 도메인 · 추론 타깃(RUL 등) · 외삽 타입 |
| 각 데이터셋 설명 | HUST·Virkler·NASA·Sunwoda… 무엇인지 |
| 데이터별 스플릿 | 좌표·hull·LOBO 등 어떻게 쪼갰는지 |
| 공통 스플릿 | unit 분리 · train support · hull-out · val-only 선택 |
| 언제 잘 되나 | 경계·방향 맞음 + val/test 이동 유사 |

## 핵심 원문 (pp-extrapolation/)

| 주제 | 파일 |
|---|---|
| 문제·분할 | `MODEL_AND_SPLIT_KO.md` |
| 추가 배터리 프로토콜 | `ADDITIONAL_REAL_DATASETS_PROTOCOL.md` |
| 최종 통합 결과 | `UNIFIED_SUPPORT_GATED_PP_RESULTS_KO.md` |
| 전체 데이터 비교 | `ALL_DATASET_COMPARISON_KO.md` |
| hull·경쟁 | `ENGRESSION_GP_EXTRAPOLATION_RESULTS_KO.md` |
| 적용 한계 | `APPLICABILITY_NOVELTY_LIMITS_KO.md` |
| 경계-몫 PP | `BOUNDARY_QUOTIENT_PP_RESULTS_KO.md` |
| 경쟁 비교 | `ALL_DATASET_EXTRAPOLATION_COMPETITORS_KO.md` |
| 통계 | `STATISTICAL_NOVELTY_EVIDENCE_KO.md` |
| 노벨티 감사 | `NOVELTY_LITERATURE_AUDIT_KO.md`, `BQ_PP_NOVELTY_AUDIT_KO.md` |
| 투고 권리 | `PP_FIRST_PUBLICATION_CLAIMS_KO.md` |
| 확증 | `MATR_BATCH2_CONFIRMATORY_RESULTS_KO.md` |
| 전체 프로그램 | `ASSUMPTION_AWARE_EXTRAPOLATION_PROGRAM_KO.md` |

## 실패·미접촉

- `XJTU_UNTOUCHED_*`
- `OXFORD_*`
- `FEMTO_PP_PROSPECTIVE_*`
- `MILLING_LOCKED_*`
- `ADDITIONAL_REAL_DATASETS_*`

## 문헌 폴더

- `ppt/외삽자료/extrapolation-papers/INDEX.md`
- `ppt/외삽자료/extrapolation-papers/paper_pdfs/`
