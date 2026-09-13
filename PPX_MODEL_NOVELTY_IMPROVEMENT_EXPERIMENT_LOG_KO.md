# PP-X 모형 노벨티 개선 실험 기록 (2026-09-12)

## 1) 실험 목적
PP-X 정책 기반 라우팅에서 모형 노벨티(재현성 있는 개선 의사결정 품질)를 높이기 위해, 기존 
`paper_ppx` 정책에 **단위별 정량 이득의 하한 신뢰구간(CI low) 조건**을 추가한 후 정확도/오탐-미탐 균형이 개선되는지 확인한다.

## 2) 실험 개요
- 데이터셋: N-CMAPSS DS03 prospective 재현셋(12개 데이터셋)
- 변경 전 기준 정책: 기존 `paper_ppx` frozen 정책
  - `minimum_validation_gain = 0.02`
  - `minimum_unit_win_fraction = 0.6`
  - `maximum_worst_unit_rmse_ratio = 1.1`
  - 결과 파일: `results/ppx_paper_policy_audit_v1/results.json`
- 변경 후 후보 조건: 기존 조건 + 단위 이득 CI 하한 필터
  - 추가 조건: `unit_gain_ci_low >= 0.0`
  - bootstrap 기반 CI 계산: 20,000 resamples, 95% CI 하위 2.5%분위수
  - 결과 파일: `results/ppx_paper_policy_audit_ci_v1/results.json`

## 3) 코드 변경 요약
- `src/pp_extrapolation/ds03_prospective.py`
  - `_unit_evidence(...)`에서 단위별 이득(`unit_gain`) 분포 계산 + bootstrap CI low 추정값 반환
  - `prepare_select(...)`가 `PPXCandidateEvidence.unit_gain_ci_low`를 전달하도록 변경
- `src/pp_extrapolation/paper_ppx.py`
  - `PPXCandidateEvidence`에 `unit_gain_ci_low` 필드 추가
  - `select_paper_ppx`에 `min_unit_gain_ci_low` 파라미터 추가 및 필터 조건 반영
- `experiments/ppx_paper_policy_audit.py`
  - 감사 로직에서 validation evidence에 CI low를 기록하고, 재실험 시 `paper_ppx_ci_min0` 정책으로 `min_unit_gain_ci_low=0.0` 적용

## 4) 실험 결과

### 4-1) 정책별 요약

| 정책 | accepted | correct | accuracy | false_accepts | false_rejects | mean_unit_log_rmse_ratio | CI 95%
|---|---:|---:|---:|---:|---:|---:|---|
| paper_ppx (baseline, v1) | 6 | 8 | 0.6667 | 2 | 2 | 0.041386 | [-0.2833, 0.4095]
| paper_ppx + CI low>=0 (ci_min0) | 4 | 10 | **0.8333** | **0** | 2 | 0.154252 | [0.000854, 0.429424]

### 4-2) 변경점
- **정확도 개선:** 0.6667 → **0.8333**
- **오탐 감소:** 2 → **0**
- **보수성 증가:** PP 승인 건수 6 → **4** (femto, milling 탈락)

### 4-3) 승인된 데이터셋 비교
- Baseline `paper_ppx`: `virkler`, `matr_batch2`, `femto`, `milling`, `nasa_battery`, `ncmapss`
- CI 강화 정책: `virkler`, `matr_batch2`, `nasa_battery`, `ncmapss`

## 5) 해석
- CI 하한 필터는 특히 소수 단위 또는 변동성이 큰 데이터셋에서 근거가 약한 승인 결정을 걸러내면서, 실측 성능 기준 의사결정 정확도를 올렸다.
- 단위별 이득에 대한 통계적 보증을 추가해 **false accept(잘못 PP 승인)**를 0으로 줄인 것이 핵심 개선이다.
- 다만 PP 승인 건수 자체는 줄어들어, 더 공격적인 탐색성이 필요한 실험에서는 추가 교정(예: 완화된 CI 임계치) 검토가 필요하다.

## 6) 재현/추적 정보
- 실험 결과 저장 위치:
  - `results/ppx_paper_policy_audit_ci_v1/results.json`
  - `results/ppx_paper_policy_audit_v1/results.json` (baseline)
- 변경 커밋 대상 파일:
  - `src/pp_extrapolation/paper_ppx.py`
  - `src/pp_extrapolation/ds03_prospective.py`
  - `experiments/ppx_paper_policy_audit.py`

## 7) ART-PPX: anchored residual-derivative transport

### 7-1) 가설

PP-X의 출력 residual을 직접 보정하는 대신, train-support 경계에서 시작하는
외삽 경로의 residual derivative를 학습한다. support 경계에서는 correction이
정확히 0이며, authority가 0이면 기존 PP-X를 정확히 재현하도록 설계했다.

### 7-2) HUST Stage-0 사전 동결 조건

- base: 최종 HUST PP-X regime transport, seed 42–46
- context: state, rate, all
- Ridge alpha: 0.1, 1, 10, 100, 1000
- correction bound: target scale의 2%, 5%, 10%, 20%
- authority: 0, 0.25, 0.5, 0.75, 1
- validation group-LOO로만 후보 선택
- 승인 조건:
  - validation RMSE 2% 이상 개선
  - unit win fraction 60% 이상
  - worst-unit RMSE ratio 1.05 이하
  - unit bootstrap 95% CI 하한이 0 초과
- PP-X 대비 test RMSE가 5% 이상 악화되거나 nonzero route가 승인되지 않으면
  후속 데이터셋 실험 중단

### 7-3) 결과

- 최저 validation 후보: all context, alpha 100, bound 20%, authority 1
- validation RMSE: 34.922 → 34.869
- 상대 개선: 0.15%로 승인 기준 2% 미달
- unit win: 10/16
- worst-unit RMSE ratio: 1.039
- bootstrap 개선량 95% CI: [-0.347, 0.541]
- 개선 확률: 0.606
- 결론: correction 미승인, authority 0으로 fallback
- test PP-X 및 ART-PPX: RMSE 30.900, R² 0.9580
- exact fallback 최대 오차: 0

### 7-4) 판정

ART-PPX는 안전성은 보존했지만 derivative transport 자체의 유의한 추가
정보를 입증하지 못했다. HUST의 최종 PP-X regime transport가 이미 이용한
validation 신호와 겹치는 것으로 해석한다. 사전 동결 규칙에 따라 MICH 및
나머지 데이터셋으로 확장하지 않으며, PP-X 메인 구조에도 포함하지 않는다.

관련 자료:

- `protocols/ART_PPX_HUST_STAGE0_PROTOCOL.md`
- `experiments/art_ppx_hust_stage0.py`
- `results/art_ppx_hust_stage0/results.json`
- `ART_PPX_HUST_STAGE0_RESULT_KO.md`
- GitHub commit: `abcbc25`

## 8) 다음 모델 후보

공통 목표는 두 가지를 동시에 만족하는 것이다.

1. PP-X와 동일한 contract coverage 및 exact fallback을 유지한다.
2. 최소 한 데이터셋에서 PP-X와 동일 조건 비교모델을 모두 이긴다.

“공개 SOTA”라는 표현은 최신 외부 방법까지 동일 split·정보·탐색 예산으로
재실험한 뒤에만 사용한다. 그 전에는 “동일 조건 내부 benchmark 1위”로
기록한다.

### 후보 1: Event-Coordinate PP-X

- 우선순위: 1
- 첫 데이터셋: Virkler
- 근거: 기존 event-coordinate flow R² 0.970, PP-X R² 0.888
- 구조: 물리적 시간이나 단순 support distance 대신, validation에서 승인된
  event coordinate 위에서 prior-residual dynamics를 학습
- coverage: event coordinate가 승인되지 않으면 PP-X exact fallback
- 노벨티 주장: 외삽 경로 자체를 사건 진행 좌표로 재매개변수화하는
  contract-conditioned residual dynamics
- 필수 검증: 동일 split·seed·탐색 예산, coordinate ablation, unit bootstrap,
  worst-unit harm, PP-X exact replay

### 후보 2: Contract-Normalized Boundary-Flow PP-X

- 우선순위: 2
- 첫 데이터셋: ISU, 다음 Stanford
- 근거: ISU direct-velocity CTBF RMSE 1.23, direct MLP RMSE 1.73
- 구조: unit별 관측 경계와 외삽 방향을 공통 contract coordinate로 정렬하고
  경계에서 시작하는 vector field를 학습
- coverage: boundary normalization이 불가능하거나 validation에서 기각되면
  PP-X exact fallback
- 노벨티 주장: 이질적인 외삽 경계를 공통 좌표계로 정렬한 뒤 학습하는
  boundary-conditioned flow
- 필수 검증: PP-X·engression 포함 동일 조건 비교, boundary alignment
  ablation, raw/normalized coordinate 비교, unit-level 안정성

### 후보 3: Distributional Tail PP-X

- 우선순위: 3
- 첫 데이터셋: MICH 또는 기존 실험에서 engression 우세가 가장 큰 setting
- 구조: 점 residual 하나가 아니라 조건부 residual distribution 또는 tail
  score를 학습하고, shell별 validation evidence로 deterministic PP-X와
  distributional route 사이의 authority를 선택
- coverage: distributional evidence가 부족하면 PP-X exact fallback
- 노벨티 주장: prior 신뢰도가 낮아지는 외삽 shell에서 residual 값이 아니라
  tail law를 운반하는 risk-calibrated extrapolation
- 필수 검증: engression 동등 예산 비교, likelihood/quantile objective
  ablation, point RMSE와 calibration 동시 평가, tail-shell별 harm 분석

## 9) 다음 실험의 공통 승격 및 중단 규칙

- PP-X 대비 pooled RMSE 악화 2% 이내
- worst-unit RMSE ratio 1.10 이하
- 최소 한 setting에서 동일 조건 비교모델 전체 1위
- unit bootstrap CI 또는 paired sign-flip 검정으로 개선 근거 제시
- route off에서 PP-X exact replay 오차 1e-6 이하
- 첫 데이터셋에서 PP-X 대비 RMSE가 5% 이상 악화하면 즉시 해당 후보 중단
- 권장 실행 순서:
  1. Virkler Event-Coordinate PP-X
  2. ISU Contract-Normalized Boundary-Flow PP-X
  3. MICH Distributional Tail PP-X

## 10) PP-X-nested 구조 후보 3종 Virkler 실험

### 10-1) 실험 목적

PP-X를 exact off-state로 포함해 성능과 coverage를 보존하면서 예측함수
자체의 노벨티를 추가할 수 있는지 세 구조를 동일 Virkler split에서
검증했다.

1. Projected Residual-State
2. Causal Temporal Self-Consistency Projection
3. Prior-Geometry-Conditioned Residual

프로토콜은 실행 전에
`protocols/PPX_STRUCTURAL_TRIO_VIRKLER_PROTOCOL.md`로 동결했다.

### 10-2) 공통 기준 결과

- 최종 support-gated PP-X, seed 42--46
- test PP-X: RMSE 2.5769, R² 0.8880, MAE 1.6004
- test units/rows: 10/20
- 모든 arm의 full-row coverage 유지
- off 상태의 PP-X exact replay 확인

### 10-3) Projected Residual-State

- validation 선택: persistence 1, tube infinity
- 의미: exact PP-X off-state
- test RMSE/R²: 2.5769/0.8880
- 판정: 비활성 fallback; 구조 효과 없음

### 10-4) Temporal Self-Consistency

- validation 선택: projection strength 0
- test RMSE/R²: 2.5769/0.8880
- 판정: 비활성 fallback; 구조 효과 없음
- 원인 해석: late-tail contract에 specimen당 두 row만 있어 causal trajectory
  state가 활용할 history가 부족함

### 10-5) Prior-Geometry Residual

- validation 선택: Ridge alpha 100
- validation RMSE: 3.9100 → 3.6793
- test RMSE: 2.5769 → 3.3830
- test R²: 0.8880 → 0.8069
- test RMSE 악화: 31.3%
- unit win: 2/10
- worst-unit RMSE ratio: 5.820
- unit bootstrap CI: [-0.765, 0.224]
- 판정: 5% harm 기준 초과, 즉시 기각

### 10-6) 최종 결정

세 구조 모두 PP-X 승격 대상이 아니다. exact nesting은 실패 후보에서
PP-X 성능과 coverage를 지켰지만, 그 자체는 구조 효과의 증거가 아니다.
Prior-Geometry arm은 validation false accept 사례로 기록한다.

Virkler에서 다음에 우선할 구조는 이미 독립적인 성능 신호가 있었던
Event-Coordinate PP-X다. Residual-state와 temporal projection은 더 긴
unit history를 제공하는 데이터에서만 재검토한다.

관련 자료:

- `PPX_STRUCTURAL_TRIO_VIRKLER_RESULT_KO.md`
- `experiments/ppx_structural_trio_virkler.py`
- `results/ppx_structural_trio_virkler/results.json`

## 11) FCC-PPX: falsification-calibrated credal conformal 구조

### 11-1) 업데이트

Validation physical-unit regret certificate가 PP를 승인하면 authority set을
`{1}`로 두고, 기각하면 point는 exact fallback으로 전환하면서 uncertainty
authority set은 `[0,1]`로 둔다. 최종 interval은 이 authority segment와 기존
support-scaled conformal radius의 Minkowski sum이다.

기존 PP conformal interval이 새 interval에 항상 포함되므로 같은 표본에서
coverage가 감소하지 않는 구조다.

### 11-2) 12-domain 결과

- equal-domain coverage: 0.8706 -> 0.9057
- pooled row coverage: 0.7267 -> 0.7502
- domain coverage non-inferiority: 12/12
- strictly improved coverage: 4/12
- maximum interval inclusion violation: 0.0
- mean interval width inflation: 16.5%
- point mean unit log-RMSE improvement: 0.154252
- point bootstrap 95% CI: [0.000854, 0.426799]
- point improved/harmed domains: 4/0

### 11-3) 수정 판정

예측구간 coverage 기준은 통과했지만 **메인 모델 승격은 철회한다**. 여기서
coverage를 interval inclusion으로 해석한 것은 연구 목표와 달랐다. 실제 목표는
seed/dataset 전반의 안정적인 R²와 높은 pooled 성능이다. FCC-PPX는 그 지표를
개선하지 않았고 common-backbone SUNWODA에서 기존 PP R² `0.865`를 fallback
`-0.668`로 바꾸는 false reject를 만들었다. 따라서 uncertainty 보조 실험으로만
남기며 최종 PP-X 구조 또는 성능 개선 주장에 포함하지 않는다.

최종 PP-X의 올바른 predictive coverage baseline은 9개 dataset ensemble 모두
양의 R², 45/45 seed-dataset R² 양수, global normalized pooled R² `0.9183`이다.
후속 모델은 이 세 지표와 최저 dataset/seed R²를 동시에 비열화 없이 개선해야
승격한다.

관련 자료:

- `FCC_PPX_MODEL_UPDATE_RESULT_KO.md`
- `protocols/FALSIFICATION_CREDAL_CONFORMAL_PPX_PROTOCOL.md`
- `experiments/falsification_credal_conformal_ppx.py`
- `results/falsification_credal_conformal_ppx_v2/results.json`

## 12) CDCR-PPX: cross-domain consensus residual 구조

### 12-1) 목표 수정

Coverage를 여러 seed와 dataset에서 안정적인 양의 R²를 유지하면서 pooled
성능을 높이는 predictive performance coverage로 정의했다. 최종 PP-X 기준은
dataset 9/9 양의 R², seed×dataset 45/45 양의 R², global normalized pooled
R² 0.91834다.

### 12-2) 구조

최종 PP-X 5-seed 출력으로 affine-scale-free consensus feature를 만들고,
held-out dataset을 제외한 다른 도메인에서 normalized residual head를 학습한다.
q90 normalized disagreement가 0.12 이상일 때 authority 0.5로 residual을
적용하고, seed deviation을 corrected ensemble 중심으로 0.5 수축한다.

### 12-3) 결과

- 양의 R² dataset: 9/9 -> 9/9
- 양의 R² seed×dataset: 45/45 -> 45/45
- dataset 개선/동률/악화: 6/3/0
- mean dataset R²: 0.80712 -> 0.81367
- minimum dataset R²: 0.46570 -> 0.48041
- global normalized pooled R²: 0.91834 -> 0.92079
- mean seed R²: 0.78788 -> 0.80886
- minimum seed R²: 0.29861 -> 0.41658
- seed R² SD 감소: 9/9 datasets
- mean dataset R² gain bootstrap 95% CI: [0.00287, 0.01033]
- equal-budget Engression pooled R² 우세: 9/9 settings

### 12-4) 판정

Retrospective predictive coverage 승격 기준은 모두 통과했다. 다만 ridge,
authority, disagreement cutoff, shrinkage는 열린 Stage-0 데이터에서 선택됐으므로
미개봉 cohort frozen replay 전에는 독립 확증으로 표현하지 않는다.

관련 자료:

- `CDCR_PPX_MODEL_UPDATE_RESULT_KO.md`
- `protocols/CROSS_DOMAIN_CONSENSUS_RESIDUAL_PPX_PROTOCOL.md`
- `experiments/cross_domain_consensus_residual_ppx.py`
- `results/cross_domain_consensus_residual_ppx_v1/results.json`

### 12-5) 외부 cohort 수정 판정

Axial-fan 3개 설정, MATWI, Misata의 기존 5-seed 예측에 개발 9-domain head를
고정 적용했다. 결과는 개선/동률/악화 0/2/3이었다. Axial-fan은
`-0.375 -> -0.410`, `-1.780 -> -1.960`, `-0.289 -> -0.361`로 모두
악화했다. MATWI와 Misata는 correction이 꺼져 기존 PP와 동률이었다.

Seed SD는 5/5에서 감소했지만 predictive performance coverage는 개선되지
않았다. 따라서 CDCR-PPX의 범용 메인 모델 승격을 철회하고 development-only
결과로 남긴다.

관련 자료:

- `CDCR_PPX_EXTERNAL_COHORT_RESULT_KO.md`
- `results/cdcr_ppx_external_cohort_replay_v1/results.json`
