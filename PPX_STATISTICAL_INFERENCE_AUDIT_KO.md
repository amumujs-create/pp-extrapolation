# PP-X 통계 추론 감사 (소표본 비모수 티어)

작성자: 박진서  
프로토콜: `protocols/PPX_STATISTICAL_INFERENCE_PROTOCOL.md`  
실험: `experiments/ppx_statistical_inference_audit.py`  
결과: `results/ppx_statistical_inference_audit_v1/results.json`

## 질문에 대한 답

**모수(표본)가 적을 때는 비모수 / exact 추론을 쓴다. 맞다.**  
다만 그건 **검정·CI 규칙**이지, 예측기를 Engression 같은 비모수 모델로
바꾸라는 뜻이 아니다.

| 구분 | 소표본에서 |
|---|---|
| 추론 | exact sign-flip, Wilcoxon, exact binomial, bootstrap CI |
| 예측기 | contract·예산에 따라 선택 (소표본만으로 클래스 전환 금지) |
| 해석 단위 | 물리 unit → dataset (row 독립 가정 금지) |

## 이번이 채운 공백

기존 `journal_statistical_evidence`에 있던 sign-flip / BH / hierarchical
bootstrap에 더해:

1. **Wilcoxon signed-rank**를 공식 보조 검정으로 고정  
2. **\(n_u\) 티어** (L0/L1/L2) 라벨  
3. **운영 정책(FA/FR/accuracy/deployed effect) dataset-bootstrap CI**  
4. “소표본 → 비모수 예측기” 혼동 금지 문장

## 12-setting 결과 요약

- 티어: 전부 **L1_exact** (unit 3–20). L0/L2 없음.
- Wilcoxon vs sign-flip α=0.05 판정: **12/12 일치** (불일치 0).
- 도메인 방향: 6/12 양수, exact binomial **p=1.0**  
  → common-backbone PP가 전 도메인에서 우세하다는 주장은 불가 (기존 가드레일과 동일).
- equal-dataset mean log-RMSE ratio ≈ **+0.082**, hierarchical 95% CI
  **[-0.314, +0.504]** (0 포함).

### BH 보정 후 두드러진 도메인 (sign-flip 주 검정)

| 데이터 | n | mean | sign p | Wilcoxon p | BH q | 해석 |
|---|---:|---:|---:|---:|---:|---|
| virkler | 10 | +1.61 | .002 | .002 | .023 | PP 이득 유의 |
| sunwoda | 9 | +1.34 | .004 | .004 | .023 | PP 이득 유의 (stage2는 FR) |
| mich | 8 | −0.33 | .008 | .008 | .031 | PP 악화 유의 → 거절이 맞음 |
| matr | 10 | −0.28 | .020 | .020 | .059 | 악화 경향 (q≈0.06) |

nasa_battery(n=4), ncmapss(n=3)는 L1이지만 **검정력이 약함** — 점추정만으로
확증하지 않는다.

## 운영 정책 bootstrap (dataset 재표집)

점추정: accept 6, accuracy 0.667, FA 2, FR 2, deployed effect +0.041

| 지표 | 95% CI |
|---|---|
| accuracy | [0.42, 0.92] |
| FA | [0, 5] |
| FR | [0, 5] |
| deployed effect | [−0.28, +0.41] |

**정책 효과 CI가 0을 포함**한다. unit-risk gate가 “미래 도메인에서 확실히
이득”이라고 말할 통계 근거는 아직 없다. 기존 registry 가드레일과 일치.

## 아직 남은 공백

1. **prior-only 예측 부재** → stage-1 OOF prior regret의 비모수 검정 불가  
   (A안 감사는 counterfactual로만 대체).
2. **메인 9-setting final executor** 경로와 common-backbone 통계 패키지의
   숫자를 한 표로 더 단단히 정렬할 여지 (이미 각각은 있음).
3. **prospective** 통계는 DS03 등 별도; 이 감사는 retrospective.
4. unit ≤2인 설정이 생기면 L0로 강제 — 현재 12개에는 해당 없음.

## 실무 규칙 (고정)

- 소표본 → **비모수/exact 추론**  
- 주 검정: **sign-flip**; 보조: **Wilcoxon**  
- 다중성: 도메인 family에 **BH**  
- 요약: **hierarchical bootstrap CI**  
- 예측기 교체는 추론 규칙과 분리
