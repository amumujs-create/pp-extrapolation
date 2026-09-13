# CDCR-PPX 모델 업데이트 결과

> **외부 cohort 후속 판정 (2026-09-13): 범용 메인 모델 승격 철회.**
> 9개 개발 데이터셋에서는 아래 성능 개선이 재현됐지만, Axial-fan 3개 설정에서
> R²가 모두 악화했고 MATWI·Misata에서는 개선이 없었다. 개발용 meta-model
> 결과로만 보존하며 최종 범용 PP-X를 대체하지 않는다.

작성일: 2026-09-13  
모델명: **Cross-Domain Consensus Residual PP-X (CDCR-PPX)**

## 1. 목표

여기서 coverage는 prediction interval coverage가 아니라 다음 의미의
**predictive performance coverage**다.

- 여러 데이터셋에서 안정적인 양의 R²
- 여러 seed에서 안정적인 양의 R²
- 낮은 seed 분산과 높은 worst-seed 성능
- 높은 pooled 성능

기준은 최종 modular PP-X의 9개 데이터셋, seed 42--46이다.

## 2. 모델 구조

최종 PP-X의 다섯 seed 출력을 `f_k`, 평균을 `m`이라고 한다. 각 도메인에서
평균·중앙값·trimmed mean·seed spread·seed range·signed seed deviation을
출력 scale로 정규화해 outcome-free consensus feature를 만든다.

다른 도메인의 정답만 사용한 domain-balanced ridge head가 PP-X 평균의
normalized residual을 예측한다.

```text
q90 normalized seed disagreement >= 0.12이면 meta residual 활성화
corrected mean = m + 0.50 * scale * predicted normalized residual
corrected seed k = corrected mean + 0.50 * (f_k - m)
```

마지막 식은 corrected ensemble mean을 정확히 유지하면서 seed별 편차를 절반으로
수축한다. 각 데이터셋의 head는 해당 데이터셋을 제외한 나머지 8개 도메인으로만
학습하는 leave-one-domain-out 방식이다.

## 3. 동결 설정

- ridge alpha: 100
- q90 normalized seed-disagreement cutoff: 0.12
- meta-residual authority: 0.50
- seed-deviation shrinkage: 0.50
- target-time label 사용: 없음
- held-out dataset label의 해당 head 학습 사용: 없음

이 설정은 열린 9개 데이터셋의 Stage-0 탐색에서 정해졌으므로 완전한 독립
확증이 아니라 retrospective development 결과다.

## 4. 데이터셋별 결과

| Dataset | 최종 PP-X R² | CDCR-PPX R² | ΔR² | seed SD 전 | seed SD 후 |
|---|---:|---:|---:|---:|---:|
| HUST | 0.95796 | 0.95796 | 0.00000 | 0.01304 | 0.00631 |
| Virkler | 0.88798 | **0.89654** | +0.00856 | 0.02118 | 0.00796 |
| NASA | 0.58375 | 0.58375 | 0.00000 | 0.00482 | 0.00242 |
| SUNWODA | 0.93945 | **0.94339** | +0.00394 | 0.02000 | 0.00434 |
| RWTH | 0.87838 | **0.88757** | +0.00919 | 0.01587 | 0.00452 |
| MICH | 0.75123 | **0.76652** | +0.01530 | 0.07663 | 0.03247 |
| MATR2019 | 0.46570 | **0.48041** | +0.01471 | 0.07294 | 0.03092 |
| MATR-b2 | 0.86239 | **0.86957** | +0.00717 | 0.06141 | 0.02885 |
| N-CMAPSS | 0.93727 | 0.93727 | 0.00000 | 0.00750 | 0.00359 |

## 5. 전체 성능 coverage

| 지표 | 최종 PP-X | CDCR-PPX |
|---|---:|---:|
| 양의 R² dataset | 9/9 | **9/9** |
| 양의 R² seed×dataset | 45/45 | **45/45** |
| 개선/동률/악화 dataset | - | **6/3/0** |
| mean dataset R² | 0.80712 | **0.81367** |
| minimum dataset R² | 0.46570 | **0.48041** |
| global normalized pooled R² | 0.91834 | **0.92079** |
| mean seed R² | 0.78788 | **0.80886** |
| minimum seed R² | 0.29861 | **0.41658** |
| seed SD 감소 dataset | - | **9/9** |

Mean dataset R² 개선의 dataset bootstrap 95% CI는
**[0.00287, 0.01033]**이다.

개별 seed pair는 38개가 개선되고 7개가 악화했지만, 모든 dataset에서 seed R²
표준편차가 감소했고 전체 최저 seed R²가 크게 상승했다.

## 6. Engression 비교

동일 equal-budget summary의 Engression과 비교하면 CDCR-PPX의 pooled R²가
9개 setting 모두 높다. 가장 작은 차이는 NASA와 N-CMAPSS다.

- NASA: 0.58375 vs 0.58350
- N-CMAPSS: 0.93727 vs 0.93228
- 전체 setting 승패: CDCR-PPX 9 / Engression 0

이 비교는 point R² 비교이며 seed 안정성 프로토콜이 Engression에 동일하게
적용된 결과는 아니다.

## 7. 판정

요청한 predictive performance coverage 기준의 retrospective 승격 조건은
모두 통과했다.

- dataset coverage 유지
- seed coverage 유지
- dataset ensemble 손실 0
- pooled R² 개선
- worst-dataset와 worst-seed R² 개선
- 9개 dataset 모두 seed SD 감소

다만 모델 hyperparameter가 열린 Stage-0 데이터에서 선택됐기 때문에 논문의
확정 일반화 근거로 쓰기 전 미개봉 cohort에서 frozen replay가 필요하다.

## 8. 재현 파일

- 모델: `src/pp_extrapolation/cross_domain_consensus.py`
- 실험: `experiments/cross_domain_consensus_residual_ppx.py`
- 프로토콜: `protocols/CROSS_DOMAIN_CONSENSUS_RESIDUAL_PPX_PROTOCOL.md`
- 테스트: `tests/test_cross_domain_consensus.py`
- 결과: `results/cross_domain_consensus_residual_ppx_v1/results.json`
- 예측: `results/cross_domain_consensus_residual_ppx_v1/predictions.npz`

관련 테스트 결과: **11 passed**.

## 9. 외부 cohort 후속 replay

- Axial-fan 1P_8F: -0.375 -> -0.410
- Axial-fan 4P_1F: -1.780 -> -1.960
- Axial-fan 4P_8F: -0.289 -> -0.361
- MATWI: -0.137 -> -0.137
- Misata: 0.833 -> 0.833
- 개선/동률/악화: 0/2/3
- seed SD 감소: 5/5 settings

Seed 안정화만으로 구조적 cohort shift를 해결하지 못했다. 자세한 판정은
`CDCR_PPX_EXTERNAL_COHORT_RESULT_KO.md`에 기록한다.
