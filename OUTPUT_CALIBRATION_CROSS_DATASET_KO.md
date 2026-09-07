# Seed-consensus output calibration 교차 데이터 결과

## 고정 규칙

각 seed의 validation prediction에 대해 unit leave-one-out으로 `identity`, `bias`,
`group_bias`, bounded `affine`을 비교한다. Seed exact test와 validation-unit paired
bootstrap을 모두 통과할 때만 calibration을 승인한다. 하나라도 실패하면 원 PP를 그대로
사용한다. Test label로 계수나 승인 여부를 선택하지 않으며 데이터셋별 열화식도 사용하지 않는다.

## 결과

| 데이터셋 | 원 PP pooled R² | 최종 consensus-calibrated PP | 판정 |
|---|---:|---:|---|
| MATR2019 latent PP | 0.257 | **0.466** | 승인, +0.208 |
| HUST | 0.773 | **0.899** | 승인, +0.127 |
| RWTH | 0.507 | **0.743** | 승인, +0.236 |
| Virkler | 0.888 | 0.888 | seed 선택 불일치, identity |
| Sunwoda | 0.865 | 0.865 | identity |
| MICH | -1.522 | -1.522 | identity, PP 실패 유지 |
| N-CMAPSS | 0.934 | 0.934 | identity |

고정한 dual-evidence 규칙은 7개 평가 중 3개를 개선했고 4개를 그대로 유지했으며 악화시킨
데이터셋은 없었다. Seed별 선택을 그대로 적용하면 Virkler가 0.888에서 0.883으로 조금
나빠지므로, seed와 물리 unit 양쪽에서 같은 scale 오류를 진단해야 승인하는 규칙이 필요하다.

NASA battery는 outer fold마다 validation unit이 하나뿐이어서 group leave-one-out 선택을
정의할 수 없다. 따라서 identity fallback을 적용하며 기존 PP pooled R² 약 0.495를 유지한다.

## FT 공정 대조와 해석

MATR에서 FT에도 동일한 validation calibration을 적용했다. FT seed의 선택은
`identity/affine/affine/identity/identity`로 합의 조건을 통과하지 못했다. 합의 조건을
무시하고 seed별 보정을 모두 적용해도 FT는 0.377로 PP의 0.466보다 낮았다.

이 결과는 PP가 일반적인 곡선 형태를 학습하고, validation이 발견한 출력 scale 이동만
제한적으로 운반할 수 있음을 보여준다. 동시에 MICH의 음수 R²는 calibration만으로 구조적
실패를 해결할 수 없다는 한계도 분명히 남긴다. 현재 결과는 MATR 하나에 맞춘 개선보다는
HUST와 RWTH에서도 반복되는 일반화 신호를 제공하지만, 최종 논문에서는 규칙을 고정한
추가 cohort 검증을 외부 확인으로 제시하는 것이 적절하다.

기계 판독 결과: `results/output_calibration_cross_dataset_v1/results.json`
