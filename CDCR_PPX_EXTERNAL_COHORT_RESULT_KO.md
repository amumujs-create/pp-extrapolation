# CDCR-PPX 외부 cohort replay 결과

작성일: 2026-09-13

## 판정

**외부 cohort 일반화 실패.** CDCR-PPX는 9개 개발 데이터셋에서는 성능
coverage를 개선했지만, 이미 열려 있던 외부 cohort replay에서는 PP보다
개선되지 않았다.

| Cohort | 기존 PP R² | CDCR-PPX R² | matched MLP R² | 판정 |
|---|---:|---:|---:|---|
| Axial-fan 1P_8F | -0.375 | -0.410 | **-0.068** | 악화 |
| Axial-fan 4P_1F | -1.780 | -1.960 | **-1.349** | 악화 |
| Axial-fan 4P_8F | -0.289 | -0.361 | **-0.132** | 악화 |
| MATWI tool-life | **-0.137** | **-0.137** | -0.151 | 동률 |
| Misata machine | 0.833 | 0.833 | **0.854** | 동률 |

전체 결과:

- R² 개선/동률/악화: 0/2/3
- 양의 R² setting: 1/5 -> 1/5
- mean setting R²: -0.349 -> -0.407
- minimum setting R²: -1.780 -> -1.960
- seed R² SD 감소: 5/5

## 해석

Seed consensus shrinkage는 모든 setting의 seed 분산을 줄였지만 정확도 문제를
해결하지 못했다. Axial-fan의 q90 normalized seed disagreement는
0.166--0.213으로 cutoff 0.12를 넘어 meta-residual이 활성화됐고, 세 설정 모두
더 악화했다.

개발 데이터에서 seed disagreement는 residual correction opportunity로 작동했지만,
Axial-fan에서는 endpoint censoring과 lifetime-scale mismatch를 나타내는
out-of-contract 경고였다. 따라서 disagreement의 크기만으로 correction을 켜는
현재 구조는 의미가 뒤집히는 domain shift를 구별하지 못한다.

MATWI와 Misata는 cutoff 미만이라 meta-residual이 꺼졌고 ensemble은 기존 PP와
정확히 같았다. Seed shrinkage로 개별 seed 분산만 감소했다.

## 결론

CDCR-PPX는 개발 데이터 전용 retrospective improvement로 남기고 범용 PP-X
업데이트로 승격하지 않는다. 다음 구조는 seed disagreement 외에 endpoint
censoring contract, lifetime-scale identifiability, train-to-target prediction
geometry를 함께 검사하고, 식별 불가능하면 correction이 아니라 matched direct
fallback 또는 abstention으로 보내야 한다.

이 replay의 외부 outcome은 이미 열려 있었으므로 새로운 prospective 증거로
세지 않는다.

재현 파일:

- `protocols/CDCR_PPX_EXTERNAL_COHORT_REPLAY_PROTOCOL.md`
- `experiments/cdcr_ppx_external_cohort_replay.py`
- `results/cdcr_ppx_external_cohort_replay_v1/results.json`
- `results/cdcr_ppx_external_cohort_replay_v1/predictions.npz`
