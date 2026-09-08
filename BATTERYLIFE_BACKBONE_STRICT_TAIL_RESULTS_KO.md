# BatteryLife 공식 backbone의 strict-tail 재실행

BatteryLife의 공식 `CPGRU`와 `CPTransformer` backbone을 PP의 고정 Sunwoda·RWTH unseen-cell tail split에 재실행했다. 원 BatteryLife task는 초기 charge/discharge curve로 전체 수명을 예측하므로, 원 data loader와 target을 그대로 사용하면 PP의 현재 시점 RUL task와 공정하지 않다. 따라서 **backbone은 공식 구현을 유지**하고, 입력만 PP와 동일하게 과거 8시점의 health·degradation-rate·상대시간으로 바꿨다. 미래 관측이나 종료 cycle은 입력하지 않았다.

각 backbone은 validation MSE로 3개 architecture/optimizer 후보 중 하나를 선택하고, seeds 42–46으로 5회 재학습했다. 수치는 test pooled R²이며 `ensemble`은 다섯 예측의 평균이다.

| 데이터셋 | 최종 PP | BatteryLife CPGRU seed R² | CPGRU ensemble | BatteryLife CPTransformer seed R² | CPTransformer ensemble |
|---|---:|---|---:|---|---:|
| HUST protocol-tail | **0.958** | 0.698, 0.006, 0.899, −0.192, 0.014 | 0.414 | 0.424, 0.667, 0.541, 0.407, 0.078 | 0.469 |
| Sunwoda unseen-cell tail | **0.865** | −0.165, 0.647, −1.355, −0.374, 0.030 | −0.134 | −3.388, −3.492, −5.116, −5.280, −7.086 | −4.791 |
| RWTH unseen-cell tail | **0.743** | 0.857, −0.408, 0.852, −0.780, −0.619 | 0.322 | −1.334, −3.083, −2.677, −2.499, −1.196 | −2.090 |
| MATR batch 2 final-PP strict tail | **0.862** | 0.721, 0.054, 0.664, 0.015, 0.476 | 0.537 | 0.851, −1.412, −1.001, 0.591, 0.565 | 0.380 |

HUST·Sunwoda·RWTH와 최종 PP의 MATRb2 train 행에서는 최종 PP가 높다. CPGRU는 일부 seed에서 양수 R²를 얻지만 재학습 분산이 크다.

앞서 보고한 MATRb2 CPGRU 0.912는 test 733행은 같지만 봉인 confirmatory train 11,553행을 사용했다. 최종 PP 0.862는 최저 경계 행을 다시 제외한 11,552행으로 개발됐으므로 둘을 직접 비교하면 안 된다. 정확히 11,552행에 맞춘 CPGRU는 0.537이다. 0.912는 삭제하지 않고 protocol-sensitivity 감사 결과로만 보존한다.

이 결과는 “BatteryLife 원 논문 benchmark를 이겼다”는 주장이 아니다. **공식 SOTA backbone을 PP의 더 엄격한 외삽 정보 조건에 맞춰 재실행한 직접 비교**다. 원 benchmark 점수와의 차이는 `LITERATURE_SOTA_MAPPING_KO.md`에 설명했다.

MATRb2에서 5-seed 개별 성능까지 맞춘 통계 비교는 `MATR_BATCH2_FIVE_SEED_COMPARISON_KO.md`에 정리했다.

재현 코드: `experiments/batterylife_strict_tail_adapter.py`  
원시 결과: `results/batterylife_strict_tail_v1/`, `results/batterylife_strict_tail_cptransformer/`, `results/batterylife_strict_tail_rwth/`
