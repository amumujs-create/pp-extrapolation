# TabPFN: Sunwoda·RWTH·MATR batch 2 외삽 결과

PP의 고정 외삽 test 행에서 로컬 TabPFN v3를 실행했다. 모델 선택과 hyperparameter 선택에 test label을 사용하지 않았으며, pooled R²를 주 지표로 계산했다.

| 데이터셋 | 분할 | TabPFN 5-seed pooled R² | TabPFN 예측 평균 ensemble pooled R² | 사용 train 행 / 전체 train 행 | test 행 |
|---|---|---:|---:|---:|---:|
| Sunwoda | unseen-cell tail | −0.886 ± 0.033 | −0.886 | 1,000 / 1,312 | 1,468 |
| RWTH | unseen-cell tail | −2.175 ± 0.106 | −2.174 | 1,000 / 2,295 | 1,859 |
| MATR batch 2 | 30/9/9 cell, capacity tail | 0.616 ± 0.041 | 0.618 | 967 / 11,553 | 733 |

실행은 seed 42–46, CPU TabPFN v3, seed당 estimator 하나로 고정했다. 로컬 CPU 프로토콜에 맞춰 train은 unit별 균등·시간순 균등 간격으로 최대 1,000행만 사용했다. 따라서 이 표는 **동일 정보·동일 test의 보조 비교**이며, TabPFN의 전체 train-data 최적 성능 주장에는 사용하지 않는다.

Sunwoda와 RWTH에서는 TabPFN이 해당 엄격 tail 외삽에서 음수 R²를 보였고, MATRb2에서는 양수지만 최종 개발 PP(0.862)와 일반 NN ensemble(0.744)보다 낮다. 혼동을 막기 위해 구분하면, 원본 봉인 confirmatory PP/support-PP는 각각 0.471/0.523이었고, 이후 validation으로 구조·transport를 선택한 최종 개발 PP가 0.862다. 따라서 최종 PP와 TabPFN의 모델 성능 비교에서는 PP가 이기지만, 독립 confirmatory PP 증거로는 0.523만 사용할 수 있다. TabPFN 설정 자체는 test label로 선택하지 않았다.

재현 산출물은 `results/tabpfn_external_batteries_v1/results.json`, 각 데이터셋의 `predictions_*.npz`, 그리고 영문 실행 표 `results/tabpfn_external_batteries_v1/RESULTS.md`에 저장했다.
