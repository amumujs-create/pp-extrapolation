# PP-X 저비용 3고호트 검증

새 학습 없이 저장된 평가 예측을 재집계했다. 결과를 이미 본 세 고호트이므로 retrospective external stress test다. Alloy/MultiStage의 CCMR 경로와 DS03 fallback을 PP-X framework의 실행 경로로 평가했으며 하나의 동일 신경망 검증은 아니다.

| 고호트 | PP-X 경로 | 개체 | PP-X R² | Engression R² | 차이 | PP-X 우위 개체 | PP-X/Eng RMSE 기하비 | unit exact p | BH q |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Alloy A | CCMR trajectory executor | 4 | 0.7643 | 0.9914 | -0.2271 | 0/4 | 5.404 | 0.1250 | 0.2500 |
| MultiStage RPT | CCMR trajectory executor | 15 | 0.9794 | 0.9794 | -0.0000 | 7/15 | 0.947 | 0.6638 | 0.6638 |
| N-CMAPSS DS03 | prospectively selected direct fallback | 6 | 0.8818 | 0.9013 | -0.0195 | 1/6 | 1.285 | 0.0625 | 0.1875 |

## 판정

PP-X가 pooled R²에서 Engression보다 높은 고호트는 0/3이다. 세 고호트 평균 R² 차이는 -0.0822, cohort exact sign-flip p=0.2500다.

이 세 고호트는 최종 9-setting 개발 결과를 외부에서 그대로 재현하지 않는다. Alloy A에서는 Engression이 명확히 우세하고, MultiStage는 사실상 동률이며, DS03에서도 Engression이 우세하다. 따라서 **외부 세 고호트에서 PP-X가 Engression보다 평균 정확도 또는 변동성이 우월하다는 주장은 지지되지 않는다.**

MultiStage의 PP-X 경로는 persistence 대비 최악 개체 regret를 증가시키지 않았다는 기존 결과가 있지만, 이것은 Engression 대비 정확도 우월과 다른 주장이다. DS03는 부적합한 prior를 거절해 PP-X 후보 중 최선 경로를 선택했지만 Engression보다 낮았다.

## 논문에서의 사용

- 주 9개 개발 설정: 높은 성능과 낮은 시드 변동의 retrospective evidence.
- 이 3개 외부 고호트: 평균 정확도 우월을 재현하지 못한 stress test와 적용 범위의 한계.
- 사용할 수 있는 메시지: PP-X는 모든 외부 고호트에서 최고 정확도를 목표로 하지 않으며, 계약에 따라 prior를 거절하거나 위험 제한 경로를 선택한다.
- 사용할 수 없는 메시지: PP-X가 외부 고호트에서도 Engression보다 덜 흔들리거나 항상 우월하다.
- 세 고호트만으로 variance 비교는 정의하기 어렵고 n=3 exact 검정의 최소 양측 p도 0.25다.

## 비용 및 재현성

- 새 학습 비용: 0.
- test prediction은 변경하지 않았다.
- 개체 ID가 있는 동일 행에서 PP-X와 Engression RMSE를 비교했다.
- exact unit 검정은 각 고호트 안에서만 수행하고 세 고호트의 개체를 하나의 독립 표본으로 합치지 않았다.
