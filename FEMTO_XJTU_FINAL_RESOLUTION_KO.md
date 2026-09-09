# FEMTO·XJTU 최종 정리와 재실행 기준

이 문서는 낮은 성능을 숨기거나 test에 맞춰 모델을 반복 선택하는 대신, 두 데이터의 유효한 최종 상태와 다음에 필요한 정보를 구분한다. 수치는 `experiments/failed_domain_data_audit.py`로 재생성한다.

## FEMTO: loader 문제는 해결했고, endpoint 정보 문제는 남는다

`femto_sensor_adapter_v2`는 여섯 열 중 실제 horizontal/vertical acceleration인 4·5열만 사용한다. `Full_Test_Set`은 읽지 않고, 공개된 truncated `Test_set`의 마지막 관측과 공식 RUL만 평가한다. 따라서 예전의 시간 열 입력 및 미래가 섞인 prefix 결과는 폐기한다.

그러나 공식 test는 bearing당 RUL endpoint 한 개뿐이다. 11개 endpoint의 RUL은 339–7,570초로 크게 다르며, bearing별 trajectory R²나 strict-tail curve coverage는 이 표본에서 계산할 수 없다. 올바른 비교는 11개 endpoint의 pooled R²·RMSE·MAE와 bearing-level paired error bootstrap이며, 이것도 작은 표본이라는 한계를 명시한다.

수정 loader로 실행한 waveform direct safety route의 ensemble pooled R²는 약 0.075였지만, 다섯 개 개별 seed R²는 모두 음수였다. 따라서 이를 PP 우월성이나 안정적인 FEMTO 성공으로 쓰지 않는다. 현 PP 논문의 주 표에서는 FEMTO를 **causal-loader repaired negative/limited-evidence case**로 두고, 유의성 표와 승패 집계에서 제외한다.

FEMTO를 완전히 해결하려면 새 아키텍처보다 먼저 다음이 필요하다.

1. test bearing의 여러 prefix에서 RUL이 공개된 multi-point holdout, 또는 별도 run-to-failure bearing cohort,
2. bearing 제조·하중·윤활 등의 lifetime-scale covariate,
3. 해당 추가 cohort에 대해 train-only waveform encoder와 route를 동결한 새 평가.

## XJTU: target-range mismatch를 분리해 해결한다

XJTU split은 train condition-1, validation condition-2, test condition-3의 operating-condition transfer다. test 행의 70% 이상이 train 최대 RUL을 넘고 test 평균 RUL은 train 평균보다 수 배 크다. 따라서 RUL을 하나의 공통 scale에서 직접 회귀하는 original PP는 fit range 밖에서 제한된다.

개발 단계에서는 `log1p(RUL / position)`을 사용하고 validation의 scale 오차를 반대쪽 condition으로 반사하는 **reflected-scale PP**가 pooled R² 0.257, seed 평균 0.252±0.006까지 올랐다. 이 구조는 raw waveform spectrum과 causal position만 사용하며 test label은 scale 계산에 쓰지 않았다. 다만 XJTU test를 이미 본 뒤 설계됐으므로 development supplement다.

최종 PP 주 표의 XJTU는 original frozen PP의 외부 transfer 실패를 유지한다. reflected-scale route는 “target-scale transport가 필요한 failure mechanism”의 ablation으로 제시하고, 같은 사전고정 route를 아직 열지 않은 **새 operating-condition bearing cohort**에 적용했을 때만 PP-X 확장 모델의 확증 결과가 된다.

## 결론

두 경우 모두 hyperparameter를 더 오래 탐색한다고 해결되는 문제가 아니다. FEMTO는 평가 표본과 lifetime identity가 부족하고, XJTU는 train과 test의 latent lifetime scale이 달라진다. PP 논문은 이 두 failure mode를 명시하고, 관측 가능한 prior contract가 있는 엄격 외삽 설정의 성능으로 범위를 제한한다.
