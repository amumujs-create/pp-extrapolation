# PP 모델링 개선 재감사

## 결론
기존 결과의 일부 수치 개선은 재현되지만, 통합 PP 신경망의 강건성 개선이나 모든 경쟁모델에 대한 우월성을 입증하지 않는다. 현재 최우선 과제는 FEMTO 입력 수정과 동일 입력 비교다. 새 cohort는 평가하지 않았다.

## 1. FEMTO: 시간 열을 진동으로 읽는 오류 확인
`../ca-css-ncmapss/femto_bearing_loader.py`의 `_rows_from_bearing_dir`는 `arr[:,0]`, `arr[:,1]`을 사용한다.
실제 `data/femto/raw/Learning_set/Bearing1_1/acc_00001.csv`는 2560×6 배열이다.
첫 두 열의 표준편차는 모두 0, RMS는 9와 39다. 캐시의 `rms_h=9`, `rms_v=39`, 양방향 첫 FFT band=0과 정확히 일치한다.
마지막 두 열의 표준편차는 0.561735와 0.435797이며 RMS는 0.561746와 0.435801이다.
따라서 진동 특징을 잘못 생성한 사실이 확인됐다. -0.571은 올바른 진동 입력에 대한 성능이 아니다.

추가로 `femto_features`는 초기 10개 관측의 중앙값을 첫 시점부터 사용한다. 첫 9개 시점에는 미래 관측이 들어간다. baseline은 각 시점까지 관측한 prefix로 계산하거나 충분한 warm-up 뒤 평가해야 한다.

개선 순서: 올바른 채널 명시 → 별도 버전 캐시 생성 → 시간/RUL 단위·stride 마지막 관측점 감사 → causal prefix 수정 → PP와 MLP·monotone 모델을 동일 입력으로 재학습. 기존 캐시는 보존한다. Full_Test_Set의 이후 관측은 공식 truncated test 입력으로 사용하지 않는다.

## 2. XJTU: 출력 상한 제거는 타당하나 새 결과는 Ridge head
`remaining_failures_structural_pp.py`의 XJTU 경로는 sklearn Ridge로 logit progress를 학습한다. NN residual은 없다. -0.843을 통합 PP NN의 구조 개선으로 부르면 부정확하다.
Elapsed time와 log elapsed 입력도 추가됐으므로 기존 경쟁모델과 입력 조건이 달라졌다.

개선 순서: 같은 causal 입력의 MLP/Ridge와 progress 변환 모델을 대조 → capped/uncapped 출력을 matched ablation → train 내부 unit holdout과 여러 시간 tail에서 loss·epoch·구조 선택 → 개체별 진행률/속도 head와 시간 일관성 loss 검증.
고정 총수명 oracle 0.231은 `max(L-t,0)` 한 모델군의 한계이며 임의 NN의 성능 상한이 아니다.

## 3. Milling: 0.341은 보정 수식의 개발 결과
실제 test 예측은 `(0.53-health)/rate`이고 NN은 꺼져 있다. 5-seed 배열은 동일 예측을 `np.repeat`한 것으로 5회 독립 신경망 재학습이 아니다.
공식 RUL label 경계 0.50은 유지했지만 예측식의 유효 경계는 0.53으로 이동했다. 이를 inspection 오차라고 단정할 직접 실험은 없다. 경험적 margin 보정으로 표현해야 한다.
대화 기록에서는 test별 offset 성능을 본 뒤 MAE 선택으로 전환했다. 코드의 마지막 선택 함수가 validation만 사용해도 전체 개발 의사결정은 test 영향을 받았다.

개선 순서: 경험적 보정으로 재표기 → train 내부 unit별 절단으로 보정법/목적함수 선택 → 원래 quotient, 보정 quotient, NN residual을 개별 ablation → 동일 boundary/rate 정보가 주어진 경쟁모델 비교. 실제 검사 간격 정보를 쓰는 지연 모델은 그다음 후보다.

## 4. 통합 구조의 현재 상태
`adaptive_routes.py`는 세 계산 함수와 width 휴리스틱을 제공한다. 데이터셋 이름 없이 경로를 고르고 하나의 학습 절차로 작동하는 통합 executor는 아직 검증되지 않았다.
FEMTO width 8 규칙은 여러 width의 test 결과를 관측한 뒤 제안됐고, group 수로 폭을 정하면 강건성이 보장된다는 이론도 없다. width는 grouped validation에서 다른 값들과 비교해야 한다.
단일 endpoint 평가나 validation 1 unit은 제한 사항이지만, 그것만으로 예측 불가능을 증명하지 않는다.

## 우선순위
1. FEMTO 센서 채널/causal prefix 오류 수정과 모든 비교모델 재실행.
2. XJTU 동일 정보·출력 상한·target 변환을 분리한 ablation.
3. Milling 수식 기여와 NN 기여, 보정 선택의 안정성을 분리.
4. 이후 공통 executor를 정의하고 기존 양호한 데이터에도 동일 규칙을 적용해 회귀 여부 확인.

이번 감사에서는 모델 재학습이나 새 성능 확보를 주장하지 않는다. 기존 성공 설정의 학습 결과를 변경하지 않았다.
