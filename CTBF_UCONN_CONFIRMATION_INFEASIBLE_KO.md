# UConn direct-velocity CTBF 확인: 평가 불가능

작성자: 박진서  
판정: protocol-infeasible exploratory exposure

## 최종 판정

UConn–ISU–ILCC 1.2 Ah cohort는 기존 동결 프로토콜의 최소 표본 조건을
충족하지 않았다.

- eligible cell 수가 기존 기준 30개 미만
- strict-tail validation: 5 rows
- strict-tail test: 7 rows, 3 physical cells
- 기존 PP-X 평가도 이미 `inconclusive`로 종료된 cohort

따라서 이 실행은 CTBF의 confirmatory 성공 또는 실패로 계산하지 않는다.

## 노출된 탐색적 수치

부적합 조건을 확인하기 전에 모델 실행이 진행되어 test 결과가 노출됐다.
삭제하지 않고 투명하게 보존한다.

- direct MLP: RMSE 15.988, R² -0.046
- direct-velocity CTBF: RMSE 17.568, R² -0.263
- CTBF unit wins: 1/3
- unit bootstrap RMSE 차이 95% CI: [-2.288, 2.197]
- boundary maximum absolute prediction: 0

수치상 CTBF가 더 나쁘지만, 7개 row와 3개 cell은 사전 최소조건에 크게
미달하므로 일반화 판단에 사용할 수 없다.

## 프로토콜 이탈

새 confirmation script가 “기존 UConn eligibility와 split을 재사용한다”는
조건은 구현했지만, 기존 프로토콜의 다음 stop rule을 fitting 전에 재검사하지
않았다.

1. eligible cell 30개 미만이면 inconclusive
2. validation/test strict-tail row가 각각 50개 미만이면 infeasible

결과 확인 직후 script에 두 검사를 추가했다. 이 변경은 노출된 결과를
재해석하거나 재실행하기 위한 것이 아니라 동일 오류의 재발 방지용이다.

## 후속 조치

- UConn은 CTBF 확인 cohort 목록에서 제외한다.
- 노출된 UConn test는 향후 pristine confirmation에 재사용하지 않는다.
- 다음 confirmation은 실행 전에 eligibility와 strict-tail 표본 수만
  outcome-free feasibility audit로 확인한다.
- Stanford·ISU에서 발견한 direct-velocity CTBF 가설은 아직 외부 확인되지
  않은 상태로 유지한다.

원시 결과:

- `results/ctbf_uconn_direct_velocity_confirmation/results.json`
- `results/ctbf_uconn_direct_velocity_confirmation/predictions.npz`
