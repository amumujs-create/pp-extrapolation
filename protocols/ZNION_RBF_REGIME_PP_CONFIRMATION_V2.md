# Deterministic RBF-regime PP 확증 v2

## 고정 모델

seed 분산이 큰 latent MLP를 제거하고 development unit descriptor를 memory로 쓰는 RBF attention head로 바꾼다. descriptor는 median/IQR로 표준화하고 `softmax(-squared_distance / 1.0)`으로 log-EOL을 합성한다. temperature 1.0은 development leave-one-unit-out 오차로 고정했다. prefix slope gate와 BQ 설정은 v1과 같고, BQ는 seed 42 하나만 쓴다. 따라서 주 결과는 ensemble-free다.

## 사전 고정 후보군

BatteryLife 공식 Zn-ion test 목록 7–12번 셀:

- `204-2_20231205230217_07_2`
- `428-2_20231212185058_01_4`
- `430-2_20231212185305_02_7`
- `436-2_20231227204653_03_6`
- `205-3_20231205230239_07_6`
- `415-1_20231209233508_06_8`

파일명은 실행을 위해 사전에 정렬하여 코드에 고정했다.

## 예측 전 eligibility 감사

1. 이전 train/validation/test와 SHA-256가 같은 원본은 제외
2. 관측된 80%-SOH EOL이 없는 오른쪽 중단 셀은 제외
3. 152-cycle 경계 뒤 tail이 50행 미만인 셀은 제외
4. 적합한 unique cell이 2개 미만이면 결과는 inconclusive

이 규칙은 모델 예측을 계산하기 전에 적용하며 후보를 교체하지 않는다.

## 성공 기준

- unique eligible cell 수 >= 2
- PP pooled R² > 0
- PP pooled R² > 같은 split의 single-seed plain MLP pooled R²

프로토콜·코드를 test 원본 다운로드 전에 commit한다.
