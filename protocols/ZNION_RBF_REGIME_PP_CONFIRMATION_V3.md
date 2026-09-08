# Deterministic RBF-regime PP 순차 확증 v3

## 목적

v2의 eligible cell이 1개에 그친 표본 부족을 해결하기 위해 BatteryLife 공식 Zn-ion test 목록에서 아직 다운로드하지 않은 나머지 13–20번 8개를 모두 후보로 고정한다. 이후 같은 데이터셋에서 추가 확증 후보를 골라 실험하지 않는다.

## 고정 후보

- `205-1_20231205230230_07_4`
- `209-1_20231205230248_07_7`
- `402-3_20231209225844_01_3`
- `406-2_20231209231604_08_6`
- `406-3_20231209231637_08_7`
- `412-1_20231209232958_09_7`
- `413-1_20231209233202_06_2`
- `446-1_20240104212538_07_2`

## 완전 동결 항목

- model: v2의 deterministic single-seed BQ + RBF-attention lifetime prior
- RBF temperature 1.0, BQ seed 42, prefix-slope gate temperature `1e-4`
- extrapolation boundary: 152 cycles
- eligibility: 기존 원본과 SHA-256 중복 제거, 관측된 80%-SOH EOL 필수, boundary-to-EOL tail 50 step 이상
- 후보군 교체 금지
- 비교군: 같은 정보와 split의 single-seed plain MLP

## 판정

eligible unique cell이 2개 미만이면 inconclusive다. 2개 이상이면 PP pooled R²가 0보다 크고 plain MLP pooled R²보다 클 때만 성공이다. unit-macro R², 셀별 R², RMSE, MAE를 모두 기록한다.

이 문서와 실행 코드를 원본 후보군 다운로드 전에 commit한다.
