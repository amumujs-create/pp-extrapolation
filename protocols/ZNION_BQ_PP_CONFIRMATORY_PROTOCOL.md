# Zn-ion BQ-PP 외부 확증 프로토콜

## 목적과 동결 시점

Na-ion prospective gate가 실패한 뒤 개발한 Boundary-Quotient PP를 다른 전지 화학계에서 확증한다. 이 문서와 `experiments/znion_bq_confirmatory.py`는 Zn-ion test xlsx를 다운로드하거나 열기 전에 동결한다.

## 데이터와 분할

- 출처: BatteryLife Raw `ZNion`, 공식 `2021` train/validation/test 목록
- train: 공식 train 목록의 처음 6개
- validation: 공식 validation 목록의 처음 3개
- untouched test: 공식 test 목록의 처음 3개
  - `ZN-coin_422-1_20231205230039_01_7`
  - `ZN-coin_438-3_20231227204754_04_5`
  - `ZN-coin_435-1_20231227204625_03_2`

오른쪽 중단 셀을 고장 라벨로 외삽하지 않는다. 형성 주기(1–9)를 제외하고 10번 주기 방전용량으로 SOH를 정규화한 뒤, 처음으로 `SOH <= 0.80`을 관측한 셀만 사용한다. 이 적합성 규칙은 모델 출력과 무관하다.

## 고정 외삽 설정

- train 셀 EOL 중앙값의 60%까지만 학습한다.
- validation/test는 그 경계 뒤의 late tail로 평가한다.
- 입력은 현재 SOH, 경과 주기, 최근 기울기와 이력 요약이며 미래 값을 사용하지 않는다.
- 주 지표는 pooled R²이며 RMSE, MAE, unit-macro R²를 같이 기록한다.

## 고정 모델과 비교군

Na-ion에서 개발한 구조를 SOH 단위로만 표현한다.

`RUL = max(SOH - 0.80, 0) * softplus(affine + bounded NN residual)`

- width 64, ridge alpha 10, residual bound 0.5
- AdamW: lr 0.001, weight decay 0.01
- max 300 epochs, patience 50
- seeds 42–46
- 비교: 같은 입력의 plain MLP, boundary-affine, BQ-PP

Validation은 프로토콜 실행 점검에만 쓰며 설정을 바꾸지 않는다. Test는 위 3개 파일을 모두 받은 뒤 `--phase test`를 한 번 실행하고, 성공과 실패를 모두 그대로 기록한다.
