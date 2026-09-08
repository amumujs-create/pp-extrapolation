# Regime-scale PP 사전 동결 확증 v1

## 동결 목적

`7edc6ae`에서 개발한 regime-conditioned latent-scale PP가 아직 열지 않은 공식 Zn-ion test cell에서 재현되는지 확인한다. 이 문서와 실행 코드를 test 파일 다운로드 전에 commit한다.

## 고정 코호트

BatteryLife의 공식 `ZNcoin_test_files` 순서에서 기존 확증 3개 다음인 4–6번 셀을 사용한다.

1. `ZN-coin_412-2_20231209233028_09_8`
2. `ZN-coin_410-2_20231209232626_09_2`
3. `ZN-coin_439-1_20231227204804_04_6`

형성 주기 1–9를 제외하고 cycle 10 용량으로 SOH를 정규화한다. 관측된 SOH가 처음 0.8 이하가 된 주기를 EOL로 삼는다. 오른쪽 중단 셀은 라벨을 외삽하지 않고 제외한다. train 수명 중앙값의 60%로 이미 고정한 152-cycle 경계 뒤만 평가한다. 경계보다 수명이 짧은 셀은 교체하지 않고 `no evaluable tail`로 기록한다.

## 고정 모델

- Boundary-quotient path: width 64, alpha 10, residual bound 0.5, lr 0.001, weight decay 0.01, 300 epochs, patience 50
- Latent lifetime head: descriptor 6 → tanh 8 → log-EOL 1, lr 0.01, weight decay 0.01, full-batch 2000 epochs
- Gate: `sigmoid(prefix_slope / 1e-4)`
- seeds: 42–46
- 비교군: 같은 기존 train/validation 절차의 plain MLP

## 사전 성공 기준

주 기준은 5-seed **mean prediction** pooled R²다. 아래 두 조건을 모두 만족해야 확증 성공으로 판정한다.

1. regime-scale PP pooled R² > 0
2. regime-scale PP pooled R² > plain MLP pooled R²

seed median, unit-macro R², 셀별 R², 단일-seed 범위도 모두 공개하지만 사후에 성공 기준을 바꾸지 않는다.
