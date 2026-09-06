# Selective extrapolation gate: 첫 고정 진단

## 규칙

Test label을 사용하지 않고 다음 조건을 모두 요구했다.

- source mechanism regime이 train에 포함됨
- held-out-unit pseudo-tail validation R² ≥ 0
- validation에서 PP ensemble MSE가 Ridge보다 최소 2% 감소
- seed 간 예측 표준편차 / validation target 표준편차 ≤ 0.25

## 결과

| 데이터셋 | validation R² | Ridge 대비 MSE gain | seed disagreement | regime covered | 판정 | 실제 test PP R² |
|---|---:|---:|---:|---:|---|---:|
| Sunwoda | 0.580 | 1.90% | 0.074 | 아니오 | ABSTAIN | 0.862 |
| RWTH | -1.599 | 29.06% | 0.058 | 예 | ABSTAIN | 0.506 |
| MICH | 0.270 | 약 0% | 0.000 | 예 | ABSTAIN | -1.522 |

MICH 실패는 test label 없이 거절했지만 성공한 두 데이터도 거절했다. 따라서 이 규칙은
현재 표본에서 failure recall 100%이면서 prediction coverage 0%인 과도하게 보수적인
certificate다. 정확도 표와 함께 내더라도 유용한 selective predictor라고 주장할 수 없다.

이 실험은 두 구조적 문제를 드러낸다. Sunwoda처럼 관측하지 않은 조건에서도 transport가
가능할 수 있으므로 단순 category overlap보다 **조건 간 관계를 검증하는 contract**가
필요하다. RWTH처럼 가까운 pseudo-tail의 절대 R²가 음수여도 더 먼 test에서 상대적 affine
경로가 작동할 수 있으므로 한 개 cutoff의 절대 R² gate는 충분하지 않다.

다음 버전은 이 결과에 threshold를 맞추지 않고, train/validation 내부의 여러 거리 shell로
calibration curve를 만든다. 각 shell에서 baseline 대비 regret와 seed disagreement를
측정하고 source 거리는 calibration된 최대 거리 안에서만 승인한다. 이 규칙을 고정한 뒤
새 untouched cohort에서 risk-coverage curve를 평가해야 한다.

현재 결과의 논문 가치는 성공을 주장하는 데 있지 않고, **mechanism coverage만으로도,
단일 pseudo-tail skill만으로도 적용 가능성을 판정할 수 없다는 반례**를 제공한다는 데 있다.
