# OOF uncertainty PP: 필수 3개 검증 결과

## 1. 구조 보류 cohort

Sunwoda는 OOF uncertainty 구조 개발에는 사용하지 않았지만 이전 PP 연구에서 이미 확인한 데이터이므로 project-level untouched가 아니라 architecture-held-out 결과다. Raw gate는 PP pooled R² `0.8621`을 `0.8612`로 낮췄다. 5개 seed 중 개선은 1개뿐이었다. 손상은 작지만 무손상 주장은 기각된다.

진짜 untouched confirmatory cohort는 현재 로컬 데이터에 없다. XJTU, MATR, Oxford, CMAPSS도 이미 확인했으므로 재사용해 독립 검증이라고 부르지 않는다.

## 2. Unit-level 반복 안정성

Test unit을 5,000회 paired bootstrap했다. HUST에서 ensemble RMSE 차이(OOF-UQ minus PP)는 평균 `-1.69`였지만 95% CI는 `[-4.24, 0.80]`으로 0을 포함했다. 다섯 seed는 모두 개선됐으나 독립 unit 기준 유의성은 확보되지 않았다. Sunwoda는 평균 `+0.24`, CI `[-0.58, 0.91]`로 개선 증거가 없다.

HUST의 protocol은 train 1–6, validation 7–8, test 9–10으로 외삽 regime 자체를 정의한다. 이를 임의로 섞은 repeated split은 연구 질문을 바꾸므로 시행하지 않고 unit bootstrap을 사용했다.

## 3. Uncertainty calibration과 risk–coverage

HUST에서 predicted uncertainty와 기존 PP absolute error의 Spearman 상관은 `ρ=0.225`였다. 낮은 uncertainty 25%의 PP RMSE는 `61.40`, 전체 RMSE는 `71.86`으로 일부 순위 정보가 있다. 그러나 risk가 coverage에 따라 완전히 단조롭지는 않았다.

Sunwoda 상관은 `ρ=-0.032` (`p=0.218`)로 calibration이 없었다. Virkler도 `ρ=-0.684`였고 gate가 꺼졌다. 따라서 uncertainty ranking의 범도메인 일반화는 입증되지 않았다.

## 가장 가능성 있는 최종 경로

기본 모델은 PP로 유지한다. OOF head는 validation에서 관측별 head가 상수 shrinkage보다 최소 0.1% 상대 MSE 개선을 보일 때만 활성화한다. 이 localization certificate를 개발 규칙으로 기록했다.

| 데이터 | certificate | PP single R² | certified PP R² |
|---|---|---:|---:|
| HUST | 승인 | 0.724 | 0.748 |
| Virkler | 거절 | 0.857 | 0.857 |
| RWTH | 거절 | 0.506 | 0.506 |
| MICH | 거절 | -1.522 | -1.522 |
| Sunwoda | 거절 | 0.862 | 0.862 |

0.1% 기준은 Sunwoda 결과를 확인한 뒤 정리한 개발 규칙이므로 Sunwoda를 이 규칙의 confirmatory evidence로 사용할 수 없다. 다음 충분한 unit 수를 가진 새 cohort에서는 이 규칙을 변경하지 않고 한 번 검증해야 한다.

현재 가장 방어 가능한 논문 주장은 “PP가 핵심 예측기이며, nested-OOF localization certificate가 있는 경우에만 ensemble-free uncertainty correction을 실행한다”이다. uncertainty head 자체가 보편적으로 정확하다는 주장은 현재 결과가 지지하지 않는다.
