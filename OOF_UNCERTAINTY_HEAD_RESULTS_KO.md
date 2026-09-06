# Ensemble-free OOF uncertainty PP

## 방법

Train unit을 3개 outer fold로 나눴다. 각 outer fold를 완전히 가리고, 남은 unit 안에서 inner train/validation을 다시 분리한 뒤 서로 다른 seed의 PP 세 개를 학습했다. 가린 outer unit에서 얻은 residual disagreement만 모아 uncertainty target을 만들었다. 작은 NN uncertainty head는 이 OOF target을 학습한다.

최종 예측은 PP 한 개와 uncertainty head 한 개만 사용한다.

\[
\hat y=\hat y_a+\exp[-\beta d-\gamma \hat u_\phi(x)]\hat r_{PP}
\]

따라서 test 추론에서 PP ensemble이나 test label을 사용하지 않는다. `beta`와 `gamma`는 기존 hull-out validation에서 seed 평균 MSE로 선택했다.

## 결과

| 데이터 | 기존 PP single pooled R² | OOF-UQ PP single pooled R² | 기존 seed SD | OOF-UQ seed SD | 선택 |
|---|---:|---:|---:|---:|---|
| HUST | 0.724 | **0.748** | 0.039 | **0.035** | β=0, γ=16 |
| Virkler | 0.857 | 0.857 | 0.019 | 0.019 | β=0, γ=0 |
| RWTH | 0.506 | 0.506 | 0.020 | 0.020 | β=0, γ=0 |
| MICH | -1.522 | -1.522 | 0.000 | 0.000 | β=0, γ=0 |
| NASA | 산출 불가 | 산출 불가 | — | — | fold당 독립 train cell 부족 |

HUST에서는 ensemble-free head가 정확도와 seed 안정성을 함께 개선했다. 나머지 세 데이터에서는 validation이 uncertainty correction을 끄면서 기존 PP를 보존했다. MICH의 residual 붕괴는 해결하지 못했다.

## 상수 shrinkage 반증 실험

HUST에서 관측별 uncertainty를 모두 validation 평균값으로 바꾼 상수 gate도 `0.745`를 얻었다. 실제 uncertainty head는 `0.748`이다. 따라서 기존 PP `0.724` 대비 총 개선 `+0.024` 중 약 `+0.021`은 전역 residual shrinkage로 설명되고, 관측별 uncertainty가 추가로 설명하는 개선은 약 `+0.003`이다.

이 결과는 head가 완전히 무의미하다는 뜻은 아니지만, 현재 데이터만으로 강한 uncertainty-localization novelty를 주장할 수 없음을 뜻한다. 논문에서는 상수 shrinkage ablation을 반드시 함께 보고해야 한다.

NASA는 각 fold에 독립 train cell이 6개 미만이라 nested unit-OOF disagreement를 구성하면 cycle을 독립 표본처럼 취급하는 pseudo-replication이 된다. 따라서 수치를 억지로 만들지 않고 `not_estimable`로 기록했다.

이 결과는 uncertainty head의 feasibility evidence다. HUST가 이미 개발에 사용된 데이터이므로 독립적인 일반화 증거는 아니다. 구조와 OOF 규칙을 고정한 뒤 충분한 train unit을 가진 untouched cohort에서 확인해야 한다.
