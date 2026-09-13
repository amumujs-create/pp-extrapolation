# MICH 외삽 실패 후속 진단: 데이터 구성 × 경계 구조 × 재학습

## 결론과 채택 판단

기존 기록 `MAIN9_TRANSFER_DIAGNOSIS_KO.md`의 미완료 비교를 이어 실행했다.
새 모델을 만든 실험이 아니라, 기존 PP-X가 회복되는 이유를 분리하는 진단이다.
따라서 **후속 모델 채택 없음, 전체 benchmark 우월성/모델 노벨티 입증 없음**이다.

기존 공동학습 dual-scale PP-X를 같은 3 seeds로 재학습한 예측은 보관 예측과
정확히 일치했다(정규화 예측 최대 절대 차이 0). MICH R²는 .71003이다.
같은 공동학습·재학습에서 경계식 없는 대조군은 -1.13583, 고정 경계 모델은
.46814였다. 이 조건에서는 경계 기반 파라미터화와 dual-scale 설정이 유효하다.

하지만 MICH 단독학습에서는 공동학습보다 높은 점수도 나왔다. 이미 확인한 test에서
가장 높은 조건을 골라 새 개선 모델이라고 발표하면 안 된다. 데이터 구성과 재학습의
효과가 구조에 따라 달라지는 것이 이번 진단의 핵심이다.

## 고정 프로토콜

- seeds 42/43/44, 11개 과거 관측 기반 특징, 동일 MICH test 202행·8개 장비.
- 모든 원본 데이터는 early-health train/validation, 더 낮은 health의 source test를
  사용한다. `mich_all`의 all은 **원본 early 학습 구간 전체**이지 test 후기 구간을
  학습에 넣었다는 뜻이 아니다. train/validation/test 장비도 분리된다.
- `mich_filtered`: 앞선 진단의 health 추가 필터를 유지한 1439 train / 259 validation.
- `mich_all`: 추가 필터 없는 원본 MICH 1924 train / 706 validation.
- `joint_all`: SUNWODA/RWTH/MICH 공동 6734 train / 2182 validation.
- 선택은 cap 500 / patience 70, 데이터셋별 정규화 validation MSE의 동일 가중 평균.
  test를 이용한 checkpoint 선택은 없다. 모든 조건을 보고한다.
- 재학습은 같은 seed에서 재초기화해 train+validation으로 선택 epoch만큼 학습한다.
  별도 checkpoint 선택은 하지 않는다. refit 없는 결과도 함께 저장한다.
- 입력 스케일은 각 데이터셋의 원본 train에서 계산해 모든 조건이 공유한다.
  앞선 filtered-only 진단의 스케일과는 다르므로 이전 수치와 단일 요인 비교는 아니다.

### 구조 대조군의 의미

`direct_softplus`와 `bq_fixed`는 같은 frozen affine + bounded SiLU 신경망,
폭 64, ridge alpha 1000, AdamW, 데이터셋/장비 가중 손실을 쓴다.
direct는 출력 계수를 1로 두며, BQ는 알려진 경계까지의 margin을 곱한다.
입력 특징에는 두 모델 모두 실제 margin이 남아 있다.
**출력 계수와 affine 초기화의 지도 target(y 대 y/m)이 함께 바뀌므로,
곱셈 하나만의 인과효과를 분리한 것은 아니다.**

`bq_dual`은 기존 PP-X의 동결된 dual-scale preset이다. 새 구조가 아니다.
Engression은 공식 구현의 hidden64/layers2/noise32, lr .001, beta1,
standardization, 128회 MC 평균을 사용한다. 같은 데이터·선택 점수·epoch 상한·
재학습 규칙이지만 native Adam/행 가중 손실은 유지한다. 동일 FLOPs나 손실함수
비교는 아니다. 모든 모델은 음수 출력만 0으로 제한하고 상한 clipping은 하지 않는다.

## MICH 결과: 3-seed 앙상블 R²

| 데이터 구성 | 모델 | 재학습 없음 | train+validation 재학습 |
|---|---|---:|---:|
| 추가 필터 MICH | direct softplus | .80346 | .48048 |
| 추가 필터 MICH | fixed BQ | .63562 | .20888 |
| 추가 필터 MICH | dual BQ | .67340 | .32501 |
| 추가 필터 MICH | Engression | -1.45199 | -1.58907 |
| 원본 MICH | direct softplus | .51793 | .72534 |
| 원본 MICH | fixed BQ | .86182 | .81655 |
| 원본 MICH | dual BQ | .85931 | .80565 |
| 원본 MICH | Engression | -1.74259 | .05999 |
| 공동학습 | direct softplus | -.16925 | -1.13583 |
| 공동학습 | fixed BQ | .43387 | .46814 |
| 공동학습 | dual BQ (기존 PP-X) | .32256 | .71003 |
| 공동학습 | Engression | -4.14518 | -3.49244 |

## 공동학습·재학습 후 세 데이터셋의 coverage

| 모델 | SUNWODA R² | RWTH R² | MICH R² | 양의 R² 데이터셋×seed | 양의 R² 장비 |
|---|---:|---:|---:|---:|---:|
| direct softplus | .70049 | .69951 | -1.13583 | 5/9 | 16/25 |
| fixed BQ | .92185 | .87090 | .46814 | 9/9 | 24/25 |
| dual BQ (기존 PP-X) | .90955 | .83941 | .71003 | 9/9 | 25/25 |
| Engression | -.05235 | .20247 | -3.49244 | 2/9 | 14/25 |

장비별 점수는 3-seed 앙상블로 계산했다. 기존 PP-X가 이 고정 Engression 설정보다
높고 coverage도 좋지만, 이는 새 후속 모델의 성과가 아니다. 또한 이 표의 9/9는
**3개 데이터셋 × 3 seeds**이며 원래 전체 9개 benchmark의 9/9와 다르다.
Engression 하이퍼파라미터 탐색 전체를 수행한 비교도 아니다.

## 확인된 해석

1. **직접 신경망 실패 전체를 경계식 부재로 설명할 수 없다.** 추가 필터 MICH에서
   direct softplus만으로 .80346을 얻었다. 앞선 CIST 직접 경로와는 초기화·활성화·
   손실·정규화 등이 다르므로, 어느 변경 하나가 회복 원인인지는 미분리 상태다.
2. **경계 파라미터화의 이득은 조건부다.** 원본 MICH와 공동학습에서는 fixed BQ가
   direct보다 좋지만, 추가 필터 MICH에서는 반대다.
3. **공동학습이 MICH 성능 향상 원인이라는 단정은 기각한다.** 재학습 후 dual BQ는
   원본 MICH 단독 .80565에서 공동 .71003으로 낮아졌다. 다만 공동 설정은 학습
   데이터뿐 아니라 validation 목적도 3개 데이터 평균으로 바뀐다. 데이터 추가만의
   인과효과가 아니라 공동 학습/선택 절차의 비교다.
4. **재학습도 일관된 개선이 아니다.** 공동 dual BQ는 .32256→.71003으로 회복되지만
   원본 MICH dual BQ는 .85931→.80565로 감소한다. 이 비교는 데이터 추가와 재초기화,
   초기 affine 재적합 및 모델 내부 표준화 재적합까지 포함한 전체 refit 절차다.
5. **평균 R²만으로 선택하면 coverage를 놓친다.** 원본 MICH/no-refit fixed BQ는
   R² .86182로 dual의 .85931보다 약간 높지만, 장비별 양의 R²는 7/8 대 8/8이고
   최악 장비 RMSE는 10.5747 대 9.8706이다. 최고 평균 점수가 유일한 채택 기준은 아니다.

## 범위와 남은 개발 과제

coverage는 여기서 데이터셋/seed/장비별 양의 R²와 최악 장비 오차를 의미한다.
확률적 예측구간의 포함률을 측정한 실험은 아니다. 동일 seed 앙상블의 비교이며
3 seeds만으로 통계적 유의성을 주장하지 않는다.

이번 결과는 기존에 열어 본 source cohort의 사후 진단이다. 고득점 조건을
test로 선택한 후 우월성이 검증됐다고 주장하지 않는다. 다음 모델링 단계의 제약은
경계식을 단순 제거하거나 전 데이터에 강제하는 것 모두 피하고, **train/validation만으로
구조의 적합성을 학습할 수 있는지**를 검증하는 것이다. 단순 조건별 스위칭은 새로운
노벨티로 계산하지 않는다. 구체적 새 구조의 신규성 검토·구현 및 잠근 외부 평가에서
full PP-X/Engression 대비 정확도와 coverage를 모두 통과하는 작업은 아직 남아 있다.

## 재현 기록

- 실행: `experiments/mich_pool_boundary_refit_factorial.py`
- 별도 검증: `experiments/verify_mich_pool_boundary_refit.py`
- 저장: `results/mich_pool_boundary_refit_factorial_v1/`
- protocol/data_audit/results/scales JSON 및 모든 조건의 seed별 예측 NPZ 보관.
- 3 pools × 4 models × 2 refit 조건 = 24개 결과, 각 3 seeds의 총 72개 학습 모델.
- 별도 검증 완료: 24조건 동일 MICH 정답·장비·행 순서, 40개 데이터셋별 결과의
  R²/RMSE/장비 평균·최악 RMSE/seed R² 재계산, 보관 PP-X 예측 일치.
  `verification.json`에 검사 결과를 저장했다. train+validation endpoint health 최솟값
  대 test 최댓값은 SUNWODA 980.70001 > 980.20001, RWTH .900016 > .899827,
  MICH .870763 > .869915로 세 데이터 모두 엄격한 lower-health 외삽 분리를 확인했다.
- 첫 JSON 저장 중 NumPy 정수 직렬화 오류가 나서 Python int로 수정 후 resume했다.
  최초 두 NPZ는 덮어쓰지 않고 재현 일치를 검사했다. 첫 Engression 실행 전에
  refit도 선택 학습과 같은 1-epoch 호출 반복으로 통일했다. 원본 protocol hash와
  수정 script hash를 resume.json에 함께 보존했다.
