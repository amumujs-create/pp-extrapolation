# 기존 전이 실패에서 이어간 진단: 입력·학습 예산·장비 분할

후속 기록: 아래 MICH 미완료 요인 비교는 이후
`MICH_POOL_BOUNDARY_REFIT_DIAGNOSIS_KO.md`에서 실행했다.
아래 내용과 수치는 이 진단 시점의 기록으로 보존한다.

## 현재 판단

이전 `main9_anchored_residual_audit_v1`의 실패를 그대로 모델 구조의 실패로
해석하면 안 된다. NASA는 입력 정보와 학습 예산을 통제하면 직접 기준선이 크게
회복된다. MICH는 같은 조치로 해결되지 않는다. N-CMAPSS 누락은 historical
loader와 새 runner의 validation unit contract 불일치로 확인했다.

새 후보를 승격하거나 PP-X를 교체하지 않았다. 이번 결과는 이미 열린 데이터의
진단이며 새 외부 우월성/노벨티/원래 9/9·45/45 coverage 회복을 주장하지 않는다.

## NASA: 같은 255개 RUL 평가 행으로 입력 × 학습 설정 비교

원래 네 battery fold, train/validation/test unit 분리, RUL 정답과 test 행을
유지했다. scalar는 원래 health_phi 1개, causal_short는 health_phi와 최근
3개 이하 관측의 평균·기울기 2개를 추가한 3개 입력이다. 각 시점의 과거만 사용한다.
현재 train/validation/test 행도 두 feature arm에서 완전히 동일하게 유지했다.

짧은 학습은 cap=60/patience=20, 긴 학습은 cap=400/patience=80이다.
따라서 cap만의 효과가 아니라 학습·조기종료 예산 설정의 효과다. 3 seeds의 동일
크기 앙상블을 비교하고, native loss/정규화/optimizer는 유지했다.

| 입력 / 학습 설정 | 직접 CIST | +적분 보정 | Latent PP | Engression |
|---|---:|---:|---:|---:|
| scalar / 60 | -3.92529 | -3.16170 | .54382 | -1.31705 |
| scalar / 400 | .12361 | .14140 | .54382 | .54079 |
| causal_short / 60 | .51190 | .51943 | .57031 | -.43670 |
| causal_short / 400 | .64079 | .63550 | .57031 | .51972 |

값은 동일 test rows의 pooled R2이며 서로 다른 데이터셋을 합친 점수가 아니다.

- scalar/60 직접·적분 예측은 예전 저장 예측과 수치 허용오차 내 정확히 재현됐다.
- scalar/60 직접 모델 12 fits 모두 마지막 60 epoch를 선택했다. 긴 설정에서는
  대부분 108~161 epoch, 한 fit은 400을 선택했다. Engression도 짧은 설정의
  12 fits 모두 60을 선택했고 긴 설정에서는 72~190을 선택했다.
- 같은 짧은 예산에서 과거 요약 추가만으로 직접 R2가 -3.925→.512가 됐다.
  기존 비교에는 구조 외에 정보량 차이가 크게 작용했다는 직접적인 근거다.
- 충분히 학습한 causal_short에서 적분 보정은 .64079→.63550으로 약화했다.
  회복된 정확도를 새 적분 구조의 성과로 귀속할 수 없다.

Latent PP는 동일 입력의 기존 executor 구조이며 separation=0으로 고정했다.
최종 NASA PP-X의 feature-preset search 전체를 재현한 것은 아니다. historical
NASA 학습은 earliest two rows를 제외하지만 이번 통제는 원래 전이 rows를 유지해
초기 1~2 관측에서는 가능한 prefix만 계산했다. 따라서 historical 최종 PP-X보다
높다는 주장도 이 표만으로 하지 않는다.

## MICH: 예산과 특징 변경으로는 실패를 설명/해결하지 못함

원래 1439 train / 259 validation / 202 test rows를 보존했다. 두 예산을 비교하고,
원래 7개 특징과 기존 boundary representation의 11개 특징도 별도로 통제했다.
11개는 margin의 현재/평균/표준편차/변화/기울기, log-rate의 같은 5개 요약,
log-cycle이다. 모든 스케일은 해당 arm의 실제 train rows에서만 계산했다.

| 특징 | 예산 | 직접 CIST | +적분 보정 | Latent PP | Engression |
|---|---|---:|---:|---:|---:|
| 7개 | 60 및 400 (동일) | -.54892 | -.73228 | -1.52219 | -2.09519 |
| 11개 | 60 및 400 (동일) | -1.45943 | -.44560 | -1.56889 | -1.64814 |

7개 입력에서 직접 모델은 epoch 22/19/20, Engression은 25/19/23을 선택했고
Latent PP는 모두 epoch 0이었다. 예산을 늘려도 선택이 같았다. NASA와 달리
단순히 epoch cap에 막힌 실패가 아니다. 11개 입력에서도 회복되지 않았다.

11개 특징에서 적분 보정은 직접 경로보다 개선되지만 최종 R2가 음수여서
사용자의 predictive coverage 조건을 통과하지 못한다.

기존 MICH PP-X는 11개 입력뿐 아니라 다음도 다르다.

1. SUNWODA·RWTH·MICH 세 데이터 공동 학습과 dataset/unit 가중.
2. boundary quotient target parameterization 및 boundary=0 구조.
3. validation에서 epoch을 선택한 후 train+validation 전체로 고정 epoch 재학습.
4. retrospectively selected dual-scale residual 설정.

이번 실험은 이 네 요인을 아직 분리하지 않았다. 따라서 어느 하나가 MICH의
원인이라고 확정하지 않는다. **다음 미완료 비교는 동일 11개 입력·동일 공동학습
및 재학습 조건에서 boundary quotient 유무를 통제하는 것**이다. 그 비교에도
Engression을 같은 train/validation/refit 정보로 포함해야 한다.

## N-CMAPSS: 중복 원인 확인 및 분리 프로토콜 실행 완료

기존 `make_tra_hard_split`은 의도적으로
`val_units = train_unit_ids + val_unit_ids`를 사용한다.

| 구분 | 장비 ID | 행 수 |
|---|---|---:|
| historical train | 2,5,10,16,18 | 5239 |
| historical validation | 2,5,10,16,18,20 | 1810 |
| historical test | 11,14,15 | 159 |
| 새 strict validation | 20 | 292 |

중복은 train↔validation이며 test unit 중복은 없다. historical loader는 다른
TRA 구간의 validation을 허용했지만, 전이 runner는 세 partition이 모두 장비별로
분리되어야 한다고 assert해 `Unit overlap`으로 중단됐다.

historical loader/결과는 바꾸지 않았다. train/test는 그대로 두고 validation만
장비20으로 제한한 **별도 프로토콜**을 저장했으며, 비교 모델을 모두 이 validation으로
다시 학습·선택했다. 60 epoch/patience20, 3 seeds 결과:

| 모델 | R2 | RMSE |
|---|---:|---:|
| 직접 CIST | .94530 | 4.56853 |
| +적분 보정 | .94530 | 4.56853 |
| Latent PP | .78729 | 9.00882 |
| Engression | .91256 | 5.77597 |

적분 보정은 3/3 기각되어 직접 경로와 같았다. 따라서 새 적분 모델의 우위가
아니다. validation unit이 하나이고 full multiscale PP-X가 아닌 Latent PP를
비교했으므로 제한 예산 진단이다. historical 8/9 표에 이 결과를 섞어 9/9 완료라고
표시하지 않는다. strict support 조건과 장비 분리 여부는 실행 중 검사했다.

## 재현과 무결성

- `experiments/main9_transfer_diagnosis.py`: NASA 4조건, MICH 7개 입력 2조건.
- `experiments/mich_representation_control.py`: MICH 11개 입력 2조건.
- `experiments/ncmapss_transfer_strict_resume.py`: 새 disjoint-validation 진단.
- 결과: `results/main9_transfer_diagnosis_v1/`,
  `results/mich_representation_control_v1/`,
  `results/ncmapss_transfer_strict_resume_v1/`.

각 폴더의 protocol.json, results.json, predictions NPZ에 설정, seed, epoch,
unit IDs, 정답과 예측을 저장했다. 표의 값은 저장 예측으로 별도 재계산했다.
이전 NASA·MICH 직접/적분 60-epoch 예측 재현도 확인했다.

main9 runner의 첫 실행은 MICH 결과 JSON 저장 시 numpy int32 unit ID 때문에
중단됐다. 진단 ID를 문자열로 변환하는 출력 수정만 하고 `--resume`으로 이어갔다.
완료된 NASA 조건은 재학습하지 않았고, 이미 저장된 MICH60 NPZ와 재현 예측 일치를
검사했다. 원래 protocol hash와 수정 후 resume hash는 함께 보존했다.

현재 3개 결과 JSON 모두 complete이며, 이번에 계획한 조건 진단은 완료됐다.
모델 노벨티나 원래 전체 benchmark의 개선이 완료됐다는 뜻은 아니다.
