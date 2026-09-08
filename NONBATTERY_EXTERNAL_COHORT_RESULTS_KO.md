# 비배터리 신규 코호트 검증 — 2026-09-08

새 도메인인 **NASA Capacitor Electrical Stress-2에서 새 소자 예측 파일럿을 완료**했다. PP가 MLP보다 높았지만 ridge와 거의 같고, health hull 밖 시험점이 0%라 **엄격한 열화 범위 외삽 성공으로 집계하지 않는다**. 기존 성공 HNEI는 재학습하지 않았다.

## 실제 실행 결과

기존 `fit_pp` 기본 frozen-affine + NN residual을 사용했다. 모든 기존 최종 PP 변형/BQ/attention을 재실행한 결과가 아니다. 앞서 개발한 safe-continuation도 고정 후보로 별도 비교했다. 모든 NN은 42–46의 5개 seed, 같은 causal history 입력, 최대300 epoch/early stopping70. 각 seed를 training target 범위로 clip한 다음 평균했다.

| 모델 | Pooled ensemble R² | RMSE (시간) |
|---|---:|---:|
| 기본 PP | 0.989896 | 5.7165 |
| Ridge | 0.989769 | 5.7523 |
| 일반 MLP | 0.975305 | 8.9368 |
| Safe-continuation PP, validation 선택 λ=0.2 | 0.977692 | 8.4938 |
| 경과시간만 이용한 진단 baseline | 0.984020 | 7.1889 |

PP seed별 R²: 0.990352, 0.989795, 0.989509, 0.989677, 0.989454.
MLP seed별 R²: 0.979328, 0.972130, 0.972960, 0.966197, 0.972121.

- Train: 소자1–3 할당. 소자1은 관측기간 내 20% 손실에 도달하지 않아 point-RUL 학습에서 제외. 실학습 소자2–3, 20개 관측.
- Validation: 소자4, 9개 관측. Test: 소자5–6, 20개 관측. 평가시점마다 그 시점까지의 실제 이력을 제공한다.
- 수명: 초기 대비 정전용량 20% 감소의 첫 도달시간. 두 실측점 사이 선형보간으로 산정한 **임계치 도달 RUL**이며, 정확하게 관측된 파괴시간은 아니다.
- PP는 시험 소자2/2에서 MLP 대비 RMSE 개선. paired exact sign-flip p=0.5. 독립 소자가 2개여서 유의성/광범위한 일반화를 주장하지 않는다.
- Health convex hull 밖 비율0%, 거리 중앙값0. 다차원 full-feature hull 결과가 아니다.
- 시간 baseline은 training 소자 평균 도달시간에서 현재 경과시간을 뺀 값이다. 결과 후 설명용으로 추가했고 test-fitting은 하지 않았다. 이것도 R²0.984여서, 높은 pooled R²의 상당 부분은 공통 수명과 경과시간 관계로 설명된다.
- **PP–MLP 차이 +0.01459, PP–ridge 차이 +0.000127**. 새로운 NN 구조의 큰 이득이나 강한 외삽 노벨티를 입증하는 자료는 아니다.

## 후보별 상태

| 후보 | 확보/검증 | 판정 |
|---|---|---|
| NASA Capacitor Stress-2 | 공식 legacy ZIP 확보, 11시점×6열, 5seed 비교 완료 | 새 소자 파일럿 가능. strict state-tail은 관측 부족 |
| NASA Capacitor ES10 | 1.2GB MAT CRC 검증, 8개 EIS 이력 추출 | 100Hz series-C 지표의 최소 health가 약0.842–0.888이라 0.8 기준 미도달. 실제 등가회로 추정 C와 동일하지 않음. 이 추출법으로 RUL 구성 불가 |
| PHM2019 피로균열 | 원본 ZIP/훈련 시편 설명 확보 | 3–9개 검사/시편, 마지막 관측=파단시간 확인 불가. RUL 성공으로 집계 안 함; 원본 T7–T8 결과 평가 안 함 |
| NASA MOSFET | 공식 압축파일 목차 확인 | 약7.85GB의 중첩 ZIP. 이번에는 원자료 전체 다운로드·학습하지 않음 |
| NASA Milling | 기존 사용 이력 확인 | 새 코호트에서 제외 |

Stress-2 strict 설정은 train health>0.9, test 0.8<health<=0.9이다. 소자6의 해당 구간 관측이 2개뿐이라 사전 최소3개 기준을 만족하지 않는다. 그 기준을 낮추지 않았고, 별도 허용한 **all-pre-event/new-device 파일럿**만 실행했다. 이는 새 소자 일반화이며 학습 범위 밖 열화 상태 검증을 대체하지 않는다.

## 출처와 재현

- [NASA PCoE 데이터 목록](https://www.nasa.gov/intelligent-systems-division/discovery-and-systems-health/pcoe/pcoe-data-set-repository/)
- [Stress-2 공식 legacy ZIP](https://data.nasa.gov/docs/legacy/EOS_DataSet.zip): 주 목록에는 다운로드 불가로 표기돼 있었으나 공식 Open Data API의 legacy 링크 GET은 실제 동작했다. 로컬 `cap2_tail.bin`은 이름과 달리 전체1648-byte ZIP이다.
- [Celaya et al., 2012, IJPHM](https://papers.phmsociety.org/index.php/ijphm/article/download/1364/347): 여섯 커패시터, 정전용량 감소율, 20% 기준 확인. 제공 행렬의 개별 열-실물 대응은 원 출처에 의존한다.
- [Renwick et al., 2015, PHM](https://papers.phmsociety.org/index.php/phmconf/article/view/2713): ES10 EIS 등가회로 추정과 임계치 이후 회복 현상. 단일 주파수 Cs와 논문의 피팅 C를 혼동하지 않는다.

```
/opt/anaconda3/bin/python experiments/extract_capacitor_stress2.py
/opt/anaconda3/bin/python experiments/nonbattery_cohort_eval.py data/nonbattery_external/capacitor_stress2_screening.json strict
/opt/anaconda3/bin/python experiments/nonbattery_cohort_eval.py data/nonbattery_external/capacitor_stress2_screening.json newunit
```

결과: `results/capacitor_stress2_screening_newunit_locked_v1/results.json` 및 `predictions.npz`.
사전 규칙: `NONBATTERY_EXTERNAL_PROTOCOL.md`, 해시: `NONBATTERY_EXTERNAL_PRETEST_SHA256.txt`.
생성 코드: `experiments/nonbattery_cohort_eval.py`, `experiments/extract_capacitor_stress2.py`, `experiments/extract_capacitor_eis.py`.
이번 시험 결과를 확인했으므로 이 코호트를 향후 모델 선택에 사용하면 이후 성능은 개발 결과로 표시해야 한다.
