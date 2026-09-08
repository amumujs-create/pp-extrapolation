# C-MAPSS regime-normalized PP 개선 결과

## 무엇을 고쳤나

첫 PP cohort 실험은 원 센서와 단순 causal 이동통계를 사용했다. 이 표현에서는 운전조건에 따른 센서 offset이 열화와 섞여 frozen affine path가 잘못된 축을 잡았다.

개선 표현은 다음을 모두 **학습 engine만으로** 계산한다.

- 운전조건 cluster 안에서 센서를 정규화
- 30-cycle causal window의 현재값·평균·표준편차·기울기
- 각 engine의 첫 20 cycle 기준 대비 현재 센서 변화
- 현재 운전조건과 log-cycle

모델은 바꾸지 않았다. 하나의 PP network 안에 frozen affine path와 tanh NN residual path가 있고 한 번의 forward pass로 예측한다. 데이터셋별 수명식이나 기성모델 mixture는 없다.

## 공정 비교

NASA C-MAPSS의 일반적인 공식 endpoint 과제에 맞춰 train RUL과 예측을 125 cycle에서 cap하고, 공식 test truth는 원값 그대로 평가했다. PP와 plain MLP는 동일한 입력, unit split, seed, target, batch size와 validation epoch selection을 사용한다. 따라서 아래 표는 기존 uncapped 실험의 숫자를 덮어쓰는 표가 아니라, **표현 개선을 matched MLP와 비교한 표준 endpoint 분석**이다.

| subset | PP ensemble R² | plain MLP ensemble R² | PP RMSE | MLP RMSE | 판정 |
|---|---:|---:|---:|---:|---|
| FD001 | **0.911** | 0.887 | **12.369** | 13.949 | PP 우세 |
| FD003 | **0.909** | 0.894 | **12.503** | 13.509 | PP 우세 |

PP는 5개 seed 각각에서도 두 subset 모두 plain MLP보다 높은 R²를 기록했다.

| subset | PP seed R² 범위 | MLP seed R² 범위 | MLP−PP RMSE gain 95% bootstrap CI | paired sign-flip p |
|---|---:|---:|---:|---:|
| FD001 | 0.905–0.913 | 0.873–0.890 | **[0.477, 2.742]** | 0.0041 |
| FD003 | 0.905–0.910 | 0.889–0.897 | **[0.361, 1.640]** | 0.0018 |

bootstrap과 permutation의 독립 단위는 공식 test engine 100대다. CI가 모두 0보다 크므로 ensemble 한 번의 우연한 오차상쇄만으로 설명되기 어렵다.

## 해석과 한계

이번 결과는 모델의 affine prior가 작동하려면 **운전조건과 열화가 분리된 좌표**가 먼저 필요하다는 정량적 증거다. 이전 uncapped 표현에서 PP는 FD001 0.641, FD003 0.249로 양수였지만 MLP보다 낮았다. 새 matched 비교에서는 두 subset 모두 역전됐다.

다만 이 표현은 첫 test 결과를 본 뒤 개발했고 C-MAPSS는 PAE 연구에서 이미 사용했다. 따라서 `untouched external confirmation`이나 새로운 데이터셋 발견으로 주장하지 않는다. 논문에서는 posthoc mechanism repair와 five-seed paired evidence로 사용하고, 다음 새 기계 cohort에는 이 표현 규칙을 동결해 적용해야 한다.
