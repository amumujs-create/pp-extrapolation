# PP 논문: 정보 한계와 주장 경계

## 무엇을 주장하는가

PP는 관측 시점까지의 인과적 이력에서 strict out-of-support RUL을 추정하는 구조다. 아핀 tail과 제한된 비선형 residual을 분리해, 학습 support 밖에서 residual의 과도한 연장을 줄이는 것이 목적이다. 이 논문은 모든 열화 도메인에서 우월하다고 주장하지 않는다.

## 관측 정보가 만드는 이론적 한계

관측 시점 `t`의 인과적 이력을 `H_t`, 남은수명을 `Y_t`, 아직 관측되지 않은 환경·제조·향후 regime 변수를 `Z`라 하자. 어떤 예측기 `f(H_t)`도 조건부 평균보다 작은 제곱오차 위험을 가질 수 없다.

\[
\mathbb{E}\{(Y_t-f(H_t))^2\}\geq \mathbb{E}[\mathrm{Var}(Y_t\mid H_t)].
\]

특히 서로 다른 unit이 거의 같은 prefix `H_t`를 보이지만 이후 knee, 부하, 또는 lifetime scale `Z`가 달라진다면, prefix만으로는 두 RUL을 구별할 수 없다. PP의 support-aware residual 제어는 **보이지 않는 `Z`를 복원하는 방법이 아니다**. 이 식은 음수 R²가 특정 신경망의 무능만을 뜻하지 않는 이유와, 새 구조를 무한히 추가해도 해소되지 않는 조건부 모호성의 원인을 설명한다.

이는 불가능성의 보편적 증명은 아니다. 더 긴 raw-signal history, 운전조건·제조 메타데이터, 더 많은 run-to-failure unit, 또는 사전에 알려진 물리적 제약이 제공되면 `H_t`가 풍부해지고 위 조건부 분산은 줄 수 있다. 다만 그런 정보는 현재 PP 입력에 없으므로 성능표에 사후적으로 추가하면 PP의 일반 외삽 주장과는 별개 실험으로 취급한다.

## 현재 실패 도메인의 진단

### XJTU

- train 최대 RUL 525보다 큰 test 행이 70.7%이고, test 조건의 lifetime scale은 train/validation과 다르다.
- 예측을 train target 범위에 완벽히 맞춘 oracle도 R²=-0.516이며, test label을 보고 맞춘 하나의 공유 총수명 scale oracle도 R²=0.231에 그쳤다.

따라서 단일 condition-level scale을 외삽하는 PP에는 구조적 mismatch가 있다. 필요한 정보는 bearing별 scale을 식별하는 load–life covariate, 해당 조건의 추가 run-to-failure bearing, 또는 사전에 고정된 물리 제약이다. 이는 test 적응이나 TTA가 아니다.

### FEMTO

기존 결과는 vibration 채널 대신 시간 열을 사용하고 일부 prefix feature가 미래를 섞은 loader/cache 문제가 발견되어 최종 주장에 사용할 수 없다. 채널을 고친 재실행도 안정적 양의 R²를 보이지 못했고, 최선 ensemble 약 0.075는 각 seed가 모두 양수라는 증거가 아니다. 11개 endpoint가 각 bearing당 한 점이어서 unit-level R²와 유의성 검정도 성립하지 않는다.

따라서 FEMTO는 “PP가 실패했다”가 아니라 “현재의 causal feature/표본 설계로 PP 우월성을 판정할 수 없다”로 보고한다. 최종 비교에는 causal raw waveform loader, unit-disjoint 다점 tail, 사전 고정한 train-only normalization, 그리고 충분한 independent bearing 수가 필요하다.

## 사전 gate와 PP 예측기의 구분

12-domain descriptor 기반 LODO gate는 balanced accuracy 0.100, permutation p=0.945로 실패했다. 따라서 현 논문은 보편적인 pre-outcome selector를 주장하지 않는다. HNEI locked-split에서 PP 자체는 pooled R² 0.994 대 plain MLP 0.961이었지만, 첫 outcome 확인 뒤 normalization repair가 있었고 certificate는 PP를 승인하지 않았다. 이는 개발적 외부 성능 증거이며 prospective gate 성공이 아니다.

## 논문에 쓰지 말아야 할 문장

- “사전 gate가 새로운 도메인에서 검증되었다.”
- “음수 R² 도메인은 어떠한 모델도 해결할 수 없다.”
- “모든 PHM/RUL 도메인에 PP가 우월하다.”

대신 PP가 유리했던 support-shift 조건, 실패한 조건, 그리고 관측정보를 늘려야 하는 조건을 각각 표와 함께 보고한다.
