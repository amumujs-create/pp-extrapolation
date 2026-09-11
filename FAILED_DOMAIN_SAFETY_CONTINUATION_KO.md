# PP-X 실패 도메인 safety-continuation 개발 감사

> **Canonical paper pointer:** 이 문서는 XJTU·FEMTO·NASA milling의
> post-test limitation/development evidence다. 논문 메인은 PP-X이며, 아래
> `PP`, affine PP, safety PP는 당시 실험 arm의 역사적 라벨이다. 세 설정은
> PP-X main superiority portfolio에 포함하지 않는다. 최신 주장 경계는
> `PP_INFORMATION_LIMITS_AND_CLAIMS_KO.md`를 우선한다.

## 목적과 증거 구분

새 cohort는 열지 않았다. 이미 결과를 본 XJTU, FEMTO, NASA milling 세 설정만
사용해 legacy PP backbone이 잘못된 prior를 받았을 때 망가지는 문제를 개발
단계에서 점검했다. 따라서 아래 수치는 **post-test development**이며 PP-X의
외부 확증 결과가 아니다.

핵심 구조는 하나의 PP 네트워크 안에 affine 경로와 direct neural 경로를 두고

\[
\hat y = \tau\hat y_{affine} + (1-\tau)\hat y_{NN}
\]

로 계산하는 것이다. `tau=0` 후보는 별도 모델을 섞은 ensemble이 아니라 동일 PP 네트워크 내부의 정확한 NN 부분공간이다. 동일 seed의 독립 MLP와 초기화·optimizer·checkpoint 선택을 맞추기 위해 residual seed replay를 사용했다. 실제 수치 감사에서 `tau=0`인 FEMTO와 milling의 최대 예측 차이는 정확히 0이었다.

## 선택 규칙

- standalone NN은 width `{32, 64}`, learning rate `{5e-4, 1e-3}`, weight decay `{0.1, 2.0}`를 validation ensemble RMSE로 독립 선택
- PP는 같은 `2×2×2×2=8` optimizer/width 조합과 affine trust `tau={0, .02, .05, .1, .2, .4}`를 결합한 48개 후보를 validation에서 **공동 튜닝**
- 최저 validation RMSE의 0.5% 안에 여러 후보가 있으면 prior가 가장 약한 후보를 선택
- 선택 seed 42--44, 최종 재학습 seed 42--46
- test label은 선택 함수에 입력하지 않음

## 결과

| 설정 | 기존 최종 PP R2 | safety PP R2 | 비교 기준 | 해석 |
|---|---:|---:|---:|---|
| XJTU condition transfer | **-1.229** | -1.894 (`tau=.4`) | linear-tail RBF -1.418 | safety 후보가 기존 PP보다 나빠 미채택; ray 반전 때문에 적용 거절 유지 |
| FEMTO endpoint transfer | -1.378 | **-1.165** (`tau=0`) | monotone NN -0.973 | 개선됐지만 경쟁모델보다 낮고 절대 R2도 음수; 적용 거절 유지 |
| NASA milling material transfer | -4.826 | **-0.476** (`tau=0`) | GroupDRO -0.691; 동일 tuned MLP -0.476 | 큰 붕괴를 제거하고 기존 최고 경쟁모델을 넘었지만, NN 경로와 동률이며 절대 성공은 아님 |

XJTU에서는 validation이 `width=64, lr=5e-4, wd=2, tau=.4`를 골랐지만 test의 운전조건 이동 방향이 validation과 반대여서 성능이 재차 무너졌다. 이는 공동 hyperparameter/trust tuning만으로 transport 방향 반전을 해결할 수 없다는 음성 결과다. 기존 PP -1.229를 최종 개발값으로 유지하되, 실제 executor는 이 설정을 예측 승인하지 않아야 한다.

FEMTO에는 causal elapsed time, 초기 신호 대비 변화, 단·중기 slope를 넣어 latent total-life와 direct RUL을 추가 진단했다. held-out learning bearing에서는 복잡한 모델이 좋아졌으나 11개 공식 test bearing에서는 반대로 악화했다. 학습 run-to-failure bearing 6개와 validation bearing 1개로는 서로 크게 다른 개체 수명 scale을 식별하지 못한다. 더 깊은 NN의 문제가 아니라 supervision과 split representativeness의 문제다.

NASA milling은 prior 자체가 부정확할 때 affine 경로를 강제로 유지한 것이 주된 붕괴 원인이었다. validation은 `tau=0`을 선택했고 PP가 정확한 direct-NN 경로로 연속 수축하면서 R2가 -4.826에서 -0.476으로 회복됐다. 다만 validation 4행·1 unit, test 10행이라는 작은 표본 때문에 이를 성공 설정으로 승격하지 않는다.

후속 domain-prior 감사에서는 데이터 프로토콜에 이미 고정된 마모 고장경계 `VB=0.50`을 모델 구조에 반영했다. 희소한 inspection이 경계 사이를 건너뛰는 현상은 validation MAE로 선택한 `+0.03` margin으로 보정했다. `RUL=(0.50+offset-health)/causal_rate + gate*NN residual`에서 material-2가 train material-1에 없다는 인증으로 test residual gate를 0으로 둔다. 그 결과 validation R2 0.638, test R2 **0.341**로 tuned NN -0.476과 GroupDRO -0.691을 넘었다. 이는 새 외부 확증이 아니라 이미 관측된 test에서의 개발 결과다.

## PP-X limitation 정책

1. 기존 9개 retrospective setting의 PP-X paper-selected route와 결과는
   변경하지 않는다.
2. 여러 validation unit에서 prior gain이 반복되면 기존 affine/residual 또는 domain-specific PP route를 사용한다.
3. prior gain이 없으면 safety-continuation의 `tau=0` direct-NN 부분공간으로 수축한다.
4. validation/test transport ray가 반대이거나 test가 unit당 endpoint 하나뿐이고 lifetime-scale 근거가 없으면 점 예측 성능표의 승인 설정에서 제외한다.

이 정책을 `certify_extrapolation` API에 코드로 고정했다. 기존 validation skill·baseline gain·seed stability에 더해 validation group 수, source의 unit당 관측 수, transport 방향 호환성을 선택적으로 검사한다. 세 값은 split metadata와 source input으로만 계산하며 source label을 사용하지 않는다.

따라서 논문 주장은 “PP-X가 모든 데이터셋에서 항상 이긴다”가 아니다. 이
감사는 prior가 실패한 setting에서 direct-NN 안전 경로 또는 사전 거절이
catastrophic negative transfer를 제한할 수 있다는 retrospective mechanism
evidence만 제공한다.

재현 코드는 `experiments/failed_domain_safety_continuation.py`, 원시 결과는 `results/failed_domain_safety_continuation_v1/results.json`에 있다.
