# PP 전 데이터셋 개선 감사

## 개선 구조

PP의 예측을 다음과 같이 확장했다.

\[
\hat y = \hat y_{\mathrm{affine}} + a\,\hat r_{\mathrm{NN}}
\]

`a`는 기성 모형을 섞는 가중치가 아니라 PP 내부 NN residual의 신뢰도다. 후보 `0, 0.25, ..., 1.5` 가운데 validation ensemble MSE로 선택한다. `a=1`은 기존 PP, `a=0`은 affine prior만 쓰는 경우다.

무조건 선택한 gain을 쓰면 일부 데이터에서 validation과 test의 관계가 달라져 악화했다. 최종 개발 규칙은 **validation ensemble MSE가 기존 PP보다 2% 이상 좋아지고 validation unit의 80% 이상에서 이길 때만 gain을 승인**한다. 두 조건 중 하나라도 실패하면 `a=1`을 유지한다. 평균 점수 하나 대신 새로운 unit으로 전달되는 일관성을 확인하는 group-robust gate다. 이 규칙은 아래 데이터를 관측한 뒤 만든 사후 개발 규칙이며 새로운 cohort에서 아직 검증되지 않았다.

시계열 window가 있는 N-CMAPSS에는 별도의 causal multiscale PP 모듈을 사용했다. 현재값과 함께 여러 시간 폭의 평균·표준편차·기울기를 계산하고 feature preset을 validation에서 선택한다.

## 동일 규칙 재실행 결과

아래 수치는 5-seed ensemble pooled R²다. `개선 PP`는 group-robust residual gate 및 N-CMAPSS temporal module을 적용한 개발 결과다.

| 데이터/프로토콜 | 기존 PP | 개선 PP | 변화 | 선택 |
|---|---:|---:|---:|---|
| HUST unseen-protocol tail | 0.773 | 0.773 | 0.000 | 기존 유지 |
| Virkler unseen-specimen tail | 0.888 | 0.888 | 0.000 | 기존 유지 |
| NASA battery health tail | 0.495 | 0.495 | 0.000 | unit-consistency 미달, 기존 유지 |
| Sunwoda unseen-cell tail | 0.865 | 0.865 | 0.000 | unit-consistency 미달, 기존 유지 |
| **RWTH unseen-cell tail** | 0.507 | **0.551** | **+0.044** | residual gain 승인 |
| MICH unseen-cell tail | -1.522 | -1.522 | 0.000 | 기존 유지 |
| MATR2019 latent PP | 0.257 | 0.257 | 0.000 | 기존 유지 |
| MATR batch 2 | 0.465 | 0.465 | 0.000 | 기존 유지 |
| **XJTU condition transfer** | -1.308 | **-1.229** | **+0.080** | residual gain 승인, 여전히 실패 |
| FEMTO endpoint | -1.378 | -1.378 | 0.000 | 기존 유지 |
| NASA milling transfer | -4.826 | -4.826 | 0.000 | 기존 유지 |
| C-MAPSS FD002+FD004 strict OP-hull | 0.747 | 0.747 | 0.000 | gain=1 유지 |
| **N-CMAPSS hard** | 0.934 | **0.937** | **+0.003** | causal multiscale PP |

관측한 13개 설정에서 3개가 개선됐고 10개는 기존 PP와 같았으며, group-robust gate를 적용한 최종 열에서는 악화가 없었다. 별도로 논문 직접 비교의 latent PP HUST·Virkler·MATR2019에 적용했을 때도 세 데이터 모두 gain=1을 선택해 기존 성능을 정확히 보존했다. XJTU·MICH·FEMTO·Milling은 개선 구조를 적용해도 음의 R²이므로 해결된 데이터로 세지 않는다.

## 폐기한 평균 기반 문턱의 Leave-one-domain-out 검증

초기에는 unit 일관성을 보지 않고 domain 평균 validation 개선 문턱만 사용했다. 승인 문턱을 각 target domain을 제외한 나머지 domain에서 선택하고 target에 적용했다. 선택 목적은 먼저 악화된 개발 domain 수를 최소화하고, 다음으로 평균 R² 개선을 최대화하며, 동률이면 높은 문턱을 택하는 것이다.

결과는 **1개 개선, 10개 동일, 1개 악화**였다. XJTU는 +0.080 개선됐고 N-CMAPSS는 -0.000024 하락했다. 평균 변화는 +0.00665였지만 RWTH 개선은 target을 제외했을 때 선택된 문턱이 너무 높아 적용되지 않았다. 이 결과 때문에 평균 문턱 방식은 폐기하고 unit-consistency 조건으로 교체했다.

## 무조건 gain을 적용했을 때의 반증

| 데이터 | 기존 PP | 무조건 gain PP | 변화 |
|---|---:|---:|---:|
| Sunwoda | 0.865 | 0.862 | -0.003 |
| N-CMAPSS | 0.934300 | 0.934277 | -0.000024 |
| latent Virkler deep-future | -5.374 | -6.558 | -1.184 |

따라서 scalar gain 자체를 보편 개선 모델이라고 주장할 수 없다. 논문의 모델은 **기본 PP + 데이터 형태별 후보 모듈 + validation evidence gate**로 정의해야 한다. 모듈이 승인되지 않으면 기존 PP를 그대로 실행한다.

## 논문에서 가능한 주장

현재 결과는 “모든 데이터에서 정확도가 상승했다”는 증거가 아니다. 방어 가능한 주장은 사후 개발 데이터에서 group-robust modular PP가 기존 PP를 보존하면서 일부 도메인의 성능을 높였다는 것이다. 2%·80% 승인 규칙과 temporal module을 고정한 뒤 새로운 untouched 데이터에 한 번 적용해야 일반화와 non-degradation 주장이 성립한다.

MATR2019에서 FT-Transformer 0.331보다 낮은 PP 0.257은 이번 gain으로 개선되지 않았다. 이 문제는 scalar calibration이 아니라 sequence representation 또는 regime transition 모델의 추가 개선이 필요하다.
