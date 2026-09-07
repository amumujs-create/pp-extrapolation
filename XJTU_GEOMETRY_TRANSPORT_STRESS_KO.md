# XJTU opposite-ray transport stress test

## 목적과 지위

Dual error-transport gate를 XJTU-SY condition transfer에 적용했다. XJTU test label은 과거
PP 연구에서 이미 확인했으므로 untouched confirmation이 아니라, 새 transport verifier의
외부 failure stress test다. 모델이나 보정 계수는 XJTU test label로 선택하지 않았다.

## 최초 dual gate 실패

- Train condition: 2250 rpm, 11 kN
- Validation condition: 2100 rpm, 12 kN
- Test condition: 2400 rpm, 10 kN
- 다섯 seed 모두 validation에서 affine 선택
- Validation-unit bootstrap MSE gain CI: `[427, 9204]`
- 기존 PP test pooled R²: `-1.308`
- affine transport를 적용한 test pooled R²: `-1.666`

Seed와 validation unit에서 모두 반복된 scale correction도 반대 방향의 condition shift에는
전이되지 않았다. Statistical replication만으로 transportability를 정의한 최초 dual gate는
이 사례에서 오승인했다.

## 원인과 geometry certificate

Train condition centroid에서 validation과 test condition centroid로 향하는 ray의 cosine을
계산했다.

\[
c=\frac{(\bar z_{val}-\bar z_{train})^\top(\bar z_{test}-\bar z_{train})}
{\|\bar z_{val}-\bar z_{train}\|\,\|\bar z_{test}-\bar z_{train}\|}.
\]

XJTU에서는 `c=-1.0`이다. Validation과 test가 train을 기준으로 정확히 반대 외삽 방향이다.
따라서 label-free geometry certificate가 correction transport를 거절한다.

최종 규칙은 다음 세 조건을 요구한다.

1. seed exact test 통과;
2. validation physical-unit bootstrap 하한 > 0;
3. predeclared extrapolation coordinate에서 validation/test shift cosine > 0.

## 최종 결과

| 방법 | transport 승인 | test pooled R² |
|---|---:|---:|
| 원 PP | — | -1.308 |
| geometry 없이 affine transport | 잘못 승인 | -1.666 |
| **geometry-compatible transport** | **거절** | **-1.308 유지** |

Geometry certificate는 실패한 PP를 성공 모델로 만들지 않는다. 기존 XJTU applicability
gate가 이미 abstain을 선택했으므로 전체 시스템 출력은 여전히 prediction abstention이다.
이번 모듈의 역할은 반대-ray validation correction으로 실패를 더 악화시키지 않는 것이다.

## 노벨티 의미

데이터 프로토콜에서 사전 선언한 extrapolation coordinate가 PP의 통계적 transport
verifier에 연결된다. PP는 같은 ray에서 반복된 오류만 보정한다. 따라서 강화된 방법은

`typed geometry eligibility → seed replication → unit replication → transport/identity`

가 된다. 이는 validation에서 좋아진 calibration을 임의의 OOD 방향으로 옮기는 일반적인
post-hoc calibration과 구별된다.

기계 판독 결과: `results/xjtu_dual_transport_stress_v1/results.json`
