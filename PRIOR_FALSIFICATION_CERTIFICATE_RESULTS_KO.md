# PP-X prior falsification certificate 실험 결과

작성자: 박진서  
상태: 12개 공개 retrospective development dataset 감사  
프로토콜: `protocols/PRIOR_FALSIFICATION_CERTIFICATE_PROTOCOL.md`

## 결론

고정된 prior falsification certificate는 기존 PP-X paper policy의
**false accept를 2건에서 0건으로 줄였다.** 정책 정확도는 8/12에서
10/12로 증가했고, 승인된 네 dataset은 test에서 모두 PP가 fallback보다
좋았다.

그러나 이 결과는 이미 공개된 12개 dataset의 retrospective 감사다. 새로운
미개봉 domain에서 확인하기 전에는 frozen paper policy를 교체하지 않는다.

## Certificate

validation physical unit \(u\)마다 fallback 대비 PP의 log-RMSE regret를
계산했다.

\[
r_u=
\log
\frac{\operatorname{RMSE}_{u,\mathrm{PP}}}
     {\operatorname{RMSE}_{u,\mathrm{fallback}}}.
\]

physical unit bootstrap으로 평균 regret의 one-sided 95% upper confidence
bound \(U_{.95}\)를 구하고, \(U_{.95}<0\)인 경우에만 prior를 승인했다.
row와 random seed는 독립 표본으로 사용하지 않았다.

Primary 정책은 기존 paper 조건인 validation MSE 2% 개선, unit win
fraction 60% 이상, worst-unit RMSE ratio 1.10 이하를 먼저 적용한 뒤
certificate를 추가했다. threshold grid나 test 기반 최적화는 사용하지 않았다.

## 정책 비교

| 정책 | 승인 | 정확 | accuracy | false accept | false reject | 선택 효과 |
|---|---:|---:|---:|---:|---:|---:|
| always direct | 0 | 6/12 | 0.500 | 0 | 6 | 0 |
| always PP | 12 | 6/12 | 0.500 | 6 | 0 | 0.0819 |
| simple validation | 9 | 7/12 | 0.583 | 4 | 1 | 0.1079 |
| 기존 paper PP-X | 6 | 8/12 | 0.667 | 2 | 2 | 0.0414 |
| certificate only | 4 | **10/12** | **0.833** | **0** | 2 | **0.1543** |
| paper + certificate | 4 | **10/12** | **0.833** | **0** | 2 | **0.1543** |
| test oracle | 6 | 12/12 | 1.000 | 0 | 0 | 0.2689 |

선택 효과는 fallback을 선택한 dataset을 0으로 두고, 승인 dataset의
test mean unit log-RMSE improvement를 12개 dataset에 동일 가중한 값이다.

Primary 정책 효과의 dataset bootstrap 95% CI는
**[0.00085, 0.42637]**였다. 하한이 0을 근소하게 넘지만 dataset이 12개뿐인
retrospective 결과이므로 강한 확증으로 해석하지 않는다.

## Dataset별 판정

| dataset | validation unit | 기존 paper | certificate | test에서 PP 개선 | \(U_{.95}\) |
|---|---:|---:|---:|---:|---:|
| HUST | 16 | reject | reject | no | +0.197 |
| Virkler | 10 | approve | approve | yes | −1.039 |
| MATR2019 | 8 | reject | reject | no | +0.719 |
| Sunwoda | 2 | reject | reject | yes | +1.102 |
| RWTH | 8 | reject | reject | no | +0.699 |
| MICH | 6 | reject | reject | no | +0.880 |
| MATR-b2 | 9 | approve | approve | yes | −0.124 |
| XJTU | 5 | reject | reject | yes | +1.172 |
| FEMTO | 1 | approve | reject | no | \(+\infty\) |
| Milling | 1 | approve | reject | no | \(+\infty\) |
| NASA battery | 4 | approve | approve | yes | −0.391 |
| N-CMAPSS | 6 | approve | approve | yes | −0.044 |

기존 정책의 false accept 두 건은 validation physical unit이 각각 하나뿐인
FEMTO와 Milling이었다. certificate는 unit 하나에서 confidence bound를
추정할 수 없으므로 두 route를 자동 거절했다. 두 dataset 모두 실제 test에서
PP가 fallback보다 나빴다.

승인된 dataset은 Virkler, MATR-b2, NASA battery, N-CMAPSS이며 모두 test에서
양의 mean unit log-RMSE improvement를 보였다. 90%, 95%, 97.5%, 99%
one-sided confidence를 사후 민감도로 확인했을 때 승인 집합은 네 dataset으로
동일했다.

## 중요한 한계

1. **효과의 상당 부분은 최소 unit 수 규칙이다.** 기존 false accept 두 건이
   모두 validation unit 하나인 dataset이었다. 이것만으로 복잡한 neural
   falsification head의 필요성을 입증하지 않는다.
2. **false reject 두 건이 남는다.** Sunwoda와 XJTU는 validation에서
   certificate를 통과하지 못했지만 test에서는 PP가 fallback보다 좋았다.
3. **N-CMAPSS margin은 작다.** \(U_{.95}=-0.044\), test mean log improvement
   0.005로 승인 경계와 실제 이득이 모두 작다.
4. **domain-level 표본 수는 12개다.** dataset bootstrap CI는 불안정하며
   meta-policy 일반화를 확증하기에 부족하다.
5. **공통 legacy backbone 감사다.** 최종 PP-X의 모든 contract별 executor를
   직접 재학습한 실험이 아니다.

## 모델적 노벨티 판정

이번 결과는 “prior falsification”이라는 개념의 유용성을 지지하지만, 현재
구현은 bootstrap certificate다. 독립적인 새 neural architecture라고 부르기에는
약하다.

논문에서 방어 가능한 표현은 다음과 같다.

> PP-X treats prior validity as a falsifiable claim: a prior-residual route is
> executed only when the upper confidence bound of its physical-unit
> fallback-relative regret is negative.

현재 첫 PP-X 논문에는 Algorithm 1의 보조 안전 조건 또는 supplementary
development result로 둘 수 있다. 메인 모형 노벨티로 승격하려면 다음 단계가
필요하다.

- 여러 executor의 nested OOF regret를 학습 target으로 생성
- train-only geometry, disagreement, curvature, contract violation을 입력
- unseen unit에서 regret upper bound를 예측하는 falsification head 구현
- fixed bootstrap certificate보다 calibration과 coverage가 우수한지 비교
- leave-one-domain-out와 새 prospective domain에서 false accept 검증

## 판정

- retrospective false-accept 감소: **통과**
- 선택된 test 효과 양수: **통과**
- 기존 paper policy 대비 accuracy 개선: **통과**
- 독립 prospective 확인: **미통과**
- 독립적인 neural model novelty: **미통과**

따라서 certificate는 **유망한 PP-X v2 challenger**로 보존하되 현재 frozen
paper model을 즉시 교체하지 않는다.
