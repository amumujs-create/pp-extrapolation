# PAE–PP 박사논문 스토리와 PP 차별 노벨티

## PAE 저장소 감사 결과

PAE는 이미 다음을 다룬다.

- typed observation contract로 도메인에서 허용되는 prior를 결정한다.
- 직접 관측 health와 boundary가 있으면 hard-boundary integral/gated head를 사용한다.
- 관측 가능한 물리 좌표가 없으면 `prior_off_sequence`로 보낸다.
- 기계 strict OP hull에서는 staged affine-tail NN을 사용한다.
- 배터리 cohort LODO에서는 새 cohort 한 unit으로 scalar output scale을 맞춘다.
- prior 강도 선택, 적용 coverage, 실패 및 prospective transport를 구분한다.

따라서 PP를 `affine + NN`, `prior routing`, `새 도메인 calibration`만으로 정의하면 PAE와
학술적으로 겹친다.

## 박사논문의 두 층

### PAE: structural assumption compilation

PAE의 질문은 **무엇을 모델 안에 넣어도 되는가**이다.

관측 계약에서 health, boundary, direction, condition axis의 존재를 검사하고, 허용된 구조
prior만 network architecture 또는 loss로 컴파일한다. 결과는 hard boundary, monotonic
direction, affine tail 또는 prior-off representation이다.

### PP: empirical error transport after compilation

PP의 질문은 **학습된 외삽기의 어떤 오류를 새 unit으로 옮겨 보정해도 되는가**이다.

PP는 예측을 세 상태로 진단한다.

1. **shape retained, scale shifted:** 예측 순서와 곡선 형태는 유지되지만 slope/offset이
   validation cohort에서 반복적으로 이동한다. 제한된 양의 affine output transport 후보다.
2. **shape and scale stable:** 보정 이득의 반복 증거가 없다. 원 prediction을 유지한다.
3. **shape failure or unsupported mechanism:** affine transport도 validation group 사이에서
   재현되지 않는다. 보정을 거절하고 applicability layer가 필요하면 abstain한다.

이 구분에서 PP는 PAE의 대체 모델이 아니다. PP 논문에서는 독립적인 neural extrapolator의
**post-training verifier and transport controller**로 정의한다.

## 강화된 PP: geometry-compatible dual-replicated error transport

최종 PP 출력은 다음과 같다.

\[
\hat y_{PP}=\operatorname{clip}(a\,f_{prior+residual}(x)+b,0,y_{max}),
\]

단, `(a,b)`는 아래 geometry 자격과 두 통계 증거를 동시에 통과할 때만 사용한다.

### 축 0: typed geometry eligibility

PP 데이터 프로토콜에서 사전 선언한 외삽 좌표를 사용해 validation과 test shift가 train
support의 같은 방향인지 검사한다. 두 shift ray의 cosine이 0 이하이면 validation
correction을 test로 운반하지 않는다. PP 논문은 이 좌표를 PAE 출력에 의존하지 않고
독립적으로 입력받는다.

### 축 1: optimization replication

각 seed가 별도 초기화로 학습된 PP prediction에 대해 group-LOO에서 affine을 identity보다 선택하는지
투표한다. Exact one-sided binomial test `alpha=.05`를 사용한다. 5 seeds에서는 5/5만
통과하며 p-value는 0.03125다.

### 축 2: physical-unit replication

Validation unit 하나를 제외하고 affine을 적합한 뒤 제외 unit의 identity 대비 MSE gain을
계산한다. Unit별 paired gain의 bootstrap 95% lower bound가 0보다 클 때만 통과한다.

Geometry와 두 통계 축을 모두 통과하지 않으면 `(a,b)=(1,0)`으로 되돌아간다. Test label과
domain-specific degradation equation은 승인에 사용하지 않는다. Test의 predeclared
condition coordinate는 예측 시 관측 가능한 geometry 자격 검사에만 사용한다.

## 실제 validation-only 이중 검증

| 데이터 | seed evidence | validation unit evidence | 최종 route |
|---|---:|---:|---|
| HUST | 5/5, p=.03125 | mean MSE gain 1213; CI **[215, 2355]** | affine transport |
| RWTH | 5/5, p=.03125 | mean MSE gain 4314; CI **[709, 7805]** | affine transport |
| MATR2019 | 5/5, p=.03125 | mean MSE gain 765; CI **[157, 1430]** | affine transport |
| Virkler | 1/5 | CI crosses 0 | identity |
| Sunwoda | 0/5 | negative CI | identity |
| MICH | 0/5 | negative CI | identity |
| N-CMAPSS | 0/5 | negative CI | identity |

이 결정 후 test 결과는 승인된 세 데이터에서만 개선됐다.

| 데이터 | base PP pooled R² | dual-evidence PP pooled R² |
|---|---:|---:|
| HUST | .773 | **.899** |
| RWTH | .507 | **.743** |
| MATR2019 | .257 | **.466** |

나머지는 identity route로 원 점수를 유지했다. 따라서 모델의 강점은 보정 평균 성능뿐 아니라
**validation의 두 상보적 변동 축에서 재현되지 않는 correction을 test로 운반하지 않는 것**이다.

## PAE와 PP가 겹치지 않는 기여 문장

### PAE 논문

> We compile admissible structural priors from typed observation contracts before
> training an extrapolation model.

### PP 논문

> We determine whether a post-training extrapolation error is transportable by requiring
> replicated evidence across both optimization seeds and held-out physical units, and
> apply a shape-preserving output correction only after this dual certificate passes.

### 박사논문 통합 문장

> The dissertation separates extrapolation into assumption compilation before training
> and evidence-certified error transport after training.

이 구조는 “물리식을 넣은 NN”보다 넓다. PAE는 prior가 없는 도메인에서 prior-off를 선택할
수 있고, PP는 어떤 backbone에도 적용 가능한 post-training transport verifier로 확장할 수
있다. PP가 FT에도 적용될 수 있다는 사실은 약점이 아니라 model-agnostic verifier라는
주장의 근거다. 다만 현재 성능 우위는 PP backbone에서 확인됐으며, FT 대조에서는 같은
보정 규칙을 적용한 결과를 공정 비교로 보고한다.

## 저널과 박사논문의 역할

- **PP 저널:** strict-tail RUL, prior-residual backbone, dual-replicated error transport,
  identity fallback, cross-domain 결과에 집중한다.
- **PAE 저널:** typed contracts와 structural prior compilation에 집중한다.
- **박사논문:** `assumption compilation → extrapolator learning → error diagnosis → certified
  transport/abstention`의 전체 생명주기로 통합한다.

## 아직 필요한 마지막 차별 실험

1. PP 외의 FT/MLP에도 dual certificate를 적용해 verifier의 model-agnostic 성질을 보인다.
2. seed-only, unit-only, dual gate를 제거 비교하여 오승인과 성능 손상을 측정한다.
3. affine이 성공하도록 고른 결과라는 반론을 막기 위해 bias·isotonic·unbounded affine과
   같은 correction family를 동일한 이중 gate 아래 비교한다.
4. 마지막 새 dataset에는 dual rule을 고정한 뒤 한 번 적용한다.

현재 결과는 PP의 방법론적 노벨티를 `새 network block`에서 **transportability를 검증하는
학습 후 통계 계층**으로 이동시킨다. 이 방향이 PAE와 가장 적게 겹치고 박사논문 전체
스토리를 가장 강하게 만든다.

XJTU opposite-condition stress에서 통계적 dual gate만 사용하면 잘못 affine을 승인해
`-1.308→-1.666`으로 악화됐다. Typed geometry를 추가하면 shift-ray cosine `-1.0`을
label 없이 검출해 transport를 거절하고 원 PP를 보존한다. 전체 applicability layer는
기존 결정대로 XJTU 예측을 abstain한다. 이 실패와 수정은 `XJTU_GEOMETRY_TRANSPORT_STRESS_KO.md`에
기록했다.

Seed들은 같은 자료와 학습 절차를 공유하므로 완전히 독립적인 과학 표본은 아니다. 따라서
seed exact p-value는 물리적 일반화의 유의확률이 아니라 고정된 알고리즘 승인 기준으로
해석한다. 물리 표본의 불확실성은 별도의 validation-unit bootstrap과 test-unit 분석으로
보고한다.
